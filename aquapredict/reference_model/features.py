"""Leak-safe feature extraction from one household's history (visible columns only)."""
from __future__ import annotations
import numpy as np
from ..schema import (DEFAULT_RATED, SEDIMENT_DP_RATIO_LIMIT, RO_FLUX_LIMIT, RO_REJECTION_LIMIT,
                      age_col, vol_col, repl_col)

TEMP_CORR_PER_C = 0.03
V1_NAMES = ["age", "vol", "u7", "u30", "u_cv", "idle_frac", "volrule_days", "miss_frac",
            "feed_cyc", "feed_14", "feed_shift", "temp_cyc", "inlet_cyc",
            "sed_level30", "sed_slope30", "sed_slope14", "sed_growth_per_L", "sed_trend_days",
            "flux_level30", "flux_slope30", "rej_level30", "rej_slope30", "flux_loss_per_load",
            "ro_load", "ro_trend_days",
            "n_prev", "prev_vol_last", "prev_vol_mean", "prev_age_last"]
# v2 adds: wear measured per LITRE (wear is driven by volume x load, not by calendar days), and
# water-source-change / variability signals.
V2_EXTRA = ["sed_slope_L", "sed_trend_days_vol", "flux_slope_L", "rej_slope_L", "ro_trend_days_vol",
            "days_since_feed_change", "feed_cv"]
# v3 adds the household's own history of water-source instability (past abrupt feed-TDS changes), which lets the
# scale model widen intervals for unstable households and keep them narrow for stable ones.
V3_EXTRA = ["n_feed_changes", "feed_change_rate", "hist_days"]
NAMES = V1_NAMES + V2_EXTRA + V3_EXTRA
PREV = {"n_prev", "prev_vol_last", "prev_vol_mean", "prev_age_last"}


def _nm(x):
    x = x[np.isfinite(x)]
    return float(x.mean()) if len(x) else np.nan


def _fit(age, y, win):
    a, v = age[-win:], y[-win:]
    m = np.isfinite(v)
    if m.sum() < 6 or np.ptp(a[m]) == 0:
        return np.nan, np.nan
    slope, level = np.polyfit(a[m] - a[-1], v[m], 1)
    return float(level), float(slope)


def _clean(y, n0=10):
    v = y[:n0]
    v = v[np.isfinite(v)]
    return float(np.median(v[:7])) if len(v) >= 5 else np.nan


def _ffill(x):
    x = x.copy()
    m = ~np.isfinite(x)
    if m.all():
        return x
    idx = np.where(~m, np.arange(len(x)), 0)
    np.maximum.accumulate(idx, out=idx)
    x = x[idx]
    x[: np.argmax(~m)] = x[np.argmax(~m)]            # leading gap takes first valid value
    return x


def _feed_change_and_cv(feed, w=7, thresh=0.25):
    """Days since the last abrupt change (>25%) in the weekly-mean feed TDS, and coefficient of variation."""
    n = len(feed)
    ff = _ffill(feed)
    if not np.isfinite(ff).any():
        return float(n), np.nan
    cv = float(np.std(ff[-30:]) / max(np.mean(ff[-30:]), 1e-9))
    if n < 2 * w:
        return float(n), cv
    cs = np.r_[0.0, np.cumsum(ff)]
    m = (cs[w:] - cs[:-w]) / w                       # m[i] = mean of ff[i:i+w]
    r = m[w:] / np.maximum(m[:-w], 1e-9)             # weekly mean vs the week before
    hit = np.flatnonzero(np.abs(r - 1.0) > thresh)
    return (float(n - (hit[-1] + 2 * w)) if len(hit) else float(n)), cv


def _change_events(feed, w=7, thresh=0.25, gap=14):
    """Number of distinct abrupt (>25%) weekly-mean changes in the whole available history."""
    ff = _ffill(feed)
    if len(ff) < 2 * w or not np.isfinite(ff).any():
        return 0
    cs = np.r_[0.0, np.cumsum(ff)]
    m = (cs[w:] - cs[:-w]) / w
    r = m[w:] / np.maximum(m[:-w], 1e-9)
    hit = np.flatnonzero(np.abs(r - 1.0) > thresh)
    return 0 if len(hit) == 0 else 1 + int((np.diff(hit) > gap).sum())


def extract(hist, stage, use_prev=False):
    """-> dict name -> float (NaN = not applicable / not available). Uses only rows of hist."""
    n = len(hist)
    age = int(hist[age_col(stage)].iloc[-1])
    cyc = hist.iloc[max(n - 1 - age, 0):]
    ag = cyc[age_col(stage)].to_numpy(float)
    usage = cyc["usage_L"].to_numpy(float)
    vol = float(cyc[vol_col(stage)].iloc[-1])
    idle = cyc["op_mode"].to_numpy() == "idle"
    f = {k: np.nan for k in NAMES}
    f.update(age=float(age), vol=vol, u7=float(usage[-7:].mean()), u30=float(usage[-30:].mean()))
    f["u_cv"] = float(usage[-30:].std() / max(f["u30"], 1.0))
    f["idle_frac"] = float(idle.mean())
    r = DEFAULT_RATED[stage]
    f["volrule_days"] = min((r["volume_L"] - vol) / max(f["u30"], 1.0), r["days"] - age)
    sens = cyc[["dp_sediment_bar", "flow_Lpm", "out_tds_ppm", "feed_tds_ppm", "temp_C"]].to_numpy(float)
    recent = sens[-14:][~idle[-14:]]
    f["miss_frac"] = float(np.isnan(recent).mean()) if len(recent) else 1.0
    feed, temp = cyc["feed_tds_ppm"].to_numpy(float), cyc["temp_C"].to_numpy(float)
    f["feed_cyc"], f["feed_14"] = _nm(feed), _nm(feed[-14:])
    f["feed_shift"] = f["feed_14"] / f["feed_cyc"] if np.isfinite(f["feed_cyc"]) and f["feed_cyc"] > 0 else np.nan
    f["temp_cyc"], f["inlet_cyc"] = _nm(temp), _nm(cyc["inlet_bar"].to_numpy(float))
    flow, dp, out = (cyc[c].to_numpy(float) for c in ("flow_Lpm", "dp_sediment_bar", "out_tds_ppm"))
    volv = cyc[vol_col(stage)].to_numpy(float)
    f["days_since_feed_change"], f["feed_cv"] = _feed_change_and_cv(feed)
    nch = _change_events(hist["feed_tds_ppm"].to_numpy(float))
    f["n_feed_changes"], f["hist_days"] = float(nch), float(n)
    f["feed_change_rate"] = nch / max(n / 365.0, 0.25)
    rate = max(f["u30"], 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        if stage == "sediment":
            y = dp / flow
            c = _clean(y)
            if np.isfinite(c) and c > 0:
                lv, sl = _fit(ag, y / c, 30)
                f["sed_level30"], f["sed_slope30"] = lv, sl
                f["sed_slope14"] = _fit(ag, y / c, 14)[1]
                f["sed_growth_per_L"] = (lv - 1.0) / max(vol, 1.0) if np.isfinite(lv) else np.nan
                if np.isfinite(sl) and sl > 1e-5:
                    f["sed_trend_days"] = max(0.0, (SEDIMENT_DP_RATIO_LIMIT - lv) / sl)
                lvL, slL = _fit(volv, y / c, 30)
                f["sed_slope_L"] = slL
                if np.isfinite(slL) and slL > 1e-9:
                    f["sed_trend_days_vol"] = max(0.0, (SEDIMENT_DP_RATIO_LIMIT - lvL) / (slL * rate))
        elif stage == "ro":
            fl = flow / (1.0 + TEMP_CORR_PER_C * (temp - 25.0))
            c = _clean(fl)
            cands, candsL = [], []
            if np.isfinite(c) and c > 0:
                lv, sl = _fit(ag, fl / c, 30)
                f["flux_level30"], f["flux_slope30"] = lv, sl
                if np.isfinite(sl) and sl < -1e-6:
                    cands.append((RO_FLUX_LIMIT - lv) / sl)
                lvL, slL = _fit(volv, fl / c, 30)
                f["flux_slope_L"] = slL
                if np.isfinite(slL) and slL < -1e-10:
                    candsL.append((RO_FLUX_LIMIT - lvL) / (slL * rate))
            rej = 1.0 - out / feed
            lv, sl = _fit(ag, rej, 30)
            f["rej_level30"], f["rej_slope30"] = lv, sl
            if np.isfinite(sl) and sl < -1e-7:
                cands.append((RO_REJECTION_LIMIT - lv) / sl)
            lvL, slL = _fit(volv, rej, 30)
            f["rej_slope_L"] = slL
            if np.isfinite(slL) and slL < -1e-11:
                candsL.append((RO_REJECTION_LIMIT - lvL) / (slL * rate))
            fe = feed.copy()
            m = ~np.isfinite(fe)
            if m.all():
                fe[:] = 350.0
            elif m.any():
                idx = np.where(~m, np.arange(len(fe)), 0)
                np.maximum.accumulate(idx, out=idx)
                fe = fe[idx]
                fe[np.isnan(fe)] = np.nanmean(fe)
            load = float(np.sum(usage * (fe / 350.0) ** 0.7))
            f["ro_load"] = load
            if np.isfinite(f["flux_level30"]):
                f["flux_loss_per_load"] = (1.0 - f["flux_level30"]) / max(load, 1.0)
            cands = [x for x in cands if np.isfinite(x)]
            if cands:
                f["ro_trend_days"] = max(0.0, min(cands))
            candsL = [x for x in candsL if np.isfinite(x)]
            if candsL:
                f["ro_trend_days_vol"] = max(0.0, min(candsL))
    if use_prev:
        rep = np.flatnonzero(hist[repl_col(stage)].to_numpy())
        rep = rep[rep > 0]
        f["n_prev"] = float(len(rep))
        if len(rep):
            pv = hist[vol_col(stage)].to_numpy()[rep - 1]
            pa = hist[age_col(stage)].to_numpy()[rep - 1]
            f.update(prev_vol_last=float(pv[-1]), prev_vol_mean=float(pv.mean()), prev_age_last=float(pa[-1]))
    return f


def vector(f, names):
    return np.array([f[k] for k in names], dtype=float)
