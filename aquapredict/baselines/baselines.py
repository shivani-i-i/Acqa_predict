"""The three baselines from the abstract. Each is a forecaster: f(hist, stage) -> dict.

hist = rows of one household up to the forecast day (VISIBLE columns only). Every baseline uses only
the CURRENT filter cycle (rows since the stage's last replacement), found from age_<stage>_d.

1. volume_rule         : replace at rated litres or rated days (fixed timer / litre count, Table 1).
2. health_20pct        : usage-based health = 1 - max(volume used / rated volume, age / rated days);
                         replacement advised when health falls to 20%.
3. trend_extrapolation : robust-ish linear trend of normalised indicators extrapolated to the
                         replacement criterion. Carbon has no sensor signal, so it (and any case where
                         the trend cannot be fitted) falls back to the volume rule; fallback is flagged.
Rated values are ILLUSTRATIVE (schema.DEFAULT_RATED).
"""
from __future__ import annotations
import numpy as np
from ..schema import (DEFAULT_RATED, SEDIMENT_DP_RATIO_LIMIT, RO_FLUX_LIMIT, RO_REJECTION_LIMIT,
                      age_col, vol_col)

TREND_WINDOW_DAYS = 30
TREND_MIN_POINTS = 8
TEMP_CORR_PER_C = 0.03          # the simple correction a practitioner would assume


def _out(point, **kw):
    return {"point": float(point), "lo": None, "hi": None, "withheld": False, "reason": "", **kw}


def _cycle(hist, stage):
    age = int(hist[age_col(stage)].iloc[-1])
    return hist.iloc[max(len(hist) - 1 - age, 0):]


def _rate(cyc, win=30):
    return max(float(cyc["usage_L"].iloc[-win:].mean()), 1.0)      # litres/day, planning rate


def volume_rule(hist, stage, rated=None):
    r = (rated or DEFAULT_RATED)[stage]
    cyc = _cycle(hist, stage)
    vol, age = float(cyc[vol_col(stage)].iloc[-1]), len(cyc) - 1
    days = min((r["volume_L"] - vol) / _rate(cyc), r["days"] - age)
    return _out(max(0.0, days))


def health_20pct(hist, stage, rated=None, threshold=0.20):
    r = (rated or DEFAULT_RATED)[stage]
    cyc = _cycle(hist, stage)
    vol, age = float(cyc[vol_col(stage)].iloc[-1]), len(cyc) - 1
    used = 1.0 - threshold
    days = min((used * r["volume_L"] - vol) / _rate(cyc), used * r["days"] - age)
    return _out(max(0.0, days))


def _clean_level(y, n0=10):
    v = y[:n0]
    v = v[np.isfinite(v)]
    return float(np.median(v[:7])) if len(v) >= 5 else np.nan


def _fit(age, y):
    a, v = age[-TREND_WINDOW_DAYS:], y[-TREND_WINDOW_DAYS:]
    m = np.isfinite(v)
    if m.sum() < TREND_MIN_POINTS or np.ptp(a[m]) == 0:
        return None
    x = a[m] - a[-1]
    slope, level_now = np.polyfit(x, v[m], 1)
    return float(level_now), float(slope)


def trend_extrapolation(hist, stage, rated=None):
    cyc = _cycle(hist, stage)
    age = cyc[age_col(stage)].to_numpy(float)
    cands = []
    with np.errstate(divide="ignore", invalid="ignore"):
        if stage == "sediment":
            y = cyc["dp_sediment_bar"].to_numpy(float) / cyc["flow_Lpm"].to_numpy(float)
            c = _clean_level(y)
            fit = _fit(age, y / c) if np.isfinite(c) and c > 0 else None
            if fit and fit[1] > 1e-5:
                cands.append((SEDIMENT_DP_RATIO_LIMIT - fit[0]) / fit[1])
        elif stage == "ro":
            T = cyc["temp_C"].to_numpy(float)
            flux = cyc["flow_Lpm"].to_numpy(float) / (1.0 + TEMP_CORR_PER_C * (T - 25.0))
            c = _clean_level(flux)
            fit = _fit(age, flux / c) if np.isfinite(c) and c > 0 else None
            if fit and fit[1] < -1e-6:
                cands.append((RO_FLUX_LIMIT - fit[0]) / fit[1])
            rej = 1.0 - cyc["out_tds_ppm"].to_numpy(float) / cyc["feed_tds_ppm"].to_numpy(float)
            fit = _fit(age, rej)
            if fit and fit[1] < -1e-7:
                cands.append((RO_REJECTION_LIMIT - fit[0]) / fit[1])
    cands = [c for c in cands if np.isfinite(c)]
    if cands:
        return _out(max(0.0, min(cands)), fallback=False)
    out = volume_rule(hist, stage, rated)
    out["fallback"] = True
    return out


BASELINES = {"volume_rule": volume_rule, "health_20pct": health_20pct, "trend": trend_extrapolation}
