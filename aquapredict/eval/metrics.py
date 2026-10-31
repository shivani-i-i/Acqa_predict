"""Metric definitions (exact formulas in each docstring).

Opportunity = one forecast for one filter at one day. Cycle = one filter life that reached its criterion.
  window error      = point - true_days_to_criterion  (we report |error|, median)  [days]
  primary subset    = opportunities with true remaining life <= cfg.primary_horizon_days
  coverage          = share of ISSUED opportunities with lo <= true <= hi
  width             = hi - lo  [days]
  alert day (cycle) = first checked day with alert_value <= cfg.lead_target_days (withheld never alerts)
  lead              = crossing_day - alert_day
  LATE warning      = no alert at all, or lead < cfg.min_notice_days
  PREMATURE alarm   = alert exists and lead > cfg.premature_days   (this is the 'false alarm')
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from .config import EvalConfig

KEYS = ["method", "household_id", "stage", "cycle"]


def issued(opps):
    return opps[(~opps["withheld"]) & np.isfinite(opps["point"])]


def _groups(df, by):
    by = list(by)
    return df.groupby(by, sort=True) if by else [((), df)]


def _rows(df, by, fn):
    out = []
    for key, g in _groups(df, by):
        key = key if isinstance(key, tuple) else (key,)
        out.append({**dict(zip(by, key)), **fn(g)})
    return pd.DataFrame(out).set_index(list(by)) if by else pd.DataFrame(out)


def window_summary(opps, cfg: EvalConfig, by=("method",)):
    iss = issued(opps).copy()
    iss["err"] = iss["point"] - iss["true_dtc"]
    iss["abs_err"] = iss["err"].abs()
    iss["primary"] = iss["true_dtc"] <= cfg.primary_horizon_days
    wh = opps.groupby(list(by))["withheld"].mean() if by else None

    def fn(g):
        p = g[g["primary"]]
        return {"n_issued": len(g), "n_primary": len(p),
                "median_abs_err_primary": p["abs_err"].median(),
                "mean_abs_err_primary": p["abs_err"].mean(),
                "p90_abs_err_primary": p["abs_err"].quantile(0.9) if len(p) else np.nan,
                "median_bias_primary": p["err"].median(),
                "median_abs_err_all": g["abs_err"].median()}
    t = _rows(iss, by, fn)
    if by:
        t["withheld_rate"] = wh.reindex(t.index).values
    return t


def interval_summary(opps, cfg: EvalConfig, by=("method",)):
    iss = issued(opps)
    iss = iss[np.isfinite(iss["lo"]) & np.isfinite(iss["hi"])].copy()
    if iss.empty:
        return pd.DataFrame()
    iss["covered"] = (iss["true_dtc"] >= iss["lo"]) & (iss["true_dtc"] <= iss["hi"])
    iss["width"] = iss["hi"] - iss["lo"]
    iss["primary"] = iss["true_dtc"] <= cfg.primary_horizon_days

    def fn(g):
        p = g[g["primary"]]
        return {"n_with_interval": len(g), "coverage": g["covered"].mean(),
                "coverage_primary": p["covered"].mean() if len(p) else np.nan,
                "median_width_all": g["width"].median(),
                "median_width_primary": p["width"].median() if len(p) else np.nan}
    return _rows(iss, by, fn)


def cycle_outcomes(opps, cfg: EvalConfig, lead_target=None):
    """One row per (method, household, stage, cycle): alert day, lead, late, premature."""
    L = cfg.lead_target_days if lead_target is None else lead_target
    base = opps.groupby(KEYS).agg(scenario=("scenario", "first"), cycle_start=("cycle_start", "first"),
                                  crossing=("crossing", "first")).reset_index()
    cand = opps[(~opps["withheld"]) & (opps["alert_value"] <= L)]
    first = cand.groupby(KEYS)["day"].min().rename("alert_day").reset_index()
    out = base.merge(first, on=KEYS, how="left")
    out["lead"] = out["crossing"] - out["alert_day"]
    has = out["alert_day"].notna()
    out["late"] = (~has) | (out["lead"] < cfg.min_notice_days)
    out["premature"] = has & (out["lead"] > cfg.premature_days)
    return out


def late_false_summary(cycles, by=("method",)):
    return _rows(cycles, by, lambda g: {
        "n_cycles": len(g), "late_rate": g["late"].mean(), "premature_rate": g["premature"].mean(),
        "never_alerted_rate": g["alert_day"].isna().mean(), "median_lead_days": g["lead"].median()})


def operating_curve(opps, cfg: EvalConfig, grid=None):
    """Late vs premature rate as the alert threshold L is swept: lets us compare at matched false-alarm rate."""
    grid = np.arange(0, 91, 3) if grid is None else grid
    rows = []
    for L in grid:
        s = late_false_summary(cycle_outcomes(opps, cfg, lead_target=float(L)))
        for m, r in s.iterrows():
            rows.append({"method": m, "lead_target": float(L), "late_rate": r["late_rate"],
                         "premature_rate": r["premature_rate"]})
    return pd.DataFrame(rows)


def bootstrap_median_diff(opps, model, baseline, cfg: EvalConfig, n_boot=300, seed=0):
    """Paired household-bootstrap CI for median|err|(model) - median|err|(baseline), primary subset.
    Negative = model better."""
    iss = issued(opps)
    iss = iss[iss["true_dtc"] <= cfg.primary_horizon_days]
    key = ["household_id", "stage", "cycle", "day"]
    a = iss[iss.method == model].set_index(key)
    b = iss[iss.method == baseline].set_index(key)
    j = a[["point", "true_dtc"]].join(b[["point"]].rename(columns={"point": "point_b"}), how="inner").reset_index()
    if j.empty:
        return {"n": 0}
    j["ea"] = (j["point"] - j["true_dtc"]).abs()
    j["eb"] = (j["point_b"] - j["true_dtc"]).abs()
    groups = [g[["ea", "eb"]].to_numpy() for _, g in j.groupby("household_id")]
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        pick = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        diffs.append(np.median(pick[:, 0]) - np.median(pick[:, 1]))
    return {"n": len(j), "diff": float(np.median(j["ea"]) - np.median(j["eb"])),
            "ci95": (float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5)))}


def household_bootstrap_ci(df, value_col, by=("method",), n_boot=500, seed=0, stat="mean"):
    """95% CI of a per-row statistic, resampling WHOLE HOUSEHOLDS (rows of one household are not independent).
    stat: 'mean' (rates) or 'median' (errors). Returns DataFrame with n, estimate, lo, hi."""
    fn = np.mean if stat == "mean" else np.median
    rng = np.random.default_rng(seed)
    out = []
    for key, g in _groups(df, by):
        key = key if isinstance(key, tuple) else (key,)
        groups = [v.to_numpy(float) for _, v in g.groupby("household_id")[value_col]]
        est = fn(np.concatenate(groups))
        boots = [fn(np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))]))
                 for _ in range(n_boot)]
        out.append({**dict(zip(by, key)), "n_rows": int(sum(len(x) for x in groups)),
                    "n_households": len(groups), "estimate": float(est),
                    "ci_lo": float(np.percentile(boots, 2.5)), "ci_hi": float(np.percentile(boots, 97.5))})
    return pd.DataFrame(out)


def fallback_summary(opps, by=("method", "stage")):
    """Share of forecasts where a method fell back to another rule (e.g. trend -> volume rule)."""
    return opps.groupby(list(by))["fallback"].mean().rename("fallback_rate").to_frame()
