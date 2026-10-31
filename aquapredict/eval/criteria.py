"""The abstract's three success criteria, checked honestly (PASS / FAIL with numbers)."""
from __future__ import annotations
import numpy as np
from .config import EvalConfig
from . import metrics as M

BASELINE_NAMES = ("volume_rule", "health_20pct", "trend")


def check_success(opps, cfg: EvalConfig, model="AquaPredict", baselines=BASELINE_NAMES):
    methods = set(opps["method"])
    if model not in methods:
        raise KeyError(f"model '{model}' not in results (have {sorted(methods)})")
    out = {}
    w = M.window_summary(opps, cfg)
    # PAIRED: a model that withholds its hard cases must not win by being scored only on easy ones, so every
    # baseline is re-scored on exactly the opportunities where the model issued a forecast.
    key = ["household_id", "stage", "cycle", "day"]
    iss = M.issued(opps)
    iss = iss[iss["true_dtc"] <= cfg.primary_horizon_days].copy()
    iss["abs_err"] = (iss["point"] - iss["true_dtc"]).abs()
    mk = iss[iss.method == model][key]
    mm = float(iss[iss.method == model]["abs_err"].median())
    comp = {}
    for b in baselines:
        if b in methods:
            comp[b] = float(iss[iss.method == b].merge(mk, on=key)["abs_err"].median())
    unpaired = {b: float(w.loc[b, "median_abs_err_primary"]) for b in comp}
    boots = {b: M.bootstrap_median_diff(opps, model, b, cfg) for b in comp}
    out["1_window_error"] = {
        "passed": bool(comp) and all(mm < v for v in comp.values()),
        "model_median_abs_err": mm, "baselines": comp, "baselines_unpaired": unpaired,
        "model_withheld_rate": float(w.loc[model, "withheld_rate"]), "bootstrap_diff_vs_baseline": boots,
        "rule": "model median |window error| < each baseline re-scored on the SAME opportunities (paired), primary horizon"}

    iv = M.interval_summary(opps, cfg)
    if model in getattr(iv, "index", []):
        cov, wid = float(iv.loc[model, "coverage"]), float(iv.loc[model, "median_width_primary"])
        ok = abs(cov - cfg.target_coverage) <= cfg.coverage_tol and wid <= cfg.max_median_width_days
        out["2_interval_coverage"] = {"passed": bool(ok), "coverage": cov, "median_width_primary_days": wid,
                                      "rule": f"|coverage-{cfg.target_coverage}|<={cfg.coverage_tol} and median width<={cfg.max_median_width_days}d"}
    else:
        out["2_interval_coverage"] = {"passed": False, "note": "model issued no intervals"}

    base_s = M.late_false_summary(M.cycle_outcomes(opps, cfg))
    curve = M.operating_curve(opps, cfg)
    if "volume_rule" in base_s.index and model in base_s.index:
        v_late, v_fa = float(base_s.loc["volume_rule", "late_rate"]), float(base_s.loc["volume_rule", "premature_rate"])
        m_late, m_fa = float(base_s.loc[model, "late_rate"]), float(base_s.loc[model, "premature_rate"])
        mc = curve[(curve.method == model) & (curve.premature_rate <= v_fa + cfg.false_alarm_tol)]
        best = float(mc["late_rate"].min()) if len(mc) else np.nan
        # PASS/FAIL uses the PRE-SET alert threshold (no threshold picked on the test data).
        out["3_late_warnings"] = {
            "passed": bool(m_late < v_late and m_fa <= v_fa + cfg.false_alarm_tol),
            "volume_rule_late": v_late, "volume_rule_premature": v_fa,
            "model_late": m_late, "model_premature": m_fa,
            "supplementary_model_late_at_matched_false_alarm_best_threshold": best,
            "rule": "at the pre-set threshold: model late rate < volume rule AND model premature rate <= volume's + tolerance"}
    out["all_passed"] = all(v["passed"] for k, v in out.items() if k[0].isdigit())
    return out
