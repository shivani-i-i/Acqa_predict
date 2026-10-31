"""Local, batch drift monitoring helpers (NOT a real-time monitoring service).

Three separate questions - do not conflate them:
  1. input_drift          : have INPUT distributions moved vs a reference? (needs no labels)
  2. prediction_error     : is forecast ERROR worse? (needs observed remaining_days)
  3. interval_coverage    : do intervals still cover outcomes? (needs observed outcomes)
Input drift is a warning sign, not proof of worse predictions; only 2 and 3 measure
actual predictive-performance degradation. PSI thresholds (0.1 / 0.25) are
conventional rules of thumb, not validated for this application.
"""
from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats


def psi(reference: Sequence[float], current: Sequence[float], bins: int = 10) -> float:
    ref = pd.Series(reference, dtype=float).dropna().to_numpy()
    cur = pd.Series(current, dtype=float).dropna().to_numpy()
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0 if np.allclose(ref.mean(), cur.mean()) else float("inf")
    edges[0], edges[-1] = -np.inf, np.inf
    eps = 1e-4
    p = np.histogram(ref, edges)[0] / len(ref) + eps
    q = np.histogram(cur, edges)[0] / len(cur) + eps
    return float(np.sum((q - p) * np.log(q / p)))


def input_drift(reference: pd.DataFrame, current: pd.DataFrame, features: Optional[Iterable[str]] = None,
                psi_threshold: float = 0.2, min_samples: int = 30) -> Dict[str, object]:
    feats = list(features) if features is not None else [
        c for c in reference.columns if c in current.columns and pd.api.types.is_numeric_dtype(reference[c])]
    result: Dict[str, object] = {"kind": "input_drift", "features": {}, "drifted_features": [],
                                 "any_drift": False, "status": "ok",
                                 "note": "Input drift is a warning sign only; it does not prove worse predictions."}
    for f in feats:
        r, c = reference[f].dropna(), current[f].dropna()
        if len(r) < min_samples or len(c) < min_samples:
            result["features"][f] = {"status": "insufficient_data", "n_reference": int(len(r)), "n_current": int(len(c))}
            continue
        ks = stats.ks_2samp(r, c)
        val = psi(r, c)
        drifted = bool(val >= psi_threshold)
        result["features"][f] = {"status": "ok", "psi": val, "ks_statistic": float(ks.statistic),
                                 "ks_pvalue": float(ks.pvalue), "drifted": drifted}
        if drifted:
            result["drifted_features"].append(f)
    result["any_drift"] = bool(result["drifted_features"])
    if not any(v.get("status") == "ok" for v in result["features"].values()):
        result["status"] = "insufficient_data"
    return result


def prediction_error(y_true, y_pred, baseline_mae: Optional[float] = None,
                     degradation_ratio: float = 1.5, min_samples: int = 30) -> Dict[str, object]:
    if y_true is None or y_pred is None:
        return {"kind": "prediction_error", "status": "no_observed_targets",
                "note": "Observed outcomes are not available yet; error cannot be monitored."}
    y, p = np.asarray(y_true, float), np.asarray(y_pred, float)
    ok = np.isfinite(y) & np.isfinite(p)
    if ok.sum() < min_samples:
        return {"kind": "prediction_error", "status": "insufficient_data", "n": int(ok.sum())}
    err = np.abs(y[ok] - p[ok])
    out = {"kind": "prediction_error", "status": "ok", "n": int(ok.sum()), "mae_days": float(err.mean()),
           "median_ae_days": float(np.median(err)), "baseline_mae_days": baseline_mae, "degraded": None}
    if baseline_mae is not None:
        out["degraded"] = bool(err.mean() > degradation_ratio * baseline_mae)
    return out


def interval_coverage(y_true, lower, upper, target: float = 0.90, tolerance: float = 0.05,
                      min_samples: int = 30) -> Dict[str, object]:
    if y_true is None or lower is None or upper is None:
        return {"kind": "interval_coverage", "status": "no_observed_targets"}
    y, lo, hi = (np.asarray(a, dtype=float) for a in (y_true, lower, upper))  # None -> NaN (withheld)
    has_interval = np.isfinite(lo) & np.isfinite(hi) & np.isfinite(y)
    n = int(has_interval.sum())
    out = {"kind": "interval_coverage", "n_with_interval": n, "n_withheld": int(len(y) - n)}
    if n < min_samples:
        return {**out, "status": "insufficient_data"}
    cov = float(((y >= lo) & (y <= hi))[has_interval].mean())
    return {**out, "status": "ok", "empirical_coverage": cov, "target_coverage": target,
            "undercovering": bool(cov < target - tolerance)}


def monitoring_report(input_result=None, error_result=None, coverage_result=None) -> Dict[str, object]:
    """Combine results, keeping input drift and real performance degradation separate."""
    perf_bad = bool((error_result or {}).get("degraded")) or bool((coverage_result or {}).get("undercovering"))
    return {"input_drift_detected": bool((input_result or {}).get("any_drift")),
            "performance_degradation_confirmed": perf_bad,
            "performance_evidence_available": any((r or {}).get("status") == "ok"
                                                  for r in (error_result, coverage_result)),
            "note": "Local batch prototype; no real-time monitoring service exists."}
