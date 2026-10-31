"""Optional SHAP explanations for single predictions.

Why several explainers are tried
--------------------------------
The estimator is scikit-learn's ``HistGradientBoostingRegressor``. SHAP's fast, exact
``TreeExplainer`` is documented for other tree libraries; for this estimator it may raise, or - worse -
accept the model but return values that do not reconstruct the prediction (HistGradientBoosting adds a
constant baseline outside its trees and routes missing values / categorical splits in its own way).
So every explainer is run and VALIDATED separately, and the first one that passes is used:

1. ``shap.TreeExplainer``        (fast; used only if its output passes validation)
2. ``shap.PermutationExplainer`` (model-agnostic SHAP estimator on the estimator's predict function)
3. ``shap.KernelExplainer``      (model-agnostic; enforces the additivity constraint by construction)

Validation (never skipped, never loosened silently): the explainer must return a finite contribution per
feature AND a base value, and ``base + sum(all contributions)`` must equal the estimator's raw output within
``ADDITIVITY_TOL_DAYS`` + 1 %. A result without a base value cannot be validated and is rejected.
If none of the explainers passes, or shap is not installed, ``available`` is False with a reason that lists
what each explainer did; no explanation is invented and the forecast itself is never affected. No other method
is ever labelled "SHAP", and ``method`` names the explainer that really produced the values.

Contributions are in days, relative to the model's average output on the stored background sample
(``base_value_days``), before the final clipping of the forecast at zero. They describe how the MODEL behaves;
they are associations, not causal effects. Feature names are the names of the model's input columns
(``filter_stage``, ``cum_usage_litres``, ...).
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from aquapredict.forecasting.models.forecast import ForecastModel

ADDITIVITY_TOL_DAYS = 0.5
N_KERNEL_BACKGROUND = 40
KERNEL_NSAMPLES = 400


def _import_shap():
    """Return the shap module or None. Separate function so tests can simulate absence."""
    try:
        import shap  # optional dependency
        return shap
    except ImportError:
        return None


def additivity_gap(base: Optional[float], values: np.ndarray, raw_pred: float) -> Optional[float]:
    """|base + sum(values) - raw_pred| in days, or None if there is no base value."""
    if base is None:
        return None
    return float(abs(base + float(np.sum(values)) - raw_pred))


def additivity_tolerance(raw_pred: float) -> float:
    return ADDITIVITY_TOL_DAYS + 0.01 * abs(raw_pred)


def _builders(shap, model: ForecastModel, transformed: pd.DataFrame) -> List[Tuple[str, Callable[[], Tuple[np.ndarray, Optional[float]]]]]:
    est = model.pipeline.named_steps["model"]
    cols = list(transformed.columns)
    x = transformed[cols].to_numpy(dtype=float)
    bg = model.background

    def predict_fn(arr):
        return est.predict(pd.DataFrame(np.asarray(arr, dtype=float), columns=cols))

    def need_bg():
        if bg is None or len(bg) == 0:
            raise RuntimeError("no background sample stored in the artifact (retrain with the current code)")
        return bg[cols].to_numpy(dtype=float)

    def tree():
        ex = shap.TreeExplainer(est)
        vals = np.asarray(ex.shap_values(transformed))
        base = getattr(ex, "expected_value", None)
        return vals.reshape(vals.shape[0], -1)[0], (None if base is None else float(np.ravel(base)[0]))

    def permutation():
        data = need_bg()
        ex = shap.PermutationExplainer(predict_fn, shap.maskers.Independent(data, max_samples=len(data)), seed=0)
        exp = ex(x, max_evals=max(500, 4 * len(cols) + 1), silent=True)
        return np.asarray(exp.values).reshape(1, -1)[0], float(np.ravel(exp.base_values)[0])

    def kernel():
        data = need_bg()[:N_KERNEL_BACKGROUND]
        ex = shap.KernelExplainer(predict_fn, data)
        vals = np.asarray(ex.shap_values(x, nsamples=KERNEL_NSAMPLES, silent=True))
        return vals.reshape(vals.shape[0], -1)[0], float(np.ravel(ex.expected_value)[0])

    return [("shap.TreeExplainer", tree), ("shap.PermutationExplainer", permutation), ("shap.KernelExplainer", kernel)]


def run_explainers(shap, model: ForecastModel, transformed: pd.DataFrame):
    """Try each explainer; return (accepted_or_None, attempts). ``accepted`` = (values, method, base, gap).

    ``attempts`` records, for every explainer tried, whether it was accepted and why not otherwise.
    """
    raw_pred = float(model.pipeline.named_steps["model"].predict(transformed)[0])
    attempts: List[Dict[str, object]] = []
    for name, build in _builders(shap, model, transformed):
        try:
            values, base = build()
        except Exception as exc:
            attempts.append({"method": name, "accepted": False, "problem": f"{type(exc).__name__}: {exc}"})
            continue
        values = np.asarray(values, dtype=float)
        if values.shape != (transformed.shape[1],) or not np.isfinite(values).all():
            attempts.append({"method": name, "accepted": False,
                             "problem": f"unexpected or non-finite output (shape {values.shape})"})
            continue
        gap = additivity_gap(base, values, raw_pred)
        if gap is None:
            attempts.append({"method": name, "accepted": False, "problem": "no base value returned; cannot validate"})
            continue
        if not np.isfinite(gap) or gap > additivity_tolerance(raw_pred):
            attempts.append({"method": name, "accepted": False, "additivity_gap_days": gap,
                             "problem": f"additivity gap {gap:.2f} days exceeds tolerance "
                                        f"{additivity_tolerance(raw_pred):.2f}"})
            continue
        attempts.append({"method": name, "accepted": True, "additivity_gap_days": gap})
        return (values, name, base, gap), attempts
    return None, attempts


def explain_prediction(model: ForecastModel, X: pd.DataFrame, row: int = -1, top_k: Optional[int] = 5) -> Dict[str, object]:
    """Explain one row. ``top_k=None`` returns every feature (used to check reconstruction in tests)."""
    unavailable = {"available": False, "method": None, "drivers": [], "summary": None, "attempts": []}
    shap = _import_shap()
    if shap is None:
        return {**unavailable, "reason": "shap is not installed (optional); explanations unavailable"}
    try:
        x_row = X[model.feature_columns].iloc[[row]]
        transformed = model.pipeline.named_steps["prep"].transform(x_row)
        accepted, attempts = run_explainers(shap, model, transformed)
    except Exception as exc:  # unexpected failure outside the explainers themselves
        return {**unavailable, "reason": f"SHAP explanation failed: {type(exc).__name__}: {exc}"}
    if accepted is None:
        why = "; ".join(f"{a['method']}: {a['problem']}" for a in attempts)
        return {**unavailable, "attempts": attempts,
                "reason": f"SHAP explanation failed - no explainer produced values that reconstruct the prediction ({why})"}
    values, method, base, gap = accepted
    order = sorted(zip(transformed.columns, values), key=lambda t: abs(t[1]), reverse=True)
    if top_k is not None:
        order = order[:top_k]
    raw = x_row.iloc[0]
    drivers = []
    for f, v in order:
        if abs(v) <= 1e-9:
            continue
        val = raw[f]
        val = val if isinstance(val, str) else (None if pd.isna(val) else float(val))
        drivers.append({"feature": f, "value": val, "contribution_days": float(v)})
    parts = [f"{d['feature']} {'raises' if d['contribution_days'] > 0 else 'lowers'} the forecast by "
             f"{abs(d['contribution_days']):.1f} days" for d in drivers[:3]]
    summary = (f"Leading drivers ({method}, relative to the model's average prediction): " + "; ".join(parts) +
               ". These are model associations, not proof of cause.") if parts else "No notable drivers."
    return {"available": True, "method": method, "drivers": drivers, "summary": summary, "reason": None,
            "base_value_days": base, "additivity_gap_days": gap, "attempts": attempts}
