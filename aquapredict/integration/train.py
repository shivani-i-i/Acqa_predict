"""Train Person 2's forecaster + conformal calibrator on Person 1's simulator data, and wrap Person 2's public
predict() as a forecaster the Person 1 evaluation harness can score."""
from __future__ import annotations
import math
import os
from pathlib import Path
import numpy as np
import pandas as pd

from ..forecasting.models.forecast import ForecastConfig, ForecastModel, baseline_predictions, regression_metrics
from ..forecasting.models.intervals import ConformalCalibrator, ConformalConfig
from ..forecasting.models.persistence import save_bundle
from ..forecasting.predict import predict as p2_predict, reset_cache
from ..schema import STAGES
from .adapter import STAGE_TO_P2, history_to_p2, training_frame, training_frames_by_stage
from ..baselines.baselines import BASELINES

_BASELINE_ALIASES = {
    "health_threshold_proxy": "health_20pct",
    "trend_extrapolation": "trend",
}


class Person2Forecaster:
    """Callable f(hist, stage) -> {point, lo, hi, withheld, reason, ...} for aquapredict.eval.harness.

    Runs Person 2's FULL pipeline (validation, sensor faults, gating, CUSUM, features, forecast, interval gate).
    status 'ok'/'degraded' -> a forecast is issued ('degraded' = no interval); 'withheld' -> none.
    explain=False skips SHAP (much faster; drivers come back empty)."""

    def __init__(self, bundle_path, pressure_source="dp", explain=False, require_interval=False,
                 selected_models=None):
        # A string remains accepted for callers with an older pooled artifact. New
        # integration training supplies one standard artifact per stage.
        self.bundle_path = bundle_path
        self.bundle_paths = (dict(bundle_path) if isinstance(bundle_path, dict)
                             else {STAGE_TO_P2[s]: str(bundle_path) for s in STAGES})
        self.pressure_source = pressure_source
        self.explain, self.require_interval = explain, require_interval
        self.selected_models = dict(selected_models or {})

    def respond(self, hist: pd.DataFrame, stage: str) -> dict:
        """Person 2's complete response dict (drivers, fault events, source-change flag, status, reason)."""
        return p2_predict({
            "filter_stage": STAGE_TO_P2[stage], "household_id": str(hist["household_id"].iloc[-1]),
            "history": history_to_p2(hist, stage, self.pressure_source),
            "model_path": self.bundle_paths.get(STAGE_TO_P2[stage], self.bundle_path),
            "explain": self.explain, "require_interval": self.require_interval})

    def __call__(self, hist: pd.DataFrame, stage: str) -> dict:
        selected = self.selected_models.get(STAGE_TO_P2[stage], "gradient_boosting")
        baseline_name = _BASELINE_ALIASES.get(selected, selected)
        if baseline_name in BASELINES:
            # Selection changes only the harness forecast. respond() deliberately
            # remains Person 2's complete public response contract.
            r = BASELINES[baseline_name](hist, stage)
            return {"point": float(r["point"]), "lo": None, "hi": None, "withheld": False,
                    "reason": r.get("reason", ""), "status": "degraded", "method": selected,
                    "n_faults": 0, "source_change": False}
        r = self.respond(hist, stage)
        iv = r["interval_days"]
        return {"point": np.nan if r["forecast_days"] is None else float(r["forecast_days"]),
                "lo": None if iv is None else float(iv[0]), "hi": None if iv is None else float(iv[1]),
                "withheld": r["status"] == "withheld", "reason": r["reason"] or "", "status": r["status"],
                "method": "person2-gbm", "n_faults": len(r["fault_events"]),
                "source_change": bool(r["source_change"].get("detected"))}


def train_person2_on_person1(train_df, cal_df, out_path, seed=0, max_iter=200, pressure_source="dp",
                             development_df=None, select_model=False, dev_df=None, **forecaster_kw) -> Person2Forecaster:
    """Train and calibrate one model per stage on disjoint household splits.

    ``development_df`` is optional and is never used for fitting or calibration.
    When ``select_model`` is true, each stage is routed to the GBM or the best
    Person 2 baseline using only that development split.
    """
    if development_df is not None and dev_df is not None:
        raise ValueError("pass only one of development_df and dev_df")
    development_df = development_df if development_df is not None else dev_df
    split_frames = [train_df, cal_df] + ([] if development_df is None else [development_df])
    ids = [set(map(str, f.household_id.unique())) for f in split_frames]
    if any(ids[i] & ids[j] for i in range(len(ids)) for j in range(i)):
        raise ValueError("training and calibration households overlap")
    if select_model and development_df is None:
        raise ValueError("development_df is required when select_model=True")

    tr_frames = training_frames_by_stage(train_df, pressure_source=pressure_source)
    ca_frames = training_frames_by_stage(cal_df, pressure_source=pressure_source)
    dev_frames = (training_frames_by_stage(development_df, pressure_source=pressure_source)
                  if development_df is not None else {})
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    paths, selected, diagnostics = {}, {}, {}
    for stage in STAGES:
        p2_stage = STAGE_TO_P2[stage]
        model = ForecastModel(ForecastConfig(seed=seed, max_iter=max_iter)).fit(tr_frames[p2_stage])
        cal = ConformalCalibrator(ConformalConfig()).fit(model, ca_frames[p2_stage])
        path = out if len(STAGES) == 1 else out.with_name(f"{out.stem}.{stage}{out.suffix or '.joblib'}")
        paths[p2_stage] = str(save_bundle(model, cal, path))
        selected[p2_stage] = "gradient_boosting"
        diagnostics[p2_stage] = {"gradient_boosting": None}
        if select_model:
            dev = dev_frames[p2_stage]
            X = __import__("aquapredict.forecasting.features.engineering", fromlist=["build_features"]).build_features(dev)
            metrics = {"gradient_boosting": regression_metrics(dev["remaining_days"], model.predict_features(X))}
            for name, pred in baseline_predictions(X).items():
                metrics[name] = regression_metrics(dev["remaining_days"], pred)
            selected[p2_stage] = min(metrics, key=lambda n: metrics[n]["median_ae_days"])
            diagnostics[p2_stage] = metrics
    reset_cache()
    forecaster = Person2Forecaster(paths, pressure_source=pressure_source,
                                   selected_models=selected, **forecaster_kw)
    forecaster.selection_metrics = diagnostics
    return forecaster
