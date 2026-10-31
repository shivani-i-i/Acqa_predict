"""Train Person 2's forecaster + conformal calibrator on Person 1's simulator data, and wrap Person 2's public
predict() as a forecaster the Person 1 evaluation harness can score."""
from __future__ import annotations
import math
import os
import numpy as np
import pandas as pd

from ..forecasting.models.forecast import ForecastConfig, ForecastModel
from ..forecasting.models.intervals import ConformalCalibrator, ConformalConfig
from ..forecasting.models.persistence import save_bundle
from ..forecasting.predict import predict as p2_predict, reset_cache
from .adapter import STAGE_TO_P2, history_to_p2, training_frame


class Person2Forecaster:
    """Callable f(hist, stage) -> {point, lo, hi, withheld, reason, ...} for aquapredict.eval.harness.

    Runs Person 2's FULL pipeline (validation, sensor faults, gating, CUSUM, features, forecast, interval gate).
    status 'ok'/'degraded' -> a forecast is issued ('degraded' = no interval); 'withheld' -> none.
    explain=False skips SHAP (much faster; drivers come back empty)."""

    def __init__(self, bundle_path, pressure_source="dp", explain=False, require_interval=False):
        self.bundle_path, self.pressure_source = str(bundle_path), pressure_source
        self.explain, self.require_interval = explain, require_interval

    def respond(self, hist: pd.DataFrame, stage: str) -> dict:
        """Person 2's complete response dict (drivers, fault events, source-change flag, status, reason)."""
        return p2_predict({
            "filter_stage": STAGE_TO_P2[stage], "household_id": str(hist["household_id"].iloc[-1]),
            "history": history_to_p2(hist, stage, self.pressure_source), "model_path": self.bundle_path,
            "explain": self.explain, "require_interval": self.require_interval})

    def __call__(self, hist: pd.DataFrame, stage: str) -> dict:
        r = self.respond(hist, stage)
        iv = r["interval_days"]
        return {"point": np.nan if r["forecast_days"] is None else float(r["forecast_days"]),
                "lo": None if iv is None else float(iv[0]), "hi": None if iv is None else float(iv[1]),
                "withheld": r["status"] == "withheld", "reason": r["reason"] or "", "status": r["status"],
                "method": "person2-gbm", "n_faults": len(r["fault_events"]),
                "source_change": bool(r["source_change"].get("detected"))}


def train_person2_on_person1(train_df, cal_df, out_path, seed=0, max_iter=200, pressure_source="dp",
                             **forecaster_kw) -> Person2Forecaster:
    """train_df / cal_df: Person 1 telemetry (with truth) of DISJOINT households."""
    if set(train_df.household_id) & set(cal_df.household_id):
        raise ValueError("training and calibration households overlap")
    tr = training_frame(train_df, pressure_source=pressure_source)
    ca = training_frame(cal_df, pressure_source=pressure_source)
    model = ForecastModel(ForecastConfig(seed=seed, max_iter=max_iter)).fit(tr)
    cal = ConformalCalibrator(ConformalConfig()).fit(model, ca)
    os.makedirs(os.path.dirname(str(out_path)) or ".", exist_ok=True)
    path = save_bundle(model, cal, out_path)
    reset_cache()
    return Person2Forecaster(path, pressure_source=pressure_source, **forecaster_kw)
