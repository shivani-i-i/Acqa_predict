"""Public interface: predict(sensor_data: dict) -> dict

Input  (sensor_data)
--------------------
{
  "filter_stage": "sediment",            # required: sediment | carbon | ro_membrane (as trained)
  "household_id": "H001",                # optional label
  "history": [ {...reading...}, ... ],   # required, oldest -> newest, ONE household + ONE filter stage,
                                         #   starting when the current filter was installed
  "now": "2026-03-01T08:00:00",          # optional; enables the stale-reading check
  "require_interval": false,             # optional policy flag; true => withhold the forecast
                                         #   whenever a justified interval cannot be issued
  "model_path": "path/to/bundle.joblib", # optional; default artifacts/aquapredict_bundle.joblib
  "explain": true                        # optional; false skips the (slow) SHAP step, e.g. in bulk evaluation
}
Reading keys: timestamp, usage_litres (required); tds_in, tds_out, pressure, flow_rate,
temperature, purification_mode (optional - absent/None is fine).

Output - ALWAYS the same keys (see README "Response schema"):
  filter_stage, household_id, forecast_days, interval_days, drivers, explanation,
  fault_events, source_change, status, reason, disclaimer

status: "ok" | "degraded" | "withheld"   (meanings in checks/gating.py and README)
predict() never raises for bad sensor data; it returns status "withheld" with a reason.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd

from aquapredict.forecasting.checks.gating import (GatingConfig, STATUS_DEGRADED, STATUS_OK, STATUS_WITHHELD,
                           gate_inputs, gate_interval, gate_model_support)
from aquapredict.forecasting.checks.sensor_faults import SensorFaultConfig, detect_sensor_faults
from aquapredict.forecasting.checks.source_change import CusumConfig, detect_shift
from aquapredict.forecasting.features.engineering import build_features
from aquapredict.forecasting.models.explain import explain_prediction
from aquapredict.forecasting.models.persistence import DEFAULT_ARTIFACT, ArtifactIncompatibleError, ArtifactNotFoundError, load_bundle

DISCLAIMER = ("Prototype trained on SYNTHETIC data; not validated for real purifier maintenance decisions.")
EXPLANATION_KEYS = {"available": False, "method": None, "summary": None, "reason": None,
                    "base_value_days": None, "additivity_gap_days": None}
_CACHE: Dict[Tuple[str, float], tuple] = {}


def reset_cache() -> None:
    _CACHE.clear()


def _load(path: Optional[str]):
    p = Path(path) if path else DEFAULT_ARTIFACT
    key = (str(p), p.stat().st_mtime if p.exists() else -1.0)
    if key not in _CACHE:
        _CACHE.clear()
        _CACHE[key] = load_bundle(p)
    return _CACHE[key]


def _response(stage=None, household=None, forecast=None, interval=None, drivers=None, explanation=None,
              faults=None, source_change=None, status=STATUS_WITHHELD, reason=None) -> Dict[str, object]:
    return {
        "filter_stage": stage, "household_id": household,
        "forecast_days": None if forecast is None else round(float(forecast), 1),
        "interval_days": None if interval is None else [round(float(interval[0]), 1), round(float(interval[1]), 1)],
        "drivers": drivers or [],
        "explanation": {**EXPLANATION_KEYS, **(explanation or {
            "available": False, "reason": "no explanation (forecast withheld)"})},
        "fault_events": faults or [],
        "source_change": source_change or {"detected": False, "evaluated": False},
        "status": status, "reason": reason, "disclaimer": DISCLAIMER,
    }


def _source_change(df: pd.DataFrame) -> Dict[str, object]:
    cfg = CusumConfig()
    if "tds_in" not in df.columns or df["tds_in"].notna().sum() <= cfg.init_n:
        return {"detected": False, "evaluated": False,
                "message": f"Not enough tds_in history (need > {cfg.init_n} valid readings) to check for a source change."}
    res = detect_shift(pd.to_numeric(df["tds_in"], errors="coerce").to_numpy(), df["timestamp"].to_numpy(), cfg)
    keep = ("detected", "direction", "alarm_index", "change_index", "alarm_timestamp", "change_timestamp",
            "baseline_mean", "message")
    return {"evaluated": True, **{k: res[k] for k in keep}}


def predict(sensor_data: dict) -> dict:
    # ---- 1. input validation -------------------------------------------------
    if not isinstance(sensor_data, dict):
        return _response(reason="Forecast withheld - invalid input: sensor_data must be a dict.")
    stage = sensor_data.get("filter_stage")
    household = sensor_data.get("household_id")
    hist = sensor_data.get("history")
    if not isinstance(stage, str) or not stage:
        return _response(household=household, reason="Forecast withheld - invalid input: 'filter_stage' (str) is required.")
    if not isinstance(hist, (list, tuple)) or len(hist) == 0 or not all(isinstance(r, dict) for r in hist):
        return _response(stage, household, reason="Forecast withheld - invalid input: 'history' must be a non-empty list of reading dicts.")
    try:
        df = pd.DataFrame(list(hist))
        if "timestamp" not in df.columns:
            raise ValueError("readings need a 'timestamp'")
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="raise")
        df["household_id"] = str(household) if household is not None else "unknown"
        df["filter_stage"] = stage
        df = df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
        dup = df["timestamp"].duplicated(keep="last")   # re-sent readings must not be double counted
        n_dup = int(dup.sum())
        df = df[~dup].reset_index(drop=True)
    except Exception as exc:
        return _response(stage, household, reason=f"Forecast withheld - invalid input: {exc}")

    # ---- 2. sensor faults + input gate -------------------------------------------
    try:
        faults = detect_sensor_faults(df, SensorFaultConfig(), now=sensor_data.get("now"))
    except Exception as exc:
        return _response(stage, household, reason=f"Forecast withheld - sensor checks failed: {exc}")
    if n_dup:
        faults.append({"type": "duplicate_timestamp", "severity": "warning",
                       "message": f"{n_dup} reading(s) shared a timestamp with a later reading and were dropped "
                                  "(the last one was kept)", "timestamp": None, "sensor": None, "value": n_dup})
    gcfg = GatingConfig(require_interval=bool(sensor_data.get("require_interval", False)))
    gate = gate_inputs(len(df), faults, gcfg)
    source = _source_change(df)
    if not gate.allowed:
        return _response(stage, household, faults=faults, source_change=source, reason=gate.reason)

    # ---- 3. model artifact ----------------------------------------------------------
    try:
        model, calibrator = _load(sensor_data.get("model_path"))
    except (ArtifactNotFoundError, ArtifactIncompatibleError) as exc:
        return _response(stage, household, faults=faults, source_change=source,
                         reason=f"Forecast withheld - model unavailable: {exc}")

    # ---- 4. features, model-support gate, point forecast ---------------------------------
    try:
        X = build_features(df)
    except Exception as exc:
        return _response(stage, household, faults=faults, source_change=source,
                         reason=f"Forecast withheld - feature engineering failed: {exc}")
    support = gate_model_support(model.out_of_range(X))
    if not support.allowed:
        return _response(stage, household, faults=faults, source_change=source, reason=support.reason)
    try:
        point = float(model.predict_features(X.iloc[[-1]])[0])
    except Exception as exc:
        return _response(stage, household, faults=faults, source_change=source,
                         reason=f"Forecast withheld - model could not make a valid prediction: {exc}")

    # ---- 5. interval gate (calibration-based; separate from model support) ----------------
    interval = calibrator.interval(point, X)
    igate = gate_interval(interval, gcfg)
    if not igate.allowed:
        return _response(stage, household, faults=faults, source_change=source, reason=igate.reason)

    # ---- 6. explanation (optional) ------------------------------------------------------
    if sensor_data.get("explain", True):
        expl = explain_prediction(model, X)
    else:
        expl = {"drivers": [], "available": False, "reason": "explanations disabled for this request (explain=false)"}
    return _response(stage, household, forecast=point, interval=interval["interval"],
                     drivers=expl["drivers"],
                     explanation={k: expl.get(k) for k in EXPLANATION_KEYS},
                     faults=faults, source_change=source, status=igate.status,
                     reason=None if igate.status == STATUS_OK else igate.reason)
