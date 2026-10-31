"""Configurable, rule-based sensor fault detection for ONE reading stream.

Severities
----------
"info"     : worth knowing (e.g. an optional sensor is simply not fitted).
"warning"  : suspicious; forecast may still be issued.
"critical" : inputs are unreliable; the gating module BLOCKS the forecast.

IMPORTANT: every threshold in SensorFaultConfig is a PLACEHOLDER chosen for the
synthetic demo. None are validated against real purifier hardware and each must
be tuned/validated with actual equipment data before real use.
A repeated reading is NOT automatically a fault: a flat-line is only a warning
unless it lasts very long (``flatline_critical_window``).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

SENSORS = ["usage_litres", "tds_in", "tds_out", "pressure", "flow_rate", "temperature"]


@dataclass
class SensorFaultConfig:
    required_sensors: Tuple[str, ...] = ("usage_litres",)
    # missing-rate: share of missing readings over the last `missing_window` readings
    missing_window: int = 14
    missing_rate_warn: float = 0.5
    # staleness (needs `now`): gap between latest reading and now
    stale_after_hours: float = 48.0
    # flat-line (exactly constant readings)
    flatline_sensors: Tuple[str, ...] = ("tds_in", "tds_out", "pressure", "flow_rate")
    flatline_window: int = 10
    flatline_critical_window: int = 40
    # spikes: robust z-score vs trailing median/MAD
    spike_sensors: Tuple[str, ...] = ("tds_in", "tds_out", "pressure", "flow_rate", "temperature")
    spike_z: float = 6.0
    spike_baseline_window: int = 30
    spike_min_baseline: int = 10
    spike_scale_floor_rel: float = 0.02
    spike_lookback: int = 3
    # TDS out > TDS in
    tds_rel_tolerance: float = 0.10
    tds_abs_tolerance_ppm: float = 10.0
    tds_inconsistency_lookback: int = 5
    tds_inconsistency_critical_count: int = 3
    # pressure / flow consistency
    pressure_near_zero_bar: float = 0.2
    flow_min_when_no_pressure_lpm: float = 0.3
    # plausibility bounds: (min, max); values outside are "invalid_value" (critical)
    plausible_bounds: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        "usage_litres": (0.0, 2000.0), "tds_in": (0.0, 5000.0), "tds_out": (0.0, 5000.0),
        "pressure": (0.0, 12.0), "flow_rate": (0.0, 30.0), "temperature": (-5.0, 80.0)})


def _ts(df: pd.DataFrame, pos: int) -> Optional[str]:
    if "timestamp" in df.columns:
        return pd.Timestamp(df["timestamp"].iloc[pos]).isoformat()
    return None


def _event(kind, severity, message, df, pos, sensor=None, value=None) -> Dict[str, object]:
    v = None if value is None or (isinstance(value, float) and np.isnan(value)) else float(value)
    return {"type": kind, "severity": severity, "message": message,
            "timestamp": _ts(df, pos) if pos is not None else None, "sensor": sensor, "value": v}


def detect_sensor_faults(history: pd.DataFrame, config: Optional[SensorFaultConfig] = None,
                         now=None) -> List[Dict[str, object]]:
    """Return a list of fault/warning event dicts for the latest readings of one stream."""
    cfg = config or SensorFaultConfig()
    if history is None or len(history) == 0:
        return [{"type": "no_data", "severity": "critical", "message": "No sensor readings supplied.",
                 "timestamp": None, "sensor": None, "value": None}]
    df = history.copy()
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.sort_values("timestamp", kind="mergesort")
    df = df.reset_index(drop=True)
    last = len(df) - 1
    events: List[Dict[str, object]] = []

    # ---- missing / unavailable -------------------------------------------------
    for s in SENSORS:
        required = s in cfg.required_sensors
        col = pd.to_numeric(df[s], errors="coerce") if s in df.columns else pd.Series(np.nan, index=df.index)
        if col.isna().all():
            events.append(_event("sensor_unavailable", "critical" if required else "info",
                                 f"{s}: no readings at all" + (" (REQUIRED sensor)" if required else
                                 " (sensor may not be fitted; forecast continues without it)"), df, None, s))
        elif np.isnan(col.iloc[last]):
            events.append(_event("missing_reading", "critical" if required else "warning",
                                 f"{s}: latest reading is missing" + (" (REQUIRED sensor)" if required else ""),
                                 df, last, s))

    # ---- high missing rate over a window (sensor has reported before) -------------------
    if len(df) >= cfg.missing_window:
        for s in SENSORS:
            if s not in df.columns:
                continue
            col = pd.to_numeric(df[s], errors="coerce")
            rate = float(col.iloc[-cfg.missing_window:].isna().mean())
            if col.notna().any() and rate >= cfg.missing_rate_warn:
                events.append(_event("high_missing_rate", "warning",
                                     f"{s}: {rate:.0%} of the last {cfg.missing_window} readings are missing",
                                     df, last, s))

    # ---- stale ----------------------------------------------------------------
    if now is not None and "timestamp" in df.columns and pd.notna(df["timestamp"].iloc[last]):
        now_ts, last_ts = pd.Timestamp(now), pd.Timestamp(df["timestamp"].iloc[last])
        if (now_ts.tzinfo is None) != (last_ts.tzinfo is None):
            # mixed naive / tz-aware: convert the aware one to UTC and treat naive times as UTC
            now_ts = now_ts.tz_convert("UTC").tz_localize(None) if now_ts.tzinfo is not None else now_ts
            last_ts = last_ts.tz_convert("UTC").tz_localize(None) if last_ts.tzinfo is not None else last_ts
        age_h = (now_ts - last_ts).total_seconds() / 3600.0
        if age_h > cfg.stale_after_hours:
            events.append(_event("stale_reading", "critical",
                                 f"latest reading is {age_h:.1f} h old (limit {cfg.stale_after_hours:g} h)",
                                 df, last))

    # ---- invalid values ----------------------------------------------------------
    for s, (lo, hi) in cfg.plausible_bounds.items():
        if s not in df.columns:
            continue
        v = pd.to_numeric(df[s], errors="coerce").iloc[last]
        if pd.notna(v) and (v < lo or v > hi):
            events.append(_event("invalid_value", "critical",
                                 f"{s}={v:g} outside plausibility bounds [{lo:g}, {hi:g}]", df, last, s, v))

    # ---- flat-line ---------------------------------------------------------------
    for s in cfg.flatline_sensors:
        if s not in df.columns:
            continue
        col = pd.to_numeric(df[s], errors="coerce")
        run = 0
        for v in col.iloc[::-1]:
            if np.isnan(v) or v != col.iloc[last] or np.isnan(col.iloc[last]):
                break
            run += 1
        if run >= cfg.flatline_window:
            sev = "critical" if run >= cfg.flatline_critical_window else "warning"
            events.append(_event("flatline", sev,
                                 f"{s}: identical value {col.iloc[last]:g} for the last {run} readings "
                                 "(a stable sensor can be legitimate; check the hardware)", df, last, s, col.iloc[last]))

    # ---- spikes ----------------------------------------------------------------------
    for s in cfg.spike_sensors:
        if s not in df.columns:
            continue
        col = pd.to_numeric(df[s], errors="coerce")
        for pos in range(max(0, last - cfg.spike_lookback + 1), last + 1):
            if np.isnan(col.iloc[pos]):
                continue
            base = col.iloc[max(0, pos - cfg.spike_baseline_window):pos].dropna()
            if len(base) < cfg.spike_min_baseline:
                continue
            med = float(base.median())
            mad = float(np.median(np.abs(base - med))) * 1.4826
            scale = max(mad, cfg.spike_scale_floor_rel * abs(med), 1e-9)
            z = abs(col.iloc[pos] - med) / scale
            if z > cfg.spike_z:
                events.append(_event("spike", "warning",
                                     f"{s}={col.iloc[pos]:g} deviates from recent median {med:g} "
                                     f"(robust z={z:.1f} > {cfg.spike_z:g})", df, pos, s, col.iloc[pos]))

    # ---- TDS out > TDS in --------------------------------------------------------------
    if {"tds_in", "tds_out"} <= set(df.columns):
        tin = pd.to_numeric(df["tds_in"], errors="coerce")
        tout = pd.to_numeric(df["tds_out"], errors="coerce")
        bad = (tout > tin * (1 + cfg.tds_rel_tolerance) + cfg.tds_abs_tolerance_ppm) & tin.notna() & tout.notna()
        recent = bad.iloc[max(0, last - cfg.tds_inconsistency_lookback + 1):]
        if bool(bad.iloc[last]):
            sev = "critical" if int(recent.sum()) >= cfg.tds_inconsistency_critical_count else "warning"
            events.append(_event("tds_out_exceeds_in", sev,
                                 f"tds_out={tout.iloc[last]:g} > tds_in={tin.iloc[last]:g} beyond tolerance "
                                 f"({int(recent.sum())} of last {len(recent)} readings)", df, last, "tds_out", tout.iloc[last]))

    # ---- pressure / flow consistency ---------------------------------------------------
    if {"pressure", "flow_rate"} <= set(df.columns):
        p = pd.to_numeric(df["pressure"], errors="coerce").iloc[last]
        f = pd.to_numeric(df["flow_rate"], errors="coerce").iloc[last]
        if pd.notna(p) and pd.notna(f) and p <= cfg.pressure_near_zero_bar and f >= cfg.flow_min_when_no_pressure_lpm:
            events.append(_event("pressure_flow_inconsistent", "warning",
                                 f"flow_rate={f:g} L/min while pressure={p:g} bar (flow without pressure)",
                                 df, last, "pressure", p))
    return events


def blocking_events(events: List[Dict[str, object]]) -> List[Dict[str, object]]:
    return [e for e in events if e.get("severity") == "critical"]
