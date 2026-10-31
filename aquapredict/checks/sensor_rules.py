"""Transparent rule-based sensor checks (rules first; a learned detector only if it finds faults these miss).

check_readings(hist, stage) -> list of events {sensor, kind, severity, blocks_forecast, reason, status}.
Status is always 'suspected' - a rule firing is not a confirmed fault.
LIMITS: flat-line rule assumes sensors are not heavily quantised (real quantised sensors need a tolerance);
slow drift and spikes are NOT detected here; source changes are not classified here.
"""
from __future__ import annotations
import numpy as np

SENSORS = ["dp_sediment_bar", "flow_Lpm", "out_tds_ppm", "feed_tds_ppm", "temp_C", "inlet_bar"]
# Sensors whose faults make a forecast for the stage unreliable. Carbon is usage/volume-based in the reference model.
RELEVANT = {"sediment": {"dp_sediment_bar", "flow_Lpm"},
            "ro": {"flow_Lpm", "out_tds_ppm", "feed_tds_ppm", "temp_C"},
            "carbon": set(), None: set(SENSORS)}


def check_readings(hist, stage=None, window=14, flat_days=7, stale_days=10, miss_thresh=0.5):
    idle = hist["op_mode"].to_numpy() == "idle"
    act = hist.loc[~idle]
    rel = RELEVANT[stage]
    events = []

    def add(sensor, kind, sev, reason):
        events.append({"sensor": sensor, "kind": kind, "severity": sev, "blocks_forecast": sensor in rel,
                       "reason": reason, "status": "suspected"})

    for s in SENSORS:
        v = act[s].to_numpy(float)
        if len(v) == 0:
            continue
        recent = v[-window:]
        last = recent[-flat_days:]
        fin = last[np.isfinite(last)]
        if len(fin) >= flat_days - 2 and np.ptp(fin) == 0.0:
            add(s, "stuck", "high", f"{s} returned an identical value for {len(fin)} readings")
        if len(v) >= stale_days and not np.isfinite(v[-stale_days:]).any():
            add(s, "stale", "high", f"no valid {s} reading in the last {stale_days} active days")
        elif len(recent) >= flat_days and np.isnan(recent).mean() > miss_thresh:
            add(s, "missing", "medium", f"{np.isnan(recent).mean():.0%} of recent {s} readings missing")
    o, f = act["out_tds_ppm"].to_numpy(float)[-5:], act["feed_tds_ppm"].to_numpy(float)[-5:]
    both = np.isfinite(o) & np.isfinite(f)
    if both.sum() >= 3 and (o[both] > f[both]).sum() >= 3:
        add("out_tds_ppm", "tds_inversion", "high", "output TDS above input TDS (physically implausible; investigate)")
    return events


def blocking_reason(hist, stage):
    ev = [e for e in check_readings(hist, stage) if e["blocks_forecast"]]
    return "; ".join(sorted({f"{e['kind']}:{e['sensor']}" for e in ev})) if ev else ""
