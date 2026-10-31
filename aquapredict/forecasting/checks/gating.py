"""Forecast gating: decide whether to issue a forecast, and whether to attach an interval.

Policy (documented in README)
-----------------------------
Three DISTINCT questions, each answered by its own function:

1. gate_inputs        - are the sensor inputs usable?  (missing required inputs,
                        critical faults, invalid values, too little history)
2. gate_model_support - are conditions inside what the MODEL was trained on?
                        (training-range check; blocks the forecast)
3. gate_interval      - is an INTERVAL justified by the conformal CALIBRATION set?
                        (calibration size / range / per-stage coverage of the data)
                        If not: status "degraded" (point forecast, no interval),
                        or "withheld" when ``require_interval=True``.

Non-critical warnings (spikes, flat-line, optional sensor missing) never block.

Statuses
--------
"ok"        forecast + interval issued.
"degraded"  forecast issued, interval withheld (reason says why).
"withheld"  no forecast (reason says why).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

STATUS_OK = "ok"
STATUS_DEGRADED = "degraded"
STATUS_WITHHELD = "withheld"
ALLOWED_STATUSES = (STATUS_OK, STATUS_DEGRADED, STATUS_WITHHELD)


@dataclass
class GatingConfig:
    min_history_rows: int = 3
    require_interval: bool = False


@dataclass
class GateDecision:
    allowed: bool
    status: str
    reason: Optional[str] = None
    reasons: List[str] = field(default_factory=list)


def gate_inputs(n_history_rows: int, fault_events: List[Dict[str, object]],
                config: Optional[GatingConfig] = None) -> GateDecision:
    cfg = config or GatingConfig()
    reasons = [f"{e['type']}: {e['message']}" for e in fault_events if e.get("severity") == "critical"]
    if n_history_rows < cfg.min_history_rows:
        reasons.insert(0, f"insufficient history: {n_history_rows} reading(s), need at least {cfg.min_history_rows}")
    if reasons:
        return GateDecision(False, STATUS_WITHHELD, "Forecast withheld - " + "; ".join(reasons), reasons)
    return GateDecision(True, STATUS_OK)


def gate_model_support(range_reasons: List[str]) -> GateDecision:
    if range_reasons:
        return GateDecision(False, STATUS_WITHHELD,
                            "Forecast withheld - conditions outside the model's training support: "
                            + "; ".join(range_reasons), list(range_reasons))
    return GateDecision(True, STATUS_OK)


def gate_interval(interval_result: Dict[str, object], config: Optional[GatingConfig] = None) -> GateDecision:
    cfg = config or GatingConfig()
    if interval_result.get("interval") is not None:
        return GateDecision(True, STATUS_OK)
    why = str(interval_result.get("reason") or "interval not available")
    if cfg.require_interval:
        return GateDecision(False, STATUS_WITHHELD,
                            f"Forecast withheld - policy requires a justified interval, but: {why}", [why])
    return GateDecision(True, STATUS_DEGRADED, f"Point forecast only; interval withheld - {why}", [why])
