"""Evaluation settings. FIX THESE BEFORE LOOKING AT RESULTS (abstract: metrics fixed beforehand)."""
from __future__ import annotations
from dataclasses import dataclass, asdict


@dataclass
class EvalConfig:
    stride: int = 7                    # forecast every N days within a filter life
    min_age_days: int = 14             # minimum history before a forecast is attempted
    lead_target_days: float = 21.0     # alert fires when alert_value <= this many days to criterion
    min_notice_days: float = 7.0       # LATE warning: alert less than this many days before the criterion
    premature_days: float = 45.0       # FALSE ALARM (premature): alert more than this many days before it
    primary_horizon_days: float = 60.0 # primary window-error metric uses forecasts with true remaining <= this
    target_coverage: float = 0.90
    coverage_tol: float = 0.05         # "close to 90%" = within +/- this
    max_median_width_days: float = 40.0  # interval not "very wide": median width (primary horizon) <= this
    false_alarm_tol: float = 0.05      # "comparable false-alarm rate" tolerance
    alert_uses_lower_bound: bool = True  # models with an interval alert on its lower bound

    def to_dict(self):
        return asdict(self)
