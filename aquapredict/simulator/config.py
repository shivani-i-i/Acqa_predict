"""Simulator configuration. Every parameter range lives here (nothing hard-coded elsewhere).

STATUS OF VALUES: all are ILLUSTRATIVE assumptions chosen to give realistic-looking household
variation. Ranges for fouling/flux should be tightened against the literature [7] by the team;
until then do not call any value "literature-derived".
"""
from __future__ import annotations
import json
from dataclasses import dataclass, asdict, fields


@dataclass
class SimConfig:
    n_days: int = 1000
    start_date: str = "2026-01-01"
    # household usage
    usage_median_L: float = 40.0
    usage_sigma: float = 0.4
    usage_min_L: float = 8.0
    usage_max_L: float = 150.0
    weekend_factor: float = 1.15
    daily_usage_sigma: float = 0.2
    vacation_rate_per_year: float = 1.5
    # feed water
    feed_tds_median_ppm: float = 350.0
    feed_tds_sigma: float = 0.6
    feed_tds_min_ppm: float = 80.0
    feed_tds_max_ppm: float = 2000.0
    ref_tds_ppm: float = 350.0
    source_change_rate_per_year: float = 0.4
    source_change_min: int = 0
    partic_coupling: float = 1.5      # particulate load changes ~ tds_factor ** coupling at a source change
    # temperature / pressure
    temp_mean_range: tuple = (20.0, 30.0)
    temp_amp_range: tuple = (2.0, 6.0)
    inlet_mean_bar: float = 2.5
    inlet_sd_bar: float = 0.4
    ref_inlet_bar: float = 2.5
    low_inlet_rate_per_year: float = 0.0
    low_inlet_factor: float = 0.45
    # hidden wear heterogeneity
    wear_log_mean: float = 0.0
    wear_log_sigma: float = 0.35
    partic_sigma: float = 0.5
    organic_sigma: float = 0.4
    rej_sens_sigma: float = 0.2
    partic_ramp_start: int = -1       # >=0 enables a gradual particulate ramp (pressure-rise scenario)
    partic_ramp_per_day: float = 0.0
    partic_ramp_cap: float = 1e9      # ceiling on the ramp multiplier (default: effectively none = original behaviour)
    partic_mult_cap: float = 8.0      # ceiling on compounded source-change particulate multiplier (original: 8)
    # clean-purifier characteristics
    flow0_Lpm: float = 0.25
    dp0_bar: float = 0.15
    rej0: float = 0.96
    # sensor noise (multiplicative sigma unless noted)
    noise_dp: float = 0.04
    noise_flow: float = 0.03
    noise_tds: float = 0.03
    noise_feed: float = 0.02
    noise_temp_C: float = 0.3         # additive
    noise_inlet: float = 0.03         # additive
    missing_rate: float = 0.01
    # replacement behaviour: 0 = filter replaced the day after it reaches its criterion (default, idealised).
    # >0 = household replaces LATE by ~N(mean, sd) days; the filter keeps running overdue (sensors keep evolving).
    replacement_delay_mean_days: float = 0.0
    replacement_delay_sd_days: float = 10.0

    @classmethod
    def calibrated(cls, **kw):
        """RECOMMENDED for final results. Caps the particulate ramp (2x) and compounded source-change particulate
        load (3x) so no scenario produces sediment filters that last only days. The DEFAULT config is kept unchanged
        because every number reported before this preset existed was produced with it."""
        base = dict(partic_ramp_cap=2.0, partic_mult_cap=3.0)
        base.update(kw)
        return cls(**base)

    def to_dict(self):
        return asdict(self)

    def to_json(self, path=None):
        s = json.dumps(self.to_dict(), indent=2)
        if path:
            with open(path, "w") as f:
                f.write(s)
        return s

    @classmethod
    def from_dict(cls, d):
        names = {f.name for f in fields(cls)}
        d = {k: (tuple(v) if isinstance(v, list) else v) for k, v in d.items() if k in names}
        return cls(**d)
