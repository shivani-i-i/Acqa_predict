"""Scenario library (10 ready-made + custom). A scenario = config overrides + fault templates."""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass(frozen=True, eq=False)   # eq=False -> hashable by identity (usable as dict key)
class Scenario:
    name: str
    description: str
    overrides: dict = field(default_factory=dict)
    fault_templates: tuple = ()


_ANY_SENSOR = ["dp_sediment_bar", "flow_Lpm", "out_tds_ppm", "feed_tds_ppm", "temp_C", "inlet_bar"]

SCENARIOS = {s.name: s for s in [
    Scenario("normal", "Normal operation"),
    Scenario("high_usage", "Increased household usage", {"usage_median_L": 90.0}),
    Scenario("accelerated_wear", "Filters wear faster than typical", {"wear_log_mean": 0.6}),
    Scenario("gradual_pressure_rise", "Particulate load ramps up -> sediment pressure drop climbs faster",
             {"partic_ramp_start": 120, "partic_ramp_per_day": 0.01}),
    Scenario("sudden_sensor_drift", "A sensor drifts 25-50% within weeks",
             fault_templates=({"kind": "drift", "prob": 1.0, "sensors": ["dp_sediment_bar", "flow_Lpm", "out_tds_ppm"],
                               "duration_range": (10, 30), "magnitude_range": (0.25, 0.5), "min_start": 60},)),
    Scenario("stuck_sensor", "A sensor flat-lines for weeks",
             fault_templates=({"kind": "stuck", "prob": 1.0, "sensors": _ANY_SENSOR,
                               "duration_range": (20, 80), "min_start": 60},)),
    Scenario("missing_readings", "High random missingness plus sensor outages",
             {"missing_rate": 0.15},
             fault_templates=({"kind": "dropout", "prob": 1.0, "count": 2, "sensors": _ANY_SENSOR,
                               "duration_range": (10, 40), "min_start": 40},)),
    Scenario("source_change", "Sudden water-source changes (e.g. tanker water)",
             {"source_change_rate_per_year": 3.0, "source_change_min": 1}),
    Scenario("unusual_operating_conditions", "Frequent low-inlet-pressure periods",
             {"low_inlet_rate_per_year": 6.0}),
    Scenario("insufficient_history", "Only 10 days of data since install (forecast must be withheld)",
             {"n_days": 10}),
    # 11th, NOT in DEFAULT_MIX: realistic replacement timing (households replace ~25 days after the criterion)
    Scenario("late_replacement", "Households replace filters late; filters run overdue. Removes the idealised "
             "'replaced exactly at the criterion' leak from previous-cycle information",
             {"replacement_delay_mean_days": 25.0, "replacement_delay_sd_days": 15.0}),
]}

DEFAULT_MIX = {
    "normal": 0.30, "high_usage": 0.08, "accelerated_wear": 0.08, "gradual_pressure_rise": 0.08,
    "sudden_sensor_drift": 0.08, "stuck_sensor": 0.08, "missing_readings": 0.08,
    "source_change": 0.08, "unusual_operating_conditions": 0.08, "insufficient_history": 0.06,
}


def custom_scenario(name, overrides=None, fault_templates=(), description="Custom scenario"):
    """Build a custom scenario, e.g. custom_scenario('my_case', {'usage_median_L': 60})."""
    return Scenario(name, description, dict(overrides or {}), tuple(fault_templates))
