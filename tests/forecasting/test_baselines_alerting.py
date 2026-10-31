"""20 % health-threshold baseline semantics and late-warning / false-alarm metric definitions."""
import numpy as np
import pandas as pd
import pytest

from aquapredict.forecasting.examples.generate_data import SimConfig, generate_dataset
from aquapredict.forecasting.models.forecast import (AlertConfig, BaselineConfig, baseline_predictions, compare_alerting,
                             evaluate_alerting)


def _X(cum, days, stage="ro_membrane", usage7=50.0):
    return pd.DataFrame({"filter_stage": [stage], "cum_usage_litres": [float(cum)],
                         "days_since_start": [float(days)], "usage_mean_7d": [usage7]})


# ------------------------------------------------------------ 20 % health baseline
def test_default_threshold_is_20_percent_proxy_health():
    cfg = BaselineConfig()
    assert cfg.health_replace_threshold == 0.20 and cfg.volume_rule_fraction == 0.80
    assert not hasattr(cfg, "health_reserve")        # old, ambiguous name is gone


def test_health_baseline_reaches_zero_exactly_at_20_percent_health():
    cap = BaselineConfig().capacity_litres["ro_membrane"]          # 9000 L
    at_threshold = baseline_predictions(_X(0.80 * cap, 99), BaselineConfig())["health_threshold_proxy"][0]
    assert at_threshold == 0.0                                      # proxy health = 20 % -> replace now
    below = baseline_predictions(_X(0.50 * cap, 99), BaselineConfig())["health_threshold_proxy"][0]
    # proxy health 50 %, lifetime rate = cum/(days+1) = 4500/100 = 45 L/day, 0.3*cap left to threshold
    assert below == pytest.approx((0.30 * cap) / 45.0)


def test_threshold_is_configurable_and_validated():
    cap = BaselineConfig().capacity_litres["ro_membrane"]
    x = _X(0.50 * cap, 99)
    t10 = baseline_predictions(x, BaselineConfig(health_replace_threshold=0.10))["health_threshold_proxy"][0]
    t20 = baseline_predictions(x, BaselineConfig(health_replace_threshold=0.20))["health_threshold_proxy"][0]
    assert t10 > t20                                                # a lower health threshold waits longer
    for bad in (-0.1, 1.0, 1.5):
        with pytest.raises(ValueError):
            BaselineConfig(health_replace_threshold=bad)


def test_volume_and_trend_baselines_use_same_replacement_volume():
    cap = BaselineConfig().capacity_litres["carbon"]
    p = baseline_predictions(_X(0.30 * cap, 40, "carbon", usage7=70.0), BaselineConfig())
    assert p["volume_rule"][0] == pytest.approx((0.50 * cap) / 60.0)      # fixed nominal 60 L/day
    assert p["trend_extrapolation"][0] == pytest.approx((0.50 * cap) / 70.0)  # recent 7-day rate


def test_simulator_default_matches_20_percent_health_and_is_configurable():
    short = generate_dataset(6, 1)                                         # default: replace at 80 % consumed
    long = generate_dataset(6, 1, SimConfig(n_households=6, seed=1, replace_at_fraction=1.0))
    assert SimConfig().replace_at_fraction == 0.80
    assert long.groupby(["household_id", "filter_stage"]).size().mean() > \
        short.groupby(["household_id", "filter_stage"]).size().mean()


# ------------------------------------------------------ alert metric definitions
def _lives(preds_by_life):
    """Build a df with one life per entry; each life has true remaining 20,19,...,0 (21 daily rows)."""
    rows, preds = [], []
    for i, p in enumerate(preds_by_life):
        for d in range(21):
            rows.append({"household_id": f"H{i}", "filter_stage": "sediment",
                         "timestamp": pd.Timestamp("2026-01-01") + pd.Timedelta(days=d), "remaining_days": 20.0 - d})
        preds += list(p)
    return pd.DataFrame(rows), np.array(preds, dtype=float)


def _pred_alerting_at(true_lead):
    """Forecast = 99 until the reading with `true_lead` days left, then 5 (<= 14 -> alert)."""
    return [99.0 if (20 - d) > true_lead else 5.0 for d in range(21)]


def test_alert_outcomes_with_hand_computed_cases():
    cfg = AlertConfig(alert_horizon_days=14, min_lead_days=7, max_early_days=28)
    never = [99.0] * 21
    df, p = _lives([_pred_alerting_at(12),     # first alert with 12 d left -> on time
                    _pred_alerting_at(3),      # 3 d < 7 -> late
                    never,                     # no alert at all -> late (and counted in no_alert)
                    _pred_alerting_at(7)])     # exactly min_lead -> NOT late (on time)
    r = evaluate_alerting(df, p, cfg)
    assert r["n_lives"] == 4
    assert r["late_warning_rate"] == 0.5 and r["no_alert_rate"] == 0.25
    assert r["false_alarm_rate"] == 0.0 and r["on_time_rate"] == 0.5
    assert r["median_lead_days"] == pytest.approx(7.0)    # leads: 12, 3, 7 (no-alert life excluded)


def test_false_alarm_means_alert_more_than_max_early_days_before_end():
    cfg = AlertConfig(alert_horizon_days=14, min_lead_days=7, max_early_days=15)
    df, p = _lives([[5.0] * 21, _pred_alerting_at(10)])   # life 0 alerts at 20 d left (> 15) -> false alarm
    r = evaluate_alerting(df, p, cfg)
    assert r["false_alarm_rate"] == 0.5 and r["on_time_rate"] == 0.5 and r["late_warning_rate"] == 0.0
    assert r["false_alarm_row_rate"] is not None and r["false_alarm_row_rate"] > 0


def test_outcome_rates_sum_to_one_and_nan_forecast_never_alerts():
    df, p = _lives([_pred_alerting_at(10), [np.nan] * 21, _pred_alerting_at(2)])
    r = evaluate_alerting(df, p)
    assert r["late_warning_rate"] + r["false_alarm_rate"] + r["on_time_rate"] == pytest.approx(1.0)
    assert r["no_alert_rate"] == pytest.approx(1 / 3)


def test_alert_config_validation_and_scaling():
    with pytest.raises(ValueError):
        AlertConfig(alert_horizon_days=14, min_lead_days=20, max_early_days=28)
    c = AlertConfig.scaled(30)
    assert (c.alert_horizon_days, c.min_lead_days, c.max_early_days) == (30, 15, 60)


def test_compare_alerting_scores_model_and_all_baselines_on_same_lives(model, splits):
    out = compare_alerting(model, splits["test"])
    assert set(out) == {"gradient_boosting", "volume_rule", "health_threshold_proxy", "trend_extrapolation"}
    assert len({v["n_lives"] for v in out.values()}) == 1
    for v in out.values():
        assert v["late_warning_rate"] + v["false_alarm_rate"] + v["on_time_rate"] == pytest.approx(1.0)


def test_interval_investigation_script_never_loads_test_households():
    import inspect
    from aquapredict.forecasting.examples import investigate_intervals as mod
    src = inspect.getsource(mod.main)
    assert 'del full, splits["test"]' in src
    assert 'splits["test"]' not in src.replace('del full, splits["test"]', "")   # no other use of the test split
