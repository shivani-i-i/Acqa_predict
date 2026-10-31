import numpy as np
import pandas as pd

from aquapredict.forecasting.checks.sensor_faults import SensorFaultConfig, blocking_events, detect_sensor_faults


def _hist(n=40, seed=0, **over):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=n, freq="D"),
        "usage_litres": rng.normal(50, 3, n), "tds_in": rng.normal(300, 5, n),
        "tds_out": rng.normal(25, 1, n), "pressure": rng.normal(2.5, 0.03, n),
        "flow_rate": rng.normal(1.5, 0.02, n), "temperature": rng.normal(26, 0.5, n)})
    for k, v in over.items():
        df[k] = v
    return df


def _types(events, kind=None):
    return [e for e in events if kind is None or e["type"] == kind]


def test_clean_data_has_no_events():
    assert detect_sensor_faults(_hist()) == []


def test_event_structure():
    df = _hist(); df.loc[df.index[-1], "tds_in"] = np.nan
    e = detect_sensor_faults(df)[0]
    assert {"type", "severity", "message", "timestamp", "sensor", "value"} <= set(e)
    assert e["severity"] in {"info", "warning", "critical"} and e["timestamp"] is not None


def test_missing_required_sensor_is_critical_missing_optional_is_warning():
    df = _hist(); df.loc[df.index[-1], ["usage_litres", "pressure"]] = np.nan
    ev = detect_sensor_faults(df)
    assert [e["severity"] for e in ev if e["sensor"] == "usage_litres"] == ["critical"]
    assert [e["severity"] for e in ev if e["sensor"] == "pressure"] == ["warning"]
    assert len(blocking_events(ev)) == 1


def test_sensor_never_reporting_is_info_if_optional_critical_if_required():
    ev = detect_sensor_faults(_hist(flow_rate=np.nan))
    assert _types(ev, "sensor_unavailable")[0]["severity"] == "info" and not blocking_events(ev)
    ev2 = detect_sensor_faults(_hist().drop(columns=["usage_litres"]))
    assert blocking_events(ev2)


def test_stale_reading_needs_now_and_threshold():
    df = _hist()
    assert not _types(detect_sensor_faults(df), "stale_reading")                        # no `now` -> no check
    last = df.timestamp.iloc[-1]
    assert not _types(detect_sensor_faults(df, now=last + pd.Timedelta(hours=10)), "stale_reading")
    ev = detect_sensor_faults(df, now=last + pd.Timedelta(hours=100))
    assert _types(ev, "stale_reading")[0]["severity"] == "critical"


def test_flatline_is_a_warning_and_critical_only_when_very_long():
    cfg = SensorFaultConfig(flatline_window=10, flatline_critical_window=30)
    df = _hist(); df.loc[df.index[-12:], "pressure"] = 2.5
    ev = _types(detect_sensor_faults(df, cfg), "flatline")
    assert len(ev) == 1 and ev[0]["severity"] == "warning" and ev[0]["sensor"] == "pressure"
    df2 = _hist(n=60, pressure=2.5)
    assert _types(detect_sensor_faults(df2, cfg), "flatline")[0]["severity"] == "critical"


def test_short_repeat_or_stable_noisy_sensor_is_not_a_fault():
    df = _hist(); df.loc[df.index[-5:], "pressure"] = 2.5          # only 5 repeats < window
    assert not _types(detect_sensor_faults(df), "flatline")
    df2 = _hist(); df2["pressure"] = np.random.default_rng(3).normal(2.5, 0.001, len(df2))  # very stable but alive
    assert not _types(detect_sensor_faults(df2), "flatline")


def test_spike_detected_and_normal_variation_ignored():
    df = _hist(); df.loc[df.index[-1], "tds_in"] = 1200.0
    sp = _types(detect_sensor_faults(df), "spike")
    assert sp and sp[0]["sensor"] == "tds_in" and sp[0]["severity"] == "warning"
    assert not _types(detect_sensor_faults(_hist(seed=5)), "spike")


def test_spike_thresholds_are_configurable():
    df = _hist(); df.loc[df.index[-1], "tds_in"] = 360.0           # +20 %
    assert _types(detect_sensor_faults(df, SensorFaultConfig(spike_z=3.0)), "spike")
    assert not _types(detect_sensor_faults(df, SensorFaultConfig(spike_z=1000.0)), "spike")


def test_tds_out_greater_than_in():
    df = _hist(); df.loc[df.index[-1], "tds_out"] = 500.0
    ev = _types(detect_sensor_faults(df), "tds_out_exceeds_in")
    assert ev and ev[0]["severity"] == "warning"
    df2 = _hist(); df2.loc[df2.index[-3:], "tds_out"] = 500.0      # persistent -> critical
    assert _types(detect_sensor_faults(df2), "tds_out_exceeds_in")[0]["severity"] == "critical"


def test_tds_small_excess_within_tolerance_is_ok():
    df = _hist(); df.loc[df.index[-1], ["tds_in", "tds_out"]] = [300.0, 310.0]
    assert not _types(detect_sensor_faults(df), "tds_out_exceeds_in")


def test_pressure_flow_inconsistency():
    df = _hist(); df.loc[df.index[-1], ["pressure", "flow_rate"]] = [0.0, 1.5]
    assert _types(detect_sensor_faults(df), "pressure_flow_inconsistent")
    df2 = _hist(); df2.loc[df2.index[-1], ["pressure", "flow_rate"]] = [0.0, 0.0]
    assert not _types(detect_sensor_faults(df2), "pressure_flow_inconsistent")


def test_invalid_value_is_critical():
    df = _hist(); df.loc[df.index[-1], "temperature"] = 400.0
    assert _types(detect_sensor_faults(df), "invalid_value")[0]["severity"] == "critical"


def test_empty_history_is_critical():
    assert detect_sensor_faults(pd.DataFrame())[0]["severity"] == "critical"


def test_high_missing_rate_is_a_warning_not_a_block():
    df = _hist(n=30)
    df.loc[df.index[-10:], "pressure"] = np.nan  # 10 of the last 14 missing, earlier data present
    ev = _types(detect_sensor_faults(df), "high_missing_rate")
    assert ev and ev[0]["severity"] == "warning" and ev[0]["sensor"] == "pressure"
    assert not blocking_events(detect_sensor_faults(df))


def test_high_missing_rate_ignores_sensor_that_never_reported():
    df = _hist(n=30); df["pressure"] = np.nan  # hardware not fitted -> info, not a missing-rate warning
    kinds = {e["type"] for e in detect_sensor_faults(df)}
    assert "sensor_unavailable" in kinds and "high_missing_rate" not in kinds
