import numpy as np
import pandas as pd

from aquapredict.forecasting.checks.drift import input_drift, interval_coverage, monitoring_report, prediction_error, psi


def _df(mean, n=300, seed=0):
    return pd.DataFrame({"x": np.random.default_rng(seed).normal(mean, 1, n)})


def test_no_input_drift_for_same_distribution():
    r = input_drift(_df(0, seed=1), _df(0, seed=2))
    assert not r["any_drift"] and r["features"]["x"]["status"] == "ok"


def test_input_drift_detected_for_shifted_distribution():
    r = input_drift(_df(0), _df(1.5, seed=3))
    assert r["any_drift"] and r["drifted_features"] == ["x"] and r["features"]["x"]["psi"] > 0.2


def test_insufficient_data_is_reported_not_guessed():
    r = input_drift(_df(0, n=10), _df(0, n=10))
    assert r["status"] == "insufficient_data" and not r["any_drift"]


def test_psi_is_near_zero_for_identical_samples():
    x = np.random.default_rng(0).normal(size=500)
    assert psi(x, x) < 1e-3


def test_prediction_error_without_targets():
    assert prediction_error(None, None)["status"] == "no_observed_targets"


def test_prediction_error_degradation_flag():
    y = np.full(100, 50.0)
    ok = prediction_error(y, y + 5, baseline_mae=5.0)
    bad = prediction_error(y, y + 30, baseline_mae=5.0)
    assert ok["degraded"] is False and bad["degraded"] is True
    assert prediction_error(y[:5], y[:5], min_samples=30)["status"] == "insufficient_data"


def test_interval_coverage_monitor_and_withheld_intervals():
    y = np.arange(100, dtype=float)
    good = interval_coverage(y, y - 1, y + 1)
    assert good["empirical_coverage"] == 1.0 and not good["undercovering"]
    bad = interval_coverage(y, y + 10, y + 20)
    assert bad["empirical_coverage"] == 0.0 and bad["undercovering"]
    lo = [None] * 10 + list(y[10:] - 1); hi = [None] * 10 + list(y[10:] + 1)
    part = interval_coverage(y, lo, hi)
    assert part["n_withheld"] == 10 and part["n_with_interval"] == 90
    assert interval_coverage(None, None, None)["status"] == "no_observed_targets"


def test_input_drift_is_not_reported_as_performance_degradation():
    drift = input_drift(_df(0), _df(2, seed=4))
    rep = monitoring_report(drift, prediction_error(None, None), interval_coverage(None, None, None))
    assert rep["input_drift_detected"] is True
    assert rep["performance_degradation_confirmed"] is False
    assert rep["performance_evidence_available"] is False
