import numpy as np
import pandas as pd
import pytest

from aquapredict.forecasting.checks.source_change import CusumConfig, detect_all_shifts, detect_shift


def _series(before=300.0, after=420.0, n1=60, n2=60, noise=9.0, seed=0):
    rng = np.random.default_rng(seed)
    return np.r_[rng.normal(before, noise, n1), rng.normal(after, noise, n2)]


def test_detects_known_upward_shift_near_change_point():
    ts = pd.date_range("2026-01-01", periods=120, freq="D")
    r = detect_shift(_series(), ts)
    assert r["detected"] and r["direction"] == "up"
    assert 58 <= r["change_index"] <= 63 and r["alarm_index"] >= 60
    assert r["alarm_timestamp"] is not None and r["change_timestamp"] is not None


def test_detects_downward_shift():
    r = detect_shift(_series(400, 250))
    assert r["detected"] and r["direction"] == "down" and 55 <= r["change_index"] <= 63


def test_no_shift_in_stable_noise():
    assert not detect_shift(_series(300, 300, seed=1))["detected"]


def test_single_spike_does_not_trigger_alarm():
    x = _series(300, 300, seed=2); x[70] = 900.0
    assert not detect_shift(x)["detected"]


def test_missing_values_are_skipped():
    x = _series(); x[::7] = np.nan
    assert detect_shift(x)["detected"]


def test_result_is_described_as_possible_not_proof():
    msg = detect_shift(_series())["message"].lower()
    assert "possible" in msg and "not proof" in msg


def test_reference_mean_configuration():
    x = _series(300, 420, n1=10, n2=40)
    r = detect_shift(x, config=CusumConfig(reference_mean=300.0, reference_std=9.0))
    assert r["detected"] and r["change_index"] <= 12 and r["baseline_mean"] == 300.0


def test_not_enough_data():
    r = detect_shift(np.ones(10))
    assert not r["detected"] and "more than" in r["message"]


def test_allowable_drift_and_threshold_are_configurable():
    x = _series(300, 330)                                  # small (~3.5 sigma) shift
    sensitive = detect_shift(x, config=CusumConfig(k_sigma=0.25, h_sigma=3.0))
    insensitive = detect_shift(x, config=CusumConfig(k_sigma=3.0, h_sigma=50.0))
    assert sensitive["detected"] and not insensitive["detected"]


def test_detect_all_restarts_after_alarm():
    rng = np.random.default_rng(4)
    x = np.r_[rng.normal(300, 9, 60), rng.normal(420, 9, 60), rng.normal(250, 9, 60)]
    found = detect_all_shifts(x)
    assert [f["direction"] for f in found] == ["up", "down"]


def test_spike_inside_baseline_window_does_not_blind_detector():
    """Regression: a single spike in the 30-reading baseline used to inflate sigma ~10x."""
    x = _series(300, 420, n1=70, n2=50, noise=6.0, seed=3)
    x[17] = 900.0  # spike inside the baseline window
    r = detect_shift(x)
    assert r["detected"] and r["direction"] == "up" and 68 <= r["change_index"] <= 74
    assert r["baseline_std"] < 30  # robust estimate, not ~170 as a plain std would give
