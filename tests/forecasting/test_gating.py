from aquapredict.forecasting.checks.gating import (ALLOWED_STATUSES, GatingConfig, STATUS_DEGRADED, STATUS_OK, STATUS_WITHHELD,
                           gate_inputs, gate_interval, gate_model_support)


def _ev(sev, kind="x"):
    return {"type": kind, "severity": sev, "message": f"{kind} problem"}


def test_clean_inputs_pass():
    d = gate_inputs(10, [])
    assert d.allowed and d.status == STATUS_OK and d.reason is None


def test_critical_fault_blocks_with_explicit_reason():
    d = gate_inputs(10, [_ev("critical", "stale_reading")])
    assert not d.allowed and d.status == STATUS_WITHHELD and "stale_reading" in d.reason


def test_warnings_and_info_do_not_block():
    assert gate_inputs(10, [_ev("warning", "spike"), _ev("info", "sensor_unavailable")]).allowed


def test_insufficient_history_blocks():
    d = gate_inputs(2, [], GatingConfig(min_history_rows=5))
    assert not d.allowed and "insufficient history" in d.reason


def test_model_support_check_is_separate_and_blocks():
    assert gate_model_support([]).allowed
    d = gate_model_support(["tds_in=4000 outside training range"])
    assert not d.allowed and "training support" in d.reason


def test_interval_policy_degraded_by_default():
    d = gate_interval({"interval": None, "reason": "calibration insufficient"})
    assert d.allowed and d.status == STATUS_DEGRADED and "calibration insufficient" in d.reason


def test_interval_policy_can_require_interval():
    d = gate_interval({"interval": None, "reason": "unsupported"}, GatingConfig(require_interval=True))
    assert not d.allowed and d.status == STATUS_WITHHELD and "requires a justified interval" in d.reason


def test_interval_present_is_ok():
    assert gate_interval({"interval": [1.0, 2.0], "reason": None}).status == STATUS_OK


def test_statuses_are_the_documented_set():
    assert set(ALLOWED_STATUSES) == {"ok", "degraded", "withheld"}
