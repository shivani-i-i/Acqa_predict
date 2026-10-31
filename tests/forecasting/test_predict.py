import copy
import json
import sys
import types

import numpy as np
import pytest

from aquapredict.forecasting.predict import predict, reset_cache

KEYS = {"filter_stage", "household_id", "forecast_days", "interval_days", "drivers", "explanation",
        "fault_events", "source_change", "status", "reason", "disclaimer"}


def _req(good_request, **changes):
    r = copy.deepcopy(good_request)
    r.update(changes)
    return r


def test_successful_forecast_path(good_request):
    r = predict(good_request)
    assert set(r) == KEYS
    assert r["status"] in {"ok", "degraded"} and r["forecast_days"] is not None and r["forecast_days"] >= 0
    if r["status"] == "ok":
        lo, hi = r["interval_days"]
        assert 0 <= lo <= r["forecast_days"] <= hi and r["reason"] is None
    json.dumps(r)                                          # must be JSON serialisable


def test_input_is_not_mutated(good_request):
    before = copy.deepcopy(good_request)
    predict(good_request)
    assert good_request == before


def test_withheld_when_required_reading_missing(good_request):
    req = _req(good_request)
    req["history"][-1]["usage_litres"] = None
    r = predict(req)
    assert set(r) == KEYS and r["status"] == "withheld"
    assert r["forecast_days"] is None and r["interval_days"] is None
    assert "usage_litres" in r["reason"] and any(e["severity"] == "critical" for e in r["fault_events"])


def test_withheld_when_required_sensor_column_absent(good_request):
    req = _req(good_request)
    for row in req["history"]:
        row.pop("usage_litres", None)
    assert predict(req)["status"] == "withheld"


def test_missing_optional_sensors_do_not_crash(good_request):
    req = _req(good_request)
    for row in req["history"]:
        for k in ("flow_rate", "temperature", "pressure"):
            row.pop(k, None)
    r = predict(req)
    assert set(r) == KEYS and r["status"] in {"ok", "degraded", "withheld"}
    assert not any(e["severity"] == "critical" for e in r["fault_events"])


def test_withheld_for_out_of_range_input(good_request):
    req = _req(good_request)
    req["history"][-1]["tds_in"] = 4000.0                  # plausible hardware value, far outside training data
    r = predict(req)
    assert r["status"] == "withheld" and r["forecast_days"] is None and "training support" in r["reason"]


def test_withheld_for_invalid_value(good_request):
    req = _req(good_request)
    req["history"][-1]["tds_in"] = 99999.0
    r = predict(req)
    assert r["status"] == "withheld" and "invalid_value" in r["reason"]


def test_withheld_for_unseen_filter_stage(good_request):
    r = predict(_req(good_request, filter_stage="uv_lamp"))
    assert r["status"] == "withheld" and "uv_lamp" in r["reason"]


def test_withheld_for_insufficient_history(good_request):
    req = _req(good_request)
    req["history"] = req["history"][:2]
    r = predict(req)
    assert r["status"] == "withheld" and "insufficient history" in r["reason"]


def test_missing_model_artifact_is_withheld_with_reason(good_request, tmp_path):
    r = predict(_req(good_request, model_path=str(tmp_path / "missing.joblib")))
    assert set(r) == KEYS and r["status"] == "withheld" and "model unavailable" in r["reason"]


@pytest.mark.parametrize("bad", [None, "text", 42, {}, {"filter_stage": "carbon"},
                                 {"filter_stage": "carbon", "history": []},
                                 {"filter_stage": "carbon", "history": ["x"]},
                                 {"filter_stage": "carbon", "history": [{"usage_litres": 1}]},
                                 {"filter_stage": 7, "history": [{"timestamp": "2026-01-01", "usage_litres": 1}]}])
def test_invalid_input_never_raises(bad):
    r = predict(bad)
    assert set(r) == KEYS and r["status"] == "withheld" and r["forecast_days"] is None and r["reason"]


def test_schema_is_identical_across_paths(good_request, tmp_path):
    outs = [predict(good_request), predict(_req(good_request, filter_stage="uv_lamp")), predict("junk"),
            predict(_req(good_request, model_path=str(tmp_path / "x.joblib")))]
    assert all(set(o) == KEYS for o in outs)
    assert {o["status"] for o in outs} <= {"ok", "degraded", "withheld"}


def test_noncritical_warning_does_not_block(good_request):
    req = _req(good_request)
    for row in req["history"][-12:]:
        row["pressure"] = 2.0                              # flat-line -> warning only
    r = predict(req)
    assert any(e["type"] == "flatline" and e["severity"] == "warning" for e in r["fault_events"])
    assert r["forecast_days"] is not None


def test_source_change_not_evaluated_with_short_history(good_request):
    req = _req(good_request)
    req["history"] = req["history"][-20:]
    r = predict(req)
    assert r["source_change"]["detected"] is False and r["source_change"].get("evaluated") is False


def test_source_change_reported_when_tds_steps_up(splits, bundle_path):
    """Long stream: +40 % step in tds_in after the 30-reading CUSUM baseline (at 60 % of the history)."""
    from aquapredict.forecasting.examples.generate_data import rows_to_history
    test = splits["test"]
    sizes = test.groupby(["household_id", "filter_stage"]).size().sort_values(ascending=False)
    (hh, stage), n_rows = next(iter(sizes.items()))
    assert n_rows >= 60, "synthetic fixture should contain a reasonably long stream"
    n = min(n_rows, 100)
    step = int(0.6 * n)                      # >= 36, i.e. after the 30-reading baseline window
    hist = rows_to_history(test[(test.household_id == hh) & (test.filter_stage == stage)], n)
    assert sum(r.get("tds_in") is not None for r in hist) > 0.9 * n
    for row in hist[step:]:
        if row.get("tds_in") is not None:
            row["tds_in"] *= 1.4
    r = predict({"filter_stage": stage, "household_id": hh, "history": hist, "model_path": str(bundle_path)})
    sc = r["source_change"]
    assert sc["evaluated"] and sc["detected"]
    assert step - 6 <= sc["change_index"] <= step + 4 and "possible" in sc["message"].lower()


# ------------------------------- SHAP behaviour -------------------------------
def test_shap_unavailable_still_forecasts(good_request, monkeypatch):
    monkeypatch.setitem(sys.modules, "shap", None)        # makes `import shap` raise ImportError
    reset_cache()
    r = predict(good_request)
    assert r["forecast_days"] is not None and r["drivers"] == []
    assert r["explanation"]["available"] is False and "shap" in r["explanation"]["reason"].lower()


def test_shap_runtime_failure_degrades_gracefully(good_request, monkeypatch):
    fake = types.ModuleType("shap")

    class Boom:
        def __init__(self, *a, **k):
            raise RuntimeError("incompatible estimator")
    fake.TreeExplainer = Boom
    monkeypatch.setitem(sys.modules, "shap", fake)
    r = predict(good_request)
    assert r["forecast_days"] is not None
    assert r["explanation"]["available"] is False and "failed" in r["explanation"]["reason"]


def test_explanation_formatting_with_a_FAKE_explainer(good_request, monkeypatch):
    """Tests only OUR formatting/sorting of driver output using a stand-in for shap (not real SHAP).
    The stand-in is additive (base + sum = model output) because non-additive results are rejected."""
    fake = types.ModuleType("shap")
    holder = {}

    class Fake:
        def __init__(self, est):
            holder["est"] = est

        @property
        def expected_value(self):
            return holder["base"]

        def shap_values(self, X):
            vals = np.zeros((len(X), X.shape[1]))
            vals[0, 2], vals[0, 5] = 12.5, -30.0
            holder["base"] = float(holder["est"].predict(X)[0]) - vals.sum()
            return vals
    fake.TreeExplainer = Fake
    monkeypatch.setitem(sys.modules, "shap", fake)
    r = predict(good_request)
    assert r["explanation"]["available"] is True
    d = r["drivers"]
    assert len(d) == 2 and abs(d[0]["contribution_days"]) == 30.0 and d[0]["contribution_days"] < 0
    assert "lowers" in r["explanation"]["summary"] and "not proof of cause" in r["explanation"]["summary"]


def test_permutation_fallback_is_used_when_tree_explainer_unsupported(good_request, monkeypatch):
    """Stand-in for shap (NOT real SHAP): checks our fallback wiring and honest method labelling."""
    fake = types.ModuleType("shap")

    class NoTree:
        def __init__(self, *a, **k):
            raise TypeError("Model type not yet supported by TreeExplainer")

    class FakeMasker:
        def __init__(self, data, max_samples=100):
            self.data = data
    fake.TreeExplainer = NoTree
    fake.maskers = types.SimpleNamespace(Independent=FakeMasker)

    class FakePermutation:
        def __init__(self, fn, masker, seed=None):
            self.fn, self.masker = fn, masker

        def __call__(self, X, max_evals=500, silent=True):
            self.fn(self.masker.data[:3])  # the wrapped predict function must accept a numeric matrix
            vals = np.zeros((len(X), X.shape[1])); vals[0, 3] = 7.0
            base = float(self.fn(np.asarray(X))[0]) - 7.0   # additive: base + sum(values) == model output
            return types.SimpleNamespace(values=vals, base_values=np.array([base]))
    fake.PermutationExplainer = FakePermutation
    monkeypatch.setitem(sys.modules, "shap", fake)
    r = predict(good_request)
    assert r["explanation"]["available"] is True
    assert r["explanation"]["method"] == "shap.PermutationExplainer"
    assert len(r["drivers"]) == 1 and r["drivers"][0]["contribution_days"] == 7.0


def test_require_interval_policy_flag(model, splits, good_request, tmp_path):
    from aquapredict.forecasting.models.intervals import ConformalCalibrator, ConformalConfig
    from aquapredict.forecasting.models.persistence import save_bundle
    weak = ConformalCalibrator(ConformalConfig(min_calibration_samples=10**7)).fit(model, splits["calibration"])
    path = str(save_bundle(model, weak, tmp_path / "weak.joblib"))
    soft = predict(_req(good_request, model_path=path))
    assert soft["status"] == "degraded" and soft["forecast_days"] is not None and soft["interval_days"] is None
    hard = predict(_req(good_request, model_path=path, require_interval=True))
    assert hard["status"] == "withheld" and hard["forecast_days"] is None and "interval" in hard["reason"]


def test_non_additive_shap_values_are_discarded_not_shown(good_request, monkeypatch):
    """Stand-in for shap: reports a base value that does not add up -> must be rejected."""
    fake = types.ModuleType("shap")

    class Bad:
        def __init__(self, model):
            self.expected_value = 1000.0     # base + sum(values) is nowhere near the model output

        def shap_values(self, X):
            vals = np.zeros((len(X), X.shape[1])); vals[0, 1] = 3.0
            return vals
    fake.TreeExplainer = Bad
    monkeypatch.setitem(sys.modules, "shap", fake)
    r = predict(good_request)
    assert r["forecast_days"] is not None and r["drivers"] == []
    assert r["explanation"]["available"] is False and "additivity" in r["explanation"]["reason"]


def test_additive_shap_values_are_accepted_and_gap_reported(good_request, model, monkeypatch):
    fake = types.ModuleType("shap")
    holder = {}

    class Good:
        def __init__(self, est):
            holder["est"] = est

        @property
        def expected_value(self):
            return holder["base"]

        def shap_values(self, X):
            vals = np.zeros((len(X), X.shape[1])); vals[0, 2], vals[0, 4] = 6.0, -2.5
            holder["base"] = float(holder["est"].predict(X)[0]) - vals.sum()
            return vals
    fake.TreeExplainer = Good
    monkeypatch.setitem(sys.modules, "shap", fake)
    r = predict(good_request)
    assert r["explanation"]["available"] is True and r["explanation"]["additivity_gap_days"] < 1e-6


def test_explanation_block_has_identical_keys_on_every_path(good_request):
    ok = predict(good_request)
    withheld = predict({**good_request, "filter_stage": "uv_lamp"})
    bad_input = predict("not a dict")
    assert ok["forecast_days"] is not None and withheld["forecast_days"] is None
    assert set(ok["explanation"]) == set(withheld["explanation"]) == set(bad_input["explanation"])


def test_duplicate_timestamps_are_dropped_not_double_counted(good_request):
    base = predict(good_request)
    dup = _req(good_request)
    dup["history"] = dup["history"] + [dict(dup["history"][-1])]
    r = predict(dup)
    assert r["forecast_days"] == base["forecast_days"]            # last duplicate removed -> same features
    assert any(e["type"] == "duplicate_timestamp" and e["severity"] == "warning" for e in r["fault_events"])


def test_timezone_aware_now_with_naive_timestamps_does_not_break_stale_check(good_request):
    last = good_request["history"][-1]["timestamp"]
    fresh = predict(_req(good_request, now=last + "Z"))
    assert fresh["status"] != "withheld" and "sensor checks failed" not in (fresh["reason"] or "")
    stale = predict(_req(good_request, now="2099-01-01T00:00:00Z"))
    assert stale["status"] == "withheld" and "stale" in stale["reason"]


def test_smuggled_remaining_days_in_history_is_ignored(good_request):
    base = predict(good_request)
    leak = _req(good_request)
    for row in leak["history"]:
        row["remaining_days"] = 0.0
    assert predict(leak)["forecast_days"] == base["forecast_days"]
