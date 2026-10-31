"""SHAP explanation path.

* Tests using a stand-in ("fake") shap module check OUR logic: every explainer is validated separately,
  a failing/non-additive explainer falls through to the next one, validation is never skipped.
  They say nothing about whether the real library accepts the model.
* Tests marked "real shap" use ``pytest.importorskip`` and only run when shap is installed.
  They exercise the actual library against the actual fitted estimator.
"""
import sys
import types

import numpy as np
import pytest

from aquapredict.forecasting.features.engineering import build_features
from aquapredict.forecasting.models import explain
from aquapredict.forecasting.models.explain import additivity_gap, additivity_tolerance, explain_prediction


def _row_inputs(model, splits):
    X = build_features(splits["test"]).iloc[:60]
    return X, 30


class _Estimator:
    """Helper giving fakes access to the estimator's raw output on the explained row."""
    def __init__(self):
        self.est = None


def _fake_module(tree=None, permutation=None, kernel=None):
    m = types.ModuleType("shap")
    if tree is not None:
        m.TreeExplainer = tree
    if permutation is not None:
        m.PermutationExplainer = permutation
    if kernel is not None:
        m.KernelExplainer = kernel
    m.maskers = types.SimpleNamespace(Independent=lambda data, max_samples=100: types.SimpleNamespace(data=data))
    return m


def _additive_tree(delta=5.0, offset=0.0):
    """TreeExplainer stand-in: contributions sum to (model output - base) + offset (offset!=0 -> NOT additive)."""
    holder = {}

    class T:
        def __init__(self, est):
            holder["est"] = est

        @property
        def expected_value(self):
            return holder["base"]

        def shap_values(self, X):
            v = np.zeros((len(X), X.shape[1])); v[0, 1] = delta
            holder["base"] = float(holder["est"].predict(X)[0]) - v.sum() + offset
            return v
    return T


def _additive_permutation(delta=3.0):
    class P:
        def __init__(self, fn, masker, seed=None):
            self.fn = fn

        def __call__(self, X, max_evals=500, silent=True):
            v = np.zeros((len(X), X.shape[1])); v[0, 0] = delta
            base = float(self.fn(np.asarray(X))[0]) - delta
            return types.SimpleNamespace(values=v, base_values=np.array([base]))
    return P


def test_additivity_helpers():
    assert additivity_gap(None, np.zeros(3), 5.0) is None
    assert additivity_gap(10.0, np.array([1.0, 2.0]), 13.0) == 0.0
    assert additivity_tolerance(100.0) == pytest.approx(0.5 + 1.0)


def test_non_additive_tree_falls_through_to_permutation(model, splits, monkeypatch):
    X, i = _row_inputs(model, splits)
    mod = _fake_module(tree=_additive_tree(offset=40.0), permutation=_additive_permutation())
    monkeypatch.setitem(sys.modules, "shap", mod)
    r = explain_prediction(model, X, row=i)
    assert r["available"] and r["method"] == "shap.PermutationExplainer"
    tried = {a["method"]: a for a in r["attempts"]}
    assert tried["shap.TreeExplainer"]["accepted"] is False and "additivity" in tried["shap.TreeExplainer"]["problem"]
    assert r["additivity_gap_days"] < 1e-6


def test_raising_tree_and_permutation_fall_through_to_kernel(model, splits, monkeypatch):
    X, i = _row_inputs(model, splits)

    class Boom:
        def __init__(self, *a, **k):
            raise TypeError("unsupported")

    class K:
        def __init__(self, fn, data):
            self.fn = fn
            self.expected_value = 20.0

        def shap_values(self, X, nsamples=0, silent=True):
            v = np.zeros((len(X), X.shape[1])); v[0, 2] = float(self.fn(np.asarray(X))[0]) - 20.0
            return v
    monkeypatch.setitem(sys.modules, "shap", _fake_module(tree=Boom, permutation=Boom, kernel=K))
    r = explain_prediction(model, X, row=i)
    assert r["available"] and r["method"] == "shap.KernelExplainer"
    assert [a["accepted"] for a in r["attempts"]] == [False, False, True]


def test_missing_base_value_cannot_be_validated_and_is_rejected(model, splits, monkeypatch):
    X, i = _row_inputs(model, splits)

    class NoBase:
        def __init__(self, est):
            pass

        def shap_values(self, X):
            return np.ones((len(X), X.shape[1]))
    monkeypatch.setitem(sys.modules, "shap", _fake_module(tree=NoBase))
    r = explain_prediction(model, X, row=i)
    assert r["available"] is False and r["drivers"] == []
    assert "no base value" in r["reason"] and "failed" in r["reason"]


def test_all_explainers_failing_reports_each_reason_and_invents_nothing(model, splits, monkeypatch):
    X, i = _row_inputs(model, splits)
    monkeypatch.setitem(sys.modules, "shap", _fake_module(tree=_additive_tree(offset=99.0)))
    r = explain_prediction(model, X, row=i)
    assert r["available"] is False and r["drivers"] == [] and r["summary"] is None and r["method"] is None
    for name in ("TreeExplainer", "PermutationExplainer", "KernelExplainer"):
        assert name in r["reason"]


def test_shap_not_installed_is_graceful(model, splits, monkeypatch):
    X, i = _row_inputs(model, splits)
    monkeypatch.setitem(sys.modules, "shap", None)
    r = explain_prediction(model, X, row=i)
    assert r["available"] is False and "not installed" in r["reason"]


def test_missing_background_blocks_model_agnostic_paths(model, splits, monkeypatch):
    X, i = _row_inputs(model, splits)
    monkeypatch.setattr(model, "background", None)
    monkeypatch.setitem(sys.modules, "shap", _fake_module(permutation=_additive_permutation()))
    r = explain_prediction(model, X, row=i)
    assert r["available"] is False and "background" in r["reason"]


# ----------------------------------------------------------------- real shap (skipped if not installed)
def test_real_shap_reconstructs_prediction_and_names_features(model, splits):
    pytest.importorskip("shap")
    X, i = _row_inputs(model, splits)
    r = explain_prediction(model, X, row=i, top_k=None)
    assert r["available"], r["reason"]
    assert r["method"] in {"shap.TreeExplainer", "shap.PermutationExplainer", "shap.KernelExplainer"}
    est = model.pipeline.named_steps["model"]
    transformed = model.pipeline.named_steps["prep"].transform(X[model.feature_columns].iloc[[i]])
    names = {d["feature"] for d in r["drivers"]}
    assert names <= set(transformed.columns) and names <= set(model.feature_columns)   # correctly named
    total = r["base_value_days"] + sum(d["contribution_days"] for d in r["drivers"])
    raw = float(est.predict(transformed)[0])
    assert abs(total - raw) <= additivity_tolerance(raw)          # the full reconstruction, not a trimmed one
    assert r["additivity_gap_days"] <= additivity_tolerance(raw)


@pytest.mark.parametrize("method", ["shap.PermutationExplainer", "shap.KernelExplainer"])
def test_real_model_agnostic_explainers_are_additive(model, splits, method):
    shap = pytest.importorskip("shap")
    X, i = _row_inputs(model, splits)
    transformed = model.pipeline.named_steps["prep"].transform(X[model.feature_columns].iloc[[i]])
    accepted, attempts = explain.run_explainers(shap, model, transformed)
    tried = {a["method"]: a for a in attempts}
    if method in tried:                 # only reached when earlier explainers were rejected
        assert tried[method]["accepted"], tried[method]
    else:                               # an earlier explainer (e.g. TreeExplainer) already passed validation
        assert accepted is not None and accepted[1] != method


def test_real_shap_through_predict(good_request):
    pytest.importorskip("shap")
    from aquapredict.forecasting.predict import predict, reset_cache
    reset_cache()
    r = predict(good_request)
    assert r["forecast_days"] is not None
    assert r["explanation"]["available"], r["explanation"]["reason"]
    assert r["drivers"] and all(set(d) == {"feature", "value", "contribution_days"} for d in r["drivers"])
