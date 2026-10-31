import numpy as np
import pandas as pd
import pytest

from aquapredict.forecasting.features.engineering import TARGET, build_features
from aquapredict.forecasting.models.forecast import (BaselineConfig, ForecastConfig, ForecastModel, baseline_predictions,
                             compare_with_baselines, regression_metrics, validate_targets)
from aquapredict.forecasting.models.intervals import ConformalCalibrator
from aquapredict.forecasting.models.persistence import (ArtifactIncompatibleError, ArtifactNotFoundError, load_bundle, save_bundle)


def test_predictions_are_nonnegative_and_finite(model, splits):
    pred = model.predict(splits["test"])
    assert len(pred) == len(splits["test"]) and np.isfinite(pred).all() and (pred >= 0).all()


def test_evaluate_returns_metrics(model, splits):
    m = model.evaluate(splits["test"])
    assert set(m) == {"n", "mae_days", "median_ae_days", "rmse_days"} and m["median_ae_days"] >= 0


@pytest.mark.parametrize("bad", [-1.0, np.nan, np.inf])
def test_invalid_training_targets_rejected(splits, bad):
    train = splits["train"].copy()
    train.loc[train.index[0], TARGET] = bad
    with pytest.raises(ValueError):
        ForecastModel().fit(train)


def test_validate_targets_rejects_empty():
    with pytest.raises(ValueError):
        validate_targets([])


def test_training_requires_target_column(splits):
    with pytest.raises(ValueError):
        ForecastModel().fit(splits["train"].drop(columns=[TARGET]))


def test_unfitted_model_refuses_to_predict(splits):
    with pytest.raises(RuntimeError):
        ForecastModel().predict(splits["test"])


def test_same_seed_gives_same_model(splits):
    a = ForecastModel(ForecastConfig(seed=1, max_iter=20)).fit(splits["train"])
    b = ForecastModel(ForecastConfig(seed=1, max_iter=20)).fit(splits["train"])
    assert np.allclose(a.predict(splits["test"]), b.predict(splits["test"]))


def test_baselines_and_comparison(model, splits):
    X = build_features(splits["test"])
    preds = baseline_predictions(X, BaselineConfig())
    assert set(preds) == {"volume_rule", "health_threshold_proxy", "trend_extrapolation"}
    assert all(len(p) == len(X) for p in preds.values())
    comp = compare_with_baselines(model, splits["test"])
    assert set(comp) == {"gradient_boosting", *preds}
    assert all(np.isfinite(v["median_ae_days"]) for v in comp.values())


def test_regression_metrics_ignore_nan_pairs():
    m = regression_metrics([1, 2, 3], [2, np.nan, 5])
    assert m["n"] == 2 and m["mae_days"] == pytest.approx(1.5)


def test_out_of_range_detection(model, splits):
    X = build_features(splits["test"]).iloc[[0]].copy()
    X["tds_in"] = 50000.0
    assert any("tds_in" in r for r in model.out_of_range(X))
    X2 = build_features(splits["test"]).iloc[[0]].copy()
    X2["filter_stage"] = "uv_lamp"
    assert any("uv_lamp" in r for r in model.out_of_range(X2))


# ------------------------------ persistence ---------------------------------
def test_predictions_identical_after_reload(model, calibrator, splits, tmp_path):
    path = save_bundle(model, calibrator, tmp_path / "m.joblib")
    m2, c2 = load_bundle(path)
    assert np.array_equal(model.predict(splits["test"]), m2.predict(splits["test"]))
    assert c2.quantile == calibrator.quantile and c2.config == calibrator.config
    assert m2.feature_columns == model.feature_columns and m2.training_ranges == model.training_ranges
    assert m2.background is not None and m2.background.equals(model.background)  # SHAP background survives


def test_missing_artifact_gives_clear_error(tmp_path):
    with pytest.raises(ArtifactNotFoundError, match="train_demo"):
        load_bundle(tmp_path / "nope.joblib")


def test_unfitted_objects_are_not_saved(model, calibrator, tmp_path):
    with pytest.raises(ValueError):
        save_bundle(ForecastModel(), calibrator, tmp_path / "a.joblib")
    with pytest.raises(ValueError):
        save_bundle(model, ConformalCalibrator(), tmp_path / "b.joblib")
    assert not (tmp_path / "a.joblib").exists() and not (tmp_path / "b.joblib").exists()


def test_corrupt_or_foreign_artifact_rejected(tmp_path):
    bad = tmp_path / "bad.joblib"
    bad.write_bytes(b"not a joblib file")
    with pytest.raises(ArtifactIncompatibleError):
        load_bundle(bad)
    import joblib
    joblib.dump({"format_version": 999}, tmp_path / "old.joblib")
    with pytest.raises(ArtifactIncompatibleError):
        load_bundle(tmp_path / "old.joblib")


def test_training_survives_a_sensor_that_is_entirely_absent(splits):
    train = splits["train"].drop(columns=["flow_rate"])          # hardware without a flow sensor
    m = ForecastModel(ForecastConfig(seed=1, max_iter=20)).fit(train)
    assert "flow_rate" in m.metadata["excluded_all_missing_features"]
    assert np.isfinite(m.predict(splits["test"].drop(columns=["flow_rate"]))).all()


def test_comparison_can_score_only_near_end_of_life(model, splits):
    from aquapredict.forecasting.models.forecast import compare_with_baselines
    full = compare_with_baselines(model, splits["test"])
    near = compare_with_baselines(model, splits["test"], max_remaining_days=30)
    assert 0 < near["gradient_boosting"]["n"] < full["gradient_boosting"]["n"]
    assert set(near) == set(full) == {"gradient_boosting", "volume_rule", "health_threshold_proxy", "trend_extrapolation"}


def test_intervals_identical_after_reload_for_every_interval_variant(model, splits, tmp_path):
    from aquapredict.forecasting.features.engineering import build_features
    from aquapredict.forecasting.models.intervals import ConformalCalibrator, ConformalConfig
    X = build_features(splits["test"])
    pred = model.predict_features(X)
    for opts in ({}, {"per_stage": True}, {"scaled": False}, {"per_stage": True, "scaled": False}):
        cal = ConformalCalibrator(ConformalConfig(**opts)).fit(model, splits["calibration"])
        m2, c2 = load_bundle(save_bundle(model, cal, tmp_path / f"m{len(opts)}{sorted(opts)}.joblib"))
        for i in (0, 50, 200):
            assert c2.interval(float(pred[i]), X, i) == cal.interval(float(pred[i]), X, i)
        assert c2.config == cal.config
