import math

import numpy as np
import pytest

from aquapredict.forecasting.features.engineering import TARGET, build_features
from aquapredict.forecasting.models.forecast import ForecastModel
from aquapredict.forecasting.models.intervals import ConformalCalibrator, ConformalConfig


def test_conformal_quantile_function_known_values():
    from aquapredict.forecasting.models.intervals import conformal_quantile
    scores = list(range(1, 11))                       # n = 10
    assert conformal_quantile(scores, 0.90)[0] == 10.0   # k = ceil(11 * 0.9) = 10
    assert conformal_quantile(scores, 0.50)[0] == 6.0    # k = ceil(11 * 0.5) = 6
    q, why = conformal_quantile(scores, 0.95)            # k = ceil(10.45) = 11 > n -> no finite quantile
    assert q is None and "cannot support" in why
    assert conformal_quantile([], 0.9)[0] is None


def test_quantile_matches_manual_split_conformal_formula_absolute(model, splits):
    cal = ConformalCalibrator(ConformalConfig(scaled=False)).fit(model, splits["calibration"])
    calib = splits["calibration"]
    resid = np.sort(np.abs(calib[TARGET].to_numpy() - model.predict(calib)))
    n = len(resid)
    k = math.ceil((n + 1) * 0.90)
    assert cal.n_calibration == n
    assert cal.quantile == pytest.approx(resid[k - 1])
    # by construction the pooled absolute interval covers >= 90 % of the calibration residuals
    assert (resid <= cal.quantile).mean() >= 0.90


def test_quantile_matches_manual_formula_scaled_default(model, calibrator, splits):
    pooled = ConformalCalibrator(ConformalConfig(per_stage=False)).fit(model, splits["calibration"])
    assert pooled.config.scaled                                            # scaled score is the default
    calib = splits["calibration"]
    pred = model.predict(calib)
    score = np.sort(np.abs(calib[TARGET].to_numpy() - pred) / (pred + pooled.config.scale_offset_days))
    k = math.ceil((len(score) + 1) * 0.90)
    assert pooled.quantile == pytest.approx(score[k - 1])


def test_default_interval_config_matches_the_variant_chosen_on_non_test_data():
    import json
    from pathlib import Path
    sel = json.loads((Path(__file__).resolve().parents[2] / "aquapredict" / "forecasting" / "artifacts" / "interval_selection_dev.json").read_text())
    opts = {"pooled_absolute": (False, False), "per_stage_absolute": (True, False),
            "pooled_scaled": (False, True), "per_stage_scaled": (True, True)}
    cfg = ConformalConfig()
    assert opts[sel["chosen_variant"]] == (cfg.per_stage, cfg.scaled)
    assert "never loaded" in sel["evaluated_on"]


def test_scaled_interval_is_wider_for_larger_forecasts(model, splits):
    pooled = ConformalCalibrator(ConformalConfig(per_stage=False)).fit(model, splits["calibration"])
    small, large = pooled.interval(10.0)["interval"], pooled.interval(100.0)["interval"]
    assert (large[1] - large[0]) > (small[1] - small[0])


def test_per_stage_quantiles_match_manual_per_stage_formula(model, splits):
    cal = ConformalCalibrator(ConformalConfig(per_stage=True, scaled=False)).fit(model, splits["calibration"])
    calib = splits["calibration"]
    resid = np.abs(calib[TARGET].to_numpy() - model.predict(calib))
    for stage, q in cal.stage_quantiles.items():
        r = np.sort(resid[(calib["filter_stage"] == stage).to_numpy()])
        k = math.ceil((len(r) + 1) * 0.90)
        assert q == pytest.approx(r[k - 1])
    X = build_features(splits["test"])
    i = 0
    res = cal.interval(50.0, X, row=i)
    if res["interval"] is not None:
        half = cal.stage_quantiles[str(X.iloc[i]["filter_stage"])]
        assert res["interval"][1] == pytest.approx(50.0 + half)
    assert cal.interval(50.0)["interval"] is None   # per-stage needs the feature row


def test_per_stage_and_scaled_roundtrip_dict(model, splits):
    cal = ConformalCalibrator(ConformalConfig(per_stage=True, scaled=True)).fit(model, splits["calibration"])
    again = ConformalCalibrator.from_dict(cal.to_dict())
    assert again.stage_quantiles == cal.stage_quantiles and again.config == cal.config
    X = build_features(splits["test"])
    assert again.interval(40.0, X, 3) == cal.interval(40.0, X, 3)


def test_coverage_report_has_per_stage_and_bucket_breakdown(model, calibrator, splits):
    rep = calibrator.evaluate_coverage(model, splits["development"] if "development" in splits else splits["test"])
    assert set(rep["by_stage"]) <= {"sediment", "carbon", "ro_membrane"} and rep["by_stage"]
    lo, hi = rep["overall_coverage_household_bootstrap_90pct_range"]
    assert 0.0 <= lo <= hi <= 1.0
    assert sum(v["n"] for v in rep["by_stage"].values()) == rep["n_test_rows"]
    assert rep["interval_method"].startswith("per-stage scaled")
    assert 0.0 <= rep["household_coverage"]["min"] <= rep["household_coverage"]["median"] <= 1.0


def test_interval_ordering_and_nonnegative(model, calibrator, splits):
    X = build_features(splits["test"])
    pred = model.predict_features(X)
    for i in range(0, len(X), 97):
        res = calibrator.interval(float(pred[i]), X, row=i)
        if res["interval"] is not None:
            lo, hi = res["interval"]
            assert 0.0 <= lo <= pred[i] <= hi


def test_lower_bound_is_clipped_at_zero(model, splits):
    pooled = ConformalCalibrator(ConformalConfig(per_stage=False)).fit(model, splits["calibration"])
    res = pooled.interval(1.0)               # no feature row -> only calibration sufficiency is checked
    assert res["interval"][0] == 0.0 and res["interval"][1] > 1.0


def test_higher_coverage_gives_wider_interval(model, splits):
    q80 = ConformalCalibrator(ConformalConfig(coverage=0.80)).fit(model, splits["calibration"]).quantile
    q95 = ConformalCalibrator(ConformalConfig(coverage=0.95)).fit(model, splits["calibration"]).quantile
    assert q95 > q80


def test_insufficient_calibration_withholds_interval(model, splits):
    cal = ConformalCalibrator(ConformalConfig(min_calibration_samples=10**6)).fit(model, splits["calibration"])
    res = cal.interval(30.0)
    assert res["interval"] is None and "insufficient" in res["reason"]
    assert cal.evaluate_coverage(model, splits["test"])["available"] is False


def test_too_few_samples_for_requested_coverage(model, splits):
    tiny = splits["calibration"].iloc[:5]
    cal = ConformalCalibrator(ConformalConfig(min_calibration_samples=1, coverage=0.99)).fit(model, tiny)
    assert cal.quantile is None and "cannot support" in cal.insufficient_reason


def test_calibration_households_must_be_unseen(model, splits):
    with pytest.raises(ValueError, match="overlap"):
        ConformalCalibrator().fit(model, splits["train"])


def test_unsupported_conditions_withhold_interval(model, calibrator, splits):
    X = build_features(splits["test"]).iloc[[0]].copy()
    X["pressure"] = 500.0
    res = calibrator.interval(40.0, X)
    assert res["interval"] is None and "calibration range" in res["reason"]


def test_stage_with_too_few_calibration_samples_is_unsupported(model, splits):
    X = build_features(splits["test"]).iloc[[0]]
    pooled = ConformalCalibrator(ConformalConfig(per_stage=False, min_samples_per_stage=10**6)).fit(model, splits["calibration"])
    res = pooled.interval(40.0, X)
    assert res["interval"] is None and "calibration samples for stage" in res["reason"]
    per_stage = ConformalCalibrator(ConformalConfig(per_stage=True, min_samples_per_stage=10**6)).fit(model, splits["calibration"])
    res = per_stage.interval(40.0, X)       # per-stage mode: no stage has a usable quantile
    assert res["interval"] is None and "no usable quantile for stage" in res["reason"]


def test_unfitted_calibrator_withholds():
    assert ConformalCalibrator().interval(10.0)["interval"] is None


def test_invalid_coverage_rejected():
    with pytest.raises(ValueError):
        ConformalCalibrator(ConformalConfig(coverage=1.5))


def test_empirical_coverage_report(model, calibrator, splits):
    rep = calibrator.evaluate_coverage(model, splits["test"])
    assert rep["available"] and 0.0 <= rep["empirical_coverage"] <= 1.0
    assert rep["mean_interval_width_days"] > 0 and "not guaranteed" in rep["note"]


def test_calibrator_roundtrip_dict(calibrator):
    again = ConformalCalibrator.from_dict(calibrator.to_dict())
    assert again.quantile == calibrator.quantile and again.fitted
