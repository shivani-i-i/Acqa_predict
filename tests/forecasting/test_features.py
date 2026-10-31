import numpy as np
import pandas as pd
import pytest

from aquapredict.forecasting.features.engineering import (FEATURE_COLUMNS, TARGET, build_features, split_by_household,
                                  validate_columns)
from aquapredict.forecasting.models.forecast import ForecastModel


def _stream(n=20, hh="h1", stage="sediment", **extra):
    df = pd.DataFrame({"timestamp": pd.date_range("2026-01-01", periods=n, freq="D"),
                       "household_id": hh, "filter_stage": stage, "usage_litres": 10.0})
    for k, v in extra.items():
        df[k] = v
    return df


def test_tds_rejection_and_division_by_zero():
    df = _stream(3, tds_in=[300.0, 0.0, 200.0], tds_out=[30.0, 5.0, np.nan])
    f = build_features(df)
    assert f["tds_rejection"].iloc[0] == pytest.approx(0.9)
    assert np.isnan(f["tds_rejection"].iloc[1])   # tds_in == 0 -> not computed
    assert np.isnan(f["tds_rejection"].iloc[2])   # tds_out missing -> not computed
    assert np.isfinite(f.select_dtypes("number").to_numpy()[~np.isnan(f.select_dtypes("number").to_numpy())]).all()


def test_target_is_never_a_feature():
    assert TARGET not in FEATURE_COLUMNS
    df = _stream(5)
    df[TARGET] = 99.0
    f = build_features(df)
    assert TARGET not in f.columns and list(f.columns) == FEATURE_COLUMNS


def test_no_future_leakage_features_match_prefix_computation():
    rng = np.random.default_rng(0)
    df = _stream(40, tds_in=rng.normal(300, 10, 40), tds_out=rng.normal(30, 2, 40),
                 pressure=rng.normal(2.5, 0.1, 40), flow_rate=rng.normal(1.5, 0.05, 40))
    full = build_features(df)
    for t in (5, 17, 30):
        prefix = build_features(df.iloc[: t + 1])
        pd.testing.assert_series_equal(full.iloc[t], prefix.iloc[t], check_names=False)


def test_changing_future_rows_does_not_change_past_features():
    rng = np.random.default_rng(1)
    df = _stream(30, tds_in=rng.normal(300, 10, 30), pressure=rng.normal(2.5, 0.1, 30))
    altered = df.copy()
    altered.loc[20:, ["usage_litres", "tds_in", "pressure"]] = 9999.0
    a, b = build_features(df), build_features(altered)
    pd.testing.assert_frame_equal(a.iloc[:20], b.iloc[:20])


def test_absent_sensors_stay_nan_not_invented():
    f = build_features(_stream(10))
    for col in ("tds_in", "tds_out", "pressure", "flow_rate", "temperature", "tds_rejection",
                "flow_pressure_ratio", "pressure_change_7d"):
        assert f[col].isna().all(), col


def test_negative_and_infinite_values_become_nan():
    df = _stream(3, pressure=[2.0, -1.0, np.inf], tds_in=[300.0, 300.0, 300.0])
    f = build_features(df)
    assert f["pressure"].iloc[0] == 2.0 and f["pressure"].iloc[1:].isna().all()


def test_cumulative_usage_is_per_household_and_stage():
    df = pd.concat([_stream(5, hh="a"), _stream(5, hh="b", stage="carbon")], ignore_index=True)
    f = build_features(df)
    assert f["cum_usage_litres"].tolist() == [10, 20, 30, 40, 50, 10, 20, 30, 40, 50]


def test_missing_required_column_rejected():
    with pytest.raises(ValueError):
        validate_columns(_stream(3).drop(columns=["usage_litres"]))
    with pytest.raises(ValueError):
        build_features(_stream(3).drop(columns=["timestamp"]))


def test_household_split_is_disjoint_complete_and_deterministic(dataset):
    a = split_by_household(dataset, seed=3)
    b = split_by_household(dataset, seed=3)
    ids = {k: set(v.household_id) for k, v in a.items()}
    assert not (ids["train"] & ids["calibration"]) and not (ids["train"] & ids["test"]) \
        and not (ids["calibration"] & ids["test"])
    assert sum(len(v) for v in a.values()) == len(dataset)
    assert all(len(v) for v in ids.values())
    assert ids == {k: set(v.household_id) for k, v in b.items()}


def test_split_needs_three_households():
    with pytest.raises(ValueError):
        split_by_household(_stream(5), seed=0)


def test_preprocessing_is_fitted_on_training_data_only():
    tr = pd.concat([_stream(40, hh="a", stage="sediment", tds_in=300.0), _stream(40, hh="b", stage="carbon", tds_in=310.0)])
    tr[TARGET] = np.arange(len(tr), 0, -1, dtype=float)
    m = ForecastModel().fit(tr)
    cats = m.pipeline.named_steps["prep"].named_transformers_["cat"].categories_[0]
    assert sorted(cats) == ["carbon", "sediment"]         # nothing from outside the training set
    unseen = _stream(10, hh="c", stage="ro_membrane", tds_in=300.0)
    assert np.isfinite(m.predict(unseen)).all()           # unseen category tolerated (treated as missing)


def test_four_way_household_split_is_disjoint_complete_deterministic_and_named(dataset):
    a = split_by_household(dataset, seed=3, fractions=(0.45, 0.20, 0.15, 0.20))
    b = split_by_household(dataset, seed=3, fractions=(0.45, 0.20, 0.15, 0.20))
    assert list(a) == ["train", "calibration", "development", "test"]
    ids = {k: set(v["household_id"]) for k, v in a.items()}
    names = list(ids)
    for i, x in enumerate(names):
        for y in names[i + 1:]:
            assert not (ids[x] & ids[y]), (x, y)             # no household in two sets
    assert set().union(*ids.values()) == set(dataset["household_id"])   # nothing dropped
    assert sum(len(v) for v in a.values()) == len(dataset)               # no row duplicated or lost
    assert all(ids[k] == set(b[k]["household_id"]) for k in a)
    assert all(len(v) >= 1 for v in ids.values())


def test_split_fraction_validation():
    import pandas as pd
    df = pd.DataFrame({"household_id": list("ABCDEFGH")})
    for bad in [(0.5, 0.5), (0.5, 0.2, 0.2), (0.5, 0.5, 0.0), (0.4, 0.2, 0.2, 0.1, 0.1)]:
        with pytest.raises(ValueError):
            split_by_household(df, fractions=bad)
