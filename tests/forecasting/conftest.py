"""Shared fixtures. Everything is SYNTHETIC and deterministic; no internet needed."""
import pytest

from aquapredict.forecasting.examples.generate_data import generate_dataset, rows_to_history
from aquapredict.forecasting.features.engineering import split_by_household
from aquapredict.forecasting.models.forecast import ForecastConfig, ForecastModel
from aquapredict.forecasting.models.intervals import ConformalCalibrator, ConformalConfig
from aquapredict.forecasting.models.persistence import save_bundle


@pytest.fixture(scope="session")
def dataset():
    return generate_dataset(30, 7)


@pytest.fixture(scope="session")
def splits(dataset):
    return split_by_household(dataset, seed=7)


@pytest.fixture(scope="session")
def model(splits):
    return ForecastModel(ForecastConfig(seed=7, max_iter=60)).fit(splits["train"])


@pytest.fixture(scope="session")
def calibrator(model, splits):
    return ConformalCalibrator(ConformalConfig()).fit(model, splits["calibration"])


@pytest.fixture(scope="session")
def bundle_path(model, calibrator, tmp_path_factory):
    return save_bundle(model, calibrator, tmp_path_factory.mktemp("art") / "bundle.joblib")


@pytest.fixture(scope="session")
def good_request(splits, bundle_path):
    """A request that really yields a forecast (found by trying prefixes of a test stream)."""
    from aquapredict.forecasting.predict import predict
    test = splits["test"]
    sizes = test.groupby(["household_id", "filter_stage"]).size().sort_values(ascending=False)
    for (hh, stage), n_rows in sizes.items():
        g = test[(test.household_id == hh) & (test.filter_stage == stage)]
        for n in range(60, 30, -1):
            if n > len(g):
                continue
            req = {"filter_stage": stage, "household_id": hh, "history": rows_to_history(g, n),
                   "model_path": str(bundle_path)}
            if predict(req)["forecast_days"] is not None:
                return req
    raise RuntimeError("fixture could not find a forecastable stream")
