"""Save / load the fitted forecaster + conformal calibration with joblib.

SECURITY: joblib files are pickles. Only load artifacts you created or trust.
Loading an untrusted file can execute arbitrary code.
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Tuple, Union

import joblib
import sklearn

from aquapredict.forecasting.features.engineering import FEATURE_COLUMNS
from aquapredict.forecasting.models.forecast import ForecastConfig, ForecastModel
from aquapredict.forecasting.models.intervals import ConformalCalibrator

FORMAT_VERSION = 2
DEFAULT_ARTIFACT = Path(__file__).resolve().parents[1] / "artifacts" / "aquapredict_bundle.joblib"


class ArtifactNotFoundError(FileNotFoundError):
    pass


class ArtifactIncompatibleError(ValueError):
    pass


def save_bundle(model: ForecastModel, calibrator: ConformalCalibrator,
                path: Union[str, Path] = DEFAULT_ARTIFACT) -> Path:
    if not model.is_fitted:
        raise ValueError("Refusing to save: forecaster is not fitted.")
    if not calibrator.fitted:
        raise ValueError("Refusing to save: conformal calibrator is not fitted.")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "format_version": FORMAT_VERSION,
        "sklearn_version": sklearn.__version__,
        "feature_columns": list(model.feature_columns),
        "forecast_config": asdict(model.config),
        "training_ranges": model.training_ranges,
        "known_stages": model.known_stages,
        "metadata": model.metadata,
        "pipeline": model.pipeline,
        "shap_background": model.background,
        "calibration": calibrator.to_dict(),
        "data_origin": "SYNTHETIC",
    }
    joblib.dump(bundle, path)
    return path


def load_bundle(path: Union[str, Path] = DEFAULT_ARTIFACT) -> Tuple[ForecastModel, ConformalCalibrator]:
    path = Path(path)
    if not path.exists():
        raise ArtifactNotFoundError(
            f"Model artifact not found at {path}. Train one first: python examples/train_demo.py")
    try:
        b = joblib.load(path)
        if b.get("format_version") != FORMAT_VERSION:
            raise ArtifactIncompatibleError(f"Artifact format {b.get('format_version')} != expected {FORMAT_VERSION}.")
        if b["feature_columns"] != list(FEATURE_COLUMNS):
            raise ArtifactIncompatibleError("Artifact feature schema differs from the current code. Retrain.")
        if b["sklearn_version"].split(".")[:2] != sklearn.__version__.split(".")[:2]:
            raise ArtifactIncompatibleError(
                f"Artifact built with scikit-learn {b['sklearn_version']}, installed {sklearn.__version__}. Retrain.")
        cfg = dict(b["forecast_config"]); cfg["range_quantiles"] = tuple(cfg["range_quantiles"])
        model = ForecastModel(ForecastConfig(**cfg))
        model.pipeline = b["pipeline"]; model.training_ranges = b["training_ranges"]
        model.known_stages = b["known_stages"]; model.metadata = b["metadata"]
        model.feature_columns = b["feature_columns"]; model.background = b.get("shap_background")
        calibrator = ConformalCalibrator.from_dict(b["calibration"])
    except (ArtifactIncompatibleError, ArtifactNotFoundError):
        raise
    except Exception as exc:  # corrupt / foreign file
        raise ArtifactIncompatibleError(f"Could not read artifact {path}: {exc}") from exc
    if not model.is_fitted:
        raise ArtifactIncompatibleError("Artifact contains no fitted pipeline.")
    return model, calibrator
