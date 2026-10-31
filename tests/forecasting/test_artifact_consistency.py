"""Shipped metrics, README tables and sample output must come from the same run (identified by run_id).

If you re-run train_demo.py locally (other library versions can change the numbers slightly) your metrics.json gets a
different run_id than the one the README was rendered from; the README-consistency tests then skip with a message
instead of failing - run `python examples/update_readme_results.py` to refresh the README from your artifacts.
"""
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2] / "aquapredict" / "forecasting"
METRICS, SAMPLE, README = ROOT / "artifacts" / "metrics.json", ROOT / "examples" / "sample_prediction.json", ROOT / "README.md"

pytestmark = pytest.mark.skipif(not METRICS.exists(), reason="artifacts/metrics.json not present")


def _metrics():
    return json.loads(METRICS.read_text(encoding="utf-8"))


def _readme_run_id():
    m = re.search(r"<!-- RESULTS:START run_id=([0-9a-f]+) -->", README.read_text(encoding="utf-8"))
    return m.group(1) if m else None


def test_metrics_run_id_is_the_hash_of_its_own_content():
    from aquapredict.forecasting.examples.train_demo import run_id_of
    m = _metrics()
    assert m["run_id"] == run_id_of(m)
    tampered = dict(m, seed=m["seed"] + 1)          # any manual edit of the numbers/config changes the hash
    assert run_id_of(tampered) != m["run_id"]


def test_metrics_record_the_environment_that_produced_them():
    env = _metrics()["environment"]
    for key in ("python", "numpy", "pandas", "scikit-learn", "scipy", "joblib", "shap"):
        assert key in env


def test_readme_results_match_artifacts_of_the_same_run():
    if _readme_run_id() != _metrics()["run_id"]:
        pytest.skip("metrics.json was regenerated locally; README tables refer to the shipped run "
                    "(run python examples/update_readme_results.py to refresh them)")
    from aquapredict.forecasting.examples.update_readme_results import build_readme
    text = README.read_text(encoding="utf-8")
    assert build_readme(text) == text, "README tables differ from artifacts; run examples/update_readme_results.py"


def test_sample_output_comes_from_the_same_run_as_metrics():
    if not SAMPLE.exists() or _readme_run_id() != _metrics()["run_id"]:
        pytest.skip("artifacts regenerated locally or sample missing")
    assert json.loads(SAMPLE.read_text(encoding="utf-8"))["model_run_id"] == _metrics()["run_id"]


def test_readme_headline_numbers_are_the_metrics_numbers():
    if _readme_run_id() != _metrics()["run_id"]:
        pytest.skip("metrics.json regenerated locally")
    c = _metrics()["test_interval_coverage"]
    text = README.read_text(encoding="utf-8")
    assert f"{100 * c['empirical_coverage']:.1f} %" in text and f"{c['mean_interval_width_days']:.1f} d" in text
    assert f"{_metrics()['test_comparison']['gradient_boosting']['median_ae_days']:.1f}" in text


def test_trained_model_artifact_carries_the_same_run_id_if_present():
    from aquapredict.forecasting.models.persistence import DEFAULT_ARTIFACT, load_bundle
    if not DEFAULT_ARTIFACT.exists():
        pytest.skip("no locally trained model artifact")
    assert load_bundle(DEFAULT_ARTIFACT)[0].metadata.get("run_id") == _metrics()["run_id"]
