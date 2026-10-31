"""Train + calibrate + evaluate on SYNTHETIC data and save the model artifact.

    python examples/train_demo.py [--households 120] [--seed 42] [--coverage 0.9]

Households are split as TRAIN / CALIBRATION / DEVELOPMENT / TEST (whole households only):
  * TRAIN        fits the gradient-boosting forecaster,
  * DEVELOPMENT  was used (by examples/investigate_intervals.py) to choose the interval variant,
  * CALIBRATION + DEVELOPMENT  together fit the FINAL conformal quantile (more households =
                 a less noisy quantile; the choice of variant is already made at that point),
  * TEST         is evaluated exactly once, here. Nothing is tuned against it.

Writes: artifacts/aquapredict_bundle.joblib  and  artifacts/metrics.json
All numbers come from the synthetic simulator - they say nothing about real purifiers.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

from aquapredict.forecasting.checks.drift import input_drift, monitoring_report, prediction_error
from aquapredict.forecasting.examples.generate_data import DEMO_HOUSEHOLDS, demo_splits
from aquapredict.forecasting.features.engineering import TARGET, build_features
from aquapredict.forecasting.models.forecast import (AlertConfig, BaselineConfig, ForecastConfig, ForecastModel, compare_alerting,
                             compare_with_baselines)
from aquapredict.forecasting.models.intervals import ConformalCalibrator, ConformalConfig
from aquapredict.forecasting.models.persistence import DEFAULT_ARTIFACT, save_bundle

ALERT_HORIZONS = (14.0, 30.0)


def environment_info() -> dict:
    """Library versions the numbers were produced with (results can differ slightly across versions)."""
    import joblib, numpy, scipy, sklearn
    try:
        import shap
        shap_version = getattr(shap, "__version__", "installed")
    except ImportError:
        shap_version = None
    return {"python": platform.python_version(), "numpy": numpy.__version__, "pandas": pd.__version__,
            "scikit-learn": sklearn.__version__, "scipy": scipy.__version__, "joblib": joblib.__version__,
            "shap": shap_version}


def run_id_of(metrics: dict) -> str:
    """12-hex content hash of the metrics (excluding run_id itself): same numbers -> same id."""
    body = {k: v for k, v in metrics.items() if k != "run_id"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:12]


def _print_regression(title, table):
    print(title)
    print(f"{'method':26s} {'median AE':>10s} {'MAE':>8s} {'RMSE':>8s}")
    for name, m in table.items():
        print(f"{name:26s} {m['median_ae_days']:10.1f} {m['mae_days']:8.1f} {m['rmse_days']:8.1f}   n={m['n']}")


def _pct(x):
    return "   n/a" if x is None else f"{100 * x:5.1f}%"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--households", type=int, default=DEMO_HOUSEHOLDS)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--coverage", type=float, default=0.90)
    a = ap.parse_args()

    _, splits = demo_splits(a.households, a.seed)
    ids = {k: set(v["household_id"]) for k, v in splits.items()}
    names = list(ids)
    for i, x in enumerate(names):
        for y in names[i + 1:]:
            assert not (ids[x] & ids[y]), f"household leak between {x} and {y}"

    model = ForecastModel(ForecastConfig(seed=a.seed)).fit(splits["train"])
    final_cal_df = pd.concat([splits["calibration"], splits["development"]], ignore_index=True)
    calib = ConformalCalibrator(ConformalConfig(coverage=a.coverage)).fit(model, final_cal_df)
    test = splits["test"]

    bcfg = BaselineConfig()
    comparison = compare_with_baselines(model, test, bcfg)
    near_end = compare_with_baselines(model, test, bcfg, max_remaining_days=30)
    alerting = {f"horizon_{int(h)}d": compare_alerting(model, test, bcfg, AlertConfig.scaled(h)) for h in ALERT_HORIZONS}
    alerting["horizon_14d"] = compare_alerting(model, test, bcfg, AlertConfig())  # default definition
    coverage = calib.evaluate_coverage(model, test)
    drift = input_drift(build_features(splits["train"]), build_features(test),
                        features=["cum_usage_litres", "usage_mean_7d", "tds_in", "pressure"])
    perf = prediction_error(test[TARGET].to_numpy(), model.predict(test),
                            baseline_mae=comparison["gradient_boosting"]["mae_days"])

    metrics = {
        "DATA_ORIGIN": "SYNTHETIC - simulator output, not real telemetry", "seed": a.seed,
        "split_policy": "whole households: train / calibration / development / test; test evaluated once here",
        "rows": {k: int(len(v)) for k, v in splits.items()},
        "households": {k: len(v) for k, v in ids.items()},
        "replacement_criterion": "simulator replaces at 80 % of nominal capacity consumed (= 20 % proxy health left)",
        "baseline_config": {"health_replace_threshold": bcfg.health_replace_threshold,
                            "volume_rule_fraction": bcfg.volume_rule_fraction,
                            "nominal_daily_litres": bcfg.nominal_daily_litres},
        "interval_method": calib.method_name(), "interval_calibration_households": calib.n_calibration_households,
        "conformal_quantile": calib.quantile, "n_calibration": calib.n_calibration,
        "test_comparison": comparison, "test_comparison_last_30_days_of_life": near_end,
        "test_alerting": alerting, "test_interval_coverage": coverage,
        "input_drift_train_vs_test": drift, "monitoring": monitoring_report(drift, perf),
        "environment": environment_info()}
    metrics["run_id"] = run_id_of(metrics)
    model.metadata["run_id"] = metrics["run_id"]          # the saved model carries the id of THIS run
    path = save_bundle(model, calib, DEFAULT_ARTIFACT)
    (ROOT / "artifacts").mkdir(exist_ok=True)
    (ROOT / "artifacts" / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")

    print("=== AquaPredict training (SYNTHETIC data) ===")
    print("rows:", metrics["rows"], "| households:", metrics["households"])
    _print_regression("\n-- Remaining-days error, all test rows (days) --", comparison)
    _print_regression("\n-- Same test households, rows with <= 30 true days left (retrospective) --", near_end)
    for key, table in alerting.items():
        d = table["gradient_boosting"]["definition"]
        print(f"\n-- Alerting, alert when forecast <= {d['alert_horizon_days']:g} d; late = lead < {d['min_lead_days']:g} d "
              f"or no alert; false alarm = lead > {d['max_early_days']:g} d --")
        print(f"{'method':26s} {'late':>7s} {'false':>7s} {'on time':>8s} {'no alert':>9s} {'median lead':>12s}  (lives={table['gradient_boosting']['n_lives']})")
        for name, r in table.items():
            ml = "n/a" if r["median_lead_days"] is None else f"{r['median_lead_days']:.1f} d"
            print(f"{name:26s} {_pct(r['late_warning_rate'])} {_pct(r['false_alarm_rate'])} {_pct(r['on_time_rate']):>8s} "
                  f"{_pct(r['no_alert_rate']):>9s} {ml:>12s}")
    if coverage.get("available"):
        print(f"\n-- Intervals ({coverage['interval_method']}), target {coverage['target_coverage']:.0%}, "
              f"final calibration on {calib.n_calibration_households} households --")
        lo, hi = coverage["overall_coverage_household_bootstrap_90pct_range"]
        print(f"overall: coverage {coverage['empirical_coverage']:.1%} (household-bootstrap 90% range {lo:.1%} - {hi:.1%}), "
              f"mean width {coverage['mean_interval_width_days']:.1f} days")
        print(f"  share of test households with coverage >= {coverage['target_coverage'] - 0.10:.0%}: "
              f"{coverage['household_coverage']['share_at_or_above_target_minus_10pts']:.0%}")
        for k, v in coverage["by_stage"].items():
            print(f"  {k:12s} coverage {v['empirical_coverage']:.1%}  width {v['mean_interval_width_days']:.1f} d  n={v['n']}")
        for k, v in coverage["by_true_remaining_bucket"].items():
            print(f"  {k:12s} coverage {v['empirical_coverage']:.1%}  width {v['mean_interval_width_days']:.1f} d  n={v['n']}")
        print("  (coverage is NOT guaranteed on new households or real data)")
    print(f"\nrun_id {metrics['run_id']}  (environment: {metrics['environment']})")
    print("saved model ->", path)
    print("saved metrics ->", ROOT / "artifacts" / "metrics.json")


if __name__ == "__main__":
    main()
