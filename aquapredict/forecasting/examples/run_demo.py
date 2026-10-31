"""Run predict() on several SYNTHETIC scenarios. Run examples/train_demo.py first.

    python examples/run_demo.py [--households 60] [--seed 42]

Prints each response and writes examples/sample_prediction.json.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

from aquapredict.forecasting.examples.generate_data import DEMO_HOUSEHOLDS, demo_splits, rows_to_history
from aquapredict.forecasting.models.persistence import DEFAULT_ARTIFACT, load_bundle
from aquapredict.forecasting.predict import predict


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--households", type=int, default=DEMO_HOUSEHOLDS)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    if not DEFAULT_ARTIFACT.exists():
        sys.exit(f"No trained model at {DEFAULT_ARTIFACT}.\nRun first:  python examples/train_demo.py")
    # same deterministic household split as train_demo.py: the demo only uses TEST households,
    # which the model, the calibration and the interval selection never saw
    test = demo_splits(a.households, a.seed)[1]["test"]
    sizes = test.groupby(["household_id", "filter_stage"]).size()
    long_enough = sizes[sizes >= 70]
    if long_enough.empty:
        sys.exit("No test filter life with >= 70 readings; use more --households.")
    hh, stage = long_enough.index[0]
    group = test[(test.household_id == hh) & (test.filter_stage == stage)]
    history = rows_to_history(group, 60)  # first 60 readings; the true target is NOT passed in
    truth = float(group["remaining_days"].iloc[59])

    scenarios = {}
    scenarios["1_normal"] = {"filter_stage": stage, "household_id": hh, "history": history}

    missing = copy.deepcopy(history); missing[-1]["usage_litres"] = None
    scenarios["2_required_reading_missing"] = {"filter_stage": stage, "household_id": hh, "history": missing}

    unsupported = copy.deepcopy(history); unsupported[-1]["tds_in"] = 4000.0
    scenarios["3_value_outside_training_range"] = {"filter_stage": stage, "household_id": hh, "history": unsupported}

    warn = copy.deepcopy(history)
    for r in warn[-12:]:
        r["pressure"] = 2.0  # flat-lined pressure -> warning only
    scenarios["4_flatline_warning_still_forecasts"] = {"filter_stage": stage, "household_id": hh, "history": warn}

    shift = copy.deepcopy(history)
    for r in shift[35:]:
        if r.get("tds_in") is not None:
            r["tds_in"] *= 1.25  # persistent step in input TDS
    scenarios["5_possible_source_change"] = {"filter_stage": stage, "household_id": hh, "history": shift}

    scenarios["6_unknown_filter_stage"] = {"filter_stage": "uv_lamp", "household_id": hh, "history": history}

    model_run_id = load_bundle(DEFAULT_ARTIFACT)[0].metadata.get("run_id")   # same id as artifacts/metrics.json
    out = {"DATA_ORIGIN": "SYNTHETIC", "model_run_id": model_run_id, "note": f"scenario 1 true remaining_days (hidden from the model) = {truth}"}
    for name, req in scenarios.items():
        res = predict(req)
        out[name] = res
        print(f"\n--- {name} ---")
        print(json.dumps({k: res[k] for k in ("filter_stage", "forecast_days", "interval_days", "status", "reason")}, indent=2))
        print("fault events:", [(e["type"], e["severity"]) for e in res["fault_events"]],
              "| source_change detected:", res["source_change"].get("detected"))
    print(f"\n(scenario 1 true remaining days, synthetic: {truth})")
    path = ROOT / "examples" / "sample_prediction.json"
    path.write_text(json.dumps(out, indent=2, default=str))
    print("wrote", path)


if __name__ == "__main__":
    main()
