"""End-to-end run on held-out households. Person 2: add your model to `forecasters` below."""
from __future__ import annotations
import argparse
import os
import pandas as pd
from ..simulator.generate import generate_dataset
from ..baselines.baselines import BASELINES
from ..planner.cost import policy_table, sensitivity
from ..export import export_results
from . import metrics as M
from .config import EvalConfig
from .harness import evaluate
from .splits import split_households, subset
from .report import render_markdown
from .criteria import check_success


def run(n=150, seed=0, law="A", out="results", forecasters=None, model_name="AquaPredict", verbose=True, cfg_sim=None):
    cfg = EvalConfig()
    df, meta = generate_dataset(n, law=law, seed=seed, cfg=cfg_sim)
    sp = split_households(df.household_id.unique(), seed=seed)
    test = subset(df, sp["test"])
    fc = dict(BASELINES)
    if forecasters:
        fc.update(forecasters)              # e.g. {"AquaPredict": my_model.predict}
    res = evaluate(test, fc, cfg=cfg)
    cyc = M.cycle_outcomes(res.opps, cfg)
    tables = {"window_summary": M.window_summary(res.opps, cfg),
              "late_false_summary": M.late_false_summary(cyc),
              "late_false_by_scenario": M.late_false_summary(cyc, by=("method", "scenario")),
              "window_by_stage": M.window_summary(res.opps, cfg, by=("method", "stage")),
              "operating_curve": M.operating_curve(res.opps, cfg).set_index(["method", "lead_target"]),
              "cost_policies": policy_table(cyc), "cost_sensitivity": sensitivity(cyc).set_index(["method"])}
    iv = M.interval_summary(res.opps, cfg)
    if len(iv):
        tables["interval_summary"] = iv
    success = check_success(res.opps, cfg, model=model_name) if model_name in fc else None
    export_results(tables, out)
    with open(os.path.join(out, "report.md"), "w") as f:
        f.write(render_markdown(res, success, data_note=f"Data: SYNTHETIC, law {law}, seed {seed}, "
                                f"{n} households ({len(sp['train'])} train / {len(sp['calibration'])} cal / {len(sp['test'])} test)."))
    if verbose:
        pd.set_option("display.width", 200); pd.set_option("display.max_columns", 20)
        print(f"test households: {len(sp['test'])}  opportunities: {len(res.opps):,}\n")
        print("WINDOW ERROR (days)\n", tables["window_summary"].round(2), "\n")
        print("LATE / PREMATURE\n", tables["late_false_summary"].round(3), "\n")
        print("COST (illustrative prices, simulated)\n", tables["cost_policies"].round(2))
        if success:
            print("\nSUCCESS CRITERIA\n", success)
    return res, tables, success


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--law", choices=["A", "B"], default="A"); ap.add_argument("--out", default="results")
    ap.add_argument("--calibrated", action="store_true", help="use SimConfig.calibrated() (recommended)")
    a = ap.parse_args()
    from ..simulator.config import SimConfig
    run(a.n, a.seed, a.law, a.out, cfg_sim=SimConfig.calibrated() if a.calibrated else None)
