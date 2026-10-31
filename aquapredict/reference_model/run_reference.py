"""Train the reference model on train households, calibrate on calibration households, evaluate on test
households against the 3 baselines, and check the abstract's success criteria. Honest PASS/FAIL.
    python -m aquapredict.reference_model.run_reference --n 250 --seed 0
"""
from __future__ import annotations
import argparse, os, time
import pandas as pd
from ..simulator.generate import generate_dataset
from ..baselines.baselines import BASELINES
from ..eval.config import EvalConfig
from ..eval.harness import evaluate
from ..eval.splits import split_households, subset
from ..eval.criteria import check_success
from ..eval.report import render_markdown
from ..eval import metrics as M
from .model import ReferenceForecaster, build_datasets


def run(n=250, seed=0, law="A", out="results_reference", verbose=True, with_prev=False):
    t0 = time.time()
    cfg = EvalConfig()
    df, _ = generate_dataset(n, law=law, seed=seed)
    sp = split_households(df.household_id.unique(), seed=seed)
    tr, ca, te = (subset(df, sp[k]) for k in ("train", "calibration", "test"))
    ds = build_datasets(tr, ca)                       # feature extraction once, reused by every variant
    base = ReferenceForecaster(seed=seed).fit(datasets=ds)
    v1 = ReferenceForecaster(seed=seed, feature_set="v1", select=False, adaptive=False,
                             max_useful_width_days=1e9).fit(datasets=ds)
    fc = dict(BASELINES)
    fc["v1_original"] = v1.variant(gate=True)
    fc["AquaPredict"] = base.variant(gate=True)
    fc["AquaPredict_nogate"] = base.variant(gate=False)
    prev = None
    if with_prev:       # DIAGNOSTIC ONLY: works because the simulator replaces filters exactly at the criterion
        prev = ReferenceForecaster(use_prev_cycles=True, seed=seed).fit(datasets=ds)
        fc["AquaPredict_prevcycles"] = prev.variant(gate=True)
    if verbose:
        print(f"trained in {time.time()-t0:.0f}s; households train/cal/test = {len(sp['train'])}/{len(sp['calibration'])}/{len(sp['test'])}")
    res = evaluate(te, fc, cfg=cfg)
    cyc = M.cycle_outcomes(res.opps, cfg)
    pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)
    tabs = {"window": M.window_summary(res.opps, cfg), "late_false": M.late_false_summary(cyc),
            "intervals": M.interval_summary(res.opps, cfg),
            "window_by_stage": M.window_summary(res.opps, cfg, by=("method", "stage")),
            "late_by_stage": M.late_false_summary(cyc, by=("method", "stage")),
            "interval_by_stage": M.interval_summary(res.opps, cfg, by=("method", "stage"))}
    success = {m: check_success(res.opps, cfg, model=m) for m in fc if m.startswith("AquaPredict") or m == "v1_original"}
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "report.md"), "w") as fh:
        fh.write(render_markdown(res, success["AquaPredict"], model_version=base.version,
                                 data_note=f"SYNTHETIC law {law} seed {seed}"))
    base.save(os.path.join(out, "reference_model.pkl"))
    if verbose:
        for k, v in tabs.items():
            print(f"\n== {k} ==\n{v.round(3)}")
        for m, s in success.items():
            print(f"\n{m}: all criteria met = {s['all_passed']}")
            for k, v in s.items():
                if k != "all_passed":
                    print(f"   {k}: {'PASS' if v['passed'] else 'FAIL'}  " + str({a: b for a, b in v.items() if a not in ('passed', 'rule', 'baselines', 'bootstrap_diff_vs_baseline')}))
    return res, tabs, success, base, (tr, ca, te)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=250); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--law", choices=["A", "B"], default="A"); ap.add_argument("--out", default="results_reference")
    ap.add_argument("--with-prev", action="store_true", help="also run the diagnostic previous-cycles variant")
    a = ap.parse_args()
    run(a.n, a.seed, a.law, a.out, with_prev=a.with_prev)
