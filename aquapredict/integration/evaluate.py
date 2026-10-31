"""Score Person 2's real model with Person 1's strict evaluation, next to the baselines and the stand-in model.
    python -m aquapredict.integration.evaluate --n 200 --seed 21 --calibrated
Everything is SYNTHETIC (Person 1's simulator). Person 2's model is RE-TRAINED on Person 1's training households
and calibrated on separate calibration households; the test households are never touched until scoring."""
from __future__ import annotations
import argparse
import os
import re
import time
import pandas as pd

from ..simulator.generate import generate_dataset
from ..simulator.config import SimConfig
from ..baselines.baselines import BASELINES
from ..eval.config import EvalConfig
from ..eval.harness import evaluate_batched
from ..eval.splits import split_households, subset
from ..eval.criteria import check_success
from ..eval.report import render_markdown
from ..eval import metrics as M
from ..export import export_results
from .train import train_person2_on_person1


def _normalise(reason: str) -> str:
    return re.sub(r"\d+(\.\d+)?", "#", reason)[:110]


def run(n=200, seed=21, calibrated=True, pressure_source="dp", with_reference=True, out="results_integration",
        max_iter=200, verbose=True):
    t0 = time.time()
    cfg = EvalConfig()
    df, _ = generate_dataset(n, seed=seed, cfg=SimConfig.calibrated() if calibrated else None)
    sp = split_households(df.household_id.unique(), seed=seed)
    tr, ca, te = (subset(df, sp[k]) for k in ("train", "calibration", "test"))
    os.makedirs(out, exist_ok=True)
    p2 = train_person2_on_person1(tr, ca, os.path.join(out, "person2_bundle.joblib"), seed=seed, max_iter=max_iter,
                                  pressure_source=pressure_source)
    fc = dict(BASELINES)
    if with_reference:
        from ..reference_model.model import ReferenceForecaster, build_datasets
        ref = ReferenceForecaster(seed=seed).fit(datasets=build_datasets(tr, ca))
        fc["ref_standin"] = ref.variant(gate=True)
    fc["Person2"] = p2
    if verbose:
        print(f"households train/cal/test = {len(sp['train'])}/{len(sp['calibration'])}/{len(sp['test'])}; "
              f"trained in {time.time()-t0:.0f}s", flush=True)
    def _progress(done, total, rss):
        if verbose:
            print(f"  scored {done}/{total} households  ({time.time()-t0:.0f}s, peak memory {rss:.0f} MB)", flush=True)
    res = evaluate_batched(te, fc, cfg=cfg, batch_households=8, progress=_progress,
                           checkpoint=os.path.join(out, "opportunities_partial.pkl"))
    cyc = M.cycle_outcomes(res.opps, cfg)
    o = res.opps
    tabs = {"window": M.window_summary(o, cfg), "late_false": M.late_false_summary(cyc),
            "intervals": M.interval_summary(o, cfg),
            "window_by_stage": M.window_summary(o, cfg, by=("method", "stage")),
            "late_by_stage": M.late_false_summary(cyc, by=("method", "stage")),
            "person2_status": o[o.method == "Person2"].groupby(["stage", "status"]).size().unstack(fill_value=0)}
    wh = o[(o.method == "Person2") & o.withheld]
    tabs["person2_withheld_reasons"] = (wh.assign(r=wh.reason.map(_normalise)).groupby("r").size()
                                        .sort_values(ascending=False).head(10).to_frame("n"))
    success = {m: check_success(o, cfg, model=m) for m in fc if m not in BASELINES}
    export_results(tabs, out)
    note = (f"SYNTHETIC; Person 1 simulator ({'calibrated' if calibrated else 'default'} config), seed {seed}; Person 2 model "
            f"re-trained on Person 1 training households, pressure channel = '{pressure_source}'")
    with open(os.path.join(out, "report.md"), "w") as fh:
        fh.write(render_markdown(res, success.get("Person2"), model_version=f"person2-gbm(max_iter={max_iter})", data_note=note))
    if verbose:
        pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)
        print(f"\nhold-out households={res.n_households} forecasts={len(o):,} short-excluded={res.n_short_excluded}  ({time.time()-t0:.0f}s)")
        for k, v in tabs.items():
            print(f"\n== {k} ==\n{v.round(3)}")
        for m, s in success.items():
            print(f"\n{m}: all criteria met = {s['all_passed']}")
            for k, v in s.items():
                if k != "all_passed":
                    print(f"   {k}: {'PASS' if v['passed'] else 'FAIL'}  " + str({a: (round(b, 3) if isinstance(b, float) else b) for a, b in v.items() if a not in ('passed', 'rule', 'bootstrap_diff_vs_baseline', 'baselines_unpaired')}))
    o.to_pickle(os.path.join(out, "opportunities.pkl"))
    return res, tabs, success


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200); ap.add_argument("--seed", type=int, default=21)
    ap.add_argument("--default-config", action="store_true", help="use the default simulator config (default: calibrated)")
    ap.add_argument("--pressure-source", choices=["downstream", "dp", "inlet"], default="dp")
    ap.add_argument("--no-reference", action="store_true", help="skip the stand-in reference model")
    ap.add_argument("--out", default="results_integration"); ap.add_argument("--max-iter", type=int, default=200)
    a = ap.parse_args()
    run(a.n, a.seed, not a.default_config, a.pressure_source, not a.no_reference, a.out, a.max_iter)
