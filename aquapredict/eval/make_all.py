"""One command: multi-seed evaluation with CIs + figures + tables.  python -m aquapredict.eval.make_all"""
from __future__ import annotations
import argparse, os
from ..simulator.generate import generate_household
from ..export import export_results
from .config import EvalConfig
from .multiseed import run_multiseed
from . import metrics as M, figures as F


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5); ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--law", choices=["A", "B"], default="A"); ap.add_argument("--out", default="results")
    a = ap.parse_args(argv)
    cfg = EvalConfig()
    opps, cyc, tb, excl = run_multiseed(seeds=range(a.seeds), n=a.n, law=a.law, cfg=cfg)
    os.makedirs(os.path.join(a.out, "figures"), exist_ok=True)
    export_results({k: v.set_index(list(v.columns[:1])) if "method" in v.columns else v for k, v in tb.items()}, a.out)
    F.fig_comparison(tb, os.path.join(a.out, "figures", "comparison.png"))
    F.fig_operating_curve(M.operating_curve(opps, cfg), os.path.join(a.out, "figures", "operating_curve.png"))
    for i, sc in enumerate(["normal", "source_change", "sudden_sensor_drift", "stuck_sensor"]):
        df, _ = generate_household(i, sc, seed=3)
        F.fig_household(df, os.path.join(a.out, "figures", f"household_{sc}.png"))
    print(tb["median_abs_err_ci"].round(2).to_string(index=False))
    print(f"pooled {len(opps):,} forecasts, {cyc.groupby('method').size().iloc[0]:,} filter lives, "
          f"{excl} too-short lives excluded. Outputs in {a.out}/")


if __name__ == "__main__":
    main()
