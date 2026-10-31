"""Ablation: what does each upgrade of the reference model buy? Uses quick offline metrics on cached features
(stride-4 samples; the harness in run_reference.py is the official evaluation).
    python -m aquapredict.reference_model.ablation --n 250 --seed 0      (~5 min)
"""
from __future__ import annotations
import argparse
import pandas as pd
from ..simulator.generate import generate_dataset
from ..eval.splits import split_households, subset
from ..schema import STAGES
from .model import ReferenceForecaster, build_dataset, quick_metrics

VARIANTS = {
    "1_original (v1 features, no selection, Mondrian intervals)": dict(feature_set="v1", select=False, adaptive=False, max_useful_width_days=1e9),
    "2_+per-litre and source-change features": dict(feature_set="v2", select=False, adaptive=False, max_useful_width_days=1e9),
    "3_+model selection (GBM only where it beats the baseline)": dict(feature_set="v2", select=True, adaptive=False, max_useful_width_days=1e9),
    "4_+scale-adaptive conformal": dict(feature_set="v2", select=True, adaptive=True, max_useful_width_days=1e9),
    "5_+instability history + hide useless intervals (v3)": dict(feature_set="v3", select=True, adaptive=True),
}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=250); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    df, _ = generate_dataset(a.n, law="A", seed=a.seed)
    sp = split_households(df.household_id.unique(), seed=a.seed)
    sets = {k: subset(df, sp[v]) for k, v in (("train", "train"), ("cal", "calibration"), ("test", "test"))}
    ds = {k: {s: build_dataset(sets[k], s, 4) for s in STAGES} for k in sets}
    pd.set_option("display.width", 200)
    for name, kw in VARIANTS.items():
        m = ReferenceForecaster(seed=a.seed, **kw).fit(datasets=ds)
        print(f"\n### {name}\n{quick_metrics(m, ds['test']).round(3).to_string()}", flush=True)


if __name__ == "__main__":
    main()
