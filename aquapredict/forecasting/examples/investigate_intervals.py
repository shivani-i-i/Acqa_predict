"""Choose the conformal interval variant WITHOUT touching the final test households (SYNTHETIC data).

    python examples/investigate_intervals.py          # about 1-2 minutes

Why repeated household resplits: with only a couple of dozen calibration/evaluation households, the coverage
of ONE split is very noisy (rows of a household are correlated, so the effective sample is the number of
households). Judging a method on one development split - or on one test split - can mislead in either direction.
This script therefore re-splits the households R times and looks at the DISTRIBUTION of coverage.

Data discipline: the households are split train / calibration / development / test. The test households are
dropped from memory before anything is fitted and never read here. The remaining (train + calibration +
development) households are pooled and, R times, randomly re-split into the same sizes as the real pipeline
(train / calibration / evaluation). Per repeat: fit the forecaster on train, fit every interval variant on
calibration, score it on the evaluation households. ``train_demo.py`` evaluates the chosen variant on the
test households once.

Selection rule (version 2):
  qualifying = mean coverage over repeats >= target - 0.01          (not biased low on average)
               AND mean coverage of EVERY filter stage >= target - 0.03   (no stage left clearly under-covered)
  choose the qualifying variant with the smallest mean interval width;
  if none qualifies, choose the one with the highest mean coverage.
The rule says nothing about the best or worst single split.

Disclosure: version 1 of the rule had only the first condition and selected "pooled_scaled". After seeing the
repeated-resplit results (non-test households) the stage condition was added, because pooled variants clearly
under-cover filter stage ro_membrane on average while the report is required to cover stages. The earlier
test report of the old default had also been seen. The test households are still never read in this script, and
the chosen variant is evaluated on them once in train_demo.py - but that evaluation is not a blind one.

Writes artifacts/interval_selection_dev.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

from aquapredict.forecasting.examples.generate_data import DEMO_HOUSEHOLDS, demo_splits
from aquapredict.forecasting.models.forecast import ForecastConfig, ForecastModel
from aquapredict.forecasting.models.intervals import ConformalCalibrator, ConformalConfig

# Each variant: options for ConformalConfig. ``level`` > target means "calibrate at a more conservative
# nominal level" (a possible remedy for household-level noise); it is compared, not assumed.
VARIANTS = {
    "pooled_absolute": dict(per_stage=False, scaled=False),
    "per_stage_absolute": dict(per_stage=True, scaled=False),
    "pooled_scaled": dict(per_stage=False, scaled=True),
    "per_stage_scaled": dict(per_stage=True, scaled=True),
    "pooled_scaled_level_0.93": dict(per_stage=False, scaled=True, level=0.93),
}
SELECTION_TOLERANCE = 0.01
STAGE_TOLERANCE = 0.03


def choose(results: dict, target: float) -> str:
    ok = [k for k, r in results.items()
          if r["mean_coverage"] >= target - SELECTION_TOLERANCE
          and min(r["mean_coverage_by_stage"].values()) >= target - STAGE_TOLERANCE]
    if ok:
        return min(ok, key=lambda k: results[k]["mean_width_days"])
    return max(results, key=lambda k: results[k]["mean_coverage"])


def _mean_of(dicts, key):
    return {k: float(np.mean([d[k][key] for d in dicts])) for k in dicts[0]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--households", type=int, default=DEMO_HOUSEHOLDS)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--coverage", type=float, default=0.90)
    ap.add_argument("--repeats", type=int, default=12)
    a = ap.parse_args()

    full, splits = demo_splits(a.households, a.seed)
    del full, splits["test"]  # the final test households are dropped from memory and never touched here
    sizes = {k: int(v["household_id"].nunique()) for k, v in splits.items()}
    pool = pd.concat([splits["train"], splits["calibration"], splits["development"]], ignore_index=True)
    ids = np.array(sorted(pool["household_id"].unique()))
    n_tr, n_ca = sizes["train"], sizes["calibration"]

    per_variant = {k: [] for k in VARIANTS}
    for rep in range(a.repeats):
        perm = np.random.default_rng(1000 + rep).permutation(ids)
        part = {"train": perm[:n_tr], "cal": perm[n_tr:n_tr + n_ca], "eval": perm[n_tr + n_ca:]}
        sub = {k: pool[pool["household_id"].isin(v)] for k, v in part.items()}
        model = ForecastModel(ForecastConfig(seed=a.seed)).fit(sub["train"])
        for name, opts in VARIANTS.items():
            opts = dict(opts)
            level = opts.pop("level", a.coverage)
            cal = ConformalCalibrator(ConformalConfig(coverage=level, **opts)).fit(model, sub["cal"])
            per_variant[name].append(cal.evaluate_coverage(model, sub["eval"]))
        print(f"repeat {rep + 1}/{a.repeats}: " + ", ".join(
            f"{k}={per_variant[k][-1]['empirical_coverage']:.3f}" for k in ("pooled_absolute", "pooled_scaled")), flush=True)

    results = {}
    for name, reps in per_variant.items():
        cov = np.array([r["empirical_coverage"] for r in reps])
        results[name] = {
            "mean_coverage": float(cov.mean()), "sd_coverage": float(cov.std(ddof=1)),
            "min_coverage": float(cov.min()), "max_coverage": float(cov.max()),
            "share_of_repeats_at_or_above_target": float((cov >= a.coverage).mean()),
            "mean_width_days": float(np.mean([r["mean_interval_width_days"] for r in reps])),
            "mean_coverage_by_stage": _mean_of([r["by_stage"] for r in reps], "empirical_coverage"),
            "mean_width_by_stage_days": _mean_of([r["by_stage"] for r in reps], "mean_interval_width_days"),
            "mean_coverage_by_true_remaining_bucket": _mean_of([r["by_true_remaining_bucket"] for r in reps], "empirical_coverage"),
            "mean_width_by_true_remaining_bucket_days": _mean_of([r["by_true_remaining_bucket"] for r in reps], "mean_interval_width_days"),
        }
    chosen = choose(results, a.coverage)

    out = {"DATA_ORIGIN": "SYNTHETIC", "households_in_pool": int(len(ids)),
           "resplit_sizes_households": {"train": n_tr, "calibration": n_ca, "evaluation": int(len(ids) - n_tr - n_ca)},
           "repeats": a.repeats, "evaluated_on": "non-test households only (test households never loaded)",
           "target_coverage": a.coverage, "selection_rule": (f"mean coverage >= target - {SELECTION_TOLERANCE} AND every stage's mean coverage >= "
                              f"target - {STAGE_TOLERANCE}, then smallest mean width (rule v2, see script docstring)"),
           "variants": results, "chosen_variant": chosen}
    (ROOT / "artifacts").mkdir(exist_ok=True)
    (ROOT / "artifacts" / "interval_selection_dev.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"\n=== Interval variants over {a.repeats} random household resplits (SYNTHETIC, test households excluded) ===")
    print(f"{'variant':26s} {'mean cov':>8s} {'sd':>6s} {'min':>6s} {'max':>6s} {'>=target':>9s} {'width':>9s}")
    for k, r in results.items():
        print(f"{k:26s} {r['mean_coverage']:8.1%} {r['sd_coverage']:6.1%} {r['min_coverage']:6.1%} {r['max_coverage']:6.1%} "
              f"{r['share_of_repeats_at_or_above_target']:9.0%} {r['mean_width_days']:7.1f} d")
    r = results[chosen]
    print(f"\nchosen by selection rule v2: {chosen}")
    print("  mean coverage by stage:  " + ", ".join(f"{k} {v:.1%}" for k, v in r["mean_coverage_by_stage"].items()))
    print("  mean coverage by bucket: " + ", ".join(f"{k} {v:.1%}" for k, v in r["mean_coverage_by_true_remaining_bucket"].items()))
    print("wrote", ROOT / "artifacts" / "interval_selection_dev.json")


if __name__ == "__main__":
    main()
