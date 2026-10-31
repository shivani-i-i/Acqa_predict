"""Pool several independent synthetic datasets (different seeds) so conclusions don't rest on one draw.
Household ids are prefixed with the seed to keep them unique. Reports pooled metrics WITH household-bootstrap CIs."""
from __future__ import annotations
import pandas as pd
from ..simulator.generate import generate_dataset
from ..baselines.baselines import BASELINES
from .config import EvalConfig
from .harness import evaluate
from .splits import split_households, subset
from . import metrics as M


def run_multiseed(seeds=(0, 1, 2, 3, 4), n=150, law="A", forecasters=None, cfg=None, n_boot=300):
    cfg = cfg or EvalConfig()
    fc = dict(BASELINES)
    fc.update(forecasters or {})
    parts, excluded = [], 0
    for sd in seeds:
        df, _ = generate_dataset(n, law=law, seed=sd)
        sp = split_households(df.household_id.unique(), seed=sd)
        res = evaluate(subset(df, sp["test"]), fc, cfg=cfg)
        res.opps["household_id"] = f"s{sd}_" + res.opps["household_id"]
        parts.append(res.opps); excluded += res.n_short_excluded
    opps = pd.concat(parts, ignore_index=True)
    cyc = M.cycle_outcomes(opps, cfg)
    cyc["late_f"] = cyc["late"].astype(float); cyc["premature_f"] = cyc["premature"].astype(float)
    iss = M.issued(opps)
    iss = iss[iss["true_dtc"] <= cfg.primary_horizon_days].copy()
    iss["abs_err"] = (iss["point"] - iss["true_dtc"]).abs()
    tables = {
        "median_abs_err_ci": M.household_bootstrap_ci(iss, "abs_err", ("method",), n_boot, stat="median"),
        "late_rate_ci": M.household_bootstrap_ci(cyc, "late_f", ("method",), n_boot),
        "premature_rate_ci": M.household_bootstrap_ci(cyc, "premature_f", ("method",), n_boot),
        "late_rate_by_scenario_ci": M.household_bootstrap_ci(cyc, "late_f", ("method", "scenario"), n_boot),
        "fallback": M.fallback_summary(opps),
    }
    return opps, cyc, tables, excluded
