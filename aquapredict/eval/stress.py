"""Stress tests: re-run the evaluation under harder conditions. The forecasters are already trained."""
from __future__ import annotations
import pandas as pd
from ..simulator.generate import generate_dataset
from ..simulator.config import SimConfig
from . import metrics as M
from .harness import evaluate
from .config import EvalConfig


def _drift(mag):
    return [{"kind": "drift", "prob": 1.0, "sensors": ["dp_sediment_bar", "flow_Lpm", "out_tds_ppm"],
             "duration_range": (10, 30), "magnitude_range": (mag, mag), "min_start": 60}]


def variants():
    """name -> kwargs for generate_dataset (controlled 'normal' households + one stressor)."""
    v = {"baseline_lawA": {}, "second_simulator_lawB": {"law": "B"}}
    for r in (0.1, 0.3, 0.5):
        v[f"missing_{int(r * 100)}pct"] = {"cfg": SimConfig(missing_rate=r)}
    for mag in (0.1, 0.25, 0.5):
        v[f"drift_{int(mag * 100)}pct"] = {"extra_faults": _drift(mag)}
    v["source_change_shocks"] = {"cfg": SimConfig(source_change_rate_per_year=3.0, source_change_min=1)}
    return v


def run_stress_suite(forecasters, n_households=40, seed=100, cfg: EvalConfig | None = None, only=None):
    cfg = cfg or EvalConfig()
    rows = []
    for name, kw in variants().items():
        if only and name not in only:
            continue
        df, _ = generate_dataset(n_households, mix="normal", seed=seed, **kw)
        res = evaluate(df, forecasters, cfg=cfg)
        w = M.window_summary(res.opps, cfg)
        lf = M.late_false_summary(M.cycle_outcomes(res.opps, cfg))
        iv = M.interval_summary(res.opps, cfg)
        for m in w.index:
            rows.append({"variant": name, "method": m,
                         "median_abs_err_primary": w.loc[m, "median_abs_err_primary"],
                         "withheld_rate": w.loc[m, "withheld_rate"],
                         "late_rate": lf.loc[m, "late_rate"], "premature_rate": lf.loc[m, "premature_rate"],
                         "coverage": iv.loc[m, "coverage"] if m in getattr(iv, "index", []) else float("nan")})
    return pd.DataFrame(rows)
