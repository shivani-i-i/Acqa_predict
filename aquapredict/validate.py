"""Schema validation for telemetry tables (synthetic now, real telemetry later).
Returns a list of problems; empty list = valid. Does NOT silently repair anything."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .schema import VISIBLE_COLS, TRUTH_COLS, STAGES, OP_MODES, repl_col, age_col, vol_col

RANGES = {"usage_L": (0, 1000), "feed_tds_ppm": (0, 20000), "out_tds_ppm": (0, 20000), "temp_C": (-10, 70),
          "inlet_bar": (0, 20), "flow_Lpm": (0, 10), "dp_sediment_bar": (0, 20)}


def validate_telemetry(df: pd.DataFrame, require_truth=False) -> list[str]:
    problems = []
    need = list(VISIBLE_COLS) + (list(TRUTH_COLS) + ["scenario"] if require_truth else [])
    missing = [c for c in need if c not in df.columns]
    if missing:
        return [f"missing columns: {missing}"]
    if df.empty:
        return ["table is empty"]
    if df.duplicated(["household_id", "day"]).any():
        problems.append("duplicate (household_id, day) rows")
    bad_modes = set(df["op_mode"].dropna().unique()) - set(OP_MODES)
    if bad_modes:
        problems.append(f"unknown op_mode values: {sorted(bad_modes)}")
    for hid, g in df.groupby("household_id", sort=False):
        d = g["day"].to_numpy()
        if not (np.diff(d) == 1).all() or d[0] != 0:
            problems.append(f"{hid}: days must be contiguous integers starting at 0")
            break
    for col, (lo, hi) in RANGES.items():
        v = df[col].to_numpy(float)
        v = v[np.isfinite(v)]
        if len(v) and (v.min() < lo or v.max() > hi):
            problems.append(f"{col} outside plausible range [{lo}, {hi}] (min {v.min():.3g}, max {v.max():.3g})")
    for s in STAGES:
        if (df[age_col(s)] < 0).any() or (df[vol_col(s)] < 0).any():
            problems.append(f"negative age/volume for stage {s}")
        if df[repl_col(s)].dtype != bool:
            problems.append(f"{repl_col(s)} must be boolean")
    if not np.issubdtype(df["day"].dtype, np.integer):
        problems.append("day must be an integer column")
    return problems


def assert_valid(df, require_truth=False):
    p = validate_telemetry(df, require_truth)
    if p:
        raise ValueError("invalid telemetry:\n  - " + "\n  - ".join(p))
