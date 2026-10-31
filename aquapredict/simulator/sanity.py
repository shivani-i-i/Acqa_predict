"""Quick realism / balance report for a generated dataset. python -m aquapredict.simulator.sanity --n 200"""
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
from ..schema import STAGES
from .generate import generate_dataset


def sanity_report(df, meta=None) -> dict:
    out = {"households": int(df.household_id.nunique()), "rows": int(len(df)),
           "scenario_households": df.groupby("scenario").household_id.nunique().to_dict()}
    stage = {}
    for s in STAGES:
        life, vol = [], []
        for _, g in df.groupby("household_id", sort=False):
            d = g[f"true_dtc_{s}"].to_numpy()
            idx = np.flatnonzero(d == 0)
            life += list(g[f"age_{s}_d"].to_numpy()[idx]); vol += list(g[f"cum_vol_{s}_L"].to_numpy()[idx])
        life, vol = np.array(life), np.array(vol)
        stage[s] = {"completed_cycles": len(life),
                    "life_days_p10_med_p90": [float(np.percentile(life, q)) for q in (10, 50, 90)] if len(life) else None,
                    "volume_L_p10_med_p90": [float(np.percentile(vol, q)) for q in (10, 50, 90)] if len(vol) else None}
    out["stages"] = stage
    fl = rj = 0
    for _, g in df.groupby("household_id", sort=False):
        for i in np.flatnonzero(g.true_dtc_ro.to_numpy() == 0):
            f = g.true_flux_rel.iloc[i] <= 0.7 + 1e-9
            r = g.true_rejection.iloc[i] <= 0.85 + 1e-9
            fl += f and not r; rj += r and not f
    out["ro_crossing_cause"] = {"flux_first": int(fl), "rejection_first": int(rj)}
    out["fault_days_by_type"] = df[df.fault_active].fault_type.value_counts().to_dict()
    out["share_idle_days"] = float((df.op_mode == "idle").mean())
    out["share_low_inlet_days"] = float((df.op_mode == "low_inlet").mean())
    sens = ["dp_sediment_bar", "flow_Lpm", "out_tds_ppm", "feed_tds_ppm", "temp_C", "inlet_bar"]
    act = df[df.op_mode != "idle"]
    out["missing_share_active_days"] = {s: float(act[s].isna().mean()) for s in sens}
    out["overdue_row_share"] = float((df[[f"true_dtc_{s}" for s in STAGES]] < 0).any(axis=1).mean())
    cyc = df[df.true_dtc_sediment == 0].groupby("scenario").size()
    out["sediment_cycle_share_by_scenario"] = (cyc / cyc.sum()).round(3).to_dict()
    out["household_share_by_scenario"] = (df.groupby("scenario").household_id.nunique() / df.household_id.nunique()).round(3).to_dict()
    out["source_changes_per_household"] = float(df.source_change_flag.sum() / df.household_id.nunique())
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=100); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--law", choices=["A", "B"], default="A"); ap.add_argument("--scenario", default=None)
    a = ap.parse_args()
    df, meta = generate_dataset(a.n, mix=a.scenario, law=a.law, seed=a.seed)
    import json
    print(json.dumps(sanity_report(df, meta), indent=2, default=str))
