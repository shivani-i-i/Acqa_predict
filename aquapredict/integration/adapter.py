"""Translate Person 1 telemetry (simulator schema) into Person 2's input schema, and back.

Column mapping (Person 1 -> Person 2):
    usage_L          -> usage_litres
    feed_tds_ppm     -> tds_in
    out_tds_ppm      -> tds_out
    temp_C           -> temperature
    flow_Lpm         -> flow_rate
    date             -> timestamp
    inlet_bar, dp_sediment_bar -> pressure   (see PRESSURE_SOURCES)
    (no equivalent) purification_mode -> missing

PRESSURE: Person 2's `pressure` is a single channel (its simulator's pressure FALLS as filters load). Person 1 has the
inlet pressure and the sediment-stage pressure DROP (which RISES with clogging). Options (PRESSURE_SOURCES):
  'dp'         (DEFAULT) the raw sediment pressure drop. Held-out result: median error 30.0 days, vs 59.1 days for 'downstream'.
  'downstream' inlet_bar - dp_sediment_bar. Looks physically natural, but the ~2.4 bar level moves with household and day
               (low-pressure episodes) far more than the 0.15-0.27 bar drop, which buries the wear signal.
  'inlet'      no wear signal at all.
'dp' did NOT trip Person 2's near-zero-pressure rule in practice (no forecasts withheld for it).
CAVEAT: 'dp' was chosen after seeing held-out results of seed 21 (two real options, large effect); it was then replicated on a
fresh seed (see docs_results/RESULTS.md). It is a modelling assumption about how a real sensor would be wired, not a fact.

Idle (vacation) days carry no sensor readings in Person 1's data. They are omitted here so Person 2's missing-data
and stale-reading checks do not mistake a holiday for a failing sensor (timestamps keep their real gaps).
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from ..schema import STAGES, age_col, repl_col, dtc_col

STAGE_TO_P2 = {"sediment": "sediment", "carbon": "carbon", "ro": "ro_membrane"}
P2_TO_STAGE = {v: k for k, v in STAGE_TO_P2.items()}
PRESSURE_SOURCES = ("downstream", "dp", "inlet")
P2_COLUMNS = ["timestamp", "usage_litres", "tds_in", "tds_out", "pressure", "flow_rate", "temperature",
              "purification_mode"]


def _pressure(df: pd.DataFrame, source: str) -> pd.Series:
    if source == "downstream":
        return df["inlet_bar"] - df["dp_sediment_bar"]
    if source == "dp":
        return df["dp_sediment_bar"]
    if source == "inlet":
        return df["inlet_bar"]
    raise ValueError(f"pressure_source must be one of {PRESSURE_SOURCES}")


def to_p2_readings(frame: pd.DataFrame, pressure_source: str = "dp", drop_idle: bool = True) -> pd.DataFrame:
    """Person 1 rows (any subset of one household) -> Person 2 reading columns. Only visible columns are read."""
    d = frame[frame["op_mode"] != "idle"] if drop_idle else frame
    out = pd.DataFrame({
        "timestamp": pd.to_datetime(d["date"]).to_numpy(),
        "usage_litres": d["usage_L"].to_numpy(float),
        "tds_in": d["feed_tds_ppm"].to_numpy(float),
        "tds_out": d["out_tds_ppm"].to_numpy(float),
        "pressure": _pressure(d, pressure_source).to_numpy(float),
        "flow_rate": d["flow_Lpm"].to_numpy(float),
        "temperature": d["temp_C"].to_numpy(float),
        "purification_mode": None,
    })
    return out[P2_COLUMNS]


def history_to_p2(hist: pd.DataFrame, stage: str, pressure_source: str = "dp") -> list[dict]:
    """A harness history (all rows of one household up to the forecast day) -> Person 2 `history` list.
    Person 2 requires the history to start when the CURRENT filter was installed, so rows before the stage's
    last replacement are cut using the visible age column."""
    age = int(hist[age_col(stage)].iloc[-1])
    cyc = hist.iloc[max(len(hist) - 1 - age, 0):]
    return to_p2_readings(cyc, pressure_source).to_dict("records")


def training_frame(df: pd.DataFrame, stages=STAGES, pressure_source: str = "dp") -> pd.DataFrame:
    """Person 1 telemetry (WITH truth) -> Person 2 long-format training frame.

    One filter life = one group: household_id becomes '<household>~c<cycle>' so Person 2's (household, stage)
    grouping never mixes two filters. Only completed lives (criterion reached) are used; rows run from
    installation to the crossing day; remaining_days = crossing day - day (>= 0). SPLIT BY REAL HOUSEHOLD BEFORE
    calling this (all cycles of a household must stay in the same split)."""
    parts = []
    for hid, g in df.groupby("household_id", sort=False):
        g = g.sort_values("day").reset_index(drop=True)
        p2 = to_p2_readings(g, pressure_source, drop_idle=False)
        for stage in stages:
            rep, dtc = g[repl_col(stage)].to_numpy(), g[dtc_col(stage)].to_numpy()
            rp = np.flatnonzero(rep)
            starts = np.r_[0, rp[rp > 0]]
            ends = np.r_[starts[1:], len(g)]
            for k, (s, e) in enumerate(zip(starts, ends)):
                hit = np.flatnonzero(dtc[s:e] == 0)
                if len(hit) == 0:
                    continue                                  # censored life: no target
                c = int(s + hit[0])
                seg = p2.iloc[s:c + 1].copy()
                keep = (g["op_mode"].to_numpy()[s:c + 1] != "idle")
                seg = seg[keep]
                if len(seg) < 5:
                    continue
                seg["household_id"] = f"{hid}~c{k}"
                seg["filter_stage"] = STAGE_TO_P2[stage]
                seg["remaining_days"] = (c - g["day"].to_numpy()[s:c + 1][keep]).astype(float)
                parts.append(seg)
    if not parts:
        raise ValueError("no completed filter lives to build a training frame from")
    out = pd.concat(parts, ignore_index=True)
    return out[["household_id", "filter_stage"] + P2_COLUMNS + ["remaining_days"]]
