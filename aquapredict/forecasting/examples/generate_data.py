"""Deterministic SYNTHETIC household simulator (NOT a validated physical model).

Every number produced here is simulated. Nothing is real Atomberg telemetry.

How the target is defined
-------------------------
Each (household, filter stage) accumulates a hidden "effective load":
    load_day = usage_litres * wear_multiplier * (1 + tds_coef[stage] * (tds_in/300 - 1))
The filter is replaced on the first day cumulative load >= capacity[stage] *
replace_at_fraction (configurable replacement criterion). The default 0.80 means
"replace when the proxy filter health = 1 - load/capacity falls to 20 %", matching the
project's 20 % health replacement threshold. Use --replace-at-fraction 1.0 for
"replace at 0 % health" instead.
    remaining_days(day) = replacement_day - day      (>= 0)
The sensors are noisy functions of that hidden load: pressure and flow fall and
TDS rejection degrades as the filter loads up, so the target is learnable but
not trivially readable from one column.

Injected data problems: absent sensors (per household), random missing readings,
spikes, a persistent TDS shift in some households (a "source change"), and
occasional flat-lined pressure sensors.

Usage:  python examples/generate_data.py --households 120 --seed 42
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd


STAGES = ["sediment", "carbon", "ro_membrane"]
MODES = ["ro", "ro_uv", "mineral"]


@dataclass
class SimConfig:
    n_households: int = 60
    seed: int = 42
    start_date: str = "2026-01-01"
    max_days: int = 500
    replace_at_fraction: float = 0.80  # replacement criterion: fraction of capacity consumed (0.80 = 20 % health left)
    capacity_litres: Dict[str, float] = field(
        default_factory=lambda: {"sediment": 5000.0, "carbon": 7000.0, "ro_membrane": 9000.0}
    )
    tds_coef: Dict[str, float] = field(
        default_factory=lambda: {"sediment": 0.3, "carbon": 0.5, "ro_membrane": 1.0}
    )


def _household_stage(rng, cfg, hh, stage, p):
    n = cfg.max_days
    days = np.arange(n)
    weekday = (days % 7) < 5
    usage = rng.gamma(shape=6.0, scale=p["usage_mean"] / 6.0, size=n) * np.where(weekday, 1.0, 1.15)
    tds_true = np.full(n, p["tds0"], dtype=float)
    if p["shift_day"] is not None:
        tds_true[p["shift_day"]:] *= p["shift_factor"]
    tds_true *= 1 + rng.normal(0, 0.02, n)
    wear = p["wear"][stage]
    load = usage * wear * np.clip(1 + cfg.tds_coef[stage] * (tds_true / 300 - 1), 0.1, None)
    cum = np.cumsum(load)
    cap = cfg.capacity_litres[stage] * cfg.replace_at_fraction
    hit = np.nonzero(cum >= cap)[0]
    if len(hit) == 0:
        return None  # filter never reached end of life in the window -> drop (no censored targets)
    end = int(hit[0])
    m = end + 1
    frac = np.clip(cum[:m] / cap, 0, 1)
    rejection = 0.96 - 0.12 * frac**2
    sn = p["noise"]
    ts = pd.date_range(cfg.start_date, periods=m, freq="D")
    temp = p["temp"] + 4 * np.sin(2 * np.pi * days[:m] / 365) + rng.normal(0, 0.8, m)
    df = pd.DataFrame({
        "timestamp": ts, "household_id": hh, "filter_stage": stage,
        "usage_litres": usage[:m] * (1 + rng.normal(0, 0.03 * sn, m)),
        "tds_in": tds_true[:m] * (1 + rng.normal(0, 0.03 * sn, m)),
        "tds_out": tds_true[:m] * (1 - rejection) * (1 + rng.normal(0, 0.05 * sn, m)),
        "pressure": p["pressure"] * (1 - 0.30 * frac) + rng.normal(0, 0.05 * sn, m),
        "flow_rate": p["flow"] * (1 - 0.35 * frac) + rng.normal(0, 0.03 * sn, m),
        "temperature": temp,
        "purification_mode": rng.choice(MODES, size=m, p=[0.6, 0.3, 0.1]),
        "remaining_days": (end - np.arange(m)).astype(float),
    })
    for col in ["usage_litres", "tds_in", "tds_out", "pressure", "flow_rate", "temperature"]:
        v = df[col].to_numpy().copy()
        spikes = rng.random(m) < p["spike_prob"]
        if col in ("tds_in", "pressure", "flow_rate"):
            v[spikes] *= rng.uniform(2.5, 4.0, spikes.sum())
        v[rng.random(m) < p["miss_prob"]] = np.nan
        df[col] = v
    for col in ("pressure", "flow_rate", "temperature"):
        if col in p["absent"]:
            df[col] = np.nan  # sensor not fitted in this household's hardware
    if p["flatline_start"] is not None and "pressure" not in p["absent"] and m > p["flatline_start"] + 12:
        s = p["flatline_start"]
        df.loc[s:s + 11, "pressure"] = df.loc[s, "pressure"]
    return df


def generate_dataset(n_households: int = 60, seed: int = 42, config: SimConfig | None = None) -> pd.DataFrame:
    cfg = config or SimConfig(n_households=n_households, seed=seed)
    rng = np.random.default_rng(cfg.seed)
    frames = []
    for i in range(cfg.n_households):
        hh = f"H{i:03d}"
        shifted = rng.random() < 0.3
        p = {
            "tds0": rng.uniform(150, 600), "usage_mean": rng.uniform(25, 110),
            "temp": rng.uniform(18, 34), "pressure": rng.uniform(1.6, 3.4),
            "flow": rng.uniform(1.0, 2.0), "noise": rng.uniform(0.6, 1.6),
            "miss_prob": rng.uniform(0.0, 0.05), "spike_prob": rng.uniform(0.0, 0.006),
            "wear": {s: float(rng.lognormal(0, 0.2)) for s in STAGES},
            "shift_day": int(rng.integers(40, 140)) if shifted else None,
            "shift_factor": float(rng.choice([0.6, 1.5])),
            "absent": {c for c in ("pressure", "flow_rate", "temperature") if rng.random() < 0.1},
            "flatline_start": int(rng.integers(20, 90)) if rng.random() < 0.15 else None,
        }
        for stage in STAGES:
            part = _household_stage(rng, cfg, hh, stage, p)
            if part is not None:
                frames.append(part)
    df = pd.concat(frames, ignore_index=True)
    return df.sort_values(["household_id", "filter_stage", "timestamp"]).reset_index(drop=True)


# Household-level split used by BOTH train_demo.py and run_demo.py (so the demo only ever uses
# households the model never saw): train / calibration / development / test.
DEMO_FRACTIONS = (0.45, 0.20, 0.15, 0.20)
DEMO_HOUSEHOLDS = 120


def demo_splits(n_households: int = DEMO_HOUSEHOLDS, seed: int = 42, replace_at_fraction: float = 0.80):
    from aquapredict.forecasting.features.engineering import split_by_household
    df = generate_dataset(n_households, seed, SimConfig(n_households=n_households, seed=seed,
                                                        replace_at_fraction=replace_at_fraction))
    return df, split_by_household(df, seed=seed, fractions=DEMO_FRACTIONS)


def rows_to_history(group: pd.DataFrame, n: int | None = None) -> list:
    """Convert rows of ONE (household, stage) into the list-of-dicts expected by predict().
    Drops the target and id columns; NaN -> None; timestamps -> ISO strings."""
    g = group.iloc[:n] if n is not None else group
    g = g.drop(columns=[c for c in ("remaining_days", "household_id", "filter_stage") if c in g.columns])
    recs = []
    for r in g.to_dict("records"):
        r["timestamp"] = pd.Timestamp(r["timestamp"]).isoformat()
        recs.append({k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in r.items()})
    return recs


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate SYNTHETIC purifier data.")
    ap.add_argument("--households", type=int, default=DEMO_HOUSEHOLDS)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--replace-at-fraction", type=float, default=0.80)
    ap.add_argument("--out", default="examples/data/synthetic_readings.csv")
    a = ap.parse_args()
    df = generate_dataset(a.households, a.seed, SimConfig(n_households=a.households, seed=a.seed,
                                                           replace_at_fraction=a.replace_at_fraction))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"[SYNTHETIC DATA] wrote {len(df)} rows, {df.household_id.nunique()} households -> {out}")


if __name__ == "__main__":
    main()
