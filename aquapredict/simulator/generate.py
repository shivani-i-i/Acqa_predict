"""Dataset generation (reproducible: household i depends only on (seed, i))."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import numpy as np
import pandas as pd

from ..schema import ALL_COLS
from .config import SimConfig
from .household import simulate_household
from .scenarios import SCENARIOS, DEFAULT_MIX, Scenario


def _resolve_scenario(s):
    if isinstance(s, Scenario):
        return s
    if s not in SCENARIOS:
        raise KeyError(f"unknown scenario '{s}'. Available: {sorted(SCENARIOS)}")
    return SCENARIOS[s]


def scenario_config(scenario: Scenario, base: SimConfig) -> SimConfig:
    d = asdict(base)
    d.update(scenario.overrides)
    return SimConfig.from_dict(d)


def generate_household(index, scenario="normal", cfg=None, law="A", seed=0, extra_faults=(),
                       forced_replacements=None, user_faults=None):
    sc = _resolve_scenario(scenario)
    c = scenario_config(sc, cfg or SimConfig())
    df, meta = simulate_household(index, sc, c, law=law, seed=seed, extra_faults=extra_faults,
                                  forced_replacements=forced_replacements, user_faults=user_faults)
    return df[ALL_COLS], meta


def _assign(mix, seed, index):
    names = list(mix)
    p = np.array([mix[k] for k in names], float)
    p = p / p.sum()
    u = np.random.default_rng(np.random.SeedSequence([int(seed), int(index), 1])).random()
    return names[int(np.searchsorted(np.cumsum(p), u, side="right").clip(0, len(names) - 1))]


def generate_dataset(n_households, mix=None, law="A", seed=0, cfg=None, n_days=None, extra_faults=()):
    """mix: dict scenario->weight, a single scenario name, or None for DEFAULT_MIX.
    Returns (telemetry_df, households_meta_df)."""
    if int(n_households) < 1:
        raise ValueError("n_households must be >= 1")
    base = cfg or SimConfig()
    if n_days is not None:
        d = asdict(base); d["n_days"] = int(n_days); base = SimConfig.from_dict(d)
    if mix is None:
        mix = DEFAULT_MIX
    elif isinstance(mix, (str, Scenario)):
        mix = {_resolve_scenario(mix).name: 1.0}
    custom = {m.name: m for m in mix if isinstance(m, Scenario)}
    dfs, metas = [], []
    for i in range(int(n_households)):
        name = _assign({getattr(k, "name", k): v for k, v in mix.items()}, seed, i)
        sc = custom.get(name) or _resolve_scenario(name)
        df, meta = generate_household(i, sc, base, law, seed, extra_faults)
        dfs.append(df); metas.append(meta)
    return pd.concat(dfs, ignore_index=True), pd.DataFrame(metas)


def main(argv=None):
    from ..export import write_dataset
    ap = argparse.ArgumentParser(description="Generate a reproducible SYNTHETIC AquaPredict dataset")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--law", choices=["A", "B"], default="A")
    ap.add_argument("--scenario", default=None, help="single scenario name (default: mixed)")
    ap.add_argument("--days", type=int, default=None)
    ap.add_argument("--calibrated", action="store_true", help="use SimConfig.calibrated() (recommended for final results)")
    ap.add_argument("--out", default="data/synth")
    a = ap.parse_args(argv)
    cfg = SimConfig.calibrated() if a.calibrated else SimConfig()
    if a.days is not None:
        cfg.n_days = a.days
    df, meta = generate_dataset(a.n, mix=a.scenario, law=a.law, seed=a.seed, cfg=cfg)
    paths = write_dataset(df, meta, cfg, a.out, law=a.law, seed=a.seed)
    print(f"{len(df):,} rows, {df.household_id.nunique()} households -> {paths}")


if __name__ == "__main__":
    main()
