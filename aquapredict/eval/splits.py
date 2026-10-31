"""Household-level splits (never row-level)."""
from __future__ import annotations
import numpy as np


def split_households(household_ids, seed=0, fractions=(0.5, 0.2, 0.3)):
    """-> dict(train, calibration, test) of disjoint household-id lists."""
    ids = np.array(sorted(set(household_ids)))
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    n = len(ids)
    a = int(round(fractions[0] * n))
    b = a + int(round(fractions[1] * n))
    out = {"train": ids[:a].tolist(), "calibration": ids[a:b].tolist(), "test": ids[b:].tolist()}
    assert_disjoint(out)
    return out


def assert_disjoint(split):
    seen = set()
    for k, v in split.items():
        s = set(v)
        if seen & s:
            raise AssertionError(f"household leakage between splits (in '{k}')")
        seen |= s


def subset(df, ids):
    return df[df["household_id"].isin(set(ids))].reset_index(drop=True)
