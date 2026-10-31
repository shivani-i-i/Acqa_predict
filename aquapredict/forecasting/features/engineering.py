"""Leakage-safe feature engineering for filter-life forecasting.

ALL DATA IN THIS PROJECT IS SYNTHETIC. See README.

Input schema (one row = one reading of one filter stage of one household)
------------------------------------------------------------------------
Required : timestamp, household_id, filter_stage, usage_litres
Optional : tds_in, tds_out, pressure, flow_rate, temperature, purification_mode
Target   : remaining_days  (TRAINING TARGET ONLY - never used as an input)

Assumptions (each is also a limitation)
---------------------------------------
A1. The history supplied for a (household, filter_stage) starts when the
    current filter was installed, so ``cum_usage_litres`` is litres since
    installation. If history is truncated this feature is underestimated.
A2. Every feature at time t uses only readings with timestamp <= t of the
    same household and filter stage. Trailing windows never look forward.
A3. Optional sensors that are absent become NaN columns. We never invent
    values. Gradient boosting handles NaN natively (no imputer needed).
A4. Physically impossible values (negative, non-finite) become NaN. Merely
    unusual values are kept; the sensor-fault checks / gating deal with them.
A5. Missing ``usage_litres`` contributes 0 to cumulative/window sums (we cannot
    know the true value), so sums are lower bounds when readings are missing.
A6. ``purification_mode`` missing -> explicit category "missing".
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

TARGET = "remaining_days"
REQUIRED_COLUMNS = ["timestamp", "household_id", "filter_stage", "usage_litres"]
OPTIONAL_NUMERIC = ["tds_in", "tds_out", "pressure", "flow_rate", "temperature"]
CATEGORICAL_FEATURES = ["filter_stage", "purification_mode"]
NUMERIC_FEATURES = [
    "usage_litres", "tds_in", "tds_out", "pressure", "flow_rate", "temperature",
    "tds_rejection", "cum_usage_litres", "usage_7d", "usage_30d", "usage_mean_7d",
    "tds_in_mean_7d", "tds_rejection_change_7d", "pressure_change_7d",
    "flow_change_7d", "flow_pressure_ratio", "days_since_start",
]
FEATURE_COLUMNS: List[str] = CATEGORICAL_FEATURES + NUMERIC_FEATURES
GROUP_KEYS = ["household_id", "filter_stage"]


@dataclass(frozen=True)
class FeatureConfig:
    short_window_days: int = 7
    long_window_days: int = 30


def validate_columns(df: pd.DataFrame, require_target: bool = False) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if require_target and TARGET not in df.columns:
        missing.append(TARGET)
    if missing:
        raise ValueError(f"Missing required column(s): {missing}")


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    if out["timestamp"].isna().any():
        raise ValueError("Unparseable or missing timestamp values are not allowed.")
    for col in ["usage_litres"] + OPTIONAL_NUMERIC:
        if col not in out.columns:
            out[col] = np.nan  # A3: absent sensor stays NaN
        out[col] = pd.to_numeric(out[col], errors="coerce").astype(float)
        out[col] = out[col].where(np.isfinite(out[col]))
        out.loc[out[col] < 0, col] = np.nan  # A4
    if "purification_mode" not in out.columns:
        out["purification_mode"] = "missing"
    out["purification_mode"] = out["purification_mode"].fillna("missing").astype(str)
    out["filter_stage"] = out["filter_stage"].astype(str)
    return out


def _safe_ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    valid = den > 0
    res = pd.Series(np.nan, index=num.index, dtype=float)
    res[valid] = num[valid] / den[valid]
    return res


def _past_mean_change(s: pd.Series, window: str) -> pd.Series:
    """current value minus mean of PREVIOUS readings inside the window (A2)."""
    prev_mean = s.shift(1).rolling(window, min_periods=1).mean()
    return s - prev_mean


def _group_features(g: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    g = g.sort_values("timestamp", kind="mergesort")
    idx = g.index
    ts = g.set_index("timestamp")
    sw, lw = f"{cfg.short_window_days}D", f"{cfg.long_window_days}D"
    usage = ts["usage_litres"].fillna(0.0)  # A5
    rejection = ts["tds_rejection"]
    feats = pd.DataFrame(index=ts.index)
    feats["cum_usage_litres"] = usage.cumsum()
    feats["usage_7d"] = usage.rolling(sw, min_periods=1).sum()
    feats["usage_30d"] = usage.rolling(lw, min_periods=1).sum()
    feats["usage_mean_7d"] = usage.rolling(sw, min_periods=1).mean()
    feats["tds_in_mean_7d"] = ts["tds_in"].rolling(sw, min_periods=1).mean()
    feats["tds_rejection_change_7d"] = _past_mean_change(rejection, sw)
    feats["pressure_change_7d"] = _past_mean_change(ts["pressure"], sw)
    feats["flow_change_7d"] = _past_mean_change(ts["flow_rate"], sw)
    feats["days_since_start"] = (ts.index - ts.index[0]).total_seconds() / 86400.0
    feats.index = idx
    return feats


def build_features(df: pd.DataFrame, config: FeatureConfig | None = None) -> pd.DataFrame:
    """Return a DataFrame[FEATURE_COLUMNS] aligned to ``df.index``.

    The target column, if present, is ignored and never copied to the output.
    """
    cfg = config or FeatureConfig()
    validate_columns(df)
    data = _clean(df)
    data["tds_rejection"] = np.where(
        (data["tds_in"] > 0) & data["tds_out"].notna(),
        1.0 - data["tds_out"] / data["tds_in"].where(data["tds_in"] > 0),
        np.nan,
    )
    data["flow_pressure_ratio"] = _safe_ratio(data["flow_rate"], data["pressure"])
    parts = [_group_features(g, cfg) for _, g in data.groupby(GROUP_KEYS, sort=False)]
    derived = pd.concat(parts).reindex(df.index) if parts else pd.DataFrame(index=df.index)
    out = pd.concat([data.drop(columns=[c for c in derived.columns if c in data.columns]), derived], axis=1)
    out = out[FEATURE_COLUMNS].copy()
    assert TARGET not in out.columns
    return out


def split_by_household(
    df: pd.DataFrame,
    seed: int = 42,
    fractions: Tuple[float, ...] = (0.6, 0.2, 0.2),
) -> Dict[str, pd.DataFrame]:
    """Split WHOLE households (never single rows) into disjoint sets.

    3 fractions -> train / calibration / test.
    4 fractions -> train / calibration / development / test, where
      * calibration : fits the conformal quantile,
      * development : used ONLY to choose between modelling / interval options,
      * test        : the final, untouched evaluation set (look at it once).
    """
    if "household_id" not in df.columns:
        raise ValueError("household_id column is required for splitting.")
    if len(fractions) not in (3, 4):
        raise ValueError("fractions must have 3 (train, calibration, test) or 4 "
                         "(train, calibration, development, test) entries.")
    if abs(sum(fractions) - 1.0) > 1e-9 or any(f <= 0 for f in fractions):
        raise ValueError("fractions must be positive and sum to 1.")
    ids = np.array(sorted(df["household_id"].unique()))
    if len(ids) < len(fractions):
        raise ValueError(f"Need at least {len(fractions)} households to make {len(fractions)} splits.")
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    n = len(ids)
    names = ["train", "calibration", "test"] if len(fractions) == 3 else ["train", "calibration", "development", "test"]
    sizes = [max(1, int(round(n * f))) for f in fractions[1:]]
    n_train = n - sum(sizes)
    if n_train < 1:
        raise ValueError("Not enough households for the requested split.")
    bounds = np.cumsum([n_train] + sizes)
    parts = np.split(ids, bounds[:-1])
    return {k: df[df["household_id"].isin(v)].copy() for k, v in zip(names, parts)}
