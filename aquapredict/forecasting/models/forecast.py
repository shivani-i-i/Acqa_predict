"""Gradient-boosted remaining-life forecaster + baselines (SYNTHETIC evaluation only).

* Preprocessing is a scikit-learn ColumnTransformer inside a Pipeline, so it is
  fitted on TRAINING rows only (ordinal encoding of categories; unknown
  categories at inference become NaN, i.e. "missing").
* HistGradientBoostingRegressor handles NaN natively, so missing sensors are
  never imputed with invented numbers.
* Predictions are clipped to >= 0 days.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from aquapredict.forecasting.features.engineering import (
    CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES, TARGET,
    build_features, validate_columns,
)

# Features whose training range defines "model-supported operating conditions".
RANGE_FEATURES = ["cum_usage_litres", "usage_mean_7d", "tds_in", "tds_out",
                  "pressure", "flow_rate", "temperature"]


@dataclass
class ForecastConfig:
    seed: int = 42
    max_iter: int = 200
    learning_rate: float = 0.06
    min_samples_leaf: int = 30
    range_quantiles: tuple = (0.001, 0.999)
    range_margin: float = 0.25  # extra margin (fraction of span) before flagging


@dataclass
class BaselineConfig:
    """Settings for the three simple baselines (all evaluated on SYNTHETIC data).

    capacity_litres : nominal rated capacity per stage. ASSUMPTION: equal to the simulator's
        nominal capacities, which flatters the baselines (a real rated figure would be less
        accurate). Configure per product.
    health_replace_threshold : the project's replacement rule - replace when the PROXY filter
        health falls to this fraction. Proxy health = 1 - cum_usage_litres / capacity_litres
        (it is NOT a measured health percentage; no real health data exists). Default 0.20 =
        replace at 20 % health. Used by baseline B2.
    volume_rule_fraction : B1 replaces after this fraction of nominal capacity has been used.
        Default 0.80, i.e. the same volume as the 20 % health rule, so all baselines target the
        same replacement event as the simulator (see examples/generate_data.py).
    nominal_daily_litres : B1 assumes this fixed household usage per day.
    """
    capacity_litres: Dict[str, float] = field(
        default_factory=lambda: {"sediment": 5000.0, "carbon": 7000.0, "ro_membrane": 9000.0})
    nominal_daily_litres: float = 60.0
    health_replace_threshold: float = 0.20
    volume_rule_fraction: float = 0.80

    def __post_init__(self):
        if not 0.0 <= self.health_replace_threshold < 1.0:
            raise ValueError("health_replace_threshold must be in [0, 1).")
        if not 0.0 < self.volume_rule_fraction <= 1.0:
            raise ValueError("volume_rule_fraction must be in (0, 1].")


@dataclass
class AlertConfig:
    """Operational definition of the maintenance alert (see evaluate_alerting).

    An alert is raised on a reading when the forecast remaining life is <= alert_horizon_days.
    For each filter life, the FIRST alert defines the warning; its lead time is the TRUE
    remaining days at that moment.
      * late warning     : no alert at all, or lead time < min_lead_days  (too little notice)
      * false alarm      : lead time > max_early_days (premature: alert far too early, so
                           filter life is wasted / the user is bothered needlessly)
      * on time          : everything else
    These three outcomes are mutually exclusive and sum to 1 over filter lives.
    """
    alert_horizon_days: float = 14.0
    min_lead_days: float = 7.0
    max_early_days: float = 28.0

    @classmethod
    def scaled(cls, horizon: float) -> "AlertConfig":
        """Same shape for another horizon: min lead = H/2, max early = 2H."""
        return cls(float(horizon), horizon / 2.0, horizon * 2.0)

    def __post_init__(self):
        if not 0 < self.min_lead_days <= self.alert_horizon_days <= self.max_early_days:
            raise ValueError("need 0 < min_lead_days <= alert_horizon_days <= max_early_days.")


def validate_targets(y) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    if y.size == 0:
        raise ValueError("Training target is empty.")
    if not np.isfinite(y).all():
        raise ValueError("Training target contains NaN or infinite values.")
    if (y < 0).any():
        raise ValueError("Training target contains negative remaining_days.")
    return y


def regression_metrics(y_true, y_pred) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ok = np.isfinite(y_true) & np.isfinite(y_pred)
    if not ok.any():
        return {"n": 0, "mae_days": float("nan"), "median_ae_days": float("nan"), "rmse_days": float("nan")}
    err = y_pred[ok] - y_true[ok]
    return {"n": int(ok.sum()), "mae_days": float(np.mean(np.abs(err))),
            "median_ae_days": float(np.median(np.abs(err))), "rmse_days": float(np.sqrt(np.mean(err**2)))}


def _build_pipeline(cfg: ForecastConfig, numeric_cols: List[str]) -> Pipeline:
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan,
                         encoded_missing_value=np.nan)
    prep = ColumnTransformer(
        [("cat", enc, CATEGORICAL_FEATURES), ("num", "passthrough", numeric_cols)],
        verbose_feature_names_out=False)
    prep.set_output(transform="pandas")
    mask = [True] * len(CATEGORICAL_FEATURES) + [False] * len(numeric_cols)
    model = HistGradientBoostingRegressor(
        max_iter=cfg.max_iter, learning_rate=cfg.learning_rate,
        min_samples_leaf=cfg.min_samples_leaf, categorical_features=mask,
        early_stopping=False, random_state=cfg.seed)
    return Pipeline([("prep", prep), ("model", model)])


class ForecastModel:
    def __init__(self, config: Optional[ForecastConfig] = None):
        self.config = config or ForecastConfig()
        self.pipeline: Optional[Pipeline] = None
        self.feature_columns: List[str] = list(FEATURE_COLUMNS)
        self.training_ranges: Dict[str, List[float]] = {}
        self.known_stages: List[str] = []
        self.metadata: Dict[str, object] = {}
        # Small sample of TRAINING rows (after preprocessing); used only as the SHAP
        # background distribution. Not used for prediction.
        self.background: Optional[pd.DataFrame] = None

    @property
    def is_fitted(self) -> bool:
        return self.pipeline is not None

    def fit(self, train_df: pd.DataFrame) -> "ForecastModel":
        validate_columns(train_df, require_target=True)
        y = validate_targets(train_df[TARGET])
        X = build_features(train_df)
        # A numeric feature that is entirely missing in the training data (sensor never fitted)
        # cannot be learned from (and crashes the booster), so it is excluded - not imputed.
        active_numeric = [c for c in NUMERIC_FEATURES if X[c].notna().any()]
        if not active_numeric:
            raise ValueError("No numeric feature has any observed value in the training data.")
        pipe = _build_pipeline(self.config, active_numeric)
        pipe.fit(X, y)
        lo_q, hi_q = self.config.range_quantiles
        self.training_ranges = {}
        for col in RANGE_FEATURES:
            v = X[col].dropna()
            if len(v):
                self.training_ranges[col] = [float(v.quantile(lo_q)), float(v.quantile(hi_q))]
        self.known_stages = sorted(X["filter_stage"].unique().tolist())
        self.metadata = {"n_train_rows": int(len(X)),
                         "n_train_households": int(train_df["household_id"].nunique()),
                         "train_households": sorted(map(str, train_df["household_id"].unique())),
                         "seed": self.config.seed, "data_origin": "SYNTHETIC",
                         "excluded_all_missing_features": [c for c in NUMERIC_FEATURES if c not in active_numeric]}
        n_bg = min(100, len(X))
        self.background = pipe.named_steps["prep"].transform(
            X.sample(n=n_bg, random_state=self.config.seed)).reset_index(drop=True)
        self.pipeline = pipe  # only assigned after a successful fit
        return self

    def _check_ready(self, X: pd.DataFrame) -> None:
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted; refusing to predict.")
        missing = [c for c in self.feature_columns if c not in X.columns]
        if missing:
            raise ValueError(f"Feature frame is missing columns: {missing}")

    def predict_features(self, X: pd.DataFrame) -> np.ndarray:
        self._check_ready(X)
        return np.clip(self.pipeline.predict(X[self.feature_columns]), 0.0, None)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.predict_features(build_features(df))

    def evaluate(self, df: pd.DataFrame) -> Dict[str, float]:
        validate_columns(df, require_target=True)
        y = validate_targets(df[TARGET])
        return regression_metrics(y, self.predict(df))

    def out_of_range(self, X: pd.DataFrame, row: int = -1) -> List[str]:
        """Human-readable reasons why a feature row is outside the TRAINING support."""
        reasons = []
        r = X.iloc[row]
        if r["filter_stage"] not in self.known_stages:
            reasons.append(f"filter_stage '{r['filter_stage']}' was not seen in training")
        for col, (lo, hi) in self.training_ranges.items():
            val = r[col]
            if pd.isna(val):
                continue
            margin = self.config.range_margin * max(hi - lo, 1e-9)
            if val < lo - margin or val > hi + margin:
                reasons.append(f"{col}={val:.3g} outside training range [{lo:.3g}, {hi:.3g}]")
        return reasons


# ----------------------------- baselines ---------------------------------
def baseline_predictions(X: pd.DataFrame, cfg: Optional[BaselineConfig] = None) -> Dict[str, np.ndarray]:
    """Three simple baselines computed from engineered features (past data only)."""
    cfg = cfg or BaselineConfig()
    cap = X["filter_stage"].map(cfg.capacity_litres).astype(float)
    cum = X["cum_usage_litres"]

    def div(a, b):
        b = b.where(b > 0)
        return (a / b).to_numpy(dtype=float)

    # B1 volume rule: replace after volume_rule_fraction of nominal capacity, fixed nominal usage.
    left_b1 = (cap * cfg.volume_rule_fraction - cum).clip(lower=0)
    b1 = div(left_b1, pd.Series(cfg.nominal_daily_litres, index=X.index))
    # B2 health-threshold proxy: proxy health = 1 - cum/cap; replace when health <= threshold
    # (i.e. cum >= cap * (1 - threshold)); usage rate = lifetime average so far.
    lifetime_rate = cum / (X["days_since_start"] + 1.0)
    left_b2 = (cap * (1.0 - cfg.health_replace_threshold) - cum).clip(lower=0)
    b2 = div(left_b2, lifetime_rate)
    # B3 trend extrapolation: same replacement volume as B1, extrapolating the RECENT 7-day usage rate.
    b3 = div(left_b1, X["usage_mean_7d"])
    return {"volume_rule": b1, "health_threshold_proxy": b2, "trend_extrapolation": b3}


def compare_with_baselines(model: ForecastModel, df: pd.DataFrame,
                           cfg: Optional[BaselineConfig] = None,
                           max_remaining_days: Optional[float] = None) -> Dict[str, Dict[str, float]]:
    """Metrics for the model and the three baselines on the SAME rows (SYNTHETIC data).

    ``max_remaining_days`` restricts the SCORING to rows whose true remaining life is at most
    that many days (a retrospective "near end of life" view). Features are still built from
    the full history, so no information is lost or leaked by the restriction.
    """
    y = validate_targets(df[TARGET])
    X = build_features(df)
    keep = np.ones(len(y), dtype=bool) if max_remaining_days is None else (y <= max_remaining_days)
    out = {"gradient_boosting": regression_metrics(y[keep], model.predict_features(X)[keep])}
    for name, pred in baseline_predictions(X, cfg).items():
        out[name] = regression_metrics(y[keep], pred[keep])
    return out


# ------------------------- late-warning / false-alarm evaluation -------------------------
def evaluate_alerting(df: pd.DataFrame, pred, cfg: Optional[AlertConfig] = None) -> Dict[str, object]:
    """Alert-level evaluation of one forecaster (SYNTHETIC data only).

    ``df`` needs household_id, filter_stage, timestamp and the true remaining_days; ``pred`` is the
    forecast for each row (NaN = no forecast -> no alert). Readings are processed in time order per
    filter life (household, stage), exactly as a daily monitor would see them.

    Life-level outcomes (see AlertConfig): late_warning_rate, false_alarm_rate, on_time_rate
    (sum to 1), plus no_alert_rate (a subset of late warnings) and median/p10 lead time.
    Row-level rates: false_alarm_row_rate = share of readings with MORE than max_early_days left
    that raise an alert; late_row_rate = share of readings with at most min_lead_days left that
    do NOT raise one.
    """
    cfg = cfg or AlertConfig()
    d = df[["household_id", "filter_stage", "timestamp", TARGET]].copy()
    d["pred"] = np.asarray(pred, dtype=float)
    d["alert"] = (d["pred"] <= cfg.alert_horizon_days).fillna(False) & d["pred"].notna()
    d = d.sort_values(["household_id", "filter_stage", "timestamp"], kind="mergesort")
    leads, n_lives, no_alert = [], 0, 0
    for _, g in d.groupby(["household_id", "filter_stage"], sort=False):
        n_lives += 1
        hit = g[g["alert"]]
        if hit.empty:
            no_alert += 1
        else:
            leads.append(float(hit[TARGET].iloc[0]))
    leads_a = np.asarray(leads, dtype=float)
    late = no_alert + int((leads_a < cfg.min_lead_days).sum())
    false_alarm = int((leads_a > cfg.max_early_days).sum())
    on_time = n_lives - late - false_alarm
    r = d[TARGET].to_numpy(dtype=float)
    alert = d["alert"].to_numpy()
    far, near = r > cfg.max_early_days, r <= cfg.min_lead_days

    def rate(a, b):
        return float(a / b) if b else None

    return {"n_lives": n_lives, "late_warning_rate": rate(late, n_lives),
            "false_alarm_rate": rate(false_alarm, n_lives), "on_time_rate": rate(on_time, n_lives),
            "no_alert_rate": rate(no_alert, n_lives),
            "median_lead_days": float(np.median(leads_a)) if len(leads_a) else None,
            "p10_lead_days": float(np.quantile(leads_a, 0.10)) if len(leads_a) else None,
            "false_alarm_row_rate": rate(int((alert & far).sum()), int(far.sum())),
            "late_row_rate": rate(int((~alert & near).sum()), int(near.sum())),
            "definition": {"alert_horizon_days": cfg.alert_horizon_days, "min_lead_days": cfg.min_lead_days,
                           "max_early_days": cfg.max_early_days}}


def compare_alerting(model: ForecastModel, df: pd.DataFrame, baseline_cfg: Optional[BaselineConfig] = None,
                     alert_cfg: Optional[AlertConfig] = None) -> Dict[str, Dict[str, object]]:
    """evaluate_alerting for the model and all three baselines on the SAME households/rows."""
    df = df.reset_index(drop=True)
    X = build_features(df)
    preds = {"gradient_boosting": model.predict_features(X), **baseline_predictions(X, baseline_cfg)}
    return {name: evaluate_alerting(df, p, alert_cfg) for name, p in preds.items()}
