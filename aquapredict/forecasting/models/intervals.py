"""Split conformal prediction intervals for remaining_days.

Procedure
---------
1. The forecaster is fitted on TRAINING households.
2. Absolute residuals |y - y_hat| are computed on CALIBRATION households that
   the model has never seen (enforced: overlap raises ValueError).
3. q = the ceil((n+1)*coverage)-th smallest residual (finite-sample correction).
   Interval = [max(y_hat - q, 0), y_hat + q].

Honest caveats
--------------
* Marginal coverage ~ ``coverage`` holds only if future data are exchangeable
  with the calibration data. Under distribution shift (new households, new
  hardware, seasons, source changes) coverage can degrade. It is NOT guaranteed.
* Default ("absolute") intervals have constant width (before clipping at 0). Residuals here grow with
  the remaining horizon and differ by filter stage, so a pooled constant width over-covers some
  groups and under-covers others. Two optional, still split-conformal variants address that:
  ``per_stage`` (one quantile per filter stage, a Mondrian-style partition) and ``scaled``
  (score = |y - yhat| / (yhat + scale_offset_days); width grows with the forecast).
  Which variant is the default was chosen on NON-TEST households only (repeated household resplits,
  examples/investigate_intervals.py); the final test households are not used for that choice.
* Rows of one household are correlated, so the effective calibration size is closer to the number of
  calibration HOUSEHOLDS than to the number of rows; the quantile is noisy when few households are used.
* "Interval support" here (calibration-range) is a different question from
  "model support" (training-range, in forecast.py). Both are reported separately.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from aquapredict.forecasting.features.engineering import TARGET, build_features
from aquapredict.forecasting.models.forecast import RANGE_FEATURES, ForecastModel, validate_targets


@dataclass
class ConformalConfig:
    coverage: float = 0.90
    min_calibration_samples: int = 200
    min_samples_per_stage: int = 30
    range_quantiles: tuple = (0.001, 0.999)
    range_margin: float = 0.25
    per_stage: bool = True           # DEFAULT chosen on non-test households; one conformal quantile per filter stage
    scaled: bool = True              # DEFAULT chosen on development households; normalised score |y - yhat| / (yhat + scale_offset_days)
    scale_offset_days: float = 14.0  # keeps the scale positive when the forecast is near 0


def conformal_quantile(scores, coverage: float):
    """Finite-sample split-conformal quantile: the ceil((n+1)*coverage)-th smallest score.

    Returns (quantile or None, reason or None). None when n is too small for the requested
    coverage (the index would exceed n), in which case no valid finite quantile exists.
    """
    s = np.sort(np.asarray(scores, dtype=float))
    n = len(s)
    k = math.ceil((n + 1) * coverage)
    if n == 0 or k > n:
        return None, f"{n} samples cannot support {coverage:.0%} coverage (finite-sample quantile index exceeds n)"
    return float(s[k - 1]), None


class ConformalCalibrator:
    def __init__(self, config: Optional[ConformalConfig] = None):
        self.config = config or ConformalConfig()
        if not 0 < self.config.coverage < 1:
            raise ValueError("coverage must be in (0, 1).")
        self.quantile: Optional[float] = None   # days (absolute) or relative factor (scaled)
        self.stage_quantiles: Dict[str, Optional[float]] = {}
        self.n_calibration = 0
        self.n_calibration_households = 0
        self.stage_counts: Dict[str, int] = {}
        self.calibration_ranges: Dict[str, List[float]] = {}
        self.insufficient_reason: Optional[str] = None
        self.fitted = False

    # -------------------------------------------------------------- fit
    def fit(self, model: ForecastModel, calib_df: pd.DataFrame) -> "ConformalCalibrator":
        if not model.is_fitted:
            raise RuntimeError("Fit the forecaster before calibrating.")
        overlap = set(map(str, calib_df["household_id"].unique())) & set(model.metadata.get("train_households", []))
        if overlap:
            raise ValueError(f"Calibration households overlap training households: {sorted(overlap)[:5]}")
        y = validate_targets(calib_df[TARGET])
        X = build_features(calib_df)
        pred = model.predict_features(X)
        scores = self._score(y, pred)
        n = len(scores)
        self.n_calibration = n
        self.n_calibration_households = int(calib_df["household_id"].nunique())
        self.insufficient_reason = None
        if n < self.config.min_calibration_samples:
            self.insufficient_reason = (f"only {n} calibration samples; need at least "
                                        f"{self.config.min_calibration_samples}")
            self.quantile = None
        else:
            self.quantile, self.insufficient_reason = conformal_quantile(scores, self.config.coverage)
        self.stage_quantiles = {}
        if self.config.per_stage:
            stages = X["filter_stage"].astype(str).to_numpy()
            for st in sorted(set(stages)):
                sc = scores[stages == st]
                q, _ = (conformal_quantile(sc, self.config.coverage)
                        if len(sc) >= self.config.min_samples_per_stage else (None, None))
                self.stage_quantiles[st] = q
        self.stage_counts = {str(k_): int(v) for k_, v in X["filter_stage"].value_counts().items()}
        lo_q, hi_q = self.config.range_quantiles
        self.calibration_ranges = {}
        for col in RANGE_FEATURES:
            v = X[col].dropna()
            if len(v):
                self.calibration_ranges[col] = [float(v.quantile(lo_q)), float(v.quantile(hi_q))]
        self.fitted = True
        return self

    # ------------------------------------------------------- inference
    def unsupported_reasons(self, X: pd.DataFrame, row: int = -1) -> List[str]:
        """Reasons the CALIBRATION set does not justify an interval for this row."""
        reasons = []
        r = X.iloc[row]
        stage = str(r["filter_stage"])
        if self.stage_counts.get(stage, 0) < self.config.min_samples_per_stage:
            reasons.append(f"only {self.stage_counts.get(stage, 0)} calibration samples for stage '{stage}' "
                           f"(need {self.config.min_samples_per_stage})")
        for col, (lo, hi) in self.calibration_ranges.items():
            val = r[col]
            if pd.isna(val):
                continue
            margin = self.config.range_margin * max(hi - lo, 1e-9)
            if val < lo - margin or val > hi + margin:
                reasons.append(f"{col}={val:.3g} outside calibration range [{lo:.3g}, {hi:.3g}]")
        return reasons

    def _score(self, y, pred) -> np.ndarray:
        r = np.abs(np.asarray(y, dtype=float) - np.asarray(pred, dtype=float))
        return r / (np.asarray(pred, dtype=float) + self.config.scale_offset_days) if self.config.scaled else r

    def _half_width(self, pred, q):
        """Half-width in days for forecast(s) ``pred`` given quantile(s) ``q``."""
        return q * (np.asarray(pred, dtype=float) + self.config.scale_offset_days) if self.config.scaled else q

    def interval(self, point: float, X: Optional[pd.DataFrame] = None, row: int = -1) -> Dict[str, object]:
        """Return {"interval": [lo, hi] | None, "reason": str | None}."""
        if not self.fitted:
            return {"interval": None, "reason": "interval calibrator has not been fitted"}
        if self.quantile is None:
            return {"interval": None, "reason": f"calibration insufficient: {self.insufficient_reason}"}
        q = self.quantile
        if self.config.per_stage:
            if X is None:
                return {"interval": None, "reason": "per-stage calibration needs the feature row to know the stage"}
            stage = str(X.iloc[row]["filter_stage"])
            q = self.stage_quantiles.get(stage)
            if q is None:
                return {"interval": None, "reason": f"calibration insufficient: no usable quantile for stage '{stage}'"}
        if X is not None:
            bad = self.unsupported_reasons(X, row)
            if bad:
                return {"interval": None, "reason": "unsupported conditions for interval: " + "; ".join(bad)}
        half = float(self._half_width(point, q))
        lo = max(point - half, 0.0)
        hi = max(point + half, lo)
        return {"interval": [float(lo), float(hi)], "reason": None}

    # ------------------------------------------------------ evaluation
    def evaluate_coverage(self, model: ForecastModel, test_df: pd.DataFrame) -> Dict[str, object]:
        """Empirical coverage / width on held-out households (SYNTHETIC data).

        Reports overall, per filter stage, by TRUE remaining-life bucket (retrospective view) and the
        spread of per-household coverage. Does not apply the calibration-range check (see
        ``unsupported_reasons``); rows whose stage has no usable quantile are excluded and counted.
        """
        if self.quantile is None:
            return {"available": False, "reason": self.insufficient_reason}
        y = validate_targets(test_df[TARGET])
        X = build_features(test_df)
        pred = model.predict_features(X)
        stages = X["filter_stage"].astype(str)
        if self.config.per_stage:
            q = stages.map(self.stage_quantiles).astype(float).to_numpy()
        else:
            q = np.full(len(y), self.quantile, dtype=float)
        ok = np.isfinite(q)
        half = np.where(ok, self._half_width(pred, np.where(ok, q, 0.0)), np.nan)
        lo, hi = np.clip(pred - half, 0, None), pred + half
        covered = (y >= lo) & (y <= hi)
        d = pd.DataFrame({"stage": stages.to_numpy(), "hh": test_df["household_id"].astype(str).to_numpy(),
                          "cov": covered, "w": hi - lo, "y": y})[ok]

        def summ(g):
            return {"n": int(len(g)), "empirical_coverage": float(g["cov"].mean()),
                    "mean_interval_width_days": float(g["w"].mean())}
        bucket = pd.cut(d["y"], [-0.1, 30, 90, np.inf], labels=["<=30 d left", "31-90 d left", ">90 d left"])
        hh = d.groupby("hh")["cov"].mean()
        # uncertainty of the coverage ESTIMATE itself: resample whole households (rows are correlated)
        per_hh = d.groupby("hh")["cov"].agg(["sum", "count"]).to_numpy(dtype=float)
        rng = np.random.default_rng(0)
        boots = []
        for _ in range(500):
            pick = per_hh[rng.integers(0, len(per_hh), len(per_hh))]
            boots.append(pick[:, 0].sum() / pick[:, 1].sum())
        return {"available": True, "target_coverage": self.config.coverage,
                "interval_method": self.method_name(),
                "empirical_coverage": float(d["cov"].mean()),
                "mean_interval_width_days": float(d["w"].mean()), "n_test_rows": int(len(d)),
                "n_rows_without_interval": int((~ok).sum()),
                "by_stage": {k: summ(g) for k, g in d.groupby("stage")},
                "by_true_remaining_bucket": {str(k): summ(g) for k, g in d.groupby(bucket, observed=True)},
                "overall_coverage_household_bootstrap_90pct_range": [float(np.quantile(boots, 0.05)),
                                                                     float(np.quantile(boots, 0.95))],
                "household_coverage": {"n_households": int(len(hh)), "min": float(hh.min()),
                                       "p10": float(hh.quantile(0.1)), "median": float(hh.median()),
                                       "share_at_or_above_target_minus_10pts":
                                           float((hh >= self.config.coverage - 0.10).mean())},
                "note": "SYNTHETIC test households; coverage is not guaranteed on future/real data."}

    def method_name(self) -> str:
        return ("per-stage " if self.config.per_stage else "pooled ") + \
               ("scaled " if self.config.scaled else "absolute-residual ") + "split conformal"

    # --------------------------------------------------- serialisation
    def to_dict(self) -> Dict[str, object]:
        return {"config": asdict(self.config), "quantile": self.quantile, "n_calibration": self.n_calibration,
                "stage_quantiles": self.stage_quantiles, "n_calibration_households": self.n_calibration_households,
                "stage_counts": self.stage_counts, "calibration_ranges": self.calibration_ranges,
                "insufficient_reason": self.insufficient_reason, "fitted": self.fitted}

    @classmethod
    def from_dict(cls, d: Dict[str, object]) -> "ConformalCalibrator":
        cfg = dict(d["config"])
        cfg["range_quantiles"] = tuple(cfg["range_quantiles"])
        obj = cls(ConformalConfig(**cfg))
        obj.quantile = d["quantile"]; obj.n_calibration = d["n_calibration"]
        obj.stage_quantiles = d.get("stage_quantiles", {}); obj.n_calibration_households = d.get("n_calibration_households", 0)
        obj.stage_counts = d["stage_counts"]; obj.calibration_ranges = d["calibration_ranges"]
        obj.insufficient_reason = d["insufficient_reason"]; obj.fitted = d["fitted"]
        return obj
