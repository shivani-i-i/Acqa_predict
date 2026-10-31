"""REFERENCE forecaster v2 (starting point for Person 2).

Pipeline per stage:
  1. features (leak-safe, past rows only)                                  -> features.py
  2. gradient-boosted regression of log(1+days to criterion)
  3. MODEL SELECTION (abstract: "kept only if it beats the baseline on unseen households"):
     cross-fitted (GroupKFold by household) out-of-fold error of the GBM is compared with the best simple
     baseline available from the same features; the GBM is used only where it wins.
  4. scale model: a second GBM predicts the expected |log error| from out-of-fold residuals
  5. normalised split-conformal intervals calibrated on SEPARATE households:
        score = |resid| / sigma_hat(x),   interval = point +/- q * sigma_hat(x)   (log space)
  6. gating: forecast withheld (with reason) for short history, missing data, suspected sensor faults,
     or operating conditions outside the calibration range.
Conformal coverage assumes exchangeable households; it is NOT a guarantee under distribution shift, and the
interval is a prediction interval for days-to-criterion, not a probability of failure.
"""
from __future__ import annotations
import math
import pickle
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

from ..schema import STAGES, VISIBLE_COLS, repl_col, dtc_col
from ..checks.sensor_rules import blocking_reason
from .features import extract, vector, NAMES, V1_NAMES, V2_EXTRA, PREV

BINS = (0.0, 20.0, 60.0, np.inf)            # Mondrian bins (non-adaptive intervals only)
TREND_FEATURE = {"sediment": "sed_trend_days", "ro": "ro_trend_days", "carbon": None}
MIN_SIGMA = 0.05


def build_dataset(df, stage, stride=4, min_age=14, min_notice=7):
    """Feature matrix over ALL feature names (incl. previous-cycle ones). Reusable across model variants.
    Returns dict(X, y, g=household ids, cyc, t)."""
    X, y, g_, cyc_, t_ = [], [], [], [], []
    for hid, g in df.groupby("household_id", sort=False):
        g = g.sort_values("day").reset_index(drop=True)
        vis = g[VISIBLE_COLS]
        rep, dtc = g[repl_col(stage)].to_numpy(), g[dtc_col(stage)].to_numpy()
        rp = np.flatnonzero(rep)
        starts = np.r_[0, rp[rp > 0]]
        ends = np.r_[starts[1:], len(g)]
        for k, (s, e) in enumerate(zip(starts, ends)):
            hit = np.flatnonzero(dtc[s:e] == 0)
            if len(hit) == 0:
                continue
            c = int(s + hit[0])
            if c - (int(s) + min_age) < min_notice:
                continue
            for t in range(int(s) + min_age, c + 1, stride):
                X.append(vector(extract(vis.iloc[: t + 1], stage, True), NAMES))
                y.append(c - t); g_.append(hid); cyc_.append(k); t_.append(t)
    return {"X": np.array(X), "y": np.array(y, float), "g": np.array(g_), "cyc": np.array(cyc_), "t": np.array(t_)}


def build_datasets(train_df, cal_df, stride=4):
    return {"train": {s: build_dataset(train_df, s, stride) for s in STAGES},
            "cal": {s: build_dataset(cal_df, s, stride) for s in STAGES}}


def _gbm(seed, max_iter, lr=0.06):
    return HistGradientBoostingRegressor(max_iter=max_iter, learning_rate=lr, max_leaf_nodes=15,
                                         min_samples_leaf=40, l2_regularization=1.0, random_state=seed)


class ReferenceForecaster:
    def __init__(self, use_prev_cycles=False, feature_set="v3", select=True, adaptive=True,
                 alpha=0.10, seed=0, max_iter=250, stride=4, n_folds=4, max_useful_width_days=100.0):
        self.use_prev, self.feature_set, self.select, self.adaptive = use_prev_cycles, feature_set, select, adaptive
        self.alpha, self.seed, self.max_iter, self.stride, self.n_folds = alpha, seed, max_iter, stride, n_folds
        self.max_useful_width = max_useful_width_days     # fixed beforehand; see show_interval
        base = {"v1": V1_NAMES, "v2": V1_NAMES + V2_EXTRA, "v3": NAMES}[feature_set]
        self.show_interval = {}
        self.want = [n for n in base if use_prev_cycles or n not in PREV]
        self.cols, self.models, self.scale, self.mode, self.q, self.range, self.diag, self._cache = {}, {}, {}, {}, {}, {}, {}, {}
        self.version = (f"ref-v3(features={feature_set},select={select},adaptive={adaptive},"
                        f"prev={use_prev_cycles},alpha={alpha},seed={seed})")

    # ------------------------------------------------------------------ helpers
    def _ix(self, names):
        return [NAMES.index(n) for n in names]

    def _baseline_days(self, stage, X):
        """Best simple baseline reconstructed from the feature matrix (trend where available, else volume rule)."""
        vol = X[:, NAMES.index("volrule_days")]
        tf = TREND_FEATURE[stage]
        b = vol.copy()
        if tf is not None:
            t = X[:, NAMES.index(tf)]
            b = np.where(np.isfinite(t), t, vol)
        return np.maximum(b, 0.0)

    def _lp(self, stage, X):
        """Point forecast in log1p space for a (full-feature) matrix."""
        if self.mode[stage] == "gbm":
            return self.models[stage].predict(X[:, self._ix(self.cols[stage])])
        return np.log1p(self._baseline_days(stage, X))

    def _quantile(self, r):
        n = len(r)
        k = math.ceil((n + 1) * (1 - self.alpha))
        return float(np.sort(r)[min(k, n) - 1])

    # ------------------------------------------------------------------ fit
    def fit(self, train_df=None, cal_df=None, datasets=None):
        ds = datasets or build_datasets(train_df, cal_df, self.stride)
        for st in STAGES:
            tr, ca = ds["train"][st], ds["cal"][st]
            Xw = tr["X"][:, self._ix(self.want)]
            keep = [n for j, n in enumerate(self.want) if np.isfinite(Xw[:, j]).any()]   # drop non-applicable columns
            self.cols[st] = keep
            Xtr, ytr = tr["X"][:, self._ix(keep)], np.log1p(tr["y"])
            # --- cross-fitted out-of-fold predictions (by household) ---
            oof = np.zeros(len(ytr))
            gkf = GroupKFold(n_splits=min(self.n_folds, len(set(tr["g"]))))
            for a, b in gkf.split(Xtr, ytr, tr["g"]):
                oof[b] = _gbm(self.seed, self.max_iter).fit(Xtr[a], ytr[a]).predict(Xtr[b])
            base_days = self._baseline_days(st, tr["X"])
            prim = tr["y"] <= 60
            e_gbm = float(np.median(np.abs(np.expm1(oof[prim]) - tr["y"][prim])))
            e_base = float(np.median(np.abs(base_days[prim] - tr["y"][prim])))
            self.mode[st] = "baseline" if (self.select and e_base < e_gbm) else "gbm"
            self.diag[st] = {"mode": self.mode[st], "oof_median_err_gbm": e_gbm, "oof_median_err_baseline": e_base,
                             "n_train": int(len(ytr)), "n_cal": int(len(ca["y"]))}
            if self.mode[st] == "gbm":
                self.models[st] = _gbm(self.seed, self.max_iter).fit(Xtr, ytr)
            lp_oof = oof if self.mode[st] == "gbm" else np.log1p(base_days)
            ix = {n: i for i, n in enumerate(NAMES)}
            self.range[st] = {k: (float(np.nanmin(tr["X"][:, ix[k]])), float(np.nanmax(tr["X"][:, ix[k]])))
                              for k in ("u30", "feed_cyc", "temp_cyc")}
            # --- scale model on out-of-fold |log residual| ---
            if self.adaptive:
                self.scale[st] = _gbm(self.seed, 120, 0.05).fit(Xtr, np.abs(ytr - lp_oof))
            # --- conformal calibration on SEPARATE households ---
            lp_c = self._lp(st, ca["X"])
            res = np.abs(np.log1p(ca["y"]) - lp_c)
            if self.adaptive:
                sig = self._sigma(st, ca["X"])
                self.q[st] = {"all": self._quantile(res / sig)}
            else:
                days = np.expm1(lp_c)
                self.q[st] = {"all": self._quantile(res)}
                for lo, hi in zip(BINS[:-1], BINS[1:]):
                    sel = (days >= lo) & (days < hi)
                    self.q[st][(lo, hi)] = self._quantile(res[sel]) if sel.sum() >= 30 else self.q[st]["all"]
            self.diag[st]["q"] = self.q[st]["all"]
            # "a prediction interval where calibration supports one": hide intervals whose calibrated median
            # width at the primary horizon exceeds max_useful_width (preset), instead of showing a meaningless range.
            self.show_interval[st] = True
            _, clo, chi = self.predict_matrix(st, ca["X"])
            prim_c = ca["y"] <= 60
            med_w = float(np.median((chi - clo)[prim_c]))
            self.show_interval[st] = med_w <= self.max_useful_width
            self.diag[st].update(cal_median_width=med_w, show_interval=self.show_interval[st])
        return self

    def _sigma(self, stage, X):
        return np.maximum(self.scale[stage].predict(X[:, self._ix(self.cols[stage])]), MIN_SIGMA)

    # ------------------------------------------------------------------ inference
    def predict_matrix(self, stage, X):
        """(point, lo, hi) arrays in days for a full-feature matrix (no gating). Used by quick evaluation."""
        lp = self._lp(stage, X)
        if self.adaptive:
            w = self.q[stage]["all"] * self._sigma(stage, X)
        else:
            d = np.expm1(lp)
            w = np.array([next((self.q[stage][(lo, hi)] for lo, hi in zip(BINS[:-1], BINS[1:]) if lo <= x < hi),
                               self.q[stage]["all"]) for x in d])
        p, lo, hi = np.maximum(0, np.expm1(lp)), np.maximum(0, np.expm1(lp - w)), np.expm1(lp + w)
        if not self.show_interval.get(stage, True):
            lo, hi = np.full_like(p, np.nan), np.full_like(p, np.nan)
        return p, lo, hi

    def _features(self, hist, stage, use_cache=False):
        """Caching is for the evaluation harness only (immutable slices of one dataset). Interactive callers
        (what-if edits to a history) must leave use_cache=False or they get stale features."""
        if not use_cache:
            return extract(hist, stage, self.use_prev)
        key = (hist["household_id"].iloc[-1], len(hist), stage)
        if key not in self._cache:
            if len(self._cache) > 200_000:
                self._cache.clear()
            self._cache[key] = extract(hist, stage, self.use_prev)
        return self._cache[key]

    def predict(self, hist, stage, gate=True, use_cache=False):
        def withheld(reason):
            return {"point": np.nan, "lo": None, "hi": None, "withheld": True, "reason": reason}
        if stage not in self.mode:
            raise RuntimeError("model not fitted")
        f = self._features(hist, stage, use_cache)
        if gate:
            if f["age"] < 14:
                return withheld("insufficient history since installation (< 14 days)")
            if f["miss_frac"] > 0.5:
                return withheld("too many missing readings in the last 14 days")
            why = blocking_reason(hist, stage)
            if why:
                return withheld(f"suspected sensor problem ({why})")
            for k, (lo, hi) in self.range[stage].items():
                if np.isfinite(f[k]) and not (lo * 0.8 <= f[k] <= hi * 1.2):
                    return withheld(f"operating condition '{k}' outside the calibration range")
        p, lo, hi = (a[0] for a in self.predict_matrix(stage, vector(f, NAMES).reshape(1, -1)))
        shown = bool(np.isfinite(lo))
        return {"point": float(p), "lo": float(lo) if shown else None, "hi": float(hi) if shown else None,
                "withheld": False, "reason": "", "method": self.mode[stage],
                "interval_status": ("split-conformal on separate households; assumes exchangeability; not a guarantee"
                                    if shown else "not shown: calibrated interval too wide to be useful for this stage")}

    def __call__(self, hist, stage):
        return self.predict(hist, stage)

    def variant(self, **kw):
        """Same fitted model with different inference settings (e.g. gate=False). Uses the feature cache:
        safe for harness evaluation only."""
        kw.setdefault("use_cache", True)
        return lambda hist, stage: self.predict(hist, stage, **kw)

    def feature_importance(self, stage, D, n=3000, seed=0):
        """Permutation importance (global, model-based; not causal). Only meaningful where mode == 'gbm'."""
        from sklearn.inspection import permutation_importance
        if self.mode[stage] != "gbm":
            return []
        X = D["X"][:, self._ix(self.cols[stage])]
        idx = np.random.default_rng(seed).choice(len(X), min(n, len(X)), replace=False)
        r = permutation_importance(self.models[stage], X[idx], np.log1p(D["y"][idx]), n_repeats=3, random_state=seed)
        return sorted(zip(self.cols[stage], r.importances_mean), key=lambda t: -t[1])

    def save(self, path):
        self._cache = {}
        with open(path, "wb") as fh:
            pickle.dump(self, fh)

    @staticmethod
    def load(path):
        with open(path, "rb") as fh:
            return pickle.load(fh)


def quick_metrics(model, test_ds, horizon=60.0, target=0.90):
    """Fast offline metrics from cached features (stride-4 samples; the harness is the official evaluation)."""
    rows = []
    for st in STAGES:
        D = test_ds[st]
        p, lo, hi = model.predict_matrix(st, D["X"])
        y = D["y"]
        prim = y <= horizon
        shown = bool(np.isfinite(lo).any())
        cov = (y >= lo) & (y <= hi)
        rows.append({"stage": st, "mode": model.mode[st], "n": int(prim.sum()),
                     "median_abs_err": float(np.median(np.abs(p - y)[prim])),
                     "coverage_all": float(cov.mean()) if shown else np.nan,
                     "coverage_primary": float(cov[prim].mean()) if shown else np.nan,
                     "median_width_primary": float(np.median((hi - lo)[prim])) if shown else np.nan})
    import pandas as pd
    out = pd.DataFrame(rows).set_index("stage")
    allp = np.concatenate([np.abs(model.predict_matrix(s, test_ds[s]["X"])[0] - test_ds[s]["y"])[test_ds[s]["y"] <= horizon] for s in STAGES])
    out.loc["ALL", ["median_abs_err"]] = float(np.median(allp))
    return out
