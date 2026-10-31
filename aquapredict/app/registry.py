"""Model registry: one interface over the 3 baselines, the reference model (Person 1) and Person 2's model.

SECURITY: trained artifacts are pickles/joblib files. Only load artifacts created on this machine (or that you trust).
"""
from __future__ import annotations
import datetime as dt
import json
import os
import pickle
import sklearn
import numpy as np

from ..schema import age_col
from ..baselines.baselines import BASELINES
from ..simulator.config import SimConfig
from ..simulator.generate import generate_dataset
from ..eval.splits import split_households, subset

MIN_HISTORY_DAYS = 14
DISCLAIMER = ("SYNTHETIC-data prototype: not validated on real purifier telemetry. Predicts filter wear only; it does "
              "not assess water safety, potability or contaminant removal.")
BASELINE_LABELS = {"trend": "Trend extrapolation (baseline)", "volume_rule": "Volume rule (baseline)",
                   "health_20pct": "20% health threshold (baseline)"}
LABELS = {**BASELINE_LABELS, "reference_v3": "Reference model v3 (Person 1 stand-in)", "person2": "Person 2 model (integrated)"}

# Reported results on the synthetic benchmark, with provenance. They are NOT live numbers: the Validation Lab recomputes.
MODEL_CARDS = {
    "trend": {"what": "Fits a line to a normalised wear indicator (sediment pressure drop per flow; RO flux and rejection) "
                      "and extrapolates to the replacement criterion. No sensor signal for carbon, so carbon falls back to the volume rule.",
              "measured": "Median |window error| 13.7-16.0 days on held-out synthetic households; late 22-24%, premature 22-31%.",
              "limits": "Point estimate only (no interval). Degrades with missing data."},
    "volume_rule": {"what": "Replace at the rated litres or rated days (illustrative ratings set near the simulated median life).",
                    "measured": "Median |window error| 30-36 days; late 43-52%.", "limits": "Ignores the household's actual wear."},
    "health_20pct": {"what": "Usage-based health = 1 - max(volume/rated volume, age/rated days); alerts at 20%.",
                     "measured": "Median |window error| 25-32 days; late 30-42%; premature 43-51%.",
                     "limits": "Early-alert rule, not a wear measurement."},
    "reference_v3": {"what": "Gradient-boosted regression + model selection (used only where it beats the simple rule) + scale-adaptive "
                             "split-conformal intervals calibrated on separate households + gating.",
                     "measured": "Fresh synthetic households (default simulator, 80 households): median |error| 12.0 d, late 11.5%, premature 34.3%, "
                                 "coverage 89.2%, median interval width 54.9 d. Criteria: error PASS, interval width FAIL (preset 40 d cap), "
                                 "late-warning PASS.",
                     "limits": "Intervals are wide. No better than the volume rule for carbon (no carbon sensor signal). Slow sensor drift not detected."},
    "person2": {"what": "Person 2's pipeline (validation, sensor-fault events, gating, CUSUM source-change signal, gradient-boosted forecast, "
                        "conformal interval, SHAP drivers when installed), retrained on this app's synthetic households.",
                "measured": "Fresh synthetic households (48): median |error| 37.0 d vs 13.7 d for the trend baseline; fails all three success "
                            "criteria; late 2.0%, premature 71.6%; coverage 91.2%, median width 135 d.",
                "limits": "Over-predicts remaining life in every stage (cause not yet diagnosed). Its pressure input is wired to the sediment "
                          "pressure drop, an assumption (see docs_results/RESULTS.md). Fault/drivers quality is not scored by that table."},
}


class ModelUnavailable(RuntimeError):
    pass


def normalise(model, version, stage, *, status, point=None, interval=None, reason=None, interval_status="", drivers=None,
              explanation=None, fault_events=None, source_change=None):
    return {"model": model, "model_version": version, "stage": stage, "status": status, "point_days": point,
            "interval_days": interval, "reason": reason, "interval_status": interval_status, "drivers": drivers or [],
            "explanation": explanation, "fault_events": fault_events or [], "source_change": source_change,
            "disclaimer": DISCLAIMER}


class Forecaster:
    name = "?"
    version = "?"

    def run(self, hist, stage) -> dict:        # full normalised response
        raise NotImplementedError

    def __call__(self, hist, stage) -> dict:   # evaluation-harness contract
        r = self.run(hist, stage)
        iv = r["interval_days"]
        return {"point": np.nan if r["point_days"] is None else float(r["point_days"]),
                "lo": None if iv is None else float(iv[0]), "hi": None if iv is None else float(iv[1]),
                "withheld": r["status"] == "withheld", "reason": r["reason"] or "", "status": r["status"]}


def _too_short(hist, stage):
    return int(hist[age_col(stage)].iloc[-1]) < MIN_HISTORY_DAYS


class BaselineForecaster(Forecaster):
    def __init__(self, name):
        self.name, self.fn, self.version = name, BASELINES[name], f"baseline:{name}"

    def run(self, hist, stage):
        if _too_short(hist, stage):
            return normalise(self.name, self.version, stage, status="withheld",
                             reason=f"insufficient history since installation (< {MIN_HISTORY_DAYS} days)")
        r = self.fn(hist, stage)
        note = "no calibrated interval for this baseline" + (" (fell back to the volume rule)" if r.get("fallback") else "")
        return normalise(self.name, self.version, stage, status="point_only", point=r["point"], interval_status=note)


class ReferenceWrapper(Forecaster):
    name = "reference_v3"

    def __init__(self, model, version):
        self.model, self.version = model, version

    def run(self, hist, stage):
        r = self.model.predict(hist, stage)
        if r["withheld"]:
            return normalise(self.name, self.version, stage, status="withheld", reason=r["reason"])
        iv = None if r["lo"] is None else [r["lo"], r["hi"]]
        return normalise(self.name, self.version, stage, status="interval" if iv else "point_only", point=r["point"],
                         interval=iv, interval_status=r["interval_status"])


class Person2Wrapper(Forecaster):
    name = "person2"

    def __init__(self, bundle_path, version, explain=True):
        from ..integration.train import Person2Forecaster
        self.p2, self.version = Person2Forecaster(bundle_path, pressure_source="dp", explain=explain), version

    def run(self, hist, stage):
        if _too_short(hist, stage):
            return normalise(self.name, self.version, stage, status="withheld",
                             reason=f"insufficient history since installation (< {MIN_HISTORY_DAYS} days)")
        r = self.p2.respond(hist, stage)
        iv = r["interval_days"]
        status = {"ok": "interval" if iv else "point_only", "degraded": "point_only", "withheld": "withheld"}[r["status"]]
        return normalise(self.name, self.version, stage, status=status, point=r["forecast_days"],
                         interval=None if iv is None else [float(iv[0]), float(iv[1])], reason=r["reason"],
                         interval_status="calibrated conformal interval (assumes exchangeable households; not a guarantee)" if iv else
                         "no interval available", drivers=r["drivers"], explanation=r["explanation"],
                         fault_events=r["fault_events"], source_change=r["source_change"])


class ModelRegistry:
    def __init__(self, artifact_dir: str):
        self.dir = artifact_dir
        os.makedirs(artifact_dir, exist_ok=True)
        self._cache = {}

    # ---------------------------------------------------------------- meta
    @property
    def _meta_path(self):
        return os.path.join(self.dir, "models_meta.json")

    def meta(self) -> dict:
        return json.load(open(self._meta_path)) if os.path.exists(self._meta_path) else {}

    def _paths(self):
        return {"reference_v3": os.path.join(self.dir, "reference_v3.pkl"), "person2": os.path.join(self.dir, "person2_bundle.joblib")}

    def available(self) -> list[dict]:
        out = [{"name": n, "label": BASELINE_LABELS[n], "available": True, "reason": "", "trained_with": None} for n in BASELINES]
        meta = self.meta()
        for n, p in self._paths().items():
            ok = os.path.exists(p)
            out.append({"name": n, "label": LABELS[n], "available": ok, "trained_with": meta.get(n),
                        "reason": "" if ok else "not trained yet - use 'Train models' (a few minutes) or `make models`"})
        return out

    def default_model(self) -> str:
        return "reference_v3" if os.path.exists(self._paths()["reference_v3"]) else "trend"

    def train_seed(self, name):
        return (self.meta().get(name) or {}).get("train_seed")

    # ---------------------------------------------------------------- get
    def get(self, name: str, explain: bool = True) -> Forecaster:
        key = (name, explain)
        if key in self._cache:
            return self._cache[key]
        if name in BASELINES:
            f = BaselineForecaster(name)
        elif name in self._paths():
            path = self._paths()[name]
            if not os.path.exists(path):
                raise ModelUnavailable(f"model '{name}' has not been trained yet")
            ver = f"{name}@{(self.meta().get(name) or {}).get('trained_at', 'unknown')}"
            if name == "reference_v3":
                try:
                    with open(path, "rb") as fh:
                        f = ReferenceWrapper(pickle.load(fh), ver)
                except Exception as e:                  # e.g. scikit-learn version mismatch
                    raise ModelUnavailable(f"could not load '{name}' ({type(e).__name__}); retrain it") from e
            else:
                f = Person2Wrapper(path, ver, explain=explain)
        else:
            raise KeyError(f"unknown model '{name}'")
        self._cache[key] = f
        return f

    # ---------------------------------------------------------------- train
    def train(self, which=("reference_v3", "person2"), n=120, seed=1000, max_iter=200, stride=4, log=print) -> dict:
        """Train on a fresh synthetic dataset (calibrated simulator). Train/calibration households are disjoint.
        Records `train_seed` so the Validation Lab can refuse to evaluate on the same households."""
        df, _ = generate_dataset(n, law="A", seed=seed, cfg=SimConfig.calibrated())
        sp = split_households(df.household_id.unique(), seed=seed, fractions=(0.6, 0.4, 0.0))
        tr, ca = subset(df, sp["train"]), subset(df, sp["calibration"])
        meta = self.meta()
        stamp = dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()
        base = {"train_seed": seed, "n_households": n, "trained_at": stamp, "max_iter": max_iter,
                "sklearn": sklearn.__version__, "simulator_config": "calibrated"}
        if "reference_v3" in which:
            from ..reference_model.model import ReferenceForecaster
            log(f"training reference_v3 on {len(sp['train'])} households, calibrating on {len(sp['calibration'])} ...")
            m = ReferenceForecaster(seed=seed, max_iter=max_iter, stride=stride).fit(tr, ca)
            m._cache = {}
            with open(self._paths()["reference_v3"], "wb") as fh:
                pickle.dump(m, fh)
            meta["reference_v3"] = dict(base)
        if "person2" in which:
            from ..integration.train import train_person2_on_person1
            log("training Person 2's model ...")
            train_person2_on_person1(tr, ca, self._paths()["person2"], seed=seed, max_iter=max_iter, pressure_source="dp")
            meta["person2"] = dict(base)
        with open(self._meta_path, "w") as fh:
            json.dump(meta, fh, indent=2)
        self._cache.clear()
        return meta
