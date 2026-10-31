"""Application service: ALL logic used by both the FastAPI backend and the Streamlit UI (so they can never disagree).
Every number returned comes from the real simulator, forecasters and rule checks - nothing is mocked.
Synthetic households are labelled source='synthetic'; real telemetry would be source='uploaded'."""
from __future__ import annotations
import datetime as dt
import io
import math
import os
from dataclasses import fields

import numpy as np
import pandas as pd

from ..schema import (STAGES, VISIBLE_COLS, ALL_COLS, TRUTH_COLS, FAULTABLE_SENSORS, age_col, vol_col, repl_col, dtc_col,
                      DEFAULT_RATED, SEDIMENT_DP_RATIO_LIMIT, RO_FLUX_LIMIT, RO_REJECTION_LIMIT)
from ..validate import validate_telemetry
from ..simulator.config import SimConfig
from ..simulator.generate import generate_household, generate_dataset
from ..simulator.scenarios import SCENARIOS, custom_scenario
from ..simulator.faults import KINDS
from ..checks.sensor_rules import check_readings, SENSORS
from ..baselines.baselines import _cycle, _clean_level, _fit
from ..eval.config import EvalConfig
from ..eval.harness import evaluate
from ..eval import metrics as M
from ..eval.criteria import check_success
from ..planner.cost import CostConfig, policy_table, sensitivity
from . import alerts as AL
from .db import Database, now_iso, dumps, loads
from .registry import ModelRegistry, ModelUnavailable, MODEL_CARDS, DISCLAIMER
from .online_sources import fetch_usgs_observations

BOOL_COLS = [repl_col(s) for s in STAGES] + ["fault_active", "missing_injected", "source_change_flag"]
STAGE_LABELS = {"sediment": "Sediment filter", "carbon": "Carbon filter", "ro": "RO membrane"}
SIM_FIELDS = {f.name for f in fields(SimConfig)}
DEFAULT_NOW_DAY = 60


class NotFound(KeyError):
    pass


class BadRequest(ValueError):
    pass


class Conflict(RuntimeError):
    pass


def clean(o):
    """Make any result JSON-safe: numpy -> python, NaN/inf -> None, datetimes -> ISO, DataFrame -> records."""
    if isinstance(o, pd.DataFrame):
        return clean(o.reset_index().to_dict("records"))
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return None if not math.isfinite(float(o)) else float(o)
    if isinstance(o, (pd.Timestamp, dt.datetime, dt.date)):
        return o.isoformat()
    if o is pd.NaT:
        return None
    return o


class AquaService:
    def __init__(self, db_path: str = ":memory:", artifact_dir: str | None = None):
        self.db = Database(db_path)
        home = os.environ.get("AQUAPREDICT_HOME", "aquapredict_app_data")
        self.registry = ModelRegistry(artifact_dir or os.path.join(home, "artifacts"))

    def online_water_quality(self, site: str = "01646500") -> dict:
        """Return live public water-quality context with source provenance."""
        return fetch_usgs_observations(site)

    # =============================================================== households
    def scenarios(self):
        return [{"name": s.name, "description": s.description, "overrides": s.overrides,
                 "fault_templates": list(s.fault_templates)} for s in SCENARIOS.values()]

    def _scenario(self, name, overrides):
        if name not in SCENARIOS:
            raise BadRequest(f"unknown scenario '{name}'. Available: {sorted(SCENARIOS)}")
        bad = [k for k in (overrides or {}) if k not in SIM_FIELDS]
        if bad:
            raise BadRequest(f"unknown simulator parameter(s) {bad}. Valid: {sorted(SIM_FIELDS)}")
        sc = SCENARIOS[name]
        return custom_scenario(sc.name, {**sc.overrides, **(overrides or {})}, sc.fault_templates, sc.description)

    def _validate_faults(self, faults):
        for f in faults or []:
            if f.get("kind") not in KINDS:
                raise BadRequest(f"unknown fault kind '{f.get('kind')}'; use one of {KINDS}")
            if f["kind"] != "tds_inversion" and f.get("sensor") not in FAULTABLE_SENSORS:
                raise BadRequest(f"cannot inject into '{f.get('sensor')}'; use one of {FAULTABLE_SENSORS}")
            if int(f.get("start", -1)) < 0 or int(f.get("duration", 1)) < 1:
                raise BadRequest("fault needs start >= 0 and duration >= 1")

    def _generate(self, h, extra_overrides=None, extra_faults=None):
        ov = {**h["overrides"], **(extra_overrides or {})}
        sc = self._scenario(h["scenario"], ov)
        base = SimConfig.calibrated() if h["calibrated"] else SimConfig()
        df, _ = generate_household(h["sim_index"], sc, base, h["law"], h["seed"],
                                   forced_replacements=h["forced"], user_faults=h["user_faults"] + (extra_faults or []))
        df = df.copy()
        df["household_id"] = h["id"]
        return df

    def _store(self, hid, df):
        d = df.copy()
        d["date"] = pd.to_datetime(d["date"]).dt.strftime("%Y-%m-%dT%H:%M:%S")
        with self.db.lock:
            self.db.conn.execute("DELETE FROM readings WHERE household_id=?", (hid,)) if self._has_readings() else None
            d.to_sql("readings", self.db.conn, if_exists="append", index=False)
            self.db.conn.execute("CREATE INDEX IF NOT EXISTS idx_readings ON readings(household_id, day)")
            self.db.conn.commit()

    def _has_readings(self):
        return bool(self.db.one("SELECT name FROM sqlite_master WHERE type='table' AND name='readings'"))

    def _parse(self, r):
        r = dict(r)
        r["overrides"] = loads(r.pop("overrides_json"), {})
        r["user_faults"] = loads(r.pop("user_faults_json"), [])
        r["forced"] = loads(r.pop("forced_json"), {})
        r["calibrated"] = bool(r["calibrated"])
        return r

    def household(self, hid) -> dict:
        r = self.db.one("SELECT * FROM households WHERE id=?", (hid,))
        if not r:
            raise NotFound(f"household '{hid}' not found")
        return self._parse(r)

    def list_households(self):
        return [self._parse(r) for r in self.db.q("SELECT * FROM households ORDER BY id")]

    def create_household(self, name, scenario="normal", law="A", seed=0, sim_index=None, overrides=None,
                         calibrated=True, now_day=DEFAULT_NOW_DAY, user_faults=None) -> dict:
        if law not in ("A", "B"):
            raise BadRequest("law must be 'A' or 'B'")
        self._validate_faults(user_faults)
        self._scenario(scenario, overrides)
        n = self.db.one("SELECT COUNT(*) AS n FROM households")["n"]
        hid = f"H{n + 1:03d}"
        while self.db.one("SELECT 1 FROM households WHERE id=?", (hid,)):
            n += 1
            hid = f"H{n + 1:03d}"
        sim_index = n if sim_index is None else int(sim_index)
        row = {"id": hid, "name": name or hid, "source": "synthetic", "scenario": scenario, "law": law, "seed": int(seed),
               "sim_index": sim_index, "calibrated": bool(calibrated), "overrides": dict(overrides or {}),
               "user_faults": list(user_faults or []), "forced": {}}
        df = self._generate(row)
        now_day = max(0, min(int(now_day), len(df) - 1))
        self.db.exec("INSERT INTO households VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                     (hid, row["name"], "synthetic", scenario, law, int(seed), sim_index, int(calibrated), dumps(row["overrides"]),
                      dumps(row["user_faults"]), dumps({}), len(df), now_day, now_iso()))
        self._store(hid, df)
        return self.household(hid)

    def delete_household(self, hid):
        self.household(hid)
        for t in ("households", "predictions", "alerts", "maintenance"):
            self.db.exec(f"DELETE FROM {t} WHERE " + ("id" if t == "households" else "household_id") + "=?", (hid,))
        if self._has_readings():
            self.db.exec("DELETE FROM readings WHERE household_id=?", (hid,))

    def _regenerate(self, hid):
        h = self.household(hid)
        self._store(hid, self._generate(h))

    # =============================================================== readings / history
    def readings(self, hid, upto_day=None, include_truth=False, last=None) -> pd.DataFrame:
        h = self.household(hid)
        upto = h["now_day"] if upto_day is None else int(upto_day)
        if upto > h["now_day"]:
            raise BadRequest("cannot read beyond the household's current day (no peeking at the future)")
        df = pd.read_sql_query("SELECT * FROM readings WHERE household_id=? AND day<=? ORDER BY day", self.db.conn,
                               params=(hid, upto), parse_dates=["date"])
        for c in BOOL_COLS:
            if c in df.columns:
                df[c] = df[c].fillna(0).astype(bool)
        if not include_truth:
            df = df[[c for c in VISIBLE_COLS if c in df.columns]]
        if last:
            df = df.tail(int(last))
        return df.reset_index(drop=True)

    def history(self, hid, upto_day=None) -> pd.DataFrame:
        return self.readings(hid, upto_day, include_truth=False)

    def set_now_day(self, hid, day):
        h = self.household(hid)
        day = int(day)
        if not 0 <= day < h["n_days"]:
            raise BadRequest(f"day must be in [0, {h['n_days'] - 1}]")
        self.db.exec("UPDATE households SET now_day=? WHERE id=?", (day, hid))

    # =============================================================== real-telemetry boundary
    def ingest_telemetry(self, name, rows, replacements=None) -> dict:
        """Accept daily readings for a REAL (uploaded) purifier. Needs per-day usage and sensor readings (null allowed) plus,
        for each stage, the days its filter was replaced (day 0 = installed at the start of the data).
        age / cumulative-volume / replaced columns are derived here; schema is validated; nothing is repaired silently."""
        if not rows:
            raise BadRequest("no rows supplied")
        df = pd.DataFrame(rows).sort_values("day").reset_index(drop=True)
        if "usage_L" not in df.columns or "day" not in df.columns:
            raise BadRequest("rows need at least 'day' and 'usage_L'")
        for c in SENSORS:
            if c not in df.columns:
                df[c] = np.nan
        df["date"] = pd.to_datetime(df["date"]) if "date" in df.columns else pd.Timestamp("2026-01-01") + pd.to_timedelta(df["day"], unit="D")
        usage = df["usage_L"].fillna(0).to_numpy(float)
        df["op_mode"] = df["op_mode"] if "op_mode" in df.columns else np.where(usage == 0, "idle", "normal")
        repl = {s: sorted(set(int(d) for d in (replacements or {}).get(s, []) if int(d) > 0)) for s in STAGES}
        for s in STAGES:
            ages, vols, flags, install, vol = [], [], [], 0, 0.0
            rdays = set(repl[s])
            for d, u in zip(df["day"], usage):
                if int(d) in rdays:
                    install, vol = int(d), 0.0
                vol += u
                ages.append(int(d) - install); vols.append(vol); flags.append(int(d) in rdays)
            df[age_col(s)], df[vol_col(s)], df[repl_col(s)] = np.array(ages, float), np.array(vols, float), np.array(flags, bool)
        df["household_id"] = "tmp"
        problems = validate_telemetry(df[VISIBLE_COLS])
        if problems:
            raise BadRequest("invalid telemetry: " + "; ".join(problems))
        n = self.db.one("SELECT COUNT(*) AS n FROM households")["n"]
        hid = f"H{n + 1:03d}"
        while self.db.one("SELECT 1 FROM households WHERE id=?", (hid,)):
            n += 1
            hid = f"H{n + 1:03d}"
        df["household_id"] = hid
        self.db.exec("INSERT INTO households VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                     (hid, name or hid, "uploaded", "uploaded", None, None, None, 0, dumps({}), dumps([]), dumps(repl),
                      len(df), len(df) - 1, now_iso()))
        self._store(hid, df[VISIBLE_COLS])
        return self.household(hid)

    # =============================================================== data quality, indicators
    def sensor_events(self, hid, stage=None, day=None) -> list[dict]:
        hist = self.history(hid, day)
        return check_readings(hist, stage)

    def data_quality(self, hid, day=None) -> dict:
        hist = self.history(hid, day)
        last_day = int(hist["day"].iloc[-1])
        act = hist[hist["op_mode"] != "idle"]
        fresh, miss = {}, {}
        for s in SENSORS:
            ok = act[act[s].notna()]
            fresh[s] = None if ok.empty else last_day - int(ok["day"].iloc[-1])
            tail = act[s].to_numpy(float)[-14:]
            miss[s] = float(np.isnan(tail).mean()) if len(tail) else None
        return {"last_day": last_day, "days_since_valid_reading": fresh, "missing_share_last_14_active_days": miss,
                "idle_today": bool(hist["op_mode"].iloc[-1] == "idle")}

    def indicators(self, hid, stage, day=None) -> dict:
        """Filter CONDITION from sensor evidence (distinct from sensor health and forecast reliability)."""
        hist = self.history(hid, day)
        cyc = _cycle(hist, stage)
        age = cyc[age_col(stage)].to_numpy(float)
        vol = float(cyc[vol_col(stage)].iloc[-1])
        none = {"available": False, "progress": None, "text": "", "stage": stage}
        with np.errstate(divide="ignore", invalid="ignore"):
            if stage == "sediment":
                y = cyc["dp_sediment_bar"].to_numpy(float) / cyc["flow_Lpm"].to_numpy(float)
                c = _clean_level(y)
                fit = _fit(age, y / c) if np.isfinite(c) and c > 0 else None
                if fit is None:
                    return {**none, "text": "not enough valid pressure/flow readings yet"}
                prog = (fit[0] - 1.0) / (SEDIMENT_DP_RATIO_LIMIT - 1.0)
                return {"available": True, "progress": float(prog), "stage": stage, "ratio": float(fit[0]),
                        "text": f"Pressure drop per flow is {fit[0]:.2f}x its clean value (assumed criterion {SEDIMENT_DP_RATIO_LIMIT}x)."}
            if stage == "ro":
                T = cyc["temp_C"].to_numpy(float)
                flux = cyc["flow_Lpm"].to_numpy(float) / (1.0 + 0.03 * (T - 25.0))
                c = _clean_level(flux)
                fit = _fit(age, flux / c) if np.isfinite(c) and c > 0 else None
                rej = 1.0 - cyc["out_tds_ppm"].to_numpy(float) / cyc["feed_tds_ppm"].to_numpy(float)
                rc, fr = _clean_level(rej), _fit(age, rej)
                progs, parts = [], []
                if fit is not None:
                    progs.append((1.0 - fit[0]) / (1.0 - RO_FLUX_LIMIT)); parts.append(f"flux at {fit[0]:.0%} of clean (criterion {RO_FLUX_LIMIT:.0%})")
                if fr is not None and np.isfinite(rc) and rc > RO_REJECTION_LIMIT:
                    progs.append((rc - fr[0]) / (rc - RO_REJECTION_LIMIT)); parts.append(f"rejection {fr[0]:.1%} (criterion {RO_REJECTION_LIMIT:.0%})")
                if not progs:
                    return {**none, "text": "not enough valid flow/TDS readings yet"}
                return {"available": True, "progress": float(max(progs)), "stage": stage, "text": "; ".join(parts).capitalize() + "."}
        r = DEFAULT_RATED[stage]
        return {"available": True, "progress": float(vol / r["volume_L"]), "stage": stage, "usage_based": True,
                "text": f"Usage-based estimate only: {vol:,.0f} L of an illustrative rated {r['volume_L']:,.0f} L. No sensor tracks this stage."}

    # =============================================================== forecasting
    def _stage(self, stage):
        if stage not in STAGES:
            raise BadRequest(f"stage must be one of {STAGES}")

    def models(self):
        out = []
        for m in self.registry.available():
            out.append({**m, "card": MODEL_CARDS.get(m["name"])})
        return out

    def _forecaster(self, model, explain=True):
        name = model or self.registry.default_model()
        try:
            return self.registry.get(name, explain=explain)
        except ModelUnavailable as e:
            raise Conflict(str(e))
        except KeyError as e:
            raise BadRequest(str(e))

    @staticmethod
    def _recommendation(fc, ind, events):
        st = fc["status"]
        if st == "withheld":
            return ("No forecast was issued, so no replacement decision should be based on this stage right now. "
                    f"Reason: {fc['reason']}. Fix the data problem or wait for more history.")
        iv = fc["interval_days"]
        basis = iv[0] if iv else fc["point_days"]
        if basis <= AL.URGENT_DAYS:
            return "Replacement is likely needed within about a week. Schedule it now and confirm it in the planner."
        if basis <= AL.LEAD_DAYS:
            return "Plan the replacement inside the predicted window and confirm a date in the planner."
        extra = " Sensor warnings are present; treat the estimate cautiously." if any(e.get("blocks_forecast") for e in events) else ""
        return "No action needed yet. Re-check as the filter ages." + extra

    @staticmethod
    def _reliability(fc):
        if fc["status"] == "withheld":
            return "none - forecast withheld"
        if fc["status"] == "point_only":
            return "low - point estimate without a validated interval"
        return "interval calibrated on separate synthetic households (assumes exchangeability; not a guarantee)"

    def forecast(self, hid, stage, model=None, day=None, store=True, explain=True) -> dict:
        self._stage(stage)
        h = self.household(hid)
        day = h["now_day"] if day is None else int(day)
        hist = self.history(hid, day)
        fc = self._forecaster(model, explain).run(hist, stage)
        events = check_readings(hist, stage)
        ind = self.indicators(hid, stage, day)
        fc.update({"household_id": hid, "day": day, "indicator": ind, "sensor_events": events,
                   "sensor_health": ("fault_suspected" if any(e["blocks_forecast"] for e in events)
                                     else "warning" if events else "ok"),
                   "reliability": self._reliability(fc), "recommendation": self._recommendation(fc, ind, events),
                   "data_source": h["source"], "stage_label": STAGE_LABELS[stage]})
        if store:
            iv = fc["interval_days"]
            self.db.exec("INSERT OR REPLACE INTO predictions(household_id,stage,day,model,model_version,status,point,lo,hi,reason,"
                         "extra_json,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                         (hid, stage, day, fc["model"], fc["model_version"], fc["status"], fc["point_days"],
                          None if not iv else iv[0], None if not iv else iv[1], fc["reason"],
                          dumps({"interval_status": fc["interval_status"], "sensor_health": fc["sensor_health"]}), now_iso()))
        return fc

    def forecast_all(self, hid, model=None, day=None, store=True, explain=True) -> dict:
        return {s: self.forecast(hid, s, model, day, store, explain) for s in STAGES}

    def truth(self, hid, stage, day=None) -> dict:
        """DEMO ONLY: the simulator's hidden ground truth. Never available for uploaded (real) telemetry."""
        h = self.household(hid)
        if h["source"] != "synthetic":
            return {"available": False, "note": "ground truth does not exist for real telemetry"}
        df = self.readings(hid, day, include_truth=True)
        v = df[dtc_col(stage)].iloc[-1]
        if not np.isfinite(v):
            return {"available": True, "censored": True, "note": "this filter was replaced before reaching its criterion"}
        return {"available": True, "censored": False, "remaining_days_true": float(v), "criterion_reached": bool(v <= 0),
                "overdue_days": float(max(0.0, -v))}

    def prediction_history(self, hid, stage=None, model=None) -> pd.DataFrame:
        sql, args = "SELECT * FROM predictions WHERE household_id=?", [hid]
        if stage:
            sql += " AND stage=?"; args.append(stage)
        if model:
            sql += " AND model=?"; args.append(model)
        return pd.DataFrame(self.db.q(sql + " ORDER BY day, stage", tuple(args)))

    # =============================================================== alerts (model-generated; user actions kept separate)
    CLEAR_AFTER_CHECKS = 2          # hysteresis: a condition must be absent on this many consecutive checks to auto-resolve

    def _sync_alerts(self, hid, stage, cands, day, counts, force_clear=False):
        existing = self.db.q("SELECT * FROM alerts WHERE household_id=? AND stage=? AND origin='model' AND cleared=0 "
                             "AND status IN ('open','acknowledged','dismissed')", (hid, stage))
        keys = {c["dedupe_key"] for c in cands}
        for c in cands:
            ex = next((e for e in existing if e["dedupe_key"] == c["dedupe_key"]), None)
            if ex:                                     # same condition still present: update, never duplicate
                self.db.exec("UPDATE alerts SET severity=?, message=?, details_json=?, last_day=?, clear_streak=0, updated_at=? WHERE id=?",
                             (c["severity"], c["message"], dumps(c["details"]), day, now_iso(), ex["id"]))
                counts["updated"] += 1
            else:
                self.db.exec("INSERT INTO alerts(household_id,stage,kind,dedupe_key,severity,status,origin,title,message,"
                             "recommended_action,details_json,first_day,last_day,created_at,updated_at) "
                             "VALUES(?,?,?,?,?,'open','model',?,?,?,?,?,?,?,?)",
                             (hid, stage, c["kind"], c["dedupe_key"], c["severity"], c["title"], c["message"],
                              c["recommended_action"], dumps(c["details"]), day, day, now_iso(), now_iso()))
                counts["created"] += 1
        for ex in existing:
            if ex["dedupe_key"] not in keys:           # condition absent on this check
                if not force_clear and ex["clear_streak"] + 1 < self.CLEAR_AFTER_CHECKS:
                    self.db.exec("UPDATE alerts SET clear_streak=clear_streak+1, updated_at=? WHERE id=?", (now_iso(), ex["id"]))
                    continue                            # absent once: wait for a second check before clearing (no flapping)
                new_status = "resolved" if ex["status"] in ("open", "acknowledged") else ex["status"]
                self.db.exec("UPDATE alerts SET cleared=1, status=?, note=COALESCE(note,'') || ?, updated_at=? WHERE id=?",
                             (new_status, " [auto: condition cleared]", now_iso(), ex["id"]))
                counts["resolved"] += 1

    def refresh_alerts(self, hid, model=None, day=None, forecasts=None, force_clear=False) -> dict:
        h = self.household(hid)
        day = h["now_day"] if day is None else int(day)
        fcs = forecasts or self.forecast_all(hid, model, day, store=True, explain=False)
        counts = {"created": 0, "updated": 0, "resolved": 0}
        for stage, fc in fcs.items():
            cands = AL.candidates(stage, fc, fc["sensor_events"], fc["indicator"])
            # a brand-new filter always has short history; that is expected, not an alert
            cands = [c for c in cands if not (c["kind"] == "forecast_withheld" and "insufficient history" in (fc["reason"] or ""))]
            self._sync_alerts(hid, stage, cands, day, counts, force_clear)
        return counts

    def alerts(self, hid=None, status=None, stage=None) -> list[dict]:
        sql, args = "SELECT * FROM alerts WHERE 1=1", []
        for col, v in (("household_id", hid), ("status", status), ("stage", stage)):
            if v:
                sql += f" AND {col}=?"; args.append(v)
        rows = self.db.q(sql + " ORDER BY CASE severity WHEN 'high' THEN 0 WHEN 'medium' THEN 1 ELSE 2 END, id DESC", tuple(args))
        for r in rows:
            r["details"] = loads(r.pop("details_json"), {})
        return rows

    def alert_action(self, alert_id, action, note=None) -> dict:
        a = self.db.one("SELECT * FROM alerts WHERE id=?", (alert_id,))
        if not a:
            raise NotFound(f"alert {alert_id} not found")
        allowed = {"acknowledge": ({"open"}, "acknowledged"), "resolve": ({"open", "acknowledged"}, "resolved"),
                   "dismiss": ({"open", "acknowledged"}, "dismissed")}
        if action not in allowed:
            raise BadRequest("action must be acknowledge, resolve or dismiss")
        src, dst = allowed[action]
        if a["status"] not in src:
            raise Conflict(f"cannot {action} an alert that is '{a['status']}'")
        tag = f" [user {action}{': ' + note if note else ''}]"
        self.db.exec("UPDATE alerts SET status=?, note=COALESCE(note,'') || ?, updated_at=? WHERE id=?", (dst, tag, now_iso(), alert_id))
        return next(x for x in self.alerts(a["household_id"]) if x["id"] == alert_id)

    # =============================================================== maintenance planner (predicted -> recommended -> confirmed -> completed)
    def refresh_plan(self, hid, forecasts, model=None):
        h = self.household(hid)
        day = h["now_day"]
        for stage, fc in forecasts.items():
            open_row = self.db.one("SELECT * FROM maintenance WHERE household_id=? AND stage=? AND status IN ('predicted','recommended')",
                                   (hid, stage))
            if fc["status"] == "withheld":
                if open_row:
                    self.db.exec("UPDATE maintenance SET note=?, updated_at=? WHERE id=?",
                                 (f"latest forecast withheld: {fc['reason']}", now_iso(), open_row["id"]))
                continue
            iv = fc["interval_days"]
            lo = day + int(math.ceil(iv[0] if iv else fc["point_days"]))
            hi = day + int(math.ceil(iv[1] if iv else fc["point_days"]))
            basis = iv[0] if iv else fc["point_days"]
            status = "recommended" if basis <= AL.LEAD_DAYS else "predicted"
            note = "from a calibrated interval" if iv else "from a point estimate (no calibrated interval)"
            if open_row:
                self.db.exec("UPDATE maintenance SET status=?, window_lo=?, window_hi=?, model=?, note=?, updated_at=? WHERE id=?",
                             (status, lo, hi, fc["model"], note, now_iso(), open_row["id"]))
            elif not self.db.one("SELECT 1 FROM maintenance WHERE household_id=? AND stage=? AND status='confirmed'", (hid, stage)):
                self.db.exec("INSERT INTO maintenance(household_id,stage,status,window_lo,window_hi,created_day,model,note,created_at,updated_at)"
                             " VALUES(?,?,?,?,?,?,?,?,?,?)", (hid, stage, status, lo, hi, day, fc["model"], note, now_iso(), now_iso()))

    def plan(self, hid, include_closed=True) -> list[dict]:
        sql = "SELECT * FROM maintenance WHERE household_id=?" + ("" if include_closed else " AND status NOT IN ('completed','cancelled')")
        return self.db.q(sql + " ORDER BY id DESC", (hid,))

    def _plan_row(self, plan_id):
        r = self.db.one("SELECT * FROM maintenance WHERE id=?", (plan_id,))
        if not r:
            raise NotFound(f"maintenance entry {plan_id} not found")
        return r

    def confirm_plan(self, plan_id, planned_day=None) -> dict:
        r = self._plan_row(plan_id)
        if r["status"] not in ("predicted", "recommended"):
            raise Conflict(f"only predicted/recommended entries can be confirmed (this one is '{r['status']}')")
        now = self.household(r["household_id"])["now_day"]
        day = r["window_lo"] if planned_day is None else int(planned_day)
        if day < now:
            raise BadRequest("the planned day cannot be in the past")
        self.db.exec("UPDATE maintenance SET status='confirmed', planned_day=?, updated_at=? WHERE id=?", (day, now_iso(), plan_id))
        return self._plan_row(plan_id)

    def cancel_plan(self, plan_id) -> dict:
        r = self._plan_row(plan_id)
        if r["status"] in ("completed", "cancelled"):
            raise Conflict(f"entry is already {r['status']}")
        self.db.exec("UPDATE maintenance SET status='cancelled', updated_at=? WHERE id=?", (now_iso(), plan_id))
        return self._plan_row(plan_id)

    def record_replacement(self, hid, stage, note=None, plan_id=None) -> dict:
        """Record that the filter was replaced TODAY. For a synthetic household this re-runs the simulator with a forced
        replacement on this day: readings up to yesterday are unchanged, the new filter's readings follow."""
        self._stage(stage)
        h = self.household(hid)
        if h["source"] != "synthetic":
            raise Conflict("for uploaded telemetry, send new readings with the replacement day instead (ingest_telemetry)")
        day = h["now_day"]
        if day <= 0:
            raise BadRequest("cannot record a replacement on day 0")
        forced = h["forced"]
        if day in forced.get(stage, []):
            raise Conflict(f"a {stage} replacement is already recorded for day {day}")
        forced.setdefault(stage, []).append(day)
        self.db.exec("UPDATE households SET forced_json=? WHERE id=?", (dumps(forced), hid))
        self._regenerate(hid)
        rows = self.db.q("SELECT id FROM maintenance WHERE household_id=? AND stage=? AND status IN ('predicted','recommended','confirmed')",
                         (hid, stage))
        for r in rows:
            self.db.exec("UPDATE maintenance SET status='completed', completed_day=?, updated_at=?, note=COALESCE(note,'') || ? WHERE id=?",
                         (day, now_iso(), " [replacement recorded by user]" + (f": {note}" if note else ""), r["id"]))
        if not rows:
            self.db.exec("INSERT INTO maintenance(household_id,stage,status,created_day,completed_day,note,created_at,updated_at) "
                         "VALUES(?,?,'completed',?,?,?,?,?)", (hid, stage, day, day, note or "replacement recorded by user", now_iso(), now_iso()))
        self.refresh_alerts(hid, force_clear=True)      # a recorded replacement is a certain fix: clear immediately
        return {"household_id": hid, "stage": stage, "day": day, "plan": self.plan(hid)}

    def complete_plan(self, plan_id, note=None) -> dict:
        r = self._plan_row(plan_id)
        if r["status"] not in ("predicted", "recommended", "confirmed"):
            raise Conflict(f"entry is already {r['status']}")
        return self.record_replacement(r["household_id"], r["stage"], note, plan_id)

    def cost_estimate(self, hid, horizon_days=90, cost: CostConfig | None = None) -> dict:
        """Spend for entries whose predicted window starts within the horizon. ILLUSTRATIVE prices (CostConfig)."""
        cost = cost or CostConfig()
        now = self.household(hid)["now_day"]
        items = [p for p in self.plan(hid, include_closed=False) if p["window_lo"] is not None and p["window_lo"] <= now + horizon_days]
        lines = [{"stage": p["stage"], "status": p["status"], "window": [p["window_lo"], p["window_hi"]],
                  "replacement_cost": cost.replacement_cost[p["stage"]]} for p in items]
        parts = sum(x["replacement_cost"] for x in lines)
        visit = cost.service_visit_cost if lines else 0.0
        return {"horizon_days": horizon_days, "lines": lines, "parts": parts, "service_visit": visit, "total": parts + visit,
                "note": cost.status}

    # =============================================================== monitoring loop, replay, simulator controls
    def update_monitoring(self, hid, model=None, day=None) -> dict:
        fcs = self.forecast_all(hid, model, day, store=True, explain=False)
        alerts = self.refresh_alerts(hid, model, day, forecasts=fcs)
        if day is None or day == self.household(hid)["now_day"]:
            self.refresh_plan(hid, fcs, model)
        return {"alerts": alerts, "statuses": {s: f["status"] for s, f in fcs.items()}}

    def advance(self, hid, days, step=7, model=None) -> dict:
        """Demo replay: move the household's clock forward, computing forecasts, alerts and plan at each step."""
        h = self.household(hid)
        if h["source"] != "synthetic":
            raise Conflict("only synthetic demo households have a simulated clock")
        if not 1 <= int(step) <= 30 or not 1 <= int(days) <= 365:
            raise BadRequest("days must be 1-365 and step 1-30")
        target = min(h["now_day"] + int(days), h["n_days"] - 1)
        if target <= h["now_day"]:
            raise Conflict("already at the end of the simulated timeline")
        stops = list(range(h["now_day"] + int(step), target + 1, int(step)))
        if not stops or stops[-1] != target:
            stops.append(target)
        for d in stops:
            self.db.exec("UPDATE households SET now_day=? WHERE id=?", (d, hid))
            self.update_monitoring(hid, model, d)
        return {"household_id": hid, "now_day": target, "steps": len(stops)}

    def inject_fault(self, hid, spec) -> dict:
        """Add a sensor fault starting today or later (the past is fixed): readings before `start` are unchanged."""
        h = self.household(hid)
        if h["source"] != "synthetic":
            raise Conflict("faults can only be injected into synthetic households")
        self._validate_faults([spec])
        if int(spec["start"]) < h["now_day"]:
            raise BadRequest(f"a fault can only start on day {h['now_day']} (today) or later; earlier readings cannot be rewritten")
        self.db.exec("UPDATE households SET user_faults_json=? WHERE id=?", (dumps(h["user_faults"] + [dict(spec)]), hid))
        self._regenerate(hid)
        return self.household(hid)

    def what_if(self, hid, overrides=None, user_faults=None, model=None, stage="sediment") -> dict:
        """HYPOTHETICAL: re-simulate this household with changed parameters / faults (nothing is saved) and compare the
        forecast at the same day. It describes a changed synthetic world, not the user's actual purifier."""
        self._stage(stage)
        h = self.household(hid)
        if h["source"] != "synthetic":
            raise Conflict("what-if analysis needs a synthetic household")
        self._validate_faults(user_faults)
        alt = self._generate(h, extra_overrides=overrides, extra_faults=user_faults or [])
        day = min(h["now_day"], len(alt) - 1)
        hist_alt = alt[alt["day"] <= day][VISIBLE_COLS].reset_index(drop=True)
        f = self._forecaster(model, explain=False)
        base = self.forecast(hid, stage, model, store=False, explain=False)
        hyp = f.run(hist_alt, stage)
        t_alt = alt[dtc_col(stage)].iloc[day]
        return {"label": "HYPOTHETICAL scenario: a re-simulated synthetic world, not your purifier", "stage": stage, "day": day,
                "changes": {"overrides": overrides or {}, "user_faults": user_faults or []},
                "base": {k: base[k] for k in ("status", "point_days", "interval_days", "reason")},
                "hypothetical": {k: hyp[k] for k in ("status", "point_days", "interval_days", "reason")},
                "simulator_truth_base": self.truth(hid, stage),
                "simulator_truth_hypothetical": None if not np.isfinite(t_alt) else float(t_alt)}

    # =============================================================== dashboard
    def dashboard(self, hid, model=None) -> dict:
        h = self.household(hid)
        fcs = self.forecast_all(hid, model, store=False, explain=False)
        dq = self.data_quality(hid)
        stages = {}
        for s, fc in fcs.items():
            iv = fc["interval_days"]
            basis = None if fc["status"] == "withheld" else (iv[0] if iv else fc["point_days"])
            stages[s] = {"label": STAGE_LABELS[s], "condition": fc["indicator"], "sensor_health": fc["sensor_health"],
                         "reliability": fc["reliability"], "status": fc["status"], "point_days": fc["point_days"],
                         "interval_days": iv, "reason": fc["reason"], "approaching": bool(basis is not None and basis <= 30),
                         "recommendation": fc["recommendation"]}
        open_alerts = self.alerts(hid, status="open")
        meta = self.registry.meta()
        name = model or self.registry.default_model()
        return {"household": {k: h[k] for k in ("id", "name", "source", "scenario", "now_day", "n_days")},
                "data_source_label": ("SYNTHETIC demo data (replayed simulator output, not real-time telemetry)"
                                      if h["source"] == "synthetic" else "UPLOADED telemetry"),
                "model": {"name": name, "trained_with": meta.get(name), "card": MODEL_CARDS.get(name)},
                "stages": stages, "n_filter_stages": len(STAGES),
                "approaching": [s for s, v in stages.items() if v["approaching"]],
                "withheld": [s for s, v in stages.items() if v["status"] == "withheld"],
                "sensor_warnings": [s for s, v in stages.items() if v["sensor_health"] != "ok"],
                "data_quality": dq, "open_alerts": len(open_alerts), "recent_alerts": open_alerts[:5],
                "upcoming_maintenance": [p for p in self.plan(hid, include_closed=False)][:6],
                "recent_maintenance": [p for p in self.plan(hid) if p["status"] == "completed"][:5], "disclaimer": DISCLAIMER}

    # =============================================================== validation lab (real harness, fresh households)
    def run_validation(self, n_households=10, seed=2000, methods=None, calibrated=True, cost: dict | None = None) -> dict:
        """Score forecasters on FRESH synthetic households with the project's strict evaluation (past data only, paired
        criteria, preset thresholds). Learned models are refused if the seed equals their training seed (no leakage)."""
        import time
        n_households = int(n_households)
        if not 4 <= n_households <= 60:
            raise BadRequest("n_households must be between 4 and 60")
        baselines = ["trend", "volume_rule", "health_20pct"]
        methods = list(methods or baselines)
        for b in baselines:
            if b not in methods:
                methods.append(b)
        learned = [m for m in methods if m in ("reference_v3", "person2")]
        for m in methods:
            if m not in baselines + ["reference_v3", "person2"]:
                raise BadRequest(f"unknown method '{m}'")
        for m in learned:
            if not os.path.exists(self.registry._paths()[m]):
                raise Conflict(f"model '{m}' has not been trained yet")
            if self.registry.train_seed(m) == int(seed):
                raise BadRequest(f"seed {seed} was used to train '{m}'; pick a different seed so the households are unseen")
        df, _ = generate_dataset(n_households, law="A", seed=int(seed), cfg=SimConfig.calibrated() if calibrated else None)
        fc = {m: self.registry.get(m, explain=False) for m in methods}
        t0 = time.time()
        res = evaluate(df, fc, cfg=EvalConfig())
        cfg = res.cfg
        if res.opps.empty:
            raise Conflict("no forecast opportunities were generated; use more households")
        cyc = M.cycle_outcomes(res.opps, cfg)
        cc = CostConfig(**cost) if cost else CostConfig()
        out = {"params": {"n_households": n_households, "seed": int(seed), "methods": methods, "calibrated": calibrated},
               "n_forecasts": int(len(res.opps)), "n_filter_lives": int(cyc.groupby("method").size().iloc[0]),
               "n_short_excluded": res.n_short_excluded, "seconds": round(time.time() - t0, 1),
               "window": M.window_summary(res.opps, cfg), "late_false": M.late_false_summary(cyc),
               "by_stage": M.window_summary(res.opps, cfg, by=("method", "stage")),
               "intervals": M.interval_summary(res.opps, cfg), "cost_policies": policy_table(cyc, cc),
               "criteria": {m: check_success(res.opps, cfg, model=m) for m in learned},
               "definitions": M.__doc__.strip(),
               "caveat": "Synthetic data from an internal simulator: shows algorithmic feasibility, not real-world accuracy. "
                         "Small samples have wide uncertainty."}
        out = clean(out)
        out["id"] = self.db.exec("INSERT INTO experiments(created_at,kind,params_json,result_json) VALUES(?,?,?,?)",
                                 (now_iso(), "validation", dumps(out["params"]), dumps(out)))
        return out

    def experiments(self) -> list[dict]:
        return [{"id": r["id"], "created_at": r["created_at"], "kind": r["kind"], "params": loads(r["params_json"])}
                for r in self.db.q("SELECT * FROM experiments ORDER BY id DESC")]

    def experiment(self, exp_id) -> dict:
        r = self.db.one("SELECT * FROM experiments WHERE id=?", (exp_id,))
        if not r:
            raise NotFound(f"experiment {exp_id} not found")
        return loads(r["result_json"])

    # =============================================================== exports
    def export_csv(self, kind, hid=None, include_truth=False) -> str:
        if kind == "telemetry":
            if not hid:
                raise BadRequest("telemetry export needs a household id")
            h = self.household(hid)
            df = self.readings(hid, include_truth=include_truth and h["source"] == "synthetic")
            df = df.drop(columns=[c for c in ("data_source",) if c in df.columns])   # the simulator table already carries one
            df.insert(0, "data_source", h["source"])
        elif kind in ("predictions", "alerts", "maintenance"):
            sql, args = f"SELECT * FROM {kind}", ()
            if hid:
                self.household(hid)
                sql, args = sql + " WHERE household_id=?", (hid,)
            cols = [r["name"] for r in self.db.q(f"PRAGMA table_info({kind})")]
            df = pd.DataFrame(self.db.q(sql + " ORDER BY id", args), columns=cols)       # header even when there are no rows
        elif kind == "experiments":
            df = pd.DataFrame(self.experiments(), columns=["id", "created_at", "kind", "params"])
            df["params"] = df["params"].map(dumps)
        else:
            raise BadRequest("kind must be telemetry, predictions, alerts, maintenance or experiments")
        buf = io.StringIO()
        df.to_csv(buf, index=False)
        return buf.getvalue()

    def export_household_config(self, hid) -> dict:
        h = self.household(hid)
        return {k: h[k] for k in ("id", "name", "source", "scenario", "law", "seed", "sim_index", "calibrated", "overrides",
                                  "user_faults", "forced", "n_days", "now_day")} | {"note": "Re-creating a household from this config "
                                                                                     "reproduces its telemetry exactly (synthetic only)."}
