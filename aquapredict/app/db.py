"""SQLite persistence. Local prototype: one file, no server. All timestamps are ISO strings (UTC)."""
from __future__ import annotations
import datetime as dt
import json
import sqlite3
import threading

SCHEMA = """
CREATE TABLE IF NOT EXISTS households(
  id TEXT PRIMARY KEY, name TEXT NOT NULL, source TEXT NOT NULL,           -- 'synthetic' | 'uploaded'
  scenario TEXT, law TEXT, seed INTEGER, sim_index INTEGER, calibrated INTEGER,
  overrides_json TEXT, user_faults_json TEXT, forced_json TEXT,
  n_days INTEGER, now_day INTEGER, created_at TEXT);
CREATE TABLE IF NOT EXISTS predictions(
  id INTEGER PRIMARY KEY AUTOINCREMENT, household_id TEXT NOT NULL, stage TEXT NOT NULL, day INTEGER NOT NULL,
  model TEXT NOT NULL, model_version TEXT, status TEXT NOT NULL, point REAL, lo REAL, hi REAL, reason TEXT,
  extra_json TEXT, created_at TEXT, UNIQUE(household_id, stage, day, model));
CREATE TABLE IF NOT EXISTS alerts(
  id INTEGER PRIMARY KEY AUTOINCREMENT, household_id TEXT NOT NULL, stage TEXT, kind TEXT NOT NULL,
  dedupe_key TEXT NOT NULL, severity TEXT NOT NULL, status TEXT NOT NULL,   -- open|acknowledged|resolved|dismissed
  origin TEXT NOT NULL DEFAULT 'model', title TEXT, message TEXT, recommended_action TEXT, details_json TEXT,
  first_day INTEGER, last_day INTEGER, cleared INTEGER NOT NULL DEFAULT 0, clear_streak INTEGER NOT NULL DEFAULT 0, note TEXT, created_at TEXT, updated_at TEXT);
CREATE TABLE IF NOT EXISTS maintenance(
  id INTEGER PRIMARY KEY AUTOINCREMENT, household_id TEXT NOT NULL, stage TEXT NOT NULL,
  status TEXT NOT NULL,                       -- predicted|recommended|confirmed|completed|cancelled
  window_lo INTEGER, window_hi INTEGER, planned_day INTEGER, created_day INTEGER, completed_day INTEGER,
  model TEXT, note TEXT, created_at TEXT, updated_at TEXT);
CREATE TABLE IF NOT EXISTS experiments(
  id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT, kind TEXT, params_json TEXT, result_json TEXT);
"""


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


class Database:
    def __init__(self, path: str = ":memory:"):
        self.path = path
        self.lock = threading.RLock()
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        with self.lock:
            self.conn.executescript(SCHEMA)
            self.conn.commit()

    def q(self, sql: str, args=()) -> list[dict]:
        with self.lock:
            return [dict(r) for r in self.conn.execute(sql, args).fetchall()]

    def one(self, sql: str, args=()) -> dict | None:
        r = self.q(sql, args)
        return r[0] if r else None

    def exec(self, sql: str, args=()) -> int:
        with self.lock:
            cur = self.conn.execute(sql, args)
            self.conn.commit()
            return cur.lastrowid

    def close(self):
        with self.lock:
            self.conn.close()


def dumps(o) -> str:
    return json.dumps(o, default=lambda x: None if x != x else (float(x) if hasattr(x, "__float__") else str(x)))


def loads(s, default=None):
    return default if s in (None, "") else json.loads(s)
