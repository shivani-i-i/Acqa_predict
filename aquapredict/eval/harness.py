"""Evaluation harness: calls forecasters on held-out households with PAST data only.

Leakage protection: forecasters receive only schema.VISIBLE_COLS rows up to the forecast day.
Truth (true_*, fault_*, scenario, ...) is read by the harness only to score the forecast.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd

from ..schema import STAGES, VISIBLE_COLS, repl_col, dtc_col
from .config import EvalConfig
from ..validate import assert_valid


def _rss_mb() -> float:
    """Return the current process RSS in MB on Unix and Windows."""
    try:
        import resource
    except ImportError:
        import ctypes
        import ctypes.wintypes

        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.wintypes.DWORD),
                ("PageFaultCount", ctypes.wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(counters)
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        ok = ctypes.windll.psapi.GetProcessMemoryInfo(
            handle, ctypes.byref(counters), counters.cb
        )
        if not ok:
            return 0.0
        return counters.WorkingSetSize / (1024 * 1024)
    usage = resource.getrusage(resource.RUSAGE_SELF)
    # Linux reports KiB; macOS reports bytes.
    return usage.ru_maxrss / (1024 if usage.ru_maxrss < 10**9 else 1024**2)


@dataclass
class EvalResult:
    opps: pd.DataFrame          # one row per (method, forecast opportunity)
    cfg: EvalConfig
    n_households: int
    n_short_excluded: int = 0   # filter lives too short to forecast with >= min_notice_days of notice


def _f(x):
    return np.nan if x is None else float(x)


def evaluate(df, forecasters: dict, stages=STAGES, cfg: EvalConfig | None = None) -> EvalResult:
    """df: telemetry (with truth) of HELD-OUT households. forecasters: name -> f(hist, stage)->dict."""
    cfg = cfg or EvalConfig()
    assert_valid(df, require_truth=True)           # fail loudly on malformed input instead of mis-scoring it
    rows, n_short = [], 0
    for hid, g in df.groupby("household_id", sort=False):
        g = g.sort_values("day").reset_index(drop=True)
        assert (g["day"].to_numpy() == np.arange(len(g))).all(), "days must be contiguous from 0"
        vis = g[VISIBLE_COLS]
        scen = g["scenario"].iloc[0]
        for stage in stages:
            rep = g[repl_col(stage)].to_numpy()
            dtc = g[dtc_col(stage)].to_numpy()
            starts = np.r_[0, np.flatnonzero(rep)[np.flatnonzero(rep) > 0]]
            ends = np.r_[starts[1:], len(g)]
            for k, (s, e) in enumerate(zip(starts, ends)):
                hit = np.flatnonzero(dtc[s:e] == 0)
                if len(hit) == 0:
                    continue                       # censored cycle: criterion never reached
                c = int(s + hit[0])
                if c - (int(s) + cfg.min_age_days) < cfg.min_notice_days:
                    n_short += 1       # unwinnable for ANY method; excluded for all methods and reported
                    continue
                for t in range(int(s) + cfg.min_age_days, c + 1, cfg.stride):
                    hist = vis.iloc[: t + 1]
                    for name, f in forecasters.items():
                        r = f(hist, stage)
                        rows.append({
                            "method": name, "household_id": hid, "scenario": scen, "stage": stage,
                            "cycle": k, "cycle_start": int(s), "crossing": c, "day": t,
                            "true_dtc": float(c - t), "point": _f(r.get("point")),
                            "lo": _f(r.get("lo")), "hi": _f(r.get("hi")),
                            "withheld": bool(r.get("withheld", False)),
                            "fallback": bool(r.get("fallback", False)),
                            "status": str(r.get("status", "")), "reason": str(r.get("reason", "") or ""),
                        })
    opps = pd.DataFrame(rows)
    if len(opps):
        use_lo = cfg.alert_uses_lower_bound & np.isfinite(opps["lo"])
        av = np.where(use_lo, opps["lo"], opps["point"])
        opps["alert_value"] = np.where(opps["withheld"], np.nan, av)
    return EvalResult(opps=opps, cfg=cfg, n_households=df["household_id"].nunique(), n_short_excluded=n_short)


def evaluate_batched(df, forecasters: dict, stages=STAGES, cfg: EvalConfig | None = None,
                     batch_households: int = 10, progress=None, checkpoint: str | None = None) -> EvalResult:
    """Same result as evaluate(), but processes households in batches so long runs report progress, can be
    checkpointed (partial opportunities written to `checkpoint` after every batch) and hold less memory.
    progress(done_households, total_households, rss_mb) is called after each batch."""
    cfg = cfg or EvalConfig()
    assert_valid(df, require_truth=True)
    ids = list(df["household_id"].unique())
    parts, n_short = [], 0
    for i in range(0, len(ids), batch_households):
        chunk = ids[i:i + batch_households]
        r = evaluate(df[df["household_id"].isin(chunk)], forecasters, stages=stages, cfg=cfg)
        parts.append(r.opps)
        n_short += r.n_short_excluded
        if checkpoint:
            pd.concat(parts, ignore_index=True).to_pickle(checkpoint)
        if progress:
            progress(min(i + batch_households, len(ids)), len(ids), _rss_mb())
    opps = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    return EvalResult(opps=opps, cfg=cfg, n_households=len(ids), n_short_excluded=n_short)
