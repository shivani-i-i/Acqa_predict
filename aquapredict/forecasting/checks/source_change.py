"""Two-sided CUSUM for persistent shifts in a signal such as input TDS.

A detected shift is reported as a POSSIBLE source or operating-condition change
(new water supply, tanker refill, sensor re-calibration, ...). It is NOT proof
that the water source changed.

Algorithm (standardised CUSUM)
------------------------------
z_t = clip((x_t - mu) / sigma, -z_clip, +z_clip)
S+_t = max(0, S+_{t-1} + z_t - k)      S-_t = max(0, S-_{t-1} - z_t - k)
alarm when S+ > h (upward shift) or S- > h (downward shift).

* mu, sigma: supplied (``reference_mean`` / ``reference_std``) or estimated from the
  first ``init_n`` valid readings, which are then NOT monitored (so a shift inside
  that baseline window cannot be detected). The estimate is ROBUST (median and
  1.4826*MAD, falling back to the sample std only if MAD is zero) so that one spike
  inside the baseline window cannot inflate sigma and blind the detector.
* k (allowance) and h (threshold) are in sigma units. Defaults k=1.0, h=8.0 are
  placeholders. A one-off simulation (1,500 streams of 150 readings, Gaussian noise
  with 3 % relative sd, seed 1) gave ~1.7 % false-alarm streams, ~99.5 % detection of a
  25 % step at reading 75 and a median delay of 2 readings. That is SYNTHETIC noise
  only; real TDS data will differ, so re-tune on real data.
* z_clip stops one isolated spike from triggering an alarm alone.
* NaN readings are skipped (statistic unchanged).
* Reset: ``detect_shift`` reports the FIRST alarm and stops. ``detect_all_shifts``
  restarts after each alarm, re-estimating the baseline from the following readings.
* Estimated change index = last index where the statistic was zero before the alarm.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

DISCLAIMER = ("Possible source or operating-condition change (statistical signal only; "
              "not proof that the water source changed).")


@dataclass
class CusumConfig:
    reference_mean: Optional[float] = None
    reference_std: Optional[float] = None
    init_n: int = 30
    k_sigma: float = 1.0
    h_sigma: float = 8.0
    z_clip: float = 4.0
    min_sigma_rel: float = 0.02   # sigma floor = 2 % of |mean| (avoids zero-variance blow-ups)


def _robust_location_scale(values: np.ndarray):
    """Median and MAD-based sigma; falls back to std when MAD is 0 (e.g. quantised data)."""
    mu = float(np.median(values))
    sigma = 1.4826 * float(np.median(np.abs(values - mu)))
    if sigma <= 0.0:
        sigma = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return mu, sigma


def _stamp(timestamps, i):
    if timestamps is None:
        return None
    return pd.Timestamp(timestamps[i]).isoformat()


def detect_shift(values: Sequence[float], timestamps: Optional[Sequence] = None,
                 config: Optional[CusumConfig] = None, start: int = 0) -> Dict[str, object]:
    cfg = config or CusumConfig()
    x = np.asarray(values, dtype=float)
    base = {"detected": False, "direction": None, "alarm_index": None, "change_index": None,
            "alarm_timestamp": None, "change_timestamp": None, "baseline_mean": None,
            "baseline_std": None, "statistic": 0.0, "message": "No persistent shift detected."}
    valid_idx = [i for i in range(start, len(x)) if np.isfinite(x[i])]
    if cfg.reference_mean is not None:
        mu = float(cfg.reference_mean)
        monitored = valid_idx
        sigma = cfg.reference_std
        if sigma is None:
            if len(valid_idx) < cfg.init_n:
                base["message"] = "Not enough readings to estimate variability."
                return base
            _, sigma = _robust_location_scale(x[valid_idx[:cfg.init_n]])
    else:
        if len(valid_idx) <= cfg.init_n:
            base["message"] = f"Need more than {cfg.init_n} valid readings to monitor for a shift."
            return base
        init = x[valid_idx[:cfg.init_n]]
        mu, init_sigma = _robust_location_scale(init)
        sigma = cfg.reference_std if cfg.reference_std is not None else init_sigma
        monitored = valid_idx[cfg.init_n:]
    sigma = max(float(sigma), cfg.min_sigma_rel * abs(mu), 1e-9)
    base.update(baseline_mean=mu, baseline_std=sigma)
    s_pos = s_neg = 0.0
    last_zero_pos = last_zero_neg = monitored[0] if monitored else 0
    for i in monitored:
        z = float(np.clip((x[i] - mu) / sigma, -cfg.z_clip, cfg.z_clip))
        s_pos = max(0.0, s_pos + z - cfg.k_sigma)
        s_neg = max(0.0, s_neg - z - cfg.k_sigma)
        if s_pos == 0.0:
            last_zero_pos = i
        if s_neg == 0.0:
            last_zero_neg = i
        for stat, direction, lz in ((s_pos, "up", last_zero_pos), (s_neg, "down", last_zero_neg)):
            if stat > cfg.h_sigma:
                change = min(len(x) - 1, lz + 1) if lz < i else i
                base.update(detected=True, direction=direction, alarm_index=int(i), change_index=int(change),
                            alarm_timestamp=_stamp(timestamps, i), change_timestamp=_stamp(timestamps, change),
                            statistic=float(stat),
                            message=f"{DISCLAIMER} Sustained {direction}ward shift in the signal "
                                    f"(baseline mean {mu:.4g}).")
                return base
        base["statistic"] = float(max(s_pos, s_neg))
    return base


def detect_all_shifts(values: Sequence[float], timestamps: Optional[Sequence] = None,
                      config: Optional[CusumConfig] = None) -> List[Dict[str, object]]:
    """Repeatedly detect shifts, restarting (new baseline) after each alarm."""
    cfg = config or CusumConfig()
    # a fixed reference only makes sense for the first segment; later segments re-estimate
    first = cfg
    later = CusumConfig(None, None, cfg.init_n, cfg.k_sigma, cfg.h_sigma, cfg.z_clip, cfg.min_sigma_rel)
    out, start, current = [], 0, first
    while True:
        res = detect_shift(values, timestamps, current, start=start)
        if not res["detected"]:
            return out
        out.append(res)
        start = res["alarm_index"] + 1
        current = later
