"""Sensor-fault injection with ground-truth labels."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from ..schema import FAULTABLE_SENSORS

KINDS = ("stuck", "drift", "spike", "dropout", "tds_inversion")


@dataclass
class FaultSpec:
    kind: str
    sensor: str
    start: int
    duration: int
    magnitude: float = 0.0


def sample_faults(rng, templates, n_days):
    """templates: dicts {kind, prob, count, sensors, duration_range, magnitude_range, min_start}."""
    specs = []
    for tpl in templates or []:
        if rng.random() >= tpl.get("prob", 1.0):
            continue
        for _ in range(int(tpl.get("count", 1))):
            kind = tpl["kind"]
            sensor = "out_tds_ppm" if kind == "tds_inversion" else str(rng.choice(tpl.get("sensors", FAULTABLE_SENSORS)))
            d_lo, d_hi = tpl.get("duration_range", (10, 40))
            dur = int(rng.integers(d_lo, d_hi + 1))
            lo = int(tpl.get("min_start", 30))
            if n_days < lo + dur + 2:
                continue
            start = int(rng.integers(lo, max(lo + 1, n_days - dur - 1)))
            m_lo, m_hi = tpl.get("magnitude_range", (0.0, 0.0))
            mag = float(rng.uniform(m_lo, m_hi))
            if kind == "drift" and rng.random() < 0.5:
                mag = -mag
            specs.append(FaultSpec(kind, sensor, start, dur, mag))
    return specs


def apply_faults(meas, specs, idle):
    """Modify `meas` (dict of arrays) in place. Returns (fault_type, fault_sensor) label arrays.
    Idle (vacation) days have no readings, so faults never write there."""
    n = len(idle)
    ftype = np.full(n, "none", dtype=object)
    fsensor = np.full(n, "none", dtype=object)

    def mark(idx, kind, sensor):
        idx = idx[~idle[idx]]
        free = idx[ftype[idx] == "none"]
        ftype[free] = kind
        fsensor[free] = sensor

    for sp in specs:
        x = meas[sp.sensor]
        s = sp.start
        e = min(n, s + sp.duration)
        if s >= n:
            continue
        if sp.kind == "stuck":
            idx = np.arange(s, e)
            valid = [i for i in range(s, n) if np.isfinite(x[i]) and not idle[i]]
            if not valid:
                continue
            x[idx[~idle[idx]]] = x[valid[0]]
            mark(idx, "stuck", sp.sensor)
        elif sp.kind == "drift":
            idx = np.arange(s, n)               # drift persists after the ramp
            ramp = np.minimum(1.0, (idx - s) / max(1, sp.duration))
            x[idx] = x[idx] * (1.0 + sp.magnitude * ramp)
            mark(idx, "drift", sp.sensor)
        elif sp.kind == "spike":
            if not idle[s]:
                x[s] = x[s] * max(1.5, abs(sp.magnitude) if sp.magnitude else 3.0)
                mark(np.array([s]), "spike", sp.sensor)
        elif sp.kind == "dropout":
            idx = np.arange(s, e)
            x[idx[~idle[idx]]] = np.nan
            mark(idx, "dropout", sp.sensor)
        elif sp.kind == "tds_inversion":
            idx = np.arange(s, e)
            fac = max(1.05, sp.magnitude if sp.magnitude else 1.2)
            idx = idx[~idle[idx]]
            meas["out_tds_ppm"][idx] = meas["feed_tds_ppm"][idx] * fac
            mark(idx, "tds_inversion", "out_tds_ppm")
    return ftype, fsensor
