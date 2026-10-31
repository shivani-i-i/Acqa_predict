"""Wear laws. Law "A" = primary simulator, law "B" = second simulator with different functional
forms (used to test generalisation). Both are illustrative models, not validated physics.

x = (hidden household wear multiplier * accumulated load) / reference load; the criterion is crossed at x = 1
for sediment, carbon and RO flux (rejection can cross first).
"""
from __future__ import annotations
import numpy as np
from ..schema import SEDIMENT_DP_RATIO_LIMIT as SED_LIMIT, RO_FLUX_LIMIT, RO_REJECTION_LIMIT

LAWS = ("A", "B")


def sediment_ratio(x, law):
    """(dp/flow) relative to clean. A: linear cake growth. B: super-linear (pore blocking -> cake)."""
    x = np.asarray(x, dtype=float)
    return 1.0 + 0.8 * (x if law == "A" else x ** 1.6)


def ro_flux(x, law):
    """Fouling-only relative flux. A: hyperbolic. B: stretched exponential. Both equal 0.70 at x = 1."""
    x = np.asarray(x, dtype=float)
    if law == "A":
        return 1.0 / (1.0 + (3.0 / 7.0) * x)
    return np.exp(-np.log(1.0 / 0.70) * x ** 1.2)


def ro_rejection(x, rej0, sens, law):
    x = np.asarray(x, dtype=float)
    if law == "A":
        return rej0 - 0.09 * sens * x ** 2.5
    return rej0 - 0.07 * sens * x ** 1.8


def temp_factor(temp_c, law):
    """True temperature effect on flow. A: +3%/degC (what a simple 'corrected flux' feature assumes).
    B: exponential 2.5%/degC (so a fixed 3% correction is slightly wrong on B)."""
    t = np.asarray(temp_c, dtype=float) - 25.0
    return 1.0 + 0.03 * t if law == "A" else np.exp(0.025 * t)


def ro_tds_exponent(law):
    return 0.7 if law == "A" else 1.0


def carbon_temp_weight(temp_c, law):
    return 1.0 if law == "A" else 1.0 + 0.02 * (float(temp_c) - 25.0)


def sediment_crossed(ratio):
    return ratio >= SED_LIMIT - 1e-9


def ro_crossed(flux, rej):
    return (flux <= RO_FLUX_LIMIT + 1e-9) or (rej <= RO_REJECTION_LIMIT + 1e-9)


def carbon_crossed(x):
    return x >= 1.0 - 1e-9
