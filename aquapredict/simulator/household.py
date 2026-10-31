"""Simulate ONE household for n_days (daily time step), with ground-truth replacement labels."""
from __future__ import annotations
import numpy as np
import pandas as pd

from ..schema import (STAGES, FAULTABLE_SENSORS, SIM_REF_VOLUME_L, DATA_SOURCE_SYNTHETIC,
                      age_col, vol_col, repl_col, dtc_col)
from . import wear as W
from .faults import sample_faults, apply_faults, FaultSpec, KINDS
from .config import SimConfig


def simulate_household(index, scenario, cfg: SimConfig, law="A", seed=0, extra_faults=(),
                       forced_replacements=None, user_faults=None):
    """forced_replacements: {stage: [day, ...]} - the household replaces that stage's filter at the start of that day
    (used by the app's "record replacement" workflow). user_faults: [{kind, sensor, start, duration, magnitude}] -
    explicit, deterministic faults that consume NO random numbers. Both leave every reading BEFORE the event unchanged."""
    if law not in W.LAWS:
        raise ValueError(f"law must be one of {W.LAWS}")
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(index), 7]))
    n = int(cfg.n_days)
    days = np.arange(n)

    # ---------------- exogenous drivers ----------------
    base_use = float(np.clip(rng.lognormal(np.log(cfg.usage_median_L), cfg.usage_sigma),
                             cfg.usage_min_L, cfg.usage_max_L))
    weekly = np.where(days % 7 >= 5, cfg.weekend_factor, 1.0)
    usage = base_use * weekly * rng.lognormal(0, cfg.daily_usage_sigma, n)
    idle = np.zeros(n, bool)
    for _ in range(rng.poisson(cfg.vacation_rate_per_year * n / 365.0)):
        s, L = int(rng.integers(0, n)), int(rng.integers(4, 15))
        idle[s:s + L] = True
    usage[idle] = 0.0

    tmean = rng.uniform(*cfg.temp_mean_range)
    amp = rng.uniform(*cfg.temp_amp_range)
    phase = rng.uniform(0, 2 * np.pi)
    temp = tmean + amp * np.sin(2 * np.pi * days / 365.0 + phase) + rng.normal(0, 1.0, n)

    inlet_base = max(1.2, rng.normal(cfg.inlet_mean_bar, cfg.inlet_sd_bar))
    inlet = np.maximum(0.8, inlet_base + rng.normal(0, 0.1, n))
    low = np.zeros(n, bool)
    for _ in range(rng.poisson(cfg.low_inlet_rate_per_year * n / 365.0)):
        s, L = int(rng.integers(0, n)), int(rng.integers(5, 21))
        low[s:s + L] = True
    inlet = np.where(low, inlet * cfg.low_inlet_factor, inlet)

    feed_base = float(np.clip(rng.lognormal(np.log(cfg.feed_tds_median_ppm), cfg.feed_tds_sigma),
                              cfg.feed_tds_min_ppm, cfg.feed_tds_max_ppm))
    k_changes = max(int(rng.poisson(cfg.source_change_rate_per_year * n / 365.0)), int(cfg.source_change_min))
    change_days = []
    if n > 60 and k_changes > 0:
        change_days = sorted(rng.choice(np.arange(20, n - 20), size=min(k_changes, n - 40), replace=False).tolist())
    tds_mult = np.ones(n)
    partic_mult = np.ones(n)
    change_flag = np.zeros(n, bool)
    regime = np.zeros(n, int)
    cur_t = cur_p = 1.0
    for r, d in enumerate(change_days, 1):
        up = rng.random() < 0.75
        f = float(np.exp(rng.uniform(0.3, 0.9)) if up else np.exp(-rng.uniform(0.2, 0.5)))
        cur_t = float(np.clip(cur_t * f, 0.4, 4.0))
        cur_p = float(np.clip(cur_p * f ** cfg.partic_coupling * rng.lognormal(0, 0.2), 0.3, cfg.partic_mult_cap))
        tds_mult[d:] = cur_t
        partic_mult[d:] = cur_p
        change_flag[d] = True
        regime[d:] = r
    feed = np.maximum(20.0, feed_base * tds_mult * (1 + rng.normal(0, 0.02, n)))

    partic_base = float(rng.lognormal(0, cfg.partic_sigma))
    ramp = np.ones(n)
    if cfg.partic_ramp_start >= 0:
        ramp = np.minimum(cfg.partic_ramp_cap, 1.0 + cfg.partic_ramp_per_day * np.maximum(0, days - cfg.partic_ramp_start))
    partic = partic_base * partic_mult * ramp * rng.lognormal(0, 0.1, n)
    organic_base = float(rng.lognormal(0, cfg.organic_sigma))
    organic = organic_base * rng.lognormal(0, 0.1, n)

    # hidden per-household wear multipliers (what a model must infer)
    w = np.exp(rng.normal(cfg.wear_log_mean, cfg.wear_log_sigma, 3))   # sediment, carbon, ro
    s_rej = float(np.exp(rng.normal(0, cfg.rej_sens_sigma)))

    # ---------------- physics loop with replacement cycles ----------------
    ref = [SIM_REF_VOLUME_L[s] for s in STAGES]
    alpha = W.ro_tds_exponent(law)
    loads = [0.0, 0.0, 0.0]            # particulate, organic, scaling-load (stage order)
    forced = [set(int(d) for d in (forced_replacements or {}).get(st, []) if int(d) > 0) for st in STAGES]
    cross_by_cycle = [{}, {}, {}]
    install, vol, cur_cycle = [0, 0, 0], [0.0] * 3, [0, 0, 0]
    due, crossed = [None] * 3, [False] * 3        # scheduled replacement day; criterion already reached this cycle
    rng_delay = np.random.default_rng(np.random.SeedSequence([int(seed), int(index), 13]))   # separate stream: default data unchanged
    cyc = np.zeros((3, n), int)
    age = np.zeros((3, n))
    cumv = np.zeros((3, n))
    repl = np.zeros((3, n), bool)
    ratio = np.empty(n); flux = np.empty(n); rej = np.empty(n); xc = np.empty(n)
    crossings = [[], [], []]
    for t in range(n):
        for j in range(3):
            if (due[j] is not None and t >= due[j]) or t in forced[j]:    # filter replaced at start of this day
                due[j], crossed[j] = None, False
                loads[j], vol[j], install[j] = 0.0, 0.0, t
                cur_cycle[j] += 1
                repl[j, t] = True
        u = usage[t]
        loads[0] += u * partic[t]
        loads[1] += u * organic[t] * W.carbon_temp_weight(temp[t], law)
        loads[2] += u * (feed[t] / cfg.ref_tds_ppm) ** alpha
        for j in range(3):
            vol[j] += u
            age[j, t], cumv[j, t], cyc[j, t] = t - install[j], vol[j], cur_cycle[j]
        ratio[t] = float(W.sediment_ratio(w[0] * loads[0] / ref[0], law))
        xc[t] = w[1] * loads[1] / ref[1]
        x_ro = w[2] * loads[2] / ref[2]
        flux[t] = float(W.ro_flux(x_ro, law))
        rej[t] = float(W.ro_rejection(x_ro, cfg.rej0, s_rej, law))
        crossed_now = (W.sediment_crossed(ratio[t]), W.carbon_crossed(xc[t]), W.ro_crossed(flux[t], rej[t]))
        for j in range(3):
            if crossed_now[j] and not crossed[j]:     # first crossing of this cycle
                crossings[j].append(t)
                cross_by_cycle[j][cur_cycle[j]] = t
                crossed[j] = True
                delay = 0
                if cfg.replacement_delay_mean_days > 0:
                    delay = max(0, int(round(rng_delay.normal(cfg.replacement_delay_mean_days,
                                                              cfg.replacement_delay_sd_days))))
                due[j] = t + 1 + delay

    dtc = np.full((3, n), np.nan)
    for j in range(3):                            # a cycle ended early by a forced replacement has no crossing (NaN = censored)
        dtc[j] = np.array([cross_by_cycle[j].get(int(c), np.nan) for c in cyc[j]], dtype=float) - days

    # ---------------- measurement layer ----------------
    flow_true = cfg.flow0_Lpm * flux * W.temp_factor(temp, law) * np.sqrt(inlet / cfg.ref_inlet_bar)
    dp_true = cfg.dp0_bar * ratio * (flow_true / cfg.flow0_Lpm)
    out_true = feed * (1.0 - rej)
    meas = {
        "usage_L": usage * rng.lognormal(0, 0.01, n),
        "feed_tds_ppm": feed * rng.lognormal(0, cfg.noise_feed, n),
        "out_tds_ppm": out_true * rng.lognormal(0, cfg.noise_tds, n),
        "temp_C": temp + rng.normal(0, cfg.noise_temp_C, n),
        "inlet_bar": inlet + rng.normal(0, cfg.noise_inlet, n),
        "flow_Lpm": flow_true * rng.lognormal(0, cfg.noise_flow, n),
        "dp_sediment_bar": dp_true * rng.lognormal(0, cfg.noise_dp, n),
    }
    for s in FAULTABLE_SENSORS:
        meas[s][idle] = np.nan                    # no water flowing -> no reading

    templates = list(scenario.fault_templates) + list(extra_faults or [])
    specs = sample_faults(rng, templates, n)
    for uf in user_faults or []:
        kind = uf["kind"]
        if kind not in KINDS:
            raise ValueError(f"unknown fault kind '{kind}'; use one of {KINDS}")
        sensor = "out_tds_ppm" if kind == "tds_inversion" else uf["sensor"]
        if sensor not in FAULTABLE_SENSORS:
            raise ValueError(f"cannot inject a fault into '{sensor}'; use one of {FAULTABLE_SENSORS}")
        specs.append(FaultSpec(kind, sensor, int(uf["start"]), max(1, int(uf.get("duration", 20))), float(uf.get("magnitude", 0.0))))
    ftype, fsensor = apply_faults(meas, specs, idle)

    miss_any = np.zeros(n, bool)
    for s in FAULTABLE_SENSORS:
        m = (rng.random(n) < cfg.missing_rate) & ~idle & np.isfinite(meas[s])
        meas[s][m] = np.nan
        miss_any |= m

    # ---------------- assemble ----------------
    hid = f"HH{int(index):04d}"
    df = pd.DataFrame({
        "household_id": hid, "scenario": scenario.name, "data_source": DATA_SOURCE_SYNTHETIC,
        "day": days,
        "date": pd.to_datetime(cfg.start_date) + pd.to_timedelta(days, unit="D"),
        **meas,
        "op_mode": np.where(idle, "idle", np.where(low, "low_inlet", "normal")),
    })
    for j, s in enumerate(STAGES):
        df[age_col(s)] = age[j]
        df[vol_col(s)] = cumv[j]
        df[repl_col(s)] = repl[j]
    for j, s in enumerate(STAGES):
        df[dtc_col(s)] = dtc[j]
    df["true_dp_ratio"], df["true_flux_rel"], df["true_rejection"], df["true_carbon_ratio"] = ratio, flux, rej, xc
    df["fault_type"], df["fault_sensor"] = ftype, fsensor
    df["fault_active"] = ftype != "none"
    df["missing_injected"] = miss_any
    df["source_change_flag"] = change_flag
    df["source_regime"] = regime

    meta = {
        "household_id": hid, "scenario": scenario.name, "law": law, "seed": int(seed),
        "base_usage_L": base_use, "feed_tds_base_ppm": feed_base,
        "wear_sediment": float(w[0]), "wear_carbon": float(w[1]), "wear_ro": float(w[2]),
        "particulate_base": partic_base, "organic_base": organic_base, "rejection_sensitivity": s_rej,
        "n_source_changes": len(change_days), "n_faults": len(specs),
        **{f"n_cycles_{s}": len(crossings[j]) for j, s in enumerate(STAGES)},
    }
    return df, meta
