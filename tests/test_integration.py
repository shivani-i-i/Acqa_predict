import numpy as np
import pandas as pd
import pytest
from aquapredict.schema import VISIBLE_COLS, TRUTH_COLS, STAGES
from aquapredict.simulator.generate import generate_dataset
from aquapredict.eval.splits import split_households, subset
from aquapredict.eval.config import EvalConfig
from aquapredict.eval.harness import evaluate
from aquapredict.baselines.baselines import BASELINES
from aquapredict.eval.criteria import check_success
from aquapredict.integration import (to_p2_readings, history_to_p2, training_frame, train_person2_on_person1,
                                     Person2Forecaster, STAGE_TO_P2, PRESSURE_SOURCES)
from aquapredict.integration import train as train_mod
from aquapredict.integration.adapter import P2_COLUMNS

RESPONSE_KEYS = {"filter_stage", "household_id", "forecast_days", "interval_days", "drivers", "explanation",
                 "fault_events", "source_change", "status", "reason", "disclaimer"}


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    df, _ = generate_dataset(36, mix="normal", seed=41)
    sp = split_households(df.household_id.unique(), seed=41)
    tr, ca, te = (subset(df, sp[k]) for k in ("train", "calibration", "test"))
    f = train_person2_on_person1(tr, ca, tmp_path_factory.mktemp("b") / "bundle.joblib", seed=1, max_iter=40)
    return {"df": df, "tr": tr, "ca": ca, "te": te, "f": f}


def test_column_mapping_and_units():
    df, _ = generate_dataset(2, mix="normal", seed=3)
    g = df[(df.household_id == df.household_id.iloc[0]) & (df.op_mode != "idle")].iloc[:50]
    out = to_p2_readings(g)
    assert list(out.columns) == P2_COLUMNS
    np.testing.assert_allclose(out.usage_litres, g.usage_L)
    np.testing.assert_allclose(out.tds_in, g.feed_tds_ppm); np.testing.assert_allclose(out.tds_out, g.out_tds_ppm)
    np.testing.assert_allclose(out.flow_rate, g.flow_Lpm); np.testing.assert_allclose(out.temperature, g.temp_C)
    np.testing.assert_allclose(out.pressure, g.dp_sediment_bar)                          # default: raw sediment pressure drop
    np.testing.assert_allclose(to_p2_readings(g, "downstream").pressure, g.inlet_bar - g.dp_sediment_bar)
    np.testing.assert_allclose(to_p2_readings(g, "dp").pressure, g.dp_sediment_bar)
    np.testing.assert_allclose(to_p2_readings(g, "inlet").pressure, g.inlet_bar)
    assert out.purification_mode.isna().all() and out.timestamp.is_monotonic_increasing
    with pytest.raises(ValueError):
        to_p2_readings(g, "nonsense")
    assert set(PRESSURE_SOURCES) == {"downstream", "dp", "inlet"}


def test_downstream_pressure_falls_as_the_filter_clogs():
    df, _ = generate_dataset(12, mix="normal", seed=6)
    drops = []
    for _, g in df.groupby("household_id"):
        c = g[(g.age_sediment_d <= g.age_sediment_d.max()) & (g.op_mode != "idle")]
        first = c[c.age_sediment_d < 30]; late = c[(c.true_dtc_sediment >= 0) & (c.true_dtc_sediment < 10)]
        if len(first) > 10 and len(late) > 3:
            p = to_p2_readings(pd.concat([first, late]), "downstream").pressure
            drops.append(p.iloc[:len(first)].mean() - p.iloc[len(first):].mean())
    assert len(drops) > 5 and np.median(drops) > 0


def test_default_pressure_is_the_raw_drop_which_rises_as_the_filter_clogs():
    """The default ('dp') carries the wear signal directly: it RISES with clogging (the held-out evidence for choosing it
    over 'downstream' is in docs_results/RESULTS.md)."""
    df, _ = generate_dataset(12, mix="normal", seed=6)
    rises = []
    for _, g in df.groupby("household_id"):
        c = g[g.op_mode != "idle"]
        first = c[c.age_sediment_d < 30]; late = c[(c.true_dtc_sediment >= 0) & (c.true_dtc_sediment < 10)]
        if len(first) > 10 and len(late) > 3:
            p = to_p2_readings(pd.concat([first, late])).pressure
            rises.append(p.iloc[len(first):].mean() - p.iloc[:len(first)].mean())
    assert len(rises) > 5 and np.median(rises) > 0


def test_idle_days_are_omitted_but_timestamps_keep_the_gap():
    df, _ = generate_dataset(30, mix="normal", seed=11)
    g = df[df.household_id.isin(df[df.op_mode == "idle"].household_id.unique()[:1])].reset_index(drop=True)
    out = to_p2_readings(g)
    assert len(out) == int((g.op_mode != "idle").sum()) < len(g)
    assert out.timestamp.diff().dt.days.max() > 1                                  # the vacation gap is preserved


def test_history_is_cut_to_the_current_filter_cycle():
    df, _ = generate_dataset(10, mix="normal", seed=7)
    for _, g in df.groupby("household_id"):
        g = g.reset_index(drop=True)
        rp = np.flatnonzero(g.replaced_sediment.to_numpy())
        if len(rp) and rp[0] + 20 < len(g):
            t = rp[0] + 20
            h = history_to_p2(g[VISIBLE_COLS].iloc[:t + 1], "sediment")
            n_active = int((g.op_mode.iloc[rp[0]:t + 1] != "idle").sum())
            assert len(h) == n_active
            assert pd.Timestamp(h[0]["timestamp"]) >= pd.Timestamp(g.date.iloc[rp[0]])
            assert pd.Timestamp(h[-1]["timestamp"]) <= pd.Timestamp(g.date.iloc[t])
            return
    pytest.fail("no replaced cycle found")


def test_training_frame_structure_and_no_cross_cycle_mixing(world):
    tf = training_frame(world["tr"])
    assert set(tf.filter_stage) == set(STAGE_TO_P2.values())
    assert (tf.remaining_days >= 0).all() and (tf.groupby(["household_id", "filter_stage"]).remaining_days.last() == 0).all()
    assert tf.groupby(["household_id", "filter_stage"]).timestamp.apply(lambda s: s.is_monotonic_increasing).all()
    assert set(tf.columns) == set(["household_id", "filter_stage"] + P2_COLUMNS + ["remaining_days"])
    real = tf.household_id.str.split("~").str[0]
    assert set(real) <= set(world["tr"].household_id)                              # only training households appear
    assert not set(real) & set(world["te"].household_id)


def test_overlapping_households_between_train_and_calibration_are_refused(world, tmp_path):
    with pytest.raises(ValueError, match="overlap"):
        train_person2_on_person1(world["tr"], world["tr"].iloc[:2000], tmp_path / "x.joblib", max_iter=5)


def test_person2_forecaster_returns_harness_contract_and_full_response(world):
    f, te = world["f"], world["te"]
    g = te[te.household_id == te.household_id.iloc[0]].reset_index(drop=True)
    hist = g[VISIBLE_COLS].iloc[:90]
    out = f(hist, "sediment")
    assert {"point", "lo", "hi", "withheld", "reason", "status"} <= set(out)
    assert out["status"] in {"ok", "degraded", "withheld"} and isinstance(out["withheld"], bool)
    if not out["withheld"]:
        assert np.isfinite(out["point"]) and out["point"] >= 0
        if out["lo"] is not None:
            assert out["lo"] <= out["hi"]
    r = f.respond(hist, "sediment")
    assert set(r) == RESPONSE_KEYS and r["filter_stage"] == "sediment"
    assert r["drivers"] == [] and r["explanation"]["available"] is False           # explain=False by default


def test_ro_stage_name_is_translated(world):
    f, te = world["f"], world["te"]
    g = te[te.household_id == te.household_id.iloc[0]].reset_index(drop=True)
    assert f.respond(g[VISIBLE_COLS].iloc[:90], "ro")["filter_stage"] == "ro_membrane"


def test_short_history_is_withheld_with_a_reason(world):
    """Person 2's documented gate: at least GatingConfig.min_history_rows (=3) readings. Below it: withheld + reason."""
    f, te = world["f"], world["te"]
    g = te[te.household_id == te.household_id.iloc[0]].reset_index(drop=True)
    out = f(g[VISIBLE_COLS].iloc[:2], "sediment")
    assert out["withheld"] and out["status"] == "withheld" and "insufficient history" in out["reason"] and np.isnan(out["point"])


def test_person2_receives_only_past_visible_readings(world, monkeypatch):
    f, te = world["f"], world["te"]
    seen = []
    real = train_mod.p2_predict
    monkeypatch.setattr(train_mod, "p2_predict", lambda req: (seen.append(req), real(req))[1])
    g = te[te.household_id == te.household_id.iloc[0]].reset_index(drop=True)
    f(g[VISIBLE_COLS].iloc[:70], "carbon")
    req = seen[0]
    keys = set().union(*[set(r) for r in req["history"]])
    assert keys <= set(P2_COLUMNS) and not keys & set(TRUTH_COLS)
    assert pd.Timestamp(req["history"][-1]["timestamp"]) <= pd.Timestamp(g.date.iloc[69])
    assert req["explain"] is False


def test_end_to_end_through_the_person1_harness_and_criteria(world):
    cfg = EvalConfig(stride=21)
    te = world["te"]
    sub = te[te.household_id.isin(te.household_id.unique()[:3])]
    res = evaluate(sub, {**BASELINES, "Person2": world["f"]}, cfg=cfg)
    p2 = res.opps[res.opps.method == "Person2"]
    assert len(p2) > 20 and {"status", "reason"} <= set(res.opps.columns)
    assert set(p2.status) <= {"ok", "degraded", "withheld"}
    s = check_success(res.opps, cfg, model="Person2")
    assert {"1_window_error", "2_interval_coverage", "3_late_warnings"} <= set(s)
