import math
import numpy as np
import pandas as pd
import pytest
from aquapredict.schema import VISIBLE_COLS
from aquapredict.simulator.generate import generate_dataset
from aquapredict.checks.sensor_rules import check_readings, blocking_reason
from aquapredict.reference_model.features import extract, NAMES
from aquapredict.reference_model.model import ReferenceForecaster
from aquapredict.eval.splits import split_households, subset


@pytest.fixture(scope="module")
def small():
    df, _ = generate_dataset(36, mix="normal", seed=31)
    sp = split_households(df.household_id.unique(), seed=1)
    tr, ca, te = (subset(df, sp[k]) for k in ("train", "calibration", "test"))
    model = ReferenceForecaster(max_iter=40, stride=8).fit(tr, ca)
    return model, te


def _hist(df, hid, upto):
    g = df[df.household_id == hid].sort_values("day").reset_index(drop=True)
    return g[VISIBLE_COLS].iloc[:upto].copy()


def test_rules_no_false_flags_on_clean_normal_data():
    df, _ = generate_dataset(10, mix="normal", seed=5)
    bad = 0
    for hid in df.household_id.unique():
        g = df[df.household_id == hid].reset_index(drop=True)
        if g.fault_active.any():
            continue
        for t in range(40, len(g), 25):
            ev = [e for e in check_readings(g[VISIBLE_COLS].iloc[:t]) if e["kind"] in ("stuck", "tds_inversion", "stale")]
            bad += len(ev)
    assert bad == 0


def test_rules_detect_stuck_inversion_and_stale():
    df, _ = generate_dataset(1, mix="normal", seed=2, n_days=80)
    h = df[VISIBLE_COLS].copy()
    h = h[h.op_mode != "idle"].reset_index(drop=True)
    stuck = h.copy(); stuck.loc[stuck.index[-9:], "dp_sediment_bar"] = 0.2
    assert {e["kind"] for e in check_readings(stuck, "sediment")} >= {"stuck"}
    assert "stuck:dp_sediment_bar" in blocking_reason(stuck, "sediment")
    assert blocking_reason(stuck, "carbon") == ""            # carbon forecast does not use that sensor
    inv = h.copy(); inv.loc[inv.index[-4:], "out_tds_ppm"] = inv.loc[inv.index[-4:], "feed_tds_ppm"] * 1.3
    assert any(e["kind"] == "tds_inversion" for e in check_readings(inv, "ro"))
    stale = h.copy(); stale.loc[stale.index[-12:], "flow_Lpm"] = np.nan
    assert any(e["kind"] == "stale" and e["sensor"] == "flow_Lpm" for e in check_readings(stale, "ro"))
    assert all(e["status"] == "suspected" for e in check_readings(stuck))


def test_features_use_only_past_rows(small):
    model, te = small
    hid = te.household_id.iloc[0]
    full = te[te.household_id == hid].sort_values("day")[VISIBLE_COLS].reset_index(drop=True)
    t = 70
    a = extract(full.iloc[: t + 1], "sediment")
    future_tampered = full.copy()
    future_tampered.loc[t + 1:, ["dp_sediment_bar", "flow_Lpm", "usage_L"]] = 999.0
    b = extract(future_tampered.iloc[: t + 1], "sediment")
    assert all((np.isnan(a[k]) and np.isnan(b[k])) or a[k] == b[k] for k in NAMES)


def test_prediction_structure_and_interval_ordering(small):
    model, te = small
    hid = te.household_id.iloc[0]
    h = _hist(te, hid, 60)
    for st in ("sediment", "carbon", "ro"):
        r = model.predict(h, st)
        if not r["withheld"]:
            if r["lo"] is None:                                   # interval intentionally hidden for this stage
                assert r["hi"] is None and "not shown" in r["interval_status"] and r["point"] >= 0
            else:
                assert 0 <= r["lo"] <= r["point"] <= r["hi"]
                assert "not a guarantee" in r["interval_status"]


def test_withholds_with_reason_when_history_short_or_sensor_stuck(small):
    model, te = small
    h = _hist(te, te.household_id.iloc[0], 60)
    short = _hist(te, te.household_id.iloc[0], 9)             # only 9 days of history
    r = model.predict(short, "sediment")
    assert r["withheld"] and "insufficient history" in r["reason"] and np.isnan(r["point"])
    stuck = h[h.op_mode != "idle"].copy()
    stuck.loc[stuck.index[-9:], "flow_Lpm"] = 0.2
    r = model.predict(stuck, "sediment")
    assert r["withheld"] and "sensor" in r["reason"]
    assert not model.predict(stuck, "sediment", gate=False)["withheld"]       # gate can be switched off


def test_save_load_roundtrip(small, tmp_path):
    model, te = small
    p = tmp_path / "m.pkl"
    model.save(str(p))
    m2 = ReferenceForecaster.load(str(p))
    h = _hist(te, te.household_id.iloc[0], 60)
    a, b = model.predict(h, "ro"), m2.predict(h, "ro")
    assert a["withheld"] == b["withheld"] and (a["withheld"] or math.isclose(a["point"], b["point"]))


def test_conformal_quantile_formula():
    m = ReferenceForecaster(alpha=0.1)
    r = np.arange(1, 101, dtype=float)             # n=100 -> k = ceil(101*0.9)=91 -> 91st smallest
    assert m._quantile(r) == 91.0


def test_unfitted_model_raises():
    with pytest.raises(RuntimeError):
        ReferenceForecaster().predict(pd.DataFrame(), "ro")


def test_degenerate_history_does_not_crash(small):
    model, te = small
    h = _hist(te, te.household_id.iloc[0], 60)
    h["age_sediment_d"] = 30.0                                  # constant age: line fit is degenerate
    r = model.predict(h, "sediment", gate=False)
    assert isinstance(r["withheld"], bool)


# ---------------- v3 behaviour ----------------
def test_nonapplicable_columns_dropped_per_stage(small):
    model, _ = small
    assert "sed_slope30" in model.cols["sediment"] and "sed_slope30" not in model.cols["ro"]
    assert "flux_level30" in model.cols["ro"] and "flux_level30" not in model.cols["carbon"]
    for st in model.cols:
        assert len(model.cols[st]) == len(set(model.cols[st]))


def test_feature_change_detector_counts_abrupt_steps():
    from aquapredict.reference_model.features import _change_events, _feed_change_and_cv
    flat = np.full(120, 300.0)
    assert _change_events(flat) == 0
    stepped = np.r_[np.full(40, 300.0), np.full(40, 600.0), np.full(40, 300.0)]
    assert _change_events(stepped) == 2
    days, cv = _feed_change_and_cv(stepped)
    assert 20 <= days <= 45 and cv >= 0
    assert _change_events(np.full(5, np.nan)) == 0


def test_model_selection_picks_baseline_when_gbm_is_worse(small):
    model, _ = small
    for st, d in model.diag.items():
        better_gbm = d["oof_median_err_gbm"] <= d["oof_median_err_baseline"]
        assert (d["mode"] == "gbm") == better_gbm            # select=True: use whichever wins out-of-fold


def test_interval_hidden_when_calibrated_width_too_large(small):
    model, te = small
    tight = ReferenceForecaster(max_iter=20, stride=8, max_useful_width_days=0.5)
    ds = {"train": {s: __import__("aquapredict.reference_model.model", fromlist=["x"]).build_dataset(subset(
        generate_dataset(30, mix="normal", seed=32)[0], split_households(generate_dataset(30, mix="normal", seed=32)[0].household_id.unique(), seed=1)["train"]), s, 8) for s in ("sediment", "carbon", "ro")}}
    ds["cal"] = ds["train"]
    tight.fit(datasets=ds)
    h = _hist(te, te.household_id.iloc[0], 60)
    r = tight.predict(h, "sediment")
    assert not r["withheld"] and r["lo"] is None and r["hi"] is None and np.isfinite(r["point"])
    assert "not shown" in r["interval_status"]


def test_adaptive_and_nonadaptive_both_produce_ordered_intervals():
    df, _ = generate_dataset(30, mix="normal", seed=33)
    sp = split_households(df.household_id.unique(), seed=2)
    tr, ca, te = (subset(df, sp[k]) for k in ("train", "calibration", "test"))
    for adaptive in (True, False):
        m = ReferenceForecaster(max_iter=20, stride=8, adaptive=adaptive, max_useful_width_days=1e9).fit(tr, ca)
        h = _hist(te, te.household_id.iloc[0], 70)
        r = m.predict(h, "sediment")
        assert 0 <= r["lo"] <= r["point"] <= r["hi"]
