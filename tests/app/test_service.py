import io
import numpy as np
import pandas as pd
import pytest
from aquapredict.app.service import AquaService, NotFound, BadRequest, Conflict, clean
from aquapredict.schema import STAGES
from .conftest import uploaded_rows


def make(svc, scenario="normal", seed=3, now=90, **kw):
    return svc.create_household("t", scenario, seed=seed, now_day=now, **kw)["id"]


# ---------------- households, readings, no peeking ----------------
def test_create_and_read_back_only_up_to_now(svc):
    hid = make(svc)
    h = svc.household(hid)
    assert h["source"] == "synthetic" and h["now_day"] == 90 and h["n_days"] == 1000
    hist = svc.history(hid)
    assert len(hist) == 91 and hist["day"].iloc[-1] == 90 and not [c for c in hist.columns if c.startswith("true_")]
    assert hist["replaced_sediment"].dtype == bool
    with pytest.raises(BadRequest):
        svc.readings(hid, upto_day=91)                       # the future is not readable
    assert any(c.startswith("true_") for c in svc.readings(hid, include_truth=True).columns)


def test_unknown_inputs_are_rejected_with_clear_errors(svc):
    with pytest.raises(BadRequest):
        svc.create_household("x", "no_such_scenario")
    with pytest.raises(BadRequest):
        svc.create_household("x", "normal", overrides={"not_a_param": 1})
    with pytest.raises(BadRequest):
        svc.create_household("x", "normal", law="Z")
    with pytest.raises(NotFound):
        svc.household("H999")
    with pytest.raises(BadRequest):
        svc.forecast(make(svc), "nope")


def test_user_overrides_beat_scenario_defaults(svc):
    a = make(svc, "high_usage")                                         # scenario sets 90 L/day
    b = make(svc, "high_usage", overrides={"usage_median_L": 15.0})     # user value must win
    assert svc.history(a)["usage_L"].mean() > 2 * svc.history(b)["usage_L"].mean()


def test_same_config_reproduces_identical_telemetry(svc):
    a, b = make(svc, seed=9, sim_index=4), make(svc, seed=9, sim_index=4)
    pd.testing.assert_frame_equal(svc.history(a).drop(columns="household_id"), svc.history(b).drop(columns="household_id"))


def test_delete_household_removes_everything(svc):
    hid = make(svc)
    svc.update_monitoring(hid, "trend")
    svc.delete_household(hid)
    assert svc.list_households() == [] and svc.alerts() == [] and svc.prediction_history(hid).empty


# ---------------- forecasting ----------------
def test_forecast_contract_and_three_separate_signals(svc):
    hid = make(svc)
    f = svc.forecast(hid, "sediment", "trend")
    for k in ("status", "point_days", "interval_days", "reason", "indicator", "sensor_health", "reliability", "recommendation",
              "model_version", "disclaimer", "data_source", "sensor_events"):
        assert k in f
    assert f["status"] == "point_only" and f["interval_days"] is None and "without a validated interval" in f["reliability"]
    assert f["indicator"]["available"] and "water safety" in f["disclaimer"]
    assert f["sensor_health"] in ("ok", "warning", "fault_suspected")


def test_new_household_with_short_history_is_withheld_not_guessed(svc):
    hid = svc.create_household("short", "insufficient_history", seed=1, now_day=9)["id"]
    for s in STAGES:
        f = svc.forecast(hid, s, "trend")
        assert f["status"] == "withheld" and f["point_days"] is None and "insufficient history" in f["reason"]
        assert "No forecast was issued" in f["recommendation"]


def test_forecast_storage_is_idempotent_and_never_beyond_now(svc):
    hid = make(svc)
    svc.forecast(hid, "ro", "trend"); svc.forecast(hid, "ro", "trend")
    assert len(svc.prediction_history(hid, "ro", "trend")) == 1
    svc.forecast(hid, "ro", "volume_rule")
    assert len(svc.prediction_history(hid, "ro")) == 2
    with pytest.raises(BadRequest):
        svc.forecast(hid, "ro", "trend", day=500)


def test_truth_is_demo_only(svc):
    hid = make(svc)
    t = svc.truth(hid, "sediment")
    assert t["available"] and isinstance(t["remaining_days_true"], float)
    rows, repl = uploaded_rows()
    up = svc.ingest_telemetry("real", rows, repl)["id"]
    assert svc.truth(up, "sediment")["available"] is False


def test_truth_handles_missing_simulator_label_without_crashing(svc, monkeypatch):
    hid = make(svc)
    original = svc.readings

    def readings_with_missing_truth(household_id, upto_day=None, include_truth=False, last=None):
        frame = original(household_id, upto_day, include_truth, last)
        if include_truth:
            frame = frame.copy()
            frame.loc[frame.index[-1], "true_dtc_sediment"] = None
        return frame

    monkeypatch.setattr(svc, "readings", readings_with_missing_truth)
    result = svc.truth(hid, "sediment")
    assert result == {
        "available": True,
        "censored": True,
        "note": "this filter was replaced before reaching its criterion",
    }


def test_untrained_model_is_a_clean_conflict(svc):
    hid = make(svc)
    with pytest.raises(Conflict):
        svc.forecast(hid, "ro", "person2")
    with pytest.raises(BadRequest):
        svc.forecast(hid, "ro", "does_not_exist")


# ---------------- alerts: de-duplication and lifecycle ----------------
def test_alerts_deduplicate_across_refreshes_and_resolve_when_condition_clears(svc):
    hid = make(svc, now=60)
    svc.advance(hid, 160, step=10, model="trend")
    first = svc.alerts(hid)
    assert first, "expected alerts after 160 days"
    for _ in range(3):
        svc.update_monitoring(hid, "trend")
    again = svc.alerts(hid)
    assert len(again) == len(first)                                  # refreshing never duplicates
    assert len({(a["stage"], a["dedupe_key"]) for a in again if a["status"] in ("open", "acknowledged")}) == \
        len([a for a in again if a["status"] in ("open", "acknowledged")])
    due = next(a for a in again if a["kind"] in ("maintenance_due", "rated_usage_reached", "indicator_past_criterion"))
    svc.record_replacement(hid, due["stage"])                        # fixes the condition
    after = {a["id"]: a for a in svc.alerts(hid)}
    assert after[due["id"]]["status"] == "resolved" and "condition cleared" in after[due["id"]]["note"]


def test_alert_actions_and_invalid_transitions(svc):
    hid = make(svc, now=60)
    svc.advance(hid, 160, step=10, model="trend")
    a = svc.alerts(hid, status="open")[0]
    ack = svc.alert_action(a["id"], "acknowledge", "seen")
    assert ack["status"] == "acknowledged" and "user acknowledge" in ack["note"] and ack["origin"] == "model"
    with pytest.raises(Conflict):
        svc.alert_action(a["id"], "acknowledge")                     # already acknowledged
    assert svc.alert_action(a["id"], "dismiss")["status"] == "dismissed"
    with pytest.raises(Conflict):
        svc.alert_action(a["id"], "resolve")                         # dismissed is closed
    with pytest.raises(BadRequest):
        svc.alert_action(a["id"], "explode")
    with pytest.raises(NotFound):
        svc.alert_action(99999, "resolve")
    n = len(svc.alerts(hid))
    svc.update_monitoring(hid, "trend")                              # a dismissed alert must not come straight back
    assert len(svc.alerts(hid)) == n


def test_sensor_fault_creates_alert_and_withholds_forecast(svc):
    hid = make(svc, now=70)
    svc.inject_fault(hid, {"kind": "stuck", "sensor": "dp_sediment_bar", "start": 70, "duration": 60})
    svc.advance(hid, 25, step=5, model="reference_v3" if False else "trend")
    ev = svc.sensor_events(hid, "sediment")
    assert any(e["kind"] == "stuck" and e["sensor"] == "dp_sediment_bar" for e in ev)
    kinds = {a["kind"] for a in svc.alerts(hid, stage="sediment")}
    assert "sensor_fault" in kinds
    assert svc.dashboard(hid, "trend")["stages"]["sediment"]["sensor_health"] == "fault_suspected"
    assert "sediment" in svc.dashboard(hid, "trend")["sensor_warnings"]


# ---------------- simulator controls never rewrite the past ----------------
def test_inject_fault_changes_only_the_future(svc):
    hid = make(svc, now=100)
    before = svc.history(hid)
    with pytest.raises(BadRequest):
        svc.inject_fault(hid, {"kind": "stuck", "sensor": "flow_Lpm", "start": 50, "duration": 10})
    with pytest.raises(BadRequest):
        svc.inject_fault(hid, {"kind": "stuck", "sensor": "usage_L", "start": 100, "duration": 10})
    svc.inject_fault(hid, {"kind": "drift", "sensor": "dp_sediment_bar", "start": 120, "duration": 10, "magnitude": 0.4})
    pd.testing.assert_frame_equal(before.iloc[:100].reset_index(drop=True), svc.history(hid).iloc[:100].reset_index(drop=True))
    svc.advance(hid, 40, step=20, model="trend")
    full = svc.readings(hid, include_truth=True)
    assert (full["fault_type"] == "drift").sum() > 0 and full.loc[full.day < 120, "fault_type"].eq("none").all()


def test_record_replacement_resets_filter_keeps_history_and_updates_plan(svc):
    hid = make(svc, now=60)
    svc.advance(hid, 100, step=10, model="trend")
    day = svc.household(hid)["now_day"]
    before = svc.history(hid)
    plan_before = [p for p in svc.plan(hid) if p["stage"] == "sediment" and p["status"] in ("predicted", "recommended")]
    assert plan_before
    svc.record_replacement(hid, "sediment", "new cartridge")
    after = svc.history(hid)
    pd.testing.assert_frame_equal(before.iloc[:day].reset_index(drop=True), after.iloc[:day].reset_index(drop=True))
    assert after["replaced_sediment"].iloc[-1] and after["age_sediment_d"].iloc[-1] == 0 and after["age_ro_d"].iloc[-1] > 0
    done = [p for p in svc.plan(hid) if p["stage"] == "sediment" and p["status"] == "completed"]
    assert done and done[0]["completed_day"] == day
    with pytest.raises(Conflict):
        svc.record_replacement(hid, "sediment")                      # same day twice
    f = svc.forecast(hid, "sediment", "trend")
    assert f["status"] == "withheld"                                 # brand-new filter: not enough history, not an alert
    assert not [a for a in svc.alerts(hid, stage="sediment", status="open") if a["kind"] == "forecast_withheld"]


def test_advance_bounds_and_stored_forecasts(svc):
    hid = make(svc, now=60)
    r = svc.advance(hid, 30, step=10, model="trend")
    assert r["now_day"] == 90 and r["steps"] == 3 and len(svc.prediction_history(hid, "ro", "trend")) == 3
    with pytest.raises(BadRequest):
        svc.advance(hid, 0)
    end = svc.create_household("end", "insufficient_history", seed=1, now_day=9)["id"]
    with pytest.raises(Conflict):
        svc.advance(end, 5)                                          # already at the end of a 10-day timeline


# ---------------- planner state machine ----------------
def test_plan_states_confirm_cancel_complete(svc):
    hid = make(svc, now=60)
    svc.advance(hid, 160, step=10, model="trend")
    rec = [p for p in svc.plan(hid) if p["status"] == "recommended"]
    assert rec, "a close window should be 'recommended'"
    p = rec[0]
    with pytest.raises(BadRequest):
        svc.confirm_plan(p["id"], planned_day=1)                     # past date
    c = svc.confirm_plan(p["id"], svc.household(hid)["now_day"] + 3)
    assert c["status"] == "confirmed" and c["planned_day"] == svc.household(hid)["now_day"] + 3
    with pytest.raises(Conflict):
        svc.confirm_plan(p["id"])                                    # only predicted/recommended
    svc.update_monitoring(hid, "trend")                              # a refresh must not overwrite the user's confirmed entry
    assert next(x for x in svc.plan(hid) if x["id"] == p["id"])["status"] == "confirmed"
    svc.complete_plan(p["id"], "done")
    assert next(x for x in svc.plan(hid) if x["id"] == p["id"])["status"] == "completed"
    with pytest.raises(Conflict):
        svc.cancel_plan(p["id"])
    with pytest.raises(NotFound):
        svc.cancel_plan(99999)


def test_cost_estimate_uses_editable_illustrative_prices(svc):
    from aquapredict.planner.cost import CostConfig
    hid = make(svc, now=60)
    svc.advance(hid, 160, step=10, model="trend")
    e = svc.cost_estimate(hid, 90)
    assert e["lines"] and e["total"] == e["parts"] + e["service_visit"] and "ILLUSTRATIVE" in e["note"]
    cheap = svc.cost_estimate(hid, 90, CostConfig(replacement_cost={"sediment": 1, "carbon": 1, "ro": 1}, service_visit_cost=0))
    assert cheap["total"] == len(cheap["lines"])


# ---------------- what-if ----------------
def test_what_if_is_hypothetical_unsaved_and_consistent(svc):
    hid = make(svc, now=120)
    same = svc.what_if(hid, {}, [], "trend", "sediment")
    assert "HYPOTHETICAL" in same["label"]
    assert same["base"]["point_days"] == pytest.approx(same["hypothetical"]["point_days"])    # no change => identical forecast
    hot = svc.what_if(hid, {"usage_median_L": 140.0}, [], "trend", "sediment")
    assert hot["hypothetical"]["point_days"] != same["base"]["point_days"]
    assert svc.household(hid)["overrides"] == {} and len(svc.readings(hid)) == 121          # nothing was saved
    rows, repl = uploaded_rows()
    up = svc.ingest_telemetry("real", rows, repl)["id"]
    with pytest.raises(Conflict):
        svc.what_if(up, {}, [], "trend", "sediment")


# ---------------- real-telemetry boundary ----------------
def test_ingest_uploaded_telemetry_end_to_end(svc):
    rows, repl = uploaded_rows(days=120)
    h = svc.ingest_telemetry("Kitchen purifier", rows, repl)
    assert h["source"] == "uploaded" and h["now_day"] == 119
    hist = svc.history(h["id"])
    assert (hist["age_sediment_d"].diff().dropna().isin([1.0]) | (hist["replaced_sediment"].iloc[1:].to_numpy())).all()
    f = svc.forecast(h["id"], "sediment", "trend")
    assert f["status"] in ("point_only", "withheld") and f["data_source"] == "uploaded"
    assert "UPLOADED" in svc.dashboard(h["id"], "trend")["data_source_label"]
    with pytest.raises(Conflict):
        svc.record_replacement(h["id"], "ro")
    with pytest.raises(Conflict):
        svc.advance(h["id"], 5)
    assert "data_source" in svc.export_csv("telemetry", h["id"]).splitlines()[0]


def test_ingest_rejects_bad_telemetry_with_reasons(svc):
    rows, repl = uploaded_rows()
    with pytest.raises(BadRequest, match="usage_L"):
        svc.ingest_telemetry("x", [{"day": 0}, {"day": 1}])
    gap = [r for r in rows if r["day"] != 10]
    with pytest.raises(BadRequest, match="contiguous"):
        svc.ingest_telemetry("x", gap, repl)
    hot = [dict(r) for r in rows]; hot[5]["temp_C"] = 900.0
    with pytest.raises(BadRequest, match="temp_C"):
        svc.ingest_telemetry("x", hot, repl)
    with pytest.raises(BadRequest):
        svc.ingest_telemetry("x", [])
    assert svc.list_households() == []                               # nothing half-saved


# ---------------- dashboard, data quality, exports, JSON safety ----------------
def test_dashboard_structure(svc):
    hid = make(svc)
    D = svc.dashboard(hid, "trend")
    assert D["n_filter_stages"] == 3 and set(D["stages"]) == set(STAGES) and "SYNTHETIC" in D["data_source_label"]
    for s, v in D["stages"].items():
        assert {"condition", "sensor_health", "reliability", "status"} <= set(v)       # three distinct concepts
    assert D["model"]["name"] == "trend" and D["disclaimer"]


def test_data_quality_handles_idle_days(svc):
    hid = make(svc, now=300)
    dq = svc.data_quality(hid)
    assert set(dq["days_since_valid_reading"]) >= {"flow_Lpm", "dp_sediment_bar"} and dq["last_day"] == 300


def test_exports_are_valid_csv_and_hide_truth_by_default(svc):
    hid = make(svc, now=60)
    svc.advance(hid, 40, step=10, model="trend")
    plain = pd.read_csv(io.StringIO(svc.export_csv("telemetry", hid)))
    withtruth = pd.read_csv(io.StringIO(svc.export_csv("telemetry", hid, include_truth=True)))
    assert plain["data_source"].eq("synthetic").all() and not [c for c in plain.columns if c.startswith("true_")]
    assert any(c.startswith("true_") for c in withtruth.columns) and len(plain) == 101
    for kind in ("predictions", "alerts", "maintenance", "experiments"):
        pd.read_csv(io.StringIO(svc.export_csv(kind, hid if kind != "experiments" else None)))
    with pytest.raises(BadRequest):
        svc.export_csv("passwords")
    with pytest.raises(BadRequest):
        svc.export_csv("telemetry")
    cfg = svc.export_household_config(hid)
    assert cfg["seed"] == 3 and "reproduces" in cfg["note"]


def test_clean_makes_everything_json_safe():
    out = clean({"a": float("nan"), "b": np.float64(2.5), "c": np.int64(3), "d": [float("inf"), np.bool_(True)],
                 "e": pd.DataFrame({"x": [1.0, np.nan]}).set_index(pd.Index(["r1", "r2"], name="k"))})
    import json
    json.dumps(out, allow_nan=False)
    assert out["a"] is None and out["b"] == 2.5 and out["d"] == [None, True] and out["e"][1]["x"] is None


def test_alert_hysteresis_prevents_flapping_but_known_fixes_clear_at_once(svc):
    hid = make(svc)
    cand = {"kind": "maintenance_due", "severity": "medium", "title": "t", "message": "m", "recommended_action": "a",
            "dedupe_key": "ro|maintenance_due", "details": {}}
    c = {"created": 0, "updated": 0, "resolved": 0}
    svc._sync_alerts(hid, "ro", [cand], 100, c)
    a = svc.alerts(hid)[0]
    assert a["status"] == "open" and c["created"] == 1
    svc._sync_alerts(hid, "ro", [], 101, c)                       # absent once: stays open
    assert svc.alerts(hid)[0]["status"] == "open"
    svc._sync_alerts(hid, "ro", [cand], 102, c)                   # returns: streak resets, still ONE alert
    assert len(svc.alerts(hid)) == 1
    svc._sync_alerts(hid, "ro", [], 103, c)
    svc._sync_alerts(hid, "ro", [], 104, c)                       # absent twice in a row: auto-resolved
    assert svc.alerts(hid)[0]["status"] == "resolved" and svc.alerts(hid)[0]["cleared"] == 1
    svc._sync_alerts(hid, "ro", [cand], 105, c)                   # genuinely new episode => new alert
    assert len(svc.alerts(hid)) == 2
    svc._sync_alerts(hid, "ro", [], 106, c, force_clear=True)     # a known fix clears immediately
    assert svc.alerts(hid, status="open") == []
