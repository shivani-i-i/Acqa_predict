"""Drives the real Streamlit app headlessly (streamlit.testing.AppTest): clicks the actual buttons, checks the database."""
import os
import shutil
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from aquapredict.app.service import AquaService
from .conftest import uploaded_rows

APP = os.path.join(os.path.dirname(__file__), "..", "..", "aquapredict", "app", "ui", "streamlit_app.py")
PAGES = ["Dashboard", "Filter detail", "Simulator & what-if", "Alerts", "Planner & costs", "Validation lab", "Models", "Data & export"]


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("AQUAPREDICT_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("AQUAPREDICT_DB", str(tmp_path / "home" / "app.db"))
    (tmp_path / "home").mkdir()
    st.cache_resource.clear()
    svc = AquaService(str(tmp_path / "home" / "app.db"), str(tmp_path / "home" / "artifacts"))
    yield svc, tmp_path
    st.cache_resource.clear()


def run():
    return AppTest.from_file(APP, default_timeout=180).run()


def text(at):
    parts = []
    for kind in ("markdown", "caption", "warning", "info", "success", "error", "header", "subheader", "title", "metric"):
        for e in getattr(at, kind):
            parts.append(str(getattr(e, "value", "")) + " " + str(getattr(e, "label", "")))
    return "\n".join(parts)


def button(at, key):
    for b in list(at.button) + list(at.sidebar.button):
        if b.key == key:
            return b
    raise AssertionError(f"no button {key}; have {[b.key for b in list(at.button) + list(at.sidebar.button)]}")


def goto(at, page):
    at.sidebar.radio(key="page").set_value(page)
    return at.run()


def demo(svc, now=60, advance=160, scenario="normal", seed=3):
    hid = svc.create_household("ui demo", scenario, seed=seed, now_day=now)["id"]
    if advance:
        svc.advance(hid, advance, 10, "trend")
    return hid


def no_errors(at):
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]


def test_empty_state_then_create_household_through_the_ui(env):
    svc, _ = env
    at = run()
    no_errors(at)
    assert svc.list_households() == [] and "No households yet" in text(at)
    at = button(at, "create_btn").click().run()
    no_errors(at)
    hs = svc.list_households()
    assert len(hs) == 1 and hs[0]["scenario"] == "normal" and hs[0]["source"] == "synthetic"
    assert "Dashboard" in text(at) and "SYNTHETIC" in text(at)
    assert svc.alerts(hs[0]["id"]) is not None and len(svc.prediction_history(hs[0]["id"])) >= 3     # monitoring ran on creation


def test_bad_overrides_show_a_clear_error_not_a_crash(env):
    svc, _ = env
    at = run()
    at.text_area(key="new_overrides").set_value("{not json").run()
    at = button(at, "create_btn").click().run()
    assert any("not valid JSON" in e.value for e in at.error) and svc.list_households() == []
    at.text_area(key="new_overrides").set_value('{"bogus_param": 1}').run()
    at = button(at, "create_btn").click().run()
    assert any("unknown simulator parameter" in e.value for e in at.error) and svc.list_households() == []


@pytest.mark.parametrize("page", PAGES)
def test_every_page_renders_without_errors(env, page):
    svc, _ = env
    demo(svc)
    at = goto(run(), page)
    no_errors(at)
    assert at.header or at.subheader


def test_dashboard_shows_three_separate_signals_and_labels(env):
    svc, _ = env
    demo(svc)
    t = text(run())
    for needle in ("Condition:", "Sensor health:", "Forecast reliability:", "Sediment filter", "Carbon filter", "RO membrane",
                   "SYNTHETIC", "not real-time telemetry", "water safety"):
        assert needle in t, needle


def test_withheld_forecast_shows_the_reason(env):
    svc, _ = env
    svc.create_household("short", "insufficient_history", seed=1, now_day=9)
    t = text(run())
    assert "Forecast withheld" in t and "Why withheld" in t and "insufficient history" in t


def test_advance_button_moves_the_clock_and_updates_monitoring(env):
    svc, _ = env
    hid = demo(svc, advance=0)
    at = run()
    at.sidebar.number_input(key="adv_days").set_value(30).run()
    at = button(at, "adv_btn").click().run()
    no_errors(at)
    assert svc.household(hid)["now_day"] == 90 and "Advanced to day 90" in text(at)
    assert len(svc.prediction_history(hid, "ro")) >= 4


def test_record_replacement_button_resets_filter(env):
    svc, _ = env
    hid = demo(svc, advance=100)
    at = goto(run(), "Simulator & what-if")
    at = button(at, "rep_sediment").click().run()
    no_errors(at)
    last = svc.history(hid).iloc[-1]
    assert last["replaced_sediment"] and last["age_sediment_d"] == 0 and "replacement recorded" in text(at)
    assert any(p["stage"] == "sediment" and p["status"] == "completed" for p in svc.plan(hid))
    at = button(at, "rep_sediment").click().run()                   # the same day again must be refused, with the reason shown
    assert any("already recorded" in e.value for e in at.error)


def test_inject_fault_button_and_rejection_of_past_start(env):
    svc, _ = env
    hid = demo(svc, advance=0, now=80)
    at = goto(run(), "Simulator & what-if")
    at.number_input(key="inj_start").set_value(10)
    at = button(at.run(), "inj_btn").click().run()
    assert any("can only start" in e.value for e in at.error) and svc.household(hid)["user_faults"] == []
    at.number_input(key="inj_start").set_value(80)
    at.selectbox(key="inj_kind").set_value("stuck")
    at.selectbox(key="inj_sensor").set_value("flow_Lpm")
    at = button(at.run(), "inj_btn").click().run()
    no_errors(at)
    f = svc.household(hid)["user_faults"]
    assert len(f) == 1 and f[0]["kind"] == "stuck" and f[0]["sensor"] == "flow_Lpm" and f[0]["start"] == 80
    assert "Fault injected" in text(at)


def test_what_if_runs_and_is_labelled_hypothetical(env):
    svc, _ = env
    hid = demo(svc, advance=40)
    at = goto(run(), "Simulator & what-if")
    at = button(at, "w_btn").click().run()
    no_errors(at)
    assert "HYPOTHETICAL" in " ".join(w.value for w in at.warning)
    assert svc.household(hid)["overrides"] == {}                    # nothing saved
    assert at.dataframe                                              # comparison table is shown


def test_alert_buttons_change_status_in_the_database(env):
    svc, _ = env
    hid = demo(svc)
    open_ids = [a["id"] for a in svc.alerts(hid, status="open")]
    assert open_ids
    at = goto(run(), "Alerts")
    no_errors(at)
    key = f"acknowledge_{open_ids[0]}"
    at = button(at, key).click().run()
    no_errors(at)
    assert svc.alerts(hid)[[a["id"] for a in svc.alerts(hid)].index(open_ids[0])]["status"] == "acknowledged"
    at = button(at, f"dismiss_{open_ids[0]}").click().run()
    assert next(a for a in svc.alerts(hid) if a["id"] == open_ids[0])["status"] == "dismissed"


def test_planner_confirm_then_complete(env):
    svc, _ = env
    hid = demo(svc)
    plan = [p for p in svc.plan(hid) if p["status"] in ("predicted", "recommended")]
    assert plan
    pid = plan[0]["id"]
    at = goto(run(), "Planner & costs")
    no_errors(at)
    at.number_input(key=f"pd_{pid}").set_value(svc.household(hid)["now_day"] + 5)
    at = button(at.run(), f"cf_{pid}").click().run()
    no_errors(at)
    p = next(x for x in svc.plan(hid) if x["id"] == pid)
    assert p["status"] == "confirmed" and p["planned_day"] == svc.household(hid)["now_day"] + 5
    assert "ILLUSTRATIVE" in " ".join(w.value for w in at.warning)
    at = button(at, f"cp_{pid}").click().run()
    assert next(x for x in svc.plan(hid) if x["id"] == pid)["status"] == "completed"


def test_validation_lab_runs_from_the_ui_and_persists(env):
    svc, _ = env
    demo(svc, advance=0)
    at = goto(run(), "Validation lab")
    at.number_input(key="v_n").set_value(4)
    at = button(at.run(), "v_run").click().run()
    no_errors(at)
    assert "forecasts on 4 fresh households" in text(at) and len(svc.experiments()) == 1
    assert len(at.dataframe) >= 3 and "synthetic" in text(at).lower()


def test_filter_detail_each_stage_and_backfill(env):
    svc, _ = env
    hid = demo(svc, advance=100)
    at = goto(run(), "Filter detail")
    for s in ("sediment", "carbon", "ro"):
        at.radio(key="stage").set_value(s)
        at = at.run()
        no_errors(at)
        assert "Recommended next action" in text(at)
    before = len(svc.prediction_history(hid, "ro", "trend"))
    at = button(at, "backfill").click().run()
    no_errors(at)
    assert len(svc.prediction_history(hid, "ro", "trend")) > before and "past forecasts" in text(at)


def test_uploaded_household_is_labelled_and_has_no_simulator_clock(env):
    svc, _ = env
    rows, repl = uploaded_rows(days=100)
    svc.ingest_telemetry("real", rows, repl)
    at = run()
    no_errors(at)
    assert "UPLOADED" in " ".join(i.value for i in at.info)
    assert "adv_btn" not in [b.key for b in at.sidebar.button]
    assert "SYNTHETIC DEMO" not in " ".join(w.value for w in at.warning)


def test_trained_models_appear_and_switching_models_works(env, trained_dir):
    svc, tmp = env
    shutil.copytree(trained_dir, tmp / "home" / "artifacts", dirs_exist_ok=True)
    st.cache_resource.clear()
    demo(svc, advance=0, now=150)
    at = run()
    no_errors(at)
    sb = at.sidebar.selectbox(key="model")
    labels = list(sb.options)
    assert any("Reference model v3" in o for o in labels) and any("Person 2" in o for o in labels)
    assert sb.value == "reference_v3"                              # best-evidence model is the default once trained
    for needle in ("Person 2", "Reference model v3", "Trend"):
        at.sidebar.selectbox(key="model").select_index(next(i for i, o in enumerate(labels) if needle in o))
        at = goto(at.run(), "Filter detail")
        no_errors(at)
        assert "Recommended next action" in text(at)
    at = goto(at, "Models")
    no_errors(at)
    assert "Train" in " ".join(b.label for b in at.button)


def test_export_page_offers_every_download(env):
    svc, _ = env
    demo(svc)
    at = goto(run(), "Data & export")
    no_errors(at)
    assert len(at.get("download_button")) == 6 and "data_source" in text(at)
