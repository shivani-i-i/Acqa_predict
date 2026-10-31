import io
import warnings
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from aquapredict.app.api import create_app
from aquapredict.app.service import AquaService, BadRequest, Conflict
from .conftest import uploaded_rows

warnings.filterwarnings("ignore")


# ---------------- learned models inside the app ----------------
def test_registry_reports_trained_models_and_defaults_to_best_evidence(trained_dir, tmp_path):
    s = AquaService(str(tmp_path / "a.db"), trained_dir)
    avail = {m["name"]: m for m in s.models()}
    assert avail["reference_v3"]["available"] and avail["person2"]["available"]
    assert avail["reference_v3"]["trained_with"]["train_seed"] == 77 and avail["reference_v3"]["card"]["limits"]
    assert s.registry.default_model() == "reference_v3"


def test_reference_model_gives_ordered_interval_and_status(trained_dir, tmp_path):
    s = AquaService(str(tmp_path / "a.db"), trained_dir)
    hid = s.create_household("m", "normal", seed=5, now_day=150)["id"]
    seen = set()
    for st in ("sediment", "carbon", "ro"):
        f = s.forecast(hid, st, "reference_v3")
        seen.add(f["status"])
        if f["interval_days"]:
            assert f["status"] == "interval" and 0 <= f["interval_days"][0] <= f["point_days"] <= f["interval_days"][1]
            assert "not a guarantee" in f["interval_status"]
    assert seen <= {"interval", "point_only", "withheld"}


def test_person2_model_returns_drivers_faults_and_status(trained_dir, tmp_path):
    s = AquaService(str(tmp_path / "a.db"), trained_dir)
    hid = s.create_household("p2", "normal", seed=6, now_day=150)["id"]
    f = s.forecast(hid, "sediment", "person2", explain=True)
    assert f["model"] == "person2" and f["status"] in ("interval", "point_only", "withheld")
    assert isinstance(f["drivers"], list) and isinstance(f["fault_events"], list) and "source_change" in f
    assert f["reliability"] and f["disclaimer"]


def test_person2_withholds_when_sensor_is_stuck(trained_dir, tmp_path):
    s = AquaService(str(tmp_path / "a.db"), trained_dir)
    hid = s.create_household("p2", "normal", seed=6, now_day=100)["id"]
    s.inject_fault(hid, {"kind": "stuck", "sensor": "flow_Lpm", "start": 100, "duration": 60})
    s.advance(hid, 20, step=20, model="trend")
    assert s.forecast(hid, "ro", "person2")["status"] in ("withheld", "point_only", "interval")   # runs end-to-end
    assert s.forecast(hid, "ro", "reference_v3")["status"] == "withheld"                          # gate: suspected sensor problem


def test_validation_lab_refuses_the_training_seed_and_scores_fresh_households(trained_dir, tmp_path):
    s = AquaService(str(tmp_path / "a.db"), trained_dir)
    with pytest.raises(BadRequest, match="unseen"):
        s.run_validation(4, seed=77, methods=["reference_v3"])               # same households the model was trained on
    r = s.run_validation(4, seed=5151, methods=["reference_v3"])
    assert set(r["criteria"]) == {"reference_v3"} and "all_passed" in r["criteria"]["reference_v3"]
    names = {w["method"] for w in r["window"]}
    assert names == {"trend", "volume_rule", "health_20pct", "reference_v3"}
    assert r["caveat"] and r["n_filter_lives"] > 5 and s.experiment(r["id"])["params"]["seed"] == 5151
    assert [e["id"] for e in s.experiments()] == [r["id"]]


def test_validation_lab_input_checks(svc):
    with pytest.raises(BadRequest):
        svc.run_validation(2)
    with pytest.raises(BadRequest):
        svc.run_validation(4, methods=["magic"])
    with pytest.raises(Conflict):
        svc.run_validation(4, methods=["person2"])                          # not trained


# ---------------- REST API ----------------
@pytest.fixture
def client(svc):
    return TestClient(create_app(svc))


def test_api_end_to_end_flow(client):
    assert client.get("/health").json()["status"] == "ok"
    assert len(client.get("/scenarios").json()) == 11
    r = client.post("/households", json={"name": "api", "scenario": "normal", "seed": 3, "now_day": 60})
    assert r.status_code == 201
    hid = r.json()["id"]
    assert client.get("/households").json()[0]["id"] == hid
    adv = client.post(f"/households/{hid}/advance", json={"days": 160, "step": 10, "model": "trend"})
    assert adv.status_code == 200 and adv.json()["now_day"] == 220
    dash = client.get(f"/households/{hid}/dashboard", params={"model": "trend"}).json()
    assert set(dash["stages"]) == {"sediment", "carbon", "ro"} and "SYNTHETIC" in dash["data_source_label"]
    f = client.get(f"/households/{hid}/forecast", params={"model": "trend"}).json()
    assert set(f) == {"sediment", "carbon", "ro"} and all("recommendation" in v for v in f.values())
    assert len(client.get(f"/households/{hid}/predictions").json()) >= 3
    plan = client.get(f"/households/{hid}/plan").json()
    assert plan and client.get(f"/households/{hid}/cost-estimate").json()["total"] >= 0
    alerts = client.get("/alerts", params={"household_id": hid, "status": "open"}).json()
    assert alerts
    ack = client.post(f"/alerts/{alerts[0]['id']}/acknowledge", json={"note": "seen"})
    assert ack.status_code == 200 and ack.json()["status"] == "acknowledged"
    assert client.post(f"/alerts/{alerts[0]['id']}/acknowledge", json={}).status_code == 409
    p = next(x for x in plan if x["status"] in ("predicted", "recommended"))
    assert client.post(f"/plan/{p['id']}/confirm", json={}).json()["status"] == "confirmed"
    rep = client.post(f"/households/{hid}/replace", json={"stage": p["stage"], "note": "done"})
    assert rep.status_code == 200 and client.post(f"/households/{hid}/replace", json={"stage": p["stage"]}).status_code == 409
    wi = client.post(f"/households/{hid}/what-if", json={"stage": "ro", "overrides": {"usage_median_L": 90}, "model": "trend"})
    assert wi.status_code == 200 and "HYPOTHETICAL" in wi.json()["label"]
    assert client.delete(f"/households/{hid}").status_code == 204 and client.get(f"/households/{hid}").status_code == 404


def test_api_validation_errors_and_status_codes(client):
    hid = client.post("/households", json={"name": "x", "seed": 2}).json()["id"]
    assert client.get("/households/H404").status_code == 404
    assert client.post("/households", json={"scenario": "bogus"}).status_code == 422
    assert client.post("/households", json={"law": "Q"}).status_code == 422                       # pydantic
    assert client.post("/households", json={"seed": -1}).status_code == 422
    assert client.get(f"/households/{hid}/forecast", params={"stage": "nope"}).status_code == 422
    assert client.post(f"/households/{hid}/advance", json={"days": 0}).status_code == 422
    assert client.post(f"/households/{hid}/faults", json={"kind": "stuck", "sensor": "usage_L", "start": 70}).status_code == 422
    assert client.post(f"/households/{hid}/faults", json={"kind": "stuck", "sensor": "flow_Lpm", "start": 5}).status_code == 422
    assert client.post(f"/households/{hid}/faults", json={"kind": "stuck", "sensor": "flow_Lpm", "start": 70, "duration": 10}).status_code == 200
    assert client.get(f"/households/{hid}/forecast", params={"stage": "ro", "model": "person2"}).status_code == 409
    assert client.post("/alerts/9999/resolve", json={}).status_code == 404
    assert client.post("/validation/run", json={"n_households": 1}).status_code == 422
    assert client.get("/export/secrets").status_code == 422


def test_api_responses_are_strict_json_no_nan(client):
    hid = client.post("/households", json={"name": "x", "seed": 3, "now_day": 100}).json()["id"]
    r = client.get(f"/households/{hid}/readings", params={"last": 50, "include_truth": True})
    import json
    json.loads(r.text, parse_constant=lambda c: (_ for _ in ()).throw(AssertionError(f"non-JSON constant {c}")))


def test_api_export_csv_and_downloads(client):
    hid = client.post("/households", json={"name": "x", "seed": 3}).json()["id"]
    client.post(f"/households/{hid}/advance", json={"days": 30, "step": 10, "model": "trend"})
    r = client.get("/export/telemetry", params={"household_id": hid})
    assert r.headers["content-type"].startswith("text/csv") and "attachment" in r.headers["content-disposition"]
    df = pd.read_csv(io.StringIO(r.text))
    assert len(df) == 91 and df["data_source"].eq("synthetic").all()
    assert pd.read_csv(io.StringIO(client.get("/export/predictions", params={"household_id": hid}).text)).shape[0] >= 9


def test_api_ingest_real_telemetry_and_forecast(client):
    rows, repl = uploaded_rows(days=100)
    r = client.post("/telemetry/ingest", json={"name": "real", "rows": rows, "replacements": repl})
    assert r.status_code == 201 and r.json()["source"] == "uploaded"
    hid = r.json()["id"]
    f = client.get(f"/households/{hid}/forecast", params={"stage": "sediment", "model": "trend"})
    assert f.status_code == 200 and f.json()["data_source"] == "uploaded"
    assert client.get(f"/households/{hid}/truth", params={"stage": "sediment"}).json()["available"] is False
    assert client.post(f"/households/{hid}/advance", json={"days": 5}).status_code == 409
    assert client.post("/telemetry/ingest", json={"name": "x", "rows": rows[:5]}).status_code == 422           # too few rows
    bad = [dict(x) for x in rows]; bad[3]["temp_C"] = 999.0
    r = client.post("/telemetry/ingest", json={"name": "x", "rows": bad, "replacements": repl})
    assert r.status_code == 422 and "temp_C" in r.json()["detail"]
    assert len(client.get("/households").json()) == 1                                                        # rejected upload not stored


def test_api_validation_run_and_experiments(client):
    r = client.post("/validation/run", json={"n_households": 4, "seed": 999})
    assert r.status_code == 200
    body = r.json()
    assert body["n_forecasts"] > 1000 and {w["method"] for w in body["window"]} == {"trend", "volume_rule", "health_20pct"}
    assert client.get("/experiments").json()[0]["id"] == body["id"]
    assert client.get(f"/experiments/{body['id']}").json()["caveat"] == body["caveat"]
    assert client.get("/experiments/9999").status_code == 404
