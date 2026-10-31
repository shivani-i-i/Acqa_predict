"""FastAPI backend. Run:  uvicorn aquapredict.app.api:app --reload   (interactive docs at /docs).
Local prototype: no authentication, one SQLite file. Do not expose to a network as-is."""
from __future__ import annotations
import os
from typing import Optional

from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import JSONResponse

from . import schemas as S
from .registry import ModelUnavailable
from .service import AquaService, NotFound, BadRequest, Conflict, clean
from .. import __version__

STAGES = ("sediment", "carbon", "ro")


def create_app(service: Optional[AquaService] = None) -> FastAPI:
    svc = service
    app = FastAPI(title="AquaPredict API", version=__version__,
                  description="Filter-maintenance forecasting prototype. Household purifier data are source-labeled: uploaded "
                              "telemetry is real user data, synthetic households are simulator data, and USGS observations are "
                              "environmental context. Forecasts predict filter wear only; they do not assess water safety.")

    def S_() -> AquaService:
        nonlocal svc
        if svc is None:
            home = os.environ.get("AQUAPREDICT_HOME", "aquapredict_app_data")
            os.makedirs(home, exist_ok=True)
            svc = AquaService(os.environ.get("AQUAPREDICT_DB", os.path.join(home, "app.db")))
        return svc

    app.state.service = S_

    @app.exception_handler(NotFound)
    async def _nf(_: Request, e: NotFound):
        return JSONResponse(status_code=404, content={"detail": str(e.args[0])})

    @app.exception_handler(BadRequest)
    async def _br(_: Request, e: BadRequest):
        return JSONResponse(status_code=422, content={"detail": str(e)})

    @app.exception_handler(Conflict)
    async def _cf(_: Request, e: Conflict):
        return JSONResponse(status_code=409, content={"detail": str(e)})

    @app.exception_handler(ModelUnavailable)
    async def _mu(_: Request, e: ModelUnavailable):
        return JSONResponse(status_code=409, content={"detail": str(e)})

    # ------------------------------------------------------------------ meta
    @app.get("/health")
    def health():
        return {"status": "ok", "version": __version__,
                "data": "source-labeled: uploaded purifier telemetry, simulator data, and optional USGS environmental context"}

    @app.get("/scenarios")
    def scenarios():
        return clean(S_().scenarios())

    @app.get("/models")
    def models():
        return clean(S_().models())

    @app.get("/sources/usgs/observations")
    def usgs_observations(site: str = "01646500"):
        """Live environmental context from the USGS NWIS instantaneous-values API."""
        return clean(S_().online_water_quality(site))

    @app.post("/models/train")
    def train(body: S.TrainIn):
        """Trains on fresh synthetic households (takes minutes). Learned models are refused in the Validation Lab on the same seed."""
        return clean(S_().registry.train(tuple(body.which), n=body.n, seed=body.seed, max_iter=body.max_iter, log=lambda *_: None))

    # ------------------------------------------------------------------ households
    @app.get("/households")
    def households():
        return clean(S_().list_households())

    @app.post("/households", status_code=201)
    def create_household(body: S.HouseholdIn):
        return clean(S_().create_household(body.name, body.scenario, body.law, body.seed, body.sim_index, body.overrides,
                                           body.calibrated, body.now_day, [f.model_dump() for f in body.user_faults]))

    @app.get("/households/{hid}")
    def household(hid: str):
        return clean(S_().household(hid))

    @app.delete("/households/{hid}", status_code=204)
    def delete_household(hid: str):
        S_().delete_household(hid)
        return Response(status_code=204)

    @app.get("/households/{hid}/config")
    def config(hid: str):
        return clean(S_().export_household_config(hid))

    @app.get("/households/{hid}/readings")
    def readings(hid: str, last: Optional[int] = Query(default=None, ge=1, le=2000), include_truth: bool = False):
        return clean(S_().readings(hid, include_truth=include_truth, last=last).to_dict("records"))

    @app.get("/households/{hid}/dashboard")
    def dashboard(hid: str, model: Optional[str] = None):
        return clean(S_().dashboard(hid, model))

    @app.get("/households/{hid}/forecast")
    def forecast(hid: str, stage: Optional[S.Stage] = None, model: Optional[str] = None, explain: bool = True):
        s = S_()
        return clean(s.forecast(hid, stage, model, explain=explain) if stage else s.forecast_all(hid, model, explain=explain))

    @app.get("/households/{hid}/truth")
    def truth(hid: str, stage: S.Stage):
        return clean(S_().truth(hid, stage))

    @app.get("/households/{hid}/data-quality")
    def data_quality(hid: str):
        return clean(S_().data_quality(hid))

    @app.get("/households/{hid}/sensor-events")
    def sensor_events(hid: str, stage: Optional[S.Stage] = None):
        return clean(S_().sensor_events(hid, stage))

    @app.get("/households/{hid}/predictions")
    def predictions(hid: str, stage: Optional[S.Stage] = None, model: Optional[str] = None):
        return clean(S_().prediction_history(hid, stage, model).to_dict("records"))

    # ------------------------------------------------------------------ simulator controls
    @app.post("/households/{hid}/advance")
    def advance(hid: str, body: S.AdvanceIn):
        return clean(S_().advance(hid, body.days, body.step, body.model))

    @app.post("/households/{hid}/faults")
    def inject(hid: str, body: S.FaultIn):
        return clean(S_().inject_fault(hid, body.model_dump()))

    @app.post("/households/{hid}/replace")
    def replace(hid: str, body: S.ReplaceIn):
        return clean(S_().record_replacement(hid, body.stage, body.note))

    @app.post("/households/{hid}/what-if")
    def what_if(hid: str, body: S.WhatIfIn):
        return clean(S_().what_if(hid, body.overrides, [f.model_dump() for f in body.user_faults], body.model, body.stage))

    @app.post("/households/{hid}/monitor")
    def monitor(hid: str, model: Optional[str] = None):
        return clean(S_().update_monitoring(hid, model))

    # ------------------------------------------------------------------ planner
    @app.get("/households/{hid}/plan")
    def plan(hid: str, include_closed: bool = True):
        S_().household(hid)
        return clean(S_().plan(hid, include_closed))

    @app.get("/households/{hid}/cost-estimate")
    def cost_estimate(hid: str, horizon_days: int = Query(default=90, ge=1, le=730)):
        return clean(S_().cost_estimate(hid, horizon_days))

    @app.post("/plan/{plan_id}/confirm")
    def confirm(plan_id: int, body: S.ConfirmIn):
        return clean(S_().confirm_plan(plan_id, body.planned_day))

    @app.post("/plan/{plan_id}/cancel")
    def cancel(plan_id: int):
        return clean(S_().cancel_plan(plan_id))

    @app.post("/plan/{plan_id}/complete")
    def complete(plan_id: int, body: S.NoteIn):
        return clean(S_().complete_plan(plan_id, body.note))

    # ------------------------------------------------------------------ alerts
    @app.get("/alerts")
    def alerts(household_id: Optional[str] = None, status: Optional[str] = None, stage: Optional[S.Stage] = None):
        return clean(S_().alerts(household_id, status, stage))

    @app.post("/alerts/{alert_id}/{action}")
    def alert_action(alert_id: int, action: str, body: S.NoteIn):
        return clean(S_().alert_action(alert_id, action, body.note))

    # ------------------------------------------------------------------ real telemetry boundary, validation lab, export
    @app.post("/telemetry/ingest", status_code=201)
    def ingest(body: S.IngestIn):
        return clean(S_().ingest_telemetry(body.name, body.rows, {k: v for k, v in body.replacements.items()}))

    @app.post("/validation/run")
    def validation(body: S.ValidationIn):
        return S_().run_validation(body.n_households, body.seed, body.methods, body.calibrated, body.cost)

    @app.get("/experiments")
    def experiments():
        return clean(S_().experiments())

    @app.get("/experiments/{exp_id}")
    def experiment(exp_id: int):
        return S_().experiment(exp_id)

    @app.get("/export/{kind}")
    def export(kind: str, household_id: Optional[str] = None, include_truth: bool = False):
        text = S_().export_csv(kind, household_id, include_truth)
        return Response(text, media_type="text/csv", headers={"Content-Disposition": f'attachment; filename="aquapredict_{kind}.csv"'})

    return app


app = create_app()
