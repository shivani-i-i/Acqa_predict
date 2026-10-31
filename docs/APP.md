# AquaPredict application (Person 3): API, dashboard, persistence

**Everything runs on synthetic data.** The app is a local prototype: no authentication, one SQLite file, not meant to be exposed to a network.
Forecasts predict filter wear only; nothing here assesses water safety.

## Run it
```bash
pip install -r requirements.txt
streamlit run aquapredict/app/ui/streamlit_app.py          # dashboard  (http://localhost:8501)
uvicorn aquapredict.app.api:app --reload --port 8000        # REST API   (interactive docs: http://localhost:8000/docs)
```
Both use the same service layer and the same database (`aquapredict_app_data/app.db`; change with `AQUAPREDICT_HOME` / `AQUAPREDICT_DB`).
The baselines work immediately. To use the learned models open **Models -> Train** (about 2-4 min for 120 households) or call `POST /models/train`.
Trained artifacts are pickles: only load files created on this machine. They are not shipped (they depend on library versions).

## Architecture
```
Streamlit UI  --\                                    +-- simulator (Person 1)  -> telemetry + hidden truth
                  +--> AquaService (all logic) ------+-- registry -> baselines | reference model | Person 2's predict()
FastAPI       --/        |                           +-- rule checks, alert rules, cost planner, strict evaluation harness
                         v
                      SQLite: households, readings, predictions, alerts, maintenance, experiments
```
`aquapredict/app/`: `service.py` (logic) · `registry.py` (model wrappers, training, model cards) · `alerts.py` (pure alert rules) ·
`db.py` · `schemas.py` (request validation) · `api.py` · `ui/streamlit_app.py`.

## What the dashboard does (every control calls real code)
| Page | Contents |
|---|---|
| Dashboard | Counts, data freshness, per-filter **condition**, **sensor health** and **forecast reliability** shown separately, window + interval or "withheld" with the reason, recommendation, alerts, upcoming maintenance, model and evaluation status |
| Filter detail | Forecast, interval basis, drivers and explanation (Person 2's model), sensor/data-quality events, trend and forecast-history charts, backfill of past forecasts, simulator ground truth (demo only) |
| Simulator & what-if | Create households (11 scenarios, 2 wear laws, overrides), record a replacement today, inject a sensor fault, hypothetical what-if comparison |
| Alerts | De-duplicated, severity-ordered; acknowledge / resolve / dismiss with notes |
| Planner & costs | predicted -> recommended -> confirmed -> completed; editable illustrative prices; spend estimate |
| Validation lab | Runs the project's strict evaluation on FRESH synthetic households; criteria PASS/FAIL; policy costs; saved experiments |
| Models | Availability, model cards with reported results and limitations, training |
| Data & export | CSV downloads (telemetry, predictions, alerts, maintenance, experiments), household config JSON, upload of real telemetry |

## Design rules that are enforced (and tested)
- **No peeking:** a household's readings can only be read up to its current day; forecasts use past data only.
- **The past is fixed:** injecting a fault or recording a replacement leaves every earlier reading unchanged (simulator re-run with the event applied from today on).
- **Ground truth is demo-only** and never exists for uploaded telemetry.
- **Withheld means withheld:** no number is invented. Insufficient history, missing data, suspected sensor faults and out-of-range conditions produce a reason.
- **Alerts:** one alert per condition, updated not duplicated; auto-resolve only after the condition is absent on two consecutive checks (no flapping),
  or immediately when a replacement is recorded. Model-generated alerts and user actions are recorded separately.
- **Planner states** are distinct. The app never books a service, buys a cartridge or contacts anyone.
- **Validation lab refuses a learned model on its own training seed.**
- Absurd extrapolations are shown as "more than 2 years away (very uncertain)", not as a precise number.

## REST API (see /docs for schemas)
`GET /health /scenarios /models` · `POST /models/train` · `GET|POST /households`, `GET|DELETE /households/{id}`, `.../readings`, `.../dashboard`,
`.../forecast?stage=&model=`, `.../truth`, `.../data-quality`, `.../sensor-events`, `.../predictions`, `.../plan`, `.../cost-estimate`, `.../config` ·
`POST .../advance|faults|replace|what-if|monitor` · `POST /plan/{id}/confirm|cancel|complete` · `GET /alerts`, `POST /alerts/{id}/acknowledge|resolve|dismiss` ·
`POST /telemetry/ingest` · `POST /validation/run`, `GET /experiments[/{id}]` · `GET /export/{kind}`.
Status codes: 404 not found, 409 conflict / model not trained / invalid state change, 422 invalid input.

## Real-telemetry boundary
`POST /telemetry/ingest` (or the upload box) accepts daily rows (`day`, `usage_L`, sensor columns with nulls allowed) and the days each filter was replaced.
It derives filter ages and volumes, validates the schema, rejects bad files with reasons and stores the household as `uploaded` (no ground truth, no simulated clock).
It has never been exercised with a real device: device protocols, authentication, time zones and sensor quantisation are not handled.

## Known limitations
- **I could only test the UI headlessly** (Streamlit's AppTest: every page and button, plus real server start-up). I have not seen it rendered in a browser, so layout and look are unreviewed.
- The "demo clock" replays simulator output; it is not real-time. Model cards quote results from `docs_results/RESULTS.md` (static text); the Validation lab recomputes.
- Forecast quality is whatever the selected model delivers. On the synthetic benchmark the reference model has wide intervals and Person 2's integrated model fails the success
  criteria (`docs_results/RESULTS.md`); the baselines' raw forecasts are noisy, which is why alert hysteresis exists.
- Carbon has no sensor signal in the simulator; its condition and forecast are usage-based.
- Single shared SQLite connection and no user accounts: one user at a time.
- Prices are illustrative placeholders. Person 2's drivers need `pip install shap`.
