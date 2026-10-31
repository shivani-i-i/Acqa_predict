# AquaPredict - combined project (Person 1 + Person 2 + Person 3)

**Atomberg Innovation Challenge - Track 2: Water Purifiers.** Software prototype that forecasts filter maintenance for connected
purifiers: *when* a filter reaches its replacement criterion, *how sure* (calibrated interval where justified), *why* (drivers), and
whether a reading looks like a *sensor fault* instead of wear.

> Purifier telemetry is source-labeled. Synthetic households are simulator data and are not real Atomberg telemetry. Uploaded
> household telemetry remains separate from the optional live USGS environmental context. Neither public environmental data nor
> simulator output is converted into purifier filter-life labels. Results show algorithmic feasibility, not real-world accuracy.
> Nothing here assesses water safety.

## What is in the repo
| Package | Owner | What it does |
|---|---|---|
| `aquapredict/simulator`, `baselines`, `eval`, `planner`, `export.py`, `validate.py` | Person 1 | Household simulator (10 scenarios + custom, 2 wear laws, labelled faults), the 3 baselines, strict household-level evaluation harness with the abstract's 3 success criteria, stress tests, cost planner, exports |
| `aquapredict/forecasting` | Person 2 | `predict()` -> forecast, conformal interval, SHAP drivers, sensor-fault events, CUSUM source-change flag, status (`ok`/`degraded`/`withheld`) + reason; drift monitoring; its own simulator, training and demo scripts. Its own README: `aquapredict/forecasting/README.md` |
| `aquapredict/integration` | bridge | Adapter from Person 1's data to Person 2's schema, trainer, and a forecaster wrapper so **Person 2's real `predict()` is scored by Person 1's evaluation** |
| `aquapredict/reference_model`, `aquapredict/checks` | Person 1 | A *stand-in* forecaster I built before Person 2's code existed. Kept as a comparison; it does not replace Person 2's model |
| `aquapredict/app` | Person 3 | SQLite persistence, service layer, FastAPI backend, Streamlit dashboard (dashboard, filter detail, simulator and what-if, alerts, planner and costs, validation lab, models, export), model registry and training. See `docs/APP.md` |
| `docs/`, `docs_results/` | | Simulator documentation (every parameter + evidence status), measured results and caveats |

## How the pieces fit
```
Person 1 simulator --> telemetry (+ hidden truth) --> household-level split (train / calibration / test)
                                   |                                   |
                                   |                    integration.adapter (column mapping, cut to current filter)
                                   |                                   v
                                   |              Person 2: features -> forecast -> conformal interval -> gating,
                                   |                        sensor faults, CUSUM, SHAP     (predict())
                                   v                                   |
              Person 1 harness: gives forecasters PAST, label-free data only, scores against hidden truth
                      -> window error, late / premature warnings, interval coverage + width, PASS/FAIL criteria
```
Person 2 ships its **own** simulator (volume-based replacement at 80% capacity). Both simulators are kept: Person 2's demos use theirs;
the integration evaluation uses Person 1's (replacement criteria from sensor-like signals: pressure drop, flux, rejection, carbon load).

## Setup (Python 3.10+)
Windows PowerShell:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install shap        # optional: SHAP drivers
```
Mac/Linux: `python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`

## Run the application (Person 3)
```bash
streamlit run aquapredict/app/ui/streamlit_app.py      # dashboard; create a demo household on first launch
uvicorn aquapredict.app.api:app --port 8000            # REST API, docs at /docs
```
Details, architecture and limitations: `docs/APP.md`.

### Deploying

The FastAPI backend includes a Vercel entrypoint at `api/index.py` and can be
deployed to Vercel as a Python Function. After deployment, use:

```text
https://YOUR-VERCEL-DOMAIN.vercel.app/docs
```

The Streamlit dashboard is not a Vercel runtime. Deploy it separately on
Streamlit Community Cloud (select
`aquapredict/app/ui/streamlit_app.py`) or run it with the local command above.
The default SQLite database is local/ephemeral in serverless environments;
use an external database before treating a Vercel deployment as production.

### Public online water-quality context

The Data & export page and `GET /sources/usgs/observations?site=01646500` can fetch the latest
observations from the official USGS National Water Information System. The default station is
the Potomac River near Washington, DC (`01646500`). The response preserves the station,
timestamp, parameter code, unit, qualifier, and source URL.

USGS observations are environmental measurements, not household purifier telemetry. In
particular, specific conductance is not silently converted to TDS, and the observations
cannot produce filter replacement labels. Upload purifier telemetry through the existing
CSV workflow or `POST /telemetry/ingest` for filter forecasting.

## Run the other parts
```bash
python -m pytest -q                                              # whole suite (Person 1 + Person 2 + integration)
python -m aquapredict.eval.run_baselines --n 150 --seed 0 --calibrated --out results_calibrated   # baselines, strict metrics
python -m aquapredict.forecasting.examples.train_demo            # Person 2: train + calibrate + evaluate on ITS simulator
python -m aquapredict.forecasting.examples.run_demo              # Person 2: six predict() scenarios
python -m aquapredict.integration.evaluate --n 160 --seed 21     # Person 2's model on Person 1's simulator, strict criteria (~10-15 min on one CPU)
```
`make test | test-p1 | test-p2 | integrate | ...` where `make` exists.

## Using Person 2's model on Person 1's data
```python
from aquapredict.integration import train_person2_on_person1, Person2Forecaster
f = train_person2_on_person1(train_df, cal_df, "artifacts/bundle.joblib")   # disjoint households
f.respond(history_df, "sediment")      # Person 2's full response: forecast, interval, drivers, fault events, status, reason
f(history_df, "ro")                    # the harness contract: {point, lo, hi, withheld, reason, status}
```

## Assumptions made in the integration (state them wherever results are quoted)
- **Pressure channel (matters a lot).** Person 2 has one `pressure` input; Person 1 has inlet pressure and the sediment pressure *drop*. The adapter
  default is the raw drop (`dp`). The more "natural" `inlet - drop` mapping buries the wear signal under inlet-pressure noise and doubled the median
  error (59 vs 30 days, 48 held-out households). `--pressure-source downstream|dp|inlet` switches it. `dp` was chosen after seeing held-out results from
  seed 21 and then replicated on a fresh seed (`docs_results/RESULTS.md`). It is an assumption about sensor wiring, not a fact about real hardware.
- **Vacation (no-flow) days** carry no readings in Person 1's data; the adapter omits them so a holiday is not read as a failing sensor.
- **History must start at filter installation** (Person 2's requirement); the adapter cuts to the current filter using the visible age column.
- Person 2's model is **re-trained on Person 1's training households** (pooled across stages) and calibrated on separate households; test
  households are never used before scoring.
- Person 2's gate allows a forecast from as few as 3 readings; Person 1's harness never asks before 14 days. For real use, raise it.

More detail: `docs/PERSON1.md` (Person 1 module guide), `docs/SIMULATOR.md`, `docs_results/RESULTS.md`, `aquapredict/forecasting/README.md`.
