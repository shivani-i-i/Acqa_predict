# AquaPredict

## Synthetic filter-life forecasting for connected water purifiers

AquaPredict is an Atomberg Innovation Challenge Track 2 prototype for forecasting
when purifier filters may reach a simulated replacement criterion. It combines
simulation, baseline rules, machine-learning forecasting, sensor-quality checks,
an evaluation harness, a FastAPI service, and a Streamlit dashboard.

> **Important:** all benchmark households and labels in this repository are
> synthetic. Reported metrics describe algorithmic behavior on these simulators,
> not real-world accuracy or production reliability. AquaPredict does not assess
> water safety, potability, contaminant removal, or health risk.

## What the project provides

### 1. Synthetic household simulator

`aquapredict/simulator` generates reproducible purifier households with:

- usage patterns including normal, vacation-heavy, high-use, and irregular homes;
- multiple sensor channels such as pressure drop, flow, TDS, temperature, and
  inlet pressure;
- filter replacement events and hidden simulator truth;
- configurable wear laws, scenarios, seeds, and labelled fault injections;
- separate simulated stages for sediment, carbon, and RO membrane behavior.

The simulator is a controlled research environment. Its hidden end-of-life
labels are used only by evaluation code and demo views; they are not inferred
from USGS observations.

### 2. Three transparent baselines

`aquapredict/baselines` implements simple methods that are deliberately easy to
inspect:

- **Trend:** extrapolates a normalized wear indicator;
- **Volume rule:** uses rated cumulative usage and age;
- **20% health rule:** raises a maintenance recommendation when usage or age
  reaches the configured health threshold.

Every learned model is compared with these baselines. A complex model is not
automatically considered better than a transparent rule.

### 3. Forecasting pipeline

`aquapredict/forecasting` exposes the stable public function:

```python
from aquapredict.forecasting.predict import predict

response = predict({
    "filter_stage": "sediment",
    "household_id": "H001",
    "history": readings,
    "explain": True,
})
```

The response contains a point forecast, an optional conformal interval, sensor
fault events, a source-change signal, status, reason, and disclaimer. The
pipeline includes:

- input and history validation;
- feature engineering from label-free telemetry;
- gradient-boosted forecasting;
- split-conformal uncertainty intervals where justified;
- sensor-fault detection for stuck, drifting, spiking, dropped, or inverted
  signals;
- CUSUM-style source or operating-condition change detection;
- optional SHAP drivers;
- explicit `ok`, `degraded`, and `withheld` statuses.

The function does not silently convert invalid input into a success-shaped
forecast. If data quality or history is insufficient, it withholds the result.

### 4. Integration layer

`aquapredict/integration` adapts the simulator's telemetry to the forecasting
pipeline and lets the real forecasting interface run through the project
evaluation harness.

The integration supports:

- current-filter history cutting;
- vacation/no-flow handling;
- stage-specific training artifacts;
- disjoint training and calibration households;
- development-only pressure-source and model selection;
- baseline fallback when a learned model does not win on development data;
- the original harness contract:
  `{point, lo, hi, withheld, reason, status}`.

The pressure mapping is an explicit sensor-wiring assumption. It must not be
interpreted as a fact about deployed purifier hardware.

### 5. Reference comparison model

`aquapredict/reference_model` is a separately maintained comparison forecaster.
It is retained because a simpler model can outperform the integrated pipeline on
some synthetic benchmark splits. It is not deleted or presented as a
production model.

### 6. Strict evaluation

`aquapredict/eval` evaluates forecasts using past readings only and scores fresh
synthetic households. The evaluation reports:

- median absolute window error;
- late-warning percentage;
- premature-warning percentage;
- conformal coverage;
- interval width;
- error as a percentage of simulated filter lifetime;
- stage-level and method-level summaries;
- asymmetric alert-cost summaries.

The success criteria are evaluated against held-out synthetic households. No
test household is used for model fitting, pressure selection, or threshold
tuning.

### 7. FastAPI backend

`aquapredict/app/api.py` provides:

- health and scenario metadata;
- model availability and training;
- household creation and simulator controls;
- telemetry ingestion;
- forecasts, prediction history, and data-quality views;
- sensor events and alert actions;
- maintenance planning and cost estimates;
- validation-lab runs;
- CSV export;
- optional USGS environmental context.

The API is a prototype boundary. Configure `AQUAPREDICT_API_KEY` for
environment-based API-key authentication and `AQUAPREDICT_RATE_LIMIT` for a
per-client request limit. In production, long model-training and validation
jobs should run through a durable background worker and persistent database.

### 8. Streamlit dashboard

`aquapredict/app/ui/streamlit_app.py` provides a working operational view with:

- a dashboard for system health and maintenance windows;
- filter detail pages with forecasts, drivers, intervals, and sensor events;
- simulator controls and what-if comparisons;
- alert acknowledgement, resolution, and dismissal;
- maintenance planning and cost analysis;
- a validation lab with fresh synthetic households;
- model cards and availability;
- data upload, export, and USGS context.

Carbon is intentionally handled differently: the simulator has no direct carbon
wear sensor. The UI routes carbon to a volume rule, hides a misleading
confidence interval, and explains that policy to the user.

## Architecture

```text
Synthetic simulator
        |
        v
Telemetry + hidden demo truth
        |
        +--> Transparent baselines --------------------+
        |                                               |
        +--> Reference model                            |
        |                                               v
        +--> Integration adapter --> Forecasting --> Evaluation harness
                                      |       |
                                      |       +--> intervals, faults, drivers
                                      |
                                      +--> Model registry --> FastAPI
                                                           |
                                                           +--> Streamlit UI
```

Uploaded purifier telemetry is kept separate from simulator truth. Optional
USGS data is environmental context only and is never used to create filter-life
labels.

## Function-level data flow

The following is the runtime path for a forecast. Names below are the public
functions or the main orchestration functions a maintainer should inspect.

1. **Create or load a household**
   - `AquaService.create_household()` validates the requested scenario and
     stores source metadata in SQLite.
   - `Simulator.generate_household()` creates reproducible readings from the
     seed. Synthetic truth is stored separately from observed telemetry.
   - `AquaService.history()` returns only the readings available at the
     requested day; it is the boundary that prevents future readings entering a
     forecast.
2. **Check telemetry**
   - `data_quality()` measures the latest valid reading, recent missingness,
     and idle days.
   - `check_readings()` identifies stuck, drifting, spiking, dropped, and
     inverted sensors.
   - `AquaService.indicators()` derives stage condition from sensor evidence.
     Carbon deliberately uses cumulative volume because it has no direct wear
     sensor in the simulator.
3. **Generate a forecast**
   - `AquaService.forecast()` selects the requested registry model, applies the
     carbon volume-rule policy, runs the stable forecasting interface, and
     attaches sensor events, reliability, recommendation, and provenance.
   - `AquaService.forecast_all()` calls that function for all three stages.
   - `ModelRegistry.get()` loads only verified artifacts; the SHA-256 sidecar
     is checked before joblib/pickle loading.
4. **Build the dashboard**
   - `AquaService.dashboard()` combines household metadata, stage forecasts,
     data quality, open alerts, and maintenance windows into one response.
   - `web/lib/api.ts` normalizes the backend's flat stage response into the
     typed `Dashboard` shape used by the Next.js UI. This keeps API contracts
     in one client instead of duplicating them in components.
   - `streamlit_app.py::page_dashboard()` renders the same service response,
     including the Data health & provenance panel.
5. **Evaluate without leakage**
   - `generate_dataset()` creates fresh synthetic households.
   - `evaluate()` cuts each history at the forecast day and scores only
     information available at that point.
   - `run_validation()` rejects a learned model when the requested validation
     seed equals its training seed.
   - `check_success()` and the metric helpers report window error, late and
     premature alerts, interval coverage/width, and lifetime-normalized error.

### Endpoint-to-function map

| HTTP route | Service function | Purpose |
|---|---|---|
| `GET /health` | `health()` | Liveness and source-boundary metadata |
| `GET /households/{id}/dashboard` | `dashboard()` | UI-ready household snapshot |
| `GET /households/{id}/forecast` | `forecast()` / `forecast_all()` | Stage forecast and explanation |
| `GET /households/{id}/data-quality` | `data_quality()` | Missingness and freshness |
| `GET /households/{id}/sensor-events` | `sensor_events()` | Sensor-fault evidence |
| `GET /alerts` | `alerts()` | Alert list and audit state |
| `POST /alerts/{id}/{action}` | `alert_action()` | Acknowledge, resolve, or dismiss |
| `POST /households/{id}/what-if` | `what_if()` | Scenario-only forecast comparison |
| `POST /validation/run` | `run_validation()` | Fresh synthetic evaluation |
| `POST /models/train` | `ModelRegistry.train()` | Train and persist verified artifacts |

The browser frontend does not contain a second forecasting implementation:
`web/lib/api.ts` is an HTTP adapter, `web/app/page.tsx` is presentation/state
orchestration, and `web/components/PurifierScene.tsx` is a visualization of
the API values. A missing forecast is shown as withheld rather than replaced
with a fabricated number.

## Setup

Python 3.10 or newer is recommended.

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-full.txt
pip install shap
```

SHAP is optional. The application still runs without it, but driver explanations
may be unavailable.

### macOS/Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-full.txt
pip install shap
```

## Run the application

```bash
streamlit run aquapredict/app/ui/streamlit_app.py
uvicorn aquapredict.app.api:app --port 8000
```

The API's interactive documentation is available at `/docs`.

### Optional 3D web frontend

The repository also includes a standalone Next.js frontend in [`web/`](web/).
It provides a responsive procedural 3D purifier view and a browser dashboard
for households, filter forecasts, alerts, planning, validation, models, and
data boundaries. It uses the same FastAPI service; it does not duplicate model
logic in JavaScript.

Start the API first, then run:

```bash
cd web
copy .env.example .env.local          # Windows
# cp .env.example .env.local          # macOS/Linux
npm install
npm run dev
```

Open `http://localhost:3000`. Set
`NEXT_PUBLIC_API_URL` to the deployed API URL when the backend is not local.
For browser requests, configure the backend's
`AQUAPREDICT_FRONTEND_ORIGIN` to the frontend origin. The 3D view is procedural
and has no external model or asset license dependency; reduced-motion users
receive the same information without relying on animation.

The Vercel entrypoint is intentionally limited to the public health and USGS
context routes. The full FastAPI service and Streamlit dashboard should be
deployed separately, with persistent storage and authentication configured
before any non-demo use.

### Deploy the 3D web frontend and full API

The Next.js frontend is deployable as a separate Vercel project:

1. Import this repository into Vercel.
2. Set **Root Directory** to `web`.
3. Set `NEXT_PUBLIC_API_URL` to the public Render API URL.
4. If `AQUAPREDICT_API_KEY` is enabled on Render, set the matching
   `NEXT_PUBLIC_API_KEY` in Vercel. This key is exposed to the browser, so use
   it only for a controlled demo; production deployments should use a
   user-session authentication layer.
5. Deploy.

The full FastAPI service is configured in [`render.yaml`](render.yaml):

1. Create a Render Blueprint from this repository.
2. Set `AQUAPREDICT_FRONTEND_ORIGIN` to the Vercel URL.
3. Set a strong `AQUAPREDICT_API_KEY`.
4. Redeploy the Vercel project with the same API key if protected API calls are
   enabled in the frontend.

The existing root Vercel function remains the small public health/USGS
endpoint. It is not the full forecasting backend. Use the Render URL for
households, forecasts, alerts, planning, validation, and model operations.

## Run tests and experiments

```bash
python -m pytest -q
python -m aquapredict.eval.run_baselines --n 150 --seed 0 --calibrated --out results_calibrated
python -m aquapredict.forecasting.examples.train_demo
python -m aquapredict.forecasting.examples.run_demo
python -m aquapredict.integration.evaluate --n 160 --seed 21
```

The integration evaluation can take several minutes on one CPU. Use a fresh
seed when measuring generalization. Never tune a model or selection policy on
the households used for the final report.

## Model artifacts and integrity

Learned artifacts are pickle/joblib files and therefore must only be loaded
from trusted locations. Saved artifacts have SHA-256 sidecars. Loading rejects
missing or mismatched sidecars so accidental or unauthorized changes do not
silently become executable model code.

Retrain artifacts when the feature schema, scikit-learn version, or training
configuration changes.

## Synthetic benchmark status

The repository's earlier integrated benchmark showed that the integrated model
did not meet all three target criteria: it over-predicted remaining life,
produced too many premature alerts, and produced wide intervals. The reference
comparison model performed better on that synthetic split. These findings are
kept as limitations rather than presented as real-world accuracy claims.

Always read:

- [`docs/AUDIT.md`](docs/AUDIT.md) for the evidence-based audit;
- [`docs_results/RESULTS.md`](docs_results/RESULTS.md) for benchmark definitions
  and caveats;
- [`docs/APP.md`](docs/APP.md) for application behavior;
- [`aquapredict/forecasting/README.md`](aquapredict/forecasting/README.md) for
  the forecasting API and response schema.

## Data and safety boundaries

- Simulator data is synthetic and labelled for controlled evaluation.
- Uploaded telemetry is treated as source-labelled household data.
- USGS observations are public environmental context only.
- USGS observations are never turned into filter-life labels.
- Forecasts are maintenance estimates, not water-quality or health decisions.
- A forecast interval is not a guarantee.
- Sensor faults and distribution shift can invalidate a forecast.

## License and contribution notes

This repository is an innovation-challenge prototype. Before production use,
review the data-governance, authentication, persistence, artifact-loading,
observability, and deployment requirements for the target environment.
