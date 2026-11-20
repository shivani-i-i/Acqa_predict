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

## Valid model families and data sources

The repository ships trained synthetic artifacts:

- `reference_v3.pkl` — gradient-boosted reference model with conformal-style
  intervals and sensor/range gating.
- `person2_bundle.<stage>.joblib` — per-stage HistGradientBoosting synthetic
  forecasting bundles.
- `models_meta.json` — training seed, household count, timestamp, and
  development-set stage selection.

At runtime AquaPredict prefers these repository artifacts and falls back to
the transparent Trend baseline only when no repository or local trained model
is available. The artifacts are synthetic artifacts, not evidence of real
purifier accuracy, and must be retrained before deployment on real telemetry.

### Fresh-seed synthetic model check

A fresh validation run used seed `5151`, 20 households, and a new test split
that was not used to train the shipped artifacts.

| Model | Median absolute error, 0-60 true days left | Late warning rate | Premature warning rate | Interval coverage | Median interval width |
|---|---:|---:|---:|---:|---:|
| **Reference v3 (shipped)** | **14.0 d** | **27.4%** | **32.1%** | 87.5% | 154.5 d all / 45.6 d primary |
| Person 2 (shipped) | 21.2 d | 32.4% | 55.1% | 92.6% | 223.5 d all / 77.1 d primary |
| Trend baseline | 18.4 d | 35.8% | 21.3% | n/a | n/a |
| Volume-rule baseline | 35.9 d | 53.0% | 34.8% | n/a | n/a |
| 20% health baseline | 27.3 d | 45.6% | 43.6% | n/a | n/a |

This table proves model behavior on the project simulator. It does **not**
prove real purifier accuracy. It also shows the reference model has useful
synthetic point performance but interval width remains larger than desired.

AquaPredict separates three kinds of analysis so that users can distinguish what
is learned, what is rule-based, and what is merely exploratory:

1. **Regression forecasting**
   - The main learned model is an actual scikit-learn
     `HistGradientBoostingRegressor` trained on synthetic household rows.
   - The target is the simulator's remaining-days-until-filter-criterion value.
   - Training, calibration, development, and test households are kept separate.
   - The implementation also uses conformal calibration for uncertainty bands.
2. **Transparent baselines**
   - Trend extrapolation, volume rule, and 20% health threshold are real,
     inspectable baseline methods.
   - They are never presented as machine-learning breakthroughs or real-world
     accuracy.
3. **Unsupervised operating-profile analysis**
   - The app fits a deterministic Lloyd-style KMeans model to 36 generated
     synthetic households using standardized mean usage, usage variability,
     flow, output TDS, and sediment pressure drop.
   - Standardization is performed only on the synthetic population used for that
     analysis.
   - KMeans assigns the current household to a low-use, typical-use, or high-use
     cluster and reports silhouette quality and distance from the cluster centre.
   - A robust standardized-distance threshold marks a pattern as rare. This is an
     exploratory operating-profile model, not a fault detector, filter-life
     model, or measurement of real equipment.

No LLM, generic chatbot, or external search result is used as a model for
filter-life training. No Kaggle dataset, USGS observation, or online water
record is converted into a filter replacement label.

## Developer verification

From the repository root:

```bash
python -m pip install -r requirements-full.txt
python -m pip install -e ".[dev]"
ruff check aquapredict tests
pytest -q
```

GitHub Actions runs the same lint and test gates on `main`, `streamlit-host`, and
pull requests. The full test suite is synthetic and does not require live purifier
hardware or real-world filter-life ground truth.

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

### 9. Hackathon evidence mode

The **Validation lab → Hackathon demo: live fail-safe evidence** panel is a
repeatable camera-ready demonstration of the core safety behavior:

1. click **Run fail-safe demonstration**;
2. the service injects a stuck differential-pressure sensor fault into the
   selected synthetic household;
3. the synthetic clock advances through the fault window;
4. monitoring and the sediment forecast run through the same service/API path;
5. the screen shows the measured status, sensor health, point estimate, event
   count, reason, and raw sensor-event evidence.

The expected safe behavior is `withheld` or `degraded` with a
`fault_suspected` sensor state. If the result is an unqualified forecast, the
panel displays an explicit failure message rather than hiding it. This is a
software fail-safe demonstration on synthetic telemetry, not a physical
water-safety test. For a presentation, record the panel, then run the
Validation lab on a fresh seed and show the measured metrics table and its
synthetic-data caveat.

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

## Complete operating guide

This section describes the project from a user's first click through the
forecast, alert, planner, and evaluation paths. It is intended to be enough
context for a new contributor, judge, or operator to understand what the
prototype does without opening the source code first.

### The central product promise

AquaPredict is a **filter-wear forecasting prototype** for a connected water
purifier. Given source-labelled purifier telemetry, it estimates how many days
remain before a simulated or user-defined replacement criterion is reached.
It does not decide whether water is safe to drink, does not certify a
purifier, and does not replace a manufacturer's maintenance instructions.

Every displayed result has three separate concepts:

1. **Condition** — evidence about current wear, such as pressure drop per flow,
   normalized RO flux/rejection, or cumulative carbon volume.
2. **Sensor health** — whether the input signals appear usable, including
   missing, stuck, drifting, spiking, dropped, or inverted readings.
3. **Forecast reliability** — whether the model has enough trustworthy history
   and whether a calibrated interval is available.

These concepts must not be collapsed into one “accuracy” score. A filter may
look worn while its sensor evidence is unreliable; in that case the forecast
can be withheld.

### End-to-end request lifecycle

For a normal dashboard request, the following sequence occurs:

```text
Browser/Streamlit
    |
    | GET /households/{id}/dashboard
    v
FastAPI route in app/api.py
    |
    v
AquaService.dashboard()
    |
    +--> household metadata and source label
    +--> history() cuts readings at the current day
    +--> forecast_all()
    |       +--> registry.get()
    |       +--> stage forecaster.run()
    |       +--> sensor checks and reliability policy
    |       +--> recommendation and optional prediction record
    +--> data_quality()
    +--> alerts()
    +--> plan()
    v
Typed response
    |
    +--> Streamlit renders directly
    +--> Next.js api.ts normalizes the response
         and renders text plus the procedural 3D scene
```

The browser never trains a model and never calculates a replacement label.
JavaScript displays values returned by FastAPI. This keeps model behavior in
Python and makes the Streamlit and browser interfaces use the same service
layer.

### Source and label policy

There are three source categories:

| Source | What it contains | Can it create filter-life labels? |
|---|---|---|
| Synthetic simulator | Reproducible readings, injected faults, hidden simulated truth | Yes, for controlled evaluation only |
| Uploaded telemetry | User-provided purifier readings and optional replacement days | Only from replacement days explicitly supplied by the user |
| USGS observations | Public environmental station context | **Never** |

Generic public water-quality data, including Kaggle datasets, do not become
purifier replacement labels merely because they contain TDS, conductivity,
pressure, or flow-like columns. A dataset can be used for external context or
feature stress testing only when its semantics, units, provenance, and license
are verified. Real-world accuracy requires a real dataset with observed
purifier filter replacements or end-of-life events.

## Streamlit dashboard: every tab

The Streamlit application is
[`aquapredict/app/ui/streamlit_app.py`](aquapredict/app/ui/streamlit_app.py).
It is the most complete operational interface because it exposes simulator
controls, training, validation, exports, and the camera-ready fail-safe demo.

### Shared command bar

The command bar appears when at least one household exists:

- **AquaPredict brand** — identifies the prototype and its purifier-care scope.
- **Page navigation** — switches the active Streamlit view.
- **Purifier/household selector** — changes the household used by all pages.
- **Forecast model selector** — chooses a baseline or an available learned
  artifact.
- **Settings popover** — exposes simulator clock advancement and monitoring
  refresh for synthetic households.

The selected household and model are stored in Streamlit session state. The
SQLite service database stores durable household, prediction, alert,
maintenance, and experiment records.

### Dashboard

`page_dashboard()` is the main operational view.

It shows:

- source label and current synthetic/demo day;
- latest flow, output TDS, and usage readings;
- estimated stage wear signals;
- open alerts and approaching maintenance windows;
- the procedural purifier/digital-twin view;
- selectable sediment, carbon, and RO stages;
- condition, sensor-health, reliability, forecast window, and recommendation
  for each stage;
- the **Data health & provenance** panel;
- the always-visible synthetic-data and water-safety boundary.

The 3D/procedural view is a visual aid. It is not a second model. Cartridge
colors and selected-stage emphasis are driven by the dashboard response.

### Filter detail

`page_filter_detail()` focuses on one stage. It provides:

- stage selector;
- current condition and progress;
- point forecast and interval when available;
- model status (`ok`, `degraded`, `point_only`, or `withheld`);
- reliability and reason text;
- model drivers when the selected forecaster supplies them;
- sensor events and their blocking/non-blocking effect;
- recent prediction history;
- simulator truth only for synthetic demo households.

Simulator truth is explicitly marked as demo-only. It is never available for
uploaded telemetry and is not used as a live production input.

### Simulator & what-if

`page_simulator()` is the reproducible demonstration environment.

**Create a synthetic household**

- choose a scenario such as normal, high-use, irregular, or vacation-heavy;
- choose the simulator wear law;
- choose a seed;
- optionally set supported parameter overrides;
- choose the starting day and calibrated/un-calibrated simulator mode.

The same seed and configuration reproduce the same simulated household.

**Controls**

- advance the synthetic clock;
- record a sediment, carbon, or RO replacement today;
- inject a fault beginning today or later;
- refresh monitoring, forecasts, alerts, and maintenance plans.

Past readings are not rewritten when a fault or replacement is added. This is
important for demonstrating that the forecast only receives information that
would have existed at the forecast time.

**What-if**

What-if scenarios re-simulate a hypothetical world with changed usage or
fault assumptions. They do not overwrite the actual household. The UI labels
the result as hypothetical and can show simulator truth only for comparison.

### Alerts

`page_alerts()` lists model-generated alert records and separates them from
user actions.

Filters:

- status: open, acknowledged, resolved, or dismissed;
- stage: sediment, carbon, RO, or all;
- refresh monitoring.

Each alert includes severity, title, message, day range, origin, kind,
recommended action, and an optional note. Actions are constrained to the
backend enum:

- **Acknowledge** — the operator has seen the alert;
- **Resolve** — the condition is no longer active or has been handled;
- **Dismiss** — the operator intentionally removes it from the active queue.

Alerts are de-duplicated and use hysteresis so a transient clean reading does
not immediately flap an alert open and closed.

### Planner & costs

`page_planner()` exposes the maintenance state machine:

```text
predicted -> recommended -> confirmed -> completed
                         \-> cancelled
```

The planner never books service or purchases a cartridge. It displays the
forecast window, lets the operator confirm a planned day, records completion,
or cancels an entry.

Cost figures are explicitly illustrative. They are useful for comparing
policies on synthetic households, not for claiming manufacturer pricing or
measured savings.

### Validation lab

`page_validation()` runs the strict evaluation harness on fresh synthetic
households. It reports:

- number of forecast opportunities and filter lives;
- short histories excluded equally for every method;
- window error;
- late and premature warnings;
- conformal interval coverage and width;
- stage-level results;
- success criteria per learned model;
- illustrative policy cost;
- metric definitions and caveat;
- downloadable JSON.

The validation seed must differ from a learned model's training seed. The
service rejects a matching seed to prevent accidental leakage.

**Hackathon demo: live fail-safe evidence**

The expanded panel in this tab is a camera-ready software demonstration:

1. click **Run fail-safe demonstration**;
2. the service injects a stuck `dp_sediment_bar` fault;
3. the synthetic clock advances through the fault;
4. monitoring and the sediment forecast execute;
5. the panel displays the actual status, sensor health, estimate, event count,
   reason, and raw sensor evidence.

The desired safe response is a withheld or degraded forecast with
`fault_suspected` health. If the result is not safe, the panel says so instead
of hiding the failure.

### Models

`page_models()` displays model cards, availability, training metadata, and
limitations. It also exposes training controls:

- model selection;
- number of synthetic training households;
- training seed;
- boosting iterations.

Training can take minutes. Artifacts are written with SHA-256 sidecars. The
registry refuses artifacts that are missing or fail integrity verification.
The integrated model is trained per stage and now compares its GBM against
transparent baselines on a disjoint development household split. The selected
stage route and development metrics are persisted in the model metadata and
reused at runtime; a baseline selection is not replaced by a GBM merely
because the GBM is more complex. The best-evidence default remains the
reference comparison model when it is available; the integrated model is not
silently treated as superior.

### Data & export

`page_data()` contains two separate areas:

**Public context**

- fetch latest USGS NWIS observations by station ID;
- display source, site, retrieval URL, and observations;
- explain that USGS values are not purifier telemetry or filter-life labels.

**Data boundary and downloads**

- export telemetry, predictions, alerts, maintenance, and experiments as CSV;
- export household configuration as JSON;
- optionally include simulator truth only for demo analysis;
- upload daily purifier telemetry with explicit replacement-day metadata.

Uploaded data is rejected when required columns, row count, or replacement
metadata are invalid. It is never silently repaired into a successful
forecast.

## Next.js 3D frontend: every surface

The optional browser application is in [`web/`](web/). Its entry point is
[`web/app/page.tsx`](web/app/page.tsx), the API adapter is
[`web/lib/api.ts`](web/lib/api.ts), and the procedural scene is
[`web/components/PurifierScene.tsx`](web/components/PurifierScene.tsx).

### Browser startup state

On load, the browser requests the household list. It then:

1. selects the first household;
2. requests its dashboard and alerts in parallel;
3. shows loading, error, empty, or populated state;
4. allows a synthetic demo household to be created when none exists.

If the FastAPI URL is missing or unreachable, the frontend shows an error
instead of substituting fake forecasts.

### Browser navigation

- **Dashboard** — 3D purifier, selectable cartridges, forecast readout,
  confidence band, and stage cards.
- **Filter detail** — the selected stage's forecast and honest interval state.
- **Alerts** — alert list with acknowledgement actions.
- **Planner** — current maintenance windows and statuses.
- **Validation** — starts a fresh validation run and shows its JSON result.
- **Models** — explains reference and integrated model roles.
- **Data** — shows source, data health, missingness, stale sensors, exports
  boundary, and the no-water-safety disclaimer.

Carbon is displayed without a fabricated interval because the simulator has no
direct carbon sensor. The UI labels it as a usage-rule forecast.

### 3D scene behavior

`PurifierScene.tsx` uses procedural Three.js geometry:

- transparent purifier housing;
- three selectable cartridge groups;
- floating/wear color driven by API condition and sensor health;
- flow particles;
- camera and orbit controls;
- a text fallback when WebGL is unavailable.

The scene is dynamically loaded with server-side rendering disabled. It is
decorative and explanatory; all meaningful values remain available as HTML
text for accessibility and reduced-motion users.

## Data contracts and important fields

### Household request

`POST /households` accepts a name, simulator scenario, wear law, seed,
starting day, calibration flag, numeric overrides, and optional user faults.
The response contains the household ID, source, scenario, and current day.

### Forecast response

A stage forecast contains:

- `stage_label`;
- `status`;
- `point_days`, possibly `null`;
- `interval_days`, possibly `null`;
- `reliability`;
- `reason`;
- `recommendation`;
- optional drivers;
- sensor events;
- sensor-health state;
- data-source label.

`null` is meaningful. It means the system did not issue that value. Frontends
must render it as withheld/unknown rather than converting it to zero.

### Status meanings

| Status | Meaning |
|---|---|
| `ok` | Forecast issued with the normal policy |
| `degraded` | Forecast issued but sensor/model evidence requires caution |
| `point_only` | Point estimate exists but no validated interval is available |
| `interval` | Forecast includes a calibrated interval |
| `withheld` | No forecast should be used because evidence is insufficient or blocked |

### Sensor event meanings

Sensor checks can identify:

- **stuck** — a value remains implausibly constant;
- **drift** — a slow shift is inconsistent with expected behavior;
- **spike** — an isolated abnormal value;
- **dropout** — missing values interrupt the signal;
- **TDS inversion** — a physically inconsistent input/output relationship.

Some events degrade confidence; blocking events withhold the forecast.

## Training, calibration, and evaluation rules

The project separates:

1. training households;
2. development/model-selection households;
3. calibration households;
4. final evaluation households.

Pressure-source choice and learned-model-vs-baseline choice happen on the
development split. The final evaluation households are not used to select
features, thresholds, pressure semantics, or model winners.

Conformal intervals are calibrated on separate synthetic households. Their
coverage is an empirical property of the tested distribution, not a guarantee
under distribution shift. Vacation-heavy, new-scenario, and simulator-law
changes must be reported as shift tests.

The integrated model and reference model can disagree. The registry and UI
show the selected model and comparison context rather than hiding a weaker
result. If a criterion fails, the validation table says **not all met**.

## Security, reliability, and operational limits

- Set `AQUAPREDICT_API_KEY` before exposing FastAPI beyond localhost.
- Set `AQUAPREDICT_FRONTEND_ORIGIN` to the exact browser origin.
- Set `AQUAPREDICT_RATE_LIMIT` to bound requests per client window.
- Keep artifact directories private and retain SHA-256 sidecars.
- Do not put a long-lived production secret in `NEXT_PUBLIC_API_KEY`; browser
  variables are visible to users and are suitable only for a controlled demo.
- Configure persistent storage for deployed FastAPI; the default SQLite path
  is local application storage.
- Training and validation routes are computationally long-running. A
  production deployment should move them to a durable background worker.
- The Vercel root endpoint is not the full backend. Use the Render/Fly FastAPI
  deployment for household, forecast, alert, planner, and validation routes.

## Presentation checklist

For a hackathon recording or live judging session:

1. Start on **Dashboard** and point out the synthetic-data notice.
2. Open **Filter detail** and explain condition, sensor health, reliability,
   point value, and interval separately.
3. Open **Simulator & what-if** and show a reproducible synthetic household.
4. Open **Validation lab** and run the **Hackathon demo: live fail-safe
   evidence** panel.
5. Show the fault event and the withheld/degraded response on camera.
6. Run validation using a fresh seed and show the measured table.
7. State the caveat exactly: these are synthetic algorithmic measurements, not
   real-world accuracy or water-safety certification.
8. Open **Data & export** and show that USGS context is separate from labels.

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
