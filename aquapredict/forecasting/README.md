# AquaPredict - Person 2: Forecasting and Sensor Intelligence

**Atomberg Innovation Challenge - Track 2: Water Purifiers**

> ## IMPORTANT DISCLOSURE
> Everything in this repository runs on **synthetic data produced by a simulator**
> (`examples/generate_data.py`). There is **no real Atomberg purifier telemetry** here, and the
> simulator is **not a validated physical model**. Every metric, interval and forecast shown below
> is computed on synthetic households and says **nothing** about accuracy on real purifiers.
> This prototype is **not validated for real filter-maintenance decisions**. Every threshold
> (fault rules, CUSUM, drift, bounds) is a placeholder that must be re-tuned on real equipment data.

## 1. What this is

A self-contained Python module that, for one filter stage of one household, takes a history of sensor
readings and returns:

* a **forecast** of remaining days until replacement,
* a **prediction interval** (split conformal) - or an explicit reason why none is given,
* the **top drivers** of the forecast (SHAP, optional),
* **sensor fault events** (missing, stale, flat-line, spike, TDS inconsistency, ...),
* a **possible source-change** flag (CUSUM on input TDS),
* a **status** (`ok` / `degraded` / `withheld`) and a human-readable **reason**.

It deliberately **abstains** (returns no number) when inputs are unreliable or outside what the model
was trained on.

## 2. Folder structure

```
AquaPredict_Person2/
|-- predict.py                  PUBLIC ENTRY POINT: predict(sensor_data) -> dict
|-- features/
|   `-- engineering.py          leakage-safe features + household-level 3- or 4-way split
|-- models/
|   |-- forecast.py             HistGradientBoosting pipeline, 3 baselines, error + alert metrics
|   |-- intervals.py            split conformal intervals (pooled/per-stage, absolute/scaled) + support check
|   |-- explain.py              optional SHAP explanations with an additivity check
|   `-- persistence.py          joblib save/load with schema + version checks
|-- checks/
|   |-- sensor_faults.py        rule-based fault events (info / warning / critical)
|   |-- source_change.py        two-sided CUSUM (possible source / operating-condition change)
|   |-- gating.py               decides: forecast? interval? (3 distinct questions)
|   `-- drift.py                input drift, error monitoring, interval-coverage monitoring
|-- examples/
|   |-- generate_data.py        deterministic SYNTHETIC household simulator (+ CSV writer, demo split)
|   |-- investigate_intervals.py  chooses the interval variant on DEVELOPMENT households only
|   |-- train_demo.py           train + calibrate + evaluate (test set used once) + save artifacts
|   |-- run_demo.py             six predict() scenarios -> examples/sample_prediction.json
|   |-- verify_shap.py          checks the REAL shap library against the trained model
|   `-- sample_prediction.json  output of run_demo.py (synthetic; produced in the build sandbox)
|-- tests/                      pytest suite (no internet, no real data)
|-- artifacts/                  metrics.json, interval_selection_dev.json (shipped); model file is created locally
|-- requirements.txt
|-- pytest.ini
`-- README.md
```

## 3. Setup on Windows (VS Code + PowerShell)

Requires **Python 3.11 - 3.13** (64-bit) from python.org with "Add python.exe to PATH" ticked.
Tested here with Python 3.13.17, numpy 2.5.3, pandas 3.0.5, scikit-learn 1.9.1, scipy 1.18.1,
joblib 1.6.0, pytest 9.1.1 (pandas 2.x / older scikit-learn are allowed by `requirements.txt` but were
**not** tested).

1. Unzip `AquaPredict_Person2.zip`, then in VS Code: **File > Open Folder...** and pick `AquaPredict_Person2`.
2. Open a terminal (**Terminal > New Terminal**, PowerShell) and run:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell refuses to activate the venv ("running scripts is disabled"):
`Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` and activate again.
In VS Code choose the interpreter: **Ctrl+Shift+P > Python: Select Interpreter > .venv**.

3. **Optional - SHAP explanations:**

```powershell
pip install shap
```

The project works fully without it (explanations are then reported as unavailable). If `pip install shap`
fails on your machine, skip it; nothing else breaks.

## 4. Run it

All commands are run from the project root, with the venv active. **Order matters: train first.**

```powershell
python -m aquapredict.forecasting.examples.generate_data            # optional: writes examples/data/synthetic_readings.csv
python -m aquapredict.forecasting.examples.investigate_intervals    # optional: re-runs the interval-variant choice (development data only)
python -m aquapredict.forecasting.examples.train_demo               # REQUIRED before the demo: trains, calibrates, evaluates, saves the model (~10 s)
python -m aquapredict.forecasting.examples.run_demo                 # six predict() scenarios + examples/sample_prediction.json
python -m aquapredict.forecasting.examples.verify_shap              # after `pip install shap`: checks real SHAP end to end (exit 0 = passed)
python -m aquapredict.forecasting.examples.update_readme_results    # optional: refresh the generated README tables from your artifacts
python -m pytest                            # runs the whole test suite
```

`run_demo.py` needs the model file created by `train_demo.py`; if it is missing the demo stops and prints
`Run first: python -m aquapredict.forecasting.examples.train_demo`. The CSV step is only for inspecting data; the scripts build the
same deterministic dataset internally (120 households, seed 42). Options: `--households N --seed S`
(`train_demo.py` also `--coverage 0.9`; `generate_data.py` also `--replace-at-fraction`).

**Artifacts written locally by `train_demo.py`:** `artifacts/aquapredict_bundle.joblib` (fitted pipeline + feature
schema + config + conformal calibration + SHAP background sample; not shipped because pickles depend on the
scikit-learn version) and `artifacts/metrics.json`. `investigate_intervals.py` writes
`artifacts/interval_selection_dev.json`.

**Security:** a `.joblib` file is a pickle. **Only load artifacts you created or trust** - loading a malicious
file can execute arbitrary code.

### What is evaluated, and how (data discipline)

Households (never single rows) are split 45 / 20 / 15 / 20 % into **train / calibration / development / test**
(54 / 24 / 18 / 24 households with the defaults).

* **train** fits the forecaster.
* **development** is the only set used to choose between interval variants (`investigate_intervals.py`; the test
  households are deleted from memory in that script before anything is fitted). The selection rule was fixed in
  the code before looking at results.
* **calibration + development** together fit the final conformal quantile (42 households, because the quantile
  is noisy with few households); this happens after the variant choice is made.
* **test** is evaluated once, in `train_demo.py`. Nothing is tuned against it. Your re-run gives the same numbers
  with the same library versions (the pipeline is deterministic).

**Replacement criterion (simulator target):** a filter is replaced when 80 % of its nominal capacity is consumed,
i.e. proxy health = `1 - consumed/capacity` falls to **20 %**. `remaining_days` = days until that event
(`examples/generate_data.py`, `--replace-at-fraction 1.0` gives "replace at 0 % health" instead).

### Baselines (all three target the same replacement event)

| Baseline | Rule | Usage-rate assumption |
|---|---|---|
| `volume_rule` | days until `volume_rule_fraction` (0.80) of nominal capacity is used | fixed nominal 60 L/day |
| `health_threshold_proxy` | days until proxy health <= `health_replace_threshold` (**0.20**) | lifetime average usage so far |
| `trend_extrapolation` | same volume as the volume rule | most recent 7-day average usage |

`health_replace_threshold` is the **proxy-health level at which replacement is due** (0.20 = replace at 20 % health;
so cumulative use at replacement = `capacity * (1 - 0.20)`). The earlier name `health_reserve` (value 0.10) was
ambiguous and is removed. "Health" is a usage-based proxy, not a measured health percentage. Because all three use the
same replacement volume they differ only in how daily usage is estimated, and they receive the simulator's nominal
capacities but ignore the TDS/household wear multipliers that the simulator applies. They are deliberately simple.

### Results (generated from one run)

Everything numeric below is **generated** by `python -m aquapredict.forecasting.examples.update_readme_results` from
`artifacts/metrics.json` and `artifacts/interval_selection_dev.json`, so README, metrics and sample output come from the same
run (`run_id`). `train_demo.py` stamps the `run_id` into the saved model, and `run_demo.py` copies it into
`examples/sample_prediction.json`. A test checks that the README matches the shipped artifacts. If you re-run
`train_demo.py` with other library versions your `metrics.json` can differ slightly from the shipped one (the shipped one
records the versions it was produced with); run `python -m aquapredict.forecasting.examples.update_readme_results` to refresh the README from yours.

<!-- RESULTS:START run_id=94adb330a5ad -->
*Generated from `artifacts/metrics.json` (run_id `94adb330a5ad`). Environment: Python 3.13.17, numpy 2.5.3, pandas 3.0.5, scikit-learn 1.9.1, scipy 1.18.1, joblib 1.6.0, shap not installed. Other library versions can give slightly different numbers; re-running `train_demo.py` rewrites `artifacts/metrics.json` with your own.*

Households (rows): train 54 (16142), calibration 24 (6820), development 18 (4272), **test 24 (8134)**. Test filter lives: 72.

**Remaining-days error, all test rows (days)**

| Method | Median AE | MAE | RMSE | n |
|---|---|---|---|---|
| Gradient boosting (ours) | 8.8 | 19.3 | 35.2 | 8134 |
| Volume rule | 28.6 | 55.4 | 88.6 | 8134 |
| Health-threshold proxy (20 %) | 25.5 | 47.5 | 72.2 | 8131 |
| Trend extrapolation | 27.6 | 49.0 | 73.6 | 8131 |

**Remaining-days error, rows with <= 30 true days left (retrospective view, days)**

| Method | Median AE | MAE | RMSE | n |
|---|---|---|---|---|
| Gradient boosting (ours) | 3.0 | 4.7 | 7.8 | 2215 |
| Volume rule | 14.5 | 18.1 | 23.5 | 2215 |
| Health-threshold proxy (20 %) | 13.4 | 21.0 | 31.3 | 2215 |
| Trend extrapolation | 13.0 | 21.3 | 33.0 | 2215 |

**Alerting, H = 14 d** (late: no alert or lead < 7 d; false alarm: lead > 28 d)

| Method | Late | False alarm | On time | No alert (part of late) | Median lead | Late rows | False-alarm rows |
|---|---|---|---|---|---|---|---|
| Gradient boosting (ours) | 6.9 % | 2.8 % | 90.3 % | 1.4 % | 13.0 d | 6.4 % | 0.1 % |
| Volume rule | 48.6 % | 22.2 % | 29.2 % | 34.7 % | 21.0 d | 42.2 % | 15.3 % |
| Health-threshold proxy (20 %) | 48.6 % | 22.2 % | 29.2 % | 31.9 % | 19.0 d | 39.4 % | 13.7 % |
| Trend extrapolation | 45.8 % | 23.6 % | 30.6 % | 33.3 % | 19.0 d | 40.1 % | 13.5 % |

**Alerting, H = 30 d** (late: no alert or lead < 15 d; false alarm: lead > 60 d)

| Method | Late | False alarm | On time | No alert (part of late) | Median lead | Late rows | False-alarm rows |
|---|---|---|---|---|---|---|---|
| Gradient boosting (ours) | 2.8 % | 0.0 % | 97.2 % | 0.0 % | 31.0 d | 1.4 % | 0.0 % |
| Volume rule | 40.3 % | 18.1 % | 41.7 % | 12.5 % | 30.0 d | 26.6 % | 21.1 % |
| Health-threshold proxy (20 %) | 31.9 % | 16.7 % | 51.4 % | 16.7 % | 30.5 d | 24.0 % | 15.7 % |
| Trend extrapolation | 27.8 % | 18.1 % | 54.2 % | 13.9 % | 32.0 d | 24.0 % | 15.6 % |

**Prediction intervals on the test households** - method: per-stage scaled split conformal, target 90%, final calibration on 42 households (11092 rows). **The target is NOT met on these test households.**

| Group | Coverage | Mean width | Rows |
|---|---|---|---|
| **Overall** | **86.2 %** (household-bootstrap 90 % range 81.5 % - 90.8 %) | **67.7 d** | 8134 |
| stage `carbon` | 82.9 % | 45.7 d | 2682 |
| stage `ro_membrane` | 88.5 % | 98.7 d | 3561 |
| stage `sediment` | 86.4 % | 40.5 d | 1891 |
| true remaining <=30 d | 91.7 % | 20.4 d | 2215 |
| true remaining 31-90 d | 90.8 % | 49.3 d | 3169 |
| true remaining >90 d | 76.3 % | 126.9 d | 2750 |

Per-household coverage on the test households: min 59.4 %, 10th percentile 75.8 %, median 92.0 %; 83.3 % of households at or above 80%. Mean width before this change was 133.1 d.

**Interval variants compared on non-test households** (12 random household resplits of the 96-household pool; resplit sizes {'train': 54, 'calibration': 24, 'evaluation': 18}; `artifacts/interval_selection_dev.json`)

| Variant | Mean coverage | SD | Min | Max | Repeats >= target | Mean width |
|---|---|---|---|---|---|---|
| pooled_absolute | 89.5 % | 5.6 % | 78.7 % | 98.3 % | 50.0 % | 82.5 d |
| per_stage_absolute | 89.5 % | 5.9 % | 78.4 % | 98.3 % | 50.0 % | 86.9 d |
| pooled_scaled | 90.0 % | 4.8 % | 80.3 % | 97.0 % | 41.7 % | 68.1 d |
| per_stage_scaled (chosen) | 90.1 % | 4.5 % | 80.9 % | 97.7 % | 50.0 % | 70.3 d |
| pooled_scaled_level_0.93 | 92.8 % | 4.1 % | 84.2 % | 98.6 % | 75.0 % | 79.0 d |

Chosen variant, mean coverage over those resplits by stage: carbon 90.5 %, ro_membrane 89.4 %, sediment 90.8 %; by true remaining life: <=30 d 96.5 %, 31-90 d 94.2 %, >90 d 75.8 %.
<!-- RESULTS:END -->

### Late-warning and false-alarm metrics - operational definitions

Implemented in `models/forecast.py: evaluate_alerting`. Readings of each filter life (household x stage) are processed in
time order, as a daily monitor would see them. An **alert** is raised on a reading when the forecast remaining life is
<= `alert_horizon_days` (H). The **first** alert of a life defines the warning; its **lead time** is the TRUE remaining days
at that moment.

* **Late warning:** no alert at all, or lead time < `min_lead_days` (H/2). Too little notice to act.
* **False alarm (premature):** lead time > `max_early_days` (2H). Alerting this early wastes filter life / bothers the user.
* **On time:** everything else. The three rates sum to 100 % over filter lives; "no alert" is the part of "late" with no alert at all.
* Row-level companions: false-alarm rows = share of readings with more than 2H days left that raise an alert; late rows = share
  of readings with at most H/2 days left that do not.

The same definition is applied to the model and all three baselines on the same households. With only a few dozen filter
lives, one life is a couple of percentage points, so small differences are noise. The baselines' high "no alert" rate comes
from the simulator wearing filters faster or slower than their litre-based rules assume (TDS and household wear), so a
better-tuned rule would do better; this compares against simple rules on a simulator, not against real devices.

### Prediction intervals: what was investigated, and the honest conclusion

*Is the calculation wrong?* No. The quantile is the `ceil((n+1)*coverage)`-th smallest score; a test compares it with a manual
computation, and pooled absolute-residual coverage on the calibration set is >= 90 % by construction.

*Why is single-split coverage below 90 %?* Three separate effects, all measured on **non-test** households
(`examples/investigate_intervals.py`, table above):

1. **Sampling noise across households.** Rows of one household are correlated, so a split with a couple of dozen
   calibration/evaluation households behaves like a small sample. Over repeated random household resplits the chosen variants average
   about 90 % coverage, but one split can land several points either side (see SD, min, max). A single development split above
   90 %, or a single test split below it, is therefore weak evidence either way; the selection rule uses the mean over
   resplits, never one split.
2. **Stage heterogeneity.** Pooled variants over-cover `sediment`/`carbon` and under-cover `ro_membrane` on average. A per-stage
   quantile removes most of that imbalance at a small cost in width.
3. **Horizon heterogeneity.** Filters with more than 90 days left are under-covered by every variant tried, while filters near
   the end of life are over-covered. Marginal conformal coverage only holds on average over all rows, not for each horizon.

*What was changed.* Five variants (pooled / per-stage x absolute / scaled score, plus a pooled-scaled variant calibrated at a
more conservative 93 % level) were compared on the resplits. The selection rule is in the script docstring. **Disclosure:** the
rule had a first version (mean coverage only) that picked `pooled_scaled`; after seeing that `ro_membrane` was under-covered on
average the stage condition was added (rule v2), which picks `per_stage_scaled`. This was done on non-test households, but the
earlier test report of the old default had already been seen, so the single test evaluation of the final variant is not
a blind one. Nothing was tuned against the test set after that.

*Result and verdict.* See the generated interval table above. **The 90 % target is not achieved on the final test households**;
the household-bootstrap range shows how uncertain that estimate is. Both the previous and the final default come out near the
same value on these test households while averaging about 90 % over the resplits, which is consistent with this particular
household draw being harder than average rather than with a calculation fault, but that explanation is not proven.
Coverage is good near the end of filter life (the maintenance-relevant regime) and poor for filters far from replacement.
Further calibration work needed: many more calibration households (cheap in the simulator, expensive with real data),
household-grouped (cross-)conformal methods, conformal partitions by forecast horizon, conformalised quantile regression, and
online recalibration once real outcomes exist. Conformal coverage assumes calibration and future data are exchangeable and can
degrade under distribution shift.

## 5. How `predict()` works

```
sensor_data -> validate -> sensor faults -> input gate -> load model -> features
            -> MODEL-support gate -> point forecast -> INTERVAL gate -> SHAP (optional) -> response
                                \--- CUSUM source-change check runs alongside ---/
```

### Calling it (for teammates)

Copy the folder into your repo (or add it to `PYTHONPATH`), then:

```python
from aquapredict.forecasting.predict import predict

response = predict({
    "filter_stage": "ro_membrane",    # sediment | carbon | ro_membrane (as trained)
    "household_id": "H001",           # optional label
    "history": [                      # oldest -> newest; ONE household, ONE stage; start at filter install
        {"timestamp": "2026-01-01T08:00:00", "usage_litres": 52.1, "tds_in": 310.0, "tds_out": 14.2,
         "pressure": 2.6, "flow_rate": 1.4, "temperature": 27.0, "purification_mode": "ro"},
        # ... more readings; any optional sensor may be missing or None
    ],
    # optional: "now": "2026-03-01T09:00:00"   -> enables the stale-reading check
    # optional: "require_interval": True        -> withhold forecast if no justified interval
    # optional: "model_path": "path/to/aquapredict_bundle.joblib"
})
print(response["status"], response["forecast_days"], response["interval_days"], response["reason"])
```

Note: package names `features`, `models`, `checks` are top level, so if your team repo already has
modules with those names, rename the folders and fix the imports (a find/replace).

* **Required** per reading: `timestamp`, `usage_litres`. **Optional**: `tds_in`, `tds_out`, `pressure`,
  `flow_rate`, `temperature`, `purification_mode`. Absent optional sensors are never invented; they
  stay missing and the gradient-boosting model handles that natively.
* `history` must start when the **current filter was installed** (cumulative usage is a key feature).
  A truncated history underestimates it. This is an assumption of the prototype.
* Units used by the simulator and bounds: TDS in ppm, pressure in bar, flow in L/min, temperature in
  deg C, usage in litres per reading (daily in the simulator).
* `predict()` never raises for bad sensor data; it returns `status: "withheld"` with a reason.

### Response schema (always the same keys)

| Key | Meaning |
|---|---|
| `filter_stage`, `household_id` | echoed from the request |
| `forecast_days` | point forecast of remaining days (>= 0), or `null` if withheld |
| `interval_days` | `[lower, upper]` (lower >= 0), or `null` if no interval is justified |
| `drivers` | list of `{feature, value, contribution_days}`, largest effect first; empty if SHAP is unavailable |
| `explanation` | `{available, method, summary, reason, base_value_days, additivity_gap_days}` - same keys on every path; `method` names the explainer that really ran |
| `fault_events` | list of `{type, severity, message, timestamp, sensor, value}` |
| `source_change` | `{evaluated, detected, direction, alarm_index, change_index, alarm_timestamp, change_timestamp, baseline_mean, message}` |
| `status` | see below |
| `reason` | `null` when `ok`; otherwise a specific human-readable explanation |
| `disclaimer` | reminder that the model is synthetic-trained |

**Statuses**

* `ok` - forecast **and** interval issued. Non-critical warnings (spike, flat-line, an optional sensor
  missing) may still be listed in `fault_events`; they do not block.
* `degraded` - point forecast issued but the **interval is withheld** (`interval_days: null`) because the
  calibration set cannot justify one (too few calibration samples, or conditions outside the calibration
  range); `reason` says why. This is the default policy.
* `withheld` - **no forecast** (`forecast_days: null`): a required reading is missing, a critical fault
  exists, a value is invalid or out of bounds, history is too short, the stage or conditions are outside
  the model's training support, the model artifact is missing/incompatible, or (only if
  `require_interval: true`) no justified interval exists.

Fault severities: `info` (e.g. optional sensor not fitted), `warning` (suspicious, does not block),
`critical` (inputs unreliable - blocks the forecast).

### Example responses (generated from `examples/sample_prediction.json`)

<!-- SAMPLE:START run_id=94adb330a5ad -->
*Generated from `examples/sample_prediction.json` (model run_id `94adb330a5ad`); synthetic data. Scenario 1 true remaining_days (hidden from the model) = 28.0.*

Successful response (scenario `1_normal`):

```json
{
  "filter_stage": "ro_membrane",
  "household_id": "H006",
  "forecast_days": 24.0,
  "interval_days": [
    8.2,
    39.8
  ],
  "drivers": [],
  "explanation": {
    "available": false,
    "method": null,
    "summary": null,
    "reason": "shap is not installed (optional); explanations unavailable",
    "base_value_days": null,
    "additivity_gap_days": null
  },
  "fault_events": [
    {
      "type": "missing_reading",
      "severity": "warning",
      "message": "tds_out: latest reading is missing",
      "timestamp": "2026-03-01T00:00:00",
      "sensor": "tds_out",
      "value": null
    }
  ],
  "source_change": {
    "evaluated": true,
    "detected": false,
    "message": "No persistent shift detected."
  },
  "status": "ok",
  "reason": null
}
```

Withheld response (scenario `2_required_reading_missing`):

```json
{
  "filter_stage": "ro_membrane",
  "household_id": "H006",
  "forecast_days": null,
  "interval_days": null,
  "drivers": [],
  "fault_events": [
    {
      "type": "missing_reading",
      "severity": "critical",
      "message": "usage_litres: latest reading is missing (REQUIRED sensor)",
      "timestamp": "2026-03-01T00:00:00",
      "sensor": "usage_litres",
      "value": null
    },
    {
      "type": "missing_reading",
      "severity": "warning",
      "message": "tds_out: latest reading is missing",
      "timestamp": "2026-03-01T00:00:00",
      "sensor": "tds_out",
      "value": null
    }
  ],
  "status": "withheld",
  "reason": "Forecast withheld - missing_reading: usage_litres: latest reading is missing (REQUIRED sensor)"
}
```
<!-- SAMPLE:END -->

With `pip install shap`, `drivers` and `explanation` are filled in (see section 6 and `examples/verify_shap.py`).
`run_demo.py` also shows: a value outside the model's training range, a flat-lined sensor that only warns, a possible
source change, and an unknown filter stage.

## 6. Design notes (what each module guarantees)

**Features (`features/engineering.py`).** Per (household, filter stage), sorted by time: TDS rejection
`1 - tds_out/tds_in` (only when `tds_in > 0` and both valid, else NaN), cumulative usage, 7- and 30-day usage
sums, 7-day mean usage, 7-day mean input TDS, 7-day changes (current value minus mean of the *previous*
readings in the window) for TDS rejection, pressure and flow, flow/pressure ratio, days since start.
Leakage rules: `remaining_days` is never a feature; windows only look backwards (tests recompute
features on prefixes and mutate future rows); splits are by whole household; the ordinal encoder is
fitted inside the sklearn `Pipeline` on training rows only. Negative or non-finite readings become NaN
(not imputed). Missing `usage_litres` counts as 0 in sums (documented lower bound).

**Forecast (`models/forecast.py`).** `HistGradientBoostingRegressor` (no early stopping, fixed seed),
predictions clipped to >= 0, training targets validated (finite, non-negative). Three baselines (volume
rule, 20 % health-threshold proxy, trend extrapolation) with metrics computed on the same rows, plus alert-level metrics.

**Intervals (`models/intervals.py`).** Split conformal: fit on training households, scores on unseen calibration
households, quantile = the `ceil((n+1)*coverage)`-th smallest score. Default score is scaled
(`|y - f| / (f + 14 d)`, interval `[max(f - q*(f+14), 0), f + q*(f+14)]`); `scaled=False` gives the plain absolute-residual interval
`[max(f - q, 0), f + q]` and `per_stage=True` one quantile per filter stage. Withheld if fewer than `min_calibration_samples` (default 200), if the
requested coverage is impossible for `n`, if the stage has too few calibration rows, or if the row is
outside the **calibration** range. This is separate from the **model-support** check (training range),
which blocks the forecast itself. **Coverage is not guaranteed on future data**: it assumes
calibration and new data are exchangeable; rows from one household are correlated, and distribution shift
(new hardware, different water, seasons) can degrade coverage.

**Sensor faults (`checks/sensor_faults.py`).** Missing (latest reading, never-reporting sensor, high
missing rate), stale (needs `now`), invalid/out-of-bounds, flat-line (warning after 10 identical readings,
critical only after 40; a stable sensor is not automatically faulty), spikes (robust median/MAD z-score),
`tds_out` > `tds_in` beyond tolerance (critical only if repeated), flow without pressure. All thresholds
live in `SensorFaultConfig` and are placeholders.

**Source change (`checks/source_change.py`).** Two-sided standardized CUSUM with robust baseline
(median / 1.4826*MAD of the first 30 valid readings, or a supplied reference), allowance `k = 1`, threshold
`h = 8` sigma, z-clipping so one spike cannot alarm, NaN skipped, reports first alarm and an estimated change
index; `detect_all_shifts` restarts with a new baseline after each alarm. Always described as a *possible*
source or operating-condition change, never proof. A one-off simulation on synthetic Gaussian noise gave about
1.7 % false-alarm streams and ~99.5 % detection of a 25 % step (see docstring); real data will differ.

**Gating (`checks/gating.py`).** Three separate questions - inputs usable? (`gate_inputs`), inside model
training support? (`gate_model_support`), interval justified by calibration? (`gate_interval`). Interval
policy: default = issue the point forecast with `status: "degraded"`; set `require_interval: true` to
withhold instead.

**Drift (`checks/drift.py`).** PSI + KS test for input drift (needs no labels), error monitoring and
interval-coverage monitoring (need observed outcomes; "no_observed_targets" / "insufficient_data" are reported
explicitly). `monitoring_report` keeps "input drift detected" separate from "performance degradation
confirmed". This is a local batch helper, **not** a real-time monitoring service.

**SHAP (`models/explain.py`).** The estimator is scikit-learn's `HistGradientBoostingRegressor`, which `shap.TreeExplainer`
may reject or - worse - accept while returning values that do not reconstruct the prediction (it keeps a constant baseline
outside its trees and has its own missing-value / categorical routing). So three explainers are tried in turn
(`TreeExplainer`, `PermutationExplainer`, `KernelExplainer`) and **each is validated separately**: finite values, a base
value, and `base + sum(all contributions) = model output` within 0.5 days + 1 %. The first accepted one is used and named in
`method`; a result without a base value cannot be validated and is rejected; if none is accepted the response says
"unavailable" and lists what each explainer did. An earlier version stopped at the first explainer that did not raise, so a
non-additive TreeExplainer result made every explanation fail without trying the fallbacks; that is fixed. Contributions are
model associations, not proof of cause.

## 7. Tests

```powershell
python -m pytest
```

Last run in the build sandbox (no `shap` installed): **154 passed, 4 skipped**. The 4 skipped tests are the ones that use the
REAL shap library (`tests/test_explain.py`, `pytest.importorskip("shap")`); after `pip install shap` they run, and they check that
an explainer's contributions plus base value reconstruct the model prediction, that feature names are the model's input
columns, and that `predict()` returns drivers. The tests cover: leakage (prefix recomputation, mutated future rows, target never a
feature, `remaining_days` smuggled into a request is ignored), 3- and 4-way household splits (disjoint, complete, deterministic,
validated), preprocessing fitted on train only, invalid/negative targets, save -> load gives identical predictions and intervals
for every interval variant, missing/corrupt/incompatible artifacts, the conformal quantile against a manual computation
(pooled, scaled, per-stage), the code default matching the variant chosen on non-test data, interval ordering and
non-negativity, insufficient calibration, unsupported conditions, the 20 % health-threshold baseline, alert metrics against
hand-computed cases, each sensor-fault rule, CUSUM on a known change point, gating, response-schema consistency, duplicate
timestamps, mixed time zones, the SHAP explainer chain (non-additive / raising / base-less explainers are rejected and the next
one is tried; stand-ins for the shap module are used for this, which tests our logic only), and that `metrics.json`, the saved model,
the sample output and the README tables share one `run_id`. They use only generated synthetic data and need no internet.

## 8. Known limitations

* **All data is synthetic**; results do not transfer to real purifiers. The simulator is unvalidated, and its replacement
  criterion (80 % of nominal capacity consumed) is a stated assumption.
* **Real SHAP was never executed in the build sandbox** (no package-index access, so `shap` could not be installed there). The
  explainer chain is unit-tested with stand-ins and the real-shap tests are skipped there. Whether the real library produces
  additive values for this model is verified only when you run `python -m aquapredict.forecasting.examples.verify_shap` (exit 0 and
  "SHAP verification: PASSED") and `python -m pytest` with shap installed. The script prints what each explainer did, so a failure
  is diagnosable. If it fails, explanations stay "unavailable" and forecasts are unaffected.
* **The 90 % interval target is not achieved on the final test households** (see the generated table). Single-split coverage is
  noisy with few households, and filters far from replacement and some stages remain under-covered. The interval-variant
  rule was revised once after seeing non-test results (disclosed above), so the final test evaluation is not a blind one.
* The late-warning / false-alarm comparison uses a few dozen filter lives and alert thresholds chosen as a plausible definition,
  not tuned; baselines are simple litre-based rules.
* `history` must start at filter installation; the 30-reading CUSUM baseline assumes the first readings are "normal".
* Which sensors real Atomberg hardware has is unconfirmed; the code tolerates missing optional sensors but `usage_litres` is required.
* Thresholds (faults, bounds, CUSUM, PSI 0.2, gating, alert horizons) are placeholders needing real equipment data.
* The saved `.joblib` is tied to the scikit-learn version that created it; retrain after upgrading. Numbers in `metrics.json`
  can differ slightly with other library versions; the shipped file records the versions it came from.
* Not tested here: pandas 2.x, Python 3.11/3.12, a fresh virtual-environment install (no package-index access in the build
  sandbox), and running on Windows itself. The code uses only `pathlib` and ASCII output.
