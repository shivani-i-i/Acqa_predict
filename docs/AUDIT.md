# AquaPredict Phase 1 Audit

**Scope:** audit only. This document is the only file changed in Phase 1; no
application or test code was changed. All data and all experiments below are
synthetic. None of these results is real-world purifier accuracy, and USGS
observations are not filter-life labels.

## Executive summary

The integrated `aquapredict/integration` model is not a safe default for the
Person 1 benchmark. The reported fresh-seed result is 37.0 median window-error
days versus 13.7 for trend, with 71.6% premature alerts, 2.0% late warnings,
91.2% coverage and 135.4-day median interval width. It fails all three stated
success criteria. The reference model is a useful comparison, not evidence of
real-world accuracy.

The strongest verified causes and contributors are:

1. `training_frame()` removes every life that has not crossed its criterion.
   In an 80-household calibrated seed-21 audit run, the training split had
   40/268 sediment, 40/200 carbon and 40/155 RO cycles censored and omitted.
2. The integrated model is one pooled model across stages. A small,
   non-official held-out experiment using the same train/test household split
   found isolated per-stage models materially better for sediment (median AE
   41.0 vs 55.4 days) and RO (35.7 vs 46.2 days); carbon was effectively
   unchanged (52.3 vs 55.2 days). This is diagnostic evidence, not a result
   suitable for publication.
3. Pressure semantics are different. In Person 1, raw sediment pressure drop
   rises with wear; downstream pressure falls because inlet variation dominates.
   The adapter documents that `dp` was selected after seeing seed-21 held-out
   results, so the choice is test-contaminated as a model-selection procedure.
4. Carbon has no wear sensor in Person 1. The observed carbon target is a
   hidden organic-load process; the honest policy is the usage/volume rule,
   with no learned interval presented as if it were sensor-supported.
5. The public predictor gates at three readings, while the app registry and
   harness policy use 14 days. This is a contract/policy mismatch.
6. Split-conformal calibration is not shift robust by construction. In a
   separate Person 2-simulator experiment, coverage was 89.1% on a normal
   held-out draw and 82.9% after changing the replacement criterion from 80%
   to 100% of capacity. The current tests check calculation and same-family
   behavior, not scenario-wise coverage under shift or vacation-heavy homes.

## Folder-by-folder audit

| Area | Purpose | Bugs / correctness risks | Leakage risks | Dead code / unused surface | Test gaps | Ranked improvements |
|---|---|---|---|---|---|---|
| [`aquapredict/simulator/`](../aquapredict/simulator/) | Generates Person 1 synthetic telemetry, hidden wear, faults, replacements and two wear laws. | Default benchmark overweights very short sediment lives; parameters are illustrative, not literature-derived. Forced replacement creates censored lives. | Truth columns are present in generated frames; callers must use `VISIBLE_COLS`. | No module proven dead; `sanity.py` and `validate.py` are reporting/validation utilities rather than runtime paths. | No broad property tests for both laws, censored-cycle distributions, or vacation-heavy coverage. | **P1** make one explicit EOL contract (or tabulate both); **P1** calibrated benchmark by default for reported results; **P2** stress distributions and sensor drift. |
| [`aquapredict/baselines/`](../aquapredict/baselines/) | Volume, 20%-health and trend comparison forecasters. | Rated capacities and 60 L/day are simulator assumptions; trend falls back to volume for carbon, which is correct but easy to misread as learned performance. | Baselines use current-cycle visible history, but benchmark ratings are close to simulated medians and may flatter them. | `_cycle`, `_fit` and helpers are shared by service indicators; no dead function established. | No tests for policy sensitivity, stage-specific fallback reporting, or alternative EOL definitions. | **P1** document baseline assumptions beside every result; **P2** make carbon fallback explicit in every metric table. |
| [`aquapredict/eval/`](../aquapredict/eval/) | Household splits, past-only harness, metrics, criteria, stress, multi-seed reports and figures. | Two EOL definitions and two simulator families are compared in different places; the harness excludes short lives and reports them separately. Criteria are fixed, but supplementary operating curves can invite threshold cherry-picking. | Harness correctly supplies visible past rows; model-selection code must never consume test households. | `figures.py`, `report.py`, `make_all.py` are batch/report utilities; no unused runtime path proven. | Stress suite does not assert conformal coverage under each shift; no end-to-end test that pressure selection is development-only. | **P1** one split/criterion configuration and a single honest table; **P1** development-only selection assertions; **P2** shift and bootstrap coverage tests. |
| [`aquapredict/planner/`](../aquapredict/planner/) | Illustrative replacement policy and cost calculations. | Costs and prices are placeholders; forecast uncertainty is converted to windows without a calibrated asymmetric action cost. | No truth access in planner logic, but app can show simulator truth in demo views. | Small and used by service validation/planner paths. | No tests for cost sensitivity under withheld/carbon policy or late-vs-early asymmetric costs. | **P2** use the same alert/action loss as selection; preserve illustrative labels. |
| [`aquapredict/forecasting/`](../aquapredict/forecasting/) | Person 2 features, GBM, conformal intervals, public `predict()`, sensor checks, CUSUM, drift, persistence and demo simulator. | Integrated transfer model over-predicts; pooled stages and simulator mismatch are core risks. Carbon interval is currently shown despite no sensor signal. Slow drift is explicitly not detected. Artifact loading has no SHA-256 integrity verification. | Feature code is leakage-aware, but `training_frame()` removes censored lives and target distribution is simulator-specific. Conformal exchangeability is assumed, not verified under shift. | SHAP is optional; demo/update scripts are not production paths. No proven dead module. | No slow-drift rule test; no artifact tamper test; no coverage tests for new scenarios, law changes, or vacation-heavy data. | **P1** per-stage training and development-only selection; **P1** carbon volume route/no interval; **P1** integrity sidecars; **P2** drift and shift-robustness tests. |
| [`aquapredict/integration/`](../aquapredict/integration/) | Maps Person 1 telemetry to Person 2 schema, trains on Person 1 and wraps real `predict()` for the harness. | `training_frame()` drops uncensored/censored lives; model is pooled across stages; `dp` was selected after held-out seed-21 results. Person 2's pressure and EOL semantics differ from Person 1's. | Household disjointness is checked, and histories are past-only, but pressure selection procedure was test-informed. | `inlet` pressure option is diagnostic only and not used by the app; retain only if clearly marked as a research comparison. | No test that trained target includes a known censored-policy audit; no selection provenance test; no per-stage-vs-pooled regression test. | **P1** train/select per stage on development households; **P1** record pressure choice before test; **P1** align/tabulate EOL; **P2** remove or isolate diagnostic options. |
| [`aquapredict/reference_model/`](../aquapredict/reference_model/) | Person 1 stand-in reference forecaster with model selection, adaptive conformal intervals and gating. | It is stronger than integrated Person 2 on the reported benchmark but still synthetic-only; interval width criterion fails in reported v3 results. Previous-cycle diagnostic features can exploit persistent simulator regimes and are not real-world evidence. | Grouped splits and calibration are implemented; using `variant()` cache is safe only for immutable harness slices. | `ablation.py` and `run_reference.py` are research/report commands, not app runtime. Do not delete without approval. | No calibrated-benchmark re-evaluation in shipped results; no shift coverage test. | **P1** keep as explicit comparator; re-run on the chosen benchmark; **P2** isolate diagnostic previous-cycle variant. |
| [`aquapredict/checks/`](../aquapredict/checks/) | Person 1 sensor rules for data quality, plus shared validation helpers. | Slow drift is not detected; quantized real sensors may false-positive on exact flat-line checks. | Checks should see visible telemetry only; no target leakage observed. | Some checks are parallel to forecasting checks (`checks/sensor_rules.py` vs `forecasting/checks/sensor_faults.py`), creating policy duplication. | Missing slow drift, quantization tolerance, and cross-check parity are untested. | **P1** add slow-drift rule and focused tests; **P2** consolidate or explicitly distinguish both rule sets. |
| [`aquapredict/app/api.py`](../aquapredict/app/api.py) | FastAPI local REST surface. | No authentication; `/models/train` and `/validation/run` run synchronously; `/telemetry/ingest` has no request/row/byte cap; alert action is a free-form path parameter. | `include_truth` is exposed for synthetic demo data and must never be enabled for uploaded data; current service guards source, but public exposure needs auth/authorization. | All routes are registered; no dead route proven. | No auth/rate-limit/size-limit/background-job tests; no enum route rejection test beyond service behavior. | **P0** env API key/auth and rate limit; **P0** body/row limits; **P1** background jobs; **P1** typed alert action enum. |
| [`aquapredict/app/service.py`](../aquapredict/app/service.py) | Shared business logic for API/UI, persistence, forecasting, alerts, planner and validation. | Very large module with mixed responsibilities; long training/validation calls block request/UI; alert action validates string late. | Past-only reads are guarded; simulator truth is demo-only. Uploaded telemetry has no truth, but the same service exposes many simulator controls. | No obvious dead path; module is a maintenance hotspot rather than dead code. | Integration coverage is broad, but no concurrency, malformed large upload, or job cancellation tests. | **P1** split into households/forecasts/alerts/plans/validation; **P1** background-job boundary; **P2** explicit authorization by source. |
| [`aquapredict/app/db.py`](../aquapredict/app/db.py) | SQLite schema, locking and JSON persistence. | Single SQLite file/connection is not suitable for concurrent/network deployment; schema has no migrations or foreign keys. | No direct leakage; data isolation is application-enforced. | `close()` exists but lifecycle ownership is not centralized. | No concurrent writer/read tests, migration tests, or Postgres compatibility tests. | **P1** document local-only deployment or add Postgres option via env; **P2** migrations/foreign keys and connection lifecycle. |
| [`aquapredict/app/registry.py`](../aquapredict/app/registry.py) | Model cards, artifact paths, training and wrapper selection. | Loads pickle/joblib artifacts without integrity check; training is synchronous; default model can change based on file existence, not an evaluated deployment decision. | Training seed is refused by validation, but artifact provenance is only metadata and not cryptographically bound. | `os`, `pickle`, and joblib paths are all intentional; no dead registry entry proven. | No tamper/integrity test, partial-training recovery test, or concurrent training test. | **P0** SHA-256 sidecars and verification; **P1** background jobs; **P1** explicit model version/provenance. |
| [`aquapredict/app/ui/`](../aquapredict/app/ui/) | Single Streamlit dashboard for simulator, forecasts, alerts, planner, validation, models and uploads. | One ~2,000-line page is difficult to test/maintain; heavy calls are not cached with `st.cache_data`; UI can show a forecast/interval for carbon unless policy is changed. | UI correctly labels synthetic truth and USGS context, but truth controls require strict source separation. | No page module separation means no isolated dead-page analysis. | Headless AppTest exists, but browser layout is explicitly unreviewed; no performance/cache tests. | **P1** split pages; **P1** cache heavy immutable calls; **P1** carbon policy/UI wording; **P2** browser smoke test. |
| [`api/`](../api/) | Vercel serverless public entrypoint. | Serves only `/`, `/health`, and `/sources/usgs/observations`; root README's broader API link can imply more. CORS is wildcard, though only GET methods are allowed. | USGS is correctly source-labelled and not used as filter truth. | Full local API is intentionally not imported due to bundle limits; this is a deployment boundary, not dead code. | No deployed endpoint contract test and no test that forecast routes are intentionally absent. | **P1** either deploy full API elsewhere or reword README; **P2** add deployment smoke check and restrictive CORS if browser clients are known. |
| [`tests/`](../tests/) | Unit/integration/UI tests for simulator, evaluation, forecasting, integration and app. | Strong contract coverage but tests encode current mismatches (3-reading Person 2 gate, 14-day registry gate) without a single policy assertion. | Several tests explicitly enforce no-peeking and disjoint households; artifact consistency tests depend on tracked JSON artifacts. | No test module is proven dead; slow model/app fixtures are mixed into normal pytest. | Missing security, integrity, rate/size, async job, slow-drift, scenario coverage, and calibrated-reference benchmark tests. | **P0** add security/limits/integrity tests; **P1** mark slow tests and add shift tests; **P1** make one min-history config contract. |
| [`docs/`](../docs/) | Human-facing simulator, Person 1 and app documentation. | Documentation is generally honest, but README still presents the full local API beside a Vercel endpoint that only serves health/USGS; result mapping and limitations are not one page. | Docs consistently state synthetic-only and separate USGS context; retain that boundary in all future edits. | No dead docs identified; `PERSON1.md`, `SIMULATOR.md`, `APP.md` are active references. | No documentation consistency test for API surface, Person labels, or all three criteria. | **P1** update README deployment wording and add honest criteria table/diagram/screenshot; **P2** rename Person labels after approval. |

## Requested investigations and evidence

### 1. Why the integrated model over-predicts remaining life

**Censored lives.** `integration/adapter.py:74-105` only appends a cycle when
`dtc == 0`; uncrossed cycles are silently skipped. In the isolated seed-21
calibrated run:

| split | stage | cycles | censored | retained |
|---|---:|---:|---:|---:|
| train | sediment | 268 | 40 | 228 |
| train | carbon | 200 | 40 | 160 |
| train | ro | 155 | 40 | 115 |

This is not necessarily a bug if the estimator is deliberately trained only
on completed lives, but it is a selection problem: slower/low-wear households
and cycles disappear. The model then learns conditional behavior among
crossers, while the test includes the full household population. Censoring
must be represented or explicitly quantified before fitting.

**Pooled stages.** The current `ForecastModel.fit()` receives a single frame
with `filter_stage` as a categorical feature. A quick diagnostic trained
separate models per stage and compared them on untouched test households:

| stage | per-stage median AE | pooled median AE | per-stage median bias | pooled median bias |
|---|---:|---:|---:|---:|
| sediment | 41.0 d | 55.4 d | +25.3 d | +34.3 d |
| carbon | 52.3 d | 55.2 d | +16.6 d | +21.9 d |
| ro | 35.7 d | 46.2 d | +4.4 d | +2.7 d |

These are row-level quick diagnostics, not the official harness metrics, and
were not used to tune a reported test result. They support per-stage training
as a high-priority experiment.

**Pressure semantics.** Person 1's simulator computes sediment pressure drop
from the wear ratio (`simulator/household.py:153-162`), while the adapter
documents that Person 2's pressure falls with load and offers `dp`,
`downstream`, and `inlet` mappings (`integration/adapter.py:10-45`). A fresh
40-household diagnostic found:

| law | median change in raw `dp` (late - early) | median change in downstream pressure |
|---|---:|---:|
| A | +0.083 bar | -0.079 bar |
| B | +0.090 bar | -0.087 bar |

The mapping is therefore a real semantic choice, not a harmless rename.
`docs_results/RESULTS.md` and the adapter state that `dp` was selected after
seed-21 held-out results. That choice must be made on a development split,
frozen, and evaluated once on test.

**Missing per-stage fallback.** Person 1's reference model explicitly selects
the baseline where GBM does not win (`reference_model/model.py:105-145`), and
the result report says this routes carbon to volume. The integrated path
trains one GBM and has no corresponding per-stage baseline selection
(`integration/train.py:44-59`). Carbon therefore receives a learned point and
interval despite no carbon wear sensor, producing the observed positive bias.

### 2. Pressure-source selection must use development data

This is a process defect, not merely a metric defect. The adapter's own
commentary records the held-out seed-21 result that motivated `dp`, and the
fresh seed-33 result in `docs_results/RESULTS.md` shows the choice did not
generalize (37.0 days integrated versus 30.0 on seed 21). Correct procedure:
train on train households, compare pressure mappings on development
households, freeze the mapping, calibrate separately, and score test once.
The test set must not choose among `downstream`, `dp`, or `inlet`.

### 3. Carbon has no sensor signal

Person 1's simulator documents carbon as hidden organic load with no sensor
signal (`docs/SIMULATOR.md:20-25`). The actual loop increments carbon load from
usage and organic factors, while visible sensors are generated from sediment
and RO quantities (`simulator/household.py:109-162`). A fresh 60-household
Spearman diagnostic, using all visible rows and the carbon `true_dtc`, found
weak correlations: usage -0.256, feed TDS -0.126, output TDS -0.139,
sediment pressure drop -0.083, flow +0.036, inlet pressure -0.013 and
temperature -0.054. These are associations in synthetic data, not proof of
independence, but there is no direct carbon wear measurement.

**Policy decision for Phase 2:** route carbon to the volume rule, hide the
conformal interval, label it “usage-based estimate only,” and do not claim
sensor-driven carbon accuracy. Previous-cycle lifetime features remain a
diagnostic simulator artifact, not a real-world claim.

### 4. Minimum-history gate mismatch

`forecasting/checks/gating.py:35-55` defaults to three rows and `predict()`
uses it. The registry uses `MIN_HISTORY_DAYS = 14`
(`app/registry.py:18`), and the evaluation harness uses `EvalConfig.min_age_days
= 14` (`eval/config.py:8`). A direct check returned `withheld` at two rows and
`ok` at three rows for the public gate, while the app wrappers still withhold
before 14 days. This is a real policy mismatch, even though the harness avoids
the earliest opportunities. Phase 2 should define one configuration source and
keep the public `predict()` response shape stable.

### 5. Two simulators and two end-of-life definitions

Person 1 replaces sediment at a pressure-drop criterion, RO at flux/rejection,
and carbon at hidden load (`docs/SIMULATOR.md:16-25`). Person 2 replaces every
stage when a volume-based capacity fraction is reached; its default is 80%
consumed / 20% proxy health (`forecasting/examples/generate_data.py:8-15`,
`:47-70`). Person 2 sensors also make pressure and flow decline with a hidden
load fraction, unlike Person 1's sediment drop channel. These are not
independent physical validations. Reports must name the simulator, law,
criterion and calibration population, or the numbers are not comparable.

### 6. Conformal coverage under distribution shift

The forecasting README correctly states that split conformal assumes
exchangeability and can degrade under new hardware, water or seasons
(`forecasting/README.md`, “Prediction intervals”). The shipped tests verify
the quantile, calibration sufficiency, stage/range gates and a generic
coverage report, but do not evaluate coverage by stress scenario.

An isolated Person 2-simulator experiment trained/calibrated on separate
households gave:

| evaluation distribution | coverage | median width | interval availability |
|---|---:|---:|---:|
| normal, same replacement criterion | 89.1% | 45.0 d | 99.6% |
| changed criterion, 100% capacity | 82.9% | 53.9 d | 99.3% |

The repository's stress variants cover law B, missingness, injected drift and
source shocks (`eval/stress.py`), but not coverage assertions for those
variants and not vacation-heavy homes. Person 1 vacation days are explicitly
no-reading periods (`simulator/household.py:24-35`); this needs a dedicated
coverage stress split, not row-level random splitting.

### 7. API risks

`app/api.py` has no authentication middleware. `POST /models/train` calls
training synchronously (`app/api.py:68-73`), and `POST /validation/run` calls
the full harness synchronously (`app/api.py:184-186`). `IngestIn.rows` has a
minimum length but no maximum (`app/schemas.py:64-67`), and service ingestion
materializes the entire list as a DataFrame (`app/service.py:206-235`).
`alert_action` takes an unconstrained `action: str` path parameter
(`app/api.py:176-178`), even though service validation later rejects unknown
actions. The SQLite backend is explicitly single-file/local-only
(`app/db.py:1`, `docs/APP.md:65-66`).

Phase 2 should add env-based API-key authentication, request/row/byte limits,
background job records and polling, rate limiting, a typed action enum, and an
env-selectable Postgres implementation.

### 8. Pickle/joblib integrity

`forecasting/models/persistence.py:52-77` calls `joblib.load()` after format,
feature-schema and scikit-learn-version checks. `app/registry.py:196-207` calls
`pickle.load()` for the reference model. These checks do not authenticate file
contents; a modified pickle can execute code on load. The README warning to
load only trusted local artifacts is not an integrity control. Phase 2 should
write a SHA-256 sidecar at save time and verify it before any load, with an
explicit error on missing or mismatched sidecars.

### 9. Vercel entrypoint scope

`api/index.py:4-8` explicitly limits deployment to lightweight health and USGS
context endpoints. The root route advertises only `/health` and
`/sources/usgs/observations` (`api/index.py:31-39`). The full FastAPI service
remains local. This is internally consistent with the Vercel file, but the
README's public API wording can lead users to expect household/forecast
routes. Either deploy the full API on a suitable service or clearly describe
Vercel as the public health/USGS endpoint only.

### 10. Tracked artifact and `.gitignore` discrepancy

In the current checkout, both
`aquapredict/forecasting/artifacts/metrics.json` and
`aquapredict/forecasting/artifacts/interval_selection_dev.json` are tracked by
Git, and the visible `.gitignore` ignores only `*.joblib` under that directory,
not either JSON file. The test
`tests/forecasting/test_intervals.py:43-51` reads
`interval_selection_dev.json`, and
`tests/forecasting/test_artifact_consistency.py` reads `metrics.json`.
Therefore the requested “ignored but tracked” condition is **not reproduced in
this checkout**; the files are tracked and not ignored by the current rules.
The underlying fragility remains: tests depend on generated JSON artifacts
whose contents must stay synchronized with the README. If a different branch
or global exclude contains the old ignore rules, remove that rule or make the
fixture generation explicit; do not silently delete the artifacts.

## Phase 1 conclusions and approval gates

No Phase 2 implementation should begin until the following are approved:

1. Whether to model censoring explicitly or report completed-life training as a
   restricted estimand.
2. Per-stage training and development-only model/pressure selection.
3. Carbon volume-rule policy with hidden interval and explicit UI wording.
4. One minimum-history policy and one EOL definition, or a side-by-side table
   for both definitions.
5. Whether the reference model remains as a comparison (no deletion requested).

The integrated model currently fails all three success criteria. Any later
improvement must be evaluated on a new seed with train/development/calibration
decisions frozen before test scoring; if a criterion still fails, the result
must say so plainly.
