# AquaPredict - Person 1 module (Simulator, Baselines, Evaluation, Cost, Exports)

All data are **synthetic**. Results show algorithmic feasibility, not real-world accuracy.

## Quick start
```bash
pip install -r requirements.txt
python -m pytest -q                                   # 65 tests
python -m aquapredict.simulator.generate --n 200 --seed 1 --law A --out data/synth_A            # add --calibrated for the recommended benchmark
python -m aquapredict.simulator.sanity --n 200 --seed 1                                         # realism/balance report
python -m aquapredict.eval.run_baselines --n 150 --seed 0 --law A --out results
python -m aquapredict.eval.make_all --seeds 5 --n 150 --out results   # pooled seeds + 95% CIs + figures (~2 min)
```

## Layout
| Path | What |
|---|---|
| `schema.py` | Shared contract (columns, units, criteria, visible-vs-truth columns, forecaster interface). **Team must agree before changing.** |
| `simulator/config.py` | Every parameter range (`SimConfig`, JSON export). Values are illustrative. |
| `simulator/wear.py` | Wear laws A (primary) and B (second simulator, different functional forms). |
| `simulator/household.py` | One household: usage, vacations, temp, pressure, source changes, hidden wear, replacement cycles, ground-truth labels, sensor layer. |
| `simulator/faults.py` | stuck, drift, spike, dropout, tds_inversion with labels. |
| `simulator/scenarios.py` | 10 scenarios + `late_replacement` + `custom_scenario()`. |
| `simulator/sanity.py`, `validate.py` | Dataset realism/balance report; telemetry schema validator (used by the harness). |
| `simulator/generate.py` | `generate_dataset(n, mix, law, seed, cfg, extra_faults)` and CLI. Household i depends only on (seed, i). |
| `baselines/baselines.py` | volume_rule, health_20pct, trend_extrapolation. |
| `eval/` | `splits` (by household), `harness` (past-only, visible columns only), `metrics`, `criteria` (3 success tests), `stress`, `report`, `run_baselines`. |
| `eval/multiseed.py`, `eval/figures.py`, `eval/make_all.py` | Pool several seeds, household-bootstrap 95% CIs, fallback rates, figures (all labelled SYNTHETIC). |
| `checks/sensor_rules.py` | Rule-based sensor checks (stuck, stale, missing, TDS inversion). Does NOT detect drift. |
| `reference_model/` | Reference forecaster v3 (GBM + model selection + adaptive conformal + gating), `run_reference` (official harness evaluation), `ablation` (what each upgrade buys). A starting point for Person 2, not a finished model. |
| `planner/cost.py` | Policy cost comparison + price sensitivity (illustrative prices). |
| `export.py` | Telemetry / config / results export with units and SYNTHETIC notice. |

## How Person 2 plugs the model in
A forecaster is `f(hist: DataFrame, stage: str) -> {"point","lo","hi","withheld","reason"}` (see `schema.py`).
```python
from aquapredict.eval.run_baselines import run
run(n=200, seed=0, forecasters={"AquaPredict": my_predict})   # prints + writes report with PASS/FAIL criteria
```
Train on `split["train"]`, calibrate on `split["calibration"]` (use `eval.splits`), never on test.
`hist` has only `schema.VISIBLE_COLS` and only rows up to the forecast day. Use `age_<stage>_d` to find the current filter cycle.

## How Person 3 uses it
`generate_household(i, scenario, seed=...)` for the simulator page and what-if (custom overrides via `custom_scenario`, faults via `extra_faults`),
`export.py` for CSV downloads, `planner.cost.policy_table/sensitivity` for the planner. Ground-truth columns drive the "compare with simulated replacement" view.

## Metric definitions (fixed beforehand, in `eval/config.py` + `eval/metrics.py`)
Window error = point - true days to criterion; primary subset = true remaining <= 60 d. LATE = no alert or < 7 days notice;
PREMATURE (false alarm) = alert > 45 days early; alert fires when (lower bound or point) <= 21 d. Filter lives too short to forecast
with >= 7 days notice (history 14 d + 7 d notice) are excluded for all methods and counted in the report. Success criterion 3 compares late rate
at matched premature rate via an operating curve.

## Audit notes (read before quoting numbers)
- Default simulator config is kept byte-identical for reproducibility, but it over-weights very short sediment lives. **Use `SimConfig.calibrated()` / `--calibrated` for final results** (see `docs/SIMULATOR.md`).
- Success criteria are strict: criterion 1 is paired on the model's issued opportunities; criterion 3 uses the pre-set alert threshold.
- `docs/SIMULATOR.md` lists every parameter and states that none is literature-derived yet.

## Known limitations
Simulator parameters, rated volumes and prices are illustrative. Rated volumes are set near median simulated life so the volume rule is not a strawman.
Both simulators are ours. Carbon has no sensor signal in the simulator, so carbon forecasts can only use volume and feed-water history.

## Reference model (start for Person 2)
`make reference` (~8 min) trains, calibrates on separate households, evaluates vs baselines and the original v1 model, and prints PASS/FAIL;
`make ablation` (~5 min) shows what each upgrade bought. Measured results and caveats: `docs_results/RESULTS.md`. Interactive callers (what-if edits)
must call `predict(hist, stage)` with the default `use_cache=False`; only `variant()` (harness) enables the feature cache.
