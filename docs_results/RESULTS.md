# Measured results - reference model v3 (ALL SYNTHETIC; algorithmic feasibility only, not real-world accuracy)

Design/tuning used households from seed 0 (train 125 / calibration 50 / test 75). The final numbers below are on
**80 fresh households (seed 999, 1,084 filter lives, 155k forecasts) that were never used for training, calibration or
design**, run through the official harness. Metrics, thresholds and the horizon were fixed beforehand in `eval/config.py`
and were not changed after seeing results.

## Headline (fresh households)
| | Volume rule | 20% health | Trend | v1 (original model) | **v3 (this version)** |
|---|---|---|---|---|---|
| Median abs window error, days (<=60 d horizon) | 35.7 | 32.2 | 15.9 | 12.4 | **12.0** |
| 90th-percentile abs error, days | 119 | 76 | 60 | 60 | **48** |
| Late-warning rate | 48% | 38% | 23% | 5.3% | 11.5% |
| Premature (false-alarm) rate | 39% | 50% | 29% | 46% | 34% |
| Interval coverage (target 90%) | - | - | - | 89.3% | 89.2% |
| Median interval width near-term, days | - | - | - | 81.6 | 54.9 |

Window error by stage (days): sediment 8.3 (trend 10.5), RO 7.9 (trend 13.5), **carbon 27.0 (= volume rule; no gain)**.
Success criteria for v3 (scored with the **stricter** rules described below): (1) lower error than all three baselines, paired on the same
opportunities **PASS** (vs trend -3.7 d, 95% CI -4.7 to -2.4); (2) coverage near 90% without very wide intervals **FAIL** (coverage 89.2% fine,
median width 54.9 d > preset 40 d cap); (3) fewer late warnings than the volume rule at the pre-set alert threshold with a comparable
false-alarm rate **PASS** (late 11.5% vs 48.1%, premature 34.3% vs 39.2%).
Under the same strict rules **v1 (the original model) FAILS criterion 3** (premature 46.3% > 39.2% + 5%). It previously "passed" only because the
old rule let the best alert threshold be picked on the test data.

### How the criteria were tightened (Person 1 audit)
- Criterion 1 is now **paired**: baselines are re-scored on exactly the opportunities where the model issued a forecast, so a model cannot win by
  withholding its hard cases (unit-tested with a cherry-picking model).
- Criterion 3 is judged at the **pre-set** alert threshold; the old "best threshold on the operating curve" is kept only as a supplementary number.

## What each upgrade bought (ablation, seed-0 test split, stride-4 quick metrics; median abs error in days)
| Step | Overall | Sediment | Carbon | RO |
|---|---|---|---|---|
| 1 original | 12.76 | 9.51 | 31.43 | 8.38 |
| 2 + per-litre / source-change features | 12.52 | 9.29 | 32.98 | 8.23 |
| 3 + model selection (GBM only where it beats the baseline) | 11.63 | 9.29 | **25.00** | 8.23 |
| 4 + scale-adaptive conformal | 11.63 | 9.29 | 25.00 | 8.23 |
| 5 + instability history + hide useless intervals (v3) | **11.33** | 8.96 | 25.00 | 8.02 |
Model selection was the largest single gain (it routes carbon to the volume rule). Features helped a little. Adaptive conformal improved
coverage slightly (sediment 0.858 -> 0.889) but did not narrow intervals.

## Honest caveats
- **v3 is not strictly better than v1.** It has better and tighter point forecasts, fewer premature alerts, and a smaller p90 error,
  but MORE late warnings (11.5% vs 5.3%): v1's very wide lower bounds made it alert early. That is a trade-off, not a free win.
- **The width improvement is mostly from hiding carbon's interval** (v1 showed ~368-day intervals for carbon). Sediment widths are unchanged
  (61.6 vs 62.3 d) and RO improved modestly (45 vs 51 d). Near-term coverage (forecasts within 60 d) is ~87%, below the 90% target.
- **I expected the width to come from water-source instability and it does not**: stable 'normal' households get intervals as wide as unstable
  ones (sediment 75 d vs 39 d for source_change). The width is the relative (x/÷ ~2.2) interval plus heavy-tailed errors (vacation gaps, sensor noise).
  The preset 40-day cap may simply be unattainable at 90% coverage in this simulator; it was not loosened.
- **Carbon cannot be forecast better than the volume rule without information the simulator does not expose** (no sensor signal). The model says so
  (selection picks the baseline, interval hidden). `AquaPredict_prevcycles` (previous filters' lifetimes) halves carbon error and looks much better
  overall. **Correction to an earlier note:** I said this works "only because" the simulator replaces filters exactly at the criterion. That was wrong:
  the correlation between a filter's lifetime and the next one stays high with late replacement (0.88 at 0 days delay, 0.85 at 25 days, 0.78 at 60,
  0.65 at 120) because household wear and water-source regimes persist across filters in the simulator. Whether real households behave this way is
  unknown, so the variant is diagnostic only and should not be reported as a result.
- Gating withheld <1% of near-term forecasts and left overall metrics unchanged; it matters for safety, not for these averages.
- Sensor rules detect stuck sensors, outages and TDS inversion (100% once they last >= 4-10 days, 0 false flags on clean synthetic data; real quantised
  sensors will be worse). **Slow drift is still not detected.**
- Both simulators are ours. Nothing here shows real-world accuracy.

## Benchmark realism audit (Person 1)
The default simulator over-weights very short-lived sediment filters: `gradual_pressure_rise` (uncapped particulate ramp) gives ~23 sediment lives per
household (median ~16 days) and supplied 28% of pooled sediment cycles from 8% of households; `source_change` gives ~13 lives per household. All results above
use this default (unchanged so they stay reproducible). `SimConfig.calibrated()` caps the ramp (2x) and compounded source-change particulates (3x):
`gradual_pressure_rise` median life 16 -> 73 d, share of lives < 21 d 56% -> 22%, and filter lives too short to forecast fall from 246 to 6 (5 seeds x 120 households).
**Baseline conclusions survive on the calibrated benchmark** (median error, days, 95% household-bootstrap CI): trend 15.5 (14.0-17.0) -> 16.3 (14.7-18.0);
20% health 29.4 -> 27.9; volume rule 36.0 -> 32.7. The reference model has NOT been re-trained or re-evaluated on the calibrated benchmark; use `--calibrated`
and re-run `make reference` before quoting final model numbers.

## Integration: Person 2's real model scored by Person 1's strict harness (ALL SYNTHETIC)
Setup: Person 1's *calibrated* simulator, 160 households split 80 train / 32 calibration / 48 test. Person 2's model is **re-trained** on Person 1's
training households (pooled across stages), calibrated on the separate calibration households, and its real `predict()` (validation, sensor faults,
gating, CUSUM, forecast, interval gate) is scored on the 48 test households with the same paired / preset-threshold criteria as everything else.
Window error = median |error| in days for forecasts within 60 days of the criterion.

| Run | Person 2 median err | Trend | 20% health | Volume rule | Person 2 late / premature | Coverage / median width |
|---|---|---|---|---|---|---|
| seed 21, pressure = inlet - drop | 59.1 | 16.0 | 25.0 | 33.0 | 1.1% / 71% | 87.3% / 196.5 d |
| seed 21, pressure = raw drop (`dp`) | 30.0 | 16.0 | 25.0 | 33.0 | 1.7% / 68% | 88.6% / 105.8 d |
| **seed 33 (fresh), pressure = raw drop** | **37.0** | 13.7 | 27.0 | 30.0 | 2.0% / 71.6% | 91.2% / 135.4 d |

By stage on the fresh seed (median |error|, days; Person 2 vs trend vs volume rule): sediment 32.4 vs 10.3 vs 28.0; RO 21.7 vs 11.4 vs 43.8;
carbon 60.7 vs 22.0 vs 22.0. Seed 21 with the stand-in model for reference: stand-in 12.9 days (passes criteria 1 and 3, fails the width criterion).
Person 2 withheld 0.1% (seed 21) and 2.1% (seed 33) of forecasts, mostly "conditions outside the model's operating range".

**Verdict: Person 2's model, as integrated, FAILS all three success criteria in every run.** On the fresh seed it does not beat even the volume rule overall.

What this does and does not show:
- **The pressure mapping was worth a factor of two** (59 -> 30 days on the same households). The raw sediment pressure drop is the default now. That choice was made
  after seeing seed-21 held-out results (two real options, large effect), which is why the fresh-seed run exists. The seed-21 figure of 30.0 (just below the volume
  rule's 33.0) did **not** replicate: 37.0 vs 30.0 on seed 33.
- **A consistent positive bias**: Person 2 over-predicts remaining life in every stage and both seeds (median bias about +32 d sediment, +21 d RO, +61 d carbon).
  I have not diagnosed the cause. Untested leads: its features were designed for its own simulator (pressure that falls with load, a capacity-based criterion),
  and it has no per-stage "use the baseline where the model does not beat it" step, which would have routed carbon to the volume rule.
- **Carbon is the worst stage** (nothing in Person 1's sensors tracks carbon wear); an interval is still shown at ~135 days wide.
- This table scores **forecast accuracy only**. Person 2's sensor-fault events, CUSUM source-change flag, SHAP drivers and drift monitor are not evaluated here;
  their own test-suite covers behaviour on their simulator. It is also not a verdict on Person 2's work on its own simulator (see its README), only on how
  well it transfers to Person 1's simulator under the stated pressure-channel assumption.
- Both simulators are ours. Nothing here shows real-world accuracy.
