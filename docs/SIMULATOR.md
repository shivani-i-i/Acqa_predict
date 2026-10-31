# AquaPredict simulator: model, parameters and evidence status

**All data are synthetic.** The simulator supports algorithmic feasibility only. It is **not** a validated physical digital twin.
**No parameter below is literature-derived yet** - the abstract promises ranges "to be taken from the fouling literature [7]";
that step is still TODO. Treat every value as an *illustrative assumption*.

## Structure (one household, daily steps)
1. Exogenous drivers: usage (weekday pattern, vacations), temperature (seasonal), inlet pressure (low-pressure episodes), feed TDS with
   water-source changes (abrupt multiplicative steps that persist), particulate and organic load.
2. Hidden per-household wear multipliers (sediment, carbon, RO) and rejection sensitivity - what a model must infer.
3. Wear laws (`simulator/wear.py`), with x = hidden multiplier x accumulated load / reference load:
   | | Law A (primary) | Law B (second simulator) |
   |---|---|---|
   | Sediment dp/flow ratio | 1 + 0.8 x | 1 + 0.8 x^1.6 |
   | RO relative flux (fouling) | 1 / (1 + 3/7 x) | exp(-ln(1/0.7) x^1.2) |
   | RO rejection | rej0 - 0.09 s x^2.5 | rej0 - 0.07 s x^1.8 |
   | Temperature effect on flow | 1 + 0.03 (T-25) | exp(0.025 (T-25)) |
   | RO scaling load per litre | (feed/ref)^0.7 | (feed/ref)^1.0 |
4. Replacement criteria: sediment dp/flow >= 1.8x clean; RO flux <= 70% of clean OR rejection <= 85%; carbon load >= capacity.
   Carbon has **no sensor signal** (only volume and feed-water history can inform it).
5. Replacement: default = the day after the criterion is reached. `replacement_delay_mean_days > 0` makes households replace late; the
   filter keeps running overdue (`true_dtc` goes negative).
6. Measurement layer: noise, then injected faults (stuck, drift, spike, dropout, TDS inversion; all labelled), then random missingness.
   Idle (vacation) days have no sensor readings.

## Known limitations / findings from auditing the simulator
- **Default config over-weights short-lived filters.** `gradual_pressure_rise` (uncapped ramp) gives ~23 sediment lives per household
  (median ~16 days) and `source_change` ~13; they supply far more pooled cycles than their share of households. Use
  `SimConfig.calibrated()` (caps ramp at 2x and compounded source-change particulates at 3x) for final results. The default is kept
  unchanged because all earlier reported numbers used it.
- **Previous filter lifetimes predict the next one strongly** (correlation of previous replacement volume and next crossing volume:
  0.88 at zero delay, 0.85 at 25-day delay, 0.78 at 60 days, 0.65 at 120 days) because household wear and water-source regimes persist
  across filters in this simulator. Whether real households behave this way is unknown; do not rely on previous-cycle features for claims.
- Rated volumes used by the baselines are set near the median simulated life (not manufacturer data).
- Both wear laws are ours; agreement between A and B is not independent validation.

## Parameters (default vs `SimConfig.calibrated()`; all ILLUSTRATIVE)
| parameter | default | calibrated | meaning |
|---|---|---|---|
| `n_days` | 1000 | 1000 | days simulated per household |
| `start_date` | 2026-01-01 | 2026-01-01 | first simulated date |
| `usage_median_L` | 40.0 | 40.0 | median household water use, litres/day |
| `usage_sigma` | 0.4 | 0.4 | log-sd of household base usage |
| `usage_min_L` | 8.0 | 8.0 | lower clip of base usage |
| `usage_max_L` | 150.0 | 150.0 | upper clip of base usage |
| `weekend_factor` | 1.15 | 1.15 | weekend usage multiplier |
| `daily_usage_sigma` | 0.2 | 0.2 | day-to-day log-sd of usage |
| `vacation_rate_per_year` | 1.5 | 1.5 | vacations per year (4-14 idle days, no readings) |
| `feed_tds_median_ppm` | 350.0 | 350.0 | median household feed TDS |
| `feed_tds_sigma` | 0.6 | 0.6 | log-sd of feed TDS across households |
| `feed_tds_min_ppm` | 80.0 | 80.0 | clip |
| `feed_tds_max_ppm` | 2000.0 | 2000.0 | clip |
| `ref_tds_ppm` | 350.0 | 350.0 | TDS at which RO scaling load accrues at the reference rate |
| `source_change_rate_per_year` | 0.4 | 0.4 | water-source changes per year |
| `source_change_min` | 0 | 0 | minimum changes per household |
| `partic_coupling` | 1.5 | 1.5 | particulate load factor = TDS factor ** this at a source change |
| `temp_mean_range` | (20.0, 30.0) | (20.0, 30.0) | household mean temperature range, degC |
| `temp_amp_range` | (2.0, 6.0) | (2.0, 6.0) | seasonal amplitude range, degC |
| `inlet_mean_bar` | 2.5 | 2.5 | mean inlet pressure |
| `inlet_sd_bar` | 0.4 | 0.4 | sd across households |
| `ref_inlet_bar` | 2.5 | 2.5 | reference inlet pressure for flow scaling |
| `low_inlet_rate_per_year` | 0.0 | 0.0 | low-pressure episodes per year (5-20 days) |
| `low_inlet_factor` | 0.45 | 0.45 | inlet pressure multiplier during an episode |
| `wear_log_mean` | 0.0 | 0.0 | mean of log household wear multiplier (hidden) |
| `wear_log_sigma` | 0.35 | 0.35 | sd of log household wear multiplier (hidden) |
| `partic_sigma` | 0.5 | 0.5 | log-sd of household particulate level (hidden) |
| `organic_sigma` | 0.4 | 0.4 | log-sd of household organic level (hidden) |
| `rej_sens_sigma` | 0.2 | 0.2 | log-sd of household rejection-decline sensitivity (hidden) |
| `partic_ramp_start` | -1 | -1 | day a gradual particulate ramp starts (-1 = off) |
| `partic_ramp_per_day` | 0.0 | 0.0 | ramp slope per day |
| `partic_ramp_cap` | 1000000000.0 | 2.0 | ceiling on ramp multiplier |
| `partic_mult_cap` | 8.0 | 3.0 | ceiling on compounded source-change particulate multiplier |
| `flow0_Lpm` | 0.25 | 0.25 | clean permeate flow, L/min |
| `dp0_bar` | 0.15 | 0.15 | clean sediment pressure drop, bar |
| `rej0` | 0.96 | 0.96 | clean RO rejection |
| `noise_dp` | 0.04 | 0.04 | multiplicative sensor noise (log-sd) |
| `noise_flow` | 0.03 | 0.03 | multiplicative sensor noise (log-sd) |
| `noise_tds` | 0.03 | 0.03 | multiplicative sensor noise (log-sd) |
| `noise_feed` | 0.02 | 0.02 | multiplicative sensor noise (log-sd) |
| `noise_temp_C` | 0.3 | 0.3 | additive sensor noise, degC |
| `noise_inlet` | 0.03 | 0.03 | additive sensor noise, bar |
| `missing_rate` | 0.01 | 0.01 | random per-reading dropout probability |
| `replacement_delay_mean_days` | 0.0 | 0.0 | mean days a household replaces AFTER the criterion (0 = next day) |
| `replacement_delay_sd_days` | 10.0 | 10.0 | sd of that delay |
