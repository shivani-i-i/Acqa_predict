# AquaPredict evaluation report
*Model version:* ref-v3(features=v3,select=True,adaptive=True,prev=False,alpha=0.1,seed=0)  
*Held-out households:* 80  *Forecast opportunities:* 155,412  *Filter lives scored:* 1,084 (*excluded as too short to forecast with >= 7 days notice, all methods:* 226)
SYNTHETIC law A (default simulator config), fresh seed 999; scored with the stricter paired / preset-threshold criteria

## Definitions (fixed beforehand)
```
Metric definitions (exact formulas in each docstring).

Opportunity = one forecast for one filter at one day. Cycle = one filter life that reached its criterion.
  window error      = point - true_days_to_criterion  (we report |error|, median)  [days]
  primary subset    = opportunities with true remaining life <= cfg.primary_horizon_days
  coverage          = share of ISSUED opportunities with lo <= true <= hi
  width             = hi - lo  [days]
  alert day (cycle) = first checked day with alert_value <= cfg.lead_target_days (withheld never alerts)
  lead              = crossing_day - alert_day
  LATE warning      = no alert at all, or lead < cfg.min_notice_days
  PREMATURE alarm   = alert exists and lead > cfg.premature_days   (this is the 'false alarm')
```
Config: `{'stride': 7, 'min_age_days': 14, 'lead_target_days': 21.0, 'min_notice_days': 7.0, 'premature_days': 45.0, 'primary_horizon_days': 60.0, 'target_coverage': 0.9, 'coverage_tol': 0.05, 'max_median_width_days': 40.0, 'false_alarm_tol': 0.05, 'alert_uses_lower_bound': True}`

## Window error
| method | n_issued | n_primary | median_abs_err_primary | mean_abs_err_primary | p90_abs_err_primary | median_bias_primary | median_abs_err_all | withheld_rate |
|---|---|---|---|---|---|---|---|---|
| AquaPredict | 25789 | 8374 | 12.000 | 20.785 | 48.128 | 1.083 | 30.833 | 0.004 |
| AquaPredict_nogate | 25902 | 8405 | 12.023 | 20.909 | 48.754 | 1.107 | 30.876 | 0.000 |
| health_20pct | 25902 | 8405 | 32.200 | 38.218 | 75.747 | -11.000 | 75.235 | 0.000 |
| trend | 25902 | 8405 | 15.888 | 30.510 | 60.289 | 1.974 | 47.000 | 0.000 |
| v1_original | 25789 | 8374 | 12.408 | 23.553 | 59.797 | 6.893 | 30.654 | 0.004 |
| volume_rule | 25902 | 8405 | 35.648 | 50.828 | 119.059 | -4.000 | 76.544 | 0.000 |

## Late / premature warnings
| method | n_cycles | late_rate | premature_rate | never_alerted_rate | median_lead_days |
|---|---|---|---|---|---|
| AquaPredict | 1084 | 0.115 | 0.343 | 0.095 | 37.000 |
| AquaPredict_nogate | 1084 | 0.114 | 0.351 | 0.094 | 37.000 |
| health_20pct | 1084 | 0.379 | 0.504 | 0.354 | 86.000 |
| trend | 1084 | 0.232 | 0.289 | 0.131 | 25.500 |
| v1_original | 1084 | 0.053 | 0.463 | 0.034 | 44.000 |
| volume_rule | 1084 | 0.481 | 0.392 | 0.450 | 74.000 |

## Window error by stage
| method | stage | n_issued | n_primary | median_abs_err_primary | mean_abs_err_primary | p90_abs_err_primary | median_bias_primary | median_abs_err_all | withheld_rate |
|---|---|---|---|---|---|---|---|---|---|
| AquaPredict | carbon | 8888 | 2905 | 27.000 | 33.678 | 69.628 | -10.000 | 55.330 | 0.001 |
| AquaPredict | ro | 8161 | 2036 | 7.910 | 11.711 | 27.012 | 2.356 | 22.533 | 0.007 |
| AquaPredict | sediment | 8740 | 3433 | 8.290 | 15.255 | 32.496 | 3.537 | 20.785 | 0.005 |
| AquaPredict_nogate | carbon | 8897 | 2909 | 27.000 | 33.674 | 69.512 | -10.000 | 55.348 | 0.000 |
| AquaPredict_nogate | ro | 8217 | 2043 | 7.867 | 11.693 | 26.977 | 2.332 | 22.730 | 0.000 |
| AquaPredict_nogate | sediment | 8788 | 3453 | 8.329 | 15.607 | 33.442 | 3.599 | 20.923 | 0.000 |
| health_20pct | carbon | 8897 | 2909 | 27.938 | 28.427 | 53.000 | -19.000 | 60.000 | 0.000 |
| health_20pct | ro | 8217 | 2043 | 35.000 | 42.642 | 88.594 | -9.000 | 88.990 | 0.000 |
| health_20pct | sediment | 8788 | 3453 | 36.167 | 43.850 | 96.613 | -4.000 | 81.000 | 0.000 |
| trend | carbon | 8897 | 2909 | 27.000 | 33.674 | 69.512 | -10.000 | 55.348 | 0.000 |
| trend | ro | 8217 | 2043 | 13.499 | 25.136 | 52.375 | 1.499 | 59.888 | 0.000 |
| trend | sediment | 8788 | 3453 | 10.450 | 31.024 | 57.158 | 6.657 | 30.246 | 0.000 |
| v1_original | carbon | 8888 | 2905 | 30.988 | 39.954 | 87.403 | 27.384 | 53.644 | 0.001 |
| v1_original | ro | 8161 | 2036 | 7.694 | 11.793 | 27.399 | 2.339 | 22.859 | 0.007 |
| v1_original | sediment | 8740 | 3433 | 8.544 | 16.649 | 36.513 | 3.871 | 21.188 | 0.005 |
| volume_rule | carbon | 8897 | 2909 | 27.000 | 33.674 | 69.512 | -10.000 | 55.348 | 0.000 |
| volume_rule | ro | 8217 | 2043 | 41.028 | 62.102 | 143.985 | -2.000 | 85.296 | 0.000 |
| volume_rule | sediment | 8788 | 3453 | 43.000 | 58.610 | 137.183 | 11.374 | 88.000 | 0.000 |

## Late warnings by scenario
| method | scenario | n_cycles | late_rate | premature_rate | never_alerted_rate | median_lead_days |
|---|---|---|---|---|---|---|
| AquaPredict | accelerated_wear | 184 | 0.288 | 0.174 | 0.266 | 31.000 |
| AquaPredict | gradual_pressure_rise | 161 | 0.000 | 0.255 | 0.000 | 29.000 |
| AquaPredict | high_usage | 157 | 0.038 | 0.395 | 0.006 | 40.000 |
| AquaPredict | missing_readings | 74 | 0.068 | 0.419 | 0.068 | 41.000 |
| AquaPredict | normal | 271 | 0.107 | 0.376 | 0.074 | 40.000 |
| AquaPredict | source_change | 82 | 0.098 | 0.390 | 0.098 | 42.000 |
| AquaPredict | stuck_sensor | 37 | 0.216 | 0.378 | 0.108 | 36.000 |
| AquaPredict | sudden_sensor_drift | 85 | 0.106 | 0.459 | 0.106 | 46.000 |
| AquaPredict | unusual_operating_conditions | 33 | 0.212 | 0.576 | 0.212 | 61.500 |
| AquaPredict_nogate | accelerated_wear | 184 | 0.288 | 0.174 | 0.266 | 31.000 |
| AquaPredict_nogate | gradual_pressure_rise | 161 | 0.000 | 0.261 | 0.000 | 30.000 |
| AquaPredict_nogate | high_usage | 157 | 0.038 | 0.408 | 0.006 | 40.000 |
| AquaPredict_nogate | missing_readings | 74 | 0.054 | 0.486 | 0.054 | 46.500 |
| AquaPredict_nogate | normal | 271 | 0.107 | 0.376 | 0.074 | 40.000 |
| AquaPredict_nogate | source_change | 82 | 0.098 | 0.390 | 0.098 | 42.000 |
| AquaPredict_nogate | stuck_sensor | 37 | 0.216 | 0.378 | 0.108 | 36.000 |
| AquaPredict_nogate | sudden_sensor_drift | 85 | 0.106 | 0.459 | 0.106 | 46.000 |
| AquaPredict_nogate | unusual_operating_conditions | 33 | 0.212 | 0.576 | 0.212 | 61.500 |
| health_20pct | accelerated_wear | 184 | 0.679 | 0.255 | 0.641 | 66.500 |
| health_20pct | gradual_pressure_rise | 161 | 0.590 | 0.354 | 0.565 | 120.500 |
| health_20pct | high_usage | 157 | 0.166 | 0.675 | 0.121 | 61.000 |
| health_20pct | missing_readings | 74 | 0.270 | 0.568 | 0.230 | 97.000 |
| health_20pct | normal | 271 | 0.251 | 0.609 | 0.244 | 86.000 |
| health_20pct | source_change | 82 | 0.585 | 0.256 | 0.549 | 68.000 |
| health_20pct | stuck_sensor | 37 | 0.000 | 0.676 | 0.000 | 97.000 |
| health_20pct | sudden_sensor_drift | 85 | 0.212 | 0.718 | 0.200 | 143.500 |
| health_20pct | unusual_operating_conditions | 33 | 0.333 | 0.667 | 0.333 | 110.000 |
| trend | accelerated_wear | 184 | 0.364 | 0.114 | 0.283 | 20.000 |
| trend | gradual_pressure_rise | 161 | 0.180 | 0.248 | 0.062 | 17.000 |
| trend | high_usage | 157 | 0.146 | 0.376 | 0.019 | 28.000 |
| trend | missing_readings | 74 | 0.257 | 0.392 | 0.149 | 34.000 |
| trend | normal | 271 | 0.229 | 0.292 | 0.118 | 27.000 |
| trend | source_change | 82 | 0.244 | 0.232 | 0.146 | 22.000 |
| trend | stuck_sensor | 37 | 0.243 | 0.216 | 0.135 | 29.000 |
| trend | sudden_sensor_drift | 85 | 0.165 | 0.471 | 0.118 | 49.000 |
| trend | unusual_operating_conditions | 33 | 0.273 | 0.545 | 0.212 | 63.000 |
| v1_original | accelerated_wear | 184 | 0.125 | 0.277 | 0.049 | 33.000 |
| v1_original | gradual_pressure_rise | 161 | 0.025 | 0.311 | 0.025 | 33.000 |
| v1_original | high_usage | 157 | 0.013 | 0.510 | 0.013 | 46.000 |
| v1_original | missing_readings | 74 | 0.054 | 0.405 | 0.054 | 40.000 |
| v1_original | normal | 271 | 0.041 | 0.561 | 0.022 | 52.000 |
| v1_original | source_change | 82 | 0.049 | 0.549 | 0.049 | 47.000 |
| v1_original | stuck_sensor | 37 | 0.027 | 0.676 | 0.027 | 62.500 |
| v1_original | sudden_sensor_drift | 85 | 0.082 | 0.529 | 0.071 | 54.000 |
| v1_original | unusual_operating_conditions | 33 | 0.030 | 0.727 | 0.030 | 58.000 |
| volume_rule | accelerated_wear | 184 | 0.750 | 0.168 | 0.739 | 72.500 |
| volume_rule | gradual_pressure_rise | 161 | 0.621 | 0.261 | 0.615 | 78.500 |
| volume_rule | high_usage | 157 | 0.229 | 0.529 | 0.191 | 50.000 |
| volume_rule | missing_readings | 74 | 0.473 | 0.446 | 0.473 | 142.000 |
| volume_rule | normal | 271 | 0.387 | 0.472 | 0.343 | 73.000 |
| volume_rule | source_change | 82 | 0.720 | 0.220 | 0.671 | 156.000 |
| volume_rule | stuck_sensor | 37 | 0.216 | 0.541 | 0.108 | 83.000 |
| volume_rule | sudden_sensor_drift | 85 | 0.329 | 0.600 | 0.294 | 134.500 |
| volume_rule | unusual_operating_conditions | 33 | 0.364 | 0.576 | 0.333 | 72.000 |

## Prediction intervals
| method | n_with_interval | coverage | coverage_primary | median_width_all | median_width_primary |
|---|---|---|---|---|---|
| AquaPredict | 16901 | 0.892 | 0.867 | 133.977 | 54.876 |
| AquaPredict_nogate | 17005 | 0.892 | 0.867 | 134.500 | 54.934 |
| v1_original | 25789 | 0.893 | 0.845 | 250.482 | 81.588 |

## Success criteria
- **1_window_error**: PASS - {'passed': True, 'model_median_abs_err': 12.0, 'baselines': {'volume_rule': 35.41928018840516, 'health_20pct': 32.06068794092178, 'trend': 15.73336525261266}, 'baselines_unpaired': {'volume_rule': 35.64786782948721, 'health_20pct': 32.19996243206424, 'trend': 15.887884672727637}, 'model_withheld_rate': 0.00436259748281986, 'bootstrap_diff_vs_baseline': {'volume_rule': {'n': 8374, 'diff': -23.41928018840516, 'ci95': (-28.723200698999705, -18.29672611822788)}, 'health_20pct': {'n': 8374, 'diff': -20.06068794092178, 'ci95': (-24.05242697212059, -16.26213911016905)}, 'trend': {'n': 8374, 'diff': -3.7333652526126606, 'ci95': (-4.727856427265193, -2.3923690432536837)}}, 'rule': 'model median |window error| < each baseline re-scored on the SAME opportunities (paired), primary horizon'}
- **2_interval_coverage**: FAIL - {'passed': False, 'coverage': 0.8923732323531152, 'median_width_primary_days': 54.87552300321321, 'rule': '|coverage-0.9|<=0.05 and median width<=40.0d'}
- **3_late_warnings**: PASS - {'passed': True, 'volume_rule_late': 0.4806273062730627, 'volume_rule_premature': 0.39206642066420666, 'model_late': 0.11531365313653137, 'model_premature': 0.34317343173431736, 'supplementary_model_late_at_matched_false_alarm_best_threshold': 0.09686346863468635, 'rule': "at the pre-set threshold: model late rate < volume rule AND model premature rate <= volume's + tolerance"}

**All criteria met: False**

> All results come from an internally developed simulator. They establish algorithmic feasibility, NOT real-world accuracy. Prediction intervals are not guarantees. Filter-wear prediction does not assess water safety.