# PASSIVE_POCKET_RANKING

candidates=7 ranked=7
candidate_parse total_rows_seen=26 table_rows_seen=22 rows_with_pass_yes=7 candidates_parsed=7 candidates_unique=7 rows_skipped_missing_fields=0
fee_grid=[0.0] adverse_mult_grid=[1.0]
pass_threshold=0.330
mitigation_profile=anti_adverse_v3 gate_config min_intensity_strong=0.000000 min_imbalance_strong=0.000000 max_spread_tight=0.000000 max_volatility_extreme=None vol_quantile_reject=0.010000

| rank | symbol | rule | horizon | min_imb | min_int | max_spread | score | robust_core | robust_stress | pass_rate_core | pass_rate_stress | npa_core | npa_stress | attempt_fill_rate | attempts_per_min | score_raw_core | score_raw_stress | score_raw_min | baseline_pass_rate_core | baseline_pass_rate_stress | baseline_npa_core | baseline_score_raw_core | delta_pass_rate_core | delta_npa_core | delta_score_raw_core | failure_reason_top | stability_std_bps | best_fee_survive | insufficient_fill_rate |
|---:|---|---|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|
| 1 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.50 | 2500 | 0.000500 | 0.000000e+00 | False | False | 50.00% | 50.00% | -4.363656e-05 | -4.363656e-05 | 56.58% | 0.06 | -0.00007774 | -0.00007774 | -0.00007774 | 50.00% | 50.00% | -4.169072e-05 | -0.00007463 | +0.00% | -1.945836e-06 | -0.00000311 | adverse_dominates | 0.493 | 0.00 | 0.00% |
| 2 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.40 | 2500 | 0.000500 | 0.000000e+00 | False | False | 50.00% | 50.00% | -4.774080e-05 | -4.774080e-05 | 57.74% | 0.06 | -0.00007946 | -0.00007946 | -0.00007946 | 50.00% | 50.00% | -3.955650e-05 | -0.00006570 | +0.00% | -8.184307e-06 | -0.00001376 | adverse_dominates | 0.429 | 0.00 | 0.00% |
| 3 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.30 | 2500 | 0.000500 | 0.000000e+00 | False | False | 50.00% | 50.00% | -5.113911e-05 | -5.113911e-05 | 58.20% | 0.07 | -0.00009506 | -0.00009506 | -0.00009506 | 50.00% | 50.00% | -5.951670e-05 | -0.00010027 | +0.00% | +8.377589e-06 | +0.00000522 | adverse_dominates | 0.392 | 0.00 | 0.00% |
| 4 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.20 | 2500 | 0.000500 | 0.000000e+00 | False | False | 50.00% | 50.00% | -5.228793e-05 | -5.228793e-05 | 55.84% | 0.07 | -0.00009465 | -0.00009465 | -0.00009465 | 50.00% | 50.00% | -5.086040e-05 | -0.00009067 | +0.00% | -1.427527e-06 | -0.00000398 | adverse_dominates | 0.378 | 0.00 | 0.00% |
| 5 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.40 | 2500 | 0.000300 | 0.000000e+00 | False | False | 50.00% | 50.00% | -5.311228e-05 | -5.311228e-05 | 61.72% | 0.05 | -0.00008094 | -0.00008094 | -0.00008094 | 50.00% | 50.00% | -4.746533e-05 | -0.00007255 | +0.00% | -5.646943e-06 | -0.00000840 | adverse_dominates | 0.500 | 0.00 | 0.00% |
| 6 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.50 | 2500 | 0.000300 | 0.000000e+00 | False | False | 50.00% | 50.00% | -5.580043e-05 | -5.580043e-05 | 60.36% | 0.04 | -0.00008705 | -0.00008705 | -0.00008705 | 50.00% | 50.00% | -5.015082e-05 | -0.00007774 | +0.00% | -5.649606e-06 | -0.00000931 | adverse_dominates | 0.627 | 0.00 | 0.00% |
| 7 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.50 | 3500 | 0.000300 | 0.000000e+00 | False | False | 44.44% | 44.44% | -1.012504e-04 | -1.012504e-04 | 61.72% | 0.02 | -0.00016205 | -0.00016205 | -0.00016205 | 44.44% | 44.44% | -9.491996e-05 | -0.00015844 | +0.00% | -6.330431e-06 | -0.00000361 | adverse_dominates | 1.396 | 0.00 | 0.00% |

survive_fee1_passrate_ge_0.5=0
