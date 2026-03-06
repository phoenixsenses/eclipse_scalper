# PASSIVE_POCKET_RANKING

candidates=7 ranked=7
candidate_parse total_rows_seen=26 table_rows_seen=22 rows_with_pass_yes=7 candidates_parsed=7 candidates_unique=7 rows_skipped_missing_fields=0
fee_grid=[0.0] adverse_mult_grid=[1.0]
pass_threshold=0.330
mitigation_profile=baseline gate_config min_intensity_strong=0.000000 min_imbalance_strong=0.000000 max_spread_tight=0.000000 max_volatility_extreme=None vol_quantile_reject=0.010000

| rank | symbol | rule | horizon | min_imb | min_int | max_spread | score | robust_core | robust_stress | pass_rate_core | pass_rate_stress | npa_core | npa_stress | attempt_fill_rate | attempts_per_min | score_raw_core | score_raw_stress | score_raw_min | baseline_pass_rate_core | baseline_pass_rate_stress | baseline_npa_core | baseline_score_raw_core | delta_pass_rate_core | delta_npa_core | delta_score_raw_core | failure_reason_top | stability_std_bps | best_fee_survive | insufficient_fill_rate |
|---:|---|---|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|
| 1 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.40 | 2500 | 0.000300 | 0.000000e+00 | False | False | 50.00% | 50.00% | -1.537155e-05 | -1.537155e-05 | 59.53% | 0.05 | -0.00002705 | -0.00002705 | -0.00002705 | 50.00% | 50.00% | -1.537155e-05 | -0.00002705 | +0.00% | +0.000000e+00 | +0.00000000 | adverse_dominates | 0.515 | 0.00 | 0.00% |
| 2 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.50 | 2500 | 0.000500 | 0.000000e+00 | False | False | 50.00% | 50.00% | -1.879321e-05 | -1.879321e-05 | 55.51% | 0.06 | -0.00003510 | -0.00003510 | -0.00003510 | 50.00% | 50.00% | -1.879321e-05 | -0.00003510 | +0.00% | +0.000000e+00 | +0.00000000 | adverse_dominates | 0.557 | 0.00 | 0.00% |
| 3 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.40 | 2500 | 0.000500 | 0.000000e+00 | False | False | 50.00% | 50.00% | -2.967889e-05 | -2.967889e-05 | 55.97% | 0.06 | -0.00005451 | -0.00005451 | -0.00005451 | 50.00% | 50.00% | -2.967889e-05 | -0.00005451 | +0.00% | +0.000000e+00 | +0.00000000 | adverse_dominates | 0.545 | 0.00 | 0.00% |
| 4 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.50 | 2500 | 0.000300 | 0.000000e+00 | False | False | 50.00% | 50.00% | -3.364785e-05 | -3.364785e-05 | 58.92% | 0.05 | -0.00005386 | -0.00005386 | -0.00005386 | 50.00% | 50.00% | -3.364785e-05 | -0.00005386 | +0.00% | +0.000000e+00 | +0.00000000 | adverse_dominates | 0.554 | 0.00 | 0.00% |
| 5 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.20 | 2500 | 0.000500 | 0.000000e+00 | False | False | 50.00% | 50.00% | -3.706349e-05 | -3.706349e-05 | 55.30% | 0.07 | -0.00006789 | -0.00006789 | -0.00006789 | 50.00% | 50.00% | -3.706349e-05 | -0.00006789 | +0.00% | +0.000000e+00 | +0.00000000 | adverse_dominates | 0.468 | 0.00 | 0.00% |
| 6 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.30 | 2500 | 0.000500 | 0.000000e+00 | False | False | 50.00% | 50.00% | -4.085055e-05 | -4.085055e-05 | 56.73% | 0.07 | -0.00007116 | -0.00007116 | -0.00007116 | 50.00% | 50.00% | -4.085055e-05 | -0.00007116 | +0.00% | +0.000000e+00 | +0.00000000 | adverse_dominates | 0.489 | 0.00 | 0.00% |
| 7 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.50 | 3500 | 0.000300 | 0.000000e+00 | False | False | 44.44% | 44.44% | -1.071846e-05 | -1.071846e-05 | 62.39% | 0.02 | -0.00001827 | -0.00001827 | -0.00001827 | 44.44% | 44.44% | -1.071846e-05 | -0.00001827 | +0.00% | +0.000000e+00 | +0.00000000 | adverse_dominates | 1.203 | 0.00 | 0.00% |

survive_fee1_passrate_ge_0.5=0
