# PASSIVE_POCKET_RANKING

candidates=7 ranked=5
candidate_parse total_rows_seen=26 table_rows_seen=22 rows_with_pass_yes=7 candidates_parsed=7 candidates_unique=7 rows_skipped_missing_fields=0
fee_grid=[1.0] adverse_mult_grid=[1.2, 1.5]
pass_threshold=0.500

| rank | symbol | rule | horizon | min_imb | min_int | max_spread | score | robust_core | robust_stress | pass_rate_core | pass_rate_stress | npa_core | npa_stress | attempt_fill_rate | attempts_per_min | score_raw_core | score_raw_stress | score_raw_min | stability_std_bps | best_fee_survive | insufficient_fill_rate |
|---:|---|---|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.50 | 2500 | 0.000500 | 0.000000e+00 | False | False | 36.67% | 36.67% | +1.545704e-05 | +1.238462e-05 | 56.59% | 0.08 | +0.00002668 | +0.00002139 | +0.00002139 | 1.292 | 1.00 | 26.67% |
| 2 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.40 | 2500 | 0.000500 | 0.000000e+00 | False | False | 33.33% | 30.00% | -2.892463e-05 | -3.233756e-05 | 55.84% | 0.09 | -0.00004833 | -0.00005397 | -0.00005397 | 1.162 | 1.00 | 20.00% |
| 3 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.40 | 2500 | 0.000300 | 0.000000e+00 | False | False | 26.67% | 26.67% | -3.132704e-05 | -3.367632e-05 | 61.08% | 0.07 | -0.00005139 | -0.00005697 | -0.00005697 | 1.012 | 1.00 | 36.67% |
| 4 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.30 | 2500 | 0.000500 | 0.000000e+00 | False | False | 23.33% | 20.00% | -2.651299e-05 | -3.032820e-05 | 57.01% | 0.09 | -0.00004998 | -0.00005588 | -0.00005588 | 1.041 | 1.00 | 6.67% |
| 5 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.20 | 2500 | 0.000500 | 0.000000e+00 | False | False | 26.67% | 20.00% | -2.726776e-05 | -3.128499e-05 | 54.76% | 0.10 | -0.00004918 | -0.00005642 | -0.00005642 | 0.956 | 1.00 | 6.67% |

survive_fee1_passrate_ge_0.5=0
