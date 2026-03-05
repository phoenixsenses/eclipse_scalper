# PASSIVE_POCKET_RANKING

candidates=7 ranked=7
candidate_parse total_rows_seen=26 table_rows_seen=22 rows_with_pass_yes=7 candidates_parsed=7 candidates_unique=7 rows_skipped_missing_fields=0
fee_grid=[1.0] adverse_mult_grid=[1.2, 1.5]
pass_threshold=0.330

| rank | symbol | rule | horizon | min_imb | min_int | max_spread | score | robust_core | robust_stress | pass_rate_core | pass_rate_stress | npa_core | npa_stress | attempt_fill_rate | attempts_per_min | score_raw_core | score_raw_stress | score_raw_min | stability_std_bps | best_fee_survive | insufficient_fill_rate |
|---:|---|---|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.50 | 3500 | 0.000300 | 0.000000e+00 | False | False | 41.67% | 41.67% | -4.910019e-05 | -5.624696e-05 | 64.55% | 0.03 | -0.00008041 | -0.00009149 | -0.00009149 | 0.918 | 1.00 | 0.00% |
| 2 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.50 | 2500 | 0.000300 | 0.000000e+00 | False | False | 33.33% | 25.00% | -5.077750e-05 | -5.578791e-05 | 63.44% | 0.06 | -0.00008109 | -0.00008890 | -0.00008890 | 0.572 | 1.00 | 0.00% |
| 3 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.50 | 2500 | 0.000500 | 0.000000e+00 | False | False | 25.00% | 25.00% | -6.020578e-05 | -6.454094e-05 | 58.28% | 0.07 | -0.00010519 | -0.00011276 | -0.00011276 | 0.626 | 1.00 | 0.00% |
| 4 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.40 | 2500 | 0.000500 | 0.000000e+00 | False | False | 8.33% | 8.33% | -7.460786e-05 | -7.905577e-05 | 58.35% | 0.07 | -0.00012954 | -0.00013726 | -0.00013726 | 0.677 | 1.00 | 0.00% |
| 5 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.40 | 2500 | 0.000300 | 0.000000e+00 | False | False | 8.33% | 0.00% | -7.489448e-05 | -7.962653e-05 | 62.04% | 0.06 | -0.00011799 | -0.00012516 | -0.00012516 | 0.683 | 1.00 | 0.00% |
| 6 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.30 | 2500 | 0.000500 | 0.000000e+00 | False | False | 0.00% | 0.00% | -7.760358e-05 | -8.226628e-05 | 58.89% | 0.08 | -0.00013184 | -0.00013942 | -0.00013942 | 0.520 | 0.00 | 0.00% |
| 7 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.20 | 2500 | 0.000500 | 0.000000e+00 | False | False | 0.00% | 0.00% | -8.323223e-05 | -8.778470e-05 | 56.98% | 0.09 | -0.00014111 | -0.00014883 | -0.00014883 | 0.487 | 0.00 | 0.00% |

survive_fee1_passrate_ge_0.5=0
