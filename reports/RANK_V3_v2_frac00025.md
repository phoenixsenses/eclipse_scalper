# PASSIVE_POCKET_RANKING

candidates=7 ranked=7
candidate_parse total_rows_seen=26 table_rows_seen=22 rows_with_pass_yes=7 candidates_parsed=7 candidates_unique=7 rows_skipped_missing_fields=0
fee_grid=[0.2, 0.5, 1.0] adverse_mult_grid=[1.0, 1.2, 1.5]
pass_threshold=0.330

| rank | symbol | rule | horizon | min_imb | min_int | max_spread | score | robust_core | robust_stress | pass_rate_core | pass_rate_stress | npa_core | npa_stress | attempt_fill_rate | attempts_per_min | score_raw_core | score_raw_stress | score_raw_min | baseline_pass_rate_core | baseline_pass_rate_stress | baseline_npa_core | baseline_score_raw_core | delta_pass_rate_core | delta_npa_core | delta_score_raw_core | stability_std_bps | best_fee_survive | insufficient_fill_rate |
|---:|---|---|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.50 | 2500 | 0.000300 | 0.000000e+00 | False | False | 33.33% | 33.33% | -6.555703e-05 | -7.309937e-05 | 60.51% | 0.04 | +0.00003364 | -0.00014086 | -0.00014086 | 33.33% | 33.33% | -6.555703e-05 | +0.00003364 | +0.00% | +0.000000e+00 | +0.00000000 | 0.902 | 1.00 | 50.00% |
| 2 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.50 | 2500 | 0.000500 | 0.000000e+00 | False | False | 33.33% | 33.33% | -6.737665e-05 | -7.409556e-05 | 55.21% | 0.05 | +0.00003061 | -0.00014190 | -0.00014190 | 33.33% | 33.33% | -6.737665e-05 | +0.00003061 | +0.00% | +0.000000e+00 | +0.00000000 | 0.693 | 1.00 | 50.00% |
| 3 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.40 | 2500 | 0.000500 | 0.000000e+00 | False | False | 27.78% | 27.78% | -5.708044e-05 | -6.380354e-05 | 55.37% | 0.06 | +0.00005050 | -0.00012215 | -0.00012215 | 27.78% | 27.78% | -5.708044e-05 | +0.00005050 | +0.00% | +0.000000e+00 | +0.00000000 | 0.952 | 1.00 | 50.00% |
| 4 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.40 | 2500 | 0.000300 | 0.000000e+00 | False | False | 27.78% | 27.78% | -5.885440e-05 | -6.587654e-05 | 60.35% | 0.05 | +0.00005083 | -0.00012254 | -0.00012254 | 27.78% | 27.78% | -5.885440e-05 | +0.00005083 | +0.00% | +0.000000e+00 | +0.00000000 | 0.948 | 1.00 | 50.00% |
| 5 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.50 | 3500 | 0.000300 | 0.000000e+00 | False | False | 27.78% | 27.78% | -9.743910e-05 | -1.080539e-04 | 59.10% | 0.02 | -0.00000427 | -0.00018244 | -0.00018244 | 27.78% | 27.78% | -9.743910e-05 | -0.00000427 | +0.00% | +0.000000e+00 | +0.00000000 | 1.424 | 1.00 | 50.00% |
| 6 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.20 | 2500 | 0.000500 | 0.000000e+00 | False | False | 16.67% | 16.67% | -8.866907e-05 | -9.539592e-05 | 53.02% | 0.06 | -0.00000908 | -0.00018190 | -0.00018190 | 16.67% | 16.67% | -8.866907e-05 | -0.00000908 | +0.00% | +0.000000e+00 | +0.00000000 | 1.015 | 1.00 | 16.67% |
| 7 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.30 | 2500 | 0.000500 | 0.000000e+00 | False | False | 16.67% | 16.67% | -9.846200e-05 | -1.054425e-04 | 54.88% | 0.06 | -0.00001646 | -0.00018895 | -0.00018895 | 16.67% | 16.67% | -9.846200e-05 | -0.00001646 | +0.00% | +0.000000e+00 | +0.00000000 | 1.034 | 1.00 | 33.33% |

survive_fee1_passrate_ge_0.5=0
