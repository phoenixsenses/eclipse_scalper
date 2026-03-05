# PASSIVE_POCKET_RANKING

candidates=7 ranked=7
candidate_parse total_rows_seen=26 table_rows_seen=22 rows_with_pass_yes=7 candidates_parsed=7 candidates_unique=7 rows_skipped_missing_fields=0
fee_grid=[0.2, 0.5, 1.0] adverse_mult_grid=[1.0, 1.2, 1.5]
pass_threshold=0.330

| rank | symbol | rule | horizon | min_imb | min_int | max_spread | score | robust_core | robust_stress | pass_rate_core | pass_rate_stress | npa_core | npa_stress | attempt_fill_rate | attempts_per_min | score_raw_core | score_raw_stress | score_raw_min | stability_std_bps | best_fee_survive | insufficient_fill_rate |
|---:|---|---|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.50 | 2500 | 0.000300 | 0.000000e+00 | False | False | 41.67% | 41.67% | -1.198245e-05 | -2.046301e-05 | 62.77% | 0.06 | +0.00014102 | -0.00003211 | -0.00003211 | 0.823 | 1.00 | 0.00% |
| 2 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.50 | 3500 | 0.000300 | 0.000000e+00 | False | False | 50.00% | 41.67% | -5.419730e-05 | -6.601163e-05 | 60.98% | 0.03 | +0.00006752 | -0.00011032 | -0.00011032 | 1.195 | 1.00 | 0.00% |
| 3 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.50 | 2500 | 0.000500 | 0.000000e+00 | False | False | 41.67% | 33.33% | -3.429781e-05 | -4.215471e-05 | 57.85% | 0.07 | +0.00010344 | -0.00006955 | -0.00006955 | 0.889 | 1.00 | 0.00% |
| 4 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.40 | 2500 | 0.000300 | 0.000000e+00 | False | False | 33.33% | 25.00% | -2.678007e-05 | -3.448076e-05 | 62.17% | 0.06 | +0.00011851 | -0.00005353 | -0.00005353 | 0.917 | 1.00 | 0.00% |
| 5 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.40 | 2500 | 0.000500 | 0.000000e+00 | False | False | 41.67% | 25.00% | -3.084411e-05 | -3.820822e-05 | 58.04% | 0.08 | +0.00010616 | -0.00006662 | -0.00006662 | 0.958 | 1.00 | 0.00% |
| 6 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.20 | 2500 | 0.000500 | 0.000000e+00 | False | False | 25.00% | 25.00% | -5.244629e-05 | -5.971221e-05 | 56.77% | 0.09 | +0.00006629 | -0.00010669 | -0.00010669 | 0.997 | 1.00 | 0.00% |
| 7 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.30 | 2500 | 0.000500 | 0.000000e+00 | False | False | 25.00% | 25.00% | -5.314794e-05 | -6.060280e-05 | 58.95% | 0.08 | +0.00006959 | -0.00010310 | -0.00010310 | 0.957 | 1.00 | 0.00% |

survive_fee1_passrate_ge_0.5=1
