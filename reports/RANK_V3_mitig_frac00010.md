# PASSIVE_POCKET_RANKING

candidates=7 ranked=7
candidate_parse total_rows_seen=26 table_rows_seen=22 rows_with_pass_yes=7 candidates_parsed=7 candidates_unique=7 rows_skipped_missing_fields=0
fee_grid=[0.2, 0.5, 1.0] adverse_mult_grid=[1.0, 1.2, 1.5]
pass_threshold=0.330

| rank | symbol | rule | horizon | min_imb | min_int | max_spread | score | robust_core | robust_stress | pass_rate_core | pass_rate_stress | npa_core | npa_stress | attempt_fill_rate | attempts_per_min | score_raw_core | score_raw_stress | score_raw_min | stability_std_bps | best_fee_survive | insufficient_fill_rate |
|---:|---|---|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.40 | 2500 | 0.000300 | 0.000000e+00 | False | False | 50.00% | 50.00% | -2.320241e-05 | -3.142300e-05 | 63.86% | 0.04 | +0.00012077 | -0.00005221 | -0.00005221 | 0.921 | 1.00 | 0.00% |
| 2 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.50 | 2500 | 0.000500 | 0.000000e+00 | False | False | 50.00% | 50.00% | -2.926983e-05 | -3.611935e-05 | 57.46% | 0.04 | +0.00011221 | -0.00005939 | -0.00005939 | 0.896 | 1.00 | 0.00% |
| 3 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.40 | 2500 | 0.000500 | 0.000000e+00 | False | False | 50.00% | 50.00% | -3.953621e-05 | -4.670064e-05 | 58.31% | 0.05 | +0.00009274 | -0.00007940 | -0.00007940 | 0.879 | 1.00 | 0.00% |
| 4 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.50 | 2500 | 0.000300 | 0.000000e+00 | False | False | 50.00% | 50.00% | -6.667240e-05 | -7.440784e-05 | 63.96% | 0.03 | +0.00004863 | -0.00012383 | -0.00012383 | 0.927 | 1.00 | 0.00% |
| 5 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.50 | 3500 | 0.000300 | 0.000000e+00 | False | False | 44.44% | 44.44% | -1.163684e-04 | -1.265369e-04 | 64.86% | 0.01 | -0.00003943 | -0.00021654 | -0.00021654 | 1.002 | 1.00 | 0.00% |
| 6 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.30 | 2500 | 0.000500 | 0.000000e+00 | False | False | 38.89% | 33.33% | -1.875589e-05 | -2.575830e-05 | 58.73% | 0.05 | +0.00012696 | -0.00004532 | -0.00004532 | 0.713 | 1.00 | 0.00% |
| 7 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.20 | 2500 | 0.000500 | 0.000000e+00 | False | False | 27.78% | 27.78% | -2.572960e-05 | -3.345249e-05 | 57.09% | 0.06 | +0.00011587 | -0.00005724 | -0.00005724 | 0.647 | 1.00 | 0.00% |

survive_fee1_passrate_ge_0.5=4
