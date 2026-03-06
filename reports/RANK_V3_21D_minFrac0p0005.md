# PASSIVE_POCKET_RANKING

candidates=7 ranked=5
candidate_parse total_rows_seen=26 table_rows_seen=22 rows_with_pass_yes=7 candidates_parsed=7 candidates_unique=7 rows_skipped_missing_fields=0
fee_grid=[1.0] adverse_mult_grid=[1.2, 1.5]

| rank | symbol | rule | horizon | min_imb | min_int | max_spread | score | robust_core | robust_stress | net_per_attempt | attempt_fill_rate | attempts_per_min | score_raw_core | score_raw_stress | score_raw_min | stability_std_bps | best_fee_survive | insufficient_fill_rate |
|---:|---|---|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.40 | 2500 | 0.000500 | 4.651151e-02 | True | True | +9.767016e-06 | 57.55% | 0.09 | +0.00001742 | +0.00001141 | +0.00001141 | 0.938 | 1.00 | 16.67% |
| 2 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.50 | 2500 | 0.000500 | 4.596652e-02 | True | True | +1.000694e-05 | 58.13% | 0.08 | +0.00002055 | +0.00001658 | +0.00001658 | 0.866 | 1.00 | 33.33% |
| 3 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.20 | 2500 | 0.000500 | 0.000000e+00 | False | False | -3.111682e-06 | 54.50% | 0.10 | -0.00000658 | -0.00001133 | -0.00001133 | 0.988 | 1.00 | 6.67% |
| 4 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.30 | 2500 | 0.000500 | 0.000000e+00 | False | False | -7.626723e-06 | 57.27% | 0.09 | -0.00001296 | -0.00001892 | -0.00001892 | 1.038 | 1.00 | 10.00% |
| 5 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.40 | 2500 | 0.000300 | 0.000000e+00 | False | False | -7.339044e-05 | 62.19% | 0.07 | -0.00012253 | -0.00012811 | -0.00012811 | 0.911 | 1.00 | 43.33% |

survive_fee1_passrate_ge_0.5=0
