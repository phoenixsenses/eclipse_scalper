# PASSIVE_POCKET_RANKING

candidates=7 ranked=6
candidate_parse total_rows_seen=26 table_rows_seen=22 rows_with_pass_yes=7 candidates_parsed=7 candidates_unique=7 rows_skipped_missing_fields=0
fee_grid=[1.0] adverse_mult_grid=[1.2, 1.5]
pass_threshold=0.500

| rank | symbol | rule | horizon | min_imb | min_int | max_spread | score | robust_core | robust_stress | pass_rate_core | pass_rate_stress | npa_core | npa_stress | attempt_fill_rate | attempts_per_min | score_raw_core | score_raw_stress | score_raw_min | stability_std_bps | best_fee_survive | insufficient_fill_rate |
|---:|---|---|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.50 | 2500 | 0.000300 | 0.000000e+00 | False | False | 46.67% | 46.67% | -5.778527e-07 | -3.722803e-06 | 60.51% | 0.07 | -0.00000101 | -0.00000645 | -0.00000645 | 1.012 | 1.00 | 0.00% |
| 2 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.40 | 2500 | 0.000300 | 0.000000e+00 | False | False | 43.33% | 40.00% | -3.484617e-05 | -3.712972e-05 | 61.45% | 0.07 | -0.00006313 | -0.00006726 | -0.00006726 | 1.020 | 1.00 | 0.00% |
| 3 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.50 | 2500 | 0.000500 | 0.000000e+00 | False | False | 36.67% | 33.33% | -2.004594e-05 | -2.343334e-05 | 58.04% | 0.08 | -0.00003455 | -0.00004028 | -0.00004028 | 0.748 | 1.00 | 0.00% |
| 4 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.40 | 2500 | 0.000500 | 0.000000e+00 | False | False | 23.33% | 23.33% | -3.355339e-05 | -3.729525e-05 | 58.37% | 0.09 | -0.00006017 | -0.00006666 | -0.00006666 | 0.842 | 1.00 | 0.00% |
| 5 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.20 | 2500 | 0.000500 | 0.000000e+00 | False | False | 20.00% | 16.67% | -3.712652e-05 | -4.092684e-05 | 56.00% | 0.10 | -0.00006751 | -0.00007441 | -0.00007441 | 0.830 | 1.00 | 0.00% |
| 6 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.30 | 2500 | 0.000500 | 0.000000e+00 | False | False | 20.00% | 13.33% | -5.136791e-05 | -5.577480e-05 | 58.36% | 0.10 | -0.00008754 | -0.00009507 | -0.00009507 | 0.932 | 1.00 | 0.00% |

survive_fee1_passrate_ge_0.5=0
