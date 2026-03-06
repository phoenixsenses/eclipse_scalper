# PASSIVE_POCKET_RANKING

candidates=7 ranked=7
candidate_parse total_rows_seen=26 table_rows_seen=22 rows_with_pass_yes=7 candidates_parsed=7 candidates_unique=7 rows_skipped_missing_fields=0
fee_grid=[0.2, 0.5, 1.0] adverse_mult_grid=[1.0, 1.2, 1.5]
pass_threshold=0.330

| rank | symbol | rule | horizon | min_imb | min_int | max_spread | score | robust_core | robust_stress | pass_rate_core | pass_rate_stress | npa_core | npa_stress | attempt_fill_rate | attempts_per_min | score_raw_core | score_raw_stress | score_raw_min | baseline_pass_rate_core | baseline_pass_rate_stress | baseline_npa_core | baseline_score_raw_core | delta_pass_rate_core | delta_npa_core | delta_score_raw_core | stability_std_bps | best_fee_survive | insufficient_fill_rate |
|---:|---|---|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.50 | 3500 | 0.000300 | 0.000000e+00 | False | False | 50.00% | 50.00% | -5.444051e-05 | -6.427758e-05 | 62.02% | 0.02 | +0.00004161 | -0.00013568 | -0.00013568 | 50.00% | 50.00% | -5.444051e-05 | +0.00004161 | +0.00% | +0.000000e+00 | +0.00000000 | 0.796 | 1.00 | 50.00% |
| 2 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.50 | 2500 | 0.000300 | 0.000000e+00 | False | False | 50.00% | 50.00% | -7.112421e-05 | -7.934704e-05 | 61.64% | 0.04 | +0.00004005 | -0.00013320 | -0.00013320 | 50.00% | 50.00% | -7.112421e-05 | +0.00004005 | +0.00% | +0.000000e+00 | +0.00000000 | 0.702 | 1.00 | 44.44% |
| 3 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.50 | 2500 | 0.000500 | 0.000000e+00 | False | False | 50.00% | 50.00% | -7.702277e-05 | -8.348420e-05 | 58.26% | 0.05 | -0.00002028 | -0.00019274 | -0.00019274 | 50.00% | 50.00% | -7.702277e-05 | -0.00002028 | +0.00% | +0.000000e+00 | +0.00000000 | 0.526 | 1.00 | 5.56% |
| 4 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.40 | 2500 | 0.000300 | 0.000000e+00 | False | False | 44.44% | 44.44% | -9.472613e-05 | -1.029197e-04 | 61.54% | 0.05 | +0.00000686 | -0.00016636 | -0.00016636 | 44.44% | 44.44% | -9.472613e-05 | +0.00000686 | +0.00% | +0.000000e+00 | +0.00000000 | 0.702 | 1.00 | 5.56% |
| 5 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.40 | 2500 | 0.000500 | 0.000000e+00 | False | False | 44.44% | 38.89% | -9.367416e-05 | -1.003670e-04 | 58.52% | 0.06 | -0.00002145 | -0.00019435 | -0.00019435 | 44.44% | 38.89% | -9.367416e-05 | -0.00002145 | +0.00% | +0.000000e+00 | +0.00000000 | 0.582 | 1.00 | 0.00% |
| 6 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.30 | 2500 | 0.000500 | 0.000000e+00 | False | False | 27.78% | 27.78% | -9.248095e-05 | -9.965517e-05 | 58.15% | 0.06 | -0.00000453 | -0.00017735 | -0.00017735 | 27.78% | 27.78% | -9.248095e-05 | -0.00000453 | +0.00% | +0.000000e+00 | +0.00000000 | 0.741 | 1.00 | 0.00% |
| 7 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.20 | 2500 | 0.000500 | 0.000000e+00 | False | False | 27.78% | 27.78% | -9.360308e-05 | -1.002069e-04 | 56.00% | 0.06 | -0.00001102 | -0.00018425 | -0.00018425 | 27.78% | 27.78% | -9.360308e-05 | -0.00001102 | +0.00% | +0.000000e+00 | +0.00000000 | 0.730 | 1.00 | 0.00% |

survive_fee1_passrate_ge_0.5=3
