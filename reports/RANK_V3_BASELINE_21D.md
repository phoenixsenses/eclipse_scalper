# PASSIVE_POCKET_RANKING

candidates=7 ranked=7
candidate_parse total_rows_seen=26 table_rows_seen=22 rows_with_pass_yes=7 candidates_parsed=7 candidates_unique=7 rows_skipped_missing_fields=0
fee_grid=[0.2, 0.5, 1.0] adverse_mult_grid=[1.0, 1.2, 1.5]
pass_threshold=0.330
mitigation_profile=baseline gate_config min_intensity_strong=0.000000 min_imbalance_strong=0.000000 max_spread_tight=0.000000 max_volatility_extreme=None vol_quantile_reject=0.010000

| rank | symbol | rule | horizon | min_imb | min_int | max_spread | score | robust_core | robust_stress | pass_rate_core | pass_rate_stress | npa_core | npa_stress | attempt_fill_rate | attempts_per_min | score_raw_core | score_raw_stress | score_raw_min | baseline_pass_rate_core | baseline_pass_rate_stress | baseline_npa_core | baseline_score_raw_core | delta_pass_rate_core | delta_npa_core | delta_score_raw_core | stability_std_bps | best_fee_survive | insufficient_fill_rate |
|---:|---|---|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.50 | 2500 | 0.000300 | 0.000000e+00 | False | False | 22.22% | 22.22% | -4.640835e-05 | -5.512757e-05 | 59.87% | 0.05 | +0.00007909 | -0.00009644 | -0.00009644 | 22.22% | 22.22% | -4.640835e-05 | +0.00007909 | +0.00% | +0.000000e+00 | +0.00000000 | 1.037 | 1.00 | 0.00% |
| 2 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.50 | 2500 | 0.000500 | 0.000000e+00 | False | False | 16.67% | 16.67% | -5.265950e-05 | -5.985541e-05 | 53.70% | 0.06 | +0.00005663 | -0.00011751 | -0.00011751 | 16.67% | 16.67% | -5.265950e-05 | +0.00005663 | +0.00% | +0.000000e+00 | +0.00000000 | 1.067 | 1.00 | 0.00% |
| 3 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.40 | 2500 | 0.000500 | 0.000000e+00 | False | False | 16.67% | 16.67% | -6.333937e-05 | -7.023335e-05 | 54.06% | 0.06 | +0.00003346 | -0.00014031 | -0.00014031 | 16.67% | 16.67% | -6.333937e-05 | +0.00003346 | +0.00% | +0.000000e+00 | +0.00000000 | 1.180 | 1.00 | 0.00% |
| 4 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.50 | 3500 | 0.000300 | 0.000000e+00 | False | False | 11.11% | 11.11% | -5.594832e-05 | -6.702115e-05 | 63.71% | 0.03 | +0.00006395 | -0.00011503 | -0.00011503 | 11.11% | 11.11% | -5.594832e-05 | +0.00006395 | +0.00% | +0.000000e+00 | +0.00000000 | 1.218 | 1.00 | 0.00% |
| 5 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.30 | 2500 | 0.000500 | 0.000000e+00 | False | False | 11.11% | 11.11% | -7.301155e-05 | -8.017954e-05 | 55.41% | 0.07 | +0.00002225 | -0.00015061 | -0.00015061 | 11.11% | 11.11% | -7.301155e-05 | +0.00002225 | +0.00% | +0.000000e+00 | +0.00000000 | 1.150 | 1.00 | 0.00% |
| 6 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.20 | 2500 | 0.000500 | 0.000000e+00 | False | False | 11.11% | 11.11% | -7.443896e-05 | -8.149013e-05 | 53.39% | 0.07 | +0.00002319 | -0.00015010 | -0.00015010 | 11.11% | 11.11% | -7.443896e-05 | +0.00002319 | +0.00% | +0.000000e+00 | +0.00000000 | 1.126 | 1.00 | 0.00% |
| 7 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.40 | 2500 | 0.000300 | 0.000000e+00 | False | False | 11.11% | 5.56% | -7.960859e-05 | -8.768314e-05 | 60.68% | 0.05 | +0.00002315 | -0.00015090 | -0.00015090 | 11.11% | 5.56% | -7.960859e-05 | +0.00002315 | +0.00% | +0.000000e+00 | +0.00000000 | 1.040 | 1.00 | 0.00% |

survive_fee1_passrate_ge_0.5=0
