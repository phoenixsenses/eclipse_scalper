# PASSIVE_POCKET_RANKING

candidates=7 ranked=7
candidate_parse total_rows_seen=26 table_rows_seen=22 rows_with_pass_yes=7 candidates_parsed=7 candidates_unique=7 rows_skipped_missing_fields=0
fee_grid=[0.2, 0.5, 1.0] adverse_mult_grid=[1.0, 1.2, 1.5]
pass_threshold=0.330
mitigation_profile=anti_adverse_v2 gate_config min_intensity_strong=0.000000 min_imbalance_strong=0.000000 max_spread_tight=0.000000 max_volatility_extreme=0.004 vol_quantile_reject=0.010000

| rank | symbol | rule | horizon | min_imb | min_int | max_spread | score | robust_core | robust_stress | pass_rate_core | pass_rate_stress | npa_core | npa_stress | attempt_fill_rate | attempts_per_min | score_raw_core | score_raw_stress | score_raw_min | baseline_pass_rate_core | baseline_pass_rate_stress | baseline_npa_core | baseline_score_raw_core | delta_pass_rate_core | delta_npa_core | delta_score_raw_core | stability_std_bps | best_fee_survive | insufficient_fill_rate |
|---:|---|---|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.50 | 3500 | 0.000300 | 0.000000e+00 | False | False | 33.33% | 27.78% | -8.912320e-05 | -9.967460e-05 | 60.62% | 0.02 | -0.00000060 | -0.00017899 | -0.00017899 | 33.33% | 27.78% | -8.912320e-05 | -0.00000060 | +0.00% | +0.000000e+00 | +0.00000000 | 1.085 | 1.00 | 0.00% |
| 2 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.50 | 2500 | 0.000500 | 0.000000e+00 | False | False | 27.78% | 22.22% | -8.845408e-05 | -9.590538e-05 | 54.64% | 0.06 | +0.00000169 | -0.00017150 | -0.00017150 | 27.78% | 22.22% | -8.845408e-05 | +0.00000169 | +0.00% | +0.000000e+00 | +0.00000000 | 0.885 | 1.00 | 0.00% |
| 3 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.40 | 2500 | 0.000300 | 0.000000e+00 | False | False | 22.22% | 22.22% | -9.632153e-05 | -1.041122e-04 | 58.74% | 0.05 | -0.00001488 | -0.00018882 | -0.00018882 | 22.22% | 22.22% | -9.632153e-05 | -0.00001488 | +0.00% | +0.000000e+00 | +0.00000000 | 0.892 | 1.00 | 0.00% |
| 4 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.50 | 2500 | 0.000300 | 0.000000e+00 | False | False | 22.22% | 22.22% | -9.654989e-05 | -1.050447e-04 | 60.16% | 0.05 | -0.00000691 | -0.00018109 | -0.00018109 | 22.22% | 22.22% | -9.654989e-05 | -0.00000691 | +0.00% | +0.000000e+00 | +0.00000000 | 0.852 | 1.00 | 0.00% |
| 5 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.40 | 2500 | 0.000500 | 0.000000e+00 | False | False | 22.22% | 22.22% | -1.056016e-04 | -1.124424e-04 | 54.71% | 0.06 | -0.00004131 | -0.00021438 | -0.00021438 | 22.22% | 22.22% | -1.056016e-04 | -0.00004131 | +0.00% | +0.000000e+00 | +0.00000000 | 1.008 | 1.00 | 0.00% |
| 6 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.30 | 2500 | 0.000500 | 0.000000e+00 | False | False | 11.11% | 11.11% | -1.569702e-04 | -1.640490e-04 | 55.45% | 0.07 | -0.00012971 | -0.00030269 | -0.00030269 | 11.11% | 11.11% | -1.569702e-04 | -0.00012971 | +0.00% | +0.000000e+00 | +0.00000000 | 0.635 | 1.00 | 0.00% |
| 7 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.20 | 2500 | 0.000500 | 0.000000e+00 | False | False | 5.56% | 0.00% | -1.511839e-04 | -1.580777e-04 | 53.51% | 0.07 | -0.00013223 | -0.00030541 | -0.00030541 | 5.56% | 0.00% | -1.511839e-04 | -0.00013223 | +0.00% | +0.000000e+00 | +0.00000000 | 0.640 | 1.00 | 0.00% |

survive_fee1_passrate_ge_0.5=0
