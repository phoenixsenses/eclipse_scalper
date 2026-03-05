# PASSIVE_POCKET_RANKING

candidates=7 ranked=7
candidate_parse total_rows_seen=26 table_rows_seen=22 rows_with_pass_yes=7 candidates_parsed=7 candidates_unique=7 rows_skipped_missing_fields=0
fee_grid=[1.0] adverse_mult_grid=[1.2]
pass_threshold=0.330
mitigation_profile=anti_adverse_v3 gate_config min_intensity_strong=0.000000 min_imbalance_strong=0.000000 max_spread_tight=0.000000 max_volatility_extreme=None vol_quantile_reject=0.010000

| rank | symbol | rule | horizon | min_imb | min_int | max_spread | score | robust_core | robust_stress | pass_rate_core | pass_rate_stress | npa_core | npa_stress | attempt_fill_rate | attempts_per_min | score_raw_core | score_raw_stress | score_raw_min | baseline_pass_rate_core | baseline_pass_rate_stress | baseline_npa_core | baseline_score_raw_core | delta_pass_rate_core | delta_npa_core | delta_score_raw_core | stability_std_bps | best_fee_survive | insufficient_fill_rate |
|---:|---|---|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.50 | 2500 | 0.000500 | 0.000000e+00 | False | False | 27.78% | 27.78% | -1.106937e-04 | -1.106937e-04 | 54.77% | 0.06 | -0.00020382 | -0.00020382 | -0.00020382 | 27.78% | 27.78% | -1.206339e-04 | -0.00022192 | +0.00% | +9.940211e-06 | +0.00001810 | 0.749 | 1.00 | 0.00% |
| 2 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.40 | 2500 | 0.000500 | 0.000000e+00 | False | False | 27.78% | 27.78% | -1.120517e-04 | -1.120517e-04 | 55.78% | 0.06 | -0.00020287 | -0.00020287 | -0.00020287 | 27.78% | 27.78% | -1.107336e-04 | -0.00020013 | +0.00% | -1.318103e-06 | -0.00000274 | 0.729 | 1.00 | 0.00% |
| 3 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.50 | 3500 | 0.000300 | 0.000000e+00 | False | False | 22.22% | 22.22% | -7.320009e-05 | -7.320009e-05 | 63.41% | 0.02 | -0.00012811 | -0.00012811 | -0.00012811 | 22.22% | 22.22% | -6.709259e-05 | -0.00011779 | +0.00% | -6.107497e-06 | -0.00001032 | 1.238 | 1.00 | 0.00% |
| 4 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.50 | 2500 | 0.000300 | 0.000000e+00 | False | False | 22.22% | 22.22% | -1.031186e-04 | -1.031186e-04 | 60.95% | 0.05 | -0.00017453 | -0.00017453 | -0.00017453 | 22.22% | 22.22% | -1.049205e-04 | -0.00017718 | +0.00% | +1.801831e-06 | +0.00000264 | 0.804 | 1.00 | 0.00% |
| 5 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.30 | 2500 | 0.000500 | 0.000000e+00 | False | False | 22.22% | 22.22% | -1.414516e-04 | -1.414516e-04 | 56.54% | 0.07 | -0.00024691 | -0.00024691 | -0.00024691 | 16.67% | 16.67% | -1.556135e-04 | -0.00027145 | +5.56% | +1.416191e-05 | +0.00002453 | 0.890 | 1.00 | 0.00% |
| 6 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.20 | 2500 | 0.000500 | 0.000000e+00 | False | False | 22.22% | 22.22% | -1.420214e-04 | -1.420214e-04 | 54.19% | 0.07 | -0.00025692 | -0.00025692 | -0.00025692 | 11.11% | 11.11% | -1.549834e-04 | -0.00028052 | +11.11% | +1.296194e-05 | +0.00002360 | 0.847 | 1.00 | 0.00% |
| 7 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.40 | 2500 | 0.000300 | 0.000000e+00 | False | False | 16.67% | 16.67% | -1.109673e-04 | -1.109673e-04 | 60.98% | 0.05 | -0.00018962 | -0.00018962 | -0.00018962 | 16.67% | 16.67% | -9.915474e-05 | -0.00016832 | +0.00% | -1.181254e-05 | -0.00002130 | 0.798 | 1.00 | 0.00% |

survive_fee1_passrate_ge_0.5=0
