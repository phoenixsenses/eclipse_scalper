# PASSIVE_POCKET_RANKING

candidates=7 ranked=7
candidate_parse total_rows_seen=26 table_rows_seen=22 rows_with_pass_yes=7 candidates_parsed=7 candidates_unique=7 rows_skipped_missing_fields=0
fee_grid=[1.0] adverse_mult_grid=[1.2]
pass_threshold=0.330
mitigation_profile=anti_adverse_v3 gate_config min_intensity_strong=0.000000 min_imbalance_strong=0.000000 max_spread_tight=0.000000 max_volatility_extreme=None vol_quantile_reject=0.010000

| rank | symbol | rule | horizon | min_imb | min_int | max_spread | score | robust_core | robust_stress | pass_rate_core | pass_rate_stress | npa_core | npa_stress | attempt_fill_rate | attempts_per_min | score_raw_core | score_raw_stress | score_raw_min | baseline_pass_rate_core | baseline_pass_rate_stress | baseline_npa_core | baseline_score_raw_core | delta_pass_rate_core | delta_npa_core | delta_score_raw_core | stability_std_bps | best_fee_survive | insufficient_fill_rate |
|---:|---|---|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.40 | 2500 | 0.000500 | 0.000000e+00 | False | False | 27.78% | 27.78% | -8.198869e-05 | -8.198869e-05 | 55.25% | 0.06 | -0.00016321 | -0.00016321 | -0.00016321 | 11.11% | 11.11% | -9.570224e-05 | -0.00018719 | +16.67% | +1.371356e-05 | +0.00002398 | 0.704 | 1.00 | 0.00% |
| 2 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.50 | 2500 | 0.000500 | 0.000000e+00 | False | False | 27.78% | 27.78% | -8.567462e-05 | -8.567462e-05 | 54.12% | 0.06 | -0.00017316 | -0.00017316 | -0.00017316 | 27.78% | 27.78% | -9.133274e-05 | -0.00018307 | +0.00% | +5.658121e-06 | +0.00000991 | 0.794 | 1.00 | 0.00% |
| 3 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.50 | 2500 | 0.000300 | 0.000000e+00 | False | False | 22.22% | 22.22% | -9.756690e-05 | -9.756690e-05 | 58.68% | 0.05 | -0.00017416 | -0.00017416 | -0.00017416 | 22.22% | 22.22% | -1.021748e-04 | -0.00018212 | +0.00% | +4.607906e-06 | +0.00000797 | 0.778 | 1.00 | 0.00% |
| 4 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.40 | 2500 | 0.000300 | 0.000000e+00 | False | False | 16.67% | 16.67% | -1.070994e-04 | -1.070994e-04 | 59.40% | 0.05 | -0.00019322 | -0.00019322 | -0.00019322 | 16.67% | 16.67% | -9.194350e-05 | -0.00016431 | +0.00% | -1.515592e-05 | -0.00002891 | 0.775 | 1.00 | 0.00% |
| 5 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.50 | 3500 | 0.000300 | 0.000000e+00 | False | False | 16.67% | 16.67% | -1.710224e-04 | -1.710224e-04 | 59.89% | 0.02 | -0.00028951 | -0.00028951 | -0.00028951 | 16.67% | 16.67% | -1.626412e-04 | -0.00027141 | +0.00% | -8.381252e-06 | -0.00001809 | 1.033 | 1.00 | 0.00% |
| 6 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.30 | 2500 | 0.000500 | 0.000000e+00 | False | False | 11.11% | 11.11% | -1.161195e-04 | -1.161195e-04 | 56.74% | 0.07 | -0.00021543 | -0.00021543 | -0.00021543 | 5.56% | 5.56% | -1.235827e-04 | -0.00023286 | +5.56% | +7.463179e-06 | +0.00001743 | 0.535 | 1.00 | 0.00% |
| 7 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.20 | 2500 | 0.000500 | 0.000000e+00 | False | False | 5.56% | 5.56% | -1.077796e-04 | -1.077796e-04 | 54.61% | 0.07 | -0.00020726 | -0.00020726 | -0.00020726 | 5.56% | 5.56% | -1.180827e-04 | -0.00022487 | +0.00% | +1.030307e-05 | +0.00001761 | 0.551 | 1.00 | 0.00% |

survive_fee1_passrate_ge_0.5=0
