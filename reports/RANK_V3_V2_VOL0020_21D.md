# PASSIVE_POCKET_RANKING

candidates=7 ranked=7
candidate_parse total_rows_seen=26 table_rows_seen=22 rows_with_pass_yes=7 candidates_parsed=7 candidates_unique=7 rows_skipped_missing_fields=0
fee_grid=[0.2, 0.5, 1.0] adverse_mult_grid=[1.0, 1.2, 1.5]
pass_threshold=0.330
mitigation_profile=anti_adverse_v2 gate_config min_intensity_strong=0.000000 min_imbalance_strong=0.000000 max_spread_tight=0.000000 max_volatility_extreme=0.002 vol_quantile_reject=0.010000

| rank | symbol | rule | horizon | min_imb | min_int | max_spread | score | robust_core | robust_stress | pass_rate_core | pass_rate_stress | npa_core | npa_stress | attempt_fill_rate | attempts_per_min | score_raw_core | score_raw_stress | score_raw_min | baseline_pass_rate_core | baseline_pass_rate_stress | baseline_npa_core | baseline_score_raw_core | delta_pass_rate_core | delta_npa_core | delta_score_raw_core | stability_std_bps | best_fee_survive | insufficient_fill_rate |
|---:|---|---|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.50 | 2500 | 0.000300 | 0.000000e+00 | False | False | 27.78% | 27.78% | -4.497261e-05 | -5.295514e-05 | 59.91% | 0.05 | +0.00008721 | -0.00008572 | -0.00008572 | 27.78% | 27.78% | -4.497261e-05 | +0.00008721 | +0.00% | +0.000000e+00 | +0.00000000 | 0.927 | 1.00 | 0.00% |
| 2 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.40 | 2500 | 0.000300 | 0.000000e+00 | False | False | 27.78% | 27.78% | -5.242905e-05 | -6.066667e-05 | 59.75% | 0.05 | +0.00007340 | -0.00010056 | -0.00010056 | 27.78% | 27.78% | -5.242905e-05 | +0.00007340 | +0.00% | +0.000000e+00 | +0.00000000 | 0.988 | 1.00 | 0.00% |
| 3 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.50 | 3500 | 0.000300 | 0.000000e+00 | False | False | 33.33% | 27.78% | -7.727099e-05 | -8.940165e-05 | 61.48% | 0.03 | +0.00004144 | -0.00013718 | -0.00013718 | 33.33% | 27.78% | -7.727099e-05 | +0.00004144 | +0.00% | +0.000000e+00 | +0.00000000 | 1.119 | 1.00 | 0.00% |
| 4 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.40 | 2500 | 0.000500 | 0.000000e+00 | False | False | 16.67% | 16.67% | -8.173145e-05 | -8.919586e-05 | 55.10% | 0.06 | +0.00001473 | -0.00015854 | -0.00015854 | 16.67% | 16.67% | -8.173145e-05 | +0.00001473 | +0.00% | +0.000000e+00 | +0.00000000 | 0.908 | 1.00 | 0.00% |
| 5 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.30 | 2500 | 0.000500 | 0.000000e+00 | False | False | 11.11% | 11.11% | -1.073633e-04 | -1.143835e-04 | 55.09% | 0.07 | -0.00003330 | -0.00020607 | -0.00020607 | 11.11% | 11.11% | -1.073633e-04 | -0.00003330 | +0.00% | +0.000000e+00 | +0.00000000 | 0.921 | 1.00 | 0.00% |
| 6 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.50 | 2500 | 0.000500 | 0.000000e+00 | False | False | 16.67% | 5.56% | -5.746920e-05 | -6.440929e-05 | 54.89% | 0.06 | +0.00005777 | -0.00011462 | -0.00011462 | 16.67% | 5.56% | -5.746920e-05 | +0.00005777 | +0.00% | +0.000000e+00 | +0.00000000 | 0.796 | 1.00 | 0.00% |
| 7 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.20 | 2500 | 0.000500 | 0.000000e+00 | False | False | 5.56% | 5.56% | -1.046531e-04 | -1.115342e-04 | 53.04% | 0.07 | -0.00003974 | -0.00021289 | -0.00021289 | 5.56% | 5.56% | -1.046531e-04 | -0.00003974 | +0.00% | +0.000000e+00 | +0.00000000 | 0.830 | 1.00 | 0.00% |

survive_fee1_passrate_ge_0.5=0
