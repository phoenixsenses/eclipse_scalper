# PASSIVE_POCKET_RANKING

candidates=7 ranked=7
candidate_parse total_rows_seen=26 table_rows_seen=22 rows_with_pass_yes=7 candidates_parsed=7 candidates_unique=7 rows_skipped_missing_fields=0
fee_grid=[1.0] adverse_mult_grid=[1.2]
pass_threshold=0.330
mitigation_profile=anti_adverse_v3 gate_config min_intensity_strong=0.000000 min_imbalance_strong=0.000000 max_spread_tight=0.000000 max_volatility_extreme=None vol_quantile_reject=0.008000

| rank | symbol | rule | horizon | min_imb | min_int | max_spread | score | robust_core | robust_stress | pass_rate_core | pass_rate_stress | npa_core | npa_stress | attempt_fill_rate | attempts_per_min | score_raw_core | score_raw_stress | score_raw_min | baseline_pass_rate_core | baseline_pass_rate_stress | baseline_npa_core | baseline_score_raw_core | delta_pass_rate_core | delta_npa_core | delta_score_raw_core | stability_std_bps | best_fee_survive | insufficient_fill_rate |
|---:|---|---|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.50 | 2500 | 0.000500 | 0.000000e+00 | False | False | 27.78% | 27.78% | -1.267331e-04 | -1.267331e-04 | 56.45% | 0.06 | -0.00021969 | -0.00021969 | -0.00021969 | 27.78% | 27.78% | -1.280745e-04 | -0.00022394 | +0.00% | +1.341435e-06 | +0.00000425 | 0.537 | 1.00 | 0.00% |
| 2 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.50 | 2500 | 0.000300 | 0.000000e+00 | False | False | 22.22% | 22.22% | -1.338996e-04 | -1.338996e-04 | 62.46% | 0.05 | -0.00021258 | -0.00021258 | -0.00021258 | 16.67% | 16.67% | -1.266370e-04 | -0.00020281 | +5.56% | -7.262696e-06 | -0.00000977 | 0.699 | 1.00 | 0.00% |
| 3 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.40 | 2500 | 0.000500 | 0.000000e+00 | False | False | 16.67% | 16.67% | -1.369208e-04 | -1.369208e-04 | 56.39% | 0.06 | -0.00023413 | -0.00023413 | -0.00023413 | 11.11% | 11.11% | -1.323019e-04 | -0.00022952 | +5.56% | -4.618823e-06 | -0.00000460 | 0.600 | 1.00 | 0.00% |
| 4 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.40 | 2500 | 0.000300 | 0.000000e+00 | False | False | 16.67% | 16.67% | -1.539409e-04 | -1.539409e-04 | 62.14% | 0.05 | -0.00024309 | -0.00024309 | -0.00024309 | 16.67% | 16.67% | -1.400901e-04 | -0.00022249 | +0.00% | -1.385080e-05 | -0.00002059 | 0.797 | 1.00 | 0.00% |
| 5 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.50 | 3500 | 0.000300 | 0.000000e+00 | False | False | 11.11% | 11.11% | -1.457519e-04 | -1.457519e-04 | 62.39% | 0.02 | -0.00022396 | -0.00022396 | -0.00022396 | 11.11% | 11.11% | -1.333966e-04 | -0.00020537 | +0.00% | -1.235530e-05 | -0.00001859 | 1.216 | 1.00 | 0.00% |
| 6 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.20 | 2500 | 0.000500 | 0.000000e+00 | False | False | 0.00% | 0.00% | -1.510509e-04 | -1.510509e-04 | 55.60% | 0.07 | -0.00027224 | -0.00027224 | -0.00027224 | 0.00% | 0.00% | -1.459400e-04 | -0.00026407 | +0.00% | -5.110953e-06 | -0.00000817 | 0.487 | 0.00 | 0.00% |
| 7 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.30 | 2500 | 0.000500 | 0.000000e+00 | False | False | 0.00% | 0.00% | -1.585140e-04 | -1.585140e-04 | 57.85% | 0.07 | -0.00027079 | -0.00027079 | -0.00027079 | 0.00% | 0.00% | -1.522950e-04 | -0.00026203 | +0.00% | -6.219091e-06 | -0.00000876 | 0.503 | 0.00 | 0.00% |

survive_fee1_passrate_ge_0.5=0
