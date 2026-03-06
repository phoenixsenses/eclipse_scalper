# PASSIVE_POCKET_RANKING

candidates=7 ranked=7
candidate_parse total_rows_seen=26 table_rows_seen=22 rows_with_pass_yes=7 candidates_parsed=7 candidates_unique=7 rows_skipped_missing_fields=0
fee_grid=[0.2, 0.5, 1.0] adverse_mult_grid=[0.8, 1.0, 1.2]
pass_threshold=0.330
mitigation_profile=anti_adverse_v3 gate_config min_intensity_strong=0.000000 min_imbalance_strong=0.000000 max_spread_tight=0.000000 max_volatility_extreme=None vol_quantile_reject=0.010000

| rank | symbol | rule | horizon | min_imb | min_int | max_spread | score | robust_core | robust_stress | pass_rate_core | pass_rate_stress | npa_core | npa_stress | attempt_fill_rate | attempts_per_min | score_raw_core | score_raw_stress | score_raw_min | baseline_pass_rate_core | baseline_pass_rate_stress | baseline_npa_core | baseline_score_raw_core | delta_pass_rate_core | delta_npa_core | delta_score_raw_core | stability_std_bps | best_fee_survive | insufficient_fill_rate |
|---:|---|---|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.50 | 2500 | 0.000500 | 0.000000e+00 | False | False | 33.33% | 33.33% | -1.019324e-04 | -1.047543e-04 | 56.84% | 0.06 | -0.00003449 | -0.00019965 | -0.00019965 | 27.78% | 22.22% | -1.183969e-04 | -0.00006372 | +5.56% | +1.646455e-05 | +0.00002923 | 0.695 | 1.00 | 0.00% |
| 2 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.40 | 2500 | 0.000300 | 0.000000e+00 | False | False | 27.78% | 27.78% | -1.459934e-04 | -1.492991e-04 | 60.98% | 0.05 | -0.00009034 | -0.00025581 | -0.00025581 | 27.78% | 22.22% | -1.223417e-04 | -0.00004959 | +0.00% | -2.365175e-05 | -0.00004075 | 0.686 | 1.00 | 0.00% |
| 3 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.40 | 2500 | 0.000500 | 0.000000e+00 | False | False | 27.78% | 22.22% | -1.046150e-04 | -1.075151e-04 | 56.45% | 0.06 | -0.00003024 | -0.00019539 | -0.00019539 | 11.11% | 11.11% | -1.004152e-04 | -0.00002148 | +16.67% | -4.199792e-06 | -0.00000876 | 0.585 | 1.00 | 0.00% |
| 4 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.50 | 2500 | 0.000300 | 0.000000e+00 | False | False | 27.78% | 22.22% | -1.219859e-04 | -1.253640e-04 | 62.06% | 0.05 | -0.00005252 | -0.00021815 | -0.00021815 | 22.22% | 16.67% | -1.242747e-04 | -0.00005366 | +5.56% | +2.288796e-06 | +0.00000114 | 0.758 | 1.00 | 0.00% |
| 5 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.50 | 3500 | 0.000300 | 0.000000e+00 | False | False | 16.67% | 16.67% | -2.170205e-04 | -2.212300e-04 | 59.77% | 0.02 | -0.00020057 | -0.00036778 | -0.00036778 | 16.67% | 16.67% | -2.182330e-04 | -0.00021196 | +0.00% | +1.212512e-06 | +0.00001139 | 0.787 | 1.00 | 0.00% |
| 6 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.30 | 2500 | 0.000500 | 0.000000e+00 | False | False | 11.11% | 11.11% | -1.125323e-04 | -1.154885e-04 | 57.96% | 0.07 | -0.00003850 | -0.00020358 | -0.00020358 | 11.11% | 11.11% | -1.076028e-04 | -0.00002927 | +0.00% | -4.929529e-06 | -0.00000923 | 0.563 | 1.00 | 0.00% |
| 7 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.20 | 2500 | 0.000500 | 0.000000e+00 | False | False | 5.56% | 5.56% | -1.062705e-04 | -1.091450e-04 | 55.35% | 0.07 | -0.00004037 | -0.00020563 | -0.00020563 | 0.00% | 0.00% | -1.048146e-04 | -0.00003345 | +5.56% | -1.455888e-06 | -0.00000692 | 0.568 | 1.00 | 0.00% |

survive_fee1_passrate_ge_0.5=0
