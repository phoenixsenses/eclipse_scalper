# PASSIVE_POCKET_RANKING

candidates=7 ranked=7
candidate_parse total_rows_seen=26 table_rows_seen=22 rows_with_pass_yes=7 candidates_parsed=7 candidates_unique=7 rows_skipped_missing_fields=0
fee_grid=[0.2, 0.5, 1.0] adverse_mult_grid=[1.0, 1.2, 1.5]
pass_threshold=0.330
mitigation_profile=anti_adverse_v3 gate_config min_intensity_strong=0.000000 min_imbalance_strong=0.000000 max_spread_tight=0.000000 max_volatility_extreme=None vol_quantile_reject=0.005000

| rank | symbol | rule | horizon | min_imb | min_int | max_spread | score | robust_core | robust_stress | pass_rate_core | pass_rate_stress | npa_core | npa_stress | attempt_fill_rate | attempts_per_min | score_raw_core | score_raw_stress | score_raw_min | baseline_pass_rate_core | baseline_pass_rate_stress | baseline_npa_core | baseline_score_raw_core | delta_pass_rate_core | delta_npa_core | delta_score_raw_core | stability_std_bps | best_fee_survive | insufficient_fill_rate |
|---:|---|---|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.50 | 3500 | 0.000300 | 0.000000e+00 | False | False | 38.89% | 33.33% | -5.489278e-05 | -6.605889e-05 | 62.20% | 0.03 | +0.00006673 | -0.00011183 | -0.00011183 | 33.33% | 27.78% | -5.648171e-05 | +0.00006381 | +5.56% | +1.588927e-06 | +0.00000293 | 1.087 | 1.00 | 0.00% |
| 2 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.50 | 2500 | 0.000300 | 0.000000e+00 | False | False | 33.33% | 33.33% | -6.317641e-05 | -7.168470e-05 | 61.01% | 0.05 | +0.00006164 | -0.00011263 | -0.00011263 | 33.33% | 33.33% | -6.367947e-05 | +0.00006094 | +0.00% | +5.030608e-07 | +0.00000070 | 0.910 | 1.00 | 0.00% |
| 3 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.50 | 2500 | 0.000500 | 0.000000e+00 | False | False | 33.33% | 33.33% | -7.477178e-05 | -8.148294e-05 | 54.85% | 0.06 | -0.00000053 | -0.00017340 | -0.00017340 | 27.78% | 22.22% | -7.460433e-05 | +0.00000013 | +5.56% | -1.674524e-07 | -0.00000066 | 0.696 | 1.00 | 0.00% |
| 4 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.40 | 2500 | 0.000300 | 0.000000e+00 | False | False | 38.89% | 27.78% | -6.750923e-05 | -7.503298e-05 | 60.21% | 0.05 | +0.00003682 | -0.00013734 | -0.00013734 | 38.89% | 27.78% | -4.702063e-05 | +0.00007210 | +0.00% | -2.048860e-05 | -0.00003528 | 0.774 | 1.00 | 0.00% |
| 5 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.40 | 2500 | 0.000500 | 0.000000e+00 | False | False | 33.33% | 27.78% | -7.525054e-05 | -8.219117e-05 | 55.06% | 0.06 | +0.00000211 | -0.00017076 | -0.00017076 | 27.78% | 11.11% | -7.446054e-05 | +0.00001207 | +5.56% | -7.899941e-07 | -0.00000996 | 0.635 | 1.00 | 0.00% |
| 6 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.30 | 2500 | 0.000500 | 0.000000e+00 | False | False | 16.67% | 16.67% | -1.132975e-04 | -1.205569e-04 | 55.62% | 0.07 | -0.00005376 | -0.00022660 | -0.00022660 | 11.11% | 0.00% | -1.434666e-04 | -0.00008539 | +5.56% | +3.016914e-05 | +0.00003164 | 0.575 | 1.00 | 0.00% |
| 7 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.20 | 2500 | 0.000500 | 0.000000e+00 | False | False | 16.67% | 5.56% | -1.101565e-04 | -1.168655e-04 | 53.91% | 0.07 | -0.00005880 | -0.00023193 | -0.00023193 | 0.00% | 0.00% | -1.359341e-04 | -0.00008205 | +16.67% | +2.577760e-05 | +0.00002325 | 0.530 | 1.00 | 0.00% |

survive_fee1_passrate_ge_0.5=0
