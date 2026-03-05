# PASSIVE_POCKET_RANKING

candidates=7 ranked=7
candidate_parse total_rows_seen=26 table_rows_seen=22 rows_with_pass_yes=7 candidates_parsed=7 candidates_unique=7 rows_skipped_missing_fields=0
fee_grid=[1.0] adverse_mult_grid=[1.2]
pass_threshold=0.330
mitigation_profile=anti_adverse_v3 gate_config min_intensity_strong=0.000000 min_imbalance_strong=0.000000 max_spread_tight=0.000000 max_volatility_extreme=None vol_quantile_reject=0.012000

| rank | symbol | rule | horizon | min_imb | min_int | max_spread | score | robust_core | robust_stress | pass_rate_core | pass_rate_stress | npa_core | npa_stress | attempt_fill_rate | attempts_per_min | score_raw_core | score_raw_stress | score_raw_min | baseline_pass_rate_core | baseline_pass_rate_stress | baseline_npa_core | baseline_score_raw_core | delta_pass_rate_core | delta_npa_core | delta_score_raw_core | stability_std_bps | best_fee_survive | insufficient_fill_rate |
|---:|---|---|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.50 | 2500 | 0.000300 | 0.000000e+00 | False | False | 38.89% | 38.89% | -9.349476e-05 | -9.349476e-05 | 60.46% | 0.05 | -0.00016951 | -0.00016951 | -0.00016951 | 33.33% | 33.33% | -8.844979e-05 | -0.00015935 | +5.56% | -5.044979e-06 | -0.00001016 | 0.829 | 1.00 | 0.00% |
| 2 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.50 | 2500 | 0.000500 | 0.000000e+00 | False | False | 38.89% | 38.89% | -1.012048e-04 | -1.012048e-04 | 54.90% | 0.06 | -0.00019765 | -0.00019765 | -0.00019765 | 33.33% | 33.33% | -9.758509e-05 | -0.00018967 | +5.56% | -3.619727e-06 | -0.00000799 | 0.737 | 1.00 | 0.00% |
| 3 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.50 | 3500 | 0.000300 | 0.000000e+00 | False | False | 27.78% | 27.78% | -7.736525e-05 | -7.736525e-05 | 62.47% | 0.02 | -0.00012979 | -0.00012979 | -0.00012979 | 27.78% | 27.78% | -7.529967e-05 | -0.00012693 | +0.00% | -2.065578e-06 | -0.00000286 | 1.174 | 1.00 | 0.00% |
| 4 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.40 | 2500 | 0.000500 | 0.000000e+00 | False | False | 27.78% | 27.78% | -1.190772e-04 | -1.190772e-04 | 55.38% | 0.06 | -0.00022003 | -0.00022003 | -0.00022003 | 27.78% | 27.78% | -1.064601e-04 | -0.00019428 | +0.00% | -1.261703e-05 | -0.00002575 | 0.589 | 1.00 | 0.00% |
| 5 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.40 | 2500 | 0.000300 | 0.000000e+00 | False | False | 11.11% | 11.11% | -1.075920e-04 | -1.075920e-04 | 59.59% | 0.05 | -0.00018906 | -0.00018906 | -0.00018906 | 5.56% | 5.56% | -9.674117e-05 | -0.00017127 | +5.56% | -1.085087e-05 | -0.00001779 | 0.687 | 1.00 | 0.00% |
| 6 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.20 | 2500 | 0.000500 | 0.000000e+00 | False | False | 11.11% | 11.11% | -1.308798e-04 | -1.308798e-04 | 53.84% | 0.07 | -0.00024149 | -0.00024149 | -0.00024149 | 5.56% | 5.56% | -1.351611e-04 | -0.00024763 | +5.56% | +4.281324e-06 | +0.00000614 | 0.593 | 1.00 | 0.00% |
| 7 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.30 | 2500 | 0.000500 | 0.000000e+00 | False | False | 11.11% | 11.11% | -1.337462e-04 | -1.337462e-04 | 56.51% | 0.07 | -0.00023732 | -0.00023732 | -0.00023732 | 5.56% | 5.56% | -1.385739e-04 | -0.00024354 | +5.56% | +4.827617e-06 | +0.00000622 | 0.574 | 1.00 | 0.00% |

survive_fee1_passrate_ge_0.5=0
