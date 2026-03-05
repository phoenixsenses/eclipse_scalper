# PASSIVE_POCKET_RANKING

candidates=7 ranked=7
candidate_parse total_rows_seen=26 table_rows_seen=22 rows_with_pass_yes=7 candidates_parsed=7 candidates_unique=7 rows_skipped_missing_fields=0
fee_grid=[0.2, 0.5, 1.0] adverse_mult_grid=[1.0, 1.2, 1.5]
pass_threshold=0.330

| rank | symbol | rule | horizon | min_imb | min_int | max_spread | score | robust_core | robust_stress | pass_rate_core | pass_rate_stress | npa_core | npa_stress | attempt_fill_rate | attempts_per_min | score_raw_core | score_raw_stress | score_raw_min | baseline_pass_rate_core | baseline_pass_rate_stress | baseline_npa_core | baseline_score_raw_core | delta_pass_rate_core | delta_npa_core | delta_score_raw_core | stability_std_bps | best_fee_survive | insufficient_fill_rate |
|---:|---|---|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.50 | 3500 | 0.000300 | 0.000000e+00 | False | False | 50.00% | 50.00% | -8.104493e-05 | -9.150547e-05 | 65.78% | 0.02 | +0.00000770 | -0.00017018 | -0.00017018 | 50.00% | 50.00% | -8.104493e-05 | +0.00000770 | +0.00% | +0.000000e+00 | +0.00000000 | 1.248 | 1.00 | 0.00% |
| 2 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.40 | 2500 | 0.000300 | 0.000000e+00 | False | False | 44.44% | 44.44% | -8.038518e-05 | -8.808825e-05 | 62.83% | 0.05 | +0.00001530 | -0.00015816 | -0.00015816 | 44.44% | 44.44% | -8.038518e-05 | +0.00001530 | +0.00% | +0.000000e+00 | +0.00000000 | 0.896 | 1.00 | 0.00% |
| 3 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.50 | 2500 | 0.000300 | 0.000000e+00 | False | False | 44.44% | 44.44% | -9.448819e-05 | -1.020637e-04 | 61.97% | 0.04 | -0.00001820 | -0.00019160 | -0.00019160 | 44.44% | 44.44% | -9.448819e-05 | -0.00001820 | +0.00% | +0.000000e+00 | +0.00000000 | 0.906 | 1.00 | 0.00% |
| 4 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.40 | 2500 | 0.000500 | 0.000000e+00 | False | False | 44.44% | 44.44% | -1.259854e-04 | -1.328246e-04 | 57.23% | 0.06 | -0.00007501 | -0.00024800 | -0.00024800 | 44.44% | 44.44% | -1.259854e-04 | -0.00007501 | +0.00% | +0.000000e+00 | +0.00000000 | 0.955 | 1.00 | 0.00% |
| 5 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.50 | 2500 | 0.000500 | 0.000000e+00 | False | False | 44.44% | 44.44% | -1.424343e-04 | -1.489465e-04 | 57.65% | 0.05 | -0.00011331 | -0.00028585 | -0.00028585 | 44.44% | 44.44% | -1.424343e-04 | -0.00011331 | +0.00% | +0.000000e+00 | +0.00000000 | 0.951 | 1.00 | 0.00% |
| 6 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.30 | 2500 | 0.000500 | 0.000000e+00 | False | False | 38.89% | 38.89% | -1.269476e-04 | -1.339030e-04 | 58.21% | 0.06 | -0.00007528 | -0.00024803 | -0.00024803 | 38.89% | 38.89% | -1.269476e-04 | -0.00007528 | +0.00% | +0.000000e+00 | +0.00000000 | 0.850 | 1.00 | 0.00% |
| 7 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.20 | 2500 | 0.000500 | 0.000000e+00 | False | False | 38.89% | 38.89% | -1.283386e-04 | -1.354224e-04 | 56.20% | 0.06 | -0.00007826 | -0.00025139 | -0.00025139 | 38.89% | 38.89% | -1.283386e-04 | -0.00007826 | +0.00% | +0.000000e+00 | +0.00000000 | 0.809 | 1.00 | 0.00% |

survive_fee1_passrate_ge_0.5=1
