# PASSIVE_POCKET_RANKING

candidates=7 ranked=7
candidate_parse total_rows_seen=26 table_rows_seen=22 rows_with_pass_yes=7 candidates_parsed=7 candidates_unique=7 rows_skipped_missing_fields=0
fee_grid=[0.2, 0.5, 1.0] adverse_mult_grid=[1.0, 1.2, 1.5]
pass_threshold=0.330

| rank | symbol | rule | horizon | min_imb | min_int | max_spread | score | robust_core | robust_stress | pass_rate_core | pass_rate_stress | npa_core | npa_stress | attempt_fill_rate | attempts_per_min | score_raw_core | score_raw_stress | score_raw_min | baseline_pass_rate_core | baseline_pass_rate_stress | baseline_npa_core | baseline_score_raw_core | delta_pass_rate_core | delta_npa_core | delta_score_raw_core | stability_std_bps | best_fee_survive | insufficient_fill_rate |
|---:|---|---|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.50 | 3500 | 0.000300 | 0.000000e+00 | False | False | 50.00% | 50.00% | -3.363012e-05 | -4.479814e-05 | 62.36% | 0.02 | +0.00010205 | -0.00007554 | -0.00007554 | 50.00% | 50.00% | -3.363012e-05 | +0.00010205 | +0.00% | +0.000000e+00 | +0.00000000 | 0.816 | 1.00 | 0.00% |
| 2 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.50 | 2500 | 0.000500 | 0.000000e+00 | False | False | 50.00% | 50.00% | -4.688906e-05 | -5.355986e-05 | 54.97% | 0.05 | +0.00006476 | -0.00010801 | -0.00010801 | 50.00% | 50.00% | -4.688906e-05 | +0.00006476 | +0.00% | +0.000000e+00 | +0.00000000 | 0.903 | 1.00 | 0.00% |
| 3 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.50 | 2500 | 0.000300 | 0.000000e+00 | False | False | 50.00% | 50.00% | -7.028222e-05 | -7.775494e-05 | 60.45% | 0.04 | +0.00002540 | -0.00014796 | -0.00014796 | 50.00% | 50.00% | -7.028222e-05 | +0.00002540 | +0.00% | +0.000000e+00 | +0.00000000 | 0.635 | 1.00 | 0.00% |
| 4 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.40 | 2500 | 0.000300 | 0.000000e+00 | False | False | 38.89% | 38.89% | -6.322793e-05 | -7.114339e-05 | 59.93% | 0.04 | +0.00004661 | -0.00012706 | -0.00012706 | 38.89% | 38.89% | -6.322793e-05 | +0.00004661 | +0.00% | +0.000000e+00 | +0.00000000 | 0.879 | 1.00 | 0.00% |
| 5 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.40 | 2500 | 0.000500 | 0.000000e+00 | False | False | 38.89% | 33.33% | -8.547862e-05 | -9.267601e-05 | 56.62% | 0.06 | -0.00000093 | -0.00017423 | -0.00017423 | 38.89% | 33.33% | -8.547862e-05 | -0.00000093 | +0.00% | +0.000000e+00 | +0.00000000 | 1.023 | 1.00 | 0.00% |
| 6 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.30 | 2500 | 0.000500 | 0.000000e+00 | False | False | 33.33% | 27.78% | -9.416610e-05 | -1.013065e-04 | 56.91% | 0.06 | -0.00001386 | -0.00018686 | -0.00018686 | 33.33% | 27.78% | -9.416610e-05 | -0.00001386 | +0.00% | +0.000000e+00 | +0.00000000 | 1.067 | 1.00 | 0.00% |
| 7 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.20 | 2500 | 0.000500 | 0.000000e+00 | False | False | 27.78% | 27.78% | -9.554745e-05 | -1.027151e-04 | 54.97% | 0.06 | -0.00001886 | -0.00019221 | -0.00019221 | 27.78% | 27.78% | -9.554745e-05 | -0.00001886 | +0.00% | +0.000000e+00 | +0.00000000 | 1.017 | 1.00 | 0.00% |

survive_fee1_passrate_ge_0.5=3
