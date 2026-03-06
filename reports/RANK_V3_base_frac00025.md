# PASSIVE_POCKET_RANKING

candidates=7 ranked=7
candidate_parse total_rows_seen=26 table_rows_seen=22 rows_with_pass_yes=7 candidates_parsed=7 candidates_unique=7 rows_skipped_missing_fields=0
fee_grid=[0.2, 0.5, 1.0] adverse_mult_grid=[1.0, 1.2, 1.5]
pass_threshold=0.330

| rank | symbol | rule | horizon | min_imb | min_int | max_spread | score | robust_core | robust_stress | pass_rate_core | pass_rate_stress | npa_core | npa_stress | attempt_fill_rate | attempts_per_min | score_raw_core | score_raw_stress | score_raw_min | stability_std_bps | best_fee_survive | insufficient_fill_rate |
|---:|---|---|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.40 | 2500 | 0.000300 | 0.000000e+00 | False | False | 50.00% | 50.00% | -7.749440e-06 | -1.515512e-05 | 59.67% | 0.05 | +0.00013971 | -0.00003339 | -0.00003339 | 0.768 | 1.00 | 0.00% |
| 2 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.50 | 2500 | 0.000500 | 0.000000e+00 | False | False | 50.00% | 50.00% | -4.754863e-05 | -5.459595e-05 | 55.62% | 0.05 | +0.00007731 | -0.00009505 | -0.00009505 | 0.761 | 1.00 | 0.00% |
| 3 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.50 | 3500 | 0.000300 | 0.000000e+00 | False | False | 50.00% | 50.00% | -5.159090e-05 | -6.331445e-05 | 65.48% | 0.02 | +0.00008064 | -0.00009699 | -0.00009699 | 0.765 | 1.00 | 50.00% |
| 4 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.50 | 2500 | 0.000300 | 0.000000e+00 | False | False | 50.00% | 50.00% | -6.077581e-05 | -6.904622e-05 | 61.05% | 0.04 | +0.00006929 | -0.00010399 | -0.00010399 | 0.700 | 1.00 | 11.11% |
| 5 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.40 | 2500 | 0.000500 | 0.000000e+00 | False | False | 38.89% | 33.33% | -4.995533e-05 | -5.699263e-05 | 56.57% | 0.06 | +0.00006552 | -0.00010748 | -0.00010748 | 0.680 | 1.00 | 0.00% |
| 6 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.30 | 2500 | 0.000500 | 0.000000e+00 | False | False | 27.78% | 22.22% | -5.359534e-05 | -6.072014e-05 | 56.70% | 0.06 | +0.00006298 | -0.00010981 | -0.00010981 | 0.637 | 1.00 | 0.00% |
| 7 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.20 | 2500 | 0.000500 | 0.000000e+00 | False | False | 22.22% | 22.22% | -5.563962e-05 | -6.283211e-05 | 54.84% | 0.07 | +0.00005844 | -0.00011467 | -0.00011467 | 0.596 | 1.00 | 0.00% |

survive_fee1_passrate_ge_0.5=4
