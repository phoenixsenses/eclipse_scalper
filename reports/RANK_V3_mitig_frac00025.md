# PASSIVE_POCKET_RANKING

candidates=7 ranked=6
candidate_parse total_rows_seen=26 table_rows_seen=22 rows_with_pass_yes=7 candidates_parsed=7 candidates_unique=7 rows_skipped_missing_fields=0
fee_grid=[0.2, 0.5, 1.0] adverse_mult_grid=[1.0, 1.2, 1.5]
pass_threshold=0.330

| rank | symbol | rule | horizon | min_imb | min_int | max_spread | score | robust_core | robust_stress | pass_rate_core | pass_rate_stress | npa_core | npa_stress | attempt_fill_rate | attempts_per_min | score_raw_core | score_raw_stress | score_raw_min | stability_std_bps | best_fee_survive | insufficient_fill_rate |
|---:|---|---|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.40 | 2500 | 0.000300 | 3.725741e-02 | True | True | 50.00% | 50.00% | +9.767250e-06 | +1.281593e-06 | 64.29% | 0.03 | +0.00017679 | +0.00000367 | +0.00000367 | 1.097 | 1.00 | 50.00% |
| 2 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.40 | 2500 | 0.000500 | 0.000000e+00 | False | False | 50.00% | 50.00% | -1.708248e-05 | -2.451294e-05 | 56.98% | 0.05 | +0.00013583 | -0.00003644 | -0.00003644 | 0.832 | 1.00 | 11.11% |
| 3 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.50 | 2500 | 0.000500 | 0.000000e+00 | False | False | 50.00% | 50.00% | -2.144890e-05 | -2.842803e-05 | 56.10% | 0.04 | +0.00013033 | -0.00004154 | -0.00004154 | 0.770 | 1.00 | 38.89% |
| 4 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.50 | 2500 | 0.000300 | 0.000000e+00 | False | False | 50.00% | 50.00% | -2.279704e-05 | -3.102358e-05 | 63.60% | 0.03 | +0.00011511 | -0.00005781 | -0.00005781 | 0.904 | 1.00 | 50.00% |
| 5 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.30 | 2500 | 0.000500 | 0.000000e+00 | False | False | 38.89% | 33.33% | -2.845908e-05 | -3.551098e-05 | 57.51% | 0.05 | +0.00011235 | -0.00005944 | -0.00005944 | 0.877 | 1.00 | 0.00% |
| 6 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.20 | 2500 | 0.000500 | 0.000000e+00 | False | False | 27.78% | 22.22% | -5.035279e-05 | -5.813984e-05 | 56.27% | 0.06 | +0.00006541 | -0.00010923 | -0.00010923 | 0.936 | 1.00 | 0.00% |

survive_fee1_passrate_ge_0.5=4
