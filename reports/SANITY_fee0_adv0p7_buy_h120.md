# PASSIVE_POCKET_RANKING

candidates=7 ranked=7
candidate_parse total_rows_seen=26 table_rows_seen=22 rows_with_pass_yes=7 candidates_parsed=7 candidates_unique=7 rows_skipped_missing_fields=0
fee_grid=[0.0] adverse_mult_grid=[0.7]
pass_threshold=0.500
mitigation_profile=baseline gate_config min_intensity_strong=0.000000 min_imbalance_strong=0.000000 max_spread_tight=0.000000 max_volatility_extreme=None vol_quantile_reject=0.010000 scratch_bps=0.0000 scratch_window_sec=0 scratch_taker_fee_bps=0.0000 scratch_slippage_bps=0.0000 horizon_sec_override=120

| rank | symbol | rule | horizon | min_imb | min_int | max_spread | score | robust_core | robust_stress | pass_rate_core | pass_rate_stress | npa_core | npa_stress | attempt_fill_rate | attempts_per_min | score_raw_core | score_raw_stress | score_raw_min | baseline_pass_rate_core | baseline_pass_rate_stress | baseline_npa_core | baseline_score_raw_core | delta_pass_rate_core | delta_npa_core | delta_score_raw_core | failure_reason_top | gate_reject_ratio | fill_rate_after_gate | avg_fee_bps | avg_adverse_bps_on_fills | avg_raw_return_bps_on_fills | avg_net_return_bps_on_fills | stability_std_bps | best_fee_survive | insufficient_fill_rate |
|---:|---|---|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.40 | 2500 | 0.000500 | 6.049931e-02 | True | True | 50.00% | 50.00% | +9.306249e-06 | +9.306249e-06 | 55.78% | 0.08 | +0.00001493 | +0.00001493 | +0.00001493 | 50.00% | 50.00% | +9.306249e-06 | +0.00001493 | +0.00% | +0.000000e+00 | +0.00000000 | mixed | 0.00% | 56.15% | 0.000 | 0.192 | +0.194 | +0.002 | 0.538 | 0.00 | 0.00% |
| 2 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.40 | 2500 | 0.000300 | 0.000000e+00 | False | False | 50.00% | 50.00% | -3.105734e-06 | -3.105734e-06 | 60.90% | 0.07 | -0.00000658 | -0.00000658 | -0.00000658 | 50.00% | 50.00% | -3.105734e-06 | -0.00000658 | +0.00% | +0.000000e+00 | +0.00000000 | adverse_dominates | 0.00% | 60.77% | 0.000 | 0.201 | -0.119 | -0.320 | 0.552 | 0.00 | 0.00% |
| 3 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.50 | 2500 | 0.000500 | 0.000000e+00 | False | False | 50.00% | 50.00% | -8.610664e-06 | -8.610664e-06 | 55.95% | 0.07 | -0.00002085 | -0.00002085 | -0.00002085 | 50.00% | 50.00% | -8.610664e-06 | -0.00002085 | +0.00% | +0.000000e+00 | +0.00000000 | adverse_dominates | 0.00% | 55.62% | 0.000 | 0.196 | +0.173 | -0.023 | 0.549 | 0.00 | 0.00% |
| 4 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.20 | 2500 | 0.000500 | 0.000000e+00 | False | False | 50.00% | 50.00% | -8.785869e-06 | -8.785869e-06 | 54.85% | 0.09 | -0.00001699 | -0.00001699 | -0.00001699 | 50.00% | 50.00% | -8.785869e-06 | -0.00001699 | +0.00% | +0.000000e+00 | +0.00000000 | adverse_dominates | 0.00% | 54.73% | 0.000 | 0.193 | +0.164 | -0.029 | 0.614 | 0.00 | 0.00% |
| 5 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.30 | 2500 | 0.000500 | 0.000000e+00 | False | False | 50.00% | 50.00% | -1.178854e-05 | -1.178854e-05 | 56.76% | 0.09 | -0.00002236 | -0.00002236 | -0.00002236 | 50.00% | 50.00% | -1.178854e-05 | -0.00002236 | +0.00% | +0.000000e+00 | +0.00000000 | adverse_dominates | 0.00% | 56.98% | 0.000 | 0.190 | +0.088 | -0.103 | 0.580 | 0.00 | 0.00% |
| 6 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.50 | 3500 | 0.000300 | 0.000000e+00 | False | False | 50.00% | 50.00% | -2.472401e-05 | -2.472401e-05 | 63.19% | 0.03 | -0.00004053 | -0.00004053 | -0.00004053 | 50.00% | 50.00% | -2.472401e-05 | -0.00004053 | +0.00% | +0.000000e+00 | +0.00000000 | adverse_dominates | 0.00% | 63.91% | 0.000 | 0.250 | -0.038 | -0.287 | 0.992 | 0.00 | 0.00% |
| 7 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.50 | 2500 | 0.000300 | 0.000000e+00 | False | False | 50.00% | 50.00% | -3.001918e-05 | -3.001918e-05 | 60.26% | 0.06 | -0.00005244 | -0.00005244 | -0.00005244 | 50.00% | 50.00% | -3.001918e-05 | -0.00005244 | +0.00% | +0.000000e+00 | +0.00000000 | adverse_dominates | 0.00% | 60.25% | 0.000 | 0.203 | -0.202 | -0.405 | 0.674 | 0.00 | 0.00% |

survive_fee1_passrate_ge_0.5=0

## Decomposition

| rank | symbol | rule | h | gross_edge_npa | fee_cost_npa | adverse_cost_npa | scratch_cost_npa | net_npa | observed_npa | residual_npa | reject_rate | n_events | n_after_gate | n_filled |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | +1.091339e-05 | +0.000000e+00 | +1.080871e-05 | +0.000000e+00 | +1.046792e-07 | +9.306249e-06 | +9.201570e-06 | 0.00% | 4691 | 4691 | 2634 |
| 2 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | -7.245913e-06 | +0.000000e+00 | +1.219381e-05 | +0.000000e+00 | -1.943973e-05 | -3.105734e-06 | +1.633399e-05 | 0.00% | 3752 | 3752 | 2280 |
| 3 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | +9.628958e-06 | +0.000000e+00 | +1.091653e-05 | +0.000000e+00 | -1.287568e-06 | -8.610664e-06 | -7.323097e-06 | 0.00% | 4241 | 4241 | 2359 |
| 4 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | +8.999900e-06 | +0.000000e+00 | +1.058149e-05 | +0.000000e+00 | -1.581593e-06 | -8.785869e-06 | -7.204276e-06 | 0.00% | 5396 | 5396 | 2953 |
| 5 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | +4.995682e-06 | +0.000000e+00 | +1.085127e-05 | +0.000000e+00 | -5.855585e-06 | -1.178854e-05 | -5.932952e-06 | 0.00% | 5009 | 5009 | 2854 |
| 6 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | -2.399858e-06 | +0.000000e+00 | +1.596970e-05 | +0.000000e+00 | -1.836956e-05 | -2.472401e-05 | -6.354456e-06 | 0.00% | 1884 | 1884 | 1204 |
| 7 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | -1.218583e-05 | +0.000000e+00 | +1.220938e-05 | +0.000000e+00 | -2.439521e-05 | -3.001918e-05 | -5.623975e-06 | 0.00% | 3381 | 3381 | 2037 |
