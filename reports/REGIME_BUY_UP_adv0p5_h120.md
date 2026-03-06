# PASSIVE_POCKET_RANKING

candidates=7 ranked=7
candidate_parse total_rows_seen=26 table_rows_seen=22 rows_with_pass_yes=7 candidates_parsed=7 candidates_unique=7 rows_skipped_missing_fields=0
fee_grid=[0.0] adverse_mult_grid=[0.5]
pass_threshold=0.500
mitigation_profile=baseline gate_config min_intensity_strong=0.000000 min_imbalance_strong=0.000000 max_spread_tight=0.000000 max_volatility_extreme=None vol_quantile_reject=0.010000 scratch_bps=0.0000 scratch_window_sec=0 scratch_taker_fee_bps=0.0000 scratch_slippage_bps=0.0000 horizon_sec_override=120

| rank | symbol | rule | horizon | min_imb | min_int | max_spread | score | robust_core | robust_stress | pass_rate_core | pass_rate_stress | npa_core | npa_stress | attempt_fill_rate | attempts_per_min | score_raw_core | score_raw_stress | score_raw_min | baseline_pass_rate_core | baseline_pass_rate_stress | baseline_npa_core | baseline_score_raw_core | delta_pass_rate_core | delta_npa_core | delta_score_raw_core | failure_reason_top | gate_reject_ratio | fill_rate_after_gate | avg_fee_bps | avg_adverse_bps_on_fills | avg_raw_return_bps_on_fills | avg_net_return_bps_on_fills | stability_std_bps | best_fee_survive | insufficient_fill_rate |
|---:|---|---|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.50 | 2500 | 0.000300 | 7.587735e-01 | True | True | 55.56% | 55.56% | +1.166362e-04 | +1.166362e-04 | 61.06% | 0.03 | +0.00019054 | +0.00019054 | +0.00019054 | 55.56% | 55.56% | +1.166362e-04 | +0.00019054 | +0.00% | +0.000000e+00 | +0.00000000 | mixed | 0.00% | 60.11% | 0.000 | 0.146 | +1.885 | +1.739 | 0.537 | 0.00 | 0.00% |
| 2 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.30 | 2500 | 0.000500 | 6.789175e-01 | True | True | 61.11% | 61.11% | +1.158883e-04 | +1.158883e-04 | 58.39% | 0.04 | +0.00020031 | +0.00020031 | +0.00020031 | 61.11% | 61.11% | +1.158883e-04 | +0.00020031 | +0.00% | +0.000000e+00 | +0.00000000 | mixed | 0.00% | 58.03% | 0.000 | 0.136 | +2.064 | +1.928 | 0.707 | 0.00 | 0.00% |
| 3 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.50 | 2500 | 0.000500 | 6.328907e-01 | True | True | 55.56% | 55.56% | +9.943637e-05 | +9.943637e-05 | 58.16% | 0.04 | +0.00018667 | +0.00018667 | +0.00018667 | 55.56% | 55.56% | +9.943637e-05 | +0.00018667 | +0.00% | +0.000000e+00 | +0.00000000 | mixed | 0.00% | 56.06% | 0.000 | 0.142 | +1.865 | +1.723 | 0.571 | 0.00 | 0.00% |
| 4 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.20 | 2500 | 0.000500 | 6.014400e-01 | True | True | 61.11% | 61.11% | +1.032197e-04 | +1.032197e-04 | 56.08% | 0.05 | +0.00018822 | +0.00018822 | +0.00018822 | 61.11% | 61.11% | +1.032197e-04 | +0.00018822 | +0.00% | +0.000000e+00 | +0.00000000 | mixed | 0.00% | 55.12% | 0.000 | 0.139 | +1.987 | +1.848 | 0.716 | 0.00 | 0.00% |
| 5 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.40 | 2500 | 0.000500 | 5.981164e-01 | True | True | 72.22% | 72.22% | +1.049453e-04 | +1.049453e-04 | 59.49% | 0.04 | +0.00018006 | +0.00018006 | +0.00018006 | 72.22% | 72.22% | +1.049453e-04 | +0.00018006 | +0.00% | +0.000000e+00 | +0.00000000 | mixed | 0.00% | 57.68% | 0.000 | 0.138 | +1.979 | +1.840 | 0.755 | 0.00 | 0.00% |
| 6 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.40 | 2500 | 0.000300 | 5.742867e-01 | True | True | 61.11% | 61.11% | +9.524901e-05 | +9.524901e-05 | 61.47% | 0.03 | +0.00015617 | +0.00015617 | +0.00015617 | 61.11% | 61.11% | +9.524901e-05 | +0.00015617 | +0.00% | +0.000000e+00 | +0.00000000 | mixed | 0.00% | 60.93% | 0.000 | 0.145 | +1.776 | +1.631 | 0.659 | 0.00 | 0.00% |
| 7 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.50 | 3500 | 0.000300 | 4.139474e-01 | True | True | 61.11% | 61.11% | +1.201338e-04 | +1.201338e-04 | 60.80% | 0.02 | +0.00019546 | +0.00019546 | +0.00019546 | 61.11% | 61.11% | +1.201338e-04 | +0.00019546 | +0.00% | +0.000000e+00 | +0.00000000 | mixed | 0.00% | 60.69% | 0.000 | 0.189 | +2.404 | +2.214 | 1.322 | 0.00 | 50.00% |

survive_fee1_passrate_ge_0.5=0

## Decomposition

| rank | symbol | rule | h | gross_edge_npa | fee_cost_npa | adverse_cost_npa | scratch_cost_npa | net_npa | observed_npa | residual_npa | reject_rate | n_events | n_after_gate | n_filled |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | +1.133123e-04 | +0.000000e+00 | +8.788470e-06 | +0.000000e+00 | +1.045238e-04 | +1.166362e-04 | +1.211241e-05 | 0.00% | 1760 | 1760 | 1058 |
| 2 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | +1.198046e-04 | +0.000000e+00 | +7.902809e-06 | +0.000000e+00 | +1.119018e-04 | +1.158883e-04 | +3.986570e-06 | 0.00% | 2621 | 2621 | 1521 |
| 3 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | +1.045367e-04 | +0.000000e+00 | +7.953075e-06 | +0.000000e+00 | +9.658360e-05 | +9.943637e-05 | +2.852769e-06 | 0.00% | 2210 | 2210 | 1239 |
| 4 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | +1.095080e-04 | +0.000000e+00 | +7.645774e-06 | +0.000000e+00 | +1.018623e-04 | +1.032197e-04 | +1.357461e-06 | 0.00% | 2821 | 2821 | 1555 |
| 5 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | +1.141314e-04 | +0.000000e+00 | +7.977462e-06 | +0.000000e+00 | +1.061539e-04 | +1.049453e-04 | -1.208560e-06 | 0.00% | 2455 | 2455 | 1416 |
| 6 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | +1.082007e-04 | +0.000000e+00 | +8.835270e-06 | +0.000000e+00 | +9.936541e-05 | +9.524901e-05 | -4.116392e-06 | 0.00% | 1994 | 1994 | 1215 |
| 7 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | +1.458888e-04 | +0.000000e+00 | +1.149624e-05 | +0.000000e+00 | +1.343926e-04 | +1.201338e-04 | -1.425875e-05 | 0.00% | 987 | 987 | 599 |
