# PASSIVE_POCKET_RANKING

candidates=7 ranked=7
candidate_parse total_rows_seen=26 table_rows_seen=22 rows_with_pass_yes=7 candidates_parsed=7 candidates_unique=7 rows_skipped_missing_fields=0
fee_grid=[0.0] adverse_mult_grid=[0.5]
pass_threshold=0.500
mitigation_profile=baseline gate_config min_intensity_strong=0.000000 min_imbalance_strong=0.000000 max_spread_tight=0.000000 max_volatility_extreme=None vol_quantile_reject=0.010000 scratch_bps=0.0000 scratch_window_sec=0 scratch_taker_fee_bps=0.0000 scratch_slippage_bps=0.0000 horizon_sec_override=120

| rank | symbol | rule | horizon | min_imb | min_int | max_spread | score | robust_core | robust_stress | pass_rate_core | pass_rate_stress | npa_core | npa_stress | attempt_fill_rate | attempts_per_min | score_raw_core | score_raw_stress | score_raw_min | baseline_pass_rate_core | baseline_pass_rate_stress | baseline_npa_core | baseline_score_raw_core | delta_pass_rate_core | delta_npa_core | delta_score_raw_core | failure_reason_top | gate_reject_ratio | fill_rate_after_gate | avg_fee_bps | avg_adverse_bps_on_fills | avg_raw_return_bps_on_fills | avg_net_return_bps_on_fills | stability_std_bps | best_fee_survive | insufficient_fill_rate |
|---:|---|---|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.40 | 2500 | 0.000500 | 6.871863e-01 | True | True | 61.11% | 61.11% | +1.157264e-04 | +1.157264e-04 | 56.78% | 0.04 | +0.00020216 | +0.00020216 | +0.00020216 | 61.11% | 61.11% | +1.157264e-04 | +0.00020216 | +0.00% | +0.000000e+00 | +0.00000000 | mixed | 0.00% | 56.60% | 0.000 | 0.138 | +1.914 | +1.775 | 0.684 | 0.00 | 0.00% |
| 2 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.50 | 3500 | 0.000300 | 6.309460e-01 | True | True | 77.78% | 77.78% | +1.730265e-04 | +1.730265e-04 | 60.24% | 0.02 | +0.00029535 | +0.00029535 | +0.00029535 | 77.78% | 77.78% | +1.730265e-04 | +0.00029535 | +0.00% | +0.000000e+00 | +0.00000000 | mixed | 0.00% | 61.46% | 0.000 | 0.190 | +2.350 | +2.160 | 1.194 | 0.00 | 50.00% |
| 3 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.50 | 2500 | 0.000500 | 5.062571e-01 | True | True | 50.00% | 50.00% | +8.927965e-05 | +8.927965e-05 | 55.35% | 0.04 | +0.00016137 | +0.00016137 | +0.00016137 | 50.00% | 50.00% | +8.927965e-05 | +0.00016137 | +0.00% | +0.000000e+00 | +0.00000000 | mixed | 0.00% | 55.41% | 0.000 | 0.142 | +1.691 | +1.549 | 0.764 | 0.00 | 0.00% |
| 4 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.40 | 2500 | 0.000300 | 4.664530e-01 | True | True | 66.67% | 66.67% | +8.169993e-05 | +8.169993e-05 | 61.04% | 0.03 | +0.00013455 | +0.00013455 | +0.00013455 | 66.67% | 66.67% | +8.169993e-05 | +0.00013455 | +0.00% | +0.000000e+00 | +0.00000000 | mixed | 0.00% | 61.11% | 0.000 | 0.146 | +1.738 | +1.592 | 0.752 | 0.00 | 0.00% |
| 5 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.30 | 2500 | 0.000500 | 4.308316e-01 | True | True | 50.00% | 50.00% | +8.343783e-05 | +8.343783e-05 | 56.69% | 0.04 | +0.00014498 | +0.00014498 | +0.00014498 | 50.00% | 50.00% | +8.343783e-05 | +0.00014498 | +0.00% | +0.000000e+00 | +0.00000000 | mixed | 0.00% | 57.02% | 0.000 | 0.136 | +1.744 | +1.608 | 0.937 | 0.00 | 0.00% |
| 6 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.20 | 2500 | 0.000500 | 3.996139e-01 | True | True | 50.00% | 50.00% | +7.877616e-05 | +7.877616e-05 | 54.31% | 0.05 | +0.00014654 | +0.00014654 | +0.00014654 | 50.00% | 50.00% | +7.877616e-05 | +0.00014654 | +0.00% | +0.000000e+00 | +0.00000000 | mixed | 0.00% | 54.28% | 0.000 | 0.139 | +1.707 | +1.569 | 0.971 | 0.00 | 0.00% |
| 7 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.50 | 2500 | 0.000300 | 3.934119e-01 | True | True | 55.56% | 55.56% | +7.495784e-05 | +7.495784e-05 | 59.55% | 0.03 | +0.00012564 | +0.00012564 | +0.00012564 | 55.56% | 55.56% | +7.495784e-05 | +0.00012564 | +0.00% | +0.000000e+00 | +0.00000000 | mixed | 0.00% | 60.10% | 0.000 | 0.146 | +1.637 | +1.491 | 0.905 | 0.00 | 0.00% |

survive_fee1_passrate_ge_0.5=0

## Decomposition

| rank | symbol | rule | h | gross_edge_npa | fee_cost_npa | adverse_cost_npa | scratch_cost_npa | net_npa | observed_npa | residual_npa | reject_rate | n_events | n_after_gate | n_filled |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | +1.083210e-04 | +0.000000e+00 | +7.835094e-06 | +0.000000e+00 | +1.004859e-04 | +1.157264e-04 | +1.524055e-05 | 0.00% | 2509 | 2509 | 1420 |
| 2 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | +1.444335e-04 | +0.000000e+00 | +1.170361e-05 | +0.000000e+00 | +1.327299e-04 | +1.730265e-04 | +4.029659e-05 | 0.00% | 1012 | 1012 | 622 |
| 3 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | +9.371007e-05 | +0.000000e+00 | +7.864233e-06 | +0.000000e+00 | +8.584583e-05 | +8.927965e-05 | +3.433813e-06 | 0.00% | 2254 | 2254 | 1249 |
| 4 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | +1.062124e-04 | +0.000000e+00 | +8.905193e-06 | +0.000000e+00 | +9.730719e-05 | +8.169993e-05 | -1.560726e-05 | 0.00% | 2029 | 2029 | 1240 |
| 5 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | +9.945466e-05 | +0.000000e+00 | +7.773260e-06 | +0.000000e+00 | +9.168140e-05 | +8.343783e-05 | -8.243566e-06 | 0.00% | 2664 | 2664 | 1519 |
| 6 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | +9.268577e-05 | +0.000000e+00 | +7.539706e-06 | +0.000000e+00 | +8.514606e-05 | +7.877616e-05 | -6.369897e-06 | 0.00% | 2861 | 2861 | 1553 |
| 7 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | +9.837360e-05 | +0.000000e+00 | +8.792997e-06 | +0.000000e+00 | +8.958060e-05 | +7.495784e-05 | -1.462276e-05 | 0.00% | 1802 | 1802 | 1083 |
