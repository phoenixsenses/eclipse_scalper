# PASSIVE_POCKET_RANKING

candidates=7 ranked=7
candidate_parse total_rows_seen=26 table_rows_seen=22 rows_with_pass_yes=7 candidates_parsed=7 candidates_unique=7 rows_skipped_missing_fields=0
fee_grid=[0.8] adverse_mult_grid=[0.5]
pass_threshold=0.500
mitigation_profile=baseline gate_config min_intensity_strong=0.000000 min_imbalance_strong=0.000000 max_spread_tight=0.000000 max_volatility_extreme=None vol_quantile_reject=0.010000 scratch_bps=0.0000 scratch_window_sec=0 scratch_taker_fee_bps=0.0000 scratch_slippage_bps=0.0000 horizon_sec_override=120

| rank | symbol | rule | horizon | min_imb | min_int | max_spread | score | robust_core | robust_stress | pass_rate_core | pass_rate_stress | npa_core | npa_stress | attempt_fill_rate | attempts_per_min | score_raw_core | score_raw_stress | score_raw_min | baseline_pass_rate_core | baseline_pass_rate_stress | baseline_npa_core | baseline_score_raw_core | delta_pass_rate_core | delta_npa_core | delta_score_raw_core | failure_reason_top | gate_reject_ratio | fill_rate_after_gate | avg_fee_bps | avg_adverse_bps_on_fills | avg_raw_return_bps_on_fills | avg_net_return_bps_on_fills | stability_std_bps | best_fee_survive | insufficient_fill_rate |
|---:|---|---|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.50 | 2500 | 0.000500 | 1.596709e-01 | True | True | 50.00% | 50.00% | +2.706414e-05 | +2.706414e-05 | 57.18% | 0.04 | +0.00004176 | +0.00004176 | +0.00004176 | 50.00% | 50.00% | +2.706414e-05 | +0.00004176 | +0.00% | +0.000000e+00 | +0.00000000 | fees_dominate | 0.00% | 56.62% | 1.600 | 0.142 | +1.701 | -0.041 | 0.695 | 0.80 | 0.00% |
| 2 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.40 | 2500 | 0.000500 | 1.386043e-01 | True | True | 50.00% | 50.00% | +2.196919e-05 | +2.196919e-05 | 58.30% | 0.04 | +0.00003051 | +0.00003051 | +0.00003051 | 50.00% | 50.00% | +2.196919e-05 | +0.00003051 | +0.00% | +0.000000e+00 | +0.00000000 | fees_dominate | 0.00% | 57.87% | 1.600 | 0.138 | +1.702 | -0.037 | 0.585 | 0.80 | 0.00% |
| 3 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.50 | 2500 | 0.000300 | 1.148889e-01 | True | True | 55.56% | 55.56% | +2.347757e-05 | +2.347757e-05 | 61.57% | 0.03 | +0.00004014 | +0.00004014 | +0.00004014 | 55.56% | 55.56% | +2.347757e-05 | +0.00004014 | +0.00% | +0.000000e+00 | +0.00000000 | fees_dominate | 0.00% | 60.23% | 1.600 | 0.146 | +1.395 | -0.351 | 1.044 | 0.80 | 0.00% |
| 4 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.40 | 2500 | 0.000300 | 1.145032e-01 | True | True | 55.56% | 55.56% | +2.041898e-05 | +2.041898e-05 | 61.31% | 0.03 | +0.00003443 | +0.00003443 | +0.00003443 | 55.56% | 55.56% | +2.041898e-05 | +0.00003443 | +0.00% | +0.000000e+00 | +0.00000000 | fees_dominate | 0.00% | 61.25% | 1.600 | 0.145 | +1.449 | -0.296 | 0.783 | 0.80 | 0.00% |
| 5 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.20 | 2500 | 0.000500 | 1.073186e-01 | True | True | 50.00% | 50.00% | +1.646225e-05 | +1.646225e-05 | 56.11% | 0.05 | +0.00002259 | +0.00002259 | +0.00002259 | 50.00% | 50.00% | +1.646225e-05 | +0.00002259 | +0.00% | +0.000000e+00 | +0.00000000 | fees_dominate | 0.00% | 55.67% | 1.600 | 0.139 | +1.690 | -0.049 | 0.534 | 0.80 | 0.00% |
| 6 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.30 | 2500 | 0.000500 | 1.021506e-01 | True | True | 50.00% | 50.00% | +1.611413e-05 | +1.611413e-05 | 59.15% | 0.04 | +0.00001609 | +0.00001609 | +0.00001609 | 50.00% | 50.00% | +1.611413e-05 | +0.00001609 | +0.00% | +0.000000e+00 | +0.00000000 | mixed | 0.00% | 58.47% | 1.600 | 0.137 | +1.767 | +0.031 | 0.577 | 0.80 | 0.00% |
| 7 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.50 | 3500 | 0.000300 | 9.995454e-02 | True | True | 55.56% | 55.56% | +2.894656e-05 | +2.894656e-05 | 61.27% | 0.02 | +0.00004666 | +0.00004666 | +0.00004666 | 55.56% | 55.56% | +2.894656e-05 | +0.00004666 | +0.00% | +0.000000e+00 | +0.00000000 | mixed | 0.00% | 61.02% | 1.600 | 0.190 | +1.993 | +0.203 | 1.317 | 0.80 | 50.00% |

survive_fee1_passrate_ge_0.5=0

## Decomposition

| rank | symbol | rule | h | gross_edge_npa | fee_cost_npa | adverse_cost_npa | scratch_cost_npa | net_npa | observed_npa | residual_npa | reject_rate | n_events | n_after_gate | n_filled |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | +9.629926e-05 | +9.059722e-05 | +8.045916e-06 | +0.000000e+00 | -2.343867e-06 | +2.706414e-05 | +2.940800e-05 | 0.00% | 2227 | 2227 | 1261 |
| 2 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | +9.846845e-05 | +9.259438e-05 | +8.003904e-06 | +0.000000e+00 | -2.129828e-06 | +2.196919e-05 | +2.409902e-05 | 0.00% | 2490 | 2490 | 1441 |
| 3 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | +8.401101e-05 | +9.637584e-05 | +8.793840e-06 | +0.000000e+00 | -2.115867e-05 | +2.347757e-05 | +4.463624e-05 | 0.00% | 1788 | 1788 | 1077 |
| 4 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | +8.876745e-05 | +9.799308e-05 | +8.876887e-06 | +0.000000e+00 | -1.810252e-05 | +2.041898e-05 | +3.852150e-05 | 0.00% | 2023 | 2023 | 1239 |
| 5 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | +9.408710e-05 | +8.907042e-05 | +7.731178e-06 | +0.000000e+00 | -2.714500e-06 | +1.646225e-05 | +1.917675e-05 | 0.00% | 2840 | 2840 | 1581 |
| 6 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | +1.033457e-04 | +9.355522e-05 | +7.983987e-06 | +0.000000e+00 | +1.806499e-06 | +1.611413e-05 | +1.430763e-05 | 0.00% | 2644 | 2644 | 1546 |
| 7 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | +1.216303e-04 | +9.762712e-05 | +1.158790e-05 | +0.000000e+00 | +1.241531e-05 | +2.894656e-05 | +1.653124e-05 | 0.00% | 1003 | 1003 | 612 |
