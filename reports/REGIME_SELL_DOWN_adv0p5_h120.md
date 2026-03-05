# PASSIVE_POCKET_RANKING

candidates=7 ranked=7
candidate_parse total_rows_seen=26 table_rows_seen=22 rows_with_pass_yes=7 candidates_parsed=7 candidates_unique=7 rows_skipped_missing_fields=0
fee_grid=[0.0] adverse_mult_grid=[0.5]
pass_threshold=0.500
mitigation_profile=baseline gate_config min_intensity_strong=0.000000 min_imbalance_strong=0.000000 max_spread_tight=0.000000 max_volatility_extreme=None vol_quantile_reject=0.010000 scratch_bps=0.0000 scratch_window_sec=0 scratch_taker_fee_bps=0.0000 scratch_slippage_bps=0.0000 horizon_sec_override=120

| rank | symbol | rule | horizon | min_imb | min_int | max_spread | score | robust_core | robust_stress | pass_rate_core | pass_rate_stress | npa_core | npa_stress | attempt_fill_rate | attempts_per_min | score_raw_core | score_raw_stress | score_raw_min | baseline_pass_rate_core | baseline_pass_rate_stress | baseline_npa_core | baseline_score_raw_core | delta_pass_rate_core | delta_npa_core | delta_score_raw_core | failure_reason_top | gate_reject_ratio | fill_rate_after_gate | avg_fee_bps | avg_adverse_bps_on_fills | avg_raw_return_bps_on_fills | avg_net_return_bps_on_fills | stability_std_bps | best_fee_survive | insufficient_fill_rate |
|---:|---|---|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.20 | 2500 | 0.000500 | 0.000000e+00 | False | False | 50.00% | 50.00% | -2.866954e-05 | -2.866954e-05 | 55.57% | 0.05 | -0.00005357 | -0.00005357 | -0.00005357 | 50.00% | 50.00% | -2.866954e-05 | -0.00005357 | +0.00% | +0.000000e+00 | +0.00000000 | adverse_dominates | 0.00% | 55.36% | 0.000 | 0.136 | -0.617 | -0.752 | 0.599 | 0.00 | 0.00% |
| 2 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.30 | 2500 | 0.000500 | 0.000000e+00 | False | False | 50.00% | 50.00% | -3.099216e-05 | -3.099216e-05 | 57.46% | 0.05 | -0.00005535 | -0.00005535 | -0.00005535 | 50.00% | 50.00% | -3.099216e-05 | -0.00005535 | +0.00% | +0.000000e+00 | +0.00000000 | adverse_dominates | 0.00% | 57.45% | 0.000 | 0.134 | -0.663 | -0.796 | 0.611 | 0.00 | 0.00% |
| 3 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.40 | 2500 | 0.000500 | 0.000000e+00 | False | False | 38.89% | 38.89% | -1.583244e-05 | -1.583244e-05 | 56.25% | 0.04 | -0.00002847 | -0.00002847 | -0.00002847 | 38.89% | 38.89% | -1.583244e-05 | -0.00002847 | +0.00% | +0.000000e+00 | +0.00000000 | adverse_dominates | 0.00% | 56.84% | 0.000 | 0.135 | -0.616 | -0.750 | 0.786 | 0.00 | 0.00% |
| 4 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.50 | 3500 | 0.000300 | 0.000000e+00 | False | False | 38.89% | 38.89% | -6.145881e-05 | -6.145881e-05 | 64.96% | 0.02 | -0.00010082 | -0.00010082 | -0.00010082 | 38.89% | 38.89% | -6.145881e-05 | -0.00010082 | +0.00% | +0.000000e+00 | +0.00000000 | adverse_dominates | 0.00% | 65.21% | 0.000 | 0.170 | -0.903 | -1.073 | 1.638 | 0.00 | 0.00% |
| 5 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.50 | 2500 | 0.000500 | 0.000000e+00 | False | False | 27.78% | 27.78% | -2.173904e-05 | -2.173904e-05 | 55.32% | 0.04 | -0.00003984 | -0.00003984 | -0.00003984 | 27.78% | 27.78% | -2.173904e-05 | -0.00003984 | +0.00% | +0.000000e+00 | +0.00000000 | adverse_dominates | 0.00% | 55.99% | 0.000 | 0.137 | -0.833 | -0.970 | 0.864 | 0.00 | 0.00% |
| 6 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.40 | 2500 | 0.000300 | 0.000000e+00 | False | False | 22.22% | 22.22% | -4.499975e-05 | -4.499975e-05 | 59.89% | 0.03 | -0.00007179 | -0.00007179 | -0.00007179 | 22.22% | 22.22% | -4.499975e-05 | -0.00007179 | +0.00% | +0.000000e+00 | +0.00000000 | adverse_dominates | 0.00% | 61.18% | 0.000 | 0.142 | -1.024 | -1.166 | 0.905 | 0.00 | 0.00% |
| 7 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.50 | 2500 | 0.000300 | 0.000000e+00 | False | False | 22.22% | 22.22% | -6.608991e-05 | -6.608991e-05 | 59.76% | 0.03 | -0.00011149 | -0.00011149 | -0.00011149 | 22.22% | 22.22% | -6.608991e-05 | -0.00011149 | +0.00% | +0.000000e+00 | +0.00000000 | adverse_dominates | 0.00% | 60.84% | 0.000 | 0.143 | -1.439 | -1.582 | 0.958 | 0.00 | 0.00% |

survive_fee1_passrate_ge_0.5=0

## Decomposition

| rank | symbol | rule | h | gross_edge_npa | fee_cost_npa | adverse_cost_npa | scratch_cost_npa | net_npa | observed_npa | residual_npa | reject_rate | n_events | n_after_gate | n_filled |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | -3.413492e-05 | +0.000000e+00 | +7.516327e-06 | +0.000000e+00 | -4.165125e-05 | -2.866954e-05 | +1.298171e-05 | 0.00% | 3080 | 3080 | 1705 |
| 2 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | -3.807067e-05 | +0.000000e+00 | +7.685023e-06 | +0.000000e+00 | -4.575570e-05 | -3.099216e-05 | +1.476353e-05 | 0.00% | 2832 | 2832 | 1627 |
| 3 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | -3.499739e-05 | +0.000000e+00 | +7.658930e-06 | +0.000000e+00 | -4.265632e-05 | -1.583244e-05 | +2.682387e-05 | 0.00% | 2646 | 2646 | 1504 |
| 4 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | -5.888021e-05 | +0.000000e+00 | +1.111398e-05 | +0.000000e+00 | -6.999419e-05 | -6.145881e-05 | +8.535380e-06 | 0.00% | 1055 | 1055 | 688 |
| 5 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | -4.662815e-05 | +0.000000e+00 | +7.676614e-06 | +0.000000e+00 | -5.430476e-05 | -2.173904e-05 | +3.256573e-05 | 0.00% | 2404 | 2404 | 1346 |
| 6 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | -6.264122e-05 | +0.000000e+00 | +8.698248e-06 | +0.000000e+00 | -7.133947e-05 | -4.499975e-05 | +2.633972e-05 | 0.00% | 2084 | 2084 | 1275 |
| 7 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | -8.756329e-05 | +0.000000e+00 | +8.711053e-06 | +0.000000e+00 | -9.627435e-05 | -6.608991e-05 | +3.018444e-05 | 0.00% | 1895 | 1895 | 1153 |
