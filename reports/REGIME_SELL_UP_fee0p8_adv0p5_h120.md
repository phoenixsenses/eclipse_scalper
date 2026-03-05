# PASSIVE_POCKET_RANKING

candidates=7 ranked=7
candidate_parse total_rows_seen=26 table_rows_seen=22 rows_with_pass_yes=7 candidates_parsed=7 candidates_unique=7 rows_skipped_missing_fields=0
fee_grid=[0.8] adverse_mult_grid=[0.5]
pass_threshold=0.500
mitigation_profile=baseline gate_config min_intensity_strong=0.000000 min_imbalance_strong=0.000000 max_spread_tight=0.000000 max_volatility_extreme=None vol_quantile_reject=0.010000 scratch_bps=0.0000 scratch_window_sec=0 scratch_taker_fee_bps=0.0000 scratch_slippage_bps=0.0000 horizon_sec_override=120

| rank | symbol | rule | horizon | min_imb | min_int | max_spread | score | robust_core | robust_stress | pass_rate_core | pass_rate_stress | npa_core | npa_stress | attempt_fill_rate | attempts_per_min | score_raw_core | score_raw_stress | score_raw_min | baseline_pass_rate_core | baseline_pass_rate_stress | baseline_npa_core | baseline_score_raw_core | delta_pass_rate_core | delta_npa_core | delta_score_raw_core | failure_reason_top | gate_reject_ratio | fill_rate_after_gate | avg_fee_bps | avg_adverse_bps_on_fills | avg_raw_return_bps_on_fills | avg_net_return_bps_on_fills | stability_std_bps | best_fee_survive | insufficient_fill_rate |
|---:|---|---|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.50 | 3500 | 0.000300 | 8.429005e-02 | True | True | 50.00% | 50.00% | +2.216546e-05 | +2.216546e-05 | 60.25% | 0.02 | +0.00003136 | +0.00003136 | +0.00003136 | 50.00% | 50.00% | +2.216546e-05 | +0.00003136 | +0.00% | +0.000000e+00 | +0.00000000 | fees_dominate | 0.00% | 61.20% | 1.600 | 0.191 | +1.736 | -0.055 | 1.104 | 0.80 | 50.00% |
| 2 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.40 | 2500 | 0.000500 | 0.000000e+00 | False | False | 50.00% | 50.00% | -3.671110e-06 | -3.671110e-06 | 57.76% | 0.04 | -0.00000807 | -0.00000807 | -0.00000807 | 50.00% | 50.00% | -3.671110e-06 | -0.00000807 | +0.00% | +0.000000e+00 | +0.00000000 | fees_dominate | 0.00% | 57.73% | 1.600 | 0.139 | +1.402 | -0.337 | 0.796 | 0.80 | 0.00% |
| 3 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.50 | 2500 | 0.000500 | 0.000000e+00 | False | False | 50.00% | 50.00% | -4.517271e-06 | -4.517271e-06 | 55.90% | 0.04 | -0.00000238 | -0.00000238 | -0.00000238 | 50.00% | 50.00% | -4.517271e-06 | -0.00000238 | +0.00% | +0.000000e+00 | +0.00000000 | fees_dominate | 0.00% | 56.28% | 1.600 | 0.143 | +1.181 | -0.562 | 0.736 | 0.80 | 0.00% |
| 4 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.30 | 2500 | 0.000500 | 0.000000e+00 | False | False | 50.00% | 50.00% | -2.298256e-05 | -2.298256e-05 | 58.60% | 0.04 | -0.00004100 | -0.00004100 | -0.00004100 | 50.00% | 50.00% | -2.298256e-05 | -0.00004100 | +0.00% | +0.000000e+00 | +0.00000000 | fees_dominate | 0.00% | 58.26% | 1.600 | 0.137 | +1.383 | -0.354 | 0.920 | 0.80 | 0.00% |
| 5 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.40 | 2500 | 0.000300 | 0.000000e+00 | False | False | 50.00% | 50.00% | -2.511754e-05 | -2.511754e-05 | 60.51% | 0.03 | -0.00003930 | -0.00003930 | -0.00003930 | 50.00% | 50.00% | -2.511754e-05 | -0.00003930 | +0.00% | +0.000000e+00 | +0.00000000 | fees_dominate | 0.00% | 60.17% | 1.600 | 0.146 | +1.471 | -0.275 | 0.650 | 0.80 | 0.00% |
| 6 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.20 | 2500 | 0.000500 | 0.000000e+00 | False | False | 50.00% | 50.00% | -2.991644e-05 | -2.991644e-05 | 56.38% | 0.05 | -0.00004727 | -0.00004727 | -0.00004727 | 50.00% | 50.00% | -2.991644e-05 | -0.00004727 | +0.00% | +0.000000e+00 | +0.00000000 | fees_dominate | 0.00% | 55.58% | 1.600 | 0.139 | +1.386 | -0.354 | 0.902 | 0.80 | 0.00% |
| 7 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.50 | 2500 | 0.000300 | 0.000000e+00 | False | False | 50.00% | 50.00% | -3.333708e-05 | -3.333708e-05 | 59.94% | 0.03 | -0.00005063 | -0.00005063 | -0.00005063 | 50.00% | 50.00% | -3.333708e-05 | -0.00005063 | +0.00% | +0.000000e+00 | +0.00000000 | fees_dominate | 0.00% | 59.64% | 1.600 | 0.147 | +1.297 | -0.450 | 0.664 | 0.80 | 0.00% |

survive_fee1_passrate_ge_0.5=0

## Decomposition

| rank | symbol | rule | h | gross_edge_npa | fee_cost_npa | adverse_cost_npa | scratch_cost_npa | net_npa | observed_npa | residual_npa | reject_rate | n_events | n_after_gate | n_filled |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | +1.062574e-04 | +9.792000e-05 | +1.168740e-05 | +0.000000e+00 | -3.350007e-06 | +2.216546e-05 | +2.551546e-05 | 0.00% | 1000 | 1000 | 612 |
| 2 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | +8.093755e-05 | +9.236246e-05 | +8.038743e-06 | +0.000000e+00 | -1.946365e-05 | -3.671110e-06 | +1.579254e-05 | 0.00% | 2472 | 2472 | 1427 |
| 3 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | +6.647468e-05 | +9.004517e-05 | +8.075142e-06 | +0.000000e+00 | -3.164563e-05 | -4.517271e-06 | +2.712836e-05 | 0.00% | 2214 | 2214 | 1246 |
| 4 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | +8.056838e-05 | +9.322163e-05 | +7.983118e-06 | +0.000000e+00 | -2.063637e-05 | -2.298256e-05 | -2.346190e-06 | 0.00% | 2626 | 2626 | 1530 |
| 5 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | +8.848560e-05 | +9.627051e-05 | +8.765493e-06 | +0.000000e+00 | -1.655040e-05 | -2.511754e-05 | -8.567138e-06 | 0.00% | 2011 | 2011 | 1210 |
| 6 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | +7.700341e-05 | +8.892045e-05 | +7.742418e-06 | +0.000000e+00 | -1.965946e-05 | -2.991644e-05 | -1.025698e-05 | 0.00% | 2816 | 2816 | 1565 |
| 7 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | +7.737541e-05 | +9.542114e-05 | +8.775037e-06 | +0.000000e+00 | -2.682077e-05 | -3.333708e-05 | -6.516315e-06 | 0.00% | 1769 | 1769 | 1055 |
