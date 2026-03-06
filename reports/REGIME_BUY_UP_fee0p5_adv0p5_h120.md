# PASSIVE_POCKET_RANKING

candidates=7 ranked=7
candidate_parse total_rows_seen=26 table_rows_seen=22 rows_with_pass_yes=7 candidates_parsed=7 candidates_unique=7 rows_skipped_missing_fields=0
fee_grid=[0.5] adverse_mult_grid=[0.5]
pass_threshold=0.500
mitigation_profile=baseline gate_config min_intensity_strong=0.000000 min_imbalance_strong=0.000000 max_spread_tight=0.000000 max_volatility_extreme=None vol_quantile_reject=0.010000 scratch_bps=0.0000 scratch_window_sec=0 scratch_taker_fee_bps=0.0000 scratch_slippage_bps=0.0000 horizon_sec_override=120

| rank | symbol | rule | horizon | min_imb | min_int | max_spread | score | robust_core | robust_stress | pass_rate_core | pass_rate_stress | npa_core | npa_stress | attempt_fill_rate | attempts_per_min | score_raw_core | score_raw_stress | score_raw_min | baseline_pass_rate_core | baseline_pass_rate_stress | baseline_npa_core | baseline_score_raw_core | delta_pass_rate_core | delta_npa_core | delta_score_raw_core | failure_reason_top | gate_reject_ratio | fill_rate_after_gate | avg_fee_bps | avg_adverse_bps_on_fills | avg_raw_return_bps_on_fills | avg_net_return_bps_on_fills | stability_std_bps | best_fee_survive | insufficient_fill_rate |
|---:|---|---|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.30 | 2500 | 0.000500 | 2.950743e-01 | True | True | 50.00% | 50.00% | +5.280310e-05 | +5.280310e-05 | 58.30% | 0.04 | +0.00008439 | +0.00008439 | +0.00008439 | 50.00% | 50.00% | +5.280310e-05 | +0.00008439 | +0.00% | +0.000000e+00 | +0.00000000 | mixed | 0.00% | 58.45% | 1.000 | 0.136 | +1.867 | +0.731 | 0.789 | 0.50 | 0.00% |
| 2 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.50 | 3500 | 0.000300 | 2.806105e-01 | True | True | 55.56% | 55.56% | +6.886708e-05 | +6.886708e-05 | 61.72% | 0.02 | +0.00010607 | +0.00010607 | +0.00010607 | 55.56% | 55.56% | +6.886708e-05 | +0.00010607 | +0.00% | +0.000000e+00 | +0.00000000 | mixed | 0.00% | 62.51% | 1.000 | 0.192 | +1.928 | +0.736 | 0.963 | 0.50 | 50.00% |
| 3 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.40 | 2500 | 0.000500 | 2.598995e-01 | True | True | 50.00% | 50.00% | +4.843980e-05 | +4.843980e-05 | 58.10% | 0.04 | +0.00008393 | +0.00008393 | +0.00008393 | 50.00% | 50.00% | +4.843980e-05 | +0.00008393 | +0.00% | +0.000000e+00 | +0.00000000 | mixed | 0.00% | 57.79% | 1.000 | 0.139 | +1.958 | +0.819 | 0.864 | 0.50 | 0.00% |
| 4 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.20 | 2500 | 0.000500 | 2.500280e-01 | True | True | 50.00% | 50.00% | +4.429049e-05 | +4.429049e-05 | 55.71% | 0.05 | +0.00007428 | +0.00007428 | +0.00007428 | 50.00% | 50.00% | +4.429049e-05 | +0.00007428 | +0.00% | +0.000000e+00 | +0.00000000 | mixed | 0.00% | 55.50% | 1.000 | 0.139 | +1.870 | +0.732 | 0.771 | 0.50 | 0.00% |
| 5 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.40 | 2500 | 0.000300 | 1.565923e-01 | True | True | 50.00% | 50.00% | +2.828680e-05 | +2.828680e-05 | 60.44% | 0.03 | +0.00004550 | +0.00004550 | +0.00004550 | 50.00% | 50.00% | +2.828680e-05 | +0.00004550 | +0.00% | +0.000000e+00 | +0.00000000 | mixed | 0.00% | 60.56% | 1.000 | 0.145 | +1.581 | +0.436 | 0.806 | 0.50 | 0.00% |
| 6 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.50 | 2500 | 0.000500 | 1.359638e-01 | True | True | 50.00% | 50.00% | +2.449751e-05 | +2.449751e-05 | 56.80% | 0.04 | +0.00004483 | +0.00004483 | +0.00004483 | 50.00% | 50.00% | +2.449751e-05 | +0.00004483 | +0.00% | +0.000000e+00 | +0.00000000 | mixed | 0.00% | 56.71% | 1.000 | 0.142 | +1.681 | +0.539 | 0.802 | 0.50 | 0.00% |
| 7 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.50 | 2500 | 0.000300 | 3.484904e-02 | True | True | 50.00% | 50.00% | +6.224445e-06 | +6.224445e-06 | 60.21% | 0.03 | +0.00001322 | +0.00001322 | +0.00001322 | 50.00% | 50.00% | +6.224445e-06 | +0.00001322 | +0.00% | +0.000000e+00 | +0.00000000 | mixed | 0.00% | 59.49% | 1.000 | 0.147 | +1.473 | +0.326 | 0.786 | 0.50 | 0.00% |

survive_fee1_passrate_ge_0.5=0

## Decomposition

| rank | symbol | rule | h | gross_edge_npa | fee_cost_npa | adverse_cost_npa | scratch_cost_npa | net_npa | observed_npa | residual_npa | reject_rate | n_events | n_after_gate | n_filled |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | +1.091573e-04 | +5.845283e-05 | +7.954165e-06 | +0.000000e+00 | +4.275032e-05 | +5.280310e-05 | +1.005278e-05 | 0.00% | 2650 | 2650 | 1549 |
| 2 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | +1.204960e-04 | +6.251246e-05 | +1.197144e-05 | +0.000000e+00 | +4.601210e-05 | +6.886708e-05 | +2.285498e-05 | 0.00% | 1003 | 1003 | 627 |
| 3 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | +1.131382e-04 | +5.778935e-05 | +8.014161e-06 | +0.000000e+00 | +4.733472e-05 | +4.843980e-05 | +1.105078e-06 | 0.00% | 2497 | 2497 | 1443 |
| 4 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | +1.038181e-04 | +5.550475e-05 | +7.690247e-06 | +0.000000e+00 | +4.062315e-05 | +4.429049e-05 | +3.667341e-06 | 0.00% | 2843 | 2843 | 1578 |
| 5 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | +9.577041e-05 | +6.056130e-05 | +8.809163e-06 | +0.000000e+00 | +2.639995e-05 | +2.828680e-05 | +1.886855e-06 | 0.00% | 2031 | 2031 | 1230 |
| 6 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | +9.532019e-05 | +5.670841e-05 | +8.067203e-06 | +0.000000e+00 | +3.054458e-05 | +2.449751e-05 | -6.047073e-06 | 0.00% | 2236 | 2236 | 1268 |
| 7 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | +8.759519e-05 | +5.948661e-05 | +8.738645e-06 | +0.000000e+00 | +1.936994e-05 | +6.224445e-06 | -1.314550e-05 | 0.00% | 1792 | 1792 | 1066 |
