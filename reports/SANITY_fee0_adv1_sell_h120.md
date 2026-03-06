# PASSIVE_POCKET_RANKING

candidates=7 ranked=7
candidate_parse total_rows_seen=26 table_rows_seen=22 rows_with_pass_yes=7 candidates_parsed=7 candidates_unique=7 rows_skipped_missing_fields=0
fee_grid=[0.0] adverse_mult_grid=[1.0]
pass_threshold=0.500
mitigation_profile=baseline gate_config min_intensity_strong=0.000000 min_imbalance_strong=0.000000 max_spread_tight=0.000000 max_volatility_extreme=None vol_quantile_reject=0.010000 scratch_bps=0.0000 scratch_window_sec=0 scratch_taker_fee_bps=0.0000 scratch_slippage_bps=0.0000 horizon_sec_override=120

| rank | symbol | rule | horizon | min_imb | min_int | max_spread | score | robust_core | robust_stress | pass_rate_core | pass_rate_stress | npa_core | npa_stress | attempt_fill_rate | attempts_per_min | score_raw_core | score_raw_stress | score_raw_min | baseline_pass_rate_core | baseline_pass_rate_stress | baseline_npa_core | baseline_score_raw_core | delta_pass_rate_core | delta_npa_core | delta_score_raw_core | failure_reason_top | gate_reject_ratio | fill_rate_after_gate | avg_fee_bps | avg_adverse_bps_on_fills | avg_raw_return_bps_on_fills | avg_net_return_bps_on_fills | stability_std_bps | best_fee_survive | insufficient_fill_rate |
|---:|---|---|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.20 | 2500 | 0.000500 | 3.731727e-02 | True | True | 50.00% | 50.00% | +5.289587e-06 | +5.289587e-06 | 54.61% | 0.09 | +0.00000881 | +0.00000881 | +0.00000881 | 50.00% | 50.00% | +5.289587e-06 | +0.00000881 | +0.00% | +0.000000e+00 | +0.00000000 | adverse_dominates | 0.00% | 55.47% | 0.000 | 0.274 | +0.212 | -0.062 | 0.417 | 0.00 | 0.00% |
| 2 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.40 | 2500 | 0.000300 | 2.123064e-02 | True | True | 50.00% | 50.00% | +3.160276e-06 | +3.160276e-06 | 60.75% | 0.06 | +0.00000408 | +0.00000408 | +0.00000408 | 50.00% | 50.00% | +3.160276e-06 | +0.00000408 | +0.00% | +0.000000e+00 | +0.00000000 | adverse_dominates | 0.00% | 61.80% | 0.000 | 0.285 | -0.011 | -0.296 | 0.489 | 0.00 | 0.00% |
| 3 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.50 | 3500 | 0.000300 | 8.252985e-03 | True | True | 50.00% | 50.00% | +1.647288e-06 | +1.647288e-06 | 64.72% | 0.03 | +0.00000161 | +0.00000161 | +0.00000161 | 50.00% | 50.00% | +1.647288e-06 | +0.00000161 | +0.00% | +0.000000e+00 | +0.00000000 | adverse_dominates | 0.00% | 63.64% | 0.000 | 0.357 | -0.490 | -0.847 | 0.996 | 0.00 | 0.00% |
| 4 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.30 | 2500 | 0.000500 | 0.000000e+00 | False | False | 50.00% | 50.00% | -1.089809e-06 | -1.089809e-06 | 56.72% | 0.09 | -0.00000249 | -0.00000249 | -0.00000249 | 50.00% | 50.00% | -1.089809e-06 | -0.00000249 | +0.00% | +0.000000e+00 | +0.00000000 | adverse_dominates | 0.00% | 57.80% | 0.000 | 0.271 | +0.142 | -0.128 | 0.419 | 0.00 | 0.00% |
| 5 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.40 | 2500 | 0.000500 | 0.000000e+00 | False | False | 50.00% | 50.00% | -2.875481e-06 | -2.875481e-06 | 55.18% | 0.08 | -0.00000604 | -0.00000604 | -0.00000604 | 50.00% | 50.00% | -2.875481e-06 | -0.00000604 | +0.00% | +0.000000e+00 | +0.00000000 | adverse_dominates | 0.00% | 56.91% | 0.000 | 0.274 | +0.208 | -0.065 | 0.572 | 0.00 | 0.00% |
| 6 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.50 | 2500 | 0.000500 | 0.000000e+00 | False | False | 50.00% | 50.00% | -1.763260e-05 | -1.763260e-05 | 54.73% | 0.07 | -0.00003338 | -0.00003338 | -0.00003338 | 50.00% | 50.00% | -1.763260e-05 | -0.00003338 | +0.00% | +0.000000e+00 | +0.00000000 | adverse_dominates | 0.00% | 55.98% | 0.000 | 0.279 | -0.036 | -0.314 | 0.579 | 0.00 | 0.00% |
| 7 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.50 | 2500 | 0.000300 | 0.000000e+00 | False | False | 50.00% | 50.00% | -3.554112e-05 | -3.554112e-05 | 60.00% | 0.06 | -0.00006019 | -0.00006019 | -0.00006019 | 50.00% | 50.00% | -3.554112e-05 | -0.00006019 | +0.00% | +0.000000e+00 | +0.00000000 | adverse_dominates | 0.00% | 60.77% | 0.000 | 0.287 | -0.314 | -0.600 | 0.496 | 0.00 | 0.00% |

survive_fee1_passrate_ge_0.5=0

## Decomposition

| rank | symbol | rule | h | gross_edge_npa | fee_cost_npa | adverse_cost_npa | scratch_cost_npa | net_npa | observed_npa | residual_npa | reject_rate | n_events | n_after_gate | n_filled |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | +1.174518e-05 | +0.000000e+00 | +1.518826e-05 | +0.000000e+00 | -3.443085e-06 | +5.289587e-06 | +8.732672e-06 | 0.00% | 5214 | 5210 | 2890 |
| 2 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | -7.091036e-07 | +0.000000e+00 | +1.761344e-05 | +0.000000e+00 | -1.832254e-05 | +3.160276e-06 | +2.148282e-05 | 0.00% | 3644 | 3644 | 2252 |
| 3 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | -3.120660e-05 | +0.000000e+00 | +2.268914e-05 | +0.000000e+00 | -5.389573e-05 | +1.647288e-06 | +5.554302e-05 | 0.00% | 1826 | 1826 | 1162 |
| 4 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | +8.231788e-06 | +0.000000e+00 | +1.564393e-05 | +0.000000e+00 | -7.412140e-06 | -1.089809e-06 | +6.322331e-06 | 0.00% | 4840 | 4836 | 2795 |
| 5 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | +1.186049e-05 | +0.000000e+00 | +1.558770e-05 | +0.000000e+00 | -3.727204e-06 | -2.875481e-06 | +8.517231e-07 | 0.00% | 4549 | 4549 | 2589 |
| 6 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | -1.993726e-06 | +0.000000e+00 | +1.559652e-05 | +0.000000e+00 | -1.759025e-05 | -1.763260e-05 | -4.235415e-08 | 0.00% | 4125 | 4125 | 2309 |
| 7 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | -1.907301e-05 | +0.000000e+00 | +1.742010e-05 | +0.000000e+00 | -3.649310e-05 | -3.554112e-05 | +9.519828e-07 | 0.00% | 3291 | 3291 | 2000 |
