# PASSIVE_POCKET_RANKING

candidates=7 ranked=7
candidate_parse total_rows_seen=26 table_rows_seen=22 rows_with_pass_yes=7 candidates_parsed=7 candidates_unique=7 rows_skipped_missing_fields=0
fee_grid=[0.0] adverse_mult_grid=[0.3]
pass_threshold=0.500
mitigation_profile=baseline gate_config min_intensity_strong=0.000000 min_imbalance_strong=0.000000 max_spread_tight=0.000000 max_volatility_extreme=None vol_quantile_reject=0.010000 scratch_bps=0.0000 scratch_window_sec=0 scratch_taker_fee_bps=0.0000 scratch_slippage_bps=0.0000 horizon_sec_override=120

| rank | symbol | rule | horizon | min_imb | min_int | max_spread | score | robust_core | robust_stress | pass_rate_core | pass_rate_stress | npa_core | npa_stress | attempt_fill_rate | attempts_per_min | score_raw_core | score_raw_stress | score_raw_min | baseline_pass_rate_core | baseline_pass_rate_stress | baseline_npa_core | baseline_score_raw_core | delta_pass_rate_core | delta_npa_core | delta_score_raw_core | failure_reason_top | gate_reject_ratio | fill_rate_after_gate | avg_fee_bps | avg_adverse_bps_on_fills | avg_raw_return_bps_on_fills | avg_net_return_bps_on_fills | stability_std_bps | best_fee_survive | insufficient_fill_rate |
|---:|---|---|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.50 | 3500 | 0.000300 | 3.159561e-01 | True | True | 55.56% | 55.56% | +6.315793e-05 | +6.315793e-05 | 61.66% | 0.03 | +0.00010112 | +0.00010112 | +0.00010112 | 55.56% | 55.56% | +6.315793e-05 | +0.00010112 | +0.00% | +0.000000e+00 | +0.00000000 | mixed | 0.00% | 62.99% | 0.000 | 0.107 | +0.256 | +0.148 | 0.999 | 0.00 | 0.00% |
| 2 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.20 | 2500 | 0.000500 | 8.427610e-02 | True | True | 50.00% | 50.00% | +1.314524e-05 | +1.314524e-05 | 56.08% | 0.09 | +0.00002216 | +0.00002216 | +0.00002216 | 50.00% | 50.00% | +1.314524e-05 | +0.00002216 | +0.00% | +0.000000e+00 | +0.00000000 | adverse_dominates | 0.00% | 56.08% | 0.000 | 0.083 | -0.001 | -0.084 | 0.560 | 0.00 | 0.00% |
| 3 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.30 | 2500 | 0.000500 | 6.336823e-02 | True | True | 50.00% | 50.00% | +9.874868e-06 | +9.874868e-06 | 58.65% | 0.09 | +0.00001474 | +0.00001474 | +0.00001474 | 50.00% | 50.00% | +9.874868e-06 | +0.00001474 | +0.00% | +0.000000e+00 | +0.00000000 | adverse_dominates | 0.00% | 58.23% | 0.000 | 0.082 | -0.046 | -0.128 | 0.558 | 0.00 | 0.00% |
| 4 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.40 | 2500 | 0.000500 | 3.335226e-02 | True | True | 50.00% | 50.00% | +4.898652e-06 | +4.898652e-06 | 57.73% | 0.08 | +0.00000504 | +0.00000504 | +0.00000504 | 50.00% | 50.00% | +4.898652e-06 | +0.00000504 | +0.00% | +0.000000e+00 | +0.00000000 | mixed | 0.00% | 57.47% | 0.000 | 0.082 | +0.132 | +0.050 | 0.469 | 0.00 | 0.00% |
| 5 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.50 | 2500 | 0.000500 | 0.000000e+00 | False | False | 50.00% | 50.00% | -3.532111e-06 | -3.532111e-06 | 56.00% | 0.07 | -0.00001272 | -0.00001272 | -0.00001272 | 50.00% | 50.00% | -3.532111e-06 | -0.00001272 | +0.00% | +0.000000e+00 | +0.00000000 | adverse_dominates | 0.00% | 56.54% | 0.000 | 0.084 | -0.008 | -0.091 | 0.508 | 0.00 | 0.00% |
| 6 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.50 | 2500 | 0.000300 | 0.000000e+00 | False | False | 50.00% | 50.00% | -1.217667e-05 | -1.217667e-05 | 60.28% | 0.06 | -0.00002200 | -0.00002200 | -0.00002200 | 50.00% | 50.00% | -1.217667e-05 | -0.00002200 | +0.00% | +0.000000e+00 | +0.00000000 | adverse_dominates | 0.00% | 60.93% | 0.000 | 0.086 | -0.148 | -0.235 | 0.505 | 0.00 | 0.00% |
| 7 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.40 | 2500 | 0.000300 | 0.000000e+00 | False | False | 44.44% | 44.44% | -6.758029e-06 | -6.758029e-06 | 61.45% | 0.06 | -0.00001222 | -0.00001222 | -0.00001222 | 44.44% | 44.44% | -6.758029e-06 | -0.00001222 | +0.00% | +0.000000e+00 | +0.00000000 | adverse_dominates | 0.00% | 61.96% | 0.000 | 0.086 | -0.014 | -0.100 | 0.442 | 0.00 | 0.00% |

survive_fee1_passrate_ge_0.5=0

## Decomposition

| rank | symbol | rule | h | gross_edge_npa | fee_cost_npa | adverse_cost_npa | scratch_cost_npa | net_npa | observed_npa | residual_npa | reject_rate | n_events | n_after_gate | n_filled |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | +1.609582e-05 | +0.000000e+00 | +6.761133e-06 | +0.000000e+00 | +9.334684e-06 | +6.315793e-05 | +5.382325e-05 | 0.00% | 1832 | 1832 | 1154 |
| 2 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | -4.383527e-08 | +0.000000e+00 | +4.640235e-06 | +0.000000e+00 | -4.684071e-06 | +1.314524e-05 | +1.782931e-05 | 0.00% | 5235 | 5235 | 2936 |
| 3 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | -2.688207e-06 | +0.000000e+00 | +4.747576e-06 | +0.000000e+00 | -7.435783e-06 | +9.874868e-06 | +1.731065e-05 | 0.00% | 4862 | 4862 | 2831 |
| 4 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | +7.605574e-06 | +0.000000e+00 | +4.734610e-06 | +0.000000e+00 | +2.870964e-06 | +4.898652e-06 | +2.027689e-06 | 0.00% | 4562 | 4562 | 2622 |
| 5 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | -4.487911e-07 | +0.000000e+00 | +4.723758e-06 | +0.000000e+00 | -5.172550e-06 | -3.532111e-06 | +1.640438e-06 | 0.00% | 4137 | 4137 | 2339 |
| 6 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | -9.039829e-06 | +0.000000e+00 | +5.264216e-06 | +0.000000e+00 | -1.430405e-05 | -1.217667e-05 | +2.127374e-06 | 0.00% | 3297 | 3297 | 2009 |
| 7 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | -8.745204e-07 | +0.000000e+00 | +5.317302e-06 | +0.000000e+00 | -6.191823e-06 | -6.758029e-06 | -5.662066e-07 | 0.00% | 3641 | 3641 | 2256 |
