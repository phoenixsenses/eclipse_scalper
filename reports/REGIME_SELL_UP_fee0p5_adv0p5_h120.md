# PASSIVE_POCKET_RANKING

candidates=7 ranked=7
candidate_parse total_rows_seen=26 table_rows_seen=22 rows_with_pass_yes=7 candidates_parsed=7 candidates_unique=7 rows_skipped_missing_fields=0
fee_grid=[0.5] adverse_mult_grid=[0.5]
pass_threshold=0.500
mitigation_profile=baseline gate_config min_intensity_strong=0.000000 min_imbalance_strong=0.000000 max_spread_tight=0.000000 max_volatility_extreme=None vol_quantile_reject=0.010000 scratch_bps=0.0000 scratch_window_sec=0 scratch_taker_fee_bps=0.0000 scratch_slippage_bps=0.0000 horizon_sec_override=120

| rank | symbol | rule | horizon | min_imb | min_int | max_spread | score | robust_core | robust_stress | pass_rate_core | pass_rate_stress | npa_core | npa_stress | attempt_fill_rate | attempts_per_min | score_raw_core | score_raw_stress | score_raw_min | baseline_pass_rate_core | baseline_pass_rate_stress | baseline_npa_core | baseline_score_raw_core | delta_pass_rate_core | delta_npa_core | delta_score_raw_core | failure_reason_top | gate_reject_ratio | fill_rate_after_gate | avg_fee_bps | avg_adverse_bps_on_fills | avg_raw_return_bps_on_fills | avg_net_return_bps_on_fills | stability_std_bps | best_fee_survive | insufficient_fill_rate |
|---:|---|---|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.50 | 2500 | 0.000500 | 3.468670e-01 | True | True | 55.56% | 55.56% | +7.092942e-05 | +7.092942e-05 | 55.33% | 0.04 | +0.00013628 | +0.00013628 | +0.00013628 | 55.56% | 55.56% | +7.092942e-05 | +0.00013628 | +0.00% | +0.000000e+00 | +0.00000000 | mixed | 0.00% | 55.06% | 1.000 | 0.141 | +1.451 | +0.309 | 1.045 | 0.50 | 0.00% |
| 2 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.50 | 2500 | 0.000300 | 3.396287e-01 | True | True | 55.56% | 55.56% | +7.169040e-05 | +7.169040e-05 | 57.19% | 0.03 | +0.00012496 | +0.00012496 | +0.00012496 | 55.56% | 55.56% | +7.169040e-05 | +0.00012496 | +0.00% | +0.000000e+00 | +0.00000000 | mixed | 0.00% | 57.99% | 1.000 | 0.146 | +1.408 | +0.262 | 1.111 | 0.50 | 0.00% |
| 3 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.40 | 2500 | 0.000500 | 2.299252e-01 | True | True | 50.00% | 50.00% | +3.814756e-05 | +3.814756e-05 | 56.27% | 0.04 | +0.00007150 | +0.00007150 | +0.00007150 | 50.00% | 50.00% | +3.814756e-05 | +0.00007150 | +0.00% | +0.000000e+00 | +0.00000000 | mixed | 0.00% | 56.59% | 1.000 | 0.138 | +1.578 | +0.440 | 0.659 | 0.50 | 0.00% |
| 4 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.40 | 2500 | 0.000300 | 1.264168e-01 | True | True | 50.00% | 50.00% | +2.458255e-05 | +2.458255e-05 | 59.39% | 0.03 | +0.00004261 | +0.00004261 | +0.00004261 | 50.00% | 50.00% | +2.458255e-05 | +0.00004261 | +0.00% | +0.000000e+00 | +0.00000000 | mixed | 0.00% | 59.32% | 1.000 | 0.145 | +1.416 | +0.271 | 0.945 | 0.50 | 0.00% |
| 5 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.50 | 3500 | 0.000300 | 1.050265e-01 | True | True | 50.00% | 50.00% | +3.144516e-05 | +3.144516e-05 | 58.13% | 0.02 | +0.00005600 | +0.00005600 | +0.00005600 | 50.00% | 50.00% | +3.144516e-05 | +0.00005600 | +0.00% | +0.000000e+00 | +0.00000000 | mixed | 0.00% | 59.08% | 1.000 | 0.191 | +1.681 | +0.490 | 1.395 | 0.50 | 50.00% |
| 6 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.20 | 2500 | 0.000500 | 6.583096e-02 | True | True | 50.00% | 50.00% | +1.129151e-05 | +1.129151e-05 | 54.85% | 0.05 | +0.00002344 | +0.00002344 | +0.00002344 | 50.00% | 50.00% | +1.129151e-05 | +0.00002344 | +0.00% | +0.000000e+00 | +0.00000000 | mixed | 0.00% | 54.65% | 1.000 | 0.139 | +1.477 | +0.338 | 0.715 | 0.50 | 0.00% |
| 7 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.30 | 2500 | 0.000500 | 5.761154e-02 | True | True | 50.00% | 50.00% | +1.035337e-05 | +1.035337e-05 | 56.94% | 0.04 | +0.00002021 | +0.00002021 | +0.00002021 | 50.00% | 50.00% | +1.035337e-05 | +0.00002021 | +0.00% | +0.000000e+00 | +0.00000000 | mixed | 0.00% | 57.27% | 1.000 | 0.136 | +1.495 | +0.359 | 0.797 | 0.50 | 0.00% |

survive_fee1_passrate_ge_0.5=0

## Decomposition

| rank | symbol | rule | h | gross_edge_npa | fee_cost_npa | adverse_cost_npa | scratch_cost_npa | net_npa | observed_npa | residual_npa | reject_rate | n_events | n_after_gate | n_filled |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | +7.986943e-05 | +5.506217e-05 | +7.784392e-06 | +0.000000e+00 | +1.702287e-05 | +7.092942e-05 | +5.390655e-05 | 0.00% | 2252 | 2252 | 1240 |
| 2 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | +8.163619e-05 | +5.799112e-05 | +8.441954e-06 | +0.000000e+00 | +1.520311e-05 | +7.169040e-05 | +5.648728e-05 | 0.00% | 1802 | 1802 | 1045 |
| 3 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | +8.928053e-05 | +5.658575e-05 | +7.823311e-06 | +0.000000e+00 | +2.487147e-05 | +3.814756e-05 | +1.327610e-05 | 0.00% | 2513 | 2513 | 1422 |
| 4 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | +8.398295e-05 | +5.932287e-05 | +8.588202e-06 | +0.000000e+00 | +1.607188e-05 | +2.458255e-05 | +8.510673e-06 | 0.00% | 2038 | 2038 | 1209 |
| 5 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | +9.931250e-05 | +5.907753e-05 | +1.126963e-05 | +0.000000e+00 | +2.896535e-05 | +3.144516e-05 | +2.479808e-06 | 0.00% | 1019 | 1019 | 602 |
| 6 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | +8.070699e-05 | +5.465360e-05 | +7.573188e-06 | +0.000000e+00 | +1.848020e-05 | +1.129151e-05 | -7.188691e-06 | 0.00% | 2858 | 2858 | 1562 |
| 7 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | +8.563372e-05 | +5.727170e-05 | +7.804173e-06 | +0.000000e+00 | +2.055784e-05 | +1.035337e-05 | -1.020448e-05 | 0.00% | 2661 | 2661 | 1524 |
