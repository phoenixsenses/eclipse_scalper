# PASSIVE_POCKET_RANKING

candidates=7 ranked=7
candidate_parse total_rows_seen=26 table_rows_seen=22 rows_with_pass_yes=7 candidates_parsed=7 candidates_unique=7 rows_skipped_missing_fields=0
fee_grid=[0.0] adverse_mult_grid=[0.0]
pass_threshold=0.500
mitigation_profile=baseline gate_config min_intensity_strong=0.000000 min_imbalance_strong=0.000000 max_spread_tight=0.000000 max_volatility_extreme=None vol_quantile_reject=0.010000 scratch_bps=0.0000 scratch_window_sec=0 scratch_taker_fee_bps=0.0000 scratch_slippage_bps=0.0000 horizon_sec_override=120

| rank | symbol | rule | horizon | min_imb | min_int | max_spread | score | robust_core | robust_stress | pass_rate_core | pass_rate_stress | npa_core | npa_stress | attempt_fill_rate | attempts_per_min | score_raw_core | score_raw_stress | score_raw_min | baseline_pass_rate_core | baseline_pass_rate_stress | baseline_npa_core | baseline_score_raw_core | delta_pass_rate_core | delta_npa_core | delta_score_raw_core | failure_reason_top | gate_reject_ratio | fill_rate_after_gate | avg_fee_bps | avg_adverse_bps_on_fills | avg_raw_return_bps_on_fills | avg_net_return_bps_on_fills | stability_std_bps | best_fee_survive | insufficient_fill_rate |
|---:|---|---|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.50 | 3500 | 0.000300 | 2.845432e-01 | True | True | 55.56% | 55.56% | +5.984778e-05 | +5.984778e-05 | 63.93% | 0.03 | +0.00009181 | +0.00009181 | +0.00009181 | 55.56% | 55.56% | +5.984778e-05 | +0.00009181 | +0.00% | +0.000000e+00 | +0.00000000 | mixed | 0.00% | 63.66% | 0.000 | 0.000 | +0.513 | +0.513 | 1.103 | 0.00 | 0.00% |
| 2 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.40 | 2500 | 0.000300 | 1.595176e-01 | True | True | 55.56% | 55.56% | +2.474847e-05 | +2.474847e-05 | 62.25% | 0.06 | +0.00003843 | +0.00003843 | +0.00003843 | 55.56% | 55.56% | +2.474847e-05 | +0.00003843 | +0.00% | +0.000000e+00 | +0.00000000 | mixed | 0.00% | 61.94% | 0.000 | 0.000 | +0.111 | +0.111 | 0.551 | 0.00 | 0.00% |
| 3 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.50 | 2500 | 0.000300 | 1.113136e-01 | True | True | 50.00% | 50.00% | +1.716995e-05 | +1.716995e-05 | 61.52% | 0.06 | +0.00002547 | +0.00002547 | +0.00002547 | 50.00% | 50.00% | +1.716995e-05 | +0.00002547 | +0.00% | +0.000000e+00 | +0.00000000 | adverse_dominates | 0.00% | 61.43% | 0.000 | 0.000 | -0.022 | -0.022 | 0.542 | 0.00 | 0.00% |
| 4 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.40 | 2500 | 0.000500 | 1.068448e-01 | True | True | 50.00% | 50.00% | +1.575860e-05 | +1.575860e-05 | 57.79% | 0.08 | +0.00002593 | +0.00002593 | +0.00002593 | 50.00% | 50.00% | +1.575860e-05 | +0.00002593 | +0.00% | +0.000000e+00 | +0.00000000 | mixed | 0.00% | 57.70% | 0.000 | 0.000 | +0.187 | +0.187 | 0.475 | 0.00 | 0.00% |
| 5 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.50 | 2500 | 0.000500 | 6.940558e-02 | True | True | 50.00% | 50.00% | +1.034082e-05 | +1.034082e-05 | 56.86% | 0.07 | +0.00001761 | +0.00001761 | +0.00001761 | 50.00% | 50.00% | +1.034082e-05 | +0.00001761 | +0.00% | +0.000000e+00 | +0.00000000 | mixed | 0.00% | 57.06% | 0.000 | 0.000 | +0.120 | +0.120 | 0.490 | 0.00 | 0.00% |
| 6 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.20 | 2500 | 0.000500 | 4.980748e-02 | True | True | 50.00% | 50.00% | +7.739213e-06 | +7.739213e-06 | 55.52% | 0.09 | +0.00001526 | +0.00001526 | +0.00001526 | 50.00% | 50.00% | +7.739213e-06 | +0.00001526 | +0.00% | +0.000000e+00 | +0.00000000 | mixed | 0.00% | 56.18% | 0.000 | 0.000 | +0.121 | +0.121 | 0.554 | 0.00 | 0.00% |
| 7 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.30 | 2500 | 0.000500 | 0.000000e+00 | False | False | 50.00% | 50.00% | -1.851079e-06 | -1.851079e-06 | 57.85% | 0.08 | -0.00000303 | -0.00000303 | -0.00000303 | 50.00% | 50.00% | -1.851079e-06 | -0.00000303 | +0.00% | +0.000000e+00 | +0.00000000 | mixed | 0.00% | 58.55% | 0.000 | 0.000 | +0.067 | +0.067 | 0.615 | 0.00 | 0.00% |

survive_fee1_passrate_ge_0.5=0

## Decomposition

| rank | symbol | rule | h | gross_edge_npa | fee_cost_npa | adverse_cost_npa | scratch_cost_npa | net_npa | observed_npa | residual_npa | reject_rate | n_events | n_after_gate | n_filled |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | +3.265051e-05 | +0.000000e+00 | +0.000000e+00 | +0.000000e+00 | +3.265051e-05 | +5.984778e-05 | +2.719727e-05 | 0.00% | 1816 | 1816 | 1156 |
| 2 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | +6.883564e-06 | +0.000000e+00 | +0.000000e+00 | +0.000000e+00 | +6.883564e-06 | +2.474847e-05 | +1.786491e-05 | 0.00% | 3642 | 3642 | 2256 |
| 3 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | -1.335489e-06 | +0.000000e+00 | +0.000000e+00 | +0.000000e+00 | -1.335489e-06 | +1.716995e-05 | +1.850544e-05 | 0.00% | 3280 | 3280 | 2015 |
| 4 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | +1.078086e-05 | +0.000000e+00 | +0.000000e+00 | +0.000000e+00 | +1.078086e-05 | +1.575860e-05 | +4.977744e-06 | 0.00% | 4539 | 4539 | 2619 |
| 5 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | +6.874520e-06 | +0.000000e+00 | +0.000000e+00 | +0.000000e+00 | +6.874520e-06 | +1.034082e-05 | +3.466304e-06 | 0.00% | 4113 | 4113 | 2347 |
| 6 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | +6.796461e-06 | +0.000000e+00 | +0.000000e+00 | +0.000000e+00 | +6.796461e-06 | +7.739213e-06 | +9.427520e-07 | 0.00% | 5221 | 5217 | 2931 |
| 7 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | +3.912878e-06 | +0.000000e+00 | +0.000000e+00 | +0.000000e+00 | +3.912878e-06 | -1.851079e-06 | -5.763957e-06 | 0.00% | 4848 | 4844 | 2836 |
