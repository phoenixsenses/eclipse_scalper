# PASSIVE_POCKET_RANKING

candidates=7 ranked=7
candidate_parse total_rows_seen=26 table_rows_seen=22 rows_with_pass_yes=7 candidates_parsed=7 candidates_unique=7 rows_skipped_missing_fields=0
fee_grid=[0.0] adverse_mult_grid=[0.3]
pass_threshold=0.500
mitigation_profile=baseline gate_config min_intensity_strong=0.000000 min_imbalance_strong=0.000000 max_spread_tight=0.000000 max_volatility_extreme=None vol_quantile_reject=0.010000 scratch_bps=0.0000 scratch_window_sec=0 scratch_taker_fee_bps=0.0000 scratch_slippage_bps=0.0000 horizon_sec_override=120

| rank | symbol | rule | horizon | min_imb | min_int | max_spread | score | robust_core | robust_stress | pass_rate_core | pass_rate_stress | npa_core | npa_stress | attempt_fill_rate | attempts_per_min | score_raw_core | score_raw_stress | score_raw_min | baseline_pass_rate_core | baseline_pass_rate_stress | baseline_npa_core | baseline_score_raw_core | delta_pass_rate_core | delta_npa_core | delta_score_raw_core | failure_reason_top | gate_reject_ratio | fill_rate_after_gate | avg_fee_bps | avg_adverse_bps_on_fills | avg_raw_return_bps_on_fills | avg_net_return_bps_on_fills | stability_std_bps | best_fee_survive | insufficient_fill_rate |
|---:|---|---|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.40 | 2500 | 0.000300 | 4.356384e-02 | True | True | 50.00% | 50.00% | +7.021345e-06 | +7.021345e-06 | 61.04% | 0.06 | +0.00001122 | +0.00001122 | +0.00001122 | 50.00% | 50.00% | +7.021345e-06 | +0.00001122 | +0.00% | +0.000000e+00 | +0.00000000 | adverse_dominates | 0.00% | 61.39% | 0.000 | 0.086 | -0.119 | -0.205 | 0.612 | 0.00 | 0.00% |
| 2 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.50 | 2500 | 0.000500 | 0.000000e+00 | False | False | 50.00% | 50.00% | -9.624895e-07 | -9.624895e-07 | 56.03% | 0.07 | -0.00000769 | -0.00000769 | -0.00000769 | 50.00% | 50.00% | -9.624895e-07 | -0.00000769 | +0.00% | +0.000000e+00 | +0.00000000 | adverse_dominates | 0.00% | 55.99% | 0.000 | 0.083 | -0.123 | -0.206 | 0.642 | 0.00 | 0.00% |
| 3 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.40 | 2500 | 0.000500 | 0.000000e+00 | False | False | 50.00% | 50.00% | -4.472670e-06 | -4.472670e-06 | 56.69% | 0.08 | -0.00000979 | -0.00000979 | -0.00000979 | 50.00% | 50.00% | -4.472670e-06 | -0.00000979 | +0.00% | +0.000000e+00 | +0.00000000 | adverse_dominates | 0.00% | 56.80% | 0.000 | 0.082 | +0.032 | -0.051 | 0.587 | 0.00 | 0.00% |
| 4 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.20 | 2500 | 0.000500 | 0.000000e+00 | False | False | 50.00% | 50.00% | -8.368986e-06 | -8.368986e-06 | 54.96% | 0.09 | -0.00001691 | -0.00001691 | -0.00001691 | 50.00% | 50.00% | -8.368986e-06 | -0.00001691 | +0.00% | +0.000000e+00 | +0.00000000 | adverse_dominates | 0.00% | 55.27% | 0.000 | 0.083 | +0.039 | -0.043 | 0.577 | 0.00 | 0.00% |
| 5 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.30 | 2500 | 0.000500 | 0.000000e+00 | False | False | 50.00% | 50.00% | -1.120754e-05 | -1.120754e-05 | 57.49% | 0.09 | -0.00002150 | -0.00002150 | -0.00002150 | 50.00% | 50.00% | -1.120754e-05 | -0.00002150 | +0.00% | +0.000000e+00 | +0.00000000 | adverse_dominates | 0.00% | 57.51% | 0.000 | 0.082 | +0.030 | -0.052 | 0.556 | 0.00 | 0.00% |
| 6 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.50 | 2500 | 0.000300 | 0.000000e+00 | False | False | 50.00% | 50.00% | -1.386063e-05 | -1.386063e-05 | 60.48% | 0.06 | -0.00002367 | -0.00002367 | -0.00002367 | 50.00% | 50.00% | -1.386063e-05 | -0.00002367 | +0.00% | +0.000000e+00 | +0.00000000 | adverse_dominates | 0.00% | 60.57% | 0.000 | 0.087 | -0.412 | -0.498 | 0.720 | 0.00 | 0.00% |
| 7 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.50 | 3500 | 0.000300 | 0.000000e+00 | False | False | 44.44% | 44.44% | -1.354689e-05 | -1.354689e-05 | 62.50% | 0.03 | -0.00002167 | -0.00002167 | -0.00002167 | 44.44% | 44.44% | -1.354689e-05 | -0.00002167 | +0.00% | +0.000000e+00 | +0.00000000 | adverse_dominates | 0.00% | 63.39% | 0.000 | 0.108 | -0.266 | -0.374 | 1.211 | 0.00 | 0.00% |

survive_fee1_passrate_ge_0.5=0

## Decomposition

| rank | symbol | rule | h | gross_edge_npa | fee_cost_npa | adverse_cost_npa | scratch_cost_npa | net_npa | observed_npa | residual_npa | reject_rate | n_events | n_after_gate | n_filled |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | -7.321217e-06 | +0.000000e+00 | +5.285869e-06 | +0.000000e+00 | -1.260709e-05 | +7.021345e-06 | +1.962843e-05 | 0.00% | 3642 | 3642 | 2236 |
| 2 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | -6.882426e-06 | +0.000000e+00 | +4.674684e-06 | +0.000000e+00 | -1.155711e-05 | -9.624895e-07 | +1.059462e-05 | 0.00% | 4140 | 4140 | 2318 |
| 3 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | +1.803092e-06 | +0.000000e+00 | +4.674234e-06 | +0.000000e+00 | -2.871142e-06 | -4.472670e-06 | -1.601528e-06 | 0.00% | 4579 | 4579 | 2601 |
| 4 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | +2.170635e-06 | +0.000000e+00 | +4.571196e-06 | +0.000000e+00 | -2.400561e-06 | -8.368986e-06 | -5.968425e-06 | 0.00% | 5258 | 5258 | 2906 |
| 5 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | +1.700787e-06 | +0.000000e+00 | +4.689806e-06 | +0.000000e+00 | -2.989019e-06 | -1.120754e-05 | -8.218517e-06 | 0.00% | 4888 | 4888 | 2811 |
| 6 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | -2.494048e-05 | +0.000000e+00 | +5.247521e-06 | +0.000000e+00 | -3.018800e-05 | -1.386063e-05 | +1.632738e-05 | 0.00% | 3289 | 3289 | 1992 |
| 7 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | -1.686452e-05 | +0.000000e+00 | +6.832117e-06 | +0.000000e+00 | -2.369664e-05 | -1.354689e-05 | +1.014975e-05 | 0.00% | 1830 | 1830 | 1160 |
