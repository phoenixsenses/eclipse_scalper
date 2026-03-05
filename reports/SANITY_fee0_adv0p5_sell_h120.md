# PASSIVE_POCKET_RANKING

candidates=7 ranked=7
candidate_parse total_rows_seen=26 table_rows_seen=22 rows_with_pass_yes=7 candidates_parsed=7 candidates_unique=7 rows_skipped_missing_fields=0
fee_grid=[0.0] adverse_mult_grid=[0.5]
pass_threshold=0.500
mitigation_profile=baseline gate_config min_intensity_strong=0.000000 min_imbalance_strong=0.000000 max_spread_tight=0.000000 max_volatility_extreme=None vol_quantile_reject=0.010000 scratch_bps=0.0000 scratch_window_sec=0 scratch_taker_fee_bps=0.0000 scratch_slippage_bps=0.0000 horizon_sec_override=120

| rank | symbol | rule | horizon | min_imb | min_int | max_spread | score | robust_core | robust_stress | pass_rate_core | pass_rate_stress | npa_core | npa_stress | attempt_fill_rate | attempts_per_min | score_raw_core | score_raw_stress | score_raw_min | baseline_pass_rate_core | baseline_pass_rate_stress | baseline_npa_core | baseline_score_raw_core | delta_pass_rate_core | delta_npa_core | delta_score_raw_core | failure_reason_top | gate_reject_ratio | fill_rate_after_gate | avg_fee_bps | avg_adverse_bps_on_fills | avg_raw_return_bps_on_fills | avg_net_return_bps_on_fills | stability_std_bps | best_fee_survive | insufficient_fill_rate |
|---:|---|---|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.50 | 3500 | 0.000300 | 1.569682e-01 | True | True | 55.56% | 55.56% | +3.313094e-05 | +3.313094e-05 | 63.89% | 0.03 | +0.00005261 | +0.00005261 | +0.00005261 | 55.56% | 55.56% | +3.313094e-05 | +0.00005261 | +0.00% | +0.000000e+00 | +0.00000000 | mixed | 0.00% | 64.80% | 0.000 | 0.179 | +0.404 | +0.225 | 1.111 | 0.00 | 0.00% |
| 2 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.40 | 2500 | 0.000300 | 1.394954e-01 | True | True | 55.56% | 55.56% | +2.258473e-05 | +2.258473e-05 | 60.66% | 0.06 | +0.00003890 | +0.00003890 | +0.00003890 | 55.56% | 55.56% | +2.258473e-05 | +0.00003890 | +0.00% | +0.000000e+00 | +0.00000000 | mixed | 0.00% | 61.67% | 0.000 | 0.143 | +0.242 | +0.098 | 0.619 | 0.00 | 0.00% |
| 3 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.40 | 2500 | 0.000500 | 1.132255e-01 | True | True | 55.56% | 55.56% | +1.758517e-05 | +1.758517e-05 | 56.18% | 0.08 | +0.00003158 | +0.00003158 | +0.00003158 | 55.56% | 55.56% | +1.758517e-05 | +0.00003158 | +0.00% | +0.000000e+00 | +0.00000000 | mixed | 0.00% | 56.51% | 0.000 | 0.137 | +0.364 | +0.227 | 0.553 | 0.00 | 0.00% |
| 4 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.50 | 2500 | 0.000300 | 9.837404e-02 | True | True | 50.00% | 50.00% | +1.697953e-05 | +1.697953e-05 | 60.88% | 0.06 | +0.00002579 | +0.00002579 | +0.00002579 | 50.00% | 50.00% | +1.697953e-05 | +0.00002579 | +0.00% | +0.000000e+00 | +0.00000000 | adverse_dominates | 0.00% | 61.29% | 0.000 | 0.144 | +0.087 | -0.058 | 0.726 | 0.00 | 0.00% |
| 5 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.50 | 2500 | 0.000500 | 9.485411e-02 | True | True | 50.00% | 50.00% | +1.532197e-05 | +1.532197e-05 | 55.81% | 0.07 | +0.00002504 | +0.00002504 | +0.00002504 | 50.00% | 50.00% | +1.532197e-05 | +0.00002504 | +0.00% | +0.000000e+00 | +0.00000000 | mixed | 0.00% | 56.05% | 0.000 | 0.140 | +0.163 | +0.023 | 0.615 | 0.00 | 0.00% |
| 6 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.20 | 2500 | 0.000500 | 0.000000e+00 | False | False | 50.00% | 50.00% | -1.151491e-05 | -1.151491e-05 | 54.16% | 0.09 | -0.00002376 | -0.00002376 | -0.00002376 | 50.00% | 50.00% | -1.151491e-05 | -0.00002376 | +0.00% | +0.000000e+00 | +0.00000000 | mixed | 0.00% | 54.89% | 0.000 | 0.138 | +0.180 | +0.042 | 0.641 | 0.00 | 0.00% |
| 7 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.30 | 2500 | 0.000500 | 0.000000e+00 | False | False | 44.44% | 44.44% | -1.984819e-05 | -1.984819e-05 | 56.43% | 0.09 | -0.00003736 | -0.00003736 | -0.00003736 | 44.44% | 44.44% | -1.984819e-05 | -0.00003736 | +0.00% | +0.000000e+00 | +0.00000000 | adverse_dominates | 0.00% | 57.20% | 0.000 | 0.136 | +0.044 | -0.091 | 0.641 | 0.00 | 0.00% |

survive_fee1_passrate_ge_0.5=0

## Decomposition

| rank | symbol | rule | h | gross_edge_npa | fee_cost_npa | adverse_cost_npa | scratch_cost_npa | net_npa | observed_npa | residual_npa | reject_rate | n_events | n_after_gate | n_filled |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | +2.616533e-05 | +0.000000e+00 | +1.160895e-05 | +0.000000e+00 | +1.455638e-05 | +3.313094e-05 | +1.857456e-05 | 0.00% | 1835 | 1835 | 1189 |
| 2 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | +1.489632e-05 | +0.000000e+00 | +8.832123e-06 | +0.000000e+00 | +6.064195e-06 | +2.258473e-05 | +1.652053e-05 | 0.00% | 3688 | 3684 | 2272 |
| 3 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | +2.056615e-05 | +0.000000e+00 | +7.731215e-06 | +0.000000e+00 | +1.283493e-05 | +1.758517e-05 | +4.750238e-06 | 0.00% | 4623 | 4619 | 2610 |
| 4 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | +5.316516e-06 | +0.000000e+00 | +8.856263e-06 | +0.000000e+00 | -3.539747e-06 | +1.697953e-05 | +2.051928e-05 | 0.00% | 3321 | 3317 | 2033 |
| 5 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | +9.113604e-06 | +0.000000e+00 | +7.829828e-06 | +0.000000e+00 | +1.283776e-06 | +1.532197e-05 | +1.403820e-05 | 0.00% | 4175 | 4171 | 2338 |
| 6 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | +9.889413e-06 | +0.000000e+00 | +7.570925e-06 | +0.000000e+00 | +2.318488e-06 | -1.151491e-05 | -1.383339e-05 | 0.00% | 5326 | 5322 | 2921 |
| 7 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | +2.532408e-06 | +0.000000e+00 | +7.753774e-06 | +0.000000e+00 | -5.221366e-06 | -1.984819e-05 | -1.462683e-05 | 0.00% | 4939 | 4935 | 2823 |
