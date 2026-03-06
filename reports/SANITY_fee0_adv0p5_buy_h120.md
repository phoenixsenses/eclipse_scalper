# PASSIVE_POCKET_RANKING

candidates=7 ranked=7
candidate_parse total_rows_seen=26 table_rows_seen=22 rows_with_pass_yes=7 candidates_parsed=7 candidates_unique=7 rows_skipped_missing_fields=0
fee_grid=[0.0] adverse_mult_grid=[0.5]
pass_threshold=0.500
mitigation_profile=baseline gate_config min_intensity_strong=0.000000 min_imbalance_strong=0.000000 max_spread_tight=0.000000 max_volatility_extreme=None vol_quantile_reject=0.010000 scratch_bps=0.0000 scratch_window_sec=0 scratch_taker_fee_bps=0.0000 scratch_slippage_bps=0.0000 horizon_sec_override=120

| rank | symbol | rule | horizon | min_imb | min_int | max_spread | score | robust_core | robust_stress | pass_rate_core | pass_rate_stress | npa_core | npa_stress | attempt_fill_rate | attempts_per_min | score_raw_core | score_raw_stress | score_raw_min | baseline_pass_rate_core | baseline_pass_rate_stress | baseline_npa_core | baseline_score_raw_core | delta_pass_rate_core | delta_npa_core | delta_score_raw_core | failure_reason_top | gate_reject_ratio | fill_rate_after_gate | avg_fee_bps | avg_adverse_bps_on_fills | avg_raw_return_bps_on_fills | avg_net_return_bps_on_fills | stability_std_bps | best_fee_survive | insufficient_fill_rate |
|---:|---|---|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.40 | 2500 | 0.000300 | 1.265500e-01 | True | True | 55.56% | 55.56% | +2.268391e-05 | +2.268391e-05 | 61.33% | 0.07 | +0.00003796 | +0.00003796 | +0.00003796 | 55.56% | 55.56% | +2.268391e-05 | +0.00003796 | +0.00% | +0.000000e+00 | +0.00000000 | mixed | 0.00% | 61.41% | 0.000 | 0.143 | +0.269 | +0.126 | 0.792 | 0.00 | 0.00% |
| 2 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.40 | 2500 | 0.000500 | 6.843127e-02 | True | True | 55.56% | 55.56% | +1.164895e-05 | +1.164895e-05 | 56.44% | 0.08 | +0.00002045 | +0.00002045 | +0.00002045 | 55.56% | 55.56% | +1.164895e-05 | +0.00002045 | +0.00% | +0.000000e+00 | +0.00000000 | mixed | 0.00% | 56.57% | 0.000 | 0.137 | +0.360 | +0.224 | 0.702 | 0.00 | 0.00% |
| 3 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.50 | 3500 | 0.000300 | 5.707862e-02 | True | True | 50.00% | 50.00% | +1.113094e-05 | +1.113094e-05 | 63.35% | 0.03 | +0.00001820 | +0.00001820 | +0.00001820 | 50.00% | 50.00% | +1.113094e-05 | +0.00001820 | +0.00% | +0.000000e+00 | +0.00000000 | mixed | 0.00% | 63.59% | 0.000 | 0.178 | +0.498 | +0.319 | 0.950 | 0.00 | 0.00% |
| 4 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.20 | 2500 | 0.000500 | 3.955910e-02 | True | True | 55.56% | 55.56% | +7.112488e-06 | +7.112488e-06 | 55.46% | 0.09 | +0.00001280 | +0.00001280 | +0.00001280 | 55.56% | 55.56% | +7.112488e-06 | +0.00001280 | +0.00% | +0.000000e+00 | +0.00000000 | mixed | 0.00% | 55.07% | 0.000 | 0.138 | +0.359 | +0.221 | 0.798 | 0.00 | 0.00% |
| 5 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.30 | 2500 | 0.000500 | 3.105063e-02 | True | True | 50.00% | 50.00% | +5.453625e-06 | +5.453625e-06 | 57.19% | 0.09 | +0.00000896 | +0.00000896 | +0.00000896 | 50.00% | 50.00% | +5.453625e-06 | +0.00000896 | +0.00% | +0.000000e+00 | +0.00000000 | mixed | 0.00% | 57.30% | 0.000 | 0.135 | +0.328 | +0.193 | 0.756 | 0.00 | 0.00% |
| 6 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.50 | 2500 | 0.000500 | 0.000000e+00 | False | False | 50.00% | 50.00% | -3.480300e-06 | -3.480300e-06 | 55.54% | 0.07 | -0.00000750 | -0.00000750 | -0.00000750 | 50.00% | 50.00% | -3.480300e-06 | -0.00000750 | +0.00% | +0.000000e+00 | +0.00000000 | mixed | 0.00% | 56.03% | 0.000 | 0.140 | +0.278 | +0.139 | 0.812 | 0.00 | 0.00% |
| 7 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.50 | 2500 | 0.000300 | 0.000000e+00 | False | False | 44.44% | 44.44% | -6.096464e-06 | -6.096464e-06 | 60.91% | 0.06 | -0.00001018 | -0.00001018 | -0.00001018 | 44.44% | 44.44% | -6.096464e-06 | -0.00001018 | +0.00% | +0.000000e+00 | +0.00000000 | adverse_dominates | 0.00% | 60.77% | 0.000 | 0.144 | +0.076 | -0.068 | 0.867 | 0.00 | 0.00% |

survive_fee1_passrate_ge_0.5=0

## Decomposition

| rank | symbol | rule | h | gross_edge_npa | fee_cost_npa | adverse_cost_npa | scratch_cost_npa | net_npa | observed_npa | residual_npa | reject_rate | n_events | n_after_gate | n_filled |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | +1.649777e-05 | +0.000000e+00 | +8.788560e-06 | +0.000000e+00 | +7.709214e-06 | +2.268391e-05 | +1.497469e-05 | 0.00% | 3682 | 3682 | 2261 |
| 2 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | +2.037580e-05 | +0.000000e+00 | +7.725775e-06 | +0.000000e+00 | +1.265002e-05 | +1.164895e-05 | -1.001075e-06 | 0.00% | 4600 | 4600 | 2602 |
| 3 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | +3.164801e-05 | +0.000000e+00 | +1.134713e-05 | +0.000000e+00 | +2.030088e-05 | +1.113094e-05 | -9.169942e-06 | 0.00% | 1843 | 1843 | 1172 |
| 4 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | +1.976538e-05 | +0.000000e+00 | +7.593558e-06 | +0.000000e+00 | +1.217182e-05 | +7.112488e-06 | -5.059330e-06 | 0.00% | 5301 | 5301 | 2919 |
| 5 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | +1.880333e-05 | +0.000000e+00 | +7.762675e-06 | +0.000000e+00 | +1.104066e-05 | +5.453625e-06 | -5.587031e-06 | 0.00% | 4918 | 4918 | 2818 |
| 6 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | +1.559730e-05 | +0.000000e+00 | +7.818080e-06 | +0.000000e+00 | +7.779216e-06 | -3.480300e-06 | -1.125952e-05 | 0.00% | 4157 | 4157 | 2329 |
| 7 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | +4.610410e-06 | +0.000000e+00 | +8.755720e-06 | +0.000000e+00 | -4.145310e-06 | -6.096464e-06 | -1.951154e-06 | 0.00% | 3319 | 3319 | 2017 |
