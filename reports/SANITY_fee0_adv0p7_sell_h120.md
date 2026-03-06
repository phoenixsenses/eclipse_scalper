# PASSIVE_POCKET_RANKING

candidates=7 ranked=7
candidate_parse total_rows_seen=26 table_rows_seen=22 rows_with_pass_yes=7 candidates_parsed=7 candidates_unique=7 rows_skipped_missing_fields=0
fee_grid=[0.0] adverse_mult_grid=[0.7]
pass_threshold=0.500
mitigation_profile=baseline gate_config min_intensity_strong=0.000000 min_imbalance_strong=0.000000 max_spread_tight=0.000000 max_volatility_extreme=None vol_quantile_reject=0.010000 scratch_bps=0.0000 scratch_window_sec=0 scratch_taker_fee_bps=0.0000 scratch_slippage_bps=0.0000 horizon_sec_override=120

| rank | symbol | rule | horizon | min_imb | min_int | max_spread | score | robust_core | robust_stress | pass_rate_core | pass_rate_stress | npa_core | npa_stress | attempt_fill_rate | attempts_per_min | score_raw_core | score_raw_stress | score_raw_min | baseline_pass_rate_core | baseline_pass_rate_stress | baseline_npa_core | baseline_score_raw_core | delta_pass_rate_core | delta_npa_core | delta_score_raw_core | failure_reason_top | gate_reject_ratio | fill_rate_after_gate | avg_fee_bps | avg_adverse_bps_on_fills | avg_raw_return_bps_on_fills | avg_net_return_bps_on_fills | stability_std_bps | best_fee_survive | insufficient_fill_rate |
|---:|---|---|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.50 | 3500 | 0.000300 | 4.091152e-02 | True | True | 50.00% | 50.00% | +8.709023e-06 | +8.709023e-06 | 61.81% | 0.03 | +0.00001337 | +0.00001337 | +0.00001337 | 50.00% | 50.00% | +8.709023e-06 | +0.00001337 | +0.00% | +0.000000e+00 | +0.00000000 | adverse_dominates | 0.00% | 63.04% | 0.000 | 0.251 | -0.091 | -0.342 | 1.129 | 0.00 | 0.00% |
| 2 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.20 | 2500 | 0.000500 | 0.000000e+00 | False | False | 50.00% | 50.00% | -1.380301e-06 | -1.380301e-06 | 55.17% | 0.09 | -0.00000285 | -0.00000285 | -0.00000285 | 50.00% | 50.00% | -1.380301e-06 | -0.00000285 | +0.00% | +0.000000e+00 | +0.00000000 | adverse_dominates | 0.00% | 55.30% | 0.000 | 0.193 | +0.123 | -0.070 | 0.380 | 0.00 | 0.00% |
| 3 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.30 | 2500 | 0.000500 | 0.000000e+00 | False | False | 50.00% | 50.00% | -4.122921e-06 | -4.122921e-06 | 57.14% | 0.09 | -0.00000771 | -0.00000771 | -0.00000771 | 50.00% | 50.00% | -4.122921e-06 | -0.00000771 | +0.00% | +0.000000e+00 | +0.00000000 | adverse_dominates | 0.00% | 57.54% | 0.000 | 0.190 | +0.071 | -0.119 | 0.398 | 0.00 | 0.00% |
| 4 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.40 | 2500 | 0.000500 | 0.000000e+00 | False | False | 50.00% | 50.00% | -7.217582e-06 | -7.217582e-06 | 56.82% | 0.08 | -0.00001548 | -0.00001548 | -0.00001548 | 50.00% | 50.00% | -7.217582e-06 | -0.00001548 | +0.00% | +0.000000e+00 | +0.00000000 | adverse_dominates | 0.00% | 56.79% | 0.000 | 0.192 | +0.137 | -0.055 | 0.456 | 0.00 | 0.00% |
| 5 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.40 | 2500 | 0.000300 | 0.000000e+00 | False | False | 50.00% | 50.00% | -1.148727e-05 | -1.148727e-05 | 60.88% | 0.06 | -0.00002043 | -0.00002043 | -0.00002043 | 50.00% | 50.00% | -1.148727e-05 | -0.00002043 | +0.00% | +0.000000e+00 | +0.00000000 | adverse_dominates | 0.00% | 61.34% | 0.000 | 0.200 | -0.109 | -0.310 | 0.463 | 0.00 | 0.00% |
| 6 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.50 | 2500 | 0.000500 | 0.000000e+00 | False | False | 50.00% | 50.00% | -2.128849e-05 | -2.128849e-05 | 55.95% | 0.07 | -0.00003976 | -0.00003976 | -0.00003976 | 50.00% | 50.00% | -2.128849e-05 | -0.00003976 | +0.00% | +0.000000e+00 | +0.00000000 | adverse_dominates | 0.00% | 55.81% | 0.000 | 0.195 | +0.022 | -0.172 | 0.557 | 0.00 | 0.00% |
| 7 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.50 | 2500 | 0.000300 | 0.000000e+00 | False | False | 44.44% | 44.44% | -3.549098e-05 | -3.549098e-05 | 61.03% | 0.06 | -0.00006271 | -0.00006271 | -0.00006271 | 44.44% | 44.44% | -3.549098e-05 | -0.00006271 | +0.00% | +0.000000e+00 | +0.00000000 | adverse_dominates | 0.00% | 60.67% | 0.000 | 0.201 | -0.285 | -0.486 | 0.577 | 0.00 | 0.00% |

survive_fee1_passrate_ge_0.5=0

## Decomposition

| rank | symbol | rule | h | gross_edge_npa | fee_cost_npa | adverse_cost_npa | scratch_cost_npa | net_npa | observed_npa | residual_npa | reject_rate | n_events | n_after_gate | n_filled |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | -5.728852e-06 | +0.000000e+00 | +1.581952e-05 | +0.000000e+00 | -2.154837e-05 | +8.709023e-06 | +3.025739e-05 | 0.00% | 1848 | 1848 | 1165 |
| 2 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | +6.792516e-06 | +0.000000e+00 | +1.068041e-05 | +0.000000e+00 | -3.887894e-06 | -1.380301e-06 | +2.507593e-06 | 0.00% | 5329 | 5329 | 2947 |
| 3 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | +4.059785e-06 | +0.000000e+00 | +1.093201e-05 | +0.000000e+00 | -6.872222e-06 | -4.122921e-06 | +2.749301e-06 | 0.00% | 4936 | 4936 | 2840 |
| 4 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | +7.755970e-06 | +0.000000e+00 | +1.088193e-05 | +0.000000e+00 | -3.125957e-06 | -7.217582e-06 | -4.091625e-06 | 0.00% | 4617 | 4617 | 2622 |
| 5 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | -6.705856e-06 | +0.000000e+00 | +1.228609e-05 | +0.000000e+00 | -1.899195e-05 | -1.148727e-05 | +7.504675e-06 | 0.00% | 3681 | 3681 | 2258 |
| 6 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | +1.254734e-06 | +0.000000e+00 | +1.087110e-05 | +0.000000e+00 | -9.616366e-06 | -2.128849e-05 | -1.167213e-05 | 0.00% | 4191 | 4191 | 2339 |
| 7 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | -1.726085e-05 | +0.000000e+00 | +1.221149e-05 | +0.000000e+00 | -2.947233e-05 | -3.549098e-05 | -6.018644e-06 | 0.00% | 3313 | 3313 | 2010 |
