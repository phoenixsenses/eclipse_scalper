# PASSIVE_POCKET_RANKING

candidates=7 ranked=7
candidate_parse total_rows_seen=26 table_rows_seen=22 rows_with_pass_yes=7 candidates_parsed=7 candidates_unique=7 rows_skipped_missing_fields=0
fee_grid=[0.0] adverse_mult_grid=[1.0]
pass_threshold=0.330
mitigation_profile=baseline gate_config min_intensity_strong=0.000000 min_imbalance_strong=0.000000 max_spread_tight=0.000000 max_volatility_extreme=None vol_quantile_reject=0.010000

| rank | symbol | rule | horizon | min_imb | min_int | max_spread | score | robust_core | robust_stress | pass_rate_core | pass_rate_stress | npa_core | npa_stress | attempt_fill_rate | attempts_per_min | score_raw_core | score_raw_stress | score_raw_min | baseline_pass_rate_core | baseline_pass_rate_stress | baseline_npa_core | baseline_score_raw_core | delta_pass_rate_core | delta_npa_core | delta_score_raw_core | failure_reason_top | stability_std_bps | best_fee_survive | insufficient_fill_rate |
|---:|---|---|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|
| 1 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.50 | 2500 | 0.000500 | 0.000000e+00 | False | False | 50.00% | 50.00% | -2.618071e-05 | -2.618071e-05 | 54.92% | 0.06 | -0.00005086 | -0.00005086 | -0.00005086 | 50.00% | 50.00% | -2.618071e-05 | -0.00005086 | +0.00% | +0.000000e+00 | +0.00000000 | adverse_dominates | 0.758 | 0.00 | 0.00% |
| 2 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.50 | 2500 | 0.000300 | 0.000000e+00 | False | False | 50.00% | 50.00% | -2.868305e-05 | -2.868305e-05 | 58.87% | 0.05 | -0.00005388 | -0.00005388 | -0.00005388 | 50.00% | 50.00% | -2.868305e-05 | -0.00005388 | +0.00% | +0.000000e+00 | +0.00000000 | adverse_dominates | 0.804 | 0.00 | 0.00% |
| 3 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.40 | 2500 | 0.000500 | 0.000000e+00 | False | False | 50.00% | 50.00% | -3.015347e-05 | -3.015347e-05 | 55.61% | 0.06 | -0.00005537 | -0.00005537 | -0.00005537 | 50.00% | 50.00% | -3.015347e-05 | -0.00005537 | +0.00% | +0.000000e+00 | +0.00000000 | adverse_dominates | 0.742 | 0.00 | 0.00% |
| 4 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.40 | 2500 | 0.000300 | 0.000000e+00 | False | False | 44.44% | 44.44% | -3.598338e-05 | -3.598338e-05 | 59.92% | 0.05 | -0.00006313 | -0.00006313 | -0.00006313 | 44.44% | 44.44% | -3.598338e-05 | -0.00006313 | +0.00% | +0.000000e+00 | +0.00000000 | adverse_dominates | 0.769 | 0.00 | 0.00% |
| 5 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.30 | 2500 | 0.000500 | 0.000000e+00 | False | False | 44.44% | 44.44% | -4.984153e-05 | -4.984153e-05 | 56.91% | 0.07 | -0.00008608 | -0.00008608 | -0.00008608 | 44.44% | 44.44% | -4.984153e-05 | -0.00008608 | +0.00% | +0.000000e+00 | +0.00000000 | adverse_dominates | 0.627 | 0.00 | 0.00% |
| 6 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.20 | 2500 | 0.000500 | 0.000000e+00 | False | False | 38.89% | 38.89% | -4.999896e-05 | -4.999896e-05 | 55.05% | 0.07 | -0.00009007 | -0.00009007 | -0.00009007 | 38.89% | 38.89% | -4.999896e-05 | -0.00009007 | +0.00% | +0.000000e+00 | +0.00000000 | adverse_dominates | 0.643 | 0.00 | 0.00% |
| 7 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.50 | 3500 | 0.000300 | 0.000000e+00 | False | False | 38.89% | 38.89% | -7.765208e-05 | -7.765208e-05 | 61.64% | 0.02 | -0.00013144 | -0.00013144 | -0.00013144 | 38.89% | 38.89% | -7.765208e-05 | -0.00013144 | +0.00% | +0.000000e+00 | +0.00000000 | adverse_dominates | 0.882 | 0.00 | 0.00% |

survive_fee1_passrate_ge_0.5=0
