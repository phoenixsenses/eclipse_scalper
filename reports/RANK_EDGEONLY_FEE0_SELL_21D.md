# PASSIVE_POCKET_RANKING

candidates=7 ranked=7
candidate_parse total_rows_seen=26 table_rows_seen=22 rows_with_pass_yes=7 candidates_parsed=7 candidates_unique=7 rows_skipped_missing_fields=0
fee_grid=[0.0] adverse_mult_grid=[1.0]
pass_threshold=0.330
mitigation_profile=baseline gate_config min_intensity_strong=0.000000 min_imbalance_strong=0.000000 max_spread_tight=0.000000 max_volatility_extreme=None vol_quantile_reject=0.010000

| rank | symbol | rule | horizon | min_imb | min_int | max_spread | score | robust_core | robust_stress | pass_rate_core | pass_rate_stress | npa_core | npa_stress | attempt_fill_rate | attempts_per_min | score_raw_core | score_raw_stress | score_raw_min | baseline_pass_rate_core | baseline_pass_rate_stress | baseline_npa_core | baseline_score_raw_core | delta_pass_rate_core | delta_npa_core | delta_score_raw_core | failure_reason_top | stability_std_bps | best_fee_survive | insufficient_fill_rate |
|---:|---|---|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|
| 1 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.40 | 2500 | 0.000500 | 0.000000e+00 | False | False | 50.00% | 50.00% | -4.740326e-05 | -4.740326e-05 | 56.01% | 0.06 | -0.00008410 | -0.00008410 | -0.00008410 | 50.00% | 50.00% | -4.740326e-05 | -0.00008410 | +0.00% | +0.000000e+00 | +0.00000000 | adverse_dominates | 0.610 | 0.00 | 0.00% |
| 2 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.50 | 2500 | 0.000500 | 0.000000e+00 | False | False | 50.00% | 50.00% | -6.121203e-05 | -6.121203e-05 | 55.99% | 0.06 | -0.00010742 | -0.00010742 | -0.00010742 | 50.00% | 50.00% | -6.121203e-05 | -0.00010742 | +0.00% | +0.000000e+00 | +0.00000000 | adverse_dominates | 0.629 | 0.00 | 0.00% |
| 3 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.20 | 2500 | 0.000500 | 0.000000e+00 | False | False | 50.00% | 50.00% | -7.229492e-05 | -7.229492e-05 | 55.51% | 0.07 | -0.00013144 | -0.00013144 | -0.00013144 | 50.00% | 50.00% | -7.229492e-05 | -0.00013144 | +0.00% | +0.000000e+00 | +0.00000000 | adverse_dominates | 0.540 | 0.00 | 0.00% |
| 4 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.30 | 2500 | 0.000500 | 0.000000e+00 | False | False | 50.00% | 50.00% | -8.091459e-05 | -8.091459e-05 | 57.07% | 0.07 | -0.00014205 | -0.00014205 | -0.00014205 | 50.00% | 50.00% | -8.091459e-05 | -0.00014205 | +0.00% | +0.000000e+00 | +0.00000000 | adverse_dominates | 0.580 | 0.00 | 0.00% |
| 5 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.40 | 2500 | 0.000300 | 0.000000e+00 | False | False | 44.44% | 44.44% | -4.268792e-05 | -4.268792e-05 | 61.04% | 0.05 | -0.00006903 | -0.00006903 | -0.00006903 | 44.44% | 44.44% | -4.268792e-05 | -0.00006903 | +0.00% | +0.000000e+00 | +0.00000000 | adverse_dominates | 0.691 | 0.00 | 0.00% |
| 6 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.50 | 3500 | 0.000300 | 0.000000e+00 | False | False | 44.44% | 44.44% | -6.817337e-05 | -6.817337e-05 | 63.98% | 0.02 | -0.00011790 | -0.00011790 | -0.00011790 | 44.44% | 44.44% | -6.817337e-05 | -0.00011790 | +0.00% | +0.000000e+00 | +0.00000000 | adverse_dominates | 1.024 | 0.00 | 0.00% |
| 7 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.50 | 2500 | 0.000300 | 0.000000e+00 | False | False | 38.89% | 38.89% | -6.534664e-05 | -6.534664e-05 | 61.43% | 0.05 | -0.00010610 | -0.00010610 | -0.00010610 | 38.89% | 38.89% | -6.534664e-05 | -0.00010610 | +0.00% | +0.000000e+00 | +0.00000000 | adverse_dominates | 0.798 | 0.00 | 0.00% |

survive_fee1_passrate_ge_0.5=0
