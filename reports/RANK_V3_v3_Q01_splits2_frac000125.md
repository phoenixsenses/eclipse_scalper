# PASSIVE_POCKET_RANKING

candidates=7 ranked=7
candidate_parse total_rows_seen=26 table_rows_seen=22 rows_with_pass_yes=7 candidates_parsed=7 candidates_unique=7 rows_skipped_missing_fields=0
fee_grid=[1.0] adverse_mult_grid=[1.2]
pass_threshold=0.330
mitigation_profile=anti_adverse_v3 gate_config min_intensity_strong=0.000000 min_imbalance_strong=0.000000 max_spread_tight=0.000000 max_volatility_extreme=None vol_quantile_reject=0.010000

| rank | symbol | rule | horizon | min_imb | min_int | max_spread | score | robust_core | robust_stress | pass_rate_core | pass_rate_stress | npa_core | npa_stress | attempt_fill_rate | attempts_per_min | score_raw_core | score_raw_stress | score_raw_min | baseline_pass_rate_core | baseline_pass_rate_stress | baseline_npa_core | baseline_score_raw_core | delta_pass_rate_core | delta_npa_core | delta_score_raw_core | stability_std_bps | best_fee_survive | insufficient_fill_rate |
|---:|---|---|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.40 | 2500 | 0.000500 | 0.000000e+00 | False | False | 0.00% | 0.00% | -1.543966e-04 | -1.543966e-04 | 56.77% | 0.06 | -0.00027635 | -0.00027635 | -0.00027635 | 0.00% | 0.00% | -1.769160e-04 | -0.00031247 | +0.00% | +2.251938e-05 | +0.00003612 | 0.892 | 0.00 | 0.00% |
| 2 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.20 | 2500 | 0.000500 | 0.000000e+00 | False | False | 0.00% | 0.00% | -1.593209e-04 | -1.593209e-04 | 55.73% | 0.06 | -0.00028397 | -0.00028397 | -0.00028397 | 0.00% | 0.00% | -1.845216e-04 | -0.00034359 | +0.00% | +2.520072e-05 | +0.00005961 | 0.661 | 0.00 | 0.00% |
| 3 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.30 | 2500 | 0.000500 | 0.000000e+00 | False | False | 0.00% | 0.00% | -1.611025e-04 | -1.611025e-04 | 57.78% | 0.06 | -0.00028667 | -0.00028667 | -0.00028667 | 0.00% | 0.00% | -2.029408e-04 | -0.00035538 | +0.00% | +4.183833e-05 | +0.00006871 | 0.683 | 0.00 | 0.00% |
| 4 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.50 | 2500 | 0.000500 | 0.000000e+00 | False | False | 0.00% | 0.00% | -1.969296e-04 | -1.969296e-04 | 55.97% | 0.05 | -0.00035603 | -0.00035603 | -0.00035603 | 0.00% | 0.00% | -2.074346e-04 | -0.00037106 | +0.00% | +1.050501e-05 | +0.00001503 | 0.963 | 0.00 | 0.00% |
| 5 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.40 | 2500 | 0.000300 | 0.000000e+00 | False | False | 0.00% | 0.00% | -2.037481e-04 | -2.037481e-04 | 59.33% | 0.04 | -0.00033757 | -0.00033757 | -0.00033757 | 0.00% | 0.00% | -1.839205e-04 | -0.00030336 | +0.00% | -1.982763e-05 | -0.00003421 | 0.793 | 0.00 | 0.00% |
| 6 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.50 | 3500 | 0.000300 | 0.000000e+00 | False | False | 0.00% | 0.00% | -2.128697e-04 | -2.128697e-04 | 61.76% | 0.02 | -0.00038834 | -0.00038834 | -0.00038834 | 0.00% | 0.00% | -2.223726e-04 | -0.00037814 | +0.00% | +9.502892e-06 | -0.00001020 | 1.390 | 0.00 | 0.00% |
| 7 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.50 | 2500 | 0.000300 | 0.000000e+00 | False | False | 0.00% | 0.00% | -2.360634e-04 | -2.360634e-04 | 59.11% | 0.04 | -0.00043026 | -0.00043026 | -0.00043026 | 0.00% | 0.00% | -2.286803e-04 | -0.00041278 | +0.00% | -7.383141e-06 | -0.00001748 | 0.725 | 0.00 | 0.00% |

survive_fee1_passrate_ge_0.5=0
