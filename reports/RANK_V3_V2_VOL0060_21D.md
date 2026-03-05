# PASSIVE_POCKET_RANKING

candidates=7 ranked=7
candidate_parse total_rows_seen=26 table_rows_seen=22 rows_with_pass_yes=7 candidates_parsed=7 candidates_unique=7 rows_skipped_missing_fields=0
fee_grid=[0.2, 0.5, 1.0] adverse_mult_grid=[1.0, 1.2, 1.5]
pass_threshold=0.330
mitigation_profile=anti_adverse_v2 gate_config min_intensity_strong=0.000000 min_imbalance_strong=0.000000 max_spread_tight=0.000000 max_volatility_extreme=0.006 vol_quantile_reject=0.010000

| rank | symbol | rule | horizon | min_imb | min_int | max_spread | score | robust_core | robust_stress | pass_rate_core | pass_rate_stress | npa_core | npa_stress | attempt_fill_rate | attempts_per_min | score_raw_core | score_raw_stress | score_raw_min | baseline_pass_rate_core | baseline_pass_rate_stress | baseline_npa_core | baseline_score_raw_core | delta_pass_rate_core | delta_npa_core | delta_score_raw_core | stability_std_bps | best_fee_survive | insufficient_fill_rate |
|---:|---|---|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.50 | 2500 | 0.000300 | 0.000000e+00 | False | False | 22.22% | 22.22% | -3.188546e-05 | -4.092575e-05 | 61.73% | 0.05 | +0.00011053 | -0.00006354 | -0.00006354 | 22.22% | 22.22% | -3.188546e-05 | +0.00011053 | +0.00% | +0.000000e+00 | +0.00000000 | 1.005 | 1.00 | 0.00% |
| 2 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.50 | 2500 | 0.000500 | 0.000000e+00 | False | False | 27.78% | 22.22% | -4.006138e-05 | -4.746317e-05 | 55.34% | 0.06 | +0.00008839 | -0.00008460 | -0.00008460 | 27.78% | 22.22% | -4.006138e-05 | +0.00008839 | +0.00% | +0.000000e+00 | +0.00000000 | 0.933 | 1.00 | 0.00% |
| 3 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.40 | 2500 | 0.000300 | 0.000000e+00 | False | False | 22.22% | 22.22% | -5.153094e-05 | -5.954616e-05 | 60.97% | 0.05 | +0.00007766 | -0.00009510 | -0.00009510 | 22.22% | 22.22% | -5.153094e-05 | +0.00007766 | +0.00% | +0.000000e+00 | +0.00000000 | 0.962 | 1.00 | 0.00% |
| 4 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.50 | 3500 | 0.000300 | 0.000000e+00 | False | False | 27.78% | 22.22% | -1.050536e-04 | -1.168694e-04 | 61.78% | 0.02 | -0.00000628 | -0.00018457 | -0.00018457 | 27.78% | 22.22% | -1.050536e-04 | -0.00000628 | +0.00% | +0.000000e+00 | +0.00000000 | 1.203 | 1.00 | 0.00% |
| 5 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.40 | 2500 | 0.000500 | 0.000000e+00 | False | False | 11.11% | 11.11% | -7.824885e-05 | -8.534214e-05 | 55.93% | 0.06 | +0.00001381 | -0.00015922 | -0.00015922 | 11.11% | 11.11% | -7.824885e-05 | +0.00001381 | +0.00% | +0.000000e+00 | +0.00000000 | 0.894 | 1.00 | 0.00% |
| 6 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.20 | 2500 | 0.000500 | 0.000000e+00 | False | False | 5.56% | 5.56% | -1.109350e-04 | -1.181422e-04 | 54.62% | 0.07 | -0.00004756 | -0.00022085 | -0.00022085 | 5.56% | 5.56% | -1.109350e-04 | -0.00004756 | +0.00% | +0.000000e+00 | +0.00000000 | 0.946 | 1.00 | 0.00% |
| 7 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.30 | 2500 | 0.000500 | 0.000000e+00 | False | False | 5.56% | 5.56% | -1.110013e-04 | -1.182856e-04 | 57.25% | 0.06 | -0.00004032 | -0.00021325 | -0.00021325 | 5.56% | 5.56% | -1.110013e-04 | -0.00004032 | +0.00% | +0.000000e+00 | +0.00000000 | 0.971 | 1.00 | 0.00% |

survive_fee1_passrate_ge_0.5=0
