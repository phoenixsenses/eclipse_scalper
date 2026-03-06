# PASSIVE_POCKET_RANKING

candidates=7 ranked=7
candidate_parse total_rows_seen=26 table_rows_seen=22 rows_with_pass_yes=7 candidates_parsed=7 candidates_unique=7 rows_skipped_missing_fields=0
fee_grid=[1.0] adverse_mult_grid=[1.2]
pass_threshold=0.330
mitigation_profile=anti_adverse_v3 gate_config min_intensity_strong=0.000000 min_imbalance_strong=0.000000 max_spread_tight=0.000000 max_volatility_extreme=None vol_quantile_reject=0.010000

| rank | symbol | rule | horizon | min_imb | min_int | max_spread | score | robust_core | robust_stress | pass_rate_core | pass_rate_stress | npa_core | npa_stress | attempt_fill_rate | attempts_per_min | score_raw_core | score_raw_stress | score_raw_min | baseline_pass_rate_core | baseline_pass_rate_stress | baseline_npa_core | baseline_score_raw_core | delta_pass_rate_core | delta_npa_core | delta_score_raw_core | failure_reason_top | stability_std_bps | best_fee_survive | insufficient_fill_rate |
|---:|---|---|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|
| 1 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.50 | 2500 | 0.000500 | 0.000000e+00 | False | False | 27.78% | 27.78% | -8.952855e-05 | -8.952855e-05 | 57.14% | 0.06 | -0.00016217 | -0.00016217 | -0.00016217 | 16.67% | 16.67% | -8.825496e-05 | -0.00016010 | +11.11% | -1.273588e-06 | -0.00000208 | fees_dominate | 0.741 | 1.00 | 0.00% |
| 2 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.40 | 2500 | 0.000500 | 0.000000e+00 | False | False | 16.67% | 16.67% | -1.043724e-04 | -1.043724e-04 | 57.04% | 0.06 | -0.00018526 | -0.00018526 | -0.00018526 | 11.11% | 11.11% | -1.046295e-04 | -0.00018647 | +5.56% | +2.571465e-07 | +0.00000121 | fees_dominate | 0.737 | 1.00 | 0.00% |
| 3 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.30 | 2500 | 0.000500 | 0.000000e+00 | False | False | 16.67% | 16.67% | -1.207675e-04 | -1.207675e-04 | 57.48% | 0.07 | -0.00020839 | -0.00020839 | -0.00020839 | 11.11% | 11.11% | -1.322551e-04 | -0.00023119 | +5.56% | +1.148759e-05 | +0.00002280 | fees_dominate | 0.753 | 1.00 | 0.00% |
| 4 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.20 | 2500 | 0.000500 | 0.000000e+00 | False | False | 16.67% | 16.67% | -1.223416e-04 | -1.223416e-04 | 55.35% | 0.07 | -0.00022370 | -0.00022370 | -0.00022370 | 16.67% | 16.67% | -1.306628e-04 | -0.00024551 | +0.00% | +8.321265e-06 | +0.00002181 | fees_dominate | 0.675 | 1.00 | 0.00% |
| 5 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.50 | 2500 | 0.000300 | 0.000000e+00 | False | False | 11.11% | 11.11% | -1.066863e-04 | -1.066863e-04 | 60.80% | 0.05 | -0.00017634 | -0.00017634 | -0.00017634 | 11.11% | 11.11% | -1.026004e-04 | -0.00017082 | +0.00% | -4.085822e-06 | -0.00000553 | fees_dominate | 0.717 | 1.00 | 0.00% |
| 6 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.40 | 2500 | 0.000300 | 0.000000e+00 | False | False | 5.56% | 5.56% | -1.150005e-04 | -1.150005e-04 | 60.51% | 0.05 | -0.00019811 | -0.00019811 | -0.00019811 | 5.56% | 5.56% | -1.121437e-04 | -0.00018511 | +0.00% | -2.856791e-06 | -0.00001299 | fees_dominate | 0.691 | 1.00 | 0.00% |
| 7 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.50 | 3500 | 0.000300 | 0.000000e+00 | False | False | 5.56% | 5.56% | -1.295697e-04 | -1.295697e-04 | 62.26% | 0.02 | -0.00021608 | -0.00021608 | -0.00021608 | 5.56% | 5.56% | -1.238944e-04 | -0.00020653 | +0.00% | -5.675393e-06 | -0.00000955 | fees_dominate | 1.447 | 1.00 | 0.00% |

survive_fee1_passrate_ge_0.5=0
