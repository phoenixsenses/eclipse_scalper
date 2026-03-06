# PASSIVE_POCKET_RANKING

candidates=7 ranked=7
candidate_parse total_rows_seen=26 table_rows_seen=22 rows_with_pass_yes=7 candidates_parsed=7 candidates_unique=7 rows_skipped_missing_fields=0
fee_grid=[0.0, 0.2, 0.5, 1.0] adverse_mult_grid=[1.0, 1.2, 1.5]
pass_threshold=0.330
mitigation_profile=anti_adverse_v3 gate_config min_intensity_strong=0.000000 min_imbalance_strong=0.000000 max_spread_tight=0.000000 max_volatility_extreme=None vol_quantile_reject=0.010000

| rank | symbol | rule | horizon | min_imb | min_int | max_spread | score | robust_core | robust_stress | pass_rate_core | pass_rate_stress | npa_core | npa_stress | attempt_fill_rate | attempts_per_min | score_raw_core | score_raw_stress | score_raw_min | baseline_pass_rate_core | baseline_pass_rate_stress | baseline_npa_core | baseline_score_raw_core | delta_pass_rate_core | delta_npa_core | delta_score_raw_core | failure_reason_top | stability_std_bps | best_fee_survive | insufficient_fill_rate |
|---:|---|---|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|
| 1 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.50 | 2500 | 0.000500 | 0.000000e+00 | False | False | 27.78% | 22.22% | -1.474996e-04 | -1.541490e-04 | 54.86% | 0.06 | -0.00007207 | -0.00028491 | -0.00028491 | 11.11% | 5.56% | -1.595521e-04 | -0.00010088 | +16.67% | +1.205248e-05 | +0.00002880 | fees_dominate | 0.591 | 1.00 | 0.00% |
| 2 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.40 | 2500 | 0.000500 | 0.000000e+00 | False | False | 11.11% | 11.11% | -1.295864e-04 | -1.365710e-04 | 55.60% | 0.06 | -0.00004324 | -0.00025628 | -0.00025628 | 11.11% | 5.56% | -1.220389e-04 | -0.00003029 | +0.00% | -7.547587e-06 | -0.00001296 | fees_dominate | 0.548 | 1.00 | 0.00% |
| 3 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.40 | 2500 | 0.000300 | 0.000000e+00 | False | False | 11.11% | 11.11% | -1.517528e-04 | -1.596485e-04 | 59.91% | 0.05 | -0.00006571 | -0.00027954 | -0.00027954 | 16.67% | 11.11% | -1.442552e-04 | -0.00005315 | -5.56% | -7.497572e-06 | -0.00001255 | fees_dominate | 0.913 | 1.00 | 0.00% |
| 4 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.50 | 3500 | 0.000300 | 0.000000e+00 | False | False | 22.22% | 11.11% | -2.045230e-04 | -2.146068e-04 | 63.06% | 0.02 | -0.00015490 | -0.00037292 | -0.00037292 | 5.56% | 5.56% | -2.114940e-04 | -0.00017112 | +16.67% | +6.971002e-06 | +0.00001622 | fees_dominate | 1.085 | 1.00 | 0.00% |
| 5 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.20 | 2500 | 0.000500 | 0.000000e+00 | False | False | 5.56% | 5.56% | -1.385651e-04 | -1.456586e-04 | 54.92% | 0.07 | -0.00005613 | -0.00026939 | -0.00026939 | 5.56% | 5.56% | -1.435266e-04 | -0.00007755 | +0.00% | +4.961483e-06 | +0.00002141 | fees_dominate | 0.576 | 1.00 | 0.00% |
| 6 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.30 | 2500 | 0.000500 | 0.000000e+00 | False | False | 5.56% | 5.56% | -1.542653e-04 | -1.614473e-04 | 57.14% | 0.07 | -0.00007532 | -0.00028828 | -0.00028828 | 5.56% | 5.56% | -1.503757e-04 | -0.00007940 | +0.00% | -3.889551e-06 | +0.00000408 | fees_dominate | 0.555 | 1.00 | 0.00% |
| 7 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.50 | 2500 | 0.000300 | 0.000000e+00 | False | False | 5.56% | 5.56% | -1.717042e-04 | -1.797324e-04 | 60.78% | 0.05 | -0.00009971 | -0.00031370 | -0.00031370 | 5.56% | 5.56% | -1.677578e-04 | -0.00010263 | +0.00% | -3.946413e-06 | +0.00000292 | fees_dominate | 0.874 | 1.00 | 0.00% |

survive_fee1_passrate_ge_0.5=0
