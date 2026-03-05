# PASSIVE_POCKET_RANKING

candidates=7 ranked=7
candidate_parse total_rows_seen=26 table_rows_seen=22 rows_with_pass_yes=7 candidates_parsed=7 candidates_unique=7 rows_skipped_missing_fields=0
fee_grid=[0.2, 0.5, 1.0] adverse_mult_grid=[1.0, 1.2, 1.5]
pass_threshold=0.330

| rank | symbol | rule | horizon | min_imb | min_int | max_spread | score | robust_core | robust_stress | pass_rate_core | pass_rate_stress | npa_core | npa_stress | attempt_fill_rate | attempts_per_min | score_raw_core | score_raw_stress | score_raw_min | stability_std_bps | best_fee_survive | insufficient_fill_rate |
|---:|---|---|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.50 | 3500 | 0.000300 | 1.719178e-01 | True | True | 50.00% | 50.00% | +3.657353e-05 | +2.604546e-05 | 65.81% | 0.02 | +0.00021681 | +0.00003853 | +0.00003853 | 1.127 | 1.00 | 0.00% |
| 2 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.50 | 2500 | 0.000500 | 0.000000e+00 | False | False | 50.00% | 50.00% | -4.547573e-05 | -5.201038e-05 | 57.26% | 0.05 | +0.00006245 | -0.00011006 | -0.00011006 | 0.551 | 1.00 | 0.00% |
| 3 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.50 | 2500 | 0.000300 | 0.000000e+00 | False | False | 50.00% | 50.00% | -4.491635e-05 | -5.245715e-05 | 62.44% | 0.04 | +0.00007284 | -0.00010016 | -0.00010016 | 0.696 | 1.00 | 0.00% |
| 4 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.40 | 2500 | 0.000300 | 0.000000e+00 | False | False | 44.44% | 44.44% | -4.347610e-05 | -5.142851e-05 | 62.32% | 0.05 | +0.00008401 | -0.00008935 | -0.00008935 | 0.848 | 1.00 | 0.00% |
| 5 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.40 | 2500 | 0.000500 | 0.000000e+00 | False | False | 38.89% | 38.89% | -3.930068e-05 | -4.644191e-05 | 56.26% | 0.06 | +0.00008846 | -0.00008443 | -0.00008443 | 0.657 | 1.00 | 0.00% |
| 6 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.30 | 2500 | 0.000500 | 0.000000e+00 | False | False | 27.78% | 22.22% | -5.171475e-05 | -5.902276e-05 | 56.97% | 0.06 | +0.00006885 | -0.00010402 | -0.00010402 | 0.726 | 1.00 | 0.00% |
| 7 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.20 | 2500 | 0.000500 | 0.000000e+00 | False | False | 22.22% | 22.22% | -6.061114e-05 | -6.790431e-05 | 55.14% | 0.07 | +0.00005634 | -0.00011684 | -0.00011684 | 0.694 | 1.00 | 0.00% |

survive_fee1_passrate_ge_0.5=3
