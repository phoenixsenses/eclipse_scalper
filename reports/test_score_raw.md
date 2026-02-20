# PASSIVE_POCKET_RANKING

candidates=1 ranked=1
candidate_parse total_rows_seen=1 table_rows_seen=1 rows_with_pass_yes=1 candidates_parsed=1 candidates_unique=1 rows_skipped_missing_fields=0
fee_grid=[0.5, 1.0, 1.5] adverse_mult_grid=[0.8, 1.0, 1.2]

| rank | symbol | rule | horizon | min_imb | min_int | max_spread | score | robust_core | robust_stress | net_per_attempt | attempt_fill_rate | attempts_per_min | score_raw_core | score_raw_stress | score_raw_min | stability_std_bps | best_fee_survive | insufficient_fill_rate |
|---:|---|---|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | ETHUSDT | intensity_spike_imbalance_cont | 60 | 0.50 | 2500 | 0.000250 | 0.000000e+00 | False | False | -4.000000e-06 | 40.00% | 90.00 | +0.00002000 | -0.00001000 | -0.00001000 | 0.000 | 0.50 | 0.00% |

survive_fee1_passrate_ge_0.5=0
