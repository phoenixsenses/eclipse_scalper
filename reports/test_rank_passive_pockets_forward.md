# PASSIVE_POCKET_RANKING

candidates=2 ranked=2
candidate_parse total_rows_seen=2 table_rows_seen=2 rows_with_pass_yes=2 candidates_parsed=2 candidates_unique=2 rows_skipped_missing_fields=0
fee_grid=[0.5, 1.0, 1.5] adverse_mult_grid=[0.8, 1.0, 1.2]

| rank | symbol | rule | horizon | min_imb | min_int | max_spread | score | robust_core | robust_stress | net_per_attempt | attempt_fill_rate | attempts_per_min | score_raw_core | score_raw_stress | score_raw_min | stability_std_bps | best_fee_survive | insufficient_fill_rate |
|---:|---|---|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | ETHUSDT | intensity_spike_imbalance_cont | 60 | 0.50 | 2500 | 0.000250 | 1.600000e-01 | True | True | +1.600000e-05 | 40.00% | 120.00 | +0.00001000 | -0.00001000 | -0.00001000 | 0.000 | 1.00 | 0.00% |
| 2 | BTCUSDT | intensity_spike_imbalance_cont | 60 | 0.50 | 2500 | 0.000250 | 0.000000e+00 | False | False | -1.400000e-05 | 40.00% | 120.00 | +0.00000500 | -0.00003500 | -0.00003500 | 0.150 | 0.50 | 0.00% |

survive_fee1_passrate_ge_0.5=1
