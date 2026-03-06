# PASSIVE_POCKET_RANKING

candidates=8 ranked=8
candidate_parse total_rows_seen=26 table_rows_seen=22 rows_with_pass_yes=20 candidates_parsed=20 candidates_unique=8 rows_skipped_missing_fields=0
fee_grid=[0.0, 0.1, 0.25, 0.5, 0.75, 1.0] adverse_mult_grid=[1.0]

| rank | symbol | rule | horizon | min_imb | min_int | max_spread | score | pass@fee1_adv1 | pass@fee1_adv1.2 | stability_std_bps | best_fee_survive | insufficient_fill_rate |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | ETHUSDT | micro_edge_v2_passive_alpha | 120 | 0.50 | 2500 | 0.000500 | 0.0000 | 0.00% | 0.00% | 0.663 | 0.75 | 0.00% |
| 2 | ETHUSDT | micro_edge_v2_passive_alpha | 120 | 0.40 | 2500 | 0.000500 | 0.0000 | 5.00% | 0.00% | 0.599 | 1.00 | 0.00% |
| 3 | ETHUSDT | micro_edge_v2_passive_alpha | 120 | 0.30 | 2500 | 0.000500 | 0.0000 | 10.00% | 0.00% | 0.575 | 1.00 | 0.00% |
| 4 | ETHUSDT | micro_edge_v2_passive_alpha | 120 | 0.20 | 2500 | 0.000500 | 0.0000 | 10.00% | 0.00% | 0.504 | 1.00 | 0.00% |
| 5 | ETHUSDT | micro_edge_v2_passive_alpha | 120 | 0.40 | 2500 | 0.000300 | 0.0000 | 0.00% | 0.00% | 0.461 | 0.75 | 0.00% |
| 6 | ETHUSDT | micro_edge_v2_passive_alpha | 120 | 0.50 | 1500 | 0.000500 | 0.0000 | 0.00% | 0.00% | 0.463 | 0.50 | 0.00% |
| 7 | ETHUSDT | micro_edge_v2_passive_alpha | 120 | 0.40 | 1500 | 0.000500 | 0.0000 | 0.00% | 0.00% | 0.460 | 0.75 | 0.00% |
| 8 | ETHUSDT | micro_edge_v2_passive_alpha | 120 | 0.50 | 2500 | 0.000300 | 0.0000 | 0.00% | 0.00% | 0.407 | 0.75 | 0.00% |

survive_fee1_passrate_ge_0.5=0
