# PASSIVE_POCKET_RANKING

candidates=4 ranked=4
fee_grid=[0.5] adverse_mult_grid=[0.8, 1.0, 1.2]

| rank | symbol | horizon | min_imb | min_int | max_spread | score | pass@fee1_adv1 | pass@fee1_adv1.2 | stability_std_bps | best_fee_survive |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | BTCUSDT | 120 | 0.90 | 10000 | 0.000150 | 0.0000 | 0.00% | 0.00% | 0.000 | 0.00 | 0.00% |
| 2 | BTCUSDT | 120 | 0.70 | 10000 | 0.000150 | 0.0000 | 0.00% | 0.00% | 0.000 | 0.50 | 0.00% |
| 3 | BTCUSDT | 120 | 0.85 | 10000 | 0.000150 | 0.0000 | 0.00% | 0.00% | 0.000 | 0.50 | 0.00% |
| 4 | BTCUSDT | 120 | 0.70 | 10000 | 0.000200 | 0.0000 | 0.00% | 0.00% | 0.000 | 0.50 | 0.00% |

survive_fee1_passrate_ge_0.5=0
