# Walkforward Eval Summary

- db: `eclipse_scalper\localtests\wf_regime_stability\a4d4d8ce8f5740e99f1465d4d471b028\db.sqlite`
- symbols: `ETHUSDT`
- strategy: `baseline`
- strategy_config: `{"period": 2}`

## Stability

| slices_count | pos_slices_count | pos_slices_frac | pnl_net_sum_total | pnl_net_sum_mean | pnl_net_sum_median | pnl_net_sum_std | pnl_net_sum_min | pnl_net_sum_max | worst_pnl_net_per_fill | fill_rate_mean | stability_score |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 1 | 0.500000 | 0.000000000000 | 0.000000000000 | 0.000000000000 | 2.000000000000 | -2.000000000000 | 2.000000000000 | -0.250000000000 | 1.000000000000 | -1.400000000000 |

## Top slices

| rank | slice_id | pnl_net_sum | pnl_net_per_fill | fills_count | fill_rate |
|---:|---|---:|---:|---:|---:|
| 1 | `slice_001_20240301T000000Z_20240301T000015Z` | 2.000000000000 | 0.250000000000 | 8 | 1.000000 |
| 2 | `slice_002_20240301T000020Z_20240301T000035Z` | -2.000000000000 | -0.250000000000 | 8 | 1.000000 |

## Worst slices

| rank | slice_id | pnl_net_sum | pnl_net_per_fill | fills_count | fill_rate |
|---:|---|---:|---:|---:|---:|
| 1 | `slice_002_20240301T000020Z_20240301T000035Z` | -2.000000000000 | -0.250000000000 | 8 | 1.000000 |
| 2 | `slice_001_20240301T000000Z_20240301T000015Z` | 2.000000000000 | 0.250000000000 | 8 | 1.000000 |
