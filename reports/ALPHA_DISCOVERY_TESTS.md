# Alpha Discovery Tests

- db: `data\microstructure.db`
- candidates_tested: `690`
- runtime_sec: `1.578`
- verdict_counts: `{"CONFIRMED_REJECTION": 3, "PROMOTE_SHADOW": 19, "REJECT": 599, "WATCH_ONLY": 69}`

## Promote Shadow

| candidate | kind | n | WR | mean | net8 | folds8 | uplift | reasons |
|---|---|---:|---:|---:|---:|---:|---:|---|
| BTCUSDT_BUY100000_SHORT_900_UTC07 | sibling_lane | 22 | 90.91% | 45.19 | 37.19 | 5/5 | 40.54 |  |
| ETHUSDT_BUY250000_SHORT_900_UTC14 | sibling_lane | 33 | 75.76% | 43.13 | 35.13 | 4/5 | 34.81 |  |
| ETHUSDT_BUY1000000_SHORT_900_SESSION_US | sibling_lane | 20 | 70.00% | 34.74 | 26.74 | 4/5 | 29.86 |  |
| SOLUSDT_BUY50000_SHORT_900_FUNDING_NEGATIVE | sibling_lane | 20 | 85.00% | 31.83 | 23.83 | 4/5 | 16.05 |  |
| ETHUSDT_BUY250000_SHORT_900_UTC19 | sibling_lane | 26 | 73.08% | 30.45 | 22.45 | 5/5 | 22.13 |  |
| ETHUSDT_S34_SHORT_900_SESSION_US | s34_quality | 25 | 72.00% | 29.01 | 21.01 | 5/5 | 24.44 |  |
| SOLUSDT_BUY25000_SHORT_900_FUNDING_NEGATIVE | sibling_lane | 34 | 82.35% | 27.39 | 19.39 | 4/5 | 14.16 |  |
| ETHUSDT_S34_SHORT_900_BASIS_POSITIVE | s34_quality | 31 | 80.65% | 27.38 | 19.38 | 4/5 | 22.81 |  |
| ETHUSDT_BUY500000_SHORT_900_SESSION_US | sibling_lane | 62 | 72.58% | 25.86 | 17.86 | 4/5 | 18.34 |  |
| BTCUSDT_SELL250000_LONG_900_UTC13 | sibling_lane | 23 | 69.57% | 25.44 | 17.44 | 4/5 | 24.27 |  |
| BTCUSDT_SELL100000_LONG_900_UTC15 | sibling_lane | 21 | 85.71% | 25.28 | 17.28 | 4/5 | 16.20 |  |
| ETHUSDT_S34_SHORT_900_CONFIDENCE_MEDIUM | s34_quality | 22 | 72.73% | 24.63 | 16.63 | 4/5 | 20.05 |  |
| ETHUSDT_BUY250000_SHORT_900_SESSION_US | sibling_lane | 126 | 69.05% | 19.95 | 11.95 | 4/5 | 11.62 |  |
| SOLUSDT_BUY50000_SHORT_900_BASELINE | cross_asset_transfer | 46 | 73.91% | 15.78 | 7.78 | 4/5 | 0.00 |  |
| SOLUSDT_BUY100000_SHORT_900_BASELINE | cross_asset_transfer | 24 | 70.83% | 14.41 | 6.41 | 4/5 | 0.00 |  |
| ETHUSDT_BUY100000_SHORT_900_SESSION_US | sibling_lane | 83 | 63.86% | 13.61 | 5.61 | 4/5 | 8.23 |  |
| SOLUSDT_BUY25000_SHORT_900_BASELINE | cross_asset_transfer | 76 | 68.42% | 13.22 | 5.22 | 4/5 | 0.00 |  |
| BTCUSDT_SELL500000_LONG_900_SESSION_US | sibling_lane | 30 | 73.33% | 10.72 | 2.72 | 4/5 | 8.43 |  |
| ETHUSDT_SELL250000_LONG_900_SESSION_US | sibling_lane | 94 | 63.83% | 8.91 | 0.91 | 4/5 | 6.84 |  |

## Watch Only

| candidate | kind | n | WR | mean | net8 | folds8 | uplift | reasons |
|---|---|---:|---:|---:|---:|---:|---:|---|
| BTCUSDT_SELL100000_LONG_900_UTC13 | sibling_lane | 30 | 80.00% | 32.66 | 24.66 | 3/5 | 23.58 | unstable_net8_folds |
| ETHUSDT_BUY500000_SHORT_900_FUNDING_POSITIVE | sibling_lane | 47 | 63.83% | 21.42 | 13.42 | 3/5 | 13.90 | unstable_net8_folds |
| ETHUSDT_BUY100000_SHORT_900_SESSION_LATE_US | sibling_lane | 23 | 65.22% | 18.94 | 10.94 | 3/5 | 13.55 | unstable_gross_folds,unstable_net8_folds |
| ETHUSDT_BUY250000_SHORT_900_FUNDING_POSITIVE | sibling_lane | 131 | 67.94% | 18.72 | 10.72 | 3/5 | 10.40 | unstable_net8_folds |
| ETHUSDT_S34_SHORT_900_SINGLE_LARGE | s34_quality | 21 | 80.95% | 17.32 | 9.32 | 3/5 | 12.75 | unstable_net8_folds |
| BTCUSDT_SELL100000_LONG_900_SESSION_EUROPE | sibling_lane | 71 | 70.42% | 16.38 | 8.38 | 3/5 | 7.30 | unstable_net8_folds |
| SOLUSDT_SELL25000_LONG_900_FUNDING_POSITIVE | sibling_lane | 24 | 79.17% | 16.07 | 8.07 | 3/5 | 12.94 | unstable_net8_folds |
| BTCUSDT_BUY250000_SHORT_900_UTC14 | sibling_lane | 42 | 57.14% | 14.45 | 6.45 | 3/5 | 10.52 | wr_below_gate,unstable_gross_folds,unstable_net8_folds |
| BTCUSDT_SELL500000_LONG_900_SESSION_EUROPE | sibling_lane | 25 | 72.00% | 14.13 | 6.13 | 2/5 | 11.84 | unstable_net8_folds |
| ETHUSDT_BUY100000_SHORT_900_UTC14 | sibling_lane | 27 | 55.56% | 14.05 | 6.05 | 4/5 | 8.67 | wr_below_gate |
| SOLUSDT_BUY25000_SHORT_900_SESSION_US | sibling_lane | 20 | 80.00% | 13.44 | 5.44 | 3/5 | 0.22 | unstable_net8_folds |
| BTCUSDT_BUY500000_SHORT_900_SESSION_US | sibling_lane | 56 | 67.86% | 12.51 | 4.51 | 2/5 | 4.04 | unstable_net8_folds |
| BTCUSDT_BUY500000_SHORT_900_SESSION_LATE_US | sibling_lane | 22 | 63.64% | 12.09 | 4.09 | 3/5 | 3.63 | unstable_gross_folds,unstable_net8_folds |
| BTCUSDT_SELL250000_LONG_900_UTC15 | sibling_lane | 21 | 61.90% | 10.27 | 2.27 | 2/5 | 9.10 | unstable_gross_folds,unstable_net8_folds |
| ETHUSDT_SELL100000_LONG_900_SESSION_ASIA | sibling_lane | 86 | 53.49% | 10.23 | 2.23 | 2/5 | 4.94 | wr_below_gate,unstable_gross_folds,unstable_net8_folds |
| BTCUSDT_BUY500000_SHORT_900_FUNDING_POSITIVE | sibling_lane | 41 | 63.41% | 9.87 | 1.87 | 3/5 | 1.40 | unstable_net8_folds |
| BTCUSDT_SELL250000_LONG_900_SESSION_US | sibling_lane | 94 | 64.89% | 9.82 | 1.82 | 3/5 | 8.65 | unstable_net8_folds |
| BTCUSDT_SELL250000_LONG_900_SESSION_EUROPE | sibling_lane | 69 | 63.77% | 9.75 | 1.75 | 3/5 | 8.57 | unstable_net8_folds |
| ETHUSDT_SELL100000_LONG_900_SESSION_US | sibling_lane | 87 | 59.77% | 9.66 | 1.66 | 3/5 | 4.36 | wr_below_gate,unstable_net8_folds |
| BTCUSDT_SELL100000_LONG_900_FUNDING_NEGATIVE | sibling_lane | 281 | 63.35% | 9.51 | 1.51 | 2/5 | 0.43 | unstable_net8_folds |
| ETHUSDT_SELL100000_LONG_900_FUNDING_POSITIVE | sibling_lane | 62 | 48.39% | 9.51 | 1.51 | 2/5 | 4.21 | wr_below_gate,unstable_gross_folds,unstable_net8_folds |
| BTCUSDT_SELL100000_LONG_900_SESSION_ASIA | sibling_lane | 58 | 63.79% | 9.42 | 1.42 | 1/5 | 0.34 | unstable_net8_folds |
| ETHUSDT_SELL1000000_LONG_900_FUNDING_NEGATIVE | sibling_lane | 21 | 47.62% | 9.22 | 1.22 | 2/5 | 16.78 | wr_below_gate,unstable_net8_folds |
| BTCUSDT_SELL100000_LONG_900_BASELINE | cross_asset_transfer | 300 | 63.00% | 9.08 | 1.08 | 2/5 | 0.00 | unstable_net8_folds |
| ETHUSDT_BUY100000_SHORT_900_FUNDING_POSITIVE | sibling_lane | 90 | 56.67% | 9.02 | 1.02 | 3/5 | 3.64 | wr_below_gate,unstable_net8_folds |
| SOLUSDT_BUY25000_SHORT_900_SESSION_ASIA | sibling_lane | 36 | 55.56% | 8.89 | 0.89 | 3/5 | -4.33 | wr_below_gate,unstable_net8_folds |
| BTCUSDT_SELL100000_LONG_900_SESSION_US | sibling_lane | 124 | 62.90% | 8.69 | 0.69 | 2/5 | -0.39 | unstable_net8_folds |
| BTCUSDT_BUY250000_SHORT_900_SESSION_US | sibling_lane | 120 | 54.17% | 8.49 | 0.49 | 2/5 | 4.57 | wr_below_gate,unstable_net8_folds |
| BTCUSDT_BUY500000_SHORT_900_BASELINE | cross_asset_transfer | 143 | 60.14% | 8.47 | 0.47 | 2/5 | 0.00 | unstable_net8_folds |
| BTCUSDT_BUY100000_SHORT_900_UTC16 | sibling_lane | 25 | 76.00% | 8.42 | 0.42 | 2/5 | 3.76 | unstable_net8_folds |

## Confirmed Rejections

| candidate | kind | n | WR | mean | net8 | folds8 | uplift | reasons |
|---|---|---:|---:|---:|---:|---:|---:|---|
| ETHUSDT_S34_SHORT_900_CLUSTERED | anti_alpha | 44 | 47.73% | -5.67 | -13.67 | 1/5 | -10.24 | negative_or_zero_mean |
| ETHUSDT_S34_SHORT_900_SESSION_NON_US | anti_alpha | 48 | 52.08% | -8.16 | -16.16 | 1/5 | -12.73 | negative_or_zero_mean |
| ETHUSDT_S34_SHORT_900_BASIS_NONPOSITIVE | anti_alpha | 42 | 42.86% | -12.26 | -20.26 | 1/5 | -16.83 | negative_or_zero_mean |

## Shadow Telemetry

- path: `logs\telemetry.jsonl`
- rows: `0`
- no labeled shadow outcomes yet
