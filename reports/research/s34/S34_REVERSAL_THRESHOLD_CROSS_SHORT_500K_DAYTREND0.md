# S34 Entry-Offset Decomposition

Generated: `2026-06-28T10:56:51.168561+00:00`

Research-only shadow analysis. No runner/config/live-rule changes.

Scope: `ETHUSDT BUY`, `SHORT`, `cluster_notional >= 500,000`, `day_trend_bps >= 0`, `TP60/SL40/BE30`, entry anchored to feature-factory `event_ts_ms` plus offset.

## First-To-Threshold Timing

| Metric | N | Median | P25 | P75 | Mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| first_ts -> threshold_cross | 97 | 81.0 | 37.7 | 187.2 | 105.7 |
| first_ts -> cluster_end | 97 | 187.0 | 107.0 | 233.0 | 174.4 |

## Real-Fill Offset Curve

| Offset sec | Filled N | No-fill % | Median | Mean | Cum | WR | Top3W removed | Exits | MFE med | MAE med | Hold med |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| 0 | 51 | 47.4% | -49.2 | -38.2 | -1946.9 | 13.7% | -2108.5 | {'BE': 4, 'SL': 38, 'TIME': 4, 'TP': 5} | 3.2 | -41.0 | 261.4 |
| 0.5 | 51 | 47.4% | -48.8 | -38.4 | -1957.9 | 13.7% | -2120.6 | {'BE': 4, 'SL': 38, 'TIME': 4, 'TP': 5} | 2.8 | -40.7 | 260.9 |
| 1 | 51 | 47.4% | -48.7 | -38.3 | -1952.9 | 13.7% | -2115.6 | {'BE': 4, 'SL': 38, 'TIME': 4, 'TP': 5} | 2.4 | -40.7 | 260.4 |
| 2 | 51 | 47.4% | -49.5 | -39.2 | -2000.9 | 15.7% | -2165.7 | {'BE': 4, 'SL': 39, 'TIME': 3, 'TP': 5} | 2.0 | -40.8 | 259.4 |
| 5 | 50 | 48.5% | -49.3 | -37.2 | -1860.0 | 16.0% | -2024.2 | {'BE': 5, 'SL': 37, 'TIME': 2, 'TP': 6} | 2.3 | -40.7 | 265.7 |

## Knowable-Anchor Curve

`threshold_cross` is the first liquidation timestamp where cumulative cluster notional reaches 500,000. `cluster_end` is the retrospective end of the 300s feature-factory cluster.

| Anchor | Filled N | No-fill % | Median | Mean | Cum | WR | Top3W removed | Exits | MFE med | MAE med | Hold med |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| threshold_cross | 51 | 47.4% | -9.0 | -6.9 | -353.4 | 33.3% | -518.2 | {'BE': 11, 'SL': 20, 'TIME': 7, 'TP': 13} | 32.2 | -33.4 | 684.9 |
| cluster_end | 50 | 48.5% | -9.3 | -2.6 | -130.6 | 38.0% | -297.3 | {'BE': 11, 'SL': 17, 'TIME': 8, 'TP': 14} | 35.8 | -10.6 | 788.6 |

## No-Fill Counterfactual

Counterfactual uses mark-price path for events without executable book entry/exit fill at the same offset.

| Offset sec | No-fill N | CF Median | CF Mean | CF Cum | CF WR | CF Exits |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 46 | -48.0 | -33.0 | -1519.4 | 10.9% | {'BE': 6, 'SL': 33, 'TIME': 4, 'TP': 3} |
| 0.5 | 46 | -48.0 | -33.0 | -1519.3 | 10.9% | {'BE': 6, 'SL': 33, 'TIME': 4, 'TP': 3} |
| 1 | 46 | -48.0 | -31.7 | -1458.7 | 13.0% | {'BE': 5, 'SL': 33, 'TIME': 4, 'TP': 4} |
| 2 | 46 | -48.0 | -31.7 | -1457.9 | 13.0% | {'BE': 5, 'SL': 33, 'TIME': 4, 'TP': 4} |
| 5 | 47 | -48.0 | -31.9 | -1501.0 | 12.8% | {'BE': 4, 'SL': 34, 'TIME': 5, 'TP': 4} |

## Knowable-Anchor No-Fill Counterfactual

| Anchor | No-fill N | CF Median | CF Mean | CF Cum | CF WR | CF Exits |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| threshold_cross | 46 | -8.0 | -2.2 | -101.5 | 39.1% | {'BE': 10, 'SL': 16, 'TIME': 6, 'TP': 14} |
| cluster_end | 47 | -8.0 | 9.5 | 447.1 | 44.7% | {'BE': 16, 'SL': 9, 'TIME': 3, 'TP': 19} |

## Read

- This is an execution-realism diagnostic, not a new rule.
- `event_ts_ms` is the feature-factory first timestamp. The threshold-cross lag table estimates how much of the cascade has already elapsed before the 500K condition is knowable.
- If the real-fill offset curve decays sharply from 0s to 0.5/1/2s, the apparent edge is highly latency-sensitive.
- If no-fill counterfactuals outperform filled rows, the real-fill subset is likely adversely biased by missed fast winners.