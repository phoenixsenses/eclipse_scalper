# S34 Entry-Offset Decomposition

Generated: `2026-06-28T10:56:51.141863+00:00`

Research-only shadow analysis. No runner/config/live-rule changes.

Scope: `ETHUSDT BUY`, `LONG`, `cluster_notional >= 50,000`, `day_trend_bps >= -1e+06`, `TP120/SL40/BE30`, entry anchored to feature-factory `event_ts_ms` plus offset.

## First-To-Threshold Timing

| Metric | N | Median | P25 | P75 | Mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| first_ts -> threshold_cross | 450 | 31.1 | 5.8 | 72.0 | 52.0 |
| first_ts -> cluster_end | 450 | 156.1 | 78.2 | 220.8 | 149.9 |

## Real-Fill Offset Curve

| Offset sec | Filled N | No-fill % | Median | Mean | Cum | WR | Top3W removed | Exits | MFE med | MAE med | Hold med |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| 0 | 135 | 70.0% | -9.1 | 9.6 | 1298.5 | 34.1% | 888.4 | {'BE': 53, 'SL': 29, 'TIME': 30, 'TP': 23} | 43.5 | -5.8 | 1725.7 |
| 0.5 | 136 | 69.8% | -9.4 | 8.9 | 1214.4 | 33.1% | 804.9 | {'BE': 53, 'SL': 30, 'TIME': 30, 'TP': 23} | 43.1 | -7.1 | 1742.5 |
| 1 | 136 | 69.8% | -9.1 | 8.6 | 1168.9 | 32.4% | 762.0 | {'BE': 54, 'SL': 30, 'TIME': 29, 'TP': 23} | 42.9 | -7.1 | 1695.2 |
| 2 | 136 | 69.8% | -9.3 | 8.5 | 1152.1 | 33.1% | 747.4 | {'BE': 53, 'SL': 28, 'TIME': 33, 'TP': 22} | 43.8 | -7.0 | 1738.0 |
| 5 | 137 | 69.6% | -9.3 | 6.2 | 851.8 | 32.1% | 450.4 | {'BE': 50, 'SL': 34, 'TIME': 30, 'TP': 23} | 44.3 | -11.1 | 1707.8 |

## Knowable-Anchor Curve

`threshold_cross` is the first liquidation timestamp where cumulative cluster notional reaches 50,000. `cluster_end` is the retrospective end of the 300s feature-factory cluster.

| Anchor | Filled N | No-fill % | Median | Mean | Cum | WR | Top3W removed | Exits | MFE med | MAE med | Hold med |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| threshold_cross | 136 | 69.8% | -10.1 | 0.7 | 98.7 | 27.9% | -304.8 | {'BE': 40, 'SL': 47, 'TIME': 27, 'TP': 22} | 34.3 | -21.5 | 1512.1 |
| cluster_end | 138 | 69.3% | -29.3 | -13.2 | -1817.6 | 21.0% | -2209.2 | {'BE': 31, 'SL': 67, 'TIME': 26, 'TP': 14} | 14.5 | -38.2 | 1279.3 |

## No-Fill Counterfactual

Counterfactual uses mark-price path for events without executable book entry/exit fill at the same offset.

| Offset sec | No-fill N | CF Median | CF Mean | CF Cum | CF WR | CF Exits |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 315 | -8.0 | 18.4 | 5788.1 | 34.3% | {'BE': 148, 'SL': 47, 'TIME': 48, 'TP': 72} |
| 0.5 | 314 | -8.0 | 18.6 | 5834.3 | 34.4% | {'BE': 148, 'SL': 46, 'TIME': 48, 'TP': 72} |
| 1 | 314 | -8.0 | 17.6 | 5541.1 | 34.1% | {'BE': 145, 'SL': 51, 'TIME': 47, 'TP': 71} |
| 2 | 314 | -8.0 | 18.5 | 5814.1 | 35.0% | {'BE': 143, 'SL': 49, 'TIME': 50, 'TP': 72} |
| 5 | 313 | -8.0 | 17.4 | 5451.3 | 35.1% | {'BE': 135, 'SL': 58, 'TIME': 49, 'TP': 71} |

## Knowable-Anchor No-Fill Counterfactual

| Anchor | No-fill N | CF Median | CF Mean | CF Cum | CF WR | CF Exits |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| threshold_cross | 314 | -8.0 | 3.6 | 1128.7 | 26.4% | {'BE': 121, 'SL': 98, 'TIME': 42, 'TP': 53} |
| cluster_end | 312 | -48.0 | -17.9 | -5576.1 | 15.7% | {'BE': 73, 'SL': 177, 'TIME': 34, 'TP': 28} |

## Read

- This is an execution-realism diagnostic, not a new rule.
- `event_ts_ms` is the feature-factory first timestamp. The threshold-cross lag table estimates how much of the cascade has already elapsed before the 500K condition is knowable.
- If the real-fill offset curve decays sharply from 0s to 0.5/1/2s, the apparent edge is highly latency-sensitive.
- If no-fill counterfactuals outperform filled rows, the real-fill subset is likely adversely biased by missed fast winners.