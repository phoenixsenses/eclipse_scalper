# S34 Entry-Offset Decomposition

Generated: `2026-06-28T10:52:17.510910+00:00`

Research-only shadow analysis. No runner/config/live-rule changes.

Scope: `ETHUSDT BUY`, `cluster_notional >= 500K`, `day_trend_bps >= 0`, `TP60/SL40/BE30`, entry anchored to feature-factory `event_ts_ms` plus offset.

## First-To-Threshold Timing

| Metric | N | Median | P25 | P75 | Mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| first_ts -> threshold_cross | 97 | 81.0 | 37.7 | 187.2 | 105.7 |
| first_ts -> cluster_end | 97 | 187.0 | 107.0 | 233.0 | 174.4 |

## Real-Fill Offset Curve

| Offset sec | Filled N | No-fill % | Median | Mean | Cum | WR | Top3W removed | Exits | MFE med | MAE med | Hold med |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| 0 | 51 | 47.4% | 48.8 | 23.3 | 1186.0 | 60.8% | 930.9 | {'BE': 12, 'SL': 7, 'TIME': 5, 'TP': 27} | 60.2 | -2.6 | 426.7 |
| 0.5 | 51 | 47.4% | 48.8 | 23.1 | 1175.7 | 60.8% | 921.2 | {'BE': 12, 'SL': 7, 'TIME': 5, 'TP': 27} | 60.2 | -2.5 | 450.4 |
| 1 | 50 | 48.5% | 49.9 | 25.6 | 1280.3 | 64.0% | 1026.4 | {'BE': 10, 'SL': 7, 'TIME': 5, 'TP': 28} | 60.2 | -2.3 | 544.8 |
| 2 | 51 | 47.4% | 49.8 | 26.8 | 1368.7 | 64.7% | 1114.2 | {'BE': 11, 'SL': 6, 'TIME': 5, 'TP': 29} | 60.3 | -1.3 | 448.9 |
| 5 | 51 | 47.4% | 49.5 | 22.3 | 1138.8 | 60.8% | 884.2 | {'BE': 10, 'SL': 9, 'TIME': 5, 'TP': 27} | 60.1 | -2.2 | 611.5 |

## Knowable-Anchor Curve

`threshold_cross` is the first liquidation timestamp where cumulative cluster notional reaches 500K. `cluster_end` is the retrospective end of the 300s feature-factory cluster.

| Anchor | Filled N | No-fill % | Median | Mean | Cum | WR | Top3W removed | Exits | MFE med | MAE med | Hold med |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| threshold_cross | 49 | 49.5% | -14.0 | -12.2 | -599.4 | 26.5% | -781.1 | {'BE': 12, 'SL': 20, 'TIME': 7, 'TP': 10} | 33.3 | -30.2 | 658.8 |
| cluster_end | 50 | 48.5% | -20.8 | -16.4 | -822.4 | 22.0% | -1006.7 | {'BE': 10, 'SL': 22, 'TIME': 9, 'TP': 9} | 15.5 | -35.9 | 646.7 |

## No-Fill Counterfactual

Counterfactual uses mark-price path for events without executable book entry/exit fill at the same offset.

| Offset sec | No-fill N | CF Median | CF Mean | CF Cum | CF WR | CF Exits |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 46 | 37.1 | 21.1 | 971.8 | 54.3% | {'BE': 18, 'SL': 2, 'TIME': 3, 'TP': 23} |
| 0.5 | 46 | 37.1 | 21.1 | 972.1 | 54.3% | {'BE': 18, 'SL': 2, 'TIME': 3, 'TP': 23} |
| 1 | 47 | 52.0 | 20.9 | 984.1 | 55.3% | {'BE': 17, 'SL': 3, 'TIME': 3, 'TP': 24} |
| 2 | 46 | 14.7 | 19.0 | 872.7 | 52.2% | {'BE': 18, 'SL': 3, 'TIME': 3, 'TP': 22} |
| 5 | 46 | 14.2 | 19.0 | 873.5 | 52.2% | {'BE': 18, 'SL': 3, 'TIME': 3, 'TP': 22} |

## Knowable-Anchor No-Fill Counterfactual

| Anchor | No-fill N | CF Median | CF Mean | CF Cum | CF WR | CF Exits |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| threshold_cross | 48 | -8.0 | -5.3 | -253.5 | 33.3% | {'BE': 10, 'SL': 20, 'TIME': 3, 'TP': 15} |
| cluster_end | 47 | -48.0 | -25.0 | -1173.0 | 17.0% | {'BE': 7, 'SL': 31, 'TIME': 2, 'TP': 7} |

## Read

- This is an execution-realism diagnostic, not a new rule.
- `event_ts_ms` is the feature-factory first timestamp. The threshold-cross lag table estimates how much of the cascade has already elapsed before the 500K condition is knowable.
- If the real-fill offset curve decays sharply from 0s to 0.5/1/2s, the apparent edge is highly latency-sensitive.
- If no-fill counterfactuals outperform filled rows, the real-fill subset is likely adversely biased by missed fast winners.