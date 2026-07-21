# S34 Liquidation Outcome Calculator

- generated_at_utc: `2026-06-26T10:09:48+00:00`
- scope: `Current feature factory contains ETHUSDT BUY events only. Other symbols/sides require feature DB expansion.`
- selection_mode: `knn`
- candidate_events: `450`
- matched_events: `50`
- confidence: `usable`
- filters: `symbol=ETHUSDT; side=BUY; cluster_notional>=200000`

## Forward Return Distribution

| Horizon | N | Mean | Median | P25 | P75 | Positive Rate |
|---|---:|---:|---:|---:|---:|---:|
| 60s | 50 | +12.61 | +9.30 | +1.21 | +17.33 | 84.0% |
| 300s | 50 | +30.99 | +26.22 | +5.63 | +44.43 | 90.0% |
| 900s | 50 | +26.46 | +21.07 | -3.44 | +37.18 | 70.0% |
| 3600s | 50 | +32.25 | +15.77 | -15.26 | +41.31 | 64.0% |

## Route Simulation

| Route | N | Median Net | Mean Net | Cum Net | Top3 Removed | WR | TP/BE/SL/TIME | MFE Median | MAE Median |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|
| `LONG_DELAY0_TP60` | 50 | +26.07 | +20.12 | +1005.85 | +827.66 | 58.0% | 23/15/4/8 | +54.72 | -2.43 |
| `LONG_DELAY60_TP120` | 50 | -8.48 | +3.17 | +158.47 | -186.26 | 34.0% | 4/20/9/17 | +52.09 | -9.64 |
| `SHORT_DELAY0_TP40_CONTROL` | 50 | -48.37 | -36.64 | -1831.87 | -1936.47 | 14.0% | 5/0/38/7 | +3.15 | -40.37 |

## Similarity

- candidate_n: `450`
- selected_n: `50`
- k: `50`
- auto_weights_applied: `true`

| Feature | Target | Weight | Scale |
|---|---:|---:|---:|
| `log_cluster_notional` | +750000.00 | 3.00 | +0.84 |
| `day_trend_bps` | +100.00 | 1.39 | +315.68 |
| `day_range_bps` | +300.00 | 0.71 | +321.61 |
| `symbol_pre_15m_bps` | +0.00 | 1.34 | +34.45 |
| `btc_pre_15m_bps` | +0.00 | 0.52 | +30.67 |

## Suggested KNN Weights

- method: `leave_one_feature_out impact normalized to 0.65x-1.75x base weight`
- applied: `true`

| Feature | Base | Suggested | Impact |
|---|---:|---:|---:|
| `log_cluster_notional` | 2.00 | 3.00 | 19.56 |
| `day_trend_bps` | 1.40 | 1.39 | 9.41 |
| `symbol_pre_15m_bps` | 1.00 | 1.34 | 14.12 |
| `day_range_bps` | 1.00 | 0.71 | 5.65 |
| `btc_pre_15m_bps` | 0.80 | 0.52 | 4.87 |

## KNN Feature Importance

Leave-one-feature-out impact on the selected neighbor set and primary route.

| Dropped Feature | Median Delta | Cum Delta | Neighbor Overlap | Drop Median |
|---|---:|---:|---:|---:|
| `symbol_pre_15m_bps` | -10.66 | -31.78 | 78.0% | +32.00 |
| `log_cluster_notional` | +10.39 | +275.09 | 48.0% | +10.95 |
| `day_trend_bps` | -7.26 | +10.26 | 86.0% | +28.60 |
| `day_range_bps` | -4.74 | -62.19 | 96.0% | +26.07 |
| `btc_pre_15m_bps` | -3.83 | -26.96 | 94.0% | +25.17 |

## Nearest Analogs

| Event | UTC | Notional | Day Trend | Day Range | Symbol Pre15 | Distance |
|---|---|---:|---:|---:|---:|---:|
| `ETHUSDT_BUY_5936030` | 2026-06-07T05:13:48.082000+00:00 | 796075 | +185.36 | +232.35 | +9.19 | 0.1875 |
| `ETHUSDT_BUY_5920938` | 2026-04-15T19:30:28.476000+00:00 | 840432 | +186.78 | +256.65 | +2.54 | 0.1882 |
| `ETHUSDT_BUY_5913313` | 2026-03-20T08:06:06.451000+00:00 | 639000 | +52.98 | +182.32 | -8.02 | 0.2145 |
| `ETHUSDT_BUY_5916856` | 2026-04-01T15:20:34.208000+00:00 | 570259 | +110.57 | +353.35 | +1.11 | 0.2237 |
| `ETHUSDT_BUY_5916816` | 2026-04-01T12:00:41.788000+00:00 | 589483 | +139.03 | +353.35 | -7.07 | 0.2358 |
| `ETHUSDT_BUY_5922162` | 2026-04-20T01:30:27.290000+00:00 | 633075 | +88.86 | +105.69 | -2.37 | 0.2371 |
| `ETHUSDT_BUY_5924043` | 2026-04-26T14:16:48.199000+00:00 | 572496 | +65.85 | +137.68 | +5.62 | 0.2799 |
| `ETHUSDT_BUY_5920932` | 2026-04-15T19:02:55.656000+00:00 | 925538 | +160.72 | +218.44 | +14.90 | 0.2993 |
| `ETHUSDT_BUY_5908952` | 2026-03-05T04:41:06.829000+00:00 | 992573 | +30.99 | +156.36 | +9.24 | 0.3035 |
| `ETHUSDT_BUY_5922673` | 2026-04-21T20:05:24.212000+00:00 | 776416 | -49.44 | +227.86 | +15.94 | 0.3146 |

## Read

This is a conditional historical distribution, not a price forecast. It is paper/research only.
If confidence is `thin` or `too_thin`, treat the output as a hypothesis prompt, not evidence.
