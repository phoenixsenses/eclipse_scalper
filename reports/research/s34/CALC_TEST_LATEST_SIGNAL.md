# S34 Liquidation Outcome Calculator

- generated_at_utc: `2026-06-26T18:55:33+00:00`
- preset: `-`
- source_signal: `ETH_BUY_LIQ_LONG_200K_TP60_SL40_BE30:5941632`
- scope: `BTCUSDT BUY N=127  |  BTCUSDT SELL N=113  |  ETHUSDT BUY N=450  |  ETHUSDT SELL N=222  |  SOLUSDT BUY N=104  |  SOLUSDT SELL N=105`
- selection_mode: `knn`
- decision_card: `AVOID`
- model_tag: `REGIME_SHIFT_RECENCY_HELPFUL_PRELIMINARY`
- candidate_events: `450`
- matched_events: `50`
- confidence: `usable`
- filters: `symbol=ETHUSDT; side=BUY; cluster_notional>=200000`

## Forward Return Distribution

| Horizon | N | Mean | Median | P25 | P75 | Positive Rate |
|---|---:|---:|---:|---:|---:|---:|
| 60s | 50 | +12.28 | +6.90 | +0.89 | +17.95 | 76.0% |
| 300s | 50 | +15.82 | +6.62 | -2.27 | +28.12 | 68.0% |
| 900s | 50 | +17.30 | +4.40 | -9.74 | +26.79 | 54.0% |
| 3600s | 50 | +10.69 | +11.95 | -47.71 | +47.23 | 54.0% |

## Route Simulation

| Route | N | Median Net | Mean Net | Cum Net | Top3 Removed | WR | TP/BE/SL/TIME | MFE Median | MAE Median |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|
| `LONG_DELAY0_TP60` | 50 | -8.22 | +4.15 | +207.38 | +35.74 | 42.0% | 17/15/12/6 | +38.78 | -9.98 |
| `LONG_DELAY60_TP120` | 50 | -9.96 | -6.11 | -305.52 | -657.08 | 22.0% | 4/16/20/10 | +35.74 | -27.08 |
| `SHORT_DELAY0_TP40_CONTROL` | 50 | -35.89 | -19.76 | -988.11 | -1091.61 | 28.0% | 13/9/25/3 | +20.59 | -39.54 |

## Similarity

- candidate_n: `450`
- selected_n: `50`
- k: `50`

| Feature | Target | Weight | Scale |
|---|---:|---:|---:|
| `log_cluster_notional` | +637472.76 | 2.00 | +0.84 |
| `cluster_duration_sec` | +35.50 | 0.80 | +142.59 |
| `cluster_liq_count` | +16.00 | 0.80 | +13.00 |
| `max_single_liq_share` | +68.88 | 0.80 | +44.01 |
| `intensity_per_sec` | +17956.47 | 1.00 | +5689.63 |
| `inter_cluster_gap_sec` | +143.35 | 0.70 | +14954.78 |
| `day_trend_bps` | +92.46 | 1.40 | +315.68 |
| `day_range_bps` | +483.42 | 1.00 | +321.61 |

## Nearest Analogs

| Event | UTC | Notional | Day Trend | Day Range | Symbol Pre15 | Distance |
|---|---|---:|---:|---:|---:|---:|
| `ETHUSDT_BUY_5920950` | 2026-04-15T20:30:32.088000+00:00 | 474830 | +172.28 | +329.12 | +4.62 | 0.3954 |
| `ETHUSDT_BUY_5921387` | 2026-04-17T08:55:09.083000+00:00 | 500550 | +44.30 | +176.20 | +39.35 | 0.4839 |
| `ETHUSDT_BUY_5920338` | 2026-04-13T17:30:04.142000+00:00 | 452419 | +191.86 | +277.86 | +36.50 | 0.4960 |
| `ETHUSDT_BUY_5920932` | 2026-04-15T19:02:55.656000+00:00 | 925538 | +160.72 | +218.44 | +14.90 | 0.4995 |
| `ETHUSDT_BUY_5920367` | 2026-04-13T19:55:07.734000+00:00 | 1263312 | +321.91 | +412.72 | +27.76 | 0.5334 |
| `ETHUSDT_BUY_5922511` | 2026-04-21T06:35:00.356000+00:00 | 582892 | +27.55 | +114.98 | +51.21 | 0.5576 |
| `ETHUSDT_BUY_5914811` | 2026-03-25T12:55:40.202000+00:00 | 476389 | +97.53 | +241.00 | +18.13 | 0.6036 |
| `ETHUSDT_BUY_5906702` | 2026-02-25T09:10:14.886000+00:00 | 1031931 | +334.03 | +523.64 | +5.87 | 0.6106 |
| `ETHUSDT_BUY_5922148` | 2026-04-20T00:23:50.235000+00:00 | 1071244 | +58.89 | +68.61 | +25.46 | 0.6160 |
| `ETHUSDT_BUY_5937329` | 2026-06-11T17:28:55.895000+00:00 | 751665 | +137.28 | +318.09 | +49.08 | 0.6375 |

## Read

This is a conditional historical distribution, not a price forecast. It is paper/research only.
If confidence is `thin` or `too_thin`, treat the output as a hypothesis prompt, not evidence.
