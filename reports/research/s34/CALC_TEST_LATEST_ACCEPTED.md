# S34 Liquidation Outcome Calculator

- generated_at_utc: `2026-06-26T19:00:43+00:00`
- preset: `-`
- source_signal: `ETH_BUY_LIQ_LONG_500K_DAYTREND0_TP60_SL40_BE30:5941632`
- source_trade: `P680`
- scope: `BTCUSDT BUY N=127  |  BTCUSDT SELL N=113  |  ETHUSDT BUY N=450  |  ETHUSDT SELL N=222  |  SOLUSDT BUY N=104  |  SOLUSDT SELL N=105`
- selection_mode: `knn`
- decision_card: `AVOID`
- model_tag: `REGIME_SHIFT_RECENCY_HELPFUL_PRELIMINARY`
- candidate_events: `140`
- matched_events: `50`
- confidence: `usable`
- filters: `symbol=ETHUSDT; side=BUY; cluster_notional>=500000`

## Forward Return Distribution

| Horizon | N | Mean | Median | P25 | P75 | Positive Rate |
|---|---:|---:|---:|---:|---:|---:|
| 60s | 50 | +10.40 | +7.01 | +1.00 | +18.14 | 78.0% |
| 300s | 50 | +22.47 | +13.00 | +3.39 | +33.18 | 80.0% |
| 900s | 50 | +20.72 | +8.22 | -7.26 | +36.69 | 60.0% |
| 3600s | 50 | +16.73 | +15.90 | -10.92 | +38.92 | 64.0% |

## Route Simulation

| Route | N | Median Net | Mean Net | Cum Net | Top3 Removed | WR | TP/BE/SL/TIME | MFE Median | MAE Median |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|

## Decision Card

- verdict: `AVOID`
- model_tag: `REGIME_SHIFT_RECENCY_HELPFUL_PRELIMINARY`
- recommended_route: `LONG_DELAY0_TP60`
- reasons: `best_route_wr=46%; source_signal_targets_loaded`
- warnings: `best_route_median_not_positive; high_be_time_rate=50%`
| `LONG_DELAY0_TP60` | 50 | -8.05 | +8.93 | +446.54 | +275.61 | 46.0% | 17/17/8/8 | +43.70 | -5.06 |
| `LONG_DELAY60_TP120` | 50 | -8.62 | -3.25 | -162.29 | -509.81 | 24.0% | 3/20/14/13 | +43.06 | -14.22 |
| `SHORT_DELAY0_TP40_CONTROL` | 50 | -48.32 | -27.61 | -1380.65 | -1480.65 | 22.0% | 10/5/31/4 | +8.41 | -40.32 |

## Similarity

- candidate_n: `140`
- selected_n: `50`
- k: `50`

| Feature | Target | Weight | Scale |
|---|---:|---:|---:|
| `log_cluster_notional` | +637472.76 | 2.00 | +1.05 |
| `cluster_duration_sec` | +35.50 | 0.80 | +135.16 |
| `cluster_liq_count` | +16.00 | 0.80 | +15.50 |
| `max_single_liq_share` | +68.88 | 0.80 | +43.79 |
| `intensity_per_sec` | +17956.47 | 1.00 | +16216.15 |
| `inter_cluster_gap_sec` | +143.35 | 0.70 | +10377.04 |
| `day_trend_bps` | +92.46 | 1.40 | +286.80 |
| `day_range_bps` | +483.42 | 1.00 | +300.32 |

## Nearest Analogs

| Event | UTC | Notional | Day Trend | Day Range | Symbol Pre15 | Distance |
|---|---|---:|---:|---:|---:|---:|
| `ETHUSDT_BUY_5921387` | 2026-04-17T08:55:09.083000+00:00 | 500550 | +44.30 | +176.20 | +39.35 | 0.4328 |
| `ETHUSDT_BUY_5911099` | 2026-03-12T15:37:41.084000+00:00 | 959738 | +91.51 | +385.18 | +97.31 | 0.4437 |
| `ETHUSDT_BUY_5904481` | 2026-02-17T16:05:11.630000+00:00 | 544503 | -46.88 | +342.64 | +168.04 | 0.4686 |
| `ETHUSDT_BUY_5920932` | 2026-04-15T19:02:55.656000+00:00 | 925538 | +160.72 | +218.44 | +14.90 | 0.4766 |
| `ETHUSDT_BUY_5920367` | 2026-04-13T19:55:07.734000+00:00 | 1263312 | +321.91 | +412.72 | +27.76 | 0.4832 |
| `ETHUSDT_BUY_5922511` | 2026-04-21T06:35:00.356000+00:00 | 582892 | +27.55 | +114.98 | +51.21 | 0.5386 |
| `ETHUSDT_BUY_5921201` | 2026-04-16T17:25:05.675000+00:00 | 504263 | -146.38 | +369.32 | +32.03 | 0.5525 |
| `ETHUSDT_BUY_5936778` | 2026-06-09T19:30:42.209000+00:00 | 678091 | -215.10 | +508.89 | +27.36 | 0.5577 |
| `ETHUSDT_BUY_5936395` | 2026-06-08T11:36:10.305000+00:00 | 1100461 | -3.26 | +405.15 | +15.76 | 0.5597 |
| `ETHUSDT_BUY_5907927` | 2026-03-01T15:15:00.210000+00:00 | 569979 | +225.48 | +566.26 | +35.01 | 0.5618 |

## Read

This is a conditional historical distribution, not a price forecast. It is paper/research only.
If confidence is `thin` or `too_thin`, treat the output as a hypothesis prompt, not evidence.
