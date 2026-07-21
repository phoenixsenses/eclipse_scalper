# S34 Liquidation Outcome Calculator

- generated_at_utc: `2026-06-26T10:01:23+00:00`
- scope: `Current feature factory contains ETHUSDT BUY events only. Other symbols/sides require feature DB expansion.`
- selection_mode: `knn`
- candidate_events: `450`
- matched_events: `50`
- confidence: `usable`
- filters: `symbol=ETHUSDT; side=BUY; cluster_notional>=200000`

## Forward Return Distribution

| Horizon | N | Mean | Median | P25 | P75 | Positive Rate |
|---|---:|---:|---:|---:|---:|---:|
| 60s | 50 | +12.18 | +8.87 | +1.21 | +16.72 | 84.0% |
| 300s | 50 | +30.09 | +25.65 | +5.63 | +43.38 | 90.0% |
| 900s | 50 | +23.94 | +20.45 | -5.80 | +36.26 | 68.0% |
| 3600s | 50 | +32.44 | +15.77 | -15.26 | +41.31 | 64.0% |

## Route Simulation

| Route | N | Median Net | Mean Net | Cum Net | Top3 Removed | WR | TP/BE/SL/TIME | MFE Median | MAE Median |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|
| `LONG_DELAY0_TP60` | 50 | +21.34 | +18.90 | +945.24 | +767.05 | 56.0% | 22/16/4/8 | +54.17 | -2.69 |
| `LONG_DELAY60_TP120` | 50 | -8.51 | +3.16 | +157.99 | -186.73 | 34.0% | 4/20/9/17 | +51.46 | -8.26 |
| `SHORT_DELAY0_TP40_CONTROL` | 50 | -48.37 | -36.83 | -1841.26 | -1945.86 | 14.0% | 5/0/38/7 | +3.39 | -40.37 |

## Similarity

- candidate_n: `450`
- selected_n: `50`
- k: `50`

| Feature | Target | Weight | Scale |
|---|---:|---:|---:|
| `log_cluster_notional` | +750000.00 | 2.00 | +0.84 |
| `day_trend_bps` | +100.00 | 1.40 | +315.68 |
| `day_range_bps` | +300.00 | 1.00 | +321.61 |
| `symbol_pre_15m_bps` | +0.00 | 1.00 | +34.45 |
| `btc_pre_15m_bps` | +0.00 | 0.80 | +30.67 |

## Nearest Analogs

| Event | UTC | Notional | Day Trend | Day Range | Symbol Pre15 | Distance |
|---|---|---:|---:|---:|---:|---:|
| `ETHUSDT_BUY_5936030` | 2026-06-07T05:13:48.082000+00:00 | 796075 | +185.36 | +232.35 | +9.19 | 0.1924 |
| `ETHUSDT_BUY_5916856` | 2026-04-01T15:20:34.208000+00:00 | 570259 | +110.57 | +353.35 | +1.11 | 0.2014 |
| `ETHUSDT_BUY_5920938` | 2026-04-15T19:30:28.476000+00:00 | 840432 | +186.78 | +256.65 | +2.54 | 0.2079 |
| `ETHUSDT_BUY_5913313` | 2026-03-20T08:06:06.451000+00:00 | 639000 | +52.98 | +182.32 | -8.02 | 0.2239 |
| `ETHUSDT_BUY_5916816` | 2026-04-01T12:00:41.788000+00:00 | 589483 | +139.03 | +353.35 | -7.07 | 0.2273 |
| `ETHUSDT_BUY_5922162` | 2026-04-20T01:30:27.290000+00:00 | 633075 | +88.86 | +105.69 | -2.37 | 0.2711 |
| `ETHUSDT_BUY_5924043` | 2026-04-26T14:16:48.199000+00:00 | 572496 | +65.85 | +137.68 | +5.62 | 0.2859 |
| `ETHUSDT_BUY_5920932` | 2026-04-15T19:02:55.656000+00:00 | 925538 | +160.72 | +218.44 | +14.90 | 0.3010 |
| `ETHUSDT_BUY_5908952` | 2026-03-05T04:41:06.829000+00:00 | 992573 | +30.99 | +156.36 | +9.24 | 0.3014 |
| `ETHUSDT_BUY_5922673` | 2026-04-21T20:05:24.212000+00:00 | 776416 | -49.44 | +227.86 | +15.94 | 0.3258 |

## Read

This is a conditional historical distribution, not a price forecast. It is paper/research only.
If confidence is `thin` or `too_thin`, treat the output as a hypothesis prompt, not evidence.
