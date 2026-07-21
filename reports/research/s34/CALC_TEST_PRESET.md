# S34 Liquidation Outcome Calculator

- generated_at_utc: `2026-06-26T18:55:33+00:00`
- preset: `eth_sell_500k`
- source_signal: `-`
- scope: `BTCUSDT BUY N=127  |  BTCUSDT SELL N=113  |  ETHUSDT BUY N=450  |  ETHUSDT SELL N=222  |  SOLUSDT BUY N=104  |  SOLUSDT SELL N=105`
- selection_mode: `filter`
- decision_card: `BASE_RATE_ONLY`
- model_tag: `KNN_USEFUL`
- candidate_events: `222`
- matched_events: `222`
- confidence: `broad`
- filters: `symbol=ETHUSDT; side=SELL; cluster_notional>=500000`

## Forward Return Distribution

| Horizon | N | Mean | Median | P25 | P75 | Positive Rate |
|---|---:|---:|---:|---:|---:|---:|
| 60s | 222 | -12.57 | -10.38 | -19.16 | -2.31 | 21.2% |
| 300s | 222 | -30.53 | -25.14 | -48.85 | -7.56 | 15.8% |
| 900s | 222 | -33.50 | -27.20 | -51.21 | +0.93 | 25.7% |
| 3600s | 222 | -27.81 | -16.43 | -62.09 | +23.69 | 39.2% |

## Route Simulation

| Route | N | Median Net | Mean Net | Cum Net | Top3 Removed | WR | TP/BE/SL/TIME | MFE Median | MAE Median |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|
| `LONG_DELAY0_TP40_CONTROL` | 222 | -48.72 | -31.99 | -7102.17 | -7220.20 | 18.9% | 37/10/164/11 | +4.51 | -40.72 |
| `SHORT_DELAY0_TP60` | 222 | +52.17 | +19.09 | +4237.14 | +4046.45 | 59.9% | 123/39/40/20 | +60.17 | -4.74 |
| `SHORT_DELAY0_TP80` | 222 | -8.11 | +17.92 | +3978.70 | +3738.19 | 47.7% | 85/66/40/31 | +64.64 | -4.74 |

## Similarity

Filter mode. No weighted KNN similarity was applied.

## Nearest Analogs

| Event | UTC | Notional | Day Trend | Day Range | Symbol Pre15 | Distance |
|---|---|---:|---:|---:|---:|---:|
| `ETHUSDT_SELL_5937597` | 2026-06-12T15:47:00.224000+00:00 | 500751 | -24.93 | +236.21 | -51.38 | 0.0030 |
| `ETHUSDT_SELL_5916471` | 2026-03-31T07:15:10.446000+00:00 | 502628 | +116.76 | +372.26 | -61.56 | 0.0105 |
| `ETHUSDT_SELL_5941591` | 2026-06-26T12:35:14.208000+00:00 | 506083 | -86.01 | +488.61 | +36.13 | 0.0242 |
| `ETHUSDT_SELL_5938863` | 2026-06-17T01:15:56.382000+00:00 | 515085 | -8.57 | +41.88 | -8.52 | 0.0594 |
| `ETHUSDT_SELL_5939462` | 2026-06-19T03:12:15.398000+00:00 | 517610 | -47.19 | +106.56 | -27.48 | 0.0692 |
| `ETHUSDT_SELL_5936895` | 2026-06-10T05:15:16.246000+00:00 | 518143 | -108.55 | +156.17 | -30.34 | 0.0713 |
| `ETHUSDT_SELL_5921086` | 2026-04-16T07:51:34.710000+00:00 | 519179 | -58.54 | +100.83 | -25.92 | 0.0753 |
| `ETHUSDT_SELL_5914221` | 2026-03-23T11:45:43.243000+00:00 | 519389 | +486.07 | +866.47 | -91.80 | 0.0761 |
| `ETHUSDT_SELL_5939179` | 2026-06-18T03:36:49.516000+00:00 | 526618 | -35.00 | +105.87 | -36.91 | 0.1037 |
| `ETHUSDT_SELL_5910094` | 2026-03-09T03:53:57.301000+00:00 | 526909 | +193.66 | +386.58 | -27.28 | 0.1048 |

## Read

This is a conditional historical distribution, not a price forecast. It is paper/research only.
If confidence is `thin` or `too_thin`, treat the output as a hypothesis prompt, not evidence.
