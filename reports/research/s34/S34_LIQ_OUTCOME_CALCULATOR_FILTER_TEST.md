# S34 Liquidation Outcome Calculator

- generated_at_utc: `2026-06-26T10:01:23+00:00`
- scope: `Current feature factory contains ETHUSDT BUY events only. Other symbols/sides require feature DB expansion.`
- selection_mode: `filter`
- candidate_events: `97`
- matched_events: `97`
- confidence: `usable`
- filters: `symbol=ETHUSDT; side=BUY; cluster_notional>=500000; day_trend_bps>=0`

## Forward Return Distribution

| Horizon | N | Mean | Median | P25 | P75 | Positive Rate |
|---|---:|---:|---:|---:|---:|---:|
| 60s | 97 | +14.27 | +9.07 | +2.38 | +20.63 | 83.5% |
| 300s | 97 | +43.60 | +30.24 | +6.68 | +54.68 | 85.6% |
| 900s | 97 | +39.58 | +26.74 | -3.83 | +55.41 | 70.1% |
| 3600s | 97 | +39.04 | +22.01 | -13.62 | +59.02 | 63.9% |

## Route Simulation

| Route | N | Median Net | Mean Net | Cum Net | Top3 Removed | WR | TP/BE/SL/TIME | MFE Median | MAE Median |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|
| `LONG_DELAY0_TP60` | 97 | +52.21 | +24.02 | +2329.84 | +2140.04 | 60.8% | 53/27/9/8 | +60.21 | -2.70 |
| `LONG_DELAY60_TP120` | 97 | -8.34 | +13.70 | +1328.68 | +950.59 | 35.1% | 20/36/21/20 | +54.81 | -6.92 |
| `SHORT_DELAY0_TP40_CONTROL` | 97 | -49.06 | -33.90 | -3288.35 | -3392.21 | 17.5% | 15/5/71/6 | +3.79 | -41.06 |

## Similarity

Filter mode. No weighted KNN similarity was applied.

## Nearest Analogs

| Event | UTC | Notional | Day Trend | Day Range | Symbol Pre15 | Distance |
|---|---|---:|---:|---:|---:|---:|
| `ETHUSDT_BUY_5921387` | 2026-04-17T08:55:09.083000+00:00 | 500550 | +44.30 | +176.20 | +39.35 | 0.0022 |
| `ETHUSDT_BUY_5908793` | 2026-03-04T15:25:25.145000+00:00 | 509495 | +610.99 | +847.85 | +53.92 | 0.0376 |
| `ETHUSDT_BUY_5921454` | 2026-04-17T14:31:04.263000+00:00 | 515805 | +380.41 | +561.77 | +16.14 | 0.0622 |
| `ETHUSDT_BUY_5912023` | 2026-03-15T20:37:08.102000+00:00 | 517499 | +153.38 | +217.09 | +78.46 | 0.0688 |
| `ETHUSDT_BUY_5912281` | 2026-03-16T18:06:30.087000+00:00 | 535855 | +647.00 | +722.39 | +14.26 | 0.1385 |
| `ETHUSDT_BUY_5916549` | 2026-03-31T13:45:26.203000+00:00 | 536960 | +200.97 | +392.28 | +34.01 | 0.1426 |
| `ETHUSDT_BUY_5918845` | 2026-04-08T13:06:31.066000+00:00 | 537374 | +86.38 | +152.69 | +18.58 | 0.1442 |
| `ETHUSDT_BUY_5911437` | 2026-03-13T19:45:01.731000+00:00 | 555482 | +146.62 | +659.59 | +21.02 | 0.2105 |
| `ETHUSDT_BUY_5910114` | 2026-03-09T05:30:52.436000+00:00 | 556991 | +266.92 | +386.58 | +40.36 | 0.2159 |
| `ETHUSDT_BUY_5906606` | 2026-02-25T01:10:24.695000+00:00 | 559662 | +70.61 | +101.26 | +64.41 | 0.2255 |

## Read

This is a conditional historical distribution, not a price forecast. It is paper/research only.
If confidence is `thin` or `too_thin`, treat the output as a hypothesis prompt, not evidence.
