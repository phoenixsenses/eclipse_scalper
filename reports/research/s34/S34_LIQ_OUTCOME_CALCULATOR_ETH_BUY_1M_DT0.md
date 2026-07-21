# S34 Liquidation Outcome Calculator

- generated_at_utc: `2026-06-26T09:54:43+00:00`
- scope: `Current feature factory contains ETHUSDT BUY events only. Other symbols/sides require feature DB expansion.`
- matched_events: `53`
- confidence: `usable`
- filters: `symbol=ETHUSDT; side=BUY; cluster_notional>=1e+06; day_trend_bps>=0`

## Forward Return Distribution

| Horizon | N | Mean | Median | P25 | P75 | Positive Rate |
|---|---:|---:|---:|---:|---:|---:|
| 60s | 53 | +14.29 | +10.12 | +3.07 | +20.31 | 84.9% |
| 300s | 53 | +45.70 | +37.79 | +5.67 | +59.19 | 88.7% |
| 900s | 53 | +40.97 | +32.93 | -6.25 | +58.34 | 66.0% |
| 3600s | 53 | +41.79 | +25.18 | -11.22 | +90.39 | 67.9% |

## Route Simulation

| Route | N | Median Net | Mean Net | Cum Net | Top3 Removed | WR | TP/BE/SL/TIME | MFE Median | MAE Median |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|
| `LONG_DELAY0_TP60` | 53 | +52.66 | +29.16 | +1545.43 | +1362.90 | 67.9% | 33/11/4/5 | +60.66 | -2.81 |
| `LONG_DELAY60_TP120` | 53 | -8.16 | +19.32 | +1023.87 | +655.75 | 39.6% | 13/16/11/13 | +59.20 | -7.62 |
| `SHORT_DELAY0_TP40_CONTROL` | 53 | -49.06 | -32.86 | -1741.48 | -1844.01 | 18.9% | 8/4/37/4 | +3.93 | -41.06 |

## Nearest Analogs

| Event | UTC | Notional | Day Trend | Day Range | Symbol Pre15 | Distance |
|---|---|---:|---:|---:|---:|---:|
| `ETHUSDT_BUY_5938407` | 2026-06-15T11:16:11.578000+00:00 | 1698577 | +94.98 | +208.69 | +26.77 | 0.2988 |
| `ETHUSDT_BUY_5924089` | 2026-04-26T18:06:51.105000+00:00 | 1508663 | +143.53 | +201.23 | +14.84 | 0.4468 |
| `ETHUSDT_BUY_5922537` | 2026-04-21T08:47:33.124000+00:00 | 1346660 | +69.27 | +129.42 | +31.32 | 0.5229 |
| `ETHUSDT_BUY_5920913` | 2026-04-15T17:26:04.216000+00:00 | 1922618 | +107.95 | +176.31 | +12.99 | 0.5759 |
| `ETHUSDT_BUY_5922750` | 2026-04-22T02:30:03.414000+00:00 | 1784092 | +126.85 | +187.26 | +89.18 | 0.6154 |
| `ETHUSDT_BUY_5919775` | 2026-04-11T18:35:05.078000+00:00 | 1314327 | +141.10 | +213.33 | +59.95 | 0.6752 |
| `ETHUSDT_BUY_5912106` | 2026-03-16T03:30:07.565000+00:00 | 1176125 | +119.99 | +206.19 | +108.47 | 0.6864 |
| `ETHUSDT_BUY_5936509` | 2026-06-08T21:07:21.635000+00:00 | 1874090 | +66.47 | +405.15 | +93.83 | 0.7807 |
| `ETHUSDT_BUY_5910792` | 2026-03-11T14:00:25.640000+00:00 | 2013637 | +74.49 | +374.34 | -73.04 | 0.8440 |
| `ETHUSDT_BUY_5920518` | 2026-04-14T08:30:05.600000+00:00 | 1269348 | +45.59 | +145.07 | +14.18 | 0.8780 |

## Read

This is a conditional historical distribution, not a price forecast. It is paper/research only.
If confidence is `thin` or `too_thin`, treat the output as a hypothesis prompt, not evidence.
