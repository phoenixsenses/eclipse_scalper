# S34 Cross-Symbol Lag - BTC Liquidation To ETH/SOL

Generated: 2026-06-17T09:19:26.243132+00:00

Scope: BTCUSDT BUY/SELL liquidation clusters >=200K, 300s bucket, 900s minimum gap.

This is descriptive research only. `same_direction_bps` means BUY-liq -> follower long return, SELL-liq -> follower short return.

## BTC Leader Cluster Coverage

| BTC side | Events | First | Last | Median Notional | Median Duration sec |
|---|---:|---|---|---:|---:|
| BUY | 544 | 2026-02-15T14:40:48.834000+00:00 | 2026-06-17T02:22:42.383000+00:00 | 333,769 | 139.4 |
| SELL | 507 | 2026-02-15T16:24:19.405000+00:00 | 2026-06-17T08:51:38.602000+00:00 | 331,190 | 157.6 |

## Same-Direction Follower Return Matrix

| BTC side | Follower | Delay | Horizon | N | Mean | Median | Positive Rate | p25/p75 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| BUY | ETHUSDT | 0s | 60s | 544 | +10.62 | +7.21 | 73.2% | -0.72/+17.24 |
| BUY | ETHUSDT | 0s | 300s | 544 | +20.51 | +12.18 | 73.2% | -0.84/+31.55 |
| BUY | ETHUSDT | 0s | 900s | 544 | +22.71 | +13.04 | 65.8% | -8.02/+38.59 |
| BUY | ETHUSDT | 60s | 60s | 544 | +3.74 | +2.19 | 58.6% | -4.60/+10.31 |
| BUY | ETHUSDT | 60s | 300s | 544 | +10.23 | +4.51 | 57.9% | -8.68/+21.53 |
| BUY | ETHUSDT | 60s | 900s | 544 | +12.10 | +4.55 | 56.1% | -17.07/+29.69 |
| BUY | ETHUSDT | 300s | 60s | 544 | +0.36 | -0.32 | 47.6% | -5.68/+5.74 |
| BUY | ETHUSDT | 300s | 300s | 544 | +1.03 | -0.16 | 49.1% | -13.10/+11.73 |
| BUY | ETHUSDT | 300s | 900s | 544 | +0.50 | -4.02 | 44.5% | -23.25/+20.10 |
| BUY | ETHUSDT | 900s | 60s | 544 | +0.03 | -0.95 | 45.0% | -5.84/+4.68 |
| BUY | ETHUSDT | 900s | 300s | 544 | -1.70 | -2.29 | 43.4% | -13.99/+9.29 |
| BUY | ETHUSDT | 900s | 900s | 544 | -1.02 | -1.27 | 47.8% | -20.04/+16.78 |
| BUY | SOLUSDT | 0s | 60s | 544 | +2.98 | +0.00 | 24.6% | +0.00/+0.00 |
| BUY | SOLUSDT | 0s | 300s | 544 | +5.27 | +0.00 | 23.5% | +0.00/+0.00 |
| BUY | SOLUSDT | 0s | 900s | 544 | +5.39 | +0.00 | 20.6% | +0.00/+0.00 |
| BUY | SOLUSDT | 60s | 60s | 544 | +1.12 | +0.00 | 18.4% | +0.00/+0.00 |
| BUY | SOLUSDT | 60s | 300s | 544 | +2.24 | +0.00 | 17.6% | +0.00/+0.00 |
| BUY | SOLUSDT | 60s | 900s | 544 | +2.18 | +0.00 | 17.5% | +0.00/+0.00 |
| BUY | SOLUSDT | 300s | 60s | 544 | -0.04 | +0.00 | 16.4% | +0.00/+0.00 |
| BUY | SOLUSDT | 300s | 300s | 544 | -0.87 | +0.00 | 14.2% | +0.00/+0.00 |
| BUY | SOLUSDT | 300s | 900s | 544 | -0.32 | +0.00 | 14.9% | +0.00/+0.00 |
| BUY | SOLUSDT | 900s | 60s | 544 | -0.23 | +0.00 | 15.4% | +0.00/+0.00 |
| BUY | SOLUSDT | 900s | 300s | 544 | -0.46 | +0.00 | 14.0% | +0.00/+0.00 |
| BUY | SOLUSDT | 900s | 900s | 544 | -0.18 | +0.00 | 15.4% | +0.00/+0.00 |
| SELL | ETHUSDT | 0s | 60s | 507 | +9.08 | +6.15 | 71.4% | -1.54/+15.63 |
| SELL | ETHUSDT | 0s | 300s | 507 | +19.84 | +15.09 | 73.0% | -1.52/+35.58 |
| SELL | ETHUSDT | 0s | 900s | 507 | +19.85 | +12.47 | 63.3% | -8.81/+43.25 |
| SELL | ETHUSDT | 60s | 60s | 507 | +5.76 | +3.95 | 66.5% | -2.52/+12.69 |
| SELL | ETHUSDT | 60s | 300s | 507 | +10.55 | +7.00 | 61.1% | -10.50/+27.02 |
| SELL | ETHUSDT | 60s | 900s | 507 | +9.06 | +4.64 | 53.5% | -17.37/+30.84 |
| SELL | ETHUSDT | 300s | 60s | 507 | -0.22 | -0.81 | 46.9% | -7.35/+6.68 |
| SELL | ETHUSDT | 300s | 300s | 507 | +0.23 | -2.70 | 44.2% | -13.78/+11.35 |
| SELL | ETHUSDT | 300s | 900s | 507 | -0.21 | -3.54 | 45.6% | -22.56/+18.32 |
| SELL | ETHUSDT | 900s | 60s | 507 | -1.72 | -1.10 | 43.8% | -6.77/+4.16 |
| SELL | ETHUSDT | 900s | 300s | 507 | -0.23 | -0.56 | 48.7% | -11.59/+10.58 |
| SELL | ETHUSDT | 900s | 900s | 506 | -1.42 | -4.11 | 44.7% | -19.78/+14.67 |
| SELL | SOLUSDT | 0s | 60s | 507 | +1.72 | -0.00 | 19.7% | -0.00/-0.00 |
| SELL | SOLUSDT | 0s | 300s | 507 | +4.04 | -0.00 | 20.7% | -0.00/-0.00 |
| SELL | SOLUSDT | 0s | 900s | 507 | +4.37 | -0.00 | 18.9% | -0.00/-0.00 |
| SELL | SOLUSDT | 60s | 60s | 507 | +1.55 | -0.00 | 18.9% | -0.00/-0.00 |
| SELL | SOLUSDT | 60s | 300s | 507 | +2.16 | -0.00 | 16.8% | -0.00/-0.00 |
| SELL | SOLUSDT | 60s | 900s | 507 | +2.78 | -0.00 | 16.6% | -0.00/-0.00 |
| SELL | SOLUSDT | 300s | 60s | 507 | -0.16 | -0.00 | 13.0% | -0.00/-0.00 |
| SELL | SOLUSDT | 300s | 300s | 507 | +0.21 | -0.00 | 14.8% | -0.00/-0.00 |
| SELL | SOLUSDT | 300s | 900s | 507 | +0.46 | -0.00 | 13.8% | -0.00/-0.00 |
| SELL | SOLUSDT | 900s | 60s | 507 | +0.14 | -0.00 | 15.2% | -0.00/-0.00 |
| SELL | SOLUSDT | 900s | 300s | 507 | +0.13 | -0.00 | 14.2% | -0.00/-0.00 |
| SELL | SOLUSDT | 900s | 900s | 506 | -0.58 | -0.00 | 11.5% | -0.00/-0.00 |

## Nearby Follower Liquidation Synchrony

| BTC side | Follower | Window | Nearby | Before | Same +/-5s | After | Median Lag sec | Side Counts |
|---|---|---:|---:|---:|---:|---:|---:|---|
| BUY | ETHUSDT | 60s | 507/544 | 26 | 335 | 146 | +0.49 | {'BUY': 471, 'SELL': 36} |
| BUY | ETHUSDT | 300s | 540/544 | 34 | 335 | 171 | +0.59 | {'BUY': 494, 'SELL': 46} |
| BUY | ETHUSDT | 900s | 543/544 | 36 | 335 | 172 | +0.58 | {'BUY': 497, 'SELL': 46} |
| BUY | SOLUSDT | 60s | 124/544 | 8 | 63 | 53 | +2.49 | {'BUY': 110, 'SELL': 14} |
| BUY | SOLUSDT | 300s | 161/544 | 19 | 63 | 79 | +4.51 | {'BUY': 134, 'SELL': 27} |
| BUY | SOLUSDT | 900s | 177/544 | 25 | 63 | 89 | +5.02 | {'BUY': 140, 'SELL': 37} |
| SELL | ETHUSDT | 60s | 469/507 | 34 | 312 | 123 | +0.36 | {'SELL': 434, 'BUY': 35} |
| SELL | ETHUSDT | 300s | 505/507 | 46 | 312 | 147 | +0.40 | {'SELL': 464, 'BUY': 41} |
| SELL | ETHUSDT | 900s | 507/507 | 46 | 312 | 149 | +0.41 | {'SELL': 466, 'BUY': 41} |
| SELL | SOLUSDT | 60s | 108/507 | 12 | 55 | 41 | +1.42 | {'SELL': 103, 'BUY': 5} |
| SELL | SOLUSDT | 300s | 141/507 | 20 | 55 | 66 | +2.93 | {'SELL': 132, 'BUY': 9} |
| SELL | SOLUSDT | 900s | 147/507 | 22 | 55 | 70 | +3.82 | {'SELL': 136, 'BUY': 11} |

## Observations

- The table separates raw delayed follower returns from same-direction forced-flow returns.
- This report does not add a cross-symbol rule, choose a delay, or modify S34.
- Apparent BTC->ETH/SOL timing relations are hypothesis material only.
