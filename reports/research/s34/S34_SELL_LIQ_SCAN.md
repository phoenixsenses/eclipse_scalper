# S34 SELL Liquidation → SHORT Scan

Generated: `2026-06-25T18:29:54.650257+00:00`  
Lookback: 120d | TP60/SL40/BE30 | real bookTicker fills

Hypothesis: SELL liq cluster (short squeeze) → exhaustion → SHORT entry.

## Full Grid

| Symbol | Threshold | Signals | Real | No Fill | No Fill% | Median | Mean | Cum | WR | Days | 2nd Median | 2nd Cum | Top3-Rmv |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ETHUSDT | 50K | 1148 | 415 | 733 | 64% | -8.24 | +1.19 | +492 | 41% | 32 | -8.24 | +492 | +230 |
| ETHUSDT | 100K | 800 | 324 | 476 | 60% | -7.28 | +6.22 | +2017 | 46% | 32 | -7.28 | +2017 | +1750 |
| ETHUSDT | 200K | 492 | 230 | 262 | 53% | -6.98 | +8.55 | +1966 | 47% | 31 | -6.78 | +1850 | +1699 |
| ETHUSDT | 500K | 203 | 117 | 86 | 42% | +47.50 | +20.57 | +2406 | 57% | 28 | +50.23 | +1873 | +2154 |
| ETHUSDT | 1000K | 107 | 76 | 31 | 29% | +52.03 | +29.05 | +2208 | 68% | 25 | +52.49 | +1724 | +1977 |
| BTCUSDT | 50K | 1202 | 425 | 777 | 65% | -8.46 | -1.71 | -727 | 40% | 32 | -8.46 | -727 | -937 |
| BTCUSDT | 100K | 846 | 311 | 535 | 63% | -6.83 | +2.91 | +905 | 44% | 32 | -6.83 | +905 | +698 |
| BTCUSDT | 200K | 530 | 212 | 318 | 60% | -6.79 | +6.19 | +1313 | 46% | 31 | -6.83 | +1308 | +1127 |
| BTCUSDT | 500K | 198 | 106 | 92 | 46% | -6.59 | +13.49 | +1430 | 48% | 27 | +17.98 | +1323 | +1218 |
| BTCUSDT | 1000K | 105 | 68 | 37 | 35% | +7.01 | +15.76 | +1072 | 51% | 23 | +34.58 | +1063 | +865 |
| SOLUSDT | 50K | 160 | 124 | 36 | 22% | -3.84 | +10.57 | +1311 | 48% | 23 | +15.90 | +1125 | +1008 |
| SOLUSDT | 100K | 102 | 81 | 21 | 21% | +26.92 | +14.99 | +1214 | 52% | 22 | +44.53 | +921 | +911 |
| SOLUSDT | 200K | 56 | 45 | 11 | 20% | +49.95 | +28.31 | +1274 | 64% | 17 | +50.90 | +904 | +971 |
| SOLUSDT | 500K | 23 | 21 | 2 | 9% | +51.51 | +48.37 | +1016 | 81% | 10 | +53.16 | +701 | +713 |
| SOLUSDT | 1000K | 11 | 11 | 0 | 0% | +54.25 | +59.46 | +654 | 91% | 9 | +58.21 | +434 | +357 |

## Candidate Screen

Screen: N>=25, second-half N>=10, full median >0, second-half median >0, top3-removed >0.

| Rank | Symbol | Threshold | N | Median | 2nd Median | 2nd Cum | No Fill | Note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | ETHUSDT | 1000K | 76 | +52.03 | +52.49 | +1724 | 29% | passes screen |
| 2 | SOLUSDT | 200K | 45 | +49.95 | +50.90 | +904 | 20% | passes screen |
| 3 | ETHUSDT | 500K | 117 | +47.50 | +50.23 | +1873 | 42% | passes screen |
| 4 | SOLUSDT | 100K | 81 | +26.92 | +44.53 | +921 | 21% | passes screen |
| 5 | BTCUSDT | 1000K | 68 | +7.01 | +34.58 | +1063 | 35% | passes screen |

## Notes

- SELL liq = liquidation of SHORT positions (forced short-covering → price spike).
- SHORT entry = fade the squeeze: hypothesis is price reverts after forced covering exhausts.
- Fill model: entry at bid (short), exits at ask (cover). Same taker fee as BUY side.
- This is research-only. No runner or config changes.