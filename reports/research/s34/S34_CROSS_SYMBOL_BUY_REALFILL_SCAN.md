# S34 Cross-Symbol BUY-Liq Real-Fill Scan

Generated: `2026-06-25T13:31:01.211068+00:00`
Window: `2026-02-25T12:20:45.373000+00:00` to `2026-06-25T12:20:45.373000+00:00` (120d lookback)

Scope: BUY liquidation cluster -> LONG, real historical bookTicker entry/exit fills, TP60/SL40/BE30. No live runner or config changes.

## Full Grid

| Symbol | Threshold | Signals | Real | No Fill | No Fill % | Median | Mean | Cum | WR | Days | Second Median | Second Cum | Top3 Removed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTCUSDT | 50K | 1298 | 498 | 800 | 61.6% | -9.06 | -2.62 | -1303.06 | 36.7% | 32 | -9.06 | -1303.06 | -1539.82 |
| BTCUSDT | 100K | 907 | 366 | 541 | 59.6% | -8.71 | 0.26 | 96.31 | 40.4% | 32 | -8.69 | 240.26 | -140.45 |
| BTCUSDT | 200K | 560 | 246 | 314 | 56.1% | -7.53 | 4.58 | 1126.62 | 43.5% | 32 | -7.22 | 1098.78 | 889.86 |
| BTCUSDT | 500K | 235 | 127 | 108 | 46.0% | 4.39 | 12.59 | 1599.39 | 51.2% | 29 | 3.39 | 915.87 | 1388.21 |
| BTCUSDT | 1000K | 119 | 80 | 39 | 32.8% | 42.14 | 21.42 | 1713.73 | 60.0% | 26 | 38.50 | 987.96 | 1502.55 |
| ETHUSDT | 50K | 1174 | 433 | 741 | 63.1% | -8.47 | 1.73 | 749.36 | 39.7% | 32 | -8.47 | 749.36 | 442.03 |
| ETHUSDT | 100K | 806 | 310 | 496 | 61.5% | -8.06 | 6.67 | 2066.69 | 44.2% | 32 | -8.06 | 2066.69 | 1770.78 |
| ETHUSDT | 200K | 477 | 205 | 272 | 57.0% | -7.33 | 10.55 | 2162.96 | 46.8% | 31 | -7.65 | 1903.46 | 1885.52 |
| ETHUSDT | 500K | 191 | 111 | 80 | 41.9% | 48.45 | 22.25 | 2469.46 | 60.4% | 28 | 50.17 | 1964.31 | 2192.03 |
| ETHUSDT | 1000K | 107 | 66 | 41 | 38.3% | 50.63 | 28.24 | 1863.68 | 66.7% | 25 | 52.16 | 1422.22 | 1586.24 |
| SOLUSDT | 50K | 141 | 106 | 35 | 24.8% | 5.29 | 11.41 | 1209.88 | 51.9% | 23 | 19.84 | 1092.56 | 994.22 |
| SOLUSDT | 100K | 95 | 72 | 23 | 24.2% | 39.57 | 19.40 | 1397.05 | 58.3% | 20 | 48.44 | 1080.01 | 1181.39 |
| SOLUSDT | 200K | 63 | 47 | 16 | 25.4% | 48.60 | 23.49 | 1103.82 | 63.8% | 16 | 48.12 | 750.55 | 909.14 |
| SOLUSDT | 500K | 22 | 20 | 2 | 9.1% | 52.91 | 34.00 | 680.06 | 80.0% | 12 | 48.60 | 319.28 | 482.98 |
| SOLUSDT | 1000K | 10 | 8 | 2 | 20.0% | 27.21 | 15.15 | 121.20 | 62.5% | 7 | 48.60 | 106.60 | -52.97 |

## Candidate Screen

Screen: N>=25, second-half N>=10, full median >0, second-half median >0, top3-removed cumulative >0.

| Rank | Symbol | Threshold | N | Median | Second N | Second Median | Second Cum | No Fill | Reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | ETHUSDT | 1000K | 66 | 50.63 | 37 | 52.16 | 1422.22 | 38.3% | passes min-N + positive median split screen |
| 2 | ETHUSDT | 500K | 111 | 48.45 | 73 | 50.17 | 1964.31 | 41.9% | passes min-N + positive median split screen |
| 3 | SOLUSDT | 100K | 72 | 39.57 | 48 | 48.44 | 1080.01 | 24.2% | passes min-N + positive median split screen |
| 4 | SOLUSDT | 200K | 47 | 48.60 | 32 | 48.12 | 750.55 | 25.4% | passes min-N + positive median split screen |
| 5 | BTCUSDT | 1000K | 80 | 42.14 | 45 | 38.50 | 987.96 | 32.8% | passes min-N + positive median split screen |
| 6 | SOLUSDT | 50K | 106 | 5.29 | 71 | 19.84 | 1092.56 | 24.8% | passes min-N + positive median split screen |
| 7 | BTCUSDT | 500K | 127 | 4.39 | 84 | 3.39 | 915.87 | 46.0% | passes min-N + positive median split screen |

## Notes

- This is research-only and not a paper-runner change.
- Second-half split is chronological by detected signal order, not randomized.
- No-fill rows are excluded from real-fill stats; high no-fill remains a selection-bias risk.
