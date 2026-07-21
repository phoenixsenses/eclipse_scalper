# S34 SELL-Liq Delayed LONG Scan

Generated: `2026-06-19T17:36:02.388699+00:00`

Scope: `ETH SELL liquidation cluster -> LONG`, threshold x delay x TP grid; live runner/config unchanged.

Combinations evaluated: `60`
Event counts by threshold: `{'50000': 326, '100000': 252, '200000': 177, '500000': 95}`
Temporal split: `2026-04-23T02:42:51.370000+00:00`

## 1. Train-Selected Top 5, With Test Performance

| Rank | Route | Train N | Train Median | Train Cum | Train Top3 Removed | Test N | Test Median | Test Mean | Test Cum | Test WR | Test Top3 Removed | Test Exits |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | TH500K_DELAY600_TP40_SL40_BE30 | 49 | +4.68 | +64.92 | -43.30 | 46 | +18.55 | -0.84 | -38.66 | 54.3% | -156.10 | {'BE': 4, 'SL': 15, 'TIME': 6, 'TP': 21} |
| 2 | TH200K_DELAY600_TP40_SL40_BE30 | 85 | +2.05 | -42.04 | -151.83 | 92 | -8.00 | -8.09 | -744.35 | 40.2% | -861.78 | {'BE': 11, 'SL': 32, 'TIME': 18, 'TP': 31} |
| 3 | TH200K_DELAY600_TP60_SL40_BE30 | 85 | -4.89 | -46.02 | -215.25 | 92 | -8.00 | -8.63 | -794.25 | 30.4% | -981.65 | {'BE': 20, 'SL': 32, 'TIME': 23, 'TP': 17} |
| 4 | TH500K_DELAY600_TP60_SL40_BE30 | 49 | -8.00 | -52.67 | -215.73 | 46 | -8.00 | -3.19 | -146.80 | 37.0% | -332.88 | {'BE': 12, 'SL': 15, 'TIME': 8, 'TP': 11} |
| 5 | TH500K_DELAY600_TP80_SL40_BE30 | 49 | -8.00 | -36.92 | -257.73 | 46 | -8.00 | -1.34 | -61.54 | 34.8% | -291.32 | {'BE': 13, 'SL': 15, 'TIME': 9, 'TP': 9} |

## 2. Real-Fill Parity: Top 5

| Route | Total | Real Fill | No Fill | No Fill Rate | Test N | Test Median | Test Mean | Test Cum | Test Top3 Removed | Test Positive Days | Entry Adv | Exit Adv | Spread | Fee |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| TH500K_DELAY600_TP40_SL40_BE30 | 95 | 90 | 5 | 5.3% | 46 | +18.39 | -0.82 | -37.57 | -172.45 | 8/13 | -0.43 | +0.52 | +0.05 | +8.00 |
| TH200K_DELAY600_TP40_SL40_BE30 | 177 | 167 | 10 | 5.6% | 91 | -8.28 | -8.37 | -761.60 | -896.48 | 5/14 | -0.44 | +0.69 | +0.05 | +8.00 |
| TH200K_DELAY600_TP60_SL40_BE30 | 177 | 165 | 12 | 6.8% | 91 | -9.45 | -9.07 | -825.09 | -1022.80 | 3/14 | -0.42 | +0.51 | +0.05 | +8.00 |
| TH500K_DELAY600_TP60_SL40_BE30 | 95 | 90 | 5 | 5.3% | 46 | -8.72 | -4.12 | -189.74 | -386.71 | 6/13 | -0.43 | +1.09 | +0.05 | +8.00 |
| TH500K_DELAY600_TP80_SL40_BE30 | 95 | 89 | 6 | 6.3% | 46 | -8.72 | -2.48 | -114.00 | -347.15 | 6/13 | -0.42 | +1.16 | +0.05 | +8.00 |

## Read

Best real-fill test median among train-selected routes: `TH500K_DELAY600_TP40_SL40_BE30`. This is a broad retrospective scan; any promotion would need a separate pre-registration and should be treated as a new alpha family.
