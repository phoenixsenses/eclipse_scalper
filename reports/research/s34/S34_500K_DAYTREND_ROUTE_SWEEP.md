# S34 500K/daytrend Route Sweep

Generated: 2026-06-17T18:08:39.373874+00:00

Scope: `ETH BUY`, `cluster_notional >= 500K`, `day_trend_bps >= 0`, `delay0`; live runner/config unchanged.

Combinations evaluated: `75`
Events: `97`; simulated events: `97`
Temporal split: `2026-04-17T03:31:34.616000+00:00`

## 1. Train-Selected Top 5, With Test Performance

| Rank | Route | Train N | Train Median | Train Cum | Train Top3 Removed | Test N | Test Median | Test Mean | Test Cum | Test WR | Test Top3 Removed | Test Exits |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | TP40_SL50_BE20 | 45 | +32.52 | +1036.16 | +905.14 | 52 | +33.15 | +20.43 | +1062.15 | 69.2% | +910.99 | {'BE': 14, 'SL': 1, 'TIME': 2, 'TP': 35} |
| 2 | TP60_SL50_BE25 | 45 | +52.16 | +1090.67 | +905.12 | 52 | +52.79 | +28.74 | +1494.71 | 65.4% | +1312.18 | {'BE': 14, 'SL': 3, 'TIME': 3, 'TP': 32} |
| 3 | TP40_SL50_BE25 | 45 | +32.52 | +1035.61 | +904.59 | 52 | +33.31 | +20.30 | +1055.53 | 73.1% | +902.69 | {'BE': 10, 'SL': 3, 'TIME': 2, 'TP': 37} |
| 4 | TP60_SL50_BE20 | 45 | +29.91 | +1065.93 | +880.38 | 52 | +52.43 | +28.33 | +1473.16 | 61.5% | +1290.62 | {'BE': 18, 'SL': 1, 'TIME': 3, 'TP': 30} |
| 5 | TP60_SL50_BE30 | 45 | +52.16 | +1039.29 | +853.74 | 52 | +52.79 | +27.78 | +1444.71 | 65.4% | +1262.18 | {'BE': 13, 'SL': 4, 'TIME': 3, 'TP': 32} |

## 2. Current Live Route Comparator

| Route | Period | N | Median | Mean | Cum | WR | Top3 Removed | Exits |
|---|---|---:|---:|---:|---:|---:|---:|---|
| TP60_SL40_BE30 | train | 45 | +29.91 | +21.51 | +968.08 | 57.8% | +782.53 | {'BE': 14, 'SL': 4, 'TIME': 5, 'TP': 22} |
| TP60_SL40_BE30 | test | 52 | +52.63 | +26.65 | +1385.74 | 63.5% | +1203.21 | {'BE': 13, 'SL': 5, 'TIME': 3, 'TP': 31} |
| TP60_SL40_BE30 | all | 97 | +52.21 | +24.27 | +2353.82 | 60.8% | +2164.02 | {'BE': 27, 'SL': 9, 'TIME': 8, 'TP': 53} |

## 3. Real-Fill Parity: Top 5 + Current

| Route | Total | Real Fill | No Fill | No Fill Rate | Test N | Test Median | Test Mean | Test Cum | Test Top3 Removed | Test Positive Days | Entry Adv | Exit Adv | Spread | Fee |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| TP40_SL50_BE20 | 97 | 51 | 46 | 47.4% | 41 | +31.74 | +23.50 | +963.38 | +711.33 | 10/10 | +1.11 | -2.95 | +0.06 | +8.00 |
| TP60_SL50_BE25 | 97 | 51 | 46 | 47.4% | 41 | +48.48 | +29.66 | +1216.09 | +961.80 | 10/10 | +1.11 | -1.43 | +0.06 | +8.00 |
| TP40_SL50_BE25 | 97 | 51 | 46 | 47.4% | 41 | +31.95 | +24.43 | +1001.81 | +749.76 | 10/10 | +1.11 | -4.08 | +0.06 | +8.00 |
| TP60_SL50_BE20 | 97 | 51 | 46 | 47.4% | 41 | +48.45 | +28.21 | +1156.62 | +902.33 | 10/10 | +1.11 | -0.40 | +0.06 | +8.00 |
| TP60_SL50_BE30 | 97 | 51 | 46 | 47.4% | 41 | +48.48 | +28.43 | +1165.45 | +911.15 | 9/10 | +1.11 | -1.45 | +0.06 | +8.00 |
| TP60_SL40_BE30 | 97 | 51 | 46 | 47.4% | 41 | +48.45 | +27.03 | +1108.08 | +853.78 | 9/10 | +1.11 | -1.45 | +0.06 | +8.00 |

## Read

Best real-fill test median among reported routes: `TP60_SL50_BE25`. This remains a retrospective route sweep over 75 combinations; a stronger route should be pre-registered as a separate exploratory variant before any live paper promotion.
