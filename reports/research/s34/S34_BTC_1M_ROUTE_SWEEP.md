# S34 BTC 1M Route Sweep

Generated: `2026-06-27T19:39:53.227857+00:00`

Scope: `BTC BUY`, `cluster_notional >= 1M`, `delay0`; live runner/config unchanged.

Combinations evaluated: `100`
Signals: `61`; real-entry events: `44`; simulated events: `44`
Temporal split: `2026-04-27T01:01:04.387000+00:00`

## 1. Train-Selected Top 5, With Test Performance

| Rank | Route | Train N | Train Median | Train Cum | Train Top3 Removed | Test N | Test Median | Test Mean | Test Cum | Test WR | Test Top3 Removed | Test Exits |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | TP40_SL30_BE40 | 23 | +32.07 | +41.36 | -66.65 | 21 | -28.37 | -5.84 | -122.56 | 47.6% | -226.01 | {'SL': 10, 'TIME': 3, 'TP': 8} |
| 2 | TP40_SL40_BE40 | 23 | +32.09 | +40.08 | -67.93 | 21 | -27.96 | -9.34 | -196.21 | 47.6% | -299.67 | {'SL': 9, 'TIME': 4, 'TP': 8} |
| 3 | TP40_SL50_BE40 | 23 | +32.09 | -19.46 | -127.47 | 21 | +5.54 | -7.28 | -152.95 | 52.4% | -258.07 | {'SL': 6, 'TIME': 6, 'TP': 9} |
| 4 | TP40_SL40_BE20 | 23 | -8.00 | +23.47 | -78.20 | 21 | -8.00 | -7.71 | -161.88 | 28.6% | -265.34 | {'BE': 8, 'SL': 5, 'TIME': 2, 'TP': 6} |
| 5 | TP40_SL50_BE20 | 23 | -8.00 | -13.65 | -115.32 | 21 | -8.00 | -0.88 | -18.53 | 33.3% | -123.65 | {'BE': 10, 'SL': 2, 'TIME': 2, 'TP': 7} |

## 2. Current BTC Route Comparator

| Route | Period | N | Median | Mean | Cum | WR | Top3 Removed | Exits |
|---|---|---:|---:|---:|---:|---:|---:|---|
| TP60_SL30_BE30 | train | 23 | -8.00 | -4.29 | -98.67 | 26.1% | -264.97 | {'BE': 8, 'SL': 9, 'TP': 6} |
| TP60_SL30_BE30 | test | 21 | -8.00 | -9.47 | -198.89 | 23.8% | -358.10 | {'BE': 6, 'SL': 9, 'TIME': 2, 'TP': 4} |
| TP60_SL30_BE30 | all | 44 | -8.00 | -6.76 | -297.56 | 25.0% | -464.35 | {'BE': 14, 'SL': 18, 'TIME': 2, 'TP': 10} |

## 3. Real-Fill Parity: Top 5 + Current

| Route | Total | Real Fill | No Fill | No Fill Rate | Test N | Test Median | Test Mean | Test Cum | Test Top3 Removed | Test Positive Days | Entry Adv | Exit Adv | Spread | Fee |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| TP40_SL30_BE40 | 44 | 44 | 0 | 0.0% | 21 | -27.64 | -5.71 | -119.89 | -230.24 | 3/14 | +0.00 | -0.84 | +0.01 | +8.00 |
| TP40_SL40_BE40 | 44 | 44 | 0 | 0.0% | 21 | -25.44 | -9.05 | -190.00 | -300.35 | 3/14 | +0.00 | -0.74 | +0.02 | +8.00 |
| TP40_SL50_BE40 | 44 | 44 | 0 | 0.0% | 21 | +1.62 | -6.85 | -143.85 | -254.93 | 5/14 | +0.00 | -0.77 | +0.01 | +8.00 |
| TP40_SL40_BE20 | 44 | 44 | 0 | 0.0% | 21 | -8.64 | -7.71 | -161.82 | -272.17 | 4/14 | +0.00 | +0.43 | +0.01 | +8.00 |
| TP40_SL50_BE20 | 44 | 44 | 0 | 0.0% | 21 | -8.59 | -0.89 | -18.64 | -129.71 | 5/14 | +0.00 | +0.38 | +0.01 | +8.00 |
| TP60_SL30_BE30 | 44 | 43 | 1 | 2.3% | 21 | -11.51 | -9.99 | -209.69 | -368.85 | 4/14 | +0.07 | +0.62 | +0.01 | +8.00 |

## Read

Best real-fill test median among reported routes: `TP40_SL50_BE40`. This is retrospective over 75 combinations; promotion would require a separate exploratory pre-registration. Existing live runner remains unchanged by this report.
