# S34 SOL 200K Route Sweep

Generated: `2026-06-19T17:04:23.380538+00:00`

Scope: `SOL BUY`, `cluster_notional >= 200K`, `delay0`; live runner/config unchanged.

Combinations evaluated: `75`
Signals: `51`; real-entry events: `35`; simulated events: `35`
Temporal split: `2026-06-14T21:15:12.471000+00:00`

## 1. Train-Selected Top 5, With Test Performance

| Rank | Route | Train N | Train Median | Train Cum | Train Top3 Removed | Test N | Test Median | Test Mean | Test Cum | Test WR | Test Top3 Removed | Test Exits |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | TP60_SL40_BE20 | 18 | +52.35 | +537.14 | +370.35 | 17 | +52.68 | +24.88 | +423.04 | 52.9% | +255.23 | {'BE': 8, 'TP': 9} |
| 2 | TP60_SL40_BE25 | 18 | +52.35 | +530.48 | +363.69 | 17 | +52.68 | +22.52 | +382.82 | 52.9% | +215.00 | {'BE': 7, 'SL': 1, 'TP': 9} |
| 3 | TP60_SL50_BE20 | 18 | +52.35 | +528.65 | +361.85 | 17 | +52.68 | +24.88 | +423.04 | 52.9% | +255.23 | {'BE': 8, 'TP': 9} |
| 4 | TP60_SL50_BE25 | 18 | +52.35 | +521.99 | +355.20 | 17 | +52.68 | +21.94 | +372.94 | 52.9% | +205.13 | {'BE': 7, 'SL': 1, 'TP': 9} |
| 5 | TP60_SL40_BE30 | 18 | +52.35 | +490.34 | +323.55 | 17 | +52.68 | +17.58 | +298.94 | 52.9% | +131.13 | {'BE': 5, 'SL': 3, 'TP': 9} |

## 2. Current Live SOL Route Comparator

| Route | Period | N | Median | Mean | Cum | WR | Top3 Removed | Exits |
|---|---|---:|---:|---:|---:|---:|---:|---|
| TP60_SL40_BE30 | train | 18 | +52.35 | +27.24 | +490.34 | 72.2% | +323.55 | {'BE': 2, 'SL': 2, 'TIME': 3, 'TP': 11} |
| TP60_SL40_BE30 | test | 17 | +52.68 | +17.58 | +298.94 | 52.9% | +131.13 | {'BE': 5, 'SL': 3, 'TP': 9} |
| TP60_SL40_BE30 | all | 35 | +52.35 | +22.55 | +789.29 | 62.9% | +619.35 | {'BE': 7, 'SL': 5, 'TIME': 3, 'TP': 20} |

## 3. Real-Fill Parity: Top 5 + Current

| Route | Total | Real Fill | No Fill | No Fill Rate | Test N | Test Median | Test Mean | Test Cum | Test Top3 Removed | Test Positive Days | Entry Adv | Exit Adv | Spread | Fee |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| TP60_SL40_BE20 | 35 | 35 | 0 | 0.0% | 17 | +47.93 | +23.49 | +399.31 | +227.46 | 4/4 | -0.67 | +0.44 | +1.34 | +8.00 |
| TP60_SL40_BE25 | 35 | 35 | 0 | 0.0% | 17 | +47.93 | +21.47 | +365.05 | +193.20 | 4/4 | -0.67 | +0.21 | +1.34 | +8.00 |
| TP60_SL50_BE20 | 35 | 35 | 0 | 0.0% | 17 | +47.93 | +23.49 | +399.31 | +227.46 | 4/4 | -0.67 | +0.37 | +1.34 | +8.00 |
| TP60_SL50_BE25 | 35 | 35 | 0 | 0.0% | 17 | +47.93 | +20.78 | +353.19 | +181.34 | 4/4 | -0.67 | +0.19 | +1.34 | +8.00 |
| TP60_SL40_BE30 | 35 | 35 | 0 | 0.0% | 17 | +47.93 | +16.62 | +282.51 | +110.67 | 4/4 | -0.67 | +0.13 | +1.34 | +8.00 |

## Read

Best real-fill test median among reported routes: `TP60_SL40_BE20`. This is retrospective over 75 combinations; promotion would require a separate exploratory pre-registration. Existing live SOL rule remains unchanged by this report.
