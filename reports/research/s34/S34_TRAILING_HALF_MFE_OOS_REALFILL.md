# S34 Trailing Half-MFE OOS + Real-Fill Check

Generated: 2026-06-17T09:34:39.314493+00:00

Scope: ETH BUY feature-factory events with `cluster_notional >= 500K AND day_trend_bps >= 0`, route `LONG_DELAY0_TP60`.

No runner/config changes. This is research-only.

Events: `97`
Temporal split: `2026-04-17T03:31:34.616000+00:00`

## 1. Temporal OOS, Simplified Mark-Fill

| Variant | Period | N | Days | Cum | Mean | Median | WR | Top3 Removed | Positive Days | Exits |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| CURRENT_BE30 | train | 45 | 26 | +968.08 | +21.51 | +29.91 | 57.8% | +782.53 | 16/26 | {'BE': 14, 'SL': 4, 'TIME': 5, 'TP': 22} |
| CURRENT_BE30 | test | 52 | 12 | +1385.74 | +26.65 | +52.63 | 63.5% | +1203.21 | 12/12 | {'BE': 13, 'SL': 5, 'TIME': 3, 'TP': 31} |
| CURRENT_BE30 | all | 97 | 38 | +2353.82 | +24.27 | +52.21 | 60.8% | +2164.02 | 28/38 | {'BE': 27, 'SL': 9, 'TIME': 8, 'TP': 53} |
| TRAIL_HALF_MFE_ARM30 | train | 45 | 26 | +1147.73 | +25.51 | +17.10 | 88.9% | +962.18 | 25/26 | {'SL': 4, 'TIME': 3, 'TP': 20, 'TRAIL': 18} |
| TRAIL_HALF_MFE_ARM30 | test | 52 | 12 | +1310.87 | +25.21 | +18.46 | 88.5% | +1128.34 | 12/12 | {'SL': 5, 'TIME': 1, 'TP': 24, 'TRAIL': 22} |
| TRAIL_HALF_MFE_ARM30 | all | 97 | 38 | +2458.61 | +25.35 | +17.88 | 88.7% | +2268.80 | 37/38 | {'SL': 9, 'TIME': 4, 'TP': 44, 'TRAIL': 40} |

## 2. Real-Fill Parity

| Variant | Total | Real Fill | No Fill | No Fill Rate | Period | Real Cum | Real Mean | Real Median | WR | Top3 Removed | Positive Days |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| CURRENT_BE30 | 97 | 51 | 46 | 47.4% | train | +217.58 | +21.76 | +37.62 | 70.0% | +45.29 | 4/4 |
| CURRENT_BE30 | 97 | 51 | 46 | 47.4% | test | +1108.08 | +27.03 | +48.45 | 65.9% | +853.78 | 9/10 |
| CURRENT_BE30 | 97 | 51 | 46 | 47.4% | all | +1325.66 | +25.99 | +48.45 | 66.7% | +1071.36 | 13/14 |
| TRAIL_HALF_MFE_ARM30 | 97 | 51 | 46 | 47.4% | train | +166.84 | +16.68 | +11.96 | 80.0% | -5.45 | 4/4 |
| TRAIL_HALF_MFE_ARM30 | 97 | 51 | 46 | 47.4% | test | +950.44 | +23.18 | +16.76 | 82.9% | +696.15 | 9/10 |
| TRAIL_HALF_MFE_ARM30 | 97 | 51 | 46 | 47.4% | all | +1117.28 | +21.91 | +14.19 | 82.4% | +862.99 | 13/14 |

## 3. Cost Components, Real-Fill Rows

| Variant | Entry Adverse | Exit Adverse | Spread | Fee |
|---|---:|---:|---:|---:|
| CURRENT_BE30 | +1.11 | -1.45 | +0.06 | +8.00 |
| TRAIL_HALF_MFE_ARM30 | +1.11 | -0.70 | +0.06 | +8.00 |

## Read

This checks whether the trailing idea survives a temporal split and whether the result remains positive under real historical bid/ask fills where available. It is still a retrospective sweep on a discovered exit idea, not authorization to change the live runner.
