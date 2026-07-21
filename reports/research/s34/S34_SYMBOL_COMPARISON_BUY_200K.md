# S34 Symbol Comparison - BUY 200K

Generated: 2026-06-17T09:11:14.605363+00:00

Scope: BTCUSDT / ETHUSDT / SOLUSDT BUY liquidation clusters >= 200K, 300s bucket, 900s minimum gap.

Costs are the simplified Phase-1 model: net = gross - 8 bps. This is descriptive research, not a runner change.

## Coverage

| Symbol | Events | First | Last | Median Notional | Median Count | Median Duration sec |
|---|---:|---|---|---:|---:|---:|
| BTCUSDT | 544 | 2026-02-15T14:40:48.834000+00:00 | 2026-06-17T02:22:42.383000+00:00 | 333,769 | 14 | 139.4 |
| ETHUSDT | 460 | 2026-02-15T22:47:11.071000+00:00 | 2026-06-17T03:32:04.477000+00:00 | 347,538 | 15 | 155.5 |
| SOLUSDT | 49 | 2026-04-20T07:20:01.642000+00:00 | 2026-06-17T02:22:56.089000+00:00 | 343,332 | 13 | 121.4 |

## Base Route Comparison

| Symbol | Route | N | Days | Mean | Median | Cum | WR | TP/BE/SL/TIME | Mean MFE | Mean MAE |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|
| BTCUSDT | LONG_DELAY0_TP60 | 544 | 80 | +6.56 | -8.10 | +3570.01 | 43.6% | 183/150/96/115 | +41.35 | -14.46 |
| BTCUSDT | LONG_DELAY60_TP120 | 544 | 80 | -2.39 | -8.60 | -1297.77 | 29.8% | 43/163/144/194 | +44.30 | -19.88 |
| BTCUSDT | SHORT_DELAY0_TP40_CONTROL | 544 | 80 | -23.08 | -48.19 | -12555.54 | 27.9% | 109/20/307/108 | +16.58 | -32.86 |
| ETHUSDT | LONG_DELAY0_TP60 | 460 | 78 | +13.99 | -3.37 | +6437.08 | 49.8% | 210/134/78/38 | +45.77 | -13.24 |
| ETHUSDT | LONG_DELAY60_TP120 | 460 | 78 | +4.73 | -8.77 | +2174.81 | 28.9% | 74/171/131/84 | +55.01 | -19.40 |
| ETHUSDT | SHORT_DELAY0_TP40_CONTROL | 460 | 78 | -27.63 | -48.46 | -12708.46 | 23.0% | 95/30/305/30 | +15.69 | -35.14 |
| SOLUSDT | LONG_DELAY0_TP60 | 49 | 15 | +18.66 | +52.13 | +914.51 | 57.1% | 26/12/8/3 | +47.74 | -13.63 |
| SOLUSDT | LONG_DELAY60_TP120 | 49 | 15 | +6.65 | -8.78 | +325.61 | 32.7% | 9/16/15/9 | +56.94 | -20.75 |
| SOLUSDT | SHORT_DELAY0_TP40_CONTROL | 49 | 15 | -29.18 | -48.76 | -1429.91 | 16.3% | 8/5/31/5 | +16.08 | -34.78 |

## 500K + Day-Trend >= 0 Slice

| Symbol | Route | N | Days | Mean | Median | Cum | WR | TP/BE/SL/TIME |
|---|---|---:|---:|---:|---:|---:|---:|---|
| BTCUSDT | LONG_DELAY0_TP60 | 141 | 45 | +9.18 | -8.15 | +1293.68 | 45.4% | 51/46/24/20 |
| BTCUSDT | LONG_DELAY60_TP120 | 141 | 45 | -3.75 | -8.62 | -528.57 | 27.0% | 13/45/42/41 |
| BTCUSDT | SHORT_DELAY0_TP40_CONTROL | 141 | 45 | -26.10 | -48.28 | -3679.56 | 24.8% | 28/4/88/21 |
| ETHUSDT | LONG_DELAY0_TP60 | 100 | 40 | +24.25 | +52.26 | +2425.50 | 61.0% | 55/28/9/8 |
| ETHUSDT | LONG_DELAY60_TP120 | 100 | 40 | +12.60 | -8.36 | +1260.16 | 34.0% | 20/38/22/20 |
| ETHUSDT | SHORT_DELAY0_TP40_CONTROL | 100 | 40 | -34.43 | -49.07 | -3442.92 | 17.0% | 15/5/74/6 |
| SOLUSDT | LONG_DELAY0_TP60 | 10 | 8 | +22.13 | +52.23 | +221.34 | 70.0% | 6/1/2/1 |
| SOLUSDT | LONG_DELAY60_TP120 | 10 | 8 | +25.03 | +8.49 | +250.31 | 50.0% | 3/2/2/3 |
| SOLUSDT | SHORT_DELAY0_TP40_CONTROL | 10 | 8 | -28.58 | -48.69 | -285.84 | 20.0% | 2/0/6/2 |

## Observations

- This report compares the same mechanical BUY-liq cluster setup across symbols.
- It does not account for historical bid/ask real-fill parity by symbol.
- It does not promote, kill, or modify any runner rule.
