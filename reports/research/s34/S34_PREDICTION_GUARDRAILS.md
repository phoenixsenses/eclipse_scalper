# S34 Prediction Guardrails

Generated: `2026-06-22T18:29:56.129649+00:00`

Diagnostic only. This report changes no runner rules or config.

## Model Summary

| Model | N | MAE bps | Bias bps | Direction hit | False green | Warned losses |
|---|---:|---:|---:|---:|---:|---:|
| base_rate_v1 | 63 | 45.31 | -4.95 | 63.5% | 13 | 17 |
| knn_v0 | 63 | 48.08 | -1.16 | 61.9% | 13 | 17 |
| knn_v1 | 63 | 48.02 | -1.01 | 61.9% | 13 | 17 |
| knn_v2 | 63 | 48.75 | -10.47 | 61.9% | 15 | 15 |

## Latest Closed Loss

`P418` `ETH_BUY_LIQ_LONG_50K_TP120_SL40_BE30` exited `SL` at `2026-06-22T16:46:25+00:00` for `-53.37` bps.

| Component | bps/value |
|---|---:|
| gross_bps | -40.76 |
| entry_adverse_bps | 4.12 |
| exit_adverse_bps | 0.43 |
| spread_cost_bps | 0.06 |
| fee_cost_bps | 8.00 |
| net_bps | -53.37 |

Interpretation: the loss was primarily directional because gross was near the SL distance; execution cost was normal taker fee plus tiny spread.

Cluster: notional `90926`, liq_count `5`, shape `distributed_mid_duration`.

## Worst False-Green Predictions

Rows where model expected >= +30 bps but outcome was negative.


### base_rate_v1

| Trade | Rule | Exit | Expected | Actual | Error | Cluster |
|---|---|---|---:|---:|---:|---:|
| P349 | SOL_BUY_LIQ_LONG_200K_TP60_SL40_BE30 | SL | 62.22 | -63.16 | -125.38 | 690696 |
| P064 | ETH_BUY_LIQ_LONG_200K_TP60_SL40_BE30 | SL | 51.96 | -56.76 | -108.72 | 296517 |
| P347 | ETH_BUY_LIQ_LONG_500K_DAYTREND0_TP60_SL40_BE30 | SL | 52.47 | -51.64 | -104.11 | 2246396 |
| P166 | ETH_BUY_LIQ_LONG_200K_TP60_SL40_BE30 | SL | 51.60 | -50.92 | -102.52 | 223432 |
| P346 | ETH_BUY_LIQ_LONG_200K_TP60_SL40_BE30 | SL | 49.79 | -51.64 | -101.43 | 2246396 |
| P163 | ETH_BUY_LIQ_LONG_200K_TP60_SL40_BE30 | SL | 53.49 | -47.74 | -101.23 | 662072 |
| P217 | ETH_BUY_LIQ_LONG_500K_DAYTREND0_TP60_SL40_BE30 | SL | 49.86 | -48.85 | -98.71 | 2279965 |
| P358 | ETH_BUY_LIQ_LONG_200K_TP60_SL40_BE30 | TIME | 48.70 | -18.67 | -67.37 | 324826 |

### knn_v0

| Trade | Rule | Exit | Expected | Actual | Error | Cluster |
|---|---|---|---:|---:|---:|---:|
| P348 | ETH_BUY_LIQ_LONG_50K_TP120_SL40_BE30 | SL | 99.67 | -51.64 | -151.31 | 2246396 |
| P349 | SOL_BUY_LIQ_LONG_200K_TP60_SL40_BE30 | SL | 77.43 | -63.16 | -140.59 | 690696 |
| P064 | ETH_BUY_LIQ_LONG_200K_TP60_SL40_BE30 | SL | 69.32 | -56.76 | -126.08 | 296517 |
| P166 | ETH_BUY_LIQ_LONG_200K_TP60_SL40_BE30 | SL | 60.31 | -50.92 | -111.23 | 223432 |
| P346 | ETH_BUY_LIQ_LONG_200K_TP60_SL40_BE30 | SL | 49.86 | -51.64 | -101.50 | 2246396 |
| P347 | ETH_BUY_LIQ_LONG_500K_DAYTREND0_TP60_SL40_BE30 | SL | 49.86 | -51.64 | -101.50 | 2246396 |
| P217 | ETH_BUY_LIQ_LONG_500K_DAYTREND0_TP60_SL40_BE30 | SL | 49.86 | -48.85 | -98.71 | 2279965 |
| P163 | ETH_BUY_LIQ_LONG_200K_TP60_SL40_BE30 | SL | 47.69 | -47.74 | -95.43 | 662072 |

### knn_v1

| Trade | Rule | Exit | Expected | Actual | Error | Cluster |
|---|---|---|---:|---:|---:|---:|
| P348 | ETH_BUY_LIQ_LONG_50K_TP120_SL40_BE30 | SL | 99.67 | -51.64 | -151.31 | 2246396 |
| P349 | SOL_BUY_LIQ_LONG_200K_TP60_SL40_BE30 | SL | 77.43 | -63.16 | -140.59 | 690696 |
| P064 | ETH_BUY_LIQ_LONG_200K_TP60_SL40_BE30 | SL | 69.32 | -56.76 | -126.08 | 296517 |
| P166 | ETH_BUY_LIQ_LONG_200K_TP60_SL40_BE30 | SL | 60.31 | -50.92 | -111.23 | 223432 |
| P346 | ETH_BUY_LIQ_LONG_200K_TP60_SL40_BE30 | SL | 49.86 | -51.64 | -101.50 | 2246396 |
| P347 | ETH_BUY_LIQ_LONG_500K_DAYTREND0_TP60_SL40_BE30 | SL | 49.86 | -51.64 | -101.50 | 2246396 |
| P217 | ETH_BUY_LIQ_LONG_500K_DAYTREND0_TP60_SL40_BE30 | SL | 49.86 | -48.85 | -98.71 | 2279965 |
| P163 | ETH_BUY_LIQ_LONG_200K_TP60_SL40_BE30 | SL | 47.69 | -47.74 | -95.43 | 662072 |

### knn_v2

| Trade | Rule | Exit | Expected | Actual | Error | Cluster |
|---|---|---|---:|---:|---:|---:|
| P349 | SOL_BUY_LIQ_LONG_200K_TP60_SL40_BE30 | SL | 77.43 | -63.16 | -140.59 | 690696 |
| P064 | ETH_BUY_LIQ_LONG_200K_TP60_SL40_BE30 | SL | 69.32 | -56.76 | -126.08 | 296517 |
| P391 | ETH_BUY_LIQ_LONG_50K_TP120_SL40_BE30 | BE | 113.15 | -7.71 | -120.87 | 55976 |
| P394 | ETH_BUY_LIQ_LONG_50K_TP120_SL40_BE30 | BE | 99.67 | -11.77 | -111.44 | 177013 |
| P346 | ETH_BUY_LIQ_LONG_200K_TP60_SL40_BE30 | SL | 53.49 | -51.64 | -105.14 | 2246396 |
| P166 | ETH_BUY_LIQ_LONG_200K_TP60_SL40_BE30 | SL | 53.49 | -50.92 | -104.41 | 223432 |
| P347 | ETH_BUY_LIQ_LONG_500K_DAYTREND0_TP60_SL40_BE30 | SL | 49.86 | -51.64 | -101.50 | 2246396 |
| P217 | ETH_BUY_LIQ_LONG_500K_DAYTREND0_TP60_SL40_BE30 | SL | 49.86 | -48.85 | -98.71 | 2279965 |

## Model Guardrail Performance

| Level | Signals | Closed | Loss rate | Median | Mean | Cum |
|---|---:|---:|---:|---:|---:|---:|
| warning | 267 | 26 | 61.5% | -11.32 | 5.48 | 142.36 |
| ok | 88 | 34 | 35.3% | 51.49 | 26.44 | 899.10 |
| unknown | 36 | 7 | 42.9% | 34.60 | 20.50 | 143.49 |
| caution | 27 | 3 | 66.7% | -17.33 | -12.63 | -37.88 |

Guardrail levels are observation labels only. They are not live execution filters.


### Latest Guardrails

| Level | Signal | Trade | Rule | Exit | Net |
|---|---|---|---|---|---:|
| warning | ETH_BUY_LIQ_LONG_50K_TP120_SL40_BE30:5940482 | P418 | ETH_BUY_LIQ_LONG_50K_TP120_SL40_BE30 | SL | -53.37 |
| warning | ETH_BUY_LIQ_LONG_50K_TP120_SL40_BE30:5940455 | - | - | - | n/a |
| warning | ETH_BUY_LIQ_LONG_50K_TP120_SL40_BE30:5940452 | P416 | ETH_BUY_LIQ_LONG_50K_TP120_SL40_BE30 | SL | -48.03 |
| warning | ETH_BUY_LIQ_LONG_200K_BTC_PRE15_TP120_SL40_BE30_DELAY60:5940451 | - | - | - | n/a |
| ok | ETH_BUY_LIQ_LONG_200K_TP60_SL40_BE30:5940451 | - | - | - | n/a |
| ok | ETH_BUY_LIQ_LONG_500K_DAYTREND0_TP60_SL40_BE30:5940451 | P413 | ETH_BUY_LIQ_LONG_500K_DAYTREND0_TP60_SL40_BE30 | TP | 52.33 |
| warning | BTC_BUY_LIQ_LONG_1M_DISTRIBUTED_TP60_SL30_BE30:5940451 | P412 | BTC_BUY_LIQ_LONG_1M_DISTRIBUTED_TP60_SL30_BE30 | TP | 55.02 |
| warning | ETH_BUY_LIQ_LONG_50K_TP120_SL40_BE30:5940451 | - | - | - | n/a |
| warning | ETH_BUY_LIQ_LONG_50K_TP120_SL40_BE30:5940450 | - | - | - | n/a |
| ok | ETH_BUY_LIQ_LONG_200K_TP60_SL40_BE30:5940450 | P409 | ETH_BUY_LIQ_LONG_200K_TP60_SL40_BE30 | TP | 54.04 |

## Correct Loss Warnings

Recent losses where the model already expected negative net bps.


### base_rate_v1

| Trade | Rule | Exit | Expected | Actual |
|---|---|---|---:|---:|
| P418 | ETH_BUY_LIQ_LONG_50K_TP120_SL40_BE30 | SL | -11.32 | -53.37 |
| P416 | ETH_BUY_LIQ_LONG_50K_TP120_SL40_BE30 | SL | -10.86 | -48.03 |
| P394 | ETH_BUY_LIQ_LONG_50K_TP120_SL40_BE30 | BE | -10.34 | -11.77 |
| P391 | ETH_BUY_LIQ_LONG_50K_TP120_SL40_BE30 | BE | -10.86 | -7.71 |
| P361 | ETH_BUY_LIQ_LONG_50K_TP120_SL40_BE30 | TIME | -10.34 | -43.52 |
| P357 | ETH_BUY_LIQ_LONG_200K_BTC_PRE15_TP120_SL40_BE30_DELAY60 | TIME | -18.72 | -24.58 |
| P348 | ETH_BUY_LIQ_LONG_50K_TP120_SL40_BE30 | SL | -10.86 | -51.64 |
| P169 | ETH_BUY_LIQ_LONG_50K_TP120_SL40_BE30 | SL | -10.86 | -51.87 |

### knn_v0

| Trade | Rule | Exit | Expected | Actual |
|---|---|---|---:|---:|
| P418 | ETH_BUY_LIQ_LONG_50K_TP120_SL40_BE30 | SL | -11.77 | -53.37 |
| P416 | ETH_BUY_LIQ_LONG_50K_TP120_SL40_BE30 | SL | -43.52 | -48.03 |
| P394 | ETH_BUY_LIQ_LONG_50K_TP120_SL40_BE30 | BE | -46.30 | -11.77 |
| P391 | ETH_BUY_LIQ_LONG_50K_TP120_SL40_BE30 | BE | -8.85 | -7.71 |
| P389 | ETH_BUY_LIQ_LONG_200K_TP60_SL40_BE30 | TIME | -20.17 | -17.33 |
| P361 | ETH_BUY_LIQ_LONG_50K_TP120_SL40_BE30 | TIME | -45.82 | -43.52 |
| P357 | ETH_BUY_LIQ_LONG_200K_BTC_PRE15_TP120_SL40_BE30_DELAY60 | TIME | -18.72 | -24.58 |
| P169 | ETH_BUY_LIQ_LONG_50K_TP120_SL40_BE30 | SL | -45.82 | -51.87 |

### knn_v1

| Trade | Rule | Exit | Expected | Actual |
|---|---|---|---:|---:|
| P418 | ETH_BUY_LIQ_LONG_50K_TP120_SL40_BE30 | SL | -9.83 | -53.37 |
| P416 | ETH_BUY_LIQ_LONG_50K_TP120_SL40_BE30 | SL | -43.52 | -48.03 |
| P394 | ETH_BUY_LIQ_LONG_50K_TP120_SL40_BE30 | BE | -46.30 | -11.77 |
| P391 | ETH_BUY_LIQ_LONG_50K_TP120_SL40_BE30 | BE | -9.83 | -7.71 |
| P389 | ETH_BUY_LIQ_LONG_200K_TP60_SL40_BE30 | TIME | -20.17 | -17.33 |
| P361 | ETH_BUY_LIQ_LONG_50K_TP120_SL40_BE30 | TIME | -45.82 | -43.52 |
| P357 | ETH_BUY_LIQ_LONG_200K_BTC_PRE15_TP120_SL40_BE30_DELAY60 | TIME | -18.72 | -24.58 |
| P169 | ETH_BUY_LIQ_LONG_50K_TP120_SL40_BE30 | SL | -45.82 | -51.87 |

### knn_v2

| Trade | Rule | Exit | Expected | Actual |
|---|---|---|---:|---:|
| P418 | ETH_BUY_LIQ_LONG_50K_TP120_SL40_BE30 | SL | -9.83 | -53.37 |
| P416 | ETH_BUY_LIQ_LONG_50K_TP120_SL40_BE30 | SL | -45.82 | -48.03 |
| P361 | ETH_BUY_LIQ_LONG_50K_TP120_SL40_BE30 | TIME | -45.82 | -43.52 |
| P357 | ETH_BUY_LIQ_LONG_200K_BTC_PRE15_TP120_SL40_BE30_DELAY60 | TIME | -18.72 | -24.58 |
| P348 | ETH_BUY_LIQ_LONG_50K_TP120_SL40_BE30 | SL | -9.83 | -51.64 |
| P169 | ETH_BUY_LIQ_LONG_50K_TP120_SL40_BE30 | SL | -45.82 | -51.87 |
| P150 | ETH_BUY_LIQ_LONG_50K_TP120_SL40_BE30 | SL | -9.83 | -55.59 |
| P149 | ETH_BUY_LIQ_LONG_50K_TP120_SL40_BE30 | SL | -9.83 | -46.30 |

## Conditional Base Rates


### by_rule

| Bucket | N | Median | Mean | Cum | Win rate |
|---|---:|---:|---:|---:|---:|
| ETH_BUY_LIQ_LONG_200K_TP60_SL40_BE30 | 23 | 47.69 | 18.22 | 419.02 | 56.5% |
| ETH_BUY_LIQ_LONG_50K_TP120_SL40_BE30 | 22 | -27.65 | 0.74 | 16.29 | 31.8% |
| ETH_BUY_LIQ_LONG_500K_DAYTREND0_TP60_SL40_BE30 | 10 | 54.32 | 38.01 | 380.05 | 80.0% |
| SOL_BUY_LIQ_LONG_200K_TP60_SL40_BE30 | 6 | 50.59 | 26.30 | 157.77 | 66.7% |
| BTC_BUY_LIQ_LONG_1M_DISTRIBUTED_TP60_SL30_BE30 | 1 | 55.02 | 55.02 | 55.02 | 100.0% |
| ETH_BUY_LIQ_LONG_200K_BTC_PRE15_TP120_SL40_BE30_DELAY60 | 1 | -24.58 | -24.58 | -24.58 | 0.0% |

### by_cluster_bucket

| Bucket | N | Median | Mean | Cum | Win rate |
|---|---:|---:|---:|---:|---:|
| 200K-500K | 19 | 31.09 | 20.00 | 379.99 | 52.6% |
| >=1M | 19 | 52.33 | 32.75 | 622.28 | 68.4% |
| <200K | 14 | -46.06 | -24.90 | -348.60 | 14.3% |
| 500K-1M | 11 | 50.64 | 31.81 | 349.91 | 72.7% |

### by_session

| Bucket | N | Median | Mean | Cum | Win rate |
|---|---:|---:|---:|---:|---:|
| 13-17 UTC | 24 | 51.02 | 13.69 | 328.57 | 54.2% |
| 17-24 UTC | 23 | -10.86 | 0.45 | 10.41 | 39.1% |
| 07-13 UTC | 9 | 49.86 | 46.26 | 416.33 | 66.7% |
| 00-07 UTC | 7 | 56.32 | 35.47 | 248.27 | 71.4% |

### by_exit_reason

| Bucket | N | Median | Mean | Cum | Win rate |
|---|---:|---:|---:|---:|---:|
| TP | 30 | 57.81 | 68.58 | 2057.32 | 100.0% |
| SL | 18 | -51.64 | -52.14 | -938.51 | 0.0% |
| BE | 8 | -10.26 | -10.46 | -83.71 | 0.0% |
| TIME | 7 | -17.33 | -4.50 | -31.52 | 42.9% |

## Guardrail Takeaway

Use KNN as an evidence/audit surface, not as an execution trigger. When KNN and base-rate both expect negative bps, show a visible warning in the dashboard; do not change live rules from this report alone.

