# S34 Pre-Liq Detector Research

Generated: `2026-06-27T13:29:31.716750+00:00`

Research only. Builds a labeled detector dataset for ETH SELL pre-liquidation book pressure.

Positive counts: 500K=125, 1M=75. Controls=625.

Controls are matched to broad `mid_down_10s >= 5 bps` and `spread <= 1 bps`, excluding +/-900s around ETH SELL liquidation clusters.

## Feature Separation

| Feature | AUC 500K | AUC 1M | Pos500 med | Pos1M med | Control med |
| --- | ---: | ---: | ---: | ---: | ---: |
| spread_bps | 0.542 | 0.528 | 0.056 | 0.056 | 0.048 |
| ask_qty_delta_10s_pct | 0.538 | 0.518 | 0.123 | 0.104 | 0.045 |
| mid_down_1s_bps | 0.518 | 0.535 | 0.120 | 0.283 | 0.050 |
| mid_down_30s_bps | 0.478 | 0.495 | 7.013 | 7.617 | 7.421 |
| bid_qty_delta_10s_pct | 0.389 | 0.441 | -0.342 | -0.166 | 0.103 |
| mid_down_3s_bps | 0.425 | 0.435 | 1.041 | 1.251 | 1.753 |
| imb_delta_10s | 0.400 | 0.433 | -0.251 | -0.167 | 0.023 |
| book_imbalance | 0.424 | 0.421 | -0.392 | -0.394 | -0.211 |
| top_qty_usd | 0.400 | 0.421 | 386658.127 | 387911.661 | 444666.315 |
| bid_depth_usd | 0.382 | 0.382 | 77345.337 | 73175.515 | 143741.892 |
| mid_down_15s_bps | 0.340 | 0.375 | 4.762 | 5.574 | 6.871 |
| mid_down_5s_bps | 0.354 | 0.365 | 1.617 | 1.725 | 3.442 |
| mid_down_10s_bps | 0.217 | 0.218 | 3.286 | 3.286 | 6.440 |

## Transparent Detector Score

`score = 0.65*mid_down_10s + 0.35*mid_down_5s + 4*max(0,-book_imbalance) - 2*spread_bps`

- Score AUC 500K: `0.311`
- Score AUC 1M: `0.322`

### Precision / Recall 500K

| Score quantile | Cutoff | Kept | Precision | Recall |
| ---: | ---: | ---: | ---: | ---: |
| 0.50 | 6.554 | 375 | 10.7% | 32.0% |
| 0.60 | 7.323 | 300 | 11.0% | 26.4% |
| 0.70 | 8.184 | 225 | 10.7% | 19.2% |
| 0.80 | 9.116 | 150 | 14.0% | 16.8% |
| 0.90 | 10.632 | 75 | 21.3% | 12.8% |

### Precision / Recall 1M

| Score quantile | Cutoff | Kept | Precision | Recall |
| ---: | ---: | ---: | ---: | ---: |
| 0.50 | 6.660 | 350 | 7.4% | 34.7% |
| 0.60 | 7.452 | 280 | 7.1% | 26.7% |
| 0.70 | 8.251 | 210 | 7.1% | 20.0% |
| 0.80 | 9.142 | 140 | 8.6% | 16.0% |
| 0.90 | 10.591 | 70 | 12.9% | 12.0% |

## Interpretation

- AUC near 0.5 means the feature cannot separate pre-liq positives from matched controls.
- Useful detector candidates should show AUC materially above 0.65 and precision lift at high score quantiles.
- This is not a trading rule yet; it is a detector research layer.
