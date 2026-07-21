# S34 Pre-Liq Detector V2

Generated: `2026-06-27T13:45:34.144512+00:00`

Research only. V2 adds ETH taker flow, recent mini-liquidations, and BTC/SOL context to the v1 book-state detector.

Positive counts: 500K=102, 1M=61. Controls=510.

Control definition: `mid_down_10s>=5bps and spread<=1bps, excluding +/-900s around ETH SELL liquidation clusters`.

## Temporal Split Detector Results

| Threshold | Train N | Test N | Test pos | Train AUC | Test AUC | Q80 precision | Q80 lift | Q90 precision | Q90 lift |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 500K | 428 | 184 | 27 | 0.802 | 0.783 | 40.5% | 25.9% | 52.6% | 38.0% |
| 1000K | 399 | 172 | 15 | 0.795 | 0.791 | 25.7% | 17.0% | 27.8% | 19.1% |

## Selected Features Per Threshold

### 500K

| Feature | Train AUC | Direction |
| --- | ---: | ---: |
| mid_down_10s_bps | 0.184 | lower |
| btc_ret_5s_bps | 0.702 | higher |
| btc_liq_imbalance_120s | 0.685 | higher |
| sol_ret_5s_bps | 0.673 | higher |
| sol_liq_imbalance_120s | 0.666 | higher |
| mid_down_15s_bps | 0.337 | lower |
| btc_liq_imbalance_60s | 0.660 | higher |
| btc_ret_10s_bps | 0.646 | higher |

### 1000K

| Feature | Train AUC | Direction |
| --- | ---: | ---: |
| mid_down_10s_bps | 0.199 | lower |
| sol_liq_imbalance_120s | 0.686 | higher |
| btc_ret_5s_bps | 0.681 | higher |
| btc_liq_imbalance_120s | 0.673 | higher |
| sol_liq_imbalance_60s | 0.669 | higher |
| sol_ret_5s_bps | 0.668 | higher |
| btc_liq_imbalance_60s | 0.661 | higher |
| sol_liq_imbalance_30s | 0.659 | higher |

## Top Single-Feature Separators

| Feature | Best AUC 500K | Best AUC 1M | Pos500 median | Control median |
| --- | ---: | ---: | ---: | ---: |
| mid_down_10s_bps | 0.794 | 0.790 | 3.401 | 6.614 |
| sol_liq_imbalance_120s | 0.673 | 0.700 | 0.000 | 0.000 |
| btc_ret_5s_bps | 0.687 | 0.680 | -0.750 | -1.875 |
| btc_liq_imbalance_120s | 0.676 | 0.679 | 0.000 | 0.000 |
| sol_ret_5s_bps | 0.673 | 0.675 | -1.323 | -3.217 |
| sol_liq_imbalance_60s | 0.637 | 0.660 | 0.000 | 0.000 |
| mid_down_15s_bps | 0.660 | 0.619 | 4.827 | 6.934 |
| sol_liq_sell_count_120s | 0.623 | 0.659 | 0.000 | 0.000 |
| eth_taker_imbalance_10s | 0.633 | 0.656 | 0.321 | 0.480 |
| sol_liq_sell_notional_120s | 0.622 | 0.656 | 0.000 | 0.000 |
| mid_down_5s_bps | 0.655 | 0.640 | 1.759 | 3.596 |
| btc_liq_imbalance_60s | 0.648 | 0.650 | 0.000 | 0.000 |
| btc_ret_10s_bps | 0.645 | 0.623 | -1.909 | -3.255 |
| sol_ret_10s_bps | 0.640 | 0.626 | -2.668 | -5.273 |
| sol_liq_imbalance_30s | 0.614 | 0.638 | 0.000 | 0.000 |
| sol_liq_sell_count_60s | 0.603 | 0.637 | 0.000 | 0.000 |
| eth_liq_imbalance_120s | 0.620 | 0.637 | 0.000 | 0.000 |
| btc_liq_imbalance_30s | 0.637 | 0.637 | 0.000 | 0.000 |
| btc_liq_sell_count_120s | 0.625 | 0.637 | 0.500 | 0.000 |
| sol_liq_sell_notional_60s | 0.602 | 0.635 | 0.000 | 0.000 |

## Interpretation

- V1 showed top-of-book alone could not detect imminent large liquidation.
- V2 tests whether taker flow, mini-liquidation context, and cross-symbol pressure add real temporal-test separation.
- This report does not change runner rules, live execution, or config.
