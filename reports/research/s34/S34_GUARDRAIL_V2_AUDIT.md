# S34 Guardrail V2 Audit

Generated at: `2026-06-23T14:33:41.411568+00:00`

Scope: closed ledger trades only. This is research. No runner/config/live rule changed.

## Feature Inventory

Closed rows: `71`

Available signal feature counts:

| Feature | Rows |
| --- | --- |
| btc_confirm_symbol | 40 |
| btc_pre_end_mark_ts_ms | 2 |
| btc_pre_min_return_bps | 2 |
| btc_pre_return_bps | 2 |
| btc_pre_start_mark_ts_ms | 2 |
| btc_pre_window_sec | 40 |
| bucket | 71 |
| cluster_duration_sec | 33 |
| cluster_end_ts_ms | 33 |
| cluster_max_single_liq_share | 33 |
| cluster_shape_label | 33 |
| cluster_start_ts_ms | 33 |
| entry_delay_sec | 40 |
| entry_fill | 70 |
| entry_price | 71 |
| entry_reference_price | 70 |
| entry_ts_ms | 40 |
| entry_ts_utc | 40 |
| fill_error | 70 |
| liq_count | 71 |
| liq_max_notional | 71 |
| liq_max_price | 71 |
| liq_total_notional | 71 |
| mark_ts_ms | 71 |
| ts_ms | 71 |
| ts_utc | 71 |

Desired fields missing from `features_json`: day_trend_bps, day_range_bps, max_single_liq_share, intensity_per_sec, inter_cluster_gap_sec

## Warning Winner vs Warning Loser

| Bucket | N | Cum | Median | WR % | Median Min Exp | Median Dispersion |
| --- | --- | --- | --- | --- | --- | --- |
| warning_all | 27 | 93.0 | -11.77 | 37.04 | -20.17 | 17.48 |
| warning_winners | 10 | 766.79 | 77.34 | 100.0 | -38.05 | 27.19 |
| warning_losers | 17 | -673.79 | -47.76 | 0.0 | -11.77 | 1.95 |

### Warning By Rule

| Rule | N | Cum | Mean | Median | WR % |
| --- | --- | --- | --- | --- | --- |
| ETH_BUY_LIQ_LONG_50K_TP120_SL40_BE30 | 22 | 18.57 | 0.84 | -27.65 | 31.82 |
| ETH_BUY_LIQ_LONG_200K_TP60_SL40_BE30 | 3 | 43.99 | 14.66 | 47.69 | 66.67 |
| BTC_BUY_LIQ_LONG_1M_DISTRIBUTED_TP60_SL30_BE30 | 1 | 55.02 | 55.02 | 55.02 | 100.0 |
| ETH_BUY_LIQ_LONG_200K_BTC_PRE15_TP120_SL40_BE30_DELAY60 | 1 | -24.58 | -24.58 | -24.58 | 0.0 |

### Warning By Cluster Notional

| Cluster | N | Cum | Mean | Median | WR % |
| --- | --- | --- | --- | --- | --- |
| 100K-200K | 8 | -318.37 | -39.8 | -47.89 | 0.0 |
| <100K | 7 | -79.59 | -11.37 | -43.52 | 28.57 |
| 200K-500K | 5 | 117.76 | 23.55 | 31.09 | 60.0 |
| >=1M | 5 | 225.84 | 45.17 | 55.02 | 60.0 |
| 500K-1M | 2 | 147.36 | 73.68 | 73.68 | 100.0 |

### Warning By Min Expected

| Min Expected | N | Cum | Mean | Median | WR % |
| --- | --- | --- | --- | --- | --- |
| -30..0 | 15 | -31.5 | -2.1 | -24.58 | 33.33 |
| -50..-30 | 11 | 114.1 | 10.37 | -11.77 | 36.36 |
| <=-50 | 1 | 10.4 | 10.4 | 10.4 | 100.0 |

## Candidate V2 Hard Block (Research Only)

Definition: `warning AND 100K <= cluster_notional < 200K`

| Scenario | N | Cum | Mean | Median | WR % | Extra |
| --- | --- | --- | --- | --- | --- | --- |
| baseline | 71 | 1097.71 | 15.46 | 31.09 | 52.11 |  |
| blocked_bucket | 8 | -318.37 | -39.8 | -47.89 | 0.0 |  |
| kept_after_block | 63 | 1416.08 | 22.48 | 47.69 | 58.73 | delta 318.37 |

Blocked examples:

| Trade | Rule | Exit | Net | Cluster | StrongNeg | MinExp |
| --- | --- | --- | --- | --- | --- | --- |
| P056 | ETH_BUY_LIQ_LONG_50K_TP120_SL40_BE30 | SL | -53.45 | 146467.59 | 0 | -8.850088427874663 |
| P169 | ETH_BUY_LIQ_LONG_50K_TP120_SL40_BE30 | SL | -51.87 | 101877.85 | 3 | -45.82086924288744 |
| P419 | ETH_BUY_LIQ_LONG_50K_TP120_SL40_BE30 | SL | -49.36 | 151053.75 | 1 | -47.756366351064926 |
| P416 | ETH_BUY_LIQ_LONG_50K_TP120_SL40_BE30 | SL | -48.03 | 154940.25 | 3 | -45.82086924288744 |
| P063 | ETH_BUY_LIQ_LONG_50K_TP120_SL40_BE30 | SL | -47.76 | 153126.29 | 0 | -9.337548076107211 |
| P149 | ETH_BUY_LIQ_LONG_50K_TP120_SL40_BE30 | SL | -46.3 | 185562.3 | 2 | -47.756366351064926 |
| P394 | ETH_BUY_LIQ_LONG_50K_TP120_SL40_BE30 | BE | -11.77 | 177012.84 | 2 | -46.29973899129289 |
| P058 | ETH_BUY_LIQ_LONG_50K_TP120_SL40_BE30 | BE | -9.83 | 135373.6 | 1 | -31.150256507310097 |

## Read

V2 is not promoted. The audit identifies whether warning can be split into a narrower hard-block candidate. Any useful candidate must be forward-tested as a shadow rule before becoming a live filter.
