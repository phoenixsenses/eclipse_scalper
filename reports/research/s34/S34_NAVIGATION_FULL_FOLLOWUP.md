# S34 Navigation Full Follow-Up

Generated: `2026-06-29T09:18:16.179926+00:00`

Status: `RESEARCH_ONLY_NO_LIVE_CHANGE`. Navigation/paper-shadow research only. No live order/config changes.

## DANGER Reverse Stability

| K | Horizon | DANGER N | Normal Sum | Normal T3R | Reverse Sum | Reverse T3R | Reverse Tail<=150 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| k5 | 30m | 314 | -6115.3 | -6572.9 | 2975.3 | 1950.6 | 3 |
| k5 | 1h | 314 | -6821.6 | -7738.5 | 3681.6 | 2605.2 | 8 |
| k5 | 2h | 314 | -11607.8 | -13127.5 | 8467.8 | 7271.4 | 22 |
| k5 | 4h | 314 | -11525.9 | -12826.8 | 8385.9 | 6802.6 | 39 |
| k8 | 30m | 530 | -6958.3 | -7417.8 | 1658.3 | 633.6 | 5 |
| k8 | 1h | 530 | -7272.7 | -8266.1 | 1972.7 | 896.3 | 14 |
| k8 | 2h | 530 | -12826.6 | -14346.3 | 7526.6 | 6330.2 | 37 |
| k8 | 4h | 530 | -13199.4 | -14557.1 | 7899.4 | 6316.0 | 60 |
| k10 | 30m | 655 | -7757.8 | -8327.3 | 1207.8 | 177.4 | 8 |
| k10 | 1h | 655 | -7180.1 | -8508.4 | 630.1 | -446.2 | 19 |
| k10 | 2h | 655 | -11307.8 | -13156.6 | 4757.8 | 3561.3 | 50 |
| k10 | 4h | 655 | -11650.4 | -13184.2 | 5100.4 | 3517.1 | 80 |
| k12 | 30m | 754 | -8083.3 | -8675.4 | 543.3 | -487.1 | 9 |
| k12 | 1h | 754 | -7301.0 | -8629.2 | -239.0 | -1315.4 | 21 |
| k12 | 2h | 754 | -11884.7 | -13733.5 | 4344.7 | 3098.0 | 56 |
| k12 | 4h | 754 | -11661.7 | -13262.7 | 4121.7 | 2538.4 | 94 |
| k15 | 30m | 891 | -8623.7 | -9235.2 | -286.3 | -1316.7 | 12 |
| k15 | 1h | 891 | -7199.5 | -8534.0 | -1710.5 | -2803.6 | 26 |
| k15 | 2h | 891 | -11972.9 | -13821.8 | 3062.9 | 1816.2 | 66 |
| k15 | 4h | 891 | -12419.7 | -14090.1 | 3509.7 | 1926.4 | 109 |
| k20 | 30m | 1094 | -9510.8 | -10141.9 | -1429.2 | -2848.7 | 15 |
| k20 | 1h | 1094 | -7209.7 | -8566.8 | -3730.3 | -4903.8 | 42 |
| k20 | 2h | 1094 | -10258.2 | -12273.5 | -681.8 | -1929.5 | 89 |
| k20 | 4h | 1094 | -10994.3 | -12706.3 | 54.3 | -1529.0 | 145 |

## KNN CLEAN Strictness (k20)

| Strictness | CLEAN N | CLEAN Sum | CLEAN T3R | CLEAN Tail<=150 | DANGER N | DANGER Sum |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| loose | 237 | 5329.8 | 4329.9 | 10 | 1085 | -10529.7 |
| base | 176 | 2731.9 | 1899.1 | 7 | 1094 | -10573.7 |
| strict | 95 | 602.7 | 61.9 | 5 | 1094 | -10573.7 |
| ultra | 42 | 228.7 | -255.3 | 2 | 1094 | -10573.7 |

## v0.3 Shadow Readout

| Mode | N | Sum | Median | T3R | Max loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| V03_15X | 11 | 1780.1 | 165.6 | 895.1 | 7.3 |
| V03_18X | 11 | 1780.1 | 165.6 | 895.1 | 7.3 |

## Bull Thin-Depth Tail Anatomy

- N: `20`; tails: `2`
- Overall 2h: `{'n': 20, 'sum_bps': 1479.3, 'mean_bps': 74.0, 'median_bps': 50.5, 'win_rate': 0.7, 'max_loss_bps': -203.1, 'tail_lte_minus100_n': 2, 'tail_lte_minus150_n': 2, 'tail_lte_minus300_n': 0, 't3r_bps': 479.4}`
- Tail profile: `{'avg_threshold': 125000.0, 'avg_vdepth': 39.0, 'avg_bid_depth': 16494.5, 'avg_book_imbalance': -0.8, 'avg_btc4h': 121.2, 'avg_eth1h': 101.5}`
- Winner profile: `{'avg_threshold': 103571.4, 'avg_vdepth': 33.0, 'avg_bid_depth': 0.0, 'avg_book_imbalance': None, 'avg_btc4h': 250.1, 'avg_eth1h': 232.3}`

## BUY500 Fade SHORT

| Cell | N | Sum | Median | Win | Tail<=150 | Max loss | T3R |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| horizon_30m | 194 | -840.5 | 7.2 | 0.536 | 11 | -517.5 | -1368.9 |
| horizon_1h | 194 | 566.1 | 16.3 | 0.572 | 16 | -302.8 | -244.9 |
| horizon_2h | 194 | 1869.2 | 27.9 | 0.582 | 19 | -390.5 | 678.5 |
| horizon_4h | 194 | 1389.4 | 17.3 | 0.593 | 28 | -588.7 | -120.0 |
| v5_20_2h | 48 | 232.5 | 27.1 | 0.604 | 5 | -304.7 | -515.4 |
| v20_40_2h | 75 | 56.4 | 16.9 | 0.52 | 8 | -390.5 | -807.8 |
| v40_plus_2h | 71 | 1580.3 | 39.8 | 0.634 | 6 | -382.2 | 480.0 |

## Pattern Ranker Sweep

| Criteria | Combo | N | Sum | T3R | Tail<=150 |
| --- | --- | ---: | ---: | ---: | ---: |
| lenient | BULL_PULLBACK+VDEPTH_CORE+BID_DEPTH_THIN | 20 | 1479.3 | 479.4 | 2 |
| lenient | BULL_PULLBACK+VDEPTH_DANGER_HIGH+BID_DEPTH_THIN | 28 | 1117.7 | 66.0 | 5 |
| base | BULL_PULLBACK+VDEPTH_CORE+BID_DEPTH_THIN | 20 | 1479.3 | 479.4 | 2 |

## Navigation Card Ledger

- Path: `D:\eclipse_scalper\reports\research\s34\S34_NAVIGATION_CARD_LEDGER.jsonl`
- Rows: `2006`
- Latest: `{'event_id': 'ETHUSDT_SELL_50000_1782721690110', 'signal_utc': '2026-06-29T08:28:10.110000+00:00', 'route': 'ETHUSDT_SELL_50000', 'tags': ['RISK_OFF_REBOUND', 'VDEPTH_DANGER_LOW', 'BID_DEPTH_OK', 'BID_DEPTH_CORE', 'EXIT_2H_ACTUAL_BETTER', 'TAIL_HIGH_OR_UNKNOWN', 'SIZE_34X_FRAGILE'], 'knn_global_k20': 'MIXED', 'tail_risk': 'MIXED', 'actual_2h_bps': 11.7, 'reverse_2h_bps': -21.7, 'neighbor_median_bps': -14.9, 'neighbor_t3r_bps': -399.3}`
