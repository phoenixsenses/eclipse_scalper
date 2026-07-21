# S34 Navigation Advanced Results

Generated: `2026-06-29T09:09:42.150218+00:00`

Status: `RESEARCH_ONLY_NO_LIVE_CHANGE`. Navigation/paper-shadow research only. No live order/config changes.

## KNN Robustness

| Mode | CLEAN N | CLEAN Sum | CLEAN T3R | DANGER N | DANGER Sum | DANGER T3R | DANGER Reverse Sum | DANGER Reverse T3R |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| global_k10 | 198 | 5797.6 | 4842.1 | 655 | -11543.0 | -13391.2 | 4993.0 | 3799.8 |
| route_k10 | 118 | -1762.6 | -2689.9 | 648 | -1882.0 | -3730.2 | -4598.0 | -5834.0 |
| global_k20 | 176 | 2731.9 | 1899.1 | 1094 | -10573.7 | -12588.4 | -366.3 | -1609.6 |
| route_k20 | 80 | -491.2 | -1393.5 | 1200 | -3064.0 | -5078.7 | -8936.0 | -10179.0 |
| global_k30 | 111 | 891.6 | 317.8 | 1401 | -9628.1 | -11642.8 | -4381.9 | -5625.2 |
| route_k30 | 45 | -1042.2 | -1630.8 | 1561 | -2362.8 | -4377.5 | -13247.2 | -14490.2 |
| global_k50 | 45 | 56.3 | -572.8 | 1762 | -5495.1 | -7509.8 | -12124.9 | -13368.2 |
| route_k50 | 21 | -276.3 | -515.5 | 1844 | -5096.8 | -7111.5 | -13343.2 | -14586.5 |

## DANGER Reverse Test (Global k20)

- DANGER actual: `{'n': 1094, 'sum_bps': -10573.7, 'mean_bps': -9.7, 'median_bps': -0.2, 'win_rate': 0.499, 'max_loss_bps': -455.2, 'tail_lte_minus100_n': 190, 'tail_lte_minus150_n': 128, 'tail_lte_minus300_n': 32, 't3r_bps': -12588.4}`
- DANGER reverse: `{'n': 1094, 'sum_bps': -366.3, 'mean_bps': -0.3, 'median_bps': -9.8, 'win_rate': 0.434, 'max_loss_bps': -728.1, 'tail_lte_minus100_n': 173, 'tail_lte_minus150_n': 89, 'tail_lte_minus300_n': 9, 't3r_bps': -1609.6}`

## v0.3 Shadow Ledger

- Ledger: `D:\eclipse_scalper\reports\research\s34\S34_V03_SHADOW_LEDGER.jsonl`
| Mode | N | Sum | Observed End | -300 End | -507 End |
| --- | ---: | ---: | ---: | ---: | ---: |
| V03_15X | 11 | 1780.1 | 349.366 | 192.151 | 83.673 |
| V03_18X | 11 | 1780.1 | 516.51 | 237.594 | 45.143 |

## Latest Navigation Card

- Route: `ETHUSDT_SELL_50000`
- Tags: `RISK_OFF_REBOUND, VDEPTH_DANGER_LOW, BID_DEPTH_OK, BID_DEPTH_CORE, EXIT_2H_ACTUAL_BETTER, TAIL_HIGH_OR_UNKNOWN, SIZE_34X_FRAGILE`
- KNN route k20: `MIXED`
- Tail risk: `MIXED_OR_UNKNOWN`
- Exit preference: `EXIT_2H`
- Sizing preference: `SIZE_34X_FRAGILE`

## Pattern Ranker

| Verdict | Score | Combo | N | CleanFrac | 2h Sum | 2h T3R | 4hTP Sum | 4hTP T3R | Tail<=150 |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CONTEXT_ONLY | 1031.1 | BULL_PULLBACK+VDEPTH_CORE+BID_DEPTH_THIN | 20 | 0.1 | 1479.3 | 479.4 | 1636.7 | 751.7 | 2 |
| CONTEXT_ONLY | -664.9 | BULL_PULLBACK+VDEPTH_DANGER_HIGH+BID_DEPTH_THIN | 28 | 0.071 | 1117.7 | 66.0 | 654.1 | -230.9 | 5 |
| CONTEXT_ONLY | -860.5 | RISK_OFF_REBOUND+VDEPTH_CORE+BID_DEPTH_THIN | 133 | 0.09 | 105.9 | -502.6 | 1727.1 | 842.1 | 12 |
| CONTEXT_ONLY | -1448.1 | NEUTRAL_CONTEXT+VDEPTH_DANGER_LOW+BID_DEPTH_OK | 64 | 0.109 | 573.1 | -91.3 | -471.8 | -1356.8 | 0 |
| CONTEXT_ONLY | -1589.5 | NEUTRAL_CONTEXT+VDEPTH_CORE+BID_DEPTH_THIN | 97 | 0.062 | 378.3 | -565.1 | 560.6 | -324.4 | 7 |
| CONTEXT_ONLY | -2120.1 | RISK_OFF_REBOUND+VDEPTH_DANGER_LOW+BID_DEPTH_OK+BID_DEPTH_CORE | 73 | 0.027 | -1109.6 | -1734.2 | 1299.1 | 414.1 | 8 |
| CONTEXT_ONLY | -2222.0 | BULL_PULLBACK+VDEPTH_DANGER_LOW+BID_DEPTH_THIN | 98 | 0.224 | 359.7 | -489.0 | -148.0 | -1033.0 | 7 |
| CONTEXT_ONLY | -2228.2 | RISK_OFF_REBOUND+VDEPTH_DANGER_LOW+BID_DEPTH_OK+BID_DEPTH_HEAVY | 28 | 0.179 | -321.3 | -810.1 | -814.7 | -1218.1 | 2 |
| CONTEXT_ONLY | -2394.8 | NEUTRAL_CONTEXT+VDEPTH_DANGER_LOW+BID_DEPTH_OK+BID_DEPTH_HEAVY | 36 | 0.056 | -435.9 | -787.1 | -1096.2 | -1607.7 | 0 |
| CONTEXT_ONLY | -2846.8 | NEUTRAL_CONTEXT+VDEPTH_DANGER_LOW+BID_DEPTH_OK+BID_DEPTH_CORE | 37 | 0.135 | -888.8 | -1302.3 | -734.8 | -1144.5 | 4 |
| CONTEXT_ONLY | -3366.1 | RISK_OFF_REBOUND+VDEPTH_DANGER_HIGH+BID_DEPTH_THIN | 151 | 0.053 | -332.8 | -963.3 | 82.2 | -802.8 | 16 |
| CONTEXT_ONLY | -3470.5 | NEUTRAL_CONTEXT+VDEPTH_DANGER_LOW+BID_DEPTH_THIN | 466 | 0.071 | 892.7 | -365.9 | 180.4 | -704.6 | 24 |
| CONTEXT_ONLY | -3506.0 | NEUTRAL_CONTEXT+VDEPTH_DANGER_HIGH+BID_DEPTH_THIN | 56 | 0.107 | -1081.9 | -1881.2 | 160.2 | -724.8 | 9 |
| CONTEXT_ONLY | -7001.1 | RISK_OFF_REBOUND+VDEPTH_DANGER_LOW+BID_DEPTH_OK | 98 | 0.143 | -2721.5 | -3503.3 | -1212.8 | -2097.8 | 14 |
| CONTEXT_ONLY | -14869.6 | RISK_OFF_REBOUND+VDEPTH_DANGER_LOW+BID_DEPTH_THIN | 541 | 0.076 | -5363.8 | -7378.5 | -1006.1 | -1891.1 | 56 |
