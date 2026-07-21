# S34 Navigation Bridge

Generated: `2026-06-29T08:54:53.855050+00:00`

Status: `NAVIGATION_RESEARCH_ONLY_NO_LIVE_CHANGE`. Navigation bridge labels context only. It does not authorize trades.

Definition: ETH SELL liquidation navigation universe: thresholds 50K/100K/200K, vdepth>=5, deterministic tags, mark-based 2h/4h labels.

Events: `2006`

## Tag Distribution

| Tag | Count |
| --- | ---: |
| SIZE_34X_FRAGILE | 1983 |
| TAIL_HIGH_OR_UNKNOWN | 1983 |
| BID_DEPTH_THIN | 1590 |
| VDEPTH_DANGER_LOW | 1450 |
| RISK_OFF_REBOUND | 1050 |
| EXIT_4H_ACTUAL_BETTER | 1038 |
| EXIT_2H_ACTUAL_BETTER | 968 |
| NEUTRAL_CONTEXT | 792 |
| BID_DEPTH_OK | 416 |
| VDEPTH_CORE | 306 |
| TAIL_REALIZED | 296 |
| VDEPTH_DANGER_HIGH | 250 |
| BULL_PULLBACK | 164 |
| BID_DEPTH_CORE | 145 |
| BID_DEPTH_HEAVY | 80 |
| SIZE_15X_STABLE | 23 |
| TAIL_LOW_CONTEXT | 23 |

## Tail-Low Validation

| Bucket | N | Sum | Median | Win | <=-100 | <=-150 | <=-300 | Max loss | T3R |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TAIL_LOW_CONTEXT | 23 | 482.7 | 23.0 | 0.609 | 3 | 2 | 1 | -392.6 | -226.0 |
| TAIL_HIGH_OR_UNKNOWN | 1983 | -7569.7 | 2.6 | 0.509 | 293 | 170 | 35 | -455.2 | -9584.4 |

## Tail-Low By Threshold

| Bucket | N | Sum | Median | Win | <=-100 | <=-150 | <=-300 | Max loss | T3R |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TAIL_LOW_thr100000 | 4 | -444.0 | -50.8 | 0.25 | 1 | 1 | 1 | -392.6 | -392.6 |
| TAIL_LOW_thr200000 | 11 | 961.6 | 50.2 | 0.909 | 0 | 0 | 0 | -13.7 | 316.6 |
| TAIL_LOW_thr50000 | 8 | -34.9 | -46.6 | 0.375 | 2 | 1 | 0 | -152.4 | -448.4 |

## Exact v0.2 Route Approximation

| Exit | N | Sum | Median | Win | <=-100 | <=-150 | <=-300 | Max loss | T3R |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2h | 11 | 961.6 | 50.2 | 0.909 | 0 | 0 | 0 | -13.7 | 316.6 |
| 4h | 11 | 1413.4 | 104.5 | 0.818 | 0 | 0 | 0 | -12.5 | 586.5 |
| tp300_sl150_4h | 11 | 1067.8 | 104.5 | 0.727 | 2 | 2 | 0 | -155.0 | 206.6 |

## Exit Preference

| Exit | N | Sum | Median | Win | Max loss | T3R |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2h | 2006 | -7087.0 | 3.2 | 0.51 | -455.2 | -9101.7 |
| 4h | 2006 | -7013.6 | 4.4 | 0.524 | -538.7 | -8858.2 |
| tp300_sl150_4h | 2006 | 878.2 | 0.4 | 0.502 | -155.0 | -6.8 |

## Pattern Candidates

| Verdict | Combo | N | 2h Sum | 2h T3R | 4hTP Sum | 4hTP T3R | Tail<=150 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CONTEXT_ONLY | RISK_OFF_REBOUND+VDEPTH_CORE+BID_DEPTH_THIN | 133 | 105.9 | -502.6 | 1727.1 | 842.1 | 12 |
| CONTEXT_ONLY | BULL_PULLBACK+VDEPTH_CORE+BID_DEPTH_THIN | 20 | 1479.3 | 479.4 | 1636.7 | 751.7 | 2 |
| CONTEXT_ONLY | RISK_OFF_REBOUND+VDEPTH_DANGER_LOW+BID_DEPTH_OK+BID_DEPTH_CORE | 73 | -1109.6 | -1734.2 | 1299.1 | 414.1 | 8 |
| CONTEXT_ONLY | RISK_OFF_REBOUND+VDEPTH_CORE+BID_DEPTH_OK+BID_DEPTH_CORE | 11 | 362.4 | -58.5 | 948.8 | 203.3 | 1 |
| CONTEXT_ONLY | BULL_PULLBACK+VDEPTH_CORE+BID_DEPTH_OK+BID_DEPTH_CORE | 5 | 196.3 | -20.6 | 34.3 | -30.2 | 0 |
| CONTEXT_ONLY | BULL_PULLBACK+VDEPTH_DANGER_LOW+BID_DEPTH_OK+BID_DEPTH_HEAVY | 5 | 306.1 | 54.3 | 56.6 | -73.0 | 0 |
| CONTEXT_ONLY | NEUTRAL_CONTEXT+VDEPTH_CORE+BID_DEPTH_OK+BID_DEPTH_CORE | 6 | -23.2 | -319.1 | 73.0 | -212.1 | 1 |
| CONTEXT_ONLY | BULL_PULLBACK+VDEPTH_DANGER_HIGH+BID_DEPTH_THIN | 28 | 1117.7 | 66.0 | 654.1 | -230.9 | 5 |
| CONTEXT_ONLY | NEUTRAL_CONTEXT+VDEPTH_DANGER_HIGH+BID_DEPTH_OK+BID_DEPTH_CORE | 5 | -481.4 | -278.1 | -580.0 | -310.0 | 1 |
| CONTEXT_ONLY | NEUTRAL_CONTEXT+VDEPTH_CORE+BID_DEPTH_THIN | 97 | 378.3 | -565.1 | 560.6 | -324.4 | 7 |
| CONTEXT_ONLY | NEUTRAL_CONTEXT+VDEPTH_CORE+BID_DEPTH_OK+BID_DEPTH_HEAVY | 9 | -35.9 | -198.5 | -428.9 | -671.7 | 0 |
| CONTEXT_ONLY | NEUTRAL_CONTEXT+VDEPTH_CORE+BID_DEPTH_OK | 13 | -268.9 | -411.2 | -348.1 | -685.9 | 0 |
| CONTEXT_ONLY | NEUTRAL_CONTEXT+VDEPTH_DANGER_LOW+BID_DEPTH_THIN | 466 | 892.7 | -365.9 | 180.4 | -704.6 | 24 |
| CONTEXT_ONLY | NEUTRAL_CONTEXT+VDEPTH_DANGER_HIGH+BID_DEPTH_THIN | 56 | -1081.9 | -1881.2 | 160.2 | -724.8 | 9 |
| CONTEXT_ONLY | RISK_OFF_REBOUND+VDEPTH_CORE+BID_DEPTH_OK | 10 | -79.2 | -594.7 | -175.9 | -750.4 | 1 |
| CONTEXT_ONLY | RISK_OFF_REBOUND+VDEPTH_DANGER_HIGH+BID_DEPTH_THIN | 151 | -332.8 | -963.3 | 82.2 | -802.8 | 16 |
| CONTEXT_ONLY | BULL_PULLBACK+VDEPTH_DANGER_LOW+BID_DEPTH_THIN | 98 | 359.7 | -489.0 | -148.0 | -1033.0 | 7 |
| CONTEXT_ONLY | NEUTRAL_CONTEXT+VDEPTH_DANGER_LOW+BID_DEPTH_OK+BID_DEPTH_CORE | 37 | -888.8 | -1302.3 | -734.8 | -1144.5 | 4 |
| CONTEXT_ONLY | RISK_OFF_REBOUND+VDEPTH_DANGER_LOW+BID_DEPTH_OK+BID_DEPTH_HEAVY | 28 | -321.3 | -810.1 | -814.7 | -1218.1 | 2 |
| CONTEXT_ONLY | NEUTRAL_CONTEXT+VDEPTH_DANGER_LOW+BID_DEPTH_OK | 64 | 573.1 | -91.3 | -471.8 | -1356.8 | 0 |
