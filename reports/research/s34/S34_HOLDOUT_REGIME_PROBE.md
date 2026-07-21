# S34 Holdout Regime Probe — 4-Test Suite

Generated: `2026-06-30T09:58:59.081149+00:00`
Status: `RESEARCH_ONLY_NO_LIVE_CHANGE`

Cal: 1404 events (2026-02-15T18:32:18Z to 2026-06-08T01:05:38Z)
Hold: 602 events (2026-06-08T01:24:48Z to 2026-06-29T08:28:10Z)

## Test 1: Propagation-Rate Regime Probe

| Metric | Cal | Hold | Delta |
| --- | ---: | ---: | ---: |
| ETH same-side prop rate (60min) | 0.803 | 0.867 | +0.064 |
| Cross-asset prop rate (60min) | 0.801 | 0.862 | +0.061 |
| Any prop rate (60min) | 0.921 | 0.93 | +0.009 |
| SYNC rate (BTC+SOL >=200K prior 10min) | 0.262 | 0.439 | +0.177 |
| Mean sync_k (K units) | 194.6 | 569.6 | +375.0 |

*Interpretation*: Higher prop_rate in hold -> more same-side follow-through -> k5=CLEAN (built on cal history) under-represents danger in hold

## Test 2: SYNC + k5=CLEAN Composite Gate Holdout

| Group | Cal N | Cal T3R | Cal median | Cal win | Hold N | Hold T3R | Hold median | Hold win | Hold maxL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sync_k5_clean | 36 | 766.9 | 35.6 | 0.778 | 39 | -3131.3 | -58.0 | 0.385 | -412.4 |
| sync_k5_danger_reverse | 72 | 2344.4 | 0.5 | 0.5 | 17 | None | -21.5 | 0.294 | -167.3 |
| idio_k5_clean | 104 | 4848.5 | 36.0 | 0.75 | 45 | -1798.1 | -13.5 | 0.378 | -364.0 |
| k5_clean_all | 140 | 6404.6 | 36.0 | 0.757 | 84 | -4563.9 | -23.9 | 0.381 | -412.4 |
| sync_all | 368 | -5854.0 | 7.0 | 0.514 | 264 | -8148.5 | -5.7 | 0.455 | -412.4 |

## Test 3: Frequency Expansion / Timing Holdout

### Exit-timing comparison (all events)

| Split | N | 2h T3R | 2h median | 2h win | 4h T3R | 4h median | TP300/SL150 T3R | TP300/SL150 median |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Cal | 1404 | -1384.8 | 5.7 | 0.519 | -1484.6 | 4.3 | 1579.7 | 0.2 |
| Hold | 602 | -8574.9 | -2.0 | 0.488 | -9086.4 | 4.5 | -2471.5 | 0.9 |

### Tags with cal T3R > 0, ranked by hold T3R

| Tag | Cal N | Cal T3R | Cal median | Hold N | Hold T3R | Hold median | Hold win |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BULL_PULLBACK | 142 | 1243.6 | 12.4 | 22 | 665.3 | 46.7 | 0.909 |
| VDEPTH_CORE | 227 | 858.3 | 21.7 | 79 | -260.4 | 10.5 | 0.532 |
| BID_DEPTH_CORE | 55 | 291.7 | -7.6 | 90 | -2796.6 | 8.9 | 0.544 |
| NEUTRAL_CONTEXT | 559 | 3003.1 | 9.3 | 233 | -6707.0 | -15.8 | 0.391 |

Hold-positive tags: `['BULL_PULLBACK']`

## Test 4: Monthly k5=CLEAN Stability

(*) = holdout period

| Month | All N | All med | All win | k5=CLEAN N | CLEAN med | CLEAN win | CLEAN T3R | k5=DANGER REV med | REV win |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-02 | 312 | -1.0 | 0.487 | 34 | 36.4 | 0.824 | 2065.0 | 7.5 | 0.537 |
| 2026-03 | 697 | 8.7 | 0.524 | 62 | 36.3 | 0.758 | 1670.9 | -8.0 | 0.452 |
| 2026-04 | 371 | 5.7 | 0.531 | 41 | 19.3 | 0.683 | 537.8 | -15.6 | 0.474 |
| 2026-06(*) | 626 | -1.9 | 0.494 | 87 | -18.8 | 0.402 | -4075.9 | -28.5 | 0.224 |

## Overall Verdict

- If Test 1 shows hold prop rate >> cal -> regime change explains holdout failure.
- If Test 2 SYNC+k5=CLEAN has hold T3R > 0 -> composite gate has residual signal.
- If Test 3 TP300/SL150 T3R > 0 in hold -> exit management recovers some edge.
- Test 4 monthly: which month did CLEAN structure break? That month = regime break.

All results RESEARCH_ONLY. No live/paper promotion without OOS+ permutation-null.
