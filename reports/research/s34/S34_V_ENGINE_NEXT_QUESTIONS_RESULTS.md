# S34 V Engine Next Questions Results

Generated: `2026-06-29T08:34:34.286374+00:00`

Status: `RESEARCH_ONLY_NO_LIVE_CHANGE`. No live executor, leverage, size, order logic, or .env changes.

## Fine Sizing Frontier

Max ratio that survives appended -507 bps: `19`

| Ratio | Observed End | -300 End | -507 End | Ruin Tail | Survive -507 |
| ---: | ---: | ---: | ---: | ---: | --- |
| 10 | 94.657 | 66.26 | 46.666 | -1000.0 | True |
| 11 | 103.702 | 69.48 | 45.867 | -909.1 | True |
| 12 | 113.465 | 72.618 | 44.433 | -833.3 | True |
| 13 | 123.994 | 75.637 | 42.27 | -769.2 | True |
| 14 | 135.339 | 78.496 | 39.275 | -714.3 | True |
| 15 | 147.549 | 81.152 | 35.338 | -666.7 | True |
| 16 | 160.682 | 83.554 | 30.337 | -625.0 | True |
| 17 | 174.793 | 85.648 | 24.139 | -588.2 | True |
| 18 | 189.943 | 87.374 | 16.601 | -555.6 | True |
| 19 | 206.195 | 88.664 | 7.567 | -526.3 | True |
| 20 | 223.616 | 89.446 | -3.131 | -500.0 | False |

## Exit x Sizing Matrix: Top Robust (-507 survives)

| Variant | Ratio | Observed End | -300 End | -507 End | Sum bps | T3R |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| tp300_sl150_4h | 18 | 516.51 | 237.594 | 45.143 | 1780.1 | 895.1 |
| fixed_4h | 18 | 483.97 | 222.626 | 42.299 | 1740.8 | 822.6 |
| tp300_sl150_4h | 15 | 349.366 | 192.151 | 83.673 | 1780.1 | 895.1 |
| fixed_4h | 15 | 330.491 | 181.77 | 79.153 | 1740.8 | 822.6 |
| trail100_after150_4h | 18 | 301.213 | 138.558 | 26.326 | 1361.0 | 744.5 |
| fixed_8h | 18 | 245.804 | 113.07 | 21.483 | 1406.4 | 542.2 |
| tp300_sl150_4h | 12 | 231.74 | 148.314 | 90.75 | 1780.1 | 895.1 |
| fixed_4h | 12 | 221.45 | 141.728 | 86.72 | 1740.8 | 822.6 |
| trail100_after150_4h | 15 | 217.987 | 119.893 | 52.208 | 1361.0 | 744.5 |
| fixed_2h | 18 | 192.124 | 88.377 | 16.792 | 1089.9 | 406.3 |

## Exit x Sizing Matrix: Top Growth

| Variant | Ratio | Observed End | -300 End | -507 End | Survive -507 |
| --- | ---: | ---: | ---: | ---: | --- |
| tp300_sl150_4h | 34 | 3199.57 | -63.991 | -2315.848 | False |
| fixed_4h | 34 | 2874.83 | -57.497 | -2080.802 | False |
| trail100_after150_4h | 34 | 1417.445 | -28.349 | -1025.947 | False |
| tp300_sl150_4h | 20 | 663.718 | 265.487 | -9.292 | False |
| fixed_2h | 34 | 647.834 | -12.957 | -468.902 | False |
| sl150_2h | 34 | 647.834 | -12.957 | -468.902 | False |
| fixed_4h | 20 | 618.162 | 247.265 | -8.654 | False |
| partial_tp150_2h | 34 | 588.227 | -11.765 | -425.759 | False |
| fixed_8h | 34 | 556.708 | -11.134 | -402.945 | False |
| tp300_sl150_4h | 18 | 516.51 | 237.594 | 45.143 | True |

## Tail Neighbor Analysis

ETH SELL 200K, vdepth 20-50, prior4h<-20; classify v0.2 pass vs near-miss filters. Outcome is anchor mark 2h net bps, not maker fill.

| Bucket | N | Sum bps | Median | Win | <=-100 | <=-150 | <=-300 | Max loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| IN_V02_ANCHOR_CF | 11 | 959.5 | 50.2 | 0.909 | 0 | 0 | 0 | -12.6 |
| NEAR_MISS_BID_DEPTH | 37 | -1003.8 | -3.9 | 0.432 | 6 | 4 | 0 | -272.8 |
| NEAR_MISS_PRIOR4H | 6 | 134.9 | 39.2 | 0.667 | 0 | 0 | 0 | -40.1 |
| NEAR_MISS_VDEPTH | 78 | -1926.9 | -11.2 | 0.436 | 18 | 8 | 3 | -414.8 |
