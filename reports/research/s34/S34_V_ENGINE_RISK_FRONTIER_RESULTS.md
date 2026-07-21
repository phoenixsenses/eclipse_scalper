# S34 V Engine Risk Frontier Results

Generated: `2026-06-29T08:29:12.726353+00:00`

Status: `RESEARCH_ONLY_NO_LIVE_CHANGE`. No live executor, leverage, size, order logic, or .env changes.

## CURRENT_ENV Tail Break-Even

- Observed 11-trade CURRENT_ENV end: `636.074`.
- Single appended tail that makes equity <= 0: about `-294.1` bps.
- Tail that gives back profit to starting $35: `-277.9` bps.

| Tail | End Equity | Multiple | Ruined At |
| ---: | ---: | ---: | --- |
| -150 | 311.676 | 8.905 | None |
| -180 | 246.797 | 7.051 | None |
| -200 | 203.544 | 5.816 | None |
| -220 | 160.291 | 4.58 | None |
| -250 | 95.411 | 2.726 | None |
| -275 | 41.345 | 1.181 | None |
| -300 | -12.721 | -0.363 | 12 |
| -350 | -120.854 | -3.453 | 12 |
| -400 | -228.987 | -6.542 | 12 |
| -507 | -460.39 | -13.154 | 12 |

## Exit Overlap

| Hold | Signals | Overlaps | Blocked if max-one | Max concurrent |
| ---: | ---: | ---: | ---: | ---: |
| 2h | 11 | 2 | 2 | 2 |
| 4h | 11 | 2 | 2 | 2 |
| 8h | 11 | 2 | 2 | 2 |

## Intermediate Sizing Frontier

| Ratio | Observed End | -150 End | -300 End | -507 End | Ruin Tail | Survive -300 | Survive -507 |
| ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1.0 | 38.96 | 38.375 | 37.791 | 36.984 | -10000.0 | True | True |
| 2.0 | 43.285 | 41.986 | 40.688 | 38.896 | -5000.0 | True | True |
| 5.0 | 58.732 | 54.327 | 49.922 | 43.844 | -2000.0 | True | True |
| 10.0 | 94.657 | 80.458 | 66.26 | 46.666 | -1000.0 | True | True |
| 15.0 | 147.549 | 114.351 | 81.152 | 35.338 | -666.7 | True | True |
| 20.0 | 223.616 | 156.531 | 89.446 | -3.131 | -500.0 | True | False |
| 25.0 | 330.815 | 206.759 | 82.704 | -88.493 | -400.0 | True | False |
| 30.0 | 479.238 | 263.581 | 47.924 | -249.683 | -333.3 | True | False |
| 34.0 | 636.074 | 311.676 | -12.721 | -460.39 | -294.1 | False | False |

## Exit Variant Tail Stress (CURRENT_ENV proxy)

| Variant | Base Sum | Base T3R | -150 End | -300 End | -507 End |
| --- | ---: | ---: | ---: | ---: | ---: |
| fixed_2h | 1089.9 | 406.3 | 232.556 | -9.492 | -343.518 |
| fixed_4h | 1740.8 | 822.6 | 1923.042 | -78.492 | -2840.608 |
| fixed_8h | 1406.4 | 542.2 | 778.828 | -31.789 | -1150.44 |
| sl150_2h | 1089.9 | 406.3 | 232.556 | -9.492 | -343.518 |
| tp300_sl150_4h | 1780.1 | 895.1 | 2122.493 | -86.632 | -3135.226 |
| trail100_after150_4h | 1361.0 | 744.5 | 801.356 | -32.708 | -1183.718 |
| partial_tp150_2h | 1014.4 | 455.0 | 331.48 | -13.53 | -489.643 |
