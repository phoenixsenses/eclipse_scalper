# S34 V Engine Recovery Robustness

Generated: `2026-06-28T21:31:45.361110+00:00`

Research-only. Sweeps confirmation timing and hold horizon; no live/paper state changed.

Filled rows: `46`

Baseline H2: N=22 sum=1120.9 med=39.4 T3R=441.9 max_loss=-144.4

Positive neighborhood cells: `24`

| Rank | Confirm | Condition | Horizon | Summary | Delta sum | Delta T3R |
| ---: | ---: | --- | ---: | --- | ---: | ---: |
| 1 | 30m | `anchor_and_btc` | 4.0h | N=18 sum=2022.8 med=82.0 T3R=1103.6 max_loss=-24.2 | 901.9 | 661.7 |
| 2 | 20m | `anchor_and_btc` | 4.0h | N=17 sum=2000.8 med=104.9 T3R=1081.6 max_loss=-24.2 | 879.9 | 639.7 |
| 3 | 45m | `anchor_and_btc` | 4.0h | N=18 sum=1990.5 med=82.0 T3R=1071.3 max_loss=-24.2 | 869.6 | 629.4 |
| 4 | 30m | `btc_not_down_continues` | 4.0h | N=19 sum=1963.7 med=59.1 T3R=1044.5 max_loss=-59.1 | 842.8 | 602.6 |
| 5 | 45m | `btc_not_down_continues` | 4.0h | N=20 sum=1960.4 med=57.3 T3R=1041.2 max_loss=-59.1 | 839.5 | 599.3 |
| 6 | 45m | `btc_not_down_continues` | 6.0h | N=20 sum=1901.5 med=104.7 T3R=1034.8 max_loss=-137.6 | 780.6 | 592.9 |
| 7 | 15m | `btc_not_down_continues` | 4.0h | N=18 sum=1941.7 med=82.0 T3R=1022.5 max_loss=-59.1 | 820.8 | 580.6 |
| 8 | 20m | `btc_not_down_continues` | 4.0h | N=18 sum=1941.7 med=82.0 T3R=1022.5 max_loss=-59.1 | 820.8 | 580.6 |
| 9 | 20m | `anchor_reclaimed` | 4.0h | N=19 sum=1888.6 med=59.1 T3R=969.4 max_loss=-134.2 | 767.7 | 527.5 |
| 10 | 30m | `anchor_reclaimed` | 4.0h | N=19 sum=1888.6 med=59.1 T3R=969.4 max_loss=-134.2 | 767.7 | 527.5 |
| 11 | 45m | `anchor_reclaimed` | 4.0h | N=20 sum=1878.3 med=57.3 T3R=959.1 max_loss=-134.2 | 757.4 | 517.2 |
| 12 | 45m | `btc_not_down_continues` | 5.0h | N=20 sum=1867.9 med=83.2 T3R=933.9 max_loss=-131.1 | 747.0 | 492.0 |
| 13 | 15m | `all` | 4.0h | N=22 sum=1848.2 med=46.4 T3R=929.0 max_loss=-134.2 | 727.3 | 487.1 |
| 14 | 20m | `all` | 4.0h | N=22 sum=1848.2 med=46.4 T3R=929.0 max_loss=-134.2 | 727.3 | 487.1 |
| 15 | 30m | `all` | 4.0h | N=22 sum=1848.2 med=46.4 T3R=929.0 max_loss=-134.2 | 727.3 | 487.1 |
| 16 | 45m | `all` | 4.0h | N=22 sum=1848.2 med=46.4 T3R=929.0 max_loss=-134.2 | 727.3 | 487.1 |
| 17 | 60m | `all` | 4.0h | N=22 sum=1848.2 med=46.4 T3R=929.0 max_loss=-134.2 | 727.3 | 487.1 |
| 18 | 15m | `all` | 6.0h | N=21 sum=1788.9 med=91.4 T3R=922.2 max_loss=-137.6 | 668.0 | 480.3 |
| 19 | 20m | `all` | 6.0h | N=21 sum=1788.9 med=91.4 T3R=922.2 max_loss=-137.6 | 668.0 | 480.3 |
| 20 | 30m | `all` | 6.0h | N=21 sum=1788.9 med=91.4 T3R=922.2 max_loss=-137.6 | 668.0 | 480.3 |
| 21 | 45m | `all` | 6.0h | N=21 sum=1788.9 med=91.4 T3R=922.2 max_loss=-137.6 | 668.0 | 480.3 |
| 22 | 60m | `all` | 6.0h | N=21 sum=1788.9 med=91.4 T3R=922.2 max_loss=-137.6 | 668.0 | 480.3 |
| 23 | 15m | `anchor_and_btc` | 4.0h | N=16 sum=1716.9 med=82.0 T3R=901.9 max_loss=-24.2 | 596.0 | 460.0 |
| 24 | 60m | `anchor_reclaimed` | 4.0h | N=21 sum=1819.2 med=55.5 T3R=900.0 max_loss=-134.2 | 698.3 | 458.1 |
| 25 | 60m | `btc_not_down_continues` | 4.0h | N=19 sum=1720.1 med=55.5 T3R=861.5 max_loss=-59.1 | 599.2 | 419.6 |
| 26 | 45m | `anchor_and_btc` | 5.0h | N=18 sum=1791.3 med=87.0 T3R=857.3 max_loss=-131.1 | 670.4 | 415.4 |
| 27 | 45m | `anchor_and_btc` | 6.0h | N=18 sum=1723.8 med=104.7 T3R=857.1 max_loss=-137.6 | 602.9 | 415.2 |
| 28 | 15m | `btc_not_down_continues` | 6.0h | N=18 sum=1712.6 med=104.7 T3R=845.9 max_loss=-137.6 | 591.7 | 404.0 |
| 29 | 20m | `btc_not_down_continues` | 6.0h | N=18 sum=1712.6 med=104.7 T3R=845.9 max_loss=-137.6 | 591.7 | 404.0 |
| 30 | 30m | `btc_not_down_continues` | 6.0h | N=18 sum=1712.6 med=104.7 T3R=845.9 max_loss=-137.6 | 591.7 | 404.0 |
| 31 | 60m | `btc_not_down_continues` | 6.0h | N=19 sum=1664.7 med=91.4 T3R=845.3 max_loss=-137.6 | 543.8 | 403.4 |
| 32 | 60m | `anchor_and_btc` | 4.0h | N=18 sum=1691.1 med=57.3 T3R=832.5 max_loss=-59.1 | 570.2 | 390.6 |
| 33 | 45m | `bull_reclaim` | 4.0h | N=14 sum=1748.4 med=82.0 T3R=829.2 max_loss=2.2 | 627.5 | 387.3 |
| 34 | 45m | `strong_rebound` | 4.0h | N=13 sum=1742.5 med=104.9 T3R=823.3 max_loss=2.2 | 621.6 | 381.4 |
| 35 | 20m | `anchor_and_btc` | 6.0h | N=17 sum=1688.1 med=118.0 T3R=821.4 max_loss=-137.6 | 567.2 | 379.5 |
| 36 | 30m | `anchor_and_btc` | 6.0h | N=17 sum=1688.1 med=118.0 T3R=821.4 max_loss=-137.6 | 567.2 | 379.5 |
| 37 | 20m | `anchor_and_btc` | 5.0h | N=17 sum=1736.7 med=103.0 T3R=802.7 max_loss=-131.1 | 615.8 | 360.8 |
| 38 | 30m | `anchor_and_btc` | 5.0h | N=17 sum=1736.7 med=103.0 T3R=802.7 max_loss=-131.1 | 615.8 | 360.8 |
| 39 | 15m | `anchor_reclaimed` | 4.0h | N=18 sum=1604.7 med=57.3 T3R=789.7 max_loss=-134.2 | 483.8 | 347.8 |
| 40 | 15m | `btc_not_down_continues` | 5.0h | N=18 sum=1717.7 med=87.0 T3R=783.7 max_loss=-131.1 | 596.8 | 341.8 |
| 41 | 20m | `btc_not_down_continues` | 5.0h | N=18 sum=1717.7 med=87.0 T3R=783.7 max_loss=-131.1 | 596.8 | 341.8 |
| 42 | 30m | `btc_not_down_continues` | 5.0h | N=18 sum=1717.7 med=87.0 T3R=783.7 max_loss=-131.1 | 596.8 | 341.8 |
| 43 | 60m | `anchor_reclaimed` | 6.0h | N=20 sum=1635.7 med=67.9 T3R=769.0 max_loss=-137.6 | 514.8 | 327.1 |
| 44 | 60m | `btc_not_down_continues` | 5.0h | N=19 sum=1652.0 med=70.9 T3R=767.0 max_loss=-131.1 | 531.1 | 325.1 |
| 45 | 15m | `all` | 5.0h | N=21 sum=1689.8 med=70.9 T3R=755.8 max_loss=-178.1 | 568.9 | 313.9 |
| 46 | 20m | `all` | 5.0h | N=21 sum=1689.8 med=70.9 T3R=755.8 max_loss=-178.1 | 568.9 | 313.9 |
| 47 | 30m | `all` | 5.0h | N=21 sum=1689.8 med=70.9 T3R=755.8 max_loss=-178.1 | 568.9 | 313.9 |
| 48 | 45m | `all` | 5.0h | N=21 sum=1689.8 med=70.9 T3R=755.8 max_loss=-178.1 | 568.9 | 313.9 |
| 49 | 60m | `all` | 5.0h | N=21 sum=1689.8 med=70.9 T3R=755.8 max_loss=-178.1 | 568.9 | 313.9 |
| 50 | 45m | `anchor_reclaimed` | 6.0h | N=19 sum=1611.2 med=91.4 T3R=744.5 max_loss=-137.6 | 490.3 | 302.6 |

## Read

- Best cell: 30m `anchor_and_btc` 4.0h -> N=18 sum=2022.8 med=82.0 T3R=1103.6 max_loss=-24.2.
- Simple all-H4: N=22 sum=1848.2 med=46.4 T3R=929.0 max_loss=-134.2; delta T3R `487.1`.
- Frozen winner-extension cell: N=18 sum=2022.8 med=82.0 T3R=1103.6 max_loss=-24.2; delta T3R `661.7`.
