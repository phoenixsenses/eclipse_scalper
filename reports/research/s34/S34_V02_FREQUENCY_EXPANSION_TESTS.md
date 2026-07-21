# S34 V02 Frequency Expansion Tests

Generated: `2026-06-30T09:26:03.549502+00:00`

Research-only. No live executor, paper buckets, config, order logic, or sizing was changed.

## Verdict

- `FREQUENCY_MAP_BUILT_NO_LIVE_PROMOTION`

## Top Cells By All-Sample Sum

| rank | cell | N | sum | median | T3R | hold T3R |
|---:|---|---:|---:|---:|---:|---:|
| 1 | `tests.event_end_vs_maker.taker.event_end_H4` | 341 | 11525.2 | 34.0 | 10099.3 | 3098.6 |
| 2 | `tests.event_end_vs_maker.taker.event_end_H2` | 341 | 10408.2 | 26.9 | 9231.4 | 2239.1 |
| 3 | `tests.event_end_vs_maker.taker.event_end_H1` | 341 | 9582.2 | 21.7 | 8467.4 | 2129.8 |
| 4 | `tests.event_end_vs_maker.taker.reclaim_H4` | 341 | 9032.6 | 29.9 | 7470.2 | 2552.8 |
| 5 | `tests.event_end_vs_maker.taker.reclaim_H2` | 341 | 7670.0 | 19.2 | 6533.0 | 1454.4 |
| 6 | `tests.event_end_vs_maker.taker.reclaim_H1` | 341 | 6423.6 | 13.4 | 5320.5 | 1492.4 |
| 7 | `tests.event_end_vs_maker.taker.event_end_M15` | 341 | 5115.9 | 13.4 | 4583.9 | 1575.1 |
| 8 | `tests.threshold_expansion.50000.cells.tau30_H2` | 479 | 4507.4 | 9.6 | 2613.5 | -124.8 |
| 9 | `tests.threshold_expansion.50000.cells.tau30_H4` | 479 | 4331.8 | 4.4 | 2614.8 | -409.0 |
| 10 | `tests.threshold_expansion.300000.cells.tau600_H4` | 269 | 4007.0 | 6.3 | 2436.2 | 537.5 |
| 11 | `tests.threshold_expansion.300000.cells.tau900_H4` | 263 | 3731.6 | 12.7 | 2141.2 | 220.7 |
| 12 | `tests.threshold_expansion.300000.cells.tau900_H2` | 263 | 3051.1 | 9.4 | 2021.7 | -588.2 |
| 13 | `tests.threshold_expansion.300000.cells.tau600_H2` | 269 | 2906.9 | 10.9 | 1918.8 | -712.0 |
| 14 | `tests.event_end_vs_maker.taker.reclaim_M15` | 341 | 2890.9 | 7.9 | 2368.1 | 1014.7 |
| 15 | `tests.deepbid_ablation.buckets.spread_bps.LOW_<=0.1` | 102 | 2872.5 | 22.9 | 1666.5 | 1201.7 |
| 16 | `tests.threshold_expansion.300000.cells.tau30_H2` | 134 | 2809.8 | 21.4 | 1758.6 | -126.0 |
| 17 | `tests.sell_silence_lane_expansion.lanes.tau30_all_silence` | 193 | 2795.4 | 26.7 | 1522.6 | 993.3 |
| 18 | `tests.threshold_expansion.200000.cells.tau30_H4` | 193 | 2795.4 | 26.7 | 1522.6 | 993.3 |
| 19 | `tests.threshold_expansion.300000.cells.tau60_H2` | 170 | 2758.2 | 17.9 | 1692.6 | 66.6 |
| 20 | `tests.threshold_expansion.150000.cells.tau30_H4` | 222 | 2628.6 | 12.9 | 1117.7 | -655.9 |

## Full JSON

```json
{
  "cross_asset_lead": {
    "cells": {
      "BTCUSDT_BUY_prev1800s_eth_fade_H4": {
        "all": {
          "attempt_n": 125,
          "fill_rate": 1.0,
          "filled_n": 125,
          "max_bps": 413.1,
          "mean_bps": -38.6,
          "median_bps": -4.6,
          "min_bps": -533.5,
          "n": 125,
          "sum_bps": -4828.9,
          "t3r_bps": -5728.1,
          "tail_lt_-100_n": 37,
          "win_rate": 0.48
        },
        "attempt_n": 125,
        "cal": {
          "attempt_n": 76,
          "fill_rate": 1.0,
          "filled_n": 76,
          "max_bps": 413.1,
          "mean_bps": -43.8,
          "median_bps": -17.5,
          "min_bps": -533.5,
          "n": 76,
          "sum_bps": -3327.4,
          "t3r_bps": -4226.6,
          "tail_lt_-100_n": 21,
          "win_rate": 0.421
        },
        "hold": {
          "attempt_n": 49,
          "fill_rate": 1.0,
          "filled_n": 49,
          "max_bps": 224.9,
          "mean_bps": -30.6,
          "median_bps": 10.4,
          "min_bps": -428.4,
          "n": 49,
          "sum_bps": -1501.5,
          "t3r_bps": -2166.9,
          "tail_lt_-100_n": 16,
          "win_rate": 0.571
        }
      },
      "BTCUSDT_BUY_prev300s_eth_fade_H4": {
        "all": {
          "attempt_n": 56,
          "fill_rate": 1.0,
          "filled_n": 56,
          "max_bps": 413.1,
          "mean_bps": -29.9,
          "median_bps": -12.1,
          "min_bps": -490.9,
          "n": 56,
          "sum_bps": -1676.9,
          "t3r_bps": -2576.1,
          "tail_lt_-100_n": 18,
          "win_rate": 0.446
        },
        "attempt_n": 56,
        "cal": {
          "attempt_n": 32,
          "fill_rate": 1.0,
          "filled_n": 32,
          "max_bps": 413.1,
          "mean_bps": -15.6,
          "median_bps": -18.9,
          "min_bps": -490.9,
          "n": 32,
          "sum_bps": -497.7,
          "t3r_bps": -1396.9,
          "tail_lt_-100_n": 8,
          "win_rate": 0.406
        },
        "hold": {
          "attempt_n": 24,
          "fill_rate": 1.0,
          "filled_n": 24,
          "max_bps": 224.1,
          "mean_bps": -49.1,
          "median_bps": -2.6,
          "min_bps": -428.4,
          "n": 24,
          "sum_bps": -1179.2,
          "t3r_bps": -1830.5,
          "tail_lt_-100_n": 10,
          "win_rate": 0.5
        }
      },
      "BTCUSDT_BUY_prev60s_eth_fade_H4": {
        "all": {
          "attempt_n": 34,
          "fill_rate": 1.0,
          "filled_n": 34,
          "max_bps": 253.3,
          "mean_bps": -15.1,
          "median_bps": -9.8,
          "min_bps": -300.2,
          "n": 34,
          "sum_bps": -514.6,
          "t3r_bps": -1217.1,
          "tail_lt_-100_n": 9,
          "win_rate": 0.441
        },
        "attempt_n": 34,
        "cal": {
          "attempt_n": 22,
          "fill_rate": 1.0,
          "filled_n": 22,
          "max_bps": 253.3,
          "mean_bps": -13.3,
          "median_bps": -11.2,
          "min_bps": -266.7,
          "n": 22,
          "sum_bps": -292.2,
          "t3r_bps": -924.9,
          "tail_lt_-100_n": 5,
          "win_rate": 0.409
        },
        "hold": {
          "attempt_n": 12,
          "fill_rate": 1.0,
          "filled_n": 12,
          "max_bps": 216.4,
          "mean_bps": -18.5,
          "median_bps": -2.6,
          "min_bps": -300.2,
          "n": 12,
          "sum_bps": -222.4,
          "t3r_bps": -811.0,
          "tail_lt_-100_n": 4,
          "win_rate": 0.5
        }
      },
      "BTCUSDT_BUY_prev900s_eth_fade_H4": {
        "all": {
          "attempt_n": 73,
          "fill_rate": 1.0,
          "filled_n": 73,
          "max_bps": 413.1,
          "mean_bps": -31.1,
          "median_bps": -4.6,
          "min_bps": -533.5,
          "n": 73,
          "sum_bps": -2273.0,
          "t3r_bps": -3172.2,
          "tail_lt_-100_n": 22,
          "win_rate": 0.466
        },
        "attempt_n": 73,
        "cal": {
          "attempt_n": 44,
          "fill_rate": 1.0,
          "filled_n": 44,
          "max_bps": 413.1,
          "mean_bps": -27.9,
          "median_bps": -6.8,
          "min_bps": -533.5,
          "n": 44,
          "sum_bps": -1227.8,
          "t3r_bps": -2127.0,
          "tail_lt_-100_n": 11,
          "win_rate": 0.432
        },
        "hold": {
          "attempt_n": 29,
          "fill_rate": 1.0,
          "filled_n": 29,
          "max_bps": 224.9,
          "mean_bps": -36.0,
          "median_bps": 5.6,
          "min_bps": -428.4,
          "n": 29,
          "sum_bps": -1045.2,
          "t3r_bps": -1710.6,
          "tail_lt_-100_n": 11,
          "win_rate": 0.517
        }
      },
      "BTCUSDT_SELL_prev1800s_eth_fade_H4": {
        "all": {
          "attempt_n": 103,
          "fill_rate": 1.0,
          "filled_n": 103,
          "max_bps": 405.7,
          "mean_bps": -8.8,
          "median_bps": 27.0,
          "min_bps": -514.1,
          "n": 103,
          "sum_bps": -907.3,
          "t3r_bps": -2036.8,
          "tail_lt_-100_n": 25,
          "win_rate": 0.553
        },
        "attempt_n": 103,
        "cal": {
          "attempt_n": 42,
          "fill_rate": 1.0,
          "filled_n": 42,
          "max_bps": 369.9,
          "mean_bps": 0.2,
          "median_bps": 31.7,
          "min_bps": -345.8,
          "n": 42,
          "sum_bps": 6.4,
          "t3r_bps": -689.1,
          "tail_lt_-100_n": 9,
          "win_rate": 0.571
        },
        "hold": {
          "attempt_n": 61,
          "fill_rate": 1.0,
          "filled_n": 61,
          "max_bps": 405.7,
          "mean_bps": -15.0,
          "median_bps": 27.0,
          "min_bps": -514.1,
          "n": 61,
          "sum_bps": -913.7,
          "t3r_bps": -2003.5,
          "tail_lt_-100_n": 16,
          "win_rate": 0.541
        }
      },
      "BTCUSDT_SELL_prev300s_eth_fade_H4": {
        "all": {
          "attempt_n": 44,
          "fill_rate": 1.0,
          "filled_n": 44,
          "max_bps": 369.9,
          "mean_bps": -6.4,
          "median_bps": 13.8,
          "min_bps": -403.1,
          "n": 44,
          "sum_bps": -281.1,
          "t3r_bps": -1335.1,
          "tail_lt_-100_n": 10,
          "win_rate": 0.523
        },
        "attempt_n": 44,
        "cal": {
          "attempt_n": 16,
          "fill_rate": 1.0,
          "filled_n": 16,
          "max_bps": 369.9,
          "mean_bps": -10.1,
          "median_bps": 5.1,
          "min_bps": -345.8,
          "n": 16,
          "sum_bps": -161.3,
          "t3r_bps": -856.8,
          "tail_lt_-100_n": 3,
          "win_rate": 0.5
        },
        "hold": {
          "attempt_n": 28,
          "fill_rate": 1.0,
          "filled_n": 28,
          "max_bps": 353.9,
          "mean_bps": -4.3,
          "median_bps": 13.8,
          "min_bps": -403.1,
          "n": 28,
          "sum_bps": -119.8,
          "t3r_bps": -1012.1,
          "tail_lt_-100_n": 7,
          "win_rate": 0.536
        }
      },
      "BTCUSDT_SELL_prev60s_eth_fade_H4": {
        "all": {
          "attempt_n": 28,
          "fill_rate": 1.0,
          "filled_n": 28,
          "max_bps": 330.2,
          "mean_bps": 7.4,
          "median_bps": 35.1,
          "min_bps": -342.2,
          "n": 28,
          "sum_bps": 208.5,
          "t3r_bps": -514.8,
          "tail_lt_-100_n": 6,
          "win_rate": 0.607
        },
        "attempt_n": 28,
        "cal": {
          "attempt_n": 9,
          "fill_rate": 1.0,
          "filled_n": 9,
          "max_bps": 165.4,
          "mean_bps": 1.9,
          "median_bps": 43.7,
          "min_bps": -322.9,
          "n": 9,
          "sum_bps": 17.5,
          "t3r_bps": -463.1,
          "tail_lt_-100_n": 2,
          "win_rate": 0.667
        },
        "hold": {
          "attempt_n": 19,
          "fill_rate": 1.0,
          "filled_n": 19,
          "max_bps": 330.2,
          "mean_bps": 10.1,
          "median_bps": 27.0,
          "min_bps": -342.2,
          "n": 19,
          "sum_bps": 191.0,
          "t3r_bps": -532.3,
          "tail_lt_-100_n": 4,
          "win_rate": 0.579
        }
      },
      "BTCUSDT_SELL_prev900s_eth_fade_H4": {
        "all": {
          "attempt_n": 62,
          "fill_rate": 1.0,
          "filled_n": 62,
          "max_bps": 369.9,
          "mean_bps": 0.2,
          "median_bps": 14.7,
          "min_bps": -403.1,
          "n": 62,
          "sum_bps": 14.5,
          "t3r_bps": -1039.5,
          "tail_lt_-100_n": 14,
          "win_rate": 0.548
        },
        "attempt_n": 62,
        "cal": {
          "attempt_n": 25,
          "fill_rate": 1.0,
          "filled_n": 25,
          "max_bps": 369.9,
          "mean_bps": -2.9,
          "median_bps": 7.3,
          "min_bps": -345.8,
          "n": 25,
          "sum_bps": -73.3,
          "t3r_bps": -768.8,
          "tail_lt_-100_n": 5,
          "win_rate": 0.52
        },
        "hold": {
          "attempt_n": 37,
          "fill_rate": 1.0,
          "filled_n": 37,
          "max_bps": 353.9,
          "mean_bps": 2.4,
          "median_bps": 27.0,
          "min_bps": -403.1,
          "n": 37,
          "sum_bps": 87.8,
          "t3r_bps": -811.6,
          "tail_lt_-100_n": 9,
          "win_rate": 0.568
        }
      },
      "SOLUSDT_BUY_prev1800s_eth_fade_H4": {
        "all": {
          "attempt_n": 43,
          "fill_rate": 1.0,
          "filled_n": 43,
          "max_bps": 224.9,
          "mean_bps": -24.1,
          "median_bps": 22.7,
          "min_bps": -428.4,
          "n": 43,
          "sum_bps": -1035.2,
          "t3r_bps": -1668.9,
          "tail_lt_-100_n": 13,
          "win_rate": 0.581
        },
        "attempt_n": 43,
        "cal": {
          "attempt_n": 6,
          "fill_rate": 1.0,
          "filled_n": 6,
          "max_bps": 83.5,
          "mean_bps": -51.4,
          "median_bps": -65.8,
          "min_bps": -169.7,
          "n": 6,
          "sum_bps": -308.3,
          "t3r_bps": -436.2,
          "tail_lt_-100_n": 3,
          "win_rate": 0.333
        },
        "hold": {
          "attempt_n": 37,
          "fill_rate": 1.0,
          "filled_n": 37,
          "max_bps": 224.9,
          "mean_bps": -19.6,
          "median_bps": 27.2,
          "min_bps": -428.4,
          "n": 37,
          "sum_bps": -726.9,
          "t3r_bps": -1360.6,
          "tail_lt_-100_n": 10,
          "win_rate": 0.622
        }
      },
      "SOLUSDT_BUY_prev300s_eth_fade_H4": {
        "all": {
          "attempt_n": 20,
          "fill_rate": 1.0,
          "filled_n": 20,
          "max_bps": 198.0,
          "mean_bps": -26.7,
          "median_bps": 26.5,
          "min_bps": -303.0,
          "n": 20,
          "sum_bps": -534.0,
          "t3r_bps": -1020.2,
          "tail_lt_-100_n": 7,
          "win_rate": 0.6
        },
        "attempt_n": 20,
        "cal": {
          "attempt_n": 2,
          "fill_rate": 1.0,
          "filled_n": 2,
          "max_bps": -155.3,
          "mean_bps": -162.5,
          "median_bps": -162.5,
          "min_bps": -169.7,
          "n": 2,
          "sum_bps": -325.0,
          "t3r_bps": -325.0,
          "tail_lt_-100_n": 2,
          "win_rate": 0.0
        },
        "hold": {
          "attempt_n": 18,
          "fill_rate": 1.0,
          "filled_n": 18,
          "max_bps": 198.0,
          "mean_bps": -11.6,
          "median_bps": 38.0,
          "min_bps": -303.0,
          "n": 18,
          "sum_bps": -209.0,
          "t3r_bps": -695.2,
          "tail_lt_-100_n": 5,
          "win_rate": 0.667
        }
      },
      "SOLUSDT_BUY_prev60s_eth_fade_H4": {
        "all": {
          "attempt_n": 12,
          "fill_rate": 1.0,
          "filled_n": 12,
          "max_bps": 198.0,
          "mean_bps": -11.7,
          "median_bps": 31.3,
          "min_bps": -303.0,
          "n": 12,
          "sum_bps": -140.6,
          "t3r_bps": -626.8,
          "tail_lt_-100_n": 4,
          "win_rate": 0.583
        },
        "attempt_n": 12,
        "cal": {
          "attempt_n": 1,
          "fill_rate": 1.0,
          "filled_n": 1,
          "max_bps": -155.3,
          "mean_bps": -155.3,
          "median_bps": -155.3,
          "min_bps": -155.3,
          "n": 1,
          "sum_bps": -155.3,
          "t3r_bps": -155.3,
          "tail_lt_-100_n": 1,
          "win_rate": 0.0
        },
        "hold": {
          "attempt_n": 11,
          "fill_rate": 1.0,
          "filled_n": 11,
          "max_bps": 198.0,
          "mean_bps": 1.3,
          "median_bps": 48.4,
          "min_bps": -303.0,
          "n": 11,
          "sum_bps": 14.7,
          "t3r_bps": -471.5,
          "tail_lt_-100_n": 3,
          "win_rate": 0.636
        }
      },
      "SOLUSDT_BUY_prev900s_eth_fade_H4": {
        "all": {
          "attempt_n": 30,
          "fill_rate": 1.0,
          "filled_n": 30,
          "max_bps": 198.0,
          "mean_bps": -39.3,
          "median_bps": 20.2,
          "min_bps": -428.4,
          "n": 30,
          "sum_bps": -1180.4,
          "t3r_bps": -1666.6,
          "tail_lt_-100_n": 10,
          "win_rate": 0.567
        },
        "attempt_n": 30,
        "cal": {
          "attempt_n": 3,
          "fill_rate": 1.0,
          "filled_n": 3,
          "max_bps": 83.5,
          "mean_bps": -80.5,
          "median_bps": -155.3,
          "min_bps": -169.7,
          "n": 3,
          "sum_bps": -241.5,
          "t3r_bps": -241.5,
          "tail_lt_-100_n": 2,
          "win_rate": 0.333
        },
        "hold": {
          "attempt_n": 27,
          "fill_rate": 1.0,
          "filled_n": 27,
          "max_bps": 198.0,
          "mean_bps": -34.8,
          "median_bps": 22.7,
          "min_bps": -428.4,
          "n": 27,
          "sum_bps": -938.9,
          "t3r_bps": -1425.1,
          "tail_lt_-100_n": 8,
          "win_rate": 0.593
        }
      },
      "SOLUSDT_SELL_prev1800s_eth_fade_H4": {
        "all": {
          "attempt_n": 51,
          "fill_rate": 1.0,
          "filled_n": 51,
          "max_bps": 353.9,
          "mean_bps": 12.1,
          "median_bps": 36.9,
          "min_bps": -499.9,
          "n": 51,
          "sum_bps": 617.5,
          "t3r_bps": -175.6,
          "tail_lt_-100_n": 7,
          "win_rate": 0.627
        },
        "attempt_n": 51,
        "cal": {
          "attempt_n": 8,
          "fill_rate": 1.0,
          "filled_n": 8,
          "max_bps": 155.0,
          "mean_bps": 58.5,
          "median_bps": 40.9,
          "min_bps": 13.9,
          "n": 8,
          "sum_bps": 467.6,
          "t3r_bps": 165.8,
          "tail_lt_-100_n": 0,
          "win_rate": 1.0
        },
        "hold": {
          "attempt_n": 43,
          "fill_rate": 1.0,
          "filled_n": 43,
          "max_bps": 353.9,
          "mean_bps": 3.5,
          "median_bps": 35.1,
          "min_bps": -499.9,
          "n": 43,
          "sum_bps": 149.9,
          "t3r_bps": -643.2,
          "tail_lt_-100_n": 7,
          "win_rate": 0.558
        }
      },
      "SOLUSDT_SELL_prev300s_eth_fade_H4": {
        "all": {
          "attempt_n": 27,
          "fill_rate": 1.0,
          "filled_n": 27,
          "max_bps": 353.9,
          "mean_bps": 22.6,
          "median_bps": 35.1,
          "min_bps": -341.1,
          "n": 27,
          "sum_bps": 611.0,
          "t3r_bps": -133.8,
          "tail_lt_-100_n": 2,
          "win_rate": 0.63
        },
        "attempt_n": 27,
        "cal": {
          "attempt_n": 4,
          "fill_rate": 1.0,
          "filled_n": 4,
          "max_bps": 155.0,
          "mean_bps": 60.3,
          "median_bps": 36.1,
          "min_bps": 13.9,
          "n": 4,
          "sum_bps": 241.1,
          "t3r_bps": 13.9,
          "tail_lt_-100_n": 0,
          "win_rate": 1.0
        },
        "hold": {
          "attempt_n": 23,
          "fill_rate": 1.0,
          "filled_n": 23,
          "max_bps": 353.9,
          "mean_bps": 16.1,
          "median_bps": 34.2,
          "min_bps": -341.1,
          "n": 23,
          "sum_bps": 369.9,
          "t3r_bps": -374.9,
          "tail_lt_-100_n": 2,
          "win_rate": 0.565
        }
      },
      "SOLUSDT_SELL_prev60s_eth_fade_H4": {
        "all": {
          "attempt_n": 21,
          "fill_rate": 1.0,
          "filled_n": 21,
          "max_bps": 208.2,
          "mean_bps": 8.7,
          "median_bps": 34.2,
          "min_bps": -341.1,
          "n": 21,
          "sum_bps": 182.8,
          "t3r_bps": -363.1,
          "tail_lt_-100_n": 2,
          "win_rate": 0.619
        },
        "attempt_n": 21,
        "cal": {
          "attempt_n": 4,
          "fill_rate": 1.0,
          "filled_n": 4,
          "max_bps": 155.0,
          "mean_bps": 60.3,
          "median_bps": 36.1,
          "min_bps": 13.9,
          "n": 4,
          "sum_bps": 241.1,
          "t3r_bps": 13.9,
          "tail_lt_-100_n": 0,
          "win_rate": 1.0
        },
        "hold": {
          "attempt_n": 17,
          "fill_rate": 1.0,
          "filled_n": 17,
          "max_bps": 208.2,
          "mean_bps": -3.4,
          "median_bps": 12.2,
          "min_bps": -341.1,
          "n": 17,
          "sum_bps": -58.3,
          "t3r_bps": -584.3,
          "tail_lt_-100_n": 2,
          "win_rate": 0.529
        }
      },
      "SOLUSDT_SELL_prev900s_eth_fade_H4": {
        "all": {
          "attempt_n": 37,
          "fill_rate": 1.0,
          "filled_n": 37,
          "max_bps": 353.9,
          "mean_bps": 29.5,
          "median_bps": 35.3,
          "min_bps": -341.1,
          "n": 37,
          "sum_bps": 1093.2,
          "t3r_bps": 315.8,
          "tail_lt_-100_n": 2,
          "win_rate": 0.622
        },
        "attempt_n": 37,
        "cal": {
          "attempt_n": 5,
          "fill_rate": 1.0,
          "filled_n": 5,
          "max_bps": 155.0,
          "mean_bps": 59.6,
          "median_bps": 36.9,
          "min_bps": 13.9,
          "n": 5,
          "sum_bps": 298.2,
          "t3r_bps": 49.2,
          "tail_lt_-100_n": 0,
          "win_rate": 1.0
        },
        "hold": {
          "attempt_n": 32,
          "fill_rate": 1.0,
          "filled_n": 32,
          "max_bps": 353.9,
          "mean_bps": 24.8,
          "median_bps": 34.7,
          "min_bps": -341.1,
          "n": 32,
          "sum_bps": 795.0,
          "t3r_bps": 17.6,
          "tail_lt_-100_n": 2,
          "win_rate": 0.562
        }
      }
    },
    "split": {
      "holdout_months": [
        "2026-06"
      ],
      "method": "chronological_month_tail_35pct",
      "months": [
        "2026-02",
        "2026-03",
        "2026-04",
        "2026-06"
      ]
    }
  },
  "deepbid_ablation": {
    "base_attempt_n": 239,
    "buckets": {
      "bid_depth_usd": {
        "HIGH_>207439.0": {
          "all": {
            "attempt_n": 35,
            "fill_rate": 1.0,
            "filled_n": 35,
            "max_bps": 330.8,
            "mean_bps": 40.0,
            "median_bps": 22.3,
            "min_bps": -423.0,
            "n": 35,
            "sum_bps": 1399.3,
            "t3r_bps": 455.7,
            "tail_lt_-100_n": 3,
            "win_rate": 0.629
          },
          "attempt_n": 35,
          "cal": {
            "attempt_n": 14,
            "fill_rate": 1.0,
            "filled_n": 14,
            "max_bps": 305.9,
            "mean_bps": 3.0,
            "median_bps": -9.2,
            "min_bps": -206.9,
            "n": 14,
            "sum_bps": 41.9,
            "t3r_bps": -457.6,
            "tail_lt_-100_n": 1,
            "win_rate": 0.357
          },
          "hold": {
            "attempt_n": 21,
            "fill_rate": 1.0,
            "filled_n": 21,
            "max_bps": 330.8,
            "mean_bps": 64.6,
            "median_bps": 57.4,
            "min_bps": -423.0,
            "n": 21,
            "sum_bps": 1357.4,
            "t3r_bps": 492.7,
            "tail_lt_-100_n": 2,
            "win_rate": 0.81
          }
        },
        "LOW_<=115762.6": {
          "all": {
            "attempt_n": 34,
            "fill_rate": 1.0,
            "filled_n": 34,
            "max_bps": 568.3,
            "mean_bps": 30.8,
            "median_bps": 42.2,
            "min_bps": -434.5,
            "n": 34,
            "sum_bps": 1047.7,
            "t3r_bps": 54.1,
            "tail_lt_-100_n": 6,
            "win_rate": 0.676
          },
          "attempt_n": 34,
          "cal": {
            "attempt_n": 5,
            "fill_rate": 1.0,
            "filled_n": 5,
            "max_bps": 160.5,
            "mean_bps": -0.5,
            "median_bps": -46.1,
            "min_bps": -139.9,
            "n": 5,
            "sum_bps": -2.6,
            "t3r_bps": -199.1,
            "tail_lt_-100_n": 1,
            "win_rate": 0.4
          },
          "hold": {
            "attempt_n": 29,
            "fill_rate": 1.0,
            "filled_n": 29,
            "max_bps": 568.3,
            "mean_bps": 36.2,
            "median_bps": 43.1,
            "min_bps": -434.5,
            "n": 29,
            "sum_bps": 1050.3,
            "t3r_bps": 56.7,
            "tail_lt_-100_n": 5,
            "win_rate": 0.724
          }
        },
        "MID_115762.6_207439.0": {
          "all": {
            "attempt_n": 33,
            "fill_rate": 1.0,
            "filled_n": 33,
            "max_bps": 299.3,
            "mean_bps": 12.9,
            "median_bps": 15.2,
            "min_bps": -403.8,
            "n": 33,
            "sum_bps": 425.5,
            "t3r_bps": -268.5,
            "tail_lt_-100_n": 3,
            "win_rate": 0.576
          },
          "attempt_n": 33,
          "cal": {
            "attempt_n": 13,
            "fill_rate": 1.0,
            "filled_n": 13,
            "max_bps": 221.2,
            "mean_bps": 32.7,
            "median_bps": 14.3,
            "min_bps": -90.0,
            "n": 13,
            "sum_bps": 425.5,
            "t3r_bps": -43.0,
            "tail_lt_-100_n": 0,
            "win_rate": 0.615
          },
          "hold": {
            "attempt_n": 20,
            "fill_rate": 1.0,
            "filled_n": 20,
            "max_bps": 299.3,
            "mean_bps": 0.0,
            "median_bps": 18.1,
            "min_bps": -403.8,
            "n": 20,
            "sum_bps": 0.0,
            "t3r_bps": -598.1,
            "tail_lt_-100_n": 3,
            "win_rate": 0.55
          }
        }
      },
      "book_imbalance": {
        "HIGH_>0.4": {
          "all": {
            "attempt_n": 35,
            "fill_rate": 1.0,
            "filled_n": 35,
            "max_bps": 330.8,
            "mean_bps": 30.1,
            "median_bps": 24.5,
            "min_bps": -403.8,
            "n": 35,
            "sum_bps": 1052.1,
            "t3r_bps": 187.4,
            "tail_lt_-100_n": 3,
            "win_rate": 0.571
          },
          "attempt_n": 35,
          "cal": {
            "attempt_n": 15,
            "fill_rate": 1.0,
            "filled_n": 15,
            "max_bps": 151.6,
            "mean_bps": -6.3,
            "median_bps": -16.0,
            "min_bps": -206.9,
            "n": 15,
            "sum_bps": -94.4,
            "t3r_bps": -493.0,
            "tail_lt_-100_n": 1,
            "win_rate": 0.333
          },
          "hold": {
            "attempt_n": 20,
            "fill_rate": 1.0,
            "filled_n": 20,
            "max_bps": 330.8,
            "mean_bps": 57.3,
            "median_bps": 72.6,
            "min_bps": -403.8,
            "n": 20,
            "sum_bps": 1146.5,
            "t3r_bps": 281.8,
            "tail_lt_-100_n": 2,
            "win_rate": 0.75
          }
        },
        "LOW_<=-0.3": {
          "all": {
            "attempt_n": 34,
            "fill_rate": 1.0,
            "filled_n": 34,
            "max_bps": 568.3,
            "mean_bps": 44.6,
            "median_bps": 33.2,
            "min_bps": -263.6,
            "n": 34,
            "sum_bps": 1517.8,
            "t3r_bps": 518.6,
            "tail_lt_-100_n": 4,
            "win_rate": 0.647
          },
          "attempt_n": 34,
          "cal": {
            "attempt_n": 9,
            "fill_rate": 1.0,
            "filled_n": 9,
            "max_bps": 221.2,
            "mean_bps": 26.0,
            "median_bps": -16.3,
            "min_bps": -139.9,
            "n": 9,
            "sum_bps": 233.6,
            "t3r_bps": -230.2,
            "tail_lt_-100_n": 1,
            "win_rate": 0.444
          },
          "hold": {
            "attempt_n": 25,
            "fill_rate": 1.0,
            "filled_n": 25,
            "max_bps": 568.3,
            "mean_bps": 51.4,
            "median_bps": 39.1,
            "min_bps": -263.6,
            "n": 25,
            "sum_bps": 1284.2,
            "t3r_bps": 319.7,
            "tail_lt_-100_n": 3,
            "win_rate": 0.72
          }
        },
        "MID_-0.3_0.4": {
          "all": {
            "attempt_n": 33,
            "fill_rate": 1.0,
            "filled_n": 33,
            "max_bps": 305.9,
            "mean_bps": 9.2,
            "median_bps": 21.2,
            "min_bps": -434.5,
            "n": 33,
            "sum_bps": 302.6,
            "t3r_bps": -518.2,
            "tail_lt_-100_n": 5,
            "win_rate": 0.667
          },
          "attempt_n": 33,
          "cal": {
            "attempt_n": 8,
            "fill_rate": 1.0,
            "filled_n": 8,
            "max_bps": 305.9,
            "mean_bps": 40.7,
            "median_bps": 14.6,
            "min_bps": -46.3,
            "n": 8,
            "sum_bps": 325.6,
            "t3r_bps": -33.7,
            "tail_lt_-100_n": 0,
            "win_rate": 0.75
          },
          "hold": {
            "attempt_n": 25,
            "fill_rate": 1.0,
            "filled_n": 25,
            "max_bps": 299.3,
            "mean_bps": -0.9,
            "median_bps": 23.5,
            "min_bps": -434.5,
            "n": 25,
            "sum_bps": -23.0,
            "t3r_bps": -739.5,
            "tail_lt_-100_n": 5,
            "win_rate": 0.64
          }
        }
      },
      "running_accel": {
        "HIGH_>9235.0": {
          "all": {
            "attempt_n": 81,
            "fill_rate": 1.0,
            "filled_n": 81,
            "max_bps": 349.5,
            "mean_bps": 17.2,
            "median_bps": 16.1,
            "min_bps": -289.2,
            "n": 81,
            "sum_bps": 1396.7,
            "t3r_bps": 469.7,
            "tail_lt_-100_n": 9,
            "win_rate": 0.593
          },
          "attempt_n": 81,
          "cal": {
            "attempt_n": 55,
            "fill_rate": 1.0,
            "filled_n": 55,
            "max_bps": 349.5,
            "mean_bps": 4.4,
            "median_bps": 5.6,
            "min_bps": -289.2,
            "n": 55,
            "sum_bps": 240.4,
            "t3r_bps": -615.3,
            "tail_lt_-100_n": 7,
            "win_rate": 0.509
          },
          "hold": {
            "attempt_n": 26,
            "fill_rate": 1.0,
            "filled_n": 26,
            "max_bps": 306.9,
            "mean_bps": 44.5,
            "median_bps": 33.8,
            "min_bps": -174.7,
            "n": 26,
            "sum_bps": 1156.3,
            "t3r_bps": 476.8,
            "tail_lt_-100_n": 2,
            "win_rate": 0.769
          }
        },
        "LOW_<=5664.5": {
          "all": {
            "attempt_n": 79,
            "fill_rate": 1.0,
            "filled_n": 79,
            "max_bps": 270.5,
            "mean_bps": -3.9,
            "median_bps": 39.1,
            "min_bps": -506.3,
            "n": 79,
            "sum_bps": -304.6,
            "t3r_bps": -1029.9,
            "tail_lt_-100_n": 17,
            "win_rate": 0.62
          },
          "attempt_n": 79,
          "cal": {
            "attempt_n": 43,
            "fill_rate": 1.0,
            "filled_n": 43,
            "max_bps": 270.5,
            "mean_bps": -7.5,
            "median_bps": 36.0,
            "min_bps": -506.3,
            "n": 43,
            "sum_bps": -322.8,
            "t3r_bps": -1002.3,
            "tail_lt_-100_n": 9,
            "win_rate": 0.581
          },
          "hold": {
            "attempt_n": 36,
            "fill_rate": 1.0,
            "filled_n": 36,
            "max_bps": 215.6,
            "mean_bps": 0.5,
            "median_bps": 40.2,
            "min_bps": -434.5,
            "n": 36,
            "sum_bps": 18.2,
            "t3r_bps": -593.6,
            "tail_lt_-100_n": 8,
            "win_rate": 0.667
          }
        },
        "MID_5664.5_9235.0": {
          "all": {
            "attempt_n": 79,
            "fill_rate": 1.0,
            "filled_n": 79,
            "max_bps": 568.3,
            "mean_bps": 13.0,
            "median_bps": 9.4,
            "min_bps": -423.0,
            "n": 79,
            "sum_bps": 1028.4,
            "t3r_bps": -176.6,
            "tail_lt_-100_n": 17,
            "win_rate": 0.532
          },
          "attempt_n": 79,
          "cal": {
            "attempt_n": 54,
            "fill_rate": 1.0,
            "filled_n": 54,
            "max_bps": 305.9,
            "mean_bps": 1.0,
            "median_bps": 6.5,
            "min_bps": -391.3,
            "n": 54,
            "sum_bps": 53.8,
            "t3r_bps": -746.8,
            "tail_lt_-100_n": 12,
            "win_rate": 0.519
          },
          "hold": {
            "attempt_n": 25,
            "fill_rate": 1.0,
            "filled_n": 25,
            "max_bps": 568.3,
            "mean_bps": 39.0,
            "median_bps": 27.3,
            "min_bps": -423.0,
            "n": 25,
            "sum_bps": 974.6,
            "t3r_bps": -223.8,
            "tail_lt_-100_n": 5,
            "win_rate": 0.56
          }
        }
      },
      "running_rate": {
        "HIGH_>5796.8": {
          "all": {
            "attempt_n": 81,
            "fill_rate": 1.0,
            "filled_n": 81,
            "max_bps": 349.5,
            "mean_bps": 19.3,
            "median_bps": 23.5,
            "min_bps": -423.0,
            "n": 81,
            "sum_bps": 1563.0,
            "t3r_bps": 575.8,
            "tail_lt_-100_n": 13,
            "win_rate": 0.593
          },
          "attempt_n": 81,
          "cal": {
            "attempt_n": 53,
            "fill_rate": 1.0,
            "filled_n": 53,
            "max_bps": 349.5,
            "mean_bps": 15.6,
            "median_bps": 14.9,
            "min_bps": -391.3,
            "n": 53,
            "sum_bps": 825.8,
            "t3r_bps": -100.2,
            "tail_lt_-100_n": 9,
            "win_rate": 0.547
          },
          "hold": {
            "attempt_n": 28,
            "fill_rate": 1.0,
            "filled_n": 28,
            "max_bps": 330.8,
            "mean_bps": 26.3,
            "median_bps": 25.9,
            "min_bps": -423.0,
            "n": 28,
            "sum_bps": 737.2,
            "t3r_bps": -199.8,
            "tail_lt_-100_n": 4,
            "win_rate": 0.679
          }
        },
        "LOW_<=1783.7": {
          "all": {
            "attempt_n": 79,
            "fill_rate": 1.0,
            "filled_n": 79,
            "max_bps": 568.3,
            "mean_bps": 2.0,
            "median_bps": 20.4,
            "min_bps": -506.3,
            "n": 79,
            "sum_bps": 160.7,
            "t3r_bps": -943.8,
            "tail_lt_-100_n": 14,
            "win_rate": 0.57
          },
          "attempt_n": 79,
          "cal": {
            "attempt_n": 47,
            "fill_rate": 1.0,
            "filled_n": 47,
            "max_bps": 270.5,
            "mean_bps": -16.3,
            "median_bps": -14.9,
            "min_bps": -506.3,
            "n": 47,
            "sum_bps": -765.4,
            "t3r_bps": -1530.6,
            "tail_lt_-100_n": 9,
            "win_rate": 0.489
          },
          "hold": {
            "attempt_n": 32,
            "fill_rate": 1.0,
            "filled_n": 32,
            "max_bps": 568.3,
            "mean_bps": 28.9,
            "median_bps": 40.2,
            "min_bps": -403.8,
            "n": 32,
            "sum_bps": 926.1,
            "t3r_bps": -59.4,
            "tail_lt_-100_n": 5,
            "win_rate": 0.688
          }
        },
        "MID_1783.7_5796.8": {
          "all": {
            "attempt_n": 79,
            "fill_rate": 1.0,
            "filled_n": 79,
            "max_bps": 239.2,
            "mean_bps": 5.0,
            "median_bps": 16.6,
            "min_bps": -434.5,
            "n": 79,
            "sum_bps": 396.8,
            "t3r_bps": -279.1,
            "tail_lt_-100_n": 16,
            "win_rate": 0.582
          },
          "attempt_n": 79,
          "cal": {
            "attempt_n": 52,
            "fill_rate": 1.0,
            "filled_n": 52,
            "max_bps": 239.2,
            "mean_bps": -1.7,
            "median_bps": 15.5,
            "min_bps": -365.4,
            "n": 52,
            "sum_bps": -89.0,
            "t3r_bps": -725.4,
            "tail_lt_-100_n": 10,
            "win_rate": 0.558
          },
          "hold": {
            "attempt_n": 27,
            "fill_rate": 1.0,
            "filled_n": 27,
            "max_bps": 227.0,
            "mean_bps": 18.0,
            "median_bps": 35.1,
            "min_bps": -434.5,
            "n": 27,
            "sum_bps": 485.8,
            "t3r_bps": -124.4,
            "tail_lt_-100_n": 6,
            "win_rate": 0.63
          }
        }
      },
      "spread_bps": {
        "HIGH_>0.1": {
          "all": {
            "attempt_n": 0,
            "fill_rate": null,
            "filled_n": 0,
            "max_bps": null,
            "mean_bps": null,
            "median_bps": null,
            "min_bps": null,
            "n": 0,
            "sum_bps": 0.0,
            "t3r_bps": 0.0,
            "tail_lt_-100_n": 0,
            "win_rate": null
          },
          "attempt_n": 0,
          "cal": {
            "attempt_n": 0,
            "fill_rate": null,
            "filled_n": 0,
            "max_bps": null,
            "mean_bps": null,
            "median_bps": null,
            "min_bps": null,
            "n": 0,
            "sum_bps": 0.0,
            "t3r_bps": 0.0,
            "tail_lt_-100_n": 0,
            "win_rate": null
          },
          "hold": {
            "attempt_n": 0,
            "fill_rate": null,
            "filled_n": 0,
            "max_bps": null,
            "mean_bps": null,
            "median_bps": null,
            "min_bps": null,
            "n": 0,
            "sum_bps": 0.0,
            "t3r_bps": 0.0,
            "tail_lt_-100_n": 0,
            "win_rate": null
          }
        },
        "LOW_<=0.1": {
          "all": {
            "attempt_n": 102,
            "fill_rate": 1.0,
            "filled_n": 102,
            "max_bps": 568.3,
            "mean_bps": 28.2,
            "median_bps": 22.9,
            "min_bps": -434.5,
            "n": 102,
            "sum_bps": 2872.5,
            "t3r_bps": 1666.5,
            "tail_lt_-100_n": 12,
            "win_rate": 0.627
          },
          "attempt_n": 102,
          "cal": {
            "attempt_n": 32,
            "fill_rate": 1.0,
            "filled_n": 32,
            "max_bps": 305.9,
            "mean_bps": 14.5,
            "median_bps": -1.9,
            "min_bps": -206.9,
            "n": 32,
            "sum_bps": 464.8,
            "t3r_bps": -222.8,
            "tail_lt_-100_n": 2,
            "win_rate": 0.469
          },
          "hold": {
            "attempt_n": 70,
            "fill_rate": 1.0,
            "filled_n": 70,
            "max_bps": 568.3,
            "mean_bps": 34.4,
            "median_bps": 42.2,
            "min_bps": -434.5,
            "n": 70,
            "sum_bps": 2407.7,
            "t3r_bps": 1201.7,
            "tail_lt_-100_n": 10,
            "win_rate": 0.7
          }
        },
        "MID_0.1_0.1": {
          "all": {
            "attempt_n": 0,
            "fill_rate": null,
            "filled_n": 0,
            "max_bps": null,
            "mean_bps": null,
            "median_bps": null,
            "min_bps": null,
            "n": 0,
            "sum_bps": 0.0,
            "t3r_bps": 0.0,
            "tail_lt_-100_n": 0,
            "win_rate": null
          },
          "attempt_n": 0,
          "cal": {
            "attempt_n": 0,
            "fill_rate": null,
            "filled_n": 0,
            "max_bps": null,
            "mean_bps": null,
            "median_bps": null,
            "min_bps": null,
            "n": 0,
            "sum_bps": 0.0,
            "t3r_bps": 0.0,
            "tail_lt_-100_n": 0,
            "win_rate": null
          },
          "hold": {
            "attempt_n": 0,
            "fill_rate": null,
            "filled_n": 0,
            "max_bps": null,
            "mean_bps": null,
            "median_bps": null,
            "min_bps": null,
            "n": 0,
            "sum_bps": 0.0,
            "t3r_bps": 0.0,
            "tail_lt_-100_n": 0,
            "win_rate": null
          }
        }
      }
    },
    "split": {
      "holdout_months": [
        "2026-06"
      ],
      "method": "chronological_month_tail_35pct",
      "months": [
        "2026-02",
        "2026-03",
        "2026-04",
        "2026-06"
      ]
    }
  },
  "event_end_vs_maker": {
    "attempt_n": 341,
    "maker": {
      "tau1800_O0.0_H4": {
        "all": {
          "attempt_n": 341,
          "fill_rate": 0.938,
          "filled_n": 320,
          "max_bps": 669.0,
          "mean_bps": 0.3,
          "median_bps": 7.6,
          "min_bps": -487.0,
          "n": 320,
          "sum_bps": 100.7,
          "t3r_bps": -1312.4,
          "tail_lt_-100_n": 51,
          "win_rate": 0.537
        },
        "attempt_n": 341,
        "cal": {
          "attempt_n": 247,
          "fill_rate": 0.915,
          "filled_n": 226,
          "max_bps": 669.0,
          "mean_bps": -0.5,
          "median_bps": 1.7,
          "min_bps": -353.0,
          "n": 226,
          "sum_bps": -120.1,
          "t3r_bps": -1506.3,
          "tail_lt_-100_n": 37,
          "win_rate": 0.504
        },
        "hold": {
          "attempt_n": 94,
          "fill_rate": 1.0,
          "filled_n": 94,
          "max_bps": 309.0,
          "mean_bps": 2.3,
          "median_bps": 17.2,
          "min_bps": -487.0,
          "n": 94,
          "sum_bps": 220.8,
          "t3r_bps": -635.7,
          "tail_lt_-100_n": 14,
          "win_rate": 0.617
        }
      },
      "tau1800_O10.0_H4": {
        "all": {
          "attempt_n": 341,
          "fill_rate": 0.616,
          "filled_n": 210,
          "max_bps": 429.7,
          "mean_bps": -0.5,
          "median_bps": 7.2,
          "min_bps": -598.8,
          "n": 210,
          "sum_bps": -107.1,
          "t3r_bps": -1154.2,
          "tail_lt_-100_n": 38,
          "win_rate": 0.529
        },
        "attempt_n": 341,
        "cal": {
          "attempt_n": 280,
          "fill_rate": 0.532,
          "filled_n": 149,
          "max_bps": 429.7,
          "mean_bps": -1.3,
          "median_bps": -0.7,
          "min_bps": -353.9,
          "n": 149,
          "sum_bps": -194.5,
          "t3r_bps": -1171.9,
          "tail_lt_-100_n": 28,
          "win_rate": 0.497
        },
        "hold": {
          "attempt_n": 61,
          "fill_rate": 1.0,
          "filled_n": 61,
          "max_bps": 314.6,
          "mean_bps": 1.4,
          "median_bps": 25.7,
          "min_bps": -598.8,
          "n": 61,
          "sum_bps": 87.4,
          "t3r_bps": -766.3,
          "tail_lt_-100_n": 10,
          "win_rate": 0.607
        }
      },
      "tau1800_O20.0_H4": {
        "all": {
          "attempt_n": 341,
          "fill_rate": 0.39,
          "filled_n": 133,
          "max_bps": 464.0,
          "mean_bps": 10.7,
          "median_bps": -3.3,
          "min_bps": -467.7,
          "n": 133,
          "sum_bps": 1420.6,
          "t3r_bps": 351.8,
          "tail_lt_-100_n": 20,
          "win_rate": 0.496
        },
        "attempt_n": 341,
        "cal": {
          "attempt_n": 308,
          "fill_rate": 0.325,
          "filled_n": 100,
          "max_bps": 464.0,
          "mean_bps": 8.7,
          "median_bps": -9.2,
          "min_bps": -325.5,
          "n": 100,
          "sum_bps": 869.6,
          "t3r_bps": -167.4,
          "tail_lt_-100_n": 16,
          "win_rate": 0.47
        },
        "hold": {
          "attempt_n": 33,
          "fill_rate": 1.0,
          "filled_n": 33,
          "max_bps": 315.9,
          "mean_bps": 16.7,
          "median_bps": 28.3,
          "min_bps": -467.7,
          "n": 33,
          "sum_bps": 551.0,
          "t3r_bps": -187.4,
          "tail_lt_-100_n": 4,
          "win_rate": 0.576
        }
      },
      "tau1800_O5.0_H4": {
        "all": {
          "attempt_n": 341,
          "fill_rate": 0.783,
          "filled_n": 267,
          "max_bps": 447.5,
          "mean_bps": 1.8,
          "median_bps": 9.9,
          "min_bps": -552.8,
          "n": 267,
          "sum_bps": 490.1,
          "t3r_bps": -562.5,
          "tail_lt_-100_n": 43,
          "win_rate": 0.547
        },
        "attempt_n": 341,
        "cal": {
          "attempt_n": 265,
          "fill_rate": 0.721,
          "filled_n": 191,
          "max_bps": 447.5,
          "mean_bps": -1.1,
          "median_bps": 2.5,
          "min_bps": -364.7,
          "n": 191,
          "sum_bps": -205.1,
          "t3r_bps": -1210.2,
          "tail_lt_-100_n": 33,
          "win_rate": 0.518
        },
        "hold": {
          "attempt_n": 76,
          "fill_rate": 1.0,
          "filled_n": 76,
          "max_bps": 308.6,
          "mean_bps": 9.1,
          "median_bps": 21.0,
          "min_bps": -552.8,
          "n": 76,
          "sum_bps": 695.2,
          "t3r_bps": -158.6,
          "tail_lt_-100_n": 10,
          "win_rate": 0.618
        }
      },
      "tau300_O0.0_H4": {
        "all": {
          "attempt_n": 400,
          "fill_rate": 0.953,
          "filled_n": 381,
          "max_bps": 577.3,
          "mean_bps": 1.9,
          "median_bps": 14.5,
          "min_bps": -567.1,
          "n": 381,
          "sum_bps": 730.9,
          "t3r_bps": -593.4,
          "tail_lt_-100_n": 73,
          "win_rate": 0.556
        },
        "attempt_n": 400,
        "cal": {
          "attempt_n": 261,
          "fill_rate": 0.927,
          "filled_n": 242,
          "max_bps": 385.1,
          "mean_bps": -4.3,
          "median_bps": 11.6,
          "min_bps": -567.1,
          "n": 242,
          "sum_bps": -1029.1,
          "t3r_bps": -2123.3,
          "tail_lt_-100_n": 50,
          "win_rate": 0.537
        },
        "hold": {
          "attempt_n": 139,
          "fill_rate": 1.0,
          "filled_n": 139,
          "max_bps": 577.3,
          "mean_bps": 12.7,
          "median_bps": 23.7,
          "min_bps": -440.0,
          "n": 139,
          "sum_bps": 1760.0,
          "t3r_bps": 521.1,
          "tail_lt_-100_n": 23,
          "win_rate": 0.59
        }
      },
      "tau300_O10.0_H4": {
        "all": {
          "attempt_n": 400,
          "fill_rate": 0.66,
          "filled_n": 264,
          "max_bps": 390.7,
          "mean_bps": -3.5,
          "median_bps": 17.1,
          "min_bps": -576.3,
          "n": 264,
          "sum_bps": -912.3,
          "t3r_bps": -2003.3,
          "tail_lt_-100_n": 52,
          "win_rate": 0.538
        },
        "attempt_n": 400,
        "cal": {
          "attempt_n": 306,
          "fill_rate": 0.556,
          "filled_n": 170,
          "max_bps": 390.7,
          "mean_bps": -14.7,
          "median_bps": 0.0,
          "min_bps": -576.3,
          "n": 170,
          "sum_bps": -2492.2,
          "t3r_bps": -3485.5,
          "tail_lt_-100_n": 35,
          "win_rate": 0.5
        },
        "hold": {
          "attempt_n": 94,
          "fill_rate": 1.0,
          "filled_n": 94,
          "max_bps": 388.0,
          "mean_bps": 16.8,
          "median_bps": 29.4,
          "min_bps": -398.0,
          "n": 94,
          "sum_bps": 1579.9,
          "t3r_bps": 600.3,
          "tail_lt_-100_n": 17,
          "win_rate": 0.606
        }
      },
      "tau300_O20.0_H4": {
        "all": {
          "attempt_n": 400,
          "fill_rate": 0.46,
          "filled_n": 184,
          "max_bps": 398.8,
          "mean_bps": -1.7,
          "median_bps": 17.1,
          "min_bps": -582.2,
          "n": 184,
          "sum_bps": -317.0,
          "t3r_bps": -1351.2,
          "tail_lt_-100_n": 37,
          "win_rate": 0.543
        },
        "attempt_n": 400,
        "cal": {
          "attempt_n": 334,
          "fill_rate": 0.353,
          "filled_n": 118,
          "max_bps": 313.7,
          "mean_bps": -17.9,
          "median_bps": 3.3,
          "min_bps": -582.2,
          "n": 118,
          "sum_bps": -2118.0,
          "t3r_bps": -2967.2,
          "tail_lt_-100_n": 24,
          "win_rate": 0.517
        },
        "hold": {
          "attempt_n": 66,
          "fill_rate": 1.0,
          "filled_n": 66,
          "max_bps": 398.8,
          "mean_bps": 27.3,
          "median_bps": 40.2,
          "min_bps": -386.1,
          "n": 66,
          "sum_bps": 1801.0,
          "t3r_bps": 774.7,
          "tail_lt_-100_n": 13,
          "win_rate": 0.591
        }
      },
      "tau300_O5.0_H4": {
        "all": {
          "attempt_n": 400,
          "fill_rate": 0.807,
          "filled_n": 323,
          "max_bps": 378.1,
          "mean_bps": -1.9,
          "median_bps": 15.2,
          "min_bps": -564.4,
          "n": 323,
          "sum_bps": -616.9,
          "t3r_bps": -1702.0,
          "tail_lt_-100_n": 62,
          "win_rate": 0.545
        },
        "attempt_n": 400,
        "cal": {
          "attempt_n": 282,
          "fill_rate": 0.727,
          "filled_n": 205,
          "max_bps": 378.1,
          "mean_bps": -8.1,
          "median_bps": 8.7,
          "min_bps": -564.4,
          "n": 205,
          "sum_bps": -1660.3,
          "t3r_bps": -2702.1,
          "tail_lt_-100_n": 40,
          "win_rate": 0.512
        },
        "hold": {
          "attempt_n": 118,
          "fill_rate": 1.0,
          "filled_n": 118,
          "max_bps": 358.0,
          "mean_bps": 8.8,
          "median_bps": 26.5,
          "min_bps": -438.1,
          "n": 118,
          "sum_bps": 1043.4,
          "t3r_bps": 100.5,
          "tail_lt_-100_n": 22,
          "win_rate": 0.602
        }
      },
      "tau60_O0.0_H4": {
        "all": {
          "attempt_n": 239,
          "fill_rate": 0.921,
          "filled_n": 220,
          "max_bps": 568.3,
          "mean_bps": 7.6,
          "median_bps": 19.5,
          "min_bps": -504.2,
          "n": 220,
          "sum_bps": 1671.0,
          "t3r_bps": 411.3,
          "tail_lt_-100_n": 40,
          "win_rate": 0.573
        },
        "attempt_n": 239,
        "cal": {
          "attempt_n": 157,
          "fill_rate": 0.879,
          "filled_n": 138,
          "max_bps": 352.3,
          "mean_bps": -3.9,
          "median_bps": 9.9,
          "min_bps": -504.2,
          "n": 138,
          "sum_bps": -540.9,
          "t3r_bps": -1482.1,
          "tail_lt_-100_n": 25,
          "win_rate": 0.514
        },
        "hold": {
          "attempt_n": 82,
          "fill_rate": 1.0,
          "filled_n": 82,
          "max_bps": 568.3,
          "mean_bps": 27.0,
          "median_bps": 42.5,
          "min_bps": -422.1,
          "n": 82,
          "sum_bps": 2211.9,
          "t3r_bps": 997.2,
          "tail_lt_-100_n": 15,
          "win_rate": 0.671
        }
      },
      "tau60_O10.0_H4": {
        "all": {
          "attempt_n": 239,
          "fill_rate": 0.573,
          "filled_n": 137,
          "max_bps": 346.7,
          "mean_bps": 3.3,
          "median_bps": 20.0,
          "min_bps": -434.5,
          "n": 137,
          "sum_bps": 451.4,
          "t3r_bps": -485.7,
          "tail_lt_-100_n": 28,
          "win_rate": 0.577
        },
        "attempt_n": 239,
        "cal": {
          "attempt_n": 189,
          "fill_rate": 0.46,
          "filled_n": 87,
          "max_bps": 346.7,
          "mean_bps": -5.7,
          "median_bps": 3.3,
          "min_bps": -377.8,
          "n": 87,
          "sum_bps": -496.2,
          "t3r_bps": -1408.1,
          "tail_lt_-100_n": 17,
          "win_rate": 0.517
        },
        "hold": {
          "attempt_n": 50,
          "fill_rate": 1.0,
          "filled_n": 50,
          "max_bps": 283.4,
          "mean_bps": 19.0,
          "median_bps": 45.2,
          "min_bps": -434.5,
          "n": 50,
          "sum_bps": 947.6,
          "t3r_bps": 218.5,
          "tail_lt_-100_n": 11,
          "win_rate": 0.68
        }
      },
      "tau60_O20.0_H4": {
        "all": {
          "attempt_n": 239,
          "fill_rate": 0.393,
          "filled_n": 94,
          "max_bps": 293.1,
          "mean_bps": 5.5,
          "median_bps": 20.5,
          "min_bps": -436.8,
          "n": 94,
          "sum_bps": 518.6,
          "t3r_bps": -334.1,
          "tail_lt_-100_n": 19,
          "win_rate": 0.564
        },
        "attempt_n": 239,
        "cal": {
          "attempt_n": 203,
          "fill_rate": 0.286,
          "filled_n": 58,
          "max_bps": 291.3,
          "mean_bps": -7.8,
          "median_bps": -0.2,
          "min_bps": -373.0,
          "n": 58,
          "sum_bps": -450.5,
          "t3r_bps": -1272.4,
          "tail_lt_-100_n": 12,
          "win_rate": 0.5
        },
        "hold": {
          "attempt_n": 36,
          "fill_rate": 1.0,
          "filled_n": 36,
          "max_bps": 293.1,
          "mean_bps": 26.9,
          "median_bps": 55.4,
          "min_bps": -436.8,
          "n": 36,
          "sum_bps": 969.1,
          "t3r_bps": 186.2,
          "tail_lt_-100_n": 7,
          "win_rate": 0.667
        }
      },
      "tau60_O5.0_H4": {
        "all": {
          "attempt_n": 239,
          "fill_rate": 0.703,
          "filled_n": 168,
          "max_bps": 347.8,
          "mean_bps": 6.0,
          "median_bps": 23.2,
          "min_bps": -439.9,
          "n": 168,
          "sum_bps": 1012.5,
          "t3r_bps": 47.3,
          "tail_lt_-100_n": 32,
          "win_rate": 0.571
        },
        "attempt_n": 239,
        "cal": {
          "attempt_n": 181,
          "fill_rate": 0.608,
          "filled_n": 110,
          "max_bps": 347.8,
          "mean_bps": 4.7,
          "median_bps": 12.7,
          "min_bps": -383.0,
          "n": 110,
          "sum_bps": 512.3,
          "t3r_bps": -452.9,
          "tail_lt_-100_n": 19,
          "win_rate": 0.527
        },
        "hold": {
          "attempt_n": 58,
          "fill_rate": 1.0,
          "filled_n": 58,
          "max_bps": 279.3,
          "mean_bps": 8.6,
          "median_bps": 36.0,
          "min_bps": -439.9,
          "n": 58,
          "sum_bps": 500.2,
          "t3r_bps": -217.3,
          "tail_lt_-100_n": 13,
          "win_rate": 0.655
        }
      },
      "tau900_O0.0_H4": {
        "all": {
          "attempt_n": 387,
          "fill_rate": 0.941,
          "filled_n": 364,
          "max_bps": 729.8,
          "mean_bps": 7.2,
          "median_bps": 6.8,
          "min_bps": -607.9,
          "n": 364,
          "sum_bps": 2611.8,
          "t3r_bps": 920.2,
          "tail_lt_-100_n": 65,
          "win_rate": 0.541
        },
        "attempt_n": 387,
        "cal": {
          "attempt_n": 270,
          "fill_rate": 0.915,
          "filled_n": 247,
          "max_bps": 729.8,
          "mean_bps": 6.1,
          "median_bps": 5.0,
          "min_bps": -607.9,
          "n": 247,
          "sum_bps": 1504.0,
          "t3r_bps": 3.7,
          "tail_lt_-100_n": 44,
          "win_rate": 0.538
        },
        "hold": {
          "attempt_n": 117,
          "fill_rate": 1.0,
          "filled_n": 117,
          "max_bps": 550.7,
          "mean_bps": 9.5,
          "median_bps": 9.3,
          "min_bps": -474.3,
          "n": 117,
          "sum_bps": 1107.8,
          "t3r_bps": -207.8,
          "tail_lt_-100_n": 21,
          "win_rate": 0.547
        }
      },
      "tau900_O10.0_H4": {
        "all": {
          "attempt_n": 387,
          "fill_rate": 0.664,
          "filled_n": 257,
          "max_bps": 654.4,
          "mean_bps": 1.5,
          "median_bps": 7.3,
          "min_bps": -592.4,
          "n": 257,
          "sum_bps": 380.7,
          "t3r_bps": -1074.4,
          "tail_lt_-100_n": 50,
          "win_rate": 0.533
        },
        "attempt_n": 387,
        "cal": {
          "attempt_n": 308,
          "fill_rate": 0.578,
          "filled_n": 178,
          "max_bps": 654.4,
          "mean_bps": 1.2,
          "median_bps": 4.8,
          "min_bps": -592.4,
          "n": 178,
          "sum_bps": 207.4,
          "t3r_bps": -1178.1,
          "tail_lt_-100_n": 33,
          "win_rate": 0.528
        },
        "hold": {
          "attempt_n": 79,
          "fill_rate": 1.0,
          "filled_n": 79,
          "max_bps": 412.4,
          "mean_bps": 2.2,
          "median_bps": 12.2,
          "min_bps": -461.0,
          "n": 79,
          "sum_bps": 173.3,
          "t3r_bps": -772.6,
          "tail_lt_-100_n": 17,
          "win_rate": 0.544
        }
      },
      "tau900_O20.0_H4": {
        "all": {
          "attempt_n": 387,
          "fill_rate": 0.457,
          "filled_n": 177,
          "max_bps": 676.3,
          "mean_bps": -1.6,
          "median_bps": 9.6,
          "min_bps": -575.5,
          "n": 177,
          "sum_bps": -284.7,
          "t3r_bps": -1875.7,
          "tail_lt_-100_n": 39,
          "win_rate": 0.531
        },
        "attempt_n": 387,
        "cal": {
          "attempt_n": 332,
          "fill_rate": 0.367,
          "filled_n": 122,
          "max_bps": 676.3,
          "mean_bps": -1.7,
          "median_bps": 9.9,
          "min_bps": -575.5,
          "n": 122,
          "sum_bps": -211.3,
          "t3r_bps": -1688.2,
          "tail_lt_-100_n": 28,
          "win_rate": 0.541
        },
        "hold": {
          "attempt_n": 55,
          "fill_rate": 1.0,
          "filled_n": 55,
          "max_bps": 425.2,
          "mean_bps": -1.3,
          "median_bps": 4.7,
          "min_bps": -458.2,
          "n": 55,
          "sum_bps": -73.4,
          "t3r_bps": -938.0,
          "tail_lt_-100_n": 11,
          "win_rate": 0.509
        }
      },
      "tau900_O5.0_H4": {
        "all": {
          "attempt_n": 387,
          "fill_rate": 0.796,
          "filled_n": 308,
          "max_bps": 675.3,
          "mean_bps": 4.8,
          "median_bps": 6.4,
          "min_bps": -579.3,
          "n": 308,
          "sum_bps": 1468.5,
          "t3r_bps": -6.6,
          "tail_lt_-100_n": 55,
          "win_rate": 0.542
        },
        "attempt_n": 387,
        "cal": {
          "attempt_n": 294,
          "fill_rate": 0.731,
          "filled_n": 215,
          "max_bps": 675.3,
          "mean_bps": 4.9,
          "median_bps": 5.0,
          "min_bps": -579.3,
          "n": 215,
          "sum_bps": 1052.2,
          "t3r_bps": -344.5,
          "tail_lt_-100_n": 36,
          "win_rate": 0.535
        },
        "hold": {
          "attempt_n": 93,
          "fill_rate": 1.0,
          "filled_n": 93,
          "max_bps": 416.4,
          "mean_bps": 4.5,
          "median_bps": 13.1,
          "min_bps": -471.3,
          "n": 93,
          "sum_bps": 416.3,
          "t3r_bps": -654.1,
          "tail_lt_-100_n": 19,
          "win_rate": 0.559
        }
      }
    },
    "split": {
      "holdout_months": [
        "2026-06"
      ],
      "method": "chronological_month_tail_35pct",
      "months": [
        "2026-02",
        "2026-03",
        "2026-04",
        "2026-06"
      ]
    },
    "taker": {
      "event_end_H1": {
        "all": {
          "attempt_n": 341,
          "fill_rate": 1.0,
          "filled_n": 341,
          "max_bps": 486.7,
          "mean_bps": 28.1,
          "median_bps": 21.7,
          "min_bps": -311.5,
          "n": 341,
          "sum_bps": 9582.2,
          "t3r_bps": 8467.4,
          "tail_lt_-100_n": 7,
          "win_rate": 0.698
        },
        "cal": {
          "attempt_n": 242,
          "fill_rate": 1.0,
          "filled_n": 242,
          "max_bps": 486.7,
          "mean_bps": 27.7,
          "median_bps": 22.9,
          "min_bps": -311.5,
          "n": 242,
          "sum_bps": 6698.2,
          "t3r_bps": 5668.3,
          "tail_lt_-100_n": 4,
          "win_rate": 0.707
        },
        "hold": {
          "attempt_n": 99,
          "fill_rate": 1.0,
          "filled_n": 99,
          "max_bps": 296.7,
          "mean_bps": 29.1,
          "median_bps": 17.8,
          "min_bps": -175.8,
          "n": 99,
          "sum_bps": 2884.0,
          "t3r_bps": 2129.8,
          "tail_lt_-100_n": 3,
          "win_rate": 0.677
        }
      },
      "event_end_H2": {
        "all": {
          "attempt_n": 341,
          "fill_rate": 1.0,
          "filled_n": 341,
          "max_bps": 455.5,
          "mean_bps": 30.5,
          "median_bps": 26.9,
          "min_bps": -357.9,
          "n": 341,
          "sum_bps": 10408.2,
          "t3r_bps": 9231.4,
          "tail_lt_-100_n": 23,
          "win_rate": 0.648
        },
        "cal": {
          "attempt_n": 242,
          "fill_rate": 1.0,
          "filled_n": 242,
          "max_bps": 455.5,
          "mean_bps": 30.3,
          "median_bps": 26.4,
          "min_bps": -283.1,
          "n": 242,
          "sum_bps": 7336.4,
          "t3r_bps": 6159.6,
          "tail_lt_-100_n": 19,
          "win_rate": 0.636
        },
        "hold": {
          "attempt_n": 99,
          "fill_rate": 1.0,
          "filled_n": 99,
          "max_bps": 290.6,
          "mean_bps": 31.0,
          "median_bps": 27.8,
          "min_bps": -357.9,
          "n": 99,
          "sum_bps": 3071.8,
          "t3r_bps": 2239.1,
          "tail_lt_-100_n": 4,
          "win_rate": 0.677
        }
      },
      "event_end_H4": {
        "all": {
          "attempt_n": 341,
          "fill_rate": 1.0,
          "filled_n": 341,
          "max_bps": 606.2,
          "mean_bps": 33.8,
          "median_bps": 34.0,
          "min_bps": -456.2,
          "n": 341,
          "sum_bps": 11525.2,
          "t3r_bps": 10099.3,
          "tail_lt_-100_n": 40,
          "win_rate": 0.642
        },
        "cal": {
          "attempt_n": 242,
          "fill_rate": 1.0,
          "filled_n": 242,
          "max_bps": 409.7,
          "mean_bps": 29.3,
          "median_bps": 32.3,
          "min_bps": -339.6,
          "n": 242,
          "sum_bps": 7085.2,
          "t3r_bps": 5943.9,
          "tail_lt_-100_n": 30,
          "win_rate": 0.607
        },
        "hold": {
          "attempt_n": 99,
          "fill_rate": 1.0,
          "filled_n": 99,
          "max_bps": 606.2,
          "mean_bps": 44.8,
          "median_bps": 35.6,
          "min_bps": -456.2,
          "n": 99,
          "sum_bps": 4440.0,
          "t3r_bps": 3098.6,
          "tail_lt_-100_n": 10,
          "win_rate": 0.727
        }
      },
      "event_end_M15": {
        "all": {
          "attempt_n": 341,
          "fill_rate": 1.0,
          "filled_n": 341,
          "max_bps": 240.6,
          "mean_bps": 15.0,
          "median_bps": 13.4,
          "min_bps": -124.0,
          "n": 341,
          "sum_bps": 5115.9,
          "t3r_bps": 4583.9,
          "tail_lt_-100_n": 3,
          "win_rate": 0.683
        },
        "cal": {
          "attempt_n": 242,
          "fill_rate": 1.0,
          "filled_n": 242,
          "max_bps": 154.1,
          "mean_bps": 12.7,
          "median_bps": 12.7,
          "min_bps": -124.0,
          "n": 242,
          "sum_bps": 3069.1,
          "t3r_bps": 2661.9,
          "tail_lt_-100_n": 3,
          "win_rate": 0.665
        },
        "hold": {
          "attempt_n": 99,
          "fill_rate": 1.0,
          "filled_n": 99,
          "max_bps": 240.6,
          "mean_bps": 20.7,
          "median_bps": 15.0,
          "min_bps": -73.2,
          "n": 99,
          "sum_bps": 2046.8,
          "t3r_bps": 1575.1,
          "tail_lt_-100_n": 0,
          "win_rate": 0.727
        }
      },
      "reclaim_H1": {
        "all": {
          "attempt_n": 341,
          "fill_rate": 1.0,
          "filled_n": 341,
          "max_bps": 481.5,
          "mean_bps": 18.8,
          "median_bps": 13.4,
          "min_bps": -311.5,
          "n": 341,
          "sum_bps": 6423.6,
          "t3r_bps": 5320.5,
          "tail_lt_-100_n": 9,
          "win_rate": 0.625
        },
        "cal": {
          "attempt_n": 242,
          "fill_rate": 1.0,
          "filled_n": 242,
          "max_bps": 481.5,
          "mean_bps": 17.3,
          "median_bps": 13.9,
          "min_bps": -311.5,
          "n": 242,
          "sum_bps": 4175.4,
          "t3r_bps": 3163.7,
          "tail_lt_-100_n": 6,
          "win_rate": 0.628
        },
        "hold": {
          "attempt_n": 99,
          "fill_rate": 1.0,
          "filled_n": 99,
          "max_bps": 296.7,
          "mean_bps": 22.7,
          "median_bps": 12.4,
          "min_bps": -174.9,
          "n": 99,
          "sum_bps": 2248.2,
          "t3r_bps": 1492.4,
          "tail_lt_-100_n": 3,
          "win_rate": 0.616
        }
      },
      "reclaim_H2": {
        "all": {
          "attempt_n": 341,
          "fill_rate": 1.0,
          "filled_n": 341,
          "max_bps": 450.3,
          "mean_bps": 22.5,
          "median_bps": 19.2,
          "min_bps": -357.9,
          "n": 341,
          "sum_bps": 7670.0,
          "t3r_bps": 6533.0,
          "tail_lt_-100_n": 25,
          "win_rate": 0.619
        },
        "cal": {
          "attempt_n": 242,
          "fill_rate": 1.0,
          "filled_n": 242,
          "max_bps": 450.3,
          "mean_bps": 22.3,
          "median_bps": 21.1,
          "min_bps": -283.1,
          "n": 242,
          "sum_bps": 5392.5,
          "t3r_bps": 4255.5,
          "tail_lt_-100_n": 21,
          "win_rate": 0.607
        },
        "hold": {
          "attempt_n": 99,
          "fill_rate": 1.0,
          "filled_n": 99,
          "max_bps": 290.6,
          "mean_bps": 23.0,
          "median_bps": 15.4,
          "min_bps": -357.9,
          "n": 99,
          "sum_bps": 2277.5,
          "t3r_bps": 1454.4,
          "tail_lt_-100_n": 4,
          "win_rate": 0.646
        }
      },
      "reclaim_H4": {
        "all": {
          "attempt_n": 341,
          "fill_rate": 1.0,
          "filled_n": 341,
          "max_bps": 606.2,
          "mean_bps": 26.5,
          "median_bps": 29.9,
          "min_bps": -456.2,
          "n": 341,
          "sum_bps": 9032.6,
          "t3r_bps": 7470.2,
          "tail_lt_-100_n": 44,
          "win_rate": 0.607
        },
        "cal": {
          "attempt_n": 242,
          "fill_rate": 1.0,
          "filled_n": 242,
          "max_bps": 576.6,
          "mean_bps": 21.5,
          "median_bps": 25.4,
          "min_bps": -339.6,
          "n": 242,
          "sum_bps": 5202.9,
          "t3r_bps": 3885.5,
          "tail_lt_-100_n": 33,
          "win_rate": 0.566
        },
        "hold": {
          "attempt_n": 99,
          "fill_rate": 1.0,
          "filled_n": 99,
          "max_bps": 606.2,
          "mean_bps": 38.7,
          "median_bps": 31.7,
          "min_bps": -456.2,
          "n": 99,
          "sum_bps": 3829.7,
          "t3r_bps": 2552.8,
          "tail_lt_-100_n": 11,
          "win_rate": 0.707
        }
      },
      "reclaim_M15": {
        "all": {
          "attempt_n": 341,
          "fill_rate": 1.0,
          "filled_n": 341,
          "max_bps": 240.6,
          "mean_bps": 8.5,
          "median_bps": 7.9,
          "min_bps": -124.0,
          "n": 341,
          "sum_bps": 2890.9,
          "t3r_bps": 2368.1,
          "tail_lt_-100_n": 3,
          "win_rate": 0.628
        },
        "cal": {
          "attempt_n": 242,
          "fill_rate": 1.0,
          "filled_n": 242,
          "max_bps": 147.3,
          "mean_bps": 5.8,
          "median_bps": 6.9,
          "min_bps": -124.0,
          "n": 242,
          "sum_bps": 1415.4,
          "t3r_bps": 1033.0,
          "tail_lt_-100_n": 3,
          "win_rate": 0.612
        },
        "hold": {
          "attempt_n": 99,
          "fill_rate": 1.0,
          "filled_n": 99,
          "max_bps": 240.6,
          "mean_bps": 14.9,
          "median_bps": 8.7,
          "min_bps": -75.5,
          "n": 99,
          "sum_bps": 1475.5,
          "t3r_bps": 1014.7,
          "tail_lt_-100_n": 0,
          "win_rate": 0.667
        }
      }
    }
  },
  "propagation_precursor": {
    "features": {
      "bid_depth_usd": {
        "BUY": {
          "HIGH_>198017.7": {
            "fade_h4": {
              "max_bps": 286.9,
              "mean_bps": -4.0,
              "median_bps": 17.8,
              "min_bps": -524.2,
              "n": 84,
              "sum_bps": -333.3,
              "t3r_bps": -1125.9,
              "tail_lt_-100_n": 14,
              "win_rate": 0.595
            },
            "n": 84,
            "propagation_rate": 0.571
          },
          "LOW_<=73591.5": {
            "fade_h4": {
              "max_bps": 419.1,
              "mean_bps": -40.7,
              "median_bps": -22.9,
              "min_bps": -533.5,
              "n": 81,
              "sum_bps": -3299.9,
              "t3r_bps": -4132.4,
              "tail_lt_-100_n": 19,
              "win_rate": 0.395
            },
            "n": 81,
            "propagation_rate": 0.63
          },
          "MID_73591.5_198017.7": {
            "fade_h4": {
              "max_bps": 460.4,
              "mean_bps": -15.0,
              "median_bps": 6.2,
              "min_bps": -476.3,
              "n": 81,
              "sum_bps": -1212.8,
              "t3r_bps": -2317.1,
              "tail_lt_-100_n": 17,
              "win_rate": 0.531
            },
            "n": 81,
            "propagation_rate": 0.605
          }
        },
        "SELL": {
          "HIGH_>216314.1": {
            "fade_h4": {
              "max_bps": 367.0,
              "mean_bps": -32.2,
              "median_bps": -25.4,
              "min_bps": -514.1,
              "n": 94,
              "sum_bps": -3024.5,
              "t3r_bps": -4027.3,
              "tail_lt_-100_n": 26,
              "win_rate": 0.394
            },
            "n": 94,
            "propagation_rate": 0.553
          },
          "LOW_<=108277.3": {
            "fade_h4": {
              "max_bps": 611.3,
              "mean_bps": 2.4,
              "median_bps": 19.5,
              "min_bps": -499.9,
              "n": 90,
              "sum_bps": 213.1,
              "t3r_bps": -1181.9,
              "tail_lt_-100_n": 15,
              "win_rate": 0.567
            },
            "n": 91,
            "propagation_rate": 0.615
          },
          "MID_108277.3_216314.1": {
            "fade_h4": {
              "max_bps": 353.9,
              "mean_bps": 20.7,
              "median_bps": 27.0,
              "min_bps": -469.4,
              "n": 91,
              "sum_bps": 1882.1,
              "t3r_bps": 883.2,
              "tail_lt_-100_n": 11,
              "win_rate": 0.604
            },
            "n": 91,
            "propagation_rate": 0.593
          }
        }
      },
      "book_imbalance": {
        "BUY": {
          "HIGH_>0.3": {
            "fade_h4": {
              "max_bps": 286.9,
              "mean_bps": -9.6,
              "median_bps": 13.1,
              "min_bps": -524.2,
              "n": 84,
              "sum_bps": -808.4,
              "t3r_bps": -1578.6,
              "tail_lt_-100_n": 15,
              "win_rate": 0.571
            },
            "n": 84,
            "propagation_rate": 0.619
          },
          "LOW_<=-0.4": {
            "fade_h4": {
              "max_bps": 224.1,
              "mean_bps": -37.4,
              "median_bps": -20.5,
              "min_bps": -425.7,
              "n": 81,
              "sum_bps": -3031.3,
              "t3r_bps": -3668.8,
              "tail_lt_-100_n": 17,
              "win_rate": 0.407
            },
            "n": 81,
            "propagation_rate": 0.642
          },
          "MID_-0.4_0.3": {
            "fade_h4": {
              "max_bps": 460.4,
              "mean_bps": -12.4,
              "median_bps": 3.9,
              "min_bps": -533.5,
              "n": 81,
              "sum_bps": -1006.3,
              "t3r_bps": -2218.7,
              "tail_lt_-100_n": 18,
              "win_rate": 0.543
            },
            "n": 81,
            "propagation_rate": 0.543
          }
        },
        "SELL": {
          "HIGH_>0.3": {
            "fade_h4": {
              "max_bps": 367.0,
              "mean_bps": -2.8,
              "median_bps": -6.4,
              "min_bps": -514.1,
              "n": 94,
              "sum_bps": -267.5,
              "t3r_bps": -1311.8,
              "tail_lt_-100_n": 20,
              "win_rate": 0.479
            },
            "n": 94,
            "propagation_rate": 0.606
          },
          "LOW_<=-0.4": {
            "fade_h4": {
              "max_bps": 611.3,
              "mean_bps": 14.7,
              "median_bps": 27.5,
              "min_bps": -403.1,
              "n": 90,
              "sum_bps": 1326.6,
              "t3r_bps": 83.8,
              "tail_lt_-100_n": 12,
              "win_rate": 0.578
            },
            "n": 91,
            "propagation_rate": 0.571
          },
          "MID_-0.4_0.3": {
            "fade_h4": {
              "max_bps": 378.0,
              "mean_bps": -21.9,
              "median_bps": 13.5,
              "min_bps": -499.9,
              "n": 91,
              "sum_bps": -1988.4,
              "t3r_bps": -3001.4,
              "tail_lt_-100_n": 20,
              "win_rate": 0.505
            },
            "n": 91,
            "propagation_rate": 0.582
          }
        }
      },
      "event_duration_sec": {
        "BUY": {
          "HIGH_>189.6": {
            "fade_h4": {
              "max_bps": 457.8,
              "mean_bps": -32.9,
              "median_bps": -8.8,
              "min_bps": -591.7,
              "n": 190,
              "sum_bps": -6255.1,
              "t3r_bps": -7356.8,
              "tail_lt_-100_n": 53,
              "win_rate": 0.463
            },
            "n": 190,
            "propagation_rate": 0.553
          },
          "LOW_<=95.9": {
            "fade_h4": {
              "max_bps": 505.6,
              "mean_bps": -10.8,
              "median_bps": 5.4,
              "min_bps": -671.3,
              "n": 184,
              "sum_bps": -1982.3,
              "t3r_bps": -3367.4,
              "tail_lt_-100_n": 35,
              "win_rate": 0.533
            },
            "n": 184,
            "propagation_rate": 0.457
          },
          "MID_95.9_189.6": {
            "fade_h4": {
              "max_bps": 413.1,
              "mean_bps": -27.3,
              "median_bps": -13.4,
              "min_bps": -584.0,
              "n": 183,
              "sum_bps": -4994.4,
              "t3r_bps": -5895.8,
              "tail_lt_-100_n": 42,
              "win_rate": 0.475
            },
            "n": 183,
            "propagation_rate": 0.508
          }
        },
        "SELL": {
          "HIGH_>198.5": {
            "fade_h4": {
              "max_bps": 294.6,
              "mean_bps": -21.7,
              "median_bps": 2.1,
              "min_bps": -540.8,
              "n": 199,
              "sum_bps": -4324.4,
              "t3r_bps": -5181.8,
              "tail_lt_-100_n": 49,
              "win_rate": 0.513
            },
            "n": 199,
            "propagation_rate": 0.533
          },
          "LOW_<=110.1": {
            "fade_h4": {
              "max_bps": 386.8,
              "mean_bps": 17.6,
              "median_bps": 26.4,
              "min_bps": -469.4,
              "n": 194,
              "sum_bps": 3415.7,
              "t3r_bps": 2281.0,
              "tail_lt_-100_n": 33,
              "win_rate": 0.572
            },
            "n": 194,
            "propagation_rate": 0.438
          },
          "MID_110.1_198.5": {
            "fade_h4": {
              "max_bps": 611.3,
              "mean_bps": 0.4,
              "median_bps": 5.3,
              "min_bps": -514.1,
              "n": 192,
              "sum_bps": 81.7,
              "t3r_bps": -1302.3,
              "tail_lt_-100_n": 32,
              "win_rate": 0.51
            },
            "n": 193,
            "propagation_rate": 0.497
          }
        }
      },
      "post_anchor_liq_notional": {
        "BUY": {
          "HIGH_>104083.1": {
            "fade_h4": {
              "max_bps": 457.8,
              "mean_bps": -41.2,
              "median_bps": -20.4,
              "min_bps": -591.7,
              "n": 190,
              "sum_bps": -7836.2,
              "t3r_bps": -9046.0,
              "tail_lt_-100_n": 57,
              "win_rate": 0.432
            },
            "n": 190,
            "propagation_rate": 0.642
          },
          "LOW_<=8928.8": {
            "fade_h4": {
              "max_bps": 505.6,
              "mean_bps": -13.5,
              "median_bps": 0.5,
              "min_bps": -671.3,
              "n": 184,
              "sum_bps": -2478.6,
              "t3r_bps": -3857.7,
              "tail_lt_-100_n": 36,
              "win_rate": 0.5
            },
            "n": 184,
            "propagation_rate": 0.413
          },
          "MID_8928.8_104083.1": {
            "fade_h4": {
              "max_bps": 286.9,
              "mean_bps": -15.9,
              "median_bps": 8.1,
              "min_bps": -584.0,
              "n": 183,
              "sum_bps": -2917.0,
              "t3r_bps": -3704.9,
              "tail_lt_-100_n": 37,
              "win_rate": 0.541
            },
            "n": 183,
            "propagation_rate": 0.459
          }
        },
        "SELL": {
          "HIGH_>111926.6": {
            "fade_h4": {
              "max_bps": 378.0,
              "mean_bps": -5.0,
              "median_bps": 7.0,
              "min_bps": -514.1,
              "n": 198,
              "sum_bps": -997.2,
              "t3r_bps": -2096.1,
              "tail_lt_-100_n": 45,
              "win_rate": 0.515
            },
            "n": 199,
            "propagation_rate": 0.719
          },
          "LOW_<=9545.1": {
            "fade_h4": {
              "max_bps": 611.3,
              "mean_bps": 5.2,
              "median_bps": 21.1,
              "min_bps": -495.8,
              "n": 194,
              "sum_bps": 1016.6,
              "t3r_bps": -302.0,
              "tail_lt_-100_n": 33,
              "win_rate": 0.577
            },
            "n": 194,
            "propagation_rate": 0.294
          },
          "MID_9545.1_111926.6": {
            "fade_h4": {
              "max_bps": 405.7,
              "mean_bps": -4.4,
              "median_bps": 0.6,
              "min_bps": -540.8,
              "n": 193,
              "sum_bps": -846.4,
              "t3r_bps": -2008.8,
              "tail_lt_-100_n": 36,
              "win_rate": 0.503
            },
            "n": 193,
            "propagation_rate": 0.451
          }
        }
      },
      "running_accel": {
        "BUY": {
          "HIGH_>9983.3": {
            "fade_h4": {
              "max_bps": 505.6,
              "mean_bps": -8.1,
              "median_bps": 1.1,
              "min_bps": -591.7,
              "n": 190,
              "sum_bps": -1542.7,
              "t3r_bps": -2872.1,
              "tail_lt_-100_n": 37,
              "win_rate": 0.521
            },
            "n": 190,
            "propagation_rate": 0.542
          },
          "LOW_<=6910.0": {
            "fade_h4": {
              "max_bps": 413.1,
              "mean_bps": -36.9,
              "median_bps": -13.8,
              "min_bps": -671.3,
              "n": 184,
              "sum_bps": -6791.1,
              "t3r_bps": -7711.8,
              "tail_lt_-100_n": 49,
              "win_rate": 0.467
            },
            "n": 184,
            "propagation_rate": 0.554
          },
          "MID_6910.0_9983.3": {
            "fade_h4": {
              "max_bps": 460.4,
              "mean_bps": -26.8,
              "median_bps": -4.0,
              "min_bps": -584.0,
              "n": 183,
              "sum_bps": -4898.0,
              "t3r_bps": -6088.5,
              "tail_lt_-100_n": 44,
              "win_rate": 0.481
            },
            "n": 183,
            "propagation_rate": 0.421
          }
        },
        "SELL": {
          "HIGH_>9671.7": {
            "fade_h4": {
              "max_bps": 386.8,
              "mean_bps": 5.1,
              "median_bps": 15.5,
              "min_bps": -514.1,
              "n": 199,
              "sum_bps": 1005.5,
              "t3r_bps": -129.2,
              "tail_lt_-100_n": 32,
              "win_rate": 0.553
            },
            "n": 199,
            "propagation_rate": 0.477
          },
          "LOW_<=6676.8": {
            "fade_h4": {
              "max_bps": 405.7,
              "mean_bps": -8.1,
              "median_bps": 14.1,
              "min_bps": -495.8,
              "n": 194,
              "sum_bps": -1568.1,
              "t3r_bps": -2618.8,
              "tail_lt_-100_n": 42,
              "win_rate": 0.541
            },
            "n": 194,
            "propagation_rate": 0.505
          },
          "MID_6676.8_9671.7": {
            "fade_h4": {
              "max_bps": 611.3,
              "mean_bps": -1.4,
              "median_bps": -1.6,
              "min_bps": -540.8,
              "n": 192,
              "sum_bps": -264.4,
              "t3r_bps": -1596.6,
              "tail_lt_-100_n": 40,
              "win_rate": 0.5
            },
            "n": 193,
            "propagation_rate": 0.487
          }
        }
      },
      "running_rate": {
        "BUY": {
          "HIGH_>12182.9": {
            "fade_h4": {
              "max_bps": 457.8,
              "mean_bps": -30.1,
              "median_bps": -8.2,
              "min_bps": -671.3,
              "n": 190,
              "sum_bps": -5723.7,
              "t3r_bps": -6852.0,
              "tail_lt_-100_n": 50,
              "win_rate": 0.463
            },
            "n": 190,
            "propagation_rate": 0.474
          },
          "LOW_<=3197.1": {
            "fade_h4": {
              "max_bps": 413.1,
              "mean_bps": -30.1,
              "median_bps": -5.5,
              "min_bps": -533.5,
              "n": 184,
              "sum_bps": -5537.9,
              "t3r_bps": -6594.9,
              "tail_lt_-100_n": 43,
              "win_rate": 0.489
            },
            "n": 184,
            "propagation_rate": 0.5
          },
          "MID_3197.1_12182.9": {
            "fade_h4": {
              "max_bps": 505.6,
              "mean_bps": -10.8,
              "median_bps": 1.3,
              "min_bps": -584.0,
              "n": 183,
              "sum_bps": -1970.2,
              "t3r_bps": -3302.2,
              "tail_lt_-100_n": 37,
              "win_rate": 0.519
            },
            "n": 183,
            "propagation_rate": 0.546
          }
        },
        "SELL": {
          "HIGH_>10133.3": {
            "fade_h4": {
              "max_bps": 378.0,
              "mean_bps": 3.0,
              "median_bps": 14.9,
              "min_bps": -514.1,
              "n": 199,
              "sum_bps": 603.1,
              "t3r_bps": -498.7,
              "tail_lt_-100_n": 31,
              "win_rate": 0.553
            },
            "n": 199,
            "propagation_rate": 0.422
          },
          "LOW_<=2584.3": {
            "fade_h4": {
              "max_bps": 611.3,
              "mean_bps": -9.2,
              "median_bps": 10.2,
              "min_bps": -540.8,
              "n": 194,
              "sum_bps": -1787.5,
              "t3r_bps": -3099.1,
              "tail_lt_-100_n": 43,
              "win_rate": 0.536
            },
            "n": 194,
            "propagation_rate": 0.474
          },
          "MID_2584.3_10133.3": {
            "fade_h4": {
              "max_bps": 386.8,
              "mean_bps": 1.9,
              "median_bps": 7.0,
              "min_bps": -497.1,
              "n": 192,
              "sum_bps": 357.4,
              "t3r_bps": -752.1,
              "tail_lt_-100_n": 40,
              "win_rate": 0.505
            },
            "n": 193,
            "propagation_rate": 0.575
          }
        }
      },
      "single_dominance_pct": {
        "BUY": {
          "HIGH_>94.1": {
            "fade_h4": {
              "max_bps": 505.6,
              "mean_bps": -22.7,
              "median_bps": -4.1,
              "min_bps": -591.7,
              "n": 190,
              "sum_bps": -4317.9,
              "t3r_bps": -5594.9,
              "tail_lt_-100_n": 40,
              "win_rate": 0.474
            },
            "n": 190,
            "propagation_rate": 0.411
          },
          "LOW_<=63.8": {
            "fade_h4": {
              "max_bps": 419.1,
              "mean_bps": -35.5,
              "median_bps": -16.1,
              "min_bps": -524.2,
              "n": 184,
              "sum_bps": -6528.3,
              "t3r_bps": -7693.4,
              "tail_lt_-100_n": 52,
              "win_rate": 0.462
            },
            "n": 184,
            "propagation_rate": 0.603
          },
          "MID_63.8_94.1": {
            "fade_h4": {
              "max_bps": 457.8,
              "mean_bps": -13.0,
              "median_bps": 5.6,
              "min_bps": -671.3,
              "n": 183,
              "sum_bps": -2385.6,
              "t3r_bps": -3496.3,
              "tail_lt_-100_n": 38,
              "win_rate": 0.536
            },
            "n": 183,
            "propagation_rate": 0.508
          }
        },
        "SELL": {
          "HIGH_>90.6": {
            "fade_h4": {
              "max_bps": 386.8,
              "mean_bps": -10.3,
              "median_bps": -1.7,
              "min_bps": -540.8,
              "n": 199,
              "sum_bps": -2042.5,
              "t3r_bps": -3154.9,
              "tail_lt_-100_n": 35,
              "win_rate": 0.492
            },
            "n": 199,
            "propagation_rate": 0.347
          },
          "LOW_<=60.3": {
            "fade_h4": {
              "max_bps": 405.7,
              "mean_bps": 4.5,
              "median_bps": 28.2,
              "min_bps": -443.9,
              "n": 193,
              "sum_bps": 860.3,
              "t3r_bps": -266.3,
              "tail_lt_-100_n": 42,
              "win_rate": 0.57
            },
            "n": 194,
            "propagation_rate": 0.598
          },
          "MID_60.3_90.6": {
            "fade_h4": {
              "max_bps": 611.3,
              "mean_bps": 1.8,
              "median_bps": 12.2,
              "min_bps": -514.1,
              "n": 193,
              "sum_bps": 355.2,
              "t3r_bps": -985.7,
              "tail_lt_-100_n": 37,
              "win_rate": 0.534
            },
            "n": 193,
            "propagation_rate": 0.528
          }
        }
      },
      "spread_bps": {
        "BUY": {
          "HIGH_>0.1": {
            "fade_h4": {
              "max_bps": 3.4,
              "mean_bps": -215.2,
              "median_bps": -215.2,
              "min_bps": -433.7,
              "n": 2,
              "sum_bps": -430.3,
              "t3r_bps": -430.3,
              "tail_lt_-100_n": 1,
              "win_rate": 0.5
            },
            "n": 2,
            "propagation_rate": 0.5
          },
          "LOW_<=0.0": {
            "fade_h4": {
              "max_bps": 286.9,
              "mean_bps": -28.2,
              "median_bps": -10.0,
              "min_bps": -533.5,
              "n": 111,
              "sum_bps": -3132.5,
              "t3r_bps": -3898.0,
              "tail_lt_-100_n": 23,
              "win_rate": 0.45
            },
            "n": 111,
            "propagation_rate": 0.55
          },
          "MID_0.0_0.1": {
            "fade_h4": {
              "max_bps": 460.4,
              "mean_bps": -9.6,
              "median_bps": 9.9,
              "min_bps": -476.3,
              "n": 133,
              "sum_bps": -1283.2,
              "t3r_bps": -2495.6,
              "tail_lt_-100_n": 26,
              "win_rate": 0.556
            },
            "n": 133,
            "propagation_rate": 0.647
          }
        },
        "SELL": {
          "HIGH_>0.1": {
            "fade_h4": {
              "max_bps": -115.7,
              "mean_bps": -115.7,
              "median_bps": -115.7,
              "min_bps": -115.7,
              "n": 1,
              "sum_bps": -115.7,
              "t3r_bps": -115.7,
              "tail_lt_-100_n": 1,
              "win_rate": 0.0
            },
            "n": 1,
            "propagation_rate": 0.0
          },
          "LOW_<=0.0": {
            "fade_h4": {
              "max_bps": 378.0,
              "mean_bps": -2.7,
              "median_bps": -8.7,
              "min_bps": -304.2,
              "n": 96,
              "sum_bps": -262.7,
              "t3r_bps": -1147.6,
              "tail_lt_-100_n": 15,
              "win_rate": 0.458
            },
            "n": 96,
            "propagation_rate": 0.5
          },
          "MID_0.0_0.1": {
            "fade_h4": {
              "max_bps": 611.3,
              "mean_bps": -3.1,
              "median_bps": 16.1,
              "min_bps": -514.1,
              "n": 178,
              "sum_bps": -550.9,
              "t3r_bps": -1934.9,
              "tail_lt_-100_n": 36,
              "win_rate": 0.556
            },
            "n": 179,
            "propagation_rate": 0.637
          }
        }
      }
    },
    "overall": {
      "BUY": {
        "n": 557,
        "propagation_rate": 0.506
      },
      "SELL": {
        "n": 586,
        "propagation_rate": 0.49
      }
    },
    "split": {
      "holdout_months": [
        "2026-06"
      ],
      "method": "chronological_month_tail_35pct",
      "months": [
        "2026-02",
        "2026-03",
        "2026-04",
        "2026-06"
      ]
    }
  },
  "sell_silence_lane_expansion": {
    "lanes": {
      "tau120_all_silence": {
        "all": {
          "attempt_n": 310,
          "fill_rate": 1.0,
          "filled_n": 310,
          "max_bps": 539.6,
          "mean_bps": -2.0,
          "median_bps": 15.8,
          "min_bps": -509.4,
          "n": 310,
          "sum_bps": -615.6,
          "t3r_bps": -1843.3,
          "tail_lt_-100_n": 58,
          "win_rate": 0.558
        },
        "attempt_n": 310,
        "cal": {
          "attempt_n": 204,
          "fill_rate": 1.0,
          "filled_n": 204,
          "max_bps": 353.2,
          "mean_bps": -11.4,
          "median_bps": 9.2,
          "min_bps": -509.4,
          "n": 204,
          "sum_bps": -2316.2,
          "t3r_bps": -3300.2,
          "tail_lt_-100_n": 41,
          "win_rate": 0.529
        },
        "hold": {
          "attempt_n": 106,
          "fill_rate": 1.0,
          "filled_n": 106,
          "max_bps": 539.6,
          "mean_bps": 16.0,
          "median_bps": 24.2,
          "min_bps": -439.2,
          "n": 106,
          "sum_bps": 1700.6,
          "t3r_bps": 536.1,
          "tail_lt_-100_n": 17,
          "win_rate": 0.613
        }
      },
      "tau120_inside_current_v02_shadow_times": {
        "all": {
          "attempt_n": 8,
          "fill_rate": 1.0,
          "filled_n": 8,
          "max_bps": 242.8,
          "mean_bps": 113.4,
          "median_bps": 102.7,
          "min_bps": -21.7,
          "n": 8,
          "sum_bps": 907.4,
          "t3r_bps": 321.9,
          "tail_lt_-100_n": 0,
          "win_rate": 0.875
        },
        "attempt_n": 8,
        "cal": {
          "attempt_n": 3,
          "fill_rate": 1.0,
          "filled_n": 3,
          "max_bps": 169.5,
          "mean_bps": 82.3,
          "median_bps": 99.2,
          "min_bps": -21.7,
          "n": 3,
          "sum_bps": 247.0,
          "t3r_bps": 247.0,
          "tail_lt_-100_n": 0,
          "win_rate": 0.667
        },
        "hold": {
          "attempt_n": 5,
          "fill_rate": 1.0,
          "filled_n": 5,
          "max_bps": 242.8,
          "mean_bps": 132.1,
          "median_bps": 106.2,
          "min_bps": 47.5,
          "n": 5,
          "sum_bps": 660.4,
          "t3r_bps": 138.2,
          "tail_lt_-100_n": 0,
          "win_rate": 1.0
        }
      },
      "tau120_outside_current_v02_shadow_times": {
        "all": {
          "attempt_n": 302,
          "fill_rate": 1.0,
          "filled_n": 302,
          "max_bps": 539.6,
          "mean_bps": -5.0,
          "median_bps": 12.9,
          "min_bps": -509.4,
          "n": 302,
          "sum_bps": -1523.0,
          "t3r_bps": -2750.7,
          "tail_lt_-100_n": 58,
          "win_rate": 0.55
        },
        "attempt_n": 302,
        "cal": {
          "attempt_n": 201,
          "fill_rate": 1.0,
          "filled_n": 201,
          "max_bps": 353.2,
          "mean_bps": -12.8,
          "median_bps": 8.3,
          "min_bps": -509.4,
          "n": 201,
          "sum_bps": -2563.2,
          "t3r_bps": -3547.2,
          "tail_lt_-100_n": 41,
          "win_rate": 0.527
        },
        "hold": {
          "attempt_n": 101,
          "fill_rate": 1.0,
          "filled_n": 101,
          "max_bps": 539.6,
          "mean_bps": 10.3,
          "median_bps": 21.2,
          "min_bps": -439.2,
          "n": 101,
          "sum_bps": 1040.2,
          "t3r_bps": -124.3,
          "tail_lt_-100_n": 17,
          "win_rate": 0.594
        }
      },
      "tau300_all_silence": {
        "all": {
          "attempt_n": 400,
          "fill_rate": 1.0,
          "filled_n": 400,
          "max_bps": 558.0,
          "mean_bps": -1.6,
          "median_bps": 11.2,
          "min_bps": -567.9,
          "n": 400,
          "sum_bps": -636.8,
          "t3r_bps": -1935.0,
          "tail_lt_-100_n": 80,
          "win_rate": 0.545
        },
        "attempt_n": 400,
        "cal": {
          "attempt_n": 255,
          "fill_rate": 1.0,
          "filled_n": 255,
          "max_bps": 377.0,
          "mean_bps": -7.0,
          "median_bps": 9.1,
          "min_bps": -567.9,
          "n": 255,
          "sum_bps": -1783.4,
          "t3r_bps": -2869.6,
          "tail_lt_-100_n": 55,
          "win_rate": 0.533
        },
        "hold": {
          "attempt_n": 145,
          "fill_rate": 1.0,
          "filled_n": 145,
          "max_bps": 558.0,
          "mean_bps": 7.9,
          "median_bps": 18.6,
          "min_bps": -443.2,
          "n": 145,
          "sum_bps": 1146.6,
          "t3r_bps": -64.6,
          "tail_lt_-100_n": 25,
          "win_rate": 0.566
        }
      },
      "tau300_inside_current_v02_shadow_times": {
        "all": {
          "attempt_n": 8,
          "fill_rate": 1.0,
          "filled_n": 8,
          "max_bps": 286.0,
          "mean_bps": 119.9,
          "median_bps": 118.8,
          "min_bps": -10.9,
          "n": 8,
          "sum_bps": 959.2,
          "t3r_bps": 378.6,
          "tail_lt_-100_n": 0,
          "win_rate": 0.875
        },
        "attempt_n": 8,
        "cal": {
          "attempt_n": 3,
          "fill_rate": 1.0,
          "filled_n": 3,
          "max_bps": 124.9,
          "mean_bps": 77.5,
          "median_bps": 118.6,
          "min_bps": -10.9,
          "n": 3,
          "sum_bps": 232.6,
          "t3r_bps": 232.6,
          "tail_lt_-100_n": 0,
          "win_rate": 0.667
        },
        "hold": {
          "attempt_n": 5,
          "fill_rate": 1.0,
          "filled_n": 5,
          "max_bps": 286.0,
          "mean_bps": 145.3,
          "median_bps": 119.1,
          "min_bps": 45.8,
          "n": 5,
          "sum_bps": 726.6,
          "t3r_bps": 151.8,
          "tail_lt_-100_n": 0,
          "win_rate": 1.0
        }
      },
      "tau300_outside_current_v02_shadow_times": {
        "all": {
          "attempt_n": 392,
          "fill_rate": 1.0,
          "filled_n": 392,
          "max_bps": 558.0,
          "mean_bps": -4.1,
          "median_bps": 10.7,
          "min_bps": -567.9,
          "n": 392,
          "sum_bps": -1596.0,
          "t3r_bps": -2894.2,
          "tail_lt_-100_n": 80,
          "win_rate": 0.538
        },
        "attempt_n": 392,
        "cal": {
          "attempt_n": 252,
          "fill_rate": 1.0,
          "filled_n": 252,
          "max_bps": 377.0,
          "mean_bps": -8.0,
          "median_bps": 8.9,
          "min_bps": -567.9,
          "n": 252,
          "sum_bps": -2016.0,
          "t3r_bps": -3102.2,
          "tail_lt_-100_n": 55,
          "win_rate": 0.532
        },
        "hold": {
          "attempt_n": 140,
          "fill_rate": 1.0,
          "filled_n": 140,
          "max_bps": 558.0,
          "mean_bps": 3.0,
          "median_bps": 13.9,
          "min_bps": -443.2,
          "n": 140,
          "sum_bps": 420.0,
          "t3r_bps": -791.2,
          "tail_lt_-100_n": 25,
          "win_rate": 0.55
        }
      },
      "tau30_all_silence": {
        "all": {
          "attempt_n": 193,
          "fill_rate": 1.0,
          "filled_n": 193,
          "max_bps": 597.9,
          "mean_bps": 14.5,
          "median_bps": 26.7,
          "min_bps": -500.2,
          "n": 193,
          "sum_bps": 2795.4,
          "t3r_bps": 1522.6,
          "tail_lt_-100_n": 33,
          "win_rate": 0.606
        },
        "attempt_n": 193,
        "cal": {
          "attempt_n": 126,
          "fill_rate": 1.0,
          "filled_n": 126,
          "max_bps": 346.2,
          "mean_bps": 4.4,
          "median_bps": 17.2,
          "min_bps": -500.2,
          "n": 126,
          "sum_bps": 554.6,
          "t3r_bps": -363.5,
          "tail_lt_-100_n": 21,
          "win_rate": 0.556
        },
        "hold": {
          "attempt_n": 67,
          "fill_rate": 1.0,
          "filled_n": 67,
          "max_bps": 597.9,
          "mean_bps": 33.4,
          "median_bps": 42.6,
          "min_bps": -449.9,
          "n": 67,
          "sum_bps": 2240.8,
          "t3r_bps": 993.3,
          "tail_lt_-100_n": 12,
          "win_rate": 0.701
        }
      },
      "tau30_inside_current_v02_shadow_times": {
        "all": {
          "attempt_n": 5,
          "fill_rate": 1.0,
          "filled_n": 5,
          "max_bps": 166.4,
          "mean_bps": 93.7,
          "median_bps": 99.7,
          "min_bps": -13.5,
          "n": 5,
          "sum_bps": 468.6,
          "t3r_bps": 40.8,
          "tail_lt_-100_n": 0,
          "win_rate": 0.8
        },
        "attempt_n": 5,
        "cal": {
          "attempt_n": 2,
          "fill_rate": 1.0,
          "filled_n": 2,
          "max_bps": 166.4,
          "mean_bps": 76.5,
          "median_bps": 76.5,
          "min_bps": -13.5,
          "n": 2,
          "sum_bps": 152.9,
          "t3r_bps": 152.9,
          "tail_lt_-100_n": 0,
          "win_rate": 0.5
        },
        "hold": {
          "attempt_n": 3,
          "fill_rate": 1.0,
          "filled_n": 3,
          "max_bps": 161.7,
          "mean_bps": 105.2,
          "median_bps": 99.7,
          "min_bps": 54.3,
          "n": 3,
          "sum_bps": 315.7,
          "t3r_bps": 315.7,
          "tail_lt_-100_n": 0,
          "win_rate": 1.0
        }
      },
      "tau30_outside_current_v02_shadow_times": {
        "all": {
          "attempt_n": 188,
          "fill_rate": 1.0,
          "filled_n": 188,
          "max_bps": 597.9,
          "mean_bps": 12.4,
          "median_bps": 25.7,
          "min_bps": -500.2,
          "n": 188,
          "sum_bps": 2326.8,
          "t3r_bps": 1054.0,
          "tail_lt_-100_n": 33,
          "win_rate": 0.601
        },
        "attempt_n": 188,
        "cal": {
          "attempt_n": 124,
          "fill_rate": 1.0,
          "filled_n": 124,
          "max_bps": 346.2,
          "mean_bps": 3.2,
          "median_bps": 17.2,
          "min_bps": -500.2,
          "n": 124,
          "sum_bps": 401.7,
          "t3r_bps": -516.4,
          "tail_lt_-100_n": 21,
          "win_rate": 0.556
        },
        "hold": {
          "attempt_n": 64,
          "fill_rate": 1.0,
          "filled_n": 64,
          "max_bps": 597.9,
          "mean_bps": 30.1,
          "median_bps": 34.5,
          "min_bps": -449.9,
          "n": 64,
          "sum_bps": 1925.1,
          "t3r_bps": 677.6,
          "tail_lt_-100_n": 12,
          "win_rate": 0.688
        }
      },
      "tau600_all_silence": {
        "all": {
          "attempt_n": 397,
          "fill_rate": 1.0,
          "filled_n": 397,
          "max_bps": 712.4,
          "mean_bps": 5.7,
          "median_bps": 6.5,
          "min_bps": -573.6,
          "n": 397,
          "sum_bps": 2259.8,
          "t3r_bps": 573.1,
          "tail_lt_-100_n": 71,
          "win_rate": 0.539
        },
        "attempt_n": 397,
        "cal": {
          "attempt_n": 262,
          "fill_rate": 1.0,
          "filled_n": 262,
          "max_bps": 712.4,
          "mean_bps": 1.3,
          "median_bps": 1.4,
          "min_bps": -573.6,
          "n": 262,
          "sum_bps": 338.7,
          "t3r_bps": -1120.9,
          "tail_lt_-100_n": 47,
          "win_rate": 0.519
        },
        "hold": {
          "attempt_n": 135,
          "fill_rate": 1.0,
          "filled_n": 135,
          "max_bps": 581.4,
          "mean_bps": 14.2,
          "median_bps": 20.8,
          "min_bps": -477.9,
          "n": 135,
          "sum_bps": 1921.1,
          "t3r_bps": 679.4,
          "tail_lt_-100_n": 24,
          "win_rate": 0.578
        }
      },
      "tau600_inside_current_v02_shadow_times": {
        "all": {
          "attempt_n": 9,
          "fill_rate": 1.0,
          "filled_n": 9,
          "max_bps": 297.9,
          "mean_bps": 117.7,
          "median_bps": 103.3,
          "min_bps": -22.8,
          "n": 9,
          "sum_bps": 1058.9,
          "t3r_bps": 441.9,
          "tail_lt_-100_n": 0,
          "win_rate": 0.889
        },
        "attempt_n": 9,
        "cal": {
          "attempt_n": 3,
          "fill_rate": 1.0,
          "filled_n": 3,
          "max_bps": 161.5,
          "mean_bps": 77.6,
          "median_bps": 94.0,
          "min_bps": -22.8,
          "n": 3,
          "sum_bps": 232.7,
          "t3r_bps": 232.7,
          "tail_lt_-100_n": 0,
          "win_rate": 0.667
        },
        "hold": {
          "attempt_n": 6,
          "fill_rate": 1.0,
          "filled_n": 6,
          "max_bps": 297.9,
          "mean_bps": 137.7,
          "median_bps": 118.5,
          "min_bps": 32.4,
          "n": 6,
          "sum_bps": 826.2,
          "t3r_bps": 237.0,
          "tail_lt_-100_n": 0,
          "win_rate": 1.0
        }
      },
      "tau600_outside_current_v02_shadow_times": {
        "all": {
          "attempt_n": 388,
          "fill_rate": 1.0,
          "filled_n": 388,
          "max_bps": 712.4,
          "mean_bps": 3.1,
          "median_bps": 4.8,
          "min_bps": -573.6,
          "n": 388,
          "sum_bps": 1200.9,
          "t3r_bps": -485.8,
          "tail_lt_-100_n": 71,
          "win_rate": 0.531
        },
        "attempt_n": 388,
        "cal": {
          "attempt_n": 259,
          "fill_rate": 1.0,
          "filled_n": 259,
          "max_bps": 712.4,
          "mean_bps": 0.4,
          "median_bps": 1.1,
          "min_bps": -573.6,
          "n": 259,
          "sum_bps": 106.0,
          "t3r_bps": -1353.6,
          "tail_lt_-100_n": 47,
          "win_rate": 0.517
        },
        "hold": {
          "attempt_n": 129,
          "fill_rate": 1.0,
          "filled_n": 129,
          "max_bps": 581.4,
          "mean_bps": 8.5,
          "median_bps": 16.7,
          "min_bps": -477.9,
          "n": 129,
          "sum_bps": 1094.9,
          "t3r_bps": -146.8,
          "tail_lt_-100_n": 24,
          "win_rate": 0.558
        }
      },
      "tau60_all_silence": {
        "all": {
          "attempt_n": 239,
          "fill_rate": 1.0,
          "filled_n": 239,
          "max_bps": 568.3,
          "mean_bps": 8.9,
          "median_bps": 21.1,
          "min_bps": -506.3,
          "n": 239,
          "sum_bps": 2120.5,
          "t3r_bps": 871.9,
          "tail_lt_-100_n": 43,
          "win_rate": 0.582
        },
        "attempt_n": 239,
        "cal": {
          "attempt_n": 152,
          "fill_rate": 1.0,
          "filled_n": 152,
          "max_bps": 349.5,
          "mean_bps": -0.2,
          "median_bps": 14.6,
          "min_bps": -506.3,
          "n": 152,
          "sum_bps": -28.6,
          "t3r_bps": -954.6,
          "tail_lt_-100_n": 28,
          "win_rate": 0.533
        },
        "hold": {
          "attempt_n": 87,
          "fill_rate": 1.0,
          "filled_n": 87,
          "max_bps": 568.3,
          "mean_bps": 24.7,
          "median_bps": 36.6,
          "min_bps": -434.5,
          "n": 87,
          "sum_bps": 2149.1,
          "t3r_bps": 943.1,
          "tail_lt_-100_n": 15,
          "win_rate": 0.667
        }
      },
      "tau60_inside_current_v02_shadow_times": {
        "all": {
          "attempt_n": 7,
          "fill_rate": 1.0,
          "filled_n": 7,
          "max_bps": 173.5,
          "mean_bps": 95.4,
          "median_bps": 100.2,
          "min_bps": -18.5,
          "n": 7,
          "sum_bps": 667.5,
          "t3r_bps": 234.8,
          "tail_lt_-100_n": 0,
          "win_rate": 0.857
        },
        "attempt_n": 7,
        "cal": {
          "attempt_n": 3,
          "fill_rate": 1.0,
          "filled_n": 3,
          "max_bps": 151.6,
          "mean_bps": 76.3,
          "median_bps": 95.7,
          "min_bps": -18.5,
          "n": 3,
          "sum_bps": 228.8,
          "t3r_bps": 228.8,
          "tail_lt_-100_n": 0,
          "win_rate": 0.667
        },
        "hold": {
          "attempt_n": 4,
          "fill_rate": 1.0,
          "filled_n": 4,
          "max_bps": 173.5,
          "mean_bps": 109.7,
          "median_bps": 103.9,
          "min_bps": 57.4,
          "n": 4,
          "sum_bps": 438.7,
          "t3r_bps": 57.4,
          "tail_lt_-100_n": 0,
          "win_rate": 1.0
        }
      },
      "tau60_outside_current_v02_shadow_times": {
        "all": {
          "attempt_n": 232,
          "fill_rate": 1.0,
          "filled_n": 232,
          "max_bps": 568.3,
          "mean_bps": 6.3,
          "median_bps": 18.8,
          "min_bps": -506.3,
          "n": 232,
          "sum_bps": 1453.0,
          "t3r_bps": 204.4,
          "tail_lt_-100_n": 43,
          "win_rate": 0.573
        },
        "attempt_n": 232,
        "cal": {
          "attempt_n": 149,
          "fill_rate": 1.0,
          "filled_n": 149,
          "max_bps": 349.5,
          "mean_bps": -1.7,
          "median_bps": 14.3,
          "min_bps": -506.3,
          "n": 149,
          "sum_bps": -257.4,
          "t3r_bps": -1183.4,
          "tail_lt_-100_n": 28,
          "win_rate": 0.53
        },
        "hold": {
          "attempt_n": 83,
          "fill_rate": 1.0,
          "filled_n": 83,
          "max_bps": 568.3,
          "mean_bps": 20.6,
          "median_bps": 27.3,
          "min_bps": -434.5,
          "n": 83,
          "sum_bps": 1710.4,
          "t3r_bps": 504.4,
          "tail_lt_-100_n": 15,
          "win_rate": 0.651
        }
      },
      "tau900_all_silence": {
        "all": {
          "attempt_n": 387,
          "fill_rate": 1.0,
          "filled_n": 387,
          "max_bps": 742.9,
          "mean_bps": 4.3,
          "median_bps": 5.1,
          "min_bps": -601.5,
          "n": 387,
          "sum_bps": 1678.2,
          "t3r_bps": -6.8,
          "tail_lt_-100_n": 68,
          "win_rate": 0.53
        },
        "attempt_n": 387,
        "cal": {
          "attempt_n": 261,
          "fill_rate": 1.0,
          "filled_n": 261,
          "max_bps": 742.9,
          "mean_bps": 1.7,
          "median_bps": 1.4,
          "min_bps": -601.5,
          "n": 261,
          "sum_bps": 451.9,
          "t3r_bps": -1058.8,
          "tail_lt_-100_n": 46,
          "win_rate": 0.513
        },
        "hold": {
          "attempt_n": 126,
          "fill_rate": 1.0,
          "filled_n": 126,
          "max_bps": 548.4,
          "mean_bps": 9.7,
          "median_bps": 20.1,
          "min_bps": -477.4,
          "n": 126,
          "sum_bps": 1226.3,
          "t3r_bps": -65.0,
          "tail_lt_-100_n": 22,
          "win_rate": 0.563
        }
      },
      "tau900_inside_current_v02_shadow_times": {
        "all": {
          "attempt_n": 9,
          "fill_rate": 1.0,
          "filled_n": 9,
          "max_bps": 349.2,
          "mean_bps": 119.0,
          "median_bps": 83.3,
          "min_bps": -31.2,
          "n": 9,
          "sum_bps": 1071.4,
          "t3r_bps": 355.5,
          "tail_lt_-100_n": 0,
          "win_rate": 0.889
        },
        "attempt_n": 9,
        "cal": {
          "attempt_n": 3,
          "fill_rate": 1.0,
          "filled_n": 3,
          "max_bps": 223.5,
          "mean_bps": 85.1,
          "median_bps": 63.1,
          "min_bps": -31.2,
          "n": 3,
          "sum_bps": 255.4,
          "t3r_bps": 255.4,
          "tail_lt_-100_n": 0,
          "win_rate": 0.667
        },
        "hold": {
          "attempt_n": 6,
          "fill_rate": 1.0,
          "filled_n": 6,
          "max_bps": 349.2,
          "mean_bps": 136.0,
          "median_bps": 105.3,
          "min_bps": 45.5,
          "n": 6,
          "sum_bps": 816.0,
          "t3r_bps": 196.3,
          "tail_lt_-100_n": 0,
          "win_rate": 1.0
        }
      },
      "tau900_outside_current_v02_shadow_times": {
        "all": {
          "attempt_n": 378,
          "fill_rate": 1.0,
          "filled_n": 378,
          "max_bps": 742.9,
          "mean_bps": 1.6,
          "median_bps": 4.0,
          "min_bps": -601.5,
          "n": 378,
          "sum_bps": 606.8,
          "t3r_bps": -1078.2,
          "tail_lt_-100_n": 68,
          "win_rate": 0.521
        },
        "attempt_n": 378,
        "cal": {
          "attempt_n": 258,
          "fill_rate": 1.0,
          "filled_n": 258,
          "max_bps": 742.9,
          "mean_bps": 0.8,
          "median_bps": 1.4,
          "min_bps": -601.5,
          "n": 258,
          "sum_bps": 196.5,
          "t3r_bps": -1314.2,
          "tail_lt_-100_n": 46,
          "win_rate": 0.512
        },
        "hold": {
          "attempt_n": 120,
          "fill_rate": 1.0,
          "filled_n": 120,
          "max_bps": 548.4,
          "mean_bps": 3.4,
          "median_bps": 10.1,
          "min_bps": -477.4,
          "n": 120,
          "sum_bps": 410.3,
          "t3r_bps": -856.1,
          "tail_lt_-100_n": 22,
          "win_rate": 0.542
        }
      }
    },
    "live_v02_signal_count": 11,
    "split": {
      "holdout_months": [
        "2026-06"
      ],
      "method": "chronological_month_tail_35pct",
      "months": [
        "2026-02",
        "2026-03",
        "2026-04",
        "2026-06"
      ]
    }
  },
  "threshold_expansion": {
    "100000": {
      "cells": {
        "tau120_H2": {
          "all": {
            "attempt_n": 503,
            "fill_rate": 1.0,
            "filled_n": 503,
            "max_bps": 434.3,
            "mean_bps": -5.7,
            "median_bps": 4.1,
            "min_bps": -398.6,
            "n": 503,
            "sum_bps": -2850.5,
            "t3r_bps": -4032.5,
            "tail_lt_-100_n": 76,
            "win_rate": 0.513
          },
          "attempt_n": 503,
          "cal": {
            "attempt_n": 360,
            "fill_rate": 1.0,
            "filled_n": 360,
            "max_bps": 434.3,
            "mean_bps": -3.8,
            "median_bps": 6.1,
            "min_bps": -365.6,
            "n": 360,
            "sum_bps": -1357.1,
            "t3r_bps": -2539.1,
            "tail_lt_-100_n": 52,
            "win_rate": 0.519
          },
          "hold": {
            "attempt_n": 143,
            "fill_rate": 1.0,
            "filled_n": 143,
            "max_bps": 300.8,
            "mean_bps": -10.4,
            "median_bps": -0.6,
            "min_bps": -398.6,
            "n": 143,
            "sum_bps": -1493.4,
            "t3r_bps": -2320.4,
            "tail_lt_-100_n": 24,
            "win_rate": 0.497
          },
          "pass_all": false,
          "pass_hold": false
        },
        "tau120_H4": {
          "all": {
            "attempt_n": 503,
            "fill_rate": 1.0,
            "filled_n": 503,
            "max_bps": 539.6,
            "mean_bps": -8.4,
            "median_bps": -3.3,
            "min_bps": -523.5,
            "n": 503,
            "sum_bps": -4215.8,
            "t3r_bps": -5757.8,
            "tail_lt_-100_n": 98,
            "win_rate": 0.489
          },
          "attempt_n": 503,
          "cal": {
            "attempt_n": 360,
            "fill_rate": 1.0,
            "filled_n": 360,
            "max_bps": 522.3,
            "mean_bps": -9.5,
            "median_bps": -4.8,
            "min_bps": -505.7,
            "n": 360,
            "sum_bps": -3432.1,
            "t3r_bps": -4738.0,
            "tail_lt_-100_n": 73,
            "win_rate": 0.483
          },
          "hold": {
            "attempt_n": 143,
            "fill_rate": 1.0,
            "filled_n": 143,
            "max_bps": 539.6,
            "mean_bps": -5.5,
            "median_bps": 0.8,
            "min_bps": -523.5,
            "n": 143,
            "sum_bps": -783.7,
            "t3r_bps": -2192.0,
            "tail_lt_-100_n": 25,
            "win_rate": 0.503
          },
          "pass_all": false,
          "pass_hold": false
        },
        "tau300_H2": {
          "all": {
            "attempt_n": 651,
            "fill_rate": 0.998,
            "filled_n": 650,
            "max_bps": 526.8,
            "mean_bps": -4.1,
            "median_bps": 1.0,
            "min_bps": -385.7,
            "n": 650,
            "sum_bps": -2648.2,
            "t3r_bps": -4034.7,
            "tail_lt_-100_n": 92,
            "win_rate": 0.502
          },
          "attempt_n": 651,
          "cal": {
            "attempt_n": 460,
            "fill_rate": 1.0,
            "filled_n": 460,
            "max_bps": 526.8,
            "mean_bps": -1.2,
            "median_bps": 2.9,
            "min_bps": -353.5,
            "n": 460,
            "sum_bps": -568.1,
            "t3r_bps": -1954.6,
            "tail_lt_-100_n": 64,
            "win_rate": 0.509
          },
          "hold": {
            "attempt_n": 191,
            "fill_rate": 0.995,
            "filled_n": 190,
            "max_bps": 277.6,
            "mean_bps": -10.9,
            "median_bps": -1.2,
            "min_bps": -385.7,
            "n": 190,
            "sum_bps": -2080.1,
            "t3r_bps": -2886.5,
            "tail_lt_-100_n": 28,
            "win_rate": 0.484
          },
          "pass_all": false,
          "pass_hold": false
        },
        "tau300_H4": {
          "all": {
            "attempt_n": 651,
            "fill_rate": 0.998,
            "filled_n": 650,
            "max_bps": 558.0,
            "mean_bps": -7.8,
            "median_bps": -0.5,
            "min_bps": -561.2,
            "n": 650,
            "sum_bps": -5069.9,
            "t3r_bps": -6596.0,
            "tail_lt_-100_n": 128,
            "win_rate": 0.497
          },
          "attempt_n": 651,
          "cal": {
            "attempt_n": 460,
            "fill_rate": 1.0,
            "filled_n": 460,
            "max_bps": 515.8,
            "mean_bps": -10.8,
            "median_bps": -1.0,
            "min_bps": -561.2,
            "n": 460,
            "sum_bps": -4986.3,
            "t3r_bps": -6312.8,
            "tail_lt_-100_n": 94,
            "win_rate": 0.496
          },
          "hold": {
            "attempt_n": 191,
            "fill_rate": 0.995,
            "filled_n": 190,
            "max_bps": 558.0,
            "mean_bps": -0.4,
            "median_bps": 0.8,
            "min_bps": -477.6,
            "n": 190,
            "sum_bps": -83.6,
            "t3r_bps": -1524.6,
            "tail_lt_-100_n": 34,
            "win_rate": 0.5
          },
          "pass_all": false,
          "pass_hold": false
        },
        "tau30_H2": {
          "all": {
            "attempt_n": 306,
            "fill_rate": 1.0,
            "filled_n": 306,
            "max_bps": 415.3,
            "mean_bps": 1.9,
            "median_bps": 6.9,
            "min_bps": -455.0,
            "n": 306,
            "sum_bps": 573.6,
            "t3r_bps": -481.6,
            "tail_lt_-100_n": 37,
            "win_rate": 0.533
          },
          "attempt_n": 306,
          "cal": {
            "attempt_n": 214,
            "fill_rate": 1.0,
            "filled_n": 214,
            "max_bps": 415.3,
            "mean_bps": 6.7,
            "median_bps": 11.1,
            "min_bps": -372.9,
            "n": 214,
            "sum_bps": 1423.7,
            "t3r_bps": 374.1,
            "tail_lt_-100_n": 24,
            "win_rate": 0.547
          },
          "hold": {
            "attempt_n": 92,
            "fill_rate": 1.0,
            "filled_n": 92,
            "max_bps": 314.5,
            "mean_bps": -9.2,
            "median_bps": -3.0,
            "min_bps": -455.0,
            "n": 92,
            "sum_bps": -850.1,
            "t3r_bps": -1696.7,
            "tail_lt_-100_n": 13,
            "win_rate": 0.5
          },
          "pass_all": false,
          "pass_hold": false
        },
        "tau30_H4": {
          "all": {
            "attempt_n": 306,
            "fill_rate": 1.0,
            "filled_n": 306,
            "max_bps": 597.9,
            "mean_bps": 6.1,
            "median_bps": 0.4,
            "min_bps": -525.2,
            "n": 306,
            "sum_bps": 1874.0,
            "t3r_bps": 363.1,
            "tail_lt_-100_n": 52,
            "win_rate": 0.5
          },
          "attempt_n": 306,
          "cal": {
            "attempt_n": 214,
            "fill_rate": 1.0,
            "filled_n": 214,
            "max_bps": 506.0,
            "mean_bps": 5.7,
            "median_bps": -3.0,
            "min_bps": -394.4,
            "n": 214,
            "sum_bps": 1215.9,
            "t3r_bps": -97.0,
            "tail_lt_-100_n": 36,
            "win_rate": 0.486
          },
          "hold": {
            "attempt_n": 92,
            "fill_rate": 1.0,
            "filled_n": 92,
            "max_bps": 597.9,
            "mean_bps": 7.2,
            "median_bps": 8.8,
            "min_bps": -525.2,
            "n": 92,
            "sum_bps": 658.1,
            "t3r_bps": -690.4,
            "tail_lt_-100_n": 16,
            "win_rate": 0.533
          },
          "pass_all": true,
          "pass_hold": false
        },
        "tau600_H2": {
          "all": {
            "attempt_n": 676,
            "fill_rate": 0.999,
            "filled_n": 675,
            "max_bps": 584.7,
            "mean_bps": -4.0,
            "median_bps": 0.9,
            "min_bps": -499.7,
            "n": 675,
            "sum_bps": -2681.4,
            "t3r_bps": -4101.5,
            "tail_lt_-100_n": 94,
            "win_rate": 0.505
          },
          "attempt_n": 676,
          "cal": {
            "attempt_n": 483,
            "fill_rate": 1.0,
            "filled_n": 483,
            "max_bps": 584.7,
            "mean_bps": -1.9,
            "median_bps": 1.8,
            "min_bps": -370.4,
            "n": 483,
            "sum_bps": -902.0,
            "t3r_bps": -2322.1,
            "tail_lt_-100_n": 66,
            "win_rate": 0.513
          },
          "hold": {
            "attempt_n": 193,
            "fill_rate": 0.995,
            "filled_n": 192,
            "max_bps": 276.6,
            "mean_bps": -9.3,
            "median_bps": -3.0,
            "min_bps": -499.7,
            "n": 192,
            "sum_bps": -1779.4,
            "t3r_bps": -2585.9,
            "tail_lt_-100_n": 28,
            "win_rate": 0.484
          },
          "pass_all": false,
          "pass_hold": false
        },
        "tau600_H4": {
          "all": {
            "attempt_n": 676,
            "fill_rate": 0.999,
            "filled_n": 675,
            "max_bps": 712.4,
            "mean_bps": -7.7,
            "median_bps": -2.7,
            "min_bps": -580.9,
            "n": 675,
            "sum_bps": -5204.3,
            "t3r_bps": -6935.7,
            "tail_lt_-100_n": 132,
            "win_rate": 0.483
          },
          "attempt_n": 676,
          "cal": {
            "attempt_n": 483,
            "fill_rate": 1.0,
            "filled_n": 483,
            "max_bps": 712.4,
            "mean_bps": -11.3,
            "median_bps": -6.2,
            "min_bps": -580.9,
            "n": 483,
            "sum_bps": -5477.2,
            "t3r_bps": -6998.3,
            "tail_lt_-100_n": 98,
            "win_rate": 0.472
          },
          "hold": {
            "attempt_n": 193,
            "fill_rate": 0.995,
            "filled_n": 192,
            "max_bps": 581.4,
            "mean_bps": 1.4,
            "median_bps": 6.9,
            "min_bps": -487.8,
            "n": 192,
            "sum_bps": 272.9,
            "t3r_bps": -1160.7,
            "tail_lt_-100_n": 34,
            "win_rate": 0.51
          },
          "pass_all": false,
          "pass_hold": false
        },
        "tau60_H2": {
          "all": {
            "attempt_n": 391,
            "fill_rate": 1.0,
            "filled_n": 391,
            "max_bps": 429.4,
            "mean_bps": -2.9,
            "median_bps": 1.3,
            "min_bps": -475.7,
            "n": 391,
            "sum_bps": -1130.8,
            "t3r_bps": -2203.2,
            "tail_lt_-100_n": 52,
            "win_rate": 0.512
          },
          "attempt_n": 391,
          "cal": {
            "attempt_n": 282,
            "fill_rate": 1.0,
            "filled_n": 282,
            "max_bps": 429.4,
            "mean_bps": 0.4,
            "median_bps": 1.5,
            "min_bps": -378.9,
            "n": 282,
            "sum_bps": 122.9,
            "t3r_bps": -941.8,
            "tail_lt_-100_n": 35,
            "win_rate": 0.511
          },
          "hold": {
            "attempt_n": 109,
            "fill_rate": 1.0,
            "filled_n": 109,
            "max_bps": 316.6,
            "mean_bps": -11.5,
            "median_bps": 1.1,
            "min_bps": -475.7,
            "n": 109,
            "sum_bps": -1253.7,
            "t3r_bps": -2089.6,
            "tail_lt_-100_n": 17,
            "win_rate": 0.514
          },
          "pass_all": false,
          "pass_hold": false
        },
        "tau60_H4": {
          "all": {
            "attempt_n": 391,
            "fill_rate": 1.0,
            "filled_n": 391,
            "max_bps": 568.3,
            "mean_bps": -0.8,
            "median_bps": -1.5,
            "min_bps": -539.5,
            "n": 391,
            "sum_bps": -304.5,
            "t3r_bps": -1853.5,
            "tail_lt_-100_n": 70,
            "win_rate": 0.494
          },
          "attempt_n": 391,
          "cal": {
            "attempt_n": 282,
            "fill_rate": 1.0,
            "filled_n": 282,
            "max_bps": 495.8,
            "mean_bps": -3.4,
            "median_bps": -4.5,
            "min_bps": -391.3,
            "n": 282,
            "sum_bps": -952.3,
            "t3r_bps": -2226.9,
            "tail_lt_-100_n": 53,
            "win_rate": 0.472
          },
          "hold": {
            "attempt_n": 109,
            "fill_rate": 1.0,
            "filled_n": 109,
            "max_bps": 568.3,
            "mean_bps": 5.9,
            "median_bps": 9.1,
            "min_bps": -539.5,
            "n": 109,
            "sum_bps": 647.8,
            "t3r_bps": -793.6,
            "tail_lt_-100_n": 17,
            "win_rate": 0.55
          },
          "pass_all": false,
          "pass_hold": false
        },
        "tau900_H2": {
          "all": {
            "attempt_n": 679,
            "fill_rate": 0.999,
            "filled_n": 678,
            "max_bps": 416.4,
            "mean_bps": -5.9,
            "median_bps": -2.6,
            "min_bps": -446.0,
            "n": 678,
            "sum_bps": -3982.4,
            "t3r_bps": -5125.7,
            "tail_lt_-100_n": 101,
            "win_rate": 0.487
          },
          "attempt_n": 679,
          "cal": {
            "attempt_n": 486,
            "fill_rate": 1.0,
            "filled_n": 486,
            "max_bps": 416.4,
            "mean_bps": -4.2,
            "median_bps": -1.1,
            "min_bps": -356.8,
            "n": 486,
            "sum_bps": -2021.0,
            "t3r_bps": -3164.3,
            "tail_lt_-100_n": 70,
            "win_rate": 0.496
          },
          "hold": {
            "attempt_n": 193,
            "fill_rate": 0.995,
            "filled_n": 192,
            "max_bps": 272.2,
            "mean_bps": -10.2,
            "median_bps": -4.7,
            "min_bps": -446.0,
            "n": 192,
            "sum_bps": -1961.4,
            "t3r_bps": -2758.7,
            "tail_lt_-100_n": 31,
            "win_rate": 0.464
          },
          "pass_all": false,
          "pass_hold": false
        },
        "tau900_H4": {
          "all": {
            "attempt_n": 679,
            "fill_rate": 0.999,
            "filled_n": 678,
            "max_bps": 742.9,
            "mean_bps": -7.8,
            "median_bps": -4.1,
            "min_bps": -588.6,
            "n": 678,
            "sum_bps": -5280.8,
            "t3r_bps": -7022.1,
            "tail_lt_-100_n": 136,
            "win_rate": 0.482
          },
          "attempt_n": 679,
          "cal": {
            "attempt_n": 486,
            "fill_rate": 1.0,
            "filled_n": 486,
            "max_bps": 742.9,
            "mean_bps": -10.3,
            "median_bps": -8.1,
            "min_bps": -588.6,
            "n": 486,
            "sum_bps": -5018.4,
            "t3r_bps": -6629.0,
            "tail_lt_-100_n": 103,
            "win_rate": 0.463
          },
          "hold": {
            "attempt_n": 193,
            "fill_rate": 0.995,
            "filled_n": 192,
            "max_bps": 548.4,
            "mean_bps": -1.4,
            "median_bps": 8.7,
            "min_bps": -483.4,
            "n": 192,
            "sum_bps": -262.4,
            "t3r_bps": -1706.9,
            "tail_lt_-100_n": 33,
            "win_rate": 0.531
          },
          "pass_all": false,
          "pass_hold": false
        }
      },
      "event_n": 1892,
      "split": {
        "holdout_months": [
          "2026-06"
        ],
        "method": "chronological_month_tail_35pct",
        "months": [
          "2026-02",
          "2026-03",
          "2026-04",
          "2026-06"
        ]
      }
    },
    "150000": {
      "cells": {
        "tau120_H2": {
          "all": {
            "attempt_n": 361,
            "fill_rate": 0.997,
            "filled_n": 360,
            "max_bps": 445.6,
            "mean_bps": -0.7,
            "median_bps": 8.3,
            "min_bps": -417.3,
            "n": 360,
            "sum_bps": -264.4,
            "t3r_bps": -1457.7,
            "tail_lt_-100_n": 53,
            "win_rate": 0.55
          },
          "attempt_n": 361,
          "cal": {
            "attempt_n": 252,
            "fill_rate": 1.0,
            "filled_n": 252,
            "max_bps": 445.6,
            "mean_bps": 5.5,
            "median_bps": 12.9,
            "min_bps": -365.6,
            "n": 252,
            "sum_bps": 1381.3,
            "t3r_bps": 188.0,
            "tail_lt_-100_n": 36,
            "win_rate": 0.567
          },
          "hold": {
            "attempt_n": 109,
            "fill_rate": 0.991,
            "filled_n": 108,
            "max_bps": 268.2,
            "mean_bps": -15.2,
            "median_bps": 0.8,
            "min_bps": -417.3,
            "n": 108,
            "sum_bps": -1645.7,
            "t3r_bps": -2380.0,
            "tail_lt_-100_n": 17,
            "win_rate": 0.509
          },
          "pass_all": false,
          "pass_hold": false
        },
        "tau120_H4": {
          "all": {
            "attempt_n": 361,
            "fill_rate": 0.997,
            "filled_n": 360,
            "max_bps": 539.6,
            "mean_bps": -0.3,
            "median_bps": 10.0,
            "min_bps": -505.7,
            "n": 360,
            "sum_bps": -101.1,
            "t3r_bps": -1650.6,
            "tail_lt_-100_n": 60,
            "win_rate": 0.525
          },
          "attempt_n": 361,
          "cal": {
            "attempt_n": 252,
            "fill_rate": 1.0,
            "filled_n": 252,
            "max_bps": 522.3,
            "mean_bps": -1.4,
            "median_bps": 8.3,
            "min_bps": -505.7,
            "n": 252,
            "sum_bps": -362.7,
            "t3r_bps": -1633.3,
            "tail_lt_-100_n": 44,
            "win_rate": 0.516
          },
          "hold": {
            "attempt_n": 109,
            "fill_rate": 0.991,
            "filled_n": 108,
            "max_bps": 539.6,
            "mean_bps": 2.4,
            "median_bps": 12.9,
            "min_bps": -502.7,
            "n": 108,
            "sum_bps": 261.6,
            "t3r_bps": -1100.5,
            "tail_lt_-100_n": 16,
            "win_rate": 0.546
          },
          "pass_all": false,
          "pass_hold": false
        },
        "tau300_H2": {
          "all": {
            "attempt_n": 488,
            "fill_rate": 0.998,
            "filled_n": 487,
            "max_bps": 545.0,
            "mean_bps": -0.6,
            "median_bps": 4.5,
            "min_bps": -399.1,
            "n": 487,
            "sum_bps": -274.1,
            "t3r_bps": -1678.3,
            "tail_lt_-100_n": 66,
            "win_rate": 0.522
          },
          "attempt_n": 488,
          "cal": {
            "attempt_n": 328,
            "fill_rate": 1.0,
            "filled_n": 328,
            "max_bps": 545.0,
            "mean_bps": 7.1,
            "median_bps": 8.9,
            "min_bps": -352.1,
            "n": 328,
            "sum_bps": 2328.7,
            "t3r_bps": 924.5,
            "tail_lt_-100_n": 42,
            "win_rate": 0.546
          },
          "hold": {
            "attempt_n": 160,
            "fill_rate": 0.994,
            "filled_n": 159,
            "max_bps": 277.6,
            "mean_bps": -16.4,
            "median_bps": -4.2,
            "min_bps": -399.1,
            "n": 159,
            "sum_bps": -2602.8,
            "t3r_bps": -3405.3,
            "tail_lt_-100_n": 24,
            "win_rate": 0.472
          },
          "pass_all": false,
          "pass_hold": false
        },
        "tau300_H4": {
          "all": {
            "attempt_n": 488,
            "fill_rate": 0.998,
            "filled_n": 487,
            "max_bps": 558.0,
            "mean_bps": -3.4,
            "median_bps": 8.8,
            "min_bps": -561.2,
            "n": 487,
            "sum_bps": -1660.1,
            "t3r_bps": -3152.0,
            "tail_lt_-100_n": 93,
            "win_rate": 0.524
          },
          "attempt_n": 488,
          "cal": {
            "attempt_n": 328,
            "fill_rate": 1.0,
            "filled_n": 328,
            "max_bps": 515.8,
            "mean_bps": -1.6,
            "median_bps": 8.4,
            "min_bps": -561.2,
            "n": 328,
            "sum_bps": -530.6,
            "t3r_bps": -1825.6,
            "tail_lt_-100_n": 64,
            "win_rate": 0.527
          },
          "hold": {
            "attempt_n": 160,
            "fill_rate": 0.994,
            "filled_n": 159,
            "max_bps": 558.0,
            "mean_bps": -7.1,
            "median_bps": 9.2,
            "min_bps": -504.7,
            "n": 159,
            "sum_bps": -1129.5,
            "t3r_bps": -2451.3,
            "tail_lt_-100_n": 29,
            "win_rate": 0.516
          },
          "pass_all": false,
          "pass_hold": false
        },
        "tau30_H2": {
          "all": {
            "attempt_n": 223,
            "fill_rate": 0.996,
            "filled_n": 222,
            "max_bps": 415.3,
            "mean_bps": 9.4,
            "median_bps": 11.3,
            "min_bps": -468.2,
            "n": 222,
            "sum_bps": 2077.0,
            "t3r_bps": 1027.4,
            "tail_lt_-100_n": 25,
            "win_rate": 0.559
          },
          "attempt_n": 223,
          "cal": {
            "attempt_n": 158,
            "fill_rate": 1.0,
            "filled_n": 158,
            "max_bps": 415.3,
            "mean_bps": 16.9,
            "median_bps": 14.8,
            "min_bps": -372.9,
            "n": 158,
            "sum_bps": 2666.5,
            "t3r_bps": 1616.9,
            "tail_lt_-100_n": 14,
            "win_rate": 0.582
          },
          "hold": {
            "attempt_n": 65,
            "fill_rate": 0.985,
            "filled_n": 64,
            "max_bps": 285.0,
            "mean_bps": -9.2,
            "median_bps": -3.5,
            "min_bps": -468.2,
            "n": 64,
            "sum_bps": -589.5,
            "t3r_bps": -1331.9,
            "tail_lt_-100_n": 11,
            "win_rate": 0.5
          },
          "pass_all": true,
          "pass_hold": false
        },
        "tau30_H4": {
          "all": {
            "attempt_n": 223,
            "fill_rate": 0.996,
            "filled_n": 222,
            "max_bps": 597.9,
            "mean_bps": 11.8,
            "median_bps": 12.9,
            "min_bps": -449.9,
            "n": 222,
            "sum_bps": 2628.6,
            "t3r_bps": 1117.7,
            "tail_lt_-100_n": 36,
            "win_rate": 0.545
          },
          "attempt_n": 223,
          "cal": {
            "attempt_n": 158,
            "fill_rate": 1.0,
            "filled_n": 158,
            "max_bps": 506.0,
            "mean_bps": 12.9,
            "median_bps": 4.8,
            "min_bps": -394.4,
            "n": 158,
            "sum_bps": 2031.7,
            "t3r_bps": 753.5,
            "tail_lt_-100_n": 24,
            "win_rate": 0.519
          },
          "hold": {
            "attempt_n": 65,
            "fill_rate": 0.985,
            "filled_n": 64,
            "max_bps": 597.9,
            "mean_bps": 9.3,
            "median_bps": 24.3,
            "min_bps": -449.9,
            "n": 64,
            "sum_bps": 596.9,
            "t3r_bps": -655.9,
            "tail_lt_-100_n": 12,
            "win_rate": 0.609
          },
          "pass_all": true,
          "pass_hold": false
        },
        "tau600_H2": {
          "all": {
            "attempt_n": 494,
            "fill_rate": 0.998,
            "filled_n": 493,
            "max_bps": 580.3,
            "mean_bps": 0.0,
            "median_bps": 1.8,
            "min_bps": -458.7,
            "n": 493,
            "sum_bps": 23.4,
            "t3r_bps": -1388.7,
            "tail_lt_-100_n": 63,
            "win_rate": 0.513
          },
          "attempt_n": 494,
          "cal": {
            "attempt_n": 342,
            "fill_rate": 1.0,
            "filled_n": 342,
            "max_bps": 580.3,
            "mean_bps": 4.1,
            "median_bps": 2.5,
            "min_bps": -370.4,
            "n": 342,
            "sum_bps": 1416.5,
            "t3r_bps": 4.4,
            "tail_lt_-100_n": 44,
            "win_rate": 0.52
          },
          "hold": {
            "attempt_n": 152,
            "fill_rate": 0.993,
            "filled_n": 151,
            "max_bps": 274.9,
            "mean_bps": -9.2,
            "median_bps": -2.7,
            "min_bps": -458.7,
            "n": 151,
            "sum_bps": -1393.1,
            "t3r_bps": -2128.3,
            "tail_lt_-100_n": 19,
            "win_rate": 0.497
          },
          "pass_all": false,
          "pass_hold": false
        },
        "tau600_H4": {
          "all": {
            "attempt_n": 494,
            "fill_rate": 0.998,
            "filled_n": 493,
            "max_bps": 712.4,
            "mean_bps": -0.3,
            "median_bps": 1.7,
            "min_bps": -580.9,
            "n": 493,
            "sum_bps": -133.1,
            "t3r_bps": -1847.2,
            "tail_lt_-100_n": 88,
            "win_rate": 0.509
          },
          "attempt_n": 494,
          "cal": {
            "attempt_n": 342,
            "fill_rate": 1.0,
            "filled_n": 342,
            "max_bps": 712.4,
            "mean_bps": -1.6,
            "median_bps": -1.5,
            "min_bps": -580.9,
            "n": 342,
            "sum_bps": -554.2,
            "t3r_bps": -2054.2,
            "tail_lt_-100_n": 61,
            "win_rate": 0.494
          },
          "hold": {
            "attempt_n": 152,
            "fill_rate": 0.993,
            "filled_n": 151,
            "max_bps": 581.4,
            "mean_bps": 2.8,
            "median_bps": 14.7,
            "min_bps": -549.0,
            "n": 151,
            "sum_bps": 421.1,
            "t3r_bps": -935.1,
            "tail_lt_-100_n": 27,
            "win_rate": 0.543
          },
          "pass_all": false,
          "pass_hold": false
        },
        "tau60_H2": {
          "all": {
            "attempt_n": 284,
            "fill_rate": 0.996,
            "filled_n": 283,
            "max_bps": 429.4,
            "mean_bps": 5.1,
            "median_bps": 7.2,
            "min_bps": -466.0,
            "n": 283,
            "sum_bps": 1432.7,
            "t3r_bps": 272.8,
            "tail_lt_-100_n": 40,
            "win_rate": 0.548
          },
          "attempt_n": 284,
          "cal": {
            "attempt_n": 197,
            "fill_rate": 1.0,
            "filled_n": 197,
            "max_bps": 429.4,
            "mean_bps": 13.7,
            "median_bps": 12.3,
            "min_bps": -378.9,
            "n": 197,
            "sum_bps": 2693.1,
            "t3r_bps": 1533.2,
            "tail_lt_-100_n": 25,
            "win_rate": 0.558
          },
          "hold": {
            "attempt_n": 87,
            "fill_rate": 0.989,
            "filled_n": 86,
            "max_bps": 275.6,
            "mean_bps": -14.7,
            "median_bps": 1.3,
            "min_bps": -466.0,
            "n": 86,
            "sum_bps": -1260.4,
            "t3r_bps": -1984.0,
            "tail_lt_-100_n": 15,
            "win_rate": 0.523
          },
          "pass_all": true,
          "pass_hold": false
        },
        "tau60_H4": {
          "all": {
            "attempt_n": 284,
            "fill_rate": 0.996,
            "filled_n": 283,
            "max_bps": 568.3,
            "mean_bps": 7.0,
            "median_bps": 8.0,
            "min_bps": -462.8,
            "n": 283,
            "sum_bps": 1968.0,
            "t3r_bps": 424.3,
            "tail_lt_-100_n": 51,
            "win_rate": 0.523
          },
          "attempt_n": 284,
          "cal": {
            "attempt_n": 197,
            "fill_rate": 1.0,
            "filled_n": 197,
            "max_bps": 495.8,
            "mean_bps": 7.8,
            "median_bps": 4.8,
            "min_bps": -427.4,
            "n": 197,
            "sum_bps": 1545.9,
            "t3r_bps": 254.0,
            "tail_lt_-100_n": 35,
            "win_rate": 0.503
          },
          "hold": {
            "attempt_n": 87,
            "fill_rate": 0.989,
            "filled_n": 86,
            "max_bps": 568.3,
            "mean_bps": 4.9,
            "median_bps": 13.2,
            "min_bps": -462.8,
            "n": 86,
            "sum_bps": 422.1,
            "t3r_bps": -956.6,
            "tail_lt_-100_n": 16,
            "win_rate": 0.57
          },
          "pass_all": true,
          "pass_hold": false
        },
        "tau900_H2": {
          "all": {
            "attempt_n": 488,
            "fill_rate": 0.998,
            "filled_n": 487,
            "max_bps": 415.2,
            "mean_bps": -1.3,
            "median_bps": 1.3,
            "min_bps": -453.4,
            "n": 487,
            "sum_bps": -640.5,
            "t3r_bps": -1773.5,
            "tail_lt_-100_n": 71,
            "win_rate": 0.505
          },
          "attempt_n": 488,
          "cal": {
            "attempt_n": 342,
            "fill_rate": 1.0,
            "filled_n": 342,
            "max_bps": 415.2,
            "mean_bps": 1.6,
            "median_bps": 1.5,
            "min_bps": -356.8,
            "n": 342,
            "sum_bps": 549.0,
            "t3r_bps": -584.0,
            "tail_lt_-100_n": 50,
            "win_rate": 0.509
          },
          "hold": {
            "attempt_n": 146,
            "fill_rate": 0.993,
            "filled_n": 145,
            "max_bps": 264.4,
            "mean_bps": -8.2,
            "median_bps": -2.2,
            "min_bps": -453.4,
            "n": 145,
            "sum_bps": -1189.5,
            "t3r_bps": -1882.7,
            "tail_lt_-100_n": 21,
            "win_rate": 0.497
          },
          "pass_all": false,
          "pass_hold": false
        },
        "tau900_H4": {
          "all": {
            "attempt_n": 488,
            "fill_rate": 0.998,
            "filled_n": 487,
            "max_bps": 742.9,
            "mean_bps": 1.2,
            "median_bps": 2.7,
            "min_bps": -588.6,
            "n": 487,
            "sum_bps": 594.0,
            "t3r_bps": -1152.0,
            "tail_lt_-100_n": 89,
            "win_rate": 0.515
          },
          "attempt_n": 488,
          "cal": {
            "attempt_n": 342,
            "fill_rate": 1.0,
            "filled_n": 342,
            "max_bps": 742.9,
            "mean_bps": -0.9,
            "median_bps": -0.8,
            "min_bps": -588.6,
            "n": 342,
            "sum_bps": -298.7,
            "t3r_bps": -1869.1,
            "tail_lt_-100_n": 66,
            "win_rate": 0.497
          },
          "hold": {
            "attempt_n": 146,
            "fill_rate": 0.993,
            "filled_n": 145,
            "max_bps": 548.4,
            "mean_bps": 6.2,
            "median_bps": 13.6,
            "min_bps": -483.0,
            "n": 145,
            "sum_bps": 892.7,
            "t3r_bps": -483.2,
            "tail_lt_-100_n": 23,
            "win_rate": 0.559
          },
          "pass_all": false,
          "pass_hold": false
        }
      },
      "event_n": 1421,
      "split": {
        "holdout_months": [
          "2026-06"
        ],
        "method": "chronological_month_tail_35pct",
        "months": [
          "2026-02",
          "2026-03",
          "2026-04",
          "2026-06"
        ]
      }
    },
    "200000": {
      "cells": {
        "tau120_H2": {
          "all": {
            "attempt_n": 310,
            "fill_rate": 1.0,
            "filled_n": 310,
            "max_bps": 440.6,
            "mean_bps": -3.8,
            "median_bps": 9.3,
            "min_bps": -417.3,
            "n": 310,
            "sum_bps": -1181.7,
            "t3r_bps": -2176.4,
            "tail_lt_-100_n": 46,
            "win_rate": 0.539
          },
          "attempt_n": 310,
          "cal": {
            "attempt_n": 204,
            "fill_rate": 1.0,
            "filled_n": 204,
            "max_bps": 440.6,
            "mean_bps": -2.1,
            "median_bps": 13.9,
            "min_bps": -365.6,
            "n": 204,
            "sum_bps": -432.0,
            "t3r_bps": -1419.4,
            "tail_lt_-100_n": 30,
            "win_rate": 0.549
          },
          "hold": {
            "attempt_n": 106,
            "fill_rate": 1.0,
            "filled_n": 106,
            "max_bps": 268.2,
            "mean_bps": -7.1,
            "median_bps": 2.9,
            "min_bps": -417.3,
            "n": 106,
            "sum_bps": -749.7,
            "t3r_bps": -1484.0,
            "tail_lt_-100_n": 16,
            "win_rate": 0.519
          },
          "pass_all": false,
          "pass_hold": false
        },
        "tau120_H4": {
          "all": {
            "attempt_n": 310,
            "fill_rate": 1.0,
            "filled_n": 310,
            "max_bps": 539.6,
            "mean_bps": -2.0,
            "median_bps": 15.8,
            "min_bps": -509.4,
            "n": 310,
            "sum_bps": -615.6,
            "t3r_bps": -1843.3,
            "tail_lt_-100_n": 58,
            "win_rate": 0.558
          },
          "attempt_n": 310,
          "cal": {
            "attempt_n": 204,
            "fill_rate": 1.0,
            "filled_n": 204,
            "max_bps": 353.2,
            "mean_bps": -11.4,
            "median_bps": 9.2,
            "min_bps": -509.4,
            "n": 204,
            "sum_bps": -2316.2,
            "t3r_bps": -3300.2,
            "tail_lt_-100_n": 41,
            "win_rate": 0.529
          },
          "hold": {
            "attempt_n": 106,
            "fill_rate": 1.0,
            "filled_n": 106,
            "max_bps": 539.6,
            "mean_bps": 16.0,
            "median_bps": 24.2,
            "min_bps": -439.2,
            "n": 106,
            "sum_bps": 1700.6,
            "t3r_bps": 536.1,
            "tail_lt_-100_n": 17,
            "win_rate": 0.613
          },
          "pass_all": false,
          "pass_hold": true
        },
        "tau300_H2": {
          "all": {
            "attempt_n": 400,
            "fill_rate": 1.0,
            "filled_n": 400,
            "max_bps": 443.6,
            "mean_bps": -2.5,
            "median_bps": 5.4,
            "min_bps": -376.6,
            "n": 400,
            "sum_bps": -1001.7,
            "t3r_bps": -2126.4,
            "tail_lt_-100_n": 55,
            "win_rate": 0.527
          },
          "attempt_n": 400,
          "cal": {
            "attempt_n": 255,
            "fill_rate": 1.0,
            "filled_n": 255,
            "max_bps": 443.6,
            "mean_bps": 0.1,
            "median_bps": 6.7,
            "min_bps": -351.5,
            "n": 255,
            "sum_bps": 33.6,
            "t3r_bps": -1091.1,
            "tail_lt_-100_n": 36,
            "win_rate": 0.545
          },
          "hold": {
            "attempt_n": 145,
            "fill_rate": 1.0,
            "filled_n": 145,
            "max_bps": 276.8,
            "mean_bps": -7.1,
            "median_bps": -0.1,
            "min_bps": -376.6,
            "n": 145,
            "sum_bps": -1035.3,
            "t3r_bps": -1837.0,
            "tail_lt_-100_n": 19,
            "win_rate": 0.497
          },
          "pass_all": false,
          "pass_hold": false
        },
        "tau300_H4": {
          "all": {
            "attempt_n": 400,
            "fill_rate": 1.0,
            "filled_n": 400,
            "max_bps": 558.0,
            "mean_bps": -1.6,
            "median_bps": 11.2,
            "min_bps": -567.9,
            "n": 400,
            "sum_bps": -636.8,
            "t3r_bps": -1935.0,
            "tail_lt_-100_n": 80,
            "win_rate": 0.545
          },
          "attempt_n": 400,
          "cal": {
            "attempt_n": 255,
            "fill_rate": 1.0,
            "filled_n": 255,
            "max_bps": 377.0,
            "mean_bps": -7.0,
            "median_bps": 9.1,
            "min_bps": -567.9,
            "n": 255,
            "sum_bps": -1783.4,
            "t3r_bps": -2869.6,
            "tail_lt_-100_n": 55,
            "win_rate": 0.533
          },
          "hold": {
            "attempt_n": 145,
            "fill_rate": 1.0,
            "filled_n": 145,
            "max_bps": 558.0,
            "mean_bps": 7.9,
            "median_bps": 18.6,
            "min_bps": -443.2,
            "n": 145,
            "sum_bps": 1146.6,
            "t3r_bps": -64.6,
            "tail_lt_-100_n": 25,
            "win_rate": 0.566
          },
          "pass_all": false,
          "pass_hold": false
        },
        "tau30_H2": {
          "all": {
            "attempt_n": 193,
            "fill_rate": 1.0,
            "filled_n": 193,
            "max_bps": 444.3,
            "mean_bps": 8.8,
            "median_bps": 15.8,
            "min_bps": -372.9,
            "n": 193,
            "sum_bps": 1691.4,
            "t3r_bps": 623.7,
            "tail_lt_-100_n": 24,
            "win_rate": 0.58
          },
          "attempt_n": 193,
          "cal": {
            "attempt_n": 126,
            "fill_rate": 1.0,
            "filled_n": 126,
            "max_bps": 444.3,
            "mean_bps": 13.6,
            "median_bps": 20.6,
            "min_bps": -372.9,
            "n": 126,
            "sum_bps": 1708.3,
            "t3r_bps": 640.6,
            "tail_lt_-100_n": 13,
            "win_rate": 0.611
          },
          "hold": {
            "attempt_n": 67,
            "fill_rate": 1.0,
            "filled_n": 67,
            "max_bps": 285.0,
            "mean_bps": -0.3,
            "median_bps": 5.4,
            "min_bps": -357.0,
            "n": 67,
            "sum_bps": -16.9,
            "t3r_bps": -759.3,
            "tail_lt_-100_n": 11,
            "win_rate": 0.522
          },
          "pass_all": true,
          "pass_hold": false
        },
        "tau30_H4": {
          "all": {
            "attempt_n": 193,
            "fill_rate": 1.0,
            "filled_n": 193,
            "max_bps": 597.9,
            "mean_bps": 14.5,
            "median_bps": 26.7,
            "min_bps": -500.2,
            "n": 193,
            "sum_bps": 2795.4,
            "t3r_bps": 1522.6,
            "tail_lt_-100_n": 33,
            "win_rate": 0.606
          },
          "attempt_n": 193,
          "cal": {
            "attempt_n": 126,
            "fill_rate": 1.0,
            "filled_n": 126,
            "max_bps": 346.2,
            "mean_bps": 4.4,
            "median_bps": 17.2,
            "min_bps": -500.2,
            "n": 126,
            "sum_bps": 554.6,
            "t3r_bps": -363.5,
            "tail_lt_-100_n": 21,
            "win_rate": 0.556
          },
          "hold": {
            "attempt_n": 67,
            "fill_rate": 1.0,
            "filled_n": 67,
            "max_bps": 597.9,
            "mean_bps": 33.4,
            "median_bps": 42.6,
            "min_bps": -449.9,
            "n": 67,
            "sum_bps": 2240.8,
            "t3r_bps": 993.3,
            "tail_lt_-100_n": 12,
            "win_rate": 0.701
          },
          "pass_all": true,
          "pass_hold": true
        },
        "tau600_H2": {
          "all": {
            "attempt_n": 397,
            "fill_rate": 1.0,
            "filled_n": 397,
            "max_bps": 395.1,
            "mean_bps": 2.2,
            "median_bps": 6.4,
            "min_bps": -355.2,
            "n": 397,
            "sum_bps": 882.3,
            "t3r_bps": -126.8,
            "tail_lt_-100_n": 48,
            "win_rate": 0.539
          },
          "attempt_n": 397,
          "cal": {
            "attempt_n": 262,
            "fill_rate": 1.0,
            "filled_n": 262,
            "max_bps": 395.1,
            "mean_bps": 3.9,
            "median_bps": 6.2,
            "min_bps": -355.2,
            "n": 262,
            "sum_bps": 1011.6,
            "t3r_bps": 2.5,
            "tail_lt_-100_n": 32,
            "win_rate": 0.553
          },
          "hold": {
            "attempt_n": 135,
            "fill_rate": 1.0,
            "filled_n": 135,
            "max_bps": 274.9,
            "mean_bps": -1.0,
            "median_bps": 6.4,
            "min_bps": -334.2,
            "n": 135,
            "sum_bps": -129.3,
            "t3r_bps": -864.5,
            "tail_lt_-100_n": 16,
            "win_rate": 0.511
          },
          "pass_all": false,
          "pass_hold": false
        },
        "tau600_H4": {
          "all": {
            "attempt_n": 397,
            "fill_rate": 1.0,
            "filled_n": 397,
            "max_bps": 712.4,
            "mean_bps": 5.7,
            "median_bps": 6.5,
            "min_bps": -573.6,
            "n": 397,
            "sum_bps": 2259.8,
            "t3r_bps": 573.1,
            "tail_lt_-100_n": 71,
            "win_rate": 0.539
          },
          "attempt_n": 397,
          "cal": {
            "attempt_n": 262,
            "fill_rate": 1.0,
            "filled_n": 262,
            "max_bps": 712.4,
            "mean_bps": 1.3,
            "median_bps": 1.4,
            "min_bps": -573.6,
            "n": 262,
            "sum_bps": 338.7,
            "t3r_bps": -1120.9,
            "tail_lt_-100_n": 47,
            "win_rate": 0.519
          },
          "hold": {
            "attempt_n": 135,
            "fill_rate": 1.0,
            "filled_n": 135,
            "max_bps": 581.4,
            "mean_bps": 14.2,
            "median_bps": 20.8,
            "min_bps": -477.9,
            "n": 135,
            "sum_bps": 1921.1,
            "t3r_bps": 679.4,
            "tail_lt_-100_n": 24,
            "win_rate": 0.578
          },
          "pass_all": true,
          "pass_hold": true
        },
        "tau60_H2": {
          "all": {
            "attempt_n": 239,
            "fill_rate": 1.0,
            "filled_n": 239,
            "max_bps": 462.3,
            "mean_bps": 6.1,
            "median_bps": 12.5,
            "min_bps": -378.9,
            "n": 239,
            "sum_bps": 1466.1,
            "t3r_bps": 371.0,
            "tail_lt_-100_n": 31,
            "win_rate": 0.561
          },
          "attempt_n": 239,
          "cal": {
            "attempt_n": 152,
            "fill_rate": 1.0,
            "filled_n": 152,
            "max_bps": 462.3,
            "mean_bps": 11.3,
            "median_bps": 16.1,
            "min_bps": -378.9,
            "n": 152,
            "sum_bps": 1721.1,
            "t3r_bps": 626.0,
            "tail_lt_-100_n": 18,
            "win_rate": 0.586
          },
          "hold": {
            "attempt_n": 87,
            "fill_rate": 1.0,
            "filled_n": 87,
            "max_bps": 275.6,
            "mean_bps": -2.9,
            "median_bps": 4.5,
            "min_bps": -373.6,
            "n": 87,
            "sum_bps": -255.0,
            "t3r_bps": -978.6,
            "tail_lt_-100_n": 13,
            "win_rate": 0.517
          },
          "pass_all": true,
          "pass_hold": false
        },
        "tau60_H4": {
          "all": {
            "attempt_n": 239,
            "fill_rate": 1.0,
            "filled_n": 239,
            "max_bps": 568.3,
            "mean_bps": 8.9,
            "median_bps": 21.1,
            "min_bps": -506.3,
            "n": 239,
            "sum_bps": 2120.5,
            "t3r_bps": 871.9,
            "tail_lt_-100_n": 43,
            "win_rate": 0.582
          },
          "attempt_n": 239,
          "cal": {
            "attempt_n": 152,
            "fill_rate": 1.0,
            "filled_n": 152,
            "max_bps": 349.5,
            "mean_bps": -0.2,
            "median_bps": 14.6,
            "min_bps": -506.3,
            "n": 152,
            "sum_bps": -28.6,
            "t3r_bps": -954.6,
            "tail_lt_-100_n": 28,
            "win_rate": 0.533
          },
          "hold": {
            "attempt_n": 87,
            "fill_rate": 1.0,
            "filled_n": 87,
            "max_bps": 568.3,
            "mean_bps": 24.7,
            "median_bps": 36.6,
            "min_bps": -434.5,
            "n": 87,
            "sum_bps": 2149.1,
            "t3r_bps": 943.1,
            "tail_lt_-100_n": 15,
            "win_rate": 0.667
          },
          "pass_all": true,
          "pass_hold": true
        },
        "tau900_H2": {
          "all": {
            "attempt_n": 387,
            "fill_rate": 1.0,
            "filled_n": 387,
            "max_bps": 366.9,
            "mean_bps": -0.0,
            "median_bps": 4.3,
            "min_bps": -391.7,
            "n": 387,
            "sum_bps": -8.7,
            "t3r_bps": -1004.2,
            "tail_lt_-100_n": 53,
            "win_rate": 0.517
          },
          "attempt_n": 387,
          "cal": {
            "attempt_n": 261,
            "fill_rate": 1.0,
            "filled_n": 261,
            "max_bps": 366.9,
            "mean_bps": 1.2,
            "median_bps": 5.4,
            "min_bps": -345.4,
            "n": 261,
            "sum_bps": 316.0,
            "t3r_bps": -679.5,
            "tail_lt_-100_n": 35,
            "win_rate": 0.529
          },
          "hold": {
            "attempt_n": 126,
            "fill_rate": 1.0,
            "filled_n": 126,
            "max_bps": 264.4,
            "mean_bps": -2.6,
            "median_bps": -2.0,
            "min_bps": -391.7,
            "n": 126,
            "sum_bps": -324.7,
            "t3r_bps": -1017.9,
            "tail_lt_-100_n": 18,
            "win_rate": 0.492
          },
          "pass_all": false,
          "pass_hold": false
        },
        "tau900_H4": {
          "all": {
            "attempt_n": 387,
            "fill_rate": 1.0,
            "filled_n": 387,
            "max_bps": 742.9,
            "mean_bps": 4.3,
            "median_bps": 5.1,
            "min_bps": -601.5,
            "n": 387,
            "sum_bps": 1678.2,
            "t3r_bps": -6.8,
            "tail_lt_-100_n": 68,
            "win_rate": 0.53
          },
          "attempt_n": 387,
          "cal": {
            "attempt_n": 261,
            "fill_rate": 1.0,
            "filled_n": 261,
            "max_bps": 742.9,
            "mean_bps": 1.7,
            "median_bps": 1.4,
            "min_bps": -601.5,
            "n": 261,
            "sum_bps": 451.9,
            "t3r_bps": -1058.8,
            "tail_lt_-100_n": 46,
            "win_rate": 0.513
          },
          "hold": {
            "attempt_n": 126,
            "fill_rate": 1.0,
            "filled_n": 126,
            "max_bps": 548.4,
            "mean_bps": 9.7,
            "median_bps": 20.1,
            "min_bps": -477.4,
            "n": 126,
            "sum_bps": 1226.3,
            "t3r_bps": -65.0,
            "tail_lt_-100_n": 22,
            "win_rate": 0.563
          },
          "pass_all": false,
          "pass_hold": false
        }
      },
      "event_n": 1143,
      "split": {
        "holdout_months": [
          "2026-06"
        ],
        "method": "chronological_month_tail_35pct",
        "months": [
          "2026-02",
          "2026-03",
          "2026-04",
          "2026-06"
        ]
      }
    },
    "300000": {
      "cells": {
        "tau120_H2": {
          "all": {
            "attempt_n": 214,
            "fill_rate": 1.0,
            "filled_n": 214,
            "max_bps": 440.6,
            "mean_bps": 6.8,
            "median_bps": 15.6,
            "min_bps": -413.4,
            "n": 214,
            "sum_bps": 1455.6,
            "t3r_bps": 478.4,
            "tail_lt_-100_n": 28,
            "win_rate": 0.589
          },
          "attempt_n": 214,
          "cal": {
            "attempt_n": 133,
            "fill_rate": 1.0,
            "filled_n": 133,
            "max_bps": 440.6,
            "mean_bps": 13.0,
            "median_bps": 17.1,
            "min_bps": -257.2,
            "n": 133,
            "sum_bps": 1725.0,
            "t3r_bps": 747.8,
            "tail_lt_-100_n": 16,
            "win_rate": 0.609
          },
          "hold": {
            "attempt_n": 81,
            "fill_rate": 1.0,
            "filled_n": 81,
            "max_bps": 258.0,
            "mean_bps": -3.3,
            "median_bps": 9.3,
            "min_bps": -413.4,
            "n": 81,
            "sum_bps": -269.4,
            "t3r_bps": -906.0,
            "tail_lt_-100_n": 12,
            "win_rate": 0.556
          },
          "pass_all": true,
          "pass_hold": false
        },
        "tau120_H4": {
          "all": {
            "attempt_n": 214,
            "fill_rate": 1.0,
            "filled_n": 214,
            "max_bps": 408.9,
            "mean_bps": 9.4,
            "median_bps": 11.6,
            "min_bps": -436.3,
            "n": 214,
            "sum_bps": 2015.1,
            "t3r_bps": 822.7,
            "tail_lt_-100_n": 42,
            "win_rate": 0.57
          },
          "attempt_n": 214,
          "cal": {
            "attempt_n": 133,
            "fill_rate": 1.0,
            "filled_n": 133,
            "max_bps": 370.4,
            "mean_bps": 5.5,
            "median_bps": 10.1,
            "min_bps": -419.2,
            "n": 133,
            "sum_bps": 731.6,
            "t3r_bps": -325.5,
            "tail_lt_-100_n": 27,
            "win_rate": 0.564
          },
          "hold": {
            "attempt_n": 81,
            "fill_rate": 1.0,
            "filled_n": 81,
            "max_bps": 408.9,
            "mean_bps": 15.8,
            "median_bps": 13.8,
            "min_bps": -436.3,
            "n": 81,
            "sum_bps": 1283.5,
            "t3r_bps": 91.1,
            "tail_lt_-100_n": 15,
            "win_rate": 0.58
          },
          "pass_all": true,
          "pass_hold": true
        },
        "tau300_H2": {
          "all": {
            "attempt_n": 271,
            "fill_rate": 1.0,
            "filled_n": 271,
            "max_bps": 443.6,
            "mean_bps": 6.8,
            "median_bps": 9.1,
            "min_bps": -377.5,
            "n": 271,
            "sum_bps": 1848.8,
            "t3r_bps": 802.7,
            "tail_lt_-100_n": 33,
            "win_rate": 0.55
          },
          "attempt_n": 271,
          "cal": {
            "attempt_n": 171,
            "fill_rate": 1.0,
            "filled_n": 171,
            "max_bps": 443.6,
            "mean_bps": 12.8,
            "median_bps": 13.6,
            "min_bps": -299.0,
            "n": 171,
            "sum_bps": 2185.9,
            "t3r_bps": 1165.8,
            "tail_lt_-100_n": 22,
            "win_rate": 0.579
          },
          "hold": {
            "attempt_n": 100,
            "fill_rate": 1.0,
            "filled_n": 100,
            "max_bps": 253.4,
            "mean_bps": -3.4,
            "median_bps": 1.1,
            "min_bps": -377.5,
            "n": 100,
            "sum_bps": -337.1,
            "t3r_bps": -957.8,
            "tail_lt_-100_n": 11,
            "win_rate": 0.5
          },
          "pass_all": true,
          "pass_hold": false
        },
        "tau300_H4": {
          "all": {
            "attempt_n": 271,
            "fill_rate": 1.0,
            "filled_n": 271,
            "max_bps": 405.8,
            "mean_bps": 6.1,
            "median_bps": 10.8,
            "min_bps": -429.3,
            "n": 271,
            "sum_bps": 1663.9,
            "t3r_bps": 475.6,
            "tail_lt_-100_n": 51,
            "win_rate": 0.554
          },
          "attempt_n": 271,
          "cal": {
            "attempt_n": 171,
            "fill_rate": 1.0,
            "filled_n": 171,
            "max_bps": 377.0,
            "mean_bps": 4.4,
            "median_bps": 10.5,
            "min_bps": -381.6,
            "n": 171,
            "sum_bps": 744.3,
            "t3r_bps": -342.0,
            "tail_lt_-100_n": 33,
            "win_rate": 0.544
          },
          "hold": {
            "attempt_n": 100,
            "fill_rate": 1.0,
            "filled_n": 100,
            "max_bps": 405.8,
            "mean_bps": 9.2,
            "median_bps": 12.8,
            "min_bps": -429.3,
            "n": 100,
            "sum_bps": 919.6,
            "t3r_bps": -268.7,
            "tail_lt_-100_n": 18,
            "win_rate": 0.57
          },
          "pass_all": true,
          "pass_hold": false
        },
        "tau30_H2": {
          "all": {
            "attempt_n": 134,
            "fill_rate": 1.0,
            "filled_n": 134,
            "max_bps": 444.3,
            "mean_bps": 21.0,
            "median_bps": 21.4,
            "min_bps": -267.6,
            "n": 134,
            "sum_bps": 2809.8,
            "t3r_bps": 1758.6,
            "tail_lt_-100_n": 14,
            "win_rate": 0.627
          },
          "attempt_n": 134,
          "cal": {
            "attempt_n": 89,
            "fill_rate": 1.0,
            "filled_n": 89,
            "max_bps": 444.3,
            "mean_bps": 26.0,
            "median_bps": 22.3,
            "min_bps": -267.6,
            "n": 89,
            "sum_bps": 2312.9,
            "t3r_bps": 1261.7,
            "tail_lt_-100_n": 9,
            "win_rate": 0.64
          },
          "hold": {
            "attempt_n": 45,
            "fill_rate": 1.0,
            "filled_n": 45,
            "max_bps": 247.1,
            "mean_bps": 11.0,
            "median_bps": 15.8,
            "min_bps": -169.6,
            "n": 45,
            "sum_bps": 496.9,
            "t3r_bps": -126.0,
            "tail_lt_-100_n": 5,
            "win_rate": 0.6
          },
          "pass_all": true,
          "pass_hold": false
        },
        "tau30_H4": {
          "all": {
            "attempt_n": 134,
            "fill_rate": 1.0,
            "filled_n": 134,
            "max_bps": 346.2,
            "mean_bps": 11.8,
            "median_bps": 18.2,
            "min_bps": -507.2,
            "n": 134,
            "sum_bps": 1582.6,
            "t3r_bps": 608.4,
            "tail_lt_-100_n": 25,
            "win_rate": 0.582
          },
          "attempt_n": 134,
          "cal": {
            "attempt_n": 89,
            "fill_rate": 1.0,
            "filled_n": 89,
            "max_bps": 346.2,
            "mean_bps": 4.9,
            "median_bps": 5.2,
            "min_bps": -507.2,
            "n": 89,
            "sum_bps": 437.7,
            "t3r_bps": -517.5,
            "tail_lt_-100_n": 16,
            "win_rate": 0.517
          },
          "hold": {
            "attempt_n": 45,
            "fill_rate": 1.0,
            "filled_n": 45,
            "max_bps": 311.5,
            "mean_bps": 25.4,
            "median_bps": 40.0,
            "min_bps": -444.9,
            "n": 45,
            "sum_bps": 1144.9,
            "t3r_bps": 355.9,
            "tail_lt_-100_n": 9,
            "win_rate": 0.711
          },
          "pass_all": true,
          "pass_hold": true
        },
        "tau600_H2": {
          "all": {
            "attempt_n": 269,
            "fill_rate": 1.0,
            "filled_n": 269,
            "max_bps": 395.1,
            "mean_bps": 10.8,
            "median_bps": 10.9,
            "min_bps": -316.9,
            "n": 269,
            "sum_bps": 2906.9,
            "t3r_bps": 1918.8,
            "tail_lt_-100_n": 27,
            "win_rate": 0.558
          },
          "attempt_n": 269,
          "cal": {
            "attempt_n": 173,
            "fill_rate": 1.0,
            "filled_n": 173,
            "max_bps": 395.1,
            "mean_bps": 17.3,
            "median_bps": 15.7,
            "min_bps": -273.3,
            "n": 173,
            "sum_bps": 2996.3,
            "t3r_bps": 2013.6,
            "tail_lt_-100_n": 19,
            "win_rate": 0.595
          },
          "hold": {
            "attempt_n": 96,
            "fill_rate": 1.0,
            "filled_n": 96,
            "max_bps": 255.0,
            "mean_bps": -0.9,
            "median_bps": -2.5,
            "min_bps": -316.9,
            "n": 96,
            "sum_bps": -89.4,
            "t3r_bps": -712.0,
            "tail_lt_-100_n": 8,
            "win_rate": 0.49
          },
          "pass_all": true,
          "pass_hold": false
        },
        "tau600_H4": {
          "all": {
            "attempt_n": 269,
            "fill_rate": 1.0,
            "filled_n": 269,
            "max_bps": 707.5,
            "mean_bps": 14.9,
            "median_bps": 6.3,
            "min_bps": -443.9,
            "n": 269,
            "sum_bps": 4007.0,
            "t3r_bps": 2436.2,
            "tail_lt_-100_n": 47,
            "win_rate": 0.539
          },
          "attempt_n": 269,
          "cal": {
            "attempt_n": 173,
            "fill_rate": 1.0,
            "filled_n": 173,
            "max_bps": 707.5,
            "mean_bps": 12.9,
            "median_bps": 0.8,
            "min_bps": -415.5,
            "n": 173,
            "sum_bps": 2238.4,
            "t3r_bps": 783.7,
            "tail_lt_-100_n": 29,
            "win_rate": 0.509
          },
          "hold": {
            "attempt_n": 96,
            "fill_rate": 1.0,
            "filled_n": 96,
            "max_bps": 455.7,
            "mean_bps": 18.4,
            "median_bps": 22.6,
            "min_bps": -443.9,
            "n": 96,
            "sum_bps": 1768.6,
            "t3r_bps": 537.5,
            "tail_lt_-100_n": 18,
            "win_rate": 0.594
          },
          "pass_all": true,
          "pass_hold": true
        },
        "tau60_H2": {
          "all": {
            "attempt_n": 170,
            "fill_rate": 1.0,
            "filled_n": 170,
            "max_bps": 462.3,
            "mean_bps": 16.2,
            "median_bps": 17.9,
            "min_bps": -227.7,
            "n": 170,
            "sum_bps": 2758.2,
            "t3r_bps": 1692.6,
            "tail_lt_-100_n": 20,
            "win_rate": 0.6
          },
          "attempt_n": 170,
          "cal": {
            "attempt_n": 107,
            "fill_rate": 1.0,
            "filled_n": 107,
            "max_bps": 462.3,
            "mean_bps": 19.3,
            "median_bps": 19.4,
            "min_bps": -227.7,
            "n": 107,
            "sum_bps": 2069.0,
            "t3r_bps": 1003.4,
            "tail_lt_-100_n": 13,
            "win_rate": 0.598
          },
          "hold": {
            "attempt_n": 63,
            "fill_rate": 1.0,
            "filled_n": 63,
            "max_bps": 243.7,
            "mean_bps": 10.9,
            "median_bps": 13.9,
            "min_bps": -185.6,
            "n": 63,
            "sum_bps": 689.2,
            "t3r_bps": 66.6,
            "tail_lt_-100_n": 7,
            "win_rate": 0.603
          },
          "pass_all": true,
          "pass_hold": true
        },
        "tau60_H4": {
          "all": {
            "attempt_n": 170,
            "fill_rate": 1.0,
            "filled_n": 170,
            "max_bps": 435.3,
            "mean_bps": 13.4,
            "median_bps": 14.8,
            "min_bps": -472.2,
            "n": 170,
            "sum_bps": 2278.8,
            "t3r_bps": 1131.8,
            "tail_lt_-100_n": 31,
            "win_rate": 0.588
          },
          "attempt_n": 170,
          "cal": {
            "attempt_n": 107,
            "fill_rate": 1.0,
            "filled_n": 107,
            "max_bps": 349.5,
            "mean_bps": 5.0,
            "median_bps": 10.5,
            "min_bps": -472.2,
            "n": 107,
            "sum_bps": 536.9,
            "t3r_bps": -404.2,
            "tail_lt_-100_n": 19,
            "win_rate": 0.542
          },
          "hold": {
            "attempt_n": 63,
            "fill_rate": 1.0,
            "filled_n": 63,
            "max_bps": 435.3,
            "mean_bps": 27.6,
            "median_bps": 24.5,
            "min_bps": -427.4,
            "n": 63,
            "sum_bps": 1741.9,
            "t3r_bps": 637.5,
            "tail_lt_-100_n": 12,
            "win_rate": 0.667
          },
          "pass_all": true,
          "pass_hold": true
        },
        "tau900_H2": {
          "all": {
            "attempt_n": 263,
            "fill_rate": 1.0,
            "filled_n": 263,
            "max_bps": 366.9,
            "mean_bps": 11.6,
            "median_bps": 9.4,
            "min_bps": -271.9,
            "n": 263,
            "sum_bps": 3051.1,
            "t3r_bps": 2021.7,
            "tail_lt_-100_n": 32,
            "win_rate": 0.555
          },
          "attempt_n": 263,
          "cal": {
            "attempt_n": 173,
            "fill_rate": 1.0,
            "filled_n": 173,
            "max_bps": 366.9,
            "mean_bps": 17.5,
            "median_bps": 11.3,
            "min_bps": -271.9,
            "n": 173,
            "sum_bps": 3023.4,
            "t3r_bps": 1994.0,
            "tail_lt_-100_n": 20,
            "win_rate": 0.584
          },
          "hold": {
            "attempt_n": 90,
            "fill_rate": 1.0,
            "filled_n": 90,
            "max_bps": 218.9,
            "mean_bps": 0.3,
            "median_bps": -0.1,
            "min_bps": -235.9,
            "n": 90,
            "sum_bps": 27.7,
            "t3r_bps": -588.2,
            "tail_lt_-100_n": 12,
            "win_rate": 0.5
          },
          "pass_all": true,
          "pass_hold": false
        },
        "tau900_H4": {
          "all": {
            "attempt_n": 263,
            "fill_rate": 1.0,
            "filled_n": 263,
            "max_bps": 738.9,
            "mean_bps": 14.2,
            "median_bps": 12.7,
            "min_bps": -523.4,
            "n": 263,
            "sum_bps": 3731.6,
            "t3r_bps": 2141.2,
            "tail_lt_-100_n": 46,
            "win_rate": 0.555
          },
          "attempt_n": 263,
          "cal": {
            "attempt_n": 173,
            "fill_rate": 1.0,
            "filled_n": 173,
            "max_bps": 738.9,
            "mean_bps": 13.4,
            "median_bps": 8.3,
            "min_bps": -390.0,
            "n": 173,
            "sum_bps": 2321.5,
            "t3r_bps": 814.8,
            "tail_lt_-100_n": 30,
            "win_rate": 0.532
          },
          "hold": {
            "attempt_n": 90,
            "fill_rate": 1.0,
            "filled_n": 90,
            "max_bps": 467.2,
            "mean_bps": 15.7,
            "median_bps": 22.0,
            "min_bps": -523.4,
            "n": 90,
            "sum_bps": 1410.1,
            "t3r_bps": 220.7,
            "tail_lt_-100_n": 16,
            "win_rate": 0.6
          },
          "pass_all": true,
          "pass_hold": true
        }
      },
      "event_n": 809,
      "split": {
        "holdout_months": [
          "2026-06"
        ],
        "method": "chronological_month_tail_35pct",
        "months": [
          "2026-02",
          "2026-03",
          "2026-04",
          "2026-06"
        ]
      }
    },
    "50000": {
      "cells": {
        "tau120_H2": {
          "all": {
            "attempt_n": 755,
            "fill_rate": 1.0,
            "filled_n": 755,
            "max_bps": 675.0,
            "mean_bps": 0.6,
            "median_bps": 4.9,
            "min_bps": -401.4,
            "n": 755,
            "sum_bps": 473.4,
            "t3r_bps": -1505.4,
            "tail_lt_-100_n": 103,
            "win_rate": 0.518
          },
          "attempt_n": 755,
          "cal": {
            "attempt_n": 548,
            "fill_rate": 1.0,
            "filled_n": 548,
            "max_bps": 675.0,
            "mean_bps": 3.3,
            "median_bps": 5.3,
            "min_bps": -366.4,
            "n": 548,
            "sum_bps": 1794.0,
            "t3r_bps": -184.8,
            "tail_lt_-100_n": 73,
            "win_rate": 0.524
          },
          "hold": {
            "attempt_n": 207,
            "fill_rate": 1.0,
            "filled_n": 207,
            "max_bps": 327.3,
            "mean_bps": -6.4,
            "median_bps": 0.5,
            "min_bps": -401.4,
            "n": 207,
            "sum_bps": -1320.6,
            "t3r_bps": -2216.2,
            "tail_lt_-100_n": 30,
            "win_rate": 0.502
          },
          "pass_all": false,
          "pass_hold": false
        },
        "tau120_H4": {
          "all": {
            "attempt_n": 755,
            "fill_rate": 1.0,
            "filled_n": 755,
            "max_bps": 557.3,
            "mean_bps": 0.8,
            "median_bps": -0.8,
            "min_bps": -525.5,
            "n": 755,
            "sum_bps": 606.4,
            "t3r_bps": -1035.3,
            "tail_lt_-100_n": 145,
            "win_rate": 0.493
          },
          "attempt_n": 755,
          "cal": {
            "attempt_n": 548,
            "fill_rate": 1.0,
            "filled_n": 548,
            "max_bps": 557.3,
            "mean_bps": 1.7,
            "median_bps": -2.5,
            "min_bps": -505.7,
            "n": 548,
            "sum_bps": 950.4,
            "t3r_bps": -672.6,
            "tail_lt_-100_n": 108,
            "win_rate": 0.485
          },
          "hold": {
            "attempt_n": 207,
            "fill_rate": 1.0,
            "filled_n": 207,
            "max_bps": 538.9,
            "mean_bps": -1.7,
            "median_bps": 6.6,
            "min_bps": -525.5,
            "n": 207,
            "sum_bps": -344.0,
            "t3r_bps": -1661.5,
            "tail_lt_-100_n": 37,
            "win_rate": 0.512
          },
          "pass_all": false,
          "pass_hold": false
        },
        "tau300_H2": {
          "all": {
            "attempt_n": 970,
            "fill_rate": 0.999,
            "filled_n": 969,
            "max_bps": 680.2,
            "mean_bps": -0.0,
            "median_bps": 3.1,
            "min_bps": -453.7,
            "n": 969,
            "sum_bps": -41.2,
            "t3r_bps": -2037.8,
            "tail_lt_-100_n": 127,
            "win_rate": 0.508
          },
          "attempt_n": 970,
          "cal": {
            "attempt_n": 701,
            "fill_rate": 1.0,
            "filled_n": 701,
            "max_bps": 680.2,
            "mean_bps": 2.1,
            "median_bps": 3.5,
            "min_bps": -353.5,
            "n": 701,
            "sum_bps": 1450.1,
            "t3r_bps": -546.5,
            "tail_lt_-100_n": 95,
            "win_rate": 0.511
          },
          "hold": {
            "attempt_n": 269,
            "fill_rate": 0.996,
            "filled_n": 268,
            "max_bps": 291.0,
            "mean_bps": -5.6,
            "median_bps": -0.1,
            "min_bps": -453.7,
            "n": 268,
            "sum_bps": -1491.3,
            "t3r_bps": -2339.2,
            "tail_lt_-100_n": 32,
            "win_rate": 0.5
          },
          "pass_all": false,
          "pass_hold": false
        },
        "tau300_H4": {
          "all": {
            "attempt_n": 970,
            "fill_rate": 0.999,
            "filled_n": 969,
            "max_bps": 605.2,
            "mean_bps": -1.9,
            "median_bps": 0.6,
            "min_bps": -561.2,
            "n": 969,
            "sum_bps": -1879.6,
            "t3r_bps": -3629.0,
            "tail_lt_-100_n": 188,
            "win_rate": 0.502
          },
          "attempt_n": 970,
          "cal": {
            "attempt_n": 701,
            "fill_rate": 1.0,
            "filled_n": 701,
            "max_bps": 605.2,
            "mean_bps": -2.4,
            "median_bps": -2.2,
            "min_bps": -561.2,
            "n": 701,
            "sum_bps": -1698.3,
            "t3r_bps": -3436.9,
            "tail_lt_-100_n": 136,
            "win_rate": 0.498
          },
          "hold": {
            "attempt_n": 269,
            "fill_rate": 0.996,
            "filled_n": 268,
            "max_bps": 559.4,
            "mean_bps": -0.7,
            "median_bps": 5.8,
            "min_bps": -481.8,
            "n": 268,
            "sum_bps": -181.3,
            "t3r_bps": -1592.9,
            "tail_lt_-100_n": 52,
            "win_rate": 0.511
          },
          "pass_all": false,
          "pass_hold": false
        },
        "tau30_H2": {
          "all": {
            "attempt_n": 479,
            "fill_rate": 1.0,
            "filled_n": 479,
            "max_bps": 705.8,
            "mean_bps": 9.4,
            "median_bps": 9.6,
            "min_bps": -371.6,
            "n": 479,
            "sum_bps": 4507.4,
            "t3r_bps": 2613.5,
            "tail_lt_-100_n": 60,
            "win_rate": 0.539
          },
          "attempt_n": 479,
          "cal": {
            "attempt_n": 352,
            "fill_rate": 1.0,
            "filled_n": 352,
            "max_bps": 705.8,
            "mean_bps": 10.7,
            "median_bps": 9.6,
            "min_bps": -371.6,
            "n": 352,
            "sum_bps": 3778.0,
            "t3r_bps": 1884.1,
            "tail_lt_-100_n": 45,
            "win_rate": 0.537
          },
          "hold": {
            "attempt_n": 127,
            "fill_rate": 1.0,
            "filled_n": 127,
            "max_bps": 313.6,
            "mean_bps": 5.7,
            "median_bps": 10.2,
            "min_bps": -324.9,
            "n": 127,
            "sum_bps": 729.4,
            "t3r_bps": -124.8,
            "tail_lt_-100_n": 15,
            "win_rate": 0.543
          },
          "pass_all": true,
          "pass_hold": false
        },
        "tau30_H4": {
          "all": {
            "attempt_n": 479,
            "fill_rate": 1.0,
            "filled_n": 479,
            "max_bps": 596.7,
            "mean_bps": 9.0,
            "median_bps": 4.4,
            "min_bps": -520.3,
            "n": 479,
            "sum_bps": 4331.8,
            "t3r_bps": 2614.8,
            "tail_lt_-100_n": 90,
            "win_rate": 0.522
          },
          "attempt_n": 479,
          "cal": {
            "attempt_n": 352,
            "fill_rate": 1.0,
            "filled_n": 352,
            "max_bps": 581.5,
            "mean_bps": 9.9,
            "median_bps": 3.1,
            "min_bps": -392.5,
            "n": 352,
            "sum_bps": 3480.1,
            "t3r_bps": 1840.2,
            "tail_lt_-100_n": 69,
            "win_rate": 0.514
          },
          "hold": {
            "attempt_n": 127,
            "fill_rate": 1.0,
            "filled_n": 127,
            "max_bps": 596.7,
            "mean_bps": 6.7,
            "median_bps": 12.2,
            "min_bps": -520.3,
            "n": 127,
            "sum_bps": 851.7,
            "t3r_bps": -409.0,
            "tail_lt_-100_n": 21,
            "win_rate": 0.543
          },
          "pass_all": true,
          "pass_hold": false
        },
        "tau600_H2": {
          "all": {
            "attempt_n": 1012,
            "fill_rate": 0.999,
            "filled_n": 1011,
            "max_bps": 710.2,
            "mean_bps": 0.5,
            "median_bps": -0.1,
            "min_bps": -505.5,
            "n": 1011,
            "sum_bps": 458.3,
            "t3r_bps": -1471.1,
            "tail_lt_-100_n": 136,
            "win_rate": 0.499
          },
          "attempt_n": 1012,
          "cal": {
            "attempt_n": 743,
            "fill_rate": 1.0,
            "filled_n": 743,
            "max_bps": 710.2,
            "mean_bps": 2.2,
            "median_bps": -0.2,
            "min_bps": -406.1,
            "n": 743,
            "sum_bps": 1604.5,
            "t3r_bps": -324.9,
            "tail_lt_-100_n": 99,
            "win_rate": 0.497
          },
          "hold": {
            "attempt_n": 269,
            "fill_rate": 0.996,
            "filled_n": 268,
            "max_bps": 292.5,
            "mean_bps": -4.3,
            "median_bps": 1.1,
            "min_bps": -505.5,
            "n": 268,
            "sum_bps": -1146.2,
            "t3r_bps": -1991.6,
            "tail_lt_-100_n": 37,
            "win_rate": 0.504
          },
          "pass_all": false,
          "pass_hold": false
        },
        "tau600_H4": {
          "all": {
            "attempt_n": 1012,
            "fill_rate": 0.999,
            "filled_n": 1011,
            "max_bps": 720.0,
            "mean_bps": -2.5,
            "median_bps": -1.9,
            "min_bps": -580.9,
            "n": 1011,
            "sum_bps": -2572.1,
            "t3r_bps": -4492.2,
            "tail_lt_-100_n": 196,
            "win_rate": 0.483
          },
          "attempt_n": 1012,
          "cal": {
            "attempt_n": 743,
            "fill_rate": 1.0,
            "filled_n": 743,
            "max_bps": 720.0,
            "mean_bps": -3.4,
            "median_bps": -3.2,
            "min_bps": -580.9,
            "n": 743,
            "sum_bps": -2548.0,
            "t3r_bps": -4453.9,
            "tail_lt_-100_n": 144,
            "win_rate": 0.478
          },
          "hold": {
            "attempt_n": 269,
            "fill_rate": 0.996,
            "filled_n": 268,
            "max_bps": 581.4,
            "mean_bps": -0.1,
            "median_bps": -0.6,
            "min_bps": -495.4,
            "n": 268,
            "sum_bps": -24.1,
            "t3r_bps": -1449.7,
            "tail_lt_-100_n": 52,
            "win_rate": 0.496
          },
          "pass_all": false,
          "pass_hold": false
        },
        "tau60_H2": {
          "all": {
            "attempt_n": 586,
            "fill_rate": 1.0,
            "filled_n": 586,
            "max_bps": 700.0,
            "mean_bps": 3.6,
            "median_bps": 5.2,
            "min_bps": -380.1,
            "n": 586,
            "sum_bps": 2103.1,
            "t3r_bps": 247.1,
            "tail_lt_-100_n": 75,
            "win_rate": 0.531
          },
          "attempt_n": 586,
          "cal": {
            "attempt_n": 428,
            "fill_rate": 1.0,
            "filled_n": 428,
            "max_bps": 700.0,
            "mean_bps": 5.2,
            "median_bps": 5.2,
            "min_bps": -380.1,
            "n": 428,
            "sum_bps": 2235.5,
            "t3r_bps": 379.5,
            "tail_lt_-100_n": 54,
            "win_rate": 0.533
          },
          "hold": {
            "attempt_n": 158,
            "fill_rate": 1.0,
            "filled_n": 158,
            "max_bps": 316.7,
            "mean_bps": -0.8,
            "median_bps": 5.5,
            "min_bps": -355.4,
            "n": 158,
            "sum_bps": -132.4,
            "t3r_bps": -1030.8,
            "tail_lt_-100_n": 21,
            "win_rate": 0.525
          },
          "pass_all": true,
          "pass_hold": false
        },
        "tau60_H4": {
          "all": {
            "attempt_n": 586,
            "fill_rate": 1.0,
            "filled_n": 586,
            "max_bps": 570.4,
            "mean_bps": 3.6,
            "median_bps": 1.5,
            "min_bps": -539.5,
            "n": 586,
            "sum_bps": 2109.3,
            "t3r_bps": 433.2,
            "tail_lt_-100_n": 116,
            "win_rate": 0.502
          },
          "attempt_n": 586,
          "cal": {
            "attempt_n": 428,
            "fill_rate": 1.0,
            "filled_n": 428,
            "max_bps": 569.5,
            "mean_bps": 3.2,
            "median_bps": -1.8,
            "min_bps": -391.0,
            "n": 428,
            "sum_bps": 1371.0,
            "t3r_bps": -229.3,
            "tail_lt_-100_n": 88,
            "win_rate": 0.486
          },
          "hold": {
            "attempt_n": 158,
            "fill_rate": 1.0,
            "filled_n": 158,
            "max_bps": 570.4,
            "mean_bps": 4.7,
            "median_bps": 10.2,
            "min_bps": -539.5,
            "n": 158,
            "sum_bps": 738.3,
            "t3r_bps": -662.4,
            "tail_lt_-100_n": 28,
            "win_rate": 0.544
          },
          "pass_all": true,
          "pass_hold": false
        },
        "tau900_H2": {
          "all": {
            "attempt_n": 1029,
            "fill_rate": 0.999,
            "filled_n": 1028,
            "max_bps": 684.7,
            "mean_bps": -1.6,
            "median_bps": -1.8,
            "min_bps": -421.2,
            "n": 1028,
            "sum_bps": -1642.0,
            "t3r_bps": -3458.3,
            "tail_lt_-100_n": 136,
            "win_rate": 0.494
          },
          "attempt_n": 1029,
          "cal": {
            "attempt_n": 760,
            "fill_rate": 1.0,
            "filled_n": 760,
            "max_bps": 684.7,
            "mean_bps": 0.2,
            "median_bps": -0.3,
            "min_bps": -356.8,
            "n": 760,
            "sum_bps": 184.2,
            "t3r_bps": -1632.1,
            "tail_lt_-100_n": 101,
            "win_rate": 0.5
          },
          "hold": {
            "attempt_n": 269,
            "fill_rate": 0.996,
            "filled_n": 268,
            "max_bps": 393.4,
            "mean_bps": -6.8,
            "median_bps": -3.6,
            "min_bps": -421.2,
            "n": 268,
            "sum_bps": -1826.2,
            "t3r_bps": -2745.0,
            "tail_lt_-100_n": 35,
            "win_rate": 0.478
          },
          "pass_all": false,
          "pass_hold": false
        },
        "tau900_H4": {
          "all": {
            "attempt_n": 1029,
            "fill_rate": 0.999,
            "filled_n": 1028,
            "max_bps": 671.8,
            "mean_bps": -2.6,
            "median_bps": -2.7,
            "min_bps": -588.6,
            "n": 1028,
            "sum_bps": -2710.9,
            "t3r_bps": -4632.9,
            "tail_lt_-100_n": 203,
            "win_rate": 0.488
          },
          "attempt_n": 1029,
          "cal": {
            "attempt_n": 760,
            "fill_rate": 1.0,
            "filled_n": 760,
            "max_bps": 671.8,
            "mean_bps": -2.0,
            "median_bps": -3.0,
            "min_bps": -588.6,
            "n": 760,
            "sum_bps": -1556.5,
            "t3r_bps": -3478.5,
            "tail_lt_-100_n": 156,
            "win_rate": 0.486
          },
          "hold": {
            "attempt_n": 269,
            "fill_rate": 0.996,
            "filled_n": 268,
            "max_bps": 547.6,
            "mean_bps": -4.3,
            "median_bps": -1.3,
            "min_bps": -473.9,
            "n": 268,
            "sum_bps": -1154.4,
            "t3r_bps": -2540.5,
            "tail_lt_-100_n": 47,
            "win_rate": 0.496
          },
          "pass_all": false,
          "pass_hold": false
        }
      },
      "event_n": 2753,
      "split": {
        "holdout_months": [
          "2026-06"
        ],
        "method": "chronological_month_tail_35pct",
        "months": [
          "2026-02",
          "2026-03",
          "2026-04",
          "2026-06"
        ]
      }
    }
  }
}
```

## Read

- The strongest executable expansion remains SELL silence/reclaim fade; broad state is stronger than executable entry after book staleness and holdout.
- BUY-side mirror still does not become a deployable short/fade lane.
- Propagation tags are useful as navigation/danger labels but not yet as live order logic.
