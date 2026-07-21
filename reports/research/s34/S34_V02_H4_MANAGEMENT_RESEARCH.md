# S34 V02 H4 Management Research

Generated: `2026-06-29T19:53:25.861249+00:00`
Scope: `{'rule': 'S34_V_ENGINE_V0_2_ETH_SELL_MAKER_LONG_H2_O20_W300_O5_DEEPBID', 'source_ledger': 'D:\\eclipse_scalper\\reports\\research\\s34\\S34_V02_ALPHA_NAVIGATION_OVERLAY_120D.json', 'n': 11, 'research_only': True}`

## Executive Read

H4 dominates current V02 ledger: H2 sum 1083.5 / T3R 392.5 vs H4 sum 1742.6 / T3R 826.5. Runner recognition is suggestive but small-N. Giveback/trailing does not beat fixed H4. SL100 damages the edge; SL150+ behaves as catastrophic-only in this sample.

## 1. H4 Forward Shadow Backtest

```json
{
  "h2": {
    "n": 11,
    "sum": 1083.5,
    "mean": 98.5,
    "median": 46.7,
    "win_rate": 1.0,
    "t3r": 392.5,
    "top1_removed": 777.1,
    "min": 15.0,
    "max": 306.4
  },
  "h3": {
    "n": 11,
    "sum": 1319.3,
    "mean": 119.9,
    "median": 104.6,
    "win_rate": 0.909,
    "t3r": 578.9,
    "top1_removed": 981.2,
    "min": -27.3,
    "max": 338.1
  },
  "h4": {
    "n": 11,
    "sum": 1742.6,
    "mean": 158.4,
    "median": 167.3,
    "win_rate": 1.0,
    "t3r": 826.5,
    "top1_removed": 1345.1,
    "min": 9.7,
    "max": 397.5
  },
  "h4_minus_h2": {
    "n": 11,
    "sum": 659.1,
    "mean": 59.9,
    "median": 86.0,
    "win_rate": 0.545,
    "t3r": 131.5,
    "top1_removed": 435.4,
    "min": -59.4,
    "max": 223.7
  },
  "per_trade": [
    {
      "fill_utc": "2026-04-16T13:52:52+00:00",
      "h2": 83.6,
      "h3": -27.3,
      "h4": 31.5,
      "delta_h4_h2": -52.2
    },
    {
      "fill_utc": "2026-04-18T03:57:08+00:00",
      "h2": 29.5,
      "h3": 32.9,
      "h4": 9.7,
      "delta_h4_h2": -19.8
    },
    {
      "fill_utc": "2026-04-20T14:51:09.004000+00:00",
      "h2": 15.0,
      "h3": 104.6,
      "h4": 167.3,
      "delta_h4_h2": 152.3
    },
    {
      "fill_utc": "2026-04-20T14:51:13+00:00",
      "h2": 16.6,
      "h3": 101.6,
      "h4": 168.2,
      "delta_h4_h2": 151.6
    },
    {
      "fill_utc": "2026-06-16T04:44:40.005000+00:00",
      "h2": 46.7,
      "h3": 56.4,
      "h4": 132.7,
      "delta_h4_h2": 86.0
    },
    {
      "fill_utc": "2026-06-17T01:21:26.001000+00:00",
      "h2": 44.2,
      "h3": 58.7,
      "h4": 39.7,
      "delta_h4_h2": -4.5
    },
    {
      "fill_utc": "2026-06-20T14:14:30+00:00",
      "h2": 154.8,
      "h3": 124.2,
      "h4": 107.0,
      "delta_h4_h2": -47.8
    },
    {
      "fill_utc": "2026-06-21T23:44:28.001000+00:00",
      "h2": 229.7,
      "h3": 219.4,
      "h4": 170.3,
      "delta_h4_h2": -59.4
    },
    {
      "fill_utc": "2026-06-26T02:53:32.010000+00:00",
      "h2": 137.7,
      "h3": 183.0,
      "h4": 275.7,
      "delta_h4_h2": 138.0
    },
    {
      "fill_utc": "2026-06-26T11:50:59.001000+00:00",
      "h2": 19.1,
      "h3": 127.8,
      "h4": 242.8,
      "delta_h4_h2": 223.7
    },
    {
      "fill_utc": "2026-06-26T13:19:32.001000+00:00",
      "h2": 306.4,
      "h3": 338.1,
      "h4": 397.5,
      "delta_h4_h2": 91.1
    }
  ]
}
```

## 2. H4 Runner Recognition

```json
{
  "runner_count": 6,
  "by_rebound50_30m": {
    "False": {
      "n": 8,
      "sum": 631.9,
      "mean": 79.0,
      "median": 112.0,
      "win_rate": 0.625,
      "t3r": 104.2,
      "top1_removed": 408.2,
      "min": -52.2,
      "max": 223.7
    },
    "True": {
      "n": 3,
      "sum": 27.2,
      "mean": 9.1,
      "median": -4.5,
      "win_rate": 0.333,
      "t3r": 27.2,
      "top1_removed": -63.9,
      "min": -59.4,
      "max": 91.1
    }
  },
  "by_rebound20_15m": {
    "False": {
      "n": 5,
      "sum": 475.8,
      "mean": 95.2,
      "median": 138.0,
      "win_rate": 0.8,
      "t3r": 33.9,
      "top1_removed": 323.5,
      "min": -52.2,
      "max": 152.3
    },
    "True": {
      "n": 6,
      "sum": 183.4,
      "mean": 30.6,
      "median": -12.1,
      "win_rate": 0.333,
      "t3r": -127.0,
      "top1_removed": -40.4,
      "min": -59.4,
      "max": 223.7
    }
  },
  "by_btc_no_dump30": {
    "False": {
      "n": 1,
      "sum": -52.2,
      "mean": -52.2,
      "median": -52.2,
      "win_rate": 0.0,
      "t3r": -52.2,
      "top1_removed": -52.2,
      "min": -52.2,
      "max": -52.2
    },
    "True": {
      "n": 10,
      "sum": 711.3,
      "mean": 71.1,
      "median": 88.6,
      "win_rate": 0.6,
      "t3r": 183.6,
      "top1_removed": 487.6,
      "min": -59.4,
      "max": 223.7
    }
  },
  "by_sol_no_dump30": {
    "True": {
      "n": 11,
      "sum": 659.1,
      "mean": 59.9,
      "median": 86.0,
      "win_rate": 0.545,
      "t3r": 131.5,
      "top1_removed": 435.4,
      "min": -59.4,
      "max": 223.7
    }
  },
  "by_cross_no_dump30": {
    "False": {
      "n": 1,
      "sum": -52.2,
      "mean": -52.2,
      "median": -52.2,
      "win_rate": 0.0,
      "t3r": -52.2,
      "top1_removed": -52.2,
      "min": -52.2,
      "max": -52.2
    },
    "True": {
      "n": 10,
      "sum": 711.3,
      "mean": 71.1,
      "median": 88.6,
      "win_rate": 0.6,
      "t3r": 183.6,
      "top1_removed": 487.6,
      "min": -59.4,
      "max": 223.7
    }
  },
  "candidate_policies": {
    "hold_h4_if_rebound50_30m_else_h2": {
      "n": 11,
      "sum": 1110.7,
      "mean": 101.0,
      "median": 46.7,
      "win_rate": 1.0,
      "t3r": 388.0,
      "top1_removed": 713.2,
      "min": 15.0,
      "max": 397.5
    },
    "hold_h4_if_cross_no_dump_else_h2": {
      "n": 11,
      "sum": 1794.7,
      "mean": 163.2,
      "median": 167.3,
      "win_rate": 1.0,
      "t3r": 878.7,
      "top1_removed": 1397.2,
      "min": 9.7,
      "max": 397.5
    },
    "hold_h4_if_rebound20_and_btc_no_dump_else_h2": {
      "n": 11,
      "sum": 1266.8,
      "mean": 115.2,
      "median": 83.6,
      "win_rate": 1.0,
      "t3r": 456.1,
      "top1_removed": 869.3,
      "min": 9.7,
      "max": 397.5
    }
  }
}
```

## 3. H2 Checkpoint Decision Engine

```json
{
  "always_h2": {
    "n": 11,
    "sum": 1083.5,
    "mean": 98.5,
    "median": 46.7,
    "win_rate": 1.0,
    "t3r": 392.5,
    "top1_removed": 777.1,
    "min": 15.0,
    "max": 306.4
  },
  "always_h3": {
    "n": 11,
    "sum": 1319.3,
    "mean": 119.9,
    "median": 104.6,
    "win_rate": 0.909,
    "t3r": 578.9,
    "top1_removed": 981.2,
    "min": -27.3,
    "max": 338.1
  },
  "always_h4": {
    "n": 11,
    "sum": 1742.6,
    "mean": 158.4,
    "median": 167.3,
    "win_rate": 1.0,
    "t3r": 826.5,
    "top1_removed": 1345.1,
    "min": 9.7,
    "max": 397.5
  },
  "h4_if_cross_no_dump_else_h2": {
    "n": 11,
    "sum": 1794.7,
    "mean": 163.2,
    "median": 167.3,
    "win_rate": 1.0,
    "t3r": 878.7,
    "top1_removed": 1397.2,
    "min": 9.7,
    "max": 397.5
  },
  "h4_if_rebound50_else_h2": {
    "n": 11,
    "sum": 1110.7,
    "mean": 101.0,
    "median": 46.7,
    "win_rate": 1.0,
    "t3r": 388.0,
    "top1_removed": 713.2,
    "min": 15.0,
    "max": 397.5
  },
  "partial_50_h2_50_h4": {
    "n": 11,
    "sum": 1413.0,
    "mean": 128.5,
    "median": 92.4,
    "win_rate": 1.0,
    "t3r": 654.3,
    "top1_removed": 1061.1,
    "min": 19.6,
    "max": 352.0
  },
  "partial_30_h2_70_h4": {
    "n": 11,
    "sum": 1544.8,
    "mean": 140.4,
    "median": 121.7,
    "win_rate": 1.0,
    "t3r": 752.2,
    "top1_removed": 1174.7,
    "min": 15.6,
    "max": 370.2
  },
  "h4_if_h2_lt100_else_h2": {
    "n": 11,
    "sum": 1672.9,
    "mean": 152.1,
    "median": 154.8,
    "win_rate": 1.0,
    "t3r": 893.9,
    "top1_removed": 1366.5,
    "min": 9.7,
    "max": 306.4
  }
}
```

## 4. H4 Giveback Protection

```json
{
  "fixed_h4": {
    "n": 11,
    "sum": 1742.6,
    "mean": 158.4,
    "median": 167.3,
    "win_rate": 1.0,
    "t3r": 826.5,
    "top1_removed": 1345.1,
    "min": 9.7,
    "max": 397.5
  },
  "giveback_distribution": {
    "n": 11,
    "sum": 702.6,
    "mean": 63.9,
    "median": 41.7,
    "win_rate": 1.0,
    "t3r": 310.0,
    "top1_removed": 554.6,
    "min": 9.9,
    "max": 148.0
  },
  "trail_peak40_gb25": {
    "result": {
      "n": 11,
      "sum": 406.9,
      "mean": 37.0,
      "median": 36.0,
      "win_rate": 1.0,
      "t3r": 261.3,
      "top1_removed": 356.6,
      "min": 25.8,
      "max": 50.3
    },
    "exits": {
      "TRAIL": 11,
      "TIME": 0
    }
  },
  "trail_peak40_gb40": {
    "result": {
      "n": 11,
      "sum": 554.4,
      "mean": 50.4,
      "median": 38.2,
      "win_rate": 1.0,
      "t3r": 285.3,
      "top1_removed": 428.9,
      "min": 22.4,
      "max": 125.5
    },
    "exits": {
      "TRAIL": 11,
      "TIME": 0
    }
  },
  "trail_peak80_gb25": {
    "result": {
      "n": 11,
      "sum": 934.5,
      "mean": 85.0,
      "median": 68.2,
      "win_rate": 1.0,
      "t3r": 492.6,
      "top1_removed": 758.9,
      "min": 9.7,
      "max": 175.6
    },
    "exits": {
      "TRAIL": 10,
      "TIME": 1
    }
  },
  "trail_peak80_gb40": {
    "result": {
      "n": 11,
      "sum": 822.2,
      "mean": 74.7,
      "median": 71.0,
      "win_rate": 1.0,
      "t3r": 463.5,
      "top1_removed": 682.8,
      "min": 9.7,
      "max": 139.4
    },
    "exits": {
      "TRAIL": 10,
      "TIME": 1
    }
  },
  "trail_after_h2_peak80_gb40": {
    "result": {
      "n": 11,
      "sum": 1197.1,
      "mean": 108.8,
      "median": 78.4,
      "win_rate": 1.0,
      "t3r": 534.4,
      "top1_removed": 799.6,
      "min": 9.7,
      "max": 397.5
    },
    "exits": {
      "TRAIL": 9,
      "TIME": 2
    }
  },
  "partial_50_h2_50_h4": {
    "n": 11,
    "sum": 1413.0,
    "mean": 128.5,
    "median": 92.4,
    "win_rate": 1.0,
    "t3r": 654.3,
    "top1_removed": 1061.1,
    "min": 19.6,
    "max": 352.0
  },
  "partial_30_h2_70_h4": {
    "n": 11,
    "sum": 1544.8,
    "mean": 140.4,
    "median": 121.7,
    "win_rate": 1.0,
    "t3r": 752.2,
    "top1_removed": 1174.7,
    "min": 15.6,
    "max": 370.2
  }
}
```

## 5. MAE / Catastrophic Stop Research

```json
{
  "h4_sl100": {
    "result": {
      "n": 11,
      "sum": 878.1,
      "mean": 79.8,
      "median": 107.0,
      "win_rate": 0.727,
      "t3r": 142.0,
      "top1_removed": 480.5,
      "min": -106.6,
      "max": 397.5
    },
    "exits": {
      "SL": 3,
      "TIME": 8
    }
  },
  "h4_sl125": {
    "result": {
      "n": 11,
      "sum": 1202.2,
      "mean": 109.3,
      "median": 132.7,
      "win_rate": 0.818,
      "t3r": 358.6,
      "top1_removed": 804.7,
      "min": -134.2,
      "max": 397.5
    },
    "exits": {
      "SL": 2,
      "TIME": 9
    }
  },
  "h4_sl150": {
    "result": {
      "n": 11,
      "sum": 1742.6,
      "mean": 158.4,
      "median": 167.3,
      "win_rate": 1.0,
      "t3r": 826.5,
      "top1_removed": 1345.1,
      "min": 9.7,
      "max": 397.5
    },
    "exits": {
      "SL": 0,
      "TIME": 11
    }
  },
  "h4_sl175": {
    "result": {
      "n": 11,
      "sum": 1742.6,
      "mean": 158.4,
      "median": 167.3,
      "win_rate": 1.0,
      "t3r": 826.5,
      "top1_removed": 1345.1,
      "min": 9.7,
      "max": 397.5
    },
    "exits": {
      "SL": 0,
      "TIME": 11
    }
  },
  "h4_sl200": {
    "result": {
      "n": 11,
      "sum": 1742.6,
      "mean": 158.4,
      "median": 167.3,
      "win_rate": 1.0,
      "t3r": 826.5,
      "top1_removed": 1345.1,
      "min": 9.7,
      "max": 397.5
    },
    "exits": {
      "SL": 0,
      "TIME": 11
    }
  },
  "h4_sl150_delay5m": {
    "result": {
      "n": 11,
      "sum": 1742.6,
      "mean": 158.4,
      "median": 167.3,
      "win_rate": 1.0,
      "t3r": 826.5,
      "top1_removed": 1345.1,
      "min": 9.7,
      "max": 397.5
    },
    "exits": {
      "SL": 0,
      "TIME": 11
    }
  },
  "h4_sl150_delay15m": {
    "result": {
      "n": 11,
      "sum": 1742.6,
      "mean": 158.4,
      "median": 167.3,
      "win_rate": 1.0,
      "t3r": 826.5,
      "top1_removed": 1345.1,
      "min": 9.7,
      "max": 397.5
    },
    "exits": {
      "SL": 0,
      "TIME": 11
    }
  },
  "h4_mae_distribution": {
    "n": 11,
    "sum": -607.5,
    "mean": -55.2,
    "median": -19.0,
    "win_rate": 0.0,
    "t3r": -597.1,
    "top1_removed": -605.4,
    "min": -147.2,
    "max": -2.2
  }
}
```

## 6. Verdict

```json
{
  "horizon": "H4_FORWARD_SHADOW_LEAD_SMALL_N",
  "runner_recognition": "CROSS_NO_DUMP_POLICY_SLIGHTLY_BEATS_H4_IN_SAMPLE_BUT_SMALL_N",
  "checkpoint": "ALWAYS_H4_OR_CROSS_NO_DUMP_H4_SHADOW; NO_LIVE_CHANGE",
  "giveback": "FIXED_H4_STILL_BEST; TRAILING_REDUCES_UPSIDE",
  "stop": "SL100_BAD; SL125_TOUCHES_ONE; SL150_PLUS_CATASTROPHIC_ONLY",
  "next_action": "create/track H3/H4 shadow buckets; keep live H2 unchanged until forward N grows and queue realism is tested"
}
```

## 7. Forward Shadow Spec

```json
{
  "buckets": [
    "V02_H2_CURRENT",
    "V02_H3_SHADOW",
    "V02_H4_SHADOW",
    "V02_H4_CROSS_NO_DUMP_SHADOW"
  ],
  "minimum_review": {
    "N_lt_10": "observe_only",
    "N_10_to_20": "early_confidence",
    "N_ge_30": "paper_candidate_review"
  },
  "kill": [
    "H4 T3R < H2 T3R after N>=10",
    "two H4 losses below -150bps",
    "queue realism rejects fills"
  ],
  "live_change_allowed": false
}
```
