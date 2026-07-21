# S34 V02 Next-Gen Alpha Research

Generated: `2026-06-29T18:56:44.852431+00:00`
Scope: `{'rule': 'S34_V_ENGINE_V0_2_ETH_SELL_MAKER_LONG_H2_O20_W300_O5_DEEPBID', 'days': 14, 'start_utc': '2026-06-15T18:53:40+00:00', 'end_utc': '2026-06-29T18:53:40+00:00', 'maker_fee_bps': -0.5, 'taker_fee_bps': 3.05, 'cross_margin_bps': 2.0, 'anchors_total': 7, 'filled_n': 7, 'note': 'Research-only. No live executor/config/order logic touched.'}`

## Executive Read

V02 baseline filled N=7 sum=933.9 median=137.4 T3R=247.5. Best fixed horizon is 240m with T3R=449.0. Best giveback is peak80_gb40 with T3R=274.1. Best execution cell is O15_W300. Mechanism expansion filled N=0 T3R=0.0.

## 1. Event Graph Anatomy

```json
{
  "cards": [
    {
      "fill_utc": "2026-06-16T04:44:40.005000+00:00",
      "net_2h_bps": 46.2,
      "fill_delay_sec": 808.5,
      "mae_bps": -19.0,
      "mae_time_sec": 670.0,
      "mfe_bps": 62.5,
      "mfe_time_sec": 6768.0,
      "first_nav_high_sec": 120.0,
      "first_buy_spike_sec": null,
      "state_sequence_5m": "LLMHMH",
      "nav_recommendation": "SCALP_OR_REDUCE"
    },
    {
      "fill_utc": "2026-06-17T01:21:26.001000+00:00",
      "net_2h_bps": 44.2,
      "fill_delay_sec": 264.2,
      "mae_bps": -18.4,
      "mae_time_sec": 75.0,
      "mfe_bps": 153.0,
      "mfe_time_sec": 3754.0,
      "first_nav_high_sec": 120.0,
      "first_buy_spike_sec": 3694.0,
      "state_sequence_5m": "MMHHHH",
      "nav_recommendation": "SCALP_ONLY"
    },
    {
      "fill_utc": "2026-06-20T14:14:30+00:00",
      "net_2h_bps": 154.8,
      "fill_delay_sec": 338.8,
      "mae_bps": -3.7,
      "mae_time_sec": 14.0,
      "mfe_bps": 238.3,
      "mfe_time_sec": 3959.0,
      "first_nav_high_sec": 60.0,
      "first_buy_spike_sec": 1830.0,
      "state_sequence_5m": "LMLLHM",
      "nav_recommendation": "SCALP_OR_REDUCE"
    },
    {
      "fill_utc": "2026-06-21T23:44:28.001000+00:00",
      "net_2h_bps": 229.5,
      "fill_delay_sec": 645.3,
      "mae_bps": -2.2,
      "mae_time_sec": 0.0,
      "mfe_bps": 318.3,
      "mfe_time_sec": 6178.0,
      "first_nav_high_sec": 0.0,
      "first_buy_spike_sec": 5912.0,
      "state_sequence_5m": "HMMMMM",
      "nav_recommendation": "BASELINE"
    },
    {
      "fill_utc": "2026-06-26T02:53:32.010000+00:00",
      "net_2h_bps": 137.4,
      "fill_delay_sec": 301.5,
      "mae_bps": -100.7,
      "mae_time_sec": 575.0,
      "mfe_bps": 237.9,
      "mfe_time_sec": 3789.0,
      "first_nav_high_sec": 120.0,
      "first_buy_spike_sec": 568.0,
      "state_sequence_5m": "MMHMHL",
      "nav_recommendation": "BASELINE"
    },
    {
      "fill_utc": "2026-06-26T11:50:59.001000+00:00",
      "net_2h_bps": 19.7,
      "fill_delay_sec": 4285.5,
      "mae_bps": -147.2,
      "mae_time_sec": 5315.0,
      "mfe_bps": 91.7,
      "mfe_time_sec": 2591.0,
      "first_nav_high_sec": 0.0,
      "first_buy_spike_sec": 2521.0,
      "state_sequence_5m": "MLLHHM",
      "nav_recommendation": "SCALP_ONLY"
    },
    {
      "fill_utc": "2026-06-26T13:19:32.001000+00:00",
      "net_2h_bps": 302.2,
      "fill_delay_sec": 37.1,
      "mae_bps": -13.7,
      "mae_time_sec": 2.0,
      "mfe_bps": 334.4,
      "mfe_time_sec": 4162.0,
      "first_nav_high_sec": 60.0,
      "first_buy_spike_sec": 1408.0,
      "state_sequence_5m": "LHMHHH",
      "nav_recommendation": "SCALP_OR_REDUCE"
    }
  ],
  "baseline": {
    "n": 7,
    "sum": 933.9,
    "mean": 133.4,
    "median": 137.4,
    "win_rate": 1.0,
    "t3r": 247.5,
    "min": 19.7,
    "max": 302.2
  }
}
```

## 2. Phase Timing

```json
{
  "mae_time_sec": {
    "n": 7,
    "sum": 6651.0,
    "mean": 950.1,
    "median": 75.0,
    "win_rate": 0.857,
    "t3r": 91.0,
    "min": 0.0,
    "max": 5315.0
  },
  "mfe_time_sec": {
    "n": 7,
    "sum": 31201.0,
    "mean": 4457.3,
    "median": 3959.0,
    "win_rate": 1.0,
    "t3r": 14093.0,
    "min": 2591.0,
    "max": 6768.0
  },
  "rebound_20bps_time_sec": {
    "n": 7,
    "sum": 3688.0,
    "mean": 526.9,
    "median": 320.0,
    "win_rate": 1.0,
    "t3r": 698.0,
    "min": 29.0,
    "max": 1110.0
  },
  "rebound_50bps_time_sec": {
    "n": 7,
    "sum": 15696.0,
    "mean": 2242.3,
    "median": 1806.0,
    "win_rate": 1.0,
    "t3r": 4366.0,
    "min": 78.0,
    "max": 6741.0
  },
  "first_nav_high_sec": {
    "n": 7,
    "sum": 480.0,
    "mean": 68.6,
    "median": 60.0,
    "win_rate": 0.714,
    "t3r": 120.0,
    "min": 0.0,
    "max": 120.0
  },
  "first_buy_spike_sec": {
    "n": 6,
    "sum": 15933.0,
    "mean": 2655.5,
    "median": 2175.5,
    "win_rate": 1.0,
    "t3r": 3806.0,
    "min": 568.0,
    "max": 5912.0
  }
}
```

## 3. Fixed Horizon Surface

```json
{
  "15m": {
    "n": 7,
    "sum": 96.9,
    "mean": 13.8,
    "median": 9.6,
    "win_rate": 0.714,
    "t3r": -7.1,
    "min": -12.4,
    "max": 59.3
  },
  "30m": {
    "n": 7,
    "sum": 300.4,
    "mean": 42.9,
    "median": 39.6,
    "win_rate": 0.857,
    "t3r": 52.4,
    "min": -6.7,
    "max": 134.8
  },
  "60m": {
    "n": 7,
    "sum": 753.3,
    "mean": 107.6,
    "median": 123.2,
    "win_rate": 0.857,
    "t3r": 166.6,
    "min": -47.5,
    "max": 257.0
  },
  "90m": {
    "n": 7,
    "sum": 740.9,
    "mean": 105.8,
    "median": 114.1,
    "win_rate": 0.857,
    "t3r": 111.8,
    "min": -84.2,
    "max": 290.3
  },
  "120m": {
    "n": 7,
    "sum": 933.9,
    "mean": 133.4,
    "median": 137.4,
    "win_rate": 1.0,
    "t3r": 247.5,
    "min": 19.7,
    "max": 302.2
  },
  "180m": {
    "n": 7,
    "sum": 1109.2,
    "mean": 158.5,
    "median": 130.2,
    "win_rate": 1.0,
    "t3r": 369.0,
    "min": 56.0,
    "max": 337.9
  },
  "240m": {
    "n": 7,
    "sum": 1364.8,
    "mean": 195.0,
    "median": 170.3,
    "win_rate": 1.0,
    "t3r": 449.0,
    "min": 39.7,
    "max": 397.5
  }
}
```

## 4. MFE/Giveback Exit

```json
{
  "peak40_gb25": {
    "n": 7,
    "sum": 241.2,
    "mean": 34.5,
    "median": 31.5,
    "win_rate": 1.0,
    "t3r": 112.5,
    "min": 25.8,
    "max": 45.7
  },
  "peak40_gb40": {
    "n": 7,
    "sum": 395.1,
    "mean": 56.4,
    "median": 38.2,
    "win_rate": 1.0,
    "t3r": 130.1,
    "min": 24.4,
    "max": 125.5
  },
  "peak40_gb60": {
    "n": 7,
    "sum": 535.0,
    "mean": 76.4,
    "median": 58.1,
    "win_rate": 1.0,
    "t3r": 143.6,
    "min": 15.0,
    "max": 170.3
  },
  "peak80_gb25": {
    "n": 7,
    "sum": 695.3,
    "mean": 99.3,
    "median": 66.5,
    "win_rate": 1.0,
    "t3r": 253.3,
    "min": 58.8,
    "max": 175.6
  },
  "peak80_gb40": {
    "n": 7,
    "sum": 632.7,
    "mean": 90.4,
    "median": 88.8,
    "win_rate": 1.0,
    "t3r": 274.1,
    "min": 50.7,
    "max": 139.4
  },
  "peak80_gb60": {
    "n": 7,
    "sum": 795.7,
    "mean": 113.7,
    "median": 88.3,
    "win_rate": 1.0,
    "t3r": 217.0,
    "min": 33.9,
    "max": 275.7
  }
}
```

## 5. MAE Survival

```json
{
  "mae_5m_bps_le_-20": {
    "False": {
      "n": 5,
      "sum": 776.8,
      "mean": 155.4,
      "median": 154.8,
      "win_rate": 1.0,
      "t3r": 90.4,
      "min": 44.2,
      "max": 302.2
    },
    "True": {
      "n": 2,
      "sum": 157.1,
      "mean": 78.5,
      "median": 78.5,
      "win_rate": 1.0,
      "t3r": 157.1,
      "min": 19.7,
      "max": 137.4
    }
  },
  "mae_5m_bps_le_-50": {
    "False": {
      "n": 7,
      "sum": 933.9,
      "mean": 133.4,
      "median": 137.4,
      "win_rate": 1.0,
      "t3r": 247.5,
      "min": 19.7,
      "max": 302.2
    }
  },
  "mae_5m_bps_le_-100": {
    "False": {
      "n": 7,
      "sum": 933.9,
      "mean": 133.4,
      "median": 137.4,
      "win_rate": 1.0,
      "t3r": 247.5,
      "min": 19.7,
      "max": 302.2
    }
  },
  "mae_10m_bps_le_-20": {
    "False": {
      "n": 5,
      "sum": 776.8,
      "mean": 155.4,
      "median": 154.8,
      "win_rate": 1.0,
      "t3r": 90.4,
      "min": 44.2,
      "max": 302.2
    },
    "True": {
      "n": 2,
      "sum": 157.1,
      "mean": 78.5,
      "median": 78.5,
      "win_rate": 1.0,
      "t3r": 157.1,
      "min": 19.7,
      "max": 137.4
    }
  },
  "mae_10m_bps_le_-50": {
    "False": {
      "n": 6,
      "sum": 796.5,
      "mean": 132.8,
      "median": 100.5,
      "win_rate": 1.0,
      "t3r": 110.1,
      "min": 19.7,
      "max": 302.2
    },
    "True": {
      "n": 1,
      "sum": 137.4,
      "mean": 137.4,
      "median": 137.4,
      "win_rate": 1.0,
      "t3r": 137.4,
      "min": 137.4,
      "max": 137.4
    }
  },
  "mae_10m_bps_le_-100": {
    "False": {
      "n": 6,
      "sum": 796.5,
      "mean": 132.8,
      "median": 100.5,
      "win_rate": 1.0,
      "t3r": 110.1,
      "min": 19.7,
      "max": 302.2
    },
    "True": {
      "n": 1,
      "sum": 137.4,
      "mean": 137.4,
      "median": 137.4,
      "win_rate": 1.0,
      "t3r": 137.4,
      "min": 137.4,
      "max": 137.4
    }
  },
  "mae_15m_bps_le_-20": {
    "False": {
      "n": 5,
      "sum": 776.8,
      "mean": 155.4,
      "median": 154.8,
      "win_rate": 1.0,
      "t3r": 90.4,
      "min": 44.2,
      "max": 302.2
    },
    "True": {
      "n": 2,
      "sum": 157.1,
      "mean": 78.5,
      "median": 78.5,
      "win_rate": 1.0,
      "t3r": 157.1,
      "min": 19.7,
      "max": 137.4
    }
  },
  "mae_15m_bps_le_-50": {
    "False": {
      "n": 6,
      "sum": 796.5,
      "mean": 132.8,
      "median": 100.5,
      "win_rate": 1.0,
      "t3r": 110.1,
      "min": 19.7,
      "max": 302.2
    },
    "True": {
      "n": 1,
      "sum": 137.4,
      "mean": 137.4,
      "median": 137.4,
      "win_rate": 1.0,
      "t3r": 137.4,
      "min": 137.4,
      "max": 137.4
    }
  },
  "mae_15m_bps_le_-100": {
    "False": {
      "n": 6,
      "sum": 796.5,
      "mean": 132.8,
      "median": 100.5,
      "win_rate": 1.0,
      "t3r": 110.1,
      "min": 19.7,
      "max": 302.2
    },
    "True": {
      "n": 1,
      "sum": 137.4,
      "mean": 137.4,
      "median": 137.4,
      "win_rate": 1.0,
      "t3r": 137.4,
      "min": 137.4,
      "max": 137.4
    }
  },
  "mae_30m_bps_le_-20": {
    "False": {
      "n": 5,
      "sum": 776.8,
      "mean": 155.4,
      "median": 154.8,
      "win_rate": 1.0,
      "t3r": 90.4,
      "min": 44.2,
      "max": 302.2
    },
    "True": {
      "n": 2,
      "sum": 157.1,
      "mean": 78.5,
      "median": 78.5,
      "win_rate": 1.0,
      "t3r": 157.1,
      "min": 19.7,
      "max": 137.4
    }
  },
  "mae_30m_bps_le_-50": {
    "False": {
      "n": 6,
      "sum": 796.5,
      "mean": 132.8,
      "median": 100.5,
      "win_rate": 1.0,
      "t3r": 110.1,
      "min": 19.7,
      "max": 302.2
    },
    "True": {
      "n": 1,
      "sum": 137.4,
      "mean": 137.4,
      "median": 137.4,
      "win_rate": 1.0,
      "t3r": 137.4,
      "min": 137.4,
      "max": 137.4
    }
  },
  "mae_30m_bps_le_-100": {
    "False": {
      "n": 6,
      "sum": 796.5,
      "mean": 132.8,
      "median": 100.5,
      "win_rate": 1.0,
      "t3r": 110.1,
      "min": 19.7,
      "max": 302.2
    },
    "True": {
      "n": 1,
      "sum": 137.4,
      "mean": 137.4,
      "median": 137.4,
      "win_rate": 1.0,
      "t3r": 137.4,
      "min": 137.4,
      "max": 137.4
    }
  }
}
```

## 6. NAV/BUY Spike Phase Sensor

```json
{
  "buy_spike_post_5m": {
    "False": {
      "n": 7,
      "sum": 933.9,
      "mean": 133.4,
      "median": 137.4,
      "win_rate": 1.0,
      "t3r": 247.5,
      "min": 19.7,
      "max": 302.2
    }
  },
  "buy_spike_post_15m": {
    "False": {
      "n": 6,
      "sum": 796.5,
      "mean": 132.8,
      "median": 100.5,
      "win_rate": 1.0,
      "t3r": 110.1,
      "min": 19.7,
      "max": 302.2
    },
    "True": {
      "n": 1,
      "sum": 137.4,
      "mean": 137.4,
      "median": 137.4,
      "win_rate": 1.0,
      "t3r": 137.4,
      "min": 137.4,
      "max": 137.4
    }
  },
  "nav_high_fill": {
    "False": {
      "n": 5,
      "sum": 684.7,
      "mean": 136.9,
      "median": 137.4,
      "win_rate": 1.0,
      "t3r": 90.4,
      "min": 44.2,
      "max": 302.2
    },
    "True": {
      "n": 2,
      "sum": 249.2,
      "mean": 124.6,
      "median": 124.6,
      "win_rate": 1.0,
      "t3r": 249.2,
      "min": 19.7,
      "max": 229.5
    }
  },
  "nav_high_holds_5m": {
    "False": {
      "n": 3,
      "sum": 521.6,
      "mean": 173.9,
      "median": 154.8,
      "win_rate": 1.0,
      "t3r": 521.6,
      "min": 137.4,
      "max": 229.5
    },
    "True": {
      "n": 4,
      "sum": 412.3,
      "mean": 103.1,
      "median": 45.2,
      "win_rate": 1.0,
      "t3r": 19.7,
      "min": 19.7,
      "max": 302.2
    }
  },
  "rebound_confirmed_5m": {
    "False": {
      "n": 7,
      "sum": 933.9,
      "mean": 133.4,
      "median": 137.4,
      "win_rate": 1.0,
      "t3r": 247.5,
      "min": 19.7,
      "max": 302.2
    }
  }
}
```

## 7. Tail / Weak-Trade Detection

```json
{
  "actual_negative_tail_n": 0,
  "weak_trade_n_lt_50bps": 3,
  "weak_by_replenish_120s": {
    "high": {
      "n": 3,
      "sum": 319.9,
      "mean": 106.6,
      "median": 46.2,
      "win_rate": 1.0,
      "t3r": 319.9,
      "min": 44.2,
      "max": 229.5
    },
    "low": {
      "n": 4,
      "sum": 614.0,
      "mean": 153.5,
      "median": 146.1,
      "win_rate": 1.0,
      "t3r": 19.7,
      "min": 19.7,
      "max": 302.2
    }
  },
  "weak_by_sell_liq_5m": {
    "high": {
      "n": 4,
      "sum": 503.4,
      "mean": 125.9,
      "median": 90.8,
      "win_rate": 1.0,
      "t3r": 19.7,
      "min": 19.7,
      "max": 302.2
    },
    "low": {
      "n": 3,
      "sum": 430.5,
      "mean": 143.5,
      "median": 154.8,
      "win_rate": 1.0,
      "t3r": 430.5,
      "min": 46.2,
      "max": 229.5
    }
  },
  "weak_by_btc_after": {
    "btc_up": {
      "n": 7,
      "sum": 933.9,
      "mean": 133.4,
      "median": 137.4,
      "win_rate": 1.0,
      "t3r": 247.5,
      "min": 19.7,
      "max": 302.2
    }
  }
}
```

## 8. Execution Surface

```json
{
  "O10_W120": {
    "fills": 7,
    "legs": {
      "replacement": 5,
      "initial": 2
    },
    "result": {
      "n": 7,
      "sum": 918.7,
      "mean": 131.2,
      "median": 150.4,
      "win_rate": 1.0,
      "t3r": 240.6,
      "min": 19.7,
      "max": 293.6
    }
  },
  "O10_W180": {
    "fills": 7,
    "legs": {
      "replacement": 4,
      "initial": 3
    },
    "result": {
      "n": 7,
      "sum": 927.1,
      "mean": 132.4,
      "median": 150.4,
      "win_rate": 1.0,
      "t3r": 249.3,
      "min": 19.7,
      "max": 293.6
    }
  },
  "O10_W300": {
    "fills": 7,
    "legs": {
      "replacement": 4,
      "initial": 3
    },
    "result": {
      "n": 7,
      "sum": 927.1,
      "mean": 132.4,
      "median": 150.4,
      "win_rate": 1.0,
      "t3r": 249.3,
      "min": 19.7,
      "max": 293.6
    }
  },
  "O10_W600": {
    "fills": 6,
    "legs": {
      "replacement": 3,
      "initial": 3
    },
    "result": {
      "n": 6,
      "sum": 772.3,
      "mean": 128.7,
      "median": 98.3,
      "win_rate": 1.0,
      "t3r": 98.9,
      "min": 19.7,
      "max": 293.6
    }
  },
  "O15_W120": {
    "fills": 7,
    "legs": {
      "replacement": 6,
      "initial": 1
    },
    "result": {
      "n": 7,
      "sum": 911.5,
      "mean": 130.2,
      "median": 139.1,
      "win_rate": 1.0,
      "t3r": 229.3,
      "min": 19.7,
      "max": 297.7
    }
  },
  "O15_W180": {
    "fills": 7,
    "legs": {
      "replacement": 6,
      "initial": 1
    },
    "result": {
      "n": 7,
      "sum": 913.5,
      "mean": 130.5,
      "median": 139.1,
      "win_rate": 1.0,
      "t3r": 231.5,
      "min": 19.7,
      "max": 297.7
    }
  },
  "O15_W300": {
    "fills": 7,
    "legs": {
      "replacement": 4,
      "initial": 3
    },
    "result": {
      "n": 7,
      "sum": 932.6,
      "mean": 133.2,
      "median": 147.3,
      "win_rate": 1.0,
      "t3r": 250.7,
      "min": 19.7,
      "max": 297.7
    }
  },
  "O15_W600": {
    "fills": 6,
    "legs": {
      "replacement": 3,
      "initial": 3
    },
    "result": {
      "n": 6,
      "sum": 777.9,
      "mean": 129.6,
      "median": 96.7,
      "win_rate": 1.0,
      "t3r": 103.4,
      "min": 19.7,
      "max": 297.7
    }
  },
  "O20_W120": {
    "fills": 7,
    "legs": {
      "replacement": 6,
      "initial": 1
    },
    "result": {
      "n": 7,
      "sum": 916.0,
      "mean": 130.9,
      "median": 139.1,
      "win_rate": 1.0,
      "t3r": 229.3,
      "min": 19.7,
      "max": 302.2
    }
  },
  "O20_W180": {
    "fills": 7,
    "legs": {
      "replacement": 6,
      "initial": 1
    },
    "result": {
      "n": 7,
      "sum": 918.0,
      "mean": 131.1,
      "median": 139.1,
      "win_rate": 1.0,
      "t3r": 231.5,
      "min": 19.7,
      "max": 302.2
    }
  },
  "O20_W300": {
    "fills": 7,
    "legs": {
      "replacement": 5,
      "initial": 2
    },
    "result": {
      "n": 7,
      "sum": 933.9,
      "mean": 133.4,
      "median": 137.4,
      "win_rate": 1.0,
      "t3r": 247.5,
      "min": 19.7,
      "max": 302.2
    }
  },
  "O20_W600": {
    "fills": 6,
    "legs": {
      "replacement": 3,
      "initial": 3
    },
    "result": {
      "n": 6,
      "sum": 791.2,
      "mean": 131.9,
      "median": 97.8,
      "win_rate": 1.0,
      "t3r": 110.1,
      "min": 19.7,
      "max": 302.2
    }
  },
  "O25_W120": {
    "fills": 7,
    "legs": {
      "replacement": 6,
      "initial": 1
    },
    "result": {
      "n": 7,
      "sum": 921.2,
      "mean": 131.6,
      "median": 139.1,
      "win_rate": 1.0,
      "t3r": 229.3,
      "min": 19.7,
      "max": 307.3
    }
  },
  "O25_W180": {
    "fills": 7,
    "legs": {
      "replacement": 6,
      "initial": 1
    },
    "result": {
      "n": 7,
      "sum": 923.1,
      "mean": 131.9,
      "median": 139.1,
      "win_rate": 1.0,
      "t3r": 231.5,
      "min": 19.7,
      "max": 307.3
    }
  },
  "O25_W300": {
    "fills": 7,
    "legs": {
      "replacement": 6,
      "initial": 1
    },
    "result": {
      "n": 7,
      "sum": 924.7,
      "mean": 132.1,
      "median": 137.4,
      "win_rate": 1.0,
      "t3r": 233.1,
      "min": 19.7,
      "max": 307.3
    }
  },
  "O25_W600": {
    "fills": 6,
    "legs": {
      "replacement": 3,
      "initial": 3
    },
    "result": {
      "n": 6,
      "sum": 811.8,
      "mean": 135.3,
      "median": 104.6,
      "win_rate": 1.0,
      "t3r": 117.4,
      "min": 19.7,
      "max": 307.3
    }
  },
  "O30_W120": {
    "fills": 7,
    "legs": {
      "replacement": 6,
      "initial": 1
    },
    "result": {
      "n": 7,
      "sum": 926.3,
      "mean": 132.3,
      "median": 139.1,
      "win_rate": 1.0,
      "t3r": 229.3,
      "min": 19.7,
      "max": 312.5
    }
  },
  "O30_W180": {
    "fills": 7,
    "legs": {
      "replacement": 6,
      "initial": 1
    },
    "result": {
      "n": 7,
      "sum": 928.3,
      "mean": 132.6,
      "median": 139.1,
      "win_rate": 1.0,
      "t3r": 231.5,
      "min": 19.7,
      "max": 312.5
    }
  },
  "O30_W300": {
    "fills": 7,
    "legs": {
      "replacement": 6,
      "initial": 1
    },
    "result": {
      "n": 7,
      "sum": 929.9,
      "mean": 132.8,
      "median": 137.4,
      "win_rate": 1.0,
      "t3r": 233.1,
      "min": 19.7,
      "max": 312.5
    }
  },
  "O30_W600": {
    "fills": 6,
    "legs": {
      "replacement": 3,
      "initial": 3
    },
    "result": {
      "n": 6,
      "sum": 824.3,
      "mean": 137.4,
      "median": 108.2,
      "win_rate": 1.0,
      "t3r": 118.5,
      "min": 19.7,
      "max": 312.5
    }
  }
}
```

## 9. Fill Delay Quality

```json
{
  "by_leg": {
    "initial": {
      "n": 2,
      "sum": 346.4,
      "mean": 173.2,
      "median": 173.2,
      "win_rate": 1.0,
      "t3r": 346.4,
      "min": 44.2,
      "max": 302.2
    },
    "replacement": {
      "n": 5,
      "sum": 587.5,
      "mean": 117.5,
      "median": 137.4,
      "win_rate": 1.0,
      "t3r": 65.9,
      "min": 19.7,
      "max": 229.5
    }
  },
  "by_delay_bin": {
    "5to15m": {
      "n": 4,
      "sum": 567.9,
      "mean": 142.0,
      "median": 146.1,
      "win_rate": 1.0,
      "t3r": 46.2,
      "min": 46.2,
      "max": 229.5
    },
    "gt15m": {
      "n": 1,
      "sum": 19.7,
      "mean": 19.7,
      "median": 19.7,
      "win_rate": 1.0,
      "t3r": 19.7,
      "min": 19.7,
      "max": 19.7
    },
    "lt5m": {
      "n": 2,
      "sum": 346.4,
      "mean": 173.2,
      "median": 173.2,
      "win_rate": 1.0,
      "t3r": 346.4,
      "min": 44.2,
      "max": 302.2
    }
  },
  "fill_delay_sec": {
    "n": 7,
    "sum": 6680.9,
    "mean": 954.4,
    "median": 338.8,
    "win_rate": 1.0,
    "t3r": 941.6,
    "min": 37.1,
    "max": 4285.5
  }
}
```

## 10. Regime Identity

```json
{
  "by_eth_daily": {
    "eth_daily_down": {
      "n": 5,
      "sum": 732.9,
      "mean": 146.6,
      "median": 137.4,
      "win_rate": 1.0,
      "t3r": 63.9,
      "min": 19.7,
      "max": 302.2
    },
    "eth_daily_up": {
      "n": 2,
      "sum": 201.0,
      "mean": 100.5,
      "median": 100.5,
      "win_rate": 1.0,
      "t3r": 201.0,
      "min": 46.2,
      "max": 154.8
    }
  },
  "by_btc_daily": {
    "btc_daily_down": {
      "n": 5,
      "sum": 732.9,
      "mean": 146.6,
      "median": 137.4,
      "win_rate": 1.0,
      "t3r": 63.9,
      "min": 19.7,
      "max": 302.2
    },
    "btc_daily_up": {
      "n": 2,
      "sum": 201.0,
      "mean": 100.5,
      "median": 100.5,
      "win_rate": 1.0,
      "t3r": 201.0,
      "min": 46.2,
      "max": 154.8
    }
  },
  "prior_eth_daily_bps": {
    "n": 7,
    "sum": -1781.8,
    "mean": -254.5,
    "median": -215.1,
    "win_rate": 0.286,
    "t3r": -2040.6,
    "min": -701.4,
    "max": 261.1
  },
  "prior_btc_daily_bps": {
    "n": 7,
    "sum": -1341.7,
    "mean": -191.7,
    "median": -158.5,
    "win_rate": 0.286,
    "t3r": -1272.9,
    "min": -441.9,
    "max": 26.1
  }
}
```

## 11. Similarity Memory / KNN

```json
{
  "n": 7,
  "k": 3,
  "corr": -0.3,
  "mae_bps": 94.0,
  "details": [
    {
      "ts": "2026-06-16T04:44:40.005000+00:00",
      "actual": 46.2,
      "pred": 142.8
    },
    {
      "ts": "2026-06-17T01:21:26.001000+00:00",
      "actual": 44.2,
      "pred": 143.5
    },
    {
      "ts": "2026-06-20T14:14:30+00:00",
      "actual": 154.8,
      "pred": 106.6
    },
    {
      "ts": "2026-06-21T23:44:28.001000+00:00",
      "actual": 229.5,
      "pred": 81.7
    },
    {
      "ts": "2026-06-26T02:53:32.010000+00:00",
      "actual": 137.4,
      "pred": 122.7
    },
    {
      "ts": "2026-06-26T11:50:59.001000+00:00",
      "actual": 19.7,
      "pred": 81.7
    },
    {
      "ts": "2026-06-26T13:19:32.001000+00:00",
      "actual": 302.2,
      "pred": 112.8
    }
  ]
}
```

## 12. Mechanism Expansion

```json
{
  "sell_notional_p95": 42887149.9,
  "candidates_total": 0,
  "filled_n": 0,
  "result": {
    "n": 0,
    "sum": 0.0,
    "mean": null,
    "median": null,
    "win_rate": null,
    "t3r": 0.0,
    "min": null,
    "max": null
  },
  "sample": []
}
```

## Decision Tags

```json
[
  "MANAGEMENT_LEAD_FIXED_HORIZON",
  "MANAGEMENT_LEAD_GIVEBACK",
  "EXECUTION_SURFACE_LEAD"
]
```
