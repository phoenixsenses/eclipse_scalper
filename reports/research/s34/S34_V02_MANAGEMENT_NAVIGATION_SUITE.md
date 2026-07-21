# S34 V02 Management / Navigation Test Suite

Generated: `2026-06-30T07:54:44.042249+00:00`

All 10 management/navigation tests were run on the current V02 shadow sample. H4 remains the best management hypothesis, but every positive result is still small-N. No live promotion is justified before forward N>=30.

## Summary

- H4 rows: `11`
- NAV rows: `11`
- research only: `True`
- live executor touched: `False`

## 01_h4_forward_validation

```json
{
  "h2": {
    "n": 11,
    "sum_bps": 1081.6,
    "mean_bps": 98.3,
    "median_bps": 46.3,
    "win_rate": 1.0,
    "t3r_bps": 402.6,
    "top1_removed_bps": 781.9,
    "min_bps": 13.3,
    "max_bps": 299.7
  },
  "h3": {
    "n": 11,
    "sum_bps": 1397.6,
    "mean_bps": 127.1,
    "median_bps": 127.7,
    "win_rate": 0.909,
    "t3r_bps": 651.4,
    "top1_removed_bps": 1062.1,
    "min_bps": -30.2,
    "max_bps": 335.5
  },
  "h4": {
    "n": 11,
    "sum_bps": 1738.6,
    "mean_bps": 158.1,
    "median_bps": 164.6,
    "win_rate": 1.0,
    "t3r_bps": 819.2,
    "top1_removed_bps": 1343.5,
    "min_bps": 6.0,
    "max_bps": 395.1
  },
  "h4_minus_h2": {
    "n": 11,
    "sum": 657.0,
    "mean": 59.7,
    "median": 83.8,
    "win_rate": 0.545,
    "t3r": 130.0,
    "min": -59.2,
    "max": 223.2
  },
  "verdict": "SMALL_N_SHADOW_ONLY",
  "read": "H4 leads H2 in current shadow sample, but N<30 so it remains shadow-only."
}
```

## 02_h4_runner_predictor

```json
{
  "runner_count": 6,
  "non_runner_count": 5,
  "runner_h4_minus_h2": {
    "n": 6,
    "sum": 840.2,
    "mean": 140.0,
    "median": 142.7,
    "win_rate": 1.0,
    "t3r": 313.2,
    "min": 83.8,
    "max": 223.2
  },
  "non_runner_h4_minus_h2": {
    "n": 5,
    "sum": -183.2,
    "mean": -36.6,
    "median": -47.3,
    "win_rate": 0.0,
    "t3r": -111.3,
    "min": -59.2,
    "max": -4.5
  },
  "by_cross_no_dump": {
    "False": {
      "n": 1,
      "sum": -52.1,
      "mean": -52.1,
      "median": -52.1,
      "win_rate": 0.0,
      "t3r": -52.1,
      "min": -52.1,
      "max": -52.1
    },
    "True": {
      "n": 10,
      "sum": 709.1,
      "mean": 70.9,
      "median": 89.6,
      "win_rate": 0.6,
      "t3r": 182.1,
      "min": -59.2,
      "max": 223.2
    }
  },
  "by_fill_delay": {
    "FAST_0_60": {
      "n": 2,
      "sum": 43.3,
      "mean": 21.7,
      "median": 21.6,
      "win_rate": 0.5,
      "t3r": 43.3,
      "min": -52.1,
      "max": 95.4
    },
    "LATE_GT900": {
      "n": 2,
      "sum": 375.7,
      "mean": 187.8,
      "median": 187.8,
      "win_rate": 1.0,
      "t3r": 375.7,
      "min": 152.5,
      "max": 223.2
    },
    "NORMAL_60_900": {
      "n": 7,
      "sum": 238.0,
      "mean": 34.0,
      "median": -4.5,
      "win_rate": 0.429,
      "t3r": -131.1,
      "min": -59.2,
      "max": 151.3
    }
  },
  "by_mae": {
    "CLEAN_LT50": {
      "n": 6,
      "sum": 48.1,
      "mean": 8.0,
      "median": -12.3,
      "win_rate": 0.333,
      "t3r": -126.6,
      "min": -59.2,
      "max": 95.4
    },
    "PAIN_50_100": {
      "n": 3,
      "sum": 437.8,
      "mean": 145.9,
      "median": 151.3,
      "win_rate": 1.0,
      "t3r": 437.8,
      "min": 134.0,
      "max": 152.5
    },
    "PAIN_GE100": {
      "n": 2,
      "sum": 171.1,
      "mean": 85.5,
      "median": 85.6,
      "win_rate": 0.5,
      "t3r": 171.1,
      "min": -52.1,
      "max": 223.2
    }
  },
  "by_rebound50": {
    "REBOUND50_FAST_30M": {
      "n": 5,
      "sum": 318.2,
      "mean": 63.6,
      "median": 95.4,
      "win_rate": 0.6,
      "t3r": -63.7,
      "min": -59.2,
      "max": 152.5
    },
    "REBOUND50_LATE": {
      "n": 6,
      "sum": 338.8,
      "mean": 56.5,
      "median": 31.9,
      "win_rate": 0.5,
      "t3r": -119.5,
      "min": -52.1,
      "max": 223.2
    }
  },
  "verdict": "HYPOTHESIS_ONLY_SMALL_N"
}
```

## 03_cross_no_dump_observer

```json
{
  "observer": {
    "cross_no_dump_true": {
      "n": 10,
      "sum_bps": 1709.6,
      "mean_bps": 171.0,
      "median_bps": 166.2,
      "win_rate": 1.0,
      "t3r_bps": 790.2,
      "top1_removed_bps": 1314.5,
      "min_bps": 6.0,
      "max_bps": 395.1
    },
    "cross_no_dump_false": {
      "n": 1,
      "sum_bps": 29.0,
      "mean_bps": 29.0,
      "median_bps": 29.0,
      "win_rate": 1.0,
      "t3r_bps": 29.0,
      "top1_removed_bps": 29.0,
      "min_bps": 29.0,
      "max_bps": 29.0
    },
    "policy_h4_if_cross_no_dump_else_h2": {
      "n": 11,
      "sum_bps": 1790.7,
      "mean_bps": 162.8,
      "median_bps": 164.6,
      "win_rate": 1.0,
      "t3r_bps": 871.3,
      "top1_removed_bps": 1395.6,
      "min_bps": 6.0,
      "max_bps": 395.1
    }
  },
  "verdict": "HYPOTHESIS_ONLY_SMALL_N",
  "read": "Cross-no-dump improves policy sum in-sample, but false bucket has N=1."
}
```

## 04_catastrophic_stop_reality

```json
{
  "observer": {
    "SL100": {
      "touch_count": 2,
      "policy_if_applied_to_h4": {
        "n": 11,
        "sum_bps": 1259.2,
        "mean_bps": 114.5,
        "median_bps": 130.1,
        "win_rate": 0.818,
        "t3r_bps": 400.7,
        "top1_removed_bps": 864.1,
        "min_bps": -105.0,
        "max_bps": 395.1
      }
    },
    "SL125": {
      "touch_count": 2,
      "policy_if_applied_to_h4": {
        "n": 11,
        "sum_bps": 1209.2,
        "mean_bps": 109.9,
        "median_bps": 130.1,
        "win_rate": 0.818,
        "t3r_bps": 350.7,
        "top1_removed_bps": 814.1,
        "min_bps": -130.0,
        "max_bps": 395.1
      }
    },
    "SL150": {
      "touch_count": 0,
      "policy_if_applied_to_h4": {
        "n": 11,
        "sum_bps": 1738.6,
        "mean_bps": 158.1,
        "median_bps": 164.6,
        "win_rate": 1.0,
        "t3r_bps": 819.2,
        "top1_removed_bps": 1343.5,
        "min_bps": 6.0,
        "max_bps": 395.1
      }
    },
    "SL175": {
      "touch_count": 0,
      "policy_if_applied_to_h4": {
        "n": 11,
        "sum_bps": 1738.6,
        "mean_bps": 158.1,
        "median_bps": 164.6,
        "win_rate": 1.0,
        "t3r_bps": 819.2,
        "top1_removed_bps": 1343.5,
        "min_bps": 6.0,
        "max_bps": 395.1
      }
    },
    "SL200": {
      "touch_count": 0,
      "policy_if_applied_to_h4": {
        "n": 11,
        "sum_bps": 1738.6,
        "mean_bps": 158.1,
        "median_bps": 164.6,
        "win_rate": 1.0,
        "t3r_bps": 819.2,
        "top1_removed_bps": 1343.5,
        "min_bps": 6.0,
        "max_bps": 395.1
      }
    }
  },
  "verdict": "SL150_CATASTROPHIC_ONLY_IN_SAMPLE",
  "read": "SL100/125 degrade current sample; SL150+ never touched in current N=11."
}
```

## 05_queue_fill_realism

```json
{
  "queue": {
    "status": "PROXY_ONLY_TOP_OF_BOOK",
    "n": 11,
    "fill_delay_sec": {
      "n": 11,
      "sum_bps": 9394.0,
      "mean_bps": 854.0,
      "median_bps": 333.0,
      "win_rate": 1.0,
      "t3r_bps": 2354.0,
      "top1_removed_bps": 5109.0,
      "min_bps": 37.0,
      "max_bps": 4285.0
    },
    "late_fill_gt_900s_n": 2,
    "bid_vanished_warning_n": 3,
    "high_quality_fill_n": 5,
    "limitation": "No real queue position from top-of-book snapshots; 600GB/tick queue replay is still required before treating fills as executable."
  },
  "late_fill_rows": [
    {
      "bid_depth_usd": 177537.4,
      "book_imbalance": 0.557,
      "btc30_bps": -32.9,
      "bucket": "H4_SHADOW",
      "cross_no_dump": true,
      "entry_price": 2295.3,
      "entry_quality_tags": "POST_ARM_FILL,PULLBACK_FILL,RETEST_BAND_2_25,SPREAD_CLEAN,NO_LARGE_SELL_LIQ_RESTART",
      "entry_quality_warnings": "BID_VANISHED,LATE_RETEST_FILL",
      "exit_horizon_sec": 14400,
      "exit_mode": "H4_SHADOW",
      "exit_price": 2337.61,
      "exit_price_source": "book_bid",
      "exit_ts_ms": 1776710471038,
      "fill_delay_sec": 1951.0,
      "fill_leg": "replacement",
      "h2_net_bps": 27.0,
      "h4_net_bps": 179.5,
      "mae_bps": -75.8,
      "maker_fill_ts_ms": 1776696071001,
      "maker_fill_utc": "2026-04-20T14:41:11.001000+00:00",
      "mfe_bps": 190.6,
      "net_bps": 179.5,
      "notes": "shadow_observation_only_no_order",
      "observation_status": "CLOSED",
      "prior_4h_bps": -75.9,
      "protocol_id": "S34_V_ENGINE_V0_2_ETH_SELL_MAKER_LONG_H2_O20_W300_O5_DEEPBID",
      "rebound50_sec": 537.0,
      "retest_quality_bucket": "RETEST_QUALITY_LOW",
      "signal_ts_ms": "1776694119155",
      "signal_utc": "2026-04-20T14:08:39.155000+00:00",
      "sol30_bps": -36.5,
      "source_observation_id": "6af7c4aab495fedd0e257602",
      "spread_bps": 0.0,
      "state_path_v2": "ANCHOR>LATE_FILL>PAIN_GE50>REBOUND100>CROSS_OK>RUNNER_H4",
      "vdepth_bps": 31.4,
      "h4_minus_h2_bps": 152.5,
      "runner_h4": true,
      "fill_delay_bucket": "LATE_GT900",
      "mae_bucket": "PAIN_50_100",
      "rebound50_bucket": "REBOUND50_FAST_30M",
      "month": "2026-04"
    },
    {
      "bid_depth_usd": 293221.2,
      "book_imbalance": 0.903,
      "btc30_bps": 27.7,
      "bucket": "H4_SHADOW",
      "cross_no_dump": true,
      "entry_price": 1544.3,
      "entry_quality_tags": "POST_ARM_FILL,PULLBACK_FILL,RETEST_BAND_2_25,SPREAD_CLEAN,NO_LARGE_SELL_LIQ_RESTART",
      "entry_quality_warnings": "BID_VANISHED,LATE_RETEST_FILL",
      "exit_horizon_sec": 14400,
      "exit_mode": "H4_SHADOW",
      "exit_price": 1582.23,
      "exit_price_source": "book_bid",
      "exit_ts_ms": 1782489059012,
      "fill_delay_sec": 4285.0,
      "fill_leg": "replacement",
      "h2_net_bps": 17.2,
      "h4_net_bps": 240.4,
      "mae_bps": -147.2,
      "maker_fill_ts_ms": 1782474659001,
      "maker_fill_utc": "2026-06-26T11:50:59.001000+00:00",
      "mfe_bps": 276.5,
      "net_bps": 240.4,
      "notes": "shadow_observation_only_no_order",
      "observation_status": "CLOSED",
      "prior_4h_bps": -141.6,
      "protocol_id": "S34_V_ENGINE_V0_2_ETH_SELL_MAKER_LONG_H2_O20_W300_O5_DEEPBID",
      "rebound50_sec": 2530.0,
      "retest_quality_bucket": "RETEST_QUALITY_LOW",
      "signal_ts_ms": "1782470373530",
      "signal_utc": "2026-06-26T10:39:33.530000+00:00",
      "sol30_bps": 62.5,
      "source_observation_id": "0b4c2340a7b04cb90c9556c9",
      "spread_bps": 0.1,
      "state_path_v2": "ANCHOR>LATE_FILL>PAIN_GE50>REBOUND100>CROSS_OK>RUNNER_H4",
      "vdepth_bps": 34.4,
      "h4_minus_h2_bps": 223.2,
      "runner_h4": true,
      "fill_delay_bucket": "LATE_GT900",
      "mae_bucket": "PAIN_GE100",
      "rebound50_bucket": "REBOUND50_LATE",
      "month": "2026-06"
    }
  ],
  "by_delay_h4": {
    "FAST_0_60": {
      "n": 2,
      "sum": 424.1,
      "mean": 212.1,
      "median": 212.1,
      "win_rate": 1.0,
      "t3r": 424.1,
      "min": 29.0,
      "max": 395.1
    },
    "LATE_GT900": {
      "n": 2,
      "sum": 419.9,
      "mean": 209.9,
      "median": 209.9,
      "win_rate": 1.0,
      "t3r": 419.9,
      "min": 179.5,
      "max": 240.4
    },
    "NORMAL_60_900": {
      "n": 7,
      "sum": 894.6,
      "mean": 127.8,
      "median": 130.1,
      "win_rate": 1.0,
      "t3r": 278.3,
      "min": 6.0,
      "max": 283.9
    }
  },
  "verdict": "PROXY_ONLY_NEEDS_TICK_QUEUE_REPLAY"
}
```

## 06_shadow_paper_bucket_health

```json
{
  "mirror_rows_total": 11,
  "mirror_closed_filled": 11,
  "h4_closed_filled": 11,
  "h4_ledger_rows": 44,
  "expected_h4_ledger_rows": 44,
  "paper_status": {
    "updated_at_utc": "2026-06-30T07:54:08.839947+00:00",
    "total_trades": 855,
    "open_trades": 0,
    "closed_trades": 130
  },
  "live_state": {
    "mode": "LIVE",
    "active": null,
    "orders_count": 0
  },
  "verdict": "PARITY_OK"
}
```

## 07_regime_drift

```json
{
  "by_month_h2": {
    "2026-04": {
      "n": 4,
      "sum": 147.5,
      "mean": 36.9,
      "median": 26.6,
      "win_rate": 1.0,
      "t3r": 13.3,
      "min": 13.3,
      "max": 81.1
    },
    "2026-06": {
      "n": 7,
      "sum": 934.1,
      "mean": 133.4,
      "median": 149.9,
      "win_rate": 1.0,
      "t3r": 255.1,
      "min": 17.2,
      "max": 299.7
    }
  },
  "by_month_h4": {
    "2026-04": {
      "n": 4,
      "sum": 379.1,
      "mean": 94.8,
      "median": 96.8,
      "win_rate": 1.0,
      "t3r": 6.0,
      "min": 6.0,
      "max": 179.5
    },
    "2026-06": {
      "n": 7,
      "sum": 1359.5,
      "mean": 194.2,
      "median": 167.8,
      "win_rate": 1.0,
      "t3r": 440.1,
      "min": 37.2,
      "max": 395.1
    }
  },
  "mirror_weekly": [
    {
      "week": "2026-W16",
      "signals": 2,
      "closed": 2,
      "pending": 0,
      "filled": 2,
      "fill_rate": 1.0,
      "summary": {
        "n": 2,
        "sum_bps": 107.2,
        "mean_bps": 53.6,
        "median_bps": 53.6,
        "p10_bps": 31.6,
        "p90_bps": 75.6,
        "win_rate": 1.0,
        "profit_factor": null,
        "max_win_bps": 81.1,
        "max_loss_bps": 26.1,
        "top3_winner_removed_sum_bps": 107.2,
        "bottom3_loser_removed_sum_bps": 107.2
      }
    },
    {
      "week": "2026-W17",
      "signals": 2,
      "closed": 2,
      "pending": 0,
      "filled": 2,
      "fill_rate": 1.0,
      "summary": {
        "n": 2,
        "sum_bps": 40.3,
        "mean_bps": 20.1,
        "median_bps": 20.1,
        "p10_bps": 14.7,
        "p90_bps": 25.6,
        "win_rate": 1.0,
        "profit_factor": null,
        "max_win_bps": 27.0,
        "max_loss_bps": 13.3,
        "top3_winner_removed_sum_bps": 40.3,
        "bottom3_loser_removed_sum_bps": 40.3
      }
    },
    {
      "week": "2026-W25",
      "signals": 4,
      "closed": 4,
      "pending": 0,
      "filled": 4,
      "fill_rate": 1.0,
      "summary": {
        "n": 4,
        "sum_bps": 467.3,
        "mean_bps": 116.8,
        "median_bps": 99.3,
        "p10_bps": 43.1,
        "p90_bps": 204.6,
        "win_rate": 1.0,
        "profit_factor": null,
        "max_win_bps": 227.0,
        "max_loss_bps": 41.7,
        "top3_winner_removed_sum_bps": 41.7,
        "bottom3_loser_removed_sum_bps": 227.0
      }
    },
    {
      "week": "2026-W26",
      "signals": 3,
      "closed": 3,
      "pending": 0,
      "filled": 3,
      "fill_rate": 1.0,
      "summary": {
        "n": 3,
        "sum_bps": 466.8,
        "mean_bps": 155.6,
        "median_bps": 149.9,
        "p10_bps": 43.7,
        "p90_bps": 269.7,
        "win_rate": 1.0,
        "profit_factor": null,
        "max_win_bps": 299.7,
        "max_loss_bps": 17.2,
        "top3_winner_removed_sum_bps": 466.8,
        "bottom3_loser_removed_sum_bps": 466.8
      }
    }
  ],
  "verdict": "TOO_FEW_MONTHS_FOR_DRIFT_DECISION"
}
```

## 08_navigation_indicator_context

```json
{
  "nav_overlay_baseline": {
    "n": 11,
    "sum": 1077.7,
    "mean": 98.0,
    "median": 46.2,
    "win_rate": 1.0,
    "t3r": 391.2,
    "min": 15.0,
    "max": 302.2
  },
  "by_nav_high_fill": null,
  "by_nav_high_holds_5m": null,
  "by_nav_score_bucket": {
    "NAV_HIGH": {
      "n": 3,
      "sum": 277.7,
      "mean": 92.6,
      "median": 28.6,
      "win_rate": 1.0,
      "t3r": 277.7,
      "min": 19.7,
      "max": 229.5
    },
    "NAV_LOW": {
      "n": 6,
      "sum": 618.4,
      "mean": 103.1,
      "median": 64.9,
      "win_rate": 1.0,
      "t3r": 77.8,
      "min": 15.0,
      "max": 302.2
    },
    "NAV_MID": {
      "n": 2,
      "sum": 181.6,
      "mean": 90.8,
      "median": 90.8,
      "win_rate": 1.0,
      "t3r": 181.6,
      "min": 44.2,
      "max": 137.4
    }
  },
  "state_sequences_top": {
    "LMHHHM": {
      "n": 2,
      "sum": 31.6,
      "mean": 15.8,
      "median": 15.8,
      "win_rate": 1.0,
      "t3r": 31.6,
      "min": 15.0,
      "max": 16.6
    },
    "LLLLHH": {
      "n": 1,
      "sum": 83.6,
      "mean": 83.6,
      "median": 83.6,
      "win_rate": 1.0,
      "t3r": 83.6,
      "min": 83.6,
      "max": 83.6
    },
    "HHHMHH": {
      "n": 1,
      "sum": 28.6,
      "mean": 28.6,
      "median": 28.6,
      "win_rate": 1.0,
      "t3r": 28.6,
      "min": 28.6,
      "max": 28.6
    },
    "LLMHMH": {
      "n": 1,
      "sum": 46.2,
      "mean": 46.2,
      "median": 46.2,
      "win_rate": 1.0,
      "t3r": 46.2,
      "min": 46.2,
      "max": 46.2
    },
    "MMHHHH": {
      "n": 1,
      "sum": 44.2,
      "mean": 44.2,
      "median": 44.2,
      "win_rate": 1.0,
      "t3r": 44.2,
      "min": 44.2,
      "max": 44.2
    },
    "LMLLHM": {
      "n": 1,
      "sum": 154.8,
      "mean": 154.8,
      "median": 154.8,
      "win_rate": 1.0,
      "t3r": 154.8,
      "min": 154.8,
      "max": 154.8
    },
    "HMMMMM": {
      "n": 1,
      "sum": 229.5,
      "mean": 229.5,
      "median": 229.5,
      "win_rate": 1.0,
      "t3r": 229.5,
      "min": 229.5,
      "max": 229.5
    },
    "MMHMHL": {
      "n": 1,
      "sum": 137.4,
      "mean": 137.4,
      "median": 137.4,
      "win_rate": 1.0,
      "t3r": 137.4,
      "min": 137.4,
      "max": 137.4
    },
    "MLLHHM": {
      "n": 1,
      "sum": 19.7,
      "mean": 19.7,
      "median": 19.7,
      "win_rate": 1.0,
      "t3r": 19.7,
      "min": 19.7,
      "max": 19.7
    },
    "LHMHHH": {
      "n": 1,
      "sum": 302.2,
      "mean": 302.2,
      "median": 302.2,
      "win_rate": 1.0,
      "t3r": 302.2,
      "min": 302.2,
      "max": 302.2
    }
  },
  "verdict": "NAV_CONTEXT_ONLY_NOT_ENTRY_RULE"
}
```

## 09_state_sequence_model

```json
{
  "h4_state_counts": {
    "ANCHOR>QUALITY_FILL>PAIN_GE50>REBOUND100>CROSS_DUMP>H2_BETTER": 1,
    "ANCHOR>QUALITY_FILL>CLEAN_PATH>REBOUND50>CROSS_OK>H2_BETTER": 1,
    "ANCHOR>LATE_FILL>PAIN_GE50>REBOUND100>CROSS_OK>RUNNER_H4": 2,
    "ANCHOR>QUALITY_FILL>PAIN_GE50>REBOUND100>CROSS_OK>RUNNER_H4": 2,
    "ANCHOR>QUALITY_FILL>CLEAN_PATH>REBOUND100>CROSS_OK>RUNNER_H4": 1,
    "ANCHOR>QUALITY_FILL>CLEAN_PATH>REBOUND100>CROSS_OK>H2_BETTER": 2,
    "ANCHOR>BASE_FILL>CLEAN_PATH>REBOUND100>CROSS_OK>H2_BETTER": 1,
    "ANCHOR>BID_VANISHED>CLEAN_PATH>REBOUND100>CROSS_OK>RUNNER_H4": 1
  },
  "h4_by_state": {
    "ANCHOR>BASE_FILL>CLEAN_PATH>REBOUND100>CROSS_OK>H2_BETTER": {
      "n": 1,
      "sum": 105.0,
      "mean": 105.0,
      "median": 105.0,
      "win_rate": 1.0,
      "t3r": 105.0,
      "min": 105.0,
      "max": 105.0
    },
    "ANCHOR>BID_VANISHED>CLEAN_PATH>REBOUND100>CROSS_OK>RUNNER_H4": {
      "n": 1,
      "sum": 395.1,
      "mean": 395.1,
      "median": 395.1,
      "win_rate": 1.0,
      "t3r": 395.1,
      "min": 395.1,
      "max": 395.1
    },
    "ANCHOR>LATE_FILL>PAIN_GE50>REBOUND100>CROSS_OK>RUNNER_H4": {
      "n": 2,
      "sum": 419.9,
      "mean": 209.9,
      "median": 209.9,
      "win_rate": 1.0,
      "t3r": 419.9,
      "min": 179.5,
      "max": 240.4
    },
    "ANCHOR>QUALITY_FILL>CLEAN_PATH>REBOUND100>CROSS_OK>H2_BETTER": {
      "n": 2,
      "sum": 205.0,
      "mean": 102.5,
      "median": 102.5,
      "win_rate": 1.0,
      "t3r": 205.0,
      "min": 37.2,
      "max": 167.8
    },
    "ANCHOR>QUALITY_FILL>CLEAN_PATH>REBOUND100>CROSS_OK>RUNNER_H4": {
      "n": 1,
      "sum": 130.1,
      "mean": 130.1,
      "median": 130.1,
      "win_rate": 1.0,
      "t3r": 130.1,
      "min": 130.1,
      "max": 130.1
    },
    "ANCHOR>QUALITY_FILL>CLEAN_PATH>REBOUND50>CROSS_OK>H2_BETTER": {
      "n": 1,
      "sum": 6.0,
      "mean": 6.0,
      "median": 6.0,
      "win_rate": 1.0,
      "t3r": 6.0,
      "min": 6.0,
      "max": 6.0
    },
    "ANCHOR>QUALITY_FILL>PAIN_GE50>REBOUND100>CROSS_DUMP>H2_BETTER": {
      "n": 1,
      "sum": 29.0,
      "mean": 29.0,
      "median": 29.0,
      "win_rate": 1.0,
      "t3r": 29.0,
      "min": 29.0,
      "max": 29.0
    },
    "ANCHOR>QUALITY_FILL>PAIN_GE50>REBOUND100>CROSS_OK>RUNNER_H4": {
      "n": 2,
      "sum": 448.5,
      "mean": 224.2,
      "median": 224.2,
      "win_rate": 1.0,
      "t3r": 448.5,
      "min": 164.6,
      "max": 283.9
    }
  },
  "h4_delta_by_state": {
    "ANCHOR>BASE_FILL>CLEAN_PATH>REBOUND100>CROSS_OK>H2_BETTER": {
      "n": 1,
      "sum": -47.3,
      "mean": -47.3,
      "median": -47.3,
      "win_rate": 0.0,
      "t3r": -47.3,
      "min": -47.3,
      "max": -47.3
    },
    "ANCHOR>BID_VANISHED>CLEAN_PATH>REBOUND100>CROSS_OK>RUNNER_H4": {
      "n": 1,
      "sum": 95.4,
      "mean": 95.4,
      "median": 95.4,
      "win_rate": 1.0,
      "t3r": 95.4,
      "min": 95.4,
      "max": 95.4
    },
    "ANCHOR>LATE_FILL>PAIN_GE50>REBOUND100>CROSS_OK>RUNNER_H4": {
      "n": 2,
      "sum": 375.7,
      "mean": 187.8,
      "median": 187.8,
      "win_rate": 1.0,
      "t3r": 375.7,
      "min": 152.5,
      "max": 223.2
    },
    "ANCHOR>QUALITY_FILL>CLEAN_PATH>REBOUND100>CROSS_OK>H2_BETTER": {
      "n": 2,
      "sum": -63.7,
      "mean": -31.9,
      "median": -31.9,
      "win_rate": 0.0,
      "t3r": -63.7,
      "min": -59.2,
      "max": -4.5
    },
    "ANCHOR>QUALITY_FILL>CLEAN_PATH>REBOUND100>CROSS_OK>RUNNER_H4": {
      "n": 1,
      "sum": 83.8,
      "mean": 83.8,
      "median": 83.8,
      "win_rate": 1.0,
      "t3r": 83.8,
      "min": 83.8,
      "max": 83.8
    },
    "ANCHOR>QUALITY_FILL>CLEAN_PATH>REBOUND50>CROSS_OK>H2_BETTER": {
      "n": 1,
      "sum": -20.1,
      "mean": -20.1,
      "median": -20.1,
      "win_rate": 0.0,
      "t3r": -20.1,
      "min": -20.1,
      "max": -20.1
    },
    "ANCHOR>QUALITY_FILL>PAIN_GE50>REBOUND100>CROSS_DUMP>H2_BETTER": {
      "n": 1,
      "sum": -52.1,
      "mean": -52.1,
      "median": -52.1,
      "win_rate": 0.0,
      "t3r": -52.1,
      "min": -52.1,
      "max": -52.1
    },
    "ANCHOR>QUALITY_FILL>PAIN_GE50>REBOUND100>CROSS_OK>RUNNER_H4": {
      "n": 2,
      "sum": 285.3,
      "mean": 142.7,
      "median": 142.7,
      "win_rate": 1.0,
      "t3r": 285.3,
      "min": 134.0,
      "max": 151.3
    }
  },
  "nav_state_sequences_top": {
    "LMHHHM": {
      "n": 2,
      "sum": 31.6,
      "mean": 15.8,
      "median": 15.8,
      "win_rate": 1.0,
      "t3r": 31.6,
      "min": 15.0,
      "max": 16.6
    },
    "LLLLHH": {
      "n": 1,
      "sum": 83.6,
      "mean": 83.6,
      "median": 83.6,
      "win_rate": 1.0,
      "t3r": 83.6,
      "min": 83.6,
      "max": 83.6
    },
    "HHHMHH": {
      "n": 1,
      "sum": 28.6,
      "mean": 28.6,
      "median": 28.6,
      "win_rate": 1.0,
      "t3r": 28.6,
      "min": 28.6,
      "max": 28.6
    },
    "LLMHMH": {
      "n": 1,
      "sum": 46.2,
      "mean": 46.2,
      "median": 46.2,
      "win_rate": 1.0,
      "t3r": 46.2,
      "min": 46.2,
      "max": 46.2
    },
    "MMHHHH": {
      "n": 1,
      "sum": 44.2,
      "mean": 44.2,
      "median": 44.2,
      "win_rate": 1.0,
      "t3r": 44.2,
      "min": 44.2,
      "max": 44.2
    },
    "LMLLHM": {
      "n": 1,
      "sum": 154.8,
      "mean": 154.8,
      "median": 154.8,
      "win_rate": 1.0,
      "t3r": 154.8,
      "min": 154.8,
      "max": 154.8
    },
    "HMMMMM": {
      "n": 1,
      "sum": 229.5,
      "mean": 229.5,
      "median": 229.5,
      "win_rate": 1.0,
      "t3r": 229.5,
      "min": 229.5,
      "max": 229.5
    },
    "MMHMHL": {
      "n": 1,
      "sum": 137.4,
      "mean": 137.4,
      "median": 137.4,
      "win_rate": 1.0,
      "t3r": 137.4,
      "min": 137.4,
      "max": 137.4
    },
    "MLLHHM": {
      "n": 1,
      "sum": 19.7,
      "mean": 19.7,
      "median": 19.7,
      "win_rate": 1.0,
      "t3r": 19.7,
      "min": 19.7,
      "max": 19.7
    },
    "LHMHHH": {
      "n": 1,
      "sum": 302.2,
      "mean": 302.2,
      "median": 302.2,
      "win_rate": 1.0,
      "t3r": 302.2,
      "min": 302.2,
      "max": 302.2
    }
  },
  "verdict": "PROMISING_STRUCTURE_BUT_SMALL_N"
}
```

## 10_kill_promote_rules

```json
{
  "promote_gate": {
    "min_forward_closed_fills": 30,
    "min_calendar_days": 30,
    "requires_h4_sum_gt_0": true,
    "requires_h4_t3r_gt_0": true,
    "requires_no_single_winner_dependence": true,
    "requires_operator_approval": true
  },
  "current_status": {
    "closed_fills": 11,
    "h4_sum": 1738.6,
    "h4_t3r": 819.2,
    "decision": "DO_NOT_PROMOTE_YET_SMALL_N"
  },
  "kill_gate": {
    "30_or_60_day_forward_sum_lt_0": "pause_or_disarm_review",
    "forward_t3r_lt_0_after_min_3": "pause_review",
    "tail_or_stop_budget_breach": "operator_size_review"
  }
}
```
