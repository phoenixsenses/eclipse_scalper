# S34 V02 H4 Shadow Control Plane

Generated: `2026-06-29T20:03:05.558057+00:00`

H4 shadow remains the strongest bucket on the current mirror sample: H2 sum 1081.6 / T3R 402.6 vs H4 sum 1738.6 / T3R 819.2. This is shadow-only and small-N; it is not a live promotion.

## Bucket Results

| Bucket | N | Sum | Median | Win | T3R | Min | Max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| H2_CURRENT | 11 | 1081.6 | 46.3 | 1.0 | 402.6 | 13.3 | 299.7 |
| H3_SHADOW | 11 | 1397.6 | 127.7 | 0.909 | 651.4 | -30.2 | 335.5 |
| H4_SHADOW | 11 | 1738.6 | 164.6 | 1.0 | 819.2 | 6.0 | 395.1 |
| H4_CROSS_NO_DUMP_SHADOW | 11 | 1790.7 | 164.6 | 1.0 | 871.3 | 6.0 | 395.1 |

## Cross-No-Dump Observer

```json
{
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
}
```

## Catastrophic Stop Observer

```json
{
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
}
```

## Queue / Fill Realism

```json
{
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
}
```

## Live / Shadow Parity

```json
{
  "status": "PASS",
  "live_executor_path": "D:\\eclipse_scalper\\tools\\s34_v_engine_live_executor.py",
  "checks": [
    {
      "field": "RULE_NAME/protocol_id",
      "live": "S34_V_ENGINE_V0_2_ETH_SELL_MAKER_LONG_H2_O20_W300_O5_DEEPBID",
      "shadow": "S34_V_ENGINE_V0_2_ETH_SELL_MAKER_LONG_H2_O20_W300_O5_DEEPBID",
      "match": true
    },
    {
      "field": "symbol",
      "live": "ETHUSDT",
      "shadow": "ETHUSDT",
      "match": true
    },
    {
      "field": "liq_side",
      "live": "SELL",
      "shadow": "SELL",
      "match": true
    },
    {
      "field": "threshold_usd",
      "live": 200000.0,
      "shadow": 200000.0,
      "match": true
    },
    {
      "field": "initial_offset_bps",
      "live": 20.0,
      "shadow": 20.0,
      "match": true
    },
    {
      "field": "replace_offset_bps",
      "live": 5.0,
      "shadow": 5.0,
      "match": true
    },
    {
      "field": "wait_sec",
      "live": 300.0,
      "shadow": 300.0,
      "match": true
    }
  ],
  "note": "Read-only parity audit; no live files were modified."
}
```

## State Machine v2

```json
{
  "state_counts": {
    "ANCHOR>QUALITY_FILL>PAIN_GE50>REBOUND100>CROSS_DUMP>H2_BETTER": 1,
    "ANCHOR>QUALITY_FILL>CLEAN_PATH>REBOUND50>CROSS_OK>H2_BETTER": 1,
    "ANCHOR>LATE_FILL>PAIN_GE50>REBOUND100>CROSS_OK>RUNNER_H4": 2,
    "ANCHOR>QUALITY_FILL>PAIN_GE50>REBOUND100>CROSS_OK>RUNNER_H4": 2,
    "ANCHOR>QUALITY_FILL>CLEAN_PATH>REBOUND100>CROSS_OK>RUNNER_H4": 1,
    "ANCHOR>QUALITY_FILL>CLEAN_PATH>REBOUND100>CROSS_OK>H2_BETTER": 2,
    "ANCHOR>BASE_FILL>CLEAN_PATH>REBOUND100>CROSS_OK>H2_BETTER": 1,
    "ANCHOR>BID_VANISHED>CLEAN_PATH>REBOUND100>CROSS_OK>RUNNER_H4": 1
  },
  "by_state": {
    "ANCHOR>QUALITY_FILL>PAIN_GE50>REBOUND100>CROSS_DUMP>H2_BETTER": {
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
    "ANCHOR>QUALITY_FILL>CLEAN_PATH>REBOUND50>CROSS_OK>H2_BETTER": {
      "n": 1,
      "sum_bps": 6.0,
      "mean_bps": 6.0,
      "median_bps": 6.0,
      "win_rate": 1.0,
      "t3r_bps": 6.0,
      "top1_removed_bps": 6.0,
      "min_bps": 6.0,
      "max_bps": 6.0
    },
    "ANCHOR>LATE_FILL>PAIN_GE50>REBOUND100>CROSS_OK>RUNNER_H4": {
      "n": 2,
      "sum_bps": 419.9,
      "mean_bps": 209.9,
      "median_bps": 209.9,
      "win_rate": 1.0,
      "t3r_bps": 419.9,
      "top1_removed_bps": 179.5,
      "min_bps": 179.5,
      "max_bps": 240.4
    },
    "ANCHOR>QUALITY_FILL>PAIN_GE50>REBOUND100>CROSS_OK>RUNNER_H4": {
      "n": 2,
      "sum_bps": 448.5,
      "mean_bps": 224.2,
      "median_bps": 224.2,
      "win_rate": 1.0,
      "t3r_bps": 448.5,
      "top1_removed_bps": 164.6,
      "min_bps": 164.6,
      "max_bps": 283.9
    },
    "ANCHOR>QUALITY_FILL>CLEAN_PATH>REBOUND100>CROSS_OK>RUNNER_H4": {
      "n": 1,
      "sum_bps": 130.1,
      "mean_bps": 130.1,
      "median_bps": 130.1,
      "win_rate": 1.0,
      "t3r_bps": 130.1,
      "top1_removed_bps": 130.1,
      "min_bps": 130.1,
      "max_bps": 130.1
    },
    "ANCHOR>QUALITY_FILL>CLEAN_PATH>REBOUND100>CROSS_OK>H2_BETTER": {
      "n": 2,
      "sum_bps": 205.0,
      "mean_bps": 102.5,
      "median_bps": 102.5,
      "win_rate": 1.0,
      "t3r_bps": 205.0,
      "top1_removed_bps": 37.2,
      "min_bps": 37.2,
      "max_bps": 167.8
    },
    "ANCHOR>BASE_FILL>CLEAN_PATH>REBOUND100>CROSS_OK>H2_BETTER": {
      "n": 1,
      "sum_bps": 105.0,
      "mean_bps": 105.0,
      "median_bps": 105.0,
      "win_rate": 1.0,
      "t3r_bps": 105.0,
      "top1_removed_bps": 105.0,
      "min_bps": 105.0,
      "max_bps": 105.0
    },
    "ANCHOR>BID_VANISHED>CLEAN_PATH>REBOUND100>CROSS_OK>RUNNER_H4": {
      "n": 1,
      "sum_bps": 395.1,
      "mean_bps": 395.1,
      "median_bps": 395.1,
      "win_rate": 1.0,
      "t3r_bps": 395.1,
      "top1_removed_bps": 395.1,
      "min_bps": 395.1,
      "max_bps": 395.1
    }
  }
}
```

## Forced-Flow Expansion

```json
{
  "status": "LOADED_EXISTING_SCAN",
  "path": "D:\\eclipse_scalper\\reports\\research\\s34\\S34_V02_NEXT_GEN_ALPHA_RESEARCH_30D.json",
  "summary": {
    "sell_notional_p95": 43289106.1,
    "candidates_total": 2,
    "filled_n": 2,
    "result": {
      "n": 2,
      "sum": 113.4,
      "mean": 56.7,
      "median": 56.7,
      "win_rate": 1.0,
      "t3r": 113.4,
      "min": 1.5,
      "max": 111.9
    },
    "sample": [
      {
        "status": "FILLED",
        "bucket": 1780245000000,
        "utc": "2026-05-31T16:30:00+00:00",
        "sell_notional": 104841477.5,
        "buy_notional": 39014994.4,
        "depth_bps": 32.2,
        "prior4h_bps": -79.5,
        "bid_depth_usd": 401765.1,
        "fill_delay_sec": 2116.0,
        "net_2h_bps": 1.4806019893148363
      },
      {
        "status": "FILLED",
        "bucket": 1780326900000,
        "utc": "2026-06-01T15:15:00+00:00",
        "sell_notional": 70292334.3,
        "buy_notional": 35571981.2,
        "depth_bps": 34.0,
        "prior4h_bps": -67.2,
        "bid_depth_usd": 764272.1,
        "fill_delay_sec": 313.0,
        "net_2h_bps": 111.93661092802381
      }
    ]
  },
  "interpretation": "Existing forced-flow expansion remains small-N/research-only; not a live route."
}
```
