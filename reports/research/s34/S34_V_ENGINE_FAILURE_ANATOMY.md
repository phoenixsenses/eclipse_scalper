# S34 V Engine Failure Anatomy

Generated: `2026-06-28T19:20:12.675141+00:00`

Protocol: `S34_V_ENGINE_V0_1_ETH_SELL_MAKER_LONG_H2_O20_V28_40_P4D`

Diagnostic only. This report explains closed maker-fill observations; it does not change v0.1.

## Sample

- ledger rows: `47`
- closed filled rows: `19`
- closed no-fill rows: `7`
- data incomplete rows: `21`
- overall closed-fill labels: N=19 sum=876.1 med=37.0 T3R=348.4

## Winner vs Loser Profile

```json
{
  "winners": {
    "n": 16,
    "net_bps": {
      "median": 43.2,
      "mean": 67.1,
      "p25": 30.0,
      "p75": 72.6
    },
    "fill_delay_sec": {
      "median": 301.0,
      "mean": 1025.2,
      "p25": 51.2,
      "p75": 857.2
    },
    "vdepth_bps": {
      "median": 32.0,
      "mean": 32.6,
      "p25": 29.3,
      "p75": 34.7
    },
    "prior_4h_bps": {
      "median": -83.8,
      "mean": -147.3,
      "p25": -183.2,
      "p75": -67.6
    },
    "ret_15m_bps": {
      "median": 29.9,
      "mean": 19.1,
      "p25": 5.4,
      "p75": 37.2
    },
    "mae_30m_bps": {
      "median": -16.0,
      "mean": -34.2,
      "p25": -60.3,
      "p75": -4.9
    },
    "mfe_30m_bps": {
      "median": 54.2,
      "mean": 58.4,
      "p25": 38.2,
      "p75": 64.6
    },
    "btc_after_15m_bps": {
      "median": 19.9,
      "mean": 18.4,
      "p25": 12.2,
      "p75": 31.9
    },
    "candle5_lower_wick_bps": {
      "median": 6.3,
      "mean": 11.9,
      "p25": 1.8,
      "p75": 15.9
    },
    "candle5_close_ret_bps": {
      "median": 10.2,
      "mean": 5.6,
      "p25": -12.6,
      "p75": 20.9
    }
  },
  "losers": {
    "n": 3,
    "net_bps": {
      "median": -35.9,
      "mean": -66.0,
      "p25": -91.0,
      "p75": -26.0
    },
    "fill_delay_sec": {
      "median": 426.0,
      "mean": 428.3,
      "p25": 219.5,
      "p75": 636.0
    },
    "vdepth_bps": {
      "median": 34.0,
      "mean": 34.1,
      "p25": 32.3,
      "p75": 35.9
    },
    "prior_4h_bps": {
      "median": -273.4,
      "mean": -258.8,
      "p25": -347.5,
      "p75": -177.3
    },
    "ret_15m_bps": {
      "median": -19.4,
      "mean": -77.3,
      "p25": -115.7,
      "p75": -10.0
    },
    "mae_30m_bps": {
      "median": -40.6,
      "mean": -108.1,
      "p25": -142.3,
      "p75": -40.2
    },
    "mfe_30m_bps": {
      "median": 7.8,
      "mean": 2.1,
      "p25": -0.9,
      "p75": 7.9
    },
    "btc_after_15m_bps": {
      "median": -15.7,
      "mean": -37.7,
      "p25": -63.1,
      "p75": -1.3
    },
    "candle5_lower_wick_bps": {
      "median": 10.3,
      "mean": 8.9,
      "p25": 6.9,
      "p75": 11.6
    },
    "candle5_close_ret_bps": {
      "median": -0.3,
      "mean": -50.1,
      "p25": -75.8,
      "p75": 0.5
    }
  }
}
```

## Trap / Leading Area Screens

| Feature | Value | N | Loser% | Summary |
| --- | --- | ---: | ---: | --- |
| `fill_delay_bucket` | `fill_2_10m` | 6 | 16.7 | N=6 sum=311.3 med=47.7 T3R=51.6 |
| `fill_delay_bucket` | `fill_10_30m` | 4 | 25.0 | N=4 sum=-24.2 med=23.5 T3R=-146.0 |
| `fill_delay_bucket` | `fill_30_120s` | 3 | 0.0 | N=3 sum=425.4 med=81.1 T3R=425.4 |
| `fill_delay_bucket` | `fill_30m_plus` | 3 | 0.0 | N=3 sum=132.1 med=32.3 T3R=132.1 |
| `fill_delay_bucket` | `fill_0_30s` | 3 | 33.3 | N=3 sum=31.5 med=30.4 T3R=31.5 |
| `first_15m_bucket` | `ret15_rebound` | 9 | 0.0 | N=9 sum=642.8 med=44.6 T3R=212.1 |
| `first_15m_bucket` | `ret15_soft_green` | 4 | 0.0 | N=4 sum=293.7 med=58.2 T3R=30.4 |
| `first_15m_bucket` | `ret15_dump` | 4 | 25.0 | N=4 sum=-8.3 med=28.3 T3R=-146.0 |
| `first_15m_bucket` | `ret15_soft_red` | 2 | 100.0 | N=2 sum=-52.1 med=-26.0 T3R=-52.1 |
| `low_rebreak_15m` | `True` | 13 | 23.1 | N=13 sum=657.2 med=44.6 T3R=129.5 |
| `low_rebreak_15m` | `False` | 6 | 0.0 | N=6 sum=218.9 med=31.3 T3R=74.9 |
| `low_rebreak_30m` | `True` | 13 | 23.1 | N=13 sum=657.2 med=44.6 T3R=129.5 |
| `low_rebreak_30m` | `False` | 6 | 0.0 | N=6 sum=218.9 med=31.3 T3R=74.9 |
| `anchor_reclaimed_15m` | `True` | 14 | 0.0 | N=14 sum=846.2 med=39.4 T3R=399.9 |
| `anchor_reclaimed_15m` | `False` | 5 | 60.0 | N=5 sum=29.9 med=-16.2 T3R=-181.9 |
| `prior4h_intensity_bucket` | `prior4h_mild_down` | 10 | 10.0 | N=10 sum=309.1 med=29.5 T3R=132.6 |
| `prior4h_intensity_bucket` | `prior4h_medium_down` | 4 | 0.0 | N=4 sum=451.3 med=59.7 T3R=32.3 |
| `prior4h_intensity_bucket` | `prior4h_hard_down` | 3 | 33.3 | N=3 sum=72.8 med=71.9 T3R=72.8 |
| `prior4h_intensity_bucket` | `prior4h_extreme_down` | 2 | 50.0 | N=2 sum=42.9 med=21.4 T3R=42.9 |
| `btc_context_bucket` | `btc_down_then_stable` | 8 | 0.0 | N=8 sum=747.6 med=65.5 T3R=226.3 |
| `btc_context_bucket` | `btc_supportive` | 6 | 16.7 | N=6 sum=153.0 med=33.7 T3R=20.6 |
| `btc_context_bucket` | `btc_down_continues` | 3 | 66.7 | N=3 sum=-81.1 med=-16.2 T3R=-81.1 |
| `btc_context_bucket` | `btc_softening` | 2 | 0.0 | N=2 sum=56.6 med=28.3 T3R=56.6 |
| `candle5_pattern` | `bull_reclaim` | 8 | 0.0 | N=8 sum=581.7 med=39.4 T3R=153.6 |
| `candle5_pattern` | `bear_followthrough` | 4 | 25.0 | N=4 sum=-8.3 med=28.3 T3R=-146.0 |
| `candle5_pattern` | `failed_hammer` | 3 | 0.0 | N=3 sum=175.6 med=59.1 T3R=175.6 |
| `candle5_pattern` | `neutral` | 2 | 0.0 | N=2 sum=179.2 med=89.6 T3R=179.2 |
| `candle5_pattern` | `hammer_reversal` | 2 | 100.0 | N=2 sum=-52.1 med=-26.0 T3R=-52.1 |
| `candle15_pattern` | `bull_reclaim` | 11 | 0.0 | N=11 sum=747.9 med=44.6 T3R=301.6 |
| `candle15_pattern` | `bear_followthrough` | 5 | 40.0 | N=5 sum=-24.5 med=27.9 T3R=-162.2 |
| `candle15_pattern` | `hammer_reversal` | 2 | 50.0 | N=2 sum=111.0 med=55.5 T3R=111.0 |

## Top Losers

| Net | UTC | Tags | Fill delay | V-depth | Prior4h | Ret15 | MAE30 | BTC | C5 | C15 |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| -146.0 | 2026-06-23T07:59:44.477000+00:00 | `low_rebreak_15m,late_fill_gt10m,weak_first_15m,candle5_bear_followthrough,btc_down_continues` | 846.0 | 30.6 | -273.4 | -211.9 | -244.0 | btc_down_continues | bear_followthrough | bear_followthrough |
| -35.9 | 2026-06-16T02:27:55.467000+00:00 | `low_rebreak_15m,weak_first_15m` | 13.0 | 34.0 | -81.3 | -0.7 | -40.6 | btc_supportive | hammer_reversal | hammer_reversal |
| -16.2 | 2026-06-25T16:32:03.169000+00:00 | `low_rebreak_15m,weak_first_15m,btc_down_continues` | 426.0 | 37.8 | -421.7 | -19.4 | -39.8 | btc_down_continues | hammer_reversal | bear_followthrough |

## Top Winners

| Net | UTC | Tags | Fill delay | V-depth | Prior4h | Ret15 | MFE30 | BTC | C5 | C15 |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| 299.7 | 2026-06-26T13:18:54.877000+00:00 | `low_rebreak_15m` | 37.0 | 36.3 | -225.6 | 61.9 | 166.9 | btc_down_then_stable | bull_reclaim | bull_reclaim |
| 146.9 | 2026-06-26T02:48:30.475000+00:00 | `low_rebreak_15m` | 338.0 | 39.6 | -278.6 | 1.8 | 62.0 | btc_down_then_stable | neutral | hammer_reversal |
| 81.1 | 2026-04-16T13:52:14.594000+00:00 | `low_rebreak_15m,weak_first_15m,candle5_bear_followthrough,btc_down_continues` | 37.0 | 29.4 | -60.9 | -53.1 | -1.4 | btc_down_continues | bear_followthrough | bear_followthrough |
| 74.7 | 2026-06-16T04:31:11.525000+00:00 | `late_fill_gt10m` | 1473.0 | 28.2 | -169.0 | 16.6 | 37.5 | btc_down_then_stable | bull_reclaim | bull_reclaim |
| 71.9 | 2026-06-18T15:57:31.634000+00:00 | `low_rebreak_15m,late_fill_gt10m,candle5_failed_hammer` | 5148.0 | 29.2 | -361.7 | 27.1 | 52.8 | btc_down_then_stable | failed_hammer | bull_reclaim |
| 59.1 | 2026-06-25T15:03:23.104000+00:00 | `low_rebreak_15m,candle5_failed_hammer` | 570.0 | 37.5 | -462.3 | 44.1 | 65.4 | btc_down_then_stable | failed_hammer | bull_reclaim |
| 53.7 | 2026-06-12T15:56:42.488000+00:00 | `low_rebreak_15m` | 188.0 | 34.3 | -75.9 | 76.2 | 100.1 | btc_supportive | bull_reclaim | bull_reclaim |
| 44.6 | 2026-04-21T14:57:31.255000+00:00 | `low_rebreak_15m,candle5_failed_hammer` | 56.0 | 31.7 | -106.4 | 36.2 | 59.1 | btc_down_then_stable | failed_hammer | bull_reclaim |
| 41.7 | 2026-06-17T01:17:01.753000+00:00 | `low_rebreak_15m` | 264.0 | 28.4 | -69.8 | 6.6 | 79.1 | btc_supportive | bull_reclaim | neutral |
| 37.0 | 2026-06-21T11:18:26.629000+00:00 | `clean_or_unclassified` | 13.0 | 32.2 | -73.1 | 34.8 | 50.9 | btc_supportive | bull_reclaim | bull_reclaim |

## No-Fill Counterfactual

Closed no-fill mark counterfactual: N=7 sum=568.9 med=57.3 T3R=115.6

| CF mark net | UTC | V-depth | Prior4h | Accel | Dominance |
| ---: | --- | ---: | ---: | ---: | ---: |
| 225.9 | 2026-06-21T23:33:42.690000+00:00 | 29.8 | -168.8 | 8618.0 | 61.9 |
| 152.2 | 2026-06-20T14:08:51.159000+00:00 | 29.2 | -99.7 | 2299.7 | 37.9 |
| 75.2 | 2026-03-13T18:19:10.642000+00:00 | 38.5 | -435.8 | 19273.2 | 82.6 |
| 57.3 | 2026-02-20T15:16:53.428000+00:00 | 39.2 | -96.1 | -3481.8 | 61.9 |
| 32.6 | 2026-03-19T14:37:50.236000+00:00 | 35.7 | -342.8 | 7146.9 | 43.5 |
| 26.6 | 2026-04-05T06:13:18.366000+00:00 | 34.9 | -114.7 | 4906.2 | 49.8 |
| -0.9 | 2026-04-16T07:54:07.129000+00:00 | 35.0 | -85.5 | 9474.5 | 51.7 |
