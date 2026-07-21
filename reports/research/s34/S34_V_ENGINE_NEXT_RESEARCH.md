# S34 V Engine Next Research

Generated: `2026-06-28T20:50:21.218972+00:00`

Research-only continuation of the portfolio roadmap. No live/paper state changed.

## 1. BTC BUY -> Maker SHORT Weak Lead

| Rank | Route | Eligible | Filled | Cal | Hold | Overall |
| ---: | --- | ---: | ---: | --- | --- | --- |
| 1 | `BTC_BUY_FADE_SHORT_T250K_V28_40_P4GT100_H4` | 25 | 5 | N=3 sum=-53.8 med=-3.3 T3R=-53.8 | N=2 sum=147.9 med=73.9 T3R=147.9 | N=5 sum=94.1 med=16.0 T3R=-69.7 |
| 2 | `BTC_BUY_FADE_SHORT_T250K_V28_40_P4GT150_H4` | 15 | 3 | N=2 sum=47.1 med=23.5 T3R=47.1 | N=1 sum=97.5 med=97.5 T3R=97.5 | N=3 sum=144.5 med=50.4 T3R=144.5 |
| 3 | `BTC_BUY_FADE_SHORT_T250K_V28_40_P4GT200_H4` | 8 | 1 | N=0 sum=0.0 med=None T3R=0.0 | N=1 sum=97.5 med=97.5 T3R=97.5 | N=1 sum=97.5 med=97.5 T3R=97.5 |
| 4 | `BTC_BUY_FADE_SHORT_T250K_V28_40_P4GT100_H2` | 25 | 5 | N=3 sum=95.7 med=47.0 T3R=95.7 | N=2 sum=6.0 med=3.0 T3R=6.0 | N=5 sum=101.7 med=18.1 T3R=-20.1 |
| 5 | `BTC_BUY_FADE_SHORT_T250K_V28_40_P4GT50_H4` | 37 | 12 | N=8 sum=43.5 med=9.3 T3R=-64.1 | N=4 sum=277.3 med=73.9 T3R=0.9 | N=12 sum=320.8 med=15.0 T3R=29.0 |
| 6 | `BTC_BUY_FADE_SHORT_T250K_V28_40_P4GT0_H4` | 45 | 15 | N=11 sum=-62.4 med=-3.3 T3R=-169.9 | N=4 sum=277.3 med=73.9 T3R=0.9 | N=15 sum=214.9 med=4.7 T3R=-76.8 |
| 7 | `BTC_BUY_FADE_SHORT_T250K_V28_40_P4GT50_H2` | 37 | 12 | N=8 sum=158.8 med=34.0 T3R=-8.3 | N=4 sum=15.2 med=3.0 T3R=-12.1 | N=12 sum=174.0 med=19.7 T3R=6.9 |
| 8 | `BTC_BUY_FADE_SHORT_T250K_V28_40_P4GT150_H2` | 15 | 3 | N=2 sum=10.1 med=5.0 T3R=10.1 | N=1 sum=-12.1 med=-12.1 T3R=-12.1 | N=3 sum=-2.0 med=-8.0 T3R=-2.0 |
| 9 | `BTC_BUY_FADE_SHORT_T250K_V28_40_P4GT200_H2` | 8 | 1 | N=0 sum=0.0 med=None T3R=0.0 | N=1 sum=-12.1 med=-12.1 T3R=-12.1 | N=1 sum=-12.1 med=-12.1 T3R=-12.1 |
| 10 | `BTC_BUY_FADE_SHORT_T250K_V28_40_P4GT0_H2` | 45 | 14 | N=10 sum=54.3 med=12.4 T3R=-112.8 | N=4 sum=15.2 med=3.0 T3R=-12.1 | N=14 sum=69.5 med=5.0 T3R=-97.6 |

## 2. ETH Threshold Redundancy

| Threshold | Events | Filled | Unique clusters | Summary |
| ---: | ---: | ---: | ---: | --- |
| 150K | 38 | 12 | 38 | N=12 sum=493.4 med=36.8 T3R=165.3 |
| 200K | 47 | 19 | 47 | N=19 sum=887.7 med=37.0 T3R=357.0 |
| 300K | 40 | 17 | 40 | N=17 sum=560.4 med=27.9 T3R=83.2 |

| Pair | Shared clusters | Jaccard | Higher subset of lower |
| --- | ---: | ---: | ---: |
| 150K vs 200K | 23 | 0.371 | 0.489 |
| 200K vs 300K | 23 | 0.359 | 0.575 |
| 150K vs 300K | 14 | 0.219 | 0.35 |

Shared by all three: `13` clusters.

## 3. ETH Pattern / State Layer

Overall core filled sample: N=19 sum=876.1 med=37.0 T3R=348.4

| Rank | Feature | Value | N | Loser% | Summary |
| ---: | --- | --- | ---: | ---: | --- |
| 1 | `anchor_reclaimed_15m` | `True` | 14 | 0.0 | N=14 sum=846.2 med=39.4 T3R=399.9 |
| 2 | `candle15_pattern` | `bull_reclaim` | 11 | 0.0 | N=11 sum=747.9 med=44.6 T3R=301.6 |
| 3 | `btc_context_bucket` | `btc_down_then_stable` | 8 | 0.0 | N=8 sum=747.6 med=65.5 T3R=226.3 |
| 4 | `first_15m_bucket` | `ret15_rebound` | 9 | 0.0 | N=9 sum=642.8 med=44.6 T3R=212.1 |
| 5 | `prior4h_intensity_bucket` | `prior4h_mild_down` | 10 | 10.0 | N=10 sum=309.1 med=29.5 T3R=132.6 |
| 6 | `low_rebreak_15m` | `True` | 13 | 23.1 | N=13 sum=657.2 med=44.6 T3R=129.5 |
| 7 | `candle15_pattern` | `hammer_reversal` | 2 | 50.0 | N=2 sum=111.0 med=55.5 T3R=111.0 |
| 8 | `low_rebreak_15m` | `False` | 6 | 0.0 | N=6 sum=218.9 med=31.3 T3R=74.9 |
| 9 | `prior4h_intensity_bucket` | `prior4h_hard_down` | 3 | 33.3 | N=3 sum=72.8 med=71.9 T3R=72.8 |
| 10 | `btc_context_bucket` | `btc_softening` | 2 | 0.0 | N=2 sum=56.6 med=28.3 T3R=56.6 |
| 11 | `prior4h_intensity_bucket` | `prior4h_extreme_down` | 2 | 50.0 | N=2 sum=42.9 med=21.4 T3R=42.9 |
| 12 | `prior4h_intensity_bucket` | `prior4h_medium_down` | 4 | 0.0 | N=4 sum=451.3 med=59.7 T3R=32.3 |
| 13 | `first_15m_bucket` | `ret15_soft_green` | 4 | 0.0 | N=4 sum=293.7 med=58.2 T3R=30.4 |
| 14 | `btc_context_bucket` | `btc_supportive` | 6 | 16.7 | N=6 sum=153.0 med=33.7 T3R=20.6 |
| 15 | `first_15m_bucket` | `ret15_soft_red` | 2 | 100.0 | N=2 sum=-52.1 med=-26.0 T3R=-52.1 |

## Read

- If BTC rows only work in holdout or fail T3R, keep them as observation lane, not candidate.
- If 150/200/300K share the same clusters, they should be a threshold-response curve, not separate portfolio engines.
- Pattern states are confirmation/permission candidates only; any waited entry must pay price deterioration in a separate test.
