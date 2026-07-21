# S34 BTC BUY -> Maker SHORT Weak-Lead Anatomy

Generated: `2026-06-28T21:31:16.865161+00:00`

Research-only anatomy for `BTCUSDT_BUY_FADE_SHORT_T250K_V28_40_H4`; no live/paper state changed.

## Summary

- Calibration: N=8 sum=43.5 med=9.3 T3R=-64.1 max_loss=-66.4
- Holdout: N=4 sum=277.3 med=73.9 T3R=0.9 max_loss=0.9
- Overall: N=12 sum=320.8 med=15.0 T3R=29.0 max_loss=-66.4

## Group Breakdowns

### session

| Group | Summary |
| --- | --- |
| `late_20_24` | N=6 sum=97.9 med=15.0 T3R=5.7 max_loss=-13.0 |
| `asia_00_08` | N=4 sum=223.6 med=49.2 T3R=-3.3 max_loss=-3.3 |
| `us_13_20` | N=1 sum=65.7 med=65.7 T3R=65.7 max_loss=65.7 |
| `eu_08_13` | N=1 sum=-66.4 med=-66.4 T3R=-66.4 max_loss=-66.4 |

### vdepth_bin

| Group | Summary |
| --- | --- |
| `v28_32` | N=6 sum=168.3 med=2.8 T3R=-15.5 max_loss=-13.0 |
| `v32_36` | N=4 sum=41.1 med=20.9 T3R=-66.4 max_loss=-66.4 |
| `v36_40` | N=2 sum=111.4 med=55.7 T3R=111.4 max_loss=14.0 |

### prior_bin

| Group | Summary |
| --- | --- |
| `p50_100` | N=7 sum=226.7 med=14.0 T3R=6.5 max_loss=-13.0 |
| `p100_200` | N=4 sum=-3.4 med=6.3 T3R=-66.4 max_loss=-66.4 |
| `p200_400` | N=1 sum=97.5 med=97.5 T3R=97.5 max_loss=97.5 |

### fill_delay_bin

| Group | Summary |
| --- | --- |
| `fill_5_15m` | N=9 sum=244.4 med=16.0 T3R=-32.0 max_loss=-66.4 |
| `fill_1_5m` | N=2 sum=62.4 med=31.2 T3R=62.4 max_loss=-3.3 |
| `fill_lt60s` | N=1 sum=14.0 med=14.0 T3R=14.0 max_loss=14.0 |

## Top Winners

| UTC | Split | Net | Vdepth | Prior4h | Fill delay | Session |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 2026-06-26T04:17:39.056000+00:00 | holdout | 128.6 | 31.0 | 67.2 | 8788.0 | asia_00_08 |
| 2026-06-26T06:24:07.608000+00:00 | holdout | 97.5 | 36.1 | 290.7 | 910.0 | asia_00_08 |
| 2026-06-20T15:18:39.359000+00:00 | calibration | 65.7 | 34.7 | 85.8 | 106.0 | us_13_20 |
| 2026-06-24T20:49:41.286000+00:00 | holdout | 50.4 | 31.8 | 164.5 | 4245.0 | late_20_24 |
| 2026-06-19T23:56:37.596000+00:00 | calibration | 25.8 | 33.7 | 79.1 | 2294.0 | late_20_24 |
| 2026-04-15T22:04:41.359000+00:00 | calibration | 16.0 | 35.7 | 103.3 | 2073.0 | late_20_24 |
| 2026-06-13T21:41:25.987000+00:00 | calibration | 14.0 | 36.6 | 76.4 | 55.0 | late_20_24 |
| 2026-04-26T22:15:31.216000+00:00 | calibration | 4.7 | 28.4 | 86.5 | 9578.0 | late_20_24 |

## Top Losers

| UTC | Split | Net | Vdepth | Prior4h | Fill delay | Session |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 2026-04-17T08:48:42.878000+00:00 | calibration | -66.4 | 34.0 | 106.0 | 365.0 | eu_08_13 |
| 2026-04-26T21:55:04.205000+00:00 | calibration | -13.0 | 29.5 | 67.0 | 10737.0 | late_20_24 |
| 2026-04-22T05:16:47.122000+00:00 | calibration | -3.3 | 28.8 | 197.8 | 89.0 | asia_00_08 |
| 2026-06-22T01:18:24.664000+00:00 | holdout | 0.9 | 29.4 | 99.8 | 319.0 | asia_00_08 |
| 2026-04-26T22:15:31.216000+00:00 | calibration | 4.7 | 28.4 | 86.5 | 9578.0 | late_20_24 |
| 2026-06-13T21:41:25.987000+00:00 | calibration | 14.0 | 36.6 | 76.4 | 55.0 | late_20_24 |
| 2026-04-15T22:04:41.359000+00:00 | calibration | 16.0 | 35.7 | 103.3 | 2073.0 | late_20_24 |
| 2026-06-19T23:56:37.596000+00:00 | calibration | 25.8 | 33.7 | 79.1 | 2294.0 | late_20_24 |

## Read

- N is still thin; anatomy is hypothesis-generating, not confirmation.
