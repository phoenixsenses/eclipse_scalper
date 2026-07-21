# S34 All-Bucket Feature Screen

Generated: `2026-06-26T22:14:29.257539+00:00`

Research-only screen over `data/s34_feature_factory.db`. No runner/config changes.

Important limitation: this uses the route labels already present in the feature factory. Some live runner rules have exact label parity; others are screened with the nearest available route label.

## Feature DB Coverage

| Symbol | Side | Events | First | Last |
| --- | --- | ---: | --- | --- |
| BTCUSDT | BUY | 127 | 2026-02-17T16:05:39.128000+00:00 | 2026-06-26T13:35:31.100000+00:00 |
| BTCUSDT | SELL | 113 | 2026-02-18T11:06:33.546000+00:00 | 2026-06-26T12:40:01.099000+00:00 |
| ETHUSDT | BUY | 450 | 2026-02-15T22:47:11.071000+00:00 | 2026-06-16T08:15:58.161000+00:00 |
| ETHUSDT | SELL | 222 | 2026-02-16T11:10:12.218000+00:00 | 2026-06-26T13:16:54.455000+00:00 |
| SOLUSDT | BUY | 104 | 2026-04-20T06:40:08.582000+00:00 | 2026-06-26T13:50:45.406000+00:00 |
| SOLUSDT | SELL | 105 | 2026-04-18T08:41:11.463000+00:00 | 2026-06-26T12:40:01.200000+00:00 |

## Best Available Route Per Active Bucket

| Rule | Events | Filters | Best route | Exact? | N | Median | Mean | WR | Cum | Top3 removed | Pos days | Avg hold | Giveback | Verdict |
| --- | ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| ETH_BUY_LIQ_LONG_200K_TP60_SL40_BE30 | 450 | - | LONG_DELAY0_TP60 | yes | 450 | -5.1 | +13.8 | 50% | +6228.3 | +6025.4 | 52/77 | +1010 | 30% | negative_median |
| ETH_BUY_LIQ_LONG_200K_BTC_PRE15_TP120_SL40_BE30_DELAY60 | 356 | btc_pre15>=0 | LONG_DELAY0_TP60 | nearest | 356 | +2.3 | +14.4 | 50% | +5126.7 | +4926.7 | 49/74 | +1053 | 30% | candidate |
| ETH_BUY_LIQ_LONG_500K_DAYTREND0_TP60_SL40_BE30 | 97 | day_trend>=0 | LONG_DELAY0_TP60 | yes | 97 | +52.2 | +24.0 | 61% | +2329.8 | +2140.0 | 28/38 | +790 | 28% | candidate |
| ETH_BUY_LIQ_LONG_500K_NEGTREND_STRETCHED_TP60_SL40_BE30 | 19 | day_trend<=0, shape=stretched_120s | LONG_DELAY0_TP60 | yes | 19 | +52.2 | +40.1 | 79% | +761.5 | +590.8 | 13/14 | +400 | 21% | thin |
| SOL_BUY_LIQ_LONG_100K_TP60_SL40_BE30 | 104 | - | LONG_DELAY0_TP60 | yes | 104 | +52.1 | +19.5 | 57% | +2028.6 | +1845.9 | 20/25 | +760 | 29% | candidate |
| SOL_BUY_LIQ_LONG_200K_TP60_SL40_BE30 | 64 | - | LONG_DELAY0_TP60 | yes | 64 | +52.3 | +22.5 | 59% | +1441.2 | +1258.5 | 15/19 | +713 | 27% | candidate |
| BTC_BUY_LIQ_LONG_1M_DISTRIBUTED_TP60_SL30_BE30 | 66 | max_share<=50 | LONG_DELAY0_TP60 | nearest | 66 | +52.1 | +25.6 | 64% | +1687.6 | +1508.7 | 25/36 | +1341 | 32% | candidate |
| ETH_SELL_LIQ_SHORT_500K_TP60_SL40_BE40 | 222 | - | SHORT_DELAY0_TP60 | yes | 222 | +52.2 | +19.1 | 60% | +4237.1 | +4046.4 | 48/67 | +1137 | 25% | candidate |
| ETH_SELL_LIQ_SHORT_1M_TP80_SL40_BE40 | 106 | - | SHORT_DELAY0_TP60 | nearest | 106 | +52.6 | +29.8 | 72% | +3155.9 | +2965.2 | 35/43 | +1065 | 18% | candidate |
| SOL_SELL_LIQ_SHORT_200K_TP60_SL30_BE30 | 51 | - | SHORT_DELAY0_TP60 | nearest | 51 | +52.2 | +25.4 | 63% | +1297.9 | +1059.8 | 16/20 | +801 | 25% | candidate |
| SOL_SELL_LIQ_SHORT_100K_TP60_SL30_BE40 | 105 | - | SHORT_DELAY0_TP40 | nearest | 105 | +32.4 | +11.5 | 71% | +1209.8 | +1086.9 | 21/27 | +871 | 14% | candidate |

## All Route Labels By Bucket

### ETH_BUY_LIQ_LONG_200K_TP60_SL40_BE30

| Route | Exact? | N | Median | WR | Exits | Giveback | Verdict |
| --- | --- | ---: | ---: | ---: | --- | ---: | --- |
| LONG_DELAY0_TP60 | yes | 450 | -5.1 | 50% | BE=133, SL=76, TIME=37, TP=204 | 30% | negative_median |
| LONG_DELAY60_TP120 | nearest | 450 | -8.8 | 29% | BE=169, SL=128, TIME=81, TP=72 | 13% | negative_median |

### ETH_BUY_LIQ_LONG_200K_BTC_PRE15_TP120_SL40_BE30_DELAY60

| Route | Exact? | N | Median | WR | Exits | Giveback | Verdict |
| --- | --- | ---: | ---: | ---: | --- | ---: | --- |
| LONG_DELAY0_TP60 | nearest | 356 | +2.3 | 50% | BE=106, SL=57, TIME=32, TP=161 | 30% | candidate |
| LONG_DELAY60_TP120 | yes | 356 | -8.8 | 31% | BE=126, SL=100, TIME=72, TP=58 | 13% | negative_median |

### ETH_BUY_LIQ_LONG_500K_DAYTREND0_TP60_SL40_BE30

| Route | Exact? | N | Median | WR | Exits | Giveback | Verdict |
| --- | --- | ---: | ---: | ---: | --- | ---: | --- |
| LONG_DELAY0_TP60 | yes | 97 | +52.2 | 61% | BE=27, SL=9, TIME=8, TP=53 | 28% | candidate |
| LONG_DELAY60_TP120 | nearest | 97 | -8.3 | 35% | BE=36, SL=21, TIME=20, TP=20 | 18% | negative_median |

### ETH_BUY_LIQ_LONG_500K_NEGTREND_STRETCHED_TP60_SL40_BE30

| Route | Exact? | N | Median | WR | Exits | Giveback | Verdict |
| --- | --- | ---: | ---: | ---: | --- | ---: | --- |
| LONG_DELAY0_TP60 | yes | 19 | +52.2 | 79% | BE=4, TP=15 | 21% | thin |
| LONG_DELAY60_TP120 | nearest | 19 | -8.2 | 37% | BE=9, SL=3, TIME=4, TP=3 | 21% | thin |

### SOL_BUY_LIQ_LONG_100K_TP60_SL40_BE30

| Route | Exact? | N | Median | WR | Exits | Giveback | Verdict |
| --- | --- | ---: | ---: | ---: | --- | ---: | --- |
| LONG_DELAY0_TP60 | yes | 104 | +52.1 | 57% | BE=30, SL=14, TIME=6, TP=54 | 29% | candidate |

### SOL_BUY_LIQ_LONG_200K_TP60_SL40_BE30

| Route | Exact? | N | Median | WR | Exits | Giveback | Verdict |
| --- | --- | ---: | ---: | ---: | --- | ---: | --- |
| LONG_DELAY0_TP60 | yes | 64 | +52.3 | 59% | BE=17, SL=8, TIME=3, TP=36 | 27% | candidate |

### BTC_BUY_LIQ_LONG_1M_DISTRIBUTED_TP60_SL30_BE30

| Route | Exact? | N | Median | WR | Exits | Giveback | Verdict |
| --- | --- | ---: | ---: | ---: | --- | ---: | --- |
| LONG_DELAY0_TP60 | nearest | 66 | +52.1 | 64% | BE=21, SL=3, TIME=8, TP=34 | 32% | candidate |

### ETH_SELL_LIQ_SHORT_500K_TP60_SL40_BE40

| Route | Exact? | N | Median | WR | Exits | Giveback | Verdict |
| --- | --- | ---: | ---: | ---: | --- | ---: | --- |
| SHORT_DELAY0_TP60 | yes | 222 | +52.2 | 60% | BE=39, SL=40, TIME=20, TP=123 | 25% | candidate |
| SHORT_DELAY0_TP80 | nearest | 222 | -8.1 | 48% | BE=66, SL=40, TIME=31, TP=85 | 30% | negative_median |

### ETH_SELL_LIQ_SHORT_1M_TP80_SL40_BE40

| Route | Exact? | N | Median | WR | Exits | Giveback | Verdict |
| --- | --- | ---: | ---: | ---: | --- | ---: | --- |
| SHORT_DELAY0_TP60 | nearest | 106 | +52.6 | 72% | BE=15, SL=11, TIME=9, TP=71 | 18% | candidate |
| SHORT_DELAY0_TP80 | yes | 106 | +44.4 | 58% | BE=29, SL=11, TIME=15, TP=51 | 27% | candidate |

### SOL_SELL_LIQ_SHORT_200K_TP60_SL30_BE30

| Route | Exact? | N | Median | WR | Exits | Giveback | Verdict |
| --- | --- | ---: | ---: | ---: | --- | ---: | --- |
| SHORT_DELAY0_TP40 | nearest | 51 | +32.9 | 84% | SL=7, TIME=1, TP=43 | 4% | candidate |
| SHORT_DELAY0_TP60 | nearest | 51 | +52.2 | 63% | BE=11, SL=7, TIME=2, TP=31 | 25% | candidate |

### SOL_SELL_LIQ_SHORT_100K_TP60_SL30_BE40

| Route | Exact? | N | Median | WR | Exits | Giveback | Verdict |
| --- | --- | ---: | ---: | ---: | --- | ---: | --- |
| SHORT_DELAY0_TP40 | nearest | 105 | +32.4 | 71% | SL=25, TIME=5, TP=75 | 14% | candidate |
| SHORT_DELAY0_TP60 | nearest | 105 | +26.0 | 53% | BE=19, SL=25, TIME=10, TP=51 | 25% | candidate |

## Next-Step Interpretation Rules

- `candidate` means the bucket deserves forward collection or a deeper exact-route sweep.
- `nearest` means do not promote directly; first generate exact labels or run a runner-helper parity check.
- High `giveback` flags the fast-exit question: the route often sees MFE but closes negative.