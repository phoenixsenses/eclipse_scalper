# S34 v3 Route Node Map

Generated: `2026-06-28T23:46:01.167790+00:00`

Research-only. No live/paper/executor changes.

## Discipline

- Min split N: `40` calibration AND holdout.
- Rule: N<40 in either calibration or holdout is HYPOTHESIS only, never edge.

## Overall

- All: N=541 sum=-2614.6 med=4.4 T3R=-4497.3 max_loss=-507.2 tail<-100=101
- Calibration: N=166 sum=1752.9 med=-6.9 T3R=608.8 max_loss=-285.3 tail<-100=11
- Holdout: N=375 sum=-4367.5 med=10.7 T3R=-6250.2 max_loss=-507.2 tail<-100=90
- Node counts: `{'HYPOTHESIS': 5, 'NO_HOLDOUT': 1, 'WEAK_OR_DEAD': 30}`

## Ranked Routes

| Rank | Status | Route | Cal | Hold | Score | Best holdout leverage point |
| ---: | --- | --- | --- | --- | ---: | --- |
| 1 | `HYPOTHESIS` | `SOLUSDT_SELL_FADE_LONG_T25K_v40_60_H4` | N=2 sum=37.1 med=18.6 T3R=37.1 max_loss=-71.4 tail<-100=0 | N=12 sum=1185.0 med=48.8 T3R=54.3 max_loss=-239.7 tail<-100=1 | -449.4 | `absorption_gate:mixed>absorbed` dT3R=171.4 dSum=1253.0 |
| 2 | `HYPOTHESIS` | `BTCUSDT_SELL_FADE_LONG_T1000K_v20_28_H4` | N=4 sum=77.1 med=21.7 T3R=-4.8 max_loss=-4.8 tail<-100=0 | N=8 sum=737.3 med=102.6 T3R=56.8 max_loss=-73.8 tail<-100=0 | -558.9 | `absorption_gate:mixed>absorbed` dT3R=328.1 dSum=-225.8 |
| 3 | `WEAK_OR_DEAD` | `ETHUSDT_SELL_FADE_LONG_T200K_v28_40_H4` | N=12 sum=375.8 med=-7.5 T3R=-90.6 max_loss=-59.9 tail<-100=0 | N=24 sum=593.1 med=22.2 T3R=-238.7 max_loss=-291.8 tail<-100=3 | -790.4 | `imbalance_gate:bid_support>ask_heavy` dT3R=636.6 dSum=1003.8 |
| 4 | `HYPOTHESIS` | `ETHUSDT_SELL_FADE_LONG_T100K_v60_inf_H4` | N=1 sum=47.6 med=47.6 T3R=47.6 max_loss=47.6 tail<-100=0 | N=2 sum=154.0 med=77.0 T3R=154.0 max_loss=-147.1 tail<-100=1 | -857.5 | `sync_gate:sync>idio` dT3R=154.0 dSum=154.0 |
| 5 | `HYPOTHESIS` | `ETHUSDT_SELL_FADE_LONG_T150K_v60_inf_H4` | N=1 sum=45.1 med=45.1 T3R=45.1 max_loss=45.1 tail<-100=0 | N=2 sum=154.0 med=77.0 T3R=154.0 max_loss=-147.1 tail<-100=1 | -857.5 | `sync_gate:sync>idio` dT3R=154.0 dSum=154.0 |
| 6 | `HYPOTHESIS` | `ETHUSDT_SELL_FADE_LONG_T200K_v60_inf_H4` | N=1 sum=45.1 med=45.1 T3R=45.1 max_loss=45.1 tail<-100=0 | N=2 sum=151.1 med=75.5 T3R=151.1 max_loss=-147.1 tail<-100=1 | -861.1 | `imbalance_gate:bid_support>ask_heavy` dT3R=445.3 dSum=445.3 |
| 7 | `WEAK_OR_DEAD` | `SOLUSDT_SELL_FADE_LONG_T50K_v40_60_H4` | N=1 sum=0.9 med=0.9 T3R=0.9 max_loss=0.9 tail<-100=0 | N=12 sum=976.8 med=14.3 T3R=-329.1 max_loss=-251.7 tail<-100=1 | -884.9 | `bid_depth_gate:deep_bid>shallow_bid` dT3R=160.6 dSum=-1150.0 |
| 8 | `NO_HOLDOUT` | `BTCUSDT_SELL_FADE_LONG_T250K_v60_inf_H4` | N=1 sum=41.0 med=41.0 T3R=41.0 max_loss=41.0 tail<-100=0 | N=0 sum=0.0 med=None T3R=0.0 max_loss=None tail<-100=0 | -1000.0 | `sync_gate:sync>idio` dT3R=0.0 dSum=0.0 |
| 9 | `WEAK_OR_DEAD` | `BTCUSDT_SELL_FADE_LONG_T500K_v60_inf_H4` | N=1 sum=41.0 med=41.0 T3R=41.0 max_loss=41.0 tail<-100=0 | N=1 sum=-66.0 med=-66.0 T3R=-66.0 max_loss=-66.0 tail<-100=0 | -1057.5 | `bid_depth_gate:deep_bid>shallow_bid` dT3R=66.0 dSum=66.0 |
| 10 | `WEAK_OR_DEAD` | `BTCUSDT_SELL_FADE_LONG_T500K_v40_60_H4` | N=7 sum=540.5 med=36.5 T3R=-30.0 max_loss=-92.7 tail<-100=0 | N=6 sum=244.2 med=58.3 T3R=-198.7 max_loss=-213.0 tail<-100=1 | -1087.7 | `bid_depth_gate:deep_bid>shallow_bid` dT3R=540.0 dSum=540.0 |
| 11 | `WEAK_OR_DEAD` | `BTCUSDT_SELL_FADE_LONG_T1000K_v60_inf_H4` | N=1 sum=41.0 med=41.0 T3R=41.0 max_loss=41.0 tail<-100=0 | N=3 sum=-52.0 med=49.0 T3R=-52.0 max_loss=-207.8 tail<-100=1 | -1090.0 | `absorption_gate:mixed>absorbed` dT3R=256.8 dSum=256.8 |
| 12 | `WEAK_OR_DEAD` | `ETHUSDT_SELL_FADE_LONG_T150K_v28_40_H4` | N=14 sum=449.8 med=-1.0 T3R=-52.7 max_loss=-70.5 tail<-100=0 | N=16 sum=434.0 med=35.0 T3R=-344.2 max_loss=-296.2 tail<-100=3 | -1135.7 | `absorption_gate:mixed>absorbed` dT3R=320.5 dSum=158.4 |
| 13 | `WEAK_OR_DEAD` | `SOLUSDT_SELL_FADE_LONG_T50K_v60_inf_H4` | N=0 sum=0.0 med=None T3R=0.0 max_loss=None tail<-100=0 | N=1 sum=-103.3 med=-103.3 T3R=-103.3 max_loss=-103.3 tail<-100=1 | -1204.1 | `bid_depth_gate:deep_bid>shallow_bid` dT3R=103.3 dSum=103.3 |
| 14 | `WEAK_OR_DEAD` | `SOLUSDT_SELL_FADE_LONG_T100K_v40_60_H4` | N=1 sum=0.9 med=0.9 T3R=0.9 max_loss=0.9 tail<-100=0 | N=10 sum=624.9 med=16.6 T3R=-429.8 max_loss=-234.3 tail<-100=2 | -1223.6 | `imbalance_gate:bid_support>ask_heavy` dT3R=780.3 dSum=-14.5 |
| 15 | `WEAK_OR_DEAD` | `ETHUSDT_SELL_FADE_LONG_T100K_v40_60_H4` | N=2 sum=-211.5 med=-105.7 T3R=-211.5 max_loss=-271.1 tail<-100=1 | N=4 sum=-277.4 med=-60.1 T3R=-154.3 max_loss=-154.3 tail<-100=1 | -1223.7 | `sync_gate:sync>idio` dT3R=31.2 dSum=31.2 |
| 16 | `WEAK_OR_DEAD` | `BTCUSDT_SELL_FADE_LONG_T1000K_v40_60_H4` | N=3 sum=-75.3 med=-22.2 T3R=-75.3 max_loss=-92.7 tail<-100=0 | N=4 sum=-6.1 med=43.5 T3R=-232.9 max_loss=-232.9 tail<-100=1 | -1234.4 | `imbalance_gate:bid_support>ask_heavy` dT3R=285.8 dSum=285.8 |
| 17 | `WEAK_OR_DEAD` | `BTCUSDT_SELL_FADE_LONG_T250K_v40_60_H4` | N=4 sum=409.7 med=101.6 T3R=-92.7 max_loss=-92.7 tail<-100=0 | N=4 sum=-338.2 med=-48.4 T3R=-227.9 max_loss=-227.9 tail<-100=1 | -1312.5 | `imbalance_gate:bid_support>ask_heavy` dT3R=311.0 dSum=311.0 |
| 18 | `WEAK_OR_DEAD` | `SOLUSDT_SELL_FADE_LONG_T100K_v20_28_H4` | N=3 sum=99.4 med=-15.5 T3R=99.4 max_loss=-24.9 tail<-100=0 | N=7 sum=-39.1 med=-43.9 T3R=-468.5 max_loss=-260.3 tail<-100=1 | -1403.3 | `absorption_gate:mixed>absorbed` dT3R=723.0 dSum=931.2 |
| 19 | `WEAK_OR_DEAD` | `SOLUSDT_SELL_FADE_LONG_T25K_v60_inf_H4` | N=0 sum=0.0 med=None T3R=0.0 max_loss=None tail<-100=0 | N=4 sum=-300.2 med=-93.5 T3R=-229.7 max_loss=-229.7 tail<-100=2 | -1404.8 | `bid_depth_gate:deep_bid>shallow_bid` dT3R=533.0 dSum=533.0 |
| 20 | `WEAK_OR_DEAD` | `SOLUSDT_SELL_FADE_LONG_T100K_v60_inf_H4` | N=0 sum=0.0 med=None T3R=0.0 max_loss=None tail<-100=0 | N=5 sum=-260.6 med=-10.1 T3R=-399.5 max_loss=-305.8 tail<-100=1 | -1439.7 | `bid_depth_gate:deep_bid>shallow_bid` dT3R=409.0 dSum=409.0 |
| 21 | `WEAK_OR_DEAD` | `BTCUSDT_SELL_FADE_LONG_T250K_v20_28_H4` | N=5 sum=24.0 med=-6.4 T3R=-56.6 max_loss=-46.6 tail<-100=0 | N=16 sum=-311.6 med=17.0 T3R=-682.4 max_loss=-397.0 tail<-100=2 | -1560.3 | `imbalance_gate:bid_support>ask_heavy` dT3R=1057.9 dSum=1349.2 |
| 22 | `WEAK_OR_DEAD` | `SOLUSDT_SELL_FADE_LONG_T25K_v20_28_H4` | N=3 sum=35.7 med=-13.1 T3R=35.7 max_loss=-21.3 tail<-100=0 | N=15 sum=197.3 med=-14.9 T3R=-907.4 max_loss=-374.0 tail<-100=3 | -1783.1 | `absorption_gate:mixed>absorbed` dT3R=552.5 dSum=483.6 |
| 23 | `WEAK_OR_DEAD` | `ETHUSDT_SELL_FADE_LONG_T150K_v40_60_H4` | N=2 sum=-211.5 med=-105.7 T3R=-211.5 max_loss=-271.1 tail<-100=1 | N=8 sum=-427.7 med=-56.6 T3R=-734.2 max_loss=-338.0 tail<-100=2 | -1841.1 | `bid_depth_gate:deep_bid>shallow_bid` dT3R=937.0 dSum=924.9 |
| 24 | `WEAK_OR_DEAD` | `SOLUSDT_SELL_FADE_LONG_T50K_v28_40_H4` | N=1 sum=-25.5 med=-25.5 T3R=-25.5 max_loss=-25.5 tail<-100=0 | N=14 sum=-355.0 med=13.6 T3R=-855.5 max_loss=-484.2 tail<-100=3 | -1894.2 | `absorption_gate:mixed>absorbed` dT3R=740.2 dSum=239.7 |
| 25 | `WEAK_OR_DEAD` | `BTCUSDT_SELL_FADE_LONG_T500K_v28_40_H4` | N=4 sum=32.9 med=-5.3 T3R=-54.6 max_loss=-54.6 tail<-100=0 | N=13 sum=-356.5 med=25.1 T3R=-730.4 max_loss=-417.1 tail<-100=4 | -1894.5 | `bid_depth_gate:deep_bid>shallow_bid` dT3R=37.7 dSum=-194.7 |
| 26 | `WEAK_OR_DEAD` | `BTCUSDT_SELL_FADE_LONG_T500K_v20_28_H4` | N=7 sum=84.1 med=-9.6 T3R=-115.5 max_loss=-54.1 tail<-100=0 | N=10 sum=-536.8 med=7.8 T3R=-824.6 max_loss=-397.1 tail<-100=2 | -1908.8 | `bid_depth_gate:deep_bid>shallow_bid` dT3R=895.7 dSum=699.6 |
| 27 | `WEAK_OR_DEAD` | `ETHUSDT_SELL_FADE_LONG_T100K_v20_28_H4` | N=19 sum=-399.5 med=-43.4 T3R=-958.3 max_loss=-256.9 tail<-100=3 | N=22 sum=326.6 med=14.6 T3R=-1060.7 max_loss=-439.8 tail<-100=5 | -1929.1 | `absorption_gate:mixed>absorbed` dT3R=532.1 dSum=-210.1 |
| 28 | `WEAK_OR_DEAD` | `ETHUSDT_SELL_FADE_LONG_T200K_v40_60_H4` | N=3 sum=-56.7 med=59.7 T3R=-56.7 max_loss=-271.1 tail<-100=1 | N=9 sum=-829.8 med=-67.4 T3R=-853.1 max_loss=-338.0 tail<-100=3 | -2135.6 | `bid_depth_gate:deep_bid>shallow_bid` dT3R=626.0 dSum=614.4 |
| 29 | `WEAK_OR_DEAD` | `BTCUSDT_SELL_FADE_LONG_T250K_v28_40_H4` | N=8 sum=246.0 med=27.5 T3R=-76.2 max_loss=-59.6 tail<-100=0 | N=6 sum=-639.8 med=-65.1 T3R=-860.7 max_loss=-413.4 tail<-100=3 | -2170.7 | `absorption_gate:mixed>absorbed` dT3R=534.3 dSum=891.9 |
| 30 | `WEAK_OR_DEAD` | `BTCUSDT_SELL_FADE_LONG_T1000K_v28_40_H4` | N=4 sum=75.0 med=-11.8 T3R=-44.9 max_loss=-44.9 tail<-100=0 | N=14 sum=-591.0 med=34.4 T3R=-974.1 max_loss=-384.0 tail<-100=5 | -2271.8 | `absorption_gate:mixed>absorbed` dT3R=441.0 dSum=187.4 |
| 31 | `WEAK_OR_DEAD` | `ETHUSDT_SELL_FADE_LONG_T100K_v28_40_H4` | N=16 sum=-48.5 med=-25.7 T3R=-500.2 max_loss=-285.3 tail<-100=2 | N=18 sum=-394.5 med=21.6 T3R=-1229.7 max_loss=-507.2 tail<-100=5 | -2378.3 | `absorption_gate:mixed>absorbed` dT3R=988.3 dSum=1170.0 |
| 32 | `WEAK_OR_DEAD` | `SOLUSDT_SELL_FADE_LONG_T100K_v28_40_H4` | N=3 sum=51.5 med=22.0 T3R=51.5 max_loss=-38.8 tail<-100=0 | N=14 sum=-747.2 med=-10.2 T3R=-1267.1 max_loss=-484.2 tail<-100=4 | -2503.9 | `absorption_gate:mixed>absorbed` dT3R=721.7 dSum=781.1 |
| 33 | `WEAK_OR_DEAD` | `SOLUSDT_SELL_FADE_LONG_T25K_v28_40_H4` | N=3 sum=-150.6 med=-52.4 T3R=-150.6 max_loss=-72.6 tail<-100=0 | N=17 sum=-883.0 med=-24.1 T3R=-1379.4 max_loss=-484.2 tail<-100=4 | -2575.2 | `absorption_gate:mixed>absorbed` dT3R=388.3 dSum=398.7 |
| 34 | `WEAK_OR_DEAD` | `SOLUSDT_SELL_FADE_LONG_T50K_v20_28_H4` | N=1 sum=-39.5 med=-39.5 T3R=-39.5 max_loss=-39.5 tail<-100=0 | N=22 sum=-672.6 med=-3.4 T3R=-1460.1 max_loss=-469.0 tail<-100=5 | -2578.2 | `absorption_gate:mixed>absorbed` dT3R=570.9 dSum=1311.4 |
| 35 | `WEAK_OR_DEAD` | `ETHUSDT_SELL_FADE_LONG_T200K_v20_28_H4` | N=13 sum=-24.7 med=-28.4 T3R=-591.0 max_loss=-224.0 tail<-100=2 | N=21 sum=-908.0 med=26.0 T3R=-2133.4 max_loss=-494.0 tail<-100=8 | -3635.4 | `absorption_gate:mixed>absorbed` dT3R=1867.5 dSum=1148.3 |
| 36 | `WEAK_OR_DEAD` | `ETHUSDT_SELL_FADE_LONG_T150K_v20_28_H4` | N=15 sum=194.6 med=-34.5 T3R=-475.0 max_loss=-221.3 tail<-100=1 | N=29 sum=-1049.3 med=17.1 T3R=-2351.4 max_loss=-494.0 tail<-100=9 | -3788.7 | `absorption_gate:mixed>absorbed` dT3R=1904.3 dSum=1626.9 |

## Read

- No route clears the N>=40 per-split + positive holdout + tail gate. There is no validated network node yet.
- No watch node clears the full N gate either; small-N winners remain hypotheses.
- Top ranked node by holdout-aware score: `SOLUSDT_SELL_FADE_LONG_T25K_v40_60_H4` -> status `HYPOTHESIS`.
- Keep route families separate; do not create a single pooled live rule from this map.
