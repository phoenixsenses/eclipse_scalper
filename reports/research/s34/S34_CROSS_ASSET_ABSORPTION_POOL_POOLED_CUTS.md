# S34 Cross-Asset Absorption Pool

Generated: `2026-06-28T23:26:41.853667+00:00`

Research-only. Pools BTC/ETH/SOL SELL deep-V real-fill events with T=0 book absorption; no live/paper state changed.

Events: `541`; holdout months `['2026-06']`; per-symbol cuts `False`

## Overall

- All: N=541 sum=-2614.6 mean=-4.8 med=4.4 win=51.6 T3R=-4497.3 max_loss=-507.2 tail<-100=101
- Calibration: N=166 sum=1752.9 mean=10.6 med=-6.9 win=47.6 T3R=608.8 max_loss=-285.3 tail<-100=11
- Holdout: N=375 sum=-4367.5 mean=-11.6 med=10.7 win=53.3 T3R=-6250.2 max_loss=-507.2 tail<-100=90

## symbol

| Group | All | Cal | Hold |
| --- | --- | --- | --- |
| `BTCUSDT` | N=134 sum=-379.4 mean=-2.8 med=17.9 win=58.2 T3R=-1272.5 max_loss=-417.1 tail<-100=20 | N=49 sum=1537.1 mean=31.4 med=32.3 win=59.2 T3R=786.6 max_loss=-92.7 tail<-100=0 | N=85 sum=-1916.5 mean=-22.5 med=13.8 win=57.6 T3R=-2665.6 max_loss=-417.1 tail<-100=20 |
| `SOLUSDT` | N=151 sum=-367.0 mean=-2.4 med=-13.1 win=48.3 T3R=-2228.6 max_loss=-484.2 tail<-100=28 | N=18 sum=10.0 mean=0.6 med=-18.4 win=38.9 T3R=-308.5 max_loss=-72.6 tail<-100=0 | N=133 sum=-377.0 mean=-2.8 med=-0.1 win=49.6 T3R=-2238.6 max_loss=-484.2 tail<-100=28 |
| `ETHUSDT` | N=256 sum=-1868.2 mean=-7.3 med=-0.3 win=50.0 T3R=-3706.2 max_loss=-507.2 tail<-100=53 | N=99 sum=205.8 mean=2.1 med=-11.4 win=43.4 T3R=-938.3 max_loss=-285.3 tail<-100=11 | N=157 sum=-2074.0 mean=-13.2 med=15.7 win=54.1 T3R=-3912.0 max_loss=-507.2 tail<-100=42 |

## vdepth_band

| Group | All | Cal | Hold |
| --- | --- | --- | --- |
| `v40_60` | N=94 sum=1586.0 mean=16.9 med=-2.9 win=47.9 T3R=-275.6 max_loss=-338.0 tail<-100=16 | N=25 sum=434.2 mean=17.4 med=36.5 win=64.0 T3R=-316.3 max_loss=-271.1 tail<-100=3 | N=69 sum=1151.7 mean=16.7 med=-16.8 win=42.0 T3R=-709.8 max_loss=-338.0 tail<-100=13 |
| `v60_inf` | N=26 sum=-62.1 mean=-2.4 med=41.0 win=53.8 T3R=-962.3 max_loss=-305.8 tail<-100=8 | N=6 sum=261.0 mean=43.5 med=43.1 win=100.0 T3R=123.1 max_loss=41.0 tail<-100=0 | N=20 sum=-323.1 mean=-16.2 med=-74.8 win=40.0 T3R=-1223.3 max_loss=-305.8 tail<-100=8 |
| `v28_40` | N=201 sum=-1933.6 mean=-9.6 med=12.0 win=54.7 T3R=-2971.7 max_loss=-507.2 tail<-100=36 | N=65 sum=1006.4 mean=15.5 med=-10.8 win=46.2 T3R=471.0 max_loss=-285.3 tail<-100=2 | N=136 sum=-2940.0 mean=-21.6 med=20.6 win=58.8 T3R=-3978.1 max_loss=-507.2 tail<-100=34 |
| `v20_28` | N=220 sum=-2204.9 mean=-10.0 med=-1.8 win=50.0 T3R=-4042.9 max_loss=-494.0 tail<-100=41 | N=70 sum=51.3 mean=0.7 med=-12.2 win=38.6 T3R=-1092.8 max_loss=-256.9 tail<-100=6 | N=150 sum=-2256.2 mean=-15.0 med=12.0 win=55.3 T3R=-4094.3 max_loss=-494.0 tail<-100=35 |

## imbalance_gate

| Group | All | Cal | Hold |
| --- | --- | --- | --- |
| `ask_heavy` | N=270 sum=3427.0 mean=12.7 med=19.0 win=57.4 T3R=1544.4 max_loss=-507.2 tail<-100=47 | N=72 sum=2116.2 mean=29.4 med=7.7 win=55.6 T3R=972.0 max_loss=-285.3 tail<-100=2 | N=198 sum=1310.8 mean=6.6 med=20.0 win=58.1 T3R=-571.8 max_loss=-507.2 tail<-100=45 |
| `bid_support` | N=271 sum=-6041.6 mean=-22.3 med=-11.4 win=45.8 T3R=-7572.6 max_loss=-494.0 tail<-100=54 | N=94 sum=-363.3 mean=-3.9 med=-11.4 win=41.5 T3R=-891.1 max_loss=-271.1 tail<-100=9 | N=177 sum=-5678.4 mean=-32.1 med=-10.1 win=48.0 T3R=-7209.4 max_loss=-494.0 tail<-100=45 |

## bid_depth_gate

| Group | All | Cal | Hold |
| --- | --- | --- | --- |
| `shallow_bid` | N=270 sum=4925.6 mean=18.2 med=18.1 win=56.7 T3R=3042.9 max_loss=-507.2 tail<-100=43 | N=68 sum=1448.6 mean=21.3 med=1.5 win=52.9 T3R=304.5 max_loss=-285.3 tail<-100=2 | N=202 sum=3477.0 mean=17.2 med=20.6 win=57.9 T3R=1594.3 max_loss=-507.2 tail<-100=41 |
| `deep_bid` | N=271 sum=-7540.2 mean=-27.8 med=-10.1 win=46.5 T3R=-8665.3 max_loss=-494.0 tail<-100=58 | N=98 sum=304.3 mean=3.1 med=-10.9 win=43.9 T3R=-469.6 max_loss=-271.1 tail<-100=9 | N=173 sum=-7844.5 mean=-45.3 med=-7.2 win=48.0 T3R=-8969.6 max_loss=-494.0 tail<-100=49 |

## absorption_gate

| Group | All | Cal | Hold |
| --- | --- | --- | --- |
| `mixed` | N=220 sum=4568.6 mean=20.8 med=18.1 win=58.2 T3R=3001.7 max_loss=-460.6 tail<-100=25 | N=65 sum=2091.7 mean=32.2 med=13.2 win=56.9 T3R=947.5 max_loss=-113.7 tail<-100=1 | N=155 sum=2477.0 mean=16.0 med=20.4 win=58.7 T3R=910.1 max_loss=-460.6 tail<-100=24 |
| `vacuum_like` | N=105 sum=1112.4 mean=10.6 med=10.7 win=52.4 T3R=-770.2 max_loss=-507.2 tail<-100=25 | N=21 sum=55.1 mean=2.6 med=-36.8 win=42.9 T3R=-429.6 max_loss=-285.3 tail<-100=1 | N=84 sum=1057.4 mean=12.6 med=14.6 win=54.8 T3R=-825.3 max_loss=-507.2 tail<-100=24 |
| `absorbed` | N=216 sum=-8295.7 mean=-38.4 med=-12.5 win=44.4 T3R=-9335.7 max_loss=-494.0 tail<-100=51 | N=80 sum=-393.8 mean=-4.9 med=-11.4 win=41.2 T3R=-921.7 max_loss=-271.1 tail<-100=9 | N=136 sum=-7901.9 mean=-58.1 med=-15.2 win=46.3 T3R=-8941.9 max_loss=-494.0 tail<-100=42 |

## symbol_x_absorption

| Group | All | Cal | Hold |
| --- | --- | --- | --- |
| `SOLUSDT:mixed` | N=84 sum=3762.6 mean=44.8 med=21.5 win=59.5 T3R=2213.8 max_loss=-460.6 tail<-100=7 | N=15 sum=-23.4 mean=-1.6 med=-24.9 win=40.0 T3R=-340.0 max_loss=-72.6 tail<-100=0 | N=69 sum=3786.0 mean=54.9 med=42.1 win=63.8 T3R=2237.2 max_loss=-460.6 tail<-100=7 |
| `ETHUSDT:vacuum_like` | N=68 sum=2375.1 mean=34.9 med=35.7 win=58.8 T3R=537.0 max_loss=-507.2 tail<-100=11 | N=17 sum=-44.8 mean=-2.6 med=-45.1 win=41.2 T3R=-529.4 max_loss=-285.3 tail<-100=1 | N=51 sum=2419.8 mean=47.4 med=37.0 win=64.7 T3R=581.8 max_loss=-507.2 tail<-100=10 |
| `ETHUSDT:mixed` | N=84 sum=887.7 mean=10.6 med=16.9 win=57.1 T3R=-291.8 max_loss=-433.4 tail<-100=9 | N=32 sum=1055.6 mean=33.0 med=7.7 win=56.2 T3R=-88.6 max_loss=-113.7 tail<-100=1 | N=52 sum=-167.9 mean=-3.2 med=20.0 win=57.7 T3R=-1035.5 max_loss=-433.4 tail<-100=8 |
| `BTCUSDT:absorbed` | N=67 sum=332.1 mean=5.0 med=22.1 win=59.7 T3R=-416.9 max_loss=-417.1 tail<-100=7 | N=27 sum=377.8 mean=14.0 med=4.4 win=51.9 T3R=-35.3 max_loss=-92.7 tail<-100=0 | N=40 sum=-45.7 mean=-1.1 med=28.9 win=65.0 T3R=-794.7 max_loss=-417.1 tail<-100=7 |
| `BTCUSDT:mixed` | N=52 sum=-81.6 mean=-1.6 med=10.8 win=57.7 T3R=-856.4 max_loss=-365.2 tail<-100=9 | N=18 sum=1059.5 mean=58.9 med=36.7 win=72.2 T3R=320.8 max_loss=-44.5 tail<-100=0 | N=34 sum=-1141.1 mean=-33.6 med=-8.8 win=50.0 T3R=-1624.9 max_loss=-365.2 tail<-100=9 |
| `BTCUSDT:vacuum_like` | N=15 sum=-629.9 mean=-42.0 med=10.2 win=53.3 T3R=-1042.5 max_loss=-397.1 tail<-100=4 | N=4 sum=99.8 mean=25.0 med=36.0 win=50.0 T3R=-59.6 max_loss=-59.6 tail<-100=0 | N=11 sum=-729.7 mean=-66.3 med=10.2 win=54.5 T3R=-1142.3 max_loss=-397.1 tail<-100=4 |
| `SOLUSDT:vacuum_like` | N=22 sum=-632.7 mean=-28.8 med=-63.9 win=31.8 T3R=-2117.0 max_loss=-472.5 tail<-100=10 | N=0 sum=0.0 mean=None med=None win=None T3R=0.0 max_loss=None tail<-100=0 | N=22 sum=-632.7 mean=-28.8 med=-63.9 win=31.8 T3R=-2117.0 max_loss=-472.5 tail<-100=10 |
| `SOLUSDT:absorbed` | N=45 sum=-3496.9 mean=-77.7 med=-43.9 win=35.6 T3R=-4041.4 max_loss=-484.2 tail<-100=11 | N=3 sum=33.3 mean=11.1 med=-15.5 win=33.3 T3R=33.3 max_loss=-21.3 tail<-100=0 | N=42 sum=-3530.2 mean=-84.1 med=-57.7 win=35.7 T3R=-4074.8 max_loss=-484.2 tail<-100=11 |
| `ETHUSDT:absorbed` | N=104 sum=-5130.9 mean=-49.3 med=-40.1 win=38.5 T3R=-6170.9 max_loss=-494.0 tail<-100=33 | N=50 sum=-805.0 mean=-16.1 med=-34.3 win=36.0 T3R=-1332.8 max_loss=-271.1 tail<-100=9 | N=54 sum=-4325.9 mean=-80.1 med=-70.3 win=40.7 T3R=-5365.9 max_loss=-494.0 tail<-100=24 |

## Route Candidates

| Route | All | Cal | Hold |
| --- | --- | --- | --- |
| `ETHUSDT_SELL_FADE_LONG_T200K_v28_40_H4` | N=36 sum=968.9 mean=26.9 med=20.6 win=58.3 T3R=137.2 max_loss=-291.8 tail<-100=3 | N=12 sum=375.8 mean=31.3 med=-7.5 win=41.7 T3R=-90.6 max_loss=-59.9 tail<-100=0 | N=24 sum=593.1 mean=24.7 med=22.2 win=66.7 T3R=-238.7 max_loss=-291.8 tail<-100=3 |
| `BTCUSDT_SELL_FADE_LONG_T1000K_v20_28_H4` | N=12 sum=814.4 mean=67.9 med=35.5 win=66.7 T3R=133.9 max_loss=-73.8 tail<-100=0 | N=4 sum=77.1 mean=19.3 med=21.7 win=75.0 T3R=-4.8 max_loss=-4.8 tail<-100=0 | N=8 sum=737.3 mean=92.2 med=102.6 win=62.5 T3R=56.8 max_loss=-73.8 tail<-100=0 |
| `ETHUSDT_SELL_FADE_LONG_T150K_v28_40_H4` | N=30 sum=883.8 mean=29.5 med=22.0 win=60.0 T3R=105.6 max_loss=-296.2 tail<-100=3 | N=14 sum=449.8 mean=32.1 med=-1.0 win=50.0 T3R=-52.7 max_loss=-70.5 tail<-100=0 | N=16 sum=434.0 mean=27.1 med=35.0 win=68.8 T3R=-344.2 max_loss=-296.2 tail<-100=3 |
| `SOLUSDT_SELL_FADE_LONG_T25K_v40_60_H4` | N=14 sum=1222.1 mean=87.3 med=48.8 win=71.4 T3R=91.4 max_loss=-239.7 tail<-100=1 | N=2 sum=37.1 mean=18.6 med=18.6 win=50.0 T3R=37.1 max_loss=-71.4 tail<-100=0 | N=12 sum=1185.0 mean=98.7 med=48.8 win=75.0 T3R=54.3 max_loss=-239.7 tail<-100=1 |
| `BTCUSDT_SELL_FADE_LONG_T500K_v40_60_H4` | N=13 sum=784.7 mean=60.4 med=36.5 win=69.2 T3R=85.1 max_loss=-213.0 tail<-100=1 | N=7 sum=540.5 mean=77.2 med=36.5 win=71.4 T3R=-30.0 max_loss=-92.7 tail<-100=0 | N=6 sum=244.2 mean=40.7 med=58.3 win=66.7 T3R=-198.7 max_loss=-213.0 tail<-100=1 |
| `SOLUSDT_SELL_FADE_LONG_T50K_v40_60_H4` | N=13 sum=977.8 mean=75.2 med=0.9 win=53.8 T3R=-328.1 max_loss=-251.7 tail<-100=1 | N=1 sum=0.9 mean=0.9 med=0.9 win=100.0 T3R=0.9 max_loss=0.9 tail<-100=0 | N=12 sum=976.8 mean=81.4 med=14.3 win=50.0 T3R=-329.1 max_loss=-251.7 tail<-100=1 |
| `BTCUSDT_SELL_FADE_LONG_T1000K_v40_60_H4` | N=7 sum=-81.3 mean=-11.6 med=-22.2 win=42.9 T3R=-388.5 max_loss=-232.9 tail<-100=1 | N=3 sum=-75.3 mean=-25.1 med=-22.2 win=33.3 T3R=-75.3 max_loss=-92.7 tail<-100=0 | N=4 sum=-6.1 mean=-1.5 med=43.5 win=50.0 T3R=-232.9 max_loss=-232.9 tail<-100=1 |
| `SOLUSDT_SELL_FADE_LONG_T100K_v60_inf_H4` | N=5 sum=-260.6 mean=-52.1 med=-10.1 win=40.0 T3R=-399.5 max_loss=-305.8 tail<-100=1 | N=0 sum=0.0 mean=None med=None win=None T3R=0.0 max_loss=None tail<-100=0 | N=5 sum=-260.6 mean=-52.1 med=-10.1 win=40.0 T3R=-399.5 max_loss=-305.8 tail<-100=1 |
| `SOLUSDT_SELL_FADE_LONG_T100K_v40_60_H4` | N=11 sum=625.9 mean=56.9 med=0.9 win=54.5 T3R=-428.9 max_loss=-234.3 tail<-100=2 | N=1 sum=0.9 mean=0.9 med=0.9 win=100.0 T3R=0.9 max_loss=0.9 tail<-100=0 | N=10 sum=624.9 mean=62.5 med=16.6 win=50.0 T3R=-429.8 max_loss=-234.3 tail<-100=2 |
| `BTCUSDT_SELL_FADE_LONG_T250K_v40_60_H4` | N=8 sum=71.5 mean=8.9 med=-21.6 win=37.5 T3R=-430.9 max_loss=-227.9 tail<-100=1 | N=4 sum=409.7 mean=102.4 med=101.6 win=75.0 T3R=-92.7 max_loss=-92.7 tail<-100=0 | N=4 sum=-338.2 mean=-84.6 med=-48.4 win=0.0 T3R=-227.9 max_loss=-227.9 tail<-100=1 |
| `ETHUSDT_SELL_FADE_LONG_T100K_v40_60_H4` | N=6 sum=-488.9 mean=-81.5 med=-60.1 win=16.7 T3R=-492.8 max_loss=-271.1 tail<-100=2 | N=2 sum=-211.5 mean=-105.7 med=-105.7 win=50.0 T3R=-211.5 max_loss=-271.1 tail<-100=1 | N=4 sum=-277.4 mean=-69.4 med=-60.1 win=0.0 T3R=-154.3 max_loss=-154.3 tail<-100=1 |
| `SOLUSDT_SELL_FADE_LONG_T100K_v20_28_H4` | N=10 sum=60.3 mean=6.0 med=-29.1 win=30.0 T3R=-542.2 max_loss=-260.3 tail<-100=1 | N=3 sum=99.4 mean=33.1 med=-15.5 win=33.3 T3R=99.4 max_loss=-24.9 tail<-100=0 | N=7 sum=-39.1 mean=-5.6 med=-43.9 win=28.6 T3R=-468.5 max_loss=-260.3 tail<-100=1 |
| `BTCUSDT_SELL_FADE_LONG_T250K_v20_28_H4` | N=21 sum=-287.5 mean=-13.7 med=13.8 win=61.9 T3R=-658.4 max_loss=-397.0 tail<-100=2 | N=5 sum=24.0 mean=4.8 med=-6.4 win=40.0 T3R=-56.6 max_loss=-46.6 tail<-100=0 | N=16 sum=-311.6 mean=-19.5 med=17.0 win=68.8 T3R=-682.4 max_loss=-397.0 tail<-100=2 |
| `BTCUSDT_SELL_FADE_LONG_T500K_v28_40_H4` | N=17 sum=-323.6 mean=-19.0 med=25.1 win=58.8 T3R=-697.5 max_loss=-417.1 tail<-100=4 | N=4 sum=32.9 mean=8.2 med=-5.3 win=50.0 T3R=-54.6 max_loss=-54.6 tail<-100=0 | N=13 sum=-356.5 mean=-27.4 med=25.1 win=61.5 T3R=-730.4 max_loss=-417.1 tail<-100=4 |
| `BTCUSDT_SELL_FADE_LONG_T250K_v28_40_H4` | N=14 sum=-393.8 mean=-28.1 med=15.0 win=57.1 T3R=-749.3 max_loss=-413.4 tail<-100=3 | N=8 sum=246.0 mean=30.8 med=27.5 win=62.5 T3R=-76.2 max_loss=-59.6 tail<-100=0 | N=6 sum=-639.8 mean=-106.6 med=-65.1 win=50.0 T3R=-860.7 max_loss=-413.4 tail<-100=3 |
| `BTCUSDT_SELL_FADE_LONG_T500K_v20_28_H4` | N=17 sum=-452.7 mean=-26.6 med=7.7 win=52.9 T3R=-824.4 max_loss=-397.1 tail<-100=2 | N=7 sum=84.1 mean=12.0 med=-9.6 win=42.9 T3R=-115.5 max_loss=-54.1 tail<-100=0 | N=10 sum=-536.8 mean=-53.7 med=7.8 win=60.0 T3R=-824.6 max_loss=-397.1 tail<-100=2 |
| `SOLUSDT_SELL_FADE_LONG_T25K_v20_28_H4` | N=18 sum=233.0 mean=12.9 med=-14.0 win=44.4 T3R=-871.7 max_loss=-374.0 tail<-100=3 | N=3 sum=35.7 mean=11.9 med=-13.1 win=33.3 T3R=35.7 max_loss=-21.3 tail<-100=0 | N=15 sum=197.3 mean=13.2 med=-14.9 win=46.7 T3R=-907.4 max_loss=-374.0 tail<-100=3 |
| `SOLUSDT_SELL_FADE_LONG_T50K_v28_40_H4` | N=15 sum=-380.5 mean=-25.4 med=6.0 win=53.3 T3R=-881.1 max_loss=-484.2 tail<-100=3 | N=1 sum=-25.5 mean=-25.5 med=-25.5 win=0.0 T3R=-25.5 max_loss=-25.5 tail<-100=0 | N=14 sum=-355.0 mean=-25.4 med=13.6 win=57.1 T3R=-855.5 max_loss=-484.2 tail<-100=3 |
| `BTCUSDT_SELL_FADE_LONG_T1000K_v28_40_H4` | N=18 sum=-516.0 mean=-28.7 med=13.3 win=55.6 T3R=-953.5 max_loss=-384.0 tail<-100=5 | N=4 sum=75.0 mean=18.7 med=-11.8 win=50.0 T3R=-44.9 max_loss=-44.9 tail<-100=0 | N=14 sum=-591.0 mean=-42.2 med=34.4 win=57.1 T3R=-974.1 max_loss=-384.0 tail<-100=5 |
| `ETHUSDT_SELL_FADE_LONG_T150K_v40_60_H4` | N=10 sum=-639.2 mean=-63.9 med=-56.6 win=30.0 T3R=-1008.2 max_loss=-338.0 tail<-100=3 | N=2 sum=-211.5 mean=-105.7 med=-105.7 win=50.0 T3R=-211.5 max_loss=-271.1 tail<-100=1 | N=8 sum=-427.7 mean=-53.5 med=-56.6 win=25.0 T3R=-734.2 max_loss=-338.0 tail<-100=2 |
| `ETHUSDT_SELL_FADE_LONG_T200K_v40_60_H4` | N=12 sum=-886.5 mean=-73.9 med=-56.6 win=25.0 T3R=-1161.2 max_loss=-338.0 tail<-100=4 | N=3 sum=-56.7 mean=-18.9 med=59.7 win=66.7 T3R=-56.7 max_loss=-271.1 tail<-100=1 | N=9 sum=-829.8 mean=-92.2 med=-67.4 win=11.1 T3R=-853.1 max_loss=-338.0 tail<-100=3 |
| `SOLUSDT_SELL_FADE_LONG_T100K_v28_40_H4` | N=17 sum=-695.8 mean=-40.9 med=12.0 win=52.9 T3R=-1215.7 max_loss=-484.2 tail<-100=4 | N=3 sum=51.5 mean=17.2 med=22.0 win=66.7 T3R=51.5 max_loss=-38.8 tail<-100=0 | N=14 sum=-747.2 mean=-53.4 med=-10.2 win=50.0 T3R=-1267.1 max_loss=-484.2 tail<-100=4 |
| `ETHUSDT_SELL_FADE_LONG_T100K_v28_40_H4` | N=34 sum=-443.0 mean=-13.0 med=6.4 win=52.9 T3R=-1278.1 max_loss=-507.2 tail<-100=7 | N=16 sum=-48.5 mean=-3.0 med=-25.7 win=43.8 T3R=-500.2 max_loss=-285.3 tail<-100=2 | N=18 sum=-394.5 mean=-21.9 med=21.6 win=61.1 T3R=-1229.7 max_loss=-507.2 tail<-100=5 |
| `ETHUSDT_SELL_FADE_LONG_T100K_v20_28_H4` | N=41 sum=-72.9 mean=-1.8 med=-11.4 win=46.3 T3R=-1480.9 max_loss=-439.8 tail<-100=8 | N=19 sum=-399.5 mean=-21.0 med=-43.4 win=36.8 T3R=-958.3 max_loss=-256.9 tail<-100=3 | N=22 sum=326.6 mean=14.8 med=14.6 win=54.5 T3R=-1060.7 max_loss=-439.8 tail<-100=5 |
| `SOLUSDT_SELL_FADE_LONG_T50K_v20_28_H4` | N=23 sum=-712.1 mean=-31.0 med=-14.9 win=47.8 T3R=-1499.5 max_loss=-469.0 tail<-100=5 | N=1 sum=-39.5 mean=-39.5 med=-39.5 win=0.0 T3R=-39.5 max_loss=-39.5 tail<-100=0 | N=22 sum=-672.6 mean=-30.6 med=-3.4 win=50.0 T3R=-1460.1 max_loss=-469.0 tail<-100=5 |
| `SOLUSDT_SELL_FADE_LONG_T25K_v28_40_H4` | N=20 sum=-1033.5 mean=-51.7 med=-25.6 win=40.0 T3R=-1530.0 max_loss=-484.2 tail<-100=4 | N=3 sum=-150.6 mean=-50.2 med=-52.4 win=0.0 T3R=-150.6 max_loss=-72.6 tail<-100=0 | N=17 sum=-883.0 mean=-51.9 med=-24.1 win=47.1 T3R=-1379.4 max_loss=-484.2 tail<-100=4 |
| `ETHUSDT_SELL_FADE_LONG_T150K_v20_28_H4` | N=44 sum=-854.7 mean=-19.4 med=6.3 win=52.3 T3R=-2208.7 max_loss=-494.0 tail<-100=10 | N=15 sum=194.6 mean=13.0 med=-34.5 win=40.0 T3R=-475.0 max_loss=-221.3 tail<-100=1 | N=29 sum=-1049.3 mean=-36.2 med=17.1 win=58.6 T3R=-2351.4 max_loss=-494.0 tail<-100=9 |
| `ETHUSDT_SELL_FADE_LONG_T200K_v20_28_H4` | N=34 sum=-932.7 mean=-27.4 med=-14.8 win=47.1 T3R=-2257.7 max_loss=-494.0 tail<-100=10 | N=13 sum=-24.7 mean=-1.9 med=-28.4 win=30.8 T3R=-591.0 max_loss=-224.0 tail<-100=2 | N=21 sum=-908.0 mean=-43.2 med=26.0 win=57.1 T3R=-2133.4 max_loss=-494.0 tail<-100=8 |

## Read

- Pooled deep_bid vs shallow_bid delta T3R `-11708.2`, delta max_loss `13.2`.
- Cross-asset pooling only helps if the absorption relation is directionally stable across symbols and survives holdout.
