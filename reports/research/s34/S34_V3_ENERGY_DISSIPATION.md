# S34 v3 Energy Dissipation

Generated: `2026-06-28T23:47:25.837619+00:00`

Research-only. No live/paper/executor changes.

Discipline: Post-entry features are NOT legal entry inputs. Use only for management/diagnostics/forward observation.

## Overall

- All: N=541 sum=-2614.6 med=4.4 T3R=-4497.3 max_loss=-507.2 tail<-100=101
- Calibration: N=166 sum=1752.9 med=-6.9 T3R=608.8 max_loss=-285.3 tail<-100=11
- Holdout: N=375 sum=-4367.5 med=10.7 T3R=-6250.2 max_loss=-507.2 tail<-100=90

## Ranked Dissipation Tests

| Rank | Feature | Cal cut | Cal high | Hold high | Hold low | Hold dT3R |
| ---: | --- | ---: | --- | --- | --- | ---: |
| 1 | `total_replenish_120s_pct:q90` | 111.6304 | N=18 sum=980.7 med=45.1 T3R=508.1 max_loss=-113.7 tail<-100=1 | N=31 sum=2268.4 med=65.1 T3R=1293.6 max_loss=-115.8 tail<-100=1 | N=344 sum=-6636.0 med=6.0 T3R=-8518.6 max_loss=-507.2 tail<-100=89 | 9812.2 |
| 2 | `liq_deceleration_60s:q50` | 0.6275 | N=86 sum=-238.8 med=2.1 T3R=-731.9 max_loss=-285.3 tail<-100=7 | N=188 sum=3111.1 med=23.0 T3R=1228.4 max_loss=-507.2 tail<-100=36 | N=187 sum=-7478.6 med=-7.2 T3R=-9009.6 max_loss=-494.0 tail<-100=54 | 10238.0 |
| 3 | `liq_deceleration_60s:q75` | 0.9211 | N=42 sum=183.3 med=-6.9 T3R=-295.0 max_loss=-285.3 tail<-100=1 | N=118 sum=2998.4 med=14.8 T3R=1115.7 max_loss=-507.2 tail<-100=24 | N=257 sum=-7365.9 med=3.6 T3R=-8914.7 max_loss=-494.0 tail<-100=66 | 10030.4 |
| 4 | `liq_deceleration_120s:q50` | 0.4737 | N=85 sum=-392.5 med=-9.6 T3R=-885.7 max_loss=-285.3 tail<-100=6 | N=191 sum=2889.2 med=21.1 T3R=1006.6 max_loss=-507.2 tail<-100=35 | N=184 sum=-7256.7 med=-6.7 T3R=-8805.5 max_loss=-494.0 tail<-100=55 | 9812.1 |
| 5 | `total_replenish_60s_pct:q90` | 127.4043 | N=17 sum=149.0 med=-10.0 T3R=-323.5 max_loss=-215.3 tail<-100=2 | N=30 sum=1886.1 med=52.4 T3R=854.7 max_loss=-397.0 tail<-100=4 | N=345 sum=-6253.7 med=8.6 T3R=-8136.3 max_loss=-507.2 tail<-100=86 | 8991.0 |
| 6 | `total_replenish_30s_pct:q90` | 138.1488 | N=17 sum=212.5 med=2.1 T3R=-153.5 max_loss=-59.6 tail<-100=0 | N=24 sum=969.6 med=32.9 T3R=-5.2 max_loss=-437.4 tail<-100=3 | N=351 sum=-5337.1 med=8.0 T3R=-7219.7 max_loss=-507.2 tail<-100=87 | 7214.5 |
| 7 | `liq_deceleration_120s:q75` | 0.8131 | N=43 sum=217.3 med=2.1 T3R=-261.0 max_loss=-285.3 tail<-100=1 | N=129 sum=1803.4 med=23.8 T3R=-79.2 max_loss=-507.2 tail<-100=27 | N=246 sum=-6171.0 med=3.0 T3R=-7719.7 max_loss=-494.0 tail<-100=63 | 7640.5 |
| 8 | `liq_deceleration_30s:q75` | 0.97 | N=42 sum=367.2 med=-1.0 T3R=-111.0 max_loss=-285.3 tail<-100=2 | N=104 sum=1472.3 med=17.1 T3R=-395.1 max_loss=-507.2 tail<-100=20 | N=271 sum=-5839.8 med=3.6 T3R=-7619.3 max_loss=-494.0 tail<-100=70 | 7224.2 |
| 9 | `liq_deceleration_30s:q50` | 0.7076 | N=84 sum=184.1 med=-2.1 T3R=-328.5 max_loss=-285.3 tail<-100=8 | N=199 sum=1361.2 med=13.8 T3R=-521.5 max_loss=-507.2 tail<-100=39 | N=176 sum=-5728.7 med=-1.5 T3R=-7253.0 max_loss=-494.0 tail<-100=51 | 6731.5 |
| 10 | `liq_deceleration_60s:q90` | 0.9958 | N=19 sum=180.7 med=14.3 T3R=-246.6 max_loss=-285.3 tail<-100=1 | N=44 sum=1210.2 med=13.9 T3R=-657.2 max_loss=-507.2 tail<-100=7 | N=331 sum=-5577.8 med=10.2 T3R=-7357.2 max_loss=-494.0 tail<-100=83 | 6700.0 |
| 11 | `dissipation_score_60s:q50` | 0.3665 | N=82 sum=-143.2 med=-11.1 T3R=-795.7 max_loss=-285.3 tail<-100=5 | N=214 sum=1015.1 med=16.6 T3R=-852.3 max_loss=-507.2 tail<-100=54 | N=161 sum=-5382.7 med=-1.2 T3R=-6950.0 max_loss=-494.0 tail<-100=36 | 6097.7 |
| 12 | `liq_deceleration_120s:q90` | 0.977 | N=18 sum=122.3 med=-6.9 T3R=-356.0 max_loss=-285.3 tail<-100=1 | N=57 sum=882.6 med=23.8 T3R=-984.8 max_loss=-507.2 tail<-100=13 | N=318 sum=-5250.1 med=7.7 T3R=-7029.6 max_loss=-494.0 tail<-100=77 | 6044.8 |
| 13 | `dissipation_score_30s:q90` | 2.7235 | N=17 sum=643.2 med=41.0 T3R=158.5 max_loss=-70.5 tail<-100=0 | N=83 sum=804.5 med=10.2 T3R=-1078.2 max_loss=-507.2 tail<-100=21 | N=292 sum=-5172.0 med=12.1 T3R=-6738.9 max_loss=-494.0 tail<-100=69 | 5660.7 |
| 14 | `total_replenish_120s_pct:q75` | 53.0822 | N=42 sum=1199.1 med=29.2 T3R=726.6 max_loss=-113.7 tail<-100=1 | N=77 sum=2.5 med=18.4 T3R=-1313.3 max_loss=-507.2 tail<-100=15 | N=298 sum=-4370.1 med=9.1 T3R=-6223.3 max_loss=-494.0 tail<-100=75 | 4910.0 |
| 15 | `bid_replenish_30s_pct:q50` | 7.2019 | N=83 sum=2051.7 med=2.1 T3R=907.6 max_loss=-285.3 tail<-100=3 | N=179 sum=548.3 med=10.7 T3R=-1334.4 max_loss=-507.2 tail<-100=42 | N=196 sum=-4915.8 med=8.3 T3R=-6288.3 max_loss=-484.2 tail<-100=48 | 4953.9 |
| 16 | `bid_replenish_60s_pct:q90` | 768.902 | N=19 sum=798.3 med=41.0 T3R=298.2 max_loss=-59.6 tail<-100=0 | N=39 sum=319.4 med=46.6 T3R=-1518.7 max_loss=-507.2 tail<-100=13 | N=336 sum=-4686.9 med=8.0 T3R=-6548.5 max_loss=-494.0 tail<-100=77 | 5029.8 |
| 17 | `bid_replenish_30s_pct:q75` | 116.1093 | N=42 sum=1463.8 med=21.4 T3R=319.7 max_loss=-285.3 tail<-100=3 | N=129 sum=263.5 med=10.7 T3R=-1619.1 max_loss=-507.2 tail<-100=30 | N=246 sum=-4631.1 med=10.3 T3R=-6197.9 max_loss=-494.0 tail<-100=60 | 4578.8 |
| 18 | `bid_replenish_30s_pct:q90` | 352.8316 | N=18 sum=1332.4 med=41.0 T3R=390.0 max_loss=-59.6 tail<-100=0 | N=65 sum=40.3 med=10.2 T3R=-1813.0 max_loss=-507.2 tail<-100=19 | N=310 sum=-4407.8 med=12.0 T3R=-6201.4 max_loss=-494.0 tail<-100=71 | 4388.4 |
| 19 | `dissipation_score_30s:q50` | 0.5343 | N=83 sum=1521.2 med=0.9 T3R=578.8 max_loss=-285.3 tail<-100=5 | N=194 sum=46.1 med=10.7 T3R=-1836.5 max_loss=-507.2 tail<-100=43 | N=181 sum=-4413.7 med=8.6 T3R=-5800.1 max_loss=-494.0 tail<-100=47 | 3963.6 |
| 20 | `total_replenish_60s_pct:q50` | 10.467 | N=82 sum=1330.9 med=-1.9 T3R=267.5 max_loss=-271.1 tail<-100=5 | N=172 sum=-33.7 med=10.7 T3R=-1887.0 max_loss=-507.2 tail<-100=41 | N=203 sum=-4333.9 med=10.2 T3R=-6127.4 max_loss=-494.0 tail<-100=49 | 4240.4 |
| 21 | `bid_replenish_60s_pct:q50` | 14.4709 | N=82 sum=1598.6 med=2.1 T3R=454.5 max_loss=-271.1 tail<-100=4 | N=186 sum=-108.7 med=19.4 T3R=-1946.7 max_loss=-507.2 tail<-100=48 | N=189 sum=-4258.8 med=0.9 T3R=-6088.8 max_loss=-494.0 tail<-100=42 | 4142.1 |
| 22 | `liq_deceleration_30s:q90` | 0.9999 | N=18 sum=127.2 med=-0.6 T3R=-304.4 max_loss=-285.3 tail<-100=1 | N=36 sum=-668.6 med=-31.6 T3R=-1968.6 max_loss=-507.2 tail<-100=8 | N=339 sum=-3698.9 med=12.0 T3R=-5552.2 max_loss=-494.0 tail<-100=82 | 3583.6 |
| 23 | `dissipation_score_30s:q75` | 1.4674 | N=42 sum=723.3 med=7.2 T3R=-219.1 max_loss=-285.3 tail<-100=3 | N=126 sum=-92.9 med=10.5 T3R=-1975.6 max_loss=-507.2 tail<-100=30 | N=249 sum=-4274.6 med=12.0 T3R=-5841.5 max_loss=-494.0 tail<-100=60 | 3865.9 |
| 24 | `bid_replenish_120s_pct:q90` | 375.6378 | N=17 sum=817.5 med=39.4 T3R=80.9 max_loss=-285.3 tail<-100=1 | N=54 sum=-991.4 med=14.6 T3R=-2039.2 max_loss=-507.2 tail<-100=17 | N=321 sum=-3376.2 med=8.0 T3R=-5258.8 max_loss=-494.0 tail<-100=73 | 3219.6 |
| 25 | `total_replenish_30s_pct:q75` | 78.6795 | N=43 sum=1453.9 med=2.1 T3R=511.5 max_loss=-221.3 tail<-100=1 | N=50 sum=-1081.5 med=-16.1 T3R=-2112.9 max_loss=-507.2 tail<-100=12 | N=325 sum=-3286.0 med=10.7 T3R=-5168.7 max_loss=-494.0 tail<-100=78 | 3055.8 |
| 26 | `dissipation_score_120s:q90` | 2.9162 | N=17 sum=909.0 med=59.1 T3R=256.5 max_loss=-285.3 tail<-100=1 | N=53 sum=-1127.9 med=10.7 T3R=-2147.4 max_loss=-507.2 tail<-100=16 | N=322 sum=-3239.7 med=10.3 T3R=-5122.3 max_loss=-494.0 tail<-100=74 | 2974.9 |
| 27 | `dissipation_score_60s:q90` | 4.0632 | N=17 sum=801.3 med=41.0 T3R=301.1 max_loss=-113.7 tail<-100=1 | N=54 sum=-358.4 med=18.3 T3R=-2196.5 max_loss=-507.2 tail<-100=18 | N=321 sum=-4009.1 med=8.0 T3R=-5870.7 max_loss=-494.0 tail<-100=72 | 3674.2 |
| 28 | `total_replenish_60s_pct:q75` | 54.8228 | N=41 sum=1046.0 med=2.1 T3R=515.2 max_loss=-215.3 tail<-100=2 | N=78 sum=-1186.9 med=-4.8 T3R=-2218.3 max_loss=-507.2 tail<-100=18 | N=297 sum=-3180.6 med=13.8 T3R=-5063.3 max_loss=-494.0 tail<-100=72 | 2845.0 |
| 29 | `dissipation_score_120s:q50` | 0.2919 | N=85 sum=-439.1 med=-11.4 T3R=-1091.5 max_loss=-285.3 tail<-100=5 | N=223 sum=-551.8 med=12.1 T3R=-2434.4 max_loss=-507.2 tail<-100=55 | N=152 sum=-3815.7 med=1.5 T3R=-5364.5 max_loss=-494.0 tail<-100=35 | 2930.1 |
| 30 | `dissipation_score_120s:q75` | 1.2523 | N=43 sum=422.3 med=-9.6 T3R=-230.1 max_loss=-285.3 tail<-100=2 | N=117 sum=-748.1 med=8.6 T3R=-2615.5 max_loss=-507.2 tail<-100=29 | N=258 sum=-3619.4 med=12.0 T3R=-5398.9 max_loss=-494.0 tail<-100=61 | 2783.4 |
| 31 | `dissipation_score_60s:q75` | 1.5862 | N=41 sum=135.1 med=-10.8 T3R=-517.3 max_loss=-271.1 tail<-100=4 | N=96 sum=-944.7 med=12.0 T3R=-2782.7 max_loss=-507.2 tail<-100=28 | N=279 sum=-3422.8 med=7.9 T3R=-5284.4 max_loss=-494.0 tail<-100=62 | 2501.7 |
| 32 | `total_replenish_30s_pct:q50` | 15.7462 | N=83 sum=1897.6 med=18.1 T3R=834.3 max_loss=-224.0 tail<-100=3 | N=159 sum=-970.1 med=12.0 T3R=-2808.2 max_loss=-507.2 tail<-100=41 | N=216 sum=-3397.4 med=10.5 T3R=-5227.3 max_loss=-484.2 tail<-100=49 | 2419.1 |
| 33 | `bid_replenish_120s_pct:q75` | 71.176 | N=42 sum=1325.3 med=35.9 T3R=471.1 max_loss=-285.3 tail<-100=2 | N=128 sum=-1211.2 med=8.2 T3R=-3049.2 max_loss=-507.2 tail<-100=35 | N=247 sum=-3156.4 med=12.0 T3R=-5018.0 max_loss=-494.0 tail<-100=55 | 1968.8 |
| 34 | `bid_replenish_60s_pct:q75` | 77.6982 | N=41 sum=633.5 med=20.8 T3R=-220.7 max_loss=-271.1 tail<-100=4 | N=128 sum=-1328.8 med=10.7 T3R=-3166.8 max_loss=-507.2 tail<-100=35 | N=247 sum=-3038.7 med=8.6 T3R=-4900.3 max_loss=-494.0 tail<-100=55 | 1733.5 |
| 35 | `total_replenish_120s_pct:q50` | 10.7903 | N=83 sum=2072.0 med=18.1 T3R=1335.4 max_loss=-285.3 tail<-100=2 | N=150 sum=-2390.5 med=-3.9 T3R=-3996.9 max_loss=-507.2 tail<-100=37 | N=225 sum=-1977.0 med=14.8 T3R=-3830.3 max_loss=-494.0 tail<-100=53 | -166.6 |
| 36 | `bid_replenish_120s_pct:q50` | 2.7375 | N=83 sum=1073.0 med=-4.8 T3R=-71.2 max_loss=-285.3 tail<-100=5 | N=209 sum=-2621.9 med=10.7 T3R=-4489.3 max_loss=-507.2 tail<-100=53 | N=166 sum=-1745.6 med=8.0 T3R=-3362.4 max_loss=-484.2 tail<-100=37 | -1126.9 |

## Best Holdout Feature By Symbol

Best feature: `total_replenish_120s_pct` with calibration q90 cut `111.6304`.

| Symbol | All | High | Low |
| --- | --- | --- | --- |
| `BTCUSDT` | N=85 sum=-1916.5 med=13.8 T3R=-2665.6 max_loss=-417.1 tail<-100=20 | N=12 sum=562.5 med=66.1 T3R=168.5 max_loss=-115.8 tail<-100=1 | N=73 sum=-2479.0 med=10.2 T3R=-3228.0 max_loss=-417.1 tail<-100=19 |
| `ETHUSDT` | N=157 sum=-2074.0 med=15.7 T3R=-3912.0 max_loss=-507.2 tail<-100=42 | N=17 sum=1374.0 med=37.0 T3R=505.9 max_loss=-74.6 tail<-100=0 | N=140 sum=-3448.0 med=2.2 T3R=-5286.0 max_loss=-507.2 tail<-100=42 |
| `SOLUSDT` | N=133 sum=-377.0 med=-0.1 T3R=-2238.6 max_loss=-484.2 tail<-100=28 | N=2 sum=332.0 med=166.0 T3R=332.0 max_loss=-40.7 tail<-100=0 | N=131 sum=-709.0 med=-0.1 T3R=-2570.6 max_loss=-484.2 tail<-100=28 |

## Read

- A positive dissipation test can become an exit/management observer, not an entry gate, because it is only known after entry.
- Look for high holdout T3R with lower tails and enough N; otherwise it is another in-sample separator.
