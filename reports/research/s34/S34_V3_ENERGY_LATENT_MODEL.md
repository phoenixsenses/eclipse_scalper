# S34 v3 Energy Latent Model

Generated: `2026-06-28T23:46:01.201858+00:00`

Research-only. No live/paper/executor changes.

## Discipline

- Z-score rule: Calibration route mean/std only; holdout never used to define z-score.
- Min split N for candidate promotion: `40`.

## Coverage

- `rows`: `541`
- `cal_rows`: `166`
- `hold_rows`: `375`
- `z_energy_rows`: `453`

## Correlation: Route-Normalized Features vs Net Bps

| Feature | Cal N | Cal Pearson | Cal Spearman | Hold N | Hold Pearson | Hold Spearman | All Pearson |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `running_notional_cal_route_z` | 156 | 0.101 | 0.098 | 297 | 0.102 | 0.049 | 0.089 |
| `running_accel_cal_route_z` | 156 | 0.097 | 0.051 | 297 | 0.086 | 0.065 | 0.075 |
| `energy_z` | 156 | 0.104 | 0.078 | 297 | 0.098 | 0.082 | 0.086 |
| `bid_depth_usd_cal_route_z` | 156 | -0.103 | -0.094 | 297 | 0.017 | -0.087 | 0.013 |
| `total_top_depth_usd_cal_route_z` | 156 | -0.089 | -0.068 | 297 | -0.023 | -0.095 | -0.029 |
| `book_imbalance_cal_route_z` | 156 | 0.002 | 0.047 | 297 | 0.090 | -0.076 | 0.081 |
| `spread_bps_cal_route_z` | 156 | 0.051 | 0.086 | 297 | 0.083 | 0.036 | 0.068 |
| `static_structure_z` | 156 | -0.058 | -0.054 | 297 | 0.091 | 0.013 | 0.075 |

## Gate Tests By Calibration Cuts

| Rank | Feature | Cut | Calibration High | Holdout High | Holdout Low | Hold dT3R |
| ---: | --- | ---: | --- | --- | --- | ---: |
| 1 | `running_accel_cal_route_z:q90` | 1.3615 | N=16 sum=379.2 med=24.1 T3R=7.7 max_loss=-92.7 tail<-100=0 | N=58 sum=677.2 med=20.8 T3R=-375.6 max_loss=-484.2 tail<-100=9 | N=239 sum=-5295.8 med=12.0 T3R=-7133.8 max_loss=-507.2 tail<-100=62 | 6758.2 |
| 2 | `energy_z:q90` | 1.3727 | N=16 sum=294.6 med=-18.2 T3R=-76.9 max_loss=-92.7 tail<-100=0 | N=58 sum=483.8 med=20.8 T3R=-569.0 max_loss=-484.2 tail<-100=10 | N=239 sum=-5102.4 med=12.0 T3R=-6940.4 max_loss=-507.2 tail<-100=61 | 6371.4 |
| 3 | `running_notional_cal_route_z:q90` | 1.4277 | N=16 sum=522.9 med=35.5 T3R=151.4 max_loss=-92.7 tail<-100=0 | N=54 sum=197.7 med=8.2 T3R=-1085.6 max_loss=-484.2 tail<-100=10 | N=243 sum=-4816.3 med=12.0 T3R=-6654.4 max_loss=-507.2 tail<-100=61 | 5568.8 |
| 4 | `running_accel_cal_route_z:q75` | 0.5639 | N=40 sum=1147.3 med=-2.6 T3R=517.6 max_loss=-92.7 tail<-100=0 | N=86 sum=-653.2 med=17.1 T3R=-1706.9 max_loss=-494.0 tail<-100=19 | N=211 sum=-3965.4 med=10.7 T3R=-5803.4 max_loss=-507.2 tail<-100=52 | 4096.5 |
| 5 | `energy_z:q75` | 0.45 | N=40 sum=1110.9 med=2.2 T3R=459.7 max_loss=-92.7 tail<-100=0 | N=81 sum=-543.1 med=20.4 T3R=-1827.4 max_loss=-494.0 tail<-100=18 | N=216 sum=-4075.5 med=11.4 T3R=-5913.5 max_loss=-507.2 tail<-100=53 | 4086.1 |
| 6 | `running_notional_cal_route_z:q75` | 0.3877 | N=40 sum=669.6 med=-10.9 T3R=256.8 max_loss=-92.7 tail<-100=0 | N=79 sum=-1092.5 med=12.1 T3R=-2376.7 max_loss=-494.0 tail<-100=19 | N=218 sum=-3526.1 med=12.0 T3R=-5364.2 max_loss=-507.2 tail<-100=52 | 2987.5 |
| 7 | `running_notional_cal_route_z:median` | -0.469 | N=78 sum=1106.2 med=-10.2 T3R=163.8 max_loss=-271.1 tail<-100=3 | N=165 sum=-755.0 med=21.1 T3R=-2540.3 max_loss=-494.0 tail<-100=38 | N=132 sum=-3863.6 med=-2.9 T3R=-5465.1 max_loss=-507.2 tail<-100=33 | 2924.8 |
| 8 | `energy_z:median` | -0.3195 | N=78 sum=1297.6 med=-10.2 T3R=355.2 max_loss=-224.0 tail<-100=2 | N=159 sum=-1235.8 med=20.4 T3R=-3021.1 max_loss=-494.0 tail<-100=40 | N=138 sum=-3382.8 med=-0.3 T3R=-5002.4 max_loss=-507.2 tail<-100=31 | 1981.3 |
| 9 | `running_accel_cal_route_z:median` | -0.181 | N=78 sum=1324.8 med=-10.2 T3R=382.4 max_loss=-224.0 tail<-100=2 | N=162 sum=-2665.9 med=14.8 T3R=-4285.3 max_loss=-494.0 tail<-100=41 | N=135 sum=-1952.7 med=3.6 T3R=-3717.0 max_loss=-507.2 tail<-100=30 | -568.3 |
| 10 | `static_structure_z:q75` | 0.4335 | N=40 sum=349.2 med=-4.1 T3R=-148.7 max_loss=-271.1 tail<-100=5 | N=253 sum=-2741.4 med=12.0 T3R=-4579.4 max_loss=-507.2 tail<-100=58 | N=44 sum=-1877.2 med=12.0 T3R=-2416.4 max_loss=-413.4 tail<-100=13 | -2163.0 |
| 11 | `static_structure_z:q90` | 0.8376 | N=16 sum=787.9 med=98.2 T3R=290.7 max_loss=-132.0 tail<-100=2 | N=252 sum=-2753.5 med=12.0 T3R=-4591.5 max_loss=-507.2 tail<-100=58 | N=45 sum=-1865.1 med=12.1 T3R=-2404.3 max_loss=-413.4 tail<-100=13 | -2187.2 |
| 12 | `static_structure_z:median` | -0.0461 | N=78 sum=140.6 med=-14.7 T3R=-508.9 max_loss=-271.1 tail<-100=7 | N=260 sum=-3023.6 med=12.0 T3R=-4861.7 max_loss=-507.2 tail<-100=61 | N=37 sum=-1594.9 med=10.2 T3R=-2134.1 max_loss=-413.4 tail<-100=10 | -2727.6 |

## Holdout Energy Confluence

Energy high cut: `energy_z >= 0.45` from calibration q75.

| Rank | Confluence | Holdout summary |
| ---: | --- | --- |
| 1 | `energy_high+not_absorbed` | N=42 sum=1662.8 med=35.7 T3R=610.1 max_loss=-460.6 tail<-100=6 |
| 2 | `energy_high+sync+not_absorbed` | N=40 sum=1591.2 med=35.6 T3R=538.4 max_loss=-460.6 tail<-100=6 |
| 3 | `energy_high+idio` | N=12 sum=227.1 med=17.3 T3R=-271.3 max_loss=-140.5 tail<-100=2 |
| 4 | `energy_high+mixed` | N=29 sum=469.0 med=20.4 T3R=-464.1 max_loss=-460.6 tail<-100=4 |
| 5 | `energy_high+sync+mixed` | N=27 sum=397.4 med=20.4 T3R=-535.8 max_loss=-460.6 tail<-100=4 |
| 6 | `energy_high` | N=81 sum=-543.1 med=20.4 T3R=-1827.4 max_loss=-494.0 tail<-100=18 |
| 7 | `energy_high+sync` | N=69 sum=-770.1 med=20.4 T3R=-2053.5 max_loss=-494.0 tail<-100=16 |

## Holdout By Symbol

| Symbol | All | Energy high | Energy low |
| --- | --- | --- | --- |
| `BTCUSDT` | N=85 sum=-1916.5 med=13.8 T3R=-2665.6 max_loss=-417.1 tail<-100=20 | N=26 sum=19.8 med=9.9 T3R=-660.7 max_loss=-417.1 tail<-100=5 | N=55 sum=-1818.3 med=16.2 T3R=-2391.5 max_loss=-413.4 tail<-100=14 |
| `ETHUSDT` | N=157 sum=-2074.0 med=15.7 T3R=-3912.0 max_loss=-507.2 tail<-100=42 | N=24 sum=-439.4 med=22.0 T3R=-1428.4 max_loss=-494.0 tail<-100=6 | N=127 sum=-2093.6 med=12.0 T3R=-3931.6 max_loss=-507.2 tail<-100=33 |
| `SOLUSDT` | N=133 sum=-377.0 med=-0.1 T3R=-2238.6 max_loss=-484.2 tail<-100=28 | N=31 sum=-123.4 med=8.6 T3R=-1382.7 max_loss=-484.2 tail<-100=7 | N=34 sum=-163.6 med=-28.3 T3R=-1345.6 max_loss=-484.2 tail<-100=6 |

## Best Holdout Route-Level Energy Deltas

| Rank | Route | All | Energy high | Energy low | dT3R |
| ---: | --- | --- | --- | --- | ---: |
| 1 | `ETHUSDT_SELL_FADE_LONG_T100K_v28_40_H4` | N=18 sum=-394.5 med=21.6 T3R=-1229.7 max_loss=-507.2 tail<-100=5 | N=3 sum=415.4 med=62.5 T3R=415.4 max_loss=23.6 tail<-100=0 | N=15 sum=-809.9 med=10.7 T3R=-1499.4 max_loss=-507.2 tail<-100=5 | 1914.8 |
| 2 | `ETHUSDT_SELL_FADE_LONG_T200K_v20_28_H4` | N=21 sum=-908.0 med=26.0 T3R=-2133.4 max_loss=-494.0 tail<-100=8 | N=3 sum=-220.3 med=26.0 T3R=-220.3 max_loss=-280.7 tail<-100=1 | N=18 sum=-687.6 med=19.7 T3R=-1913.1 max_loss=-494.0 tail<-100=7 | 1692.8 |
| 3 | `SOLUSDT_SELL_FADE_LONG_T100K_v28_40_H4` | N=14 sum=-747.2 med=-10.2 T3R=-1267.1 max_loss=-484.2 tail<-100=4 | N=0 sum=0.0 med=None T3R=0.0 max_loss=None tail<-100=0 | N=14 sum=-747.2 med=-10.2 T3R=-1267.1 max_loss=-484.2 tail<-100=4 | 1267.1 |
| 4 | `ETHUSDT_SELL_FADE_LONG_T150K_v20_28_H4` | N=29 sum=-1049.3 med=17.1 T3R=-2351.4 max_loss=-494.0 tail<-100=9 | N=3 sum=-740.3 med=-280.7 T3R=-740.3 max_loss=-494.0 tail<-100=2 | N=26 sum=-309.0 med=26.0 T3R=-1611.2 max_loss=-439.8 tail<-100=7 | 870.9 |
| 5 | `ETHUSDT_SELL_FADE_LONG_T200K_v40_60_H4` | N=9 sum=-829.8 med=-67.4 T3R=-853.1 max_loss=-338.0 tail<-100=3 | N=1 sum=-67.4 med=-67.4 T3R=-67.4 max_loss=-67.4 tail<-100=0 | N=8 sum=-762.4 med=-59.7 T3R=-785.7 max_loss=-338.0 tail<-100=3 | 718.3 |
| 6 | `ETHUSDT_SELL_FADE_LONG_T150K_v28_40_H4` | N=16 sum=434.0 med=35.0 T3R=-344.2 max_loss=-296.2 tail<-100=3 | N=6 sum=656.2 med=99.4 T3R=8.3 max_loss=-74.6 tail<-100=0 | N=10 sum=-222.2 med=17.1 T3R=-626.5 max_loss=-296.2 tail<-100=3 | 634.8 |
| 7 | `BTCUSDT_SELL_FADE_LONG_T1000K_v28_40_H4` | N=14 sum=-591.0 med=34.4 T3R=-974.1 max_loss=-384.0 tail<-100=5 | N=4 sum=-131.9 med=7.7 T3R=-212.4 max_loss=-212.4 tail<-100=1 | N=10 sum=-459.2 med=46.9 T3R=-842.3 max_loss=-384.0 tail<-100=4 | 629.9 |
| 8 | `BTCUSDT_SELL_FADE_LONG_T250K_v20_28_H4` | N=16 sum=-311.6 med=17.0 T3R=-682.4 max_loss=-397.0 tail<-100=2 | N=3 sum=-39.9 med=7.7 T3R=-39.9 max_loss=-59.8 tail<-100=0 | N=13 sum=-271.7 med=30.7 T3R=-642.5 max_loss=-397.0 tail<-100=2 | 602.6 |
| 9 | `ETHUSDT_SELL_FADE_LONG_T150K_v40_60_H4` | N=8 sum=-427.7 med=-56.6 T3R=-734.2 max_loss=-338.0 tail<-100=2 | N=1 sum=-67.4 med=-67.4 T3R=-67.4 max_loss=-67.4 tail<-100=0 | N=7 sum=-360.3 med=-45.8 T3R=-666.7 max_loss=-338.0 tail<-100=2 | 599.3 |
| 10 | `ETHUSDT_SELL_FADE_LONG_T200K_v28_40_H4` | N=24 sum=593.1 med=22.2 T3R=-238.7 max_loss=-291.8 tail<-100=3 | N=2 sum=156.6 med=78.3 T3R=156.6 max_loss=20.4 tail<-100=0 | N=22 sum=436.4 med=22.2 T3R=-395.3 max_loss=-291.8 tail<-100=3 | 551.9 |
| 11 | `BTCUSDT_SELL_FADE_LONG_T250K_v28_40_H4` | N=6 sum=-639.8 med=-65.1 T3R=-860.7 max_loss=-413.4 tail<-100=3 | N=2 sum=-19.6 med=-9.8 T3R=-19.6 max_loss=-140.5 tail<-100=1 | N=4 sum=-620.2 med=-148.3 T3R=-413.4 max_loss=-413.4 tail<-100=2 | 393.8 |
| 12 | `SOLUSDT_SELL_FADE_LONG_T100K_v20_28_H4` | N=7 sum=-39.1 med=-43.9 T3R=-468.5 max_loss=-260.3 tail<-100=1 | N=2 sum=21.6 med=10.8 T3R=21.6 max_loss=-43.9 tail<-100=0 | N=5 sum=-60.7 med=-75.2 T3R=-349.4 max_loss=-260.3 tail<-100=1 | 371.0 |
| 13 | `BTCUSDT_SELL_FADE_LONG_T1000K_v40_60_H4` | N=4 sum=-6.1 med=43.5 T3R=-232.9 max_loss=-232.9 tail<-100=1 | N=0 sum=0.0 med=None T3R=0.0 max_loss=None tail<-100=0 | N=4 sum=-6.1 med=43.5 T3R=-232.9 max_loss=-232.9 tail<-100=1 | 232.9 |
| 14 | `BTCUSDT_SELL_FADE_LONG_T250K_v40_60_H4` | N=4 sum=-338.2 med=-48.4 T3R=-227.9 max_loss=-227.9 tail<-100=1 | N=0 sum=0.0 med=None T3R=0.0 max_loss=None tail<-100=0 | N=4 sum=-338.2 med=-48.4 T3R=-227.9 max_loss=-227.9 tail<-100=1 | 227.9 |
| 15 | `BTCUSDT_SELL_FADE_LONG_T500K_v40_60_H4` | N=6 sum=244.2 med=58.3 T3R=-198.7 max_loss=-213.0 tail<-100=1 | N=0 sum=0.0 med=None T3R=0.0 max_loss=None tail<-100=0 | N=6 sum=244.2 med=58.3 T3R=-198.7 max_loss=-213.0 tail<-100=1 | 198.7 |

## Read

- Energy is only useful if calibration-cut high energy improves holdout without collapsing N.
- If high-energy confluence is still tail-heavy, it is a sizing/context variable, not a standalone entry rule.
- N<40 per split remains a hypothesis even when the row looks strong.
