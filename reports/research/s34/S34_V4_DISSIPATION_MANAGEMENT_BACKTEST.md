# S34 v4 Dissipation Management Backtest

Generated: `2026-06-29T00:42:17.890225+00:00`

`RESEARCH_ONLY_NO_LIVE_NO_PAPER` - this tests post-entry management only.

## Coverage

- rows: `541`
- calibration rows: `166`
- holdout rows: `375`
- split: `{'method': 'chronological_month_tail', 'months': ['2026-04', '2026-06'], 'holdout_months': ['2026-06']}`

## Baseline Hold 4h

- all: N=541 sum=-2614.6 med=4.4 T3R=-4497.3 max_loss=-507.2 tail<-100=101 tail<-200=70
- cal: N=166 sum=1752.9 med=-6.9 T3R=608.8 max_loss=-285.3 tail<-100=11 tail<-200=8
- hold: N=375 sum=-4367.5 med=10.7 T3R=-6250.2 max_loss=-507.2 tail<-100=90 tail<-200=62

## Primary Predefined Rule

- config: `tau120_dual_and_replQ50_decelQ50`
- cuts from calibration: `{'replenish_cut': 10.7903, 'decel_cut': 0.4737}`
- hold decisions: `{'hold_4h': 70, 'exit_early': 305}`
- cal: base N=165 sum=1739.7 med=-7.4 T3R=595.5 max_loss=-285.3 tail<-100=11 tail<-200=8; managed N=165 sum=-1132.5 med=-6.0 T3R=-1546.3 max_loss=-285.3 tail<-100=2 tail<-200=1; dSum=-2872.2 dT3R=-2141.8 dMaxLoss=0.0
- hold: base N=375 sum=-4367.5 med=10.7 T3R=-6250.2 max_loss=-507.2 tail<-100=90 tail<-200=62; managed N=375 sum=-2405.8 med=-4.2 T3R=-3654.4 max_loss=-507.2 tail<-100=15 tail<-200=8; dSum=1961.7 dT3R=2595.8 dMaxLoss=0.0

### Primary Holdout By Symbol

| Symbol | N | Baseline | Managed | Decisions |
| --- | ---: | --- | --- | --- |
| `BTCUSDT` | 85 | N=85 sum=-1916.5 med=13.8 T3R=-2665.6 max_loss=-417.1 tail<-100=20 tail<-200=15 | N=85 sum=357.2 med=-2.9 T3R=-86.8 max_loss=-115.8 tail<-100=1 tail<-200=0 | `{'hold_4h': 15, 'exit_early': 70}` |
| `ETHUSDT` | 157 | N=157 sum=-2074.0 med=15.7 T3R=-3912.0 max_loss=-507.2 tail<-100=42 tail<-200=27 | N=157 sum=-3578.2 med=-4.0 T3R=-3939.8 max_loss=-507.2 tail<-100=11 tail<-200=7 | `{'hold_4h': 26, 'exit_early': 131}` |
| `SOLUSDT` | 133 | N=133 sum=-377.0 med=-0.1 T3R=-2238.6 max_loss=-484.2 tail<-100=28 tail<-200=20 | N=133 sum=815.2 med=-4.6 T3R=-433.4 max_loss=-234.3 tail<-100=3 tail<-200=1 | `{'hold_4h': 29, 'exit_early': 104}` |

## Best Exploratory Config By Holdout dT3R

- config: `tau60_dual_or_replQ90_decelQ50`
- cuts from calibration: `{'replenish_cut': 127.4043, 'decel_cut': 0.6275}`
- hold decisions: `{'hold_4h': 208, 'exit_early': 167}`
- cal: base N=162 sum=1368.2 med=-9.8 T3R=224.0 max_loss=-285.3 tail<-100=11 tail<-200=8; managed N=162 sum=-1134.2 med=-7.3 T3R=-1646.8 max_loss=-285.3 tail<-100=8 tail<-200=7; dSum=-2502.4 dT3R=-1870.8 dMaxLoss=0.0
- hold: base N=375 sum=-4367.5 med=10.7 T3R=-6250.2 max_loss=-507.2 tail<-100=90 tail<-200=62; managed N=375 sum=2262.8 med=-8.1 T3R=380.1 max_loss=-507.2 tail<-100=39 tail<-200=25; dSum=6630.3 dT3R=6630.3 dMaxLoss=0.0

### Best Holdout By Symbol

| Symbol | N | Baseline | Managed | Decisions |
| --- | ---: | --- | --- | --- |
| `BTCUSDT` | 85 | N=85 sum=-1916.5 med=13.8 T3R=-2665.6 max_loss=-417.1 tail<-100=20 tail<-200=15 | N=85 sum=-458.6 med=-6.4 T3R=-942.5 max_loss=-397.1 tail<-100=7 tail<-200=5 | `{'hold_4h': 46, 'exit_early': 39}` |
| `ETHUSDT` | 157 | N=157 sum=-2074.0 med=15.7 T3R=-3912.0 max_loss=-507.2 tail<-100=42 tail<-200=27 | N=157 sum=1985.2 med=-8.1 T3R=147.2 max_loss=-507.2 tail<-100=17 tail<-200=9 | `{'hold_4h': 81, 'exit_early': 76}` |
| `SOLUSDT` | 133 | N=133 sum=-377.0 med=-0.1 T3R=-2238.6 max_loss=-484.2 tail<-100=28 tail<-200=20 | N=133 sum=736.2 med=-10.3 T3R=-931.0 max_loss=-484.2 tail<-100=15 tail<-200=11 | `{'hold_4h': 81, 'exit_early': 52}` |

## Cal + Hold Consistent Improvers

Configs with both cal and hold delta_sum > 0 and delta_T3R > 0: `3`

| Rank | Config | Cal dSum/dT3R | Hold dSum/dT3R | Hold managed |
| ---: | --- | ---: | ---: | --- |
| 1 | `tau180_replenish_only_replQ50_decelQ50` | 417.1/497.9 | 1216.8/1305.8 | N=373 sum=-2343.3 med=0.7 T3R=-4136.9 max_loss=-507.2 tail<-100=44 tail<-200=31 |
| 2 | `tau180_replenish_only_replQ50_decelQ75` | 417.1/497.9 | 1216.8/1305.8 | N=373 sum=-2343.3 med=0.7 T3R=-4136.9 max_loss=-507.2 tail<-100=44 tail<-200=31 |
| 3 | `tau180_dual_or_replQ50_decelQ75` | 110.1/190.8 | 765.2/765.1 | N=373 sum=-2794.9 med=-3.2 T3R=-4677.6 max_loss=-507.2 tail<-100=61 tail<-200=41 |

## Live v0.2 Lane Diagnostic

- config applied: `tau60_dual_or_replQ90_decelQ50`
- note: Small-N diagnostic only; this is the currently live v0.2 lane shape.
- rows: `16`
- decisions: `{'hold_4h': 7, 'exit_early': 9}`
- baseline: N=16 sum=825.4 med=16.1 T3R=96.9 max_loss=-92.8 tail<-100=0 tail<-200=0
- managed: N=16 sum=388.2 med=-6.0 T3R=-65.7 max_loss=-40.0 tail<-100=0 tail<-200=0

## Ranked Configs

| Rank | Config | Hold decisions | Cal dSum/dT3R | Hold baseline | Hold managed | Hold dSum | Hold dT3R | dMaxLoss |
| ---: | --- | --- | ---: | --- | --- | ---: | ---: | ---: |
| 1 | `tau60_dual_or_replQ90_decelQ50` | `{'hold_4h': 208, 'exit_early': 167}` | -2502.4/-1870.8 | N=375 sum=-4367.5 med=10.7 T3R=-6250.2 max_loss=-507.2 tail<-100=90 tail<-200=62 | N=375 sum=2262.8 med=-8.1 T3R=380.1 max_loss=-507.2 tail<-100=39 tail<-200=25 | 6630.3 | 6630.3 | 0.0 |
| 2 | `tau60_dual_or_replQ90_decelQ75` | `{'hold_4h': 142, 'exit_early': 233}` | -2446.7/-1815.1 | N=375 sum=-4367.5 med=10.7 T3R=-6250.2 max_loss=-507.2 tail<-100=90 tail<-200=62 | N=375 sum=2241.8 med=-6.7 T3R=359.2 max_loss=-507.2 tail<-100=28 tail<-200=16 | 6609.3 | 6609.4 | 0.0 |
| 3 | `tau120_dual_or_replQ90_decelQ50` | `{'hold_4h': 213, 'exit_early': 162}` | -2405.6/-1774.0 | N=375 sum=-4367.5 med=10.7 T3R=-6250.2 max_loss=-507.2 tail<-100=90 tail<-200=62 | N=375 sum=2154.4 med=-4.6 T3R=271.8 max_loss=-507.2 tail<-100=36 tail<-200=23 | 6521.9 | 6522.0 | 0.0 |
| 4 | `tau90_dual_or_replQ90_decelQ50` | `{'hold_4h': 209, 'exit_early': 166}` | -3739.5/-3088.6 | N=375 sum=-4367.5 med=10.7 T3R=-6250.2 max_loss=-507.2 tail<-100=90 tail<-200=62 | N=375 sum=1637.5 med=-6.5 T3R=-245.2 max_loss=-507.2 tail<-100=40 tail<-200=25 | 6005.0 | 6005.0 | 0.0 |
| 5 | `tau120_dual_or_replQ90_decelQ75` | `{'hold_4h': 156, 'exit_early': 219}` | -2049.6/-1418.1 | N=375 sum=-4367.5 med=10.7 T3R=-6250.2 max_loss=-507.2 tail<-100=90 tail<-200=62 | N=375 sum=1500.4 med=-5.8 T3R=-382.2 max_loss=-507.2 tail<-100=28 tail<-200=18 | 5867.9 | 5868.0 | 0.0 |
| 6 | `tau120_replenish_only_replQ90_decelQ50` | `{'hold_4h': 31, 'exit_early': 344}` | -2048.2/-1376.5 | N=375 sum=-4367.5 med=10.7 T3R=-6250.2 max_loss=-507.2 tail<-100=90 tail<-200=62 | N=375 sum=530.3 med=-3.0 T3R=-444.5 max_loss=-115.8 tail<-100=2 tail<-200=0 | 4897.8 | 5805.7 | 391.4 |
| 7 | `tau120_replenish_only_replQ90_decelQ75` | `{'hold_4h': 31, 'exit_early': 344}` | -2048.2/-1376.5 | N=375 sum=-4367.5 med=10.7 T3R=-6250.2 max_loss=-507.2 tail<-100=90 tail<-200=62 | N=375 sum=530.3 med=-3.0 T3R=-444.5 max_loss=-115.8 tail<-100=2 tail<-200=0 | 4897.8 | 5805.7 | 391.4 |
| 8 | `tau60_dual_or_replQ50_decelQ75` | `{'hold_4h': 241, 'exit_early': 134}` | -736.3/-655.5 | N=375 sum=-4367.5 med=10.7 T3R=-6250.2 max_loss=-507.2 tail<-100=90 tail<-200=62 | N=375 sum=983.9 med=-3.0 T3R=-898.8 max_loss=-507.2 tail<-100=53 tail<-200=32 | 5351.4 | 5351.4 | 0.0 |
| 9 | `tau90_dual_or_replQ75_decelQ50` | `{'hold_4h': 228, 'exit_early': 147}` | -3609.5/-2978.1 | N=375 sum=-4367.5 med=10.7 T3R=-6250.2 max_loss=-507.2 tail<-100=90 tail<-200=62 | N=375 sum=958.7 med=-6.5 T3R=-923.9 max_loss=-507.2 tail<-100=46 tail<-200=29 | 5326.2 | 5326.3 | 0.0 |
| 10 | `tau90_dual_or_replQ90_decelQ75` | `{'hold_4h': 159, 'exit_early': 216}` | -3079.3/-2428.4 | N=375 sum=-4367.5 med=10.7 T3R=-6250.2 max_loss=-507.2 tail<-100=90 tail<-200=62 | N=375 sum=817.3 med=-6.7 T3R=-1065.3 max_loss=-507.2 tail<-100=34 tail<-200=22 | 5184.8 | 5184.9 | 0.0 |
| 11 | `tau120_dual_or_replQ75_decelQ50` | `{'hold_4h': 235, 'exit_early': 140}` | -2616.4/-1984.8 | N=375 sum=-4367.5 med=10.7 T3R=-6250.2 max_loss=-507.2 tail<-100=90 tail<-200=62 | N=375 sum=679.0 med=-4.6 T3R=-1203.6 max_loss=-507.2 tail<-100=45 tail<-200=27 | 5046.5 | 5046.6 | 0.0 |
| 12 | `tau60_dual_or_replQ75_decelQ75` | `{'hold_4h': 180, 'exit_early': 195}` | -1386.2/-773.4 | N=375 sum=-4367.5 med=10.7 T3R=-6250.2 max_loss=-507.2 tail<-100=90 tail<-200=62 | N=375 sum=495.9 med=-6.1 T3R=-1386.8 max_loss=-507.2 tail<-100=39 tail<-200=23 | 4863.4 | 4863.4 | 0.0 |
| 13 | `tau60_dual_and_replQ50_decelQ50` | `{'hold_4h': 77, 'exit_early': 298}` | -2306.0/-1655.0 | N=375 sum=-4367.5 med=10.7 T3R=-6250.2 max_loss=-507.2 tail<-100=90 tail<-200=62 | N=375 sum=466.0 med=-6.1 T3R=-1387.3 max_loss=-507.2 tail<-100=14 tail<-200=7 | 4833.5 | 4862.9 | 0.0 |
| 14 | `tau60_replenish_only_replQ90_decelQ50` | `{'hold_4h': 30, 'exit_early': 345}` | -2250.2/-1578.5 | N=375 sum=-4367.5 med=10.7 T3R=-6250.2 max_loss=-507.2 tail<-100=90 tail<-200=62 | N=375 sum=-374.3 med=-4.6 T3R=-1405.7 max_loss=-397.0 tail<-100=4 tail<-200=2 | 3993.2 | 4844.5 | 110.2 |
| 15 | `tau60_replenish_only_replQ90_decelQ75` | `{'hold_4h': 30, 'exit_early': 345}` | -2250.2/-1578.5 | N=375 sum=-4367.5 med=10.7 T3R=-6250.2 max_loss=-507.2 tail<-100=90 tail<-200=62 | N=375 sum=-374.3 med=-4.6 T3R=-1405.7 max_loss=-397.0 tail<-100=4 tail<-200=2 | 3993.2 | 4844.5 | 110.2 |
| 16 | `tau180_dual_or_replQ90_decelQ50` | `{'hold_4h': 192, 'exit_early': 181}` | -2989.0/-2338.0 | N=373 sum=-3560.1 med=10.7 T3R=-5442.7 max_loss=-507.2 tail<-100=88 tail<-200=60 | N=373 sum=1281.3 med=-6.1 T3R=-601.4 max_loss=-507.2 tail<-100=33 tail<-200=21 | 4841.4 | 4841.3 | 0.0 |
| 17 | `tau60_dual_or_replQ75_decelQ50` | `{'hold_4h': 238, 'exit_early': 137}` | -1853.9/-1241.1 | N=375 sum=-4367.5 med=10.7 T3R=-6250.2 max_loss=-507.2 tail<-100=90 tail<-200=62 | N=375 sum=367.7 med=-7.3 T3R=-1515.0 max_loss=-507.2 tail<-100=49 tail<-200=31 | 4735.2 | 4735.2 | 0.0 |
| 18 | `tau90_dual_or_replQ75_decelQ75` | `{'hold_4h': 184, 'exit_early': 191}` | -2864.8/-2233.4 | N=375 sum=-4367.5 med=10.7 T3R=-6250.2 max_loss=-507.2 tail<-100=90 tail<-200=62 | N=375 sum=117.9 med=-7.0 T3R=-1764.8 max_loss=-507.2 tail<-100=42 tail<-200=26 | 4485.4 | 4485.4 | 0.0 |
| 19 | `tau90_replenish_only_replQ90_decelQ50` | `{'hold_4h': 21, 'exit_early': 354}` | -3113.3/-2349.4 | N=375 sum=-4367.5 med=10.7 T3R=-6250.2 max_loss=-507.2 tail<-100=90 tail<-200=62 | N=375 sum=-923.0 med=-3.3 T3R=-1823.2 max_loss=-397.0 tail<-100=4 tail<-200=2 | 3444.5 | 4427.0 | 110.2 |
| 20 | `tau90_replenish_only_replQ90_decelQ75` | `{'hold_4h': 21, 'exit_early': 354}` | -3113.3/-2349.4 | N=375 sum=-4367.5 med=10.7 T3R=-6250.2 max_loss=-507.2 tail<-100=90 tail<-200=62 | N=375 sum=-923.0 med=-3.3 T3R=-1823.2 max_loss=-397.0 tail<-100=4 tail<-200=2 | 3444.5 | 4427.0 | 110.2 |
| 21 | `tau180_replenish_only_replQ90_decelQ50` | `{'hold_4h': 23, 'exit_early': 350}` | -2774.8/-2103.9 | N=373 sum=-3560.1 med=10.7 T3R=-5442.7 max_loss=-507.2 tail<-100=88 tail<-200=60 | N=373 sum=-37.3 med=-1.7 T3R=-1068.7 max_loss=-397.0 tail<-100=2 tail<-200=1 | 3522.8 | 4374.0 | 110.2 |
| 22 | `tau180_replenish_only_replQ90_decelQ75` | `{'hold_4h': 23, 'exit_early': 350}` | -2774.8/-2103.9 | N=373 sum=-3560.1 med=10.7 T3R=-5442.7 max_loss=-507.2 tail<-100=88 tail<-200=60 | N=373 sum=-37.3 med=-1.7 T3R=-1068.7 max_loss=-397.0 tail<-100=2 tail<-200=1 | 3522.8 | 4374.0 | 110.2 |
| 23 | `tau90_replenish_only_replQ75_decelQ50` | `{'hold_4h': 63, 'exit_early': 312}` | -2630.2/-1989.8 | N=375 sum=-4367.5 med=10.7 T3R=-6250.2 max_loss=-507.2 tail<-100=90 tail<-200=62 | N=375 sum=-562.8 med=-3.3 T3R=-1882.7 max_loss=-507.2 tail<-100=14 tail<-200=7 | 3804.7 | 4367.5 | 0.0 |
| 24 | `tau90_replenish_only_replQ75_decelQ75` | `{'hold_4h': 63, 'exit_early': 312}` | -2630.2/-1989.8 | N=375 sum=-4367.5 med=10.7 T3R=-6250.2 max_loss=-507.2 tail<-100=90 tail<-200=62 | N=375 sum=-562.8 med=-3.3 T3R=-1882.7 max_loss=-507.2 tail<-100=14 tail<-200=7 | 3804.7 | 4367.5 | 0.0 |
| 25 | `tau120_dual_and_replQ90_decelQ50` | `{'hold_4h': 9, 'exit_early': 366}` | -2924.1/-2048.4 | N=375 sum=-4367.5 med=10.7 T3R=-6250.2 max_loss=-507.2 tail<-100=90 tail<-200=62 | N=375 sum=-1305.3 med=-3.4 T3R=-1950.6 max_loss=-115.8 tail<-100=2 tail<-200=0 | 3062.2 | 4299.6 | 391.4 |
| 26 | `tau120_dual_or_replQ75_decelQ75` | `{'hold_4h': 181, 'exit_early': 194}` | -1727.4/-1095.8 | N=375 sum=-4367.5 med=10.7 T3R=-6250.2 max_loss=-507.2 tail<-100=90 tail<-200=62 | N=375 sum=-135.5 med=-6.2 T3R=-2018.1 max_loss=-507.2 tail<-100=38 tail<-200=22 | 4232.0 | 4232.1 | 0.0 |
| 27 | `tau120_dual_and_replQ90_decelQ75` | `{'hold_4h': 4, 'exit_early': 371}` | -3006.5/-2017.0 | N=375 sum=-4367.5 med=10.7 T3R=-6250.2 max_loss=-507.2 tail<-100=90 tail<-200=62 | N=375 sum=-1873.0 med=-3.4 T3R=-2132.7 max_loss=-115.8 tail<-100=2 tail<-200=0 | 2494.5 | 4117.5 | 391.4 |
| 28 | `tau90_dual_and_replQ90_decelQ50` | `{'hold_4h': 4, 'exit_early': 371}` | -2972.9/-2106.4 | N=375 sum=-4367.5 med=10.7 T3R=-6250.2 max_loss=-507.2 tail<-100=90 tail<-200=62 | N=375 sum=-1971.4 med=-3.9 T3R=-2172.7 max_loss=-151.6 tail<-100=2 tail<-200=0 | 2396.1 | 4077.5 | 355.6 |
| 29 | `tau90_dual_and_replQ75_decelQ50` | `{'hold_4h': 27, 'exit_early': 348}` | -2619.8/-1892.8 | N=375 sum=-4367.5 med=10.7 T3R=-6250.2 max_loss=-507.2 tail<-100=90 tail<-200=62 | N=375 sum=-932.3 med=-4.5 T3R=-2184.8 max_loss=-507.2 tail<-100=6 tail<-200=1 | 3435.2 | 4065.4 | 0.0 |
| 30 | `tau90_dual_and_replQ90_decelQ75` | `{'hold_4h': 2, 'exit_early': 373}` | -3101.5/-2121.4 | N=375 sum=-4367.5 med=10.7 T3R=-6250.2 max_loss=-507.2 tail<-100=90 tail<-200=62 | N=375 sum=-2122.8 med=-4.0 T3R=-2262.5 max_loss=-151.6 tail<-100=2 tail<-200=0 | 2244.7 | 3987.7 | 355.6 |

## Read

- This is an actual management P&L backtest: rows either exit at tau using bid/ask or hold to 4h.
- Positive descriptive dissipation is not enough; the relevant number is holdout managed total/T3R vs baseline.
- Post-entry features remain illegal as entry inputs. Use only for management/shadow observation.
