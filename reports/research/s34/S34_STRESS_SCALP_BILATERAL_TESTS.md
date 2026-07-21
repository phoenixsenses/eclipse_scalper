# S34 Stress Scalp Bilateral Tests

Generated: `2026-06-29T13:56:08.630381+00:00`

Status: `RESEARCH_ONLY_NO_LIVE_CHANGE`

Primary state: `live_like_causal3`

## 1. Exact Mirror LONG / SHORT

### `original_holdstate_near3`

| Test | Summary | Exits |
| --- | --- | --- |
| `fixed_5m_LONG` | N=132 sum=-1691.5 med=-11.0 T3R=-1856.3 tail150=1 maxLoss=-154.3 | `{}` |
| `fixed_5m_SHORT` | N=132 sum=371.5 med=1.0 T3R=-12.9 tail150=0 maxLoss=-68.5 | `{}` |
| `fixed_15m_LONG` | N=132 sum=-3879.6 med=-22.2 T3R=-4192.6 tail150=10 maxLoss=-269.8 | `{}` |
| `fixed_15m_SHORT` | N=132 sum=2559.6 med=12.2 T3R=1809.4 tail150=0 maxLoss=-127.6 | `{}` |
| `fixed_20m_LONG` | N=132 sum=-3883.9 med=-16.1 T3R=-4152.0 tail150=13 maxLoss=-367.8 | `{}` |
| `fixed_20m_SHORT` | N=132 sum=2563.9 med=6.1 T3R=1516.4 tail150=0 maxLoss=-105.8 | `{}` |
| `fixed_30m_LONG` | N=132 sum=-3488.8 med=-6.9 T3R=-4020.9 tail150=13 maxLoss=-513.2 | `{}` |
| `fixed_30m_SHORT` | N=132 sum=2168.8 med=-3.1 T3R=748.4 tail150=4 maxLoss=-187.4 | `{}` |
| `LONG_TP40_SL200_20M` | N=132 sum=-4401.6 med=-15.4 T3R=-4506.6 tail150=16 maxLoss=-205.0 | `{'TP': 38, 'TIME': 84, 'SL': 10}` |
| `LONG_TP80_SL80_20M` | N=132 sum=-3153.3 med=-18.6 T3R=-3378.3 tail150=0 maxLoss=-85.0 | `{'TIME': 74, 'SL': 44, 'TP': 14}` |
| `LONG_TP200_SL40_20M` | N=132 sum=-2529.8 med=-45.0 T3R=-2786.0 tail150=0 maxLoss=-45.0 | `{'TIME': 54, 'SL': 78}` |
| `LONG_TP150_SL30_15M` | N=132 sum=-2118.5 med=-35.0 T3R=-2354.3 tail150=0 maxLoss=-35.0 | `{'TIME': 44, 'SL': 88}` |
| `SHORT_TP200_SL40_20M` | N=132 sum=3081.6 med=5.4 T3R=2496.6 tail150=0 maxLoss=-45.0 | `{'SL': 38, 'TIME': 84, 'TP': 10}` |

### `live_like_near3`

| Test | Summary | Exits |
| --- | --- | --- |
| `fixed_5m_LONG` | N=169 sum=-1227.5 med=-9.6 T3R=-1400.3 tail150=1 maxLoss=-154.3 | `{}` |
| `fixed_5m_SHORT` | N=169 sum=-462.5 med=-0.4 T3R=-732.1 tail150=0 maxLoss=-68.5 | `{}` |
| `fixed_15m_LONG` | N=169 sum=-1511.9 med=-3.1 T3R=-1846.4 tail150=6 maxLoss=-258.3 | `{}` |
| `fixed_15m_SHORT` | N=169 sum=-178.1 med=-6.9 T3R=-835.2 tail150=0 maxLoss=-127.6 | `{}` |
| `fixed_20m_LONG` | N=169 sum=-1995.6 med=-2.9 T3R=-2272.3 tail150=6 maxLoss=-350.0 | `{}` |
| `fixed_20m_SHORT` | N=169 sum=305.6 med=-7.1 T3R=-422.6 tail150=0 maxLoss=-105.8 | `{}` |
| `fixed_30m_LONG` | N=169 sum=-774.3 med=8.4 T3R=-1269.9 tail150=7 maxLoss=-476.7 | `{}` |
| `fixed_30m_SHORT` | N=169 sum=-915.7 med=-18.4 T3R=-1844.7 tail150=3 maxLoss=-187.4 | `{}` |
| `LONG_TP40_SL200_20M` | N=169 sum=-1975.7 med=1.7 T3R=-2080.7 tail150=6 maxLoss=-205.0 | `{'TIME': 109, 'TP': 56, 'SL': 4}` |
| `LONG_TP80_SL80_20M` | N=169 sum=-1603.0 med=-6.5 T3R=-1828.0 tail150=0 maxLoss=-85.0 | `{'TIME': 122, 'TP': 14, 'SL': 33}` |
| `LONG_TP200_SL40_20M` | N=169 sum=-1905.7 med=-19.2 T3R=-2179.5 tail150=0 maxLoss=-45.0 | `{'SL': 76, 'TIME': 93}` |
| `LONG_TP150_SL30_15M` | N=169 sum=-1322.3 med=-35.0 T3R=-1618.7 tail150=0 maxLoss=-35.0 | `{'SL': 87, 'TIME': 82}` |
| `SHORT_TP200_SL40_20M` | N=169 sum=285.7 med=-11.7 T3R=-299.3 tail150=0 maxLoss=-45.0 | `{'TIME': 109, 'SL': 56, 'TP': 4}` |

### `live_like_causal3`

| Test | Summary | Exits |
| --- | --- | --- |
| `fixed_5m_LONG` | N=97 sum=-523.5 med=-6.2 T3R=-689.8 tail150=1 maxLoss=-154.3 | `{}` |
| `fixed_5m_SHORT` | N=97 sum=-446.5 med=-3.8 T3R=-680.9 tail150=0 maxLoss=-67.9 | `{}` |
| `fixed_15m_LONG` | N=97 sum=-348.5 med=0.5 T3R=-616.2 tail150=3 maxLoss=-241.8 | `{}` |
| `fixed_15m_SHORT` | N=97 sum=-621.5 med=-10.5 T3R=-1162.9 tail150=0 maxLoss=-122.5 | `{}` |
| `fixed_20m_LONG` | N=97 sum=-445.5 med=-1.5 T3R=-707.4 tail150=3 maxLoss=-195.7 | `{}` |
| `fixed_20m_SHORT` | N=97 sum=-524.5 med=-8.5 T3R=-1026.4 tail150=0 maxLoss=-103.7 | `{}` |
| `fixed_30m_LONG` | N=97 sum=310.9 med=12.2 T3R=-177.4 tail150=2 maxLoss=-249.0 | `{}` |
| `fixed_30m_SHORT` | N=97 sum=-1280.9 med=-22.2 T3R=-1842.7 tail150=2 maxLoss=-187.4 | `{}` |
| `LONG_TP40_SL200_20M` | N=97 sum=-551.2 med=1.7 T3R=-656.2 tail150=3 maxLoss=-205.0 | `{'TP': 36, 'TIME': 60, 'SL': 1}` |
| `LONG_TP80_SL80_20M` | N=97 sum=-395.8 med=-2.9 T3R=-620.8 tail150=0 maxLoss=-85.0 | `{'TP': 9, 'TIME': 74, 'SL': 14}` |
| `LONG_TP200_SL40_20M` | N=97 sum=-616.1 med=-15.3 T3R=-877.9 tail150=0 maxLoss=-45.0 | `{'TIME': 60, 'SL': 37}` |
| `LONG_TP150_SL30_15M` | N=97 sum=-290.0 med=-16.3 T3R=-557.6 tail150=0 maxLoss=-35.0 | `{'TIME': 54, 'SL': 43}` |
| `SHORT_TP200_SL40_20M` | N=97 sum=-418.8 med=-11.7 T3R=-929.3 tail150=0 maxLoss=-45.0 | `{'SL': 36, 'TIME': 60, 'TP': 1}` |

### `live_like_causal2`

| Test | Summary | Exits |
| --- | --- | --- |
| `fixed_5m_LONG` | N=188 sum=-534.7 med=-3.0 T3R=-707.4 tail150=2 maxLoss=-220.6 | `{}` |
| `fixed_5m_SHORT` | N=188 sum=-1345.3 med=-7.0 T3R=-1754.8 tail150=0 maxLoss=-68.5 | `{}` |
| `fixed_15m_LONG` | N=188 sum=-462.6 med=5.5 T3R=-789.8 tail150=5 maxLoss=-258.3 | `{}` |
| `fixed_15m_SHORT` | N=188 sum=-1417.4 med=-15.5 T3R=-2070.9 tail150=0 maxLoss=-127.6 | `{}` |
| `fixed_20m_LONG` | N=188 sum=-763.4 med=0.5 T3R=-1070.3 tail150=5 maxLoss=-350.0 | `{}` |
| `fixed_20m_SHORT` | N=188 sum=-1116.6 med=-10.5 T3R=-1817.6 tail150=0 maxLoss=-127.3 | `{}` |
| `fixed_30m_LONG` | N=188 sum=-9.7 med=9.9 T3R=-505.3 tail150=5 maxLoss=-476.7 | `{}` |
| `fixed_30m_SHORT` | N=188 sum=-1870.3 med=-19.9 T3R=-2785.2 tail150=3 maxLoss=-187.4 | `{}` |
| `LONG_TP40_SL200_20M` | N=188 sum=-908.8 med=4.8 T3R=-1013.8 tail150=6 maxLoss=-205.0 | `{'TP': 69, 'TIME': 115, 'SL': 4}` |
| `LONG_TP80_SL80_20M` | N=188 sum=-343.5 med=0.1 T3R=-568.5 tail150=0 maxLoss=-85.0 | `{'TP': 18, 'TIME': 144, 'SL': 26}` |
| `LONG_TP200_SL40_20M` | N=188 sum=-745.3 med=-4.3 T3R=-1051.4 tail150=0 maxLoss=-45.0 | `{'TIME': 124, 'SL': 64}` |
| `LONG_TP150_SL30_15M` | N=188 sum=-271.3 med=-3.7 T3R=-567.7 tail150=0 maxLoss=-35.0 | `{'TIME': 115, 'SL': 73}` |
| `SHORT_TP200_SL40_20M` | N=188 sum=-971.2 med=-14.8 T3R=-1556.2 tail150=0 maxLoss=-45.0 | `{'SL': 69, 'TIME': 115, 'TP': 4}` |


## 2. Two-Phase Causal LONG -> Later SHORT

### `phaseA_LONG_20m_then_phaseB_SHORT_delay60s`
- `phaseA_LONG_TP80_SL80_20M`: N=97 sum=-395.8 med=-2.9 T3R=-620.8 tail150=0 maxLoss=-85.0; exits `{'TP': 9, 'TIME': 74, 'SL': 14}`
- `phaseB_SHORT_TP200_SL40_20M`: N=97 sum=-464.8 med=-4.9 T3R=-921.2 tail150=0 maxLoss=-45.0; exits `{'SL': 32, 'TIME': 64, 'TP': 1}`
- `phaseB_SHORT_fixed15m`: N=97 sum=-732.5 med=-9.6 T3R=-1139.7 tail150=0 maxLoss=-124.3; exits `{}`
- `phaseB_LONG_fixed15m`: N=97 sum=-237.5 med=-0.4 T3R=-506.6 tail150=1 maxLoss=-162.6; exits `{}`
### `phaseA_LONG_20m_then_phaseB_SHORT_delay180s`
- `phaseA_LONG_TP80_SL80_20M`: N=97 sum=-395.8 med=-2.9 T3R=-620.8 tail150=0 maxLoss=-85.0; exits `{'TP': 9, 'TIME': 74, 'SL': 14}`
- `phaseB_SHORT_TP200_SL40_20M`: N=97 sum=-998.3 med=-20.2 T3R=-1366.0 tail150=0 maxLoss=-45.0; exits `{'SL': 34, 'TIME': 63}`
- `phaseB_SHORT_fixed15m`: N=97 sum=-621.5 med=-13.6 T3R=-994.1 tail150=0 maxLoss=-80.0; exits `{}`
- `phaseB_LONG_fixed15m`: N=97 sum=-348.5 med=3.6 T3R=-547.3 tail150=0 maxLoss=-136.8; exits `{}`
### `phaseA_LONG_20m_then_phaseB_SHORT_delay300s`
- `phaseA_LONG_TP80_SL80_20M`: N=97 sum=-395.8 med=-2.9 T3R=-620.8 tail150=0 maxLoss=-85.0; exits `{'TP': 9, 'TIME': 74, 'SL': 14}`
- `phaseB_SHORT_TP200_SL40_20M`: N=97 sum=-1218.3 med=-20.7 T3R=-1628.7 tail150=0 maxLoss=-45.0; exits `{'SL': 35, 'TIME': 62}`
- `phaseB_SHORT_fixed15m`: N=97 sum=-565.5 med=-10.4 T3R=-908.0 tail150=0 maxLoss=-77.1; exits `{}`
- `phaseB_LONG_fixed15m`: N=97 sum=-404.5 med=0.4 T3R=-581.4 tail150=0 maxLoss=-137.7; exits `{}`
### `phaseA_LONG_20m_then_phaseB_SHORT_delay600s`
- `phaseA_LONG_TP80_SL80_20M`: N=97 sum=-395.8 med=-2.9 T3R=-620.8 tail150=0 maxLoss=-85.0; exits `{'TP': 9, 'TIME': 74, 'SL': 14}`
- `phaseB_SHORT_TP200_SL40_20M`: N=97 sum=-1312.7 med=-18.7 T3R=-1572.3 tail150=0 maxLoss=-45.0; exits `{'SL': 31, 'TIME': 66}`
- `phaseB_SHORT_fixed15m`: N=97 sum=-1190.2 med=-7.3 T3R=-1504.0 tail150=0 maxLoss=-107.3; exits `{}`
- `phaseB_LONG_fixed15m`: N=97 sum=220.2 med=-2.7 T3R=-66.5 tail150=0 maxLoss=-120.2; exits `{}`
### `phaseA_LONG_20m_then_phaseB_SHORT_delay900s`
- `phaseA_LONG_TP80_SL80_20M`: N=97 sum=-395.8 med=-2.9 T3R=-620.8 tail150=0 maxLoss=-85.0; exits `{'TP': 9, 'TIME': 74, 'SL': 14}`
- `phaseB_SHORT_TP200_SL40_20M`: N=97 sum=-1052.2 med=-9.3 T3R=-1288.2 tail150=0 maxLoss=-45.0; exits `{'SL': 30, 'TIME': 67}`
- `phaseB_SHORT_fixed15m`: N=97 sum=-1131.6 med=-5.3 T3R=-1303.3 tail150=0 maxLoss=-142.7; exits `{}`
- `phaseB_LONG_fixed15m`: N=97 sum=161.6 med=-4.7 T3R=-191.4 tail150=0 maxLoss=-85.0; exits `{}`

## 3. Reverse Failure Anatomy

- `all`: `{'n': 97, 'short_mfe_med': 31.2, 'short_mae_med': -25.8, 'short_mfe_sec_med': 360.0, 'short_mae_sec_med': 819.0, 'short_sl40_hit_n': 36, 'short_sl40_sec_med': 564.5, 'long_mfe_med': 25.8, 'long_tp40_hit_n': 36, 'long_tp40_sec_med': 564.5}`
- `short_losers`: `{'n': 22, 'short_mfe_med': 8.4, 'short_mae_med': -73.9, 'short_mfe_sec_med': 20.5, 'short_mae_sec_med': 1014.0, 'short_sl40_hit_n': 22, 'short_sl40_sec_med': 496.0, 'long_mfe_med': 73.9, 'long_tp40_hit_n': 22, 'long_tp40_sec_med': 496.0}`
- `short_winners`: `{'n': 44, 'short_mfe_med': 47.0, 'short_mae_med': -16.1, 'short_mfe_sec_med': 759.5, 'short_mae_sec_med': 235.0, 'short_sl40_hit_n': 4, 'short_sl40_sec_med': 303.0, 'long_mfe_med': 16.1, 'long_tp40_hit_n': 4, 'long_tp40_sec_med': 303.0}`
- `short_final`: `{'n': 97, 'sum_bps': -533.0, 'mean_bps': -5.5, 'median_bps': -9.4, 'win_rate': 0.454, 'max_loss_bps': -103.7, 'tail_lte_minus100_n': 1, 'tail_lte_minus150_n': 0, 'tail_lte_minus300_n': 0, 't3r_bps': -1033.9}`

## 4. Conflict Inversion

Conflict N: `11`
- `SHORT_TP200_SL40_20M`: N=11 sum=-260.7 med=-45.0 T3R=-295.4 tail150=0 maxLoss=-45.0; exits `{'SL': 6, 'TIME': 5}`
- `LONG_TP200_SL40_20M`: N=11 sum=89.7 med=-2.9 T3R=-141.2 tail150=0 maxLoss=-45.0; exits `{'TIME': 7, 'SL': 4}`
- `LONG_fixed2h`: N=11 sum=246.0 med=11.5 T3R=-226.3 tail150=1 maxLoss=-154.2; exits `{}`
- `SHORT_fixed20m`: N=11 sum=-389.2 med=-33.0 T3R=-423.0 tail150=0 maxLoss=-103.7; exits `{}`

## 5. Causal vs Near Spread

### `causal3_only`
- `SHORT_TP200_SL40_20M`: N=74 sum=-472.8 med=-14.3 T3R=-850.2 tail150=0 maxLoss=-45.0; exits `{'SL': 26, 'TIME': 47, 'TP': 1}`
- `LONG_TP80_SL80_20M`: N=74 sum=-203.8 med=-6.1 T3R=-428.8 tail150=0 maxLoss=-85.0; exits `{'TP': 5, 'TIME': 62, 'SL': 7}`
- `LONG_fixed15m`: N=74 sum=151.5 med=3.9 T3R=-116.2 tail150=1 maxLoss=-241.8; exits `{}`
- `SHORT_fixed15m`: N=74 sum=-891.5 med=-13.9 T3R=-1298.0 tail150=0 maxLoss=-122.5; exits `{}`
### `near3_only`
- `SHORT_TP200_SL40_20M`: N=109 sum=3027.5 med=7.8 T3R=2442.5 tail150=0 maxLoss=-45.0; exits `{'SL': 28, 'TIME': 71, 'TP': 10}`
- `LONG_TP80_SL80_20M`: N=109 sum=-2961.4 med=-19.3 T3R=-3186.4 tail150=0 maxLoss=-85.0; exits `{'TIME': 62, 'SL': 37, 'TP': 10}`
- `LONG_fixed15m`: N=109 sum=-3379.5 med=-24.3 T3R=-3692.6 tail150=8 maxLoss=-269.8; exits `{}`
- `SHORT_fixed15m`: N=109 sum=2289.5 med=14.3 T3R=1539.3 tail150=0 maxLoss=-127.6; exits `{}`
### `both_causal3_and_near3`
- `SHORT_TP200_SL40_20M`: N=23 sum=54.1 med=-9.5 T3R=-343.4 tail150=0 maxLoss=-45.0; exits `{'SL': 10, 'TIME': 13}`
- `LONG_TP80_SL80_20M`: N=23 sum=-191.9 med=-0.6 T3R=-416.9 tail150=0 maxLoss=-85.0; exits `{'TIME': 12, 'SL': 7, 'TP': 4}`
- `LONG_fixed15m`: N=23 sum=-500.0 med=-5.5 T3R=-647.5 tail150=2 maxLoss=-165.6; exits `{}`
- `SHORT_fixed15m`: N=23 sum=270.0 med=-4.5 T3R=-138.8 tail150=0 maxLoss=-68.1; exits `{}`

## 6. Chain Direction

### `COUNTER_BUY_PRESENT`
- N `8`, symbols_avg `2.5`, sell_share_avg `0.613`
- `LONG_TP80_SL80_20M`: N=8 sum=76.9 med=16.2 T3R=-132.2 tail150=0 maxLoss=-85.0; exits `{'SL': 2, 'TIME': 4, 'TP': 2}`
- `SHORT_TP200_SL40_20M`: N=8 sum=-48.0 med=-30.2 T3R=-195.4 tail150=0 maxLoss=-45.0; exits `{'TIME': 4, 'SL': 4}`
- `LONG_fixed15m`: N=8 sum=51.0 med=2.8 T3R=-71.5 tail150=0 maxLoss=-34.4; exits `{}`
- `SHORT_fixed15m`: N=8 sum=-131.0 med=-12.8 T3R=-178.1 tail150=0 maxLoss=-54.8; exits `{}`
### `SELL_DOMINANT`
- N `89`, symbols_avg `2.6`, sell_share_avg `0.963`
- `LONG_TP80_SL80_20M`: N=89 sum=-472.6 med=-6.5 T3R=-697.6 tail150=0 maxLoss=-85.0; exits `{'TP': 7, 'TIME': 70, 'SL': 12}`
- `SHORT_TP200_SL40_20M`: N=89 sum=-370.7 med=-9.5 T3R=-881.3 tail150=0 maxLoss=-45.0; exits `{'SL': 32, 'TIME': 56, 'TP': 1}`
- `LONG_fixed15m`: N=89 sum=-399.5 med=0.5 T3R=-667.1 tail150=3 maxLoss=-241.8; exits `{}`
- `SHORT_fixed15m`: N=89 sum=-490.5 med=-10.5 T3R=-1031.9 tail150=0 maxLoss=-122.5; exits `{}`
