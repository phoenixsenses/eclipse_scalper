# S34 State Navigation Overlay Tests

Generated: `2026-06-29T14:06:52.339442+00:00`

Status: `RESEARCH_ONLY_NO_LIVE_CHANGE`

Rows: `1204`

## 1. Label Coverage / Direction

| Label | N | Coverage | LONG 2h | LONG bracket | SHORT 20m | SHORT bracket |
| --- | ---: | ---: | --- | --- | --- | --- |
| `PANIC_CONTINUES` | 458 | 0.38 | N=458 sum=-5066.9 med=-6.1 T3R=-5898.4 tail150=34 maxLoss=-414.8 | N=458 sum=-9425.9 med=-15.3 T3R=-9650.9 tail150=0 maxLoss=-85.0 | N=458 sum=5684.4 med=4.3 T3R=4639.8 tail150=2 maxLoss=-234.6 | N=458 sum=4819.3 med=3.1 T3R=4234.3 tail150=0 maxLoss=-45.0 |
| `RECLAIM_CONFIRMED` | 115 | 0.096 | N=115 sum=1663.6 med=18.3 T3R=655.1 tail150=4 maxLoss=-354.9 | N=115 sum=1557.0 med=20.3 T3R=1332.0 tail150=0 maxLoss=-85.0 | N=115 sum=-2970.6 med=-30.3 T3R=-3232.5 tail150=1 maxLoss=-188.4 | N=115 sum=-1826.3 med=-45.0 T3R=-2411.3 tail150=0 maxLoss=-45.0 |
| `EXHAUSTION_PROXY` | 47 | 0.039 | N=47 sum=350.6 med=17.5 T3R=-365.6 tail150=3 maxLoss=-351.5 | N=47 sum=729.0 med=15.5 T3R=504.0 tail150=0 maxLoss=-85.0 | N=47 sum=-1180.1 med=-25.5 T3R=-1373.6 tail150=0 maxLoss=-126.4 | N=47 sum=-1069.1 med=-45.0 T3R=-1264.5 tail150=0 maxLoss=-45.0 |
| `CHAIN_BUILDING` | 318 | 0.264 | N=318 sum=-2045.6 med=-2.6 T3R=-2658.6 tail150=16 maxLoss=-405.8 | N=318 sum=-3241.7 med=-4.3 T3R=-3466.7 tail150=0 maxLoss=-85.0 | N=318 sum=575.7 med=-6.2 T3R=-326.0 tail150=1 maxLoss=-162.3 | N=318 sum=98.9 med=-7.7 T3R=-486.1 tail150=0 maxLoss=-45.0 |
| `CHAIN_COMPLETE` | 579 | 0.481 | N=579 sum=-616.5 med=10.4 T3R=-1668.8 tail150=44 maxLoss=-456.1 | N=579 sum=-1163.0 med=0.3 T3R=-1388.0 tail150=0 maxLoss=-85.0 | N=579 sum=-4196.2 med=-11.3 T3R=-4975.8 tail150=4 maxLoss=-243.0 | N=579 sum=-3246.0 med=-16.0 T3R=-3831.0 tail150=0 maxLoss=-45.0 |
| `NO_TRADE_HINDSIGHT_ZONE` | 109 | 0.091 | N=109 sum=-5207.5 med=-32.4 T3R=-5982.3 tail150=22 maxLoss=-406.7 | N=109 sum=-2954.6 med=-20.0 T3R=-3179.6 tail150=0 maxLoss=-85.0 | N=109 sum=2623.2 med=6.9 T3R=1578.7 tail150=0 maxLoss=-105.8 | N=109 sum=3030.0 med=7.4 T3R=2445.0 tail150=0 maxLoss=-45.0 |

## 2. v0.2 Conflict Resolver Policies

| Policy | Traded N | Skipped N | Traded long2h | Skipped counterfactual |
| --- | ---: | ---: | --- | --- |
| `baseline_all` | 31 | 0 | N=31 sum=697.6 med=12.0 T3R=52.6 tail150=2 maxLoss=-194.4 | N=0 sum=0.0 med=None T3R=0.0 tail150=None maxLoss=None |
| `allow_only_reclaim_or_exhaustion` | 8 | 23 | N=8 sum=408.9 med=25.4 T3R=59.1 tail150=0 maxLoss=-8.2 | N=23 sum=288.7 med=10.0 T3R=-234.2 tail150=2 maxLoss=-194.4 |
| `skip_panic` | 19 | 12 | N=19 sum=-39.8 med=4.7 T3R=-348.6 tail150=2 maxLoss=-194.4 | N=12 sum=737.4 med=25.8 T3R=173.5 tail150=0 maxLoss=-56.2 |
| `skip_panic_and_chain_building` | 19 | 12 | N=19 sum=-39.8 med=4.7 T3R=-348.6 tail150=2 maxLoss=-194.4 | N=12 sum=737.4 med=25.8 T3R=173.5 tail150=0 maxLoss=-56.2 |
| `skip_hindsight_zone` | 26 | 5 | N=26 sum=704.0 med=17.4 T3R=140.0 tail150=1 maxLoss=-154.2 | N=5 sum=-6.4 med=-12.6 T3R=-220.5 tail150=1 maxLoss=-194.4 |
| `allow_tail_low_bid_ok` | 11 | 20 | N=11 sum=959.5 med=50.2 T3R=314.4 tail150=0 maxLoss=-12.6 | N=20 sum=-261.9 med=0.0 T3R=-469.5 tail150=2 maxLoss=-194.4 |

## 3. v0.2 Label Breakdown

| Label | N | Long2h | Tighten TP120/SL40 | Exit after 60s |
| --- | ---: | --- | --- | --- |
| `PANIC_CONTINUES` | 12 | N=12 sum=737.4 med=25.8 T3R=173.5 tail150=0 maxLoss=-56.2 | N=12 sum=104.0 med=-22.9 T3R=-241.0 tail150=0 maxLoss=-45.0 | N=12 sum=-225.7 med=-15.9 T3R=-208.5 tail150=0 maxLoss=-40.3 |
| `RECLAIM_CONFIRMED` | 3 | N=3 sum=309.0 med=50.2 T3R=309.0 tail150=0 maxLoss=-8.2 | N=3 sum=25.0 med=-45.0 T3R=25.0 tail150=0 maxLoss=-45.0 | N=3 sum=8.3 med=-1.3 T3R=8.3 tail150=0 maxLoss=-12.6 |
| `EXHAUSTION_PROXY` | 5 | N=5 sum=99.9 med=24.2 T3R=16.4 tail150=0 maxLoss=0.9 | N=5 sum=166.8 med=26.8 T3R=-90.0 tail150=0 maxLoss=-45.0 | N=5 sum=-12.8 med=-2.6 T3R=-13.1 tail150=0 maxLoss=-8.9 |
| `CHAIN_BUILDING` | 0 | N=0 sum=0.0 med=None T3R=0.0 tail150=None maxLoss=None | N=0 sum=0.0 med=None T3R=0.0 tail150=None maxLoss=None | N=0 sum=0.0 med=None T3R=0.0 tail150=None maxLoss=None |
| `NO_TRADE_HINDSIGHT_ZONE` | 5 | N=5 sum=-6.4 med=-12.6 T3R=-220.5 tail150=1 maxLoss=-194.4 | N=5 sum=-33.9 med=-45.0 T3R=-90.0 tail150=0 maxLoss=-45.0 | N=5 sum=-8.7 med=-1.2 T3R=-14.6 tail150=0 maxLoss=-12.0 |

## 4. State Transition Matrix

| Transition | N | LONG 2h | LONG bracket | SHORT 20m | SHORT bracket |
| --- | ---: | --- | --- | --- | --- |
| `PANIC_CONTINUES->PANIC_CONTINUES` | 204 | N=204 sum=-1093.1 med=0.0 T3R=-1847.3 tail150=15 maxLoss=-396.3 | N=204 sum=-3721.3 med=-14.1 T3R=-3946.3 tail150=0 maxLoss=-85.0 | N=204 sum=2285.4 med=3.1 T3R=1342.2 tail150=0 maxLoss=-114.4 | N=204 sum=1854.2 med=1.7 T3R=1269.2 tail150=0 maxLoss=-45.0 |
| `CHAIN_COMPLETE->CHAIN_COMPLETE` | 84 | N=84 sum=908.3 med=17.3 T3R=76.7 tail150=7 maxLoss=-312.0 | N=84 sum=946.6 med=5.6 T3R=721.6 tail150=0 maxLoss=-85.0 | N=84 sum=-1689.4 med=-15.4 T3R=-1954.6 tail150=1 maxLoss=-243.0 | N=84 sum=-1546.2 med=-19.1 T3R=-1814.4 tail150=0 maxLoss=-45.0 |
| `PANIC_CONTINUES->CHAIN_COMPLETE` | 40 | N=40 sum=-98.6 med=1.8 T3R=-761.7 tail150=4 maxLoss=-278.0 | N=40 sum=-287.1 med=-5.6 T3R=-460.2 tail150=0 maxLoss=-85.0 | N=40 sum=-338.9 med=-5.5 T3R=-597.4 tail150=1 maxLoss=-243.0 | N=40 sum=-372.5 med=-7.4 T3R=-602.3 tail150=0 maxLoss=-45.0 |
| `CHAIN_BUILDING->CHAIN_BUILDING` | 59 | N=59 sum=222.9 med=4.4 T3R=-316.5 tail150=2 maxLoss=-326.3 | N=59 sum=-34.9 med=1.0 T3R=-259.9 tail150=0 maxLoss=-85.0 | N=59 sum=-540.0 med=-11.0 T3R=-738.3 tail150=0 maxLoss=-95.3 | N=59 sum=-610.2 med=-11.2 T3R=-796.1 tail150=0 maxLoss=-45.0 |
| `RECLAIM_CONFIRMED->RECLAIM_CONFIRMED` | 32 | N=32 sum=302.3 med=15.1 T3R=-516.3 tail150=0 maxLoss=-124.9 | N=32 sum=350.3 med=23.4 T3R=125.3 tail150=0 maxLoss=-85.0 | N=32 sum=-735.4 med=-30.6 T3R=-891.9 tail150=0 maxLoss=-108.0 | N=32 sum=-208.1 med=-45.0 T3R=-656.7 tail150=0 maxLoss=-45.0 |
| `PANIC_CONTINUES->CHAIN_BUILDING` | 31 | N=31 sum=-764.7 med=-21.2 T3R=-1103.0 tail150=2 maxLoss=-244.5 | N=31 sum=-401.6 med=-4.8 T3R=-580.2 tail150=0 maxLoss=-85.0 | N=31 sum=273.4 med=-5.2 T3R=-206.0 tail150=1 maxLoss=-162.3 | N=31 sum=166.9 med=-5.2 T3R=-191.6 tail150=0 maxLoss=-45.0 |
| `PANIC_CONTINUES->RECLAIM_CONFIRMED` | 16 | N=16 sum=863.1 med=51.5 T3R=281.2 tail150=0 maxLoss=-110.1 | N=16 sum=387.3 med=18.9 T3R=162.3 tail150=0 maxLoss=-85.0 | N=16 sum=-642.3 med=-31.6 T3R=-668.0 tail150=0 maxLoss=-126.4 | N=16 sum=-575.6 med=-45.0 T3R=-538.7 tail150=0 maxLoss=-45.0 |
| `OTHER->CHAIN_BUILDING` | 19 | N=19 sum=61.5 med=12.6 T3R=-159.4 tail150=1 maxLoss=-164.3 | N=19 sum=-283.0 med=-8.3 T3R=-401.9 tail150=0 maxLoss=-85.0 | N=19 sum=78.0 med=-1.7 T3R=-160.4 tail150=0 maxLoss=-80.8 | N=19 sum=61.1 med=-1.7 T3R=-175.2 tail150=0 maxLoss=-45.0 |
| `OTHER->PANIC_CONTINUES` | 15 | N=15 sum=-742.3 med=-7.3 T3R=-1099.2 tail150=2 maxLoss=-310.3 | N=15 sum=-328.9 med=-21.4 T3R=-415.7 tail150=0 maxLoss=-85.0 | N=15 sum=233.7 med=11.4 T3R=-42.7 tail150=0 maxLoss=-52.6 | N=15 sum=62.8 med=3.9 T3R=-112.8 tail150=0 maxLoss=-45.0 |
| `NO_TRADE_HINDSIGHT_ZONE->NO_TRADE_HINDSIGHT_ZONE` | 16 | N=16 sum=-218.6 med=4.6 T3R=-773.3 tail150=2 maxLoss=-405.8 | N=16 sum=-129.5 med=3.9 T3R=-250.4 tail150=0 maxLoss=-85.0 | N=16 sum=92.7 med=-13.8 T3R=-245.9 tail150=0 maxLoss=-58.5 | N=16 sum=125.8 med=-13.9 T3R=-263.6 tail150=0 maxLoss=-45.0 |
| `PANIC_CONTINUES->EXHAUSTION_PROXY` | 14 | N=14 sum=218.6 med=21.0 T3R=8.1 tail150=0 maxLoss=-69.9 | N=14 sum=56.8 med=15.5 T3R=-104.3 tail150=0 maxLoss=-85.0 | N=14 sum=-185.4 med=-26.3 T3R=-362.3 tail150=0 maxLoss=-103.7 | N=14 sum=-215.0 med=-41.1 T3R=-379.0 tail150=0 maxLoss=-45.0 |
| `OTHER->CHAIN_COMPLETE` | 15 | N=15 sum=-421.9 med=-32.1 T3R=-897.6 tail150=2 maxLoss=-456.1 | N=15 sum=44.5 med=5.5 T3R=-103.0 tail150=0 maxLoss=-85.0 | N=15 sum=-181.0 med=-15.3 T3R=-309.8 tail150=0 maxLoss=-89.7 | N=15 sum=-184.6 med=-17.1 T3R=-312.9 tail150=0 maxLoss=-45.0 |
| `PANIC_CONTINUES->NO_TRADE_HINDSIGHT_ZONE` | 9 | N=9 sum=-482.6 med=-47.7 T3R=-550.2 tail150=1 maxLoss=-194.4 | N=9 sum=-338.6 med=-24.3 T3R=-331.0 tail150=0 maxLoss=-85.0 | N=9 sum=167.2 med=9.2 T3R=-95.2 tail150=0 maxLoss=-95.1 | N=9 sum=174.6 med=9.3 T3R=-84.1 tail150=0 maxLoss=-45.0 |
| `NO_TRADE_HINDSIGHT_ZONE->PANIC_CONTINUES` | 7 | N=7 sum=-933.0 med=-58.0 T3R=-870.8 tail150=2 maxLoss=-414.8 | N=7 sum=-472.6 med=-85.0 T3R=-340.0 tail150=0 maxLoss=-85.0 | N=7 sum=490.0 med=57.6 T3R=62.9 tail150=0 maxLoss=-22.6 | N=7 sum=518.3 med=58.8 T3R=66.4 tail150=0 maxLoss=-20.0 |
| `CHAIN_BUILDING->PANIC_CONTINUES` | 12 | N=12 sum=-142.7 med=-18.3 T3R=-421.3 tail150=1 maxLoss=-200.7 | N=12 sum=-215.2 med=-9.9 T3R=-325.9 tail150=0 maxLoss=-85.0 | N=12 sum=219.0 med=-0.5 T3R=-111.2 tail150=0 maxLoss=-100.2 | N=12 sum=222.2 med=-0.1 T3R=-66.6 tail150=0 maxLoss=-45.0 |
| `CHAIN_BUILDING->CHAIN_COMPLETE` | 19 | N=19 sum=291.2 med=24.7 T3R=-120.3 tail150=0 maxLoss=-125.2 | N=19 sum=30.1 med=4.3 T3R=-86.7 tail150=0 maxLoss=-42.8 | N=19 sum=-217.2 med=-14.2 T3R=-307.8 tail150=0 maxLoss=-60.9 | N=19 sum=-210.7 med=-14.3 T3R=-300.8 tail150=0 maxLoss=-45.0 |
| `OTHER->RECLAIM_CONFIRMED` | 6 | N=6 sum=-36.7 med=-15.2 T3R=-100.4 tail150=0 maxLoss=-44.7 | N=6 sum=121.1 med=18.5 T3R=-67.6 tail150=0 maxLoss=-56.2 | N=6 sum=-59.0 med=-6.9 T3R=-100.4 tail150=0 maxLoss=-48.8 | N=6 sum=-97.4 med=-26.6 T3R=-135.0 tail150=0 maxLoss=-45.0 |
| `CHAIN_BUILDING->EXHAUSTION_PROXY` | 5 | N=5 sum=253.6 med=43.4 T3R=45.3 tail150=0 maxLoss=12.7 | N=5 sum=-36.1 med=14.7 T3R=-95.0 tail150=0 maxLoss=-85.0 | N=5 sum=-48.8 med=-24.7 T3R=-64.1 tail150=0 maxLoss=-38.6 | N=5 sum=-48.3 med=-24.7 T3R=-64.2 tail150=0 maxLoss=-38.7 |
| `NO_TRADE_HINDSIGHT_ZONE->EXHAUSTION_PROXY` | 5 | N=5 sum=-608.8 med=-54.1 T3R=-502.5 tail150=2 maxLoss=-351.5 | N=5 sum=86.2 med=-7.0 T3R=-56.9 tail150=0 maxLoss=-48.2 | N=5 sum=-95.9 med=-3.0 T3R=-130.6 tail150=0 maxLoss=-68.2 | N=5 sum=-98.1 med=-45.0 T3R=-90.0 tail150=0 maxLoss=-45.0 |
| `CHAIN_COMPLETE->EXHAUSTION_PROXY` | 6 | N=6 sum=-198.8 med=-26.8 T3R=-244.8 tail150=0 maxLoss=-118.9 | N=6 sum=-22.5 med=-9.4 T3R=-58.1 tail150=0 maxLoss=-23.3 | N=6 sum=-40.3 med=-1.1 T3R=-65.9 tail150=0 maxLoss=-45.6 | N=6 sum=-37.1 med=-0.6 T3R=-65.3 tail150=0 maxLoss=-45.0 |

## 5. Conflict Focus

| Label | Yes N | Yes long2h | Yes tighten | No N | No long2h |
| --- | ---: | --- | --- | ---: | --- |
| `PANIC_CONTINUES` | 12 | N=12 sum=737.4 med=25.8 T3R=173.5 tail150=0 maxLoss=-56.2 | N=12 sum=104.0 med=-22.9 T3R=-241.0 tail150=0 maxLoss=-45.0 | 19 | N=19 sum=-39.8 med=4.7 T3R=-348.6 tail150=2 maxLoss=-194.4 |
| `RECLAIM_CONFIRMED` | 3 | N=3 sum=309.0 med=50.2 T3R=309.0 tail150=0 maxLoss=-8.2 | N=3 sum=25.0 med=-45.0 T3R=25.0 tail150=0 maxLoss=-45.0 | 28 | N=28 sum=388.6 med=11.8 T3R=-134.3 tail150=2 maxLoss=-194.4 |
| `EXHAUSTION_PROXY` | 5 | N=5 sum=99.9 med=24.2 T3R=16.4 tail150=0 maxLoss=0.9 | N=5 sum=166.8 med=26.8 T3R=-90.0 tail150=0 maxLoss=-45.0 | 26 | N=26 sum=597.7 med=10.8 T3R=-47.3 tail150=2 maxLoss=-194.4 |
| `CHAIN_BUILDING` | 0 | N=0 sum=0.0 med=None T3R=0.0 tail150=None maxLoss=None | N=0 sum=0.0 med=None T3R=0.0 tail150=None maxLoss=None | 31 | N=31 sum=697.6 med=12.0 T3R=52.6 tail150=2 maxLoss=-194.4 |
| `CHAIN_COMPLETE` | 31 | N=31 sum=697.6 med=12.0 T3R=52.6 tail150=2 maxLoss=-194.4 | N=31 sum=23.0 med=-45.0 T3R=-322.0 tail150=0 maxLoss=-45.0 | 0 | N=0 sum=0.0 med=None T3R=0.0 tail150=None maxLoss=None |

## 6. Recommendations

- `PANIC_CONTINUES`: dashboard red-light / SHORT pressure / LONG caution; live_action=`notify_only`; evidence=`{'n': 458, 'sum_bps': 4819.3, 'mean_bps': 10.5, 'median_bps': 3.1, 'win_rate': 0.541, 'max_loss_bps': -45.0, 'tail_lte_minus100_n': 0, 'tail_lte_minus150_n': 0, 'tail_lte_minus300_n': 0, 't3r_bps': 4234.3}`
- `RECLAIM_CONFIRMED`: dashboard green-light for rebound state; live_action=`notify_only`; evidence=`{'n': 115, 'sum_bps': 1557.0, 'mean_bps': 13.5, 'median_bps': 20.3, 'win_rate': 0.626, 'max_loss_bps': -85.0, 'tail_lte_minus100_n': 0, 'tail_lte_minus150_n': 0, 'tail_lte_minus300_n': 0, 't3r_bps': 1332.0}`
- `EXHAUSTION_PROXY`: small-N rebound permission; keep shadow; live_action=`shadow_only`; evidence=`{'n': 47, 'sum_bps': 729.0, 'mean_bps': 15.5, 'median_bps': 15.5, 'win_rate': 0.638, 'max_loss_bps': -85.0, 'tail_lte_minus100_n': 0, 'tail_lte_minus150_n': 0, 'tail_lte_minus300_n': 0, 't3r_bps': 504.0}`
