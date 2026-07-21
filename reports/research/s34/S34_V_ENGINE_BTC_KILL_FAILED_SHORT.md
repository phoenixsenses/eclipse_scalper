# S34 V Engine BTC Kill + Failed SHORT

Generated: `2026-06-28T21:15:47.595724+00:00`

Protocol: `S34_V_ENGINE_V0_1_ETH_SELL_MAKER_LONG_H2_O20_V28_40_P4D`

Research-only. Early exits and SHORT shadows use book_ticker only; no mark fallback.

Baseline original LONG: N=19 sum=876.1 med=37.0 T3R=348.4

## Verdict

- BTC kill switch improves baseline: `NO`
- Failed-V SHORT has positive shadow expectancy: `NO`
- best kill variant: `btc_and_anchor_fail_T-20_M15` -> N=19 sum=804.5 med=37.0 T3R=276.8
- best SHORT shadow: `btc_and_anchor_fail_T-20_M15` -> N=1 sum=-79.4 med=-79.4 T3R=-79.4

## Best Kill Switch Variants

| Rank | Label | Trig | Triggered original | Not-triggered original | Kill switch | SHORT shadow |
| ---: | --- | ---: | --- | --- | --- | --- |
| 1 | `btc_and_anchor_fail_T-20_M15` | 1/19 (5.3%) | N=1 sum=-146.0 med=-146.0 T3R=-146.0 | N=18 sum=1022.1 med=39.4 T3R=494.4 | N=19 sum=804.5 med=37.0 T3R=276.8 | N=1 sum=-79.4 med=-79.4 T3R=-79.4 |
| 2 | `btc_and_candle_fail_T0_M5` | 2/19 (10.5%) | N=2 sum=-64.9 med=-32.5 T3R=-64.9 | N=17 sum=941.0 med=37.0 T3R=419.7 | N=19 sum=707.1 med=32.3 T3R=185.8 | N=2 sum=-182.8 med=-91.4 T3R=-182.8 |
| 3 | `btc_and_candle_fail_T-10_M5` | 2/19 (10.5%) | N=2 sum=-64.9 med=-32.5 T3R=-64.9 | N=17 sum=941.0 med=37.0 T3R=419.7 | N=19 sum=707.1 med=32.3 T3R=185.8 | N=2 sum=-182.8 med=-91.4 T3R=-182.8 |
| 4 | `btc_down_condition_T-20_M5` | 2/19 (10.5%) | N=2 sum=-64.9 med=-32.5 T3R=-64.9 | N=17 sum=941.0 med=37.0 T3R=419.7 | N=19 sum=707.1 med=32.3 T3R=185.8 | N=2 sum=-182.8 med=-91.4 T3R=-182.8 |
| 5 | `btc_and_anchor_fail_T-20_M5` | 2/19 (10.5%) | N=2 sum=-64.9 med=-32.5 T3R=-64.9 | N=17 sum=941.0 med=37.0 T3R=419.7 | N=19 sum=707.1 med=32.3 T3R=185.8 | N=2 sum=-182.8 med=-91.4 T3R=-182.8 |
| 6 | `btc_and_candle_fail_T-20_M5` | 2/19 (10.5%) | N=2 sum=-64.9 med=-32.5 T3R=-64.9 | N=17 sum=941.0 med=37.0 T3R=419.7 | N=19 sum=707.1 med=32.3 T3R=185.8 | N=2 sum=-182.8 med=-91.4 T3R=-182.8 |
| 7 | `btc_down_condition_T-20_M10` | 2/19 (10.5%) | N=2 sum=-64.9 med=-32.5 T3R=-64.9 | N=17 sum=941.0 med=37.0 T3R=419.7 | N=19 sum=693.2 med=32.3 T3R=171.9 | N=2 sum=-197.0 med=-98.5 T3R=-197.0 |
| 8 | `btc_and_anchor_fail_T-20_M10` | 2/19 (10.5%) | N=2 sum=-64.9 med=-32.5 T3R=-64.9 | N=17 sum=941.0 med=37.0 T3R=419.7 | N=19 sum=693.2 med=32.3 T3R=171.9 | N=2 sum=-197.0 med=-98.5 T3R=-197.0 |
| 9 | `btc_and_candle_fail_T-20_M10` | 2/19 (10.5%) | N=2 sum=-64.9 med=-32.5 T3R=-64.9 | N=17 sum=941.0 med=37.0 T3R=419.7 | N=19 sum=693.2 med=32.3 T3R=171.9 | N=2 sum=-197.0 med=-98.5 T3R=-197.0 |
| 10 | `btc_and_anchor_fail_T0_M10` | 3/19 (15.8%) | N=3 sum=-81.1 med=-16.2 T3R=-81.1 | N=16 sum=957.2 med=39.4 T3R=435.9 | N=19 sum=684.0 med=32.3 T3R=162.7 | N=3 sum=-212.5 med=-31.0 T3R=-212.5 |
| 11 | `btc_and_candle_fail_T0_M10` | 3/19 (15.8%) | N=3 sum=-81.1 med=-16.2 T3R=-81.1 | N=16 sum=957.2 med=39.4 T3R=435.9 | N=19 sum=684.0 med=32.3 T3R=162.7 | N=3 sum=-212.5 med=-31.0 T3R=-212.5 |
| 12 | `btc_down_condition_T-10_M10` | 3/19 (15.8%) | N=3 sum=-81.1 med=-16.2 T3R=-81.1 | N=16 sum=957.2 med=39.4 T3R=435.9 | N=19 sum=684.0 med=32.3 T3R=162.7 | N=3 sum=-212.5 med=-31.0 T3R=-212.5 |
| 13 | `btc_and_anchor_fail_T-10_M10` | 3/19 (15.8%) | N=3 sum=-81.1 med=-16.2 T3R=-81.1 | N=16 sum=957.2 med=39.4 T3R=435.9 | N=19 sum=684.0 med=32.3 T3R=162.7 | N=3 sum=-212.5 med=-31.0 T3R=-212.5 |
| 14 | `btc_and_candle_fail_T-10_M10` | 3/19 (15.8%) | N=3 sum=-81.1 med=-16.2 T3R=-81.1 | N=16 sum=957.2 med=39.4 T3R=435.9 | N=19 sum=684.0 med=32.3 T3R=162.7 | N=3 sum=-212.5 med=-31.0 T3R=-212.5 |
| 15 | `btc_and_anchor_fail_T0_M15` | 3/19 (15.8%) | N=3 sum=-81.1 med=-16.2 T3R=-81.1 | N=16 sum=957.2 med=39.4 T3R=435.9 | N=19 sum=656.2 med=32.3 T3R=134.9 | N=3 sum=-240.9 med=-79.4 T3R=-240.9 |
| 16 | `btc_and_anchor_fail_T-10_M15` | 3/19 (15.8%) | N=3 sum=-81.1 med=-16.2 T3R=-81.1 | N=16 sum=957.2 med=39.4 T3R=435.9 | N=19 sum=656.2 med=32.3 T3R=134.9 | N=3 sum=-240.9 med=-79.4 T3R=-240.9 |
| 17 | `btc_down_condition_T-20_M15` | 3/19 (15.8%) | N=3 sum=-89.4 med=27.9 T3R=-89.4 | N=16 sum=965.5 med=43.2 T3R=437.8 | N=19 sum=660.6 med=37.0 T3R=132.9 | N=3 sum=-236.2 med=-78.4 T3R=-236.2 |
| 18 | `btc_and_candle_fail_T-20_M15` | 3/19 (15.8%) | N=3 sum=-89.4 med=27.9 T3R=-89.4 | N=16 sum=965.5 med=43.2 T3R=437.8 | N=19 sum=660.6 med=37.0 T3R=132.9 | N=3 sum=-236.2 med=-78.4 T3R=-236.2 |
| 19 | `candle_bear_condition_T0_M5` | 4/19 (21.1%) | N=4 sum=-8.3 med=28.3 T3R=-146.0 | N=15 sum=884.4 med=41.7 T3R=363.1 | N=17 sum=650.5 med=37.0 T3R=129.2 | N=2 sum=-182.8 med=-91.4 T3R=-182.8 |
| 20 | `candle_bear_condition_T-10_M5` | 4/19 (21.1%) | N=4 sum=-8.3 med=28.3 T3R=-146.0 | N=15 sum=884.4 med=41.7 T3R=363.1 | N=17 sum=650.5 med=37.0 T3R=129.2 | N=2 sum=-182.8 med=-91.4 T3R=-182.8 |
| 21 | `candle_bear_condition_T-20_M5` | 4/19 (21.1%) | N=4 sum=-8.3 med=28.3 T3R=-146.0 | N=15 sum=884.4 med=41.7 T3R=363.1 | N=17 sum=650.5 med=37.0 T3R=129.2 | N=2 sum=-182.8 med=-91.4 T3R=-182.8 |
| 22 | `anchor_not_reclaimed_condition_T0_M30` | 4/19 (21.1%) | N=4 sum=-117.0 med=-26.0 T3R=-146.0 | N=15 sum=993.1 med=41.7 T3R=471.8 | N=19 sum=635.4 med=32.3 T3R=114.1 | N=4 sum=-268.7 med=-26.6 T3R=-210.4 |
| 23 | `candle_bear_condition_T0_M30` | 4/19 (21.1%) | N=4 sum=-117.0 med=-26.0 T3R=-146.0 | N=15 sum=993.1 med=41.7 T3R=471.8 | N=19 sum=635.4 med=32.3 T3R=114.1 | N=4 sum=-268.7 med=-26.6 T3R=-210.4 |
| 24 | `btc_and_anchor_fail_T0_M30` | 4/19 (21.1%) | N=4 sum=-117.0 med=-26.0 T3R=-146.0 | N=15 sum=993.1 med=41.7 T3R=471.8 | N=19 sum=635.4 med=32.3 T3R=114.1 | N=4 sum=-268.7 med=-26.6 T3R=-210.4 |
| 25 | `btc_and_candle_fail_T0_M30` | 4/19 (21.1%) | N=4 sum=-117.0 med=-26.0 T3R=-146.0 | N=15 sum=993.1 med=41.7 T3R=471.8 | N=19 sum=635.4 med=32.3 T3R=114.1 | N=4 sum=-268.7 med=-26.6 T3R=-210.4 |
| 26 | `anchor_not_reclaimed_condition_T-10_M30` | 4/19 (21.1%) | N=4 sum=-117.0 med=-26.0 T3R=-146.0 | N=15 sum=993.1 med=41.7 T3R=471.8 | N=19 sum=635.4 med=32.3 T3R=114.1 | N=4 sum=-268.7 med=-26.6 T3R=-210.4 |
| 27 | `candle_bear_condition_T-10_M30` | 4/19 (21.1%) | N=4 sum=-117.0 med=-26.0 T3R=-146.0 | N=15 sum=993.1 med=41.7 T3R=471.8 | N=19 sum=635.4 med=32.3 T3R=114.1 | N=4 sum=-268.7 med=-26.6 T3R=-210.4 |
| 28 | `any_failure_T-10_M30` | 4/19 (21.1%) | N=4 sum=-117.0 med=-26.0 T3R=-146.0 | N=15 sum=993.1 med=41.7 T3R=471.8 | N=19 sum=635.4 med=32.3 T3R=114.1 | N=4 sum=-268.7 med=-26.6 T3R=-210.4 |
| 29 | `anchor_not_reclaimed_condition_T-20_M30` | 4/19 (21.1%) | N=4 sum=-117.0 med=-26.0 T3R=-146.0 | N=15 sum=993.1 med=41.7 T3R=471.8 | N=19 sum=635.4 med=32.3 T3R=114.1 | N=4 sum=-268.7 med=-26.6 T3R=-210.4 |
| 30 | `candle_bear_condition_T-20_M30` | 4/19 (21.1%) | N=4 sum=-117.0 med=-26.0 T3R=-146.0 | N=15 sum=993.1 med=41.7 T3R=471.8 | N=19 sum=635.4 med=32.3 T3R=114.1 | N=4 sum=-268.7 med=-26.6 T3R=-210.4 |

## Best Failed-V SHORT Shadows

| Rank | Label | Trig | SHORT shadow | Triggered original |
| ---: | --- | ---: | --- | --- |
| 1 | `btc_and_anchor_fail_T-20_M15` | 1/19 (5.3%) | N=1 sum=-79.4 med=-79.4 T3R=-79.4 | N=1 sum=-146.0 med=-146.0 T3R=-146.0 |
| 2 | `btc_down_condition_T0_M10` | 6/19 (31.6%) | N=4 sum=-261.6 med=-40.0 T3R=-166.0 | N=6 sum=17.2 med=28.3 T3R=-134.3 |
| 3 | `btc_and_candle_fail_T0_M5` | 2/19 (10.5%) | N=2 sum=-182.8 med=-91.4 T3R=-182.8 | N=2 sum=-64.9 med=-32.5 T3R=-64.9 |
| 4 | `btc_and_candle_fail_T-10_M5` | 2/19 (10.5%) | N=2 sum=-182.8 med=-91.4 T3R=-182.8 | N=2 sum=-64.9 med=-32.5 T3R=-64.9 |
| 5 | `btc_down_condition_T-20_M5` | 2/19 (10.5%) | N=2 sum=-182.8 med=-91.4 T3R=-182.8 | N=2 sum=-64.9 med=-32.5 T3R=-64.9 |
| 6 | `btc_and_anchor_fail_T-20_M5` | 2/19 (10.5%) | N=2 sum=-182.8 med=-91.4 T3R=-182.8 | N=2 sum=-64.9 med=-32.5 T3R=-64.9 |
| 7 | `btc_and_candle_fail_T-20_M5` | 2/19 (10.5%) | N=2 sum=-182.8 med=-91.4 T3R=-182.8 | N=2 sum=-64.9 med=-32.5 T3R=-64.9 |
| 8 | `candle_bear_condition_T0_M5` | 4/19 (21.1%) | N=2 sum=-182.8 med=-91.4 T3R=-182.8 | N=4 sum=-8.3 med=28.3 T3R=-146.0 |
| 9 | `candle_bear_condition_T-10_M5` | 4/19 (21.1%) | N=2 sum=-182.8 med=-91.4 T3R=-182.8 | N=4 sum=-8.3 med=28.3 T3R=-146.0 |
| 10 | `candle_bear_condition_T-20_M5` | 4/19 (21.1%) | N=2 sum=-182.8 med=-91.4 T3R=-182.8 | N=4 sum=-8.3 med=28.3 T3R=-146.0 |
| 11 | `btc_down_condition_T-20_M10` | 2/19 (10.5%) | N=2 sum=-197.0 med=-98.5 T3R=-197.0 | N=2 sum=-64.9 med=-32.5 T3R=-64.9 |
| 12 | `btc_and_anchor_fail_T-20_M10` | 2/19 (10.5%) | N=2 sum=-197.0 med=-98.5 T3R=-197.0 | N=2 sum=-64.9 med=-32.5 T3R=-64.9 |
| 13 | `btc_and_candle_fail_T-20_M10` | 2/19 (10.5%) | N=2 sum=-197.0 med=-98.5 T3R=-197.0 | N=2 sum=-64.9 med=-32.5 T3R=-64.9 |
| 14 | `candle_bear_condition_T0_M10` | 4/19 (21.1%) | N=4 sum=-411.2 med=-98.5 T3R=-198.7 | N=4 sum=65.8 med=32.5 T3R=-146.0 |
| 15 | `candle_bear_condition_T-10_M10` | 4/19 (21.1%) | N=4 sum=-411.2 med=-98.5 T3R=-198.7 | N=4 sum=65.8 med=32.5 T3R=-146.0 |
| 16 | `candle_bear_condition_T-20_M10` | 4/19 (21.1%) | N=4 sum=-411.2 med=-98.5 T3R=-198.7 | N=4 sum=65.8 med=32.5 T3R=-146.0 |
| 17 | `anchor_not_reclaimed_condition_T0_M30` | 4/19 (21.1%) | N=4 sum=-268.7 med=-26.6 T3R=-210.4 | N=4 sum=-117.0 med=-26.0 T3R=-146.0 |
| 18 | `candle_bear_condition_T0_M30` | 4/19 (21.1%) | N=4 sum=-268.7 med=-26.6 T3R=-210.4 | N=4 sum=-117.0 med=-26.0 T3R=-146.0 |
| 19 | `btc_and_anchor_fail_T0_M30` | 4/19 (21.1%) | N=4 sum=-268.7 med=-26.6 T3R=-210.4 | N=4 sum=-117.0 med=-26.0 T3R=-146.0 |
| 20 | `btc_and_candle_fail_T0_M30` | 4/19 (21.1%) | N=4 sum=-268.7 med=-26.6 T3R=-210.4 | N=4 sum=-117.0 med=-26.0 T3R=-146.0 |

## Best Kill Trigger Cards

Best label: `btc_and_anchor_fail_T-20_M15`

| UTC | Original | Kill | SHORT | BTC ret | Ret15 | BTC context | Candle15 |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 2026-06-23T07:59:44.477000+00:00 | -146.0 | -217.6 | -79.4 | -110.5 | -211.9 | btc_down_continues | bear_followthrough |
