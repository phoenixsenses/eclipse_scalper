# S34 Navigation Next Results

Generated: `2026-06-29T09:01:52.815130+00:00`

Status: `RESEARCH_ONLY_NO_LIVE_CHANGE`. Navigation tests only. No live order/config changes.

## Route-Specific Map

| Route | All 2h N | All Sum | TailLow N | TailLow Sum | TailLow T3R | TailLow <=150 | 4hTP Sum |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ETHUSDT_SELL_100000 | 674 | -5194.7 | 4 | -444.0 | -392.6 | 1 | -1391.7 |
| ETHUSDT_SELL_200000 | 450 | -605.8 | 11 | 961.6 | 316.6 | 0 | 3169.8 |
| ETHUSDT_SELL_50000 | 882 | -1286.5 | 8 | -34.9 | -448.4 | 1 | -899.9 |

## Bull Thin-Depth Anatomy

- N: `20`
- 2h: N=20 sum=1479.3 med=50.5 T3R=479.4 tails<=150=2
- TP300/SL150/4h: sum=1636.7 med=54.8 T3R=751.7

| Threshold bucket | N | Sum | Median | Tail<=150 | T3R |
| --- | ---: | ---: | ---: | ---: | ---: |
| thr100000 | 5 | 381.3 | 45.7 | 0 | -149.6 |
| thr200000 | 5 | 306.1 | 45.7 | 1 | -145.3 |
| thr50000 | 10 | 791.9 | 63.4 | 1 | -100.0 |

## KNN Navigation

- k: `20`
- prediction counts: `{'CLEAN': 178, 'DANGER': 1094, 'MIXED': 734}`

| Prediction | N | Sum | Median | Win | Tail<=150 | Max loss | T3R |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CLEAN | 178 | 2589.2 | 15.6 | 0.607 | 7 | -356.0 | 1756.4 |
| DANGER | 1094 | -10573.7 | -0.2 | 0.499 | 128 | -455.2 | -12588.4 |
| MIXED | 734 | 897.5 | 0.6 | 0.503 | 37 | -347.2 | -171.3 |

## BUY-Side Navigation

| Side/Threshold | N | Sum | Median | Win | Tail<=150 | Max loss | T3R |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fade_SHORT_thr100000 | 656 | -3754.2 | 4.2 | 0.523 | 52 | -552.2 | -4989.0 |
| fade_SHORT_thr200000 | 441 | -2452.7 | 6.0 | 0.535 | 45 | -524.4 | -3644.0 |
| fade_SHORT_thr50000 | 905 | -5384.6 | 6.3 | 0.527 | 69 | -556.9 | -6604.4 |
| fade_SHORT_thr500000 | 194 | 1810.2 | 28.6 | 0.577 | 19 | -390.5 | 618.0 |
| continuation_LONG_thr100000 | 656 | -2805.8 | -14.2 | 0.433 | 47 | -504.9 | -4233.9 |
| continuation_LONG_thr200000 | 441 | -1957.3 | -16.0 | 0.429 | 32 | -490.9 | -3264.1 |
| continuation_LONG_thr50000 | 905 | -3665.4 | -16.3 | 0.424 | 53 | -489.9 | -5098.2 |
| continuation_LONG_thr500000 | 194 | -3750.2 | -38.5 | 0.387 | 20 | -518.1 | -4857.3 |
