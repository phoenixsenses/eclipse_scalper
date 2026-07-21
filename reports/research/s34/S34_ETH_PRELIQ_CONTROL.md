# S34 ETH Pre-Liq Book Pressure Control Test

Generated: `2026-06-27T13:25:05.038620+00:00`

Research only. Tests whether the pre-liq book-pressure pocket exists outside future liquidation labels.

Pressure: `lead5_lb10_down5`. Control timestamps exclude +/-900s around ETH SELL liq events.

## Results

| Sample | Route | Raw samples | Closed | Median | Mean | WR | Cum | T3R | Pos days | Avg hold | Pre-move | Imb | Exits |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| control | TP15_SL20_BE10_H60 | 335 | 335 | -8.6 | -9.5 | 13% | -3181 | -3204 | 2/61 | 54s | +7.3 | -0.11 | BE=13 SL=21 TIME=277 TP=24 |
| preliq_500K | TP15_SL20_BE10_H60 | 41 | 41 | +7.0 | +3.0 | 78% | +121 | +80 | 15/19 | 30s | +10.2 | -0.45 | TIME=12 TP=29 |
| preliq_1000K | TP15_SL20_BE10_H60 | 26 | 26 | +6.9 | +2.0 | 73% | +52 | +14 | 9/13 | 32s | +10.3 | -0.45 | TIME=8 TP=18 |
| control | TP20_SL20_BE10_H60 | 335 | 335 | -8.6 | -9.3 | 12% | -3125 | -3163 | 3/61 | 55s | +7.3 | -0.11 | BE=13 SL=21 TIME=285 TP=16 |
| preliq_500K | TP20_SL20_BE10_H60 | 41 | 41 | +11.6 | +5.6 | 76% | +230 | +176 | 16/19 | 34s | +10.2 | -0.45 | BE=1 TIME=13 TP=27 |
| preliq_1000K | TP20_SL20_BE10_H60 | 26 | 26 | +12.0 | +5.2 | 73% | +134 | +86 | 10/13 | 35s | +10.3 | -0.45 | TIME=9 TP=17 |

## Interpretation

- If control performance is similar to pre-liq, the edge may be a standalone book-pressure alpha.
- If control is much worse, the result depends on knowing a future liquidation cluster and is not directly tradable.
- This does not change runner/live rules.
