# S34 ETH Pre-Liquidation Executable Check

Generated: `2026-06-27T13:17:46.690445+00:00`

Research only. Future liquidation clusters are labels; entry uses only prior bookTicker state. Fills use real bid/ask bookTicker.

## Top 30 Rows

| Threshold | Pressure | TP/SL/BE/H | Events | Closed | Filtered | No-fill | Median | Mean | WR | Cum | T3R | Pos days | Avg hold | Pre-move | Imb | Adverse | Exits | Verdict |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1000K | lead5_lb10_down5 | TP20/SL20/BE10/H60 | 106 | 26 | 80 | 0 | +12.0 | +5.2 | 73% | +134 | +86 | 10/13 | 35s | +10.3 | -0.45 | +0.2 | TIME=9 TP=17 | candidate |
| 1000K | lead5_lb10_down5 | TP15/SL20/BE10/H60 | 106 | 26 | 80 | 0 | +6.9 | +2.0 | 73% | +52 | +14 | 9/13 | 32s | +10.3 | -0.45 | +0.2 | TIME=8 TP=18 | candidate |
| 500K | lead5_lb10_down5 | TP20/SL20/BE10/H60 | 222 | 41 | 181 | 0 | +11.6 | +5.6 | 76% | +230 | +176 | 16/19 | 34s | +10.2 | -0.45 | +0.5 | BE=1 TIME=13 TP=27 | watch_too_selective |
| 500K | lead5_lb10_down5 | TP15/SL20/BE10/H60 | 222 | 41 | 181 | 0 | +7.0 | +3.0 | 78% | +121 | +80 | 15/19 | 30s | +10.2 | -0.45 | +0.2 | TIME=12 TP=29 | watch_too_selective |
| 500K | lead5_lb10_down5 | TP10/SL20/BE10/H60 | 222 | 41 | 181 | 0 | +2.0 | -1.1 | 73% | -44 | -55 | 14/19 | 26s | +10.2 | -0.45 | +0.4 | TIME=9 TP=32 | reject_outlier_dependent |
| 1000K | lead5_lb10_down5 | TP10/SL20/BE10/H60 | 106 | 26 | 80 | 0 | +2.0 | -1.9 | 73% | -51 | -59 | 9/13 | 28s | +10.3 | -0.45 | +0.2 | TIME=6 TP=20 | reject_outlier_dependent |

## Best Per Threshold

| Threshold | Pressure | TP/SL/BE/H | Closed | Median | WR | T3R | Pos days | Verdict |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 500K | lead5_lb10_down5 | TP20/SL20/BE10/H60 | 41 | +11.6 | 76% | +176 | 16/19 | watch_too_selective |
| 1000K | lead5_lb10_down5 | TP20/SL20/BE10/H60 | 26 | +12.0 | 73% | +86 | 10/13 | candidate |

## Interpretation

- A `candidate` here is not deployable by itself because the sample is conditioned on future liquidation labels.
- A robust candidate would justify building a continuous pre-liq book-pressure detector and forward logging it.
- If rows are only thin or outlier-dependent, the idea stays research-only.
