# S34 Sell-Liq Bounce Research

Generated: `2026-06-26T22:05:51.951815+00:00`

**Hypothesis**: LONG entry at SELL liq cascade threshold cross captures post-cascade reversal.
Data window: `2026-02-15T14:26:28+00:00` to `2026-06-26T20:24:37+00:00`

No runner/config/pre-reg changes. Research only.

## ETH_SELL_250K

| Variant | Sigs | NF% | Closed | Median | Cum | T3R | WR | H1 | H2 | Prelim |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| TP20/SL20/BE15 | 458 | 52% | 216 | -8.9 | -1805.3 | -1868.4 | 45% | -8.3 | -9.6 | yes |
| TP25/SL25/BE15 | 458 | 52% | 215 | -9.4 | -1985.7 | -2083.8 | 38% | -9.4 | -9.6 | yes |
| TP30/SL30/BE15 | 458 | 52% | 216 | -9.5 | -2054.9 | -2158.4 | 36% | -9.3 | -9.9 | yes |
| TP30/SL40/BE20 | 459 | 52% | 216 | -7.8 | -2016.1 | -2139.7 | 43% | -6.0 | -8.5 | yes |
| TP40/SL30/BE20 | 459 | 52% | 216 | -10.1 | -1899.4 | -2032.7 | 35% | -8.1 | -11.6 | yes |
| TP40/SL40/BE20 | 459 | 52% | 216 | -8.5 | -1791.3 | -1924.6 | 38% | -7.0 | -9.2 | yes |
| TP60/SL40/BE30 | 459 | 52% | 217 | -9.2 | -1782.1 | -1983.2 | 40% | -5.7 | -9.5 | yes |

## ETH_SELL_500K

| Variant | Sigs | NF% | Closed | Median | Cum | T3R | WR | H1 | H2 | Prelim |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| TP20/SL20/BE15 | 222 | 45% | 122 | -8.8 | -983.9 | -1052.5 | 43% | -0.3 | -11.6 | yes |
| TP25/SL25/BE15 | 222 | 45% | 122 | -8.7 | -879.3 | -959.9 | 41% | -7.2 | -9.9 | yes |
| TP30/SL30/BE15 | 222 | 45% | 122 | -9.4 | -1004.3 | -1091.1 | 35% | -7.5 | -11.0 | yes |
| TP30/SL40/BE20 | 222 | 45% | 122 | -7.3 | -1083.4 | -1189.5 | 46% | +6.2 | -9.9 | yes |
| TP40/SL30/BE20 | 222 | 45% | 122 | -12.1 | -1067.2 | -1194.8 | 34% | -7.5 | -14.3 | yes |
| TP40/SL40/BE20 | 222 | 45% | 122 | -9.2 | -1014.0 | -1142.9 | 39% | -4.6 | -12.8 | yes |
| TP60/SL40/BE30 | 222 | 45% | 122 | -10.4 | -1242.8 | -1436.9 | 39% | -0.5 | -33.0 | yes |

## ETH_SELL_1M

| Variant | Sigs | NF% | Closed | Median | Cum | T3R | WR | H1 | H2 | Prelim |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| TP20/SL20/BE15 | 114 | 32% | 77 | -7.3 | -607.3 | -673.2 | 48% | +11.5 | -22.3 |  |
| TP25/SL25/BE15 | 114 | 32% | 77 | -10.4 | -663.0 | -739.9 | 43% | +14.1 | -23.6 |  |
| TP30/SL30/BE15 | 114 | 32% | 76 | -11.0 | -814.4 | -894.3 | 34% | -6.6 | -13.8 |  |
| TP30/SL40/BE20 | 114 | 32% | 76 | -9.9 | -831.9 | -918.1 | 43% | +8.0 | -11.6 |  |
| TP40/SL30/BE20 | 114 | 32% | 77 | -12.2 | -832.0 | -946.4 | 31% | -10.0 | -26.4 |  |
| TP40/SL40/BE20 | 114 | 32% | 77 | -10.6 | -804.5 | -923.7 | 36% | -8.5 | -11.8 |  |
| TP60/SL40/BE30 | 114 | 32% | 77 | -14.5 | -889.2 | -1079.8 | 38% | -8.6 | -41.3 |  |

## Top 10 by Score

| Threshold | Variant | Median | T3R Cum | WR | H1 | H2 | Prelim |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| ETH_SELL_1M | TP20/SL20/BE15 | -7.3 | -673.2 | 48% | +11.5 | -22.3 |  |
| ETH_SELL_1M | TP25/SL25/BE15 | -10.4 | -739.9 | 43% | +14.1 | -23.6 |  |
| ETH_SELL_1M | TP30/SL40/BE20 | -9.9 | -918.1 | 43% | +8.0 | -11.6 |  |
| ETH_SELL_500K | TP25/SL25/BE15 | -8.7 | -959.9 | 41% | -7.2 | -9.9 | yes |
| ETH_SELL_500K | TP20/SL20/BE15 | -8.8 | -1052.5 | 43% | -0.3 | -11.6 | yes |
| ETH_SELL_1M | TP40/SL40/BE20 | -10.6 | -923.7 | 36% | -8.5 | -11.8 |  |
| ETH_SELL_1M | TP30/SL30/BE15 | -11.0 | -894.3 | 34% | -6.6 | -13.8 |  |
| ETH_SELL_500K | TP30/SL40/BE20 | -7.3 | -1189.5 | 46% | +6.2 | -9.9 | yes |
| ETH_SELL_500K | TP30/SL30/BE15 | -9.4 | -1091.1 | 35% | -7.5 | -11.0 | yes |
| ETH_SELL_500K | TP40/SL40/BE20 | -9.2 | -1142.9 | 39% | -4.6 | -12.8 | yes |

## Viable Combos

None found in this sweep.

## Interpretation Notes

- Bounce hypothesis: 61% post-cascade reversal at avg +33.8 bps (600s window, ETH 500K SELL N=222).
- Current SELL SHORT rules enter after cascade drops ~37-42 bps; TP requires 22-38 more bps. Runner shows median -12 to -38 bps.
- This sweep tests LONG direction on same SELL liq signal. Viable = median>0, T3R>0, H2>0, N>=30.
- If viable combos found: queue for runner-parity deep-dive before any pre-reg amendment.
- Research only. No runner/config changes.