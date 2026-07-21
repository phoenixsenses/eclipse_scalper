# S34 Third Wave Research Suite

Generated: `2026-06-30T10:38:47.966381+00:00`  |  Status: `RESEARCH_ONLY_NO_LIVE_CHANGE`
Cal: 1404 (2026-02-15T18:32:18Z to 2026-06-08T01:05:38Z)
Hold: 602 (2026-06-08T01:24:48Z to 2026-06-29T08:28:10Z)

## A. Multi-Criteria Scoring (0-6 points)

Criteria: +1 each for silence30, n_prior2h>=3, btc4h<0, vdepth>=30, US session, sync_k>=200K

### By cumulative score threshold (score >= N)
| Score >= | Silence N cal | Silence T3R cal | Silence WR cal | Silence N hold | Silence T3R hold | Silence WR hold |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| score_gte_0 | 724 | 14738.2 | 0.623 | 194 | 7733.7 | 0.701 |
| score_gte_1 | 724 | 14738.2 | 0.623 | 194 | 7733.7 | 0.701 |
| score_gte_2 | 633 | 13531.5 | 0.633 | 171 | 8264.5 | 0.76 |
| score_gte_3 | 440 | 11513.1 | 0.684 | 124 | 7294.5 | 0.831 |
| score_gte_4 | 234 | 7719.3 | 0.722 | 73 | 4091.6 | 0.795 |
| score_gte_5 | 81 | 2608.3 | 0.741 | 24 | 1569.7 | 0.75 |
| score_gte_6 | 15 | 52.9 | 0.733 | 2 | None | 1.0 |

| Score >= | Short N cal | Short T3R cal | Short WR cal | Short N hold | Short T3R hold | Short WR hold |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| score_gte_0 | 605 | 7778.7 | 0.542 | 397 | 11360.3 | 0.549 |
| score_gte_1 | 554 | 8796.7 | 0.545 | 373 | 10934.1 | 0.558 |
| score_gte_2 | 440 | 7259.5 | 0.559 | 314 | 10574.7 | 0.554 |
| score_gte_3 | 248 | 5410.8 | 0.54 | 210 | 8301.0 | 0.581 |
| score_gte_4 | 85 | 966.0 | 0.494 | 107 | 6842.4 | 0.664 |
| score_gte_5 | 20 | 467.6 | 0.6 | 16 | 309.2 | 0.75 |
| score_gte_6 | 0 | None | None | 0 | None | None |

### By exact score value
| Score | Cal N | Cal T3R | Cal med | Cal win | Hold N | Hold T3R | Hold med | Hold win |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| score_0 | 64 | -622.3 | -16.9 | 0.359 | 27 | -1160.2 | -9.0 | 0.407 |
| score_1 | 225 | -3850.6 | -7.0 | 0.453 | 86 | -1807.5 | -19.9 | 0.337 |
| score_2 | 413 | -3074.8 | -8.5 | 0.46 | 151 | -2945.4 | -4.0 | 0.47 |
| score_3 | 381 | -2646.1 | 12.5 | 0.533 | 158 | 141.6 | 12.8 | 0.601 |
| score_4 | 220 | 3283.2 | 26.6 | 0.65 | 140 | -5968.0 | -5.2 | 0.486 |
| score_5 | 86 | 991.0 | 43.4 | 0.663 | 38 | -262.5 | -1.1 | 0.474 |
| score_6 | 15 | 52.9 | 13.0 | 0.733 | 2 | None | 241.4 | 1.0 |

### Perfect storm (score>=4 + silence) — permutation null
Cal: N=234 T3R=7719.3 med=35.5 win=0.722
Hold: N=73 T3R=4091.6 med=51.8 win=0.795
Perm cal: real=7719.3 null_p95=1450.2 p=0.0 -> **PASS**

### Perfect storm SHORT (score>=4 + noisy)
Cal: N=85 T3R=966.0 win=0.494
Hold: N=107 T3R=6842.4 win=0.664
Perm cal: real=966.0 p=0.001 -> **PASS**

## B. Propagation Mechanics

### B1: Cascade count in 30-min noisy window (SHORT direction)
| Count | Cal N | Cal T3R | Cal win | Hold N | Hold T3R | Hold win |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| prop_count_exactly_1 | 275 | -2362.4 | 0.462 | 95 | -1203.2 | 0.421 |
| prop_count_2_or_3 | 190 | 1050.2 | 0.532 | 112 | 2153.0 | 0.518 |
| prop_count_4plus | 140 | 7122.2 | 0.714 | 190 | 8736.3 | 0.632 |

### B2: Max propagation cascade size (SHORT direction)
| Max size | Cal N | Cal T3R | Cal win | Hold N | Hold T3R | Hold win |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| prop_max_50K_100K_SHORT | 144 | -98.4 | 0.507 | 82 | -1197.8 | 0.378 |
| prop_max_100K_200K_SHORT | 208 | 3908.5 | 0.582 | 81 | 1664.0 | 0.605 |
| prop_max_200K_plus_SHORT | 253 | 2159.4 | 0.53 | 234 | 8976.9 | 0.59 |

### B3: Timing of first propagation (SHORT direction)
| First cascade timing | Cal N | Cal T3R | Cal win | Hold N | Hold T3R | Hold win |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| prop_timing_first_0_5min_SHORT | 302 | 3126.2 | 0.546 | 224 | 4045.4 | 0.545 |
| prop_timing_first_5_15min_SHORT | 145 | 1491.9 | 0.538 | 95 | 3550.7 | 0.568 |
| prop_timing_first_15_30min_SHORT | 158 | 1149.1 | 0.538 | 78 | 1652.9 | 0.538 |

## C. BULL_PULLBACK + noisy: Long, Short, or Random?

| Condition | Cal N | Cal T3R | Cal med | Cal win | Hold N | Hold T3R | Hold med | Hold win |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cal:bull_noisy_LONG | 75 | -1740.2 | -7.0 | 0.467 | | | | |
| cal:bull_noisy_SHORT | 75 | -1097.3 | -3.0 | 0.493 | | | | |
| cal:bull_silence_LONG | 67 | 2006.1 | 35.5 | 0.627 | | | | |
| cal:noisy_not_bull_SHORT | 605 | 7778.7 | 9.1 | 0.542 | | | | |
| hold:bull_noisy_LONG | 11 | None | 9.9 | 0.909 | | | | |
| hold:bull_noisy_SHORT | 11 | None | -19.9 | 0.091 | | | | |
| hold:bull_silence_LONG | 11 | None | 72.3 | 0.909 | | | | |
| hold:noisy_not_bull_SHORT | 397 | 11360.3 | 10.5 | 0.549 | | | | |

Perm bull_noisy LONG  (cal): p=0.863 real=-1740.2 -> **ARTIFACT**
Perm bull_noisy SHORT (cal): p=0.279 real=-1097.3 -> **ARTIFACT**

## D. 200K Live Rule Subset Optimization

| Subset | Cal N | Cal T3R | Cal win | Hold N | Hold T3R | Hold win | Perm | Verdict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 200K_all | 286 | -155.9 | 0.552 | 164 | -2440.3 | 0.482 | 0.287 | **ARTIFACT** |
| 200K_silence | 134 | 2794.5 | 0.664 | 48 | 1553.4 | 0.688 | 0.0 | **PASS** |
| 200K_sil_cluster | 58 | 863.4 | 0.724 | 26 | 1156.8 | 0.769 | 0.012 | **PASS** |
| 200K_sil_US | 53 | 1050.5 | 0.679 | 15 | 161.0 | 0.667 | 0.003 | **PASS** |
| 200K_sil_btcbear | 90 | 2028.6 | 0.7 | 27 | 1030.5 | 0.741 | 0.0 | **PASS** |
| 200K_sil_sync200K | 52 | 984.8 | 0.692 | 22 | 952.5 | 0.773 | 0.007 | **PASS** |
| 200K_sil_vdepth30 | 50 | 1401.2 | 0.7 | 11 | None | 0.818 | 0.001 | **PASS** |
| 200K_sil_clust_bear | 46 | 912.2 | 0.783 | 17 | 690.7 | 0.824 | 0.007 | **PASS** |
| 200K_sil_clust_US | 29 | 678.1 | 0.793 | 11 | None | 0.727 | 0.003 | **PASS** |
| 200K_sil_score4 | 58 | 1608.2 | 0.741 | 17 | 597.1 | 0.765 | 0.001 | **PASS** |
| 200K_noisy_short | 140 | 883.4 | 0.507 | 113 | 1901.2 | 0.549 | 0.059 | **ARTIFACT** |
| 200K_noisy_short_us | 79 | -1168.4 | 0.443 | 59 | 1685.1 | 0.61 | 0.688 | **ARTIFACT** |

## E. ETH Prior 1h Context

| Condition | Cal N | Cal T3R | Cal win | Hold N | Hold T3R | Hold win |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| eth1h_bull_gt50_all | 158 | 462.2 | 0.487 | 45 | -343.3 | 0.578 |
| eth1h_bull_gt50_silence | 78 | 1451.0 | 0.538 | 20 | 467.9 | 0.6 |
| eth1h_bull_gt50_noisy_SHORT | 14 | 244.7 | 0.714 | 14 | 81.7 | 0.643 |
| eth1h_bull_0_50_all | 116 | 94.7 | 0.517 | 36 | -942.6 | 0.5 |
| eth1h_bull_0_50_silence | 53 | 597.1 | 0.585 | 16 | 34.5 | 0.75 |
| eth1h_bull_0_50_noisy_SHORT | 54 | -1591.4 | 0.444 | 20 | -77.2 | 0.65 |
| eth1h_flat_all | 82 | -392.0 | 0.585 | 32 | -425.2 | 0.344 |
| eth1h_flat_silence | 40 | -460.1 | 0.55 | 13 | 383.5 | 0.615 |
| eth1h_flat_noisy_SHORT | 42 | -1611.6 | 0.333 | 19 | 158.7 | 0.737 |
| eth1h_bear_all | 1129 | -3607.4 | 0.524 | 521 | -8415.6 | 0.48 |
| eth1h_bear_silence | 592 | 11173.3 | 0.639 | 158 | 6210.1 | 0.709 |
| eth1h_bear_noisy_SHORT | 537 | 7802.7 | 0.547 | 363 | 9802.2 | 0.54 |
| eth1h_bear_lt-50_all | 722 | -4253.8 | 0.536 | 308 | -5991.7 | 0.5 |
| eth1h_bear_lt-50_silence | 353 | 8467.9 | 0.674 | 85 | 2920.7 | 0.788 |
| eth1h_bear_lt-50_noisy_SHORT | 369 | 7518.6 | 0.558 | 223 | 5488.9 | 0.552 |
| eth1h_bear_lt-100_all | 330 | -1630.6 | 0.552 | 101 | -1608.2 | 0.475 |
| eth1h_bear_lt-100_silence | 160 | 4693.5 | 0.7 | 19 | 1027.6 | 0.789 |
| eth1h_bear_lt-100_noisy_SHORT | 170 | 3599.3 | 0.559 | 82 | 793.0 | 0.573 |

## F. High-Sync noisy SHORT Permutation Null

| Sync gate | Cal N | Cal T3R | Cal win | Hold N | Hold T3R | Hold win | Perm p | Verdict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| sync_gte_200K | 174 | 6002.6 | 0.569 | 196 | 9457.6 | 0.602 | 0.0 | **PASS** |
| sync_gte_300K | 111 | 3245.7 | 0.586 | 146 | 8040.5 | 0.63 | 0.013 | **PASS** |
| sync_gte_500K | 55 | 274.6 | 0.564 | 106 | 6058.7 | 0.632 | 0.355 | **ARTIFACT** |
| sync_gte_700K | 25 | -113.5 | 0.52 | 75 | 4555.7 | 0.667 | 0.354 | **ARTIFACT** |
| sync_gte_1000K | 16 | -559.9 | 0.438 | 58 | 3021.3 | 0.672 | 0.778 | **ARTIFACT** |
| sil+sync500K_cal | 38 | 134.7 | 0.605 | | | | | |
| sil+sync500K_hold | 19 | 279.5 | 0.737 | | | | | |

## G. bid_depth_usd Absolute Level

Cal bid_depth percentiles: p25=90146.5 p50=180474.1 p75=316921.1

| Condition | Cal N | Cal T3R | Cal win | Hold N | Hold T3R | Hold win |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| bid_q4_high_all | 67 | -468.9 | 0.463 | 66 | -1763.5 | 0.394 |
| bid_q4_high_silence | 30 | 150.5 | 0.6 | 22 | -84.1 | 0.591 |
| bid_q4_high_noisy_SHORT | 35 | -281.3 | 0.629 | 43 | 502.3 | 0.488 |
| bid_q3_all | 67 | -1239.0 | 0.478 | 109 | -3955.8 | 0.459 |
| bid_q3_silence | 30 | -120.2 | 0.567 | 33 | 1156.2 | 0.788 |
| bid_q3_noisy_SHORT | 33 | -245.0 | 0.606 | 74 | 3408.4 | 0.622 |
| bid_q2_all | 68 | 712.8 | 0.618 | 141 | -1657.7 | 0.539 |
| bid_q2_silence | 34 | 737.9 | 0.735 | 55 | 2828.5 | 0.818 |
| bid_q2_noisy_SHORT | 32 | -788.1 | 0.375 | 84 | 2431.0 | 0.595 |
| bid_q1_low_all | 1202 | -1821.4 | 0.519 | 286 | -3262.1 | 0.497 |
| bid_q1_low_silence | 630 | 13153.1 | 0.621 | 84 | 1799.4 | 0.619 |
| bid_q1_low_noisy_SHORT | 505 | 7717.0 | 0.543 | 196 | 2041.5 | 0.515 |
| bid_zero_all | 1136 | -1959.5 | 0.518 | 137 | -1369.7 | 0.496 |
| bid_zero_silence | 598 | 12656.7 | 0.62 | 37 | -146.2 | 0.486 |
| bid_zero_noisy_SHORT | 474 | 7776.6 | 0.544 | 97 | -447.5 | 0.454 |
| bid_nonzero_all | 268 | 7.1 | 0.522 | 465 | -7750.6 | 0.486 |
| bid_nonzero_silence | 126 | 1622.8 | 0.635 | 157 | 7523.3 | 0.752 |
| bid_nonzero_noisy_SHORT | 131 | -674.4 | 0.534 | 300 | 11257.9 | 0.58 |

## H. Propagation Timing Breakdown

| Segment | N | SHORT T3R | SHORT win | LONG T3R | LONG win |
| --- | ---: | ---: | ---: | ---: | ---: |
| early_0_5min_cal | 321 | 3126.2 | 0.546 | -9306.6 | 0.38 |
| med_5_15min_cal | 182 | 1491.9 | 0.538 | -5515.3 | 0.429 |
| late_15_30min_cal | 177 | 1149.1 | 0.538 | -4330.4 | 0.441 |
| ultra_early_lt1min_cal | 137 | 518.2 | 0.551 | - | - |
| early_0_5min_hold | 227 | 4045.4 | 0.545 | -8135.9 | 0.401 |
| med_5_15min_hold | 97 | 3550.7 | 0.568 | -6024.8 | 0.34 |
| late_15_30min_hold | 84 | 1652.9 | 0.538 | -3882.1 | 0.405 |
| ultra_early_lt1min_hold | 132 | -816.6 | 0.458 | - | - |

## I. Weekday Effect

| Day | Cal N | Hold N | Sil WR cal | Sil WR hold | Sil T3R hold | Short WR cal | Short WR hold | Short T3R hold |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Mon | 222 | 98 | 0.723 | 0.675 | 1389.6 | 0.444 | 0.537 | 316.1 |
| Tue | 239 | 103 | 0.637 | 0.519 | -448.1 | 0.511 | 0.581 | 2470.5 |
| Wed | 217 | 137 | 0.496 | 0.714 | 1465.4 | 0.434 | 0.636 | 4862.7 |
| Thu | 234 | 100 | 0.694 | 0.667 | -195.7 | 0.65 | 0.642 | 3388.9 |
| Fri | 207 | 92 | 0.657 | 0.73 | 1922.8 | 0.677 | 0.442 | -1162.4 |
| Sat | 98 | 22 | 0.638 | 1.0 | None | 0.429 | 0.467 | -554.3 |
| Sun | 187 | 50 | 0.531 | 0.864 | 685.2 | 0.536 | 0.179 | -1361.2 |

## J. Same-Day Sequential Trades

### cal
| Trade order | N | T3R | med | win |
| --- | ---: | ---: | ---: | ---: |
| 1st_silence_cal | 72 | 696.1 | 23.7 | 0.611 |
| 2nd_silence_cal | 69 | -23.7 | 19.2 | 0.609 |
| 3rd_plus_silence_cal | 583 | 12576.3 | 22.3 | 0.626 |
Daily event counts: n_days=74 min=1 med=18.5 max=47 p75=26.5 p90=34.0

### hold
| Trade order | N | T3R | med | win |
| --- | ---: | ---: | ---: | ---: |
| 1st_silence_hold | 21 | 50.1 | 34.4 | 0.667 |
| 2nd_silence_hold | 21 | 474.8 | 34.4 | 0.762 |
| 3rd_plus_silence_hold | 152 | 5826.0 | 33.9 | 0.697 |
Daily event counts: n_days=22 min=3 med=25.0 max=57 p75=38.5 p90=48.3

---
## New Questions After This Suite

1. **Perfect storm score>=5**: WR is very high but N small — needs more data (2026 Q3?)
2. **Session x score interaction**: score>=3 US session silence — combined permutation null?
3. **Propagation cascade SIZE predicts momentum strength** — high max prop size = stronger SHORT
4. **ETH -100bps 1h + silence = 5th signal?** — eth1h bearish context + silence interaction
5. **BTC lead + noisy SHORT** — WR=79% but N=29 in hold; is this a fifth standalone signal?
6. **bid_depth impact on SILENCE RATE** — does high bid_depth predict silence at entry?
7. **Weekend cascades** — different mechanics (lower liquidity) — should we filter?
8. **Frequency plateau** — at score>=4, monthly frequency drops. Is it still worth it?
9. **Sequential day hypothesis** — after a 'cascade storm day' (10+ events), next day?
10. **Cross-asset silence** — no BTC OR ETH cascade in 30min = strongest silence variant?

RESEARCH_ONLY. No live changes without explicit operator sign-off.
