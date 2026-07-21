# S34 Deep Expansion Suite

Generated: `2026-06-30T10:29:40.413876+00:00`  |  Status: `RESEARCH_ONLY_NO_LIVE_CHANGE`
Cal: 1404 (2026-02-15T18:32:18Z to 2026-06-08T01:05:38Z)
Hold: 602 (2026-06-08T01:24:48Z to 2026-06-29T08:28:10Z)

Baseline reference (from comprehensive final suite):
- Silence LONG hold: WR=70.1%, T3R=+7733
- Silence+sync>=200K LONG hold: WR=83.1%, T3R=+4298
- noisy_NOT_bull SHORT hold: WR=54.9%, T3R=+11360

## A. Signal Portfolio (LONG + SHORT Combined)

| Split | Signal | N | T3R | med | win | coverage |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| cal | LONG only | 724 | 14738.2 | 22.1 | 0.623 |  |
| cal | SHORT only | 605 | 7778.7 | 9.1 | 0.542 |  |
| cal | Combined | 1329 | 23546.7 | 17.9 | 0.586 | 0.947 |
| cal | Untraded (bull+noisy) | 75 | - | - | - | - |
| hold | LONG only | 194 | 7733.7 | 34.4 | 0.701 |  |
| hold | SHORT only | 397 | 11360.3 | 10.5 | 0.549 |  |
| hold | Combined | 591 | 19952.0 | 22.1 | 0.599 | 0.982 |
| hold | Untraded (bull+noisy) | 11 | - | - | - | - |

## B. Flip Strategy (SHORT first, flip to LONG on silence)

| Split | Strategy | N | T3R | med | win |
| --- | --- | ---: | ---: | ---: | ---: |
| cal | Always SHORT 2h | 1404 | -15783.1 | -15.7 | 0.429 |
| cal | Baseline (sil=LONG, noisy=SHORT) | 1404 | 19855.6 | 13.9 | 0.569 |
| cal | Flip (SHORT->LONG on silence) | 1404 | -19854.4 | -21.6 | 0.412 |
| hold | Always SHORT 2h | 602 | 453.6 | -8.0 | 0.437 |
| hold | Baseline (sil=LONG, noisy=SHORT) | 602 | 18607.0 | 16.3 | 0.588 |
| hold | Flip (SHORT->LONG on silence) | 602 | 7937.0 | -4.1 | 0.473 |

*Note*: flip approximation: silence drift assumed +25bps from prior test

## C. vdepth Gate (Overshoot Depth)

Cal vdepth percentiles: p25=11.7bps  p50=19.4bps  p75=31.9bps

| Segment | Gate | Cal N | Cal T3R | Cal win | Hold N | Hold T3R | Hold win |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| vdepth_q1_all | | 347 | -1481.5 | 0.487 | 195 | -2837.7 | 0.559 |
| vdepth_q1_silence | | 203 | 3018.1 | 0.596 | 58 | 844.9 | 0.672 |
| vdepth_q1_noisy | | 144 | -5237.0 | 0.333 | 137 | -4158.5 | 0.511 |
| vdepth_q2_all | | 351 | 475.4 | 0.553 | 191 | -4937.6 | 0.419 |
| vdepth_q2_silence | | 197 | 3915.4 | 0.64 | 60 | 770.9 | 0.633 |
| vdepth_q2_noisy | | 154 | -4224.2 | 0.442 | 131 | -6325.2 | 0.321 |
| vdepth_q3_all | | 352 | -3790.4 | 0.477 | 146 | -1702.1 | 0.5 |
| vdepth_q3_silence | | 156 | 1678.9 | 0.596 | 53 | 2688.1 | 0.717 |
| vdepth_q3_noisy | | 196 | -5958.5 | 0.383 | 93 | -5090.8 | 0.376 |
| vdepth_q4_all | | 354 | 427.9 | 0.559 | 70 | -1162.1 | 0.457 |
| vdepth_q4_silence | | 168 | 3794.5 | 0.661 | 23 | 1430.9 | 0.913 |
| vdepth_q4_noisy | | 186 | -4364.9 | 0.468 | 47 | -2964.8 | 0.234 |
| vdepth_lt_15_all | | 501 | -782.5 | 0.503 | 280 | -3874.7 | 0.529 |
| vdepth_lt_15_silence | | 290 | 4360.1 | 0.603 | 83 | 2123.7 | 0.687 |
| vdepth_lt_15_noisy | | 211 | -6092.9 | 0.365 | 197 | -6624.4 | 0.462 |
| vdepth_15_30_all | | 508 | -3916.0 | 0.502 | 238 | -4987.0 | 0.45 |
| vdepth_15_30_silence | | 239 | 3871.9 | 0.619 | 82 | 2581.5 | 0.671 |
| vdepth_15_30_noisy | | 269 | -8484.7 | 0.398 | 156 | -8269.1 | 0.333 |
| vdepth_30_60_all | | 321 | 923.5 | 0.567 | 77 | -1264.7 | 0.455 |
| vdepth_30_60_silence | | 158 | 2760.3 | 0.633 | 25 | 1073.3 | 0.8 |
| vdepth_30_60_noisy | | 163 | -2805.9 | 0.503 | 52 | -2758.7 | 0.288 |
| vdepth_gt_60_all | | 74 | -611.3 | 0.541 | 7 | None | 0.571 |
| vdepth_gt_60_silence | | 37 | 1362.9 | 0.757 | 4 | None | 1.0 |
| vdepth_gt_60_noisy | | 37 | -2510.3 | 0.324 | 3 | None | 0.0 |

## D. BTC Context (btc4h_bps)

| Condition | Cal N | Cal T3R | Cal win | Hold N | Hold T3R | Hold win |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| btc4h_bull_gt100_all | 178 | 727.9 | 0.528 | 38 | 135.5 | 0.658 |
| btc4h_bull_gt100_silence | 88 | 677.7 | 0.591 | 17 | 310.3 | 0.765 |
| btc4h_bull_gt100_noisy | 90 | -948.1 | 0.467 | 21 | -346.9 | 0.571 |
| btc4h_bull_0_100_all | 307 | 1888.6 | 0.495 | 131 | -1966.9 | 0.435 |
| btc4h_bull_0_100_silence | 156 | 2550.3 | 0.545 | 67 | 784.6 | 0.582 |
| btc4h_bull_0_100_noisy | 151 | -1378.5 | 0.444 | 64 | -3244.4 | 0.281 |
| btc4h_bear_all | 919 | -5980.7 | 0.526 | 433 | -7819.8 | 0.49 |
| btc4h_bear_silence | 480 | 9726.7 | 0.654 | 110 | 5584.1 | 0.764 |
| btc4h_bear_noisy | 439 | -16804.8 | 0.385 | 323 | -14145.9 | 0.396 |
| btc4h_bear_lt-100_all | 355 | -3241.0 | 0.501 | 171 | -2920.5 | 0.515 |
| btc4h_bear_lt-100_silence | 160 | 2753.7 | 0.662 | 42 | 3514.9 | 0.833 |
| btc4h_bear_lt-100_noisy | 195 | -6622.8 | 0.369 | 129 | -7172.7 | 0.411 |
| btc_lead_cascade_all | 10 | None | 0.5 | 31 | -2504.5 | 0.258 |
| btc_lead_cascade_silence | 3 | None | 0.0 | 2 | None | 1.0 |
| btc_lead_cascade_noisy | 7 | None | 0.714 | 29 | -2549.9 | 0.207 |
| no_btc_lead_all | 1394 | -1491.8 | 0.519 | 571 | -6411.7 | 0.501 |
| no_btc_lead_silence | 721 | 14845.3 | 0.626 | 192 | 7688.3 | 0.698 |
| no_btc_lead_noisy | 673 | -17526.8 | 0.406 | 379 | -14842.0 | 0.401 |

## E. Book Imbalance Gate

Cal book_imb percentiles: p25=0.0  p75=0.0

| Condition | Cal N | Cal T3R | Cal win | Hold N | Hold T3R | Hold win |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| bid_heavy_q4_all | 1284 | -2254.7 | 0.517 | 353 | -4870.5 | 0.484 |
| bid_heavy_q4_silence | 663 | 13310.6 | 0.62 | 108 | 3720.0 | 0.676 |
| bid_heavy_q4_noisy_short | 621 | 7690.8 | 0.541 | 245 | 4958.7 | 0.494 |
| bid_heavy_q3q4_all | 1284 | -2254.7 | 0.517 | 353 | -4870.5 | 0.484 |
| bid_heavy_q3q4_silence | 663 | 13310.6 | 0.62 | 108 | 3720.0 | 0.676 |
| bid_heavy_q3q4_noisy_short | 621 | 7690.8 | 0.541 | 245 | 4958.7 | 0.494 |
| ask_heavy_q1_all | 120 | 411.2 | 0.542 | 249 | -4492.5 | 0.494 |
| ask_heavy_q1_silence | 61 | 968.9 | 0.656 | 86 | 3225.6 | 0.733 |
| ask_heavy_q1_noisy_short | 59 | -455.8 | 0.492 | 163 | 4894.6 | 0.601 |
| bid_heavy_pos_all | 148 | -862.8 | 0.507 | 215 | -4114.9 | 0.474 |
| bid_heavy_pos_silence | 65 | 383.9 | 0.615 | 70 | 3440.9 | 0.771 |
| bid_heavy_pos_noisy_short | 83 | -565.3 | 0.566 | 145 | 4924.0 | 0.524 |
| ask_heavy_neg_all | 1256 | -1089.6 | 0.521 | 387 | -5248.1 | 0.496 |
| ask_heavy_neg_silence | 659 | 14084.3 | 0.624 | 124 | 3504.7 | 0.661 |
| ask_heavy_neg_noisy_short | 597 | 7539.4 | 0.533 | 263 | 4929.3 | 0.544 |

## F. prior4h Permutation Null

| Condition | Cal N | Cal T3R | Cal win | Hold N | Hold T3R | Hold win | Perm p | Perm verdict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| prior4h_gt100 | 208 | 824.9 | 0.514 | 61 | -1145.3 | 0.623 | 0.092 | **ARTIFACT** |
| prior4h_gt50 | 338 | 2449.4 | 0.524 | 90 | -1889.4 | 0.578 | 0.026 | **PASS** |
| prior4h_0_50 | 139 | 2187.4 | 0.597 | 74 | -2687.4 | 0.27 | 0.004 | **PASS** |
| prior4h_neg | 927 | -7801.0 | 0.506 | 438 | -4883.8 | 0.507 | 0.996 | **ARTIFACT** |
| prior4h_lt-100 | 470 | -3561.2 | 0.504 | 228 | -1092.9 | 0.561 | 0.886 | **ARTIFACT** |
| prior4h_gt100_sil | 104 | 1144.4 | 0.577 | 27 | 588.9 | 0.815 | 0.023 | **PASS** |
| prior4h_neg_sil | 483 | 8703.2 | 0.631 | 122 | 6741.3 | 0.762 | 0.0 | **PASS** |
| prior4h_gt100_noisy_short | 48 | -1278.6 | 0.479 | 24 | 989.7 | 0.583 | 0.829 | **ARTIFACT** |

## G. Cascade Sequence Counter

Cal distribution: {0: 368, 1: 304, 2: 202, 3: 157, 4: 103, 5: 270}
Hold distribution: {0: 81, 1: 92, 2: 65, 3: 63, 4: 38, 5: 263}

| Sequence position | Gate | Cal N | Cal T3R | Cal win | Hold N | Hold T3R | Hold win |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| first_in_cluster_all | | 368 | 1412.2 | 0.533 | 81 | -1761.8 | 0.469 |
| first_in_cluster_silence | | 221 | 1918.5 | 0.597 | 30 | 402.4 | 0.733 |
| first_in_cluster_noisy_short | | 128 | -3182.9 | 0.422 | 49 | 1160.5 | 0.551 |
| second_n_prior2h_1_all | | 304 | -3294.8 | 0.461 | 92 | -1796.8 | 0.446 |
| second_n_prior2h_1_silence | | 155 | 1115.3 | 0.51 | 30 | -494.8 | 0.533 |
| second_n_prior2h_1_noisy_short | | 130 | 560.8 | 0.554 | 59 | -162.7 | 0.508 |
| mid_n_prior2h_2_all | | 202 | -1886.6 | 0.47 | 65 | -1144.2 | 0.292 |
| mid_n_prior2h_2_silence | | 91 | 1694.8 | 0.615 | 25 | 351.6 | 0.44 |
| mid_n_prior2h_2_noisy_short | | 97 | 1688.4 | 0.577 | 38 | 531.2 | 0.632 |
| deep_n_prior2h_3plus_all | | 530 | -946.4 | 0.562 | 364 | -5687.1 | 0.538 |
| deep_n_prior2h_3plus_silence | | 257 | 7597.3 | 0.716 | 109 | 5984.7 | 0.798 |
| deep_n_prior2h_3plus_noisy_short | | 250 | 5857.1 | 0.584 | 251 | 8129.5 | 0.546 |
| any_prior_cascade_all | | 1036 | -4055.6 | 0.514 | 521 | -7605.9 | 0.491 |
| any_prior_cascade_silence | | 503 | 12036.9 | 0.634 | 164 | 6543.2 | 0.695 |
| any_prior_cascade_noisy_short | | 477 | 10077.4 | 0.574 | 348 | 9666.3 | 0.549 |

## H. Time of Day (Session Breakdown)

| Session/Key | Cal N | Cal T3R | Cal win | Hold N | Hold T3R | Hold win |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| EU_all | 301 | -5165.0 | 0.462 | 111 | -2858.7 | 0.468 |
| EU_silence | 172 | 671.7 | 0.564 | 39 | 629.2 | 0.667 |
| EU_noisy_short | 119 | 4329.9 | 0.681 | 69 | 1745.1 | 0.536 |
| US_all | 658 | 4951.5 | 0.564 | 285 | -8438.3 | 0.446 |
| US_silence | 295 | 9698.0 | 0.692 | 81 | 3573.5 | 0.765 |
| US_noisy_short | 315 | -1216.3 | 0.489 | 196 | 9098.1 | 0.643 |
| ASIA_all | 445 | -3503.8 | 0.492 | 206 | 1228.7 | 0.558 |
| ASIA_silence | 257 | 2676.8 | 0.584 | 74 | 2180.9 | 0.649 |
| ASIA_noisy_short | 171 | 2653.6 | 0.544 | 132 | -1236.9 | 0.417 |

### Hourly breakdown (hold only)
| Hour UTC | Sil N | Sil WR | Noisy Short N | Noisy WR |
| --- | ---: | ---: | ---: | ---: |
| 00:00 | 8 | 0.5 | 12 | 0.667 |
| 01:00 | 5 | 0.8 | 12 | 0.333 |
| 02:00 | 5 | 0.8 | 23 | 0.348 |
| 03:00 | 11 | 0.545 | 9 | 0.667 |
| 04:00 | 6 | 0.833 | 20 | 0.55 |
| 05:00 | 8 | 0.625 | 16 | 0.188 |
| 06:00 | 6 | 0.833 | 5 | 0.6 |
| 07:00 | 3 | - | 8 | 0.625 |
| 08:00 | 3 | 0.667 | 20 | 0.5 |
| 09:00 | 16 | 0.5 | 6 | 0.5 |
| 10:00 | 3 | 1.0 | 7 | 0.714 |
| 11:00 | 8 | 0.875 | 8 | - |
| 12:00 | 6 | 1.0 | 20 | 0.7 |
| 13:00 | 10 | 0.8 | 39 | 0.795 |
| 14:00 | 9 | 0.889 | 41 | 0.683 |
| 15:00 | 12 | 0.583 | 34 | 0.735 |
| 16:00 | 12 | 0.75 | 29 | 0.517 |
| 17:00 | 7 | 0.571 | 23 | 0.609 |
| 18:00 | 12 | 0.833 | 8 | 0.75 |
| 19:00 | 13 | 0.769 | 18 | 0.278 |
| 20:00 | 6 | 1.0 | 4 | 0.5 |
| 21:00 | 9 | 0.444 | 12 | 0.333 |
| 22:00 | 10 | 0.6 | 14 | 0.357 |
| 23:00 | 6 | 0.833 | 9 | 0.333 |

## I. Tail Event Audit (Losers vs Winners)

### cal

**Silence LONG Losers vs Winners** (bottom/top 10%):
| Feature | Losers | Winners |
| --- | ---: | ---: |
| avg vdepth | 23.0 | 22.9 |
| avg btc4h | -14.6 | -16.2 |
| avg sync_k | 139563.8 | 82177.9 |
| avg prior4h | -19.4 | -14.8 |
| avg book_imb | 0.0 | -0.0 |
| avg thresh | 95138.9 | 94444.4 |
| pct_btc_lead | 0.0 | 0.0 |
| sessions | {'EU': 30, 'US': 24, 'ASIA': 18} | {'EU': 17, 'US': 41, 'ASIA': 14} |

**noisy SHORT Losers vs Winners** (bottom/top 10%):
| Feature | Losers | Winners |
| --- | ---: | ---: |
| avg vdepth | 25.8 | 29.7 |
| avg btc4h | -34.7 | -70.2 |
| avg sync_k | 133041.9 | 505000.6 |
| avg prior4h | -46.3 | -78.7 |
| pct_btc_lead | 0.05 | 0.033 |
| sessions | {'EU': 7, 'US': 42, 'ASIA': 11} | {'EU': 18, 'US': 23, 'ASIA': 19} |

### hold

**Silence LONG Losers vs Winners** (bottom/top 10%):
| Feature | Losers | Winners |
| --- | ---: | ---: |
| avg vdepth | 16.6 | 25.5 |
| avg btc4h | 29.0 | -105.4 |
| avg sync_k | 693945.4 | 260005.2 |
| avg prior4h | 29.5 | -119.3 |
| avg book_imb | 0.1 | -0.0 |
| avg thresh | 84210.5 | 113157.9 |
| pct_btc_lead | 0.0 | 0.0 |
| sessions | {'EU': 3, 'US': 8, 'ASIA': 8} | {'EU': 3, 'US': 12, 'ASIA': 4} |

**noisy SHORT Losers vs Winners** (bottom/top 10%):
| Feature | Losers | Winners |
| --- | ---: | ---: |
| avg vdepth | 20.6 | 20.1 |
| avg btc4h | -135.4 | -153.0 |
| avg sync_k | 389140.2 | 1019118.6 |
| avg prior4h | -198.9 | -100.5 |
| pct_btc_lead | 0.103 | 0.154 |
| sessions | {'EU': 5, 'US': 18, 'ASIA': 16} | {'EU': 8, 'US': 26, 'ASIA': 5} |


## J. Frequency & Kelly Sizing

### cal (112.3 days, 3.688 months)
| Signal | Monthly trades | Edge bps | WR | avg_win | avg_loss | W/L ratio | Kelly | Half-Kelly |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| silence_LONG | 196.294 | 22.4 | 0.623 | 74.3 | -63.3 | 1.173 | 0.301 | 0.151 |
| noisy_short | 164.03 | 14.6 | 0.542 | 92.6 | -77.8 | 1.191 | 0.158 | 0.079 |
| silence_live_200K | 36.331 | 28.8 | 0.664 | 75.4 | -63.2 | 1.193 | 0.383 | 0.191 |
| silence_highsync | 47.718 | 20.1 | 0.636 | 59.9 | -49.4 | 1.211 | 0.336 | 0.168 |

### hold (21.3 days, 0.7 months)
| Signal | Monthly trades | Edge bps | WR | avg_win | avg_loss | W/L ratio | Kelly | Half-Kelly |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| silence_LONG | 277.325 | 44.3 | 0.701 | 84.5 | -50.0 | 1.69 | 0.524 | 0.262 |
| noisy_short | 567.516 | 31.7 | 0.549 | 103.8 | -56.0 | 1.854 | 0.306 | 0.153 |
| silence_live_200K | 68.617 | 49.3 | 0.688 | 84.9 | -28.9 | 2.934 | 0.581 | 0.29 |
| silence_highsync | 92.918 | 79.3 | 0.831 | 102.1 | -32.3 | 3.163 | 0.777 | 0.389 |

## K. BTC-Led Cascade (Cross-Asset Lead)

| Key | N | T3R | med | win |
| --- | ---: | ---: | ---: | ---: |
| btc_lead_all_cal | 10 | None | -12.5 | 0.5 |
| btc_lead_silence_cal | 3 | None | -35.7 | 0.0 |
| btc_lead_noisy_cal | 7 | None | 84.9 | 0.714 |
| no_btc_lead_all_cal | 1394 | -1491.8 | 5.7 | 0.519 |
| no_btc_lead_silence_cal | 721 | 14845.3 | 22.3 | 0.626 |
| no_btc_lead_noisy_cal | 673 | -17526.8 | -18.0 | 0.406 |
| btc_lead_all_hold | 31 | -2504.5 | -47.9 | 0.258 |
| btc_lead_silence_hold | 2 | None | 22.7 | 1.0 |
| btc_lead_noisy_hold | 29 | -2549.9 | -54.1 | 0.207 |
| no_btc_lead_all_hold | 571 | -6411.7 | 0.6 | 0.501 |
| no_btc_lead_silence_hold | 192 | 7688.3 | 34.4 | 0.698 |
| no_btc_lead_noisy_hold | 379 | -14842.0 | -15.8 | 0.401 |

## L. Weekly Holdout Stability

| Week | N events | Avg sync_k | Sil rate | Sil T3R | Sil win | SHORT T3R | SHORT win |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-W23 | 194 | 379899.4 | 0.325 | 592.2 | 0.556 | -412.8 | 0.461 |
| 2026-W24 | 181 | 450713.4 | 0.32 | 1681.4 | 0.759 | 3006.2 | 0.585 |
| 2026-W25 | 211 | 868705.0 | 0.294 | 3419.0 | 0.79 | 7531.5 | 0.61 |
| 2026-W26 | 16 | 271761.3 | 0.688 | None | 0.727 | None | 0.2 |

---
## Open Questions for Next Session

1. **vdepth optimal cut**: does vdepth>30bps + silence form a 4th validated signal?
2. **BTC-led cascade permutation null**: test K shows directional pattern — run formal perm test
3. **prior4h>100 + silence perm null (hold)**: PASS in cal — does hold confirm?
4. **Session-specific signal**: if EU silence shows strongest WR, can we trade EU-only?
5. **Flip strategy cost sensitivity**: how much drift reduction needed for flip to be worth it?
6. **Combined portfolio perm null**: silence LONG + noisy SHORT as a SINGLE combined strategy
7. **Cascade depth + sync**: deep cascade (vdepth>40) in high sync + silence = ultra signal?
8. **Next-day fade**: does silence gate work on H8/H12 (overnight hold)?
9. **Sequence signal**: first cascade after 4h quiet period — stronger silence predictor?
10. **Bid depth absolute**: high bid_depth_usd at cascade time — does it predict silence?

RESEARCH_ONLY. No live changes without explicit operator sign-off.
