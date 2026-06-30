# S34 Fifth Wave — Final Research Questions

Generated: `2026-06-30T10:56:54.151354+00:00`  |  Status: `RESEARCH_ONLY_NO_LIVE_CHANGE`
Cal: 1404 (2026-02-15T18:32:18Z to 2026-06-08T01:05:38Z)
Hold: 602 (2026-06-08T01:24:48Z to 2026-06-29T08:28:10Z)

## A. neither_silence SHORT — Formal OOS Permutation

Cal: 90 N, T3R=6042.5, WR=0.667 | Perm: p=0.0 real=6042.5 N=90 -> **PASS**
Hold: 119 N, T3R=8599.5, WR=0.731 | Perm: p=0.0 real=8599.5 N=119 -> **PASS**

| Gate | Cal N | Cal T3R | Cal win | Cal Perm | Hold N | Hold T3R | Hold win |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| neither_short_score_gte1 | 88 | 5955.5 | 0.659 | 0.0 **PASS** | 117 | 8700.1 | 0.744 |
| neither_short_score_gte2 | 77 | 5408.7 | 0.675 | 0.0 **PASS** | 105 | 8635.4 | 0.762 |
| neither_short_score_gte3 | 48 | 3574.1 | 0.688 | 0.0 **PASS** | 82 | 7377.8 | 0.793 |
| neither_short_score_gte4 | 17 | 769.6 | 0.706 | 0.0 **PASS** | 52 | 5718.8 | 0.904 |
| neither_short_sync_gte200K | 45 | 3770.8 | 0.756 | - | 80 | 6180.0 | 0.725 |
| neither_short_sync_gte300K | 26 | 2430.8 | 0.885 | - | 67 | 4978.2 | 0.731 |
| neither_short_sync_gte500K | 16 | 1105.2 | 0.875 | - | 49 | 3849.8 | 0.714 |

## B. bid_depth Data Period Analysis

First cal event with bid_dep>0: `2026-04-11T20:03:34Z`

Cal BEFORE bid_data: N=1107, bid_zero=1107, silence WR=0.615
Cal AFTER bid_data:  N=297, bid_nonzero=268
  silence_long WR=0.655 T3R=2546.3
  sil_score3_biddep WR=0.667 T3R=1211.1 N=66

### Monthly bid_depth coverage
| Month | N events | N with bid | Coverage |
| --- | ---: | ---: | ---: |
| 2026-02 | 312 | 0 | 0.0 |
| 2026-03 | 697 | 0 | 0.0 |
| 2026-04 | 371 | 268 | 0.722 |
| 2026-06 | 626 | 465 | 0.743 |

## C. Ultra-Early Exit Management

| Split | N | Managed T3R | Managed WR | Unmanaged T3R | Unmanaged WR | Improvement |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| cal | 1115 | 280.8 | 0.473 | -4204.6 | 0.525 | 4485.4 |
| hold | 489 | -7776.2 | 0.37 | -8907.8 | 0.501 | 1131.6 |

Ultra-early events specifically (enter vs flat-exit):
| Split | N ultra | Hold-2h T3R | Hold-2h WR | Flat-exit T3R | Flat-exit WR |
| --- | ---: | ---: | ---: | ---: | ---: |
| cal | 101 | -3160.5 | 0.386 | -490.0 | 0.0 |
| hold | 109 | -1917.5 | 0.477 | -530.0 | 0.0 |

## D. neither_silence + score>=3 SHORT (Cross-Asset Cascade)

| Signal | Cal N | Cal T3R | Cal win | Cal Perm | Hold N | Hold T3R | Hold win |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| neither_s3_short | 48 | 3574.1 | 0.688 | 0.0 **PASS** | 82 | 7377.8 | 0.793 |
| neither_s3_US_short | 34 | 803.7 | 0.559 | 0.002 **PASS** | 62 | 6829.5 | 0.887 |
| neither_s3_sync300_short | 21 | 2019.4 | 0.857 | 0.0 **PASS** | 60 | 4899.1 | 0.783 |
| neither_s3_prop4_short | 26 | 1242.6 | 0.654 | 0.0 **PASS** | 72 | 6978.9 | 0.806 |
| neither_s3_WedThu_short | 3 | None | 0.667 | 0.0 **PASS** | 49 | 5561.3 | 0.878 |
| neither_s2_short | 77 | 5408.7 | 0.675 | 0.0 **PASS** | 105 | 8635.4 | 0.762 |
| neither_no_ultra_short | 61 | 3654.7 | 0.656 | 0.0 **PASS** | 61 | 6888.0 | 0.852 |
| neither_s3_no_ultra_short | 31 | 2703.1 | 0.774 | 0.0 **PASS** | 41 | 5780.7 | 0.902 |

## E. Best LONG Subset — Permutation Both Splits

| Signal | Cal N | Cal T3R | Cal win | Cal Perm | Hold N | Hold T3R | Hold win | Hold Perm |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- |
| WedThu_US_s3_bid_sil | 6 | None | 0.833 | 0.0 **PASS** | 24 | 1260.4 | 0.917 | 0.0 **PASS** |
| WedThu_s3_bid_sil | 16 | 373.1 | 0.875 | 0.007 **PASS** | 32 | 1587.7 | 0.875 | 0.0 **PASS** |
| US_s3_bid_sil | 34 | 718.4 | 0.735 | 0.007 **PASS** | 52 | 3194.3 | 0.885 | 0.0 **PASS** |
| weekday_s3_bid_sil | 37 | 1334.0 | 0.892 | 0.001 **PASS** | 86 | 6162.1 | 0.884 | 0.0 **PASS** |
| MonThu_s3_bid_sil | 28 | 899.5 | 0.857 | 0.001 **PASS** | 62 | 3768.2 | 0.855 | 0.0 **PASS** |
| s3_bid_sil_cluster | 35 | 268.2 | 0.657 | 0.068 **ARTIFACT** | 81 | 5738.0 | 0.901 | 0.0 **PASS** |
| s3_bid_sil_eth1h_bear | 21 | 83.0 | 0.619 | 0.083 **ARTIFACT** | 58 | 2454.4 | 0.845 | 0.0 **PASS** |
| s4_bid_sil | 38 | 642.0 | 0.737 | 0.015 **PASS** | 55 | 3819.7 | 0.873 | 0.0 **PASS** |
| s3_bid_sil_WR_target | 26 | 51.3 | 0.615 | 0.106 **ARTIFACT** | 64 | 4199.8 | 0.891 | 0.0 **PASS** |

## F. Rolling 7-Day WR Stability

| Window start | N | Avg sync_k | Sil rate | Sil WR | Noisy SHORT WR | S3+bid Sil WR | Neither SHORT WR | Holdout? |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 2026-02-15 | 124 | 372022.2 | 0.556 | 0.565 | 0.176 | - | 0.182 |  |
| 2026-02-18 | 125 | 158164.7 | 0.608 | 0.632 | 0.479 | - | 0.667 |  |
| 2026-02-21 | 191 | 175779.3 | 0.492 | 0.596 | 0.733 | - | 0.778 |  |
| 2026-02-24 | 250 | 111731.3 | 0.488 | 0.639 | 0.706 | - | 1.0 |  |
| 2026-02-27 | 249 | 193864.4 | 0.518 | 0.643 | 0.677 | - | 1.0 |  |
| 2026-03-02 | 196 | 231682.2 | 0.587 | 0.661 | 0.687 | - | 1.0 |  |
| 2026-03-05 | 150 | 278479.2 | 0.64 | 0.677 | 0.472 | - | 0.875 |  |
| 2026-03-08 | 135 | 130432.6 | 0.615 | 0.711 | 0.468 | - | 0.818 |  |
| 2026-03-11 | 166 | 133202.3 | 0.554 | 0.641 | 0.688 | - | 0.923 |  |
| 2026-03-14 | 155 | 147644.3 | 0.503 | 0.628 | 0.594 | - | 1.0 |  |
| 2026-03-17 | 158 | 176085.5 | 0.487 | 0.571 | 0.568 | - | 0.778 |  |
| 2026-03-20 | 118 | 181267.8 | 0.551 | 0.508 | 0.532 | - | 0.667 |  |
| 2026-03-23 | 100 | 133383.3 | 0.57 | 0.561 | 0.429 | - | - |  |
| 2026-03-26 | 120 | 155045.5 | 0.467 | 0.607 | 0.508 | - | 1.0 |  |
| 2026-03-29 | 100 | 147943.1 | 0.44 | 0.636 | 0.473 | - | 1.0 |  |
| 2026-04-01 | 65 | 149913.2 | 0.415 | 0.667 | 0.594 | - | 0.889 |  |
| 2026-04-04 | 53 | 160884.9 | 0.491 | 0.654 | 0.19 | - | 0.667 |  |
| 2026-04-07 | 84 | 186951.4 | 0.464 | 0.41 | 0.686 | 0.375 | 1.0 |  |
| 2026-04-10 | 129 | 179110.0 | 0.512 | 0.652 | 0.648 | 0.741 | 0.8 |  |
| 2026-04-13 | 177 | 203077.7 | 0.52 | 0.609 | 0.649 | 0.605 | 0.824 |  |
| 2026-04-16 | 147 | 236157.0 | 0.429 | 0.635 | 0.434 | 0.595 | 0.615 |  |
| 2026-04-19 | 88 | 223121.7 | 0.352 | 0.806 | 0.333 | 0.882 | 0.429 |  |
| 2026-04-22 | 36 | 162395.8 | 0.528 | 0.842 | 0.588 | 1.0 | - |  |
| 2026-04-25 | 12 | 225318.5 | 0.5 | 1.0 | 0.5 | 1.0 | - |  |
| 2026-05-31 | 15 | 98537.7 | 0.6 | 0.778 | 1.0 | - | - |  |
| 2026-06-03 | 151 | 451900.0 | 0.305 | 0.609 | 0.469 | - | 0.72 |  |
| 2026-06-06 | 208 | 365620.8 | 0.327 | 0.574 | 0.45 | 1.0 | 0.667 |  |
| 2026-06-09 | 171 | 207004.8 | 0.339 | 0.586 | 0.519 | 0.87 | 0.889 |  |
| 2026-06-12 | 169 | 469159.7 | 0.355 | 0.75 | 0.644 | 0.825 | 0.829 |  |
| 2026-06-15 | 173 | 467463.8 | 0.295 | 0.804 | 0.617 | 0.868 | 0.795 | YES |
| 2026-06-18 | 174 | 944457.7 | 0.247 | 0.791 | 0.618 | 0.897 | 0.83 | YES |
| 2026-06-21 | 209 | 839812.2 | 0.311 | 0.8 | 0.645 | 0.884 | 0.769 | YES |
| 2026-06-24 | 141 | 573238.2 | 0.44 | 0.806 | 0.421 | 0.927 | 0.348 | YES |
| 2026-06-27 | 44 | 473987.3 | 0.432 | 0.737 | 0.08 | 0.846 | 0.071 | YES |

---
## FINAL VALIDATED SIGNAL REGISTRY

| # | Signal | Hold N | Hold WR | Hold T3R | Cal Perm | Hold Perm |
| --- | --- | ---: | ---: | ---: | --- | --- |
| 1 | Silence LONG (30min ETH quiet) | 194 | 70.1% | +7733 | p=0.0 PASS | p=0.0 PASS |
| 2 | Silence + sync>=200K LONG | 65 | 83.1% | +4298 | p=0.001 PASS | p=0.0 PASS |
| 3 | noisy_NOT_bull SHORT (ETH propagation) | 397 | 54.9% | +11360 | p=0.0 PASS | - |
| 4 | neither_silence SHORT (ETH+BTC both noisy) | 119 | 73.1% | +8599 | p=0.0 PASS | p=0.0 PASS |
| 5 | score3+bid_dep+silence LONG | 102 | 88.2% | +6952 | p=0.004 PASS | p=0.0 PASS |
| — | Combined portfolio (refined) | 233 | 75.5% | +15278 | PASS | PASS |

**Entry rule**: anchor entry at cascade detection time. No delay.
**Exit rule**: silence -> hold 4h. Noisy (>1min) -> hold 2h. Ultra-early (<1min) -> exit flat.

RESEARCH_ONLY. Live promotion requires explicit operator sign-off.
