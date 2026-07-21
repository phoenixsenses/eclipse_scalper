# S34 Fourth Wave Research Suite

Generated: `2026-06-30T10:45:57.243798+00:00`  |  Status: `RESEARCH_ONLY_NO_LIVE_CHANGE`
Cal: 1404 (2026-02-15T18:32:18Z to 2026-06-08T01:05:38Z)
Hold: 602 (2026-06-08T01:24:48Z to 2026-06-29T08:28:10Z)

## A. bid_depth=0 Filter Analysis

### cal
Total N=1404  bid_zero N=1136 (80.9%)  bid_nonzero N=268
Silence rate: bid_zero=0.526 bid_nonzero=0.47
avg_sync_k: zero=190999.8  nonzero=209678.8
avg_vdepth: zero=25.7  nonzero=19.1
avg_thresh: zero=94586.3  nonzero=108395.5

| Gate | N | T3R | med | win |
| --- | ---: | ---: | ---: | ---: |
| bid_zero_silence_LONG | 598 | 12656.7 | 22.8 | 0.62 |
| bid_zero_noisy_SHORT | 474 | 7776.6 | 11.2 | 0.544 |
| bid_nonzero_silence_LONG | 126 | 1622.8 | 19.3 | 0.635 |
| bid_nonzero_noisy_SHORT | 131 | -674.4 | 6.0 | 0.534 |

### hold
Total N=602  bid_zero N=137 (22.8%)  bid_nonzero N=465
Silence rate: bid_zero=0.27 bid_nonzero=0.338
avg_sync_k: zero=487920.6  nonzero=593719.2
avg_vdepth: zero=19.6  nonzero=18.3
avg_thresh: zero=104379.6  nonzero=109032.3

| Gate | N | T3R | med | win |
| --- | ---: | ---: | ---: | ---: |
| bid_zero_silence_LONG | 37 | -146.2 | -1.9 | 0.486 |
| bid_zero_noisy_SHORT | 97 | -447.5 | -2.9 | 0.454 |
| bid_nonzero_silence_LONG | 157 | 7523.3 | 35.2 | 0.752 |
| bid_nonzero_noisy_SHORT | 300 | 11257.9 | 21.6 | 0.58 |

Perm bid_nonzero silence (cal): p=0.011 real=1622.8 null_p95=964.9 N=126 -> **PASS**

## B. Ultra-Event: Cluster (prior 2h) + Silence & 4+ Cascades Now

| Signal | Gate | Cal N | Cal T3R | Cal win | Hold N | Hold T3R | Hold win | Perm |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| n_prior2h_gte3 | silence_LONG | 257 | 7597.3 | 0.716 | 109 | 5984.7 | 0.798 | 0.0 **PASS** |
| n_prior2h_gte3 | noisy_SHORT | 250 | 5857.1 | 0.584 | 251 | 8129.5 | 0.546 | 0.0 **PASS** |
| n_prior2h_gte4 | silence_LONG | 179 | 4194.1 | 0.709 | 85 | 4532.9 | 0.835 | 0.0 **PASS** |
| n_prior2h_gte4 | noisy_SHORT | 179 | 2588.1 | 0.581 | 215 | 6960.8 | 0.54 | 0.0 **PASS** |
| n_prior2h_gte5 | silence_LONG | 119 | 2486.7 | 0.664 | 71 | 3252.5 | 0.817 | 0.0 **PASS** |
| n_prior2h_gte5 | noisy_SHORT | 140 | 2369.8 | 0.614 | 192 | 6172.8 | 0.547 | 0.0 **PASS** |
| n_prior2h_gte2 | silence_LONG | 348 | 10099.2 | 0.69 | 134 | 6844.0 | 0.731 | 0.0 **PASS** |
| n_prior2h_gte2 | noisy_SHORT | 347 | 8543.6 | 0.582 | 289 | 9254.3 | 0.557 | 0.0 **PASS** |
| prop_count>=4 SHORT | now | 140 | 7122.2 | 0.714 | 190 | 8736.3 | 0.632 | 0.0 **PASS** |

## C. Day-of-Week Permutation Null

| Day | Sil Cal N | Sil Cal win | Sil Hold N | Sil Hold win | Sil Perm | Short Hold N | Short Hold win | Short Perm |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- |
| Mon | 101 | 0.723 | 40 | 0.675 | 0.0 **PASS** | 54 | 0.537 | 0.494 **ARTIFACT** |
| Tue | 135 | 0.637 | 27 | 0.519 | 0.001 **PASS** | 74 | 0.581 | 0.407 **ARTIFACT** |
| Wed | 119 | 0.496 | 28 | 0.714 | 0.141 **ARTIFACT** | 107 | 0.636 | 0.036 **PASS** |
| Thu | 111 | 0.694 | 33 | 0.667 | 0.013 **PASS** | 67 | 0.642 | 0.0 **PASS** |
| Fri | 102 | 0.657 | 37 | 0.73 | 0.007 **PASS** | 52 | 0.442 | 0.0 **PASS** |
| Sat | 58 | 0.638 | 7 | 1.0 | 0.098 **ARTIFACT** | 15 | 0.467 | 0.009 **PASS** |
| Sun | 98 | 0.531 | 22 | 0.864 | 0.188 **ARTIFACT** | 28 | 0.179 | 0.282 **ARTIFACT** |

## D. 5th Signal Candidate: 200K + Cluster + Bear + bid_depth

| Signal | Cal N | Cal T3R | Cal win | Cal Perm | Hold N | Hold T3R | Hold win | Hold Perm |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- |
| 200K_sil_cluster_bear_biddep | 7 | None | 0.714 | 0.0 **PASS** | 14 | 702.7 | 0.929 | 0.0 **PASS** |
| 200K_sil_cluster_bear | 46 | 912.2 | 0.783 | 0.007 **PASS** | 17 | 690.7 | 0.824 | 0.0 **PASS** |
| 200K_sil_biddep | 29 | 240.5 | 0.69 | 0.056 **ARTIFACT** | 40 | 1486.5 | 0.75 | 0.0 **PASS** |
| 200K_sil_score3 | 91 | 2310.6 | 0.703 | 0.0 **PASS** | 32 | 1458.0 | 0.781 | 0.0 **PASS** |
| 200K_sil_score4 | 58 | 1608.2 | 0.741 | 0.002 **PASS** | 17 | 597.1 | 0.765 | 0.001 **PASS** |
| any_sil_score4_biddep | 38 | 642.0 | 0.737 | 0.015 **PASS** | 55 | 3819.7 | 0.873 | 0.0 **PASS** |
| any_sil_score3_biddep | 66 | 1211.1 | 0.667 | 0.004 **PASS** | 102 | 6952.2 | 0.882 | 0.0 **PASS** |
| any_sil_cluster_bear_biddep | 26 | 51.3 | 0.615 | 0.106 **ARTIFACT** | 64 | 4199.8 | 0.891 | 0.0 **PASS** |

## E. Ultra-Early (<1min) SHORT Mechanics

### cal
Ultra-early N=137  Normal-early N=543
| Feature | Ultra (<1min) | Normal (>=1min) |
| --- | ---: | ---: |
| avg sync_k | 235565.5 | 188943.3 |
| avg vdepth | 27.4 | 25.2 |
| avg btc4h  | -34.4 | -25.9 |
| avg prop_count | 3.8 | 2.6 |
| pct US session | 0.482 | 0.547 |
| pct sync>=300K | 0.219 | 0.169 |

Ultra SHORT H2: N=127 T3R=518.2 med=7.0 win=0.551 maxL=-207.2
Normal SHORT H2: N=478 T3R=6293.5 med=9.4 win=0.54 maxL=-728.1

### hold
Ultra-early N=132  Normal-early N=276
| Feature | Ultra (<1min) | Normal (>=1min) |
| --- | ---: | ---: |
| avg sync_k | 903143.2 | 626854.0 |
| avg vdepth | 17.5 | 18.7 |
| avg btc4h  | -93.3 | -71.5 |
| avg prop_count | 9.7 | 6.7 |
| pct US session | 0.53 | 0.486 |
| pct sync>=300K | 0.439 | 0.322 |

Ultra SHORT H2: N=131 T3R=-816.6 med=-4.3 win=0.458 maxL=-260.8
Normal SHORT H2: N=266 T3R=11517.9 med=30.9 win=0.594 maxL=-275.3

## F. Refined Portfolio (score>=3 + bid_dep>0 + prop filters)

| Split | Coverage | LONG N | LONG T3R | LONG win | SHORT N | SHORT T3R | SHORT win | Combined T3R | Combined win |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cal | 0.069 | 66 | 1211.1 | 0.667 | 31 | -300.0 | 0.581 | 1253.1 | 0.639 |
| hold | 0.387 | 102 | 6952.2 | 0.882 | 131 | 7468.6 | 0.656 | 15278.8 | 0.755 |

Perm (cal):  p=0.007 real=1253.1 null_p95=384.3 N=97 -> **PASS**
Perm (hold): p=0.0 real=15278.8 null_p95=-1578.0 N=233 -> **PASS**

## G. Wed+Thu US Session + Score>=3 (Best Subset Search)

| Signal | Cal N | Cal T3R | Cal win | Hold N | Hold T3R | Hold win | Perm p | Verdict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| baseline_silence | 724 | 14738.2 | 0.623 | 194 | 7733.7 | 0.701 | 0.0 | **PASS** |
| US_session_silence | 295 | 9698.0 | 0.692 | 81 | 3573.5 | 0.765 | 0.0 | **PASS** |
| WedThu_silence | 230 | 2333.5 | 0.591 | 61 | 1811.7 | 0.689 | 0.01 | **PASS** |
| WedThu_US_silence | 97 | 2239.5 | 0.68 | 34 | 1300.6 | 0.765 | 0.001 | **PASS** |
| WedThu_US_score3 | 79 | 1213.5 | 0.671 | 32 | 1090.0 | 0.75 | 0.011 | **PASS** |
| MonFri_silence | 203 | 6523.2 | 0.69 | 77 | 4084.5 | 0.701 | 0.0 | **PASS** |
| weekday_silence | 568 | 13121.1 | 0.637 | 165 | 6399.7 | 0.667 | 0.0 | **PASS** |
| weekend_silence | 156 | 771.6 | 0.571 | 29 | 989.0 | 0.897 | 0.081 | **ARTIFACT** |
| score3_silence | 440 | 11513.1 | 0.684 | 124 | 7294.5 | 0.831 | 0.0 | **PASS** |
| score3_US_silence | 254 | 8881.7 | 0.72 | 69 | 3449.0 | 0.812 | 0.0 | **PASS** |
| score3_biddep_silence | 66 | 1211.1 | 0.667 | 102 | 6952.2 | 0.882 | 0.004 | **PASS** |
| score3_US_biddep_silence | 34 | 718.4 | 0.735 | 52 | 3194.3 | 0.885 | 0.007 | **PASS** |
| score4_US_silence | 171 | 5934.1 | 0.731 | 55 | 3183.9 | 0.782 | 0.0 | **PASS** |
| US_score3_noisy_short | 189 | 347.0 | 0.487 | 153 | 8052.9 | 0.641 | 0.007 | **PASS** |
| WedThu_US_score3_short | 66 | -327.6 | 0.47 | 92 | 8439.7 | 0.783 | 0.089 | **ARTIFACT** |

## H. ETH 1h Bear + bid_nonzero + Silence Gate

| Signal | Cal N | Cal T3R | Cal win | Hold N | Hold T3R | Hold win | Perm p | Verdict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| eth1h_lt-50_silence | 353 | 8467.9 | 0.674 | 85 | 2920.7 | 0.788 | 0.0 | **PASS** |
| eth1h_lt-50_biddep_silence | 34 | 53.7 | 0.618 | 71 | 2678.3 | 0.831 | 0.134 | **ARTIFACT** |
| eth1h_lt-100_biddep_silence | 6 | None | 0.667 | 17 | 1007.6 | 0.824 | 0.0 | **PASS** |
| eth1h_lt-50_score3_silence | 275 | 8755.3 | 0.713 | 64 | 2610.9 | 0.844 | 0.0 | **PASS** |
| eth1h_lt-50_cluster_silence | 167 | 5967.1 | 0.766 | 59 | 2534.5 | 0.814 | 0.0 | **PASS** |
| eth1h_bear_noisy_short | 369 | 7518.6 | 0.558 | 223 | 5488.9 | 0.552 | 0.0 | **PASS** |
| eth1h_bear_sync300_noisy_short | 91 | 2575.4 | 0.571 | 98 | 5087.2 | 0.673 | 0.0 | **PASS** |
| eth1h_bull_silence | 78 | 1451.0 | 0.538 | 20 | 467.9 | 0.6 | 0.005 | **PASS** |
| eth1h_flat_silence | 40 | -460.1 | 0.55 | 13 | 383.5 | 0.615 | 0.408 | **ARTIFACT** |

## I. BULL_PULLBACK + noisy LONG (Full Analysis)

### cal: N_bull_noisy=75  N_bull_sil=67
Profile: prior4h=254.4 eth1h=169.4 btc4h=190.2 sync_k=140368.2 vdepth=34.2 thresh=88666.7
H2 LONG: N=75 T3R=-1740.2 med=-7.0 win=0.467 maxL=-387.6
H4 LONG: N=75 T3R=-2660.2 med=-15.1 win=0.44 maxL=-455.9
Low-sync (<200K) H2: N=57 T3R=-1987.9 med=-10.3 win=0.456 maxL=-387.6
High-sync (>=200K) H2: N=18 T3R=-599.6 med=4.7 win=0.5 maxL=-290.6

### hold: N_bull_noisy=11  N_bull_sil=11
Profile: prior4h=285.0 eth1h=131.0 btc4h=132.6 sync_k=121403.2 vdepth=21.9 thresh=113636.4
H2 LONG: N=11 T3R=92.9 med=9.9 win=0.909 maxL=-20.7
H4 LONG: N=11 T3R=-98.8 med=54.0 win=0.727 maxL=-175.0
Low-sync (<200K) H2: N=8 T3R=None med=8.8 win=0.875 maxL=-20.7
High-sync (>=200K) H2: N=3 T3R=None med=52.1 win=1.0 maxL=43.6

Perm (cal):  p=0.867 real=-1740.2 null_p95=599.4 N=75 -> **ARTIFACT**
Perm (hold): p=0.019 real=92.9 null_p95=-6.9 N=11 -> **PASS**

## J. Cross-Asset Silence (ETH + BTC both quiet)

**cal** rates: ETH_sil=0.516 BTC_sil=0.917 Both_sil=0.505
**hold** rates: ETH_sil=0.322 BTC_sil=0.796 Both_sil=0.317

| Signal | Cal N | Cal T3R | Cal win | Hold N | Hold T3R | Hold win | Perm p | Verdict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| eth_silence_only | 15 | -729.0 | 0.2 | 3 | None | 0.0 | 0.823 | **ARTIFACT** |
| btc_silence_only | 578 | -8079.3 | 0.434 | 288 | -5997.8 | 0.448 | 1.0 | **ARTIFACT** |
| both_silence | 709 | 15243.3 | 0.632 | 191 | 7810.5 | 0.712 | 0.0 | **PASS** |
| both_silence_score3 | 426 | 11798.6 | 0.7 | 124 | 7294.5 | 0.831 | 0.0 | **PASS** |
| both_silence_biddep | 123 | 1647.5 | 0.65 | 157 | 7523.3 | 0.752 | 0.011 | **PASS** |
| both_silence_score3_biddep | 63 | 1235.8 | 0.698 | 102 | 6952.2 | 0.882 | 0.004 | **PASS** |
| eth_sil_btc_noisy_short | 0 | None | None | 0 | None | None | 0.0 | **PASS** |
| btc_sil_eth_noisy_short | 515 | 720.6 | 0.52 | 278 | 1721.3 | 0.471 | 0.0 | **PASS** |
| neither_silence_short | 90 | 6042.5 | 0.667 | 119 | 8599.5 | 0.731 | 0.0 | **PASS** |

---
## Synthesis — Running Signal Registry

| # | Signal | Hold WR | Hold T3R | Perm Status |
| --- | --- | ---: | ---: | --- |
| 1 | Silence LONG (30min) | 70.1% | +7733 | p=0.0 PASS |
| 2 | Silence + sync>=200K LONG | 83.1% | +4298 | p=0.0 PASS x2 |
| 3 | noisy_NOT_bull SHORT | 54.9% | +11360 | p=0.0 PASS |
| 4 | prior4h_neg + silence LONG | 76.2% | +6741 | p=0.0 PASS |
| 5 | Combined portfolio | 59.9% | +19952 | p=0.0 PASS |
| ? | 5th signal candidate | ? | ? | See test D |

RESEARCH_ONLY. No live changes without explicit operator sign-off.
