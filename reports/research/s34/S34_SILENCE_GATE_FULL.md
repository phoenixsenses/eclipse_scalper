# S34 Silence Gate Full Suite

Generated: `2026-06-30T10:10:41.584408+00:00`  |  Status: `RESEARCH_ONLY_NO_LIVE_CHANGE`

Cal: 1404 events (2026-02-15T18:32:18Z to 2026-06-08T01:05:38Z)
Hold: 602 events (2026-06-08T01:24:48Z to 2026-06-29T08:28:10Z)

## A. Silence Gate Permutation Null

### cal split (N_silence=724, N_noisy=680)

- Real T3R=14738.2  |  Null p95=2011.3  |  Null p99=3330.8
- p-right=0.0  ->  **PASS**
- Silence: N=724 T3R=14738.2 med=22.1 win=0.623 maxL=-280.1
- Noisy:   N=680 T3R=-17312.7 med=-17.6 win=0.409 maxL=-387.6

### hold split (N_silence=194, N_noisy=408)

- Real T3R=7733.7  |  Null p95=-1157.1  |  Null p99=-162.2
- p-right=0.0  ->  **PASS**
- Silence: N=194 T3R=7733.7 med=34.4 win=0.701 maxL=-312.0
- Noisy:   N=408 T3R=-17050.6 med=-19.4 win=0.387 maxL=-455.2

## B. Silence + BULL_PULLBACK Combo

| Group | Cal N | Cal T3R | Cal med | Cal win | Hold N | Hold T3R | Hold med | Hold win | Hold maxL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| silence_AND_bull | 67 | 2006.1 | 35.5 | 0.627 | 11 | None | 72.3 | 0.909 | -53.3 |
| silence_AND_NOT_bull | 657 | 11737.7 | 21.9 | 0.623 | 183 | 6813.2 | 29.7 | 0.689 | -312.0 |
| noisy_AND_bull | 75 | -1740.2 | -7.0 | 0.467 | 11 | None | 9.9 | 0.909 | -20.7 |
| noisy_AND_NOT_bull | 605 | -16624.2 | -19.1 | 0.402 | 397 | -17315.6 | -20.5 | 0.373 | -455.2 |
| silence_only | 724 | 14738.2 | 22.1 | 0.623 | 194 | 7733.7 | 34.4 | 0.701 | -312.0 |
| bull_only | 142 | 1243.6 | 12.4 | 0.542 | 22 | 665.3 | 46.7 | 0.909 | -53.3 |

## C. sync_k as Predictor of Silence (Live Proxy)

| sync_k bin | Split | N | Silence rate | Silence T3R | Silence med | Silence win | All T3R |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| sync_0K_to_50K | cal | 625 | 0.536 | 6980.3 | 19.2 | 0.588 | 2227.8 |
| sync_0K_to_50K | hold | 196 | 0.418 | 1310.3 | 12.7 | 0.598 | -1119.2 |
| sync_50K_to_100K | cal | 191 | 0.513 | 720.9 | 21.2 | 0.602 | -950.4 |
| sync_50K_to_100K | hold | 51 | 0.294 | 357.8 | 49.8 | 0.8 | -845.6 |
| sync_100K_to_200K | cal | 220 | 0.523 | 2157.2 | 33.5 | 0.722 | 395.3 |
| sync_100K_to_200K | hold | 91 | 0.352 | -17.9 | 19.8 | 0.656 | -448.8 |
| sync_200K_to_300K | cal | 146 | 0.521 | 1053.2 | 13.0 | 0.592 | -2300.3 |
| sync_200K_to_300K | hold | 77 | 0.325 | 2142.6 | 147.4 | 0.84 | 321.2 |
| sync_300K_to_500K | cal | 123 | 0.504 | 869.0 | 32.2 | 0.71 | -3650.9 |
| sync_300K_to_500K | hold | 62 | 0.339 | 697.0 | 47.6 | 0.905 | -1789.8 |
| sync_500K_to_1000K | cal | 69 | 0.348 | 223.3 | 31.7 | 0.708 | -603.7 |
| sync_500K_to_1000K | hold | 62 | 0.226 | None | 72.3 | 0.857 | -3157.3 |
| sync_1000K_to_infK | cal | 30 | 0.467 | None | -20.0 | 0.429 | -1012.4 |
| sync_1000K_to_infK | hold | 63 | 0.079 | None | -22.8 | 0.4 | -5366.2 |

## D. Live Rule (200K threshold) Silence Analysis

| Split | N live | Silence N | Silence rate | Silence T3R | Silence med | Silence win | Noisy T3R | Noisy med | All T3R |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cal | 286 | 134 | 0.469 | 2794.5 | 29.9 | 0.664 | -3778.2 | -8.6 | -155.9 |
| hold | 164 | 48 | 0.293 | 1553.4 | 24.5 | 0.688 | -4571.1 | -19.0 | -2440.3 |

## E. Silence by sync_k Level

| Gate | Cal N | Cal T3R | Cal med | Cal win | Hold N | Hold T3R | Hold med | Hold win | Hold maxL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| silence_AND_sync_lt_50K | 335 | 6980.3 | 19.2 | 0.588 | 82 | 1310.3 | 12.7 | 0.598 | -154.0 |
| silence_AND_sync_lt_100K | 433 | 8500.1 | 19.2 | 0.591 | 97 | 2187.9 | 17.4 | 0.629 | -154.0 |
| silence_AND_sync_lt_200K | 548 | 11195.5 | 22.5 | 0.619 | 129 | 2653.8 | 19.3 | 0.636 | -312.0 |
| silence_AND_sync_lt_300K | 624 | 12735.4 | 21.9 | 0.615 | 154 | 5578.2 | 27.7 | 0.669 | -312.0 |
| silence_AND_sync_lt_500K | 686 | 14035.5 | 21.9 | 0.624 | 175 | 6754.1 | 34.4 | 0.697 | -312.0 |
| silence_AND_sync_gte_200K | 176 | 2900.8 | 21.5 | 0.636 | 65 | 4298.1 | 60.6 | 0.831 | -227.3 |
| silence_AND_sync_gte_300K | 100 | 1397.6 | 24.6 | 0.67 | 40 | 1383.4 | 46.0 | 0.825 | -227.3 |

## F. prior4h Trend + Silence Combo

| Gate | Cal N | Cal T3R | Cal med | Cal win | Hold N | Hold T3R | Hold med | Hold win | Hold maxL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| silence_AND_prior4h_gt_-50 | 377 | 7742.2 | 24.2 | 0.605 | 101 | 1385.9 | 15.3 | 0.604 | -154.0 |
| silence_AND_prior4h_gt_0 | 239 | 4812.4 | 26.2 | 0.603 | 72 | 472.2 | 12.9 | 0.597 | -120.1 |
| silence_AND_prior4h_gt_25 | 194 | 3280.2 | 25.4 | 0.582 | 53 | 502.6 | 20.2 | 0.66 | -120.1 |
| silence_AND_prior4h_gt_50 | 170 | 2754.6 | 13.0 | 0.571 | 42 | 837.2 | 26.1 | 0.762 | -120.1 |
| silence_AND_prior4h_gt_100 | 104 | 1144.4 | 9.9 | 0.577 | 27 | 588.9 | 44.4 | 0.815 | -120.1 |
| silence_prior4h_gt_0_sync_lt_200K | 206 | 4084.1 | 29.8 | 0.602 | 46 | -888.4 | -18.5 | 0.435 | -120.1 |
| silence_prior4h_gt_0_sync_lt_300K | 218 | 3987.3 | 26.1 | 0.587 | 50 | -575.3 | -12.6 | 0.46 | -120.1 |
| silence_prior4h_gt_25_sync_lt_200K | 169 | 2752.9 | 28.9 | 0.58 | 31 | -691.4 | -4.0 | 0.484 | -120.1 |

## Key Questions Answered

- A: Is silence gate statistically real on holdout? (p-right < 0.05 = YES)
- B: Does silence+BULL_PULLBACK combo beat silence alone?
- C: Can sync_k at entry predict whether silence will occur? (live proxy)
- D: Does the 200K live-rule specific silence gate work in holdout?
- E: Does silence work better when sync_k is low? (regime interaction)
- F: Does prior4h trend + silence combo improve hold signal?

RESEARCH_ONLY. No live change without operator sign-off.
