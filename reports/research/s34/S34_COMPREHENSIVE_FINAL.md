# S34 Comprehensive Final Suite

Generated: `2026-06-30T10:20:40.182036+00:00`  |  Status: `RESEARCH_ONLY_NO_LIVE_CHANGE`
Cal: 1404 (2026-02-15T18:32:18Z to 2026-06-08T01:05:38Z)
Hold: 602 (2026-06-08T01:24:48Z to 2026-06-29T08:28:10Z)

## A. noisy_AND_NOT_bull SHORT Signal

Cal SHORT: 605 events, T3R=7778.7, med=9.1, win=0.542, maxL=-728.1
Hold SHORT: 397 events, T3R=11360.3, med=10.5, win=0.549, maxL=-275.3
Permutation null (cal): real T3R=7778.7, null p95=-3853.0, p-right=0.0 -> **PASS**

### By horizon
| Horizon | Cal N | Cal T3R | Cal med | Cal win | Hold N | Hold T3R | Hold med | Hold win |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| H1 | 605 | 8608.7 | 6.5 | 0.554 | 397 | 5807.3 | 4.1 | 0.529 |
| H2 | 605 | 7778.7 | 9.1 | 0.542 | 397 | 11360.3 | 10.5 | 0.549 |
| H4 | 603 | 5355.4 | -2.8 | 0.486 | 397 | 11680.5 | -0.3 | 0.496 |

### By cascade threshold
| Threshold | Cal N | Cal T3R | Cal win | Hold N | Hold T3R | Hold win |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| thr_50K | 465 | 5932.7 | 0.553 | 284 | 8331.5 | 0.549 |
| thr_100K | 347 | 4646.4 | 0.536 | 251 | 6697.0 | 0.546 |
| thr_200K | 140 | 883.4 | 0.507 | 113 | 1901.2 | 0.549 |

### SHORT + sync_k conditioning
| Gate | Cal N | Cal T3R | Cal win | Hold N | Hold T3R | Hold win |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| sync_lt_100K | 340 | -78.7 | 0.515 | 144 | 1337.8 | 0.535 |
| sync_lt_200K | 431 | 751.1 | 0.531 | 201 | 757.9 | 0.498 |
| sync_lt_300K | 494 | 3522.2 | 0.532 | 251 | 2146.4 | 0.502 |
| sync_lt_500K | 550 | 6502.4 | 0.54 | 291 | 4128.2 | 0.519 |

## B. 30-min Delayed Entry (Enter After Silence Confirmed)

### cal (N_silence=724)
- Price drift in silence window: median=22.4bps, mean=25.7bps, pct_up=0.758, pct_down=0.242
- Anchor 2h:      724 | T3R=14738.2 | med=22.1 | win=0.623
- Anchor 4h:      724 | T3R=13569.0 | med=21.5 | win=0.579
- Delayed 90min:  724 | T3R=-3302.7 | med=-2.2 | win=0.483
- Delayed 3.5h:   724 | T3R=-4662.2 | med=-3.2 | win=0.482

### hold (N_silence=194)
- Price drift in silence window: median=28.0bps, mean=35.1bps, pct_up=0.83, pct_down=0.17
- Anchor 2h:      194 | T3R=7733.7 | med=34.4 | win=0.701
- Anchor 4h:      194 | T3R=7888.9 | med=27.2 | win=0.68
- Delayed 90min:  194 | T3R=1106.0 | med=0.4 | win=0.505
- Delayed 3.5h:   194 | T3R=1287.7 | med=-2.6 | win=0.454

### Live rule (200K) with 30-min delayed entry
| Split | N | Delayed 3.5h T3R | Delayed 3.5h med | Delayed 3.5h win |
| --- | ---: | ---: | ---: | ---: |
| cal | 134 | -1208.3 | -4.9 | 0.47 |
| hold | 48 | 276.3 | -0.8 | 0.479 |

## C. sync_k Proxy Gate (Entry-Time Filter, No Waiting)

| Gate | Cal N | Cal T3R | Cal med | Cal win | Hold N | Hold T3R | Hold med | Hold win |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| skip_sync_gte_100K | 816 | 2465.5 | 3.4 | 0.511 | 247 | -1445.0 | -3.1 | 0.478 |
| skip_sync_gte_200K | 1036 | 3417.5 | 5.6 | 0.521 | 338 | -1212.9 | 3.4 | 0.515 |
| skip_sync_gte_300K | 1182 | 1933.9 | 5.7 | 0.522 | 415 | -105.2 | 2.2 | 0.52 |
| skip_sync_gte_500K | 1305 | -1245.4 | 5.7 | 0.522 | 477 | -1242.6 | 2.0 | 0.514 |
| skip_sync_gte_1000K | 1374 | -848.5 | 5.7 | 0.52 | 539 | -3653.3 | 1.2 | 0.508 |

### Live rule (200K) + sync_k proxy
| Gate | Cal N | Cal T3R | Cal win | Hold N | Hold T3R | Hold win |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| sync_lt_200K | 172 | 435.3 | 0.552 | 65 | -279.9 | 0.538 |
| sync_lt_300K | 209 | 748.8 | 0.565 | 90 | -239.6 | 0.522 |
| sync_lt_500K | 249 | 67.7 | 0.566 | 111 | -239.0 | 0.523 |

## D. Exit Management (Enter All, Exit Early on Noisy)

| Split | Managed T3R | Managed med | Managed win | Raw 4h T3R | Raw 4h med | Raw 4h win |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| cal all | -7223.7 | -11.8 | 0.443 | -1484.6 | 4.4 | 0.523 |
| cal live200K | -1212.9 | -10.2 | 0.455 | 226.2 | 15.8 | 0.552 |
| hold all | -2178.1 | -6.9 | 0.457 | -9086.4 | 4.5 | 0.53 |
| hold live200K | -1008.6 | -3.9 | 0.488 | -2030.8 | 14.4 | 0.561 |

## E. Silence Window Length Scan

| Window | Cal Silence N | Cal Silence rate | Cal T3R | Cal med | Cal win | Hold Silence N | Hold rate | Hold T3R | Hold med | Hold win |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 5min | 1120 | 0.798 | 7264.0 | 12.3 | 0.556 | 385 | 0.64 | -1646.9 | 7.8 | 0.54 |
| 10min | 981 | 0.699 | 8445.8 | 13.3 | 0.57 | 312 | 0.518 | 2142.1 | 12.7 | 0.583 |
| 15min | 910 | 0.648 | 10776.7 | 16.6 | 0.581 | 280 | 0.465 | 4208.6 | 18.5 | 0.607 |
| 20min | 852 | 0.607 | 12519.8 | 19.0 | 0.597 | 254 | 0.422 | 5485.3 | 20.6 | 0.626 |
| 30min | 724 | 0.516 | 14738.2 | 22.1 | 0.623 | 194 | 0.322 | 7733.7 | 34.4 | 0.701 |
| 45min | 614 | 0.437 | 16125.3 | 29.4 | 0.65 | 158 | 0.262 | 6517.1 | 34.4 | 0.728 |
| 60min | 523 | 0.373 | 13562.9 | 31.1 | 0.663 | 133 | 0.221 | 5943.5 | 38.7 | 0.797 |

## F. Silence + High Sync Permutation Null

### silence_sync_gte_200K
Cal: N=176 T3R=2900.8 med=21.5 win=0.636
Hold: N=65 T3R=4298.1 med=60.6 win=0.831
Perm (cal):  real=2900.8 null_p95=1082.1 p-right=0.001 -> **PASS**
Perm (hold): real=4298.1 null_p95=-131.4 p-right=0.0 -> **PASS**

### silence_sync_gte_300K
Cal: N=100 T3R=1397.6 med=24.6 win=0.67
Hold: N=40 T3R=1383.4 med=46.0 win=0.825

## G. Cascade Size Breakdown

| Key | Cal N | Cal T3R | Cal med | Cal win | Hold N | Hold T3R | Hold med | Hold win |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| thr_50K_silence | 351 | 6698.6 | 23.2 | 0.613 | 83 | 2670.5 | 38.7 | 0.711 |
| thr_50K_noisy | 299 | -8453.1 | -24.0 | 0.395 | 149 | -6775.1 | -19.9 | 0.383 |
| thr_50K_all | 650 | -691.2 | 5.2 | 0.512 | 232 | -3421.7 | 0.0 | 0.5 |
| thr_100K_silence | 239 | 3460.3 | 19.2 | 0.615 | 63 | 1955.9 | 37.0 | 0.698 |
| thr_100K_noisy | 229 | -7100.8 | -17.6 | 0.397 | 143 | -6821.3 | -18.8 | 0.385 |
| thr_100K_all | 468 | -2801.3 | 3.4 | 0.509 | 206 | -4282.7 | -2.7 | 0.481 |
| thr_200K+_silence | 134 | 2794.5 | 29.9 | 0.664 | 48 | 1553.4 | 24.5 | 0.688 |
| thr_200K+_noisy | 152 | -3778.2 | -8.6 | 0.454 | 116 | -4571.1 | -19.0 | 0.397 |
| thr_200K+_all | 286 | -155.9 | 12.0 | 0.552 | 164 | -2440.3 | -2.1 | 0.482 |

## H. Horizon Scan on Silence Events

| Horizon | Desc | Cal N | Cal T3R | Cal med | Cal win | Hold N | Hold T3R | Hold med | Hold win |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| H0_5 | 30min hold from anchor | 724 | 14461.0 | 17.4 | 0.709 | 194 | 5307.9 | 23.0 | 0.794 |
| H1 | 1h hold | 724 | 14552.6 | 20.2 | 0.667 | 194 | 5598.7 | 24.4 | 0.68 |
| H2 | 2h hold | 724 | 14738.2 | 22.1 | 0.623 | 194 | 7733.7 | 34.4 | 0.701 |
| H4 | 4h hold | 724 | 13569.0 | 21.5 | 0.579 | 194 | 7888.9 | 27.2 | 0.68 |
| H2_delayed | enter +30min, hold 90min total | 724 | -3302.7 | -2.2 | 0.483 | 194 | 1106.0 | 0.4 | 0.505 |
| H4_delayed | enter +30min, hold 3.5h total | 724 | -4662.2 | -3.2 | 0.482 | 194 | 1287.7 | -2.6 | 0.454 |

## I. Early Silence Signal (5-min Predictor)

**cal_prediction**: N_sil5=1120 (79.8%) | P(30min_sil | 5min_sil)=0.646 | P(30min_sil | 5min_noisy)=0.0
**hold_prediction**: N_sil5=385 (64.0%) | P(30min_sil | 5min_sil)=0.504 | P(30min_sil | 5min_noisy)=0.0

### 5-min silence -> 2h outcome
| Split | sil5+sil30 N | T3R | med | win | sil5+noisy30 N | T3R | med | win |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cal | 724 | 14738.2 | 22.1 | 0.623 | 396 | -8663.9 | -12.4 | 0.434 |
| hold | 194 | 7733.7 | 34.4 | 0.701 | 191 | -9993.5 | -15.8 | 0.377 |

*Interpretation*: If prob_30_given_5 >> base silence rate -> 5-min check is a good proxy. High predictability -> can enter after 5min instead of 30min.

## Key Verdicts

- A: noisy_NOT_bull SHORT — is it the other side of the silence trade?
- B: delayed entry — does the edge survive if we wait 30min?
- C: sync_k proxy — best entry-time filter without waiting?
- D: exit management — does early exit on noisy events improve overall?
- E: optimal window — what's the best silence confirmation time?
- F: high-sync silence — is WR 83% statistically real?
- G: size breakdown — which cascade size drives the silence alpha?
- H: best hold horizon for silence trades?
- I: 5-min early signal — can we enter after 5min instead of 30min?

RESEARCH_ONLY. All findings require permutation-null before live promotion.
