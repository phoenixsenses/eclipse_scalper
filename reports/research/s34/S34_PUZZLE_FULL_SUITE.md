# S34 Puzzle Full Suite — 6 Tests

Generated: `2026-06-30T10:06:08.535503+00:00`  |  Status: `RESEARCH_ONLY_NO_LIVE_CHANGE`

Cal: 1404 events (2026-02-15T18:32:18Z to 2026-06-08T01:05:38Z)
Hold: 602 events (2026-06-08T01:24:48Z to 2026-06-29T08:28:10Z)

## Test 1: Silence Gate Holdout

| Split | Silence N | Silence rate | Silence T3R | Silence med | Silence win | Noisy T3R | Noisy med | Noisy win |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cal | 724 | 0.516 | 14738.2 | 22.1 | 0.623 | -17312.7 | -17.6 | 0.409 |
| hold | 194 | 0.322 | 7733.7 | 34.4 | 0.701 | -17050.6 | -19.4 | 0.387 |

## Test 2: sync_k Threshold Scan

| Gate | Cal N | Cal T3R | Cal med | Cal win | Hold N | Hold T3R | Hold med | Hold win |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| all | 1404 | -1384.8 | 5.7 | 0.519 | 602 | -8574.9 | -2.0 | 0.488 |
| sync_lt_50K | 625 | 2227.8 | 5.6 | 0.522 | 196 | -1119.2 | 1.2 | 0.505 |
| sync_lt_100K | 816 | 2465.5 | 3.4 | 0.511 | 247 | -1445.0 | -3.1 | 0.478 |
| sync_lt_150K | 926 | 1810.8 | 4.6 | 0.518 | 287 | -1524.0 | -3.1 | 0.484 |
| sync_lt_200K | 1036 | 3417.5 | 5.6 | 0.521 | 338 | -1212.9 | 3.4 | 0.515 |
| sync_lt_300K | 1182 | 1933.9 | 5.7 | 0.522 | 415 | -105.2 | 2.2 | 0.52 |
| sync_lt_500K | 1305 | -1245.4 | 5.7 | 0.522 | 477 | -1242.6 | 2.0 | 0.514 |

## Test 3: BULL_PULLBACK Permutation Null + Anatomy

- Cal BULL_PULLBACK: N=142, T3R=1243.6, med=12.4, win=0.542
- Hold BULL_PULLBACK: N=22, T3R=665.3, med=46.7, win=0.909
- Cal NON-BULL_PULLBACK: N=1262, T3R=-3702.3, med=4.7

**Permutation null (cal, 1000 shuffles):** real T3R=1243.6, null p95=958.8, p-right=0.03 -> **PASS**

### Anatomy — cal feature medians

| Feature | BULL_PULLBACK | NON-BULL |
| --- | ---: | ---: |
| prior4h_bps | 257.4 | -59.1 |
| vdepth_bps | 20.9 | 19.4 |
| bid_depth_usd | 0.0 | 0.0 |
| book_imbalance | 0.1 | 0.1 |
| eth1h_bps | 127.3 | -59.6 |
| btc4h_bps | 160.9 | -51.3 |
| threshold_usd | 50000.0 | 100000.0 |
| sync_k | 37.3 | 71.3 |

### prior4h gate within hold BULL_PULLBACK subset

| Gate | Hold N | Hold T3R | Hold med | Hold win |
| --- | ---: | ---: | ---: | ---: |
| prior4h_gt_-100 | 22 | 665.3 | 46.7 | 0.909 |
| prior4h_gt_-50 | 22 | 665.3 | 46.7 | 0.909 |
| prior4h_gt_0 | 22 | 665.3 | 46.7 | 0.909 |
| prior4h_gt_25 | 22 | 665.3 | 46.7 | 0.909 |
| prior4h_gt_50 | 22 | 665.3 | 46.7 | 0.909 |
| prior4h_gt_100 | 21 | 658.4 | 49.8 | 0.905 |

## Test 4: prior4h_bps Trend Gate Holdout

| Gate | Cal N | Cal T3R | Cal med | Cal win | Hold N | Hold T3R | Hold med | Hold win |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| prior4h_gt_-100 | 934 | 917.8 | 8.4 | 0.527 | 374 | -8274.8 | -8.6 | 0.444 |
| prior4h_gt_-50 | 699 | 5285.3 | 10.2 | 0.535 | 253 | -5483.2 | -9.0 | 0.435 |
| prior4h_gt_0 | 475 | 5114.1 | 10.6 | 0.543 | 162 | -4038.7 | -8.0 | 0.444 |
| prior4h_gt_25 | 389 | 3335.3 | 9.3 | 0.53 | 125 | -3294.2 | -0.6 | 0.488 |
| prior4h_gt_50 | 338 | 2449.4 | 9.3 | 0.524 | 90 | -1889.4 | 8.2 | 0.578 |
| prior4h_gt_100 | 208 | 824.9 | 5.7 | 0.514 | 61 | -1145.3 | 10.5 | 0.623 |
| prior4h_gt_0_AND_sync_lt_200K | 389 | 5046.1 | 11.9 | 0.542 | 110 | -3890.9 | -19.4 | 0.382 |
| prior4h_gt_0_AND_sync_lt_100K | 329 | 6205.5 | 11.9 | 0.55 | 93 | -3629.6 | -23.5 | 0.344 |
| prior4h_gt_25_AND_sync_lt_200K | 324 | 3748.0 | 11.7 | 0.531 | 82 | -3320.8 | -19.4 | 0.415 |

## Test 5: KNN Augmented (+ sync_k feature)

| Pattern | Cal N | Cal T3R | Cal med | Hold N | Hold T3R | Hold med | Hold win |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| k5_CLEAN_NORMAL | 156 | 7939.9 | 36.8 | 78 | -4462.2 | -40.7 | 0.295 |
| k5_DANGER_REVERSE | 226 | 2292.2 | -7.2 | 80 | -296.7 | -21.0 | 0.438 |
| k8_CLEAN_NORMAL | 172 | 6954.0 | 35.2 | 78 | -5098.0 | -58.0 | 0.282 |
| k8_DANGER_REVERSE | 393 | 2475.3 | -7.4 | 106 | -115.4 | -19.1 | 0.415 |

## Test 6: Weekly Breakdown

(*) = holdout

| Week | Hold? | All N | All med | All win | k5=CLEAN N | CLEAN med | CLEAN win | Mean sync_k (K) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-02-09 |  | 12 | 76.2 | 0.833 | 0 | None | None | 61.6 |
| 2026-02-16 |  | 112 | 17.9 | 0.58 | 11 | 35.5 | 0.636 | 405.3 |
| 2026-02-23 |  | 222 | -25.0 | 0.41 | 24 | 37.3 | 0.917 | 158.7 |
| 2026-03-02 |  | 204 | 5.2 | 0.525 | 21 | 43.7 | 0.81 | 227.6 |
| 2026-03-09 |  | 137 | 34.7 | 0.613 | 13 | 44.8 | 0.923 | 127.7 |
| 2026-03-16 |  | 170 | -4.2 | 0.471 | 12 | -0.2 | 0.5 | 152.4 |
| 2026-03-23 |  | 106 | -0.6 | 0.491 | 8 | 16.7 | 0.625 | 181.7 |
| 2026-03-30 |  | 100 | 10.9 | 0.55 | 9 | 23.1 | 0.667 | 138.9 |
| 2026-04-06 |  | 63 | -7.6 | 0.476 | 6 | 181.4 | 1.0 | 153.5 |
| 2026-04-13 |  | 167 | -6.8 | 0.467 | 21 | 18.8 | 0.667 | 207.2 |
| 2026-04-20 |  | 84 | 27.2 | 0.738 | 12 | 27.8 | 0.667 | 236.4 |
| 2026-04-27 |  | 3 | -125.2 | 0.0 | 0 | None | None | 70.0 |
| 2026-06-01 |  | 23 | 18.2 | 0.652 | 3 | 183.0 | 1.0 | 190.3 |
| 2026-06-08 |  | 195 | 0.7 | 0.503 | 23 | -9.0 | 0.435 | 378.4 |
| 2026-06-15 | Y | 181 | -0.6 | 0.497 | 25 | -58.0 | 0.36 | 450.7 |
| 2026-06-22 | Y | 211 | -5.7 | 0.445 | 35 | -51.6 | 0.343 | 868.7 |
| 2026-06-29 | Y | 16 | 45.2 | 0.75 | 1 | 50.2 | 1.0 | 271.8 |

## Summary Verdict

- Test 1: silence gate — does no-propagation still predict fade in hold?
- Test 2: at what sync_k threshold does holdout T3R flip positive?
- Test 3: BULL_PULLBACK — permutation result + anatomy reveals the knowable gate
- Test 4: prior4h trend filter + sync_k combo — simplest possible gate
- Test 5: augmented KNN — does adding sync_k as feature rescue holdout?
- Test 6: exact week of regime break visible in sync_k spike

RESEARCH_ONLY. No live/paper promotion without OOS permutation null.
