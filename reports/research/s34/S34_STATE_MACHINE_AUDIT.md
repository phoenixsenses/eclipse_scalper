# S34 State Machine Pre-Live Audit

Generated: `2026-06-30T15:16:42.711984+00:00`

Events: 450 classified (>=200K, finite net_2h) from 2006 total. Holdout: 135.

## A. Event Distribution

Daily average: **5.2/day** total  |  SILENCE 1.9/day  |  NEITHER 0.8/day

| State | N total | N holdout | Hold WR (LONG) | Hold WR (SHORT) |
| --- | ---: | ---: | ---: | ---: |
| NEITHER | 68 | 34 | 29.4% | 76.5% |
| NEITHER_BULL | 2 | 0 | N/A | N/A |
| NOISY | 185 | 58 | 48.3% | 48.3% |
| NOISY_BULL | 13 | 2 | 100.0% | N/A |
| SILENCE | 167 | 38 | 73.7% | 23.7% |
| SILENCE_BULL | 15 | 3 | 100.0% | N/A |

## B. Ultra-Early BTC Trap (NEITHER SHORT)

NEITHER total: 68  |  Ultra-early BTC (<60s): 10  |  Normal: 58

| Condition | N holdout | WR | Mean bps | T3R |
| --- | ---: | ---: | ---: | ---: |
| Ultra-early BTC (<60s) | 7 | 85.7% | +83.1 | -43 |
| Normal BTC timing | 27 | 74.1% | +86.7 | +1255 |

BTC cascade delay: {'<2min': 13, '2-5min': 16, '5-15min': 25, '15-30min': 14}  median=5.7min

## C. LONG + NEITHER Overlap (Flip vs Hold)

SILENCE events: 167  |  With overlapping NEITHER in 4h: 12 (7.2%)

| Scenario | N holdout | WR | Mean bps | T3R |
| --- | ---: | ---: | ---: | ---: |
| HOLD_LONG | 3 | 66.7% | -90.5 | -271 |
| FLIP_SHORT | 3 | 66.7% | +210.8 | +632 |

## D. Score Filter for NEITHER SHORT

Score = n2h>=3 + btc4h<0 + vdepth>=30 + US_session + sync_k>=200K (max=5, sil_eth always 0)

### By exact score

| Score | N all | WR all | N hold | WR hold |
| --- | ---: | ---: | ---: | ---: |
| 1 | 6 | 50.0% | 2 | 50.0% |
| 2 | 11 | 54.5% | 4 | 50.0% |
| 3 | 26 | 76.9% | 12 | 75.0% |
| 4 | 19 | 78.9% | 12 | 83.3% |
| 5 | 6 | 100.0% | 4 | 100.0% |

### Cumulative (score >= threshold)

| Score >= | N all | WR all | N hold | WR hold | T3R hold |
| --- | ---: | ---: | ---: | ---: | ---: |
| >=0 | 68 | 73.5% | 34 | 76.5% | +1836 |
| >=1 | 68 | 73.5% | 34 | 76.5% | +1836 |
| >=2 | 62 | 75.8% | 32 | 78.1% | +1775 |
| >=3 | 51 | 80.4% | 28 | 82.1% | +1870 |
| >=4 | 25 | 84.0% | 16 | 87.5% | +1109 |
| >=5 | 6 | 100.0% | 4 | 100.0% | +28 |
