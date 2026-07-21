# S34 — 45m Scalp + All Horizons · CAUSAL vs LOOKAHEAD (echo)

_2026-07-20T14:39:00.043004+00:00 · READ-ONLY · anchors=697 · 5.16 mo · FEE=5bps_

> **LOOKAHEAD** arm keeps the frozen `not noisy` gate (inspects T0+60s→T0+30m = future); **CAUSAL** drops it. The gap = the lookahead's contribution (hindsight tail removal, §165). For holds ≤30m the not-noisy window even extends past exit. echo-only. CAN kill, CANNOT bless.

Kolon: N, WR, avg(net), worst, tail(<-100), mc_p, noovN.

## CAUSAL

| hold | N | WR | avg | worst | tail | mc_p | noovN |
|---|---:|---:|---:|---:|---:|---:|---:|
| 30m | 118 | 54.2% | -1.8 | -460.4 | 8 | 0.602 | 104 |
| 45m | 118 | 61.9% | +4.7 | -447.1 | 9 | 0.282 | 91 |
| 2h | 118 | 60.2% | +9.5 | -395.4 | 11 | 0.159 | 73 |
| 4h | 118 | 69.5% | +41.2 | -338.9 | 14 | 0.001 | 63 |
| 6h | 118 | 69.5% | +49.0 | -412.4 | 12 | 0.0 | 57 |
| 12h | 118 | 62.7% | +46.0 | -419.9 | 20 | 0.005 | 52 |
| 24h | 118 | 67.8% | +92.3 | -553.8 | 24 | 0.0 | 44 |
| 48h | 118 | 61.9% | +63.9 | -795.3 | 34 | 0.009 | 29 |

## LOOKAHEAD

| hold | N | WR | avg | worst | tail | mc_p | noovN |
|---|---:|---:|---:|---:|---:|---:|---:|
| 30m | 38 | 89.5% | +41.7 | -25.5 | 0 | 0.0 | 38 |
| 45m | 38 | 84.2% | +41.5 | -42.9 | 0 | 0.0 | 36 |
| 2h | 38 | 86.8% | +48.7 | -64.0 | 0 | 0.0 | 32 |
| 4h | 38 | 81.6% | +87.8 | -85.9 | 0 | 0.0 | 31 |
| 6h | 38 | 84.2% | +100.6 | -119.7 | 1 | 0.0 | 31 |
| 12h | 38 | 84.2% | +123.1 | -150.3 | 2 | 0.0 | 29 |
| 24h | 38 | 84.2% | +166.2 | -344.0 | 3 | 0.0 | 26 |
| 48h | 38 | 57.9% | +94.9 | -486.7 | 12 | 0.033 | 21 |

## REMOVED_by_noisy

| hold | N | WR | avg | worst | tail | mc_p | noovN |
|---|---:|---:|---:|---:|---:|---:|---:|
| 30m | 80 | 37.5% | -22.5 | -460.4 | 8 | 0.996 | 69 |
| 45m | 80 | 51.2% | -12.8 | -447.1 | 9 | 0.885 | 63 |
| 2h | 80 | 47.5% | -9.2 | -395.4 | 11 | 0.765 | 54 |
| 4h | 80 | 63.8% | +19.0 | -338.9 | 14 | 0.089 | 49 |
| 6h | 80 | 62.5% | +24.4 | -412.4 | 11 | 0.044 | 44 |
| 12h | 80 | 52.5% | +9.4 | -419.9 | 18 | 0.34 | 39 |
| 24h | 80 | 60.0% | +57.2 | -553.8 | 21 | 0.033 | 34 |
| 48h | 80 | 63.8% | +49.1 | -795.3 | 22 | 0.07 | 25 |

## Read
- 45m scalp: if CAUSAL avg≈0/neg or tail-heavy → the quick exit doesn't harvest the bounce (cascade rebound is slower than 45m) — scalp dead. If LOOKAHEAD >> CAUSAL at 45m, the pretty number is hindsight, not a scalp edge.
- Compare LOOKAHEAD−CAUSAL per horizon: a large positive gap = lookahead doing the work (esp. tail/worst). Forward is the only proof; no tuning here.
