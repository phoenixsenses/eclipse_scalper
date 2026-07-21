# Echo — Causal (no-lookahead) vs Frozen (with-lookahead) Diagnostic

_2026-07-20T10:01:08.845691+00:00 · READ-ONLY · anchors=695 · 5.15 mo · FEE=5bps_

> **CAN kill, CANNOT bless.** Only the `not noisy` (T+30m lookahead) gate is toggled; all other gates (echo_30_90, regime, session, dow, bull) held fixed. A positive causal arm is necessary-not-sufficient — the other gates are also in-sample. Proof is FORWARD only.

Kolon: N, /ay, WR, Avg(net bps), Worst, Tail(<-100), mc_p, WF.

## T0 hold 4h

| arm | N | /ay | WR | Avg | Worst | Tail | mc_p | WF |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FULL (with lookahead) | 38 | 7.4 | 81.6% | +87.8 | -85.9 | 0 | 0.0 | 5/5 |
| CAUSAL (no lookahead) | 118 | 22.9 | 69.5% | +41.2 | -338.9 | 14 | 0.001 | 5/5 |
| REMOVED by noisy gate | 80 | 15.5 | 63.8% | +19.0 | -338.9 | 14 | 0.089 | 3/5 |

No-overlap (realistic, hold=4h dedup):

| arm | noov N | noov /ay | noov WR | noov sum |
|---|---:|---:|---:|---:|
| FULL | 31 | 6.0 | 77.4% | 2584.5 |
| CAUSAL | 63 | 12.2 | 61.9% | 560.5 |
| REMOVED | 49 | 9.5 | 57.1% | -493.8 |

## T0 hold 6h

| arm | N | /ay | WR | Avg | Worst | Tail | mc_p | WF |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FULL (with lookahead) | 38 | 7.4 | 84.2% | +100.6 | -119.7 | 1 | 0.0 | 5/5 |
| CAUSAL (no lookahead) | 118 | 22.9 | 69.5% | +49.0 | -412.4 | 12 | 0.0 | 5/5 |
| REMOVED by noisy gate | 80 | 15.5 | 62.5% | +24.4 | -412.4 | 11 | 0.044 | 4/5 |

No-overlap (realistic, hold=4h dedup):

| arm | noov N | noov /ay | noov WR | noov sum |
|---|---:|---:|---:|---:|
| FULL | 31 | 6.0 | 80.6% | 2948.3 |
| CAUSAL | 63 | 12.2 | 58.7% | 1335.3 |
| REMOVED | 49 | 9.5 | 55.1% | 252.4 |

## Read
- If CAUSAL avg/sum collapses vs FULL **and** REMOVED-by-noisy is disproportionately
  negative/tail-heavy → the lookahead was doing the work (hindsight tail removal). KILL.
- If CAUSAL holds up → necessary-not-sufficient; proceed to FORWARD accumulation + snapshot
  enrichment (dev-list #15/#19). Still no deploy, no tuning on this burned sample.
