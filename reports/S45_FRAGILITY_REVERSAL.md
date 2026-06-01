# S45 — Fragility Zone Reversal

**Hypothesis**: GOLDILOCKS cascade following EXTREME/COLD zone is especially high quality.
Total signals: 73, GOLDILOCKS: 34

## Transition Analysis

| transition | N | WR@60s | WR@120s | mean_ret@60s |
|---|---:|---:|---:|---:|
| GOLDILOCKS (all) | 34 | 55.9% | 47.1% | +2.75 bps |
| prev=EXTREME → GOLDILOCKS | 8 | 25.0% | 12.5% | -6.05 bps |
| prev=COLD → GOLDILOCKS | 7 | 42.9% | 28.6% | -0.20 bps |
| prev=EXTREME/COLD → GOLDILOCKS | 15 | 33.3% | 20.0% | -3.32 bps |
| prev=GOLDILOCKS → GOLDILOCKS | 18 | 77.8% | 72.2% | +8.41 bps |
| EXTREME (all) | 17 | 58.8% | 64.7% | +6.49 bps |

## Full Transition Matrix (WR@60s)

| prev→curr | curr=COLD | curr=GOLD | curr=OHEAT | curr=EXTREME |
|---|---|---|---|---|
| prev=COLD | 70%(N=10) | 43%(N=7) | — | 0%(N=2) |
| prev=GOLD | 60%(N=5) | 78%(N=18) | 100%(N=1) | 70%(N=10) |
| prev=OHEAT | 100%(N=1) | — | — | 100%(N=1) |
| prev=EXTREME | 75%(N=4) | 25%(N=8) | 0%(N=1) | 50%(N=4) |
| prev=first | — | 0%(N=1) | — | — |

## Streak Analysis

2 consecutive bad zones → GOLDILOCKS: N=6, WR@60s=33.3%

EXTREME/COLD → GOLDILOCKS + HEALTHY + sr>2.0: N=1, WR=100.0%

## Verdict: NO EDGE
prev=EXTREME/COLD → GOLDILOCKS: WR=33.3% vs GOLDILOCKS flat=55.9% (delta=-22.5pp, N=15)