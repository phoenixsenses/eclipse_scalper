# S44 — BTC Contagion Filter

**Hypothesis**: Isolated ETH cascades revert more reliably than BTC-contagion cascades.
isolated: N=67, contagion: N=6

## Forward Return by Isolation State

| group | h | N | WR | mean_ret |
|---|---:|---:|---:|---:|
| isolated | 30s | 67 | 58.2% | +0.33 bps |
| contagion | 30s | 6 | 66.7% | +7.34 bps |
| **delta** | 30s | — | **-8.5pp** | — |
| isolated | 60s | 67 | 59.7% | +3.55 bps |
| contagion | 60s | 6 | 66.7% | +9.79 bps |
| **delta** | 60s | — | **-7.0pp** | — |
| isolated | 120s | 67 | 53.7% | +0.37 bps |
| contagion | 120s | 6 | 66.7% | +20.19 bps |
| **delta** | 120s | — | **-12.9pp** | — |

## Combined Filters (isolated)

| condition | N | WR@60s |
|---|---:|---:|
| isolated only | 67 | 59.7% |
| isolated + GOLDILOCKS | 32 | 56.2% |
| isolated + GOLDILOCKS + HEALTHY | 7 | 85.7% |
| isolated + GOLDILOCKS + HEALTHY + sr>2.0 | 0 | nan% |

## Isolated Signals by Regime

| regime | N | WR@60s |
|---|---:|---:|
| HEALTHY | 12 | 83.3% |
| NULL | 41 | 56.1% |

## Verdict: NO EDGE
isolated WR=59.7% vs contagion WR=66.7% (delta=-7.0pp)