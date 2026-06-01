# S49 — Single Large Liquidation Validation

## Part 1: Fill-Model WR from PKL (true fill simulation)

Matched 30/52 pkl signals to detector_signals.

| composition | N | WR | mean_ret |
|---|---:|---:|---:|
| single_large | 13 | 84.6% | +1486.11 bps |
| clustered | 17 | 52.9% | +681.04 bps |

## Part 2: Passive Limit Simulation (all 21 single_large signals)

Hold window: 120s after fill, fee=2.0 bps/side
Fill rate: 10/21 = 47.6%

| metric | value |
|---|---:|
| N filled | 10 |
| WR (net) | 80.0% |
| mean gross | +7.11 bps |
| mean net | +5.11 bps |
| NPA per attempt | +2.43 bps |

### By Fragility Zone (filled signals)

| zone | N | WR | net_NPA |
|---|---:|---:|---:|
| COLD | 5 | 80.0% | +6.43 bps |
| GOLDILOCKS | 2 | 100.0% | +10.47 bps |
| EXTREME | 3 | 66.7% | -0.67 bps |

### Individual fills

| date | zone | regime | fill | exit | gross | net | result |
|---|---|---|---:|---:|---:|---:|---|
| 02-26 | COLD | None | 2073.53 | 2075.09 | -7.52 | -9.52 | **L** |
| 03-02 | GOLDILOCKS | HEALTHY | 2072.69 | 2070.71 | +9.53 | +7.53 | **W** |
| 03-04 | COLD | None | 2184.90 | 2178.60 | +28.85 | +26.85 | **W** |
| 03-09 | COLD | HEALTHY | 2026.23 | 2023.72 | +12.39 | +10.39 | **W** |
| 03-25 | GOLDILOCKS | DEGRADED | 2186.67 | 2183.30 | +15.41 | +13.41 | **W** |
| 03-26 | EXTREME | DEGRADED | 2074.27 | 2071.95 | +11.18 | +9.18 | **W** |
| 03-30 | EXTREME | None | 2017.38 | 2011.50 | +29.12 | +27.12 | **W** |
| 04-11 | EXTREME | DEGRADED | 2288.46 | 2296.77 | -36.31 | -38.31 | **L** |
| 04-13 | COLD | None | 2197.88 | 2197.22 | +2.98 | +0.98 | **W** |
| 04-14 | COLD | None | 2374.53 | 2373.23 | +5.47 | +3.47 | **W** |

## Verdict: GO — single_large NPA per attempt: +2.43 bps (fee-adjusted positive)