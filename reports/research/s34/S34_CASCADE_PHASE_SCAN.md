# S34 Cascade Phase Scan

Generated: 2026-06-20T09:11:16.640468+00:00

Scope: ETH BUY feature-factory events, route `LONG_DELAY0_TP60`. No live runner/config changes.

Phase features are no-lookahead: only liquidation flow before the cluster timestamp is used.

## Phase Label Distribution

| Phase | N | Median | Mean | Cum | WR | Top3 Removed | Days | Exits |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| early_impulse | 7 | +53.84 | +30.78 | +215.48 | 71.4% | +49.57 | 6 | {'BE': 1, 'SL': 1, 'TP': 5} |
| fresh_start | 434 | -7.76 | +13.39 | +5810.40 | 49.1% | +5607.52 | 77 | {'BE': 129, 'SL': 74, 'TIME': 37, 'TP': 194} |
| late_saturated | 3 | +53.51 | +32.01 | +96.04 | 66.7% | +0.00 | 3 | {'BE': 1, 'TP': 2} |
| mid_cascade | 6 | +22.02 | +17.73 | +106.36 | 50.0% | -68.09 | 6 | {'BE': 2, 'SL': 1, 'TP': 3} |

## OOS Phase Candidates

| Rank | Candidate | Train N | Train Median | Train Cum | Test N | Test Median | Test Mean | Test Cum | Test Top3 Removed | Test Positive Days |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | prior15_ge_p75_109044 | 55 | +52.38 | +1108.55 | 58 | +52.17 | +17.46 | +1012.48 | +828.64 | 17/27 |
| 2 | prior15_clusters_eq_0 AND prior15_ge_p75_109044 | 55 | +52.38 | +1108.55 | 58 | +52.17 | +17.46 | +1012.48 | +828.64 | 17/27 |
| 3 | phase_fresh_start AND prior15_ge_p75_109044 | 52 | +52.36 | +1009.46 | 45 | +21.37 | +15.42 | +693.69 | +511.49 | 13/26 |
| 4 | prior15_notional_lt_500k AND prior15_ge_p75_109044 | 52 | +52.36 | +1009.46 | 45 | +21.37 | +15.42 | +693.69 | +511.49 | 13/26 |
| 5 | pressure15_ge_p75_0.27 AND prior15_ge_p75_109044 | 48 | +52.36 | +922.57 | 41 | -8.28 | +9.00 | +368.95 | +185.62 | 13/24 |
| 6 | prior15_notional_lt_1m AND prior15_ge_p75_109044 | 53 | +52.34 | +1000.91 | 51 | +21.37 | +15.46 | +788.26 | +606.06 | 14/26 |
| 7 | current_share_15m_ge_50 AND prior15_ge_p75_109044 | 47 | +52.34 | +808.28 | 44 | +52.42 | +20.60 | +906.28 | +724.08 | 15/24 |
| 8 | pressure15_le_p25_0.01 AND prior15_le_p25_3736 | 42 | +52.16 | +821.12 | 58 | +14.20 | +15.70 | +910.79 | +734.90 | 18/26 |

## Real-Fill Parity For Top Candidates

| Candidate | Total | Real Fill | No Fill | Test N | Test Median | Test Mean | Test Cum | Test Top3 Removed | Test Positive Days |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| prior15_ge_p75_109044 | 113 | 36 | 77 (68.1%) | 36 | +21.46 | +14.43 | +519.33 | +265.49 | 9/15 |
| prior15_clusters_eq_0 AND prior15_ge_p75_109044 | 113 | 36 | 77 (68.1%) | 36 | +21.46 | +14.43 | +519.33 | +265.49 | 9/15 |
| phase_fresh_start AND prior15_ge_p75_109044 | 97 | 25 | 72 (74.2%) | 25 | -8.66 | +7.23 | +180.80 | -63.09 | 5/14 |
| prior15_notional_lt_500k AND prior15_ge_p75_109044 | 97 | 25 | 72 (74.2%) | 25 | -8.66 | +7.23 | +180.80 | -63.09 | 5/14 |
| pressure15_ge_p75_0.27 AND prior15_ge_p75_109044 | 89 | 25 | 64 (71.9%) | 25 | -8.66 | +6.51 | +162.74 | -22.39 | 7/12 |
| prior15_notional_lt_1m AND prior15_ge_p75_109044 | 104 | 30 | 74 (71.2%) | 30 | -8.00 | +9.52 | +285.64 | +38.76 | 6/14 |
| current_share_15m_ge_50 AND prior15_ge_p75_109044 | 91 | 28 | 63 (69.2%) | 28 | +21.46 | +14.78 | +413.78 | +166.90 | 8/15 |
| pressure15_le_p25_0.01 AND prior15_le_p25_3736 | 100 | 40 | 60 (60.0%) | 40 | +22.93 | +17.61 | +704.34 | +515.74 | 12/17 |

## Read

This is a research scan over phase predicates and predicate pairs. Treat positives as hypothesis seeds, not live-rule proof.
