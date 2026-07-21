# S34 V02 Navigation x Liquidation Spike Tests

Generated: `2026-06-29T18:00:17.399288+00:00`
Scope: ETHUSDT, last `7` days, `10066` 1m nav points.

## Spike Thresholds

- BUY primary threshold: `320411.5` notional/min
- SELL primary threshold: `437568.9` notional/min
- Non-overlap cooldown: `5` minutes
- Spike counts: BUY `48`, SELL `49`

## 1. NAV_HIGH -> Liq Spike Lead

- within 1m: BUY hit `0.011` (25), SELL hit `0.007` (16)
- within 3m: BUY hit `0.02` (46), SELL hit `0.018` (41)
- within 5m: BUY hit `0.028` (65), SELL hit `0.026` (60)
- within 15m: BUY hit `0.068` (157), SELL hit `0.065` (151)

## 2 + 5. Spike Forward Returns / Side Symmetry

### BUY spikes
- 1m: N `48`, sum `456.1`, median `6.32`, WR `0.812`, T3R `247.9`
- 5m: N `48`, sum `446.3`, median `2.98`, WR `0.604`, T3R `73.8`
- 15m: N `48`, sum `760.7`, median `8.57`, WR `0.604`, T3R `267.1`
- 60m: N `44`, sum `-107.4`, median `-0.37`, WR `0.477`, T3R `-478.2`
- 120m: N `44`, sum `376.6`, median `-9.09`, WR `0.477`, T3R `-419.0`

### SELL spikes
- 1m: N `49`, sum `-525.5`, median `-8.31`, WR `0.224`, T3R `-620.4`
- 5m: N `49`, sum `-1027.3`, median `-5.36`, WR `0.469`, T3R `-1283.0`
- 15m: N `49`, sum `-1454.0`, median `1.61`, WR `0.551`, T3R `-1826.5`
- 60m: N `49`, sum `-1444.1`, median `12.01`, WR `0.551`, T3R `-1996.7`
- 120m: N `49`, sum `-2165.2`, median `8.66`, WR `0.551`, T3R `-2759.2`

## 3. Pre-Spike Indicator Shape

- BUY: spike avg score `6.83` vs control `6.01`; delta5m `0.23` vs control `0.1`; prev5 high-min `2.75` vs control `2.63`
- SELL: spike avg score `3.45` vs control `6.01`; delta5m `-2.2` vs control `0.1`; prev5 high-min `2.14` vs control `2.63`

## 4. Live Alpha + Spike Timing

- alpha rows in scope: `3`
- alpha NAV_HIGH: `{'n': 1, 'sum': 299.7, 'mean': 299.7, 'median': 299.7, 'win_rate': 1.0, 't3r': None, 'min': 299.7, 'max': 299.7}`
- alpha not-high: `{'n': 2, 'sum': 167.1, 'mean': 83.55, 'median': 83.55, 'win_rate': 1.0, 't3r': None, 'min': 17.2, 'max': 149.9}`

## Notes

- Research-only. No live executor/config/order logic touched.
- The indicator here is a 1-minute proxy; use the chart line for visual monitoring, not as a live trigger.