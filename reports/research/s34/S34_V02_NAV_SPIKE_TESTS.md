# S34 V02 Navigation x Liquidation Spike Tests

Generated: `2026-06-29T18:03:14.448013+00:00`
Scope: ETHUSDT, last `30` days, `34951` 1m nav points.

## Spike Thresholds

- BUY primary threshold: `302236.6` notional/min
- SELL primary threshold: `312783.9` notional/min
- Non-overlap cooldown: `5` minutes
- Spike counts: BUY `109`, SELL `133`

## 1. NAV_HIGH -> Liq Spike Lead

- within 1m: BUY hit `0.007` (55), SELL hit `0.005` (38)
- within 3m: BUY hit `0.013` (105), SELL hit `0.013` (103)
- within 5m: BUY hit `0.019` (152), SELL hit `0.02` (159)
- within 15m: BUY hit `0.045` (358), SELL hit `0.051` (403)

## 2 + 5. Spike Forward Returns / Side Symmetry

### BUY spikes
- 1m: N `109`, sum `1410.6`, median `10.92`, WR `0.826`, T3R `1180.6`
- 5m: N `109`, sum `1468.2`, median `7.19`, WR `0.633`, T3R `1045.9`
- 15m: N `109`, sum `2025.6`, median `7.67`, WR `0.56`, T3R `1428.2`
- 60m: N `105`, sum `1401.7`, median `1.78`, WR `0.505`, T3R `654.4`
- 120m: N `105`, sum `2133.7`, median `-3.37`, WR `0.495`, T3R `1086.3`

### SELL spikes
- 1m: N `133`, sum `-1176.0`, median `-5.3`, WR `0.278`, T3R `-1277.7`
- 5m: N `133`, sum `-1505.2`, median `-0.93`, WR `0.481`, T3R `-1794.0`
- 15m: N `133`, sum `-2039.7`, median `3.68`, WR `0.541`, T3R `-2426.9`
- 60m: N `133`, sum `-1656.1`, median `4.02`, WR `0.534`, T3R `-2419.6`
- 120m: N `133`, sum `-2929.3`, median `5.15`, WR `0.526`, T3R `-3699.2`

## 3. Pre-Spike Indicator Shape

- BUY: spike avg score `6.49` vs control `5.96`; delta5m `0.17` vs control `-0.08`; prev5 high-min `2.68` vs control `2.81`
- SELL: spike avg score `4.39` vs control `5.96`; delta5m `-1.23` vs control `-0.08`; prev5 high-min `2.29` vs control `2.81`

## 4. Live Alpha + Spike Timing

- alpha rows in scope: `7`
- alpha NAV_HIGH: `{'n': 2, 'sum': 341.4, 'mean': 170.7, 'median': 170.7, 'win_rate': 1.0, 't3r': None, 'min': 41.7, 'max': 299.7}`
- alpha not-high: `{'n': 5, 'sum': 592.7, 'mean': 118.54, 'median': 149.9, 'win_rate': 1.0, 't3r': 63.5, 'min': 17.2, 'max': 227.0}`

## Notes

- Research-only. No live executor/config/order logic touched.
- The indicator here is a 1-minute proxy; use the chart line for visual monitoring, not as a live trigger.