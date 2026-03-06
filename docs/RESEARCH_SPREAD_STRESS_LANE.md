# Research Spread Stress Lane

## Why This Lane

`spread_stress_regime` is the next event-intelligence candidate after liquidation.

Reason:

- directly impacts fillability and passive execution quality
- likely overlaps runtime pain faster than liquidation-only regimes
- uses already available research features:
  - `spread`
  - `trade_intensity`
  - `ret_1`

## Initial Hypothesis

Buckets with:

- unusually wide spread
- unusually low trade intensity

should be treated as execution stress context first, not as immediate alpha.

## First Deliverable

Tool:

- `python -m tools.spread_stress_alerts`
- `python -m tools.spread_stress_state`

This first version is intentionally narrow:

- detects spread stress buckets
- emits recent alerts
- emits runtime-ready state/card payload
- produces JSON/MD + `run_summary`

## Expected Use

Short term:

- research event feed
- monitoring annotation
- possible runtime caution lane

Not yet:

- direct trade trigger
- direct score boost
