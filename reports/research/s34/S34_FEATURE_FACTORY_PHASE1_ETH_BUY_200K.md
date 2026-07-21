# S34 Feature Factory Phase 1 - ETH BUY 200K

Generated: 2026-06-16T08:29:57.348869+00:00

Scope: ETHUSDT BUY liquidation clusters >= 200K, 300s bucket, 900s minimum gap.

Output DB: `data/s34_feature_factory.db`

## Lookahead Boundary

- `liq_event_features`: signal-time/no-lookahead features only.
- `liq_event_outcome_labels`: future path labels and route outcomes only.
- Wait/confirmation returns are not stored in the feature table in Phase 1. They must be modeled through route `entry_delay_sec` or added later to a separate delayed-feature table.

## Extraction Summary

- Feature rows: `450`
- Outcome label rows: `1350`
- Anchor routes: `3`

## Anchor Route Results

| Route | N | Mean Net | Median Net | Cum Net | WR | TP | BE | SL | TIME | Mean MFE | Mean MAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| LONG_DELAY0_TP60 | 450 | +13.84 | -5.09 | +6228.28 | 49.6% | 204 | 133 | 76 | 37 | +45.70 | -13.23 |
| LONG_DELAY60_TP120 | 450 | +4.48 | -8.78 | +2016.71 | 28.7% | 72 | 169 | 128 | 81 | +54.80 | -19.48 |
| SHORT_DELAY0_TP40_CONTROL | 450 | -27.58 | -48.46 | -12412.91 | 23.1% | 93 | 30 | 298 | 29 | +15.74 | -35.13 |

## Phase 1 Acceptance

- Separate feature and label tables created.
- Source `microstructure.db` read-only.
- Feature table has no future path columns.
- Only three anchor routes computed, avoiding the full combinatorial route explosion.

## Read

This is infrastructure, not a new trading decision. Use this DB as the base for a query layer. Do not promote a new paper variant from Phase 1 without outlier/day-spread checks and live-fill confirmation.
