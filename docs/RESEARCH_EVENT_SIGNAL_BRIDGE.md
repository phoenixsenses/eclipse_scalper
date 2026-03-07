# Research Event/Signal Bridge

## Purpose

`tools/validate_micro_edge_forward.py` now reports not only walk-forward quality but also event-lane context for the selected discovery and validation slices.

This closes the gap between:

- signal evaluation
- event-intelligence lanes

## What Is Added

The forward-validation payload now includes:

- `event_lane_context_impact.discovery`
- `event_lane_context_impact.validation`

Each section reports:

- `available`
- `rows_total`
- `lane_count`
- `top_lane_by_delta_avg_net`
- `by_lane`

Current lane coverage inside the bridge:

- `spread_stress`
- `return_shock`
- `volatility_burst`
- `volume_vacuum`
- `book_proxy_pressure`

## Why It Matters

Before this change, forward validation could tell us:

- whether a selection collapses
- whether liquidation-heavy slices behave differently

It could not tell us:

- which broader event regime the selected rows belong to

Now the research question becomes:

- "Is this pocket good?"
- and also
- "Under which event context is this pocket good or bad?"

## Interpretation

This bridge is intentionally descriptive, not causal.

It does not change signal scoring.
It adds context around selected rows so we can decide:

- whether a pocket is regime-specific
- whether event lanes overlap with the selected surface
- whether a future signal refinement should become:
  - a feature
  - a filter
  - a separate event lane

## Current Constraint

This bridge uses the current debug-row feature surface.

It does not infer true order-book regimes because the live collector still does not store real top-of-book depth.
That is why the lane is named `book_proxy_pressure`, not real book imbalance.
