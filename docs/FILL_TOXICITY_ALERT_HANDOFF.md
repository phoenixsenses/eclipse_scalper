# Fill Toxicity Alert Handoff

## Purpose

`fill_toxicity_regime` is a realized execution-quality lane.

It is not a direct trading trigger. The intended runtime uses are:

- dashboard annotation
- operator caution card
- execution-quality drift visibility

It should not directly mutate routing or risk behavior without separate validation.

## Runtime Input

Single payload:

- `tools.fill_toxicity_state`
- example artifact:
  - `reports/FILL_TOXICITY_STATE_REAL.json`

## What Runtime Should Render

- `state.level`
- `state.reasons`
- `recommended_action`
- `dashboard_summary`
- `card.top_side`
- `card.rows`
- `card.toxicity_score`
- `card.adverse_bps_mean`
- `card.pnl_bps_mean`

## Recommended Semantics

Values currently used:

- `monitor_only`
- `show_caution`
- `reduce_passive_aggression`

Interpretation:

- `monitor_only`
  - display only
- `show_caution`
  - visible caution card
- `reduce_passive_aggression`
  - strong operator warning
  - no automatic runtime mutation yet

## Current Real Example

Latest real artifact currently shows:

- `rows = 0`
- `state.level = quiet`
- `recommended_action = monitor_only`

This means no live trade history was available at evaluation time. Runtime should render this as no active signal, not as a broken payload.
