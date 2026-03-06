# Spread Stress Alert Handoff

## Purpose

`spread_stress_regime` is an execution-quality and monitoring lane.

It is not a direct trading trigger. The intended runtime uses are:

- dashboard annotation
- operator caution card
- multi-symbol watchlist/banner
- optional monitoring notification

It should not directly alter trade selection logic without separate validation.

## Research Conclusion

This lane measures passive execution pain context:

- unusually wide spread
- enough recurrence to matter operationally
- freshness-aware severity

Unlike the liquidation lane, this regime is naturally aligned with runtime execution quality.

## Runtime Inputs

Single-symbol state payload:

- `tools.spread_stress_state`
- example artifact:
  - `reports/SPREAD_STRESS_STATE_REAL.json`

Multi-symbol watchlist payload:

- `tools.spread_stress_watchlist`
- example artifact:
  - `reports/SPREAD_STRESS_WATCHLIST_REAL.json`

## What Runtime Should Render

For a symbol card:

- `state.level`
- `state.freshness.status`
- `recommended_action`
- `dashboard_summary`
- `card.recent_alert_count`
- `card.high_count`
- `card.medium_count`
- `card.avg_spread_tagged`

For a watchlist:

- `top_summary`
- `banner`
- `rows`

## Recommended Semantics

`recommended_action` should be treated as display guidance, not an order-routing command.

Values currently used:

- `monitor_only`
- `show_caution`
- `reduce_passive_aggression`

Interpretation:

- `monitor_only`
  - show status only
- `show_caution`
  - visible caution state in dashboard
- `reduce_passive_aggression`
  - highlight execution stress prominently
  - do not automatically mutate runtime behavior unless separately implemented and approved

## Freshness Rule

Always read severity together with freshness.

Example:

- `level=severe` and `freshness=stale`
  - render as stale severe context
  - do not treat as active stress

## Current Real Example

From the latest real watchlist artifact:

- `top_summary.symbol = ETHUSDT`
- `top_summary.state_level = severe`
- `top_summary.freshness_status = stale`
- `top_summary.recommended_action = monitor_only`

This means the symbol had strong stress context, but it is not fresh enough to escalate right now.

## Person 2 Scope

Suggested runtime task:

- add spread-stress card to dashboard
- add spread-stress watchlist section
- add top banner based on `banner.headline`

Do not:

- convert this to auto-trade logic
- blend it into risk/kill-switch logic yet
- rewrite the scoring logic on the runtime side
