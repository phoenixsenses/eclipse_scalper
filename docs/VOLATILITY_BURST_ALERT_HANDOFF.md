# Volatility Burst Alert Handoff

## Purpose

`volatility_burst` is an event-intelligence lane. It detects active expansion conditions where short-horizon returns and trade activity rise together.

## Runtime Guidance

- use it as expansion/regime context
- do not convert `dominant_direction` directly into a trade action
- combine it with return shock and spread stress if runtime wants an active-market view

## Runtime Targets

- single-symbol state card from `VOLATILITY_BURST_STATE`
- multi-symbol overview from `VOLATILITY_BURST_WATCHLIST`

## What Runtime Should Render

For a symbol card:

- `state.level`
- `state.freshness.status`
- `state.dominant_direction`
- `recommended_action`
- `dashboard_summary`
- `card.recent_alert_count`
- `card.high_count`
- `card.medium_count`
- `card.avg_abs_ret_1_tagged`
- `card.avg_trade_intensity_tagged`

For a watchlist:

- `top_summary`
- `banner`
- `rows`

## Recommended Semantics

`recommended_action` is display guidance only.

Values currently used:

- `monitor_only`
- `show_caution`
- `escalate_monitoring`

Interpretation:

- `monitor_only`
  - informational only
- `show_caution`
  - visible active-market caution
- `escalate_monitoring`
  - highlight that the market is in a fresh high-activity expansion regime

Do not auto-wire this into trade direction or order logic without separate validation.

## Freshness Rule

Always read severity together with freshness.

- fresh + severe
  - active expansion regime
- stale + severe
  - historical context, no active escalation
