# Latency Stress Alert Handoff

## Purpose

`latency_stress_regime` is an execution-quality monitoring lane derived from realized fill timing diagnostics.

It is not a direct trading trigger. The intended runtime uses are:

- dashboard annotation
- operator caution card
- latency drift visibility

It should not directly mutate routing behavior without separate validation.

## Runtime Input

Single payload:

- `tools.latency_stress_state`
- example artifact:
  - `reports/LATENCY_STRESS_STATE_REAL.json`

## What Runtime Should Render

- `state.level`
- `state.reasons`
- `recommended_action`
- `dashboard_summary`
- `card.rows`
- `card.fill_rate`
- `card.latency_fill_delay_sec_p50`
- `card.latency_fill_delay_sec_p95`
- `card.latency_impact_vs_net_corr`

## Recommended Semantics

Values currently used:

- `monitor_only`
- `show_caution`
- `escalate_monitoring`

Interpretation:

- `monitor_only`
  - display only
- `show_caution`
  - visible caution card
- `escalate_monitoring`
  - strong operator warning
  - no automatic runtime mutation yet

## Current Real Example

Latest real artifact currently shows:

- `rows = 0`
- `state.level = quiet`
- `recommended_action = monitor_only`

This means no live trade history was available at evaluation time. Runtime should render this as no active latency incident, not as a broken payload.
