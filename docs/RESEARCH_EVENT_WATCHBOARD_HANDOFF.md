# Research Event Watchboard Handoff

## Purpose

`research_event_watchboard` is the top-level aggregation layer for current research event-intelligence lanes.

Current lanes:

- liquidation
- spread stress
- fill toxicity
- latency stress

Its job is to answer one runtime question:

- what is the most important research event context right now

## Runtime Input

Single payload:

- `tools.research_event_watchboard`
- example artifact:
  - `reports/RESEARCH_EVENT_WATCHBOARD_REAL.json`

## What Runtime Should Render

- `summary.state_counts`
- `summary.top_lane`
- `top_event`
- `banner`
- `lanes`

This is intended for:

- dashboard overview card
- top banner
- research event section

## Runtime Rules

- treat this as monitoring aggregation only
- do not auto-wire into execution logic
- if top event is `stale`, keep action informational
- prefer `banner` for top strip/header
- prefer `lanes` for detail table

## Current Real Example

Latest real watchboard currently shows:

- `top_lane = liquidation`
- `state_counts = {"severe": 2, "quiet": 2}`
- top liquidation and spread-stress lanes are both `stale`
- top action remains `monitor_only`

This means the strongest research event context is currently historical/stale, not active enough for escalation.
