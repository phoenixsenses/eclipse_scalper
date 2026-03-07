# Effective Event Watchboard Handoff

## Purpose

`event_watchboard_effective` converts the raw research event watchboard into the
runtime-facing display view after suppression policy is applied.

Its job is to answer one runtime question:

- what should the operator actually see as primary versus secondary right now

## Runtime Input

Two upstream payloads:

- `tools.research_event_watchboard`
- `tools.event_lane_suppression_policy`

One preferred output:

- `tools.event_watchboard_effective`
- example artifact:
  - `reports/EVENT_WATCHBOARD_EFFECTIVE_REAL.json`

## What Runtime Should Render

- `summary.raw_top_lane`
- `summary.effective_top_lane`
- `summary.hidden_lane_count`
- `summary.degraded_lane_count`
- `summary.collapsed_lane_count`
- `top_event`
- `banner`
- `lanes`

This is intended for:

- dashboard overview card
- top banner
- prioritized lane table

## Runtime Rules

- prefer `effective_top_lane`, not `raw_top_lane`, for primary dashboard focus
- render degraded lanes below the primary lane
- do not fully suppress hidden/collapsed lanes from logs; only suppress display emphasis
- treat stale top events as informational unless another runtime lane independently escalates
- do not auto-wire this payload into execution logic

## Current Real Example

Latest real effective watchboard currently shows:

- `raw_top_lane = spread_stress`
- `effective_top_lane = spread_stress`
- `degraded_lane_count = 2`
- degraded lanes:
  - `volume_vacuum`
  - `volatility_burst`

This means the operator should still focus on `spread_stress`, while the two
overlapping secondary lanes remain visible but de-emphasized.
