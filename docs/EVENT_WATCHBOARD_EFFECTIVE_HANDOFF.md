# Effective Event Watchboard Handoff

## Purpose

`event_watchboard_effective` converts the raw research event watchboard into the
runtime-facing display view after suppression and persistence policy are applied.

Its job is to answer one runtime question:

- what should the operator actually see as primary versus secondary right now

## Runtime Input

Two upstream payloads:

- `tools.research_event_watchboard`
- `tools.event_lane_suppression_policy`
- `tools.event_lane_persistence_policy`

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
- `summary.noisy_lane_count`
- `summary.primary_noisy_lane`
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
- if `primary_noisy_lane` matches the top lane, respect `recommended_min_persist_snapshots`
  before promoting it into a sticky header/banner
- use `recommended_cooldown_snapshots` to avoid fast banner flipping for noisy lanes
- treat stale top events as informational unless another runtime lane independently escalates
- do not auto-wire this payload into execution logic

## Current Real Example

Latest real effective watchboard currently shows:

- `raw_top_lane = return_shock`
- `effective_top_lane = return_shock`
- `degraded_lane_count = 2`
- `noisy_lane_count = 1`
- `primary_noisy_lane = return_shock`
- degraded lanes:
  - `volume_vacuum`
  - `volatility_burst`

This means the operator should focus on `return_shock`, keep the two overlapping
secondary lanes de-emphasized, and treat `return_shock` as a persistence-aware
headline candidate instead of immediately re-flipping the top banner on every
snapshot.
