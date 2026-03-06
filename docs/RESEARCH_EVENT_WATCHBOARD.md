# Research Event Watchboard

## Purpose

`research_event_watchboard` aggregates the active event-intelligence lanes into one runtime-friendly summary.

Current lanes:

- liquidation
- spread stress
- return shock
- volume vacuum
- fill toxicity
- latency stress

## Why This Layer Exists

Without a watchboard, runtime would need to consume and rank multiple lane payloads separately.

The watchboard solves:

- which lane is currently most important
- what headline should appear first
- what top operator action is suggested right now

## First Deliverable

Tool:

- `python -m tools.research_event_watchboard`

It emits:

- `summary`
- `top_event`
- `banner`
- `lanes`
- `run_summary`

## Intended Use

Short term:

- dashboard header / overview card
- operator priority ranking across research event lanes
- freshness-aware top event banner
- multi-lane severity table

Not yet:

- direct execution mutation
- automatic risk actions
