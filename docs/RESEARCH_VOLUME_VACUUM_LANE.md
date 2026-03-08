# Research Volume Vacuum Lane

## Purpose

`volume_vacuum_regime` detects thin, low-activity buckets where passive execution can look calm but market depth is effectively absent.

## Why This Matters

This is an execution-context lane, not a directional signal.

It helps answer:

- is the market too empty to trust passive quality
- are we seeing low-flow conditions with widening spread
- should runtime treat the book as fragile

## Stack

1. `tools.volume_vacuum_alerts`
2. `tools.volume_vacuum_state`
3. `tools.volume_vacuum_watchlist`

## Runtime Shape

This lane now has the same runtime handoff shape as the other mature event lanes:

- alerts
- state/card
- watchlist
- watchboard aggregation

## Intended Runtime Use

- dashboard caution card
- thin-market watchlist row
- event watchboard aggregation

Not intended for:

- direct directional trading
- automatic execution mutation without separate validation
