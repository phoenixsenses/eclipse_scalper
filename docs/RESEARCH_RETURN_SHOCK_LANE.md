# Research Return Shock Lane

## Purpose

`return_shock_regime` detects short-horizon price shock buckets from mark-price returns and trade intensity.

## Why This Matters

This lane is not a trade rule yet. It is an event-intelligence lane for:

- sudden directional price shock detection
- operator context during bursty moves
- future overlap checks against execution pain and signal collapse

## Stack

1. `tools.return_shock_alerts`
2. `tools.return_shock_state`
