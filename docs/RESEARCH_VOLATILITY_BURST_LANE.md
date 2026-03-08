# Research Volatility Burst Lane

## Purpose

`volatility_burst_regime` detects active expansion buckets where short-horizon returns and trade activity rise together.

## Why This Matters

This is an event-intelligence lane, not a direct trading trigger.

It helps answer:

- is the market in an expansion burst regime
- is there enough movement and activity to warrant operator attention
- should runtime treat the symbol as active and unstable

## Stack

1. `tools.volatility_burst_alerts`
2. `tools.volatility_burst_state`
