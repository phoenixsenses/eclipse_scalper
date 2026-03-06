# Book Proxy Pressure Alert Handoff

## Purpose

`book_proxy_pressure` is an event-intelligence lane that estimates one-sided pressure from current derived bucket features.

## Important Constraint

This is not real order-book depth.

It is a proxy built from:

- imbalance proxy
- spread
- trade intensity
- short-horizon return stability

## Runtime Guidance

- use it as one-sided market-pressure context
- do not treat it as true book imbalance
- do not auto-wire `primary_side_bias` into order logic

## Runtime Targets

- single-symbol state card from `BOOK_PROXY_PRESSURE_STATE`
- multi-symbol overview from `BOOK_PROXY_PRESSURE_WATCHLIST`
