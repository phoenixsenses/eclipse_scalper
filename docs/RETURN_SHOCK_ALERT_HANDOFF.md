# Return Shock Alert Handoff

## Purpose

`return_shock` is an event-intelligence lane. It detects sudden short-horizon directional moves with enough activity to matter operationally.

## Runtime Guidance

- use it as context and monitoring
- do not auto-map `dominant_direction` to a trade action
- combine it with spread/liquidation lanes if runtime wants a composite stress view

## Runtime Targets

- single-symbol state card from `RETURN_SHOCK_STATE`
- multi-symbol overview from `RETURN_SHOCK_WATCHLIST`
