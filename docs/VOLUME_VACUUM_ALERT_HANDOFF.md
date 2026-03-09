# Volume Vacuum Alert Handoff

## Purpose

`volume_vacuum` is an event-intelligence lane. It detects thin, quiet market pockets where passive execution quality can degrade even without a directional event.

## Runtime Guidance

- use it as passive-execution caution context
- do not treat it as a directional trade signal
- combine it with spread stress to highlight thin and expensive market conditions

## Runtime Targets

- single-symbol state card from `VOLUME_VACUUM_STATE`
- multi-symbol overview from `VOLUME_VACUUM_WATCHLIST`
