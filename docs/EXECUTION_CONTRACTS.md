# Execution Contracts (PR-1 Freeze)

This document defines canonical execution events and invariants used across:
- backtest
- papertrade
- live daemon

Schema version: `1`

## Event Types
- `order_intent`
- `order_ack`
- `fill`
- `reject`

All events must include:
- `schema_version`
- `event_id`
- `ts_ms` (UTC milliseconds)
- `source`
- `symbol`
- `order_id`
- `client_order_id`
- `side` (`BUY|SELL`)

## Dataclasses
Defined in `src/microphys/contracts/events.py`:
- `OrderIntent`
- `OrderAck`
- `FillEvent`
- `RejectEvent`

Type alias:
- `ExecutionEvent = OrderIntent | OrderAck | FillEvent | RejectEvent`

## Units and Conventions
- Time: `ts_ms` in UTC milliseconds.
- Price: absolute quote price (`> 0`).
- Quantity: base-asset quantity (`> 0` for fills/intents).
- Cost: `effective_cost_bps` in basis points.
- Side: canonicalized to `BUY|SELL` at boundary (accepts `long/short` input for normalization only).

## Invariants
Per event:
- non-empty identifiers (`event_id`, `order_id`, `client_order_id`, `symbol`)
- positive `ts_ms`
- side in `BUY|SELL`
- limit intents require `limit_price > 0`
- fills require `fill_qty > 0`, `fill_price > 0`
- `cumulative_qty >= fill_qty`
- `effective_cost_bps` finite and bounded

Per order sequence:
- `ts_ms` monotonic non-decreasing
- no events after terminal states (`FILLED`, `REJECTED`, `CANCELED`)
- no fill without prior intent/ack
- no duplicate intent after state progressed

## Transition Map (order-level)
- `NONE -> INTENT`
- `INTENT -> ACKED|CANCELED|REJECTED|PARTIAL|FILLED`
- `ACKED -> PARTIAL|FILLED|CANCELED|REJECTED`
- `PARTIAL -> PARTIAL|FILLED|CANCELED`

## Golden Fixtures
Fixtures are stored under:
- `tests/fixtures/execution/golden_execution_events.jsonl`
- `tests/fixtures/execution/golden_execution_events_invalid.jsonl`

They lock schema behavior before deeper refactors.

## Validation API
- `event_from_dict(raw)`
- `validate_event(event) -> list[str]`
- `validate_event_sequence(events) -> list[str]`

