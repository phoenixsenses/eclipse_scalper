# PR-5 Runtime Discipline Plan

## Goal
- Add a minimal runtime discipline layer with:
  - typed event bus
  - strict order finite state machine
  - runtime health supervisor

## Scope
- `src/microphys/runtime/event_bus.py`
- `src/microphys/runtime/order_fsm.py`
- `src/microphys/runtime/supervisor.py`
- `src/microphys/runtime/__init__.py`
- `tests/runtime/test_order_fsm.py`
- `tests/runtime/test_idempotency.py`

## Acceptance
- Illegal order state transitions raise deterministic errors.
- Event bus supports idempotent subscribers by `event_id`.
- Runtime supervisor emits `ok/degraded/failed` and halt signal on failure.
- Targeted compile + tests pass.

## Verification
- `python -m py_compile src/microphys/runtime/event_bus.py src/microphys/runtime/order_fsm.py src/microphys/runtime/supervisor.py`
- `pytest -q tests/runtime/test_order_fsm.py tests/runtime/test_idempotency.py`

