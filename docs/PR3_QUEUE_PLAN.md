# PR-3 Implementation Plan: Queue Position Engine V2

Objective:
- Replace static queue approximation with event-step queue dynamics.
- Keep legacy queue model available with one config toggle.
- Add calibration hooks for queue parameters (symbol/regime-aware optional path).

## Scope
- `src/microphys/execution/queue_position.py` (new)
- `src/microphys/execution/queue_sim.py` (integrate v2 mode)
- `src/microphys/execution/calibration.py` (queue v2 calibration helper)
- tests:
  - `tests/execution/test_queue_position_dynamics.py`
  - `tests/execution/test_queue_calibration.py`

## Model
State:
- `queue_ahead` (visible queue ahead of our order)

Per-step updates:
- joins increase queue (`same_side_join_rate`)
- cancels reduce queue (`same_side_cancel_rate`)
- opposite aggressive flow consumes queue (`opposite_flow_rate * trade_through_prob * pressure`)

Fill condition:
- queue_ahead <= 0 before TTL expiry.

## Modes
- Legacy:
  - `queue_mode=fixed`
  - `queue_mode=adaptive`
- New:
  - `queue_mode=position_v2` (or `v2`)

Default remains legacy (`fixed`) for backward compatibility.

## Calibration Hook
Add helper in calibration module:
- `calibrate_queue_position_params(physics, symbol=None, regime_id=None) -> dict`

Output keys:
- `initial_queue_frac`
- `same_side_join_rate`
- `same_side_cancel_rate`
- `opposite_flow_scale`
- `pressure_floor`
- `ttl_bars`
- `min_depth`

## Validation
- v2 model reacts to joins/cancels/opposite flow directionally.
- legacy mode unchanged.
- calibration outputs bounded finite values.

