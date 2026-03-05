# PR-2 Implementation Plan: Latency Model V2

Objective:
- Add explicit latency modeling with deterministic sampling.
- Keep legacy behavior unchanged when latency is disabled.
- Integrate latency into passive fill simulation and backtest CLI.

## Scope
- `src/microphys/execution/latency.py` (new)
- `execution/passive_execution_simulator.py` (integration)
- `src/microphys/execution/__init__.py` (exports)
- tests:
  - `tests/execution/test_latency_model.py`
  - `tests/execution/test_latency_effect_on_fill_rate.py`

## Contract
Latency timeline fields (ms):
- `decision_ts`
- `send_ts`
- `exchange_recv_ts`
- `book_effective_ts`
- `ack_ts`
- `fill_ts`

Config modes:
- `fixed`
- `normal` (bounded jitter)
- `empirical` (weighted buckets)

## Compatibility
- `latency_enabled=0` => all stage delays are zero.
- Existing params remain valid:
  - `latency_decision_to_ack_ms`
  - `latency_queue_entry_ms`
  - `latency_feed_lag_ms`
  - jitter keys
- Backward fields preserved in simulator outputs:
  - `latency_ms_total`, `latency_bars`, stage component fields.

## Detailed Steps
1. Add latency primitives
- config parser from params dict
- deterministic sampler (`seed + event_id`)
- stage-wise sample
- timeline build helper
- utility for converting latency to bars

2. Integrate simulator
- replace ad-hoc latency draw with shared latency module
- keep same penalty semantics:
  - touch probability reduction by latency bars
  - fill offset shift by latency bars
  - optional adverse bps per sec

3. Extend outputs
- include full stage timings and timeline markers for diagnostics

4. Tests
- fixed profile deterministic outputs
- normal profile bounded outputs
- empirical profile deterministic bucket selection
- latency on/off effect on fill rate

5. Validation
- `py_compile` for modified files
- run new tests + existing passive sim tests

## Done Criteria
- default-off path unchanged
- latency-enabled path deterministic
- tests pass and document expected latency impacts

