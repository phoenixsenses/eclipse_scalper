# Execution V2 End-to-End Build Map

## Objective
Deliver execution-quality upgrades without changing validated signal logic:
- Latency realism
- Queue realism
- Replay parity
- Runtime discipline
- Backtest/paper/live contract parity
- Rollout safety

## Current Status
- PR-1 Contracts/Invariants: done
- PR-2 Latency model: done
- PR-3 Queue position v2: done
- PR-4 Replay parity tooling: done
- PR-5 Runtime event bus + order FSM: done
- PR-6 Unified execution engine parity: done
- PR-7 Diagnostics + rollout flags: done

## Build Sequence
1. PR-5 Runtime discipline
2. PR-6 Unified execution interface
3. PR-7 Diagnostics and rollout controls

## PR-5 Plan (Runtime Discipline)
- Add `src/microphys/runtime/event_bus.py`
  - typed topic publish/subscribe
  - idempotent subscriber keys
- Add `src/microphys/runtime/order_fsm.py`
  - states: NEW -> ACKED -> PARTIAL -> FILLED/CANCELED/REJECTED
  - invariant checks on illegal transitions
- Add `src/microphys/runtime/supervisor.py`
  - health gates for stale feed, stalled order updates, loop exceptions
- Tests:
  - `tests/runtime/test_order_fsm.py`
  - `tests/runtime/test_idempotency.py`

## PR-6 Plan (Unified Engine / Parity Harness)
- Add `src/microphys/execution/engine.py`
  - single execution interface for backtest/paper/live adapters
  - same request/response contracts
- Integrate with:
  - `tools/micro_edge_backtest.py`
  - `src/microphys/sim/papertrade.py`
  - `src/microphys/live/daemon.py`
- Tests:
  - `tests/parity/test_backtest_paper_parity.py`
  - `tests/parity/test_paper_live_contract_parity.py`

## PR-7 Plan (Diagnostics / Rollout)
- Add tools:
  - `tools/execution_diagnostics.py`
  - `tools/toxicity_report.py`
  - `tools/post_rollout_audit.py`
- Add flags:
  - `EXEC_LATENCY_V2`
  - `QUEUE_MODEL_V2`
  - `EXEC_ENGINE_UNIFIED`
- Add docs:
  - `docs/ROLLOUT_EXECUTION_V2.md`
- Add staged rollout checks:
  - canary symbol -> 2 symbols -> full set
  - rollback via single env toggle

## Verification Gate per PR
- Compile:
  - `python -m py_compile <changed_files>`
- Tests:
  - `pytest -q <targeted_tests>`
- Smoke:
  - run one backtest + one paper dry-run with flags on/off

## Final Done Criteria
- Fill-rate prediction error improvement >= 25%
- Adverse-selection prediction MAE improvement >= 20%
- No contract violation in 7-day soak
- P95 execution loop latency within configured budget
