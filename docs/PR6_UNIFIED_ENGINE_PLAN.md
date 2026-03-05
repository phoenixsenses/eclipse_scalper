# PR-6 Unified Execution Engine Plan

## Goal
- Introduce one execution interface reusable by backtest, paper, and live paths.
- Keep strategy logic unchanged; this is contract unification.

## Scope
- `src/microphys/execution/engine.py`
- `src/microphys/execution/__init__.py`
- `tests/parity/test_backtest_paper_parity.py`
- `tests/parity/test_paper_live_contract_parity.py`

## Interface
- `ExecutionRequest`
- `ExecutionResult`
- `ExecutionEngine`
- `ExecutionAdapter` protocol
- Deterministic default adapters for parity baseline

## Acceptance
- Same input yields same cost/return semantics for backtest and paper adapters.
- Paper and live adapters expose same output contract fields.
- Targeted compile + parity tests pass.

## Verification
- `python -m py_compile src/microphys/execution/engine.py`
- `pytest -q tests/parity/test_backtest_paper_parity.py tests/parity/test_paper_live_contract_parity.py`

