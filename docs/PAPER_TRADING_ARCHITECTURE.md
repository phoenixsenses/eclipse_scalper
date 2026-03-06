# Paper Trading Architecture

## Entry Points
- `bot/runner.py`
  - Primary launcher used in current runs.
  - Supports `--dry-run` and installs a hard order-block guard on exchange `create_order`.
- `execution/bootstrap.py`
  - Loop orchestrator that starts `data_loop`, `entry_loop`, `guardian`, and optional `exit/position_manager` loops.

## Loop Responsibilities
- `execution/data_loop.py`
  - Maintains market data cache used by signal and sizing paths.
- `execution/entry_loop.py`
  - Pulls strategy signal (`strategies.eclipse_scalper.scalper_signal`).
  - Applies safety gates (cooldown, kill switch, staleness, reliability, risk checks).
  - Sends entry orders through `execution/order_router.create_order`.
- `execution/exit.py`
  - Handles exit-side order management and telemetry.
- `execution/guardian.py`
  - Supervises periodic maintenance and high-frequency exit watch.

## Current Paper/Dry-Run Behavior
- Dry mode is controlled by `SCALPER_DRY_RUN=1` or `runner --dry-run`.
- In dry mode:
  - `bot/runner.py` monkey-patches exchange `create_order` to block live order placement.
  - `execution/order_router.py` also has dry-run route guards and returns dry-run stubs.
- This path is used as the paper-safe execution route in this repo.

## Signal Flow
1. `entry_loop` picks symbols.
2. Strategy call (`scalper_signal`) returns dict or tuple.
3. Tuple is normalized to `{action, confidence, type}`.
4. Entry gates run.
5. If allowed, order is routed via `order_router.create_order`.

## Position and Exit Flow
- Positions are tracked in `bot.state.positions`.
- Reconciliation logic in `execution/reconcile.py` aligns exchange/bot state.
- Exit events are processed in `execution/exit.py`.

## Phase 4 Integration Points Added
- `core/regime.py` (`RegimeClassifier`) is wired into `entry_loop` as a pre-entry gate.
- New env knobs in `entry_loop`:
  - `ENTRY_REGIME=none|up|down` (default `none`)
  - `ENTRY_REGIME_LOOKBACK_SEC` (default `3600`)
  - `ENTRY_REGIME_DEBOUNCE_SEC` (default `60`)
  - `ENTRY_REGIME_BLOCK_TRANSITION` (default `true`)
  - `ENTRY_REGIME_BLOCK_UNKNOWN` (default `true`)
- Behavior:
  - If regime gate blocks, `entry.blocked` is emitted with reason:
    - `regime_transition`, `regime_unknown`, or `regime_mismatch`
  - Regime metadata is attached to signal payload when available:
    - `regime`, `regime_age_sec`, `rolling_return`

## Phase 5 Risk Hook (Opt-in)
- `core/regime_risk.py` provides `RegimeRiskManager` policy engine.
- Runtime wiring in `execution/entry_loop.py` and `execution/exit.py` is feature-flagged:
  - `ENTRY_REGIME_RISK_ENABLED` (default `false`)
  - Entry-time checks block with `entry.blocked` reason `risk_regime_block`.
  - Regime flips generate `entry.blocked` reason `risk_regime_action` with action metadata.
  - Full exits notify risk manager via `exit.risk_action` telemetry events (if actions produced).
- Daily counters are reset by entry loop on UTC day change when risk manager is enabled.

## Scratch Integration Status
- A standalone `core/scratch.py` engine is implemented and tested.
- Existing research-side scratch simulation remains in `tools/micro_edge_backtest.py`.
- Runtime integration is now feature-flagged in `execution/exit.py`:
  - `EXIT_SCRATCH_ENABLED` (default `false`)
  - `EXIT_SCRATCH_ADVERSE_BPS`
  - `EXIT_SCRATCH_COOLDOWN_SEC`
  - `EXIT_SCRATCH_TRAILING_BPS`
  - `EXIT_SCRATCH_TAKE_PROFIT_BPS`
  - `EXIT_SCRATCH_HARD_HORIZON_SEC`
- With flags unset, behavior remains unchanged.

## Validation Commands
```powershell
python -m py_compile core/regime.py core/scratch.py execution/entry_loop.py tools/verify_regime_consistency.py tools/backtest_scratch.py
pytest -q tests/test_regime.py tests/test_scratch.py tests/test_passive_scratch_rule.py
python -m tools.verify_regime_consistency --db data/microstructure.db --symbol ETHUSDT --lookback-sec 3600
```
