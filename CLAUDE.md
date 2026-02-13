# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Eclipse Scalper is an async Python cryptocurrency futures trading bot targeting Binance via ccxt. It uses a reconciliation-first architecture where exchange state is the authority and local state is "belief".

## Commands

```bash
# Run all unit tests
python -m pytest tools/ -v

# Run a single test file
python -m pytest tools/test_entry_loop_unit.py -v

# Run a single test
python -m pytest tools/test_entry_loop_unit.py::EntryLoopTelemetryTests::test_recent_router_blocks_counts -v

# Start bot (dry-run safe)
python main.py --dry-run

# Start via execution bootstrap
python -m execution.bootstrap

# Telemetry dashboard
python tools/telemetry_dashboard.py --path logs/telemetry.jsonl

# Reliability gate check
python tools/reliability_gate.py
```

## Architecture

### Execution Pipeline

```
Signal Generation (strategies/eclipse_scalper.py)
  → Entry Loop (execution/entry_loop.py) — 20+ gate checks before submission
    → Order Router (execution/order_router.py) — retry, idempotency, kill-switch
      → Exchange Adapter (exchanges/binance.py) — ccxt wrapper
        → Reconcile (execution/reconcile.py) — exchange vs brain state sync
          → Position Manager (execution/position_manager.py) — stop/TP management
```

### Core Object: `bot` (EclipseEternal from bot/core.py)

Every async function takes `bot` as first argument. Key attributes:
- `bot.ex` / `bot.exchange` — exchange adapter
- `bot.state` — `PsycheState` (brain/state.py), persisted as LZ4-compressed binary
- `bot.state.positions` — position dict keyed by canonical symbol (e.g. `BTCUSDT`)
- `bot.state.run_context` — ephemeral-but-persistent runtime state dict
- `bot.data` — market data cache
- `bot.cfg` — configuration (config/settings.py)
- `bot.active_symbols` — set of symbols to trade

### Symbol Key Convention

All modules use `_symkey()` to canonicalize symbols: `BTC/USDT:USDT` → `BTCUSDT`. The single source of truth is `execution/entry_primitives.py:symkey()`. All other modules import from there.

### State Persistence

Brain state is saved as LZ4-compressed binary to `~/.blade_eternal.brain.lz4`. The `run_context` dict on `bot.state` survives restarts. Entry pending-blocks and WAL intents are stored there.

### Concurrency Model

Fully async (`asyncio`). Per-symbol locks are shared across reconcile and position_manager via `execution/shared_locks.py`. Entry loop uses per-symbol locks to prevent concurrent submit storms.

## Key Design Rules

1. **Guardian-safe contract**: All execution functions never raise — they catch and log internally
2. **Exits are privileged**: Never apply entry restrictions (kill-switch, circuit breaker) to exit/protective orders. `intent_reduce_only=True` orders must always be allowed through
3. **Reconciliation is authority**: Exchange state is truth, local state is belief
4. **Router retries are bounded**: No infinite retry loops. Errors are classified (retryable, fatal, idempotent_safe)
5. **Hedge-mode**: Binance futures uses hedge mode with `positionSide` (LONG/SHORT)
6. **Telemetry-driven**: Extensive JSONL telemetry in `logs/telemetry.jsonl`

## Test Conventions

- Tests live in `tools/test_*_unit.py` (unittest framework, pytest-compatible)
- Import pattern: `ROOT = Path(__file__).resolve().parents[2]`, then `from eclipse_scalper.execution.module import ...`
- Bot stubs use `SimpleNamespace` with `state`, `cfg`, etc.
- Tests must handle Windows codepage: `sys.stdout.reconfigure(encoding="utf-8")`
- Circuit breaker is opt-in via `bot.cfg.CIRCUIT_BREAKER_ENABLED` to avoid test pollution

## Environment

- Python 3.10+ (current: 3.13.9)
- `SCALPER_DRY_RUN=1` for simulation (safe default), `=0` for live trading
- `.env` file for API keys (`BINANCE_API_KEY`, `BINANCE_API_SECRET`)
- `ACTIVE_SYMBOLS=BTCUSDT,ETHUSDT` for symbol selection
- Binance clientOrderId limit: < 36 characters

## CI

GitHub Actions runs chaos scenarios, execution invariant suites, and reliability gate enforcement on push/PR. See `.github/workflows/ci-tests.yml`.
