# Kisi 1 (Research/Data) Status — 2026-03-09

Branch: `feat/execution-hardening-and-strategies`

## Owned Directories

### strategies/ (2 modules)
- `eclipse_scalper.py` — Primary scalper signal (910 lines after split)
- `risk.py` — Risk calculations with named constants

### data/ (10+ modules)
- `cache.py` — `GodEmperorDataOracle` (OHLCV, ticker, funding cache)
- `quality.py` — Data quality checks
- `microstructure_collector.py` — Microstructure data collection
- `microstructure_signals.py` — Signal generation from microstructure
- `microstructure_analysis.py` — Analysis utilities
- `event_diary.py` — Event diary tracking
- `data/features/` — `micro_features.py`, `registry.py`, `snapshot.py`
- `data/labels/` — `forward_return.py`
- `data/derived/` — 10 subdirectories (alpha_candidates, regimes, physics, etc.)
- `data/live/` — `online_plan.json`

### tools/ (200+ files)
Key research tools:
- `micro_edge_lib.py` — Feature computation
- `micro_edge_signal_v2.py` — Signal generation
- `micro_edge_backtest.py` — Execution-aware backtest
- `sweep_passive_realistic_filters.py` — Parameter sweep
- `validate_passive_pocket_forward.py` — Forward validation
- `rank_passive_pockets_forward.py` — Robustness ranking
- 20+ telemetry analysis tools (`telemetry_*.py`)
- 10+ event lane/watchboard tools
- Dashboard runners and health check scripts

### tests/ (200+ test files)
- Unit tests for strategies, signals, micro edge, alpha pipeline
- Parity, replay, execution, and contract tests in subdirectories
- 698 passing, 36 pre-existing failures (alpha pipeline/transfer tests)

## Pending Merge: 12 New execution/ Files
Branch `feat/execution-hardening-and-strategies` has 12 new execution files
not yet in `codex/runtime/ops-foundation`. Coordination needed before merge
to avoid conflicts with Kisi 2's runtime hardening work.

## Known Issues / TODOs
- `data/microstructure_signals.py:21` — TODO: "Implement after collecting 2+ weeks of microstructure data"
- 36 pre-existing test failures in alpha pipeline/transfer/live model tests
- `canonical_symbol()` vs `symkey()` deduplication gap (USDTUSDT case)
- No top-level `features/` directory — feature code lives in `data/features/`

## Completed Refactoring (by Kisi 1)
- `eclipse_scalper.py` split: 1521 -> 910 lines + `env_helpers.py` + `indicators.py`
- `_symkey()` duplication eliminated across 21 files
- `strategies/risk.py` magic numbers replaced with named constants
- 88 unit tests added for env helpers, indicators, signal pipeline
