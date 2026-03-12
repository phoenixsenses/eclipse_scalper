# Symkey & Helper Deduplication Plan

**Date:** 2026-03-12
**Author:** Kisi 2 (Runtime/Ops)
**Status:** Phase 0.4 — Inventory complete, dedup starts in Phase 4

---

## Problem

20 files define local `_symkey()` fallbacks. 5 core helpers (`_symkey`, `_safe_float`, `_truthy`, `_cfg`, `_now`) are duplicated across 170+ locations. Divergent implementations risk reconcile mismatches and subtle bugs.

## Centralized Sources

| Helper | Canonical Location | Notes |
|--------|--------------------|-------|
| `symkey()` | `execution/runtime_helpers.py` L46-53 | Imports `canonical_symbol` from `utils/symbols.py` |
| `_safe_float()` | `execution/runtime_helpers.py` | 101 local definitions exist |
| `_truthy()` | `execution/runtime_helpers.py` | 14 local definitions exist |
| `_cfg()` | `execution/runtime_helpers.py` | 26 local definitions exist |
| `_now()` | `execution/runtime_helpers.py` | 16 local definitions exist |

## _symkey Fallback Inventory (20 files)

### Kisi 2 Domain (14 files — try/except import from runtime_helpers)

| # | File | Lines | Pattern |
|---|------|-------|---------|
| 1 | `bot/core.py` | 64-68 | try/except import + fallback |
| 2 | `execution/belief_controller.py` | 10-14 | try/except import + fallback |
| 3 | `execution/data_loop.py` | 83-87 | try/except import + fallback |
| 4 | `execution/data_quality.py` | 38-42 | try/except import + fallback |
| 5 | `execution/entry_watch.py` | 16-20 | try/except import + fallback |
| 6 | `execution/exit.py` | 65-69 | try/except import + fallback |
| 7 | `execution/order_router.py` | 68-72 | try/except import + fallback |
| 8 | `execution/position_manager.py` | 19-23 | try/except import + fallback |
| 9 | `execution/rebuild.py` | 24-28 | try/except import + fallback |
| 10 | `execution/reconcile.py` | 18-25 | try/except import + fallback |
| 11 | `execution/telemetry.py` | 57-61 | try/except import + fallback |
| 12 | `exchanges/binance.py` | 34-38 | try/except import + fallback |
| 13 | `brain/persistence.py` | 166-170 | try/except import + fallback |
| 14 | `brain/state.py` | 50-54 | try/except import + fallback |

### Kisi 1 Domain (6 files — direct definitions, NOT try/except)

| # | File | Lines | Notes |
|---|------|-------|-------|
| 15 | `data/cache.py` | 40-51 | Direct definition, should import from utils/symbols |
| 16 | `strategies/eclipse_scalper.py` | 228-243 | Direct definition |
| 17 | `tools/corr_group_check.py` | 11+ | Direct definition |
| 18 | `tools/peek_cache.py` | 22+ | Direct definition |
| 19 | `tools/telemetry_roll_alerts.py` | 35+ | Direct definition |
| 20 | `tools/telemetry_threshold_alerts.py` | 83+ | Direct definition |

## Other Duplicated Helpers

| Helper | Definitions | Primary Locations |
|--------|-------------|-------------------|
| `_safe_float` | 101 | execution (38), tools (57), brain (3), core (3) |
| `_cfg` / `_cfg_float` / `_cfg_env` | 26 | execution (21), risk (1), tests (1), core (3) |
| `_truthy` | 14 | execution (12), risk (1), emergency (1) |
| `_now` / `_now_ts` / `_now_wall` | 16 | execution (11), brain (1), risk (1), tools (1) |

## Dedup Plan (Phase 4)

### Step 1: Kisi 2 symkey dedup (14 files)
- Replace try/except + fallback with direct `from execution.runtime_helpers import symkey as _symkey`
- One file at a time, full test suite after each
- Start with lowest-risk files: telemetry, belief_controller, data_quality

### Step 2: Kisi 2 helper dedup (_truthy, _cfg, _now, _safe_float)
- Import from `execution/runtime_helpers.py` instead of local definition
- Focus on execution/ and risk/ files first

### Step 3: Coordinate with Kisi 1
- `data/cache.py` L40 should import from `utils/symbols.py`
- `strategies/eclipse_scalper.py` L228 should import from `utils/symbols.py`
- tools/ files can be done in batch

### Risk Mitigation
- Deploy one file at a time
- Run `pytest tests/ -x --timeout=30` after each change
- If `runtime_helpers` import fails at startup, bot won't start (acceptable — better than silent divergence)
