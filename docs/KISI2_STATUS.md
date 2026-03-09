# Kisi 2 (Runtime/Ops) Status — 2026-03-09

Branch: `codex/runtime/ops-foundation`

## Completed Work Summary

### Phase 1 — Safety Hardening (18 commits)
- P0: Kill switch fail-open -> fail-closed (`entry.py:1333`)
- P0: Timeout guards (order_router 10s, reconcile 8s), heartbeat, Telegram circuit breaker
- P1: Post-scaling size validation, dashboard rate limit, OOM fixes (intent_ledger, dashboard, tailer)
- P1: Atomic disk writes (5 files), startup validation gates (symbols, hedge, API)
- P1: Margin/liquidation alerts, guardian step timeout (45s), memory caps
- P1: HTML injection fix in Telegram, /kill command
- Tech: `_symkey()` consolidation (15 files), dead code removal (-482 lines)
- Test: 31 runtime safety tests added

### Phase 2 — Ops Enhancements (5 commits)
- Alert escalation (repeated alerts auto-escalate INFO -> WARNING -> CRITICAL)
- Graceful degradation mode (exchange down -> entries blocked, exits allowed)
- Config hot-reload from JSON override (40+ safe fields)
- Structured alert rules engine with 6 default rules
- Ops runbook (incident response playbook)

### Phase 3 — Profiling & Gaps (3 commits)
- Guardian step profiling (avg/max/last ms)
- Default alert rules JSON file
- Degraded mode wired to entry gate
- Alert rules API endpoint
- Adaptive guard memory cap (P0)
- Kisi 1 sync analysis document

## Audit Status

| Directory | Files | Fixed | Clean | Deleted | N/A |
|-----------|-------|-------|-------|---------|-----|
| execution/ | 37 | 5 | 29 | 2 | 1 |
| bot/ | 2 | 1 | 1 | — | — |
| brain/ | 3 | 0 | 3 | — | — |
| risk/ | 1 | 0 | 1 | — | — |
| exchanges/ | 4 | 0 | 4 | — | — |
| notifications/ | 6 | 2 | 4 | — | — |
| dashboard/backend/ | 5 | 1 | 4 | — | — |
| monitoring/ | 1 | 1 | 0 | — | — |
| top-level | 4 | 0 | 2 | — | 2 |
| **Total** | **63** | **10** | **48** | **2** | **3** |

## PR History
- **#27** — Merged: Initial runtime safety fixes
- **#28** — Merged: Phase 2 ops enhancements
- **#29** — Open: Phase 3 profiling & gaps

## Test Coverage (tests/runtime/)
- 10 test files, 115 test functions, 1711 lines
- Covers: runtime safety, deep audit fixes, alert escalation, alert rules,
  config hot-reload, degraded mode, guardian profiling, idempotency,
  integration startup, order FSM

## Recently Added (Phase 3 — current commit)

### Invariant Tests (3 files, 28 tests)
- `tests/test_order_router_idempotency.py` — EXE-01 (9 tests, duplicate order prevention)
- `tests/test_order_router_intent_lifecycle.py` — EXE-02 (10 tests, intent terminal states)
- `tests/test_paper_mode_no_live_orders.py` — SAF-02 (9 tests, paper mode safety)

### Infra Files (3)
- `execution/preflight.py` — Startup readiness checks (referenced in CLAUDE.md)
- `execution/env_sanity.py` — Runtime environment validation
- `execution/shared_locks.py` — Per-symbol async lock coordination

## Remaining Gaps (Low Priority)

### Low Test Coverage Areas
- `exchanges/` — Only mock.py tested; binance.py lacks unit tests
- `bot/core.py` — No dedicated test file
- `notifications/` — Basic integration test only

### Optional Improvements
- `monitoring/prometheus.py` — Dedicated Prometheus exporter (currently inline in app.py)
- `scripts/deploy_checklist.sh` — Pre-deploy verification script
