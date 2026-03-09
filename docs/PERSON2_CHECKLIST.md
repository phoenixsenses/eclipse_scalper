# Person 2 (Runtime/Ops) Checklist

Branch: `codex/runtime/ops-foundation`

## Completed Fixes (committed & pushed)

| Commit | Severity | Fix |
|--------|----------|-----|
| `5cecc40` | P1 | Upgrade drift alerts, add notification fallback JSONL, fix encoding |
| `8767ae6` | P1 | Ensure critical dirs exist + early kill switch state load on boot |
| `e4bead6` | P2 | Fix encoding: remove BOM and smart quotes across 5 runtime files |
| `b4d16b2` | P1 | Add post-scaling size validation before order submission |
| `37e2367` | P1 | Harden dashboard rate limit (default ON), fix OOM in status_snapshot, fix Telegram HTML injection |
| `88ee636` | **P0** | Kill switch exception handler was fail-open -> fail-closed (`entry.py:1333`) |
| `daacdc4` | P1 | Fix OOM: intent_ledger journal fallback read entire file -> tail 512KB |
| `583790f` | Tech | Consolidate `_symkey()`: 15 duplicate defs -> single import from runtime_helpers |
| `975507f` | Tech | Remove dead code: management.py, management_omega.py, guardian.py (-482 lines) |
| `236b852` | P2 | Fix potential OOM: dashboard ops_health_history -> tail 512KB |
| `55aada3` | Test | Add runtime safety test suite: 31 tests covering all ops fixes |
| `3558b67` | P2 | Enhance status_snapshot: kill switch, positions, decisions, feed freshness |
| `3a3ac31` | P1 | Fix HTML injection in telegram_bot replies + add /kill command |
| `d32cfb6` | P2 | Fix OOM: tail-based reading in tailer, performance_monitor, telegram_bot |
| `5ffb69a` | Tech | Add runtime/ops improvement roadmap (12 items, 4 weeks) |
| `e6d4400` | P1 | Atomic disk writes: 5 files hardened with tmp+rename pattern |
| `dec1c4c` | P1 | Startup validation gates: symbols, hedge mode, API permissions |
| `059c3e9` | P1 | Margin/liquidation alerts + position stuck detection in guardian |
| `8d1925f` | Tech | Consolidated /api/risk-overview endpoint + roadmap update |

## Full Audit Status

### execution/ (38 files)

| File | Status | Notes |
|------|--------|-------|
| entry.py | FIXED | Kill switch fail-closed (P0) |
| entry_loop.py | Clean | Kill switch -> `continue` (fail-closed), CancelledError re-raised |
| entry_loop_full.py | Clean | Same pattern, shutdown/CancelledError handling |
| entry_watch.py | Clean | Lock-guarded polling, CancelledError re-raised |
| exit.py | Clean | `exit_sent` guard on all paths, CancelledError re-raised |
| order_router.py | Clean | Spread/impact/notional guards, bounded retries |
| emergency.py | Clean | Truth-first flatten, live verification, forced escalation |
| data_loop.py | Clean | CancelledError re-raised, shutdown check |
| guardian.py | Clean | Watchdog + exchange probe |
| reconcile.py | Clean | Encoding fix + priority upgrade |
| bootstrap.py | Clean | mkdir + encoding |
| position_manager.py | Clean | Stable client IDs, proper locking |
| rebuild.py | Clean | Orphan handling |
| telemetry.py | Clean | Guardian-safe |
| shutdown_control.py | Clean | Traced shutdown event |
| belief_controller.py | Clean | Never blocks exits |
| belief_evidence.py | Clean | Proper freshness scoring |
| data_quality.py | Clean | Helper functions |
| intent_ledger.py | FIXED | OOM on journal fallback -> tail 512KB |
| adaptive_guard.py | Clean | Offset-based tail reading, expired cleanup |
| alpha_gate.py | Clean | Atomic writes, fallback chain |
| anomaly_guard.py | Clean | Simple pause check |
| health_gate.py | Clean | Escalation paths, degradation tracking |
| reliability_gate_runtime.py | Clean | Cached mtime, bounded scores |
| guard_knobs.py | Clean | Simple dataclass |
| state_machine.py | Clean | Strict FSM transitions |
| replace_manager.py | Clean | Envelope cap, state machine |
| runtime_helpers.py | Clean | Central utility functions |
| error_codes.py | Clean | Constants + mapper |
| diagnostics.py | Clean | Read-only, never raises |
| bot_factory.py | Clean | Never-fatal guarantees |
| telemetry_recovery.py | Clean | Cached state with TTL |
| passive_execution_simulator.py | Clean | Research code, no side effects |
| sim/min_exec_sim.py | Clean | Research code |
| sim/price_oracle.py | Clean | Research code |
| management.py | Dead code | Not imported, bypasses router |
| management_omega.py | Dead code | Not imported, bypasses router |

### bot/

| File | Status | Notes |
|------|--------|-------|
| core.py | Clean | `_trade_allowed()` returns False on exception (fail-closed) |
| runner.py | Clean | Early kill switch state load |

### brain/

| File | Status | Notes |
|------|--------|-------|
| state.py | Clean | `KNOWN_EXIT_IDS_CAP=50000` caps unbounded growth |
| persistence.py | Clean | Atomic writes, SHA256 checksums, lz4, IO lock, backup rotation, caps |
| performance_memory.py | Clean | Atomic tmp+replace writes, online mean, EMA |

### risk/

| File | Status | Notes |
|------|--------|-------|
| kill_switch.py | Clean | Disk persistence, post-crash cooldown |

### exchanges/

| File | Status | Notes |
|------|--------|-------|
| binance.py | Clean | Proper symbol resolution |
| base.py | Clean | Abstract interface |
| validator.py | Clean | Symbol purity filter |
| mock.py | Clean | Test scaffolding |

### notifications/

| File | Status | Notes |
|------|--------|-------|
| telegram.py | FIXED | HTML injection via `html.escape()` |
| manager.py | Clean | Fallback JSONL |
| events.py | Clean | |
| health_alerts.py | Clean | |
| daily_summary.py | Clean | |
| x_twitter.py | Clean | |

### dashboard/backend/

| File | Status | Notes |
|------|--------|-------|
| app.py | FIXED | Rate limit default ON, stale key cleanup |
| control_actions.py | Clean | Whitelisted actions, path traversal protection |
| data_sources.py | Clean | Masked sensitive config, hardcoded SQL columns |
| models.py | Clean | Pydantic response models |
| tailer.py | Clean | SSE log streaming |

### monitoring/

| File | Status | Notes |
|------|--------|-------|
| status_snapshot.py | FIXED | OOM -> tail-based 256KB read |

### Top-level

| File | Status | Notes |
|------|--------|-------|
| main.py | Clean | Proper dry-run handling |
| guardian.py | Dead code | Standalone script, not imported, bypasses router |
| signal_check.py | N/A | Standalone analysis tool |
| settings.py | Clean | 5-line bridge |

## Resolved Tech Debt

| Item | Status | Commit |
|------|--------|--------|
| `_symkey()` duplication (~20 copies) | DONE — 15 files consolidated to single import | `583790f` |
| Dead code (management.py, management_omega.py, guardian.py) | DONE — deleted (-482 lines) | `975507f` |
| Dashboard ops_health_history.jsonl OOM risk | DONE — tail 512KB read | `236b852` |
| Runtime safety test coverage | DONE — 31 tests added | `55aada3` |
| Telegram bot HTML injection | DONE — html.escape + /kill command | `3a3ac31` |
| OOM: tailer, perf_monitor, tg_bot | DONE — tail 256-512KB reads | `d32cfb6` |
| Status snapshot enhancement | DONE — kill switch, positions, decisions | `3558b67` |
| Dashboard: liq alert card (issue #3) | DONE — cherry-picked from branch | `c39ecfd` |
| Dashboard: spread-stress card (issue #4) | DONE — card + watchlist | `7371805` |
| Dashboard: fill-toxicity card (issue #5) | DONE — toxicity metrics | `7371805` |
| Dashboard: latency-stress card (issue #6) | DONE — latency metrics | `7371805` |
| Dashboard: watchboard overview card (issue #7) | DONE — banner + lane table | `f7c770c` |
| Kill-switch path/schema mismatch fix | DONE — correct path + field names | `4b5f0b2` |
| Kill-switch silent exception logging | DONE — 3 except blocks now log warnings | `813dce8` |
| Main sync | DONE — merged 90+ commits from main | `bd02d84` |
