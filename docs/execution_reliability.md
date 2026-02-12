# Execution Reliability Hardening

## Reliability Findings (Read-First Audit)
- `execution/order_router.py`: retries were bounded, but retries did not emit a structured event on every retry attempt, and retry classification did not expose an explicit policy class.
- `execution/order_router.py`: duplicate `clientOrderId` handling could rotate IDs, but tracing one intent across retries/cancel paths was difficult without a stable `correlation_id`.
- `execution/order_router.py`: cancel was idempotent for unknown-order patterns, but did not cross-check fetch status to detect unknown-order false positives.
- `execution/reconcile.py`: reconcile repaired drift/stop protection but did not publish a reconcile mismatch streak for circuit-breaking.
- `risk/kill_switch.py`: kill-switch had API/data/equity checks, but no explicit reconcile-mismatch storm trigger.

## What Changed

### 1) Router Error Taxonomy + Retry Policy Surface
- Added explicit error policy module in `execution/error_policy.py`:
  - `retryable`
  - `retryable_with_modification`
  - `idempotent_safe`
  - `fatal`
- Router now consumes `classify_order_error_policy(...)` so retries carry policy metadata (`reason`, `code`, `error_class`) from a single mapping surface.
- Kept existing reason/code mapping behavior and extended reduce-only classification to be retryable-with-modification.
- Unknown/unmapped error text now defaults to conservative `fatal` classification to avoid retry storms.

### 2) Deterministic IDs + Correlation
- Added deterministic, length-safe client ID generation:
  - `_make_client_order_id(..., bucket=...)`
  - Stable format with intent-aware prefix and hash.
  - No cumulative growth across retries.
- Added intent prefix helper: `_intent_client_order_prefix(...)` (`ENTRY`, `TP`, `SL`, `EXIT` buckets).
- Added per-intent `correlation_id` generation: `_make_correlation_id(...)`.
- Routed `correlation_id` into router logs and telemetry payloads.

### 3) Retry Visibility + Backoff Boundaries
- Router now emits one structured telemetry event (`order.retry`) for each retry attempt with:
  - `error_class`, `reason`, `code`, `tries`, `attempt`, `correlation_id`.
- Existing exponential backoff + jitter + `ROUTER_RETRY_MAX_ELAPSED_SEC` cap remains in effect.

### 4) Cancel Idempotency Hardening
- `cancel_order(...)` now supports `correlation_id`.
- Added `_fetch_order_status_best_effort(...)` and conflict check:
  - unknown/already-gone cancel is success unless fetched status indicates still open/new/partially_filled.
- Extended cancel telemetry to include `status` and `correlation_id`.

### 5) Cancel/Replace Guarded Helper
- Added `cancel_replace_order(...)` helper with max attempts and clear give-up telemetry (`order.cancel_replace_giveup`), avoiding indefinite cancel/replace loops.

### 6) Reconcile as Source-of-Truth Signal
- `execution/reconcile.py` now tracks and publishes:
  - `mismatch_events`
  - `repair_actions`
  - `repair_skipped`
  - `mismatch_streak`
- Added reconcile stop repair cooldown (`RECONCILE_REPAIR_COOLDOWN_SEC`) to reduce repair thrash.
- Reconcile writes mismatch streak into `state.kill_metrics` for risk controls.
- Emits `reconcile.summary` telemetry events when mismatches/repairs occur.

### 7) Kill-Switch Circuit Breaker Extension
- Added reconcile-mismatch storm trigger in `risk/kill_switch.py`:
  - `KILL_RECONCILE_MISMATCH_STREAK_MAX`
  - `KILL_RECONCILE_MISMATCH_HALT_SEC`
- Trips halt when mismatch streak crosses threshold.

### 8) Belief Controller (Risk Posture From State Debt)
- Added `execution/belief_controller.py` and `execution/guard_knobs.py`.
- Reconcile now computes belief debt and drives a single guard-knob surface (`allow_entries`, `max_notional_usdt`, `max_leverage`, `min_entry_conf`, cooldown, mode).
- Entry loop consumes guard knobs as read-only policy inputs (entry blocks, confidence floor, cooldown, notional cap, open-order cap).
- Router now applies belief-driven `max_leverage` cap for entries only (`is_exit=False`) so protective exits are never policy-blocked.
- Belief controller emits trace fields into `execution.belief_state` for post-mortem replay.

### 9) Replace/Protection Reliability Layer
- Added `execution/replace_manager.py` with explicit bounded replace states:
  - `INTENT_CREATED`
  - `CANCEL_SENT_UNKNOWN`
  - `REPLACE_RACE`
  - `FILLED_AFTER_CANCEL`
  - `DONE`
- `order_router.cancel_replace_order(...)` now uses replace-manager outcomes and emits stateful give-up telemetry.
- Added `execution/protection_manager.py` stop-coverage evaluator:
  - detects close-position stop coverage
  - detects under-covered stop quantity
  - provides anti-churn refresh gating (`min_delta_ratio`, `max_refresh_interval_sec`)
- Reconcile now uses protection coverage to refresh undersized stops via bounded cancel/replace instead of only binary present/missing behavior.
- Added a strict protection coverage gap tracker with TTL:
  - per-symbol coverage gaps are tracked across reconcile ticks
  - TTL breaches emit critical reconcile telemetry and increase mismatch pressure
  - persistent gaps feed belief-controller posture tightening (entry-only clamp, exits still exempt)

### 10) Restart Rebuild + Chaos v2
- Added `execution/rebuild.py` for startup reconstruction:
  - rebuilds local positions from exchange positions
  - inspects open orders for non-reduce-only orphans
  - tags `state.run_context["rebuild"]` with summary/diagnostics
- Added bootstrap hook (`BOOT_REBUILD_ON_START=1` default) so rebuild executes before loops start.
- Optional fail-safe mode: `BOOT_REBUILD_FREEZE_ON_ORPHANS=1` halts entries when orphan entry intents are discovered at startup.
- Added chaos v2 recovery tests (`tools/test_execution_chaos_scenarios_v2.py`) to validate restart behavior and orphan freeze paths.

### 11) Canonical State Machine + Event Journal Skeleton
- Added `execution/state_machine.py` with typed order-intent and position-belief states plus transition validator.
- Invalid transitions raise explicit `TransitionError` (fail-fast on lifecycle bugs in tests/instrumentation).
- Added `execution/event_journal.py` append-only JSONL journal for lifecycle and rebuild events.
- Router now journals order intent transitions around submit/retry/ack/terminal paths (best-effort, non-fatal).
- Rebuild emits summary events and position-belief transitions for restart traceability.

### 12) Strict Transition Mode + Replay Tool
- `replace_manager` now validates lifecycle progression through `state_machine.transition(...)`.
- Strict mode toggle for replace paths: `ROUTER_REPLACE_STRICT_TRANSITIONS=1` (raises on invalid transition sequencing).
- Reconcile now tracks per-symbol position belief states and emits state-transition journal entries.
- Added replay helper: `tools/replay_trade.py` to reconstruct lifecycle transitions from `logs/execution_journal.jsonl`.
- Telemetry dashboard + alert/notify scripts now include replay snippets so lifecycle traces appear in HTML and Telegram summaries.

## Invariants Now Guaranteed
- Cancel path is idempotent for unknown/already-gone orders, with open-status conflict protection.
- Router retries are bounded by attempts and elapsed-time cap.
- Each retry attempt emits exactly one structured retry event.
- `clientOrderId` generation is deterministic, length-safe, and non-growing across retry variants.
- A `correlation_id` ties create/retry/cancel failure events for one order intent.
- Reconcile mismatch storms can trip kill-switch automatically.
- Protective entry safeguards (`FIRST_LIVE_SAFE`) remain entry-only (protective exits are not blocked by those caps).
- Entry-only leverage caps can be tightened by guard posture without affecting reduce-only/protective exit routing.
- Belief-controller posture transitions are bounded with persistence + recovery hysteresis.
- Entry orchestration authority is `execution.entry_loop.entry_loop`; bootstrap forces this path.
- Legacy `execution.entry.try_enter` is runtime-blocked by default via `ENTRY_ENABLE_LEGACY_TRY_ENTER=0` (set `1` only for controlled back-compat/debug use).
- Entry fill handling now emits staged protection hints into run-context (`entry_stage_hints`) so `position_manager` can prioritize urgent Stage-1 stop restoration without waiting for normal polling cadence.
- Stop/trailing placement is now routed through shared `execution.protection_manager` placement helpers, reducing divergence between `entry` and `position_manager` restore paths.
- Stage-1 emergency protection in `entry_loop` can assert into `run_context["protection_gap_state"]` (`entry.stage1_gap_assertion`) so reconcile/runtime gating can react when immediate post-fill protection placement fails.

## Assumptions
- Exchange adapter `fetch_order` may be unavailable; status conflict checks are best-effort.
- Existing strategy logic (signal/sizing math) is intentionally unchanged.
- Existing telemetry pipeline remains JSONL event-based without external dependency changes.

## Test & Repro Commands

### Unit tests (focused)
```powershell
python eclipse_scalper/tools/test_order_router_unit.py
python eclipse_scalper/tools/test_position_manager_unit.py
python eclipse_scalper/tools/test_kill_switch_reconcile_unit.py
python eclipse_scalper/tools/test_belief_controller_unit.py
python eclipse_scalper/tools/test_error_policy_unit.py
python eclipse_scalper/tools/test_replace_manager_unit.py
python eclipse_scalper/tools/test_protection_manager_unit.py
python eclipse_scalper/tools/test_rebuild_unit.py
python eclipse_scalper/tools/test_execution_chaos_scenarios_v2.py
python eclipse_scalper/tools/test_state_machine_unit.py
python eclipse_scalper/tools/test_event_journal_unit.py
python eclipse_scalper/tools/test_replay_trade_unit.py
```

### Router reliability test split

`order_router` reliability tests are split by concern to keep failures isolated and reviewable:

- `tools/test_order_router_replace_unit.py` for cancel/replace boundedness and transition enforcement.
- `tools/test_order_router_classifier_unit.py` for classifier/retry-policy safety ordering.
- `tools/test_order_router_idempotency_unit.py` for client-order-id idempotency and intent-ledger reuse.

### Full suite
```powershell
pytest
```

### Dry-run bootstrap scenario
```powershell
$env:TELEMETRY_ENABLED="1"
$env:TELEMETRY_PATH="logs/telemetry.jsonl"
$env:KILL_RECONCILE_MISMATCH_STREAK_MAX="6"
$env:PYTHONUTF8="1"
python main.py --dry-run
```
