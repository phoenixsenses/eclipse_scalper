# Execution Reliability System Summary

This document explains the current execution reliability architecture as implemented in the repo.  
Primary anchors: `docs/execution_reliability.md`, `docs/execution_principles.md`, `execution/reconcile.py`, `execution/order_router.py`, `tools/test_belief_controller_unit.py`, `tools/test_order_router_unit.py`.

## Why this exists

Execution does not run in a strongly consistent environment. Exchange APIs, websocket streams, and local process state can disagree in timing and content. The goal of this system is not perfect correctness at every instant; it is bounded wrongness with fast recovery and protected exits.

## 1) Mental model

### Exchange truth vs local truth

Local state is treated as a belief snapshot, not ground truth. Exchange-observed state is authoritative but delayed and sometimes contradictory. This is explicit in doctrine (`docs/execution_principles.md`) and implemented by reconcile-first correction in `execution/reconcile.py`.

### Placement is speculative; reconciliation is reality

`execution/order_router.py` handles create/cancel/replace attempts under uncertainty.  
`execution/reconcile.py` establishes what exists now (positions, open protective orders, drift), then updates reliability metrics and policy posture.

### Unknown order is ambiguity, not a plain exception

Unknown-order responses are routed through idempotent-safe handling and state transitions, not treated as immediate hard failure. In router flow this appears in error classification and cancel idempotency behavior (`execution/order_router.py`).

## 2) Operating doctrine and invariants

The following invariants are the system contract under stress.

### Protective exits are never blocked

- Entry safety rails are scoped to entry paths; router leverage clamps use `is_exit=False` gating (`execution/order_router.py`).
- Exit-like intents (reduce-only, stop, TP) are modeled as always allowed in controller intent logic (`execution/belief_controller.py`).
- Tests: exit paths bypass entry caps in `tools/test_order_router_unit.py` (`test_first_live_safe_caps_not_applied_to_exits`, `test_resolve_leverage_applies_belief_guard_cap_for_entries_only`).

### Cancel is idempotent

- `cancel_order` treats unknown/already-gone as success, with a best-effort status conflict check to reject false positives when order is still open (`execution/order_router.py`).
- Tests: `test_all_unknown_is_success`, `test_unknown_cancel_with_open_fetch_order_is_not_success` in `tools/test_order_router_unit.py`.

### No infinite loops

- Router retries are bounded by attempt count and elapsed-time ceiling (`ROUTER_RETRY_MAX_ELAPSED_SEC`) in `execution/order_router.py`.
- Cancel/replace attempts are bounded and emit a terminal give-up event.
- Tests: `test_retry_max_elapsed_caps_attempts`, `test_cancel_replace_is_bounded_when_cancel_fails` in `tools/test_order_router_unit.py`.

### Local state can be rebuilt from exchange state

- Reconcile continuously re-aligns local `state.positions` using exchange positions/open orders and repairs inconsistencies (`execution/reconcile.py`).
- Reliability doctrine also documents restart rebuild and reconcile-first recovery (`docs/execution_reliability.md`).

### Risk exposure decreases monotonically as belief degrades

- Belief controller maps debt score to lower notional/leverage and higher confidence/cooldown constraints (`execution/belief_controller.py`).
- Test proof: `test_monotone_risk_mapping` in `tools/test_belief_controller_unit.py`.

## 3) Belief state and belief debt

### What belief state contains

Reconcile computes and publishes:

- `belief_debt_sec`
- `belief_debt_symbols`
- `mismatch_streak`
- `belief_confidence`
- repair metadata (`repair_actions`, `repair_skipped`)

These fields are emitted in `execution.belief_state` and `reconcile.summary` (`execution/reconcile.py`).

### How debt is computed and why growth matters

In reconcile:

- mismatch evidence is tracked per symbol with first/last timestamps
- debt age is the max unresolved mismatch age
- debt breadth is active mismatched symbol count

In controller:

- `debt_score = debt_sec / BELIEF_DEBT_REF_SEC + debt_symbols * BELIEF_SYMBOL_WEIGHT + mismatch_streak * BELIEF_STREAK_WEIGHT`
- `debt_growth_per_min` is computed from score slope between updates

Growth rate captures bursts that should trigger protective posture before absolute debt fully accumulates (`execution/belief_controller.py`, `tools/test_belief_controller_unit.py` `test_growth_burst_trips_red`).

### How debt is surfaced

- Raw telemetry events: `execution.belief_state`, `reconcile.summary` (`execution/reconcile.py`)
- Dashboard and summary tooling consume these events: `tools/telemetry_dashboard_page.py`, `tools/telemetry_alert_summary.py`, `tools/telemetry_dashboard_notify.py`
- Unit coverage for telemetry parsing exists in `tools/test_telemetry_belief_state_unit.py`

## 4) Reconcile mechanics

### Evidence sources currently used

`execution/reconcile.py` relies on:

- `fetch_positions` (all or selected symbols)
- `fetch_open_orders` for protective-order checks
- local `state.positions`
- run-context stores for phantom tracking, repair cooldowns, and belief bookkeeping

### What reconcile does each cycle

1. Build exchange position map and mark confirmed beliefs.
2. Detect and optionally adopt orphan exchange positions.
3. Detect phantom local positions missing on exchange and clear after grace/miss thresholds.
4. Correct drifted size/entry/side metadata.
5. Verify stop coverage and trigger bounded repair/refresh behavior.
6. Update mismatch/debt metrics and emit telemetry.
7. Update belief controller and publish guard knobs; optionally request kill-switch halt on RED.

### How reconcile avoids happy-path optimism

- Failed evidence fetches increase mismatch pressure instead of being ignored.
- Ambiguity persists until cleared by evidence or timeout logic.
- Repair activity is throttled with cooldown to prevent order spam and thrash.

## 5) Belief controller

### Inputs

Current direct inputs to `BeliefController.update` are:

- `belief_debt_sec`
- `belief_debt_symbols`
- `mismatch_streak`

Internal controller memory provides prior score/time for growth-rate estimation (`execution/belief_controller.py`).

Inference note: "worst symbols" is not a direct controller input today. Symbol-level mismatch tracking exists in reconcile metrics, but controller currently consumes aggregated debt signals.

### Outputs (guard knobs)

Controller outputs `GuardKnobs` with:

- `allow_entries`
- `max_notional_usdt`
- `max_leverage`
- `min_entry_conf`
- `entry_cooldown_seconds`
- `max_open_orders_per_symbol`
- `mode` (`GREEN`, `YELLOW`, `ORANGE`, `RED`)
- `kill_switch_trip`

### Hysteresis and RED burst trip

- Upward transitions require persistence windows, except RED can trip immediately.
- Downward transitions require lower hysteresis threshold and sustained recovery window.
- Growth bursts can force RED even before high absolute debt.

Tests: `test_hysteresis_prevents_flapping`, `test_growth_burst_trips_red` in `tools/test_belief_controller_unit.py`.

### Critical scope rule: entry-only application

Guard knobs constrain exposure expansion. Exit/protection flows remain exempt:

- Router applies belief leverage cap only when `is_exit=False` (`execution/order_router.py`).
- Exit invariant is verified by tests in `tools/test_order_router_unit.py`.

## 6) Router hardening

### Entry clamps, exit safety

Router applies belief-controller leverage cap and entry safety rails on entry-like traffic only. Exit/protective flow paths remain pass-through for safety (`execution/order_router.py`).

### Idempotency and correlation

- Deterministic, length-safe `clientOrderId` generation (Binance-safe size constraints)
- Stable per-intent `correlation_id` carried through retry/cancel/create telemetry
- Duplicate/too-long client-order-id scenarios handled by bounded sanitized variants

### Error policy and retry discipline

- Policy classes are explicit: retryable, retryable-with-modification, idempotent-safe, fatal (via error policy integration in router)
- Retries use capped exponential backoff with jitter and max elapsed time
- Every retry emits one structured `order.retry` event rather than uncontrolled line spam

### Telemetry task hardening

Router telemetry scheduling includes safe coroutine cleanup when no active loop is available, preventing coroutine leak warnings in edge contexts (`execution/order_router.py` `_telemetry_task` behavior).

## 7) Tests as proof of behavior

### Belief controller tests

`tools/test_belief_controller_unit.py` verifies:

- monotonic risk mapping as debt rises
- hysteresis behavior to avoid mode flapping
- debt-growth burst transition to RED
- exit-intent safety invariant

### Router tests

`tools/test_order_router_unit.py` verifies:

- idempotent cancel semantics for unknown/already-gone orders
- open-status conflict rejection on unknown cancel responses
- bounded create retry and bounded cancel/replace loops
- entry-only safety rails and exit exemption behavior
- belief cap applied to entries only

### Commands and interpretation

Core commands used for this reliability slice:

- `pytest -q tools/test_belief_controller_unit.py`
- `pytest -q tools/test_order_router_unit.py`
- `pytest -q` (full suite)

Passing means these invariants are executable checks, not doctrine-only statements.

## 8) End-to-end trade narrative (unknown-order + delayed ack)

Concrete timeline:

1. Signal path requests an entry; router mints stable identity (`clientOrderId`, `correlation_id`).
2. Create call returns timeout or ambiguous transport error.
3. Router classifies the error, emits `order.retry`, and retries within bounded policy.
4. A cancel/replace path sees unknown-order on cancel; idempotent-safe rules treat it as ambiguous success unless status evidence says still open.
5. Reconcile tick fetches positions/open orders and compares them with local state.
6. If mismatch exists, reconcile records mismatch/debt and emits `execution.belief_state` and `reconcile.summary`.
7. Belief controller recomputes posture:
   - healthy: entries continue with baseline knobs
   - degraded: tighter notional/leverage and stricter confidence/cooldown
   - burst/severe: RED mode with halt request
8. Protective exits remain executable regardless of entry posture.
9. Subsequent reconcile passes clear ambiguity or keep debt elevated until evidence resolves.

This closes the loop: evidence -> belief -> policy -> constrained action -> safer evidence acquisition.

## 9) Incident reasoning guide

During incidents, debug in this order:

1. Evidence: what did exchange snapshots/events show at each reconcile cycle?
2. Belief: how did `belief_debt_sec`, symbol count, and mismatch streak evolve?
3. Policy: what mode/knobs did controller output and when did transitions occur?
4. Action: were entry actions clamped while exits stayed open?
5. Reconstruction: can one intent be replayed via `correlation_id` and structured events?

If these are consistent, the system is handling uncertainty as designed; if not, look for invariant violation.

## 10) Claim-to-code map

- Reliability doctrine and invariant wording: `docs/execution_principles.md`, `docs/execution_reliability.md`
- Debt computation and reconcile telemetry: `execution/reconcile.py`
- Policy transformation and hysteresis: `execution/belief_controller.py`
- Entry-only router clamps and idempotent cancel behavior: `execution/order_router.py`
- Controller invariant tests: `tools/test_belief_controller_unit.py`
- Router idempotency/retry/exit-scope tests: `tools/test_order_router_unit.py`

## 11) Known boundaries and explicit inferences

- The controller currently consumes aggregate debt metrics, not a direct per-symbol "worst symbols" vector.
- Current reconcile implementation is primarily REST-snapshot driven; this summary does not claim a dedicated weighted multi-sensor fusion module because none is present in the referenced anchors.
- Where behavior is inferred (for example, how downstream loops consume knobs beyond router scope), inference is based on available callsites and tests, not undocumented assumptions.
