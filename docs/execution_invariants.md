# Execution Invariants

This document defines non-negotiable reliability invariants for execution under network/exchange uncertainty.

## Core Invariants

1. Protective exits are never policy-blocked.
- Entry gates (`allow_entries`, confidence floors, cooldowns, leverage/notional caps) apply only to exposure expansion.
- Reduce-only, stop, TP, and flatten intents remain executable even when entries are halted.

2. Reconciliation is source of truth.
- Local state can be stale or wrong; reconcile can overwrite local position assumptions.
- Execution correctness is judged by eventual exchange-aligned state, not local placement success.

3. Order intent identity is stable.
- One intent carries one bounded `correlation_id` through create/retry/cancel/replace.
- `clientOrderId` variants never grow unbounded and remain exchange-safe length.

4. Cancel is idempotent by default.
- Unknown/already-gone outcomes are treated as successful cancel objective unless explicit open-status conflict exists.
- Repeated cancel requests must not create side effects or retry storms.

5. Retry behavior is bounded and classified.
- Retries occur only for retryable classes and stop at configured attempt/time boundaries.
- Fatal classes fail fast and are escalated through telemetry/error code surfaces.

6. No infinite loop on cancel/replace or guard checks.
- Loops have max-attempt bounds and explicit give-up path.
- Give-up triggers reconcile/alert semantics instead of tight repeat attempts.

7. Belief debt drives posture, not strategy alpha.
- Execution uncertainty (mismatch streak, debt age, debt growth) can tighten entries.
- Strategy signal logic is unchanged unless safety requires constraining action.

8. Kill-switch semantics are debt-aware.
- Circuit break can trip on mismatch storms, debt growth bursts, or repeated fatal classes.
- Recovery requires cooldown and fresh evidence, not a single optimistic check.

## Violation Signals (Operational)

- Any protective exit path blocked by entry guard is a critical invariant breach.
- Repeated order attempts without bounded termination is a critical invariant breach.
- Missing `correlation_id` in create/retry/cancel telemetry is an observability breach.
- Reconcile mismatch growth without posture tightening/escalation is a control-loop breach.
