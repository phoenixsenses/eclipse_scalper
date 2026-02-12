# Strategies Summary

## Purpose
This document summarizes the current strategy architecture in Eclipse Scalper as it exists today: trading logic, execution reliability controls, telemetry-driven adaptation, and safety invariants. It is written for engineers onboarding into maintenance or extension work.

The system is best understood as two coupled strategies:

- Alpha strategy: when and how to take market risk.
- Reliability strategy: how to bound risk when system belief diverges from exchange truth.

The reliability strategy is intentionally privileged over alpha under uncertainty.

## System Strategy Shape

### 1) Alpha layer (signal and execution intent)
- Generates entry/exit intents from signal components and filters.
- Uses confidence-based gating and optional confidence shaping knobs (for example power/min/max clamps) to adjust entry selectivity.
- Supports trend confirmation, regime filters, session guards, and momentum/vwap/ATR-oriented exits.

### 2) Reliability layer (belief and control posture)
- Treats order placement as speculative and reconciliation as authority.
- Tracks local confidence in exchange state through belief telemetry and debt indicators.
- Adapts entry posture (notional/leverage/confidence/cooldown) as belief quality degrades.
- Preserves protective exits as safety-critical operations that remain exempt from entry-only clamps.

## Operating Doctrine (Invariants)
The strategy stack is constrained by non-negotiable execution invariants:

- Protective exits are never blocked by entry safety posture.
- Cancel semantics are idempotent-safe under unknown/already-gone conditions.
- Retry and cancel/replace paths are bounded; no infinite loops.
- Reconciliation is the source of truth for drift correction.
- Risk posture tightens monotonically when belief health worsens.

These invariants are enforced by router/reconcile/controller logic and protected by targeted unit tests in `tools/`.

## Entry Strategy Summary

### Signal acceptance
- Entry is not signal-only; it is signal plus runtime health.
- Guard/filter checks can block, defer, or scale an otherwise valid signal.
- Near-miss and blocked-reason telemetry provide diagnostics for tuning.

### Adaptive controls
- Entry notional and leverage can be scaled down by guard pressure.
- Minimum confidence can be auto-raised when stress signals repeat.
- Cooldown windows grow under repeated pressure events.

### Pressure sources currently integrated
- Reconcile-first gate severity/streak pressure.
- Runtime reliability gate degradation.
- Partial-fill escalation and retry-alert patterns.
- Guard-history and telemetry feedback loops.

## Exit Strategy Summary

### Priority model
- Exit safety outranks entry liveness.
- Telemetry and anomaly states can accelerate/reinforce exit handling.
- Exit telemetry is aggregated into quality dashboards and summaries for tuning and incident review.

### Behavior themes
- Timeout/stagnation behavior with optional ATR scaling.
- Momentum/VWAP contextual exits.
- Telemetry-aware guard hooks for high-risk conditions.
- Post-close quality summaries (reason, duration, pnl windows).

## Sizing and Leverage Strategy

### Baseline
- Fixed notional or fixed quantity per symbol with optional overrides.

### Dynamic overlays
- Confidence-based scaling.
- Adaptive guard scaling from reliability signals.
- Correlation-aware group caps that reduce effective leverage as correlated exposure increases.

### Strategic effect
- Exposure becomes elasticity-driven: when reliability drops, risk compresses before full stop.

## Correlation and Portfolio Control
- Correlated symbol groups have dynamic leverage/notional dampening.
- Group-level constraints reduce hidden concentration during multi-symbol participation.
- Per-group overrides allow asymmetric control where market structure differs by cluster.

## Execution Reliability Strategy

### Router posture
- Uses explicit error-policy classification (retryable, retry-with-modification, idempotent-safe, fatal).
- Bounded retries with backoff/jitter and cap conditions.
- Deterministic client order identity and correlation IDs for lifecycle traceability.
- Idempotent cancel outcomes and bounded cancel/replace behavior.

### Reconcile posture
- Continuously validates local assumptions against exchange evidence.
- Tracks mismatch/debt indicators and emits belief state telemetry.
- Repairs protection gaps with cooldown/throttle to avoid thrash.

### Controller posture
- Converts belief debt and degradation trends into guard knobs:
  - `allow_entries`
  - notional/leverage caps
  - `min_entry_conf`
  - entry cooldown
  - health mode and trip posture
- Applies hysteresis and streak/burst handling to avoid flapping.

## Telemetry Strategy

### Telemetry role
- Not just observability: telemetry is now a control-plane input.
- Events feed dashboards, summaries, adaptive guard adjustments, and scheduled alerts.

### Current operational outputs
- Health and anomaly text artifacts.
- Dashboard HTML snapshots.
- Guard-history CSV/HTML timelines.
- Notifier state with transition-aware dedup semantics and reason trails.

### Alerting semantics
- Alerting distinguishes transitions from unchanged state.
- Critical alerts can be resent on worsened critical state, not on static noise.
- Decision reasons are persisted for incident audit (`initial_state`, level transition, worsened critical, unchanged skip).

## Tests-as-Strategy Proof
Strategy behavior is validated through focused tests rather than ad-hoc confidence:

- Telemetry notifier state transitions and dedup logic.
- Reconcile-first severity/streak thresholds.
- Dashboard rendering of reliability and guard-state summaries.
- Entry loop and router safety paths.
- Reliability gates and guard history wiring checks in workflow tests.

Passing tests indicate guard semantics and alert policy stayed stable under expected scenarios.

## End-to-End Narrative (Typical Stress Path)
1. Signal passes initial strategy checks and proposes entry.
2. Router attempts placement; exchange response is delayed/ambiguous.
3. Reconcile observes mismatch pressure and updates belief debt.
4. Belief controller tightens entry knobs (higher min confidence, lower exposure).
5. New entries are clamped or blocked while reconcile converges state.
6. Exits/protection remain available throughout.
7. Telemetry captures the full chain for dashboarding and alerts.

This flow demonstrates the core design intent: bounded wrongness first, opportunity second.

## What This Enables Next
The current strategy architecture is strong enough to support the next reliability enhancements cleanly:

- Durable intent persistence across restarts.
- Deeper replace-race protocol hardening.
- Multi-source evidence weighting for belief confidence.
- Chaos-scenario CI expansion as a release gate.
- Runtime replay divergence as automatic posture input.

## Related Documents
- `docs/execution_reliability.md`
- `docs/execution_system_summary.md`
- `docs/execution_invariants.md`
- `docs/belief_state_model.md`
- `docs/execution_fmea.md`
- `docs/observability_contract.md`
