# Execution Principles

Execution reliability is treated as belief management under uncertainty.

## Core Invariants
- Bounded wrongness over perfect correctness.
- Identity over speed: one intent keeps one stable identity across retries.
- Reconciliation is the source of truth; placement is provisional.
- Cancel operations are idempotent-safe, including unknown-order ambiguity.
- Protective exits are never blocked by entry safety rails.
- No unbounded loops in cancel/replace/retry paths.

## Operational Semantics
- Local state is a belief snapshot, not truth.
- Exchange responses are evidence with delay, omission, and duplication.
- Unknown-order is a valid epistemic state, not an exceptional one.
- Partial fills are default behavior, not edge behavior.

## Guardrails
- Retry logic must be classified by error class and bounded by attempts + elapsed time.
- Reconcile mismatch and belief debt are first-class signals.
- Circuit-breakers may trip on belief-debt growth, not only transport/API errors.
- Telemetry must reconstruct one intent life-cycle using correlation IDs.
