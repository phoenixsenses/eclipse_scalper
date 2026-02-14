# Execution Reliability Design (`eclipse_scalper`)

## Purpose
Define a production execution reliability design that treats execution as belief management under uncertainty and enforces safety through explicit invariants, reconciliation authority, and idempotent action protocols.

## Background Topics
- Asynchronous distributed systems and partial observability
- Safety vs liveness tradeoffs
- Idempotency and retry semantics
- Event-sourced recovery patterns
- Hysteresis and dwell-time control
- Tail-risk and correlation regime tightening

## Problem Framing
`eclipse_scalper` operates across an adversarial asynchronous boundary (exchange, network, timing skew). Request success is evidence, not truth. Reconciliation defines operational truth.

Design objectives:
- Bound worst-case risk while belief is uncertain
- Recover deterministically after faults/restarts
- Keep exits and protection always available
- Make every decision explainable and replayable

## Reference Architecture Mapped to Existing Modules

### Evidence Ingestion and Health
- `execution/belief_evidence.py`
- `execution/reconcile.py`

Responsibilities:
- Collect WS/REST/fill-derived evidence
- Score source health via freshness, error rate, and practical coverage
- Emit contradiction and confidence signals

### Belief Estimation and Debt
- `execution/reconcile.py`
- `execution/belief_controller.py`

Responsibilities:
- Maintain uncertainty-aware belief fields
- Track belief debt (age, breadth, depth, burn)
- Promote contradictions into debt, not silent overwrite

### Posture and Guard Policy
- `execution/belief_controller.py`
- `execution/reliability_gate_runtime.py`

Responsibilities:
- Convert belief + runtime gate signals into entry knobs
- Degrade fast, recover slow (hysteresis + persistence)
- Keep exits exempt from entry clamps

### Idempotent Action Plane
- `execution/order_router.py`
- `execution/replace_manager.py`
- `execution/intent_ledger.py`

Responsibilities:
- Intent-scoped create/cancel/replace behavior
- Retry policy by class with bounded backoff
- Replace safety envelope and bounded attempt budgets

### Rebuild and Orphan Handling
- `execution/rebuild.py`
- `execution/intent_ledger.py`
- `execution/event_journal.py`

Responsibilities:
- Reconstruct local state after restart
- Deterministic orphan decisions: `ADOPT/CANCEL/FREEZE`
- Preserve invariants during recovery

### Protection and Position Consistency
- `execution/protection_manager.py`
- `execution/position_manager.py`
- `execution/reconcile.py`

Responsibilities:
- Enforce protection coverage over filled exposure
- Anti-churn protection updates
- Reconcile missing/partial protection safely

### Replayability and Traceability
- `execution/event_journal.py`
- `execution/intent_ledger.py`
- telemetry tooling under `tools/`

Responsibilities:
- Append-only event trail for intent/action/evidence outcomes
- Reconstruct lifecycle from `intent_id` + `correlation_id`
- Emit dominant causes and unlock conditions for operators

## Minimal Implementable Belief Envelope

Per symbol:
- `position_interval = [pos_min, pos_max]`

Per order/intent:
- `live_state_set subset of {UNKNOWN, LIVE, PARTIAL, FILLED, CANCELED}`
- `filled_qty_interval = [fill_min, fill_max]`
- `live_qty_interval = [live_min, live_max]`

Rules:
- Ambiguity widens intervals/sets immediately
- Narrowing requires consistent higher-authority evidence
- Policy acts against worst-case member of the envelope

## Practical Coverage Model
Use a coverage model unless reliable sequence IDs exist:
- `freshness_sec`: age of last valid update
- `expected_interval_sec`: expected cadence for source
- `coverage_ratio`: observed/expected updates over sliding window
- `gap_events`: inferred missing periods or reconnect gaps
- `error_rate`: failures per window

Coverage influences confidence and contradiction severity.

## Non-Negotiable Invariants
1. Exits and protective actions are never blocked by entry posture logic.
2. Retries are bounded; no infinite action loops.
3. Cancel is idempotent for unknown/already-closed outcomes.
4. Reconciliation is authority; API acks are evidence only.
5. Local state is rebuildable from journal + exchange evidence.
6. Risk clamps are monotone conservative as belief degrades.
7. No duplicate risk-increasing side effects per `intent_id`.
8. Replace allowed only when worst-case old+new live exposure stays within budget; otherwise degrade to safer path.

## Policy Semantics

### Degrade Fast
Immediate tightening/freeze when:
- runtime gate is degraded
- contradiction burn spikes
- source coverage collapses
- protection coverage gap persists

### Recover Slow
Recovery requires:
- healthy streak + persistence window
- minimum coverage threshold
- contradiction-clear duration
- warmup stage where configured

### Entry-Only Coupling
Runtime gate and belief controller affect:
- `allow_entries`
- `max_notional_usdt`
- `max_leverage`
- `min_entry_conf`
- `entry_cooldown_seconds`

These controls never block risk-reducing exits.

## Replace and Restart Safety

### Replace Safety Envelope
Before replace:
- Compute worst-case exposure if old stays live while new becomes live
- If unsafe, degrade to `cancel -> reconcile -> create`
- Enforce per-symbol and global replace budgets

### Restart Safety
On boot:
- Rebuild intent lifecycle from journal
- Reconcile against exchange snapshots
- Classify external unknowns via `ADOPT/CANCEL/FREEZE`
- Keep posture conservative until uncertainty narrows

## Observability Contract
Each lifecycle should be reconstructable with:
- `intent_id`
- `correlation_id`
- action attempts/outcomes
- reconcile outcomes
- posture/cause summary at decision time

Operator output should include:
- dominant contributors
- current posture
- unlock conditions and remaining requirements

## Detailed Enhancements and Roadmap

### PR-1: Evidence Coverage + Contradiction Tightening
Changes:
- Refine coverage metrics in `execution/belief_evidence.py` and `execution/reconcile.py`
- Standardize contradiction fields and burn-rate output

Tests:
- Coverage monotonicity under stale/missing updates
- Contradiction severity vs staleness-only degradation
- Integration: contradictory evidence clamps faster than stale-only evidence

### PR-2: Belief Envelope Minimal Contract
Changes:
- Normalize envelope fields from reconcile outputs:
  - position interval
  - live state set
  - qty intervals
- Ensure controller debt logic consumes envelope uncertainty

Tests:
- Ambiguity widens envelopes immediately
- Narrowing requires consistent evidence
- Tightening uncertainty never loosens worst-case risk bound

### PR-3: Idempotency Hardening by `intent_id`
Changes:
- Strengthen `execution/intent_ledger.py` + router use paths
- Ensure retries and restart resend paths preserve one risk-increasing effect per intent

Tests:
- Duplicate create retry with same `intent_id` does not duplicate exposure
- Restart with pending intent does not amplify exposure
- Invariant: no duplicate risk-increasing side effects per `intent_id`

### PR-4: Replace Envelope Enforcement
Changes:
- Enforce pre-replace exposure envelope in `execution/replace_manager.py`
- Add fallback path when envelope unsafe
- Add replace attempt budgets

Tests:
- Replace denied when old+new worst-case exceeds budget
- Fill-after-cancel and reorder scenarios remain bounded
- Replace storms are bounded and escalate posture

### PR-5: Restart Rebuild + Orphan Determinism
Changes:
- Tighten `execution/rebuild.py` with journal/ledger bootstrap consistency
- Deterministic `ADOPT/CANCEL/FREEZE` decisions with reason tags

Tests:
- Restart scenarios: mid-entry, mid-replace, mid-partial-fill
- Deterministic orphan classification outcomes
- Restart does not increase net risk unexpectedly

### PR-6: Protection Coverage Hard Invariant
Changes:
- Enforce coverage TTL + anti-churn in `execution/protection_manager.py` and reconcile integration
- Escalate posture when unresolved coverage gaps persist

Tests:
- Dribble fills do not spam replaces
- Missing stop repaired within TTL or posture escalates
- Filled exposure cannot remain silently uncovered

### PR-7: Runtime Reliability Coupling Explainability
Changes:
- Continue `runtime_gate_cause_summary` alignment across reconcile/controller/alerts
- Ensure unlock conditions are explicit and stable

Tests:
- Cause summary drives expected clamps
- Reconcile -> controller -> alert summary remains consistent
- Regression: exits always remain allowed

### PR-8: Required Chaos Gate in CI
Changes:
- Add compact required chaos scenarios:
  - unknown-order ambiguity
  - replace race
  - contradiction burst
  - persistent protection gap

Gates:
- Fail on invariant violation
- Fail if posture does not degrade under required faults
- Fail on replay/rebuild inconsistency

## Definition of Done
A reliability change is complete only if:
- It explicitly strengthens at least one invariant
- It emits traceable telemetry/journal fields
- It includes at least one adversarial test
- It preserves exit-always-safe behavior
