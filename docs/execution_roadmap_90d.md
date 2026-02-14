# Execution Reliability 90-Day Roadmap

## Objective
Move from Level 4 to Level 5 execution maturity:

- From adaptive reliability controls
- To a self-regulating, adversarially hardened execution organism with CI-enforced invariants

This roadmap is ordered by risk reduction per engineering effort.

## Guiding Invariants
All milestones must preserve:

- Protective exits are never blocked.
- No infinite create/cancel/replace loops.
- Worst-case exposure stays bounded under ambiguity.
- Reconciliation remains authority over speculative request outcomes.
- Every critical decision is replayable from journal + evidence.

## Milestone 1 (Weeks 1-2): Durable Intent Ledger Persistence

### Goal
Make idempotency restart-safe and deterministic.

### PRs
1. `execution: persist intent ledger lifecycle to journal-backed store`
2. `execution: restart rebuild policy for orphan/adopt/cancel/freeze`
3. `tests: crash/restart scenarios for duplicate-exposure prevention`

### Likely file touch list
- `execution/intent_ledger.py`
- `execution/event_journal.py`
- `execution/rebuild.py`
- `execution/reconcile.py`
- `execution/bootstrap.py`
- `tools/test_intent_ledger_unit.py`
- `tools/test_rebuild_unit.py`
- `tools/test_replay_trade_unit.py`

### Test gates
- Restart during active intent does not create duplicate entry.
- Orphan classification is deterministic and journaled.
- Replay reconstructs pre-crash and post-restart state transitions.

### Exit criteria
- Crash/restart produces no net unintended exposure increase.

## Milestone 2 (Weeks 3-4): Replace Safety Envelope Hardening

### Goal
Make replace safe under adversarial interleavings.

### PRs
1. `execution: replace protocol envelope with bounded ambiguity budget`
2. `execution: replace fallback policy (cancel->reconcile->re-enter/flatten)`
3. `tests: fill-after-cancel and out-of-order replace race coverage`

### Likely file touch list
- `execution/replace_manager.py`
- `execution/order_router.py`
- `execution/state_machine.py`
- `execution/reconcile.py`
- `execution/telemetry.py`
- `tools/test_replace_manager_unit.py`
- `tools/test_order_router_unit.py`
- `tools/test_execution_chaos_scenarios.py`

### Test gates
- Replace race cannot breach configured exposure envelope.
- Replace storms terminate with bounded attempts and clear terminal state.
- Replace ambiguity increments debt and posture tightens.

### Exit criteria
- Replace path is bounded and safe under delayed/lost/out-of-order signals.

## Milestone 3 (Weeks 5-6): Multi-Source Evidence Contradiction Weighting

### Goal
Upgrade confidence/debt from staleness-only signals to weighted evidence quality.

### PRs
1. `belief: weighted evidence model for ws/rest/fills freshness and agreement`
2. `belief: contradiction severity and debt attribution`
3. `tests: contradiction-vs-staleness posture behavior`

### Likely file touch list
- `execution/belief_evidence.py`
- `execution/belief_controller.py`
- `execution/reconcile.py`
- `execution/reliability_gate_runtime.py`
- `tools/test_belief_evidence_unit.py`
- `tools/test_belief_controller_unit.py`
- `tools/test_reliability_gate_runtime_unit.py`

### Test gates
- WS down + healthy REST degrades modestly (not instant RED).
- WS/REST contradiction escalates faster than simple staleness.
- Confidence/debt changes are monotone and explainable.

### Exit criteria
- Posture reacts to evidence quality, not just elapsed time.

## Milestone 4 (Weeks 7-8): Runtime Replay Divergence to Posture Clamp

### Goal
Turn replay mismatch into immediate live mitigation input.

### PRs
1. `runtime gate: classify replay divergence categories`
2. `adaptive guard: clamp entries on critical replay divergence`
3. `dashboard/alerts: divergence category visibility and trend`

### Likely file touch list
- `execution/reliability_gate_runtime.py`
- `execution/entry_loop.py`
- `execution/adaptive_guard.py`
- `tools/replay_trade.py`
- `tools/reliability_gate.py`
- `tools/telemetry_dashboard_notify.py`
- `tools/telemetry_dashboard_page.py`
- `tools/test_reliability_gate_unit.py`
- `tools/test_telemetry_dashboard_notify_unit.py`

### Test gates
- Injected replay divergence clamps posture within one control cycle.
- Divergence categories are emitted and rendered in artifacts.
- Exits remain unaffected by entry clamp.

### Exit criteria
- Replay correctness failures trigger automatic risk reduction.

## Milestone 5 (Weeks 9-10): Chaos CI Required Gate

### Goal
Make adversarial scenarios merge-blocking quality checks.

### PRs
1. `chaos: required scenario matrix for execution reliability`
2. `ci: enforce invariant gates on chaos scenarios`
3. `docs: chaos runbook and failure triage guide`

### Required scenarios
- ACK arrives after fill.
- Cancel returns unknown while order is gone.
- Fill occurs after cancel ack.
- Replace race with reordered stream events.
- WS down while REST healthy.
- REST stale with WS lag.
- Retry storm pressure.

### Likely file touch list
- `tools/test_execution_chaos_scenarios.py`
- `tools/test_execution_chaos_scenarios_v2.py`
- `.github/workflows/ci-tests.yml`
- `docs/execution_reliability.md`

### Test gates
- No infinite loops.
- Exits always allowed.
- Exposure envelope never exceeded.
- Degradation and recovery align with policy.

### Exit criteria
- PRs cannot merge if chaos invariants fail.

## Milestone 6 (Weeks 11-12): Recovery Ladder and Operator Introspection

### Goal
Prevent whiplash recovery and reduce incident diagnosis time.

### PRs
1. `belief controller: staged RED->ORANGE->YELLOW->GREEN recovery ladder`
2. `telemetry: unlock-conditions and stability-window explanation`
3. `dashboard: why constrained + what unlocks next panel`

### Likely file touch list
- `execution/belief_controller.py`
- `execution/guard_knobs.py`
- `execution/entry_loop.py`
- `tools/telemetry_dashboard_page.py`
- `tools/telemetry_dashboard_notify.py`
- `tools/test_belief_controller_unit.py`
- `tools/test_telemetry_dashboard_page_unit.py`

### Test gates
- No mode flapping near thresholds.
- Recovery requires sustained stability window.
- No instant full-risk resume from RED.

### Exit criteria
- Recovery is slower and safer than degradation by design.

## Cross-Cutting Metrics (Track Weekly)

- Ambiguity duration (mean/95p time to resolve unknown states).
- Ambiguity surface area (active ambiguous intents/symbols).
- Replay divergence count by category.
- Replace ambiguity and terminal give-up rates.
- Protective coverage gap seconds.
- Time spent in ORANGE/RED by cause.
- Entry clamp reasons (count and trend).

## Definition of Done Per PR

- States the invariant(s) it strengthens.
- Includes targeted unit tests plus at least one adversarial scenario when applicable.
- Emits structured telemetry for new behavior.
- Updates one doc/runbook section for incident handling.

## Risk-Reduction Priority Order

1. Durable intent persistence.
2. Replace safety envelope.
3. Runtime replay divergence clamp.
4. Chaos CI required gate.
5. Multi-source contradiction weighting.
6. Recovery ladder and unlock introspection.

## Recommended Working Cadence

- Weekly: one reliability PR + one test/chaos PR.
- Biweekly: roadmap checkpoint against metrics above.
- End of each milestone: replay one real incident journal and verify improved boundedness.
