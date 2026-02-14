---
name: Reliability Incident Review
about: Post-incident review for execution reliability, belief state, and guard behavior
title: "[Incident] "
labels: reliability, incident, postmortem
assignees: ""
---

## Incident Summary

- Date/time (UTC):
- Environment (dry-run/live):
- Symbols affected:
- Severity (`low | medium | high | critical`):
- Short description:

## Detection

- Detection source (`alert | dashboard | manual | workflow`):
- First alert/event:
- Time to detect:

## Timeline (UTC)

List key events in order.

1. 
2. 
3. 

## Evidence and Replay

- Telemetry path:
- Journal path:
- Replay command used:

```bash
python tools/replay_trade.py --journal logs/execution_journal.jsonl
```

- Correlation IDs involved:
- Replay outcome summary:
- Any replay mismatch categories (`ledger | transition | belief`):

## Belief / Control Posture During Incident

- Belief debt level and trend:
- Mode transitions (`GREEN/YELLOW/ORANGE/RED`):
- Guard knobs in effect (entry confidence, notional/leverage caps, cooldown):
- Was reconcile-first active:

## Invariant Check

- [ ] Protective exits were never blocked.
- [ ] No infinite create/cancel/replace loops occurred.
- [ ] Worst-case exposure remained bounded.
- [ ] Reconcile remained authority over speculative outcomes.
- [ ] State remained replayable/auditable.

If any invariant failed, describe exactly where and why:

## Root Cause Analysis

### Primary cause
- 

### Contributing factors
- 
- 

### Why existing controls did not fully prevent it
- 

## Impact

- Execution impact:
- Financial/risk impact:
- Operational impact:

## Corrective Actions

### Immediate containment
- [ ] 
- [ ] 

### Short-term fixes (1-2 weeks)
- [ ] 
- [ ] 

### Long-term hardening
- [ ] 
- [ ] 

## Test and Validation Plan

- [ ] Added unit test(s) for the failure path.
- [ ] Added adversarial/chaos scenario for this incident class.
- [ ] Confirmed no regressions in related reliability suites.

Commands/results:
```bash
pytest -q
```

## Telemetry / Alerting Follow-up

- New/updated events:
- Dashboard updates:
- Alert threshold/policy changes:
- Notify dedup/decision changes:

## Documentation Follow-up

- [ ] Updated relevant docs (`execution_reliability`, `execution_system_summary`, roadmap/runbook).
- [ ] Added operator notes for future triage.

Docs touched:
- 
- 

## Owner and Due Dates

- Incident owner:
- Fix owner(s):
- Target completion date:

## Closure Criteria

- [ ] Corrective actions merged.
- [ ] Tests and chaos checks passing.
- [ ] Invariants verified under reproduced scenario.
- [ ] Monitoring confirms stable behavior post-fix.
