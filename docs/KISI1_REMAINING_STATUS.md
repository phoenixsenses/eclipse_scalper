# Kisi 1 Remaining Status

Date: 2026-03-09
Scope: research/data follow-up items after the main research-runtime pipeline landed in `main`

## Summary

The core Kisi 1 research/data spine is already present in the main repo.

This means the following are no longer primary backlog items:

- event lane gate
- regime recovery watchdog
- pocket promotion checklist
- daily research report
- liq reversal E2E chain
- research watchboard/dashboard integration
- canonical validation baseline
- microstructure feature pipeline baseline

The remaining work is second-wave hardening, documentation, and cleanup.

## Completed Core Spine

- Canonical validation baseline exists via `tools/validate_canonical.py`
- Microstructure feature pipeline exists via `data/features/micro_features.py`
- Event lane gate exists via `execution/event_lane_gate.py`
- Regime recovery watchdog exists via `tools/watch_regime_recovery.py`
- Pocket promotion checklist exists via `tools/pocket_promotion_checklist.py`
- Daily research report exists via `tools/daily_research_report.py`
- Dashboard research refresh exists via `tools/refresh_dashboard_research_events.py`
- Liq reversal E2E chain exists via `tools/run_liq_reversal_e2e.py`
- Alert/risk lane surface exists in monitoring and dashboard docs/tooling

## Remaining Work

### P1

- Define and publish a canonical microstructure data contract
- Add a deterministic fixture/sample DB for repeatable research tests
- Implement a research fitness validator for data/research readiness checks
- Complete `canonical_symbol()` vs `symkey()` cleanup and standardize symbol identity usage

### P2

- Close status/docs gaps for research event lanes and watchboard integration
- Decide which research-only tools/docs should be promoted from `eclipse_scalper-research`
- Define a cleanup policy for generated `reports/` and `localtests/` artifacts
- Audit and classify pre-existing test failures outside the newly landed work

## Partial But Needs Closure

- Research event lanes are implemented, but status closure is still informal
- Watchboard integration exists in code and dashboard, but status documentation is behind implementation
- Execution hardening is partially landed, but not yet summarized as a single closure checkpoint

## Recommended Order

1. Microstructure data contract
2. Deterministic fixture/sample DB
3. Research fitness validator
4. Symbol canonicalization cleanup
5. Status/docs closure
6. Research worktree promotion triage
7. Artifact cleanup policy

## Notes

- `eclipse_scalper-research` still contains many research-only docs, experimental tools, and generated outputs.
- That worktree should not be treated as a clean source of truth without filtering.
- The main repo already contains the operational subset required for runtime, dashboard, and research monitoring workflows.
