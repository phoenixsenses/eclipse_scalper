# Research Worktree Promotion Triage

Date: 2026-03-10
Status: active triage note
Scope: decide what should be promoted from `eclipse_scalper-research`, what already exists in `main`, and what should remain research-only

## Summary

The important research/runtime monitoring surface is already largely present in `main`.

This means the next step is not broad migration. The next step is selective promotion and cleanup:

- do not re-import docs/tools that already exist in `main`
- keep experimental evaluation/calibration tooling in `eclipse_scalper-research`
- only promote items that add new operational value to runtime, dashboard, or repeatable research validation

## Already In Main

These are already present in the main repo and should not be re-promoted just because they also exist in the research worktree.

Docs:

- `RESEARCH_EVENT_OPERATOR_BRIEF.md`
- `RESEARCH_EVENT_SIGNAL_BRIDGE.md`
- `RESEARCH_EVENT_WATCHBOARD.md`
- `RESEARCH_EVENT_WATCHBOARD_CYCLE.md`
- `RESEARCH_EVENT_WATCHBOARD_HISTORY.md`
- `RESEARCH_EVENT_WATCHBOARD_TREND.md`
- `RESEARCH_EVENT_WATCHBOARD_TREND_FROM_HISTORY.md`
- `RESEARCH_LIQ_REVERSAL_E2E_PLAN.md`
- `RESEARCH_LIQ_REVERSAL_SURFACE_NOTES.md`
- `RESEARCH_MICROSTRUCTURE_AUDIT.md`

Tools:

- `check_event_lanes.py`
- `daily_research_report.py`
- `validate_canonical.py`
- `validate_microstructure_contract.py`

## Promote If Operational Need Appears

These are worth promoting only if they become part of the repeatable runtime/dashboard/research loop.

- `event_lane_consolidation.py`
- `event_lane_overlap.py`
- `event_lane_persistence_policy.py`
- `event_lane_suppression_policy.py`
- `event_watchboard_effective.py`
- `event_watchboard_snapshot_append.py`
- `event_watchboard_trend.py`
- `event_watchboard_trend_from_history.py`
- `research_event_watchboard.py`
- `run_research_event_watchboard_cycle.py`

Promotion gate for this group:

- clear consumer in runtime or dashboard
- test coverage
- stable output path/contract
- no duplicate existing mainline tool

## Keep Research-Only

These should stay in `eclipse_scalper-research` unless they are explicitly adopted into a production research workflow.

- `analyze_micro_edge_debug.py`
- `analyze_micro_edge_regimes.py`
- `micro_edge_backtest.py`
- `micro_edge_gate_export.py`
- `micro_edge_lib.py`
- `micro_edge_report.py`
- `micro_edge_signal_v2.py`
- `micro_edge_smoke.py`
- `micro_edge_sweep.py`
- `sweep_micro_edge_costs.py`
- `sweep_micro_edge_exec_models.py`
- `sweep_micro_edge_gates.py`
- `validate_micro_edge_debug_split.py`
- `validate_micro_edge_forward.py`
- `validate_passive_pocket_forward.py`

Reason:

- experimental strategy evaluation
- calibration-heavy
- not part of the current operational spine

## Archive / Ignore

These do not need active promotion planning.

- generated `reports/`
- transient `localtests/`
- one-off comparison outputs
- stale handoff artifacts duplicated by newer docs in `main`

## Recommended Next Steps

1. Keep using `main` as the operational source of truth
2. Promote only one watchboard-policy tool at a time if a real consumer appears
3. Do not mix strategy-eval tooling into runtime branches by default
4. Periodically clean `eclipse_scalper-research` generated artifacts so triage stays readable

## Current Recommendation

No immediate mass-promotion is needed.

The highest-value next move after this note is to continue runtime/dashboard work from `main` and only cherry-pick focused research helpers when they have a clear operational consumer.
