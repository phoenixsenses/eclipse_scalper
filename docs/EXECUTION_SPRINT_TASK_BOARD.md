# Execution Upgrade Task Board

## Scope
This board implements:
- Sprint 1 Integration Hardening
- Sprint 2 Calibration + Reality Match
- Sprint 3 Production Discipline + Rollout

All changes stay behind `EXEC_*` flags until canary completion.

## Sprint 1 - Integration Hardening

### Ticket S1-1 - Wire `ExecutionEngine` into backtest path
- Files:
  - `tools/micro_edge_backtest.py`
- Done:
  - `EXEC_ENGINE_UNIFIED=1` routes net-return computation through `ExecutionEngine` adapter.
- Acceptance:
  - Flag off: legacy net path unchanged.
  - Flag on: no crash, same sign semantics and bounded drift.

### Ticket S1-2 - Wire `ExecutionEngine` into paper path
- Files:
  - `src/microphys/sim/papertrade.py`
- Done:
  - `PaperTradeConfig.use_unified_engine` added.
  - Env fallback `EXEC_ENGINE_UNIFIED`.
  - Emits `order_id` on unified path.
- Acceptance:
  - `tests/test_papertrade_unified_engine.py` passes.

### Ticket S1-3 - Lifecycle event publishing via `EventBus`
- Files:
  - `src/microphys/live/daemon.py`
- Done:
  - Added lifecycle bus writer (`logs/live_execution_events.jsonl`).
  - Publishes `order_intent`, `order_ack`, `fill`/`reject`.
  - Validates events via contract validators.
- Acceptance:
  - `exec_event_bus_enabled` true -> nonzero lifecycle event count in status.

### Ticket S1-4 - FSM enforcement + contract violation kill hook
- Files:
  - `src/microphys/live/daemon.py`
- Done:
  - Order lifecycle is passed through `OrderFSM`.
  - Violations logged to risk events.
  - Optional hard kill via `EXEC_KILL_ON_CONTRACT_VIOLATION=1`.
- Acceptance:
  - `tests/test_live_daemon_runtime_hooks.py` violation case passes.

### Ticket S1-5 - RuntimeSupervisor heartbeat hooks
- Files:
  - `src/microphys/live/daemon.py`
  - `src/microphys/live/config.py`
  - `tools/run_live_papertrade.py`
- Done:
  - Supervisor state evaluated each loop when enabled.
  - Fail state emits `logs/live_supervisor.json` and stops daemon with rc=2.
- Acceptance:
  - `tests/test_live_daemon_runtime_hooks.py` supervisor fail-fast passes.

## Sprint 2 - Calibration + Reality Match

### Ticket S2-1 - Daily calibration job
- Files:
  - `tools/calibrate_execution_models.py` (extend)
  - `tools/schedule_online_calibration.py` (wire daily)
- Target:
  - Persist latency distributions, queue params, adverse tables by symbol/session.
- Acceptance:
  - Daily artifact emitted under `state/` or `reports/`.

### Ticket S2-2 - Replay-vs-observed drift dashboard
- Files:
  - `tools/replay_parity_report.py`
  - `tools/execution_diagnostics.py`
  - `tools/toxicity_report.py`
- Target:
  - One report summarizing fill-rate delta, fill-delay delta, adverse delta.
- Acceptance:
  - `reports/REPLAY_PARITY_REPORT.md` + `reports/EXECUTION_HEALTH.md` generated.
  - Orchestrated via `tools/execution_e2e_pipeline.py`.

### Ticket S2-3 - Drift alert thresholds
- Files:
  - `src/microphys/live/alerts.py`
  - `src/microphys/live/daemon.py`
- Target:
  - Alert when parity/latency/toxicity drift exceeds thresholds.
- Acceptance:
  - Alerts appended to `logs/live_alerts.jsonl`.

### Ticket S2-4 - Regime/session specific params
- Files:
  - `src/microphys/execution/calibration.py`
  - `src/microphys/live/daemon.py`
- Target:
  - Select calibrated profile by volatility/session buckets.
- Acceptance:
  - Status contains active profile id and source.

## Sprint 3 - Production Discipline + Rollout

### Ticket S3-1 - Canary rollout automation
- Files:
  - `docs/ROLLOUT_EXECUTION_V2.md`
  - `tools/post_rollout_audit.py`
- Target:
  - 1 symbol -> 2/3 symbols -> full rollout gating.
  - End-to-end check command available:
    - `python -m tools.execution_e2e_pipeline`
  - Single-command wrapper available:
    - `powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\run_execution_canary.ps1 -Symbol ETHUSDT -MaxCycles 5 -OpenReport`

### Ticket S3-2 - Recovery tests
- Files:
  - `tests/runtime/*`
  - `tests/parity/*`
- Target:
  - restart mid-order, duplicate replay, delayed ack/fill ordering.

### Ticket S3-3 - Chaos tests
- Files:
  - `tests/legacy_tools/test_execution_chaos_scenarios.py` (extend)
- Target:
  - feed lag spike, DB partial outage, exchange timeout bursts.

### Ticket S3-4 - SLO + runbook triggers
- Files:
  - `docs/ROLLOUT_EXECUTION_V2.md`
  - `tools/post_rollout_audit.py`
- Target:
  - explicit rollback criteria and automated pass/fail audit.

## Non-Negotiable Gates
1. Parity tests must stay green.
2. No side effects outside adapters when `EXEC_ENGINE_UNIFIED=0`.
3. New behavior stays behind flags until canary complete.
4. Every incident leaves replay artifact/log.

