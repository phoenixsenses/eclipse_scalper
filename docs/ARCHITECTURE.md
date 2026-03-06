# ARCHITECTURE.md

System map for Eclipse Scalper / CryptoLion.

Related docs:
- Agent operating rules: `docs/AGENTS.md`
- Hard contracts: `docs/INVARIANTS.md`

## 1) High-Level Data/Control Flow
```text
                 +---------------------------+
                 |   integrations/telegram   |
                 | telegram_control/notifier |
                 +------------+--------------+
                              |
                              v
+---------+      +------------+-------------+      +----------------------+
| config/ | ---> | execution/bootstrap.py   | ---> | execution/guardian.py|
| settings|      | bot_factory + preflight  |      | health/permissions   |
+---------+      +------------+-------------+      +----------+-----------+
                              |                               |
                              v                               v
                     +--------+---------+            +--------+---------+
                     | execution/entry  |            | risk/*, circuit  |
                     | loop + decision  |            | breaker/kill     |
                     +--------+---------+            +--------+---------+
                              |                               |
                              v                               |
                     +--------+---------+                     |
                     | execution/order  |<--------------------+
                     | router + verifier|
                     +--------+---------+
                              |
                              v
                       +------+------+
                       | exchanges/* |
                       | binance/paper|
                       +------+------+
                              |
                              v
                     +--------+---------+
                     | execution/reconcile|
                     | + position_manager |
                     +--------------------+

Persistence/brain sidecar:
brain/state.py, brain/persistence.py, brain/performance_memory.py,
state/*.json, runtime/*.jsonl, execution/intent_ledger_persistence.py

Research pipeline:
data/microstructure.db -> tools/micro_edge_* -> reports/*.md|*.json
```

## 2) Major Subsystems

### Execution Subsystem
Primary files:
- `execution/bootstrap.py`
- `execution/guardian.py`
- `execution/entry_loop.py`
- `execution/order_router.py`
- `execution/reconcile.py`
- `execution/position_manager.py`
- `execution/intent_ledger.py`

Responsibilities:
- bootstrap: initialize services, restore state, start loops.
- guardian/health: readiness and environmental gating.
- entry loop: candidate generation and entry gating.
- order router: intent -> exchange action with lifecycle journaling.
- reconcile: eventually-consistent truth alignment with exchange/paper state.
- position manager: ongoing position lifecycle (stop/TP/restore paths).

### Risk and Safety Subsystem
Primary files:
- `risk/kill_switch.py`
- `execution/circuit_breaker.py`
- `execution/protection_manager.py`
- `execution/guard_knobs.py`

Responsibilities:
- hard stop/flatten controls,
- drawdown/rate constraints,
- runtime gating and protective transitions.

### Brain/State/Persistence
Primary files:
- `brain/state.py`
- `brain/persistence.py`
- `brain/performance_memory.py`
- `execution/intent_ledger_persistence.py`
- `state/*.json`

Responsibilities:
- persist recoverable state,
- restore after restart,
- keep scoring/performance memory for adaptive behavior.

### Integrations
Primary files:
- `integrations/telegram_control.py`
- `integrations/telegram_notifier.py`
- `notifications/telegram.py`
- `execution/alert_spool.py`

Responsibilities:
- operator control surface,
- asynchronous alert spooling/replay for reliability.

### Microstructure Research Subsystem
Primary files:
- `tools/micro_edge_backtest.py`
- `tools/sweep_passive_realistic_filters.py`
- `tools/validate_passive_pocket_forward.py`
- `tools/rank_passive_pockets_forward.py`
- `tools/micro_edge_signal_v2.py`
- `tools/micro_edge_lib.py`
- `execution/passive_execution_simulator.py`

Inputs/outputs:
- inputs: `data/microstructure.db`, debug JSONL under `logs/`
- outputs: `reports/*.md`, `reports/*.json`

## 3) Authoritative Loops and Ownership
- Bootstrap owns service startup order and restart restore behavior.
- Entry loop owns candidate selection and pre-router gating.
- Order router owns intent state transitions and exchange submission semantics.
- Reconcile owns eventual truth correction against exchange/paper reality.
- Position manager owns post-entry lifecycle maintenance.

Idempotency expectation:
- repeated loop ticks must not duplicate side effects for same intent/event.

## 4) State Machine Overview (Conceptual)
Intent path (contractual intent):
1. intent created
2. submission attempted
3. terminal state (filled/done/blocked/cancelled)

Position path:
1. opened/recognized
2. managed (risk/exit updates)
3. closed/reconciled

Recovery path:
- restart -> bootstrap restore -> reconcile correction -> stable loop operation.

## 5) Reliability Strategy
- Eventual consistency: reconcile is allowed to repair temporary drift.
- Retries/backoff around exchange and persistence boundaries.
- Stable identifiers (`intent_id`, event keys) for dedupe/audit.
- Journaling for causal traceability (`runtime/*.jsonl`, logs).

Failure scenarios and expected behavior:
- Network/API outage: health/guardian blocks entries; retries continue.
- Partial fill / delayed exchange ack: router + reconcile converge state.
- Crash/restart: bootstrap restore + reconcile repair incomplete paths.

## 6) Micro-Edge Research Architecture
Pipeline layers:
1. feature preparation from microstructure series (`tools/micro_edge_lib.py`, `tools/micro_edge_signal_v2.py`)
2. execution-aware simulation (`tools/micro_edge_backtest.py` + `execution/passive_execution_simulator.py`)
3. pocket search (`tools/sweep_passive_realistic_filters.py`)
4. forward validation (`tools/validate_passive_pocket_forward.py`)
5. robustness ranking (`tools/rank_passive_pockets_forward.py`)

Expected outputs:
- pocket sweeps: `reports/FILTER_SWEEP_*.md`
- forward reports: `reports/*FORWARD*.md`
- ranking JSON/MD: `reports/PASSIVE_POCKET_RANKING*.json|md`

## 7) Observability Conventions
- human logs: `logs/*.log`
- machine logs: JSONL in `logs/*.jsonl` and `runtime/*.jsonl`
- reports: `reports/*.md`, `reports/*.json`

Minimum expected diagnostics for critical tools:
- parsed row counts,
- skipped row counts with sample reasons,
- pass/fail counters,
- deterministic seed/config echo in report headers.

## 8) Where to Change What

### If changing execution flow
Touch (at minimum):
- `execution/entry_loop.py`
- `execution/order_router.py`
- `execution/reconcile.py`
- related risk/guardian modules
And update:
- lifecycle/reliability tests (if present),
- runbook/report notes under `docs/`.

### If changing signal/rule logic
Touch:
- `tools/micro_edge_lib.py`
- `tools/micro_edge_signal_v2.py`
- `tools/micro_edge_backtest.py`
And update:
- sweep/forward/rank tool compatibility,
- tests in `tests/test_micro_edge_*`,
- report docs in `reports/`.

### If changing data schema/parsers
Touch:
- data readiness/probe tools (`tools/check_data_ready.py`, `tools/data_layer_probe.py`)
- parser-dependent research tools.
Must add:
- compatibility handling and parser diagnostics,
- tests for old/new format parsing.

## 9) Verification Checklist
- compile touched Python files,
- run `pytest -q`,
- run relevant end-to-end CLI for the affected subsystem,
- confirm report/log counters are non-zero and sensible.
