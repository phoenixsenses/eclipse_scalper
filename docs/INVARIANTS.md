# INVARIANTS.md

Hard contracts for Eclipse Scalper / CryptoLion.

Related docs:
- Operating behavior: `docs/AGENTS.md`
- System map: `docs/ARCHITECTURE.md`

If any invariant fails: stop promoting changes, run incident playbook (section 6), and fix before further feature work.

## 1) Execution Invariants

### EXE-01: Order router idempotency
Contract:
- Same intent must not produce duplicate live exchange orders.

Why:
- Duplicate orders create uncontrolled exposure.

How it breaks:
- retry without idempotency key,
- race between loops,
- intent ID split without mapping.

Detect:
- intent/order logs show multiple submissions for one intent key.

Enforce:
- stable intent/correlation IDs in `execution/order_router.py` and `execution/intent_ledger.py`.
- TODO test: `tests/test_order_router_idempotency.py` (add if missing).

### EXE-02: Intent lifecycle completeness
Contract:
- No intent remains permanently in created-only limbo after actionable path starts.

Why:
- auditability and operational truth.

How it breaks:
- early return after creation,
- exception path without terminal record.

Detect:
- lifecycle debug tools show unresolved-after-created > 0.

Enforce:
- terminal transitions on all post-create branches.
- TODO test: `tests/test_order_router_intent_lifecycle.py` (add if missing).

### EXE-03: Flatten / kill-switch precedence
Contract:
- flatten and kill-switch overrides must dominate entry logic.

Why:
- safety over alpha.

How it breaks:
- entry loop bypasses kill state,
- delayed propagation of emergency flag.

Detect:
- entries created while kill-switch active.

Enforce:
- preserve checks in `risk/kill_switch.py`, `execution/circuit_breaker.py`, `execution/entry_loop.py`.

### EXE-04: Restart safety
Contract:
- restart/bootstrap must converge incomplete runtime state to safe, reconciled state.

Why:
- crash/restart is normal; stale partial state is dangerous.

How it breaks:
- persistence mismatch,
- reconcile not run or incorrectly scoped.

Detect:
- unresolved intents/positions after boot,
- repeated orphan/adoption loops.

Enforce:
- `execution/bootstrap.py` + `execution/reconcile.py` restore/reconcile paths.

### EXE-05: Risk bounds respected
Contract:
- position sizing and notional constraints must not exceed configured limits.

Why:
- prevents runaway risk.

How it breaks:
- unit conversion bugs,
- bypassed verifier path.

Detect:
- size/notional in logs exceeds configured cap.

Enforce:
- checks in `risk/risk_manager.py`, `execution/order_verifier.py`, `execution/order_router.py`.

## 2) Data + Research Invariants

### DAT-01: No lookahead bias
Contract:
- signal at `t` may use only data available at `t`.

Why:
- leakage invalidates all edge claims.

How it breaks:
- centered windows,
- future-index labels used in feature calc.

Detect:
- alignment tests fail; suspiciously high forward metrics collapse live.

Enforce:
- tests: `tests/test_micro_edge_alignment.py`.
- code review for feature windows in `tools/micro_edge_lib.py`, `tools/micro_edge_signal_v2.py`.

### DAT-02: Trade timing alignment
Contract:
- `signal_idx < entry_idx < exit_idx`; horizon mapping must be consistent.

Why:
- timing mismatch creates fake win-rate/PnL mismatch.

How it breaks:
- label and backtest use different entry conventions.

Detect:
- debug analyzer disagreements across smoke/backtest.

Enforce:
- tests: `tests/test_micro_edge_alignment.py`, `tests/test_micro_edge_backtest_signs.py`.

### DAT-03: Deterministic passive simulation
Contract:
- `execution/passive_execution_simulator.py` is deterministic per `(seed,event_id)`.

Why:
- reproducible research and ranking.

How it breaks:
- unseeded random branch,
- event_id instability.

Detect:
- repeated run with same inputs gives different outputs.

Enforce:
- tests: `tests/test_passive_realistic_sim.py`, `tests/test_passive_adverse_mult.py`.

### DAT-04: Cost unit correctness
Contract:
- bps-to-ratio conversion must be exact and single-applied.

Why:
- 10x scaling bugs can invert conclusions.

How it breaks:
- double counting spread/fee,
- treating bps as ratio.

Detect:
- impossible average costs, failed cost-model tests.

Enforce:
- tests: `tests/test_exec_cost_models.py`, `tests/test_micro_edge_backtest_metrics.py`.

### DAT-05: Debug JSONL schema stability
Contract:
- debug JSONL rows remain valid JSON and backward-compatible for core fields.

Why:
- analyzers/sweeps rely on stable keys.

How it breaks:
- renamed fields, mixed malformed rows.

Detect:
- analyzer parse errors, invalid line counts.

Enforce:
- tests: `tests/test_analyze_micro_edge_debug.py`, `tests/test_micro_edge_jsonl.py`.

## 3) Validation Invariants

### VAL-01: True forward splits
Contract:
- validation must use future slice; no overlap leakage from discovery.

Why:
- protects against optimistic overfit.

How it breaks:
- split indexing error,
- thresholds computed on full data.

Detect:
- forward validator fails synthetic collapse tests.

Enforce:
- tests: `tests/test_validate_micro_edge_forward.py`, `tests/test_validate_pocket_forward_api.py`.

### VAL-02: Candidate ranking reproducibility
Contract:
- same inputs produce same ranking order and scores.

Why:
- ranking is a decision gate.

How it breaks:
- unstable parser, nondeterministic aggregation.

Detect:
- repeated rank runs differ.

Enforce:
- tests: `tests/test_rank_passive_pockets_forward.py`.

### VAL-03: Candidate parsing integrity
Contract:
- PASS rows from sweep sources must parse to non-zero candidates when present.

Why:
- zero-candidate silent failures hide valid pockets.

How it breaks:
- fixed-index markdown parser,
- unhandled header variants.

Detect:
- parse counters: `rows_with_pass_yes > 0` but `candidates_unique == 0`.

Enforce:
- tests: `tests/test_rank_passive_pockets_forward.py`.

## 4) Safety Invariants

### SAF-01: Secrets never logged
Contract:
- API keys/tokens/private secrets must not appear in logs/reports.

Detect:
- grep scans in CI/ops checks.

Enforce:
- redaction discipline in logging code (`utils/logging.py`, integration modules).

### SAF-02: Paper-trading guard
Contract:
- paper-mode settings must prevent live order placement.

Why:
- prevents accidental live execution during research.

Detect:
- runtime logs indicate live exchange order in paper run.

Enforce:
- strict routing path checks in `exchanges/paper_trading.py` and execution controls.
- TODO test: `tests/test_paper_mode_no_live_orders.py`.

## 5) Invariant Test Suite Mapping
Current explicit coverage in `tests/`:
- DAT-01 / DAT-02: `tests/test_micro_edge_alignment.py`
- DAT-03: `tests/test_passive_realistic_sim.py`, `tests/test_passive_adverse_mult.py`
- DAT-04: `tests/test_exec_cost_models.py`, `tests/test_micro_edge_backtest_metrics.py`, `tests/test_micro_edge_backtest_signs.py`
- DAT-05: `tests/test_analyze_micro_edge_debug.py`, `tests/test_micro_edge_jsonl.py`
- VAL-01: `tests/test_validate_micro_edge_forward.py`, `tests/test_validate_pocket_forward_api.py`
- VAL-02 / VAL-03: `tests/test_rank_passive_pockets_forward.py`

TODO coverage gaps (recommended):
- EXE-01: `tests/test_order_router_idempotency.py`
- EXE-02: `tests/test_order_router_intent_lifecycle.py`
- SAF-02: `tests/test_paper_mode_no_live_orders.py`

## 6) Invariant Incident Playbook
If any invariant is violated:
1. Freeze promotion/deployment of new changes.
2. Capture evidence:
   - failing test output,
   - relevant logs (`logs/*.log`, `logs/*.jsonl`, `runtime/*.jsonl`),
   - exact command/config used.
3. Contain risk:
   - if execution-related: set paper/off mode and verify kill-switch readiness.
4. Reproduce deterministically:
   - minimal CLI/test case.
5. Patch minimally:
   - smallest possible fix with explicit regression test.
6. Re-validate:
   - compile touched files,
   - targeted tests + `pytest -q`.
7. Document:
   - update report/runbook/doc note with root cause and verification.

## 7) Quick Verification Commands
```powershell
pytest -q

python -m tools.rank_passive_pockets_forward --help
python -m tools.validate_passive_pocket_forward --help
python -m tools.micro_edge_backtest --help
```
