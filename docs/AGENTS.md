# AGENTS.md

Canonical rules for human and AI agents working in this repo.

Related docs:
- Architecture: `docs/ARCHITECTURE.md`
- Non-negotiable contracts: `docs/INVARIANTS.md`

## 1) Golden Rules
1. Never break invariants in `docs/INVARIANTS.md`.
2. Any behavior change must include:
   - at least one test update/addition,
   - a report note or runbook note with exact repro commands.
3. Determinism is mandatory for research simulations:
   - same input + same seed => same output,
   - no hidden randomness.
4. No secrets in code, reports, logs, or test fixtures:
   - do not print API keys/tokens/env secrets.
5. No silent logging schema breaks:
   - keep JSONL field names stable,
   - additive changes only unless migration is explicit.
6. Safety first for execution paths:
   - paper/dryrun validation before any live-impacting behavior.

## 2) Scope and Change Boundaries
Default-safe boundary for research work:
- allowed: `tools/`, `tests/`, `docs/`, `reports/` (generated outputs)
- execution-risk areas: `execution/`, `exchanges/`, `risk/`, `bot/`, `brain/`

If touching execution-risk areas:
- add feature flags with safe defaults,
- preserve kill-switch/circuit-breaker semantics,
- include explicit rollback path.

## 3) Workflow (Task -> Patch -> Proof)
1. Define scope:
   - list exact files to touch,
   - list invariants affected.
2. Implement minimal diff:
   - avoid broad refactors during incident/fix work,
   - prefer importable helpers for reusable logic.
3. Validate:
   - compile checks (changed Python files),
   - targeted pytest,
   - end-to-end CLI smoke for the changed pipeline.
4. Document:
   - update relevant report/runbook/docs,
   - include exact PowerShell commands.
5. Deliver:
   - summarize root cause, patch, validation evidence.

## 4) Patch Etiquette
- Keep diffs small and local.
- Prefer feature flags over behavior replacement.
- Preserve backward-compatible CLI defaults.
- Commit message format should be explicit, e.g.:
  - `fix(rank): robust markdown candidate parser with PASS alias handling`
  - `feat(micro-edge): add v2 passive alpha feature enrichment`

## 5) Logging Policy
Use structured logs with stable fields.

Required principles:
- JSONL for machine pipelines (`logs/*.jsonl`, debug outputs).
- Add fields; avoid renaming/removing existing fields.
- Keep key identifiers stable (`symbol`, `rule_name`, `seed`, `split`, `intent_id`, `event_id`).
- Never log secrets, private tokens, or full credentials.

## 6) Determinism Rules (Research)
For tools in `tools/micro_edge_*` and `execution/passive_execution_simulator.py`:
- deterministic seed usage is required,
- event-level deterministic IDs/hashing must be stable,
- no dependence on wall-clock time inside scoring/simulation unless explicitly part of input.

When adding randomness:
- expose seed in CLI and outputs,
- include seed in JSON/markdown report headers.

## 7) API/CLI Design Rules
- Put reusable logic in importable functions.
- Keep CLI wrappers thin (`main()` only orchestration).
- Follow existing CLI style:
  - explicit flags (`--min-n`, `--min-n-frac`, `--maker-fee-bps-grid`),
  - defaults safe and documented.
- If a parser/validator is importable, keep return types structured (`dict` with per-row and aggregate sections).

## 8) Execution Safety Rules
Any change that can affect order placement must satisfy all:
1. paper-mode test first,
2. no regression of lifecycle invariants,
3. kill-switch + flatten remains dominant path,
4. recovery/reconcile behavior preserved.

Relevant files:
- `execution/entry_loop.py`
- `execution/order_router.py`
- `execution/reconcile.py`
- `execution/intent_ledger.py`
- `execution/bootstrap.py`
- `risk/kill_switch.py`
- `execution/circuit_breaker.py`

## 9) Standard Validation Commands
Compile touched Python files:
```powershell
python -m py_compile <file1.py> <file2.py> ...
```

Run tests:
```powershell
pytest -q
```

Micro-edge end-to-end (example):
```powershell
python -m tools.micro_edge_backtest --db data/microstructure.db --symbols BTCUSDT,ETHUSDT --lookback-min 1440 --bucket-sec 1 --horizon-sec 60 --rule intensity_spike_imbalance_cont --side auto --exec-model passive_realistic

python -m tools.sweep_passive_realistic_filters --db data/microstructure.db --symbols BTCUSDT,ETHUSDT --lookback-min 1440 --bucket-sec 1 --horizon-grid 30,60,120 --rule intensity_spike_imbalance_cont --side auto --out-md reports/FILTER_SWEEP_PASSIVE_REALISTIC.md

python -m tools.validate_passive_pocket_forward --db data/microstructure.db --symbol ETHUSDT --lookback-min 1440 --bucket-sec 1 --horizon-sec 60 --rule intensity_spike_imbalance_cont --side auto --min-imbalance 0.5 --min-trade-intensity 2500 --max-spread 0.00025 --splits 4 --seeds 11,22,33,44,55 --min-n 50

python -m tools.rank_passive_pockets_forward --db data/microstructure.db --lookback-min 1440 --bucket-sec 1 --rule intensity_spike_imbalance_cont --side auto --candidates-md reports/FILTER_SWEEP_PASSIVE_REALISTIC_ETH.md,reports/FILTER_SWEEP_PASSIVE_REALISTIC_BTC.md --splits 4 --seeds 11,22,33,44,55 --min-n 50 --maker-fee-bps-grid 0.5,1.0,1.5 --passive-adverse-mult-grid 0.8,1.0,1.2 --out-md reports/PASSIVE_POCKET_RANKING.md --out-json reports/PASSIVE_POCKET_RANKING.json
```

## 10) Failure Modes to Check Before Merge
- Parser produced zero candidates silently.
- Forward-validation used validation leakage from discovery.
- Cost unit mismatch (bps vs ratio) causing 10x errors.
- Debug JSONL schema drift breaking analyzer tools.
- Determinism regression due to hidden non-seeded randomness.

## 11) Agent Checklist
### Pre-change
- [ ] Read `docs/INVARIANTS.md`.
- [ ] Identify exact files and invariant impact.
- [ ] Confirm scope (research-only vs execution-risk).

### Post-change
- [ ] `py_compile` for touched Python files.
- [ ] `pytest -q` (or targeted + explain).
- [ ] CLI smoke command(s) relevant to changed tool.
- [ ] Verify logs/reports contain expected counters/fields.

### Required artifacts
- [ ] Tests updated/added.
- [ ] Docs/report note updated.
- [ ] Repro commands included in delivery.
- [ ] Root cause + verification evidence included.
