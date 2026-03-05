# Parallel Layers Roadmap (Sweep-Independent)

This roadmap is designed to run in parallel with long-running ranking sweeps. It avoids coupling to `reports/_runs/*` and focuses on durable improvements that are valuable regardless of final attribution mix.

## Phase 0: Baseline Safety and Repeatability
### Objective
Lock deterministic execution for research tools and protect active sweep outputs.
### Why It Matters
Parallel work is only useful if outputs are reproducible and non-destructive. This phase reduces accidental overwrite risk and prevents silent drift while teams iterate on separate layers.
### Tasks
- [ ] Ensure all new tools write to explicit output paths, never implicit cwd scratch files.
- [ ] Add run metadata to all new reports (`timestamp_utc`, `args`, `git_commit` where available).
- [ ] Standardize fast smoke checks in `tools/smoke_all.py`.
- [ ] Ensure env-backed defaults are explicit in CLI help (`config/costs.py`, `MAKER_FEE_BPS`).
### Acceptance Criteria
- `python -m tools.smoke_all` returns `0` with or without DB.
- No tool modifies `reports/_runs/*` unless directly invoked with those paths.
- New tool outputs are deterministic for same args.
### Tests To Add
- `tests/test_smoke_all.py`: skip behavior without DB and deterministic success path.
- `tests/test_run_rank_sweep.py`: deterministic run-id and registry schema.
### CLI Validation
```powershell
python -m tools.smoke_all --db data/microstructure.db
python -m tools.run_rank_sweep --help
```
### Expected Artifacts
- `reports/RUN_RANK_SWEEP_REGISTRY.jsonl`
- `reports/_runs/<run_id>/rank.json`
- `reports/_runs/<run_id>/rank.md`

---

## Phase 1: Data Layer and Canonical Integrity
### Objective
Ship a deterministic canonical integrity gate that fails fast on schema/time/sanity violations and writes stable artifacts.
### Why It Matters
All downstream research and ranking conclusions assume canonical data is trustworthy. Silent schema drift, timestamp disorder, duplicates, or impossible values can produce fake edge and invalidate forward validation.
### Tasks
- [ ] Implement `tools/validate_canonical.py` with deterministic run-id artifacts.
- [ ] Add schema checks (required timestamp/symbol/price columns).
- [ ] Add dtype checks (numeric convertibility + timestamp convertibility).
- [ ] Add time invariants (monotonicity, duplicate timestamps per symbol, backward jumps).
- [ ] Add NaN/Inf thresholds for critical columns.
- [ ] Add basic sanity constraints (negative spread/volume, non-positive price).
- [ ] Implement graceful skip semantics for missing source/missing optional DB flag.
- [ ] Emit deterministic outputs:
  - `reports/validate_canonical_<run_id>.json`
  - `reports/validate_canonical_<run_id>.md`
### Acceptance Criteria
- Tool supports `--help` and deterministic behavior.
- Exit code `0` on pass/skip, exit code `3` on validation failure.
- Missing source path produces `status=skip` with `skipped_missing_data`.
- JSON artifact contains: `status`, `violations`, `column_stats`, `invariant_summary`.
- No timestamp-based filenames.
### Tests To Add
- `tests/test_validate_canonical.py`:
  - `test_pass_clean_synthetic`
  - `test_fail_duplicate_timestamp`
  - `test_fail_missing_required_column`
  - `test_fail_nan_threshold`
  - `test_skip_missing_source`
### CLI Validation
```powershell
python -m tools.validate_canonical --help
python -m tools.validate_canonical --in data/canonical/canonical_merged.parquet --reports-dir reports
python -m tools.validate_canonical --in data/canonical/missing.parquet --reports-dir reports
```
### Expected Artifacts
- `reports/validate_canonical_<run_id>.json`
- `reports/validate_canonical_<run_id>.md`

---

## Phase 2: Signal Layer (Calibration + Guards)
### Objective
Improve signal reliability via score calibration, regime hooks, and anti-leakage checks.
### Why It Matters
Raw directional hit-rate is insufficient; signals must remain valid under execution constraints and no-lookahead assumptions. This phase keeps model confidence honest and deterministic.
### Tasks
- [ ] Add score calibration helper for `tools/micro_edge_signal_v2.py` / v3 hooks.
- [ ] Add regime-conditioning hooks in `tools/micro_edge_lib.py` (past-only).
- [ ] Add feature sanitation pipeline (clip/winsorize/NaN policy) with explicit logging.
- [ ] Add no-lookahead unit checks for rule firing and labels.
### Acceptance Criteria
- Signal outputs deterministic for same input + seed.
- Rule generation never reads future rows.
- Calibration outputs include coverage and confidence bins.
### Tests To Add
- `tests/test_micro_edge_signal_v2.py`: deterministic outputs and thresholds.
- `tests/test_micro_edge_alignment.py`: label/entry timing invariants.
- `tests/test_signal_no_lookahead.py` (new): signal uses past-only fields.
### CLI Validation
```powershell
python -m tools.micro_edge_smoke --db data/microstructure.db --symbols BTCUSDT,ETHUSDT --lookback-min 240 --bucket-sec 1 --horizon-sec 30 --min-rule-n 100
python -m tools.micro_edge_report --in logs/micro_edge_smoke.jsonl --symbol ETHUSDT --last 200
```
### Expected Artifacts
- `logs/micro_edge_smoke.jsonl`
- `reports/MICRO_EDGE_SIGNAL_V2.md` (update/addendum)

---

## Phase 3: Execution Simulation Layer (Passive Model Diagnostics)
### Objective
Make passive execution diagnostics first-class and testable.
### Why It Matters
Edge viability is mostly decided by fill quality and adverse selection. Diagnostics must explain failures in terms of queue competition, fill probabilities, and cost decomposition.
### Tasks
- [ ] Add queue competition toggle profiles in `execution/passive_execution_simulator.py`.
- [ ] Add fill-probability diagnostics export fields per attempt/trade.
- [ ] Add adverse distribution diagnostics (quantiles, tails by regime) in backtest outputs.
- [ ] Add deterministic calibration replay checks for passive model parameters.
### Acceptance Criteria
- `passive_realistic` outputs stable diagnostics with same seed.
- Fill/adverse metrics are present in rank/forward outputs.
- Negative maker fee (rebate) flows through cost math without clamp bugs.
### Tests To Add
- `tests/test_passive_realistic_sim.py`: deterministic fill behavior.
- `tests/test_passive_adverse_mult.py`: adverse multiplier monotonic effect.
- `tests/test_costs_config.py`: env default + rebate propagation.
### CLI Validation
```powershell
python -m tools.micro_edge_backtest --db data/microstructure.db --symbols ETHUSDT --lookback-min 720 --bucket-sec 1 --horizon-sec 60 --rule micro_edge_v3_passive_alpha --exec-model passive_realistic --maker-fee-bps 1.0
python -m tools.validate_passive_pocket_forward --db data/microstructure.db --symbol ETHUSDT --lookback-min 1440 --bucket-sec 1 --horizon-sec 120 --rule micro_edge_v3_passive_alpha --min-imbalance 0.4 --min-trade-intensity 2500 --max-spread 0.0003 --splits 3 --seeds 7,11,22 --min-n 20 --min-n-frac 0.0001 --maker-fee-bps 1.0 --passive-adverse-mult 1.2
```
### Expected Artifacts
- `logs/micro_edge_debug_trades.jsonl`
- `reports/PASSIVE_POCKET_FORWARD_VALIDATION*.md`

---

## Phase 4: Cost Model Layer (Fees/Rebates/Slippage)
### Objective
Standardize cost model configuration and reporting across tools.
### Why It Matters
Cost assumptions dominate net outcomes. Inconsistent defaults or hidden clamps create contradictory conclusions and invalidate comparisons.
### Tasks
- [ ] Use `config/costs.py` across all research CLIs.
- [ ] Add slippage model placeholders in backtest args and summary schema.
- [ ] Extend attribution summary to include fee/adverse/raw/net bps everywhere.
- [ ] Add env-default audit command to print effective cost defaults.
### Acceptance Criteria
- `MAKER_FEE_BPS` is honored consistently.
- Negative fee rebate is preserved end-to-end.
- Cost breakdown fields are included in rank rows and summaries.
### Tests To Add
- `tests/test_costs_config.py`: parse + default + negative fee.
- `tests/test_rank_passive_pockets_forward.py`: attribution fields present and non-null.
- `tests/test_summarize_rank_attribution.py`: robust console summary.
### CLI Validation
```powershell
$env:MAKER_FEE_BPS = "-0.25"
python -m tools.rank_passive_pockets_forward --help
python -m tools.summarize_rank_attribution --in reports/PASSIVE_POCKET_RANKING.json --top-n 20
```
### Expected Artifacts
- `reports/PASSIVE_POCKET_RANKING*.json`
- `reports/PASSIVE_POCKET_RANKING*.md`

---

## Phase 5: Risk Layer (Guardian/Kill-Switch Integration Interfaces)
### Objective
Define and test integration points for drawdown guard, belief-debt, and panic mode semantics.
### Why It Matters
Even perfect research signals require operational risk containment. This phase defines interfaces now, so eventual runtime integration is low risk and test-first.
### Tasks
- [ ] Add interface spec doc + stub contracts for:
  - drawdown guard trigger input/output
  - belief-debt accumulation and reset policy
  - panic mode state transitions and cooldown semantics
- [ ] Add test harness under `tools/` for dry-run risk decisions (no live coupling).
- [ ] Add JSON event schema placeholders for risk state snapshots.
### Acceptance Criteria
- Stub interfaces compile and are unit-tested.
- Risk event schema is explicit and versioned.
- No change to live execution behavior (research-only path).
### Tests To Add
- `tools/test_adaptive_guard_unit.py`
- `tools/test_belief_controller_unit.py`
- `tools/test_belief_evidence_unit.py`
- `tests/test_risk_interface_contracts.py` (new): schema fields and transition rules.
### CLI Validation
```powershell
python -m tools.risk_checklist
pytest -q tools/test_adaptive_guard_unit.py tools/test_belief_controller_unit.py tools/test_belief_evidence_unit.py
```
### Expected Artifacts
- `reports/RISK_INTERFACE_CONTRACTS.md`
- `logs/risk_state_snapshots.jsonl`

---

## Phase 6: Reporting and Observability Layer
### Objective
Standardize experiment run summaries and build deterministic multi-run rollups.
### Why It Matters
Parallel experiments are only comparable with a stable schema and reproducible run metadata. This phase enables reliable portfolio-level diagnostics over many runs.
### Tasks
- [ ] Define `run_summary` JSON schema (`version`, `inputs`, `metrics`, `artifacts`).
- [ ] Build run registry readers/aggregators for `_runs` outputs.
- [ ] Add schema validation utility for report JSON files.
- [ ] Add concise markdown reporter for top-level experiment outcomes.
### Acceptance Criteria
- Rollup tool can aggregate >100 runs deterministically.
- Schema validator catches missing critical fields.
- Output order stable with same input files.
### Tests To Add
- `tests/test_run_rank_sweep.py`
- `tests/test_summarize_rank_attribution.py`
- `tests/test_report_schema_validator.py` (new)
### CLI Validation
```powershell
python -m tools.run_rank_sweep --dry-run --candidates-md reports/FILTER_SWEEP_V3_21D_ETH_h120_ADV1p2.md
python -m tools.summarize_rank_attribution --in reports/PASSIVE_POCKET_RANKING.json --top-n 10
```
### Expected Artifacts
- `reports/RUN_RANK_SWEEP_REGISTRY.jsonl`
- `reports/EXPERIMENT_RUN_ROLLUP.json`
- `reports/EXPERIMENT_RUN_ROLLUP.md`

---

## Phase 7: CI and Developer Experience
### Objective
Provide a single fast entrypoint for local confidence and stable CI behavior.
### Why It Matters
Frequent research iteration needs fast feedback loops. A stable smoke target + naming conventions reduce regressions and review overhead.
### Tasks
- [ ] Maintain `tools/smoke_all.py` as the fast deterministic smoke target.
- [ ] Enforce CLI consistency (`--db`, `--out-*`, `--help` clarity, env defaults).
- [ ] Add lightweight py_compile + pytest subset CI stage.
- [ ] Add docs links from `docs/AGENTS.md` and `docs/RUNBOOK_PARALLEL_WORK.md`.
### Acceptance Criteria
- `tools/smoke_all.py` passes in <30s without DB.
- All new tools have deterministic defaults and clear errors.
- CI smoke stage catches import/compile/schema failures.
### Tests To Add
- `tests/test_smoke_all.py`
- `tests/test_cli_help_contracts.py` (new)
### CLI Validation
```powershell
python -m tools.smoke_all --db data/microstructure.db
pytest -q tests/test_smoke_all.py
```
### Expected Artifacts
- `reports/CI_SMOKE_SUMMARY.md`

---

## Parallel Work Matrix
| Task | Owner | Dependency | Notes |
|---|---|---|---|
| Data readiness + alignment checks | friend | none | Safe while sweeps run |
| Signal no-lookahead/calibration tests | Codex | none | No sweep-output dependency |
| Passive simulator diagnostics fields | Codex | none | Deterministic seed tests required |
| Cost default/env plumbing audit | me | none | Validate `MAKER_FEE_BPS` behavior |
| Risk interface contracts/stubs | friend | none | Do not wire live paths yet |
| Run summary schema + validator | Codex | none | Works on existing reports |
| Attribution post-processing heuristics | me | requires sweep results | Tune using completed run outputs |
| Pocket promotion decisions | me | requires sweep results | Final gating thresholds only after sweep |
| Canonical dataset rebuild | friend | requires data rebuild | Separate from running sweeps |
