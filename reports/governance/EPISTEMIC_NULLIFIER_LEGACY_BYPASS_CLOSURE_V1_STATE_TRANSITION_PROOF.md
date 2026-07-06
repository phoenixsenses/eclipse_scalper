# EPISTEMIC_NULLIFIER_LEGACY_BYPASS_CLOSURE_V1_STATE_TRANSITION_PROOF

**Batch:** BATCH-EPISTEMIC-NULLIFIER-LEGACY-BYPASS-CLOSURE-V1
**Purpose:** Close the 10 legacy canonical.sqlite research modules' inline-SQL bypass and the `research.sqlite` gap identified (but not closed) by the prior batch's transition proof.
**Prior checkpoint:** `EPISTEMIC_NULLIFIER_ENFORCEMENT_WIRING_V1_INCOMPLETE`, commit `51e78673`.
**Nature:** Enforcement closure only. No CVD G2 research executed, no scientific experiment created, no TEST outcome read, no threshold chosen, no CVD data changed, canonical schema_version untouched at 12, no scientific hypothesis reinterpreted, no runtime/risk/execution/shadow/paper/forward/live file touched, no repository-wide cleanup performed, R-16 not resolved.
**Author:** Sonnet 5 · **Date:** 2026-07-06

---

## PHASE 1 — Complete bypass inventory

**Corrected count: NOT 10.** The true production-capable-registration surface, re-audited from scratch (grep for `INSERT INTO experiment_registry/experiment_results`, `ResearchRegistry(`, `.register_experiment(`, `EXPERIMENT_ID`, `def freeze`, across `ami/`, `tools/`, `tests/`):

### A. The 10 originally-reported canonical.sqlite modules (confirmed, unchanged count)

`candidate_universe.py`, `w1_cycle_integrity.py`, `w3_entry_timing_reconciliation.py`, `w4_post_event_path_taxonomy.py`, `w5a_morphology_swing_grammar.py`, `w6_compression_rs_session.py`, `w6rs_confirmation.py`, `w6rs_confound_resolution.py`, `w7a_state_structure_aging_market_clocks.py`, `w10a_multi_tf_structural_conflict.py` (all in `ami/research/`).

| Field | Value (common to all 10) |
|---|---|
| Public entry point | `freeze_and_record(conn, provenance=...)` |
| CLI entry point | `if __name__=="__main__": main()` (each standalone-executable) |
| Experiment-ID creation | hardcoded module-level `EXPERIMENT_ID` constant |
| Freeze/preregistration | none formal — each call recomputes and writes directly |
| Split/TEST-cycle resolution | `candidate_universe`/`w1`/`w3`: **none** ("no train/test split", descriptive-only); `w4`/`w5a`/`w6`/`w6rs_confirmation`/`w6rs_confound_resolution`/`w7a`/`w10a`: chronological 70/30 via shared `_split_chronological` (imported from `w4`), same underlying 252-anchor population |
| Registry/result write path (BEFORE this batch) | own inline `INSERT INTO experiment_registry ... ON CONFLICT(experiment_id) DO UPDATE SET dataset_hash=excluded.dataset_hash, completed_at=..., updated_ms=...` + `DELETE FROM experiment_results WHERE experiment_id=?` + reinsert loop |
| Database | canonical.sqlite |
| Production-capable? | **Yes** — directly executable, real historical experiment_ids already registered in the real DB |
| Historical/replay-only? | Effectively yes in current practice (each re-run recomputes against the live/growing population and refreshes the SAME experiment_id — a "living snapshot", not a new experiment) |
| Test/migration-only? | No |
| Transaction boundary (before) | each call = its own implicit commit sequence (registry INSERT, results DELETE+INSERT loop), no gate participation |
| Bypass mechanism | never called `ami/warehouse/experiment_ledger.py` at all |

### B. The `research.sqlite` gap — wider than previously characterized

`ami/research/registry.py:ResearchRegistry.register_experiment()` (blind `INSERT OR REPLACE`, no gate, no immutability check at all before this batch) has **8 real production-capable callers**, not just "a separate system" as previously noted:

| Caller | Role |
|---|---|
| `ami/latent/discovery.py` (`spec_6a`) | Phase 6A latent discovery (REJECTED verdict, historical) |
| `ami/latent/regime.py` (`spec_6ar`) | Phase 6A-R regime (PASS narrow, historical) |
| `ami/latent/risk_applicability.py` (`spec_6ar2`) | Phase 6A-R2 risk (FALSIFIES/INSUFFICIENT, historical) |
| `ami/run_forward_pipeline.py` (`ensure_binding`) | **recurring/cron** — CLAUDE.md documents this as run every session/cron, registering `E-HOUR17-FWD-001`/`E-CONVCOMP-FWD-001` |
| `ami/run_phase_checks.py` | one-time Phase demo/validation harness |
| `tools/research_ami_mfe50_experiment.py` | MFE50 experiment (historical) |
| `tools/research_s34_buyfade_reentry.py` | BUYFADE_REENTRY (FALSIFIED, historical) |
| `tools/research_s34_buyfade_silence_exit.py` | BUYFADE_SILENCE_EXIT (historical) |
| `tools/research_s34_buyfade_structural.py` | BUYFADE_STRUCTURAL (historical) |

Plus `ami/mutation_suite.py` (`ResearchRegistry(tmp/"r.sqlite")`, 10 call sites) — **TEST_ONLY_PATH**, exercises `ForwardEvidencePipeline`/`EpistemicGovernor` adversarial scenarios, never touches the real DB.

All 8 real callers construct an `ExperimentSpec` with a **fixed** `experiment_id` and re-freeze/re-register it each run — same "living snapshot" pattern as the canonical-side 10, confirmed identical in kind, not just in name.

## PHASE 2 — Path classification

| Path | Classification |
|---|---|
| 10 canonical.sqlite legacy modules | **NORMAL_RESEARCH_PATH** |
| `ResearchRegistry`'s 8 real callers (discovery/regime/risk_applicability/run_forward_pipeline/run_phase_checks/3× tools) | **NORMAL_RESEARCH_PATH** (their own historical instances are effectively **HISTORICAL_REPLAY_PATH**, but the mechanism itself remains a normal, always-callable path) |
| `ami/mutation_suite.py` | **TEST_ONLY_PATH** |
| `ami/warehouse/experiment_ledger.py:record_experiment_registry`/`record_experiment_results` | **NORMAL_RESEARCH_PATH** (unchanged low-level primitives, now internal-only in intent) |
| `ami/warehouse/experiment_ledger.py:register_experiment_with_gates`/`register_legacy_snapshot_with_gates` | the **mandatory boundary** (new) |
| `ami/lifecycle/path_metrics.py`, `path_canonical_migration.py`, `path_candle_repair_correction.py`, `ami/geometry/birth_truncated_cascade_geometry.py`, `short_noisy_v1_migration_rehearsal.py` | **MIGRATION_PATH** / out of scope (write to their own dedicated tables, never `experiment_registry`/`experiment_results`/`research.sqlite`'s `experiments` table) |

Repair/migration bypass rule (test harnesses only): `ami/mutation_suite.py`'s `_register_test_spec()` issues a direct M-0034 gate receipt before calling `ResearchRegistry.register_experiment()` — internal-only, explicit, reachable only from this adversarial test suite (never imported by a normal research CLI), and does not create scientific evidence (synthetic specs, disposable DBs only). Same pattern applied to 7 other existing test files that needed the same accommodation (§5).

## PHASE 3 — Single mandatory registration boundary (canonical.sqlite)

**`ami/warehouse/experiment_ledger.py:register_legacy_snapshot_with_gates()`** (new) is now the sole path all 10 legacy modules call. Design:

1. Always evaluates the graveyard slash-set gate (every call, not just first-ever registration — a family could be graveyarded after initial registration).
2. Compares the proposed `registry_values` against whatever is already stored for this `experiment_id`, but **only on `_LEGACY_SNAPSHOT_STRICT_COLUMNS`** (`question_ids`, `hypothesis_id`, `frozen_splits` — family identity + split methodology). Everything else (`dataset_hash`, computed population counts embedded in `frozen_population`, and prose fields like `frozen_statistical_gate`) is drift-tolerant, matching these modules' own long-standing, already-accepted `ON CONFLICT DO UPDATE SET dataset_hash=...` behavior.
3. **Only an ALREADY-REGISTERED experiment_id whose strict columns are unchanged** takes the lenient path (`_upsert_legacy_snapshot`, reproducing the exact pre-existing SQL pattern verbatim) — no nullifier touched, this is a descriptive refresh, not new evidence.
4. **A brand-new experiment_id, or any strict-column change, always requires the full gate**: if the module declares `no_test_split=True` (candidate_universe/w1/w3), it registers directly (graveyard-gated only, nothing to nullify); otherwise it must supply real `test_cycle_ids` or raises `MissingFrozenTestMetadata` (fail-closed, no data invented).
5. Every successful branch issues an M-0034 gate receipt (`ami.governance.epistemic_gates.issue_gate_receipt`) in the same atomic transaction.

**A real bug was found and fixed during implementation:** the first draft treated `existing is None` (a brand-new experiment_id) as automatically eligible for the lenient path — meaning any first-ever registration silently skipped the graveyard/nullifier gate entirely, exactly the hole this batch exists to close. Caught by a new test (`test_11_family_alias_cannot_bypass_legacy_path`) expecting `TestEvidenceReuseBlocked` on a case/whitespace-alias family reusing a TEST set under a new experiment_id — it did not raise. Root-caused and fixed: only `existing is not None and is_drift_only` now qualifies for the lenient path; `existing is None` always goes through the full gate. Re-verified against all 10 modules' own pre-existing test suites (two of them, `w4`/`w5a`, needed a one-line fixture fix — see §5 — since their "does freeze_and_record write correct SQL from a blank DB" tests exercised a first-ever-registration scenario that, in real production, never happens: all 10 experiment_ids are always already registered among the 22 historical canonical experiments).

Invariants proven (tests in `tests/test_ami_epistemic_nullifier_legacy_bypass_closure.py`, §"REQUIRED TESTS" below):
- No normal module may directly INSERT/UPDATE `experiment_registry`/`experiment_results` (verified: zero `INSERT INTO experiment_registry`/`experiment_results` strings remain in `ami/research/*.py`).
- A valid gate receipt (or the drift-only exemption) is required for every registration; reuse of a receipt for a different experiment_id is meaningless (receipts are keyed by `experiment_id`, not transferable).
- A blocked/failed attempt never produces a registered experiment_id or an orphaned result row (same atomic-transaction guarantee as `register_experiment_with_gates`).
- No normal caller has an `enforce_gates=False`-style optional bypass.

## PHASE 4 — research.sqlite disposition

**Determined role:** `research.sqlite` stores experimental specifications, experiment identities, and (via `forward_bindings`/`processed_trades`) forward-evidence projections — a **downstream projection/cache of already-frozen specs**, never itself a place where new scientific criteria are chosen (its own `ExperimentSpec.freeze()` already enforces immutability of criteria post-freeze, independent of this batch).

**Fix implemented (no schema change to `research.sqlite` itself; no canonical schema_version change):**

`ResearchRegistry.register_experiment()`:
- Historical replay (an existing `experiment_id` whose stored `frozen_hash` **matches** the proposed spec's) is **always allowed unconditionally** — every one of the 8 real callers' recurring/repeated calls hits exactly this branch, so none of them needed any change.
- A **new** `experiment_id`, or a **changed** `frozen_hash` for an existing one, now requires a matching M-0034 gate receipt (`ami.governance.epistemic_gates.has_gate_receipt`, read against the SAME knowledge.sqlite the canonical gate writes to) — raises `ResearchRegistryUnauthorized` fail-closed otherwise, with a precise remediation message (register through `register_experiment_with_gates`/`register_legacy_snapshot_with_gates` first).

Invariants satisfied:
- `research.sqlite` cannot become an alternate authoritative registry — any genuinely new/changed spec requires proof the canonical gate already passed.
- Records reference the canonical registered experiment identity implicitly (the receipt is keyed by the same `experiment_id`).
- Crash-then-retry is naturally idempotent: the receipt is a pure existence check (never consumed/deleted), and `register_experiment()`'s own historical-replay branch handles a repeat call with identical content without requiring a fresh receipt.
- Projection repair never touches `epistemic_test_nullifiers` (verified: `experiment_gate_receipts` and `epistemic_test_nullifiers` are separate tables; a research.sqlite retry only ever reads the former).
- Existing historical `research.sqlite` rows are untouched — no migration to that file was needed or performed.

**8 existing test files** (`test_ami_mutation_suite.py`'s subject module + `test_ami_latent_mutations.py`, `test_ami_regime_mutations.py`, `test_ami_risk_mutations.py`, `test_ami_states_research.py`, `test_ami_research_forward_pipeline_characterization.py`, `test_buyfade_mutations.py`, `test_buyfade_silexit_mutations.py`) needed a matching fixture fix: each constructs a synthetic `ExperimentSpec` in a **fresh, empty** disposable `research.sqlite` (so `existing is None`, requiring a receipt under the new rule) purely to exercise the freeze/`attach_evidence` immutability contract, not the gate. Each now either passes an explicit `knowledge_path=` pointing at its own disposable `KnowledgeStore` and/or issues a direct M-0034 receipt via `ami.governance.epistemic_gates.issue_gate_receipt` before registering — same internal-only pattern as `ami/mutation_suite.py`'s own fix, never touching a normal research CLI.

## REAL INCIDENT DISCOVERED AND CLOSED DURING THIS BATCH

**Accidental unintended writes to the REAL `data/ami/knowledge.sqlite`.** Post-implementation invariant checks found the real file's hash had changed and it had gained 2 new tables (`experiment_gate_receipts` with 11 real rows, `epistemic_authorization_tokens` empty) — entirely unintended.

**Root cause:** `ami/knowledge/store.py:KnowledgeStore.__init__(self, path: str | Path = DEFAULT_PATH)` had (a) no test-isolation guard at all, unlike `ami/warehouse/schema.py:connect()`, and (b) a `path=DEFAULT_PATH` parameter default **bound at function-definition time**, not re-read per call — meaning even if `tests/conftest.py` had tried to redirect `ami.knowledge.store.DEFAULT_PATH` the way it already does for `ami.warehouse.schema.DEFAULT_PATH`, bare `KnowledgeStore()` calls would still have silently resolved to the ORIGINAL real path. Several of the 10 legacy modules' own pre-existing tests open the real canonical.sqlite via the existing, correct `ami.warehouse.schema` isolation copy, but their `freeze_and_record()` calls (now routed through `register_legacy_snapshot_with_gates`, whose `knowledge_db_path=None` default correctly resolves `ami.knowledge.store.DEFAULT_PATH` at call time) had nothing on the knowledge side to redirect to — so the M-0033/M-0034 additive schema landed on the real file.

**Remediation:**
1. Rolled back: `DROP TABLE experiment_gate_receipts; DROP TABLE epistemic_authorization_tokens;` (+ their indexes) against the real file. All pre-existing tables' row counts re-verified unchanged (`knowledge`=11, `edges`=4, `audit_log`=40, `failure_archive`=22, `graveyard_slash_fingerprints`=31, `epistemic_test_nullifiers`=0).
2. Closed the root cause: `ami/knowledge/store.py` now has `REAL_KNOWLEDGE_PATH_IMMUTABLE`/`_TEST_ISOLATION_ACTIVE` (mirroring `ami/warehouse/schema.py`) and `KnowledgeStore.__init__` resolves its path default at call time and fail-closed-rejects any writable connection to the real path while isolation is active.
3. `tests/conftest.py` gained `_isolate_real_knowledge_db` (session-scoped, autouse), mirroring `_isolate_real_canonical_db` exactly: copies the real file once, redirects `ami.knowledge.store.DEFAULT_PATH`, asserts hash/mtime unchanged at session teardown.
4. Verified: two full regression passes (run3final, run4final, below) after this fix show the real `knowledge.sqlite` hash stable at `6c8905f7f014b7c1b18ea5e32462cdf69cfdf363f0db85a9d75da400d00a4c9a` across both runs, with unchanged row counts in every table and no reappearance of the two dropped tables.

**Note on byte-hash vs. logical content for `knowledge.sqlite`:** this file is opened in `journal_mode=WAL`. Investigated directly: a single read-only `sqlite3.connect(...mode=ro...)` did not by itself change the hash in a controlled test, but the file's hash DID shift once between the manual rollback (step 1) and the first post-fix regression run — traced to delayed WAL-checkpoint activity finishing the already-logically-committed `DROP TABLE` from step 1, not a new write. After that one settling shift, the hash has been byte-stable across two full regression passes. Logical row-content equality (not raw byte-hash) is therefore the authoritative invariant for this file going forward; `tests/conftest.py`'s new fixture asserts both, and passed both times post-settling.

## Exact changed-file manifest

| File | Status | What changed |
|---|---|---|
| `ami/warehouse/experiment_ledger.py` | Modified | New `register_legacy_snapshot_with_gates`, `_upsert_legacy_snapshot`, `MissingFrozenTestMetadata`, `_LEGACY_SNAPSHOT_STRICT_COLUMNS`; fixed the existing-is-None routing bug |
| `ami/governance/epistemic_gates.py` | Modified | New M-0034 `_RECEIPT_SCHEMA` (`experiment_gate_receipts`), `issue_gate_receipt`, `has_gate_receipt` |
| `ami/research/registry.py` | Modified (first git commit) | `ResearchRegistry.__init__` gained `knowledge_path=`; `register_experiment` gained the historical-replay-vs-gate-receipt fail-closed check; new `ResearchRegistryUnauthorized` |
| `ami/mutation_suite.py` | Modified (first git commit) | `_env()` passes explicit `knowledge_path`; new `_register_test_spec()` helper (direct receipt issuance); all 10 call sites updated |
| `ami/knowledge/store.py` | Modified (first git commit) | Test-isolation closure (see incident above): call-time path resolution, `REAL_KNOWLEDGE_PATH_IMMUTABLE`/`_TEST_ISOLATION_ACTIVE` guard |
| `ami/research/candidate_universe.py`, `w1_cycle_integrity.py`, `w3_entry_timing_reconciliation.py`, `w4_post_event_path_taxonomy.py`, `w5a_morphology_swing_grammar.py`, `w6_compression_rs_session.py`, `w6rs_confirmation.py`, `w6rs_confound_resolution.py`, `w7a_state_structure_aging_market_clocks.py`, `w10a_multi_tf_structural_conflict.py` | Modified (first git commit) | Each `freeze_and_record()` routed through `register_legacy_snapshot_with_gates` instead of inline SQL; scientific inputs/outputs/experiment IDs unchanged |
| `tests/conftest.py` | Modified (first git commit) | New `_isolate_real_knowledge_db` fixture (incident closure above) |
| `tests/test_ami_epistemic_nullifier_enforcement_wiring.py` | Modified | Canary test #22 updated: expected legacy-bypass set is now empty |
| `tests/test_ami_epistemic_nullifier_legacy_bypass_closure.py` | New | 26 tests, one per required scenario |
| `tests/test_ami_latent_mutations.py`, `test_ami_regime_mutations.py`, `test_ami_risk_mutations.py`, `test_ami_states_research.py`, `test_ami_research_forward_pipeline_characterization.py`, `test_buyfade_mutations.py`, `test_buyfade_silexit_mutations.py` | Modified (first git commit) | Explicit `knowledge_path=`/direct gate-receipt issuance so their synthetic freeze/attach_evidence tests keep passing under the new `ResearchRegistry` rule |
| `tests/test_ami_research_w4_post_event_path_taxonomy.py`, `test_ami_research_w5a_morphology_swing_grammar.py` | Modified (first git commit) | Added `_preseed_existing_registration()` so their from-scratch-DB tests reflect real production topology (experiment_id always already registered) instead of first-ever registration |
| `reports/governance/EPISTEMIC_NULLIFIER_LEGACY_BYPASS_CLOSURE_V1_STATE_TRANSITION_PROOF.md` | New | this document |

**Git history note:** as with the prior batch, most of these files (`ami/research/registry.py`, `ami/mutation_suite.py`, `ami/knowledge/store.py`, the 10 legacy modules, `tests/conftest.py`, and 9 of the 10 fixed test files) had never been committed before — this batch's commit is their first-ever git snapshot, containing both their pre-existing content and this batch's edits together (no prior commit to diff against; not a hunk-separation problem).

## Test results

30 → wait: **26 new tests** (`tests/test_ami_epistemic_nullifier_legacy_bypass_closure.py`), one per required scenario 1-26, all passing. All previously-passing test files re-verified green after every fix (candidate_universe, w1, w3, w4, w5a, w6, w6rs_confirmation, w6rs_confound_resolution, w7a, w10a, mutation_suite, latent_mutations, regime_mutations, risk_mutations, states_research, forward_pipeline_characterization, buyfade_mutations, buyfade_silexit_mutations, epistemic gates V1, epistemic-nullifier-enforcement-wiring V1).

## Regression

- **Collect-only** (frozen AMI file set): **923** (897 prior + 26 new).
- **Run1/Run2** (before the knowledge.sqlite isolation fix): 36/36 pairs green except `test_ami_chart_candle_builder.py::test_real_seed_against_default_trades_db_is_sane` — **922/923** both times. Root-caused (read-only, no code changed): the real, live `data/microstructure.db`'s ETHUSDT and BTCUSDT agg-trade streams had stopped advancing (~2-2.5h stale at time of investigation, confirmed via direct `MAX(ts_ms)` query and `tasklist`), while SOLUSDT continued streaming fine (1.6s old) — a live-collector data-freshness issue, unrelated to any file this batch touched (`ami/chart/candle_builder.py`, collector code, and `data/microstructure.db` were never opened for writing by this batch). Per instruction, runtime/collector was not touched or restarted.
- **run3final/run4final** (after the knowledge.sqlite isolation fix): same **922/923** both times, same single pre-existing/environmental failure, confirmed via isolated reruns as deterministic (not a race) and unrelated to code. `data/ami/knowledge.sqlite` hash held stable (`6c8905f7f014b7c1b18ea5e32462cdf69cfdf363f0db85a9d75da400d00a4c9a`) across both of these runs; `data/ami/canonical.sqlite` hash held stable (`458bc07c…`) across all four runs.
- **This batch's own dedicated closure**: every test that exercises the new gate/receipt/registry mechanism (55 tests total across the two epistemic-nullifier test files) is 100% green across all four regression passes.

## Canonical hash/version proof, experiment/result immutability, protected delta

| Check | Result |
|---|---|
| canonical.sqlite sha256 (after) | `458bc07ca5b436041e59c781a26cf502779d5dc2751a3be8a0c1cddb93e84d49` — unchanged from before this batch |
| canonical.sqlite schema_version | 12 — unchanged |
| `experiment_registry` count | 22 — unchanged |
| `integrity_check` | ok |
| Protected counts (events/signal_lifecycle/cycles/geometry) | 252/324/167/220 — unchanged |
| CVD frozen counts | repaired_trades=40934, batch_ledger=8, exact=1840, proxy=1840, exclusions=104, quality=1840, EXACT_RECONSTRUCTABLE=1828, SOURCE_GAPPED=12 — all unchanged |
| Real `knowledge.sqlite` row counts | knowledge=11, edges=4, audit_log=40, failure_archive=22, graveyard_slash_fingerprints=31, epistemic_test_nullifiers=0 — unchanged from pre-batch state (post-incident-rollback) |
| Real `knowledge.sqlite` — new tables from the incident | dropped, confirmed absent |
| Real `research.sqlite` | untouched (sha256 `926c787a965b78cbbeba072dc36b3595c81c77faf9dca57793f7f69f8c770a71`, never opened for writing by this batch — only the `ResearchRegistry` class definition changed, no code path in this batch called it against the real path) |
| New experiments created | 0 |
| New experiment_results rows | 0 |
| Runtime/risk/execution/shadow/paper/forward files touched | 0 |

## Remaining semantic family-identity bypass risk (carried over, unresolved by design)

Unchanged from the prior batch: `resolve_canonical_family_id`/`resolve_split_version` hash free-form text (`question_ids`, `hypothesis_id`, `frozen_splits`). A semantically-identical hypothesis expressed under different wording still resolves to a different family/nullifier. Closing this fully requires a real canonical hypothesis-family ontology — explicitly out of scope for an "enforcement closure" batch (would be a scientific/ontological redesign, not wiring).

## New state root

| Field | Value |
|---|---|
| canonical.sqlite hash / schema_version | `458bc07c…` / 12 (unchanged) |
| Legacy canonical.sqlite bypass | **CLOSED** — all 10 modules route through `register_legacy_snapshot_with_gates` |
| research.sqlite gap | **CLOSED** — `ResearchRegistry.register_experiment` requires a gate receipt for any new/changed registration |
| `GATES_SCHEMA_VERSION` | 2 (unchanged from M-0033; M-0034 added a table, not a version bump, matching M-0033's own convention for `_AUTH_SCHEMA`) |
| Real knowledge.sqlite | incident rolled back, isolation gap closed, hash stable across 2 post-fix regression passes |

---

## Verdict

**`EPISTEMIC_NULLIFIER_LEGACY_BYPASS_CLOSURE_V1_COMPLETE`**

All 10 originally-reported legacy canonical.sqlite modules, plus the 8 real `research.sqlite` callers (a wider set than originally scoped, discovered during Phase 1's non-assumed re-audit), now route through mandatory, fail-closed enforcement. Zero production-capable experiment-registration path found in this audit remains structurally unguarded. `ami/mutation_suite.py`'s internal test-harness receipt-issuance is the only bypass-shaped code, and it satisfies every internal-bypass-policy requirement (internal-only, explicit, caller-identified by file, creates no scientific evidence, covered by its own 20-scenario test suite). Two full regression passes obtained after the mid-batch knowledge.sqlite isolation incident was found and closed, both 922/923 with the single remaining failure conclusively isolated to an unrelated, pre-existing live-collector data-staleness condition this batch is barred from touching. Canonical state, CVD frozen counts, and protected-subsystem counts are byte-for-byte/row-for-row unchanged throughout.
