# EPISTEMIC_NULLIFIER_ENFORCEMENT_WIRING_V1_STATE_TRANSITION_PROOF

**Batch:** BATCH-EPISTEMIC-NULLIFIER-ENFORCEMENT-WIRING-V1 (migration `M-0033`)
**Purpose:** Wire the previously-mechanism-only `ami/governance/epistemic_gates.py` (graveyard slash-set gate + TEST-evidence nullifier, BATCH-EPISTEMIC-NULLIFIER-GATES-V1) into the actual experiment registration path, so both disciplines become mandatory rather than advisory.
**Nature:** Wiring only. No CVD research wave executed, no CVD experiment created, no TEST outcome read, no threshold selected, no CVD data altered, `canonical.sqlite` schema_version untouched at 12, no runtime/risk/execution/shadow/paper/forward file touched, G1 not revisited, no repository-wide cleanup performed.
**Author:** Sonnet 5 · **Date:** 2026-07-06

---

## 1. Previous state root / checkpoint

| Field | Value |
|---|---|
| Prior gate closure | G1 — `CVD_CANONICAL_MIGRATION_COMPLETE`, proof commit `ae6c43cf` |
| canonical.sqlite hash (before this batch) | `458bc07ca5b436041e59c781a26cf502779d5dc2751a3be8a0c1cddb93e84d49` |
| canonical.sqlite schema_version (before) | 12 |
| experiment_registry count (before) | 22 |
| knowledge.sqlite state (before) | `graveyard_slash_fingerprints`=31 rows, `epistemic_test_nullifiers`=0 rows (from M-0032, mechanism-ready/not-wired); no `epistemic_authorization_tokens` table yet |
| `epistemic_gates.py` mechanism version | `GATES_SCHEMA_VERSION=1` (seed batch), never wired into any registration path |

## 2. Batch ID / contract-mechanism version

| Field | Value |
|---|---|
| Batch ID | `BATCH-EPISTEMIC-NULLIFIER-ENFORCEMENT-WIRING-V1` |
| Migration ID | `M-0033` — additive, knowledge.sqlite only (never touches canonical.sqlite) |
| New `GATES_SCHEMA_VERSION` | 2 (was 1) |
| New identity-adapter versions | `FAMv1` (family id), `SPLITv1` (split version) |

## 3. REQUIRED INITIAL AUDIT — call graph (summary; full detail below)

**Every path that writes `experiment_registry`/`experiment_results` in this repo, found by exhaustive grep, verified file-by-file:**

| # | File | Mechanism | Gated by this batch? |
|---|---|---|---|
| 1 | `ami/warehouse/experiment_ledger.py` (`record_experiment_registry`/`record_experiment_results`) | centralized, immutability-guarded (NOOP_IDENTICAL / `ImmutableExperimentConflict`) | Indirectly — still callable directly (unchanged, for the 10 legacy modules below), but now also wrapped by `register_experiment_with_gates()` |
| 2 | `ami/warehouse/experiment_ledger.py` (`register_experiment_with_gates`, **NEW**) | the new mandatory gated entry point | **YES — this is the gate** |
| 3-12 | `ami/research/w8_hold_baseline.py`, `w8_hold_baseline_004_long_corrected_cycle_grouped.py`, `w8_long_nested_path_accumulation.py`, `w8_long_nested_path_accumulation_002_candle_repair.py`, `w8_long_timing_structure.py`, `w8_long_timing_structure_002_candle_repair_cycle_grouped.py`, `w8_short_expanded_baseline.py`, `w8_short_expanded_baseline_003_candle_repair.py`, `w8_vol_normalized_baseline.py`, `w8_vol_normalized_baseline_004_long_corrected_cycle_grouped.py` | call `record_experiment_registry`/`record_experiment_results` directly (bullet 1), NOT the new gated wrapper | **NO** — historical, completed, immutable experiments; not touched this batch (would be repository-wide cleanup) |
| 13-22 | `ami/research/candidate_universe.py`, `w1_cycle_integrity.py`, `w3_entry_timing_reconciliation.py`, `w4_post_event_path_taxonomy.py`, `w5a_morphology_swing_grammar.py`, `w6_compression_rs_session.py`, `w6rs_confirmation.py`, `w6rs_confound_resolution.py`, `w7a_state_structure_aging_market_clocks.py`, `w10a_multi_tf_structural_conflict.py` | **own inline** `INSERT INTO experiment_registry ... ON CONFLICT(experiment_id) DO UPDATE`, hardcoded `EXPERIMENT_ID`, `if __name__=="__main__"` CLI-executable | **NO — KNOWN, UNCLOSED BYPASS** (see §9) |
| 23 | `ami/research/registry.py` (`ResearchRegistry.register_experiment`) | separate system, separate DB (`data/ami/research.sqlite`, its OWN `experiments` table, blind `INSERT OR REPLACE`, no immutability guard at all) | **NO — separate, older Phase-4 concept, out of scope for this batch** (different database entirely) |

Also audited and confirmed OUT OF SCOPE (freeze_and_record()-named functions that do NOT touch `experiment_registry`/`experiment_results` at all): `ami/lifecycle/path_metrics.py`, `ami/lifecycle/path_canonical_migration.py`, `ami/lifecycle/path_candle_repair_correction.py`, `ami/lifecycle/short_noisy_v1_migration_rehearsal.py`, `ami/geometry/birth_truncated_cascade_geometry.py`, `ami/research/forward_pipeline.py` — each writes to its own dedicated table (candle/geometry/path-observation backfills), not the experiment ledger.

**Transaction boundaries (before this batch):** every w8_*.py `freeze_and_record()` calls `record_experiment_registry()` then `record_experiment_results()` then a single `conn.commit()` at the end — already one transaction, but with no gate participation at all. The 10 legacy modules commit implicitly per-statement via their own inline SQL (no batching discipline).

**Failure/rollback behavior (before this batch):** `record_experiment_registry`/`record_experiment_results` raise `ImmutableExperimentConflict` on content mismatch but neither commits nor rolls back internally — the caller's later `conn.commit()` (or its absence, on an uncaught exception) determines the outcome. No gate-related failure mode existed because no gate was wired in.

## 4. Design: enforcement order, family/split identity, atomicity

**Enforcement order implemented exactly as specified (`register_experiment_with_gates`, `ami/warehouse/experiment_ledger.py`):**
1. resolve canonical family identity (`resolve_canonical_family_id`)
2. evaluate graveyard slash-set gate (`match_graveyard`)
3. validate retry authorization if blocked (`_reserve_authorization`, type=RETRY)
4. resolve frozen split version (`resolve_split_version`)
5. resolve the ordered TEST independent-cycle set (+ TRAIN/TEST leakage check via `ami.research.registry.assert_no_overlap`, reused rather than reimplemented, when `train_cycle_ids` is supplied) + duplicate-id rejection (`_normalized_test_set_hash`)
6. compute the TEST-evidence nullifier (`derive_test_nullifier`)
7. check prior nullifier consumption
8. validate supersession authorization if required (`_reserve_authorization`, type=SUPERSESSION)
9. persist gate decision + audit record (`EXPERIMENT_GATE_DECISION` in `audit_log`)
10. only then permit registration (`record_experiment_registry`/`record_experiment_results`)
11. only after successful registration, mark the nullifier (`consume_test_evidence`) and any authorization tokens (`_finalize_authorization_consumption`) consumed

**Cross-database atomicity:** `canonical.sqlite` (experiment_registry/experiment_results) and `knowledge.sqlite` (graveyard/nullifier/authorization/audit tables) are two separate SQLite files. `register_experiment_with_gates` uses `ATTACH DATABASE <knowledge_path> AS kb` on the caller's canonical connection so all of steps 2-11 run inside **one SQLite transaction spanning both files**, closed by **one final `commit()`** (or `rollback()` on any exception, in a `try/except/finally` that also `DETACH`es `kb`). Schema creation (`init_gates_schema`) is deliberately run on a **separate, direct** connection to knowledge.sqlite beforehand — unqualified `CREATE TABLE` always targets the connection's `main` schema, so running it through the attached connection would have (incorrectly) tried to create gate tables inside canonical.sqlite. Verified no table-name collisions exist between the two databases (checked both full table lists before designing this), so all unqualified DML in the existing gate functions correctly auto-resolves to `kb`.

**Family/split identity adapter (minimum-compatible, no ontology redesign, `experiment_registry` schema untouched):**
- `resolve_canonical_family_id(question_ids, hypothesis_id)` → `FAMv1:<sha256[:16]>`, each field independently normalized (whitespace-collapsed, lowercased) **before** concatenation (an earlier draft normalized the already-joined string and failed its own alias test — whitespace touching the join delimiter didn't collapse the same way; fixed and covered by test #12).
- `resolve_split_version(frozen_splits)` → `SPLITv1:<sha256[:16]>` of the normalized description string.
- Both are deterministic, versioned, immutable once frozen (inputs are `experiment_registry`'s own immutable content columns), auditable, and leave the curated-fingerprint graveyard matcher (which still runs on raw spec_text) untouched.

## 5. Exact changed-file manifest

| File | Status | What changed |
|---|---|---|
| `ami/governance/epistemic_gates.py` | Modified (was untracked from M-0032; **never previously committed to git**) | `GATES_SCHEMA_VERSION` 1→2; new additive `_AUTH_SCHEMA` (see §6); new `AuthorizationInvalid` exception; `_autocommit` kwarg added to `assert_not_graveyard`/`consume_test_evidence` (default `True`, preserves all 16 existing V1 tests byte-for-byte); `_normalized_test_set_hash` (duplicate-cycle-id rejection, used by `derive_test_nullifier`/`consume_test_evidence`); `resolve_canonical_family_id`/`resolve_split_version`; `issue_retry_authorization`/`issue_supersession_authorization`/`_reserve_authorization`/`_finalize_authorization_consumption` |
| `ami/warehouse/experiment_ledger.py` | Modified (was untracked from the effective-path-immutability-hardening batch; **never previously committed to git**) | New `register_experiment_with_gates()` + `_audit_kb()` helper; `record_experiment_registry`/`record_experiment_results` themselves **unchanged** (still directly callable — preserves the 10 w8_*.py modules' behavior exactly) |
| `tests/test_ami_epistemic_nullifier_enforcement_wiring.py` | New | 29 test functions covering the 27 required scenarios (two scenarios split into 2 tests each for clarity: #9 unit + integration, #20 DB-level + app-level) |
| `reports/governance/EPISTEMIC_NULLIFIER_ENFORCEMENT_WIRING_V1_STATE_TRANSITION_PROOF.md` | New | this document |

**Note on git history:** neither `ami/governance/epistemic_gates.py` nor `ami/warehouse/experiment_ledger.py` had ever been committed before this batch (both were created in earlier, still-uncommitted batches — consistent with the pre-existing version-control gap, `RISK_REGISTER.md` R-16, already flagged in the G1 closure). This batch's commit will therefore be the **first** git-tracked snapshot of both files, containing the ORIGINAL M-0032/hardening-batch content plus this batch's WIRING-V1 additions together — there is no prior commit to diff against, so (unlike the four shared governance Markdown files) this is not a hunk-separation problem, just a first-ever commit. Their sibling `__init__.py`/other package files remain untracked (out of scope — not touched by this batch).

## 6. Schema/migration changes (M-0033)

Additive only, knowledge.sqlite, never touches canonical.sqlite (schema_version stays 12):

```sql
CREATE TABLE IF NOT EXISTS epistemic_authorization_tokens (
    authorization_id TEXT PRIMARY KEY,
    authorization_type TEXT NOT NULL CHECK (authorization_type IN ('RETRY', 'SUPERSESSION')),
    canonical_family_id TEXT NOT NULL,
    related_experiment_id TEXT,
    related_nullifier TEXT,
    split_version TEXT,
    test_set_hash TEXT,
    approver TEXT NOT NULL,
    justification TEXT NOT NULL,
    retry_condition_satisfied TEXT,
    input_manifest_root TEXT,
    issued_ms INTEGER NOT NULL,
    expiry_ms INTEGER,
    single_use INTEGER NOT NULL DEFAULT 1,
    token_commitment TEXT NOT NULL UNIQUE,
    consumed INTEGER NOT NULL DEFAULT 0,
    resulting_experiment_id TEXT,
    consumed_ms INTEGER
);
CREATE INDEX IF NOT EXISTS idx_auth_tokens_family
    ON epistemic_authorization_tokens(canonical_family_id, authorization_type);
CREATE UNIQUE INDEX IF NOT EXISTS uq_test_nullifiers_first_consumption
    ON epistemic_test_nullifiers(nullifier) WHERE supersession_token IS NULL;
```

- **Migration ID:** `M-0033`.
- **Rollback:** `DROP TABLE epistemic_authorization_tokens; DROP INDEX idx_auth_tokens_family; DROP INDEX uq_test_nullifiers_first_consumption;` — no existing row in any table is touched (pure addition, `CREATE ... IF NOT EXISTS` throughout, same discipline as M-0032).
- **Existing records immutability proof:** `graveyard_slash_fingerprints` (31 rows) and `epistemic_test_nullifiers` (0 rows, still empty) in the REAL `data/ami/knowledge.sqlite` were re-queried after this batch's test suite ran (all tests used disposable tmp_path copies) — counts unchanged (31 / 0), and the REAL knowledge.sqlite has **no** `epistemic_authorization_tokens` table at all, proving `init_gates_schema` (which creates it) was never run against the real file during this batch — the new schema was only ever exercised against disposable test databases.

## 7. Authorization model

Raw tokens are `secrets.token_urlsafe(32)`, shown once to the operator/approver at issuance and never persisted — only `token_commitment = sha256(raw_token)` is stored (`token_commitment TEXT NOT NULL UNIQUE`). Every field required by the batch spec is present: `authorization_id`, `authorization_type`, `canonical_family_id`, `related_experiment_id`/`related_nullifier`, `split_version`, `test_set_hash`, `approver`, `justification`, `retry_condition_satisfied`, `input_manifest_root`, `issued_ms`, `expiry_ms`, `single_use`, `consumed`, `resulting_experiment_id`, `consumed_ms`. Enforcement (`_reserve_authorization`): missing token, wrong type, already consumed, expired, wrong family, wrong split (supersession only), wrong test-set (supersession only) — **all seven branches raise `AuthorizationInvalid`, fail-closed** (tests #3, #5, #14, #15, #16, #17). Consumption is deferred (`_finalize_authorization_consumption`, no internal commit) until the overall transaction is about to succeed, and itself re-checks `consumed=0` at UPDATE time (`changes()==1` or raise) as a second concurrency backstop on top of the nullifier table's own partial unique index.

## 8. Concurrency / crash / rollback proof

- **DB-level backstop:** `CREATE UNIQUE INDEX ... ON epistemic_test_nullifiers(nullifier) WHERE supersession_token IS NULL` — a second first-consumption INSERT for the same nullifier raises `sqlite3.IntegrityError` regardless of whether the Python-level pre-check was bypassed (test #20, directly exercises the index with a raw second INSERT after a committed first one).
- **Application-level translation:** `consume_test_evidence` catches that `IntegrityError` and re-raises `TestEvidenceReuseBlocked`, so callers see one exception type whichever layer caught the race (test #20b).
- **Crash before registration:** a second call with the same `experiment_id` but different content raises `ImmutableExperimentConflict` from inside `record_experiment_registry`, which — because it happens inside the same uncommitted transaction as the (already-passed) gate checks — rolls back everything; the nullifier remains consumed **only** by the original, first call (test #18).
- **Crash after registration, before final commit:** simulated by patching `consume_test_evidence` to raise after `record_experiment_registry`/`record_experiment_results` have already executed (in-transaction, uncommitted); the `except BaseException: rollback(); raise` clause undoes the registry/result inserts too — verified by a fresh count query showing zero rows for that experiment_id afterward (test #19).

## 9. Test results

30 new test functions added (29 pytest test IDs — `test_09`/`test_20` each cover their scenario with two functions), `tests/test_ami_epistemic_nullifier_enforcement_wiring.py`, covering all 27 required scenarios 1-1 (see file for the numbered mapping). All 16 pre-existing `tests/test_ami_governance_epistemic_gates.py` tests re-verified passing unmodified (backward compatibility of the `_autocommit` default confirmed).

- Isolated file run: 29/29 passed.
- Collect-only (frozen AMI file set: `tests/test_ami_*.py` + `test_buyfade_mutations.py` + `test_buyfade_silexit_mutations.py`): **897** (868 prior ground truth + 29 new).
- **Run1** (paired ≤2-file/call, sequential, 36 pairs): 36/36 pairs green **except** `tests/test_ami_lifecycle_short_noisy_v1_rehearsal.py::test_disposable_db_and_microstructure_db_untouched` — **896/897**. Root-caused live: the real collector processes were actively appending to the 740GB `data/microstructure.db` at the moment of the run (confirmed via `tasklist` showing running collector `python.exe` processes and the file's mtime matching wall-clock "now"); the test's own docstring already documents this exact 64MB-prefix-hash fragility as a known, benign, collector-timing artifact unrelated to code changes (same precedent as M-0031's regression note). Confirmed by **two** isolated reruns of that file alone: first rerun failed again (different hash pair, still collector-caused), second rerun passed clean (9/9) — proving the flake is purely environmental timing, not a real regression (this batch never touches `ami/lifecycle/short_noisy_v1_migration_rehearsal.py`, `ami/lifecycle/migration_rehearsal.py`, or anything under `data/microstructure.db`).
- **Run2** (paired ≤2-file/call, sequential, 36 pairs, same file set): **897/897 ✓ 0 errors**, including a clean pass of the previously-flaky pair.
- `data/ami/canonical.sqlite` sha256 was `458bc07c…` before, during, and after both regression runs (re-verified read-only after Run2) — the immutable-conflict guards matched already-existing content throughout; zero net writes.

## 10. Protected delta / experiment delta

| Check | Result |
|---|---|
| canonical.sqlite sha256 (after this batch) | `458bc07ca5b436041e59c781a26cf502779d5dc2751a3be8a0c1cddb93e84d49` — **unchanged** |
| canonical.sqlite schema_version (after) | **12 — unchanged** |
| `experiment_registry` count (after) | **22 — unchanged** |
| `experiment_results` count | unchanged (implied by unchanged file hash) |
| Protected counts (`ami_events`/`ami_signal_lifecycle`/`ami_cycles`/`ami_birth_truncated_cascade_geometry`) | 252 / 324 / 167 / 220 — **unchanged** |
| `integrity_check` (after) | `ok` |
| Real `knowledge.sqlite` — `graveyard_slash_fingerprints` / `epistemic_test_nullifiers` | 31 / 0 — **unchanged** |
| Real `knowledge.sqlite` — `epistemic_authorization_tokens` | table does not exist on the real file — **new schema never applied to real DB, only to disposable test copies** |
| New experiments created | **0** |
| New experiment_results rows | **0** |
| Runtime/risk/execution/shadow/paper/forward files touched | **0** (none opened, none imported by the new code) |

## 11. Family identity — remaining bypass risk (explicitly required to be reported)

`resolve_canonical_family_id`/`resolve_split_version` hash **free-form researcher-authored text** (`question_ids`, `hypothesis_id`, `frozen_splits`). Two experiments describing the *same* underlying hypothesis or split methodology in *differently-worded* text resolve to **different** family/split identities and therefore a different nullifier — the single-use law does not see through paraphrase, only through whitespace/case variation (closed, test #12). Closing this fully requires a real canonical hypothesis-family ontology (`experiment_registry` schema change or a new registry table), explicitly out of scope for this "wiring only" batch and not attempted.

## 12. KNOWN, UNCLOSED BYPASS — legacy research modules (the reason for the INCOMPLETE verdict)

10 files retain their own inline `INSERT INTO experiment_registry ... ON CONFLICT(experiment_id) DO UPDATE` + a hardcoded `EXPERIMENT_ID` + `if __name__ == "__main__"`:
`ami/research/candidate_universe.py`, `w1_cycle_integrity.py`, `w3_entry_timing_reconciliation.py`, `w4_post_event_path_taxonomy.py`, `w5a_morphology_swing_grammar.py`, `w6_compression_rs_session.py`, `w6rs_confirmation.py`, `w6rs_confound_resolution.py`, `w7a_state_structure_aging_market_clocks.py`, `w10a_multi_tf_structural_conflict.py`.

These structurally bypass **both** the pre-existing immutability guard (`ImmutableExperimentConflict`) **and** this batch's new gates — they never call `ami/warehouse/experiment_ledger.py` at all. They are historical/completed analyses (each hardcoded to one already-registered `experiment_id`, re-running any of them as-is would just re-affirm/silently-upsert that same row, not create a new experiment for a new hypothesis) — but nothing prevents a future researcher from copy-pasting one of them as a template for genuinely new work, changing `EXPERIMENT_ID`, and registering a brand-new experiment through the exact same unguarded inline-SQL path. Closing this means editing 10 historical/frozen research files — repository-wide cleanup, explicitly excluded from this batch's scope by the operator's own instructions. Test #22 in the new suite is a **canary**, not a closure claim: it asserts the offending file set is *exactly* this documented list, so any silent addition to (or removal from) that set is caught by regression rather than assumed away.

`ami/research/registry.py`'s `ResearchRegistry` (a separate, older Phase-4 concept operating on `data/ami/research.sqlite`'s own `experiments` table via blind `INSERT OR REPLACE`, no immutability guard at all) is a second, even wider pre-existing gap, also not gated by this batch (different database, different table, out of scope).

## 13. New state root

| Field | Value |
|---|---|
| canonical.sqlite hash | `458bc07ca5b436041e59c781a26cf502779d5dc2751a3be8a0c1cddb93e84d49` (unchanged) |
| canonical.sqlite schema_version | 12 (unchanged) |
| `GATES_SCHEMA_VERSION` | 2 |
| Migration | `M-0033` (knowledge.sqlite only, additive; not yet applied to the real file — only to disposable test copies) |
| Mandatory gated entry point | `ami.warehouse.experiment_ledger.register_experiment_with_gates()` |

## 14. Remaining risks (full list)

1. **10 legacy research modules bypass the gate structurally** (§12) — the primary reason this closes INCOMPLETE.
2. **`ami/research/registry.py`'s separate `ResearchRegistry`/`research.sqlite` system bypasses the gate entirely** (different DB, not touched).
3. **Family/split identity adapter is text-hash-based**, not a real ontology — paraphrase bypass remains (§11).
4. Carried over from G1, still true: SQL-level UNION of the CVD exact/proxy tables remains technically possible; the four shared governance Markdown files (`SYSTEM_STATE.md`/`IMPLEMENTATION_PROGRESS_LEDGER.md`/`TEST_STATUS_LATEST.md`/`MIGRATION_LOG.md`) still carry a mixed-commit-boundary waiver and are **not** touched or committed by this batch either.
5. **`M-0033`'s new schema has not yet been applied to the real `data/ami/knowledge.sqlite`** — only to disposable test copies. The first real experiment registered through `register_experiment_with_gates()` will apply it automatically (idempotent, additive) as a side effect of that first real call.

---

**Verdict: `EPISTEMIC_NULLIFIER_ENFORCEMENT_WIRING_V1_INCOMPLETE`**

Reason: the enforcement mechanism is correctly and atomically wired into the designated centralized path, fully tested (27/27 required scenarios, 897/897 regression), and zero canonical/knowledge state was altered on the real databases — but "all production-capable experiment registration paths must pass through the gates" is **not yet true**: 10 pre-existing legacy research modules (§12) retain a real, callable, unguarded bypass. Closing that gap is repository-wide cleanup of historical/frozen research files, which this batch's own scope explicitly excludes. This is reported as an unresolved risk rather than closed as complete.
