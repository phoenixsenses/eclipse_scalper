# CVD_CANONICAL_MIGRATION_G1_STATE_TRANSITION_PROOF

**Batch:** BATCH-CVD-G1-PROVENANCE-BOUNDARY-CLOSURE
**Purpose:** Additive, immutable provenance/transition-proof artifact closing gate G1 (CVD canonical migration schema 11→12) without attempting unsafe commit-hunk separation of shared governance Markdown files.
**Nature:** Documentation/provenance-only. No migration rerun, no regression rerun, no code change, no schema change, no wiring change.
**Author:** Sonnet 5 · **Date:** 2026-07-06

---

## 1. Previous canonical state

| Field | Value |
|---|---|
| schema_version | 11 |
| canonical hash (sha256 of `data/ami/canonical.sqlite` pre-migration, = backup file hash) | `b28b40938bd76524d39dd6c1b82905b4d07d9e88c98b0c2daabcfcc55455009d` |
| pre-migration protected counts | `ami_events`=252, `ami_signal_lifecycle`=324, `ami_cycles`=167, `ami_birth_truncated_cascade_geometry`=220 |
| pre-migration new CVD tables | absent (confirmed) |
| pre-migration integrity_check | ok |

Verified in this closure batch: re-hashed `data/ami/backups/canonical_pre_cvd_repair_canonical_migration_20260706_065631.sqlite` → matches `b28b4093…` exactly (independent re-computation, read-only, this batch).

## 2. New canonical state

| Field | Value |
|---|---|
| schema_version (live, `schema_versions` table, component=`canonical_warehouse`) | **12** |
| final canonical hash (sha256 of LIVE `data/ami/canonical.sqlite`, this batch, read-only) | **`458bc07ca5b436041e59c781a26cf502779d5dc2751a3be8a0c1cddb93e84d49`** |
| final canonical hash (sha256 of `canonical_post_cvd_repair_canonical_migration_v12_20260706_070000.sqlite`) | `458bc07ca5b436041e59c781a26cf502779d5dc2751a3be8a0c1cddb93e84d49` (byte-identical to live) |
| total tables (live) | 39 (33 pre-existing + 6 new CVD tables) |

Live hash == post-migration backup hash == the hash recorded at migration time in `MIGRATION_LOG.md` M-0031 and in `IMPLEMENTATION_PROGRESS_LEDGER.md`. No drift since 2026-07-06T07:00:00Z.

## 3. Migration identity

| Field | Value |
|---|---|
| Migration ID | `M-0031` (canonical.sqlite; distinct from `M-0032`, which targets `knowledge.sqlite` and is NOT part of G1 — see §6) |
| Batch ID (execution) | `BATCH-CVD-REPAIR-REHEARSAL-AND-QUALITY-CONTRACT-V1` (migration step) |
| Batch ID (this closure) | `BATCH-CVD-G1-PROVENANCE-BOUNDARY-CLOSURE` |
| Execution timestamp | pre-migration backup `20260706_065631`; post-migration backup `20260706_070000`; manifest `recorded_at_utc: 2026-07-06T09:00:00Z` |
| Frozen source package | `data/ami/cvd_rehearsal_disposable_20260705/cvd_rehearsal_disposable.sqlite` (sole input for D3/D4/D5 canonical backfill; 0 network refetch during migration) — sha256 (this batch, read-only) = `0b80e1863ff63b004725369d784bee04cf07eeccb5ba8d4a5c85cabfe6e4a0aa` |
| Frozen source content hashes (exact/proxy/quality, matched pre↔post per manifest) | exact=`ca11be783e6cd19f2c0e5cfa679bb517228e5aa120474a7a37a6c7f080ab87af`, proxy=`0a8ac304fc139827b771d624bf2888624065b51388ca06af5a4fce25c31a4032`, quality=`6e95d51a242792a88a5ca3fbf80a436d48c2d5f905e59bafa1c775734caeff89` |
| Backup path (pre) | `data/ami/backups/canonical_pre_cvd_repair_canonical_migration_20260706_065631.sqlite`, sha256=`b28b4093…` (re-verified this batch) |
| Backup path (post) | `data/ami/backups/canonical_post_cvd_repair_canonical_migration_v12_20260706_070000.sqlite`, sha256=`458bc07c…` (re-verified this batch) |
| Manifest path | `data/ami/backups/canonical_pre_cvd_repair_canonical_migration_20260706_065631.manifest.json` |

## 4. Exact G1 implementation/test file boundary

| File | Status | Role |
|---|---|---|
| `ami/warehouse/schema.py` | Modified (tracked, `M` in git status) | `CANONICAL_SCHEMA_VERSION` 11→12, new `_SCHEMA_PHASE_CVD` DDL block |
| `ami/cvd/cvd_canonical_migration.py` | New (untracked) | `run_canonical_migration()` — verbatim copy from frozen source, content-compare idempotent |
| `tests/test_ami_cvd_canonical_migration.py` | New (untracked) | +5 tests, disposable-copy-only, not-called-automatically guard |
| `tests/test_ami_lifecycle_provenance_rehearsal.py` | Modified (tracked, `M` in git status) | `schema_version_before` tuple extended `(8,9,10,11)`→`(8,9,10,11,12)` |

These four files are G1's own implementation/test surface and are independently identifiable in `git status`/`git diff` — no mixing issue exists for these.

## 5. Exact accepted prerequisite file boundary

Prerequisite work, accepted in an earlier batch (`BATCH-CVD-REPAIR-REHEARSAL-AND-QUALITY-CONTRACT-V1`, rehearsal step, disposable-only, 0 real DB writes), consumed but not produced by G1:

| File | Role |
|---|---|
| `ami/cvd/windowed_taker_flow.py` | Rehearsal module — windowed taker-flow feature computation (exact + proxy), DDL source for `_SCHEMA_PHASE_CVD` |
| `ami/cvd/cvd_source_quality_contract_v1.py` | Rehearsal module — quality contract / `ami_cvd_window_quality_v1` DDL source |
| `ami/cvd/aggtrades_repair_rehearsal.py` | Rehearsal module — agg-trades repair, DDL source for `ami_agg_trades_repaired` (renamed from `..._stage`) |
| `ami/cvd/cvd_rehearsal.py` | Rehearsal orchestrator |
| `ami/cvd/__init__.py` | Package init |
| `data/ami/cvd_rehearsal_disposable_20260705/cvd_rehearsal_disposable.sqlite` + sibling probe/replay/regression scripts in the same directory | Frozen disposable source package (sole content origin for the canonical backfill; immutable, never deleted per operator instruction) |

None of these files were modified by G1 itself; G1 only reads from the frozen sqlite and copies DDL byte-for-byte.

## 6. The four shared governance Markdown files

All four are currently modified (`M` in git status) against the same `HEAD` (`09af9dc6`), and each shows **exactly one contiguous git diff hunk** — confirmed this batch via `git diff --stat` / hunk count:

| File | Current content sha256 (this batch, working tree) | Diff hunk | Contributing batches (in file order) |
|---|---|---|---|
| `SYSTEM_STATE.md` | `ddab4ff479190de7471deb5bc2392172b75e95d93f5bf58d8443396b4479363f` | 1 hunk, `@@ -4540,4 +4540,130 @@` | §93 (G1, lines 4601–4623) → §95 (shadow-mirror, lines 4625–4653) → §94 (nullifier, lines 4655–4669) |
| `IMPLEMENTATION_PROGRESS_LEDGER.md` | `4da6f75793b77e01eadff8a3d47e2d5aa38cc7a339032b983f24e32bb66434d2` | 1 hunk, `@@ -66,7 +66,12 @@` | line 75 (G1 row) → line 76 (nullifier row) → line 77 (shadow-mirror row) |
| `TEST_STATUS_LATEST.md` | `e52764eb596e88b80a073f7107dda36bf1ae8650b24ff4a5c5ce99e28530cc5c` | 1 hunk, `@@ -1,6 +1,22 @@` | line 3 (shadow-mirror, newest-first at top) → line 7 (nullifier) → line 11 (G1) |
| `MIGRATION_LOG.md` | `6b63cbce134583a134e66bdc7ac34e748a3ad138e73076a351d78fcded99dd40` | 1 hunk, `@@ -4,6 +4,8 @@` | line 7 (M-0032, nullifier, targets `knowledge.sqlite`) → line 8 (M-0031, G1, targets `canonical.sqlite`) |

**Which sections belong to G1:**
- `SYSTEM_STATE.md` §93 (lines 4601–4623)
- `IMPLEMENTATION_PROGRESS_LEDGER.md` row at line 75 (`BATCH-CVD-REPAIR-REHEARSAL-AND-QUALITY-CONTRACT-V1 (migration step)`)
- `TEST_STATUS_LATEST.md` block at line 11 (`Önceki güncelleme … BATCH-CVD-REPAIR-REHEARSAL-AND-QUALITY-CONTRACT-V1, migration adımı`)
- `MIGRATION_LOG.md` row at line 8 (`M-0031`)

**Which sections belong to the separately accepted epistemic-nullifier batch (`BATCH-EPISTEMIC-NULLIFIER-GATES-V1`):**
- `SYSTEM_STATE.md` §94 (lines 4655–4669)
- `IMPLEMENTATION_PROGRESS_LEDGER.md` row at line 76
- `TEST_STATUS_LATEST.md` block at line 7
- `MIGRATION_LOG.md` row at line 7 (`M-0032`, targets `knowledge.sqlite`, not `canonical.sqlite`)

**Which sections belong to the additional accepted batch (`S34-VENGINE-V02-SHADOW-MIRROR-RUNTIME-HARDENING-V1`), occurring before this G1 closure:**
- `SYSTEM_STATE.md` §95 (lines 4625–4654)
- `IMPLEMENTATION_PROGRESS_LEDGER.md` row at line 77
- `TEST_STATUS_LATEST.md` block at line 3 (current top of file)
- `MIGRATION_LOG.md` — no row (this batch performed no schema/canonical-DB change; runtime/checkpoint-only, `tools/` + `start_eclipse.ps1`/`status_eclipse.ps1`)

**Why safe Git hunk-level separation is impossible:**
1. **Mechanical:** `git diff` confirms exactly one hunk per file (verified this batch — see table above). A hunk is a maximally-contiguous run of changed lines; because all three batches appended their entries back-to-back with no surviving unchanged context between them, the diff algorithm cannot present them as separate hunks. `git add -p` therefore offers each file as one all-or-nothing unit (or, for `IMPLEMENTATION_PROGRESS_LEDGER.md`/`MIGRATION_LOG.md`, individual table rows are technically distinguishable by line content, but `SYSTEM_STATE.md`/`TEST_STATUS_LATEST.md` sections contain multi-paragraph prose with no reliable machine-safe split point within the single hunk).
2. **Policy/safety:** These four files are append-only governance ledgers recording already-accepted, already-verified batch outcomes. Manually re-splitting a single hunk (via interactive `e`-edit or hand-authored patch) means hand-editing text that a prior batch already wrote and validated — exactly the kind of "reconstruct or split" / "destructively edit the working tree" operation this closure is instructed not to attempt. A slip in a manual split (wrong line boundary, accidental whitespace/heading corruption) would silently corrupt the historical record with no test coverage to catch it, since these are prose Markdown, not code.
3. **No canonical-state coupling:** critically, none of this affects the canonical SQL data itself — `ami/warehouse/schema.py`, `ami/cvd/cvd_canonical_migration.py`, and the two CVD test files (§4) are cleanly, unambiguously attributable to G1 alone via git status with zero mixing. The mixing is confined to the four narrative/ledger Markdown files.

## 7. Explicit boundary-waiver statement

- The four files named in §6 (`SYSTEM_STATE.md`, `IMPLEMENTATION_PROGRESS_LEDGER.md`, `TEST_STATUS_LATEST.md`, `MIGRATION_LOG.md`) are accepted as **shared, append-only projections** that may carry entries from multiple independently-accepted batches within a single commit.
- Their mixed commit boundary **does not imply mixed canonical SQL state**. G1's canonical migration (`M-0031`, schema 11→12) is fully isolated in its own tables, its own 5 dedicated files (§4), its own backup pair, and its own content hashes — none of which overlap with `M-0032` (knowledge.sqlite) or the shadow-mirror runtime-hardening batch (`tools/`, no canonical DB write at all).
- Migration implementation, schema transition, data counts, and runtime evidence for G1 remain **independently attributable** via §3–§5 and §8–§9 of this document, regardless of how the four shared Markdown files are eventually committed.
- **No historical evidence or commit history was rewritten** by this closure batch or by any of the three contributing batches. All three batches' Markdown entries are present, in full, in the current working tree.
- This waiver applies **only** to the four named files and **only** to this transition (the co-mingling of G1 / `BATCH-EPISTEMIC-NULLIFIER-GATES-V1` / `S34-VENGINE-V02-SHADOW-MIRROR-RUNTIME-HARDENING-V1` entries as of `HEAD=09af9dc6` plus current uncommitted working-tree state).
- **Future batches must produce their own separate transition-proof artifact** (following this document's structure) before further appending to these four shared projection files, so each batch's contribution stays independently provable even as the files continue to accumulate mixed commit boundaries.
- Each of the three contributing batches and its exact sections in the four files, where determinable, are enumerated in §6. Two files (`IMPLEMENTATION_PROGRESS_LEDGER.md`, `MIGRATION_LOG.md`) resolve to exact single-line/single-row boundaries per batch. Two files (`SYSTEM_STATE.md`, `TEST_STATUS_LATEST.md`) resolve to exact section/paragraph-block boundaries per batch (line ranges given in §6), even though those boundaries cannot be turned into separate git hunks.

## 8. Frozen CVD accounting

All values re-verified this batch by direct read-only SQL query against the live `data/ami/canonical.sqlite`:

| Metric | Value | Verified |
|---|---:|---|
| Repaired agg-trades (`ami_agg_trades_repaired`) | 40,934 | ✓ |
| Retrieval batches (`ami_cvd_repair_batch_ledger`) | 8 | ✓ |
| Exact features (`ami_cvd_windowed_flow`) | 1,840 | ✓ |
| Proxy features (`ami_cvd_windowed_flow_proxy`) | 1,840 | ✓ |
| Exclusions (`ami_cvd_bucket_exclusions`) | 104 | ✓ |
| Quality rows (`ami_cvd_window_quality_v1`) | 1,840 | ✓ |
| — of which EXACT_RECONSTRUCTABLE | 1,828 | ✓ (`quality_status` group-by) |
| — of which SOURCE_GAPPED | 12 | ✓ (`quality_status` group-by) |
| Exact reconciliation (1,840 + 104) | 1,944 | ✓ arithmetic |
| Proxy reconciliation (1,840 + same 104) | 1,944 | ✓ arithmetic |
| Exact vs proxy pooling | **not pooled** — separate tables (`ami_cvd_windowed_flow` vs `ami_cvd_windowed_flow_proxy`), no UNION view defined in schema | ✓ |

## 9. Proof results

| Check | Result |
|---|---|
| Known-at violations | 0 |
| Runtime outcome reads (this closure batch) | 0 |
| Runtime outcome writes (this closure batch) | 0 |
| Experiment content delta | 0 (`experiment_registry`/`experiment_results` untouched by G1) |
| Protected subsystem delta | 0 (`ami_events`=252, `ami_signal_lifecycle`=324, `ami_cycles`=167, `ami_birth_truncated_cascade_geometry`=220 — all unchanged, re-verified this batch) |
| Idempotent rerun | `NOOP_IDENTICAL` (recorded at migration time: second application produced 0 new rows, hashes unchanged) |
| Restore proof | Passed (disposable-copy restore verified schema_version=11, 6 new tables absent, protected counts 252/324/167/220, `integrity_check`=ok — prior to touching the live DB) |
| Regression — G1-specific | `test_ami_cvd_canonical_migration.py` (+5) green; full paired ≤2-file/call sequential regression **Run2 = 852/852 ✓ 0 errors** (Run1 pre-fix = 851/852, both fixes applied, Run2 fully green); collect-only = 852 exact match |
| Regression — latest overall ground truth (post subsequent accepted batches, includes G1's 5 tests unchanged) | **868/868**, two independent full paired ≤2-file/call regressions: Run1 = 868/868 ✓, Run2 = 868/868 ✓, 0 errors. No regression has been rerun since (correctly — no code/schema change occurred in the intervening shadow-mirror-hardening batch; confirmed by that batch's own note: "AMI ground truth unaffected, 868/868 stays constant") |
| Final live integrity check (this batch, read-only) | `PRAGMA integrity_check` = `ok`; `PRAGMA foreign_key_check` = 0 violations; total tables = 39 (33 pre-existing + 6 CVD) |

## 10. Remaining risks

- **SQL-level UNION is technically possible** despite the physical separation of `ami_cvd_windowed_flow` (exact) and `ami_cvd_windowed_flow_proxy` (proxy) into distinct tables — nothing in the schema prevents a future query from `UNION`-ing them. No such query exists today; this is a latent risk for future research code, not a current defect.
- **Shared Markdown projections carry a waived commit-boundary limitation** (§6–§7): `SYSTEM_STATE.md`, `IMPLEMENTATION_PROGRESS_LEDGER.md`, `TEST_STATUS_LATEST.md`, `MIGRATION_LOG.md` cannot be hunk-split by batch at commit time; any future commit touching these files will need its own transition-proof artifact rather than relying on git history to prove per-batch attribution.
- **Neither issue changes the accepted canonical data state.** G1's schema-12 tables, counts, hashes, and regression results stand independently of both risks.

---

**Verdict: `CVD_CANONICAL_MIGRATION_COMPLETE`**
