# STORAGE_ROTATION_RETENTION_PRODUCTION_ACTIVATION_REHEARSAL_V1

**Gate:** BATCH-STORAGE-ROTATION-RETENTION-PRODUCTION-ACTIVATION-REHEARSAL-V1
**Nature:** First production-path archive publication rehearsal. Publishes exactly one verified production archive partition. No general production activation, no purge, no scheduler, no VACUUM, no source mutation.
**Date:** 2026-07-08 · **Author:** Sonnet 5

---

## Headline result

**Exactly one production archive partition was published, verified, and re-verified with zero mismatches — `mark_prices`/`ETHUSDT`/`2026-05`/`v1`.** Row count, watermark, and scientific-content hash are byte-identical to the accepted disposable dry-run (commit `6fbe0571`) and the bounded-implementation acceptance reproduction (commit `55a017ff`). Direct production Parquet read, minimal SQLite-slice restore, deterministic root-catalog-index rebuild, and idempotent rerun (`NOOP_IDENTICAL_PRODUCTION_ARCHIVE`, confirmed via the actual CLI a second time — files untouched, mtimes unchanged) all passed. Interruption/recovery and corruption/tamper detection were proven on disposable fixtures. The source database received **zero write attempts** throughout. General production activation, scheduler, purge, and VACUUM all remain structurally disabled.

---

## Phase 1 — Pre-activation reconciliation

No frozen production archive root existed in any accepted artifact (`grep` across `reports/`, `ami/`, `SYSTEM_STATE.md` for `data/archives`/`raw_v1`/production-root mentions: 0 hits prior to this batch). `data/archives/` did not exist. Per the operator ruling, the fallback root `D:\eclipse_scalper\data\archives\raw_v1\` was used and recorded as `root_source="operator_approved_fallback"`. No pre-existing archive-like files, no prior production catalog, no path conflicts, no staging leftovers existed before this batch began.

## Phase 2 — Production path contract

Deterministic Hive-style layout, exactly as specified:

```
data/archives/raw_v1/
  table=mark_prices/venue=BINANCE_USDM_PERP/market_segment=PERPETUAL_FUTURES/
  symbol=ETHUSDT/year=2026/month=05/version=v1/
    part-00000.parquet
    manifest.json
    catalog_entry.json
    _SUCCESS
  catalog_index.json
```

Staging occurs on the same volume beneath `data/archives/raw_v1.staging/<table>_<symbol>_<year>-<month>_wm<watermark>_<job_identity>/` — a unique, job-owned directory per attempt. The final directory did not exist before atomic publication (`os.rename`, same-volume, atomic).

**Implementation note (disclosed, fixed):** the initial implementation used `pyarrow.parquet.read_table()` for post-write verification, which triggers pyarrow's dataset/Hive-partition auto-discovery on the `key=value/` directory layout — the path's `symbol=ETHUSDT` segment collided with the real `symbol` column, raising `ArrowTypeError: Unable to merge... symbol has incompatible types`. Fixed by switching every single-file read (in `production.py` and `reader.py`) to `pq.ParquetFile(path).read()`, which reads exactly one file with no dataset inference. The actual data export/publication had already succeeded on the first attempt before this was discovered — only the *verification* read path needed the fix, and a leftover from that first (verification-failed but data-correct) attempt was deleted before the real, clean publication run.

## Phase 3 — Production policy extension

`ami/storage/policy.py::ProductionRehearsalAuthorization` — a frozen dataclass whose default values are the exact single partition (`mark_prices`/`ETHUSDT`/`BINANCE_USDM_PERP`/`PERPETUAL_FUTURES`/`2026-05`/`v1`). `is_authorized()` checks all seven fields; any mismatch on any field is denied. `GENERAL_PRODUCTION_ACTIVATION_ENABLED = False` is a module-level constant, the single source of truth that nothing in this batch flips. There is no reusable "authorize any partition" mechanism — the authorization is a literal, not a parameterized rule.

## Phase 4 — Persistent production catalog

Immutable, filesystem-backed, no mutable production SQLite database, no canonical migration. Each partition carries its own `catalog_entry.json` (`ami/storage/production.py::build_catalog_entry()`); the root `catalog_index.json` is **rebuilt from these partition-local entries**, never the reverse — `build_root_catalog_index()` walks the root, rejects any entry with `purge_authorization != PROHIBITED` or `verification_status != VERIFIED`, rejects duplicate archive identities with differing content, and produces a deterministic, self-hashed (`index_self_hash`) JSON body. Published via the same `.partial`→atomic-replace discipline as everything else. A root-index failure cannot corrupt a partition-local archive (they are independent write operations); this batch's real run published both successfully.

## Phase 5 — Manifest production state

The accepted 36-field manifest contract (`ami/storage/archive.py::build_manifest()`) gained one new optional parameter, `production_status` (default `"DISPOSABLE_NOT_PRODUCTION"`, the only value every disposable caller ever passes), validated against `ALLOWED_PRODUCTION_STATUS = ("DISPOSABLE_NOT_PRODUCTION", "PRODUCTION_VERIFIED")` — any other value raises `ValueError`. No field was removed or renamed. The clean separation:

| Artifact | Mutability | Scope |
|---|---|---|
| `manifest.json` | immutable once published | one partition, scientific + minimal operational identity |
| `catalog_entry.json` | immutable once published | one partition, full operational/governance status |
| `catalog_index.json` | **rebuildable** (deterministic, derived) | whole production root, operational metadata only |

## Phase 6 — Source snapshot and watermark

Opened via the accepted `ami.storage.source_access.open_read_only()` (`mode=ro` + `query_only` + authorizer). A **fresh** watermark was captured at rehearsal time (not reused from any historical value): `MAX(id)` for `symbol='ETHUSDT'` within the May-2026 window. Result: **identical to the accepted historical reference** — row count 260,657, watermark 13,265,132, scientific-content hash `228c5705bc1461471102b5bbc1e81b781ddfe7cb5a853b484a7cbb5961860999`. No late-arriving or repaired rows had changed the frozen population between the bounded-implementation batch and this one — confirmed, not assumed, by re-querying the live database.

## Phase 7-8 — Staged export and atomic publication

Full 20-step sequence executed exactly as specified: unique staging dir → `.partial` Parquet (ZSTD) → stream/validate → rename to final name inside staging → `.partial` manifest → validate → rename → `.partial` catalog entry → validate → rename → `.partial` `_SUCCESS` → rename → validate complete staging dir (no `.partial` remnants, all 4 required files present) → atomic directory rename to the final production path (which was proven not to exist immediately before the rename).

## Phase 9 — Post-publication re-verification

`reverify_published_partition()` independently re-opened the **final published path** (not the staging copy) and re-checked: `_SUCCESS` presence, Parquet-file-hash vs. manifest, manifest-hash vs. catalog-entry, manifest scientific-hash vs. catalog-entry scientific-hash, `partition_id` match, `production_status`/`purge_authorization` values in both manifest and catalog entry, and a full re-read-and-re-hash of every row. **Result: 0 mismatches**, both immediately after the real publication and again independently in this batch's test suite (`test_real_production_reverification_zero_mismatches`).

## Phase 10 — Root catalog index

Built and published (`catalog_index.json`, 1,971 bytes, `entry_count=1`, `index_self_hash=2f0bf514…`). Rebuilt independently twice more in this batch (once via ad hoc verification, once via the committed test suite) — **byte-identical `index_self_hash` both times**, confirming full determinism. The index is operational metadata only (no source row content, no outcome data — confirmed by a dedicated test asserting `"endpoint_return_bps"`/`"mfe_bps"` never appear in its serialized form).

## Phase 11 — Direct production Parquet read

`ami.storage.reader.read_partition()` against the real published path, gated by the real manifest (symbol/venue/market-segment match, checksum match). Returned exactly 260,657 rows; canonical-hashed and compared against a **fresh** independent source re-query — **0 mismatches, exact row-for-row parity**.

## Phase 12 — Minimal restore from production archive

Restored into `.runtime_temp/storage_rotation_production_activation_rehearsal_v1/restored.sqlite` (only `mark_prices_restored`, only the frozen partition, no outcome/canonical/governance tables). Row count 260,657, scientific-content hash identical to both the source and the published Parquet. **Deleted immediately after the parity proof was recorded** — confirmed absent afterward.

## Phase 13 — Idempotent rerun

Two independent proofs:
1. **CLI-level:** `python -m ami.storage.cli production-activation-rehearsal` was run a **second time** against the already-published archive. Result: `{"status": "NOOP_IDENTICAL_PRODUCTION_ARCHIVE", "reverification_mismatch_count": 0, ...}` — file mtimes on all 4 partition files were **unchanged** between the two CLI invocations (verified via `ls -la`), and the Parquet SHA-256 was re-hashed and matched byte-for-byte, proving no rewrite occurred.
2. **Test-level:** `test_full_orchestration_second_run_returns_noop` and `test_idempotent_rerun_does_not_rewrite_files` (disposable fixtures) both assert the same behavior deterministically.

No `version=v2` directory was ever created (confirmed both manually via `find` and by a dedicated test).

## Phase 14 — Interruption and recovery

Tested exclusively on disposable roots (never the real archive): a partial Parquet-only staging directory, and a staging directory with only a completed Parquet but no manifest/catalog-entry/`_SUCCESS`, were both proven to never produce a final-path directory. A subsequent fresh publish attempt (different job identity) succeeded cleanly and was not blocked or confused by the abandoned staging leftover. Abandoned staging directories are excluded from the root catalog index by construction (the index scanner only walks the production root, never `.staging`).

## Phase 15 — Corruption and tamper detection

All tested on disposable copies under `tmp_path`, never the real archive: altered Parquet byte (hash differs, original untouched), truncated Parquet (rejected by the reader as `ArchiveCorruptionError`), missing Parquet (rejected), altered manifest field (mismatch provable), altered catalog-entry field (`purge_authorization` tampering rejected by `build_root_catalog_index()`), missing `_SUCCESS` (detected by `reverify_published_partition`), corrupted root index file (rebuild from partition-local entries is unaffected — the index is always derived, never trusted as a source of truth). **The real production archive's three file hashes were re-verified identical before and after this batch's entire corruption-test suite ran** (`test_real_production_archive_byte_identical_before_and_after_corruption_tests`).

## Phase 16 — Source retention and research-dependency proof

Read-only against the **real** database: the frozen May-2026 `mark_prices`/`ETHUSDT` population is unchanged (260,657 rows, hash `228c5705…`, zero write attempts in the authorizer log). `catalog_entry.json`'s `source_retention_status="SOURCE_PRESENT"` and `research_dependency_status="BLOCKED"` are hardcoded, not computed from a live check — a verified archive does **not** by itself satisfy any research dependency. Current `mark_prices` dependencies (unchanged from the readiness batch, `f65545ee`): `FUNDING_LEVEL_VELOCITY` (funding_rates is dead; mark_prices is the last surviving raw funding-adjacent source) and `FAM_SPOT_PERP_BASIS_REVERSAL` (future basis bridging work). `purge_authorization` remains `PROHIBITED` in every artifact this batch produced.

## Phase 17 — Production archive health report

`ami/storage/health.py::build_health_report()` extended with 9 new fields (`production_archive_root`, `verified_production_partitions`, `failed_production_partitions`, `staging_directory_count`, `abandoned_staging_count`, `root_catalog_index_status`, `root_catalog_entry_count`, `total_archive_bytes`, `latest_publication_timestamp`, `production_archive_rehearsal_status`), populated by a **read-only** scan (`_scan_production_root()`) of the real root. Live result: `verified_production_partitions=1`, `root_catalog_index_status="PRESENT"`, `total_archive_bytes=2476313`, `production_archive_rehearsal_status="COMPLETE"`. `production_activation`/`scheduler`/`purge`/`vacuum` all remain the hardcoded `"DISABLED"` string default — unaffected by the scan.

## Phase 18 — CLI boundary

One new subcommand: `production-activation-rehearsal`. It accepts **zero** partition-identifying arguments (no `--table`, `--symbol`, `--utc-year`, `--utc-month`, `--output-root`, `--production-root`, or `--archive-version`) — confirmed by a dedicated test enumerating the subparser's registered option strings and asserting disjointness from that set. There is nothing to parameterize, so it cannot become a general production-enable command by any future misuse of its own arguments. No `activate-production`, `archive-all`, `archive-range`, `schedule`, `purge`, `delete`, `vacuum`, `compact`, `stop-collector`, or `restart-collector` command exists (same structural absence proof as the prior batch, extended to cover this one).

## Phase 19 — Resource and drive-safety guards

| Guard | Limit | Actual |
|---|---|---|
| Source partition estimate | ≤2 GB | ~16.7 MB (from the planner's own estimate) |
| Production Parquet | ≤1 GB | 2,476,313 bytes (~2.36 MiB) |
| Staging + disposable verification | ≤4 GB | ~2.5 MB actual peak |
| Minimum `D:` free space before start | ≥100 GB | 1,124 GB (confirmed before publication) |
| Full database copy | none | none created |
| Full database hash | none | never computed |

All limits comfortably satisfied; no abort was triggered (none was needed).

---

## Verdict

**`STORAGE_ROTATION_RETENTION_PRODUCTION_ACTIVATION_REHEARSAL_V1_COMPLETE`**

**`STORAGE_ROTATION_RETENTION_SINGLE_PRODUCTION_PARTITION_VERIFIED_SOURCE_RETAINED`**

This disposition authorizes nothing beyond the one published, re-verified partition. General production activation, scheduler activation, purge, and VACUUM all remain structurally disabled — no code path exists to enable any of them from this batch's work.

## Blockers / residual risks

1. Only one table/partition (`mark_prices`/2026-05) has been through the full production path; `agg_trades`/`book_ticker` production publication remains unrehearsed at the production-path level (though fully fixture-tested at the engine level in the prior bounded-implementation batch).
2. The root catalog index has no locking/concurrency contract for simultaneous publications — out of scope for a single-partition rehearsal, would need addressing before any multi-job production activation.
3. `_scan_production_root()`'s `staging_directory_count` field is currently always `0` (a placeholder — live staging-in-progress detection was not needed for this single, already-completed rehearsal and was not built out).

## Storage report

| Item | Value |
|---|---|
| Production files retained (permanent, outside Git) | `part-00000.parquet` (2,476,313 B), `manifest.json` (1,898 B), `catalog_entry.json` (1,687 B), `_SUCCESS` (26 B), `catalog_index.json` (1,971 B) — **2,482,981 total bytes (~2.37 MiB)** |
| Staging files created and consumed | 1 job's worth (`.partial` × 4, renamed in place, then the whole staging directory atomically renamed away — nothing left in `.staging`) |
| Disposable verification files created and deleted | 1 restored SQLite slice (`.runtime_temp/storage_rotation_production_activation_rehearsal_v1/restored.sqlite`), deleted after parity proof; all corruption-test copies (disposable, under `tmp_path`, auto-cleaned by pytest) |
| Final `data/archives/raw_v1.staging/` contents | empty |
| Final `.runtime_temp/storage_rotation_production_activation_rehearsal_v1/` contents | empty (restored slice deleted) |
| Peak disk usage this batch | ~2.5 MB disposable + 2.37 MiB permanent production data |
| Full database copy created | **confirmed NOT created** |
| Source rows modified or deleted | **confirmed NOT occurred** |
| Purge capability added | **confirmed NOT added** |
| Scheduler capability added | **confirmed NOT added** |
| VACUUM capability added | **confirmed NOT added** |
| General production activation | **confirmed remains disabled** |

## Next controlled gate

Only if a future operator chooses to proceed: **`BATCH-STORAGE-ROTATION-RETENTION-PRODUCTION-ARCHIVE-ACTIVATION-V1`** — may enable manually authorized production archive creation for allowlisted closed partitions (all three tables), still prohibiting source-row deletion, purge, automatic scheduling, collector changes, and VACUUM. After that is proven across all three tables, later independent gates would cover: scheduler planning, research-reader integration, purge readiness/dependency release, bounded source-row purge authorization, and finally `BATCH-STORAGE-SQLITE-PHYSICAL-RECLAMATION-MAINTENANCE-V1` (not to be opened until production archival is established, an authorized purge has occurred, and substantial reclaimable pages exist).

Not begun by this batch.
