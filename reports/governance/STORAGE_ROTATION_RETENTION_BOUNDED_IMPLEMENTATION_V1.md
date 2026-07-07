# STORAGE_ROTATION_RETENTION_BOUNDED_IMPLEMENTATION_V1

**Gate:** BATCH-STORAGE-ROTATION-RETENTION-BOUNDED-IMPLEMENTATION-V1
**Nature:** Production-quality, fail-closed, deterministic archive tooling. No production activation, no purge, no scheduler, no VACUUM, no collector change, no outcome access.
**Date:** 2026-07-07 · **Author:** Sonnet 5

---

## Headline result

A new `ami/storage/` package (13 modules, 1,478 lines) implements the full accepted archive contract as maintainable, typed repository code — policy, source-table registry, closed-month partition model, read-only planner, bounded exporter, schema/manifest contracts, layered verifier, disposable catalog, direct Parquet reader, minimal SQLite-slice restorer, job-state machine, health reporting, and a bounded CLI. **The accepted `mark_prices`/ETHUSDT/2026-05 rehearsal was reproduced live against the real database using this new implementation and matched the original disposable driver's output byte-for-byte** (scientific-content hash `228c5705…` and Parquet SHA-256 `6f919144…`, both identical) — a stronger result than required (only scientific-content equality was mandated; full byte equality was not forced and happened anyway). 144 new focused tests pass; the three prior storage-batch test suites (92 tests) remain green, unaffected.

---

## Phase 1 — Reconciliation of the disposable dry-run implementation

The disposable driver (`'.runtime_temp/storage_rotation_dry_run_v1/driver.py`, never committed) was a single-file, hardcoded-to-`mark_prices` proof script. Reconciliation:

| Disposable driver logic | Disposition in the bounded implementation |
|---|---|
| `canonical_row_hash()` | Promoted verbatim (same construction) into `ami/storage/archive.py` |
| Hardcoded `mark_prices` column list/types | Generalized into `ami/storage/registry.py::SourceTableSpec`, now covering all 3 allowlisted tables |
| Inline watermark/export/validate/publish sequence | Split into typed, independently-testable functions across `partition.py` (identity), `archive.py` (export+manifest), `verifier.py` (checks) |
| Ad hoc authorizer + `query_only` setup | Promoted into `ami/storage/source_access.py`, reusable by every future caller |
| One-off interruption/corruption simulation code | Replaced by a proper `ami/storage/job_state.py` state machine + `ami/storage/verifier.py` layered checks, not copied verbatim (the disposable simulation code was throwaway proof, not designed for reuse) |
| No CLI | New `ami/storage/cli.py` (repository `tools/*.py` convention: `parse_args`/`main`/`sys.exit`) |
| No catalog, no reader, no restorer as reusable code | New `catalog.py`, `reader.py`, `restorer.py` |

Nothing from the disposable driver was copied wholesale; every reusable piece of logic was re-implemented as a typed, independently-tested module.

## Phase 2 — Storage policy model

`ami/storage/policy.py::StoragePolicy` (plain `@dataclass`, matching repository convention — not `frozen=True`, validated in `__post_init__`). `validate_policy()` fails closed for: unknown policy version, unsupported archive format/compression, `<30`-day retention, non-UTC timezone, and — critically — **any attempt to construct a policy with `automatic_purge_enabled`, `automatic_vacuum_enabled`, `production_activation_enabled`, `scheduler_activation_enabled`, or `partial_month_purge_allowed` set `True`**. There is no override parameter anywhere in the class. `DEFAULT_POLICY` is the only pre-built instance and is safe by construction.

## Phase 3 — Source-table registry

`ami/storage/registry.py::SOURCE_TABLE_REGISTRY` — an allowlist dict, exactly `{mark_prices, agg_trades, book_ticker}`, each a `SourceTableSpec` with the exact column list, SQL/archive type mapping, nullable columns, stable ordering field (`id` for all three), expected indexes, gap-ledger/repair relationships, and research dependencies, all sourced directly from the live `sqlite_master` schema dumps taken this batch (not guessed). `get_table_spec()` raises `UnknownTableError` for anything else — the CLI's `--table` argument is `choices=allowlisted_tables()`, so an arbitrary table name is rejected by `argparse` itself before any code runs.

## Phase 4 — Closed-month partition model

`ami/storage/partition.py::PartitionIdentity` (frozen dataclass) + `build_partition_identity()`. UTC-only, half-open `[start, end)`, current/future/partial months all rejected (the current-month and partial-month conditions are proven, by a dedicated test, to be the same underlying condition for calendar months — the implementation and its tests both reflect this explicitly rather than papering over it). The 30-day active-retention-horizon protection is enforced as a fourth independent check. `partition_id`/`archive_relative_path` are deterministic (sha256-derived), reproducing the exact `1777593600000`/`1780272000000` millisecond boundaries from the accepted dry-run.

## Phase 5 — Partition planner

`ami/storage/partition.py::plan_partition()` — read-only, bounded, index-backed (`COUNT`/`MAX` on indexed columns, never a full scan). Returns an `ArchivePlan` with one of 7 states (`ARCHIVE_PLAN_ELIGIBLE`/`_BLOCKED_BY_SCHEMA`/`_BY_INDEX`/`_BY_SOURCE_GAP`/`_BY_REPAIR`/`_BY_RESOURCE_LIMIT`/`_BY_UNKNOWN_SOURCE_SEMANTICS`). Against the real database, `plan_partition(mark_prices, ETHUSDT, 2026-05)` returns exactly the accepted values: 260,657 rows, watermark 13,265,132, 3 unresolved gaps (disclosed, does not block rehearsal), `archive_rehearsal_eligible=True`, `purge_eligible=False` (hardcoded, this batch never sets it `True`).

## Phase 6 — Read-only source access

`ami/storage/source_access.py::open_read_only()` — `mode=ro` URI + `PRAGMA query_only=ON` (first layer) + SQLite authorizer denying every write-capable action and writable PRAGMA (second, independent layer). 14 focused tests prove `INSERT`/`UPDATE`/`DELETE`/`CREATE TABLE`/`DROP TABLE`/`ALTER TABLE`/`REINDEX`/`ATTACH`/writable-`PRAGMA` are all rejected — against disposable fixtures, never the live database. `assert_read_only_session_clean()` raises if any denial occurred during a session that should have been clean.

## Phase 7-8 — Bounded exporter and archive schema contract

`ami/storage/archive.py::export_partition()` — fetches via `fetch_partition_rows()` (bounded, ordered, watermark-filtered), builds a typed pyarrow schema per `SourceTableSpec.archive_types`, writes `.partial` (ZSTD), validates 6 invariants (row-count, duplicates, symbol-only, timestamp-range, watermark-ceiling, output-size-cap), and only then atomically `os.replace`s to the final name. `ProductionPathRejected` is raised before any write if `output_root` is not beneath an approved disposable root — proven by a dedicated test.

## Phase 9 — Manifest model

`ami/storage/archive.py::build_manifest()` — **36 fields**, matching the accepted dry-run's authoritative field count (not the readiness batch's earlier 29-field draft; the extra 7 fields — `venue`, `market_segment`, `quote_currency`, `partition_id`, `table_producer`, `primary_key`, `preserved_columns`/`nullable_columns` — were added during the dry-run's own disclosure discipline and are preserved here, not silently dropped). `production_status`/`purge_authorization` are hardcoded, not parameters — no caller can construct a manifest claiming production status.

## Phase 10 — Verifier

`ami/storage/verifier.py` — four independent layers (`verify_structural`, `verify_accounting`, `verify_scientific_parity`, `verify_manifest`, plus `verify_checksum`), each returning exactly one of 9 `VerificationState` values. `verify_full()` short-circuits on the first failure. `FAILED_STATES = VERIFICATION_STATES - {VERIFIED_DISPOSABLE}` — a structural guarantee (tested) that no failed state can ever equal the verified one.

## Phase 11 — Disposable archive catalog

`ami/storage/catalog.py::DisposableArchiveCatalog` — operates only beneath one explicit disposable root, rejects path escapes and production-root paths, rejects unverified registrations, rejects conflicting content hashes under the same `partition_id` (fail-closed — a repaired archive must call `register_new_version()` explicitly, which preserves the prior entry and history untouched). Idempotent re-registration of identical content is a no-op, not an error.

## Phase 12-13 — Direct reader and minimal restorer

`ami/storage/reader.py::read_partition()` requires a matching manifest (symbol/venue/segment) and a checksum match before ever opening the Parquet file; `ami/storage/restorer.py::restore_slice()` writes only beneath `.runtime_temp`/`.pytest_temp`, refuses to overwrite a non-empty destination, and independently re-verifies the restored content's scientific hash against the manifest before returning success. `cleanup_restored_slice()` re-validates the path before deleting anything.

## Phase 14-15 — Job state and corruption detection

`ami/storage/job_state.py::ArchiveJob` — a 7-state machine (`PLANNED`→`EXPORTING_PARTIAL`→`EXPORTED_UNVERIFIED`→`VERIFYING`→`VERIFIED_DISPOSABLE`, plus `FAILED`/`ABANDONED_PARTIAL`) with an explicit legal-transition table; `VERIFIED_DISPOSABLE` has **zero** outgoing transitions (structurally proven: any attempted transition out of it raises `IllegalJobTransitionError`, so a verified archive cannot be silently overwritten by code, not merely by convention). Corruption detection is the verifier's `verify_checksum`/`verify_manifest` layers, exercised against tampered fixtures in tests.

## Phase 16 — CLI

`ami/storage/cli.py`, following the repository's `tools/*.py` convention exactly (`parse_args(argv) -> Namespace`, `main(argv=None) -> int`, `sys.exit(main())`). Six commands: `policy-status`, `plan`, `disposable-dry-run`, `verify`, `read`, `restore-slice`. **`FORBIDDEN_COMMANDS = ("purge", "delete", "vacuum", "schedule", "activate-production", "stop-collector", "restart-collector")`** — a dedicated test confirms none of these strings appear among the parser's actual registered subcommands (not merely that they're undocumented — they do not exist as invokable code paths at all).

## Phase 17 — Resource guards

`StoragePolicy.max_disposable_source_bytes_estimate` (2GB default) / `max_disposable_output_bytes` (4GB default), enforced in `plan_partition()` (returns `PLAN_BLOCKED_BY_RESOURCE_LIMIT` before any export begins) and again in `export_partition()`'s post-write size check (aborts before publication if exceeded). No hidden "force" parameter exists anywhere in the exporter's signature.

## Phase 18 — Research-dependency representation

Every `SourceTableSpec.research_dependencies` tuple is non-empty for the three tables (e.g. `mark_prices` → `FUNDING_LEVEL_VELOCITY`, `FAM_SPOT_PERP_BASIS_REVERSAL`; `book_ticker` → `FAM_BOOK_SPREAD_DYNAMICS_LONG`). `ArchivePlan.purge_eligible` and `production_activation_eligible` are hardcoded `False` in this batch's planner — there is no code path that could set either `True`, so "unknown dependency" is structurally equivalent to "blocked" (nothing is ever permitted by omission).

## Phase 19 — Health/readiness reporting

`ami/storage/health.py::build_health_report()` — pure aggregation over already-collected inputs (job list, policy version, tooling versions, drive/DB sizes). `production_activation`/`scheduler`/`purge`/`vacuum` fields are hardcoded `"DISABLED"` string literals in the dataclass default, not computed from any live state — not wired to any automation.

## Phase 20 — Accepted rehearsal reproduction (live database)

```
$ python -m ami.storage.cli plan --table mark_prices --symbol ETHUSDT --utc-year 2026 --utc-month 5
{
  "plan_state": "ARCHIVE_PLAN_ELIGIBLE", "estimated_row_count": 260657,
  "unresolved_gap_count": 3, "archive_rehearsal_eligible": true, "purge_eligible": false
}

$ python -m ami.storage.cli disposable-dry-run --table mark_prices --symbol ETHUSDT \
    --utc-year 2026 --utc-month 5 --output-root .runtime_temp/storage_rotation_bounded_implementation_v1
{
  "status": "EXPORTED", "row_count": 260657,
  "scientific_content_hash": "228c5705bc1461471102b5bbc1e81b781ddfe7cb5a853b484a7cbb5961860999",
  "parquet_sha256": "6f91914400dcbe84b662c9260a24f9e5eb7f56b2d9db34adfde55a53af8e900f"
}
```

**Both hashes are byte-for-byte identical to the accepted disposable dry-run (commit `6fbe0571`).** `verify`, `read`, and `restore-slice` were then run end-to-end against this output (manually, via the CLI) and all succeeded: `verify` → `VERIFIED_DISPOSABLE`; `read` → 260,657 rows; `restore-slice` → 260,657 rows, scientific hash matching. No difference to explain at any level (schema/scientific-content/row-accounting/writer-byte) — full equality was achieved without being forced.

## Phase 21 — Fixture coverage for all three tables

`tests/test_ami_storage_acceptance.py` — `mark_prices` reproduced live; `agg_trades` and `book_ticker` covered by minimal synthetic fixtures exercising: row exactly at partition start (included), row exactly at partition end (excluded — half-open boundary), rows above watermark (excluded), wrong symbol (excluded), active-horizon blocker, current-month blocker, nullable-column preservation (`book_ticker.bid_depth_usd`), no-duplicate-IDs, boolean/int64 non-narrowing (`agg_trades.is_buyer_maker`), scientific-content-hash stability, and a genuinely missing source column raising a plain `sqlite3.OperationalError` rather than being silently skipped.

## Phase 22 — VACUUM boundary (documented, not implemented)

Routine archive creation and routine purge do not require or trigger `VACUUM` — deleted SQLite pages are first reused via the normal freelist mechanism (`auto_vacuum=NONE` on `microstructure.db`, confirmed in the readiness batch: `freelist_count=0` currently, meaning no deletions have occurred yet). Physical file shrinkage requires a **separate, later, explicitly-authorized** maintenance gate:

**`BATCH-STORAGE-SQLITE-PHYSICAL-RECLAMATION-MAINTENANCE-V1`** — not opened by this batch, not recommended to open now. It becomes relevant only after production archival and an authorized purge produce substantial reclaimable free pages, and requires its own backup/restore proof, measured temporary-disk-space requirement, governed collector-downtime procedure, and a guarantee that `VACUUM` never runs against an actively-written production database without that downtime contract.

---

## Verdict

**`STORAGE_ROTATION_RETENTION_BOUNDED_IMPLEMENTATION_V1_COMPLETE`**

**`STORAGE_ROTATION_RETENTION_IMPLEMENTATION_READY_FOR_PRODUCTION_ACTIVATION_REHEARSAL`**

This disposition does **not** authorize production activation. Every module in `ami/storage/` remains disposable-output-only; production activation, scheduling, and purge exist nowhere in this codebase — there is no flag, override, or hidden parameter that enables them.

## Blockers / residual risks

1. Only `mark_prices` was exercised against the live database this batch; `agg_trades`/`book_ticker` are covered by fixtures only (matching the prior dry-run batch's own disclosed scope limitation — larger tables reserved for a future, larger-scale rehearsal).
2. The catalog is in-memory only (no persistence layer) — a future production-activation batch would need to decide the catalog's actual storage medium.
3. No production Parquet reader integration exists yet in any research-facing module (`ami/research/feature_gateway.py` and friends still read SQLite directly) — building that integration is out of this batch's scope.

## Storage report

| Item | Value |
|---|---|
| Disposable files created this batch | 1 retained (`manifest.json`, 1,911 bytes, under `.runtime_temp/storage_rotation_bounded_implementation_v1/`); 2 bulk binaries created then deleted after hash-recording (Parquet 2,476,313 bytes, restored SQLite slice 12,902,400 bytes) |
| Disposable files deleted | 2 (both bulk binaries, hashes recorded before deletion: Parquet `6f919144…`, SQLite slice `27ef1205…`) |
| Disposable files retained | `manifest.json` only |
| Peak disposable disk usage | ~15.4 MB (before cleanup), 1.9 KB (after) |
| Final `.runtime_temp/storage_rotation_bounded_implementation_v1/` contents | `manifest.json` only |
| Final `.pytest_temp/` contents | unchanged (empty; all test fixtures used pytest's own `tmp_path`, auto-cleaned) |
| Full database copy created | **confirmed NOT created** |
| Production archive created | **confirmed NOT created** |
| Source row modified or deleted | **confirmed NOT occurred** |
| Purge available | **confirmed NOT available** (no code path exists) |
| Scheduler available | **confirmed NOT available** (no code path exists) |
| VACUUM available | **confirmed NOT available** (no code path exists) |

## Next controlled gate

**`BATCH-STORAGE-ROTATION-RETENTION-PRODUCTION-ACTIVATION-REHEARSAL-V1`** (recommended, not begun). Must still prohibit source-row deletion, purge, scheduler activation, collector changes, and VACUUM; must prove production path/permissions, atomic publication, catalog registration, re-verification after publication, direct research read, restore proof, idempotency, and interrupted-publication recovery — with zero source mutation throughout.

Not begun by this batch.
