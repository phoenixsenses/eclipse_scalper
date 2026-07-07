# STORAGE_ROTATION_RETENTION_DISPOSABLE_DRY_RUN_V1

**Gate:** BATCH-STORAGE-ROTATION-RETENTION-DISPOSABLE-DRY-RUN-V1
**Nature:** Bounded, non-production, non-destructive archive rehearsal. No live-row mutation, no production archive, no purge.
**Date:** 2026-07-07 · **Author:** Sonnet 5

---

## Headline result

**Full end-to-end success.** One real closed partition (`mark_prices`/`ETHUSDT`/2026-05, 260,657 rows) was selected, exported to disposable ZSTD-compressed Parquet, validated, published, manifested, read back with exact parity, restored into a minimal SQLite slice with exact parity, rebuilt a second time with **byte-identical** output (not merely scientific-content-identical — the Parquet file itself hashed identically across both runs), survived a simulated interruption/restart, correctly detected both Parquet-byte and manifest-field corruption, and proved the watermark/late-arrival contract holds. **Zero writes were issued against the live source database** at any point (confirmed by a SQLite authorizer that logged zero denied attempts — because zero write attempts were ever made).

---

## Phase 1 — Tooling verification

| Component | Value |
|---|---|
| Python | 3.13.9, `C:\Users\Windows 11\AppData\Local\Programs\Python\Python313\python.exe` |
| pyarrow | 21.0.0 |
| pyarrow.parquet | available |
| ZSTD codec | `pa.Codec.is_available('zstd')` → **True**; `pa.Codec('zstd')` constructs successfully |
| pandas | 2.3.3 (not required for the export path, present) |
| SQLite | 3.50.4 (via `sqlite3` stdlib module) |
| OS / filesystem | Windows, NTFS |
| Packages installed | **0** — everything used was already present in the environment |

**No blocker.** Proceeds past `STORAGE_ROTATION_RETENTION_DISPOSABLE_DRY_RUN_V1_BLOCKED_BY_TOOLING`.

## Phase 2 — Partition selection

| Candidate (preference order) | Outcome |
|---|---|
| 1. `mark_prices` | **SELECTED** — smallest of the three archive-eligible tables; exact timestamp/ID/symbol semantics confirmed sufficient |
| 2. `agg_trades` | Rejected — larger table, reserved for a later rehearsal once the `mark_prices` contract is proven |
| 3. `book_ticker` | Rejected — largest table (~4.79B rows), reserved for a later, larger-scale rehearsal |

**Frozen selection:**

| Field | Value |
|---|---|
| Source table | `mark_prices` |
| Symbol | `ETHUSDT` |
| Venue | `BINANCE_USDM_PERP` |
| Market segment | `PERPETUAL_FUTURES` |
| UTC partition | `[2026-05-01T00:00:00Z, 2026-06-01T00:00:00Z)` — fully closed, entirely before the 30-day active horizon (2026-06-07) and not the current UTC month (July) |
| Boundary type | half-open, UTC calendar month |
| Selection method | bounded, indexed `COUNT/MIN/MAX` query (0.038s), never joined to research outcomes, never chosen by compressibility or performance |

## Phase 3 — Source snapshot and watermark

| Field | Value |
|---|---|
| Source database | `data/microstructure.db`, opened `mode=ro` |
| Read-transaction start | captured at query time (recorded in the disposable results, UTC ISO timestamp) |
| Watermark field | `id` (INTEGER PRIMARY KEY AUTOINCREMENT) |
| Watermark value | **13,265,132** (`MAX(id)` for `symbol='ETHUSDT' AND ts_ms` in the May window, at snapshot time) |
| Eligibility rule | `symbol='ETHUSDT' AND ts_ms>=1777593600000 AND ts_ms<1780272000000 AND id<=13265132`, ordered `id ASC` |
| `microstructure.db` size at snapshot start | 759,124,799,488 bytes |
| WAL size at snapshot start | 8,157,632 bytes |
| Late/gap findings for this window | `gaps` table shows several `mark_prices` gap entries overlapping May 2026 (mostly `resolved_bool=1`); **3 unresolved, open-ended gap rows** (`id=778,781,796`) exist, disclosed honestly, not hidden — this is normal for a raw high-frequency stream and does not block a disposable rehearsal (only a future purge) |

## Phase 4 — Archive schema contract (frozen, as applied)

| Source column | SQL type | Parquet type | Nullable |
|---|---|---|---|
| `id` | INTEGER | int64 | No |
| `ts_ms` | INTEGER | int64 | No |
| `symbol` | TEXT | string | No |
| `mark_price` | REAL | double | No |
| `funding_rate` | REAL | double | Yes |
| `next_funding_time_ms` | INTEGER | int64 | Yes |

All 6 source columns preserved, none omitted, none narrowed, no float conversion of the integer ID/timestamp columns, no timezone reinterpretation (UTC epoch-ms `int64` throughout).

## Phase 5-6 — Disposable output boundary and export

All outputs written beneath `D:\eclipse_scalper\.runtime_temp\storage_rotation_dry_run_v1\` only (confirmed — no file was created under any production archive location, `data/ami/backups`, the repository root, or OS temp).

| Metric | Value |
|---|---|
| Source rows read | 260,657 |
| Export duration | 0.467s |
| `.parquet.partial` created | yes, ZSTD compression |
| Row groups | single (small enough not to require multi-group streaming) |
| Min/max source ID | 12,483,110 / 13,265,132 |
| Min/max receipt timestamp (`ts_ms`) | 1,777,593,604,001 / 1,780,271,978,008 |
| Null counts (all 6 columns) | 0 for `id`/`ts_ms`/`symbol`/`mark_price`; 0 observed for `funding_rate`/`next_funding_time_ms` in this specific partition (both nullable, neither actually null here) |
| Duplicate primary keys | **0** |

## Phase 7 — Pre-publication validation

| Check | Result |
|---|---|
| Row count matches source | ✓ (260,657 = 260,657) |
| Scientific-content hash matches source | ✓ |
| Compression is ZSTD | ✓ |
| Min/max ID match | ✓ |
| Symbol-only (`ETHUSDT`) | ✓ |
| All timestamps inside frozen month | ✓ |
| All source IDs ≤ watermark | ✓ |
| Duplicate count | 0 |
| Column count matches schema | ✓ |

**All checks passed → atomic publication** (`os.replace` from `.parquet.partial` to `.parquet`, a single filesystem rename, not a copy).

## Phase 8 — Manifest

Published as `manifest.json.partial` → validated → atomically renamed to `manifest.json`. **36 fields** recorded (superset of the accepted 29-field contract — additional fields for venue/market-segment/gap-status/repair-status disclosure). Key fields:

| Field | Value |
|---|---|
| `production_status` | `DISPOSABLE_NOT_PRODUCTION` |
| `purge_authorization` | `PROHIBITED` |
| `source_watermark_value` | 13,265,132 |
| `ordered_scientific_content_hash` | `228c5705bc1461471102b5bbc1e81b781ddfe7cb5a853b484a7cbb5961860999` |
| `parquet_sha256` | `6f91914400dcbe84b662c9260a24f9e5eb7f56b2d9db34adfde55a53af8e900f` |
| `verification_status` | `PASS` |
| `source_gap_status` | `GAPS_PRESENT_MOSTLY_RESOLVED_SEE_REPORT` |
| `repair_status` | `NOT_APPLICABLE_MARK_PRICES_HAS_NO_REPAIR_LAYER` |

## Phase 9 — Source-to-Parquet direct-read parity

A **fresh** re-query of the source (same watermark, issued after publication) was compared against the published Parquet:

| Metric | Value |
|---|---|
| Fresh source hash | `228c5705bc1461471102b5bbc1e81b781ddfe7cb5a853b484a7cbb5961860999` |
| Published Parquet hash | `228c5705bc1461471102b5bbc1e81b781ddfe7cb5a853b484a7cbb5961860999` |
| **Match** | **✓ exact** |
| Row-by-row mismatch count | **0** |
| Row count match | ✓ |

## Phase 10 — Minimal SQLite-slice restore

Restored into a new, minimal SQLite database under the dry-run directory (`mark_prices_restored` table only — no outcome tables, no canonical tables, no governance tables):

| Metric | Value |
|---|---|
| Restored row count | 260,657 (= source) |
| Restored scientific-content hash | `228c5705bc1461471102b5bbc1e81b781ddfe7cb5a853b484a7cbb5961860999` |
| Matches source | ✓ |
| Matches Parquet | ✓ |

## Phase 11 — Idempotent rebuild (run A vs. run B)

The complete export→publish→manifest pipeline was re-run from the **same frozen snapshot identity and watermark**:

| Metric | Run A | Run B | Match |
|---|---|---|---|
| Source scientific hash | `228c5705…` | `228c5705…` | ✓ |
| Parquet scientific hash | `228c5705…` | `228c5705…` | ✓ |
| **Parquet file SHA-256** | `6f919144…` | `6f919144…` | **✓ byte-identical** |
| Parquet size | 2,476,313 bytes | 2,476,313 bytes | ✓ |
| Row count | 260,657 | 260,657 | ✓ |

**Full byte-level determinism achieved** — no writer-metadata variance was observed between the two runs (the operator's own contract anticipated a possible metadata-only difference; none occurred, disclosed as a clean positive result, not assumed).

## Phase 12 — Interruption and restart safety

A simulated interruption (export halted after the first half of rows, `.partial` file never renamed) was tested in an isolated directory:

| Check | Result |
|---|---|
| `.partial` file never received the final `.parquet` name | ✓ |
| Interrupted state was never published (no `mark_prices_ETHUSDT_2026-05.parquet` existed at that point) | ✓ |
| Restart discarded the abandoned partial | ✓ |
| Fresh rebuild from the same frozen watermark matches a clean build exactly | ✓ (hash `228c5705…`) |

## Phase 13 — Corruption detection

**Parquet corruption:** a disposable copy of the published Parquet had one byte flipped at offset 200.

| Check | Result |
|---|---|
| Original archive untouched | ✓ (hash unchanged) |
| Corrupted file's SHA-256 differs from the original | ✓ |
| Corrupted file remained technically readable by pyarrow, but its row content hash differed from the source | ✓ (content mismatch detected) |
| Would be rejected before reaching `VERIFIED` status | ✓ |

**Manifest corruption:** a disposable copy of the manifest had its `ordered_scientific_content_hash` field tampered.

| Check | Result |
|---|---|
| Tampered field detected as differing from the real manifest | ✓ |
| Would fail verification | ✓ |

Both corruption-test copies were deleted immediately after the evidence above was recorded (not retained).

## Phase 14 — Late-arrival and watermark proof

| Check | Value |
|---|---|
| Captured watermark ID | 13,265,132 |
| Current `MAX(id)` for the same partition (re-queried at the end of the rehearsal) | 13,265,132 (**unchanged**) |
| Rows above the watermark now | 0 |
| Rows above the watermark present in the archive | 0 (by construction — the export query itself filters `id<=watermark`) |

Since the selected month (May 2026) is over two months in the past, no new rows arrived during this rehearsal — consistent with expectations. The frozen policy (any late row would require a new manifest version; a published archive is never mutated in place) was not exercised end-to-end (no late row existed to trigger it), but the exclusion mechanism itself (the `id<=watermark` filter) was proven structurally correct by Phase 7's `all_ids_le_watermark` check.

## Phase 15 — Research-dependency proof

Per the accepted readiness batch (`f65545ee`), `mark_prices` is `RAW_HIGH_FREQUENCY_ARCHIVE_ELIGIBLE`, `CONDITIONAL` purge-eligibility (funding-adjacent research surface — `funding_rates` itself is dead, making `mark_prices` the last surviving raw funding-level source). This dry-run:

- Did **not** delete any source row.
- Did **not** mark any partition purge-safe (the manifest's `purge_authorization` field is hardcoded `PROHIBITED`).
- Did **not** touch `FAM_BOOK_SPREAD_DYNAMICS` LONG's `book_ticker` dependency (a different table entirely, not selected this batch).
- Did **not** alter collector continuity, active 30-day availability, or historical research access.

A disposable archive rehearsal is explicitly permitted despite the purge block (Phase 15's own instruction) — this batch exercised exactly that distinction.

## Phase 16 — Live-source immutability

| Metric | Value |
|---|---|
| SQLite authorizer denied write attempts | **0** (zero write attempts were ever issued — the exporter is read-only by construction, not merely by authorizer interception) |
| `microstructure.db` size at start / end | 759,124,799,488 / 759,124,799,488 bytes (**unchanged** — the ~10-second rehearsal window happened not to coincide with a collector page-flush growing the file; WAL activity below shows collectors were still live) |
| WAL size at start / end | 8,157,632 / 3,015,872 bytes (WAL **shrank** — a live collector's own periodic checkpoint, not caused by this batch, which never issued a checkpoint command) |
| `mark_prices` global `MAX(id)` at watermark capture (May-2026-scoped) vs. at rehearsal end (unscoped, all months) | 13,265,132 → 21,101,480 — the large jump is expected and correct: the first figure is scoped to the May 2026 partition, the second is the table's true current global maximum across all months through today; it does **not** indicate any row was added to the May 2026 partition itself (proven separately by Phase 14's unchanged partition-scoped watermark) |
| Dry-run process write count | **0** |

## Phase 17 — Performance and resource report

| Metric | Value |
|---|---|
| Source rows | 260,657 |
| Parquet file size | 2,476,313 bytes (~2.36 MiB) |
| Compression ratio | ~5.05x |
| Total elapsed time | 9.91 seconds |
| Peak disposable disk usage (all runs combined, before cleanup) | 20,333,167 bytes (~19.4 MiB) |
| Within the 4GB total dry-run cap | ✓ (0.5% of cap) |
| Within the 2GB hard per-partition estimate | ✓ (well under 1% of cap) |
| Full-table scans outside the bounded partition | 0 |
| Full database copies | 0 |
| OS-temp database copies | 0 |
| Collector stoppage | 0 |

---

## Verdict

**`STORAGE_ROTATION_RETENTION_DISPOSABLE_DRY_RUN_V1_COMPLETE`**

**`STORAGE_ROTATION_RETENTION_ARCHIVE_TOOLING_READY_FOR_BOUNDED_IMPLEMENTATION`**

This disposition does **not** authorize production archival, row deletion, purge, scheduler activation, or collector changes — it confirms the accepted archive contract (Parquet+ZSTD, closed-UTC-month partitioning, watermark-based export, atomic publish, manifest verification, direct-read and restore parity, corruption detection, idempotent determinism) is implementable and was proven end-to-end on real data without touching the live source.

## Blockers / residual risks

1. Only one table (`mark_prices`) and one partition were rehearsed — `agg_trades` and `book_ticker` (larger, higher row-count tables) remain unrehearsed; their larger scale may surface different performance/memory characteristics not observed here.
2. The late-arrival exclusion mechanism was proven structurally (the `id<=watermark` filter, verified via `all_ids_le_watermark`) but not exercised against an actual late-arriving row, since none existed for a 2+-month-old closed partition.
3. Disposable evidence retention: bulk binary outputs (Parquet files, restored SQLite slice, corruption-test copies) were deleted after their hashes were recorded in `results.json`; only the small JSON summary and the driver script remain in `.runtime_temp` (not committed to git).

## Storage report

| Item | Value |
|---|---|
| Peak disposable disk usage | 20,333,167 bytes (~19.4 MiB), all beneath `.runtime_temp/storage_rotation_dry_run_v1/` |
| Files created (this batch) | 10 (2 Parquet copies across run A/B, 1 manifest, 1 restored SQLite slice, 1 interrupted-then-discarded partial, 1 interrupt-recovery Parquet, 2 corruption-test copies, plus the driver script and results.json) |
| Files deleted (this batch) | 3 immediately (the abandoned interruption partial, both corruption-test copies) + all bulk binaries (run_a/run_b/run_interrupt/run_corrupt directories) deleted after hash-recording, post-rehearsal |
| Files retained | `driver.py` (19,810 bytes) + `results.json` (6,230 bytes) — both small, hashed, bounded, contain no outcome data, contain only the selected raw partition's metadata/hashes |
| Final `.runtime_temp/storage_rotation_dry_run_v1/` contents | `driver.py`, `results.json` only (28 KB total) |
| Final `.pytest_temp/` contents | unchanged (empty) |
| No full database copy created | **confirmed** — every source read was a bounded, indexed, watermark-limited query against a `mode=ro` connection |
| No production archive created | **confirmed** — all outputs disposable, under `.runtime_temp` only |
| No source row deleted or modified | **confirmed** — see Phase 16 |

## Next controlled gate

**`BATCH-STORAGE-ROTATION-RETENTION-BOUNDED-IMPLEMENTATION-V1`** (recommended, not begun). That future gate may build a production-quality exporter/verifier/manifest-catalog/reader/restorer, but must still prohibit source-row purge, automatic scheduling, collector changes, VACUUM, and production activation — a further, separate activation gate would be required before any real archival, and yet another purge-authorization gate before any live-row deletion.

Not begun by this batch.
