# S34 CVD — PROPOSED SCHEMA 11 → 12 CANONICAL MIGRATION (PROPOSAL ONLY, 2026-07-05)

**Status: PROPOSAL. NOT EXECUTED. NO APPROVAL GRANTED OR IMPLIED.**
Produced by `BATCH-CVD-REPAIR-REHEARSAL-AND-QUALITY-CONTRACT-V1`. The real
`data/ami/canonical.sqlite` was NOT written by this batch (schema_version
remains **11**); every structure below was exercised only in the disposable
rehearsal database `data/ami/cvd_rehearsal_disposable_20260705/cvd_rehearsal_disposable.sqlite`.

Execution would follow the exact `birth_truncated_geometry_canonical_migration.py`
precedent: DDL folded **verbatim** into `ami/warehouse/schema.py` as
`_SCHEMA_PHASE_CVD`, applied by a new controlled entry point
`ami/cvd/cvd_canonical_migration.py`, only after separate operator approval.

---

## 1. Proposed tables (exact names + DDL)

The DDL below is byte-for-byte the DDL already exercised in the disposable
rehearsal (`ami/cvd/windowed_taker_flow.py::_SCHEMA`,
`ami/cvd/cvd_source_quality_contract_v1.py::_SCHEMA`,
`ami/cvd/aggtrades_repair_rehearsal.py::_SCHEMA` — the staging table is
renamed for canonical use, see 1.1).

### 1.1 `ami_agg_trades_repaired` — immutable repaired trade rows

Rename of the rehearsal's `ami_agg_trades_repaired_stage` (identical columns).

- **Primary key:** `(symbol, agg_trade_id, retrieval_batch_id)`
- **Foreign keys:** none (source-layer table; references no canonical entity)
- **Uniqueness:** PK; additionally the effective view (§1.5) guarantees at most
  one effective row per `(symbol, agg_trade_id)`
- **CHECK constraints:** `is_buyer_maker IN (0,1)`; `taker_side IN ('BUY','SELL')`;
  `(is_buyer_maker = 0) = (taker_side = 'BUY')`;
  `legacy_match_status IN ('UNMATCHED','MATCHED_1TO1','AMBIGUOUS','CONFLICTING','NOT_ATTEMPTED')`;
  `data_version_id = 'aggtrades-binance-fapi-repair-r1')`
- **Immutable version columns:** `data_version_id` (frozen
  `aggtrades-binance-fapi-repair-r1`), `retrieval_batch_id`
- **Timestamps:** `ts_ms` (Binance trade time `T`), `retrieved_at_ms`
  (local retrieval), `created_ms`
- **Provenance:** `source_provenance` (`GET /fapi/v1/aggTrades`),
  `retrieval_page_index`, `first_trade_id`/`last_trade_id` (Binance `f`/`l`),
  `source_regime_id`, `legacy_match_status`, `legacy_match_fingerprint`
- **Append-only rule:** rows are never UPDATEd/DELETEd. Same
  `(symbol, agg_trade_id, retrieval_batch_id)` re-inserted with different
  content raises `ImmutableRepairRowConflict` (proven by test).
- **Supersession rule:** a corrected retrieval mints a NEW
  `retrieval_batch_id`; the old batch's rows get `superseded_by_batch_id`
  stamped by an append-style UPDATE OF THAT ONE NULLABLE POINTER COLUMN ONLY
  (the sole permitted mutation, mirroring the candle-repair supersession
  precedent), and the effective view excludes superseded rows.
- **Idempotency:** re-running the migration/backfill with identical frozen
  source packages is a NOOP (content-compare before insert).
- **Expected row count (from this rehearsal, if executed with the same
  frozen source package):** 40,934 rows, 8 retrieval spans, ETHUSDT only
  (the 35 signal-window-touched missing minutes span 8 contiguous ranges;
  every span's rerun was byte-identical and verdict `EXACT_RECONSTRUCTED`).

### 1.2 `ami_cvd_repair_batch_ledger` — retrieval-batch ledger

- **Primary key:** `retrieval_batch_id`
- **Foreign keys:** none (ledger of external retrievals)
- **Uniqueness:** PK
- **CHECK:** `exact_reconstruction_verdict IN
  ('EXACT_RECONSTRUCTED','INCOMPLETE','FAILED','PROBE_ONLY')`
- **Immutable version columns:** `data_version_id`
- **Timestamps:** `created_ms`
- **Provenance:** full request accounting — `pagination_method`, `page_count`,
  `row_count`, `first_agg_trade_id`, `last_agg_trade_id`,
  `earliest_trade_ts_ms`, `latest_trade_ts_ms`, `page_overlap_rows`,
  `missing_id_ranges` (JSON), `request_errors` (JSON), `truncation_flag`,
  `content_sha256`, `gap_manifest_sha256`, `duplicate_manifest_sha256`
- **Append-only:** ledger rows are never rewritten; a re-retrieval is a new
  batch id.
- **Expected row count:** 8 (one per repair span; the rehearsal's separate
  determinism-replay batches on 3 sample windows live only in the disposable
  replay databases, never in this ledger).

### 1.3 `ami_cvd_windowed_flow` (+ `ami_cvd_windowed_flow_proxy`, `ami_cvd_bucket_exclusions`) — CVD signal-window feature tables

- **Primary key:** `feature_id` (deterministic
  `CVDF-sha256(signal_id|window_id|feature_definition_version|layer)[:24]`)
- **Foreign keys (canonical form adds):** `signal_id -> ami_signal_lifecycle`,
  `source_event_id -> ami_events`, `independent_cycle_id -> ami_cycles`
  (rehearsal DDL omits FKs because the disposable DB does not contain the
  parent tables; the canonical fold-in adds the three FK clauses — the ONLY
  permitted delta vs the rehearsal DDL, stated here so the byte-for-byte
  equality check is defined as "rehearsal DDL + these three FK lines")
- **Uniqueness:** `UNIQUE (signal_id, window_id, feature_definition_version)`
  on both layer tables; `UNIQUE (signal_id, feature_definition_version)` on
  the exclusion ledger
- **CHECK constraints (exact layer):** window family pinned to the 6 frozen
  ids; `evidence_layer = 'EXACT'`; `window_end_ts_ms = signal_birth_ts`;
  `feature_available_ts_ms = signal_birth_ts`;
  `known_at_classification = 'KNOWN_AT_SAFE'`;
  `source_row_count = legacy_row_count + repair_row_count`;
  `repair_method IN ('NONE','AGGTRADES_REST','AGGTRADES_VISION_ARCHIVE')`;
  `normalized_cvd` bounded to [-1, 1] or NULL
- **CHECK constraints (proxy layer):** `evidence_layer = 'PROXY'`;
  `descriptive_only = 1`; `last_contained_close_ts_ms <= signal_birth_ts`;
  `(contained_candle_count = 0) = (proxy_cvd_qty IS NULL)` — zero-candle
  windows are NULL, never fabricated 0
- **Immutable version columns:** `feature_definition_version`
  (`s34-cvd-windowed-taker-flow-v1-birth-truncated`),
  `raw_interpretation_version` (`aggtrades-taker-side-v1`),
  `quality_contract_version`, `repair_population_version`
- **Timestamps:** `signal_birth_ts`, `window_start_ts_ms`, `window_end_ts_ms`,
  `feature_available_ts_ms`, `created_ms`
- **Provenance:** `source_row_manifest_sha256`, `source_regime_ids` (JSON),
  `repair_method`, `provenance`
- **Append-only / supersession:** rows immutable
  (`ImmutableCvdFeatureConflict` on same-identity different-content);
  a redefinition mints a new `feature_definition_version`, never overwrites.
- **No outcome columns exist or may ever be added under this version** — a
  new version would be required even for that (and would be rejected by the
  research OS's no-outcome-in-feature-layer law anyway).
- **Expected row counts:** exact = {{EXACT_ROW_COUNT}}; proxy =
  {{PROXY_ROW_COUNT}}; bucket exclusions = {{BUCKET_EXCLUSION_COUNT}}
  (accounting identity: exact + exclusions = 324 × 6 = 1,944).

### 1.4 `ami_cvd_window_quality_v1` — field/source-quality ledger

- **Primary key:** `quality_id` (deterministic hash id)
- **Foreign keys (canonical form):** `signal_id -> ami_signal_lifecycle`
- **Uniqueness:** `UNIQUE (signal_id, window_id, quality_contract_version,
  assessment_version)`
- **CHECK:** the five frozen statuses ONLY (`EXACT_RECONSTRUCTABLE`,
  `PROXY_ONLY`, `SOURCE_GAPPED`, `SOURCE_COVERAGE_UNRESOLVED`,
  `UNREPAIRABLE`); `window_end_ts_ms = signal_birth_ts`;
  `feature_available_ts_ms = signal_birth_ts`; `regime_spanning IN (0,1)`;
  `total_row_count = legacy_row_count + repair_row_count`
- **Row-vs-field quality:** per the accepted contract §7, the window IS the
  quality unit (Q1/Q2/Q3 share the identical source window and rows), so
  this window-level ledger IS the field-level ledger — stated explicitly.
- **Append-only:** new opinions append under a NEW `assessment_version`;
  same version + different content raises `ImmutableCvdQualityConflict`
  (proven by test). Statuses never silently upgrade.
- **Expected row count:** {{QUALITY_ROW_COUNT}} per assessment version.

### 1.5 Effective views

```sql
CREATE VIEW IF NOT EXISTS ami_agg_trades_repaired_effective AS
  SELECT * FROM ami_agg_trades_repaired WHERE superseded_by_batch_id IS NULL;

CREATE VIEW IF NOT EXISTS ami_cvd_window_quality_v1_effective AS
  SELECT q.* FROM ami_cvd_window_quality_v1 q
  WHERE q.assessed_at_ms = (
    SELECT MAX(q2.assessed_at_ms) FROM ami_cvd_window_quality_v1 q2
    WHERE q2.signal_id = q.signal_id AND q2.window_id = q.window_id
      AND q2.quality_contract_version = q.quality_contract_version);
```

Consumers use ONLY the effective views for quality state; raw ledgers remain
complete audit history.

## 2. Migration procedure (proposal)

1. **Backup:** `data/ami/backups/canonical_pre_cvd_migration_<ts>.sqlite`
   (copy + sha256 recorded in `MIGRATION_LOG.md`, M-XXXX), exactly the
   geometry-migration procedure.
2. **Disposable restore proof:** restore the backup into a disposable copy,
   verify `schema_version = 11`, table census matches pre-migration, and the
   4 new tables are ABSENT — before the real migration runs (proven pattern
   from the geometry batch; the restore is never applied to the live DB).
3. **Canonical hash verification:** sha256 of `canonical.sqlite` recorded
   before/after; the only permitted deltas are the new tables/views, the
   `schema_versions` row 12, and (by established exception) the
   feature-gateway exposure ledger.
4. **Migration body:** `_SCHEMA_PHASE_CVD` DDL (byte-compare vs rehearsal DDL
   + the three declared FK lines, programmatic equality check as in the
   geometry migration), then frozen-value backfill: repaired rows from the
   frozen source package (content hashes MUST reproduce the rehearsal's
   {{REPAIR_CONTENT_HASH_NOTE}}), feature matrix and quality ledger
   recomputed and byte-compared against the rehearsal's stored content
   hashes (`content_hash_exact` = {{EXACT_CONTENT_HASH}},
   `content_hash_proxy` = {{PROXY_CONTENT_HASH}}, quality content hash =
   {{QUALITY_CONTENT_HASH}}). ANY mismatch = hard stop + rollback.
5. **Idempotency:** rerunning the migration entry point must be
   `NOOP_IDENTICAL` (content-compare writers throughout; proven in the
   disposable DB by this batch's rerun check).
6. **Content-conflict behavior:** any same-identity/different-content write
   raises (`ImmutableRepairRowConflict` / `ImmutableCvdFeatureConflict` /
   `ImmutableCvdQualityConflict`) and aborts the transaction — fail closed,
   no partial state.
7. **Rollback:** `DROP` of exactly the 4 new tables + 2 views + removal of
   the version-12 row, restoring the pre-migration schema fingerprint
   (geometry `rollback()` precedent); backup restore as the outer fallback.
8. **Protected-invariant checks (pre + post):** `ami_events` = 252,
   `ami_signal_lifecycle` = 324, `ami_cycles` = 167,
   `ami_birth_truncated_cascade_geometry` = 220, experiment registry/results
   content hashes unchanged, `PRAGMA foreign_key_check` clean,
   `PRAGMA integrity_check` = ok.
9. **Required regression command (unchanged, frozen):**
   `pytest tests/test_ami_*.py tests/test_buyfade_mutations.py tests/test_buyfade_silexit_mutations.py`
   — must be green at the batch's own post-migration ground truth
   ({{NEW_GROUND_TRUTH}} after this rehearsal batch's test additions).
10. **Expected migration stop conditions:** backup hash mismatch; DDL
    byte-compare failure; any content-hash mismatch vs frozen rehearsal
    values; any immutability-conflict exception; protected-invariant delta;
    regression not green; `schema_version != 11` at start.

## 3. What this proposal does NOT authorize

No live-row mutation of `data/microstructure.db:agg_trades` (repair rows live
ONLY in the new separate table); no collector change; no outcome read; no
experiment registration; no inferential use of any window whose effective
quality status is not `EXACT_RECONSTRUCTABLE`; no exact/proxy pooling
(schema-level CHECKs make a pooled population unrepresentable).

**Approval state: WAIT_FOR_OPERATOR_APPROVAL.**
