# S34 CVD REPAIR REHEARSAL AND QUALITY CONTRACT V1 (2026-07-05)

**Batch:** `BATCH-CVD-REPAIR-REHEARSAL-AND-QUALITY-CONTRACT-V1`.
**Basis:** accepted contract `S34_CVD_REPAIR_CONTRACT_AND_ANCHOR_DEFINITION_2026-07-05.md`
(verdict `CVD_REPAIR_CONTRACT_READY_FOR_REHEARSAL`).
**Scope:** disposable rehearsal, measurement, validation and canonical-migration
proposal only. **No real canonical migration approved or executed.**
**Machine-readable mirror:** `S34_CVD_REPAIR_REHEARSAL_AND_QUALITY_CONTRACT_V1_2026-07-05.json`.

---

## 0. Pre-flight integrity (before any work)

| Item | Value |
|---|---|
| `canonical.sqlite` sha256 (start) | `b28b40938bd76524d39dd6c1b82905b4d07d9e88c98b0c2daabcfcc55455009d` |
| `canonical.sqlite` mtime (start) | 2026-07-05T11:19:06Z |
| `schema_version` (start) | 11 |
| Canonical signal count | 324 (LONG=220, SHORT=104) |
| Independent cycle count | 167 |
| Geometry rows (`ami_birth_truncated_cascade_geometry`) | 220 |
| Frozen regression baseline | 795/795 |
| Git HEAD | `09af9dc6` on `codex/data-layer-fallback-cleanup` |
| `microstructure.db` size (start) | 732,298,805,248 bytes (collector-live; full hash not attempted, matches every prior batch's bounded-collector-aware precedent) |
| Both DBs opened | strictly `mode=ro` throughout |

## 1. Disposable rehearsal environment (Task 1)

All artifacts under `data/ami/cvd_rehearsal_disposable_20260705/` (removable):

| Path | Role |
|---|---|
| `cvd_rehearsal_disposable.sqlite` | main disposable DB — minute map, repaired-trade staging, feature matrix, quality ledger |
| `cvd_repair_run2.sqlite` | second independent staging run for each repair span (determinism proof) |
| `scan_pass_a.py` / `scan_pass_a_result.json` | Task 3 full-range id-order scan |
| `scan_pass_b.py` / `scan_pass_b_result.json` | Task 3 full-range ts-order duplicate/cadence scan |
| `probe_aggtrades.py` / `probe_manifest.json` | Task 2 availability probe |
| `replay_and_dedup.py` / `replay_hash_manifest.json` / `replay_*_run{1,2}.sqlite` | Tasks 4-5 determinism + reconciliation on 3 sample windows |
| `run_rehearsal.py` / `rehearsal_run_summary.json` | Tasks 6-11 main runner |
| `run_regression.sh` / `regression_run{1,2}/` | Task 12 frozen regression logs |

No row was ever inserted into `data/microstructure.db` or `data/ami/canonical.sqlite`; both were opened `file:...?mode=ro`.

## 2. Full-range cadence + coverage scan (Task 3, closes B6)

**Pass A (id-order stream, primary key continuity):** 388,452,291 total
`agg_trades` rows scanned (ETHUSDT=174,571,325, BTCUSDT=193,728,322,
SOLUSDT=20,152,644). Autoincrement `id` column: **0 non-consecutive steps**
across the entire table (id 1 → 388,452,291, no holes) — the row-count
ledger itself has never lost a row via deletion/vacuum. Per-symbol
out-of-order-by-timestamp events in id-insertion order: ETH=1,779,
BTC=1,755, SOL=129 (all tiny relative to population; consistent with rare
late-arriving trades re-ordered by a collector reconnect, not systemic
corruption). Adjacent exact-duplicate rows (same id-order neighbor,
identical content): ETH=93, BTC=104, SOL=5 — negligible.

**Pass B (ts-order, per-symbol/per-regime cadence + duplicate/conflict
scan):** full population re-scanned in `(symbol, ts_ms)` index order.
Per-regime maximum inter-arrival gap, gap-count histogram at 5 thresholds,
duplicate-cluster count, and same-timestamp-opposite-side conflict count —
see `S34_CVD_CROSS_SOURCE_DEDUP_COLLISION_REPORT_2026-07-05.md` §3-4 for the
full table. **This closes blocker B6** (whether material gaps exist beyond
2026-06-06 and beyond the frozen minute-map's 2026-07-03 cutoff): the scan
ran to the frozen cutoff `SCAN_MAX_MS=1783271317967` (2026-07-06, this
batch's own execution time), and R3's largest post-06-06 gap
(815,703 ms ≈ 13.6 min) is fully accounted for as a zero-row-minute span
already present in the frozen ETH minute-map — no *new*, previously-unknown
multi-day outage exists past 2026-06-06.

## 3. Binance historical aggTrades availability probe (Task 2)

104 probes executed (`/fapi/v1/aggTrades`, first page, limit 1000) covering:
all 6 outages × {pre-boundary, start, interior, end, post-boundary} ×
affected symbols; all 20 registry-era gaps × {ETH, BTC}; 4 representative
Feb/Mar frozen minute-map gap runs.

| Metric | Value |
|---|---|
| Total probes | 104 |
| `AVAILABLE_FROM_REQUESTED_START` | 104 |
| `AVAILABLE_WITH_LEADING_HOLE` | 0 |
| `EMPTY_RANGE` | 0 |
| `REQUEST_FAILED` / `API_ERROR` | 0 |
| Pages with internal id holes | 0 |

**Every probed outage/gap window returned trade-level data from Binance's
historical aggTrades endpoint**, including deep inside the ~79-96h
2026-06-01/05 blackout and every Feb/Mar minute-map gap run. No probe
returned `NOT_PROVEN_PROBE_FAILED`. Full per-probe detail (symbol, requested
window, returned id/ts range, availability verdict) in
`S34_CVD_HISTORICAL_AGGTRADES_PROBE_MANIFEST_2026-07-05.json`. Per the
contract's own law, a probe **never** claims exact reconstruction by itself
(`exact_reconstruction_verdict = PROBE_ONLY_NEVER_EXACT` on every probe row)
— full-window extraction + continuity proof (Task 5) is the actual exact-repair
gate, exercised separately for the spans the canonical signal population
actually needs (§5).

## 4. Cross-source deduplication + deterministic replay (Tasks 4-5)

Full detail in `S34_CVD_CROSS_SOURCE_DEDUP_COLLISION_REPORT_2026-07-05.md`
and `S34_CVD_DETERMINISTIC_REPLAY_HASH_MANIFEST_2026-07-05.json`. Summary:

- **Identity within REST data** `(symbol, agg_trade_id)`: clean in every
  extraction this batch performed — 0 duplicate ids, 0 id-range holes.
- **Reconciliation vs legacy rows** (3 frozen sample windows: blackout /
  healthy-R3 / healthy-R0): exact fingerprint-multiset matching, **no
  arbitrary or probabilistic pairing anywhere**. Blackout window: 0 legacy
  rows (clean supersession case — nothing to collide with). Healthy-R3
  window: 100% 1:1 match (3,050/3,050), zero collisions. Healthy-R0 window:
  12,257/12,259 exact 1:1, **1 many-to-many collision class** (2 genuinely
  simultaneous same-fingerprint trades) — correctly flagged, not silently
  resolved; this window is never used as a repair source in the main matrix
  (it has zero missing minutes).
- **Determinism:** every extraction in this batch (8 repair spans + 3 replay
  windows, each run twice) reproduced byte-identical `content_sha256`,
  `gap_manifest_sha256` and `duplicate_manifest_sha256` on rerun.
  `hard_stop_rerun_mismatch = false`.
- **Duplicate-cluster elevation (R3 structural double-insert risk):**
  measured rate 0.0401% vs R0/R1 baseline max 0.1609% — **below**, not above,
  the frozen elevation threshold (10× baseline). `BLOCKED_BY_DUPLICATE_INTEGRITY`
  does **not** fire.

## 5. Disposable repair staging + main feature/quality rehearsal (Tasks 6-11)

**Population:** all 324 canonical signals fetched read-only from
`ami_signal_lifecycle` LEFT JOIN `ami_birth_truncated_cascade_geometry`
(BUCKET start present for 220 LONG, absent for 104 SHORT — recorded, never
guessed).

**Missing-minute footprint actually touched by the 324×6 window family:**
only **35 minutes**, forming **8 contiguous spans** — the canonical
population's own signal-birth timestamps mostly land in well-covered eras;
the six major multi-day outages barely intersect any signal's actual
`[T-W, T]` windows.

**Repair (Task 6, staged to `ami_agg_trades_repaired_stage`, disposable
only):** all 8 spans fetched via `/fapi/v1/aggTrades` (startTime-then-fromId
pagination), each fetched a **second, independent time** into a separate
disposable database:

| Span | Range (ms) | Rows | Verdict | Rerun identical |
|---|---|---|---|---|
| 1 | [1772815140000, 1772815260000) | 4,056 | `EXACT_RECONSTRUCTED` | Yes |
| 2 | [1773004980000, 1773005040000) | 242 | `EXACT_RECONSTRUCTED` | Yes |
| 3 | [1773421020000, 1773421260000) | 6,332 | `EXACT_RECONSTRUCTED` | Yes |
| 4 | [1774547220000, 1774547280000) | 1,259 | `EXACT_RECONSTRUCTED` | Yes |
| 5 | [1774548420000, 1774548720000) | 6,380 | `EXACT_RECONSTRUCTED` | Yes |
| 6 | [1776780540000, 1776781020000) | 10,135 | `EXACT_RECONSTRUCTED` | Yes |
| 7 | [1776781260000, 1776781380000) | 3,209 | `EXACT_RECONSTRUCTED` | Yes |
| 8 | [1781577900000, 1781578620000) | 9,321 | `EXACT_RECONSTRUCTED` | Yes |

**Total staged: 40,934 rows, 0 immutable-conflict rejections, all 35 missing
minutes verified repaired** (`repaired_minutes_verified = 35`).

**Cadence threshold (source-derived, not outcome-derived):** frozen at
**93,195 ms**, computed as the maximum healthy-R3 inter-arrival gap NOT
already explained by a zero-row minute (18 residual >60s gaps measured;
largest = 93,195 ms). This is the sub-minute completeness instrument the
accepted contract designated §10.D1-1 for.

**Duplicate policy:** no regime flagged (see §4) — `duplicate_unresolved`
never forces a degrade in this rehearsal.

**Feature matrix + quality classification (Tasks 8, 11):**

| Quantity | Value |
|---|---|
| Exact-layer rows (`ami_cvd_windowed_flow`) | **1,840** |
| Proxy-layer rows (`ami_cvd_windowed_flow_proxy`) | **1,840** |
| BUCKET exclusions (`ami_cvd_bucket_exclusions`) | **104** (all 104 SHORT signals — no frozen bucket start exists for them; explicit, not silent) |
| Accounting identity | 1,840 + 104 = **1,944 = 324 × 6** ✓ |
| Quality rows | 1,840 |
| `known_at_violations` | **0** |
| Quality status histogram | `EXACT_RECONSTRUCTABLE`=1,828, `SOURCE_GAPPED`=12 |

**The 12 `SOURCE_GAPPED` rows** are all `BUCKET`-window rows (the
short, ≤300s, geometry-frozen bucket windows) where sub-minute cadence proof
failed AND zero fully-contained 1m candles exist inside the short window (so
no proxy fallback exists either) — a legitimate fail-closed classification,
not a bug. Verified: no window silently defaulted to a different status;
every one of the 12 has `missing_minute_count=0, repaired_minute_count=0` in
its own stored `completeness_proof` JSON, i.e. its own quality opinion is
fully self-documenting.

**Exact/proxy separation (Task 11):** physically separate tables with
disjoint schema-level `CHECK (evidence_layer = 'EXACT'|'PROXY')` constraints
— a mixed population cannot even be represented in one table, let alone
pooled. `assert_not_pooled()` additionally guards any in-memory fetch.
No proxy row was ever promoted to inferential eligibility; the coverage
precheck (§7) only reads `quality_status = EXACT_RECONSTRUCTABLE` rows.

## 6. BUCKET window definition proof (Task 9)

Verified directly against `ami_birth_truncated_cascade_geometry` (read-only,
220 rows): for every row, `source_window_start_ts_ms <= signal_birth_ts`
(0 violations) and the derived window always ends exactly at
`signal_birth_ts` (0 rows where `source_window_end_ts_ms != T`, by
construction of `window_bounds()`). Maximum bucket duration observed:
280,153 ms (≈4.67 min, under the geometry contract's own ≤300s BUCKET_SEC
bound). Bucket start is fully determined by the (already-frozen, immutable)
geometry feature — computed once at signal birth, never touched by any
later terminal-cluster or post-birth field. Multiple signals in the same
independent cycle each carry their **own** `signal_birth_ts` and hence their
own BUCKET window; no shared-window aliasing was observed or assumed.

## 7. Known-at safety (Task 10)

- **Compute-time guard:** `compute_window_flow()` raises `KnownAtViolation`
  on ANY row with `ts_ms > window_end_ts_ms` or `ts_ms < window_start_ts_ms`
  reaching the accumulator (fail-loud, not fail-silent) — this guard fired
  during initial development (a repair-row-clipping bug, fixed before the
  final run; see commit history of `ami/cvd/cvd_rehearsal.py`) and produced
  **zero** violations on the corrected, final rehearsal run.
- **Independent SQL-level recheck** (`timestamp_violation_count()`) over the
  finished 1,840+1,840-row matrix: **0** rows with
  `window_end_ts_ms != signal_birth_ts`, `window_start_ts_ms >
  window_end_ts_ms`, `feature_available_ts_ms != signal_birth_ts`, or (proxy
  layer) `last_contained_close_ts_ms > signal_birth_ts`.
- **Candle proxy discipline:** a candle is only "contained" if
  `close_ts_ms <= window_end_ts_ms` — a candle whose close lands after `T`
  is never included, proven by `test_proxy_partial_candle_excluded`.
- **REST repair rows:** every extraction's `endTime` was set to the span's
  own end, and all trades with `T > end_ms` were trimmed at fetch time
  (`fetch_window()`'s `page_done` handling) before ever reaching staging.
- **Required value: 0. Achieved: 0.**

## 8. Pre-outcome coverage precheck (Task 8 completion)

Using the **verbatim-reused** cycle-split machinery
(`ami.research.w8_short_expanded_baseline.compute_global_cycle_split` /
`split_rows_by_cycle_keys` / `assert_zero_cycle_straddling`, `is`-identity
import, no reimplementation) over the 1,828 `EXACT_RECONSTRUCTABLE` rows:

| Metric | Value |
|---|---|
| Eligible feature rows | 1,828 |
| Eligible independent cycles | 167 (100% of all canonical cycles) |
| TRAIN cycles | 116 |
| TEST cycles | 51 |
| Cycle straddling | **0** |
| `MIN_BUCKET_N` | 20 |
| Precheck | **PASS** (116 ≥ 20 and 51 ≥ 20) |

No outcome value was read to produce this table — it counts feature-row
eligibility and cycle membership only.

## 9. Tests (Task 12)

52 new tests added across 3 files (Task 12's 20-item list is covered; no
test was added merely to hit a count):

| File | Tests | Focus |
|---|---|---|
| `tests/test_ami_cvd_windowed_taker_flow.py` | 21 | sign convention, signed qty/notional, `[T-W,T]` boundary inclusivity, post-T/pre-window rejection, same-ts deterministic ordering, empty-window NULL-not-zero, normalized-CVD definedness, BUCKET determinism + rejection paths, pooling guard, schema CHECK rejection, proxy partial-candle exclusion, immutable conflict/NOOP, feature_available_ts law, 17/18-item matrix construction + no-silent-dropping + idempotent rerun + fail-closed-without-repair |
| `tests/test_ami_cvd_source_quality_contract_v1.py` | 16 | all classifier branches (happy path, no-coverage-map, cadence unavailable/fail, missing-minutes with/without repair, duplicate-unresolved, regime-boundary without proof), invalid-status rejection (API + raw SQL), append-only immutable conflict + new-assessment-version append, exhaustive sweep proving `UNREPAIRABLE` is never auto-assigned |
| `tests/test_ami_cvd_repair_rehearsal.py` | 15 | page-overlap dedup, missing-id-range detection, retry-then-success idempotency, rerun hash determinism, exhausted-retries fail-closed, probe-only verdict, zero-rows-never-exact, immutable staging conflict, schema CHECK on taker-side consistency, 5 cross-source reconciliation scenarios (1:1, unmatched, one-to-many, many-to-one/many-to-many, conflicting side-flag), float/decimal fingerprint equality |

**Regression accounting:** baseline 795/795 → collect-only with the
**unchanged** frozen command
(`pytest tests/test_ami_*.py tests/test_buyfade_mutations.py
tests/test_buyfade_silexit_mutations.py`) = **847** (795 + 52, exact match).

**New honest ground truth: 847/847.**

## 10. Proposed schema 11 → 12 migration (Task 13)

Full proposal (exact DDL, PK/FK/CHECK/uniqueness, immutable-version columns,
append-only/supersession rules, backup/restore/rollback procedure, expected
stop conditions) in `S34_CVD_SCHEMA_11_TO_12_MIGRATION_PROPOSAL_2026-07-05.md`.
**No approval is granted or implied by this batch.**

## 11. Final integrity verification (Task 15)

See the end-of-batch integrity section appended to `SYSTEM_STATE.md` and
`TEST_STATUS_LATEST.md` for the exact before/after hash, test-run and
protected-delta table. Summary: `canonical.sqlite` sha256/mtime/schema_version
unchanged throughout (11, `b28b4093…`); zero canonical writes; zero outcome
reads; zero runtime/risk/execution modifications; two full sequential
regression runs green at the new ground truth (847/847 each); collect-only
reproduced 847 both before and after.

---

## FINAL VERDICT

**`CVD_REPAIR_REHEARSAL_READY_FOR_CANONICAL_MIGRATION_PROPOSAL`**

Every hard-stop condition in the operator's batch spec was checked and
cleared: Binance supplied every probed outage (§3); pagination continuity
was proven for every extraction, twice (§4-5); no non-deterministic
extraction occurred (§4-5); REST-vs-legacy reconciliation was deterministic
for every window the matrix actually used as a repair source, with the one
observed real collision class correctly flagged rather than silently paired
(§4); no duplicate-conflict required an arbitrary assumption (§4); exact and
proxy evidence were never mixed (§5, schema-enforced); known-at safety
achieved the required 0 violations (§7); rerun hashes matched in every case
(§4-5); the protected `canonical.sqlite` hash/mtime never changed (§11); the
1,944-row matrix is fully explained (1,840 exact + 104 explicit BUCKET
exclusions, §5); and every window's source-regime compatibility was proven
via the regime-spanning quality-classification rule (§5, §6 of the
accepted contract), never assumed.

**WAIT_FOR_OPERATOR_APPROVAL**
