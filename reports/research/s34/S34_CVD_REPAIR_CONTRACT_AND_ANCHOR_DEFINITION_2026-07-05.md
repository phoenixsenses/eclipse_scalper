# CVD / TAKER-VOLUME REPAIR CONTRACT + CANONICAL ANCHOR/EPOCH DEFINITION (2026-07-05)

**Batch type:** design, reconciliation and preregistration-readiness ONLY. No canonical.sqlite write, no backfill, no data mutation, no repair execution, no outcome read, no MFE/MAE, no alpha test, no threshold-on-outcome selection, no experiment registration, no collector change.
**Basis:** accepted audit `S34_CVD_TAKER_VOLUME_DATA_READINESS_AUDIT_2026-07-05.md` (verdict `CVD_DATA_REPAIR_REQUIRED`) and every artifact it cites, all re-read for this batch. New read-only facts established in this batch: (a) the collector's `_parse_agg_trade()` receives Binance's aggregate-trade id (`a`) and first/last trade ids (`f`/`l`) but **drops them** — only `T,s,p,q,m` are persisted, and `ts_ms` = Binance trade time `T` (not receive time); (b) the current collector has an aggTrades **REST fallback** (`ingest_rest_agg_trade`, default-enabled, 5s poll) whose per-symbol dedup cursor (`_rest_agg_last_id`) is **in-memory only** — a structural WS/REST double-insert risk given no unique trade id column exists; (c) the frozen candle-repair source package's `gap_manifest_pre_repair.json` (208 gap runs, 25,716 missing minutes, ETHUSDT 1m, 2026-02-15→07-03) is an authoritative **minute-granularity ETHUSDT agg_trades coverage map** (a candle-minute missing ⇔ zero agg_trades rows that minute), already byte-frozen with SHA-256 manifests.
**Machine-readable mirror:** `S34_CVD_REPAIR_CONTRACT_AND_ANCHOR_DEFINITION_2026-07-05.json`.

---

## 1. Canonical CVD anchor/epoch model (resolves blocker B7)

### 1.1 The frozen decision

**There is no global CVD epoch. The canonical CVD quantity is a bounded, event-relative, trailing-window net taker flow** — computed per signal, per frozen window, over `[T − W, T]` where `T = signal_birth_ts`.

A bounded trailing signed sum is translation-invariant: it needs no epoch, no reset rules, and no cross-restart state. This is the design insight that dissolves B7 rather than parameterizing it: an *unbounded* cumulative CVD is ill-defined without an arbitrary epoch choice (and couples every value's quality to the entire history since that epoch), while a *windowed* net flow is fully defined by its own window and inherits quality only from that window.

| Contract element | Frozen value |
|---|---|
| Anchor source event | The canonical signal (`ami_signal_lifecycle.signal_id`), via its `source_event_id` — the same identity chain as the birth-truncated geometry feature |
| Anchor timestamp | `signal_birth_ts` (proven == reconstructed `anchor_ts_ms` for 220/220 LONG; SHORT signals use their own `signal_birth_ts`, e.g. BTC-confirmation ts for `SHORT_NOISY_BTC200K_CONFIRMED_V1`) |
| Epoch semantics | Event-relative bounded windows `[T − W, T]`, end-inclusive at `T`, start-inclusive at `T − W`. Not fixed-clock, not rolling-persisted, not cumulative-since-anything |
| Frozen window family `W` | `{60s, 300s, 600s, 1800s, 3600s}` + `BUCKET` (= `[source_window_start_ts_ms, T]`, the signal's own reconstructed RUNNING_CLUSTER bucket window, ≤300s, already frozen in `ami_birth_truncated_cascade_geometry`) — **6 windows total** |
| Window-family justification (reconciliation-driven, NOT outcome-driven) | 60s brackets `research_s34_orderflow_lead.py`'s 15/30/60s scale; 600s = `research_ami_mfe50_experiment.py`'s exact 10-minute trailing convention (`ofi10m`/`taker_sell10m`); 3600s = `orderflow_chart.py`'s 1-hour rolling CVD panel; `BUCKET` = the geometry feature's own frozen window; 300s/1800s complete a coarse log-spaced ladder between those pre-existing conventions. No outcome value was read to choose these |
| Multiple signals in one independent cycle | Each signal computes its own windows at its own `T` — windows may overlap across same-cycle signals and that is expected; **statistical independence is never claimed at the feature layer**; it is enforced at the research layer by the existing cycle-grouped split machinery (signals sharing `independent_cycle_id` never straddle TRAIN/TEST, and cycle-count — not signal-count — drives sufficiency), identical to every W8 precedent |
| Reset rules | None needed (stateless bounded windows) |
| Symbol scope (v1) | **ETHUSDT flow only** (the canonical signal/event population is 100% ETHUSDT). BTCUSDT/SOLUSDT flow windows are explicitly DEFERRED, not implicitly included |
| Venue scope | Binance **USDT-M futures** (`fapi`) only — matching every other source in the project. No spot |
| `feature_available_ts` | `= signal_birth_ts` exactly. Every source row used satisfies `ts_ms ≤ signal_birth_ts` by construction (window end = T) |
| Allowed as-of lookup | Strictly `ts_ms <= T`; the same defensive pre-truncation discipline as `reconstruct_signal_geometry()` (filter first, compute second, assert no future row passed) |
| Collector restarts / regime changes | Never handled by resets or special-casing in the feature: a window overlapping an outage/regime problem is handled entirely by its **quality status** (§4/§8). Feature values are computed identically in all regimes; quality classification carries the regime information |

### 1.2 Rejected alternatives (recorded so they are not re-litigated)

1. **Session/day-anchored cumulative CVD (fixed-clock epochs):** rejected — epoch choice is arbitrary, values are unbounded and drift-dominated, and a gap anywhere since the epoch contaminates every later value of the day. Quality classification would degenerate to "whole day complete or nothing."
2. **Previous-accepted-anchor-relative cumulative (inter-anchor CVD):** well-defined (the geometry feature's `inter_cluster_gap_sec` proves the previous-anchor concept), but its required source window is the entire inter-anchor span — the geometry batch measured exactly this pattern making `inter_cluster_gap_sec` the *limiting field* (87 vs 94 COMPLETE). Adopting it as the primary CVD form would voluntarily inherit the worst-case quality coupling. DEFERRED as a possible v2 feature, never silently included in v1.
3. **Persisted rolling CVD series (materialized time series):** rejected for v1 — it recreates the epoch problem at materialization time, multiplies storage, and none of the three existing consumer patterns actually needs a persisted series (all three compute bounded windows on demand).

---

## 2. Authoritative CVD quantities (strict separation)

Sign convention (verified in three independent code sites: `candle_builder.py` lines 89-92, `state_reconstruct.py` `_get_trade_side()`, `orderflow_chart.py` `cvd_series()`): `is_buyer_maker = 0` → taker BUY (+); `is_buyer_maker = 1` → taker SELL (−).

| # | Quantity | Definition | Layer |
|---|---|---|---|
| Q1 | `cvd_notional_{W}` | Σ over trades in `[T−W, T]` of `sign · notional` | **EXACT trade-level** (authoritative) |
| Q2 | `cvd_qty_{W}` | Σ of `sign · quantity` (base asset) | **EXACT trade-level** (authoritative) |
| Q3 | `total_notional_{W}` | Σ of `notional` (unsigned; the normalization denominator any ratio form derives from — e.g. mfe50's `ofi10m = (2·buy−tot)/tot ≡ Q1/Q3`) | **EXACT trade-level** (auxiliary) |
| P1 | `candle_cvd_qty_{W}` | Σ over 1m candles **fully contained** in `[T−W, T]` of `(taker_buy_volume − taker_sell_volume)` | **PROXY** — minute-quantized boundaries; never equivalent to Q2 even on gap-free data (partial edge minutes excluded); additionally carries repaired-candle provenance where applicable |

**Frozen non-equivalences (contract law):**
- P1 is NEVER a substitute for Q1/Q2 in an exact-layer population. No silent substitution, ever.
- Repaired 1m klines (`candle-binance-fapi-repair-v1`) recover Binance-authoritative *per-minute net* taker volume — the intra-minute trade sequence is permanently unrecoverable from klines. Kline-based repair therefore produces P1-layer data only.
- The ONLY exact-layer repair source is trade-level Binance historical aggTrades (§4).

---

## 3. Source-regime segmentation

| Regime | Start → End (measured) | Writer / mode | Duplicate policy | Ordering | Completeness proof available | Known limitations |
|---|---|---|---|---|---|---|
| **R0** | 2026-02-15 14:26 → 2026-04-12 07:47 | old collector, per-symbol `{sym}@aggTrade` WS, single combined socket; ETH+BTC only | None in DB (no trade-id stored); WS-only path, low intrinsic duplicate risk | `ts_ms` (=Binance `T`) + autoincrement `id` tiebreak | ETH: frozen minute-map (gap manifest: Feb=42, Mar=164 missing minutes fall here); sub-minute unproven | No gap registry existed yet for this stream (its first agg row is R1); BTC/SOL minute-maps never built |
| **R1** | 2026-04-12 07:47 → 2026-04-24 10:27 | same collector + live gap registry (agg stream); SOL joins 2026-04-18 08:41 | as R0 | as R0 | Registry rows (18 resolved + 2 open) + ETH minute-map | Registry itself proven unreliable-by-class in the liquidation reconciliation (in-memory resolution state, orphaned opens) |
| **R2** | 2026-04-24 10:27 → 2026-06-06 17:43 | same code, **degraded operations**: registry dead; repeated multi-day process-wide outages (6 major windows, §4) | as R0 | as R0 | ETH minute-map ONLY (registry blind); cross-symbol simultaneous-recovery evidence from the audit | The worst era; every major outage lives here |
| **R3** | 2026-06-06 17:43 → present | current collector (`5cda3122` era): all-market forceOrder change did NOT change aggTrades (still per-symbol WS); **REST fallback for aggTrades present in current source** (`ingest_rest_agg_trade`, in-memory `_rest_agg_last_id` cursor) | **Structural WS/REST double-insert risk**: no trade-id column + non-persistent REST cursor means overlap between a WS resume and REST-polled span can insert byte-identical duplicate rows undetectably | as R0 | ETH minute-map through 07-03; nothing after; no registry | Exact deploy timestamp of the REST-fallback code is NOT provable (collector file historically untracked in git — honest limitation); duplicate-cluster rate never measured |

**Timestamp semantics (all regimes):** `ts_ms` is Binance trade time `T` — event-time, not local receive time. Late arrival therefore lands correctly by timestamp; replay `ORDER BY ts_ms, id` is deterministic.

---

## 4. Repairability taxonomy + per-outage assignment

**Taxonomy (frozen):** `EXACT_RECONSTRUCTABLE` (trade-level authoritative source retrievable) / `PROXY_ONLY` (only minute-level kline taker volume retrievable/already present) / `SOURCE_GAPPED` (confirmed loss, no repair source) / `SOURCE_COVERAGE_UNRESOLVED` (coverage not yet positively audited) / `UNREPAIRABLE` (loss confirmed AND all repair sources confirmed unavailable).

**Repair source (exact layer):** Binance official historical USDT-M futures aggTrades — (a) `GET /fapi/v1/aggTrades` with `fromId` pagination (limit 1000), and/or (b) `data.binance.vision` daily aggTrades archives. Same authoritative-source precedent as the accepted candle repair (which proved Binance retrievability over these exact calendar windows at kline granularity, 25,716/25,716). **Availability at trade granularity is asserted from Binance's documented retention, NOT yet probed — the rehearsal's first stop-condition is an explicit per-window availability probe (§10). Every `EXACT_RECONSTRUCTABLE` assignment below is provisional on that probe; probe failure ⇒ automatic reclassification to `PROXY_ONLY` (ETHUSDT klines for all these windows are ALREADY repaired locally) + operator decision.**

| Outage (from accepted audit) | Symbols | Assignment | Justification |
|---|---|---|---|
| 2026-04-30 → 05-02 (44–51h) | ETH+BTC | `EXACT_RECONSTRUCTABLE` (provisional) | Process-wide local outage; exchange-side data existed and is within Binance historical retention; kline-level retrieval over this window already proven by candle repair |
| 2026-05-08 → 05-09 (25–35h) | ETH+BTC | `EXACT_RECONSTRUCTABLE` (provisional) | same |
| 2026-05-15 (10.1h) | SOL | `EXACT_RECONSTRUCTABLE` (provisional) | same mechanism; SOL not needed for v1 population (ETHUSDT-only) — repair optional |
| 2026-05-21 → 05-22 (28.6–28.8h) | ETH+BTC+SOL | `EXACT_RECONSTRUCTABLE` (provisional) | same |
| 2026-05-26 → 05-28 (53–64h) | ETH+BTC (+3h SOL tail) | `EXACT_RECONSTRUCTABLE` (provisional) | same |
| 2026-06-01/02 → 06-05 (79–96h) | ETH+BTC | `EXACT_RECONSTRUCTABLE` (provisional) | same — largest single window |
| Registry-era small gaps (20 rows, 66s–2.2h, 2026-04-12→04-24) | mixed | `EXACT_RECONSTRUCTABLE` (provisional) | same; individually tiny |
| ETH minute-map gaps outside the above (Feb=42, Mar=164 min, residual Apr/Jun minutes) | ETH | `EXACT_RECONSTRUCTABLE` (provisional) | the frozen gap manifest enumerates them per-minute; same repair source |
| Sub-minute completeness, all eras (a minute with ≥1 trade appears "present" in the minute-map) | all | `SOURCE_COVERAGE_UNRESOLVED` | no sub-minute proof exists yet; the rehearsal's healthy-era cadence baseline (§10.D1) is the designated instrument |
| Post-2026-07-03 span (beyond the frozen minute-map) + audit blocker B6 (post-06-06 not exhaustively scanned) | all | `SOURCE_COVERAGE_UNRESOLVED` | full-range cadence scan is rehearsal step D1-1 |
| SOL before 2026-04-18 | SOL | out of scope (never subscribed — absence of regime, not an outage) | recorded to prevent misclassification as a gap |
| — | — | `UNREPAIRABLE`: **none assigned** | nothing identified is proven unrepairable; this class stays reserved |

---

## 5. Duplicate and ordering rules (frozen)

1. **Canonical raw-trade identity — repaired rows:** `(symbol, agg_trade_id)`. The repair source provides Binance's `a` id; the repaired-rows table MUST store it (making the repaired regime strictly stronger than live collection, which discards it).
2. **Canonical raw-trade identity — live-collected rows:** none exists (by schema). The composite `(symbol, ts_ms, price, quantity, is_buyer_maker)` can collide legitimately (two identical real trades in the same ms), so exact-duplicate rows are **suspect but not provably duplicates** row-by-row.
3. **Duplicate detection (live rows):** statistical only — the rehearsal measures the exact-duplicate-cluster rate per regime and compares R3 (REST-fallback era) against R0/R1 baseline; a materially elevated R3 rate triggers the `BLOCKED_BY_DUPLICATE_INTEGRITY` stop condition and an operator decision (options at that point: window-level `DUPLICATE_SUSPECT` quality flag, or R3 re-fetch from the authoritative source where probing succeeds).
4. **Late arrivals:** inert by construction — `ts_ms` is exchange event time; ordering is by `ts_ms`, not arrival.
5. **Out-of-order ingestion:** same — replay order is `ORDER BY ts_ms ASC, id ASC` (id = insertion tiebreak within a millisecond), deterministic.
6. **Correction policy:** live rows are **NEVER updated or deleted**. Repair supersedes: repaired trade rows live in a separate versioned table; consumers use an effective-selection rule (repaired window ⊃ live rows for that window), exactly the `path-v2-candle-repair-r1` precedent. Fail-closed on identity conflicts (same repaired identity, different content ⇒ immutable-conflict error), the established ledger discipline.

---

## 6. Corrected-data version contract (proposed IDs — NOT created in canonical SQL in this batch)

| Concern | Immutable version ID |
|---|---|
| Raw source interpretation (sign convention, `T`-time semantics, per-symbol WS scope) | `aggtrades-taker-side-v1` |
| Repaired trade population | `aggtrades-binance-fapi-repair-r1` |
| Feature definition | `s34-cvd-windowed-taker-flow-v1-birth-truncated` |
| Source-quality assessment | `cvd-source-quality-contract-v1` |

Semantics identical to the geometry precedent: feature values + source manifests immutable under their version; a redefinition mints a new version, never overwrites; quality reassessment appends under a new assessment version, never rewrites.

## 7. Field-level provenance and quality (per proposed feature)

One feature row per `(signal_id, window)`; the three exact quantities (Q1/Q2/Q3) of a row share the SAME source window and the SAME source rows by construction — so, unlike the geometry feature (where windows differed per field and field-level quality was mandatory), **here the window IS the quality unit: row-level quality ≡ field-level quality, stated explicitly rather than silently assumed.** Each feature row carries / is joined to:

| Element | Content |
|---|---|
| Source table(s) | `agg_trades` (live) ⊕ proposed `ami_agg_trades_repaired` (per §5.6 effective selection); proxy features additionally `ami_candles` |
| Source window | `[T − W, T]` exact bounds stored per row (`window_start_ts_ms`, `window_end_ts_ms`) |
| `feature_available_ts` | `= signal_birth_ts` |
| Quality status | from `cvd-source-quality-contract-v1`: window's minute-map coverage + repair status + cadence proof + duplicate assessment → one of the §4 taxonomy values, stored append-only |
| Repair method | `NONE` / `AGGTRADES_REST` / `AGGTRADES_VISION_ARCHIVE` / (`PROXY_KLINE` for P1 rows only) |
| Source-regime ID | R0/R1/R2/R3 of the window (window spanning a boundary records both) |
| Provenance pointer | source-row manifest SHA-256 (sorted `[ts_ms, notional, sign]` triples — geometry precedent) + field-provenance rows in the established `DETERMINISTIC_HISTORICAL_SAFE` vocabulary |
| Inferential use allowed | derived, never stored as an independent opinion: `quality == EXACT-layer COMPLETE` per §8 |

## 8. Eligibility rules for future experiments (frozen)

1. **Inferentially eligible:** only exact-layer (Q1/Q2/Q3) feature rows whose window is `EXACT_RECONSTRUCTABLE`-repaired-and-verified or natively complete under the frozen contract (minute-map complete AND cadence-proof pass AND no unresolved duplicate flag).
2. **`PROXY_ONLY` rows: descriptive only, always.** They may never enter an inferential population, never be pooled with exact rows, and never silently upgrade.
3. **Mixed exact/proxy populations: forbidden** — enforced by the same fail-closed pooling-guard pattern as `assert_not_pooled`/evidence-layer guards (a fetch returning both layers raises).
4. **Independent-cycle grouping:** `compute_global_cycle_split`/`split_rows_by_cycle_keys`/`assert_zero_cycle_straddling` reused verbatim (`is`-identity, the W8 discipline). Cycle-straddling must measure 0; sufficiency counts CYCLES, never signals.
5. **Train/test compatibility:** where an experiment pairs with an existing baseline population, the split is recomputed from the paired baseline's own population and byte-compared (`verify_split_matches_*` precedent).
6. **Minimum-coverage precheck before ANY outcome is read:** eligible-population cycle counts must satisfy `MIN_BUCKET_N = 20` in BOTH splits, reported in a pre-outcome coverage report (monthly distribution + setup composition), else the experiment records `INSUFFICIENT_SAMPLE`/blocked verdicts without opening outcomes — the standing fail-closed rule.

## 9. What this contract does NOT do

No repair was executed; no canonical row was written; no outcome/MFE/MAE/alpha value was read; no threshold was selected against outcomes; no experiment was registered; no collector code was modified. The window family in §1.1 is a pre-hoc design constant set reconciled to pre-existing code conventions, not a fitted parameter.

---

## 10. Proposed controlled implementation batch (NOT implemented here)

**Name:** `BATCH-CVD-REPAIR-REHEARSAL-AND-QUALITY-CONTRACT-V1`

### Phase D1 — disposable rehearsal (no real-DB write)
1. **Full-range cadence + coverage scan** (read-only, streaming — no full-list materialization): per symbol, per regime, whole Feb→now range; extends the audit's sampled scan; closes B6; produces the healthy-era cadence baseline from which the completeness threshold constant is frozen (from R3 healthy spans — explicitly source-derived, never outcome-derived).
2. **Duplicate-cluster baseline** (§5.3) per regime; R3-vs-baseline comparison.
3. **Repair-source availability probe:** for every §4 `EXACT_RECONSTRUCTABLE` window, fetch a small sample via `/fapi/v1/aggTrades` (and/or verify `data.binance.vision` archive presence) — per-window pass/fail recorded; failures auto-reclassify to `PROXY_ONLY`.
4. **Disposable repair build:** fetch full aggTrades for verified windows into a disposable DB table `ami_agg_trades_repaired` (with `agg_trade_id`), frozen source package (raw + request manifests + SHA-256, candle-repair Part-0 precedent), deterministic reconstruction proof.
5. **Disposable feature backfill:** compute all `(signal, window)` rows for the 324 canonical signals × 6 windows (**expected 1,944 feature rows; 1,944 quality rows; 1,944 × field-provenance rows per the provenance spec**; LONG-only view = 1,320), with write-time known-at assertions (every source `ts_ms ≤ T`).
6. **Quality classification** of all 1,944 windows under `cvd-source-quality-contract-v1`; pre-outcome coverage report (eligible cycles per split vs `MIN_BUCKET_N`) — measured, never forced.

### Phase D2 — validation
Idempotent rerun (row-count + content-hash identical); conflicting-content fail-closed probe; old-reader compatibility (all existing table counts unchanged); rollback → schema-fingerprint restore; reapply → byte-identical content (all: geometry-rehearsal precedent, reused patterns).

### Phase D3 — canonical migration proposal (separate operator approval)
`CANONICAL_SCHEMA_VERSION 11 → 12`; fold disposable-validated DDL verbatim into `ami/warehouse/schema.py` (`_SCHEMA_PHASE_CVD`); controlled entry point mirroring `birth_truncated_geometry_canonical_migration.py`.

### Phase D4/D5 — immutable backfill + quality ledger (post-D3 approval only)
Frozen-rehearsal-values contract (hashes must reproduce exactly); append-only quality ledger with effective views; `IMMUTABLE_*_CONFLICT` fail-closed writers.

### Files to create/modify (proposal)
| File | Action |
|---|---|
| `ami/cvd/__init__.py`, `ami/cvd/windowed_taker_flow.py` | NEW — feature definition + schema + backfill (quality-free, geometry pattern) |
| `ami/cvd/cvd_source_quality_contract_v1.py` | NEW — regime map, minute-map/cadence/duplicate classification, append-only quality ledger |
| `ami/cvd/aggtrades_repair_rehearsal.py` | NEW — availability probe + source-package freeze + disposable repair build (candle-gap-repair pattern) |
| `ami/cvd/cvd_rehearsal.py` | NEW — end-to-end disposable rehearsal harness |
| `tests/test_ami_cvd_windowed_taker_flow.py`, `tests/test_ami_cvd_source_quality_contract_v1.py`, `tests/test_ami_cvd_repair_rehearsal.py` | NEW — sign-convention lock, window-family lock (6 windows, no additions), known-at (future-trade invariance), no-outcome-terms static guards, proxy-never-pooled-with-exact guard, taxonomy branch tests, duplicate/ordering determinism, idempotency/rollback/reapply, real-data smoke |
| Proposed tables (disposable first) | `ami_agg_trades_repaired`, `ami_cvd_windowed_flow`, `ami_cvd_field_quality_v1`, `ami_cvd_field_provenance` (+2 effective views) |
| Version IDs | exactly §6 |

### Stop conditions (fail closed, operator decision required)
1. Availability probe fails for any `EXACT_RECONSTRUCTABLE` window → that window `PROXY_ONLY`; if the ELIGIBLE exact-layer population then fails `MIN_BUCKET_N`, record `CVD_INFERENTIAL_RESEARCH_BLOCKED_BY_SOURCE_QUALITY`.
2. R3 duplicate-cluster rate materially elevated vs baseline → `BLOCKED_BY_DUPLICATE_INTEGRITY`.
3. Healthy-era cadence threshold cannot be frozen non-arbitrarily → `BLOCKED_BY_SEMANTICS`.
4. Any disposable-vs-expected count/hash mismatch, any known-at assertion failure, any pooling-guard breach → hard stop, nothing proceeds to D3.
5. Frozen regression (795/795 command) must stay green at every phase boundary.

---

## FINAL VERDICT

**`CVD_REPAIR_CONTRACT_READY_FOR_REHEARSAL`**

All semantics required by the operator's ten work items are frozen above with no unresolved definitional dependency; the open empirical questions (availability probe, cadence baseline, duplicate baseline, B6 closure) are exactly the rehearsal phase's designated measurements, each with a fail-closed stop condition — none blocks the contract itself.
