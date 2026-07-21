# S34 CVD — CROSS-SOURCE DEDUPLICATION AND COLLISION MEASUREMENT (2026-07-05)

**Batch:** `BATCH-CVD-REPAIR-REHEARSAL-AND-QUALITY-CONTRACT-V1`, Tasks 4-5.
**Scope:** measurement only. No canonical write, no outcome read.

## 1. Identity within REST repair data

`(symbol, agg_trade_id)` is checked as the sole identity for REST-retrieved
rows (Binance's own `a` field). Across every repair span used in the main
rehearsal (8 spans, 40,934 staged rows) and every replay sample window (3
windows, 37,400 rows), **zero duplicate `agg_trade_id` values** were observed
within a single extraction, and **zero id-range holes** — every extraction's
`missing_id_ranges` was `[]` and every rerun reproduced an identical ordered
content hash. See `S34_CVD_DETERMINISTIC_REPLAY_HASH_MANIFEST_2026-07-05.json`
for the full per-window manifest (`rest_identity` block per window).

## 2. Reconciliation: REST rows vs legacy local rows

**Matching algorithm (frozen, per the accepted contract):** exact fingerprint
`(ts_ms, price, quantity, is_buyer_maker)` multiset intersection. A REST row
matches a legacy row 1:1 **iff** the fingerprint occurs exactly once on both
sides. Any fingerprint with multiplicity > 1 on either side is a collision
class — counted, never arbitrarily paired. No nearest-row or probabilistic
matching exists anywhere in this code path
(`ami/cvd/aggtrades_repair_rehearsal.py::reconcile_rest_vs_legacy`).

### 2.1 Three frozen sample windows (source-derived, not outcome-derived)

| Window | Symbol | Span | Regime | REST rows | Legacy rows | Exact 1:1 | Unmatched REST | Unmatched legacy | Many-to-many | Deterministic? |
|---|---|---|---|---|---|---|---|---|---|---|
| W_A (blackout) | ETHUSDT | 2026-06-03 00:00–00:10 | R2 | 22,091 | 0 | 0 | 22,091 | 0 | 0 | **NO** (trivially — legacy has zero rows, so "unmatched" is the expected/correct outcome, not an ambiguity; see §2.2) |
| W_B (healthy R3) | ETHUSDT | 2026-06-20 12:00–12:10 | R3 | 3,050 | 3,050 | 3,050 | 0 | 0 | 0 | **YES** — 100% 1:1, zero collisions |
| W_C (healthy R0) | ETHUSDT | 2026-03-10 12:00–12:10 | R0 | 12,259 | 12,259 | 12,257 | 0 | 0 | 1 (2 REST rows ↔ 2 legacy rows sharing one fingerprint) | **NO** — 1 many-to-many collision class present |

Full detail (per-window `duplicate_fingerprint_multiplicity_hist`,
`ambiguous_rest_rows`/`ambiguous_legacy_rows`) is in
`S34_CVD_DETERMINISTIC_REPLAY_HASH_MANIFEST_2026-07-05.json`.

### 2.2 W_A interpretation — the clean supersession case

W_A sits fully inside the confirmed June 1–5 blackout: local `agg_trades` has
**zero** rows in this span (already known from the minute-map). REST rows are
therefore never reconciled against anything — there is nothing to collide
with. This is exactly the "effective selection" case the contract designed
for (§5.6 of the accepted contract, path-v2-candle-repair-r1 precedent): a
repaired minute with zero pre-existing legacy rows is unambiguous by
construction, and the main rehearsal's matrix build restricts REST-row usage
to precisely this case (`fetch_repair_window_rows` in
`ami/cvd/cvd_rehearsal.py` only ever pulls REST rows for minutes absent from
the local minute-map). **No window in the 324×6 feature matrix mixes REST and
legacy rows for the same minute** — this was true by construction, not
discovered after the fact, but W_A's 0-vs-22,091 split is the empirical
confirmation.

### 2.3 W_C interpretation — a real, small collision class in a healthy era

W_C (a fully healthy R0 10-minute window, no missing minutes) shows **one**
many-to-many collision: 2 REST rows and 2 legacy rows share an identical
`(ts_ms, price, quantity, is_buyer_maker)` fingerprint (two genuinely
simultaneous trades at the same millisecond, same price, same size, same
side — not a data error, a real market coincidence). Because this window has
zero missing minutes, it is **never used as a repair source** in the main
matrix (repair rows are only fetched for minutes absent from the local
minute-map, per §2.2) — the collision is reported here as a measurement of
the reconciliation algorithm's honesty, not a defect that affects any
feature-matrix row. Had this window instead needed REST repair, the fail-closed
rule would apply: **a many-to-many collision blocks a `MATCHED_1TO1`
determination for those 2 rows** — `deterministic_supersession_feasible =
false` for the window as a whole, per the frozen rule "no arbitrary
nearest-row or probabilistic matching for an exact-repair claim."

## 3. Duplicate-cluster baseline per regime (Pass B, full-range scan)

Adjacent-exact-duplicate rows (`(ts_ms, price, quantity, is_buyer_maker)`
appearing more than once at the same timestamp), measured over the **entire**
local ETHUSDT population (174,571,325 rows) via the full-range ts-order scan:

| Regime | Rows | Duplicate-extra rows | Rate | vs R0/R1 baseline |
|---|---|---|---|---|
| R0 | 97,437,047 | 156,796 | 0.1609% | baseline |
| R1 | 16,567,889 | 2,050 | 0.0124% | baseline |
| R2 | 23,955,131 | 4,442 | 0.0185% | 0.11x of R0 max — **not elevated** |
| R3 | 36,611,258 | 14,667 | 0.0401% | 0.25x of R0 max — **not elevated** |

**Frozen elevation rule (per contract §5.3):** R3 rate flagged only if it
exceeds `max(R0, R1) × 10`. Measured R3 rate (0.0401%) is far below that bar
(16.09% would be the flag threshold) — **`BLOCKED_BY_DUPLICATE_INTEGRITY` is
NOT triggered.** The REST-fallback era's structural double-insert risk
(documented in the accepted contract §3, regime R3) did **not** manifest as
a measurable elevation in the adjacent-duplicate rate over this population.
This is a statistical absence-of-elevation finding, not a proof that no
double-insert has ever occurred — the contract's own duplicate-detection
law (§5.3) is explicitly statistical, not row-level provable, for legacy
rows (no trade-id column exists to prove it directly).

## 4. Conflicting duplicates (same ts/price/qty, opposite `is_buyer_maker`)

Measured as `conflict_pair_groups` in Pass B (a real, legitimate phenomenon:
Binance's aggTrade stream can report the same trade fill from both sides in
rare edge cases, or two genuinely simultaneous opposite-side trades share all
other fields). Counts: R0=876, R1=280, R2=275, R3=331 (ETHUSDT). These are
**not** treated as errors — they are recorded as a distinct, measured
category and never silently merged or resolved by preferring one side.

## 5. Float vs. decimal representation

REST rows arrive as JSON strings (`"1858.96"`); legacy rows are stored as
SQLite `REAL`. The reconciliation fingerprint applies `float()` to both sides
before comparison — verified deterministic and collision-free for every
observed price/quantity pair in this batch's samples (see
`test_float_representation_equality_of_fingerprints` in
`tests/test_ami_cvd_repair_rehearsal.py`). No representation-driven false
mismatch was observed.

## 6. Summary verdict

- Identity within REST data: **clean** (0 duplicate ids, 0 holes, in every
  extraction performed this batch).
- Cross-source reconciliation: **deterministic and collision-free for the
  minutes actually used as repair sources** (zero-legacy-row minutes only,
  by construction); a small, real, and correctly-flagged collision class
  exists in at least one fully-healthy sample window (W_C) and is handled by
  the fail-closed rule, never by arbitrary pairing.
- Duplicate-cluster elevation (R3 structural risk): **not observed** at the
  frozen elevation threshold — `BLOCKED_BY_DUPLICATE_INTEGRITY` does not
  fire in this batch.
- No exact-repair claim in the main rehearsal rests on an unresolved
  collision: the matrix only ever draws REST rows into minutes with **zero**
  legacy rows, which is the one case this measurement proves is always
  unambiguous.
