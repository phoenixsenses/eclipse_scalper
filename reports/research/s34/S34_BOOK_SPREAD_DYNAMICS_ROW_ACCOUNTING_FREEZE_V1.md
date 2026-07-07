# S34_BOOK_SPREAD_DYNAMICS_ROW_ACCOUNTING_FREEZE_V1

**Gate:** BATCH-BOOK-SPREAD-DYNAMICS-ROW-ACCOUNTING-FREEZE-V1
**Nature:** Outcome-blind immutable row-accounting and lineage freeze only. No canonical migration, no schema change, no outcome access, no experiment/result/nullifier/gate-receipt, no route/bucket promotion, no runtime/risk/execution/shadow/paper/forward/live change.
**Depends on:** operator ruling `FAM_BOOK_SPREAD_DYNAMICS_PRIMARY_DEFINITION_V1`; rehearsal commit `6a449a64`; readiness commit `f115b9c1`.
**Date:** 2026-07-07 · **Author:** Sonnet 5

---

## Accepted checkpoint

Rehearsal `6a449a64` (`BOOK_SPREAD_DYNAMICS_DISPOSABLE_REHEARSAL_V1_COMPLETE`) under the operator ruling W300 + additive `spread_change_bps` difference. `schema_version=13`, `experiment_registry=24`, `experiment_results=381`, nullifiers=2, gate_receipts=2. Regression baseline 1,027/1,013/14.

## Family / child identity, source, specification

- **Family:** `FAM_BOOK_SPREAD_DYNAMICS` · **Child:** `H-BOOK-SPREAD-CHANGE-BPS-W300-V1` · **Formula version:** `BOOK_SPREAD_CHANGE_BPS_W300_V1` · **Spec hash:** `ea611121…`
- **Source:** exact `book_ticker` L1 (`bid_price`/`ask_price`, row id `id`), `BINANCE_USDM_PERP` / ETHUSDT / `PERPETUAL_FUTURES` / USDT.
- **Formula:** `spread_change_bps_w300 = spread_bps(t0) − spread_bps(t0−300s)`; current/historical quote = latest exact valid quote at-or-before target with `id`-DESC tie-break; 5-min staleness at both endpoints; crossed/locked/zero/negative excluded; precedence `UNAVAILABLE > ZERO_NEG > CROSSED > LOCKED > STALE`; `known_at_ts = feature_available_ts = signal_birth_ts`.

## Ordering and serialization policies

- **Ordering:** `signal_birth_ts ASC, anchor_id ASC` — deterministic, immutable anchor fields only (never feature values/quality/direction/outcomes). The accepted rehearsal's own content/row-manifest hashes use `anchor_id`-only ordering and are reproduced separately.
- **Serialization:** each record = its fields (fixed order) rendered with `repr()` (full round-trippable float precision, identical discipline to the accepted rehearsal content hash), fields joined U+001F, records joined U+001E, sha256.

## Anchor accounting (324)

| Class | Count |
|---|---|
| `EXACT_RECONSTRUCTABLE` | 196 |
| `STALE_SOURCE` | 22 |
| `UNAVAILABLE_BEFORE_COLLECTION` | 106 |
| crossed / zero-neg / locked / repaired / source-gapped / proxy | 0 each |
| symbol / venue / segment / currency / duplicate-anchor mismatches | 0 each |
| **Total** | **324** = 196 + 22 + 106 |

Distinct anchor IDs = 324. Manifest hash `a77a8daf…`.

## Exact-feature accounting (196)

196 exact rows, all with a feature value; 0 excluded rows carry a value. Additive identity `change = current_spread − historical_spread` holds exactly (<1e-9). 0 non-finite, 0 non-positive mid. `spread_change_bps_w300` ∈ [−0.00084, +0.12807] (median +0.00018): 170 expansion / 26 compression / 0 zero. Current-quote age 0–13,682 ms (median 2 ms). Manifest hash `b1eb902f…`.

## Exclusion accounting (128)

`128 = 22 STALE_SOURCE + 106 UNAVAILABLE_BEFORE_COLLECTION`. Every excluded anchor carries exactly one final quality class and one primary exclusion reason (endpoint-tagged). All zero-count categories reproduced (not assumed). Manifest hash `0694e433…`.

## Cycle / representative accounting (97)

167 independent cycles total; 97 exact independent cycles (196-row membership manifest `e692ff1c…`); 97 selected representatives (manifest `edadf597…`) — 0 duplicate cycle IDs in the representative set, 0 cycles with >1 representative, 0 exact-eligible cycles without a representative. Representative rule: earliest `signal_birth_ts` per cycle among EXACT rows (anchor_id tie-break); uses no outcome/feature/direction/subgroup/route/bucket signal.

## Selected-source lineage

Each exact row binds both endpoints' `quote_id`/`quote_ts`/`bid`/`ask`/`spread_bps` plus the derived change — sufficient to reproduce and verify every selected quote. Full per-row detail in `S34_BOOK_SPREAD_DYNAMICS_ROW_ACCOUNTING_FREEZE_V1_MANIFEST.json`.

## Accepted-versus-replay (independent equality proof)

Two fresh independent rebuilds via the accepted rehearsal builder (not copies), each with an SQLite authorizer installed:

| Equality | Result |
|---|---|
| replay A ≡ replay B (content, row-manifest, all 5 manifests) | ✅ |
| replay A ≡ accepted rehearsal content hash `5e9ee58c…` | ✅ |
| replay A ≡ accepted rehearsal row-manifest hash `8e8e23ff…` | ✅ |
| replay B ≡ accepted content hash | ✅ |

Exact serialized equality (no approximate tolerance). Equality compared against the live-recomputed retained evidence (`.runtime_temp/spread_rehearsal_v1/rehearsal_run1.sqlite`). The replay uses the **same** `input_manifest_id` as the accepted rehearsal, so the content hash reproduces exactly — no field is excluded to force the match.

## Known-at and no-lookahead proof (revalidated from frozen lineage)

0 current-endpoint future quote selections, 0 historical-endpoint, 0 current-staleness violations, 0 historical-staleness violations, 0 identity violations, 0 known-at-field violations — **all zero**, re-derived directly from the frozen lineage fields (independent of the builder's in-flight check), both replays. No interpolation, no post-target nearest-neighbour, no forward-fill beyond 5 min, duplicate `ts_ms` resolved by `id DESC`.

## No-outcome-access proof

SQLite authorizer on the canonical connection for both replays → `authorizer_violations = []` (0 outcome/experiment/nullifier/gate-receipt access), corroborated by an AST static guard over the freeze module's `.execute()` literals. 0 outcome reads, 0 governance writes, 0 route/bucket promotion.

## Full hash tree + root

See `FAM_BOOK_SPREAD_DYNAMICS_ROW_ACCOUNTING_FREEZE_V1.md` (governance) and the JSON companion for the complete component-hash table. **Root hash `BOOK_SPREAD_DYNAMICS_ROW_ACCOUNTING_FREEZE_V1_ROOT = 33c4f4be3233aad399d72fc525601c7eecb2eb6ab235ecd4070ba640701c6e31`** — sha256 over the sorted `name=full-hash` pairs of the 8-component set, joined U+001E.

## Migration-contract draft (not executed)

Proposed destination tables (existing-convention names, no DDL written, no migration ID assigned): `ami_book_spread_change_windowed_flow` (feature grain, PK `feature_id`, unique `(anchor_id, formula_version)`, FK `anchor_id→ami_signal_lifecycle.signal_id`, expected 196 rows), `ami_book_spread_change_window_quality_v1` (accounting grain, 324 rows), `ami_book_spread_change_exclusions` (128 rows). Idempotency: second run `NOOP_IDENTICAL`, root hash unchanged. Immutability: insert-only; a corrected computation requires a new `formula_version`/freeze version. Rollback: byte-exact pre-migration backup + disposable restore proof before any real write. Draft only — makes the next gate deterministic; creates nothing.

## Verdict

**`BOOK_SPREAD_DYNAMICS_ROW_ACCOUNTING_FREEZE_V1_COMPLETE`**

**Disposition:** `BOOK_SPREAD_DYNAMICS_ROW_ACCOUNTING_FROZEN_FOR_CANONICAL_MIGRATION` — authorizes **no** automatic next step.

## Blockers / residual risks

None blocking. 22 STALE + 106 UNAVAILABLE are permanent, immutable, outcome-blind exclusions; the usable population is frozen at 196 rows / 97 cycles. Future-preregistration cross-family holdout-overlap disclosure remains a later-gate concern, not a freeze blocker.

## Recommended next gate

`BATCH-BOOK-SPREAD-DYNAMICS-CANONICAL-MIGRATION-V1` (do not begin automatically; must reuse this root, preserve 324/196/128/97, remain outcome-blind, assign a migration ID only inside that authorized gate).

## Storage report

Peak temporary disk ~0.9 MB under `.runtime_temp/spread_freeze_v1/` (two ~230 KB disposable replay SQLites + result/manifest JSON); one OS-scratchpad driver (deleted). The 335 KB detail manifest was copied into the repo as the committed immutable manifest. Retained accepted rehearsal evidence under `.runtime_temp/spread_rehearsal_v1/` was read-only and unmodified. No full database copy created.

Stopping after the row-accounting-freeze verdict and dedicated commit.
