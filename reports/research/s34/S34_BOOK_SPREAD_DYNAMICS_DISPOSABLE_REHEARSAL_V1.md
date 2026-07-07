# S34_BOOK_SPREAD_DYNAMICS_DISPOSABLE_REHEARSAL_V1

**Gate:** BATCH-BOOK-SPREAD-DYNAMICS-DISPOSABLE-REHEARSAL-V1
**Nature:** Outcome-blind disposable rehearsal only. No TRAIN/TEST outcome access, no experiment/result/nullifier/gate-receipt, no canonical migration, no schema change, no route/bucket promotion, no runtime/risk/execution/shadow/paper/forward/live change.
**Depends on:** operator ruling `FAM_BOOK_SPREAD_DYNAMICS_PRIMARY_DEFINITION_V1`; accepted readiness commit `f115b9c1`.
**Date:** 2026-07-07 · **Author:** Sonnet 5

---

## Accepted checkpoint

Readiness `f115b9c1` resolved `SPREAD_EXPANSION_COMPRESSION_DEFINITION_AMBIGUOUS`; the operator ruling froze **W300 + additive `spread_change_bps` difference**. `schema_version=13`, `experiment_registry=24`, `experiment_results=381`, nullifiers=2, gate_receipts=2. Deterministic regression baseline 1,027/1,013/14.

## Family / child identity and formula

- **Family:** `FAM_BOOK_SPREAD_DYNAMICS` · **Child working ID:** `H-BOOK-SPREAD-CHANGE-BPS-W300-V1`
- **Formula version:** `BOOK_SPREAD_CHANGE_BPS_W300_V1` · **Specification hash:** `ea611121291c63136860d57926389520de571ce6615bed2e1a3627e51442a212`

```
mid_price(t)  = (best_ask(t) + best_bid(t)) / 2
spread_bps(t) = 10,000 × (best_ask(t) - best_bid(t)) / mid_price(t)
spread_change_bps_w300 = spread_bps(t0) - spread_bps(t0 - 300s)
```

Sign: + = expansion, − = compression, 0 = unchanged. Units: bps of spread change.

## Source identity

`data/microstructure.db:book_ticker`, `bid_price`/`ask_price`, deterministic row id `id`, **Binance USDⓈ-M perpetual futures** (`BINANCE_USDM_PERP`), symbol `ETHUSDT`, segment `PERPETUAL_FUTURES`, quote currency `USDT`, coverage begins 2026-04-11T17:08:42.005Z. **EXACT** for spread (L1 best bid/ask *is* the spread; no proxy tier). Same tradable instrument as the anchor population (100% ETHUSDT perp).

## Specification (frozen)

Current target `t0`; historical target `t0 − 300000 ms`. Quote selection at each: `WHERE ts_ms<=target ORDER BY ts_ms DESC, id DESC LIMIT 1` (the accepted deterministic contract from `f115b9c1` — `id DESC` tie-break is materially required: ~75% of rows share a `ts_ms`, ~6.5% of collisions carry a different bid/ask). Staleness tolerance 5 min (`FEED_LIMITS["book_ticker"]`, reused). Locked→`INVALID_QUOTE_LOCKED`, crossed→`INVALID_QUOTE_CROSSED`, zero/negative→`INVALID_QUOTE_ZERO_OR_NEG`, no quote→`UNAVAILABLE_BEFORE_COLLECTION`, too-old→`STALE_SOURCE`. `known_at_ts = feature_available_ts = signal_birth_ts` (schema `CHECK`-enforced). Symbol/venue/segment pinned by `CHECK`. Independent-cycle representative rule: earliest `signal_birth_ts` per cycle among EXACT rows.

## Quality-class precedence (frozen before counts)

`UNAVAILABLE_BEFORE_COLLECTION` > `INVALID_QUOTE_ZERO_OR_NEG` > `INVALID_QUOTE_CROSSED` > `INVALID_QUOTE_LOCKED` > `STALE_SOURCE`. A row is `EXACT_RECONSTRUCTABLE` iff **both** endpoints are EXACT; otherwise its single row-level reason is the highest-precedence non-EXACT status among the two endpoints (tagged current/historical/both). Independent of feature values and outcomes.

## Anchor accounting (outcome-blind, reconciles exactly)

| Source-quality class | Count |
|---|---|
| `EXACT_RECONSTRUCTABLE` | **196** |
| `STALE_SOURCE` | 22 |
| `UNAVAILABLE_BEFORE_COLLECTION` | 106 |
| `INVALID_QUOTE_CROSSED` / `_ZERO_OR_NEG` / `_LOCKED` | 0 / 0 / 0 |
| `REPAIRED_EXACT` / `SOURCE_GAPPED` / `PROXY_ONLY` | 0 / 0 / 0 |
| symbol / venue / segment / duplicate-source / duplicate-anchor exclusions | 0 |
| **Total** | **324** = 196 + 22 + 106 |

**196 exact rows → 97 independent cycles → 97 representatives, 0 duplicates.** Est. TRAIN 67 / TEST 30 (≥ `MIN_BUCKET_N=20`). Independent cycles total (all anchors) = 167.

**Difference from the readiness estimate: none.** The strict two-endpoint rehearsal reproduces the readiness level-at-birth accounting exactly (196/22/106) — because `book_ticker` updates sub-second, the historical (T−300s) endpoint is fresh whenever the current (T) endpoint is. Any difference would have been explained by an immutable, outcome-blind reason; there is none.

## Known-at and no-lookahead proof

`known_at_violations = 0`; `known_at_field_violations = 0`. At both endpoints: 0 selected quotes postdate their target (structurally impossible under `WHERE ts_ms<=target`, with a defensive raise); no post-target nearest-neighbour; no future snapshot in lineage; no interpolation across either target or across signal birth; no future repair data; staleness bounded; `feature_available_ts` reproducible; out-of-order handled deterministically; duplicate `ts_ms` resolved by `id DESC`.

## No-outcome-access proof

A SQLite authorizer (`install_access_guard`) was installed on the canonical (read-only) connection for the entire run, denying any reference to `ami_lifecycle_path_observations`/`experiment_registry`/`experiment_results`/`epistemic_test_nullifiers`/`experiment_gate_receipts` and the outcome columns `endpoint_return_bps`/`mfe_bps`/`mae_bps`. **`authorizer_violations = []` on both runs** — 0 outcome-table reads, 0 experiment/nullifier/gate-receipt writes. Corroborated by an AST static guard over the module's `.execute()`-family literals and by the disposable schema carrying no outcome column.

## Deterministic rebuild

Two independent full builds from the same frozen inputs/specification, ordered by `anchor_id` before hashing:

| Check | Value |
|---|---|
| Content hash (bookkeeping-excluded) | `5e9ee58cd9c260c2877b05ed803dbf51767ecedc579bdc90c37b5391a867bcbb` — **identical** |
| Row-manifest hash | `8e8e23ff8af6dfd1c11199f963698d4a148583fd2b9c979dffa7f4e4fdec72f2` — **identical** |
| Counts / accounting | identical |
| Disposable `.sqlite` file hash | run1 `341677679426af38…` vs run2 `227b26040fdae021…` — **differs by `created_ms` only** (bookkeeping timestamp, excluded from content hash) |
| Verdict | **`REBUILD_IDENTICAL`** |

## Numerical validation (source-integrity only, never outcome-joined)

196 exact rows: 0 non-finite values, 0 non-positive mids, and the additive identity `spread_change = current_spread − historical_spread` holds exactly (< 1e-9). `spread_change_bps_w300` ranges −0.00084 → +0.12807 bps (median +0.00018), 170 expansion / 26 compression / 0 zero. Current-quote age 0–13,682 ms (median 2 ms). No silent clip/winsorize/smooth, no future-informed normalization. These summaries are used only to detect feed corruption; they were never joined to any outcome and never used to alter the formula or window.

## Family distinctness (carried forward, no outcome analysis)

Spread dynamics measures the cost/immediacy of crossing the top of book — short-horizon widening/narrowing of the executable L1 quote state. Distinct from CVD (executed-flow imbalance), absorption (price response per forced-flow notional), basis (cross-instrument dislocation), depth imbalance (displayed quantity asymmetry), refill/pull (replenishment/cancellation), funding (carry), day trend (direction), liquidation geometry (event structure), and graveyarded OFI momentum (signed flow pressure). It is not merely volatility, trend, CVD, depth imbalance, or absorption.

## Future scientific question (drafted, not preregistered)

> Does the frozen continuous `BOOK_SPREAD_CHANGE_BPS_W300_V1` feature contain incremental information for one already-existing frozen canonical outcome on the accepted independent-cycle population?

No outcome selected, no outcome value read. Compatible existing outcome IDs (identity strings only, not values): `endpoint_return_bps@swing_24h`, `mfe_bps@swing_24h`. Outcome resolution belongs to a later preregistration gate.

## Retained evidence

`D:\eclipse_scalper\.runtime_temp\spread_rehearsal_v1\` — `rehearsal_run1.sqlite`, `rehearsal_run2.sqlite`, `rehearsal_result.json`, `manifest.json` (~462 KB total), **contains no outcome data**, retained as immutable rehearsal evidence for a future row-accounting-freeze gate (hashes in `manifest.json`).

## Verdict

**`BOOK_SPREAD_DYNAMICS_DISPOSABLE_REHEARSAL_V1_COMPLETE`**

**Readiness disposition:** `BOOK_SPREAD_DYNAMICS_DISPOSABLE_DATA_READY_FOR_ROW_ACCOUNTING_FREEZE` — this disposition authorizes **no** automatic next step.

**Recommended next gate:** `BATCH-BOOK-SPREAD-DYNAMICS-ROW-ACCOUNTING-FREEZE-V1` (do not begin automatically; still prohibits outcome access unless separately authorized).

Stopping after the rehearsal verdict. No row-accounting freeze, replay, migration, preregistration, modelling, TEST execution, subgroup/alternate-window/alternate-feature work, bucket construction, or route promotion begins without new operator instruction.
