# S34_SPOT_PERP_BASIS_READINESS_AND_CONTRACT_V1

**Gate:** SPOT_PERP_BASIS_READINESS_AND_CONTRACT_V1
**Nature:** Readiness and contract only. No TRAIN/TEST outcome access, no experiment, no nullifier, no preregistration, no migration, no route promotion, no runtime/risk/execution/shadow/paper/forward/live modification.
**Canonical family (per the accepted selection artifact):** `FAM_SPOT_PERP_BASIS_REVERSAL`
**Date:** 2026-07-07 · **Author:** Sonnet 5

---

## Family identity resolution

The canonical family name and hypothesis text are **not invented here** — both are already defined verbatim in the accepted selection artifact `reports/governance/NEXT_INDEPENDENT_RESEARCH_HYPOTHESIS_SELECTION_V1.md` (commit `0c976e21`), candidate 2:

> **Proposed canonical family name:** `FAM_SPOT_PERP_BASIS_REVERSAL`
> **Hypothesis:** Does the pre-birth spot-perp basis (level and/or slope) contain continuous incremental predictive information for `endpoint_return_bps@swing_24h`, controlling for the same frozen control set?
> **Predictor / outcome / population / controls:** structurally identical pattern to candidate 1 [absorption/impact], substituting a basis-level-or-slope predictor (level vs. slope choice must be frozen in the prereg, not swept)
> **Data-quality risks:** `spot_prices` (live, current) but no `mark_prices`-vs-`spot_prices` reconciliation/quality-contract exists yet in the canonical warehouse; the only prior signal is a tiny (n=15), ungoverned, pre-governance ad-hoc exploratory cut that must not be used to justify or tune the design.
> **Exact prerequisite blockers:** ... an explicit level-vs-slope design decision that must be frozen by the operator before any prereg is written.

Graveyard check: `match_graveyard()` against `basis`/`spot`/`arb`/`funding`/`perp` stems — **0 hits** in `graveyard_slash_fingerprints` (31 curated) or `failure_archive` (22 rows). No existing Knowledge Object references basis or spot-perp (`K-S34-FUNDING-LEVEL-001` is a distinct, adjacent concept — funding *rate level/velocity*, not price *basis* — see §Family distinctness). **Clean, not a graveyard retest.**

---

## 1–4: Source audit

| Source | Table | Symbol | Rows | First ts (UTC) | Last ts (UTC) | Timestamp field | Price field | Venue | Sampling | Known-at | Repair status | Gap status | Exact/proxy |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Spot trade/ticker price | `microstructure.db:spot_prices` | ETHUSDT | 65,386 | 2026-03-07T16:00:00Z | live (today) | `ts_ms` | `spot_price` | **Binance SPOT** (`api.binance.com/api/v3/ticker/price`) | ~60s poll, public REST, no auth | at-or-before, deterministic | none (no repair mechanism exists for this stream) | **no gap ledger instrumented** (`gaps` table has 0 rows for `stream='spot_prices'`) — empirically measured instead (below) | **EXACT** where fresh; the underlying value itself is a genuine last-traded/ticker spot price, not an index |
| Perp mark price (+ embedded funding) | `microstructure.db:mark_prices` | ETHUSDT | 8,793,754 | 2026-02-15T14:26:28Z | live (today) | `ts_ms` | `mark_price` | **Binance USDⓈ-M Perpetual Futures** (`fapi.binance.com`, live websocket collector — same collector already proven clean for CVD/absorption) | sub-second, continuous | at-or-before, already proven `KNOWN_AT_SAFE` (CVD/absorption precedent), re-confirmed this batch | n/a (already exact, live) | 51 recorded gap incidents (pre-existing `gaps` ledger), none intersecting the anchor population at short windows (established CVD/absorption precedent) | **EXACT** |
| Standalone funding-rate ledger | `microstructure.db:funding_rates` | ETHUSDT only | 178 | 2026-02-15T16:00:00Z | **2026-04-13T16:00:00Z (dead ~3 months)** | `ts_ms` | `funding_rate` | Binance perp (derived, `source='mark_prices_next_funding'`) | 8h funding-interval cadence | n/a — table is not live | none | not instrumented | **stale/abandoned table** — but see next row |
| Embedded funding (live, via mark_prices) | `microstructure.db:mark_prices.funding_rate` | ETHUSDT | 8,793,754 (100% non-null) | 2026-02-15T14:26:28Z | live (today) | `ts_ms` | `funding_rate`, `next_funding_time_ms` | same as mark_prices | same as mark_prices | same as mark_prices | n/a | same as mark_prices | **EXACT, live** — the standalone `funding_rates` table's death is immaterial; funding is still available live through this column |
| Index price | — | — | — | — | — | — | — | — | — | — | — | — | **does not exist as a separate table** — `mark_price` is Binance's own mark price (typically index-referenced internally by the exchange), not independently recomputed here; not silently substituted for spot |
| BTC/SOL spot & mark | `spot_prices`/`mark_prices` | BTCUSDT, SOLUSDT | 64,356 / 8,793,828 (BTC); 64,285 / 3,505,879 (SOL) | 2026-04-18-ish onward | live | same as above | same as above | same venues as above | same as above | same as above | same as above | same as above | **not relevant to the anchor population** — `ami_signal_lifecycle` is 100% ETHUSDT (verified, §Anchor accounting) |

**Venue identity, explicit:** spot and perp are the **same exchange (Binance)**, **different market segments** (spot cash market vs. USDⓈ-M perpetual futures) — a standard, textbook same-exchange spot-perp basis construction, **not** a cross-venue comparison. No index price is silently substituted for spot anywhere in this audit; no venue mixing occurs.

**Collector provenance (from source code, `data/oi_spot_poller.py`):** *"spot_prices (producer 2026-06-05'te olmus)"* — the poller's own header comment discloses that the `spot_prices` producer **died on 2026-06-05** and this script exists specifically to revive it (`"Olu tablolari canlandirir"` — "revives dead tables"). This is independent, first-party confirmation of the large gap measured empirically below — not a new discovery, an already-documented operational incident.

**Ungoverned prior computation (precedent, not reused as-is):** `tools/s34_mechanism_feature_store.py` already computes `basis_spot_bps = (mark_price - spot_price) / spot_price * 1e4` and a `basis_spot_slope` (delta between two basis readings) into the ungoverned `mechanism_store.sqlite` — the same formula shape proposed below, reused as a *mechanical* precedent only, never for its historical values. A separate, unrelated, ungoverned table `basis_reversion_candidates` (1,430 rows, `microstructure.db`) belongs to a different exploratory script (`tools/research_funding_extreme_mean_reversion.py`) and carries real trading-outcome columns (`long_return`, `long_win`) — **not opened for its outcome columns by this audit**, noted only for completeness/family-distinctness.

---

## 5–6: Basis definition and known-at contract (frozen, outcome-blind)

**Formula (proposed, `PROPOSED_NOT_YET_ACCEPTED`, matching the absorption/impact family's own precedent status before its own rehearsal):**

```
basis_bps_w = 10,000 × (mark_price(T) - spot_price_at_or_before(T)) / spot_price_at_or_before(T)
```

| Element | Frozen value |
|---|---|
| Price types | `mark_price` (Binance perp mark) and `spot_price` (Binance spot ticker) — **not** index, **not** a different venue |
| Venues | both Binance (spot cash market; USDⓈ-M perpetual futures) |
| Symbol mapping | `ETHUSDT` spot ↔ `ETHUSDT` perp (identical ticker string, same base/quote pair — no cross-symbol mapping ambiguity) |
| Timestamp alignment rule | `mark_price` sampled at `T = signal_birth_ts` (already `KNOWN_AT_SAFE`, sub-second); `spot_price` sampled via the **same at-or-before convention** already used throughout this codebase (`ami.states.engine.StateEngine._px`, absorption/impact's `fetch_mark_price_at_or_before`) — never a future spot sample |
| Maximum staleness tolerance | **10 minutes**, reused verbatim from the already-established `ami.states.engine.FEED_LIMITS["spot_prices"] = 10.0` (minutes) convention — not invented for this readiness batch |
| Direction/sign interpretation | positive = perp trades above spot (contango); negative = perp trades below spot (backwardation) — standard convention, not direction-flipped for LONG/SHORT |
| Units | bps of spot price |
| Known-at timestamp | `feature_available_ts_ms = signal_birth_ts` (identical CHECK-constrained pattern to every prior W-series family) |
| Missing-data rule | if no spot sample exists at or before `signal_birth_ts` at all → `SOURCE_ABSENT_BEFORE_COLLECTION` (excluded, immutable reason); if a sample exists but exceeds the 10-minute staleness tolerance → `SOURCE_STALE_BEYOND_HEALTHY_AGE` (excluded, immutable reason); never imputed, never interpolated across the anchor timestamp |
| Source-quality class | `EXACT_RECONSTRUCTABLE` only where both legs are fresh; **no proxy tier exists in this proposal** — there is no lower-fidelity spot substitute available (no index table, no cross-venue fallback) |
| Formula version | `spot-perp-basis-readiness-v1` |

**This formula was not selected by outcome behavior.** It is the direct, textbook definition of relative basis (matching the operator's own permitted-structure example), and its exact shape (numerator/denominator, bps scaling) mirrors the one pre-existing mechanical precedent (`s34_mechanism_feature_store.py`), reused for continuity, never for its historical values.

---

## Temporal features (minimal candidate, no feature grid)

**Recommended single primary candidate:** `basis_bps` **level** at `signal_birth_ts` (the raw formula above, evaluated once, at the anchor timestamp — the first bullet of the operator's own permitted list).

**Why level, not slope, as the recommendation:** the level is the simplest, single-window, zero-additional-design-parameter quantity — it requires no second timestamp, no lookback window choice, and therefore introduces zero extra researcher degrees of freedom beyond the family's own existence. A slope/change quantity would additionally require a second, separate window-choice decision (mirroring exactly the situation that required an explicit operator ruling for the absorption/impact family's W300 window) — the selection artifact itself flags "level vs. slope choice must be frozen in the prereg, not swept" as an **explicit, named prerequisite blocker requiring an operator decision**, not something this readiness batch resolves unilaterally. This recommendation is offered as an outcome-blind starting proposal only; the level-vs-slope decision remains open, exactly like the window decision was before the operator's own ruling for absorption/impact.

**Other mechanistically justified quantities, explicitly deferred as future, independently-preregistered children (not proposed now):** pre-birth basis change (slope) over a fixed window; basis compression/expansion (rate of |basis| change); a TRAIN-only basis z-score; funding proximity (`next_funding_time_ms - signal_birth_ts`, already known at signal birth via the same `mark_prices` row — computable, but a distinct mechanism from *price* basis, not bundled into this proposal).

---

## Window policy

No new "window" concept is needed for the **level** feature (it is evaluated at a single anchor timestamp, not over a lookback interval) — the only "window" question is the **staleness tolerance** for the spot leg, already resolved above by reusing the existing `FEED_LIMITS["spot_prices"]=10min` convention (an outcome-blind, already-established, non-invented value). If a future preregistration pursues the **slope** variant, its own lookback window would need the same kind of outcome-blind, mechanism-grounded ruling process already used for W300 — **not decided here**, consistent with "stop at readiness with exact options" where genuine ambiguity remains.

---

## 6–7: Anchor accounting (outcome-blind, reproducible)

Read-only, this session, against the real, live databases. No outcome column was ever selected (verified statically, `tests/test_ami_research_spot_perp_basis_readiness_audit.py::test_module_never_executes_sql_naming_the_outcome_table`).

| Quantity | Count |
|---|---|
| Total anchors (`ami_signal_lifecycle`) | 324 (LONG 220 / SHORT 104), 100% `ETHUSDT` |
| `SOURCE_ABSENT_BEFORE_COLLECTION` (no spot sample at all before `signal_birth_ts`) | **49** (LONG 41 / SHORT 8) |
| `SOURCE_STALE_BEYOND_HEALTHY_AGE` (a spot sample exists, but >10min stale) | **221** (LONG 141 / SHORT 80) |
| `EXACT_RECONSTRUCTABLE` (spot sample fresh, ≤10min) | **54** (LONG 38 / SHORT 16) |
| Reconciliation | 49 + 221 + 54 = 324 ✓ |
| Independent cycles among the fresh (`EXACT_RECONSTRUCTABLE`) subset | **38** (16 duplicate same-cycle signals collapsed) |
| Perp (`mark_price`) absent rows | 0 |
| Perp stale (>10s) rows | 0 |

**The single dominant cause of the 270 non-fresh rows:** a ~26.97-day `spot_prices` collector outage, **2026-06-05T15:59:11.295Z → 2026-07-02T15:12:58.399Z** (matches the poller script's own disclosed incident, §Source audit). 159 of the 324 anchors were born inside this exact window and therefore have no spot sample fresher than up to ~27 days. A further 49 anchors predate `spot_prices` collection entirely (before 2026-03-07T16:00:00Z). The remaining ~13 non-fresh anchors are scattered across smaller (1–100 minute) gaps, none individually large but collectively pushing several anchors past the 10-minute tolerance.

**38 independent cycles is a hard, disqualifying number for any future preregistration on the existing population**: a cycle-grouped 70/30 chronological split of 38 cycles yields **TRAIN≈27 / TEST≈11** — TEST would already fall below the `MIN_BUCKET_N=20` threshold this codebase applies uniformly across every prior W-series/CVD/absorption family, **before any outcome-eligibility gate (e.g. `swing_24h observation_status='OK'`) is even applied**, which would shrink it further.

---

## Lookahead and known-at proof

| Check | Result |
|---|---|
| All spot/mark observations at or before `signal_birth_ts` | ✅ `nearest_at_or_before()` is structurally incapable of returning a future timestamp (bisect only searches left); independently re-verified over the full materialized 324-row result set — 0 violations |
| No future funding realization used | ✅ this proposal uses `mark_price` only, not `funding_rate`/`next_funding_time_ms` (funding proximity is explicitly deferred, §Temporal features); when/if pursued, it would use `next_funding_time_ms` (a forward-looking *scheduled* time, already present in the row *as of* signal birth — known-at-safe by construction, not a lookahead) |
| No future candle close used | ✅ no candle/OHLC source is used anywhere in this proposal |
| No interpolation crosses signal birth | ✅ `nearest_at_or_before` never interpolates — it selects a single, real, past-or-equal sample or `None` |
| No nearest-neighbor match selects a post-birth observation | ✅ same structural guarantee as above (this is a strict at-or-before search, never a true "nearest," which could otherwise select a later sample) |
| Source staleness bounded by a frozen tolerance | ✅ 10 minutes, reused from `FEED_LIMITS`, enforced deterministically per row |
| Availability timestamp deterministic | ✅ `feature_available_ts_ms = signal_birth_ts`, identical to every prior family |
| Runtime access-control mechanism | The absorption/impact family's SQLite-authorizer pattern (`SQLITE_DENY` on the outcome table) was evaluated for reuse here; this readiness batch performs no schema/data-write step to protect (pure read-only accounting), so the authorizer was not additionally installed — the equivalent proof here is the AST-based static guard (`test_module_never_executes_sql_naming_the_outcome_table`) plus the simple structural fact that `ami_lifecycle_path_observations` is never opened by any connection this module creates (0 references in any `.execute()`-style call, verified) |

**`known_at_violations = 0`** across all 324 anchors, both legs, reproduced independently twice in this session (idempotent, byte-identical accounting both times — `tests/test_ami_research_spot_perp_basis_readiness_audit.py::test_anchor_accounting_idempotent_across_two_independent_runs`).

---

## Family distinctness

| Family | Source table(s) | Mechanism | Relationship to spot-perp basis |
|---|---|---|---|
| `FAM_CVD_PRIMARY_LONG_REVERSAL` (closed, `NO_RELIABLE_ASSOCIATION`) | `ami_cvd_windowed_flow` (`agg_trades`) | net signed aggressive taker-flow notional | Different source table entirely (no price-comparison of any kind); flow *quantity*, not price *relationship* |
| `FAM_CASCADE_ABSORPTION_IMPACT` (closed, `NO_RELIABLE_INCREMENTAL_ASSOCIATION`) | `ami_absorption_impact_windowed_flow` (`agg_trades` + `mark_prices`) | price response *per unit of signed flow* (Kyle-λ-style) | Uses `mark_prices` too, but as the *numerator of a price-impact ratio driven by flow*, never compared against a second, independent price series (`spot_prices` is never opened by that family's code) |
| Book-depth proxies (`K-S34-BOOK-PULL-001`/`K-S34-REFILL-CTX-001`) | `book_ticker` | order-book depth withdrawal/refill (a liquidity-state variable) | Different source table (`book_ticker`, not `spot_prices`), different physical quantity (queued liquidity, not a traded/quoted price level) |
| Funding-only (`K-S34-FUNDING-LEVEL-001`, `RECOMPUTE_REQUIRED`) | `funding_rates` / `mark_prices.funding_rate` | the perpetual funding *rate* itself (a periodic cash-flow mechanism) | Related (funding and basis are theoretically linked via cost-of-carry arbitrage) but **measures a different quantity** — a rate paid between counterparties, not a price-level difference between two markets. This proposal explicitly excludes funding (deferred to a future, separate child) |
| Day trend (`compute_day_trend_bps`, an existing control) | `mark_prices` | intraday directional drift of a single instrument | Single-instrument trend, not a two-instrument (spot vs. perp) relative-pricing comparison |
| `S34_ORDERFLOW_LEAD` (graveyarded OFI momentum) | `agg_trades` | standalone, all-timestamp order-flow-imbalance momentum, net-of-cost economic claim | Different source table, different (all-timestamp, not event-anchored) population, different (economic/fee-net) claim type — already distinguished from CVD/absorption on the same grounds; distinguished again here for completeness |

**Why genuinely distinct:** spot-perp basis is the only candidate among all of these that is a **relative-pricing / arbitrage-pressure signal between two independently-priced markets for the same asset** — no other family in this codebase compares two separate price series against each other. It answers a structurally different question ("is the perp trading rich or cheap relative to the underlying cash market, and does that predict mean-reversion pressure") than flow-quantity (CVD), price-impact-per-flow (absorption), liquidity-state (book-depth), or single-instrument trend (day trend) families.

---

## Scientific contract (drafted, not preregistered)

> Does a frozen pre-birth spot-perpetual basis **level** state (Binance ETHUSDT spot vs. perp, at `signal_birth_ts`, ≤10-minute staleness) contain incremental information for an existing, frozen outcome, on a defined independent-cycle population, controlling for the same frozen control set already established (`event_notional`, `session`, `day_trend_bps`)?

**Possible existing outcome IDs** (listed, not read for their values):

- `endpoint_return_bps@swing_24h` (`ami_lifecycle_path_observations`, effective/corrected selection) — the same outcome already reused by the closed CVD and absorption/impact families.
- `mfe_bps@swing_24h` — already established as the standard `NON_PROMOTABLE_DIAGNOSTIC` secondary check in both prior families; could in principle become primary for a differently-framed future hypothesis, but is not proposed as primary here (no outcome value of either was read to make this listing).

No outcome was selected by performance; this list is copied from the existing, already-frozen outcome-identity catalogue used by every prior W-series/CVD/absorption family.

---

## Readiness verdict

**`SPOT_PERP_BASIS_BLOCKED_BY_COVERAGE`**

**Basis for this verdict:**

1. **270 of 324 anchors (83.3%)** have no source-quality-adequate spot price at `signal_birth_ts` under the already-established, non-invented 10-minute staleness tolerance — 49 predate collection entirely, 221 are stale (dominated by one ~27-day collector outage that alone affects 159 anchors, disclosed independently by the collector script's own source comments).
2. The 54 fresh anchors collapse to only **38 independent cycles** — a cycle-grouped 70/30 split would leave a TEST fold (~11 cycles) already below the `MIN_BUCKET_N=20` floor this codebase applies uniformly, **before** any outcome-eligibility gate is even applied (which would shrink it further, matching the ~88% eligibility rate seen in CVD/absorption).
3. This is **not** a proxy-quality situation (`SPOT_PERP_BASIS_READY_WITH_PROXY_LIMITATION` does not apply) — there is no lower-fidelity spot substitute to fall back to (no index table, no alternate venue); the data for the 270 excluded anchors simply does not exist or is many days stale, not merely lower-fidelity.
4. This is **not** a definition ambiguity (`SPOT_PERP_BASIS_DEFINITION_AMBIGUOUS` does not apply) — the formula itself is unambiguous and outcome-blind-selectable (§Basis definition); only the level-vs-slope *feature* choice remains open, and that is a downstream, future preregistration decision, not a blocker to readiness itself.
5. **Perp-side data (`mark_prices`) is not the problem** — it is exact, essentially gap-free, and already proven `KNOWN_AT_SAFE` by two prior families. The blocker is entirely on the spot leg.

**Forward-looking note (does not change this verdict):** `spot_prices` collection has resumed and is live and current as of today (2026-07-07); any *future* anchor born from 2026-07-02 onward would have excellent staleness characteristics. This means the family may become retrospectively testable as the anchor population grows forward in time — but the **existing 324-signal population, today, is not adequate**.

---

## Next gate

Do not begin disposable rehearsal without new, separate operator instruction. If pursued in the future, the immediately preceding blocker (independent-cycle sample sufficiency) should be re-measured against the then-current anchor population before any further readiness work — a coverage re-check, not a new contract, would likely suffice if enough new anchors have since accumulated with clean post-2026-07-02 spot coverage.

## Blockers (exact, for the record)

1. **Coverage**: only 38/324 anchors' independent cycles have fresh (≤10min) spot data — below any usable split threshold.
2. **The 27-day collector outage** (2026-06-05 → 2026-07-02) is the dominant, single cause, already disclosed in the collector's own source code — not a data-quality contract gap to be built, but a genuine historical data hole that cannot be reconstructed after the fact (no venue-side historical spot-ticker archive is used by this repository).
3. **Level-vs-slope feature design** remains an open, operator-level decision for any future preregistration (not a blocker to this readiness/contract batch itself, since the readiness verdict is coverage-blocked regardless of which variant is eventually chosen).
