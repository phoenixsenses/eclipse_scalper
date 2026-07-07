# S34_SPREAD_EXPANSION_COMPRESSION_READINESS_AND_CONTRACT_V1

**Gate:** SPREAD_EXPANSION_COMPRESSION_READINESS_AND_CONTRACT_V1
**Nature:** Readiness, source audit and scientific-contract work only. No TRAIN/TEST outcome access, no experiment, no gate receipt, no nullifier, no preregistration, no migration, no route promotion, no runtime/risk/execution/paper/shadow/forward/live change.
**Canonical family (per the accepted selection artifact):** `FAM_BOOK_SPREAD_DYNAMICS`
**Date:** 2026-07-07 · **Author:** Sonnet 5

---

## Family identity

Resolved from `reports/governance/NEXT_INDEPENDENT_RESEARCH_HYPOTHESIS_SELECTION_V1.md` (commit `0c976e21`), **candidate 3** (ranked 47/60, third of the three shortlisted), not invented this batch:

- **Canonical family ID / name:** `FAM_BOOK_SPREAD_DYNAMICS`
- **Hypothesis (verbatim):** *"Does pre-birth L1 spread expansion (relative to its own pre-window baseline) contain continuous incremental predictive information for `endpoint_return_bps@swing_24h`, controlling for the same frozen control set?"*
- **Mechanism:** quoted liquidity thinness (bid-ask width on the L1 book) — "quoted liquidity thinness, not executed flow."
- **Ranking:** 3rd of 3 (absorption 51, basis 47, spread 47; spread and basis tied, absorption led).

**No conflicting family identity exists in the repository** — a single frozen name (`FAM_BOOK_SPREAD_DYNAMICS`) with no competing definition. This is therefore **not** `SPREAD_EXPANSION_COMPRESSION_DEFINITION_AMBIGUOUS` on *identity* grounds (the ambiguity, below, is about the *feature definition/window*, not the family ID).

### Graveyard / prior-exposure check

- `graveyard_slash_fingerprints` (31 curated): **0 hits** for `spread`/`bid`/`ask`/`quote`/`widen`/`thin`/`liquidity`.
- `knowledge` (11 KOs): **0** spread KOs. `K-S34-BOOK-PULL-001`/`K-S34-REFILL-CTX-001` use `book_ticker` but measure **depth** (bid-quantity withdrawal/refill) — a different physical quantity than the bid-ask **price gap**; not the same family.
- `failure_archive` (22 rows): **0 direct** spread-family entries. One **incidental** mention (id=13, `MFE50 giveback separation`, `NO_EDGE`) lists `spread` as 1 of 10 candidate features in a *post-entry* (+50) single-feature MFE50-giveback rule — a different population framing, a different mechanism, and a different claim; the ruling was about the MFE50-giveback claim, **not** a terminal ruling on a pre-birth spread-dynamics predictor. Disclosed for transparency; it does not block this family.

**Family is genuinely independent and not graveyarded.**

---

## Source inventory

| Field | Value |
|---|---|
| Database / table | `data/microstructure.db:book_ticker` |
| Symbol | ETHUSDT |
| Relevant columns | `ts_ms`, `id` (PK), `bid_price`, `ask_price`, `bid_qty`, `ask_qty`, `mid_price`, `spread_pct` |
| Row count (ETHUSDT) | 2,077,780,064 (from the accepted absorption readiness audit `fc1321f5`; **not recounted** — a full `COUNT` over ~2×10⁹ rows is disallowed by the storage guardrail; MIN/MAX re-verified index-backed this batch) |
| Coverage | **2026-04-11T17:08:42.005Z → live (2026-07-07)** |
| Exchange timestamp field | **none** — a single `ts_ms` only |
| Local receipt timestamp field | `ts_ms` (local collector write time) |
| Bid/ask fields | `bid_price` / `ask_price` (+ `bid_qty`/`ask_qty`) |
| Venue | **Binance USDⓈ-M Perpetual Futures** (`fapi.binance.com` book-ticker websocket) |
| Market segment | perpetual futures — **the same segment and tradable instrument as the anchor population** |
| Update semantics | streamed best-bid/ask, continuous, sub-second (~120,000 updates per 5-minute window) |
| Deduplication rule | `id` (PK) is unique; **`ts_ms` is NOT unique** — ~75% of rows share a `ts_ms` with another row |
| Ordering guarantee | `id` monotonic with insertion; 0 out-of-order `ts_ms` in a 598,261-quote sample; no exchange-sequence field |
| Repair status | none instrumented |
| Gap status | **no gap ledger** for `book_ticker` (the `gaps` table covers only `agg_trades`/`liquidations`/`mark_prices`) — coverage/staleness measured empirically instead |
| Known-at semantics | `ts_ms` is receipt time (≥ exchange event time), so at-or-before selection is conservatively known-at-safe |
| Exact / proxy | **EXACT for spread** — L1 best bid/ask *is* the spread directly. The "L1 is proxy" caveat in `S34_MECHANISM_RESEARCH_PLAN.md` applies to **depth/absorption** inference, not to the spread itself |

**Explicitly NOT treated as exact bid/ask spread:** candles/OHLC, trade prices (`agg_trades`), mark/index/liquidation prices, or any volatility-inferred synthetic spread. None is used. `mark_prices` is a single value (no bid/ask pair) and cannot yield a spread.

### Venue and symbol identity (proven)

Exact exchange **Binance**; exact market segment **USDⓈ-M perpetual futures**; exact symbol **ETHUSDT**; exact quote currency **USDT**. The `book_ticker` ETHUSDT perp is the **same tradable instrument** as the anchor population (`ami_signal_lifecycle` is 100% ETHUSDT perp). **No** cross-venue, cross-instrument, spot-for-perp, other-exchange, or synthetic/index substitution occurs anywhere. This is a single-instrument, same-segment quote source — the ideal case, no proxy labeling required.

---

## Primary spread formula (proposed, outcome-blind)

```
mid_price  = (ask_price + bid_price) / 2
spread_bps = 10,000 × (ask_price - bid_price) / mid_price
```

| Element | Frozen value |
|---|---|
| Source / fields | `book_ticker`, `bid_price`, `ask_price` |
| Venue / symbol | Binance perp / ETHUSDT |
| Timestamp selection | latest quote **at-or-before** the target, **tie-broken by `id DESC`** |
| Max quote staleness | **5 minutes** (reused from `ami.states.engine.FEED_LIMITS["book_ticker"]=5.0`, not invented) |
| Locked book (bid==ask) | `INVALID_QUOTE_LOCKED` — flagged under its own code, never silently pooled as a normal 0-bps spread |
| Crossed book (bid>ask) | `INVALID_QUOTE_CROSSED` — excluded |
| Zero/negative price | `INVALID_QUOTE_ZERO_OR_NEG` — excluded |
| Units | bps of mid | 
| Precision | IEEE-754 double, no rounding |
| Source-quality class | `EXACT_RECONSTRUCTABLE` (no proxy tier exists) |
| Formula version | `spread-dynamics-readiness-v1` |
| Feature-availability / known-at ts | `= signal_birth_ts` |

Not selected by any outcome value.

### Quote-selection rule (the one genuine data-quality nuance)

`book_ticker` delivers multiple updates within the same millisecond: **~75% of rows share a `ts_ms`** with another row, and **~6.5% of those collisions carry a *different* bid/ask**. A naive `ORDER BY ts_ms DESC LIMIT 1` is therefore **non-deterministic**. The frozen rule adds a mandatory tie-break: `ORDER BY ts_ms DESC, id DESC LIMIT 1` — `id` is the autoincrement PK (= insertion order = the most recent update at that ms), making selection reproducible. This is a resolvable specification, not a blocker, but it **must** be frozen (it is, in `select_quote_at_or_before`, and proven by `test_duplicate_timestamp_deterministic_tiebreak_by_id`).

Ordering ts = `ts_ms`; tie-break = `id DESC`; no valid quote → `UNAVAILABLE`/`STALE` (never imputed); only a receipt timestamp exists, so **receipt time determines known-at safety** (conservative); 0 out-of-order observed; **no interpolation crosses birth**; **no future quote fills a pre-birth gap**.

---

## Data-quality audit (bounded, no full-table scan)

From a 598,261-quote sample across the pre-birth 5-minute windows of 5 anchors spanning the covered period, plus the actual quote at every one of the 324 anchors:

| Check | Result |
|---|---|
| Avg quotes per 5-min pre-birth window | ~119,652 (density is never a concern) |
| Duplicate `ts_ms` | ~75% of rows (resolved by the frozen `id`-tie-break) |
| Out-of-order `ts_ms` / receipt ts | 0 |
| bid > ask (crossed) at anchors | 0 |
| bid == ask (locked) at anchors | 0 |
| zero / negative price at anchors | 0 |
| Extreme spread spikes at anchors | none (~0.06 bps, tight liquid perp) |
| Symbol / venue / timestamp-unit mismatch | 0 |
| Silent repair / clip / smoothing applied | **none** |

No anomaly was silently repaired, clipped, or removed; every exclusion is deterministic and carries an immutable code.

---

## Anchor accounting (spread level at birth, outcome-blind)

324 anchors (LONG 220 / SHORT 104), earliest 2026-02-17, latest 2026-07-03, source `ami_signal_lifecycle` (schema v13), 0 duplicate anchors, 167 total independent cycles.

| Source-quality partition | Count | LONG | SHORT |
|---|---|---|---|
| `EXACT_RECONSTRUCTABLE` | **196** | 120 | 76 |
| `STALE_SOURCE` (>5min old quote) | 22 | 13 | 9 |
| `UNAVAILABLE_BEFORE_COLLECTION` (born before 2026-04-11) | 106 | 87 | 19 |
| `REPAIRED_EXACT` / `SOURCE_GAPPED` / `INVALID_QUOTE` / `PROXY_ONLY` | 0 | — | — |
| **Reconciliation** | **324** = 196 + 22 + 106 | | |

**196 exact rows → 97 independent cycles** (99 same-cycle duplicates collapsed). Under the repository's cycle-grouped 70/30 convention: **TRAIN ≈ 67 / TEST ≈ 30** — TEST comfortably exceeds `MIN_BUCKET_N = 20`. Known-at violations = **0** (reproduced independently twice, byte-identical). No exact/proxy pooling (no proxy tier exists).

### Window audit (candidate change-feature coverage)

For a windowed change feature, an anchor is usable only if **both** the birth quote (T) and the baseline quote (T−W) are `EXACT_RECONSTRUCTABLE`:

| Window | Both-endpoints-exact rows | Independent cycles | est. TRAIN / TEST |
|---|---|---|---|
| W60 | 196 | 97 | 67 / 30 |
| W300 | 196 | 97 | 67 / 30 |
| W600 | 196 | 97 | 67 / 30 |
| W1800 | 194 | 97 | 67 / 30 |
| W3600 | 196 | 97 | 67 / 30 |

**Coverage is window-invariant** — because `book_ticker` updates sub-second, a fresh quote exists at T−W whenever one exists at T. This is the crux of the verdict below: coverage cannot distinguish windows, **removing the one outcome-blind tiebreaker that resolved absorption's W300** (there, coverage was also flat, but a mechanism-timescale + CVD-W300-parity argument broke the tie; here that same parity argument would apply *equally* to every window, so it does not select one). No outcome metric (coefficient/correlation/MFE/MAE/WR/PnL/route/subgroup) was inspected.

---

## Known-at and no-outcome-access proof

`known_at_violations = 0` across all 324 anchors, reproduced independently twice (idempotent, byte-identical). `select_quote_at_or_before` is structurally incapable of returning a future quote (`WHERE ts_ms <= target`), with a defensive raise on any negative staleness. No future candle/snapshot, no interpolation across birth, staleness deterministically bounded (5-min), feature-availability timestamp reproducible (`= signal_birth_ts`).

**Outcome/governance access denial:** the audit module never opens `ami_lifecycle_path_observations` and never selects `endpoint_return_bps`/`mfe_bps`/`mae_bps`, nor touches `experiment_registry`/`experiment_results`/`epistemic_test_nullifiers`/`experiment_gate_receipts` — proven by an AST-based static guard (`test_module_never_executes_sql_naming_outcome_or_governance_tables`) that parses the module and inspects only string literals passed to `.execute()`-family calls (the narrower, correct check, avoiding the docstring false-positive class already fixed in the absorption rehearsal batch), plus a second real-SQL-literal guard. This batch performs no schema/data-write step, so there is nothing for a live SQLite authorizer to additionally protect.

---

## Family distinctness

| Comparison | Distinct because |
|---|---|
| `FAM_CVD_WINDOWED_TAKER_FLOW` | uses **no** trade/taker data — only resting best quotes; measures quoted width, not executed flow |
| `FAM_CASCADE_ABSORPTION_IMPACT` | absorption is price response **per unit of executed signed flow** (a trade-driven ratio); spread uses **zero** trade data |
| `FAM_SPOT_PERP_BASIS_REVERSAL` | basis compares **two markets'** prices (spot vs perp); spread is a **single-book** bid-ask gap |
| book-depth pull/refill (`K-S34-BOOK-PULL-001`/`REFILL-CTX-001`) | those measure **quantity** (bid depth) withdrawal/return; spread measures the **price** gap |
| funding | funding is a periodic **cash-flow rate**; spread is an instantaneous price width |
| day trend | trend is a **signed** single-instrument move; spread is a **directionless magnitude** |
| liquidation size/geometry | event-magnitude/shape, unrelated to resting quotes |
| graveyarded OFI momentum | order-flow-imbalance from trades; spread uses no trades |

**Why not volatility:** spread is an instantaneous cross-sectional price gap between two live quotes, not a time-series dispersion of returns. **Why not trend:** directionless magnitude, not a signed movement. **Why not order-flow imbalance:** uses no trade/taker information at all. **Why not renamed absorption:** absorption is trade-driven price-impact-per-flow; spread uses zero trade data. Spread is a genuinely distinct **liquidity-state (quoted-width)** mechanism.

---

## Future scientific question (drafted, not preregistered)

> Does a frozen, pre-birth spread expansion or compression state (Binance ETHUSDT perp L1, at `signal_birth_ts`, ≤5-minute staleness, deterministic `id`-tie-break) contain incremental information for one existing frozen outcome, on a defined independent-cycle population, controlling for the same frozen control set (`event_notional`, `session`, `day_trend_bps`)?

Possible existing outcome IDs (listed, **not read**): `endpoint_return_bps@swing_24h`; `mfe_bps@swing_24h` (established non-promotable diagnostic). No outcome value was read; no outcome selected by any statistic.

---

## Readiness assessment

| Dimension | Status |
|---|---|
| Exact anchor count | 196 |
| Exact independent cycles | 97 |
| Likely TRAIN / TEST | 67 / 30 |
| TEST meets `MIN_BUCKET_N=20` | **yes** |
| Source quality sufficient for deterministic rehearsal | **yes** (exact L1, 0 anomalies at anchors, deterministic tie-break) |
| One primary definition frozen | **no** |

**Open definition items:** (1) feature form — level vs change; (2) if change — ratio vs log-ratio vs difference; (3) baseline window — coverage-indistinguishable across {60,300,600,1800,3600}s; (4) if z-score/normalized — a full lookback/expanding-vs-rolling/min-obs/cold-start sub-contract.

---

## Verdict

**`SPREAD_EXPANSION_COMPRESSION_DEFINITION_AMBIGUOUS`**

**This is a *definition* stop, not a data stop** — stated positively and precisely:

- **NOT `BLOCKED_BY_COVERAGE`:** 97 independent cycles, TEST ≈ 30 ≥ `MIN_BUCKET_N=20`, window-invariant. Coverage is genuinely sufficient (in sharp contrast to the just-parked basis family's 38 cycles).
- **NOT `BLOCKED_BY_SOURCE_QUALITY`:** exact L1 best bid/ask, 0 crossed/locked/zero at all 196 anchor quotes, 2 ms median staleness, the sole duplicate-`ts_ms` nuance resolved deterministically by the frozen `id`-tie-break.
- **NOT `READY_FOR_DISPOSABLE_REHEARSAL`:** that verdict requires *one defensible primary definition*. The family's own named concept — spread **expansion/compression** = a windowed **change relative to a baseline** — has multiple defensible feature forms (level/ratio/log-ratio/difference/z-score) **and** a baseline window that cannot be selected outcome-blind (coverage is window-invariant; the absorption-style tiebreaker does not discriminate here). Per the batch instructions, when "multiple defensible definitions/windows remain unresolved and require operator ruling," the correct action is to *stop with the exact candidate set and request an operator ruling before rehearsal* — not to choose arbitrarily.

This mirrors the absorption precedent exactly: there, the primary window (W300) required an explicit operator ruling before the governed workflow proceeded. Here, the equivalent ruling (feature form + window) is surfaced as the readiness verdict itself. **Everything else is done** — the audit module, deterministic accounting, known-at proof, quality partitions, and focused tests are all complete and committed, so the moment the operator rules on feature-form + window, a disposable-rehearsal gate can open immediately with the contract below.

### Disposable-rehearsal contract (activates only on an operator ruling)

- **Input source:** `book_ticker` ETHUSDT, `mode=ro`, bounded per-anchor at-or-before queries only (never a full-table copy).
- **Formula:** the operator-ruled feature form over the frozen base `spread_bps`.
- **Candidate windows:** {60, 300, 600, 1800, 3600} s (all computed at rehearsal; the primary window ruled at preregistration, per the absorption precedent).
- **Anchor universe:** `ami_signal_lifecycle` 324 (LONG 220 / SHORT 104).
- **Quality classes / exclusion codes:** `EXACT_RECONSTRUCTABLE`, `STALE_SOURCE`, `UNAVAILABLE_BEFORE_COLLECTION`, `INVALID_QUOTE_CROSSED`, `INVALID_QUOTE_ZERO_OR_NEG`, `INVALID_QUOTE_LOCKED` (immutable).
- **Row-accounting identity:** `324 = EXACT + STALE + UNAVAILABLE + INVALID` per window (level-at-birth: 196 + 22 + 106 + 0).
- **Outcome access:** prohibited until a separate preregistration/execution gate. **Canonical migration:** prohibited until a later gate.

---

## Blockers (exact)

1. **DEFINITION** — feature form (level vs expansion-ratio vs log-ratio vs difference vs z-score) is unresolved and not selectable outcome-blind.
2. **WINDOW** — the baseline window is coverage-indistinguishable across all candidates; no outcome-blind tiebreaker exists.
3. **NOT a data blocker** — coverage (97 cycles) and source quality (exact, clean) are both sufficient.

## Next controlled gate

Await an operator ruling on **feature form + baseline window** for `FAM_BOOK_SPREAD_DYNAMICS`. Only then does a disposable-rehearsal gate open. **No rehearsal begins automatically.**
