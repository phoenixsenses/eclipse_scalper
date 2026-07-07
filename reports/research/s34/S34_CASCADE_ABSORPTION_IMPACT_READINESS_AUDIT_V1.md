# S34_CASCADE_ABSORPTION_IMPACT_READINESS_AUDIT_V1

**Gate:** BATCH-CASCADE-ABSORPTION-IMPACT-CANONICAL-BRIDGE-READINESS-AND-CONTRACT-V1
**Nature:** Readiness audit and contract definition only. No preregistration, no experiment ID, no nullifier action, no TEST access, no model, no migration, no schema change.
**Canonical family (correct spelling):** `FAM_CASCADE_ABSORPTION_IMPACT` (the selection artifact's typo `FAM_CASCADE_ABSORMPTION_IMPACT` is recorded here only as a known alias of the same selection decision — the immutable selection artifact itself, `reports/governance/NEXT_INDEPENDENT_RESEARCH_HYPOTHESIS_SELECTION_V1.md`, commit `0c976e21`, is not edited).
**Date:** 2026-07-07 · **Author:** Sonnet 5

All facts below were gathered read-only (`mode=ro` SQL against the real database files, file reads of real code/reports). No feature, window, or direction was chosen using any outcome value or apparent profitability — every number in Phase 6 is a coverage/count fact, never a performance number.

---

## PHASE 1 — Canonical family reconciliation

| Item | Path | Identity | Status | Source data | Outcome exposure | Relationship to `FAM_CASCADE_ABSORPTION_IMPACT` |
|---|---|---|---|---|---|---|
| `tools/research_s34_wave_absorption.py` + `S34_WAVE_ABSORPTION.{md,json}` | `tools/`, `reports/research/s34/` | ad hoc, no family/experiment ID | Ungoverned exploratory (no verdict token found; N=54, never registered) | `book_ticker` snapshot at anchor + `mark_prices`/`liquidations` | Descriptive N/WR/bps cuts only, no frozen outcome ID | **Distinct child concept**: book-depth/imbalance state classifier ("absorbed"/"vacuum"/"mixed"), not a price-impact-per-notional measure |
| 11 sibling `research_s34_*absorption*.py` scripts + their 12 report families in `S34_ALL.db` | `tools/`, `reports/research/s34/`, `S34_ALL.db.research_results` | ad hoc | Ungoverned exploratory, no governance verdict token found in any | `book_ticker` primarily; `S34_CONTINUOUS_ABSORPTION_REGRESSION` pools raw Pearson correlations (\|r\|<0.12 for all tested book-depth features) | Descriptive only | **Same family as above** (book-depth), not this one |
| `mechanism_store.sqlite.events.fl_{pre10,pre1,post1,post5,post10}_impact` | `reports/research/s34/mechanism_store.sqlite` (836 rows) | ungoverned feature columns, no Knowledge Object, no experiment | Computed, never analyzed for separator behavior (absent from `tools/s34_mechanism_taxonomy.py`'s tested feature list and from `S34_MECHANISM_TAXONOMY.md`'s results) | `agg_trades` (`is_buyer_maker`) + `mark_prices` return; formula = `\|mret\| / (total_notional/1e6)` — **unsigned total notional, not signed flow** | None | **Closest prior computation of the same concept**, but (a) uses total not signed notional (methodologically different from a Kyle-λ regression on signed flow), (b) anchor universe is ETH SELL ≥$100K (wider/different threshold than the canonical warehouse's population), (c) pre-warehouse, ungoverned, ~2 days after the original `S34_MECHANISM_RESEARCH_PLAN.md` (2026-07-02) that specified it |
| `S34_MECHANISM_RESEARCH_PLAN.md` §"FLOW" line 42 | repo root | canonical mechanism plan, not an experiment | Phase 1 (feature store) marked done; Phase 2 (taxonomy testing) explicitly did not include `impact` | n/a (plan document) | n/a | **Direct textual precedent/origin** of the concept (`impact = \|Δfiyat\| / $1M notional (Kyle-λ proxy — absorption ölçüsü)`) — this audit operationalizes that same idea into a canonical, governed form |
| `K-S34-BOOK-PULL-001` (Knowledge Object, `HOLDOUT_VALIDATED`) | `data/ami/knowledge.sqlite.knowledge` | existing KO, `RECOMPUTE_REQUIRED` under CT-005 | mechanism field literally `"LIQUIDITY_REMAINS_AND_ABSORBS_FORCED_FLOW"`; source_tables `["book_ticker","liquidations","mark_prices"]` — **no `agg_trades`** | claim: low pre-cascade bid-depth withdrawal associated with +70bps reversal, WR 70.7% | evidence_level 3/5, `permitted=[RESEARCH_ONLY, SHADOW_ALLOWED]` | **Sibling family, not the same mechanism**: this KO's "absorption" is order-book depth *holding* (a state variable), not executed-trade price-impact-per-notional (a flow-response variable). Both are legitimately called "absorption" in prose, but measure different evidence (static book depth vs. dynamic price response to signed flow) — **not silently merged here** |
| `K-S34-REFILL-CTX-001` (`REGIME_LIMITED`) | same | existing KO, `AFFECTED` under CT-005 | book_ticker-only, "refill" (liquidity return after withdrawal), regime-sign-flipped | n/a | — | Distinct, adjacent family (liquidity *dynamics* over time, not price response to a given flow) |
| `K-S34-MECH-COMPOSITE-001` (`HOLDOUT_VALIDATED`) | same | existing KO, `RECOMPUTE_REQUIRED` | 7-factor composite (funding, rv, retail-size, OFI, two-sided, refill, pull) — **does not include `impact`** | claim: composite≥4 → TEST WR 82.6% | evidence_level per KO record | Adjacent/parent-level composite; does not contain or supersede this family |
| `failure_archive` (22 rows) / `graveyard_slash_fingerprints` (31 rows) | `data/ami/knowledge.sqlite` | — | **Zero hits** for `absorption\|impact\|kyle\|exhaustion\|aggression` (full-table scan, both tables, all columns, stems included) | n/a | n/a | **Clean — not graveyarded under any name** |
| `S34_ORDERFLOW_LEAD` graveyard fingerprint ("OFI momentum") | `graveyard_slash_fingerprints` | graveyarded | standalone all-timestamp OFI-quantile momentum, net-of-cost, dead on 3 symbols | `agg_trades`-derived OFI | — | **Different family** (net order-flow-imbalance momentum, not price-impact-per-notional); already distinguished from the closed CVD test on the same grounds; distinguished again here for completeness — not the same fingerprint, no overlap |
| "Exhaustion" | whitepaper §29 SHORT-genesis feature list only | **not implemented anywhere** | No computed feature, no KO, no failure_archive row | n/a | n/a | Not a usable synonym or child of this family today — a separate, currently unimplemented concept |

### Identity ruling

**`FAM_CASCADE_ABSORPTION_IMPACT` is a genuinely new canonical family.** It is not an existing family under another name (it is methodologically distinct from `K-S34-BOOK-PULL-001`/`K-S34-REFILL-CTX-001`'s book-depth-state "absorption," and distinct from `S34_ORDERFLOW_LEAD`'s graveyarded OFI-momentum). It has one ungoverned prior computation (`mechanism_store.sqlite.fl_*_impact`) that is a useful precedent but not a reusable canonical artifact as-is (wrong anchor universe, unsigned-notional formula, no quality contract, no schema). No child-hypothesis enumeration is required at this stage: the family is scoped narrowly enough (one mechanical construct — price response per unit of signed aggressive notional, pre-birth) to support a single future preregistration; direction (LONG/SHORT/pooled-with-separate-strata) is a Phase 5 anchor-universe decision, not a reason to split into parent/child families.

---

## PHASE 2 — Source and data-path audit (verified read-only, live, 2026-07-07)

| Source | Path | Table | Coverage (symbol/time) | Rows | Resolution | Provenance | Duplicate/gap policy | Known-at | Historical/forward | Exact/proxy |
|---|---|---|---|---|---|---|---|---|---|---|
| Aggressive trades (primary) | `data/microstructure.db` | `agg_trades` | ETHUSDT, 2026-02-15T14:26:27.967 → 2026-07-06T21:09:26.704 (live) | 175,748,566 (ETHUSDT) / 391,235,560 (all 3 symbols) | trade-level (`ts_ms`, `id`) | live collector | 20 recorded gap records for this stream (18 closed, **2 open-ended/unresolved**, both at the 2026-04-24 R1 regime boundary) — see below | native, per-trade `ts_ms`; usable as-is for a `[T-W,T]` pre-birth window | historical + forward (live) | native = exact; the 20 gap windows would need the same repair/quality-contract discipline as CVD before being called exact |
| Repaired trades (gap-fill only) | `data/ami/canonical.sqlite` | `ami_agg_trades_repaired` | ETHUSDT only, 8 disjoint spans (2026-03-06→2026-06-16), ~2,100s total | 40,934 | trade-level | Binance REST repair, `EXACT_RECONSTRUCTED`×8 | none of the 8 spans intersect any of the 324 signals' pre-birth windows (verified) | n/a to this family as currently scoped | historical only | exact (repair-verified) |
| Book L1 | `data/microstructure.db` | `book_ticker` | ETHUSDT, 2026-04-11T17:08:42.005 → 2026-07-06T20:39:12.139 | 2,077,780,064 | per-update | live collector | **0 gap records for this stream at all** (gap-detection not instrumented for it, or never fired — either way, no ledger to consult) | usable only for signals born after 2026-04-11 | historical + forward | proxy-grade for absorption (L1 only, no L2 depth) |
| Mark price | `data/microstructure.db` | `mark_prices` | ETHUSDT, 2026-02-15T14:26:28 → 2026-07-06T20:44:12 | 8,784,962 | ~1s | live collector | 51 gap records (mostly pre-R1) | usable for price-return computation over the same `[T-W,T]` window | historical + forward | exact for mark-price series itself |
| Liquidation anchors | `data/ami/canonical.sqlite` | `ami_events` | ETHUSDT, 100% (1 symbol), 252 rows, 2026-02-17T14:42:45.459 → 2026-07-03T19:15:57.545 | 252 | event-level | canonical, already governed | governed (existing) | already point-in-time-safe | historical | exact (REAL_LIQUIDATION, existing contract) |
| Signal/cycle identity | `data/ami/canonical.sqlite` | `ami_signal_lifecycle`, `ami_cycles` | 324 signals (LONG 220/SHORT 104), 167 independent cycles | 324 / 167 | signal-level | canonical, already governed | governed (existing) | `signal_birth_ts` already frozen | historical | exact |
| Source-quality precedent (template, not this family's contract) | `ami/cvd/cvd_source_quality_contract_v1.py` | n/a (code) | n/a | n/a | n/a | frozen for CVD | 5-status fail-closed decision function (`classify_window`), regime boundaries R0-R3 | `feature_available_ts_ms = signal_birth_ts`, `known_at_classification` frozen literal `KNOWN_AT_SAFE` (CHECK-constrained) | template only | n/a |

**Two unresolved (open-ended) `agg_trades` gap records** (`start_ts_ms=1777010531825` / `1777026446426`, both `resolved_bool=0`, `end_ts_ms=NULL`, both landing 2026-04-24T06:02–10:27 UTC, coinciding with the existing R1→R2 regime-boundary language in `cvd_source_quality_contract_v1.py`): these are **not silently treated as closed**. Verified read-only: zero of the 324 signals have a pre-birth window (any tested size, 60–3600s, with a conservative 24h extension past the unresolved start applied) that intersects this period — the nearest signal births are 2026-04-21T22:03 / 04-24T22:29 / 04-24T23:18, all outside every tested window's reach into the unresolved zone. **This is disclosed as an open item for the future quality contract to formally classify (likely `SOURCE_COVERAGE_UNRESOLVED` for any window that ever does reach it), not as a currently-blocking condition** since no signal in the current 324-row population is affected.

### `book_ticker` coverage vs. population (Phase 5 relevance)

| Population | Signals born before `book_ticker` coverage began (2026-04-11T17:08:42) | Total | Fraction |
|---|---|---|---|
| All signals | 106 | 324 | 32.7% |
| LONG | 87 | 220 | 39.5% |
| SHORT | 19 | 104 | 18.3% |

A book-depth-based measure would structurally exclude ~40% of the LONG population. **This is why the primary bridge definition (Phase 3/8) is trades-based (`agg_trades`), not book-based** — `agg_trades` has covered the entire population with 2-day margin before the earliest signal since before this project's data collection began in its current form.

---

## PHASE 3 — Measurement definition inventory

| Definition | Formula | Units | Sign | Window | Known-at | Missing-data | Numerical stability | Measures | Exact/proxy | Prior usage |
|---|---|---|---|---|---|---|---|---|---|---|
| `research_s34_wave_absorption.py`'s book-state classifier | composite cut on `spread_pct`, `book_imbalance`, `bid_depth_usd` vs. median/quartile thresholds | categorical (`absorbed`/`vacuum`/`mixed`) | n/a | at-anchor snapshot | anchor ts | N/A (requires book_ticker at that instant) | threshold-based, not continuous | book-depth state, a **static** absorption proxy | proxy (L1 only) | ad hoc, N=54, never governed |
| `mechanism_store.sqlite`'s `fl_*_impact` (existing, ungoverned) | `\|mret_w\| / (total_notional_w / 1e6)` | bps per $1M | unsigned (absolute value of both numerator and effectively the denominator) | pre10/pre1/post1/post5/post10 (fixed, relative to event ts) | tied to mechanism_store's own event ts, not the canonical `signal_birth_ts` | `None` if `tot<=0` or return missing | division by near-zero notional possible (no floor imposed in the reviewed code) | **mixture of impact and raw activity level** (total notional includes both directions, so it does not isolate the *net* flow's price effect) | proxy (ungoverned store, wrong anchor universe) | computed for 836 events, never analyzed for separator behavior |
| **PROPOSED_NOT_YET_ACCEPTED primary bridge definition** (this audit) | `price_response_per_signed_notional_w = mark_price_return_bps([T-W,T]) / (signed_notional_w / 1e6)`, `signed_notional_w = Σ(taker_sign · notional)` over `[T-W,T]`, `taker_sign`: `is_buyer_maker=0→+1 (taker BUY)`, `=1→-1 (taker SELL)` (identical sign convention to `ami/cvd/windowed_taker_flow.py`) | bps of mark-price return per $1,000,000 of **net signed** aggressive notional | signed (can be negative — sign carries information: large price move per unit of flow = fragile/low absorption; near-zero = flow absorbed without moving price) | proposed: same fixed family as CVD, `{60, 300, 600, 1800, 3600}` seconds pre-birth, `[T-W, T]` inclusive both ends (is-identity reuse of `ami/cvd/windowed_taker_flow.py`'s window law) | `feature_available_ts_ms = signal_birth_ts`, `known_at_classification` fixed literal, identical pattern to CVD | listwise: a signal missing mark-price or trades coverage for the window is excluded, count reported, never imputed | must floor `\|signed_notional_w\|` at a small positive constant before dividing (frozen constant, not tuned) to avoid a near-zero-denominator blowup — **this floor value is not yet chosen and must be frozen before any rehearsal, from data distribution only, never from outcome** | genuine price-impact-per-net-flow (closer to a textbook Kyle-λ estimator than the mechanism_store predecessor, which uses total not net notional) | **exact** at the trades layer for windows ≤1800s (see Phase 6); requires the same fail-closed quality-contract discipline as CVD before being certified as such | **PROPOSED_NOT_YET_ACCEPTED** — no prior usage; rationale is mechanical (isolates the *net*-flow price response, matching the Kyle-λ literature definition more precisely than the existing ungoverned total-notional variant) and source-based (agg_trades near-total coverage), not chosen for any observed profitability |
| Secondary, descriptive-only alternate (not primary, to avoid a feature zoo) | mechanism_store's own total-notional variant, re-derived against the canonical population for direct comparability to the ungoverned prior computation | bps per $1M | unsigned | same windows | same | same | same floor issue | activity-normalized volatility, not net-flow impact | same | reported alongside the primary only as a descriptive bridge-validation check (does the canonical recomputation reproduce the same *shape* the ungoverned store showed), never as a second primary candidate |

No feature was chosen or excluded because it "looked profitable" — the exploratory `S34_CONTINUOUS_ABSORPTION_REGRESSION.md`'s own numbers (all book-depth correlations \|r\|<0.12) were read only to confirm that family is a *different* (book-based) concept, not to screen candidate impact definitions.

---

## PHASE 4 — Exact versus proxy ruling

**Ruling: `RECONSTRUCTABLE_HIGH_FIDELITY_PROXY`, pending confirmation of `EXACT_ABSORPTION_OR_IMPACT` at the rehearsal stage.**

Reasoning: the trades-level source (`agg_trades`) has native, continuous coverage with zero confirmed-gap overlap for the entire 324-signal population at every tested window ≤1800s (Phase 6). This is the same starting condition CVD's own source was in before its formal quality-contract classification (`cvd_source_quality_contract_v1.py`) was written and run — CVD's classification was not assumed, it was computed. **This audit does not call the impact source "exact" by assumption** — that requires the same fail-closed `classify_window()`-equivalent to actually run against real data (Phase 8/9, not this batch). Today's honest ruling is the intermediate tier: reconstructable to high fidelity (strong prior evidence — zero known gaps, full temporal margin), not yet formally exact-certified.

If a future rehearsal produces a mix of `EXACT_RECONSTRUCTABLE`/`SOURCE_GAPPED` signals (as CVD's did), the frozen contract (Phase 8) already requires:
- physically separate canonical tables for exact vs. any proxy representation (no shared table with a status flag alone — matching `ami_cvd_windowed_flow` vs. `_proxy`'s existing precedent),
- separate quality partitions and separate accounting,
- no pooling of exact and proxy rows in any future preregistration,
- no silent fallback from exact to proxy,
- proxy promotion into a primary research population forbidden without its own, separate preregistration.

A book-depth-based proxy (using `book_ticker`) is separately ruled **`LOW_FIDELITY_PROXY_ONLY`** for this population (L1-only, no L2 depth, ~40% LONG-population coverage gap) — it is not proposed as this family's primary evidence layer at all, only noted for completeness since `research_s34_wave_absorption.py` already explored it under a different (state-classifier) framing.

---

## PHASE 5 — Anchor universe and known-at contract

- **Signal/event family:** reuse the existing canonical `ami_events`(252)/`ami_signal_lifecycle`(324)/`ami_cycles`(167) identity chain verbatim — no new event population is proposed.
- **Direction applicability:** **direction-neutral at the measurement layer** (impact is computed per signal regardless of direction), with direction retained as a mandatory stratification variable for any future preregistration (never pooled — same discipline as every prior W-series module).
- **Anchor timestamp / signal birth timestamp:** `signal_birth_ts`, identical to CVD (no new identity concept).
- **Independent-cycle representative rule:** reuse the existing, already-frozen rule (earliest `signal_birth_ts` per `independent_cycle_id`, `NOCYCLE-<source_event_id>` fallback) — the same rule used for the closed CVD test and documented in `ami/research/w8_short_expanded_baseline.py`. No new rule is proposed.
- **Candidate feature windows:** `{60, 300, 600, 1800, 3600}` seconds pre-birth (is-identity reuse of `ami/cvd/windowed_taker_flow.py`'s `FIXED_WINDOWS_SEC`), boundary law `[T-W, T]` both ends inclusive. The `BUCKET` window (tied to birth-truncated cascade geometry) is **explicitly excluded from this family's scope** — reusing it would touch the geometry track, which remains `GEOMETRY_INFERENTIAL_RESEARCH_BLOCKED_BY_SOURCE_QUALITY`/`INFERENTIAL_PARKED_SOURCE_DEAD` and is out of scope per operator instruction.
- **Feature availability timestamp:** `feature_available_ts_ms = signal_birth_ts` (identical CHECK-constrained pattern to CVD).
- **Maximum permissible latency:** none beyond the window itself — no post-birth data of any kind enters the feature (pre-birth-only, matching CVD's own `window_end_ts_ms = signal_birth_ts` law).
- **Post-anchor information prohibited:** yes, absolutely — the same `KnownAtViolation` fail-closed enforcement pattern (reject, not filter) from `windowed_taker_flow.py::compute_window_flow` is the required template.
- **Incomplete windows:** excluded, count reported (no imputation), matching CVD's `SOURCE_GAPPED`/`SOURCE_COVERAGE_UNRESOLVED` vocabulary.
- **Overlapping cycles:** governed by the existing independent-cycle representative rule (one representative per cycle) — no new handling required.
- **Cross-symbol context:** none proposed for the primary bridge (ETHUSDT-only, matching the existing 252-event/324-signal population, which is 100% ETHUSDT).

**Parent/child scoping:** the family is narrow enough for one future preregistration (a single continuous predictor at one frozen window, controls reused verbatim from the CVD prereg's own frozen set: event_notional, session, day_trend_bps). No child-hypothesis enumeration beyond direction stratification (LONG-only, SHORT-only, or both-separately — a future preregistration decision, not decided here, and never to be decided using TEST performance).

---

## PHASE 6 — Coverage and quality accounting (outcome-blind; live counts, 2026-07-07)

**Candidate universe (all directions):**

| Quantity | Count |
|---|---|
| Total canonical anchors (`ami_events`) | 252 |
| Total canonical signals (`ami_signal_lifecycle`) | 324 (LONG 220, SHORT 104) |
| Total independent cycles (`ami_cycles`, cross-checked against `ami_signal_lifecycle` grouping) | 167 (matches exactly — reconciliation check passed) |
| Representative-cycle earliest-signal direction split | LONG-earliest 142, SHORT-earliest 25 (sums to 167) |

**Source-coverage accounting per candidate window (pre-birth `[T-W,T]`, `agg_trades`, ETHUSDT, all 324 signals):**

| Window | Signals overlapping a *confirmed* `agg_trades` gap | Signals overlapping the 2 *unresolved* (open-ended) gap records (24h-conservative bound) | Signals whose window starts before `agg_trades` collection began (2026-02-15T14:26:27.967) | Usable (reconciled) |
|---|---|---|---|---|
| 60s | 0 | 0 | 0 | 324 / 324 |
| 300s | 0 | 0 | 0 | 324 / 324 |
| 600s | 0 | 0 | 0 | 324 / 324 |
| 1800s | 0 | 0 | 0 | 324 / 324 |
| 3600s | 1 (LONG) | 0 | 0 | 323 / 324 |

**Reconciliation equation (per window, e.g. W=600s):**
`candidate universe (324) = usable representation (324) + immutable exclusions (0: gap-overlap) + unresolved quarantined records (0: none of the 324 signals' windows reach the 2 open-ended gaps)`. Exactly reconciles: 324 = 324 + 0 + 0.

For W=3600s: `324 = 323 + 1 (confirmed gap overlap, immutable exclusion) + 0`. Exactly reconciles: 324 = 323 + 1 + 0.

**This accounting is a necessary-but-not-sufficient precondition, not a final quality certification.** It only checks overlap against the `gaps` table's recorded incidents — it does not (this batch) run a full per-window cadence-proof/duplicate-detection pass (the actual `classify_window()`-equivalent computation), which is reserved for the rehearsal stage (Phase 9, stage A3). **Exact-reconstructable vs. proxy-only vs. source-gapped counts by the full quality contract do not exist yet** — reporting them as final numbers now would overclaim; this table reports only the coverage precondition that makes running that computation worthwhile.

**Exact + proxy are never added together**: no proxy source is proposed for this family's primary evidence layer (Phase 4), so no such summation risk exists in this contract.

**TRAIN/TEST population counts:** **not computed in this batch.** The existing cycle-grouped 70/30 split methodology (`compute_global_cycle_split`) is reusable without touching any outcome (it depends only on `signal_birth_ts` ordering), and applying it to a 324-signal, 167-cycle, direction-stratified population would very likely reproduce counts in the same neighborhood as the closed CVD test's own 131-cycle/91-TRAIN/40-TEST split (since it is close to the same underlying signal population) — but computing and freezing an exact split is a preregistration-time decision tied to a specific eligibility rule (which depends on the *feature's* availability, not yet formally classified per Phase 4/6 above), not a readiness-audit deliverable. Stating a specific number here would risk it being read as already-frozen when it is not.

---

## PHASE 7 — Open recompute and Knowledge-Object reconciliation

| Knowledge Object | Status | Old input manifest | Old feature definition | Old source-quality assumption | Why recompute is open | Does `FAM_CASCADE_ABSORPTION_IMPACT` depend on resolving it? | Stale/contaminated/exploratory/valid? | Prior TEST exposure? | Disposition |
|---|---|---|---|---|---|---|---|---|---|
| `K-S34-BOOK-PULL-001` | `HOLDOUT_VALIDATED`, `RECOMPUTE_REQUIRED` (CT-005) | `book_ticker`+`liquidations`+`mark_prices`, dataset_hash `s34-2026H1` | bid-depth-withdrawal state variable | pre-cycle-adjusted-N (naive event-level N, not independent-cycle N) | `anchor_to_cycle_ratio=0.66` issue (CT-005), not yet recomputed | **No** — different evidence table, different mechanism, no shared code path | Descriptive claim still directionally plausible but its **statistical N-claim is currently overstated** per CT-005; not contaminated, not invalid, just pending a controlled cycle-adjusted rerun | Yes (its own historical/ad hoc exposure, unrelated to this family) | **Keep quarantined under its own open recompute flag** — this batch does not touch it; it is not superseded, versioned, or refreshed by anything here |
| `K-S34-REFILL-CTX-001` | `REGIME_LIMITED`, `AFFECTED` (CT-005, descriptive-only, no N-claim) | `book_ticker` | refill dynamics | n/a (descriptive) | Same CT-005 batch, lower priority | **No** | Valid as a regime-limited descriptive finding | n/a | Unaffected, untouched |
| `K-S34-MECH-COMPOSITE-001` | `HOLDOUT_VALIDATED`, `RECOMPUTE_REQUIRED` (CT-005) | `liquidations`+`mark_prices`+`agg_trades`+`book_ticker` | 7-factor composite, excludes `impact` | same N issue | Same CT-005 batch | **No** — composite does not include an impact term, and this family does not propose adding one to that composite | Valid pending recompute | Yes | Unaffected, untouched |
| `mechanism_store.sqlite.fl_*_impact` (not a Knowledge Object — raw data product) | n/a | ETH SELL ≥$100K, 836 events, 2026-04-11→2026-07-02 | unsigned total-notional impact | none formally assessed (pre-warehouse) | n/a — was never promoted to KO status | **Yes, as precedent only** — the future bridge's rehearsal stage should recompute this concept against the canonical population/definition, not reuse these columns directly | Exploratory, ungoverned, never a scientific claim | No (no experiment ever read its values against an outcome) | **Recompute as a non-scientific data product** at bridge time (Phase 9, A2) — i.e., a fresh, canonical-population, signed-notional, quality-contracted feature table, explicitly *not* a resurrection of the old columns and *not* itself a Knowledge Object |

No existing Knowledge Object is overwritten, silently refreshed, or superseded by this batch.

---

## Readiness verdict

**`ABSORPTION_IMPACT_READY_FOR_DIRECT_REHEARSAL`**

Basis: the family is genuinely new and clean (Phase 1: no graveyard hit, no existing-KO conflict once book-depth "absorption" is correctly distinguished as a sibling, not the same, family). The primary evidence source (`agg_trades`) has full native coverage with zero confirmed-gap overlap across the entire 324-signal population at every candidate window through 1800s, and only a single, already-identified exclusion at 3600s (Phase 6) — there is no known repair backlog analogous to CVD's 8 confirmed gap spans that must be filled before a rehearsal can even begin. This verdict is **conditional**: it authorizes proceeding straight to a disposable rehearsal (which will itself run the actual fail-closed quality-contract classification against real data, per Phase 9 stage A3) rather than requiring a dedicated repair batch first — it is **not** a preregistration-readiness claim, since no canonical schema, quality contract, or table exists yet (Phase 8 defines that contract; it is not implemented in this batch).
