# S34_CASCADE_ABSORPTION_IMPACT_CANONICAL_BRIDGE_CONTRACT_V1

**Gate:** BATCH-CASCADE-ABSORPTION-IMPACT-CANONICAL-BRIDGE-READINESS-AND-CONTRACT-V1
**Status:** FROZEN CONTRACT — DEFINITION ONLY. No schema created, no migration run, no data written, `schema_version` remains 12. This document specifies what a future implementation batch must build; it does not build it.
**Depends on:** `S34_CASCADE_ABSORPTION_IMPACT_READINESS_AUDIT_V1.md` (same batch), readiness verdict `ABSORPTION_IMPACT_READY_FOR_DIRECT_REHEARSAL`.
**Canonical family:** `FAM_CASCADE_ABSORPTION_IMPACT`
**Date:** 2026-07-07 · **Author:** Sonnet 5

---

## PHASE 8 — Canonical bridge contract

### 1. Canonical family identity and aliases

- Canonical family name: `FAM_CASCADE_ABSORPTION_IMPACT`
- Known alias (typo, provenance-preserved only, never used in new code/schema): `FAM_CASCADE_ABSORMPTION_IMPACT` (as it appears verbatim in `reports/governance/NEXT_INDEPENDENT_RESEARCH_HYPOTHESIS_SELECTION_V1.md`, commit `0c976e21` — that artifact is immutable and is not corrected in place)
- Hypothesis-id convention for the eventual preregistration (not created in this batch): `H-CASCADE-ABSORPTION-IMPACT-<WINDOW>-V1`, following the exact naming convention of `H-CVD-PRIMARY-LONG-W300-EXACT-NET-TAKER-FLOW-NOTIONAL-V1`
- Sibling, explicitly-not-merged families: `K-S34-BOOK-PULL-001` (book-depth state), `K-S34-REFILL-CTX-001` (liquidity-return dynamics), `S34_ORDERFLOW_LEAD`/graveyarded OFI-momentum

### 2. Source package manifest

| Input | Path | Role |
|---|---|---|
| `agg_trades` | `data/microstructure.db` (mode=ro) | Primary trades source — signed notional, price |
| `mark_prices` | `data/microstructure.db` (mode=ro) | Price-return numerator |
| `ami_agg_trades_repaired` | `data/ami/canonical.sqlite` | Consulted only if a future signal's window intersects one of its 8 existing repair spans (none do today, per Phase 6 of the readiness audit — reconfirm at rehearsal time, not assumed) |
| `gaps` (`stream='agg_trades'`) | `data/microstructure.db` | Confirmed-gap ledger; the 2 unresolved (open-ended) rows must be carried forward as explicit unresolved records, never silently treated as closed |
| `ami_events`, `ami_signal_lifecycle`, `ami_cycles` | `data/ami/canonical.sqlite` | Anchor/signal/cycle identity (reused verbatim, no new population) |

### 3. Exact input tables/files

- `microstructure.db:agg_trades(id, ts_ms, symbol, price, quantity, notional, is_buyer_maker)` — read-only, symbol='ETHUSDT'
- `microstructure.db:mark_prices(id, ts_ms, symbol, mark_price, funding_rate, next_funding_time_ms)` — read-only, symbol='ETHUSDT'
- `microstructure.db:gaps(id, stream, start_ts_ms, end_ts_ms, duration_sec, resolved_bool)` — read-only
- `canonical.sqlite:ami_signal_lifecycle`, `ami_events`, `ami_cycles`, `ami_agg_trades_repaired` — read-only until the migration stage (A5)

### 4. Feature definition and units

Primary (frozen, `PROPOSED_NOT_YET_ACCEPTED` until an operator/prereg-time sign-off, per the readiness audit Phase 3):

```
price_response_per_signed_notional_w =
    mark_price_return_bps(window=[T-W, T]) / max(|signed_notional_w| / 1_000_000, FLOOR_USD_M)

signed_notional_w = Σ_{trades in [T-W,T]} (taker_sign(trade) × notional(trade))
taker_sign(trade) = +1 if is_buyer_maker=0 (taker BUY), -1 if is_buyer_maker=1 (taker SELL)
```

- Units: bps of mark-price return per $1,000,000 of net signed aggressive notional
- `FLOOR_USD_M`: a fixed, TRAIN-distribution-informed (never outcome-informed), small positive constant to prevent division blow-up near-zero net flow — value to be chosen at rehearsal time from the real `signed_notional_w` distribution, frozen before any TEST-adjacent use, documented with its derivation
- Secondary, descriptive-only, non-primary companion (bridge-validation only, per Phase 3 of the readiness audit): the legacy unsigned/total-notional variant, recomputed against the canonical population, reported only to confirm continuity with the prior ungoverned `mechanism_store.sqlite` computation — never promoted to primary

### 5. Evidence-layer classification

Per the readiness audit Phase 4 ruling: `RECONSTRUCTABLE_HIGH_FIDELITY_PROXY`, pending confirmation of `EXACT_ABSORPTION_OR_IMPACT` by an actual quality-contract run (not this batch). The eventual schema must therefore carry the **same 5-status vocabulary already frozen for CVD** (`EXACT_RECONSTRUCTABLE`, `PROXY_ONLY`, `SOURCE_GAPPED`, `SOURCE_COVERAGE_UNRESOLVED`, `UNREPAIRABLE`), computed by an impact-specific `classify_window()`-equivalent — not inherited by reference from the CVD quality table (a structurally separate table is required, §12).

### 6. Anchor universe

`ami_signal_lifecycle` (324 signals, LONG 220/SHORT 104), `ami_events` (252), `ami_cycles` (167) — reused verbatim, no new population, no new eligibility filter beyond feature-availability (never outcome-availability at this stage).

### 7. Windows

`{60, 300, 600, 1800, 3600}` seconds, pre-birth, `[T-W, T]` both ends inclusive — is-identity reuse of `ami/cvd/windowed_taker_flow.py`'s `FIXED_WINDOWS_SEC` boundary law. `BUCKET` window explicitly **excluded** from this family (would touch the blocked geometry track).

### 8. Known-at contract

- `window_start_ts_ms = signal_birth_ts - W`, `window_end_ts_ms = signal_birth_ts` (both inclusive)
- `feature_available_ts_ms = signal_birth_ts` (schema `CHECK` constraint, identical to CVD)
- `known_at_classification`: frozen literal `KNOWN_AT_SAFE`, schema `CHECK`-constrained (not a computed/inferred rule)
- Fail-closed enforcement: any trade or mark-price row with `ts_ms > window_end_ts_ms` or `ts_ms < window_start_ts_ms` entering the computation **raises**, matching `windowed_taker_flow.py::compute_window_flow`'s `KnownAtViolation` — rejection, not silent filtering
- No post-birth information may enter the feature under any circumstance

### 9. Row identity and primary key

`feature_id` (content-derived, same convention as `ami_cvd_windowed_flow.feature_id`), unique on `(symbol, signal_id, window_id, feature_definition_version)` — mirrors the existing CVD table's key discipline.

### 10. Duplicate resolution

Same discipline as `ami_agg_trades_repaired`/`ami_cvd_repair_batch_ledger`: `legacy_match_status` vocabulary (`UNMATCHED`/`MATCHED_1TO1`/`AMBIGUOUS`/`CONFLICTING`/`NOT_ATTEMPTED`) reused verbatim if any repair-sourced rows are ever mixed in; native collector rows require no dedup beyond the trade's own `id`.

### 11. Exclusion taxonomy

- `WINDOW_STARTS_BEFORE_COLLECTION_BEGAN` (0 signals affected today, per readiness audit Phase 6 — retained as a named category for future data expansions)
- `CONFIRMED_GAP_OVERLAP` (1 signal today, W=3600s only)
- `UNRESOLVED_GAP_PROXIMITY` (0 signals today; reserved, must never be silently dropped if it becomes nonzero as data grows)
- `BUCKET_WINDOW_NOT_IN_SCOPE` (family-level exclusion of the `BUCKET` window entirely, not a per-signal exclusion)

### 12. Quality-state taxonomy

New table `ami_impact_window_quality_v1` (name illustrative — not created in this batch), structurally parallel to `ami_cvd_window_quality_v1` but **physically separate**, carrying its own `quality_contract_version`, its own effective-view (`assessment_version`-latest-wins, identical pattern), and its own 5-status CHECK constraint. **Never a shared table with the CVD quality table** — different feature, different contract, different versioning lifecycle, even though the underlying trade source overlaps.

### 13. Exact/proxy physical separation

If any proxy-tier representation is ever proposed for this family (not currently — Phase 4 rules book-depth proxying out as `LOW_FIDELITY_PROXY_ONLY` and does not propose it as evidence): a physically separate table (`ami_impact_windowed_flow_proxy`-equivalent), `CHECK (evidence_layer='PROXY')`, `CHECK (descriptive_only=1)`, and an `assert_not_pooled()`-equivalent static guard — identical discipline to `ami/cvd/windowed_taker_flow.py`.

### 14. Immutable source and repair ledgers

If repair is ever needed (not currently anticipated — Phase 6 shows near-total native coverage), it must use the **same** frozen Binance REST pagination law already proven in `ami/cvd/aggtrades_repair_rehearsal.py` (id-cursor, gap-free-by-construction, measured not assumed) rather than a new repair mechanism — is-identity reuse, not reinvention.

### 15. Feature-availability timestamps

`feature_available_ts_ms` column, `CHECK (feature_available_ts_ms = signal_birth_ts)` — identical to CVD, not optional.

### 16. Content hashes and manifest roots

Any future migration must record: pre-migration canonical.sqlite sha256, post-migration sha256, a frozen source-package manifest (mirroring `data/ami/candle_repair_source_package/` and `data/ami/cvd_rehearsal_disposable_20260705/`'s precedent — checksum every raw file/query result the migration consumes), and a disposable-rehearsal content-hash proof reproduced independently before touching the real file.

### 17. Idempotency requirements

Migration/backfill code must be re-runnable with zero row-count/content-hash drift on a second run (same discipline as every prior M-00xx migration in `MIGRATION_LOG.md`).

### 18. Prohibited outcome access

No stage before an actual, separately-authorized preregistration may read `endpoint_return_bps`, `mfe_bps`, or any other outcome column for this family. The bridge/rehearsal/migration stages (A1-A5 below) touch only identity, source-quality, and feature-value tables — never `ami_lifecycle_path_observations`'s outcome columns.

### 19. Prohibited pooling

- Exact and proxy representations (if a proxy is ever added) may never be pooled in any population.
- LONG and SHORT signals may never be pooled in any future preregistration's primary analysis (direction is always a stratification variable, per the whole codebase's standing discipline).
- This family's evidence may never be pooled with CVD's `ami_cvd_windowed_flow` evidence in a single model (different family_id, different nullifier, different experiment).

### 20. Protected subsystem boundaries

`execution/`, `risk/`, `brain/`, `.env`, `tools/s34_state_machine_live_executor.py` — untouched by every stage of this contract, including the eventual migration.

### 21. Expected row-accounting equations

For each window `W`: `candidate universe (324) = usable representation + confirmed-gap exclusions + unresolved-gap-quarantine + (future) SOURCE_GAPPED/UNRESOLVED per the formal quality contract` — must reconcile exactly, every stage, every rerun (same standing law as every prior CVD/geometry migration's postflight check).

### 22. Controlled backup/restore expectations

Before any real-file migration (stage A5): full pre-migration backup under `data/ami/backups/`, hash-verified, restore-tested on a disposable copy (never the real file) before the real migration runs — identical to every prior M-00xx backup discipline.

### 23. Required disposable rehearsal

Stage A3 (below) is mandatory and must run entirely against disposable copies of `canonical.sqlite`/`knowledge.sqlite`, per the storage guardrail (`D:\eclipse_scalper\.runtime_temp`), before any real migration is proposed.

### 24. Future canonical migration acceptance conditions

A real migration (stage A5) may only proceed once: (a) the rehearsal (A3) reproduces the coverage accounting in this contract's Phase 6 table exactly or explains every difference, (b) the quality-contract classification has actually run and produced a disclosed `EXACT_RECONSTRUCTABLE`/`SOURCE_GAPPED`/etc. breakdown (not assumed), (c) row-accounting freezes cleanly (A4), (d) an operator explicitly approves the real migration (same approval discipline as every prior M-00xx).

---

## PHASE 9 — Future stage plan

| Stage | Inputs | Outputs | Success verdict | Stop conditions | Prohibited actions |
|---|---|---|---|---|---|
| **A1. Readiness/contract** (this batch) | Repository state, real DB read-only | This audit + this contract + transition proof | `ABSORPTION_IMPACT_READY_FOR_DIRECT_REHEARSAL` | — (complete) | Schema/migration/TEST access (none occurred) |
| **A2. Source repair or feature reconstruction (disposable only)** | `agg_trades`, `mark_prices`, this contract's feature definition | A disposable feature computation (impact per signal/window) against a **disposable copy** of the real DB; the `FLOOR_USD_M` constant chosen from the real signed-notional distribution (TRAIN-only conceptually, but at this stage no TRAIN/TEST split exists yet — chosen from the *whole* distribution, outcome-blind, and re-frozen once a split exists) | Feature computed for all reachable signals, coverage matches or explains deviation from Phase 6's table | Any unexplained coverage discrepancy vs. this contract | Reading any outcome column; writing to the real canonical.sqlite |
| **A3. Disposable rehearsal** | A2's feature computation + a disposable schema (`ami_impact_windowed_flow`-equivalent, `ami_impact_window_quality_v1`-equivalent) | The actual quality-contract classification (`EXACT_RECONSTRUCTABLE`/etc. breakdown), known-at violation proof (0 expected), idempotency proof, content hashes | All required validations pass, breakdown disclosed honestly (including if it's worse than Phase 6's coverage-only estimate) | Any known-at violation found; any content-hash non-determinism | Real-file writes; TEST outcome access |
| **A4. Row-accounting freeze** | A3's disposable results | A frozen, hash-stamped row-accounting document (this contract's §21 equation, filled with real numbers) | Reconciles exactly, reviewed | Any unreconciled row | Real-file writes; TEST outcome access |
| **A5. Controlled canonical migration** | A3/A4, operator approval | Real `canonical.sqlite` schema extension + backfill, backup taken and verified, `MIGRATION_LOG.md` entry (M-00xx) | Postflight matches disposable rehearsal exactly, `schema_version` bumped from 12, protected tables unchanged | Any postflight mismatch | TEST outcome access; any runtime/risk/execution file change |
| **A6. Independent preregistration** | A5's canonical feature table, existing frozen outcome/split machinery | A new preregistration document (`E-CASCADE-ABSORPTION-IMPACT-<WINDOW>-PREREG-001`-style), new family_id, new nullifier issued (not consumed) | `..._PREREGISTRATION_V1_COMPLETE`-equivalent, graveyard-clean, no TEST access | Graveyard hit without retry token; ambiguous split/outcome identity | TEST outcome access; model execution |
| **A7. One governed TEST execution** | A6's frozen spec | Exactly one confirmatory model run, nullifier consumed exactly once, one experiment_registry/results write | One of the four preregistered verdict-rule dispositions | Any code change after TEST access; any second model | Threshold scan; subgroup rescue; second TEST pass |
| **A8. Forward validation (if warranted)** | A7's result, if not null | A forward/shadow observation binding, `0/N` accumulation start | Per the standing forward-validation discipline (e.g. `E-HOUR17-FWD-001`'s pattern) | N/A at this distance | Live/paper/route promotion without its own separate sign-off |

No stage may be combined with another (per instruction: "do not combine research execution with data canonicalization"). Stages A1-A5 are pure data/infrastructure work; A6-A8 are the research track, structurally identical to how the CVD family was built (BATCH-CVD-A/B/C precedent) and then executed (the just-closed governed cycle).
