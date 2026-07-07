# S34_CASCADE_ABSORPTION_IMPACT_ROW_ACCOUNTING_FREEZE_V1

**Gate:** BATCH-CASCADE-ABSORPTION-IMPACT-ROW-ACCOUNTING-FREEZE-V1
**Purpose:** Freeze the outcome-blind absorption/impact reconstruction universe, formula, source commitments, quality partitions, exact row accounting and expected canonical migration contents. Row-accounting and migration-input freeze only.
**Depends on (source of truth, unedited):** readiness/contract commit `fc1321f5`, disposable rehearsal commit `fc43e972`.
**Date:** 2026-07-07 · **Author:** Sonnet 5

This batch performed no canonical migration, no preregistration, no experiment/nullifier action, no outcome/TEST read, no model run, and no alteration of the frozen formula, windows, or floor. Every number below was either transcribed unedited from the accepted artifacts or independently recomputed this batch by read-only queries against the two retained disposable databases (never against `microstructure.db` or a new disposable copy).

---

## Source of truth — exact committed files

### Readiness/contract commit `fc1321f5`

| File | sha256 (recomputed this batch) |
|---|---|
| `reports/research/s34/S34_CASCADE_ABSORPTION_IMPACT_READINESS_AUDIT_V1.md` | `fbef831fe828c4a8768bb01b884edac3f52059cc8f48e84f522a2adf4d0ba709` |
| `reports/research/s34/S34_CASCADE_ABSORPTION_IMPACT_READINESS_AUDIT_V1.json` | `7fea347b59b1c3721199c7bf3be5bc2f2d0daa2a50d751fde4b967aa18156c8e` |
| `reports/research/s34/S34_CASCADE_ABSORPTION_IMPACT_CANONICAL_BRIDGE_CONTRACT_V1.md` | `5acf0d532241f8f4197da1ac10951d6afec6539244422ceeba636674bdbfdb9a` |
| `reports/governance/CASCADE_ABSORPTION_IMPACT_READINESS_AND_CONTRACT_V1_STATE_TRANSITION_PROOF.md` | `981425798c7e13b4bf6250a4a636332aadea4e0285f074200468985e3de39482` |

### Disposable rehearsal commit `fc43e972` — the exact six files

| File | sha256 (recomputed this batch) |
|---|---|
| `ami/absorption/__init__.py` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` (empty) |
| `ami/absorption/cascade_absorption_impact_rehearsal.py` | `604947829105be47b0a425694104392a91b502e7bbff6b7ba2a71b3f881ec609` |
| `reports/governance/CASCADE_ABSORPTION_IMPACT_DISPOSABLE_REHEARSAL_V1_STATE_TRANSITION_PROOF.md` | `1d4ccd5d84e4b19d71ae0b3cab3b02548eef73603cab5d466dea38ba6d69d910` |
| `reports/research/s34/S34_CASCADE_ABSORPTION_IMPACT_DISPOSABLE_REHEARSAL_V1.json` | `c8ee11efbcaaa195d45b2fbd7f87b60c879bf3aff3cfae67a59f50fe0cceea82` |
| `reports/research/s34/S34_CASCADE_ABSORPTION_IMPACT_DISPOSABLE_REHEARSAL_V1.md` | `0c76beb68792354201d156ad3864aabbd28724bcbeac8b8bb7122b55681a4e1f` |
| `tests/test_ami_absorption_cascade_impact_rehearsal.py` | `d21a771e95f207ccc9791aa537861a4252df22b6e233a403b43e9dd9f770269a` |

All six recomputed hashes match the values already recorded in the rehearsal's own manifest/transition proof — no drift. No accepted artifact was rewritten by this batch.

---

## PHASE 1 — Formula and parameter freeze

Frozen exactly as validated by the accepted rehearsal (no alteration):

```
price_response_per_signed_notional_w =
    mark_price_return_bps([T-W, T]) / max(|signed_notional_w| / 1_000_000, FLOOR_USD_M)
signed_notional_w = Σ taker_sign(trade) * notional(trade), trades in [T-W, T]
taker_sign(trade) = +1 if is_buyer_maker=0 (taker BUY), -1 if is_buyer_maker=1 (taker SELL)
```

| Element | Frozen value |
|---|---|
| Canonical family ID | `FAM_CASCADE_ABSORPTION_IMPACT` |
| Formula version | `absorption-impact-rehearsal-v1` |
| Numerator | `mark_price_return_bps = (mark_price(T)-mark_price(T-W))/mark_price(T-W)*1e4`, both prices "nearest at-or-before" |
| Denominator | `max(\|signed_notional_w\|/1e6, FLOOR_USD_M)` — absolute value, not signed |
| Units | bps of mark-price return per $1,000,000 net signed aggressive notional |
| Sign convention | `is_buyer_maker=0→+1` (taker BUY), `=1→-1` (taker SELL) |
| Price source | `microstructure.db:mark_prices`, at-or-before |
| Signed-notional source | `microstructure.db:agg_trades` native, merged with `canonical.sqlite:ami_agg_trades_repaired` on window intersection (0 of 324 signals intersected any of the 8 repair spans) |
| Five fixed windows | `{60, 300, 600, 1800, 3600}` seconds, IDs `W60/W300/W600/W1800/W3600` |
| Anchor timestamp | `T = signal_birth_ts` |
| Window inclusivity | `[T-W, T]`, both ends inclusive |
| Feature-availability timestamp | `feature_available_ts_ms = signal_birth_ts` |
| Numerical precision | IEEE-754 double, no intermediate rounding |
| Zero-denominator policy | floored at `FLOOR_USD_M`, never divides by exactly zero |
| Near-zero-denominator policy | same floor, continuous |
| Source-gap policy | `CONFIRMED_GAP_OVERLAP` → immutable exclusion, never imputed |
| Duplicate policy | dedup by `agg_trade_id`; repaired wins on conflict |
| Quality taxonomy | `EXACT_RECONSTRUCTABLE / PROXY_ONLY / SOURCE_GAPPED / SOURCE_COVERAGE_UNRESOLVED / UNREPAIRABLE` |

**`FLOOR_USD_M = 0.01` ($10,000)** — frozen from the real `|signed_notional|` distribution at W=60s across all 324 signals (min=$37,006.70, p1=$93,985.10, max=$78,936,758.21), selected outcome-blind as the 1st percentile rounded down to one significant figure. It never bound on any of the 1,619 usable rehearsal rows (`floor_applied_rows = 0`, both runs, independently reconfirmed this batch, §Phase 4). It may not be retuned during migration, preregistration, or TEST execution.

**Status: `REHEARSAL_VALIDATED_FROZEN_FOR_CANONICAL_MIGRATION`** — a data-product status, not a scientific edge verdict. No formula upgrade, no alternative definition, no window/floor change was made or proposed this batch.

---

## PHASE 2 — Immutable source manifest

| Item | Value |
|---|---|
| `data/ami/canonical.sqlite` sha256 (unchanged before/during/after this freeze) | `25a56a98d02f84191aeb6ff46f81245d36bc0d635e916dbfac3e13d076bf5291` |
| `data/microstructure.db` | not hashed (~747GB); never copied; every read this batch and in the original rehearsal was a bounded per-window range query |
| Code hash | `604947829105be47b0a425694104392a91b502e7bbff6b7ba2a71b3f881ec609` |
| Contract hash | `5acf0d532241f8f4197da1ac10951d6afec6539244422ceeba636674bdbfdb9a` |
| Readiness audit hash | `fbef831fe828c4a8768bb01b884edac3f52059cc8f48e84f522a2adf4d0ba709` |
| Quality-contract hash (`classify_quality()` source, embedded in code — no separate quality-contract file exists for this family) | `9d38d748b08197ea153988c9dce4a85894d80483603edf70b93dcb09031353ea` |
| Package creation timestamp | `2026-07-07T06:13:51Z` UTC |
| Repository commit identities | `0c976e21` (selection), `fc1321f5` (readiness/contract), `fc43e972` (rehearsal) |
| Symbol coverage | ETHUSDT, 100% of the 324-signal population |
| Timestamp range | `2026-02-15T14:26:27.967` → `2026-07-06T21:09:26.704` |
| Anchor manifest | 324 signals (LONG 220 / SHORT 104), 252 events, 167 independent cycles |

### Table schema hashes (recomputed this batch, `sqlite_master.sql` text)

| Table | sha256 |
|---|---|
| `microstructure.db:agg_trades` | `4f402d7959ebc4e9d2eb51968d05c5ae7be523421473a412dfd58fab40134625` |
| `microstructure.db:mark_prices` | `9de41c03820d8143a797c69b0ff4f7139003355d0d0b9f6f2c4ff6ddd889deb6` |
| `microstructure.db:gaps` | `4ec87c2dd82391d68da92169cb44103c929d45053fcd4e4f4611eec706059493` |
| `canonical.sqlite:ami_signal_lifecycle` | `b4a5c75a6d56ce6cf16f204b12854b2268f44471f57e533fde9efd0115e90357` |
| `canonical.sqlite:ami_agg_trades_repaired` | `f50e679ee114d7615a24df6fb62478828e35fd5670f3136cc42315996868c49b` |
| `canonical.sqlite:ami_events` | `6daff1fbb63cbfaa7b0e8b2787a30af0ea87106a1c4486c9c202e51535d47fef` |
| `canonical.sqlite:ami_cycles` | `ec822d1bee2dba5d87ec3ea989d8454fddd6782fb125fe1c54919cfead375dd6` |

### Retained-evidence promotion check

The retained `.runtime_temp/absorption_impact_rehearsal_v1/` package (4 files, 2.5 MB) is promoted to **immutable rehearsal evidence** this batch: its manifest is complete, every one of the 4 files has an independently recomputed sha256 (matching the original manifest's claims exactly), it contains no outcome data (verified §Phase 7), and it is reproducible from the frozen source commitments (`REBUILD_IDENTICAL`, proven both originally and re-verified this batch, §Phase 5/10).

---

## PHASE 3 — Primary-key and row-identity freeze

Row identity: `(symbol, signal_id, window_id, feature_definition_version)` for usable rows, `(signal_id, symbol, window_id, reason_code)` for exclusions, `(signal_id, symbol, window_id, quality_contract_version)` for quality rows. The union of `(signal_id, window_id)` pairs across the usable and exclusion tables is the scientific row-identity space.

Independently re-verified this batch (read-only SQL against both retained disposable databases):

| Check | Run 1 | Run 2 |
|---|---|---|
| `absorption_impact_windowed_flow` row count | 1,619 | 1,619 |
| Distinct `feature_id` | 1,619 | 1,619 |
| `absorption_impact_window_quality_v1` row count / distinct `quality_id` | 1,620 / 1,620 | 1,620 / 1,620 |
| `absorption_impact_exclusions` row count / distinct `exclusion_id` | 1 / 1 | 1 / 1 |
| Union of `(signal_id, window_id)` across both tables | 1,620 | 1,620 |
| Duplicate `(signal_id, window_id)` within `windowed_flow` | 0 | 0 |
| `(signal_id, window_id)` pairs present in **both** `windowed_flow` and `exclusions` | 0 | 0 |

**Expected primary-key count = 1,620 — confirmed. Duplicate primary keys = 0. Conflicting rows = 0. One and only one terminal accounting state per expected row — confirmed.** No mutable timestamp is used as row identity anywhere in the schema.

---

## PHASE 4 — Per-window accounting freeze

| Window | Candidate anchors | Usable (`EXACT_RECONSTRUCTABLE`) | Excl. before-collection | Excl. confirmed-gap | Excl. unresolved-gap | Reconciled | Content hash (`windowed_flow`) |
|---|---|---|---|---|---|---|---|
| W60 | 324 | 324 | 0 | 0 | 0 | 324 | `fe1fe456bb…` |
| W300 | 324 | 324 | 0 | 0 | 0 | 324 | `487a7ca4f4…` |
| W600 | 324 | 324 | 0 | 0 | 0 | 324 | `d68eed8339…` |
| W1800 | 324 | 324 | 0 | 0 | 0 | 324 | `ffde744b2e…` |
| W3600 | 324 | 323 | 0 | 1 | 0 | 324 | `7778b457b7…` |
| **Total** | **1,620** | **1,619** | **0** | **1** | **0** | **1,620** | — |

Zero-denominator exclusions, near-zero-denominator exclusions, timestamp-invalid rows, duplicate/conflict rows, and quarantined rows are all **0** at every window (independently re-verified this batch). Reconciliation: **1,620 = 1,619 + 1 + 0**, exactly.

### The one source gap — full identity

| Field | Value |
|---|---|
| Signal ID | `SIG-e03382b4d82720185dfc870a` |
| Direction | LONG |
| Independent cycle | `CYC-f8a61eab111e474774583b9e` |
| Source event | `EVT-137f4705e3e807cda51c61fb` |
| `signal_birth_ts` | `1776783451255` ms = `2026-04-21T14:57:31.255Z` |
| Affected window | W3600 only: `[2026-04-21T13:57:31.255Z, 2026-04-21T14:57:31.255Z]` |
| Missing interval | `microstructure.db:gaps` row `id=766`, stream `agg_trades`, `2026-04-21T14:10:35.146Z → 2026-04-21T14:17:05.985Z` (390.839s), `resolved_bool=1` (confirmed/closed) |
| Immutable exclusion reason | `CONFIRMED_GAP_OVERLAP` |
| Repaired or filled this batch | No — retained as an immutable exclusion, not silently repaired |

---

## PHASE 5 — Content-hash freeze

**Hash-type distinctions:**

1. **Raw file byte hash** — sha256 over the literal bytes of a retained file. Differs between `rehearsal_run1.sqlite` (`b42972c7…`) and `rehearsal_run2.sqlite` (`e842f694…`) because of `created_ms`/`assessed_at_ms` bookkeeping timestamps.
2. **Deterministic scientific-content hash** — sha256 over a table's declared content columns (all columns except bookkeeping timestamps), `ORDER BY 1`. Identical across both runs.
3. **Semantic hash excluding non-scientific bookkeeping** — in this rehearsal, computation (2) and (3) are the same value per table, since `created_ms`/`assessed_at_ms` are the only volatile-but-non-scientific fields. Kept as two named concepts here only because the freeze instructions distinguish them; no additional exclusion beyond those two columns was needed or applied.

**Fields excluded from semantic hashing, with justification:**

| Table | Excluded column | Why it cannot affect scientific meaning |
|---|---|---|
| `absorption_impact_windowed_flow` | `created_ms` | Wall-clock row-insertion time; the measured `signed_notional`/`mark_return_bps`/`price_response_per_signed_notional` values are separate, included columns |
| `absorption_impact_window_quality_v1` | `assessed_at_ms` | Wall-clock assessment time; the classification result (`quality_status`, gap flags) is a separate, included column |
| `absorption_impact_exclusions` | `created_ms` | Wall-clock time; the `reason_code` itself is a separate, included column |

**Frozen content hashes:**

| Partition | sha256 |
|---|---|
| Usable feature rows (`absorption_impact_windowed_flow`) | `f7c834cc8ebe90708e308629f1921a050d58520ad5560422b09406a7d1ca8942` |
| Quality ledger (`absorption_impact_window_quality_v1`) | `5d1a205c7f79ca1b269307e34750c0d46dc104c8a799e9b4d01c862d307d7ba0` |
| Exclusion ledger (`absorption_impact_exclusions`) | `5e3ae2e524fcdbd5d045698a5a14bd397ae2c21bf0ff9ae2f54f2502c35a3ff7` |
| Complete rehearsal package (raw byte, 4 files) | `a209e84ad15592d1665e8d02b1393f9aa079c8b33207cff4cd4d567e8a92a49c` |
| Complete rehearsal package (semantic content, 3 tables) | `9ad5f49c9ae952577cb3c81a77c1a70c0f0c54df66bdb2081e5a6dc8a0d24d93` |

Per-window content hashes are recorded in the companion JSON (15 hashes: 3 tables × 5 windows), independently reproduced identical between run1 and run2 for every window.

**No file is described as byte-identical when it is not**: the two `.sqlite` files are explicitly *not* byte-identical (raw hash differs); only their scientific content is.

---

## PHASE 6 — Exact/proxy separation freeze

- Native `agg_trades`-derived reconstruction remains the sole primary `EXACT` representation.
- Book-depth absorption remains ruled `LOW_FIDELITY_PROXY_ONLY` (per the readiness audit) — **not constructed** in the rehearsal and **not constructed** in this freeze batch.
- Proxy rows in the primary table: 0. No UNION/pooling exists anywhere in the schema or code.
- Rules frozen forward: separate physical storage, separate accounting, no silent fallback, no substitution on a missing primary row, proxy promotion requires its own future preregistration.

---

## PHASE 7 — Known-at and access freeze

Independently re-verified this batch by direct read-only SQL against both retained disposable databases (which contain no outcome table at all — only `absorption_impact_windowed_flow`/`window_quality_v1`/`exclusions`/`rehearsal_manifest`):

| Check | Run 1 | Run 2 |
|---|---|---|
| `feature_available_ts_ms != signal_birth_ts` | 0 | 0 |
| `window_end_ts_ms != signal_birth_ts` | 0 | 0 |
| `known_at_classification != 'KNOWN_AT_SAFE'` | 0 | 0 |
| `evidence_layer != 'EXACT'` | 0 | 0 |
| Window duration outside `{60000,300000,600000,1800000,3600000}` ms | 0 | 0 |

`known_at_violations = 0`, `outcome_reads = 0`, `post_cutoff_source_rows = 0` — all confirmed. This batch did **not** reread outcomes; the original authorizer-based proof from the accepted rehearsal (`install_outcome_access_guard`, `SQLITE_DENY`, `outcome_access_violations = []`) stands as the outcome-access proof and was not repeated, since the retained disposable databases structurally cannot contain an outcome table.

---

## PHASE 8 — Expected canonical migration manifest

**Naming reconciliation (flagged, not resolved unilaterally):** the frozen contract (`fc1321f5`, §12) used the illustrative name `ami_impact_window_quality_v1`; the validated rehearsal (`fc43e972`) actually implemented and hash-proved `absorption_impact_windowed_flow` / `absorption_impact_window_quality_v1` / `absorption_impact_exclusions`. This freeze recommends the future A5 migration reuse the rehearsal's exact validated names for continuity between tested code and canonical schema — an explicit operator decision at A5, not decided here.

| Table | PK | Row count expected | Content hash expected |
|---|---|---|---|
| `absorption_impact_windowed_flow` | `feature_id`, unique `(symbol, signal_id, window_id, feature_definition_version)` | 1,619 | `f7c834cc…` |
| `absorption_impact_window_quality_v1` | `quality_id`, unique `(signal_id, window_id, quality_contract_version)` | 1,620 | `5d1a205c…` |
| `absorption_impact_exclusions` | `exclusion_id`, unique `(signal_id, window_id, reason_code)` | 1 | `5e3ae2e5…` |

Full column lists, CHECK constraints, evidence-layer checks, foreign keys, indexes, and immutable-update policy per table are recorded in the companion JSON (`phase8_expected_canonical_migration_manifest`). All three tables are insert-only; a second migration run must be `NOOP_IDENTICAL`. No effective-view (`assessment_version`-latest-wins) is implemented by the validated rehearsal, unlike the contract's illustrative CVD-parallel description — this is a disclosed gap for A5 to resolve, not silently added here. No broader feature-store redesign is proposed. No DDL was implemented in the live database this batch.

---

## PHASE 9 — Migration acceptance equations

| Condition | Required value |
|---|---|
| Expected feature rows | 1,619 |
| Expected exclusions | 1 |
| Expected candidate universe | 1,620 |
| Known-at violations | 0 |
| Duplicate primary keys | 0 |
| Proxy rows in primary table | 0 |
| Outcome reads | 0 |
| Experiment writes | 0 |
| Runtime/risk/execution delta | 0 |
| Second migration run | `NOOP_IDENTICAL` |
| Backup/restore proof | required before A5 |
| Source/package/code hashes | must match this freeze exactly |

Any count or hash drift at A5 blocks migration; no automatic reconciliation is permitted.

---

## PHASE 10 — Independent freeze verification

Performed read-only against the two retained disposable databases only — **no rebuild against `microstructure.db`, no new disposable database created**:

| Item | Result |
|---|---|
| All row counts | match accepted rehearsal exactly |
| All primary keys | match, 0 duplicates |
| All quality partitions | match (`EXACT_RECONSTRUCTABLE`=1,619, `SOURCE_GAPPED`=1, others=0) |
| All exclusion reasons | match (1× `CONFIRMED_GAP_OVERLAP`) |
| All content hashes | match, run1≡run2 for all 3 tables and all 5×3 per-window partitions |
| W3600 source gap | independently traced to `gaps.id=766` |
| Denominator floor | confirmed non-binding (0/1,619) |
| Known-at compliance | confirmed (0 violations, 0 CHECK-column mismatches) |
| Outcome-access count | confirmed 0 (structurally: no outcome table exists in either retained database) |

**Result: `ROW_ACCOUNTING_IDENTICAL_TO_ACCEPTED_REHEARSAL`.** No drift found; no count was altered to force a match.

---

## Live database state (unchanged, proven)

| Check | Before | After |
|---|---|---|
| `canonical.sqlite` sha256 | `25a56a98d0…` | `25a56a98d0…` (unchanged) |
| `experiment_registry` | 23 | 23 |
| `experiment_results` | 350 | 350 |
| `schema_version` | 12 | 12 |
| `researcher_exposure_ledger` | 1,176 | 1,176 |
| `epistemic_test_nullifiers` | 1 | 1 |
| `experiment_gate_receipts` | 1 | 1 |

No feature/window/floor change. No runtime/risk/execution delta. No route or bucket promotion. No canonical migration occurred.

---

## Storage guardrail

| Item | Value |
|---|---|
| Peak temporary disk usage this batch | 6,713 bytes (one working diagnostic JSON under `.runtime_temp`, used to compute the independent-verification numbers above) |
| Files created this batch | 1 (`independent_verification_run1_run2.json`, purpose: read-only recomputation scratch, deleted after its results were incorporated into this freeze's MD/JSON) |
| Files retained this batch | this MD, this JSON, and the governance state-transition proof |
| Files deleted this batch | `.runtime_temp/absorption_impact_rehearsal_v1/independent_verification_run1_run2.json` |
| Remaining under `.runtime_temp` | `absorption_impact_rehearsal_v1/{manifest.json, rehearsal_result.json, rehearsal_run1.sqlite, rehearsal_run2.sqlite}` (2.5 MB, unchanged, now promoted to immutable rehearsal evidence) |
| Remaining under `.pytest_temp` | none |
| Full `microstructure.db` copy made | never |

---

## Success verdicts

**`CASCADE_ABSORPTION_IMPACT_ROW_ACCOUNTING_FREEZE_V1_COMPLETE`**

**`ABSORPTION_IMPACT_ROW_ACCOUNTING_FROZEN_FOR_CANONICAL_MIGRATION`**

All counts, identities, and hashes match the accepted rehearsal exactly; independent re-verification found zero drift. Stopping after the freeze. No canonical migration begins without new, separate operator instruction.
