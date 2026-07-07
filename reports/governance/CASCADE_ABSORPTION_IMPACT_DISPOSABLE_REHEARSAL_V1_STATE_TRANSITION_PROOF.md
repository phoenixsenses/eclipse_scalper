# CASCADE_ABSORPTION_IMPACT_DISPOSABLE_REHEARSAL_V1_STATE_TRANSITION_PROOF

**Batch:** BATCH-CASCADE-ABSORPTION-IMPACT-DISPOSABLE-REHEARSAL-V1
**Purpose:** Implement and validate the frozen absorption/price-impact bridge contract (commit `fc1321f5`) entirely in a disposable rehearsal environment — feature reconstruction, quality validation, row-accounting rehearsal only.
**Prior checkpoint (unchanged, not reopened):** commit `fc1321f5` (`CASCADE_ABSORPTION_IMPACT_READINESS_AND_CONTRACT_V1_COMPLETE`), readiness verdict `ABSORPTION_IMPACT_READY_FOR_DIRECT_REHEARSAL`, `experiment_registry`=23, `experiment_results`=350, `epistemic_test_nullifiers`=1, `schema_version`=12.
**Nature:** No preregistration, no experiment ID, no nullifier action, no outcome/TEST access, no performance calculation, no canonical migration, no schema_version change, no runtime/risk/execution modification, no route promotion.
**Author:** Sonnet 5 · **Date:** 2026-07-07

---

## Sequence executed

1. **Implementation** (Phase 1): new module `ami/absorption/cascade_absorption_impact_rehearsal.py` (605 lines) implementing the single frozen formula exactly as contracted (absolute-value denominator, `[T-W,T]` window law, `feature_available_ts_ms=signal_birth_ts`, `KNOWN_AT_SAFE` literal) — no alternative formulas, no multi-definition comparison.
2. **`FLOOR_USD_M` derivation**: computed once from the real `|signed_notional|` distribution at W=60s across all 324 signals (0 excluded at that window), read-only, outcome-blind — frozen at `0.01` ($10,000), documented with its full derivation in code and in the rehearsal report.
3. **Disposable environment** (Phase 2): schema created only inside disposable SQLite files under `D:\eclipse_scalper\.runtime_temp\absorption_impact_rehearsal_v1\`; every source read used `mode=ro` connections to the real `data/ami/canonical.sqlite`/`data/microstructure.db`, bounded per-signal-per-window range queries only (no full-table copy of either file, ever).
4. **Anchor universe / source reconstruction / numerical-stability / evidence-layer / quality taxonomy / row accounting / known-at proof / idempotency / source-gap reconciliation** (Phases 3-11): implemented and exercised via `run_rehearsal()`, `content_hash_of_disposable()`, `row_accounting()`, and a SQLite authorizer (`install_outcome_access_guard`).
5. **Tests**: `tests/test_ami_absorption_cascade_impact_rehearsal.py` (581 lines, 26 tests) — first full run found 4 failures (all test-design issues, not implementation bugs — see "Failures found and fixed" below); after fixes, **26/26 passed** in two independent full reruns.
6. **Real-data execution**: a standalone driver script ran the full rehearsal twice against the real (read-only) source state, writing only to the disposable location, producing `rehearsal_run1.sqlite`, `rehearsal_run2.sqlite`, `manifest.json`, `rehearsal_result.json`.
7. **Reports**: this proof + `S34_CASCADE_ABSORPTION_IMPACT_DISPOSABLE_REHEARSAL_V1.md`/`.json`.

## Failures found and fixed during test development (disclosed, not concealed)

| # | Failure | Root cause | Fix |
|---|---|---|---|
| 1 | `test_outcome_table_access_raises` | Raising a custom Python exception *from inside* a SQLite authorizer callback is not reliably propagated by the sqlite3 C extension | Authorizer now returns `sqlite3.SQLITE_DENY` (the documented, correct mechanism); SQLite itself surfaces the denial as `sqlite3.DatabaseError: access to X is prohibited`, with a `violations` list as the audit trail |
| 2 | `test_outcome_column_access_raises_even_via_different_table_alias` | Same root cause | Same fix |
| 3 | `test_rehearsal_functions_never_reference_outcome_table_statically` | The test scanned the *entire* module source for the string `"ami_lifecycle_path_observations"` — but `_FORBIDDEN_TABLES`/`_FORBIDDEN_COLUMNS` **must** contain that exact string as deny-list data (the guard has to name what it forbids); this is intentional, necessary code, not a violation | Rewrote the test (`test_rehearsal_functions_never_execute_sql_naming_the_outcome_table`) to parse the AST and check only string literals passed as arguments to `.execute()`/`.executescript()`/`.executemany()` calls — the narrower, correct invariant ("no SQL query ever names the outcome table"), with a sanity check that the scan actually found real SQL (not vacuously empty) |
| 4 | `test_full_real_data_rehearsal_idempotent_and_known_at_clean` | Row-level `created_ms`/`assessed_at_ms` bookkeeping timestamps legitimately differ between two separate real-time runs, making a whole-table hash comparison spuriously fail even though the *content* was identical | `content_hash_of_disposable()` rewritten to hash only the declared content columns per table (excluding bookkeeping timestamps) — identical discipline to `ami/warehouse/experiment_ledger.py`'s `_VOLATILE_BOOKKEEPING_COLUMNS` convention already established elsewhere in this codebase |

None of these four were implementation defects in the rehearsal logic itself — all four were test-design corrections, made before the real-data execution that produced the retained evidence (the code that ran the retained evidence run is the same code all 26 fixed tests pass against, confirmed by matching sha256).

## Code/contract/evidence hash chain

| Artifact | sha256 |
|---|---|
| `ami/absorption/cascade_absorption_impact_rehearsal.py` (as run) | `604947829105be47b0a425694104392a91b502e7bbff6b7ba2a71b3f881ec609` |
| `S34_CASCADE_ABSORPTION_IMPACT_CANONICAL_BRIDGE_CONTRACT_V1.md` (frozen, commit `fc1321f5`) | `5acf0d532241f8f4197da1ac10951d6afec6539244422ceeba636674bdbfdb9a` |
| `S34_CASCADE_ABSORPTION_IMPACT_READINESS_AUDIT_V1.md` (frozen, commit `fc1321f5`) | `fbef831fe828c4a8768bb01b884edac3f52059cc8f48e84f522a2adf4d0ba709` |
| `data/ami/canonical.sqlite` (source, unchanged before/after this batch) | `25a56a98d02f84191aeb6ff46f81245d36bc0d635e916dbfac3e13d076bf5291` |
| `rehearsal_run1.sqlite` (disposable output) | `b42972c76cb8de700fb8b9addc358958a2bd5c904eaf798026db20c4799978b9` |
| `rehearsal_run2.sqlite` (disposable output) | `e842f694d930091f714de53049ec4565be546bef39ca17cdcc1ac349eb5bc923` |

The full manifest (including creation timestamp `2026-07-07T06:13:51Z` UTC) is retained at `D:\eclipse_scalper\.runtime_temp\absorption_impact_rehearsal_v1\manifest.json`.

## Real database state — unchanged (proof)

| Check | Before this batch | After this batch |
|---|---|---|
| `data/ami/canonical.sqlite` sha256 | `25a56a98d0…` | `25a56a98d0…` (unchanged) |
| `experiment_registry` | 23 | 23 |
| `experiment_results` | 350 | 350 |
| `schema_version` | 12 | 12 |
| `data/ami/knowledge.sqlite`: `epistemic_test_nullifiers` | 1 (CVD nullifier only) | 1 (unchanged) |
| Outcome-table (`ami_lifecycle_path_observations`) reads | — | **0**, proven by a live SQLite authorizer denying every attempt (0 attempts occurred) |

Every connection to the real files during the rehearsal was `mode=ro`, structurally incapable of writing (SQLite enforces this at the OS/VFS level), independent of any application-level discipline.

## Exact changed/added-file manifest (this commit)

| File | Status | Content |
|---|---|---|
| `ami/absorption/__init__.py` | New | package marker |
| `ami/absorption/cascade_absorption_impact_rehearsal.py` | New | rehearsal implementation (605 lines) |
| `tests/test_ami_absorption_cascade_impact_rehearsal.py` | New | 26 tests (581 lines) |
| `reports/research/s34/S34_CASCADE_ABSORPTION_IMPACT_DISPOSABLE_REHEARSAL_V1.md` | New | rehearsal report |
| `reports/research/s34/S34_CASCADE_ABSORPTION_IMPACT_DISPOSABLE_REHEARSAL_V1.json` | New | machine-readable companion |
| `reports/governance/CASCADE_ABSORPTION_IMPACT_DISPOSABLE_REHEARSAL_V1_STATE_TRANSITION_PROOF.md` | New | this document |

Not included: any canonical migration, preregistration artifact, TEST result, shared unrelated governance-projection change, runtime modification, or repository cleanup unrelated to this batch's own temp-file accounting.

## Storage guardrail accounting

| Item | Value |
|---|---|
| Peak temporary disk usage this batch | ~893 MB (`D:\eclipse_scalper\.pytest_temp\`, four pytest `--basetemp` directories accumulated across iterative test-fix cycles — each holding a conftest-fixture disposable copy of `canonical.sqlite`+`knowledge.sqlite`) + 2.5 MB (`D:\eclipse_scalper\.runtime_temp\absorption_impact_rehearsal_v1\`, the retained rehearsal evidence) |
| Full database copies created | 0 by the rehearsal implementation itself; the conftest test-isolation fixture (pre-existing repository infrastructure, not created by this batch) made its usual session-scoped disposable copies under `.pytest_temp` |
| Large temp DB copies under `C:\Users\...\AppData\Local\Temp` | **0** — all pytest/runtime scratch this batch was correctly routed to `D:\eclipse_scalper\.pytest_temp`/`.runtime_temp` per the storage guardrail; the OS-temp scratchpad held only two small pre-existing `.py` diagnostic scripts (~4KB total, from a prior batch), one of which (the standalone rehearsal driver script) was deleted after use |
| Files created this batch | 4 pytest `--basetemp` directories (`absorption_quickcheck`, `absorption_rehearsal_run1`, `absorption_rehearsal_run2`, `absorption_rehearsal_final`) under `.pytest_temp`; `absorption_impact_rehearsal_v1/` (2 disposable `.sqlite` files + 2 `.json` files) under `.runtime_temp`; one standalone driver script under the OS scratchpad |
| Files retained at completion | `D:\eclipse_scalper\.runtime_temp\absorption_impact_rehearsal_v1\` (2.5 MB total: `rehearsal_run1.sqlite`, `rehearsal_run2.sqlite`, `manifest.json`, `rehearsal_result.json`) — **retained as accepted immutable rehearsal evidence**, per the manifest's own retention field |
| Files deleted at completion | All 4 `.pytest_temp` `--basetemp` directories (~893 MB reclaimed — these were ordinary pytest fixture scratch, not evidence artifacts themselves); the standalone driver script under the OS scratchpad |
| Never touched | `data\microstructure.db`, `data\ami\canonical.sqlite`, `data\ami\knowledge.sqlite`, accepted `data\ami\backups\*`, any prior immutable evidence artifact, any active runtime checkpoint/ledger |

## Required validations (proven)

- Deterministic implementation: ✅ (two independent runs, identical counts and content hashes)
- Exact row reconciliation: ✅ (1,620 = 1,619 usable + 1 excluded, every window, both runs)
- `known_at_violations = 0`: ✅
- Outcome reads = 0: ✅ (SQLite-authorizer-proven, live)
- No exact/proxy pooling: ✅ (schema `CHECK (evidence_layer='EXACT')`, no proxy table exists in this rehearsal)
- Reproducible identical rerun: ✅ (`REBUILD_IDENTICAL`)
- Live canonical state unchanged: ✅ (23/350/12, byte-identical before/after)
- All focused tests green: ✅ (26/26)

---

## Verdict

**`CASCADE_ABSORPTION_IMPACT_DISPOSABLE_REHEARSAL_V1_COMPLETE`**

Success verdict: **`ABSORPTION_IMPACT_REHEARSAL_READY_FOR_ROW_ACCOUNTING_FREEZE`**

Stopping after rehearsal. No row-accounting freeze, canonical migration, preregistration, or execution begins without new, separate operator instruction.
