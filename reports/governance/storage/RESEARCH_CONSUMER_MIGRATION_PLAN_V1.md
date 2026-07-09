# Research Consumer Migration Plan V1

> BATCH-STORAGE-ROTATION-RETENTION-RESEARCH-CONSUMER-MIGRATION-PLAN-V1
> Planning-only gate: inventory, classification, risk separation, and a bounded
> batch strategy for the remaining research-script storage-reader migrations.
> No consumer scripts are migrated in this gate. Raw structural data backing this
> document: `reports/governance/storage/research_consumer_migration_inventory_v1.json`,
> produced by `tools/storage_migration_inventory_scan.py` (read-only, static text
> analysis only — never imports or executes a scanned file).

## 1. Current state

**Canonical reader stack (all shipped, all tested, all still green):**

| Component | Commit | Purpose |
|---|---|---|
| Sharded book_ticker exporter | `a74b4571` | Multi-shard Parquet export for partitions too large for one file |
| ts_ms ordering fix | `9247406e` | Sort-free canonical `(ts_ms ASC, id ASC)` export ordering |
| Guarded subprocess reverify | `748c74f9` | Full scientific reverify, isolated + RSS-guarded, explicit-invocation-only |
| Canonical research reader | `3ed35288` | `ami.storage.research_reader`: unified archive+SQLite range reads (`plan_read`/`execute_read`) |
| Invalid-receipt test amendment | `29a62575` | Closed a coverage gap in the trust-check test suite |
| Second consumer integration | `1093ed0f` | Proved the range-read helper on a second, independent consumer |
| Point-lookup primitive | `437f2b33` | `lookup_latest_at_or_before` + `select_last_row_group_at_or_before` |
| ASOF consumer migration V1 | `4f612cde` | `research_s34_consensus_composite.py` |
| ASOF consumer migration V2 | `e0d5cd0c` | `research_s34_btc_microtrend_eth_quality.py` |
| ASOF consumer migration V3 | `81d3ce88` | `research_s34_btc_microtrend_sweep.py` |

**Production archive (unchanged across every gate above and this one):** `entry_count=3` —
`mark_prices/ETHUSDT/2026-05`, `agg_trades/ETHUSDT/2026-02`, `book_ticker/SOLUSDT/2026-04`.
All three manifest hashes confirmed unchanged since gate `748c74f9`. Staging/lock clean.
Source `microstructure.db` opened `mode=ro` everywhere; source mutations remain 0 across
every gate to date.

## 2. Completed migrated consumers (5)

| Consumer | Helper used | Real archive/hybrid proven? |
|---|---|---|
| `research_ami_mfe50_experiment.py` | range-read (`execute_read`) | agg_trades + book_ticker bounded windows |
| `research_s34_100k_notmon_check.py` | range-read (`execute_read`) | agg_trades bounded window |
| `research_s34_consensus_composite.py` | point-lookup (`lookup_latest_at_or_before`) | mark_prices archive-only + hybrid (real) |
| `research_s34_btc_microtrend_eth_quality.py` | point-lookup | BTCUSDT has no archive → SQLite-only real + synthetic archive/hybrid |
| `research_s34_btc_microtrend_sweep.py` | point-lookup | mark_prices/ETHUSDT archive-only + **real hybrid boundary** |

All five: old direct-SQL path preserved as parity oracle, new reader-backed path proven
bit-identical, full storage/reader/lookup regression green after each, zero source mutation.

## 3. Remaining inventory — headline numbers

Full re-scan via `tools/storage_migration_inventory_scan.py` (static analysis, no imports,
no DB access): **314 files** in `tools/*.py` reference `microstructure.db` and are not
already migrated.

| Filter | Count |
|---|---|
| Total candidates scanned | 314 |
| Flagged **DO_NOT_TOUCH** by filename pattern (live/execution/shadow/dashboard/v_engine/state_machine) | 62 |
| Remaining, in-scope for classification | **252** |
| …of which have an `ORDER BY ts_ms DESC LIMIT 1` pattern (ASOF-fit) | 91 |
| …of which have a bounded-range scan pattern (range-fit) | 94 |
| …of which have **both** patterns in the same file | 68 |
| …of which have **neither** (other query shapes: aggregates on non-`ts_ms` keys, joins, full-table scans, or a different DB entirely) | 135 |
| Allowlisted-table-only usage (`mark_prices`/`agg_trades`/`book_ticker`, no other table) | 45 |
| Out-of-allowlist-only usage (`liquidations`/`vol_state`/etc., no allowlisted table at all) | 17 |
| **Mixed** allowlist + out-of-allowlist in the same file | 98 |
| Uses neither allowlist nor known out-of-allowlist tables (different DB, e.g. `s34_intelligence.db`, `s34_feature_factory.db`, canonical.sqlite) | 92 |
| `mode=ro` already present | 172 / 252 |
| AMI governance/knowledge/research-registry import present | 3 (already-committed Group 2 buyfade scripts — see §7) |
| `random` imported without a fixed seed (non-determinism risk flag) | 2 |
| Writes a `reports/research/s34` output file | 54 |
| Untracked (never committed) | 163 |

This reconciles and supersedes the looser "76 ORDER BY DESC LIMIT 1 scripts" / "64 bounded-range
scripts" counts used informally in the three ASOF gates and the two range-read gates before
this one — those were narrower, less rigorous greps. This is the canonical count going forward.

## 4. Classification

### 4.1 SAME_PATTERN_BATCH_SAFE_ASOF — 18 files

ASOF-fit, allowlisted-table-only, `mode=ro`, no AMI/governance import, no non-determinism flag.
Same exact shape as the three already-migrated ASOF pilots. **Direct vs. helper parity is
expected to be mechanical** — extract the inline query into a named oracle function (if not
already named), add a `_v2` reader-backed counterpart, swap the call site, prove parity on a
handful of real timestamps plus the standard synthetic suite.

Full list, smallest first (lines / path / allowlisted tables / DESC-LIMIT-1 occurrences):

```
173  research_funding_nonoverlap.py            [mark_prices]              x1
178  s34_mechanism_taxonomy.py                 [mark_prices]              x2
181  research_s34_prediction_image.py          [agg_trades, mark_prices]  x3
282  research_s34_real_fill_parity.py          [book_ticker]              x1
284  research_s34_counter_regime_realfill.py   [book_ticker]              x1
324  research_s34_day_context_scan.py          [book_ticker]              x1
375  research_s34_eth_preliq_control.py        [book_ticker]              x1
380  research_eth_provision_realism.py         [book_ticker]              x1
381  research_s34_preliq_detector.py           [book_ticker]              x1
398  research_s34_wave_absorption.py           [book_ticker]              x1
402  s34_regime_filter_shadow_eval.py          [book_ticker, mark_prices] x1
420  research_s34_500k_daytrend_route_sweep.py [book_ticker, mark_prices] x1
420  research_s34_eth_preliq_executable.py     [book_ticker]              x1
434  research_s34_trailing_oos_realfill.py     [book_ticker, mark_prices] x1
477  research_s34_cluster_geometry_features.py [book_ticker]              x1
484  research_nonpredictive_carry_provision.py [book_ticker]              x1
516  research_s34_early_confirmation_scan.py   [book_ticker, mark_prices] x1
638  research_s34_v6_management_system.py      [agg_trades, book_ticker]  x1
```

**Important sub-note:** the `book_ticker`-only entries in this list were spot-checked
(`research_s34_real_fill_parity.py`, `research_s34_counter_regime_realfill.py`,
`research_s34_day_context_scan.py`) and their `book_ticker_at()`-style helpers take `symbol`
as a runtime parameter — meaning it is **not statically knowable** from source alone whether
they're invoked with SOLUSDT (which HAS a real archive partition) or ETHUSDT (which doesn't).
Each must be checked at migration time via its actual call sites, same as done for the two
BTCUSDT-heavy pilots already migrated; whichever symbol is actually used determines whether
production smoke can show a real `ARCHIVE_ONLY`/`HYBRID` plan or must rely on the synthetic
fixture pattern established in `test_research_s34_btc_microtrend_eth_quality_lookup_migration_parity.py`.

### 4.2 SYNTHETIC_ONLY_ASOF — sub-case of §4.1, not a separately-counted bucket

Not a distinct file set: this is a **property** each §4.1 candidate is checked for at
migration time (does its real invocation ever touch an archived symbol?). Two of the three
already-migrated ASOF pilots needed this (`btc_microtrend_eth_quality.py` fully,
`btc_microtrend_sweep.py` partially, for its BTCUSDT calls only) — expect roughly half of
the 18 in §4.1 to need it too, going by the `book_ticker`-heavy composition of that list
(book_ticker's only real archive is SOLUSDT).

### 4.3 OUT_OF_ALLOWLIST_WAIT — 17 files (table-blocked) + 1 ASOF-shaped example

17 files use only out-of-allowlist tables (`liquidations`, `vol_state`, `open_interest`,
`spot_prices`, `gaps`, `s34_trades`, `ami_signal_lifecycle`) for their storage access —
these cannot be migrated to either helper without an explicit allowlist-expansion decision
(see §6). One additional file has an ASOF-shaped (`ORDER BY ts_ms DESC LIMIT 1`) pattern but
exclusively against an out-of-allowlist table — same wait condition applies.

None of these are migrated in this gate or proposed for the next one.

### 4.4 RANGE_READ_HELPER_NEEDED — 24 files (range-fit only, no ASOF pattern, allowlisted tables)

These need only the **range-read helper** (`ami.storage.research_reader.plan_read` +
`execute_read`), already built, tested, and proven on two consumers
(`research_ami_mfe50_experiment.py`, `research_s34_100k_notmon_check.py`) in the two range-read
gates before the ASOF track started. **No new helper is required** — the existing
`execute_read` contract (bounded batches, column projection, Python-side filter application,
archive/SQLite/hybrid planning) already covers every pattern seen in this bucket (bounded
`ts_ms>=? AND ts_ms<?` window scans, SUM/COUNT-style aggregates computed client-side over the
streamed rows). Top candidates by size:

```
147  funding_rate_analysis.py                    [mark_prices]              range x1
243  research_s34_source_quality_reconciliation.py [agg_trades, mark_prices] range x2
253  research_s34_orderflow_lead.py               [agg_trades, mark_prices] range x2
255  research_s34_sell_reversal_filter.py         [agg_trades, mark_prices] range x4
264  research_s34_sell_reversal_quality.py        [mark_prices]              range x2
276  research_s34_buy_reversal_short.py           [mark_prices]              range x2
281  research_s34_cross_symbol_lag.py             [mark_prices]              range x1
321  export_s34_visualization_json.py             [mark_prices]              range x1
```

A dedicated `RANGE-READ-CONSUMER-MIGRATION-V1` gate (mirroring the ASOF gates' shape exactly,
just targeting `execute_read` instead of `lookup_latest_at_or_before`) is the natural next
track once the ASOF batches below are done, or can be interleaved with them.

### 4.5 MIXED_PARTIAL_MIGRATION — 68 files (both ASOF-fit and range-fit, allowlisted tables present)

These files have BOTH shapes present (like the two already-migrated ASOF pilots that also
had an untouched bounded-range TP/SL/BE scan, or an untouched `ofir`-style aggregate).
**Safe partial migration is the established, proven pattern**: migrate only the ASOF-shaped
calls this gate's helper covers, leave the range-shaped calls on direct SQL (or migrate those
too, separately, via the range-read helper, following §4.4's track) — exactly as done for
`research_s34_consensus_composite.py`'s `ofir`/`rv5` (left untouched, documented inline) and
`research_s34_btc_microtrend_sweep.py`'s TP/SL/BE scan (left untouched, documented inline).
No new engineering needed here beyond continuing the same discipline; this bucket is a
**large pool to draw ASOF Batch 3+ candidates from** once §4.1's pure cases are exhausted.

### 4.6 DO_NOT_TOUCH_FOR_NOW — 62 files

Filtered out before classification by filename pattern (not by content — a conservative,
name-based net, deliberately over-inclusive):

| Pattern | Count | Examples |
|---|---|---|
| `v_engine` | 27 | `s34_v_engine_state_machine_management.py`, `s34_v_engine_shadow_observer.py` |
| `v02_` | 8 | `s34_v02_h4_shadow_control_plane.py` |
| `state_machine` | 7 | `research_s34_state_machine_v2_gauntlet.py`, `s34_buy_side_state_machine_gauntlet.py` |
| `shadow` | 7 | `s34_realtime_shadow_runner.py`, `s34_shadow_paper_runner.py` |
| `live_` | 4 | `s34_live_order_executor.py`, `s34_live_preflight.py` |
| `dashboard` | 2 | `s34_cascade_navigation_dashboard.py`, `s34_live_chart.py` |
| `execution` | 1 | `s34_execution_optimizer.py` |
| other (`scheduler`, `monitor`, `risk_*`, `intelligence_ledger`, etc.) | 6 | `s34_quarantine_monitor.py`, `s34_prereg_monitor.py` |

**Never touched by any research-reader migration gate**, regardless of category, until an
operator explicitly re-scopes one individually. This is the same guardrail already in force
via CLAUDE.md (`tools/s34_state_machine_live_executor.py` is inside this set) — this bucket
generalizes it to every file that even *looks* execution/live/shadow/scheduler-adjacent by
name, not just the one hard-coded guardrail file.

## 5. Risk scoring

| Risk | Criteria | Bucket(s) affected |
|---|---|---|
| **Low** | allowlisted-table-only, `mode=ro`, deterministic, no governance import, writes at most its own overwritable report | most of §4.1, §4.4 |
| **Medium** | mixed allowlist + out-of-allowlist tables (partial migration required, care needed not to touch the out-of-scope half) | §4.5 |
| **Medium** | book_ticker/mark_prices usage where the actual symbol isn't statically determinable (synthetic-only risk, needs call-site check) | subset of §4.1 |
| **High** | out-of-allowlist-only (blocked, don't attempt) | §4.3 |
| **High** | governance/knowledge/research-registry import present (writes AMI state — must re-verify no result mutation before *any* touch, even for an in-scope table) | the 3 buyfade files, §7 |
| **Excluded (no scoring attempted)** | live/execution/shadow/scheduler/state-machine name match | §4.6 |

## 6. Allowlist Expansion Decision Gate (not decided here — flagged for operator)

17 files are blocked purely because their storage access is `liquidations`/`vol_state`/
`open_interest`/`spot_prices`/etc. — tables with **no archive partition and no reader
support at all**. Extending the allowlist would require, at minimum:
- confirming whether any of these tables are ever intended for archival (none currently are —
  the archive-eligible registry in `ami/storage/registry.py` is a deliberate 3-table allowlist)
- deciding whether `research_reader`/`lookup_latest_at_or_before` should support SQLite-only
  tables that will never have an archive-side (i.e., a strictly-SQLite variant of the same
  contract, for consistency/testing benefit, without any archive-side code paths)

This is a real decision with real scope, not a small addition — **explicitly deferred**, not
started, not implied as "next." Recommendation: leave the 17 files in `OUT_OF_ALLOWLIST_WAIT`
indefinitely unless a future gate is explicitly opened for this exact question.

## 7. Special case: the 3 governance-aware buyfade scripts

`research_s34_buyfade_reentry.py`, `research_s34_buyfade_silence_exit.py`,
`research_s34_buyfade_structural.py` were already committed in this session's very first
baseline gate (Group 2), already verified `mode=ro` with no `experiment_registry` writes.
They import `ami.research`/`ami.knowledge` modules and their underlying finding is already
recorded in memory as a closed, negative result ("BUY-Fade Structural... route ALL tarihsel
negatif"). **Recommendation: do not migrate these in an ASOF or range-read batch** — even
though a migration would only touch the *data-access* layer and preserve output exactly (the
same guarantee already proven 5 times this session), touching a script tied to a closed,
governance-recorded finding carries reputational/process risk out of proportion to the
benefit. Leave them exactly as they are unless the operator explicitly reopens that finding.

## 8. Proposed bounded batches

### A. ASOF Batch 1 (next gate candidate) — 3 files, lowest risk

1. `research_funding_nonoverlap.py` (173 lines, `mark_prices` only, 1 lookup)
2. `s34_mechanism_taxonomy.py` (178 lines, `mark_prices` only, 2 lookups)
3. `research_s34_prediction_image.py` (181 lines, `agg_trades`+`mark_prices`, 3 lookups)

All three use tables with **confirmed real archive coverage** (mark_prices/ETHUSDT,
agg_trades/ETHUSDT) — expect genuine archive-only/hybrid production smoke, same as
`consensus_composite.py` and `btc_microtrend_sweep.py`, not the synthetic-only path.
Expected test approach: identical structure to the five already-shipped ASOF parity suites
(direct-SQL oracle kept, `_v2` reader-backed path added, real-window production smoke +
synthetic fixture suite for edge cases, full regression re-run).

### B. ASOF Batch 2 — next 3-5 after Batch 1, `book_ticker`-heavy

4. `research_s34_real_fill_parity.py` (282 lines)
5. `research_s34_counter_regime_realfill.py` (284 lines)
6. `research_s34_day_context_scan.py` (324 lines)

**Risk note:** all three take `symbol` as a runtime parameter for their `book_ticker` lookups
— call sites must be checked first to know whether real archive smoke (SOLUSDT) or
synthetic-only coverage (any other symbol) applies. Slightly higher batch risk than Batch A
for this reason alone; otherwise same shape (allowlisted-only, `mode=ro`, deterministic).

### C. RANGE Reader/Helper Gate

No new helper needed — `execute_read` (shipped in `3ed35288`, proven in the first two
consumer-migration gates) already covers every pattern in §4.4's 24-file bucket. Recommended
next range-track gate: **RANGE-READ-CONSUMER-MIGRATION-V1**, targeting
`funding_rate_analysis.py` (147 lines, smallest) as the third range-read pilot, using the
exact same commit/test/parity shape as `research_ami_mfe50_experiment.py` and
`research_s34_100k_notmon_check.py`.

### D. Allowlist Expansion Decision Gate

Not started. See §6. Recommendation: address only if/when an operator explicitly wants the
17 blocked files unlocked; no urgency signal exists today (they represent <7% of the
remaining 252 in-scope candidates).

### E. Do-not-touch list

62 files, filename-pattern-blocked (§4.6). Re-eligible only if an operator explicitly
names one individually and re-scopes it outside its current live/execution/shadow role —
not something a future migration gate should do on its own initiative.

## 9. Standard test/validation contract for all future migration batches

Every future ASOF or range-read consumer migration must satisfy (unchanged from the five
gates already shipped):

1. Direct-SQL oracle path preserved, unchanged, in the file
2. New reader/helper-backed path added, used by the actual call site
3. Row-exists / no-row parity
4. Row timestamp parity
5. Selected-column parity
6. Ordering / tie-break parity (`ORDER BY ts_ms DESC, id DESC` for ASOF; canonical
   `(ts_ms ASC, id ASC)` for range-read)
7. Digest/aggregate parity (exact numeric match, `pytest.approx(rel=1e-9)` where floats)
8. Final consumer output parity (stdout/report — unchanged)
9. Provenance correctness (source type, result/segment source, manifest/hash identity,
   query bounds, filters, ordering)
10. Trust-failure fail-closed (synthetic corrupt manifest/receipt → `ArchiveTrustError`,
    never silently wrong data)
11. Source mutation 0 (catalog/manifest mtime asserted unchanged)
12. No full scientific reverify triggered (static check: no `reverify_guard` import in the
    migrated module)
13. Catalog `entry_count` and all 3 manifest hashes unchanged
14. Full storage/reader/lookup regression green
15. **All previously migrated consumers' parity suites still green**

## 10. Explicit non-goals of this gate

- No consumer script was migrated
- No purge, VACUUM, scheduler activation, or bulk migration was started or implied as imminent
- No allowlist expansion decision was made (§6 flags it, doesn't resolve it)
- No `Group 4/5/6`, `reports/research/s34` bulk output, shadow/dashboard, or
  `runtime/dashboard_backend.json` file was touched
- The 3 governance-aware buyfade scripts are explicitly NOT proposed for any near-term batch

## 11. Recommended next gate

**`BATCH-STORAGE-ROTATION-RETENTION-ASOF-LOOKUP-CONSUMER-MIGRATION-V4`**, targeting
§8.A's 3-file Batch 1 (`research_funding_nonoverlap.py`, `s34_mechanism_taxonomy.py`,
`research_s34_prediction_image.py`) — same bounded, single-batch shape as V1–V3, all three
using tables with confirmed real archive coverage. Not started automatically; requires
explicit operator invocation, per this gate's own instruction.
