# Range-Read Inventory Reconciliation V1

> `BATCH-STORAGE-ROTATION-RETENTION-RANGE-READ-INVENTORY-RECONCILIATION-V1`
> Governance/reconciliation gate. **No migration, no consumer change, no alpha search, no data
> mutation.** Produces an honest, reproducible, current-state range-read migration inventory to
> replace the stale `research_consumer_migration_inventory_v1.json` (a static, one-time scan from
> `99a5bf24`, before any of Range V1–V7 existed).

## 1. Executive verdict

The prior "24 total / 14 migrated / 10 remaining / 0 blocked" figures from
`RESEARCH_CONSUMER_MIGRATION_PLAN_V1.md` §4.4 **cannot be reconciled and are retired.** They were
never a complete accounting of the codebase — the plan doc's §4.4 table names only 8 of an implied
24, and the backing JSON has no way to distinguish "migrated" from "not yet touched" (see §3). A
fresh, reproducible, table-correlated static scan of all 320 `tools/*.py` files that reference
`microstructure.db` today finds:

| Category | Count |
|---|---:|
| `MIGRATED_RANGE_READ` | 18 |
| `REMAINING_MIGRATABLE_RANGE_READ_CANDIDATE` | 56 |
| `NO_OP_NO_ALLOWLISTED_RANGE_PATTERN` | 135 |
| `DO_NOT_TOUCH_BY_NAME` | 62 |
| `BLOCKED_FORWARD_ASOF_PRIMITIVE` | 21 |
| `ASOF_ONLY_NOT_RANGE_READ_SCOPE` | 17 |
| `OUT_OF_SCOPE_UNBOUNDED_SCAN` | 6 |
| `BLOCKED_DIFFERENT_DB` | 4 |
| `NO_OP_EXECUTE_READ_CONTRACT_MISMATCH` | 1 |
| **Total scanned** | **320** |

Confidence: **HIGH** for `MIGRATED_RANGE_READ` (18/18 cross-checked against every known-migrated
file from V1–V7 plus the two pre-V1 pilots, zero mismatches) and for the two explicitly-named
special cases (`research_s34_cross_symbol_lag.py`, `walkforward_eval.py`, both manually re-read in
full, not just regex-classified). **MEDIUM** for the 56-file `REMAINING_MIGRATABLE` pool — each
entry is a verified, table-correlated static signal (an actual bounded `ts_ms>=?...ts_ms<=?`
window against `mark_prices`/`agg_trades`/`book_ticker` in the same SQL statement), but none of
the 56 have had the individual manual-review pass (call-site symbol audit, import-safety check,
`mode=ro` behavior read) that migration itself requires — that is V8+'s job, not this gate's.

## 2. V7 commit hash and manifest

**Commit:** `e66b956ae1141618ebaca0341f1e1a5e57ba2467`
**Subject:** `feat(storage): range-read consumer migration V7 -- 2 consumers migrated`
**Author/date:** Gorkem Berke Yuksel, 2026-07-09 21:22:38 +0300

```
 tests/test_research_s34_exit_giveback_sweep_reader_migration_parity.py | 389 +++++++
 tests/test_research_s34_micro_entry_scalp_reader_migration_parity.py   | 428 +++++++
 tools/research_s34_exit_giveback_sweep.py                              | 539 ++++++
 tools/research_s34_micro_entry_scalp.py                                | 521 ++++++
 4 files changed, 1877 insertions(+)
```

Matches the expected V7 content exactly: `tools/research_s34_micro_entry_scalp.py`,
`tools/research_s34_exit_giveback_sweep.py`, and their two parity test files. Confirmed ancestor of
`HEAD` (current branch `codex/data-layer-fallback-cleanup`). Both files independently confirmed
`MIGRATED_RANGE_READ` by this gate's live scan (§5) — `reader_v2_present=True`, i.e. an actual
`execute_read`/`plan_read` call site, not merely an import.

## 3. Why the old inventory became stale

`research_consumer_migration_inventory_v1.json` (commit `99a5bf24`, generated once by
`tools/storage_migration_inventory_scan.py`) has five structural problems, all confirmed by
direct inspection during this gate:

1. **Static, one-time, never re-run.** Its `ALREADY_MIGRATED` exclusion set is hardcoded to the
   5 ASOF pilots that existed when it was written (`cf7f056d`/before). It has no concept of any of
   the 18 files migrated since (2 pre-V1 range pilots + 16 across Range V1–V7) — it still lists
   every one of them as an ordinary unmigrated candidate. Confirmed empirically: all 16 V1–V7
   migrated files are present in the old JSON, indistinguishable from a never-touched file.
2. **The oracle-preservation pattern defeats whole-file regex classification.** Every migration
   gate in this project's established convention *deliberately keeps* the old direct-SQL function
   as an unchanged parity oracle (e.g. `_mark_series()` next to `_mark_series_v2()`). A whole-file
   regex for `ts_ms>=?...ts_ms<=?` matches the oracle's literal SQL string forever, regardless of
   whether a reader-backed path was added alongside it. The old scanner has no signal at all for
   "was a reader-backed path also added" — it only ever looked for the raw SQL shape.
3. **Untracked/new files did not exist at scan time.** `research_s34_micro_entry_scalp.py` and
   `research_s34_exit_giveback_sweep.py` (both untracked in git, both migrated in V7) were present
   in the working tree when the old JSON was generated (confirmed: both appear in it) — but 6
   other files migrated in V5/V6 (`research_s34_sell_liq_bounce.py`,
   `research_s34_sell_path_quality.py`, `research_s34_sell_regime_analysis.py`,
   `research_s34_exact_route_change_validation.py`, `research_s34_sol200k_sell_dayfilter.py`,
   `research_s34_symbol_compare.py`) were *also* present — the staleness here is not "file didn't
   exist," it's purely problem #2 (oracle literal still matches).
4. **No-op files were never distinguished from remaining files.** The old JSON's flat structure
   has no `classification` field at all — the plan doc's §4.4 "24 files" table was a **manually
   curated subset**, not a generated one, and it only ever named 8 of the 24 explicitly. The other
   16 were asserted by count, never listed, so there is no way to verify or reconcile them now.
5. **No table-correlation between a range/ASOF pattern and an allowlisted table.** The old scanner
   (and this gate's own first-pass scanner, before this fix — see §4) matched "has an allowlisted
   table anywhere in the file" and "has a bounded range pattern anywhere in the file" as two
   *independent* whole-file signals. This produces false positives when a file has a bounded range
   scan on an **out-of-allowlist** table (e.g. `liquidations`) and an **unrelated** allowlisted-table
   query elsewhere (e.g. an ASOF `mark_prices` lookup) — the whole-file view wrongly looks like an
   allowlisted range-read candidate. Confirmed directly on `tools/research_s34_cross_symbol_lag.py`
   (§6.2).

**The old "24 total" claim is explicitly retired, not reconciled to 24.** No forcing was applied.

## 4. Methodology

Two new read-only, static-text scripts (never import or execute a scanned file; never touch
`microstructure.db`, the archive, or any consumer file):

- **`tools/range_read_inventory_reconciliation_v1_scan.py`** — scans every `tools/*.py` file
  referencing `microstructure.db` (320 files today, vs. 314 in the old one-time scan — the count
  moved because files were added/removed in the working tree since `99a5bf24`, not because of a
  methodology change). For each `select` keyword occurrence, takes a 400-character window and
  extracts the **FROM table** in that same window, then correlates range-scan / ASOF patterns to
  that specific table — fixing problem #5 above. Also detects `reader_v2_present` (an actual
  `RR.execute_read(`/`RR.plan_read(`/`research_reader.execute_read(`/`research_reader.plan_read(`
  **call**, not just the shared import line) and `lookup_call_present`
  (`lookup_latest_at_or_before(` call) — this single signal is what correctly separates "migrated"
  from "oracle-literal-still-matches," fixing problem #2.
- **`tools/range_read_inventory_reconciliation_v1_classify.py`** — reads the scan output and
  applies the classification tree in §5.1, 100% automated except for two named, fully-documented
  manual overrides (§6.2) each backed by a complete manual read of the file, not just its regex
  signals.

**Validation of the new signal:** cross-checked `reader_v2_present` against every file named in
the operator's `KNOWN_MIGRATED` list (16 files) plus the 2 pre-V1 pilots found by the scan itself
(`research_ami_mfe50_experiment.py`, `research_s34_100k_notmon_check.py`, both from the original
`RESEARCH_CONSUMER_MIGRATION_PLAN_V1.md` §2 "Completed migrated consumers (5)" list, before the
ASOF/range tracks existed as separate gates) — **zero mismatches, 18/18**. Cross-checked against 6
known ASOF-migrated files (`lookup_call_present=True` expected) — 6/6 correct.

Reproducibility proven directly: scan + classify re-run from scratch in this session produced
byte-identical `category_counts` on the second run.

## 5. Reconciled classification

### 5.1 Classification tree (in order, first match wins)

1. `path` in the two manual overrides (§6.2) → that override's classification.
2. Filename matches the CLAUDE.md-generalized `DO_NOT_TOUCH` pattern set → `DO_NOT_TOUCH_BY_NAME`.
3. `reader_v2_present` (an actual `execute_read`/`plan_read` **call**) → `MIGRATED_RANGE_READ`.
4. A bounded `ts_ms>=?...ts_ms<=?` (or similar) range pattern correlated to an allowlisted table
   in the same statement window → `REMAINING_MIGRATABLE_RANGE_READ_CANDIDATE` (tagged
   `mixed_partial_migration: true` if `lookup_call_present` is also `True` — see §5.3).
5. No allowlisted table referenced anywhere → `BLOCKED_DIFFERENT_DB` (if a non-`microstructure.db`
   `.db`/`.sqlite` literal is present) or `NO_OP_NO_ALLOWLISTED_RANGE_PATTERN`.
6. Allowlisted table present, no allowlisted-table range pattern, but an `ASC`/dynamic-direction
   `ORDER BY ts_ms` on that table → `BLOCKED_FORWARD_ASOF_PRIMITIVE`.
7. Allowlisted table present, `DESC LIMIT 1` ASOF pattern on that table, `lookup_call_present` is
   `False` → `ASOF_ONLY_NOT_RANGE_READ_SCOPE` (belongs to a point-lookup gate, not range-read).
8. Only an unbounded `MAX(ts_ms)`/`MIN(ts_ms)` probe → `OUT_OF_SCOPE_UNBOUNDED_SCAN`.
9. Otherwise → `NO_OP_NO_ALLOWLISTED_RANGE_PATTERN` (allowlisted table mentioned in an unrelated
   context — string literal, comment, or a query shape none of the above patterns cover).

### 5.2 Resolved / migrated list (18) — `MIGRATED_RANGE_READ`

| # | Path | Gate |
|---|---|---|
| 1 | `tools/research_ami_mfe50_experiment.py` | pre-V1 pilot |
| 2 | `tools/research_s34_100k_notmon_check.py` | pre-V1 pilot |
| 3 | `tools/funding_rate_analysis.py` | V1 |
| 4 | `tools/research_s34_orderflow_lead.py` | V2 |
| 5 | `tools/research_s34_sell_reversal_filter.py` | V2 |
| 6 | `tools/research_s34_buy_reversal_short.py` | V3 |
| 7 | `tools/research_s34_sell_reversal_quality.py` | V3 |
| 8 | `tools/export_s34_visualization_json.py` | V4 |
| 9 | `tools/micro_edge_smoke.py` | V4 |
| 10 | `tools/research_s34_symbol_compare.py` | V4 |
| 11 | `tools/research_s34_sell_liq_bounce.py` | V5 |
| 12 | `tools/research_s34_sell_path_quality.py` | V5 |
| 13 | `tools/research_s34_sell_regime_analysis.py` | V5 |
| 14 | `tools/research_s34_exact_route_change_validation.py` | V6 |
| 15 | `tools/research_s34_sol200k_sell_dayfilter.py` | V6 |
| 16 | `tools/research_s34_source_quality_reconciliation.py` | `4a69880f` (dedicated prep+migration) |
| 17 | `tools/research_s34_micro_entry_scalp.py` | V7 |
| 18 | `tools/research_s34_exit_giveback_sweep.py` | V7 |

All 18 verified via `reader_v2_present=True` (an actual call site, immune to oracle-literal false
positives). Zero mismatches against the operator-supplied `KNOWN_MIGRATED` list.

### 5.3 Mixed-partial-migration subset (11 of the 56 remaining)

These already have their ASOF portion migrated (`lookup_call_present=True`) from a prior ASOF
gate, but retain a genuinely separate, allowlisted-table range-read pattern that was **deliberately
left on direct SQL at that time** (the plan doc's own §4.5 `MIXED_PARTIAL_MIGRATION` precedent —
e.g. `research_s34_consensus_composite.py`'s documented `ofir`/`rv5` computation,
`research_s34_btc_microtrend_sweep.py`'s documented TP/SL/BE scan). Not oversights — real, open,
separately-scoped range-read opportunities:

| Path | Remaining range table(s) |
|---|---|
| `tools/research_s34_btc_microtrend_sweep.py` | `mark_prices` |
| `tools/research_s34_consensus_composite.py` | `agg_trades` |
| `tools/research_s34_500k_daytrend_route_sweep.py` | `mark_prices` |
| `tools/research_s34_early_confirmation_scan.py` | `mark_prices` |
| `tools/research_s34_trailing_oos_realfill.py` | `mark_prices` |
| `tools/s34_regime_filter_shadow_eval.py` | `mark_prices` |
| `tools/research_s34_v6_management_system.py` | `agg_trades`, `book_ticker` |
| `tools/research_eth_provision_realism.py` | `book_ticker` |
| `tools/research_nonpredictive_carry_provision.py` | `book_ticker` |
| `tools/research_s34_eth_preliq_control.py` | `book_ticker` |
| `tools/research_s34_eth_preliq_executable.py` | `book_ticker` |

## 6. No-op / out-of-scope lists

### 6.1 `NO_OP_NO_ALLOWLISTED_RANGE_PATTERN` (135) and `DO_NOT_TOUCH_BY_NAME` (62)

Full lists are in the committed JSON (`records[].classification`); not reproduced here for length
— 197 of 320 files, the large majority of the codebase's `microstructure.db`-touching scripts,
either never reference an allowlisted table with a bounded range shape at all, or are excluded by
the CLAUDE.md-generalized filename guardrail (`v_engine`, `shadow_*`, `state_machine`, `live_*`,
`dashboard`, `scheduler`, etc. — unchanged from prior gates, never re-scoped here).

### 6.2 Special-cased files (manual override, full read, not just regex)

**`tools/research_s34_cross_symbol_lag.py` → `NO_OP_NO_ALLOWLISTED_RANGE_PATTERN`.** Manually
read in full. Its only bounded range pattern (`follower_liq_nearby()`,
`ts_ms>=?...ts_ms<=?...ORDER BY abs(ts_ms-?) ASC LIMIT 1`) targets `liquidations`
(out-of-allowlist) — confirmed by the per-statement scan (`ooa_range_tables=['liquidations']`,
`allowlist_range_tables=[]`). Its only allowlisted-table (`mark_prices`) query, `mark_at()`, is a
dynamic-direction ASOF point-lookup (`order by ts_ms {order}`, called exclusively with
`before=False`, i.e. forward/ASC), not a range scan. Matches the prior V3 finding exactly: "no
allowlisted range-read; only ASOF mark_at + liquidations." **Confirmed by re-reading the file, not
merely trusted from the prompt.**

**`tools/walkforward_eval.py` → `NO_OP_EXECUTE_READ_CONTRACT_MISMATCH`.** Manually read in full.
`_slice_price_window()` has the file's only allowlisted-table range pattern: `SELECT ts_ms,
symbol, price FROM agg_trades WHERE ts_ms>=? AND ts_ms<=? ORDER BY ts_ms ASC, rowid ASC` —
**with no `symbol=?` predicate in the SQL at all.** It fetches every symbol in the window and
filters a `symbol_set` client-side in Python afterward. `research_reader.plan_read()`'s contract
requires one resolved `symbol` per call (the catalog-matching key) — this shape does not fit
without either calling `plan_read` once per requested symbol (a real behavior change: today's code
issues exactly one SQL query per slice regardless of symbol count) or a new multi-symbol reader
primitive (not authorized in any range-read gate to date). Also `mode_ro=False` (plain
`sqlite3.connect`, no `?mode=ro` URI) and imports from `tools.eval_run`/`tools.replay_strategy`
(replay-adjacent) — even if the contract mismatch were resolved, an import-safety/`mode=ro` prep
step would be a **separate** gate, matching the prior V6/V7 finding referenced in the operator's
brief.

### 6.3 `BLOCKED_DIFFERENT_DB` (4)

`tools/db_maintenance.py`, `tools/run_execution_canary.py`, `tools/smoke_all.py`,
`tools/validate_env.py` — no allowlisted table reference; only non-`microstructure.db` DB literals
(operational/health-check scripts, not research consumers at all).

### 6.4 `OUT_OF_SCOPE_UNBOUNDED_SCAN` (6)

`tools/check_event_lanes.py`, `tools/prototype_ws_vs_db_latency.py`,
`tools/research_s34_prediction_image.py` (already ASOF-migrated in ASOF Batch 1 — this flag is for
an *additional*, separate unbounded probe elsewhere in the file, out of range-read scope),
`tools/research_s34_price_chart.py`, `tools/research_s34_short_setup_chart.py`,
`tools/research_s34_trade_explain_chart.py` — only unbounded `MAX(ts_ms)`/`MIN(ts_ms)` probes, no
bounded window to migrate.

### 6.5 `BLOCKED_FORWARD_ASOF_PRIMITIVE` (21) and `ASOF_ONLY_NOT_RANGE_READ_SCOPE` (17)

Neither bucket is range-read scope. The first (21 files) has an `ASC`/dynamic-direction
`ORDER BY ts_ms` on an allowlisted table (the same forward-ASOF shape documented for `mark_at()` in
`tools/research_s34_exit_giveback_sweep.py` during V7 — `research_reader.lookup_latest_at_or_before`
only supports the backward direction). The second (17 files) has an ordinary backward
`DESC LIMIT 1` ASOF pattern on an allowlisted table but has never been migrated to
`lookup_latest_at_or_before` at all — these would be new candidates for a **future ASOF/point-lookup
wave**, not for `RANGE-READ-CONSUMER-MIGRATION-V8`. No forward-ASOF helper is proposed, built, or
implied here, per explicit instruction.

## 7. True remaining migratable range-read list (56)

Full list with per-file signals is in the committed JSON
(`records[].classification == "REMAINING_MIGRATABLE_RANGE_READ_CANDIDATE"`). 11 of the 56 are the
mixed-partial subset (§5.3); the other 45 are untouched files with a genuine, verified, bounded,
allowlisted-table range pattern that has never been migrated in any gate.

## 8. Recommended V8 targets

Ranked by `mode_ro=True` first, no `ami.knowledge`/`ami.research`/`ami.governance` import, then
ascending line count (smallest/cleanest first, matching every prior gate's stated preference) —
restricted to the 45 non-mixed-partial files to keep V8's shape identical to V1–V7 (a plain,
never-touched consumer, not a partial-migration continuation, which is a materially different
review shape better suited to its own gate):

| Rank | Path | Lines | Table |
|---|---|---:|---|
| 1 | `tools/research_s34_hold_sweep.py` | 173 | `mark_prices` |
| 2 | `tools/research_s34_session_analysis.py` | 179 | `mark_prices` |
| 3 | `tools/research_s34_post_tp_continuation.py` | 181 | `mark_prices` |

All three: untracked, `mode=ro`, no governance import, single allowlisted range table, no
DO_NOT_TOUCH name collision — same shape as every one of V1–V7's targets. **Not started in this
gate** — recommendation only, per explicit instruction that this reconciliation gate does not
migrate anything.

Separately, if the operator wants to close out the mixed-partial subset (§5.3) as its own track,
`tools/research_s34_hold_sweep.py`-shaped candidates should still come first; the mixed-partial
files need an extra check (confirming the ASOF and range portions don't share mutable state)
before touching them, out of scope for this gate to assess.

## 9. Guardrail proof

Checked before and after this gate's work:

| Check | Before | After |
|---|---|---|
| Catalog `entry_count` | 3 | 3 (unchanged) |
| Catalog `index_self_hash` | `b2b26d06ff19800298c50c418892f4f5daeb8fb9a1ad9b674824503b3dc466f2` | identical |
| `data/archives/raw_v1.staging` file count | 0 | 0 |
| `.lock` files under `data/archives` | 0 | 0 |
| `microstructure.db` opens `mode=ro` | yes | yes |
| Availability audit artifacts present | all 5 confirmed (§ preflight) | unchanged, untouched |
| Source mutations | 0 | 0 |

Commit chain preflight (all 10 required commits confirmed ancestors of `HEAD`, current branch
`codex/data-layer-fallback-cleanup`): `aaebffa4`, `783e7282`, `c619d90b`, `d2bf15ef`, `4a69880f`,
`29878236`, `cf676357`, `861cf798`, `2d1746ee`, `393155fd` — plus V7 (`e66b956a`, §2).

Note: `reports/governance/alpha/ALPHA_CANDIDATE_AVAILABILITY_AUDIT_V1_FAMILY_RECORD.json` named
in the preflight checklist does not exist on disk — only the `.md` sibling
(`ALPHA_CANDIDATE_AVAILABILITY_AUDIT_V1_FAMILY_RECORD.md`) and the top-level
`ALPHA_CANDIDATE_AVAILABILITY_AUDIT_V1.json` do. This gate did not create it (not in scope) and
flags the discrepancy for the operator rather than silently treating it as present.

## 10. No data mutation statement

This gate wrote exactly two new files under `reports/governance/storage/` (this report and the
reconciled JSON) and two new read-only static-text scanner scripts under `tools/`. It did not
modify, migrate, or change the behavior of any consumer script; did not write to
`microstructure.db`, the archive, any manifest, catalog, or shard; did not run any research/alpha
script; did not touch `runtime/dashboard_backend.json`, any shadow/dashboard file, or any
production runner/scheduler path. `git status` immediately before commit shows only this gate's 4
new files as additions relative to the V7 commit; every pre-existing dirty file in the working
tree (`IMPLEMENTATION_PROGRESS_LEDGER.md`, `SYSTEM_STATE.md`, `TEST_STATUS_LATEST.md`,
`runtime/dashboard_backend.json` deletion, `tools/s34_cascade_navigation_dashboard.py`,
`tools/s34_realtime_shadow_runner.py`, and the large set of untracked docs/reports) is pre-existing
operator/session state, not touched or staged by this gate.

## 11. Recommended next gate

`BATCH-STORAGE-ROTATION-RETENTION-RANGE-READ-CONSUMER-MIGRATION-V8`, targeting the 3 files in §8,
only after this reconciliation gate closes. Not alpha rehearsal, purge, scheduler, VACUUM, or
live/paper activation.
