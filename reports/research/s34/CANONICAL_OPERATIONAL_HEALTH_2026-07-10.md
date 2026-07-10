# Canonical Operational Health — Corrective Pass, 2026-07-10

Response to: `CANONICAL_OPERATIONAL_HEALTH_V1_CORRECTIVE_REVIEW` (provisional
verdict `CANONICAL_OPERATIONAL_HEALTH_CORRECTIVE_CHANGES_REQUIRED`).

Scope discipline observed throughout: no commits made, live executors left
disabled, no full runtime restart, no `collection_watchdog` process started,
no trading state/checkpoint/`.env`/`risk/`/`brain/` file touched.
`execution/health_gate.py` was edited under explicit operator sign-off
obtained mid-session for this one scoped change (removing its overall.json
merge-write), per CLAUDE.md's execution/-directory guardrail.

---

## 1. Blocker Resolution

| Blocker | Status |
|---|---|
| A. Genuinely single writer for `overall.json` | **RESOLVED** |
| B. Research fitness safe on production-shaped DB | **RESOLVED** (deeper than originally scoped — see §5) |
| C. Health tests isolated from live runtime | **RESOLVED** (surfaced and fixed one real product defect — see §7) |
| D. Watchdog cadence vs staleness budget | **RESOLVED** — interval reduced 10s→5s on measured evidence |

## 2. Final Writer Ownership Map

| File | Sole writer | Notes |
|---|---|---|
| `logs/health/overall.json` | `tools/heartbeat_watchdog.py` (`build_canonical_overall` + `run_once`) | Enforced by `tools.health_state.write_component_health` rejecting `component="overall"`, by deletion of the generic `write_overall_health` helper, and by a source-scan test (`tests/test_health_writer_ownership.py`) |
| `logs/health/watchdog.json` | `tools/heartbeat_watchdog.py` | unchanged |
| `reports/WATCHDOG_STATUS.json` | `tools/heartbeat_watchdog.py` | unchanged |
| `logs/health/collector.json` | `data/microstructure_collector.py` | unchanged (already component-only) |
| `logs/health/bookticker.json` | `data/bookticker_collector.py` | unchanged (already component-only) |
| `logs/health/paper_trader.json` | `execution/health_gate.py::write_paper_trader_health` | **fixed**: no longer also merge-writes `overall.json` |
| `logs/health/replay.json` | `tools/replay_slice.py::_write_replay_health` | **fixed**: no longer also merge-writes `overall.json`; now also accepts `--health-root` for isolated/test use |
| `logs/health/research_fitness.json` | `tools/research_fitness_report.py` | advisory-only; **since the final corrective round, structurally enforced**: its writer rejects every protected operational basename (resolved, case-insensitive) before any write, and the production CLI no longer accepts an output path at all |
| `logs/health/heartbeat.json` | `execution/guardian.py` / `monitoring/prometheus.py` | **separate filename, no ownership conflict** — see note below |

`tools/health_state.write_overall_health` was **deleted outright** (not
deprecated) — it had exactly two callers (`execution/health_gate.py`,
`tools/replay_slice.py`), both fixed, leaving it dead and an attractive
nuisance for a future accidental second writer.

**`logs/health/heartbeat.json` (added to the map in the final corrective
round, per the independent review's inventory finding):** two dormant
modules — `execution/guardian.py` and `monitoring/prometheus.py` — each
write this file via `execution/runtime_helpers.py::atomic_write_json`, a
third, unrelated generic-writer family. This is a **different filename**
from every canonical/component output above; it does not violate
`overall.json` single-writer ownership, is not read or aggregated by
`tools/heartbeat_watchdog.py`, and both writer modules belong to the
live-executor stack, which is not running. Classification:
**documentation-only follow-up** — no code cleanup is needed for the
canonical-health architecture; if the live-executor stack is ever
activated, the two modules writing the same filename should at that point
be reviewed under the same single-writer discipline (tracked here, not
blocking).

**Protected-output enforcement (final corrective round):** research fitness
can no longer target any operational-health file even in principle.
`tools/research_fitness_report.py::_atomic_write_json` resolves its target
and raises `ProtectedOperationalOutputError` — before creating any
directory or temp file — if the resolved basename matches (case-
insensitively) any of: `overall.json`, `watchdog.json`,
`WATCHDOG_STATUS.json`, `collector.json`, `bookticker.json`,
`paper_trader.json`, `replay.json`. Relative aliases and traversal are
neutralized by full resolution; symlinks are followed to their real target
before the check; Windows case aliases are covered by the lowercase match.
The production CLI writes exactly `logs/health/research_fitness.json`
(`--out` removed); the deprecated `tools/collection_watchdog.py` wrapper
imports this same guarded writer and therefore cannot bypass it (proven by
`test_collection_watchdog_wrapper_cannot_bypass_protection`).

## 3. Paper-Trader Component Migration

`execution/health_gate.py::write_paper_trader_health` already called
`write_component_health("paper_trader", payload, root=root)` — that half was
correct. The defect was purely the *additional* block immediately after it:
a `load_overall_health` → mutate `components["paper_trader"]` → conditionally
mutate `state`/`reason` → `write_overall_health` sequence. That block is
deleted; the function now does exactly one write (its own component file)
and returns. `tools/replay_slice.py::_write_replay_health` had the identical
pattern (`write_component_health("replay", ...)` immediately followed by a
hand-built `overall` dict and `write_overall_health(overall)`) and received
the identical fix.

Proof (`tests/test_health_writer_ownership.py`,
`tests/runtime/test_health_gate_unit.py`):
- `write_paper_trader_health` called against an isolated root creates only
  `paper_trader.json`, never `overall.json`.
- A pre-existing `overall.json` is left byte-for-byte untouched by a
  `write_paper_trader_health` call (no read-then-write at all now).
- A simulated interleaving — one watchdog cycle, then a paper-trader write,
  then a second watchdog cycle — proves the paper-trader payload appears in
  the *next* canonical cycle intact, with neither writer able to clobber the
  other (`test_concurrent_paper_trader_and_watchdog_updates_cannot_lose_each_other`).

## 4. Canonical Aggregation Design

`tools/heartbeat_watchdog.py::build_canonical_overall` previously read the
*existing* `overall.json` and "preserve-merged" any component it didn't
itself own (`{k: v for k, v in existing_components.items() if k not in
{"collector","bookticker","watchdog"}}`). That is exactly the "unknown
historical placeholder copied indefinitely" failure mode Part E warns
against, and it was live in production (see §11) — a `paper_trader` entry
from a paper-trading run that stopped on **2026-04-21** was still being
copied into every `overall.json` write as of this morning.

New design: `OPTIONAL_COMPONENT_FILES = {"paper_trader": "paper_trader.json",
"replay": "replay.json"}`. Every cycle, `_read_optional_components()` reads
each file **directly from disk**, fresh — never from a previous
`overall.json`. Missing or corrupt → omitted entirely (not fabricated, not
carried forward). Present → included **verbatim**, including its own
`ts_utc`, so a stale component's age is still computable by any reader; nothing
here ever rewrites a component's own timestamp to make it look fresher than
it is (`test_stale_paper_trader_component_remains_visibly_stale_after_rewrite`).

Per-component contract:

| Component | Source | Required | Owner | Contributes to top-level state |
|---|---|---|---|---|
| `collector` | `logs/health/collector.json`, read every cycle | yes | `data/microstructure_collector.py` | yes (critical if down, warning if degraded) |
| `bookticker` | `logs/health/bookticker.json` | conditional (`--expect-bookticker`) | `data/bookticker_collector.py` | yes when expected |
| `watchdog` | self-authored in-memory each cycle | yes | `tools/heartbeat_watchdog.py` | descriptive only |
| `paper_trader` | `logs/health/paper_trader.json` | no | `execution/health_gate.py` | no (advisory/observational — see rationale below) |
| `replay` | `logs/health/replay.json` | no | `tools/replay_slice.py` | no (advisory/observational) |

`paper_trader`/`replay` deliberately do not gate top-level severity: the
actual live-safety consumer (`execution/health_gate.py::evaluate_health_gate`)
already gates on top-level `state` and `components.collector` only, and
`paper_trader`'s own status is *itself derived from* a prior cycle's
canonical `overall.json` (via `evaluate_health_gate` reading it) — folding it
back into severity would create a same-cycle feedback loop without adding
independent signal. This is an explicit design choice, not an oversight.

`runtime_mode` also changed: previously fell back to
`existing.get("mode") or "paper"` (a stale prior mode, possibly `"live"`,
could be echoed forever if `runtime_launcher_status.json` ever went
missing); now falls back straight to `"paper"` — an unknown current mode
must never echo a possibly-stale `"live"`.

## 5. Research-Fitness Query Bounding

Two independent unbounded-cost defects were found and fixed (the review's
own hypothesis — `detector_heartbeat` scanned via `inspect_tables()` — was
real but turned out **not to be the dominant cost**):

1. **`tools/check_data_ready.py::inspect_tables()`** iterated every table
   returned by `list_tables()`. Fixed: new `table_allowlist` parameter
   (`None` preserves old full-scan behavior for other callers, e.g.
   `tools/micro_collector_watchdog.py`, which this fix does not touch).
   `tools/check_data_ready.RESEARCH_FITNESS_TABLE_ALLOWLIST =
   ("mark_prices", "agg_trades", "liquidations")` — exactly the three
   tables research fitness's own sample-stats/feature-fitness logic already
   queries directly. `detector_heartbeat` (and any other table) is now never
   inspected by this path, regardless of its size or index state.
2. **`tools/validate_data_research_fitness.py::_symbol_sample_stats` /
   `_feature_fitness`** (not previously flagged) — `_symbol_sample_stats` ran
   an unbounded, all-time `COUNT(*) ... WHERE symbol=?` per table per symbol;
   `_feature_fitness` called `load_symbol_window(conn, symbol)` with **no**
   `start_ms`/`end_ms`/`limit` at all, i.e. it fetched a symbol's entire
   historical row set, across every ts+symbol-keyed table, into a Python
   list. Against the real ~792GB `data/microstructure.db` this — not
   `detector_heartbeat` — is what actually made the one-shot tool hang past
   90s even after fix #1 was applied (confirmed empirically, see §6). Fixed:
   both now use a bounded recent-activity window
   (`_RECENT_ACTIVITY_WINDOW_SEC = 600`) plus a hard row cap
   (`_FEATURE_ROW_LIMIT_PER_TABLE = 2000`), using the existing
   `(symbol, ts_ms)` composite indices (`idx_trade_symbol_ts`,
   `idx_mark_symbol_ts`, `idx_liq_symbol_ts` — confirmed present on the real
   DB) for an index range scan instead of a full/near-full scan. This is
   the semantically correct behavior too, not just a performance patch:
   research fitness asks "is there usable data *right now*", not "what has
   ever existed".

No index or migration was added — the required tables already have the
composite indices the bounded queries need; only unrelated tables (never
touched now) lacked indexing.

Additionally (found while making the production run safe, §6): every SQLite
connection to `data/microstructure.db` in this call path
(`validate_data_research_fitness.py`, `validate_microstructure_contract.py`,
`src/microphys/io/sqlite_reader.py::discover_mappings`) was a plain
read-write `sqlite3.connect()`, violating CLAUDE.md's `mode=ro` guardrail
for this database. All three now open `f"file:{path}?mode=ro", uri=True`,
matching the established pattern already used elsewhere in the codebase
(`ami/storage/source_access.py`). `mode=ro` also fails loudly instead of
silently creating an empty file when a path doesn't exist.

Tables research fitness requires and why: **`mark_prices`, `agg_trades`,
`liquidations`** — these are the only tables `_symbol_sample_stats` and the
DB-freshness check ever query; `detector_heartbeat` and any other
operational table are unrelated to research-data fitness and are never
inspected.

Tests (`tests/test_validate_data_research_fitness.py`, 14 tests total in
that file, all passing):
1. `test_large_unrelated_table_excluded_from_allowlisted_scan` — a 50k-row
   unindexed `detector_heartbeat`-shaped table never appears in the returned
   diagnostics.
2. `test_unbounded_scan_still_available_for_other_callers` — `None`
   allowlist still full-scans, proving no behavior change for other
   `inspect_tables()` callers.
3. `test_research_fitness_bounded_runtime_against_production_shaped_fixture`
   — 50 required rows + 200k-row unrelated table, asserts `elapsed < 5.0s`.
4. `test_missing_unrelated_tables_do_not_block_fitness`.
5. `test_missing_required_table_is_deterministically_blocked` — `status ==
   "blocked"` with a populated `failures` list, not a silent pass.
6. `test_cli_writes_only_the_dedicated_output_file` (pre-existing, still
   passing) — proves #6/#7.
7. `test_evaluation_never_mutates_the_database` — sha256 of the DB file
   unchanged before/after a call that hits failure paths.
8. Exit-code determinism: `_EXIT_CODE_MAP = {"ready": 0, "limited": 1,
   "blocked": 2}` (pre-existing, unchanged, re-verified by
   `tests/test_research_fitness_report.py`).

## 6. Research-Fitness Live Result

Before the deeper fix (§5 item 2), a real one-shot run against
`data/microstructure.db` was attempted with a 90s hard timeout and **timed
out (exit 124)** — proving the review's premise correct even with the
`detector_heartbeat` allowlist fix alone applied. After both fixes:

```
$ time timeout 60 python -m tools.research_fitness_report \
    --db data/microstructure.db --csv data/event_diary.csv \
    --symbols BTCUSDT,ETHUSDT --out logs/health/research_fitness.json
research_fitness status=ready raw_status=pass contract_tier=full_book db_ready=True warnings=0 failures=0
real  0m1.107s
```

- Process completed normally, exit code 0.
- `logs/health/research_fitness.json` produced atomically (temp-file +
  `os.replace`, verified no leftover `.tmp_*`).
- Elapsed time reported above: **1.107s** (vs. >90s before, i.e. >80x).
- Tables evaluated: `mark_prices`, `agg_trades`, `liquidations` only.
- DB opened `mode=ro` (see §5) — no mutation possible by construction, not
  merely by absence of INSERT statements.
- `logs/health/overall.json` (`state`) and `reports/WATCHDOG_STATUS.json`
  (`overall`) severity confirmed unchanged by this run (`state=ok`/`GREEN`
  before and after, `research_fitness` key correctly absent from
  `overall.json`'s `components`).

## 7. Test-Isolation Correction

`tests/test_health_cycle_smoke.py` previously passed
`health_file="logs/health/overall.json"` and `db_path="data/microstructure.db"`
directly — the real repo paths — with no root override at all, and spawned
`data.microstructure_collector` as a real subprocess that (before this
session's earlier Part-A fix) wrote directly into the real
`logs/health/collector.json`, racing the actually-running production
collector. Worse: because `overall.json` is now (correctly, post-Part-A)
*only* written by `heartbeat_watchdog`, this test could never have produced
a meaningful `overall.json` transition at all without silently depending on
the real, already-running production watchdog picking up its stray
`collector.json` writes within the test's ~15s window.

Fix:
- `data/microstructure_collector.py` gained a `--health-root`/`health_root`
  parameter (default `"logs/health"`, i.e. unchanged for real launches) so
  an isolated caller can redirect its `collector.json` write.
- `tools/health_cycle_smoke.py` gained `--root` (scopes `logs/health/`,
  `reports/`, `logs/collector_heartbeat.json` for this run) and
  `--seed-market-data` (creates+seeds an isolated DB only if it doesn't
  already exist — never touches a real one). It now runs one
  `heartbeat_watchdog` evaluation cycle **in-process** every poll tick,
  scoped to `--root` via a save/restore of `hw.ROOT`/`hw.LOG_HEALTH`/
  `hw.REPORTS` — no second Python process, no dependency on any real
  background watchdog.
- `tests/test_health_cycle_smoke.py` now passes an isolated `--root` and
  `--db-path` under `tmp`/`localtests`, and a new
  `test_run_smoke_creates_no_files_in_real_repo_health_dir` asserts the
  *set of filenames* under the real `logs/health/` is unchanged after an
  isolated run (filename-set, not mtime/content, so the assertion holds even
  while the live watchdog is independently rewriting its own files in
  place — verified against the actually-running production system).

**Separate deterministic product defect found and fixed** (exactly the
scenario Part C anticipated — "expose a separate deterministic product
defect, then report precisely"): the smoke test's `_validate_snapshot`
required `components.collector.status == "degraded"` during the simulated
outage. `data/microstructure_collector.py::_write_heartbeat` computes that
`status` field from a staleness threshold (`stall_timeout_sec`, default
45s) — by design it does **not** flip on a bare disconnect, only after
progress has been stale for 45s. A `down_sec=2`–`6` simulated outage can
never make it read `"degraded"`; the canonical top-level `state` correctly
still goes `"degraded"` during the outage, but via the faster,
connection-level `native_ws_policy` signal (`NATIVE_WS_DISCONNECTED`), which
is the correct fast-detection layer this architecture is supposed to have.
The test's invariant was checking the wrong signal and could only ever have
"passed" by accident, via the leaked dependency on the live production file
described above. Fixed by checking `components.collector.connected is
False` (the field that genuinely does flip immediately) instead of
`.status`.

Result after isolation + the product-defect fix: **passes deterministically,
3/3, and confirmed stable across repeated runs** — not "harmless pre-existing
flakiness", a real fix.

## 8. Watchdog Cadence Evidence

Measured directly against the live, already-running production watchdog
(`--interval-sec 10`, passive observation only, zero new processes) over 90s:
9 consecutive real inter-write deltas — `[1.00, 11.28, 11.03, 11.03, 11.03,
11.03, 11.03, 11.03, 11.03]` — i.e. **actual cycle time ≈11.03s**, not the
nominal 10s: evaluation itself (chiefly `python_process_running()`'s
PowerShell `Get-CimInstance` subprocess spawn) costs ≈1.03s per cycle. Real
margin before the 15s consumer budget was thus ≈3.97s (≈1.36x), not the
≈1.5x a naive 15/10 ratio implies.

Decision: **reduced `DEFAULT_INTERVAL_SEC` 10→5** (and `start_eclipse.ps1`'s
`--interval-sec` argument to match). Evaluation cost is the same ≈1s
regardless of interval, so real cycle time becomes ≈6s, giving ≈9s of real
margin (≈2.5x). Re-measured after activation (§11): 6 consecutive deltas —
`[4.21, 6.04, 5.83, 6.03, 5.82, 6.04]` — confirming the predicted ≈6s real
cycle and ≈9s margin. Cost: roughly double the PowerShell subprocess spawns
(≈1 extra per 6s instead of per 11s) — negligible relative to the safety
gain.

Boundary tests added (`tests/runtime/test_health_gate_unit.py`):
- `test_health_staleness_just_below_limit_allows` (age=14.9s)
- `test_health_staleness_exactly_at_limit_allows` (age=15.0s exactly — the
  comparison is strict `>`, so equality must still pass)
- `test_health_staleness_just_above_limit_blocks` (age=15.1s)
- `test_health_staleness_two_delayed_watchdog_cycles_blocks` (age=22.0s)

All four constructed with an explicit `now_ts` (not real wall-clock) for
determinism.

## 9. Files Changed

Edited this session (all listed tests re-run and passing; nothing here was
committed):

- `execution/health_gate.py` — removed `overall.json` merge-write from
  `write_paper_trader_health` (operator sign-off obtained for this scoped
  edit)
- `tools/replay_slice.py` — same fix; added `--health-root`/`health_root`
- `tools/health_state.py` — deleted `write_overall_health`;
  `write_component_health` rejects `component="overall"`
- `tools/heartbeat_watchdog.py` — `build_canonical_overall` redesigned
  (component registry, no read-from-previous-`overall.json`);
  `DEFAULT_INTERVAL_SEC` 10→5
- `tools/check_data_ready.py` — `table_allowlist` param,
  `RESEARCH_FITNESS_TABLE_ALLOWLIST`
- `tools/validate_data_research_fitness.py` — `mode=ro`; bounded
  recent-window sample-stats/feature-fitness
- `tools/validate_microstructure_contract.py` — `mode=ro`
- `src/microphys/io/sqlite_reader.py` — `mode=ro` in `discover_mappings`
- `tools/research_fitness_report.py` — docstring updated (hazard resolved)
- `data/microstructure_collector.py` — `health_root` param/`--health-root`
  CLI flag
- `tools/health_cycle_smoke.py` — isolation redesign (`--root`,
  `--seed-market-data`, in-process watchdog evaluation, fixed
  `_validate_snapshot`)
- `start_eclipse.ps1` — heartbeat_watchdog `--interval-sec` 10→5
- `tests/runtime/test_health_gate_unit.py` — updated
  `write_paper_trader_health` tests + 4 staleness boundary tests
- `tests/test_canonical_health_gate_integration.py` —
  `existing_path`→`log_health` param rename
- `tests/test_heartbeat_watchdog.py` — rewrote preserve-component tests;
  added never-reads-previous-overall / stale-component-stays-stale tests
- `tests/test_health_writer_ownership.py` — **new file**
- `tests/test_validate_data_research_fitness.py` — bounding tests, `mode=ro`
  compatibility, frozen-time fix for `test_main_writes_outputs`
- `tests/test_health_cycle_smoke.py` — full isolation rewrite
- `tests/test_replay_slice.py` — isolation fix + regression guard

Deliberately **not** touched — list derived directly from `git status` in
the final corrective round (superseding the earlier from-memory version,
which omitted `tests/test_collection_watchdog.py`). Pre-existing
working-tree modifications from before this corrective pass began,
read/verified but not further edited:
`tools/native_ws_health_policy.py`, `tests/test_native_ws_health_policy.py`,
`tools/collection_watchdog.py`, `tests/test_collection_watchdog.py`,
`tools/s34_cascade_navigation_dashboard.py`,
`tools/s34_realtime_shadow_runner.py`; additionally `status_eclipse.ps1`,
`stop_eclipse.ps1`, the deleted `runtime/dashboard_backend.json`, and
`TEST_STATUS_LATEST.md` — all likewise pre-existing and untouched by this
pass.

**Final corrective round additions (same day, after the independent
review):** `tools/research_fitness_report.py` (structural protected-output
guard: `PROTECTED_OPERATIONAL_OUTPUT_BASENAMES` +
`ProtectedOperationalOutputError` inside `_atomic_write_json`, resolved-
basename + case-insensitive matching, rejection before any write; `--out`
removed from the production CLI — output is fixed to
`logs/health/research_fitness.json`, path injection survives only in the
internal API, which itself rejects every protected name; the deprecated
`tools/collection_watchdog.py` wrapper inherits the guard automatically
because it imports this very `_atomic_write_json`, so that pre-existing
file needed no edit), `tests/test_research_fitness_report.py` (+6 guard
tests, CLI test reworked for the removed `--out`, substring module-scan
converted to ast), `tests/test_health_writer_ownership.py` (fragile
substring assertion replaced with ast-based structural analysis — comments
and docstrings are invisible to it, health_gate's read-only
`load_overall_health` default is the single allowed `overall.json` literal;
+3 new tests: docstring/line-wrap regression control, heartbeat_watchdog
positive-ownership control, protected-output rejection coverage; source
reads use `utf-8-sig` because `tools/heartbeat_watchdog.py` carries a BOM
that plain-utf-8 `ast.parse` rejects).

## 10. Exact Test Results

> **SUPERSEDED (final corrective round, same day).** The 241/241/0 total
> originally recorded here was measured before a later docstring edit to
> `tools/replay_slice.py` (made during controlled activation) and was never
> re-run after that edit; the independent review reproduced 240 passed /
> 1 failed against that state (the fragile substring assertion in
> `test_known_former_writers_no_longer_reference_overall_json`, broken by a
> harmless docstring line wrap — writer ownership itself was never
> violated). The authoritative totals are the post-correction run below,
> executed after the final code/test edit of the final corrective round,
> with zero edits afterward.

| Batch | Files | Result (final run) |
|---|---|---|
| 1 | test_health_writer_ownership.py + test_heartbeat_watchdog.py / test_canonical_health_gate_integration.py (two ≤2-file calls) | 32 + 6 = 38 passed |
| 2 | runtime/test_health_gate_unit.py | 26 passed |
| 3 | test_validate_data_research_fitness.py + test_research_fitness_report.py | 20 passed |
| 4 | test_validate_microstructure_contract.py + test_microstructure_sample_fixture.py | 7 passed |
| 5 | test_collector_simulation.py + test_collector_checkpoint_interval.py | 2 passed |
| 6 | test_health_cycle_smoke.py | 3 passed |
| 7 | test_native_ws_health_policy.py + test_collector_supervisor_cleanup.py | 22 passed |
| 8 | test_entry_loop_gate_integration.py + test_dashboard_overview_api.py | 5 passed |
| 9 | test_ami_host_health_observation.py + test_ami_host_health_evaluator.py | 98 passed |
| 10 | test_s34_live_chart_host_health.py + test_health_check.py | 11 passed |
| 11 | test_health_check_stale.py + test_collection_health.py | 6 passed |
| 12 | test_collection_watchdog.py + test_status_snapshot.py | 9 passed |
| 13 | test_push_status.py + test_replay_slice.py | 3 passed |

**Final total: 250 collected, 250 passed, 0 failed, 0 skipped** across the
same 20 test files. The expansion over the reviewed 241 is exactly the 9
new tests added by the final corrective round (+6 protected-output guard
tests in `test_research_fitness_report.py`, +3 structural-ownership tests
in `test_health_writer_ownership.py`); every one of the original 241 test
node IDs is retained by name and passes within this run — the original
reviewed set reproduces 241/241 as a strict subset.

## 11. Controlled Activation

Pre-activation process inventory (`Get-CimInstance Win32_Process`, read-only
enumeration): `collector_supervisor.py`, `data.bookticker_collector`,
`data.oi_spot_poller`, `tools.s34_shadow_paper_runner`, `tools.s34_live_chart`,
`tools.s34_v_engine_v02_shadow_mirror`, `data.event_diary`,
`tools.s34_realtime_shadow_runner`, `tools.orderflow_chart`,
`tools.s34_replay`, `data.microstructure_collector`,
`tools.heartbeat_watchdog --interval-sec 10` (PID 21352, running the
pre-fix code). Zero `collection_watchdog` process. Zero live-executor
process (`entry_loop`, `s34_live_order_executor`,
`s34_v_engine_live_executor`, `s34_state_machine_live_executor` all absent).
`s34_shadow_paper_runner.py` confirmed (grep) to have no dependency on
`execution/health_gate.py` — unaffected by the health_gate.py fix.

Only `tools.heartbeat_watchdog` had code changes that alter its behavior
(single-writer redesign, new interval), so only it was cycled: the old
process (PID 21352) was stopped and a new one started with the corrected
code and `--interval-sec 5 --max-age-sec 420 --expect-bookticker` (matching
the updated `start_eclipse.ps1`), PID and metadata re-registered under
`logs/pids/heartbeat_watchdog.{pid,json}` in the same format
`start_eclipse.ps1` itself uses. `data/microstructure_collector.py`'s change
(new `health_root` param, default identical to prior hardcoded behavior) is
behavior-neutral for the real running collector — not restarted.

Post-activation verification, live:
- Exactly one `microstructure_collector`, one `heartbeat_watchdog` (new
  PID), zero `collection_watchdog`, zero live executor, zero duplicates
  (re-enumerated).
- `overall.json.state == "ok"`, `WATCHDOG_STATUS.json.overall == "GREEN"` —
  agree.
- `native_ws_status == "GREEN"`, `rest_fallback.active == False`.
- `source_freshness`: `agg_trades` age 2.6s, `mark_prices` age 3.4s,
  `liquidations` age 3.9s — all fresh.
- Real cadence re-measured post-activation: ≈6s/cycle (§8), well inside the
  15s budget.
- `components` observed live: `collector`, `bookticker`, `watchdog`, plus
  **`paper_trader`** — genuinely present on disk from a paper-trading run
  that stopped **2026-04-21** (`reason: "alpha_metrics_missing"`, age at
  observation time ≈80 days / 6,906,834s) — a live, real-world confirmation
  that a stale optional component is surfaced with its own true age, not
  silently dropped or refreshed.
- A **`replay`** component was also observed, `age≈135s` — this was not
  legitimate: it was a side effect of this session's own
  `tests/test_replay_slice.py` run (batch 13, §10) writing into the real
  `logs/health/replay.json` before that test's isolation fix (§9) landed.
  Removed (`logs/health/replay.json` deleted) as a corrective action;
  re-verified it does not reappear (confirming components are read fresh
  from disk each cycle with no fabrication — once the source file is gone,
  the component correctly disappears from `overall.json` on the next
  cycle, while `paper_trader` correctly remains, since *its* file is real).
- Research-fitness one-shot re-confirmed complete (1.1s) with `overall.json`
  severity unchanged before/after (§6).
- No DB/checkpoint mutation: all research/contract-analysis connections
  against `data/microstructure.db` now open `mode=ro`.
- No duplicate process created.

## 12. Live Health Agreement

`overall.json.state=ok` ⇔ `WATCHDOG_STATUS.json.overall=GREEN` — confirmed
equal at multiple observation points before, during and after activation.

## 13. Process / Executor Safety

Live executors (`s34_live_order_executor`, `s34_v_engine_live_executor`,
`s34_state_machine_live_executor`) confirmed **absent** both before and
after this pass — never started, never enabled. No `.env`, `execution/`
gating logic (thresholds/leverage/sizing), `risk/`, or `brain/` file altered
beyond the one sign-offed, scoped `write_paper_trader_health` edit. No full
runtime restart — 11 of 12 pre-existing processes were never touched; only
`heartbeat_watchdog` was cycled.

## 14. Worktree Isolation

All test isolation described in §7 applies equally to worktree-style
disposable runs: `tools/health_cycle_smoke.py --root` and
`data/microstructure_collector.py --health-root` accept any path, including
one under a throwaway worktree, with no dependency on the invoking
process's cwd being the real repo beyond the (unchanged) production
defaults.

## 15. Findings

1. The originally-hypothesized `detector_heartbeat` full-scan was real but
   not the dominant cost; an unbounded per-symbol `COUNT(*)` and a fully
   unbounded `load_symbol_window()` fetch in
   `tools/validate_data_research_fitness.py` were the actual cause of the
   90s+ hang, only found by running the tool against the real production DB
   after the first fix and observing it still hang.
2. Every connection this call path opened against `data/microstructure.db`
   was read-write, violating the CLAUDE.md `mode=ro` guardrail for this
   database; none of the three sites needed write access. Fixed.
3. `tools/heartbeat_watchdog.py`'s "preserve unowned components from the
   previous `overall.json`" logic was live in production and had been
   silently echoing an 80-day-old `paper_trader` placeholder into every
   cycle's output — a direct, real-world instance of the exact failure mode
   Part E warned about.
4. `tests/test_health_cycle_smoke.py`'s degraded-phase assertion encoded a
   false invariant (`collector.status=="degraded"` on any disconnect) that
   the collector's actual staleness-gated status semantics can never
   satisfy for a short outage; it could only ever have passed by accident,
   via the very production-path leak Part C describes.
5. Running this session's own (now-fixed) `tests/test_replay_slice.py`
   during the Part F test sweep left a transient stray `replay.json` in the
   real `logs/health/`, which then legitimately appeared as an advisory
   component in the live `overall.json` — caught, explained, and cleaned up
   in §11 rather than left unexplained.

## 16. Final Verdict

`CANONICAL_OPERATIONAL_HEALTH_CORRECTED_AND_VERIFIED`

---

## 17. Commit Closure (2026-07-10, same day)

Final independent acceptance review (see the companion
`_INDEPENDENT_REVIEW.md` file and its follow-up section) returned
`CANONICAL_OPERATIONAL_HEALTH_ACCEPTED_WITH_LOW_FINDINGS` — 0 HIGH, 0
MEDIUM, both prior MEDIUM findings resolved, original 241-test set
reproduced as a passing subset, expanded 250-test set independently
reconfirmed, production research-fitness reproduced at 0.500s, a
repository-wide AST single-writer audit across 1,012 non-test modules
(plus live symlink and NTFS 8.3-short-name alias rejection probes) found
zero violations.

Committed as four content commits plus this governance commit, on branch
`codex/data-layer-fallback-cleanup` (not pushed):

| # | Hash | Subject |
|---|---|---|
| 1 | `00ef49ad5d0d8a94acaba76795fadddec5c98534` | `feat(health): enforce single-writer canonical health aggregation` |
| 2 | `81ec6d7139b4148b237fc54244c182e0c427cbc1` | `feat(research-fitness): bound read-only evaluation and protect health outputs` |
| 3 | `f3d95f5eb161e828cd106581e997056667d6c40d` | `test(health): isolate health-cycle smoke from live runtime` |
| 4 | `6faa2177648d40ecf147392d1848405282294933` | `chore(health): set watchdog interval from measured freshness budget` |

The eight pre-existing unrelated working-tree items (§9) were left
untouched and are not part of any of these commits (verified via
per-commit `git diff --cached --stat` before each commit). No process was
restarted during this commit closure (the one required watchdog cycle
happened during the earlier controlled-activation phase, §11).

**Final verdict: `CANONICAL_OPERATIONAL_HEALTH_COMMITTED_AND_CLOSED`.**
