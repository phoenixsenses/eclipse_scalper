# Canonical Operational Health — Independent Acceptance Review, 2026-07-10

Reviewing: `reports/research/s34/CANONICAL_OPERATIONAL_HEALTH_2026-07-10.md`
(claimed verdict `CANONICAL_OPERATIONAL_HEALTH_CORRECTED_AND_VERIFIED`).

No files modified during this review. No process restarted. No live
executor enabled. No DB/checkpoint/runtime state/log/PID file modified
(only read). Nothing committed.

## 1. Review Scope

Full uncommitted diff, the implementation report, all 20 claimed test
files, and current live process/health state. The six pre-existing,
unrelated working-tree modifications named by the requester —
`tools/native_ws_health_policy.py`, `tests/test_native_ws_health_policy.py`,
`tools/collection_watchdog.py`, `tests/test_collection_watchdog.py`,
`tools/s34_cascade_navigation_dashboard.py`,
`tools/s34_realtime_shadow_runner.py` — were excluded from evaluation, as
instructed; they were not touched by this corrective pass (confirmed: no
edit tool call this session or the prior one touched any of them).

## 2. Complete Diff

22 tracked files changed (+2362/-462) attributable to this corrective pass,
plus 2 new untracked test files and 1 new governance report:

| File | Classification |
|---|---|
| `tools/heartbeat_watchdog.py` | canonical overall-writer migration + cadence/configuration |
| `tools/health_state.py` | canonical overall-writer migration |
| `execution/health_gate.py` | component-writer migration |
| `tools/replay_slice.py` | component-writer migration |
| `data/microstructure_collector.py` | component-writer migration (health_root) |
| `tools/check_data_ready.py` | research-fitness bounding |
| `tools/validate_data_research_fitness.py` | research-fitness bounding |
| `tools/validate_microstructure_contract.py` | research-fitness bounding (mode=ro) |
| `src/microphys/io/sqlite_reader.py` | research-fitness bounding (mode=ro) |
| `tools/health_cycle_smoke.py` | test isolation |
| `tests/test_health_cycle_smoke.py` | test isolation |
| `tests/test_replay_slice.py` | test isolation |
| `start_eclipse.ps1` | cadence/configuration |
| `tests/runtime/test_health_gate_unit.py` | tests (writer migration + cadence boundary) |
| `tests/test_heartbeat_watchdog.py` | tests (aggregation redesign) |
| `tests/test_canonical_health_gate_integration.py` | tests (param rename) |
| `tests/test_health_writer_ownership.py` **(new)** | tests (writer ownership) |
| `tests/test_validate_data_research_fitness.py` | tests (bounding proofs) |
| `SYSTEM_STATE.md`, `IMPLEMENTATION_PROGRESS_LEDGER.md` | governance/reporting |
| `reports/research/s34/CANONICAL_OPERATIONAL_HEALTH_2026-07-10.md` **(new)** | governance/reporting |
| `tools/research_fitness_report.py` | governance/reporting (docstring only — pre-existing file, not newly created) |

No hidden or unrelated production changes found beyond the six named
exclusions and `TEST_STATUS_LATEST.md`/`status_eclipse.ps1`/
`stop_eclipse.ps1`/`runtime/dashboard_backend.json` (deleted) — all four of
these were already modified/deleted before this corrective pass began and
were not touched by it either (verified: none reference any health-writer
symbol; see §3).

**Finding (LOW / completeness):** the implementation report's "Deliberately
not touched" list names only 5 of what should be 6 pre-existing files —
`tests/test_collection_watchdog.py` is absent from that list even though it
was equally untouched. Not a functional defect; a documentation omission.

## 3. Final Writer Ownership

Verified beyond direct string matches — traced every generic JSON-write
helper repo-wide (`atomic_write_json` in `tools/health_state.py`,
`execution/runtime_helpers.py`, `src/microphys/live/registry.py`;
`_atomic_write_json` in `tools/research_fitness_report.py`; the local
`atomic_write` in `tools/heartbeat_watchdog.py`) and every module that
imports `write_component_health`/`write_overall_health`, plus all `.ps1`
scripts and the dashboard frontend/backend.

1. **`heartbeat_watchdog.py` is the only production writer of
   `overall.json`** — confirmed. Its own `atomic_write()` is the only call
   site left targeting that filename anywhere in the tracked repo.
2. **Partially confirmed — one real gap found.** `tools/health_state.py`'s
   `write_component_health` now rejects `component="overall"` (verified,
   re-tested). However, `tools/research_fitness_report.py` defines its
   **own separate** `_atomic_write_json(path, payload)` (not imported from
   `tools/health_state.py`) and its CLI `--out` argument accepts an
   **unvalidated arbitrary path** — nothing stops `--out
   logs/health/overall.json` (or `watchdog.json`, or
   `reports/WATCHDOG_STATUS.json`) from silently writing a
   research-fitness-shaped payload directly over the canonical file.
   `tools/collection_watchdog.py::main()` inherits the identical exposure
   (it imports and calls the same `_atomic_write_json`, with its own
   equally-unvalidated `--out`). **No current invocation (start_eclipse.ps1,
   any test, any doc) ever passes a non-default `--out`** — this is a
   latent hardening gap, not an active defect — but it is exactly the class
   of "generic helper that can silently restore multi-writer behavior" the
   review asked to rule out, and it is not ruled out. **Finding: MEDIUM.**
3. **Confirmed.** `execution/health_gate.py::write_paper_trader_health`
   writes only `write_component_health("paper_trader", ...)`; no
   `overall.json` reference remains in its source.
4. **Confirmed.** `tools/replay_slice.py::_write_replay_health` writes only
   `write_component_health("replay", comp, root=health_root)`; no
   `overall.json` write remains.
5. **Confirmed.** No read-merge-write against `overall.json` exists
   anywhere (traced `load_overall_health`'s only callers: `execution/`
   modules that only ever *read* it for gating decisions, never write back).
6. **Confirmed.** Both `tools/heartbeat_watchdog.py::atomic_write` and
   `tools/health_state.py::atomic_write_json` use temp-file-then-`os.replace`
   / `Path.replace`; independently re-verified no leftover `.tmp_*` files
   after two fresh test runs and one fresh production run this review.
7. **Confirmed.** With #2 as the sole caveat, no lost-update race remains
   for any code path exercised by the actual running system.

Also found, informational: `execution/guardian.py` and
`monitoring/prometheus.py` each independently write
`logs/health/heartbeat.json` — a **different filename**, not a collision,
via their own `atomic_write_json` (from `execution/runtime_helpers.py`, a
third, unrelated generic-writer family). Both modules are dormant (no
live-executor process is running); this file is not read or aggregated by
`heartbeat_watchdog.py` and is not mentioned in the implementation report's
ownership map. Not a defect — correctly not fabricated/aggregated — but an
incomplete inventory item worth naming explicitly.

## 4. Component Aggregation Review

| Component | Source | Writer | Required | Contributes to state | Missing/corrupt |
|---|---|---|---|---|---|
| `collector` | `logs/health/collector.json`, read every cycle | `data/microstructure_collector.py` | yes | yes | `read_json` → `{}` → treated unhealthy |
| `bookticker` | `logs/health/bookticker.json` | `data/bookticker_collector.py` | conditional | yes when expected | same |
| `watchdog` | self-authored in-memory | `heartbeat_watchdog.py` | yes | descriptive only | n/a |
| `paper_trader` | `logs/health/paper_trader.json` | `execution/health_gate.py` | no | no (by design, see report §4) | omitted, not fabricated |
| `replay` | `logs/health/replay.json` | `tools/replay_slice.py` | no | no | omitted, not fabricated |

Confirmed live (read-only): `heartbeat_watchdog.py` rebuilds
`components` from `_read_optional_components()` reading each dedicated file
fresh every cycle — no code path reads a previous `overall.json` at all
(the function that used to do this, `_read_existing_overall`, no longer
exists in the source). The 80-day-old `paper_trader` placeholder is
confirmed **still present on disk** (`reports/research/s34/
CANONICAL_OPERATIONAL_HEALTH_2026-07-10.md`'s claim independently
re-verified: `last_progress_ts_utc: 2026-04-21T17:38:08`, `reason:
alpha_metrics_missing`) and correctly appears in the live `overall.json`
with its own true, ancient timestamp rather than a refreshed one — proving
the "surfaced-but-not-refreshed" design in a real, non-synthetic instance.

## 5. Replay Isolation Review

`tools/replay_slice.py` previously wrote `overall.json` because its
original author copied the same (now-recognized-as-wrong) pattern used by
`execution/health_gate.py` — a component write immediately followed by a
hand-assembled `overall` dict and a second write. Both call sites are gone;
`_write_replay_health` now performs exactly one write.

- Replay cannot overwrite the live canonical verdict: confirmed (§3.4).
- Replay cannot fabricate a fresh live component: confirmed — it can only
  write its own `replay.json`, which `heartbeat_watchdog.py` folds in
  read-only, verbatim, with no severity contribution.
- Replay output is isolated from production state in the *test* path
  (`--health-root`/`health_root` parameter, defaults to real
  `logs/health` for CLI parity — same pattern as
  `data/microstructure_collector.py --health-root`).
- `tests/test_replay_slice.py` passes an isolated temp root and asserts the
  real `logs/health/replay.json` file's existence/mtime is unchanged by the
  test — independently re-run, passes.
- No compatibility consumer expects `replay_slice.py` to mutate
  `overall.json` (grep confirms zero remaining readers of a `replay`-driven
  `overall.json` write; the only consumer of the `replay` component is
  `heartbeat_watchdog.py`'s read-only aggregation).
- Confirmed live: the transient `replay.json` artifact this corrective
  session's own pre-fix test run had left in the real `logs/health/` was
  removed, and the `replay` component correctly disappeared from the live
  `overall.json` on the next cycle once its source file was gone (observed
  directly, not merely asserted).

## 6. Health-Gate Review

`execution/health_gate.py::write_paper_trader_health` writes only
`paper_trader.json` (confirmed, §3.3). `evaluate_health_gate()` itself is
**byte-for-byte unmodified** in this pass — its logic, thresholds
(`max_health_staleness_sec=15`, `max_collector_lag_sec=30`,
`max_reconnects_5m=10`, etc.), and `GateDecision`/`GateState` shapes are
untouched. Live-safety gating still reads `logs/health/overall.json`
exclusively via `load_overall_health()`. Canonical state mapping
(`GREEN→ok`, `YELLOW→degraded`, `RED→halted`) is unchanged
(`CANONICAL_OVERALL_STATE_MAP` in `heartbeat_watchdog.py`, identical to
before this pass). `degraded`/`halted` still block (`evaluate_health_gate`
returns `allow=False` for both, unchanged code path). Component freshness
(`paper_trader.ts_utc`) is never confused with top-level freshness
(`evaluate_health_gate` only ever reads `health_obj["ts_utc"]` for its
staleness check, never a component's own timestamp). No sizing, leverage,
order, or execution-behavior code was touched anywhere in this diff — the
only change inside `execution/` is the deletion of 15 lines that
constructed and wrote an `overall` dict; nothing upstream of that (the
`GateDecision` computation itself) changed.

## 7. Research-Fitness Semantics

Verified by direct code read (not the report's prose):

1. **Exact required tables:** `mark_prices`, `agg_trades`, `liquidations`
   (`RESEARCH_FITNESS_TABLE_ALLOWLIST` in `tools/check_data_ready.py`).
2. **Exact symbols:** whatever the caller passes (`BTCUSDT,ETHUSDT` by
   default) — unchanged from before this pass.
3. **Exact bounded window/row limit:** `_RECENT_ACTIVITY_WINDOW_SEC = 600`
   (10 minutes) and `_FEATURE_ROW_LIMIT_PER_TABLE = 2000`, both in
   `tools/validate_data_research_fitness.py`.
4. **Readiness determination:** unchanged logic — `check_db_fresh` (now
   allowlist-scoped) plus per-symbol row counts/feature computability
   within the bounded window.
5. **Data no longer inspected:** every table outside the 3-table allowlist
   (e.g. `detector_heartbeat`) for the table-scan path; and, for the two
   count/feature queries, any row older than 600s from evaluation time.
6. **Why irrelevant:** research-fitness asks "is there usable data *right
   now*", a recent-activity question by definition; tables outside the
   allowlist are unrelated operational tables never referenced by the
   readiness/sample/feature logic in the first place (confirmed by reading
   every remaining query in the file — none references any 4th table).
7. **Insufficient coverage:** `low_trade_rows:{symbol}` warning if
   `agg_trade_rows < min_trade_rows_per_symbol` (default 10) *within the
   600s window* — this is a real behavior change worth naming explicitly:
   a symbol with healthy historical volume but genuinely zero trades in the
   last 10 minutes (e.g., temporarily illiquid altcoin) will now warn where
   the old all-time count might not have. This is the correct tradeoff for
   a *readiness* check but is a semantic narrowing, not merely a speed
   optimization — the report does state this design choice explicitly
   (§5, "this is the semantically correct behavior too"), so it is
   disclosed, not hidden.
8. **Missing required tables:** `db_not_ready` failure → `status="fail"` →
   mapped to `"blocked"` — deterministic (re-verified via
   `test_missing_required_table_is_deterministically_blocked`).
9. **Empty windows:** `agg_trade_rows`/etc. simply come back `0` — folds
   into the existing `low_trade_rows`/`no_feature_rows` warning/failure
   paths, no special-cased crash.
10. **Deterministic disposition:** `_STATUS_MAP`/`_EXIT_CODE_MAP` unchanged,
    fixed dict lookups.

Every production-path SQLite connection in this exact chain
(`validate_data_research_fitness.py`, `validate_microstructure_contract.py`,
`src/microphys/io/sqlite_reader.py::discover_mappings`) opens
`f"file:{path}?mode=ro", uri=True` — confirmed by direct read of all three
call sites. Grepped the full call chain for `PRAGMA|CREATE TABLE|CREATE
INDEX|ANALYZE|BEGIN|COMMIT|INSERT|UPDATE|DELETE|CHECKPOINT`: zero real SQL
statements match (only comments/identifiers/function names) — no write-
capable statement exists in this path regardless of the `mode=ro` guarantee.

## 8. Production Performance Reproduction (Independent)

Not accepted on the claimed number alone — reran independently:

```
prev_checksum (from implementation session) = 76f97e8b0e83b30c9f13d6a3dc594fb996e6f95ef2315ddf6f97c1f1b48e4ec0
=== START 2026-07-10T16:29:55.056Z ===
research_fitness status=ready raw_status=pass contract_tier=full_book db_ready=True warnings=0 failures=0
real  0m0.510s
=== END 2026-07-10T16:29:55.600Z EXIT=0 ===
new_checksum = 6c09ae57a078fc787ed8dc3c2b119bff907b5f9e8565407c0863420dfaaaf5eb
```

- DB size at time of this run: 792,963,543,040 bytes (~738.6 GiB);
  792,965,529,600 bytes ~7s later (grew ~1.9MB — attributable entirely to
  the independently-running live collector, not this read-only run).
- Elapsed: **0.510s** (faster than the implementation session's own
  1.107s — both comfortably bounded; no contradiction).
- Exit code 0.
- Disposition: `ready`/`pass`, `db_ready=True`, 0 warnings, 0 failures.
- Required tables inspected: `mark_prices`, `agg_trades`, `liquidations`
  only (per code, §7).
- Report checksum differs from the prior run's (expected — new
  `evaluated_at_utc`), output path `logs/health/research_fitness.json`,
  written atomically (no leftover `.tmp_research_fitness_*`).
- `overall.json`/`WATCHDOG_STATUS.json` severity confirmed unchanged
  (`ok`/`GREEN` before and after), `research_fitness` key confirmed absent
  from `overall.json.components` both before and after.
- DB mutation: structurally impossible (`mode=ro` connection; independently
  confirmed no write-capable statement exists in the call path, §7).

**Result: claim independently reproduced and exceeded (faster). No
corrective action required for this section.**

## 9. Test-Isolation Review (Independent)

Re-read `tools/health_cycle_smoke.py` line-by-line against the report's
claims — accurate. Ran the corrected smoke test **twice** independently:

```
run 1: 3 passed in 9.61s
run 2: 3 passed in 9.61s
```

Real `logs/health/overall.json.state` observed `"ok"` immediately before
and after both runs, with its `ts_utc` continuing to advance normally
(proving the real background watchdog was undisturbed, not paused or
corrupted, by either test run).

Isolation mechanics confirmed: `--root` scopes `logs/health/`, `reports/`,
and `logs/collector_heartbeat.json`; `--db-path`/`--seed-market-data`
scope the database; the spawned `data.microstructure_collector` subprocess
receives explicit `--health-root`/`--heartbeat-path` flags pointing inside
`--root`; canonical `overall.json` is produced by one in-process
`heartbeat_watchdog` evaluation per poll tick (module-global save/restore
via try/finally) — no second Python process, no dependency on the real
watchdog. One gap found: the spawned subprocess's environment is **not**
explicitly cleared/isolated (`subprocess.Popen` with no `env=` inherits the
full parent environment) — reviewed `data/microstructure_collector.py` for
env-var reads: only `COLLECTOR_RECONNECT_ALERT_THRESHOLD_5M` (an alert-print
threshold, irrelevant to this test's actual assertions). **Finding: LOW /
informational** — technically not "fully isolated" as literally claimed,
but no identified path by which this affects test correctness today.

False-invariant defect, verified against current code
(`tools/health_cycle_smoke.py::_validate_snapshot`):

- **Exact old invariant:** during the `"degraded"` phase, the test required
  both `state=="degraded"` *and* `components.collector.status=="degraded"`.
- **Why false:** `data/microstructure_collector.py::_write_heartbeat`
  computes `collector.json`'s `status` from a staleness threshold
  (`stall_timeout_sec`, default 45s) — a brief 2–6s simulated disconnect
  can never make it read `"degraded"`; it correctly stays `"ok"` while
  `connected` correctly flips to `False`.
- **Exact corrected behavior:** the check now requires only
  `components.collector.connected is False`; the top-level `state` check
  (still `"degraded"`) is what actually proves the outage was detected — via
  the faster `native_ws_policy` connection-level signal, which is the
  intended fast-detection layer.
- **Test expectation changed, not production code** — `_write_heartbeat`'s
  staleness-gated status semantics are untouched and correct as designed.
- **Why the correction is safe:** it makes the assertion match the actual,
  intended, two-speed detection architecture (fast connection-level +
  slower staleness-level) instead of an invariant no implementation of that
  architecture could ever satisfy.
- **Regression coverage:** the corrected assertion is exercised by both
  independent reruns above; no dedicated unit test isolates
  `_validate_snapshot` itself outside the full `run_smoke` integration path
  (LOW: a direct unit test would make the invariant change harder to
  silently regress twice).

## 10. Cadence / Staleness Evidence (Independent)

Passive, read-only, 240-second sample against the live, unmodified running
watchdog (PID 22816, `--interval-sec 5`) — no process touched, no file
written except the sampler's own scratch output:

- **n = 41 distinct `ts_utc` writes observed, 40 inter-write deltas.**
- min = 4.82s, **p50 = 5.85s**, **p90 = 6.04s**, **max = 6.23s**.
- Worst-case margin against the 15s consumer budget: **8.77s**.

This independently confirms and slightly exceeds the report's own
re-measurement (~6s/cycle, ~9s margin) with a much larger sample (40 deltas
vs. the report's 6). `execution/health_gate.py::evaluate_health_gate`'s
staleness check (`(now - ts) > max_health_staleness_sec`) is unmodified and
was independently re-tested at exactly 14.9s/15.0s/15.1s/22.0s (all four
`tests/runtime/test_health_gate_unit.py` boundary tests pass, confirming
deterministic block-not-accept behavior beyond the budget).

CPU/resource check on the live watchdog process: ~3.27s cumulative CPU
after ~18 minutes of uptime (~0.3% average), ~17MB working set, stdout log
growing at a normal, one-line-per-cycle rate, empty stderr log. **No
excessive CPU, DB polling, or log spam from the 5s cadence.**

## 11. Independent Test Results

Reran all 13 batches (≤2 files/call) independently:

| Batch | Files | Result (this review) | Claimed |
|---|---|---|---|
| 1 | test_health_writer_ownership.py + test_heartbeat_watchdog.py + test_canonical_health_gate_integration.py | **34 passed, 1 failed** | 35 passed |
| 2 | runtime/test_health_gate_unit.py | 26 passed | 26 passed |
| 3 | test_validate_data_research_fitness.py + test_research_fitness_report.py | 14 passed | 14 passed |
| 4 | test_validate_microstructure_contract.py + test_microstructure_sample_fixture.py | 7 passed | 7 passed |
| 5 | test_collector_simulation.py + test_collector_checkpoint_interval.py | 2 passed | 2 passed |
| 6 | test_health_cycle_smoke.py | 3 passed (×2 independent runs) | 3 passed |
| 7 | test_native_ws_health_policy.py + test_collector_supervisor_cleanup.py | 22 passed | 22 passed |
| 8 | test_entry_loop_gate_integration.py + test_dashboard_overview_api.py | 5 passed | 5 passed |
| 9 | test_ami_host_health_observation.py + test_ami_host_health_evaluator.py | 98 passed | 98 passed |
| 10 | test_s34_live_chart_host_health.py + test_health_check.py | 11 passed | 11 passed |
| 11 | test_health_check_stale.py + test_collection_health.py | 6 passed | 6 passed |
| 12 | test_collection_watchdog.py + test_status_snapshot.py | 9 passed | 9 passed |
| 13 | test_push_status.py + test_replay_slice.py | 3 passed | 3 passed |

**Total: 241 collected (matches claim), 240 passed, 1 failed (does not
match the claimed 241 passed / 0 failed).**

Failing test:
`tests/test_health_writer_ownership.py::test_known_former_writers_no_longer_reference_overall_json`

```
AssertionError: overall.json may only appear in an ownership-explaining comment
assert "owned solely" in rs_source
```

**Root cause (traced precisely):** `tools/replay_slice.py`'s
`_write_replay_health` docstring was edited a *second* time, later in the
implementation session (adding `--health-root` support, during the
"controlled activation" phase, after the section-F full-suite run that
produced the reported 241/241/0). That second edit's line-wrapping split
the literal phrase across a line break — `"...is owned\n    solely by
tools/heartbeat_watchdog.py"` — so the single-space substring `"owned
solely"` the test requires no longer appears contiguously, even though the
underlying property being tested (no write to `overall.json`) remains true.
**The full test suite was never re-run after that later edit**, so the
report's 241/241/0 was accurate *at the time it was measured* but is not
accurate against the final, current code state.

This is a test-fragility bug (brittle exact-substring matching across a
docstring line wrap), not a functional or safety regression — the actual
invariant (`replay_slice.py` doesn't write `overall.json`) still holds, as
independently proven in §5. But the claimed count is currently false.

**Finding: MEDIUM** (a specific, reproducible, currently-true discrepancy
against an explicit, central claim in the acceptance criteria — "241
passed / 0 failed" is not what the repository's current state produces).

Coverage confirmed present across the required list: unique overall writer
(§3), component-file ownership (§4), replay isolation (§5), paper-trader
migration (§6), canonical health-gate behavior (§6),
stale/missing/corrupt components (§4, `test_corrupt_optional_component_file_is_omitted_not_fatal`,
`test_canonical_overall_omits_paper_trader_when_dedicated_file_absent`),
native GREEN/YELLOW/RED mapping (`test_two_output_files_cannot_disagree_in_severity`
et al.), research-fitness bounded queries + read-only DB access + unrelated
large table ignored (§7–8), health-cycle isolation (§9), cadence boundaries
(§10), live executors default OFF (§12).

## 12. Live State

Re-verified read-only, this review, after all reruns above:

- Exactly one `heartbeat_watchdog` — **PID 22816, unchanged** from the
  implementation session's activation (`started_at` in
  `logs/pids/heartbeat_watchdog.json` matches: `2026-07-10T16:11:37Z`).
- Exactly one `microstructure_collector` (PID 3828, unchanged).
- `collector_supervisor.py` (PID 23052, unchanged) present as parent.
- All other previously-enumerated processes present, unchanged PIDs:
  `bookticker_collector`, `oi_spot_poller`, `s34_shadow_paper_runner`,
  `s34_live_chart`, `s34_v_engine_v02_shadow_mirror`, `event_diary`,
  `s34_realtime_shadow_runner`, `orderflow_chart`, `s34_replay`.
- Zero `collection_watchdog` process.
- `overall.json.state="ok"` ⇔ `WATCHDOG_STATUS.json.overall="GREEN"` —
  agree.
- `overall.json` age at observation: **6.09s** (well within the 15s
  budget).
- `native_ws_status="GREEN"`, `rest_fallback.active=False`.
- `source_freshness`: `agg_trades` 1.00s, `mark_prices` 1.64s,
  `liquidations` 1.09s.
- Live executors OFF (confirmed absent from the process list, as before).
- No duplicate process for any role.
- No unrelated process restarted (all PIDs from the implementation
  session's own live-state check remain identical except the one
  documented `heartbeat_watchdog` cycle).
- Writer confirmed by direct observation, not just code reading: the
  `paper_trader` component (from a file no live process is currently
  writing) still shows its true ~80-day-old timestamp; the transient
  `replay` component from this review's own earlier reruns of
  `test_replay_slice.py` did **not** reappear in `overall.json` (correctly
  isolated — this review's reruns wrote only to their own temp roots).

## 13. Governance Report Review

`reports/research/s34/CANONICAL_OPERATIONAL_HEALTH_2026-07-10.md` was
checked claim-by-claim against the independent findings above. Accurate on:
undisclosed writers, the stale `paper_trader` placeholder, the
research-fitness root cause (correctly attributes the deeper cause beyond
the original `detector_heartbeat` hypothesis), the mode=ro corrections, the
smoke-test isolation defect and its false-invariant explanation, the
cadence measurement, the PID transition, both live executors OFF, and "no
commit yet" (independently confirmed — `git status` unchanged in kind from
the implementation session, nothing staged/committed).

**Not accurate:** §10 "Exact Test Results" states "241 collected, 241
passed, 0 failed" — independently reproduced as 241 collected, **240
passed, 1 failed** (§11 above). This is the report's one material
inaccuracy found by this review.

**Completeness gap (LOW):** §9 "Files Changed → Deliberately not touched"
lists 5 of the 6 pre-existing unrelated files (missing
`tests/test_collection_watchdog.py`).

## 14. Worktree Isolation

All isolation mechanisms reviewed (`tools/health_cycle_smoke.py --root`,
`data/microstructure_collector.py --health-root`,
`tools/replay_slice.py --health-root`) accept arbitrary paths with no
dependency on the invoking process's cwd beyond their unchanged,
real-path-preserving defaults — equally usable from a disposable worktree.
No file touched by this review or its rerun tests fell outside each
respective tool's own isolated temp root, confirmed by the filename-set and
mtime checks in §5/§9's independent reruns.

## 15. Findings

| # | Severity | Finding |
|---|---|---|
| 1 | **MEDIUM** | `tools/research_fitness_report.py` (and `tools/collection_watchdog.py`, which reuses it) exposes an unvalidated `--out` CLI parameter through its own local `_atomic_write_json`, bypassing `write_component_health`'s `component="overall"` guard. Nothing currently invokes it with a path matching `overall.json`/`watchdog.json`/`WATCHDOG_STATUS.json`, but no code-level guard prevents it — the single-writer invariant is not structurally enforced against this path. |
| 2 | **MEDIUM** | The reported "241 collected, 241 passed, 0 failed" is not reproducible against the current, final code state: `tests/test_health_writer_ownership.py::test_known_former_writers_no_longer_reference_overall_json` fails deterministically (240 passed, 1 failed), due to a docstring line-wrap introduced by a later edit (`tools/replay_slice.py`, made during "controlled activation") that was never followed by a final full-suite re-run. The underlying property being tested is still true; only the test's fragile exact-substring assertion is broken. |
| 3 | LOW | Governance report's "six unrelated pre-existing files" list is missing `tests/test_collection_watchdog.py` (lists 5, not 6). |
| 4 | LOW / informational | `execution/guardian.py` and `monitoring/prometheus.py` independently write `logs/health/heartbeat.json` (different filename, no collision) via a third, unrelated generic-writer family (`execution/runtime_helpers.py::atomic_write_json`). Both modules are dormant. Not accounted for in the report's component ownership map. |
| 5 | LOW | `tools/health_cycle_smoke.py`'s spawned collector subprocess inherits the full parent environment rather than an explicitly isolated one; the one env var the code path reads (`COLLECTOR_RECONNECT_ALERT_THRESHOLD_5M`) is irrelevant to the test's assertions, so no correctness impact identified today. |
| 6 | LOW | The corrected `_validate_snapshot` false-invariant fix has no dedicated unit test isolated from the full `run_smoke` integration path — a second silent regression of this specific assertion would only be caught by the (slower) end-to-end run. |
| 7 | informational | `tools/health_cycle_smoke.py --root` defaults to `"."` (real repo paths) for CLI/ops parity; confirmed no automatic/scheduled invocation anywhere in the repo uses this default — purely a manually-invoked tool today, so this is not a live risk, only a latent one if that ever changes. |

## 16. Acceptance Verdict

Two MEDIUM findings (an unenforced single-writer guard on a CLI-configurable
output path, and a currently-failing test that falsifies the report's
central "0 failed" claim). Both are narrow, well-understood, and cheaply
fixable — this is not evidence of a broken architecture; every core claim
about the writer redesign, the research-fitness bounding, the test
isolation, the cadence measurement, and the live activation was
independently reproduced and confirmed correct. But per the review's own
rule, a MEDIUM finding blocks acceptance, and the test-count discrepancy in
particular means the report's own acceptance criterion is not currently
met by the repository.

`CANONICAL_OPERATIONAL_HEALTH_CORRECTIVE_CHANGES_REQUIRED`

Corrective work needed before re-review: (a) fix the `--out` validation gap
in `tools/research_fitness_report.py`/`tools/collection_watchdog.py`
(mirror `write_component_health`'s guard); (b) fix
`test_known_former_writers_no_longer_reference_overall_json`'s fragile
assertion (normalize whitespace or check for the ownership comment
differently) and re-run the full 20-file/241-test suite to completion
without a subsequent uninspected edit; (c) optionally, complete the "six
unrelated files" list and add a direct unit test for
`_validate_snapshot`'s corrected invariant (LOW, non-blocking).

---

## FOLLOW-UP: Final Corrective Round Resolution (same day)

Both MEDIUM findings were corrected in the final corrective round (see the
implementation report's updated §2/§9/§10 for full detail):

- **Finding 1 (MEDIUM, `--out` gap): RESOLVED.** The guard was placed
  inside `tools/research_fitness_report.py::_atomic_write_json` itself
  (resolved-basename, case-insensitive rejection of all 7 protected
  operational filenames, raising `ProtectedOperationalOutputError` before
  any mkdir/temp-file), the production CLI `--out` was removed outright,
  and — because `tools/collection_watchdog.py` imports that exact writer —
  the deprecated wrapper inherits the guard with zero edits to that
  pre-existing file. Verified live: CLI attempt exits 2 (argparse), the
  internal API and wrapper attempts raise deterministically with zero
  filesystem writes, and canonical outputs remained `ok`/`GREEN`
  throughout. 6 new tests cover default path, CLI rejection, all 7
  protected names, relative aliases, case aliases, and the wrapper.
- **Finding 2 (MEDIUM, stale 241/241 claim + fragile test): RESOLVED.**
  The substring assertion was replaced with ast-based structural analysis
  (comments and docstrings are invisible; `execution/health_gate.py`'s
  read-only `load_overall_health` default is the single permitted
  `overall.json` literal; a dedicated regression test proves docstring
  prose/line-wrapping cannot affect the verdict, plus a positive-control
  test that `tools/heartbeat_watchdog.py` does own the path literal). The
  full 20-file suite was then re-run after the final code edit:
  **250 collected, 250 passed, 0 failed** — the original 241 reviewed node
  IDs are a passing strict subset; the +9 are the new guard/structural
  tests. The implementation report's §10 now records this run and
  explicitly supersedes the stale 241 claim.
- **Finding 3 (LOW, incomplete unrelated-files list): RESOLVED** — list
  re-derived from `git status` in the implementation report §9.
- **Finding 4 (LOW, `logs/health/heartbeat.json` writers unmapped):
  RESOLVED (documentation-only)** — added to the ownership map §2 with the
  explicit classification: separate filename, no `overall.json` ownership
  violation, dormant live-executor modules, review-again-if-executors-
  activate.
- **Finding 5 (LOW, smoke-test subprocess env not sanitized): CARRIED
  FORWARD, deliberately unfixed.** Full environment scrubbing for a
  Windows Python subprocess requires reconstructing a working minimal env
  (`SystemRoot`, `ComSpec`, `PATH`, etc.) — a real breakage risk for zero
  identified correctness gain, since the only environment variable the
  spawned collector reads (`COLLECTOR_RECONNECT_ALERT_THRESHOLD_5M`) is a
  print-only alert threshold with no influence on any assertion. All
  *path* isolation is explicit via flags. Remains LOW.
- **Finding 6 (LOW, no direct `_validate_snapshot` unit test): CARRIED
  FORWARD** — the corrected invariant is exercised end-to-end by the smoke
  test on every run; a dedicated unit test remains a nice-to-have.
- **Finding 7 (informational, `--root` default `"."`)**: unchanged,
  still manually-invoked-only.

---

## FINAL ACCEPTANCE (same day, second follow-up)

A final independent acceptance pass re-audited the resolved state from
scratch: full repository-wide AST single-writer scan (1,012 non-test
modules; zero violations), live rejection probes beyond the original
required set (a real symlink alias and an NTFS 8.3 short-name alias of the
actual `overall.json`, both rejected with zero writes), the original-241
subset demonstrated by explicit `--deselect` of the 9 new node IDs (12
passed + 229 from the other 18 files = 241), the expanded 250-test set
independently rerun fresh (250 passed, 0 failed, 0 skipped), and a fresh
production research-fitness reproduction (0.500s, exit 0). No HIGH or
MEDIUM finding. Two LOW findings deliberately carried forward, unchanged.

**Verdict: `CANONICAL_OPERATIONAL_HEALTH_ACCEPTED_WITH_LOW_FINDINGS`.**

Committed as four content commits (`00ef49ad`, `81ec6d71`, `f3d95f5e`,
`6faa2177`) plus one governance commit, on `codex/data-layer-fallback-cleanup`
(not pushed) — see the implementation report's §17 for the full table and
per-commit scope verification. Final state:
**`CANONICAL_OPERATIONAL_HEALTH_COMMITTED_AND_CLOSED`.**
