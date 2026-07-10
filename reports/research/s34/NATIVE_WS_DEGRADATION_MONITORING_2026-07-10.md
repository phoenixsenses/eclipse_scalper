# Native WebSocket Degradation Monitoring - 2026-07-10

## Prerequisite: Routed-Endpoint Recovery

This work follows directly from the routed-endpoint incident closure:

- Implementation: `bd7feb326b1d1e5a4aef3441ecab7fc1da8ee869`
- Governance: `a3e921447e945fa5a02780aea5f3fcfc4edfd82d`
- See `reports/research/s34/LIQUIDATION_TRANSPORT_RESTORED_2026-07-10.md` for the
  full incident record (root cause, 72h reconnect chronology, permanent data gap).

## The Original Monitoring Failure

During that ~4-day outage, native WebSocket application-frame delivery and
liquidation ingestion were completely dead while REST-fallback-covered tables
(`agg_trades`, `mark_prices`) stayed fresh. The existing health chain only
evaluated one blended "native OR REST" freshness flag and never inspected
`last_message_ts_utc`, `liquidation_transport_available`, or per-table database
freshness on their own. As a result, `watchdog_overall` (the field
`status_eclipse.ps1` and the operator actually watch) read GREEN for the
entire outage.

## Corrected Semantics

`tools/native_ws_health_policy.py` computes a deterministic RED/DEGRADED/GREEN
verdict from native-only signals, merged into the existing top-level `overall`
(`tools/heartbeat_watchdog.py`) with precedence `RED > YELLOW > GREEN`:

- native `GREEN` adds no severity -- never weakens an existing YELLOW/RED
  caused by something else (bookticker down, runtime not ready, etc.)
- native `DEGRADED` forces `overall` to at least `YELLOW`
- native `RED` forces `overall` to `RED`

Handshake/transport-connected flags are never treated as sufficient evidence
of healthy data flow; the native determination is keyed on actual
application-frame receipt (`last_message_ts_utc`).

## Heartbeat Persistence Design

`data/microstructure_collector.py`'s `_stats_loop` now refreshes the heartbeat
JSON file every `heartbeat_write_interval_sec` (default 15s), decoupled from
the (300s-in-production) `--stats-interval` cadence that still governs the
console stats print and checkpoint calls. This adds no additional console/log
output -- only the JSON heartbeat file write frequency changed. Before this
change, a read landing between two 300s writes could show up to ~300s of
apparent native-message staleness even on a fully healthy connection, which is
why the native-message threshold could not previously be tightened without
false positives.

## Thresholds

| Signal | Warning | Critical | Basis |
|---|---:|---:|---|
| Native WS message age | 60s | 180s | stall timeout (45s) + one heartbeat-write interval (15s); critical = 4x stall timeout |
| Liquidations (global, all-symbol) | 600s | 1800s | two independent bounded rowid-range historical samples (~90min and ~119min, 2,001 rows each) of known-healthy pre-outage activity; observed max gap 77.1s / 75.4s, ~8x/~23x margin applied |
| agg_trades | 60s | 180s | REST-poll cadence (5s) and stall timeout margin |
| mark_prices | 30s | 120s | markPrice@1s guaranteed per-symbol cadence |
| REST-fallback transition grace | 60s | -- | roughly one stall-timeout cycle, so a single brief reconnect is not flagged |

Liquidation freshness is evaluated globally (latest row across all symbols),
matching the collector's `!forceOrder@arr` all-market subscription scope --
deliberately not per-required-symbol, so a single naturally quiet symbol is
never mislabeled as a transport failure.

## Historical Replay

The actual captured 2026-07-06..07-10 outage state (native disconnected,
REST-covered tables fresh, liquidations frozen ~2 days) was replayed through
both the isolated policy function and the full `heartbeat_watchdog.evaluate()`
merge path. Result: `RED` in both, never `GREEN` -- this incident class can no
longer produce a top-level GREEN reading.

## File Adoption Disclosure

Five active runtime/test files were discovered to be previously untracked in
Git despite running this project's live process-management and health
monitoring: `start_eclipse.ps1`, `stop_eclipse.ps1`, `status_eclipse.ps1`,
`tools/heartbeat_watchdog.py`, `tests/test_heartbeat_watchdog.py`. No
independent backup, hash-matched copy, generator, branch, or tag provides a
byte-exact pre-task baseline for any of them. They were adopted in their
current, fully reviewed form in commit `99e9df5537e7edaf2739c604eb1a37b77f6a9d0d`
-- this intentionally includes the monitoring additions already present in
`heartbeat_watchdog.py`, `status_eclipse.ps1`, and its tests, since those
additions could not be safely separated from an unprovable prior baseline. No
historical split is claimed. All five received a full content security and
operational-safety review before adoption (no secrets/credentials/tokens/
host-specific absolute paths; live executors default OFF with active
enforcement; process-kill patterns keyed on specific full module paths).

## Tests

52 passed, 0 failed, independently re-run across two separate audit rounds:
`test_native_ws_health_policy.py` (20), `test_heartbeat_watchdog.py` (12),
`test_microstructure_rest_fallback.py` (9), `test_collector_simulation.py` (1),
`test_collector_checkpoint_interval.py` (1), `test_collector_supervisor_cleanup.py` (1),
`test_collection_watchdog.py` (4), `test_verify_data_layer_status.py` (4).

`tests/test_health_cycle_smoke.py::test_run_smoke_success_short_cycle` fails
independent of this work -- confirmed pre-existing via two separate isolation
methods (a `git stash` comparison, and a from-scratch sandboxed checkout of
the last-committed collector using `git show`, run entirely outside the
production worktree). Not fixed as part of this batch; tracked separately.

## Runtime Activation

- Collector: supervisor-mediated restart, PID `9236` -> `21348`
- Heartbeat watchdog: canonical-mechanism cycle, PID `1916` -> `21312`
- Collector's parent supervisor remained PID `23052` throughout (never restarted)
- No full runtime restart was performed or required
- Native WS remained connected, REST fallback remained false, all source
  tables remained fresh across every check
- No duplicate process introduced at any point
- Both live executors (`s34_v_engine_live_executor`, `s34_state_machine_live_executor`)
  confirmed OFF before, during, and after this work

## Low Findings Carried Forward

- Liquidation threshold evidence is drawn from two bounded samples on a
  single calendar day (2026-07-06); recommend revisiting with broader
  regime/weekend/holiday evidence as more history accumulates.
- Five adopted runtime/test files lack independently-verifiable historical
  byte-exact provenance; current-form adoption was the safe treatment used.
- `test_health_cycle_smoke.py` flakiness is pre-existing and separately
  tracked, not addressed by this work.
