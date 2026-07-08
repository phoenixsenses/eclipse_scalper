# OPERATOR_HOST_HEALTH_RESTART_READINESS_DASHBOARD_V1

**Gate:** BATCH-OPERATOR-HOST-HEALTH-AND-RESTART-READINESS-DASHBOARD-V1
**Date:** 2026-07-08 · **Author:** Sonnet 5
**Nature:** monitoring/dashboard only — read-only host observation + a pure deterministic
restart-readiness classifier. No restart, shutdown, suspend, collector stop/restart, process
kill, registry write, Windows Update change, pagefile change, SMART change, or package install
occurs anywhere in this batch's code, at any point.

---

## 1. Purpose

Answer, deterministically and explainably, on the existing operator dashboard:
*is a controlled Windows restart of this host currently unnecessary, advisable, or urgent — and why.*

## 2. Architecture audit (Phase 1) — integration point chosen

The repository has **two** independently running dashboards:

1. `tools/s34_live_chart.py` (port 5050) — a raw `http.server` chart with a single `/api/data`
   JSON endpoint and an already-computed-but-unrendered `disk_status()` payload field.
2. `dashboard/` — a FastAPI (`dashboard/backend/app.py`, port 8765) + React/Vite
   (`dashboard/frontend/`, port 5173) operator console, with an existing `/api/ops/health`
   endpoint (`read_ops_health()` in `dashboard/backend/data_sources.py`) and matching
   `Ops Health: *` cards in `dashboard/frontend/src/pages/ControlTower.tsx`, using an
   `ok`/`warning`/`critical` status convention and `badge-green|yellow|red|gray` CSS classes.

**Chosen integration point:** the FastAPI/React console (#2) — it is the actively developed
operator surface, already has a near-identical health-card pattern to imitate exactly, and
already exposes a typed Pydantic response + polling hook convention
(`usePoll`, `AsyncState`) that a new card can reuse verbatim. No second dashboard was created.

State-machine style: modeled on `ami/governance/storage_rotation_retention_readiness_v1.py`'s
`storage_health_state()` — a pure, frozenset-enum, boundary-tested classifier function that
fails toward the more severe state on disagreement. That same module's `storage_health_state()`
function is reused directly (not reimplemented) for the D:-drive storage-state input.

## 3. What was built

| Component | Path |
|---|---|
| Immutable observation model + fail-closed collector | `ami/host_health/observation.py` |
| Pure deterministic restart-readiness evaluator | `ami/host_health/evaluator.py` |
| Read-only CLI (`python -m tools.host_health_status`) | `tools/host_health_status.py` |
| Backend integration (`GET /api/host/health`, 20s cache) | `dashboard/backend/data_sources.py::read_host_health()`, `dashboard/backend/models.py::HostHealthResponse`, `dashboard/backend/app.py` |
| Frontend "PC / Host Health" card | `dashboard/frontend/src/pages/ControlTower.tsx`, types in `dashboard/frontend/src/api/types.ts`, fetcher in `dashboard/frontend/src/api/client.ts` |
| Focused tests (98 total) | `tests/test_ami_host_health_evaluator.py` (62), `tests/test_ami_host_health_observation.py` (36) |

## 4. Observation sources (Phase 2/3)

* **psutil** (already a repo dependency, `requirements.txt`) — `boot_time()`, `virtual_memory()`,
  `swap_memory()` (pagefile), `cpu_percent()`, `disk_usage("C:\\")`, `disk_usage("D:\\")`. No
  subprocess, near-zero cost.
* **One consolidated, read-only PowerShell subprocess call per observation** (`_POWERSHELL_SCRIPT`
  in `observation.py`), invoked via `-EncodedCommand` (UTF-16LE base64) with a 25s timeout,
  UTF-8 console output forced (`[Console]::OutputEncoding`). Covers: `Win32_OperatingSystem`
  commit-limit/committed-bytes, three pending-reboot registry-evidence checks (`Test-Path` /
  `Get-ItemProperty`, never a write), `Get-PhysicalDisk` + `Get-StorageReliabilityCounter`
  (SSD temperature/SMART, best-effort — `Get-StorageReliabilityCounter` requires elevation on
  this host and fails closed to `UNKNOWN` when unavailable, confirmed live), and five bounded
  `Get-WinEvent -FilterHashtable` queries (`-MaxEvents 500`, 168h window) for unexpected
  shutdown (System 41/6008), WHEA (`Microsoft-Windows-WHEA-Logger`), disk/NTFS provider errors,
  application crashes (`Application Error`/1000), and resource-exhaustion events (2004/2005).
  **Every cmdlet in the script is `Get-`/`Test-`/`ConvertTo-`/`Write-`-shaped** — proven
  structurally by a dedicated test that strips string literals and asserts every remaining
  cmdlet-shaped token starts with an allowed read-only verb (`tests/test_ami_host_health_observation.py::test_powershell_script_only_uses_readonly_cmdlets`).
* **Repository-local read-only file reads** (no subprocess): `logs/health/overall.json` +
  `logs/collector_heartbeat.json` for collector status (reusing the same HEALTHY/DEGRADED/FAILED
  vocabulary and staleness convention as `tools/health_check.py`), `data/microstructure.db` and
  `data/microstructure.db-wal` `os.stat()` size.
* **Critical-operation heuristic:** best-effort — scans live `python.exe` command lines (already
  collected in the same PowerShell call) for keywords (`purge`, `vacuum`, `archive_export`,
  `rehearsal`, `production_activation`, `rotation_retention_apply`, `migration`). This is
  explicitly a heuristic, not a canonical "active batch" registry (none exists in the repo today)
  — disclosed as a residual risk in §8. It only ever adds an informational
  `RESTART_RECOMMENDED_BUT_DEFER_UNTIL_SAFE_CHECKPOINT` caution; it never blocks or changes the
  underlying GREEN/YELLOW/RED state.

Every field that could not be determined is `None`/`"UNKNOWN"` and listed in
`unknown_fields` — nothing is fabricated. Confirmed live on this host: `Get-StorageReliabilityCounter`
denies access without elevation, so `ssd_temp_c` correctly comes back `None` with
`SSD_SENSOR_UNKNOWN` in the reason codes, and the dashboard does **not** turn red because of it.

## 5. Restart-readiness state machine (Phase 8/9)

Exact states: `HOST_RESTART_GREEN`, `HOST_RESTART_YELLOW`, `HOST_RESTART_RED`,
`HOST_RESTART_UNKNOWN` (`ami/host_health/evaluator.py::HOST_RESTART_STATES`).

**Fail-closed UNKNOWN gate** (checked first, wins over every other rule): boot time
undeterminable, pending-reboot evidence contradictory, memory observation unavailable,
collector health unevaluable, or the observation is materially stale (>120s old at evaluation
time). Confirmed by `test_unknown_gate_wins_over_red_tier_conditions` — even with WHEA errors
*and* `STORAGE_EMERGENCY` simultaneously present, an UNKNOWN-gate trigger still wins.

**Uptime never causes RED by itself** — `UPTIME_HIGH` (≥14 days) alone classifies YELLOW; RED
requires it combined with a *confirmed* pending reboot. Proven by
`test_uptime_alone_never_red_even_at_extreme_values` (365 days, still not RED).

**Sustained vs. instantaneous pressure:** RAM/commit critical-tier readings only reach RED when
a caller-supplied sustained-window value confirms it; a lone instantaneous spike lands on
YELLOW instead. The dashboard backend (`read_host_health()`) maintains a small in-memory
(non-persisted, no new background service) rolling history and calls
`ami/host_health/observation.py::sustained_value()` with a 15-minute window before evaluating;
the one-shot CLI has no history, so every reading there is instantaneous-only by construction.

Full threshold table, reason-code registry (35 codes), and per-state trigger list are in
`ami/host_health/evaluator.py` module docstring/constants and are exercised by the boundary
tests in §7.

## 6. Real observation captured this batch (2026-07-08, this host)

```
state: HOST_RESTART_YELLOW
primary_reason: WINDOWS_REBOOT_PENDING
reasons: COLLECTORS_HEALTHY, COMMIT_PRESSURE_NORMAL, PAGEFILE_PRESSURE_NORMAL, RAM_NORMAL,
         SSD_SENSOR_UNKNOWN, STORAGE_HEALTHY, UPTIME_NORMAL, WINDOWS_REBOOT_PENDING
recommended_action: "Restart in the next safe maintenance window, preferably within 24 hours."
uptime: 1d 20h 10m           pending_reboot: TRUE (PendingFileRenameOperations evidence)
ram_used_pct: 77.9           commit_used_pct: 84.72        pagefile_used_pct: 56.7
d_drive_free_gb: 1041.99     distance to 800GB threshold: +241.99 GB (healthy side)
microstructure.db: 765,074,128,896 bytes    storage_health_state: STORAGE_HEALTHY
collector_status: HEALTHY    ssd_990pro_detected: True     ssd_temp_c: None (elevation-gated)
recent_unexpected_shutdown_24h: 0   recent_app_crash_24h: 2   recent_disk_ntfs/whea_24h: 0
event_log_access: OK         critical_operation_active: False
unknown_fields: ['ssd_temp_c']
```

This is a real, live, machine-produced signal (a genuine pending-file-rename registry entry was
present) — not a synthetic example. It correctly demonstrates the YELLOW tier working exactly
as specified: advisable-not-urgent, reasons fully explained, no automatic action taken.

## 7. Focused tests (Phase 17)

* `tests/test_ami_host_health_evaluator.py` — **62/62 passed.** Pure logic only; AST guards
  confirm the module imports none of `subprocess`/`socket`/`winreg`/`sqlite3`/`urllib`/`requests`/`os`
  and defines no forbidden call (`system`, `Popen`, `run`, `kill`, `remove`, …). Covers every
  threshold boundary (RAM/commit/pagefile at their elevated/critical edges), instantaneous-vs-
  sustained RAM/commit/SSD-temp, all four uptime brackets including "alone never RED", all four
  storage states, SSD sensor-unavailable non-escalation, collector HEALTHY/DEGRADED/FAILED/
  UNKNOWN (including repeated-failure → RED), critical-operation deferral text, every event-log
  category (unexpected shutdown single-vs-repeated, disk/NTFS, WHEA, OOM, app-crash threshold),
  the fail-closed UNKNOWN gate (including its priority over simultaneous RED conditions),
  determinism (identical inputs → identical output, twice), sorted/deterministic reason-code
  ordering, and `no_automatic_action is True` on every path.
* `tests/test_ami_host_health_observation.py` — **36/36 passed.** `sustained_value()` window
  semantics (empty/partial/full coverage), `_run_powershell` fail-closed on non-zero exit /
  timeout / malformed JSON / non-Windows, `_event_bucket_counts` None-vs-empty-vs-populated,
  `_collect_pids_health` over missing/ok/degraded/halted/stale/malformed `overall.json`,
  `_storage_health_state` delegation and fail-closed None/zero-total handling, `build_health_inputs`
  mapping + staleness detection, `collect_host_observation` end-to-end with psutil fully absent
  or raising on every call (never raises itself, reduces to `HOST_RESTART_UNKNOWN` end-to-end),
  and three structural guards over `_POWERSHELL_SCRIPT` itself: no `Set-`/`Remove-`/`Stop-`/
  `Restart-`/`New-`/`Clear-`/`Install-`/`Uninstall-`/`Disable-`/`Enable-` cmdlet prefix anywhere,
  no `shutdown.exe`/`Restart-Computer`/`Stop-Computer` invocation token, and every cmdlet-shaped
  token in the script (string literals excluded) starts with an allowed read-only verb.
* Combined: **98/98 passed**, run together per the repository's ≤2-test-files-per-`pytest`-call
  convention, `--basetemp=.runtime_temp/pytest_scratch -p no:cacheprovider`.

## 8. Residual risks / known gaps (honest disclosure, not hidden)

* **Production archive / catalog-index / staging-directory health** (Phase 2's `production
  archive health`, `catalog-index health`, `staging-directory health` fields) are **not** wired
  into this V1's observation model. `ami/storage/health.py::build_health_report()` exists and
  could supply them, but it requires a live `jobs` list from `ami/storage/job_state.py` that this
  read-only dashboard batch does not construct. The dashboard's "storage state" field today is
  the D:-drive free-space classifier only (reusing `storage_rotation_retention_readiness_v1.storage_health_state()`
  directly). Operators should continue to check the existing `Ops Health: Data Integrity` card
  for backup/WAL detail until a future gate wires the production-archive scan in.
* **Critical-operation-active detection is a command-line keyword heuristic**, not a canonical
  "active batch" registry (none exists in the repo). False positives/negatives are possible; it
  only ever adds a cautionary defer note, never blocks or hides the underlying state.
* **SSD temperature/SMART requires administrator elevation** on this host
  (`Get-StorageReliabilityCounter` returns access-denied under the current token) — correctly
  reported as `SSD_SENSOR_UNKNOWN` rather than fabricated, but the dashboard cannot show a live
  990 Pro temperature reading until the operator either elevates the dashboard backend process or
  a future gate adds an alternative (e.g. already-installed, already-trusted `smartctl`, per the
  spec's constraint against installing new tooling this batch).
* **Windows Event Log categorization is provider/ID-based, not exhaustively validated against
  every possible OOM/crash signature** — the resource-exhaustion (2004/2005) and application-
  crash (`Application Error`/1000) filters cover the standard, most common signatures; more
  exotic failure modes may not be captured.
* React/TypeScript card was verified via `npm run typecheck` (clean) and by directly exercising
  the backend `read_host_health()` function (real payload captured in §6); the frontend was not
  opened in an actual browser this batch (no interactive browser session available in this
  environment) — see §"Dashboard rendering verification" below for what was and wasn't checked.

## 9. Dashboard rendering verification

`npm run typecheck` (`tsc -p tsconfig.json --noEmit`) passed clean after the new card, types,
and client fetcher were added — no type errors introduced anywhere in the frontend. The backend
route was exercised directly (not via a running `uvicorn` process, to avoid starting a second
long-lived process during this batch) by calling `dashboard.backend.data_sources.read_host_health()`
in-process and confirming the exact JSON shape the React card consumes, including the 20-second
cache behavior (`assert DS.read_host_health() is payload` on the second call). A live-browser
screenshot was not captured this batch.

## 10. Verdict

**`OPERATOR_HOST_HEALTH_RESTART_READINESS_DASHBOARD_V1_COMPLETE`**

All Phase-15 completion criteria are met: observation model exists and is immutable
(frozen dataclass), pending-reboot detection exists (three-source registry evidence, fails
closed to `UNKNOWN`/`CONTRADICTORY`), uptime/memory observations exist, storage health is
integrated (D:/C: drive classifier reused from existing governance module + microstructure.db/WAL
size), collector health is integrated (reuses `logs/health/overall.json` +
`logs/collector_heartbeat.json` conventions), deterministic GREEN/YELLOW/RED/UNKNOWN evaluation
exists with 35 reason codes, the dashboard card is rendered in the existing FastAPI/React
cockpit (not a new one), machine-readable status exists (`/api/host/health` and
`python -m tools.host_health_status`), a controlled-restart checklist exists referencing only the
repository's own canonical `stop_eclipse.ps1`/`start_eclipse.ps1`/`status_eclipse.ps1`
procedure, no restart/shutdown action exists anywhere in the new code (proven structurally, not
just behaviorally), 98/98 focused tests pass, and no regression was introduced (additive-only —
see the accompanying `_STATE_TRANSITION_PROOF.md`).

**Next controlled gate (not begun):** none requested; if pursued later, the natural next step is
wiring `ami/storage/health.py::build_health_report()`'s production-archive/catalog/staging fields
into the observation model (§8), or adding an elevated (or `smartctl`-based, if the operator
already trusts and has that tool installed) SSD-temperature source.

## 11. Addendum (2026-07-08, same day) — operator-confirmed primary dashboard is `s34_live_chart.py` (`:5050`)

After this gate's first commit, the operator clarified in-session that `tools/s34_live_chart.py`
(port 5050, page title "Eclipse S34 Control") — not the FastAPI/React console — is the dashboard
actually used day to day, and asked for the same Host Health section there, with the exact
placement (own section vs. folded into "Process health") left to this agent's judgment. The
FastAPI/React `/api/host/health` route and `ControlTower.tsx` card built in §3 remain in place
and correct (harmless, unused today), but are no longer the primary consumer of this batch's
observation/evaluator modules.

**What was added, narrowly, on top of the already-committed `ami/host_health/` package:**

* `host_health_payload()` in `tools/s34_live_chart.py` — same 20s in-process cache pattern as the
  FastAPI backend's `read_host_health()` (module-level tuple `_HOST_HEALTH_CACHE`, bounded
  200-sample RAM/commit history lists for the 15-minute sustained-window calculation), wrapping
  `ami.host_health.observation.collect_host_observation()` +
  `ami.host_health.observation.build_health_inputs()` +
  `ami.host_health.evaluator.evaluate_restart_readiness()` — the exact same evaluator, same
  reason codes, same fail-closed behavior; nothing was reimplemented.
* A new `"host_health"` key added to `build_payload()`'s returned dict, in both the success path
  and the exception-fallback path (host health has no dependency on the price-chart DB query, so
  it stays available even if that query fails).
* A new **"PC / Host health"** `<div class="panel">` section (own section, chosen over folding
  into "Process health" — Process health is specifically PID-liveness; Host Health carries
  ~15 additional independent fields and reads better as its own card) plus a `renderHostHealth()`
  JS function, called from the existing `refresh()` tick alongside `renderProcesses()`.
* One dedicated test file, `tests/test_s34_live_chart_host_health.py` (9 tests): payload shape,
  20s cache hit/expiry, fail-closed behavior when the underlying collector raises, presence of
  `"host_health"` in `build_payload()`, bounded history length, HTML panel/renderer presence, and
  a structural guard that the HTML/JS never contains a restart/shutdown token or an
  `/api/restart`-`/api/shutdown`-shaped endpoint.

**Live verification performed this addendum (not just unit tests):** the operator's actual
running `:5050` process (PID 12456, up since 2026-07-06) predated this code change and would not
have picked it up without a restart. With the operator's direction in this session, the process
was stopped and immediately relaunched with the byte-identical command line
`start_eclipse.ps1` itself uses for this role (`python -u tools\s34_live_chart.py --host
127.0.0.1 --port 5050 --no-browser`) — new PID 20684. Verified live, post-restart:
`GET http://127.0.0.1:5050/api/data` returns `"host_health": {"available": true, "state":
"HOST_RESTART_YELLOW", ...}`; `GET http://127.0.0.1:5050/` (the HTML page) contains both the
`"PC / Host health"` panel title and the `renderHostHealth` function; exactly one
`s34_live_chart.py` process exists afterward (no duplicate, no orphan); no other collector or
Eclipse process was touched.

**Disclosure required by this gate's own zero-mutation-proof discipline:** this addendum's one
process action (`Stop-Process` + `Start-Process` on `s34_live_chart.py` alone) is **not** a
"collector" restart in the sense §"Do not stop or restart collectors" means — `s34_live_chart.py`
is a read-only HTTP viewer with no write path to any database, log, or collector state; it
neither ingests nor writes market data. It is nonetheless a real process stop/start, explicitly
directed by the operator in this session (not automatic, not silent), and is recorded honestly as
`processes_killed_this_addendum: 1` in the accompanying JSON and proof documents rather than
rounded down to zero — this repository's governance convention (see prior storage-rotation
batches) is to disclose exactly what happened, not to force a round number.

Updated verdict scope: `OPERATOR_HOST_HEALTH_RESTART_READINESS_DASHBOARD_V1_COMPLETE` now covers
**two** working dashboard integration points (FastAPI/React, and the operator-confirmed primary
`s34_live_chart.py`), 107/107 combined focused tests (98 from the original commit + 9 new),
still zero collector/database/registry/Windows-configuration mutation.
