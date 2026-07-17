# S34 State Machine — Canonical Operator Dashboard

**This is the canonical, continuously-used operator dashboard for Eclipse.** It is
an operator **safety and decision-support surface**, not a cosmetic chart. New
operational-visibility and safety panels integrate here by default. Alternate
dashboards remain secondary unless an independently-accepted governance migration
explicitly supersedes it (see `SYSTEM_STATE.md` and `CLAUDE.md`).

> The FastAPI/React app under `dashboard/backend/app.py` + `dashboard/frontend/`
> (documented in `dashboard/README.md`) is the **legacy secondary** subsystem. It
> is **not** the canonical operator surface and is **never imported** by the
> canonical entrypoint below.

## Canonical entrypoint

```
tools/s34_cascade_navigation_dashboard.py
```

- **Default invocation is unchanged**: it still produces the point-in-time,
  no-lookahead CLI **JSON + Markdown** report (`--json-out` / `--md-out`, all
  existing CLI arguments and output-path contract preserved).
- **Web operator UI** is opt-in via a separate, explicit mode:

```powershell
python tools/s34_cascade_navigation_dashboard.py --serve            # 127.0.0.1:8770
python tools/s34_cascade_navigation_dashboard.py --serve --serve-port 8771
```

CLI report mode and web serve mode are explicitly separated; `--serve`
short-circuits to the read-only web surface and never runs the report path.

## Read-only safety contract (non-negotiable)

The web surface serves **GET/HEAD only** and binds **127.0.0.1** only:

| Method | Result |
|---|---|
| `GET /`, `GET /api/overview`, `GET /api/panel/<key>`, `GET /static/*` | 200 (or typed fail-closed) |
| `HEAD` | 200, no body |
| `POST`, `PUT`, `PATCH`, `DELETE`, `OPTIONS` | 405 |
| unknown route | 404 |

It has **no control actions**: no trade/order placement, no cancel-order, no live
executor activation, no scheduler activation, no process start/stop/restart, no
config/env mutation, no DB writes. **All SQLite access is `mode=ro`.** No secrets
or credentials are rendered. Mutating routes do not exist.

## Data sources, freshness & trust model

Every panel is a typed `PanelViewModel` (`dashboard/backend/view_models.py`)
carrying: `source_path`, `source_timestamp`, `read_timestamp`, `age_seconds`,
`freshness_threshold_seconds`, `parse_status`, `provenance_status`, `trust_state`,
`reason_codes`, `raw_evidence`.

`trust_state ∈ {current, stale, missing, malformed, inferred}` is computed
centrally (`dashboard/backend/freshness.py`). **A non-`current` panel is visually
muted with an explicit badge — stale is never shown as a bare GREEN.** Missing /
stale / malformed / ambiguous inputs **fail closed**.

Freshness thresholds are configured in `dashboard/backend/sources.py`
(`FRESHNESS`). Large stores (the 650 GB+ `microstructure.db`, JSONL ledgers,
`SYSTEM_STATE.md`) are read with **bounded** stat / tail / single mode=ro queries
— never wholesale.

## Information architecture (10 sections)

1. Executive Safety Header (sticky) — live-executor OFF/ON, scheduler, branch/HEAD,
   canonical governance state, watchdog, oldest-artifact age, open findings.
2. Runtime Process Topology — per-role PID/identity/expected-vs-actual/duplicate/orphan.
3. S34 State Machine (primary) — state, closed-ids/duplicate-close guard, bounded history.
4. Execution Safety — live-executor count (expected 0), pending/cancel, `NO CONTROL ACTIONS AVAILABLE`.
5. Liquidation-Silence / Gate 2 — detector health, continuity, scheduler, launcher-validation-pending.
6. Native WebSocket & Collectors — connection/age/backoff; **does not absorb the separately-owned dirty native-WS policy package**.
7. Storage & Data Integrity — DB mode=ro accessibility, size, free space, bounded newest-row probe.
8. Research & Strategy Readiness — validated registry; research is **never** shown as execution-approved.
9. Governance — canonical state, findings by severity, branch/HEAD, foreign-owned display.
10. EXECMGMT / STOPPROT — the accepted read-only sizing/stop panels (see below).

Panels also cover the **Artifact Trust Model**: each artifact-backed panel shows
source path/timestamp/age/threshold/parse/provenance and its trust state.

## EXECMGMT / STOPPROT and the `-175.7 bps` reference

The accepted EXECMGMT/STOPPROT panels are preserved (self-contained port). The
worst-real-fill value **`-175.7 bps`** is explicitly labelled as *Historical
reference / research worst-fill evidence* with its source path/timestamp, and as
a *fallback historical reference* when the live execution-management audit is
absent. It is never presented as active config or a current runtime limit
(`worst_real_fill_bps_is_active_config = false`).

## Architecture

```
tools/s34_cascade_navigation_dashboard.py   # canonical entrypoint (CLI + --serve dispatch)
dashboard/backend/
  view_models.py        # TrustState/Severity/PanelViewModel typed contract
  freshness.py          # central trust/freshness evaluation
  sources.py            # DashboardContext + source paths + freshness thresholds
  readers.py            # fail-closed bounded JSON/JSONL readers; open_ro (mode=ro); numeric-only env
  process_identity.py   # read-only psutil role/duplicate/orphan topology
  canonical_state.py    # bounded SYSTEM_STATE.md parse + git plumbing (no subprocess)
  aggregator.py         # runs each adapter isolated; partial-failure isolation
  server.py             # GET/HEAD-only stdlib http.server; loopback-only
  adapters/*.py         # one read-only adapter per information-architecture section
dashboard/templates/index.html
dashboard/static/app.css, app.js     # dark ops theme, sticky header, trust overlays, bounded auto-refresh
```

Partial-failure isolation: a failing adapter degrades only its own panel (typed
fail-closed) — the page never 500s, and only the exception **class name** is
surfaced (no message/secret/full-environment dump).

## Testing

Committed tests under `tests/dashboard/` (safety, HTTP methods, trust/freshness,
process topology, adapters, EXECMGMT/STOPPROT, aggregator isolation). Run per the
repository guardrails (max 2 test files per invocation, external `--basetemp`,
`-p no:cacheprovider`):

```powershell
python -m pytest tests/dashboard/test_safety_contract.py tests/dashboard/test_http_server.py `
  -p no:cacheprovider --basetemp <external> -q
```

## Rollback / disable

The web UI is opt-in: simply do not pass `--serve` (the CLI report is the
default and is unaffected). To stop a running dev/test server, use its documented
graceful shutdown (`Ctrl-C`, or `httpd.shutdown()` in-process) — never a hard
kill. Removing/reverting the `--serve` args restores the pre-existing entrypoint
exactly; nothing else in the trading system depends on this package.

## No-control guarantee

This surface cannot place/cancel orders, arm the live executor, start/stop the
scheduler, manage processes, mutate config/env, or write any database or source
artifact. Enforced by construction (stdlib GET/HEAD server, mode=ro DB access,
numeric-only env reads) and by `tests/dashboard/test_safety_contract.py`.
