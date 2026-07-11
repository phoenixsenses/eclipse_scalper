# Liquidation Silence / Transport-Outage Detector — Implementation, 2026-07-11

**Verdict: `LIQUIDATION_SILENCE_DETECTOR_CORRECTED_AWAITING_REREVIEW`.**

> **Corrective pass (2026-07-11, Opus 4.8):** an independent read-only review
> returned `LIQUIDATION_SILENCE_DETECTOR_CORRECTIVE_IMPLEMENTATION_REQUIRED`
> (three MEDIUM findings + LOW items). This report has been corrected in
> place; the full corrective record is **§19 (Corrective Implementation)** at
> the end. Headline changes: complete-symbol-evidence gating (no more
> all-symbol transport-outage claim from an incomplete symbol set), a
> decision-logic policy fingerprint (`POLICY_SPEC` sha256, not thresholds
> only), an explicit `LIVE` / `HISTORICAL_REPLAY` evaluation mode that
> structurally blocks present-day evidence contaminating a historical replay,
> structured DB error propagation, future-timestamp handling, and symbol
> normalization. Frozen thresholds (3600 / 7200 / 9000 / 300) are UNCHANGED.
> `POLICY_VERSION` is now `liquidation_silence_policy_v2_2026-07-11`; the v1
> fingerprint is recorded as superseded in §5.

Implementation-complete, tested (86/86), replayed against real production
history under the corrected safe historical mode, performance-rehearsed. Not
activated, not scheduled, not wired into `tools/heartbeat_watchdog.py`. Live
executors remained OFF throughout (never touched). No production database
mutation. No process restarted.

---

## 1. Motivating Incident

A confirmed all-tracked-symbol liquidation outage ran
**2026-04-27T14:24:51.798Z (SOLUSDT last good) / 14:27:26.345Z (ETHUSDT,
BTCUSDT last good) → 2026-06-06T17:47:03.630Z (ETHUSDT) / 17:47:04.094Z
(BTCUSDT) / 17:47:05.191Z (SOLUSDT)** — **≈40 days 3 hours**. All three
tracked symbols entered and exited the gap within seconds of each other
(single shared root cause, not per-symbol). Full forensic detail:
`reports/research/s34/S34_HOUR17_CYCLE_ADJUSTED_RECOMPUTE_AND_MAY_GAP_FORENSIC_2026-07-11.md`
§4 (root-cause confidence: `ROOT_CAUSE_PROBABLE`, network/VPN transport-path
blocker, first-party 2026-05-03 diagnostic confirmed connection-successful/
subscription-confirmed/zero-frames on liquidation **and control** endpoints
at that specific instant; archive/rotation ruled out; no deletion evidence).

A second, shorter, mechanistically distinct outage
(**2026-07-06T10:06:39.307Z → 2026-07-10T11:24:37.685Z**, ≈4 days — a routed-
WS-endpoint regression, `BINANCE_WS` reverted from `/market/stream` to
`/stream` in commit `5cda3122`) is documented in
`reports/research/s34/LIQUIDATION_TRANSPORT_RESTORED_2026-07-10.md` and is
already partially covered by `tools/native_ws_health_policy.py`'s existing
**global** (all-744-symbol-table) `liquidations` freshness check
(`LIQUIDATIONS_WARNING_AGE_SEC=600`/`LIQUIDATIONS_CRITICAL_AGE_SEC=1800`,
wired into `logs/health/overall.json` since the 2026-07-10 corrective pass).
That existing check has **no per-symbol granularity** and does not itself
branch to an evidence-insufficient state when control streams are
simultaneously stale — the gap this detector fills. `tools/
native_ws_health_policy.py` and its test file are currently dirty/uncommitted
under another session's ownership (see §2) and were **not modified** by this
batch; they were read only, for calibration reference.

---

## 2. Preflight and Ownership

- Branch `codex/data-layer-fallback-cleanup`, HEAD `884bbb30443b0a42417e9ff6e64173e0d5e3cfbc`.
- Required accepted commits (`5448e856`, `1c1c12ac`, `884bbb30`, `f02b7d88`) all confirmed ancestors of HEAD.
- Dirty/foreign-owned files present at session start (per the task's own inventory, independently reconfirmed): `.claude/settings.local.json`, `runtime/dashboard_backend.json` (deleted), `status_eclipse.ps1`, `stop_eclipse.ps1`, `tests/test_native_ws_health_policy.py`, `tools/native_ws_health_policy.py`, `tools/s34_cascade_navigation_dashboard.py`. Cross-confirmed by `reports/research/s34/CANONICAL_OPERATIONAL_HEALTH_2026-07-10.md` §9's own "deliberately not touched" list, which names the identical set.
- The canonical health architecture's sole-writer boundary (`tools/heartbeat_watchdog.py::build_canonical_overall` owns `logs/health/overall.json`; `tools/health_state.py::write_component_health` is the shared component-file writer and structurally rejects `component="overall"`) lives in **clean, already-committed** files (commits `00ef49ad`, `81ec6d71`, `f3d95f5e`, `6faa2177`, `9b43d182`) — neither `tools/health_state.py` nor `tools/heartbeat_watchdog.py` was dirty.
- **Resolution:** the complete detector was implementable in entirely new, isolated files (`tools/liquidation_silence_policy.py`, `tools/liquidation_silence_detector.py`, two new test files, this report) using only the existing, clean `write_component_health` contract — zero edits to any dirty file, zero edits even to the clean `tools/heartbeat_watchdog.py` (see §9 for why activation-wiring was deliberately deferred rather than added-but-disabled there). No `LIQUIDATION_SILENCE_DETECTOR_BLOCKED_BY_HEALTH_OWNERSHIP` condition applied.
- Live PIDs at session start (verified via read-only `Get-CimInstance Win32_Process` enumeration): collector `3828`, collector_supervisor `23052`, heartbeat_watchdog `22816`, bookticker `5624`, oi_spot_poller `23124`, s34_live_chart `23728`, s34_v_engine_v02_shadow_mirror `11672`, event_diary `19224`, orderflow_chart `5244`, s34_replay `14444`, s34_shadow_paper_runner `19504`, s34_realtime_shadow_runner `21576` — 12 processes, no duplicates, zero live-executor process. Re-enumerated identical after implementation, testing, and performance rehearsal (§8).

---

## 3. Historical Calibration (read-only)

All queries below used `data/microstructure.db` opened `mode=ro`, bounded via
the existing `idx_liq_ts`/`idx_liq_symbol_ts`/`idx_mark_ts`/`idx_trade_ts`
indices (`ORDER BY ts_ms [ASC|DESC] LIMIT 1` or `WHERE ts_ms BETWEEN`/`<= `
range scans) — never a full table scan, confirmed via `EXPLAIN QUERY PLAN`
(§8).

**Tracked-symbol universe** discovered at evaluation time from
`logs/pids/collector_supervisor.json`'s own `"symbols"` field (written by
`start_eclipse.ps1`, UTF-8-BOM-prefixed — handled): `BTCUSDT,ETHUSDT,SOLUSDT`
as of this session. Never hardcoded except as an explicit, reported
last-resort fallback (`symbol_source="fallback_default"`) when that file is
unreadable.

**Data coverage:** `liquidations`/`mark_prices`/`agg_trades` span
2026-02-15T14:30Z → 2026-07-11 (live). SOLUSDT liquidations begin
2026-04-18T08:41Z (joined later than BTC/ETH).

**Architecture-change caveat (excluded from calibration, reason recorded):**
the collector's `liquidation_stream_mode` default became `all_market_arr`
(subscribing to the full-market `!forceOrder@arr` stream) on **2026-06-06**
as part of the first outage recovery
(`LIQUIDATION_TRANSPORT_RESTORED_2026-06-06.md`). A 2026-04-20→04-27 sample
window (pre-change) showed all-symbol combined event density **≈25.6x lower**
than any post-2026-06-06 healthy window despite comparable per-symbol
(BTC/ETH) counts — consistent with a narrower pre-change subscription scope,
not organic volatility. **Excluded from the frozen-policy calibration
population** as non-representative of the current architecture; per-symbol
counts from that window were reviewed but not used to set thresholds either,
for consistency.

**Calibration duration (corrected).** The healthy calibration population is
**307 hours of included healthy coverage** (168h + 120h + 19h across the
three windows below) = **≈12.8 elapsed days**, spanning **14 distinct
calendar days**. (An earlier draft summarized this loosely as "~19 days";
307h / ≈12.8 elapsed days / 14 calendar days is the exact form.) Calibration
adequacy is **`ADEQUATE_WITH_LOW_LIMITATION`**: the frozen thresholds are
validated for the **current** (`all_market_arr`, post-2026-06-06)
subscription architecture. The **0 false positives** figure applies only to
the evaluated healthy windows (not a universal guarantee). When replayed
against the **excluded** older, lower-density per-symbol subscription regime,
one genuine `ALL_SYMBOL_SILENCE_WARNING` (YELLOW) can occur — that is **not a
false positive under the current architecture**, but it means threshold
*portability to the older regime is unproven*. Thresholds were not revised in
light of this.

**Healthy calibration population (all three windows post-2026-06-06,
2026-07-06..07-10 outage excluded):**

| Window | Purpose |
|---|---|
| 2026-06-10T00:00 .. 2026-06-17T00:00Z | post-first-recovery healthy |
| 2026-06-28T00:00 .. 2026-07-03T00:00Z | pre-second-outage healthy |
| 2026-07-10T12:00 .. 2026-07-11T07:00Z | post-second-recovery healthy (partial, ~19h) |

**Excluded intervals and reasons:**

| Interval | Reason |
|---|---|
| 2026-04-20 .. 2026-04-27T14:00Z | pre-architecture-change subscription scope, non-representative (see above) |
| 2026-04-27T14:24 .. 2026-06-06T17:48Z | confirmed all-symbol liquidation outage #1 (~40d3h) |
| 2026-07-06T10:06 .. 2026-07-10T11:25Z | confirmed all-market liquidation outage #2 (routed-endpoint regression, ~4d) |

**Per-symbol inter-arrival gaps** (combined across the three healthy
windows, BTCUSDT/ETHUSDT/SOLUSDT):

| Symbol | n gaps | max | p50 | p90 | p95 | p99 | p999 |
|---|---:|---:|---:|---:|---:|---:|---:|
| BTCUSDT | 13,738 | 6018.6s | 6.04s | 224.4s | 430.6s | 1072.4s | 2159.8s |
| ETHUSDT | 13,166 | 4926.6s | 6.85s | 241.7s | 451.3s | 1087.2s | 1938.3s |
| SOLUSDT | 7,610 | 6733.2s | 9.19s | 434.2s | 807.6s | 1832.3s | 3705.0s |

**All-tracked-symbol combined** (chronological freshest-of-3
reconstruction — the age of "how long since ANY tracked symbol last
ticked", the exact quantity `all_symbol_silence_age_sec` measures):

| Window | n | max | p50 | p95 | p99 | p999 |
|---|---:|---:|---:|---:|---:|---:|
| post_recovery_healthy | 18,575 | 2508.9s | 1.74s | 183.8s | 519.8s | 1040.9s |
| pre_july_outage_healthy | 14,874 | 1968.5s | 1.71s | 160.6s | 473.0s | 1081.2s |
| post_july_recovery_recent | 1,077 | 2073.1s | 2.60s | 347.6s | 1117.4s | 1645.6s |
| **combined** | **34,526** | **2508.9s** | — | — | **518.9s** | **1141.4s** |

Distribution by UTC hour / weekday (all-symbol combined stream, max gap per
bucket): every bucket's max stayed under 820s; no hour or weekday showed
materially elevated silence risk (hour range 73.8s–818.0s, weekday range
70.0s–818.0s) — no low-volatility-period or session-specific adjustment was
warranted.

**Control-stream continuity evidence (point-sampled, 16 points across
outage #1, 10 points across outage #2):** `mark_prices`/`agg_trades`
remained fresh (0–11s lag, one isolated ≈1700s blip at 2026-05-21T16:26Z —
see §5) throughout **both** confirmed liquidation outages, via REST
fallback (`data/microstructure_collector.py` has REST coverage for
`aggTrade`/`markPrice` but not `forceOrder`/liquidations — see
`LIQUIDATION_TRANSPORT_RESTORED_2026-07-10.md` "Permanent Data Gap"). This
is the direct empirical basis for classifying both historical incidents as
liquidation-**transport-specific**, not general data-layer outages.

**Existing canonical threshold reviewed (not blindly reused):**
`tools/native_ws_health_policy.py`'s global `LIQUIDATIONS_WARNING_AGE_SEC=
600`/`LIQUIDATIONS_CRITICAL_AGE_SEC=1800` (derived from a different, narrower
~90-120min pre-2026-07-06 sample, all-744-symbol scope, no per-symbol
granularity). This detector's own all-tracked-symbol combined empirical
maximum (2508.9s) already exceeds that 600s warning value, confirming the
two detectors measure genuinely different quantities and independent
calibration was necessary, not optional.

---

## 4. Frozen Detector Policy

`tools/liquidation_silence_policy.py`, `POLICY_VERSION =
"liquidation_silence_policy_v2_2026-07-11"` (numeric thresholds UNCHANGED
from v1; only the surrounding evidence semantics were corrected — see §19).

| Threshold | Value | Margin over observed historical max | Basis |
|---|---:|---:|---|
| `SYMBOL_SILENCE_WARNING_AGE_SEC` | 9000s (150min) | 1.34x over 6733.2s (SOLUSDT max) | per-symbol gap calibration |
| `ALL_SYMBOL_SILENCE_WARNING_AGE_SEC` | 3600s (60min) | 1.43x over 2508.9s | all-symbol combined calibration |
| `ALL_SYMBOL_SILENCE_CRITICAL_AGE_SEC` | 7200s (120min) | 2.87x over 2508.9s | all-symbol combined calibration |
| `CONTROL_STREAM_FRESH_AGE_SEC` | 300s (5min) | 1.67x over native_ws_health_policy's own 180s AGG_TRADES_CRITICAL; above the observed ≈1700s mid-outage control blip (correctly classified as stale, not silently absorbed) | cross-check against existing, already-validated control-freshness convention |

No threshold was chosen or adjusted by looking at replay outcomes (§6) —
frozen before replay, replay run once against the frozen values, and (per
task instruction) not revised afterward. Not optimized against strategy
P&L or any trading outcome.

## 5. Policy Fingerprint

**Current (v2, decision-logic fingerprint):**

```
POLICY_VERSION     = liquidation_silence_policy_v2_2026-07-11
POLICY_FINGERPRINT = e117cf132bce3bd180af3c718670d3c75910dd69206588d4b7f1b341aadf2291
```

**Superseded (v1, thresholds-only fingerprint) — retained for provenance:**

```
POLICY_VERSION     = liquidation_silence_policy_v1_2026-07-11
POLICY_FINGERPRINT = 9781e0ed8f7b4950e62bdb6b4e64773ef1f9f6e383749b92ac20641dec4ed9d8   (SUPERSEDED)
```

The v1 fingerprint was `sha256` over only the four threshold values + a
manually maintained version string — so a material *logic* change (a boundary
operator, the classification precedence, the all-symbol aggregation method,
the native-WS precedence, ...) could occur without moving it (review MEDIUM
2). The **v2** fingerprint is `sha256` over the canonical (`sort_keys=True`)
JSON of the full **`POLICY_SPEC`** decision-logic specification
(`tools/liquidation_silence_policy.py::_policy_spec` /
`_fingerprint_of`): thresholds, boundary operators, age-clamping / future-
timestamp semantics, the complete-evidence requirement, the all-symbol
aggregation method, the per-symbol rule, the decision precedence list, the
native-WS/collector precedence, stale/missing/corrupt-evidence behavior, the
status→severity map, and the output schema version. Any semantic change now
necessarily changes the fingerprint; it is insensitive to key ordering /
whitespace (proven by unit tests). Because semantics changed in this
corrective pass, the v2 fingerprint **differs** from the superseded v1 value
above (also asserted by a unit test). `SUPERSEDED_V1_POLICY_FINGERPRINT` is
kept as a module constant for provenance.

---

## 6. Detector Semantics

Seven classifications, exactly matching the task's conceptual set (internal
enum names differ trivially per repo convention):

| Classification | Severity | Trigger |
|---|---|---|
| `HEALTHY` | GREEN | no symbol/all-symbol/control/native-WS condition triggered |
| `SYMBOL_SILENCE_WARNING` | YELLOW | ≥1 (not all) tracked symbol ≥ symbol-warning age |
| `ALL_SYMBOL_SILENCE_WARNING` | YELLOW | all tracked symbols ≥ all-symbol-warning age, AND controls advancing |
| `LIQUIDATION_TRANSPORT_OUTAGE` | RED | all tracked symbols ≥ all-symbol-critical age, AND controls advancing |
| `CONTROL_STREAMS_STALE` | UNKNOWN | all-symbol age ≥ warning/critical, but controls are **not** advancing (or missing) — refuses to claim liquidation-specific causation |
| `NATIVE_WS_UNHEALTHY` | RED (native RED) / YELLOW (native DEGRADED) | existing canonical `native_ws_status` already RED/DEGRADED — preserved/explained, never downgraded, never re-evaluated as if it were a liquidation-specific issue |
| `UNKNOWN_INSUFFICIENT_EVIDENCE` | UNKNOWN | no tracked symbols configured, all symbol evidence missing, malformed input, or collector process confirmed down |

Core rule enforced structurally (not just by convention): control-stream
consultation only happens **inside** the all-symbol-silence branches — a
missing/stale control reading is irrelevant noise while liquidation itself
is healthy (verified by `test_missing_control_evidence_irrelevant_when_
liquidation_healthy`), and is the deciding factor only once liquidation
silence is already material. A lone quiet symbol never escalates past
YELLOW by construction (`SYMBOL_SILENCE_WARNING` has no RED path).

---

## 7. Component Output Contract

`logs/health/liquidation_silence.json`, written **only** via
`tools.health_state.write_component_health("liquidation_silence", payload,
root=...)` (atomic temp-file + `os.replace`, matching every other component
in this repository). Fields: `schema_version`, `component`, `policy_version`,
`policy_fingerprint`, `evaluated_at_utc`/`evaluated_at_ms`/`ts_utc`
(gradeable by the existing `component_fresh()` convention),
`status`/`severity`/`reason_codes`, `tracked_symbols`/`symbol_source`,
`last_liquidation_ts_ms`, `per_symbol_silence_age_sec`,
`all_symbol_silence_age_sec`, `control_stream_ages_sec` (+ the two named
ages), `native_ws_status`, `collector_component_alive`, `thresholds`,
`data_sources` (non-sensitive local paths only), `detector_runtime_sec`,
`error`.

---

## 8. Canonical-Integration Design (disabled)

Nothing in this batch calls `tools/liquidation_silence_detector.py` from
any scheduled or production code path: not imported by
`tools/heartbeat_watchdog.py`, not added to `start_eclipse.ps1`, no looping
mode exists in the module itself. `tools/heartbeat_watchdog.py` was **not
edited at all** — deliberately, even for a disabled flag: it is the
sole-writer file for `overall.json`, was just hardened and closed out in the
2026-07-10 corrective pass (250/250 tests, independent review accepted), and
touching it — even inertly — was judged unnecessary risk for a batch this
task explicitly scopes as implementation-only.

`test_disabled_integration_has_zero_effect_on_canonical_overall` proves
this concretely against the real, unmodified `tools.heartbeat_watchdog`
module: `"liquidation_silence"` is absent from its
`OPTIONAL_COMPONENT_FILES` registry, and a real `liquidation_silence.json`
written to an isolated health root has zero effect on
`build_canonical_overall`'s output.

**Intended future composition** (defined, tested in isolation, not wired):
`tools/liquidation_silence_policy.py::compose_with_overall_severity(existing,
detector)` — detector YELLOW/UNKNOWN raises overall to at least YELLOW,
detector RED forces overall RED, detector GREEN never downgrades a more
severe existing verdict. For a future controlled-activation batch: add
`"liquidation_silence": "liquidation_silence.json"` to
`OPTIONAL_COMPONENT_FILES`, and fold `compose_with_overall_severity` into
`evaluate()`'s critical/warning list construction the same way
`native_ws_policy` already is.

---

## 9. Files Changed

New files only:

- `tools/liquidation_silence_policy.py` — pure decision policy
- `tools/liquidation_silence_detector.py` — bounded read-only snapshot + one-shot CLI + component writer
- `tests/test_liquidation_silence_policy.py` — 28 pure-policy unit tests
- `tests/test_liquidation_silence_detector.py` — 19 I/O / writer-ownership / bounded-query / historical-replay test functions (24 collected items — one is 6-way parametrized)
- `reports/research/s34/LIQUIDATION_SILENCE_DETECTOR_2026-07-11.md` — this report

No existing file was modified. No file listed as foreign-owned (§2) was
touched.

---

## 10. Test Results

```
pytest tests/test_liquidation_silence_policy.py tests/test_liquidation_silence_detector.py \
  -p no:cacheprovider --basetemp=<scratchpad>/pytest_basetemp -v
52 passed in 1.39s
```

All 25 required scenarios covered (several with multiple tests):
recent-healthy→GREEN, single-symbol-silence→warning-not-all-symbol,
all-symbol-warning→YELLOW, all-symbol-critical→RED, controls-stale→UNKNOWN
(both warning- and critical-tier), native-WS-RED-preserved,
native-WS-DEGRADED-preserved, exact warning/critical boundaries (inclusive
`>=`) + just-below-boundary, per-symbol boundary inclusivity, missing-symbol
evidence (partial + total), missing-control evidence (material +
irrelevant-when-healthy), malformed input (bad type, empty universe,
non-numeric timestamp, collector-down), corrupt-predecessor overwrite,
atomic-write-no-tmp-leftover, deterministic fingerprint + byte-identical
repeat payload, never-writes-overall.json, mode=ro write-rejection, DB
mutation-checksum (fixture), disabled-integration zero-effect (real
`heartbeat_watchdog` module), `compose_with_overall_severity` YELLOW/RED/
GREEN-never-downgrades mapping, ts_utc staleness-gradeable, symbol-universe
discovery (canonical + BOM handling + missing/corrupt fallback + arbitrary
universe), bounded-query plan (`EXPLAIN QUERY PLAN` asserts covering-index
search, not `liquidations` table scan) + 50k-unrelated-row timing (<2s),
and two real-production-DB historical replays (April-June outage → RED;
July routed-endpoint outage → RED) plus a 6-point healthy-period sweep
(never RED).

---

## 11. Historical Replay

All replay evaluations used `evaluate_once(...)` with `pid_meta_path`/
`overall_path`/`collector_component_path` pointed at nonexistent paths (so
no *current* live-system file leaks into a *historical* point-in-time
evaluation — point-in-time, no-lookahead by construction; the `ts_ms <=
now_ts*1000` bound added to both DB-read functions during this batch, see
§13 finding 1, is what makes this correct). 5-minute step granularity
below explains the ~2-8 minute overshoot past the exact 3600s/7200s
threshold instants.

**Outage #1 onset (2026-04-27):** last good liquidation 14:27:26.345Z.
First `ALL_SYMBOL_SILENCE_WARNING` (YELLOW) at 15:30:00Z (latency 3753.7s
≈62.6min). First `LIQUIDATION_TRANSPORT_OUTAGE` (RED) at 16:30:00Z (latency
7353.7s ≈122.6min) — **vs. the ~40 days it actually took to notice.**

**Outage #1 recovery (2026-06-06):** RED continuously through 17:47:00Z
(age 3,467,973.7s). At 17:47:05Z (SOLUSDT's confirmed recovery instant) →
immediately `SYMBOL_SILENCE_WARNING` (age 0.3s), then `HEALTHY` by
17:47:10Z. **Recovery-to-GREEN latency: ≈5-10s.**

**Outage #2 onset (2026-07-06):** last good liquidation 10:06:39.307Z.
First YELLOW at 11:10:00Z (latency 3800.7s ≈63.3min). First RED at
12:10:00Z (latency 7400.7s ≈123.3min).

**Outage #2 recovery (2026-07-10):** already `HEALTHY` at the first probed
point (11:24:00Z, age 79.6s) — the freshest tracked symbol (not necessarily
BTCUSDT, whose own individual recovery lagged slightly per
`LIQUIDATION_TRANSPORT_RESTORED_2026-07-10.md`'s BTC-specific evidence)
had already ticked, correctly collapsing `all_symbol_silence_age_sec` (a
`min()` over tracked symbols) even before every individual symbol had
recovered — the intended per-symbol-vs-all-symbol distinction working as
designed on real data, not just in unit tests.

**Healthy-period false-positive sweep:** 310 hourly evaluations across the
three calibration windows (2026-06-10..17, 2026-06-28..07-03,
2026-07-10T12:00..07-11T07:00) → **310/310 HEALTHY, 0 YELLOW, 0 RED, 0
UNKNOWN.**

**Control-stream-stale moment (2026-05-21T16:30Z, inside outage #1):**
classified `CONTROL_STREAMS_STALE`/UNKNOWN (`mark_prices_age≈15950s`,
`agg_trades_age≈15956s` at that instant), **not**
`LIQUIDATION_TRANSPORT_OUTAGE` — correctly refusing to claim
liquidation-specific causation at the one moment inside the confirmed
outage where the evidence didn't support it.

**Native-WS-unhealthy replay:** no historical `logs/health/overall.json`
snapshots are retained from the incident period (only the current live
file exists), so this dimension could not be replayed against real
historical data — it is covered instead by
`test_native_ws_red_is_preserved_even_when_liquidation_healthy` /
`test_native_ws_degraded_preserved_when_no_liquidation_specific_evidence`
(pure-policy unit tests). Documented explicitly rather than silently
omitted.

Single frozen policy version was replayed (`liquidation_silence_policy_v1_
2026-07-11`); no threshold was revised after seeing these results.

---

## 12. False-Positive Analysis

Zero false YELLOW/RED/UNKNOWN across 310 independent hourly healthy-period
evaluations (0.0%). Longest false warning: none observed. This is expected
given thresholds carry 1.34x-2.87x margin over the corresponding calibration
population's observed historical maximum (§4) — not a coincidence of the
particular sweep points chosen.

---

## 13. Performance Rehearsal

One-shot `run_once(...)` against the real, live, 796,167,065,600-byte
(≈741GiB) `data/microstructure.db`, `mode=ro`, output directed to an
isolated scratch path (never the real `logs/health/`):

```
import_time_sec            = 0.0334
run_once_wall_time_sec      = 0.00346
detector_reported_runtime   = 0.00233
query_count                 = 5  (3x per-symbol liquidation lookup, 2x control-table lookup)
```

Each query confirmed via `EXPLAIN QUERY PLAN` to use a covering index
search (`idx_liq_symbol_ts` / `idx_mark_ts` / `idx_trade_ts`), never a table
scan — reconfirmed directly against the real production DB (not just the
test fixture), including with the `ts_ms <= ?` no-lookahead bound applied.

**DB/state immutability (§15 also):** whole-file `sha256` was judged
infeasible and non-diagnostic for this specific 741GiB, continuously-and-
legitimately-written-by-the-live-collector database (a live-collector
before/after hash would differ regardless of this detector's actions,
making it a meaningless signal here) — the same reasoning
`tools/native_ws_health_policy.py`'s own comments give for avoiding
`MAX(ts_ms)`-style full scans on this database. Instead: a fixed historical
window row count (`BTCUSDT`/`ETHUSDT`/`SOLUSDT` liquidations,
2026-04-01..02, a period no live process can still be writing into) was
captured before (1092/1318/0) and after (1092/1318/0, identical) the
rehearsal run — proving no historical mutation. `mode=ro` itself is proven
structurally write-rejecting by `test_production_db_connection_uses_read_
only_mode_and_rejects_writes` (raises `sqlite3.OperationalError` on INSERT).
DB file size grew by 73,728 bytes over the 35.6s before/after window,
consistent with ordinary ongoing collector throughput (~1-3 rows/sec across
three tables) and not attributable to this read-only detector.

`logs/health/{overall,watchdog,collector,bookticker}.json` checksums
changed between before/after captures — expected and unrelated to this
work: `tools/heartbeat_watchdog.py` (PID `22816`, `--interval-sec 5`) and
the live collector (PID `3828`) continuously rewrite these files as normal
production operation; this detector never opens any of them for writing
(only `overall.json` and `collector.json` are ever opened, both strictly
read-only). `logs/health/paper_trader.json` was unchanged (no paper-trading
activity in the window), a useful negative control. The rehearsal's own
output (`liquidation_silence.json`) was written only to the isolated
scratch path — confirmed absent from the real `logs/health/` directory
both before and after.

---

## 14. DB/State Immutability Summary

- `mode=ro` connection: write attempt raises `sqlite3.OperationalError` (test-proven).
- Fixed historical window row counts: byte-identical before/after rehearsal.
- No checkpoint/PID/trade-store file touched (this detector reads only `logs/pids/collector_supervisor.json`, `logs/health/overall.json`, `logs/health/collector.json`, `data/microstructure.db` — all read-only).
- Post-rehearsal PID enumeration: identical 12-process set to preflight (§2), zero duplicates, zero live-executor process.

---

## 15. Live State

- Live executors: OFF / OFF, before, during, and after this batch (repeatedly reconfirmed via read-only process enumeration).
- Canonical health: GREEN (unaffected — this detector's component file was never written to the real `logs/health/` directory).
- Native WS health: GREEN.
- No process restarted, stopped, or started by this batch.
- No `.env`/`execution/`/`risk/`/`brain/`/`tools/s34_state_machine_live_executor.py` file touched.

---

## 16. Worktree Isolation

`evaluate_once`/`run_once` accept `db_path`/`health_root`/`pid_meta_path`/
`overall_path`/`collector_component_path` as explicit parameters (no hidden
cwd-relative default beyond the module's own `ROOT`-derived constants) —
safe to invoke from a throwaway worktree with all paths redirected, matching
the precedent `tools/health_cycle_smoke.py --root` and
`data/microstructure_collector.py --health-root` already set.

---

## 17. Findings

1. **Lookahead bug found and fixed during this batch** (not present in the
   final implementation): the first draft of `read_last_liquidation_ts`/
   `read_control_freshness` had no `now_ts` upper bound, so a historical
   replay call would silently pick up the row nearest the *real* wall clock
   regardless of the requested historical instant — caught immediately by
   the two replay tests failing (`HEALTHY` instead of the expected
   `LIQUIDATION_TRANSPORT_OUTAGE`), fixed by adding a `ts_ms <= now_ts*1000`
   bound to both functions (still index-bounded, reconfirmed via
   `EXPLAIN QUERY PLAN`). A live one-shot run is unaffected (its own real
   `now_ts` is already an upper bound on any row that exists).
2. `logs/pids/collector_supervisor.json` carries a UTF-8 BOM; a first
   naive `utf-8` `json.loads` silently failed and fell back to the default
   symbol list. Fixed by reading `utf-8-sig`.
3. The existing `tools/native_ws_health_policy.py` global liquidations
   check (600s/1800s) and this detector's all-tracked-symbol check
   (3600s/7200s) measure different populations (744-symbol table-wide vs.
   3-symbol tracked-only) and were independently calibrated — the existing
   600s value would already have false-positive-warned inside this
   detector's own combined-healthy population (observed max 2508.9s), which
   is exactly why the task instructed against blind reuse.
4. Both confirmed historical incidents (40-day and 4-day) share the same
   detectable signature — all-tracked-symbol liquidation silence with
   REST-covered control streams (`mark_prices`/`agg_trades`) continuing to
   advance — and this detector classifies both correctly as
   `LIQUIDATION_TRANSPORT_OUTAGE` on real historical data, with zero false
   positives across 310 independent healthy-period samples.

---

## 18. Next Action

`REREVIEW_LIQUIDATION_SILENCE_DETECTOR` (see §19 for the corrective pass that
supersedes the original `REVIEW_...` action).

Independent re-review required before any controlled-activation batch (adding
`"liquidation_silence"` to `tools/heartbeat_watchdog.py`'s
`OPTIONAL_COMPONENT_FILES` and wiring `compose_with_overall_severity` into
its severity composition). No runtime change has occurred. No execution
authorization is implied or requested by this batch.

---

## 19. Corrective Implementation (2026-07-11, Opus 4.8) — post-independent-review

Independent read-only review verdict:
**`LIQUIDATION_SILENCE_DETECTOR_CORRECTIVE_IMPLEMENTATION_REQUIRED`** (three
MEDIUM findings + LOW items). The review changed nothing; this pass
implements the corrections. Frozen thresholds (3600 / 7200 / 9000 / 300) were
**not** changed and the detector was **not** activated, scheduled, or
restarted.

### MEDIUM 1 — Partial-symbol evidence overclaim → COMPLETE-EVIDENCE GATING
An "all tracked symbols silent" claim now requires usable evidence for
**every** tracked symbol. The policy computes `complete_symbol_evidence`,
`missing_symbols`, `known_symbol_count`, `tracked_symbol_count`. When any
tracked symbol's evidence is missing while the known symbols are silent
beyond warning/critical, the classification is the new visibly-uncertain
`PARTIAL_SYMBOL_EVIDENCE` / UNKNOWN (reason `KNOWN_SYMBOL_SILENCE_BEYOND_
WARNING|CRITICAL` + `PARTIAL_SYMBOL_EVIDENCE_MISSING:<syms>`) — never
`LIQUIDATION_TRANSPORT_OUTAGE` and never the `ALL_SYMBOL_SILENCE_BEYOND_*`
reason codes. The exact review reproduction (BTC+ETH critically silent, SOL
missing) is now a regression test → `PARTIAL_SYMBOL_EVIDENCE`, not RED.
Native-WS RED still preserves upstream RED before any liquidation analysis
(item D); isolated known-symbol warnings under complete evidence are retained
(item E). Output explicitly reports tracked symbols, missing symbols,
`complete_symbol_evidence`, and known/expected counts.

### MEDIUM 2 — Fingerprint incompleteness → DECISION-LOGIC FINGERPRINT
`POLICY_FINGERPRINT` is now `sha256` of the full deterministic `POLICY_SPEC`
(see §5) rather than the four thresholds + a version string. Unit tests prove:
same semantics → same fingerprint; threshold / boundary-operator / precedence
/ complete-evidence / native-WS-precedence change → fingerprint changes; key
ordering does not change it; v2 ≠ superseded v1. New value:
`e117cf132bce3bd180af3c718670d3c75910dd69206588d4b7f1b341aadf2291`.

### MEDIUM 3 — Historical replay temporal contamination → EVALUATION MODE
`evaluate_once` / `run_once` take an explicit `evaluation_mode`
(`LIVE` | `HISTORICAL_REPLAY`).
- **LIVE**: `now_ts` must be within `LIVE_MODE_WALLCLOCK_TOLERANCE_SEC`
  (900s) of wall-clock; may read the live default overall/collector/pid
  files. A historical `now_ts` in LIVE mode is rejected (fail-visible
  UNKNOWN, live files never consulted).
- **HISTORICAL_REPLAY**: the live default health/pid files may **not** be
  read (they are present-day snapshots with no historical version) — explicit
  historical evidence paths (or nonexistent/disabled paths) are required, or
  the call fails with `HISTORICAL_EVIDENCE_REQUIRED`. Component timestamps are
  validated against `now_ts`; a component timestamp after `now_ts` beyond
  tolerance is rejected as `CONTROL_COMPONENT_TEMPORAL_MISMATCH`, so a
  present-day `collector.json` cannot influence an April/May replay. All
  replay helpers/tests were updated to this mode. This is a structural API
  safeguard, not a documentation warning.

### ERROR PROPAGATION
DB read helpers now return `(data, errors)` with deterministic codes
(`DB_TABLE_MISSING`, `DB_SCHEMA_MISMATCH`, `DB_LOCKED`, `DB_READ_ERROR`,
`DB_CONNECT_ERROR`, `DB_PERMISSION_DENIED`) surfaced centrally in the
payload `error` field instead of being swallowed to `None`. `error` stays
`null` on a normal healthy evaluation (tested). Output-write failures raise
rather than silently no-op (tested). No secrets / sensitive paths exposed
(only local non-sensitive paths, as before).

### FUTURE-TIMESTAMP HANDLING
Source timestamps in the future within `FUTURE_TIMESTAMP_TOLERANCE_SEC` (60s)
clamp to age 0 (benign skew); beyond tolerance they are recorded as an
evidence anomaly (`FUTURE_LIQUIDATION_TS_ANOMALY` / `FUTURE_CONTROL_TS_
ANOMALY`) and the reading is treated as unusable — never silently clamped to
a healthy age-0. Applied to liquidation, control-stream, and component
timestamps. Exact-boundary tests included.

### SYMBOL NORMALIZATION
`normalize_tracked_symbols` upper-cases / strips / de-duplicates (first-seen
order) / drops empties before evaluation and output; an empty normalized
universe fails visibly (UNKNOWN + `EMPTY_NORMALIZED_UNIVERSE`). The detector's
documented last-resort fallback is retained and still reported via
`symbol_source`. Duplicate-symbol test included.

### TESTS (corrective)
`tests/test_liquidation_silence_policy.py` (48) +
`tests/test_liquidation_silence_detector.py` (38) = **86 passed** (`-p
no:cacheprovider`, `--basetemp` in scratchpad). Canonical health regression
(`test_health_writer_ownership.py` + `test_canonical_health_gate_integration.py`)
= 16 passed. Added coverage: partial-evidence + critical/warning silence,
missing-symbol + native-WS RED, duplicate symbols, empty normalized universe,
future liquidation/control ts within/beyond tolerance, all four evaluation-
mode paths, component-ts-after-now_ts temporal mismatch, present-day-file
non-contamination, missing table / schema mismatch / DB locked / permission-
classification / output-write failure, healthy error-field null, and the six
fingerprint-sensitivity tests.

### REPLAY REVALIDATION (corrected safe HISTORICAL_REPLAY mode)
| Event | Result |
|---|---|
| Outage #1 onset (last good 2026-04-27T14:27:26Z) | first YELLOW 15:30 (latency ≈3753.7s ≈62.6min); first RED 16:30 (≈7353.7s ≈122.6min) |
| 2026-05-03 / 2026-05-15 deep-gap | `LIQUIDATION_TRANSPORT_OUTAGE` / RED (controls fresh) |
| 2026-05-21T16:30 control-stale blip | `CONTROL_STREAMS_STALE` / UNKNOWN (controls ≈15950s stale) — no over-claim |
| Outage #1 recovery (2026-06-06T17:47:05Z) | RED through 17:47, HEALTHY by 17:48 (recovery ≈<1min) |
| Outage #2 onset (last good 2026-07-06T10:06:39Z) | first YELLOW 11:10 (≈3837.6s); first RED 12:10 (≈7437.6s) |
| Outage #2 recovery (2026-07-10T11:24:37Z) | HEALTHY at 11:24 (age 79.6s) |
| Healthy current-architecture sweep (9 probes) | 9/9 HEALTHY, 0 YELLOW/RED/UNKNOWN |
| Excluded older per-symbol regime (7 probes) | 7/7 HEALTHY at these points (portability still unproven — a genuine YELLOW can occur elsewhere in that regime) |

Historical-mode proof: every replay used `evaluation_mode=HISTORICAL_REPLAY`
with nonexistent overall/collector/pid paths → no present-day live component
evidence read; all outage-window classifications had
`complete_symbol_evidence=True`. Thresholds not changed based on results.

### PERFORMANCE / IMMUTABILITY (corrective rehearsal)
One-shot LIVE-mode `run_once` against the real `mode=ro` production DB, output
to an isolated scratch path: wall **0.0043s**, detector_runtime **0.0031s**,
**5** indexed queries, output **1693 bytes**, `error=null`, status HEALTHY.
Never wrote to the real `logs/health/` (verified absent). Frozen historical
row counts (BTC/ETH/SOL liquidations 2026-04-01..03) **687 / 798 / 0**
identical before and after. Live PIDs unchanged (12 python procs, 0 duplicate,
0 detector process, 0 live executor).

### FINAL VERDICT
`LIQUIDATION_SILENCE_DETECTOR_CORRECTED_AWAITING_REREVIEW`. Detector remains
disabled by default, not scheduled, not wired into `heartbeat_watchdog.py`.
Next action: **`REREVIEW_LIQUIDATION_SILENCE_DETECTOR`**. No execution
authorization implied.
