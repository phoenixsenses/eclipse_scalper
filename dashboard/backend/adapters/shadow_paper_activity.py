"""Panel — Shadow/Paper Activity (read-only).

Gives the operator a per-runner, per-bucket operational-state read so a
healthy-but-idle runner (no eligible signal this cycle) is never confused
with a dead one, and a genuinely missing evaluation cycle is never confused
with "quiet market, nothing to trade".

Every bucket is classified into exactly one of eight mutually-exclusive
states (see ``STATE_*`` below). The classification is intentionally a
precedence chain (first match wins) rather than independent flags, so a
panel always has one unambiguous answer to "what is this bucket doing right
now" instead of a soup of booleans.

Three runners are covered, each with a different natural granularity:

- ``s34_shadow_paper_runner``: one regime-gated strategy. Its full per-trade
  history (``S34_SHADOW_PAPER_TRADES.json``) is already at/over this
  package's wholesale-JSON size guard (see ``readers.MAX_JSON_BYTES``) and
  keeps growing, so this adapter deliberately does NOT parse it; only the
  small, already-bounded ``S34_SHADOW_PAPER_STATUS.json`` aggregate is read.
  Per-signal win/loss/net-bps breakdown is reported as unavailable rather
  than fabricated or produced by an unbounded read.
- ``s34_v_engine_v02_shadow_mirror``: one protocol/bucket by construction.
- ``s34_realtime_shadow_runner`` (state machine shadow): genuinely
  multi-bucket — the state artifact already carries a per-``signal``-name
  pnl breakdown (n/wins/wr/avg_net/total_net), which this adapter surfaces
  directly rather than recomputing from the ledger.

Evidence gaps (see :func:`detect_evidence_gaps`) are computed ONLY from a
source with genuinely uniform, signal-independent cadence (a runner's own
per-cycle stdout log, which writes one line every tick regardless of
whether anything was found) — never from ledger/trade timestamps. A
low-frequency detector's real signals can legitimately be many hours apart
(e.g. an hour-of-day rule that fires once a day); treating those natural
gaps as "evidence of an outage" would be a false-positive firehose, not a
useful signal, and was caught and removed during this panel's own
implementation (see the module's git history / review notes). Per-cycle
stdout logs are truncated on every process restart, so this only detects
stalls *within the current process's lifetime* — it cannot see through a
restart. Confirmed historical outages (like a real multi-day process
death) are instead recorded once, explicitly, in the bounded
``runtime/dashboard_incident_log.json`` record (see
:func:`read_incident_log`) and surfaced separately, never re-derived from
noisy event timestamps.
"""
from __future__ import annotations

import time
from typing import Any

from dashboard.backend.process_identity import scan_topology
from dashboard.backend.readers import iter_jsonl_tail, load_json, parse_iso
from dashboard.backend.sources import DashboardContext
from dashboard.backend.view_models import (
    ParseStatus,
    PanelViewModel,
    ProvenanceStatus,
    Severity,
    TrustState,
)
from dashboard.backend.adapters import native_ws

# --- the eight mutually-exclusive bucket states -----------------------------
STATE_DATA_STALE = "DATA_STALE"
STATE_PROCESS_DEAD = "PROCESS_DEAD"
STATE_EVALUATION_STALE = "EVALUATION_STALE"
STATE_NO_ELIGIBLE_SIGNAL = "NO_ELIGIBLE_SIGNAL"
STATE_SIGNAL_SEEN_NO_ENTRY = "SIGNAL_SEEN_NO_ENTRY"
STATE_VIRTUAL_POSITION_OPEN = "VIRTUAL_POSITION_OPEN"
STATE_VIRTUAL_TRADE_CLOSED = "VIRTUAL_TRADE_CLOSED"
STATE_ERROR = "ERROR"

ALL_STATES = (
    STATE_DATA_STALE, STATE_PROCESS_DEAD, STATE_EVALUATION_STALE,
    STATE_NO_ELIGIBLE_SIGNAL, STATE_SIGNAL_SEEN_NO_ENTRY,
    STATE_VIRTUAL_POSITION_OPEN, STATE_VIRTUAL_TRADE_CLOSED, STATE_ERROR,
)

# A bucket state that is itself evidence of unhealthy operation (used to
# derive panel severity from the worst bucket present).
_UNHEALTHY_STATES = frozenset({STATE_DATA_STALE, STATE_PROCESS_DEAD, STATE_EVALUATION_STALE, STATE_ERROR})

RECENT_CLOSE_WINDOW_SEC = 900.0

STATUS_REL = ("reports", "research", "s34", "S34_SHADOW_PAPER_STATUS.json")
V02_STATE_REL = ("runtime", "s34_v_engine_v02_shadow_mirror_state.json")
V02_LEDGER_REL = ("reports", "research", "s34", "S34_V_ENGINE_V0_2_SHADOW_MIRROR_LEDGER.jsonl")
SM_STATE_REL = ("reports", "shadow", "s34_state_machine_shadow_state.json")
SM_LEDGER_REL = ("reports", "shadow", "s34_state_machine_shadow.jsonl")
SHADOW_PAPER_LOG_REL = ("logs", "s34_shadow_paper_runner.stdout.log")
V02_MIRROR_LOG_REL = ("logs", "s34_v_engine_v02_shadow_mirror.out.log")
INCIDENT_LOG_REL = ("runtime", "dashboard_incident_log.json")

# Bounded tail read for per-cycle stdout logs (uniform-cadence source used
# for gap detection). These logs are truncated on every process restart, so
# this bound is generous relative to realistic single-lifetime log sizes.
MAX_LOG_TAIL_BYTES = 4 * 1024 * 1024

_LEADING_ISO_TS_RE = None  # compiled lazily to keep import-time cost near zero


def _tail_log_timestamps(path, *, max_bytes: int = MAX_LOG_TAIL_BYTES) -> list[float]:
    """Bounded tail-read of a per-cycle stdout log; parse each line's leading
    ISO-8601 timestamp (the convention used by every per-cycle log this
    adapter reads). Malformed/non-matching lines are skipped, not fatal.
    Returns an empty list if the file is absent -- callers must not treat
    that as "no gaps found", only as "no data available"."""
    global _LEADING_ISO_TS_RE
    if _LEADING_ISO_TS_RE is None:
        import re
        _LEADING_ISO_TS_RE = re.compile(r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:\+00:00|Z))")
    if not path.exists():
        return []
    try:
        size = path.stat().st_size
    except OSError:
        return []
    out: list[float] = []
    try:
        with path.open("rb") as f:
            if size > max_bytes:
                f.seek(size - max_bytes)
                f.readline()  # discard partial first line
            raw = f.read()
    except OSError:
        return []
    for line in raw.decode("utf-8", errors="replace").splitlines():
        m = _LEADING_ISO_TS_RE.match(line)
        if not m:
            # Also try a leading JSON object's own updated_at_utc key, which
            # is the shadow_paper_runner log's actual per-line format.
            ts = parse_iso(_extract_json_field(line, "updated_at_utc"))
            if ts is not None:
                out.append(ts)
            continue
        ts = parse_iso(m.group(1))
        if ts is not None:
            out.append(ts)
    return out


def _extract_json_field(line: str, key: str) -> str | None:
    """Best-effort extraction of one string field from a JSON line without a
    full parse of every line (cheap pre-filter); falls back to a real
    ``json.loads`` only on lines that look like JSON objects."""
    if not line.strip().startswith("{"):
        return None
    try:
        import json as _json
        obj = _json.loads(line)
        val = obj.get(key) if isinstance(obj, dict) else None
        return str(val) if val is not None else None
    except Exception:
        return None


def read_incident_log(ctx: DashboardContext) -> dict[str, Any]:
    """Read the small, explicit, hand/tool-maintained incident record. This
    is the ONLY source of confirmed historical (cross-restart) outages --
    never re-derived from noisy per-event timestamps. A missing file is
    reported honestly as "no incident log found", not as "zero incidents
    ever happened"."""
    path = ctx.p(*INCIDENT_LOG_REL)
    data, ps, mtime = load_json(path)
    if ps == ParseStatus.MISSING:
        return {"parse_status": ps.value, "source_path": str(path), "incidents": [],
                "note": "no incident log file found -- this is NOT evidence that no incident occurred"}
    if ps != ParseStatus.OK or not isinstance(data, dict):
        return {"parse_status": ps.value, "source_path": str(path), "incidents": [],
                "note": "incident log unparseable (fail-closed)"}
    incidents = data.get("incidents")
    return {
        "parse_status": ps.value,
        "source_path": str(path),
        "source_mtime": mtime,
        "incidents": incidents if isinstance(incidents, list) else [],
    }


def classify_bucket_state(
    *,
    process_alive: bool,
    market_data_stale: bool = False,
    artifact_parse_status: ParseStatus = ParseStatus.OK,
    artifact_age_sec: float | None = None,
    artifact_threshold_sec: float | None = None,
    error_present: bool = False,
    has_open_position: bool = False,
    last_event_kind: str | None = None,  # "ENTRY" | "CLOSE" | "REJECT" | None
    last_event_age_sec: float | None = None,
    recent_close_window_sec: float = RECENT_CLOSE_WINDOW_SEC,
) -> str:
    """Classify one bucket/runner into exactly one of the eight states.

    Precedence (first match wins): a reported error always wins (it is the
    most specific, most actionable signal); then process liveness (nothing
    else can be trusted if the process itself is gone); then whether we can
    even see fresh output from it at all (artifact trust); then whether the
    market data it depends on is stale; only once all of that is healthy do
    we look at what the runner actually reported doing this cycle.
    """
    if error_present:
        return STATE_ERROR
    if not process_alive:
        return STATE_PROCESS_DEAD
    if artifact_parse_status in (ParseStatus.MISSING, ParseStatus.MALFORMED):
        # Process is alive but we cannot see what it is doing -- this is a
        # failure of observability into evaluation, not of the market data.
        return STATE_EVALUATION_STALE
    if (
        artifact_threshold_sec is not None
        and artifact_age_sec is not None
        and artifact_age_sec > artifact_threshold_sec
    ):
        return STATE_EVALUATION_STALE
    if market_data_stale:
        return STATE_DATA_STALE
    if has_open_position:
        return STATE_VIRTUAL_POSITION_OPEN
    if last_event_kind == "CLOSE":
        if last_event_age_sec is not None and last_event_age_sec <= recent_close_window_sec:
            return STATE_VIRTUAL_TRADE_CLOSED
        # An old close with nothing since is an idle runner, same as never
        # having traded -- fall through to NO_ELIGIBLE_SIGNAL.
    elif last_event_kind == "REJECT":
        return STATE_SIGNAL_SEEN_NO_ENTRY
    return STATE_NO_ELIGIBLE_SIGNAL


def detect_evidence_gaps(
    timestamps_sec: list[float],
    *,
    gap_threshold_sec: float,
    now: float | None = None,
) -> list[dict[str, Any]]:
    """Find gaps > ``gap_threshold_sec`` in a sorted-or-unsorted timestamp
    sequence. Each gap is reported as an explicit evidence-gap window rather
    than silently read as "no activity" -- the caller should label the
    surrounding period as evidence-gap, not zero-signal.

    A trailing gap between the last known timestamp and ``now`` is also
    reported (an ongoing/just-ended gap), so a currently-stale runner shows
    its gap immediately rather than only after its next event.
    """
    ts = sorted(float(t) for t in timestamps_sec if t is not None)
    gaps: list[dict[str, Any]] = []
    for prev, cur in zip(ts, ts[1:]):
        delta = cur - prev
        if delta > gap_threshold_sec:
            gaps.append({
                "gap_start_epoch": prev,
                "gap_end_epoch": cur,
                "duration_sec": round(delta, 1),
                "ongoing": False,
            })
    if now is not None and ts:
        trailing = now - ts[-1]
        if trailing > gap_threshold_sec:
            gaps.append({
                "gap_start_epoch": ts[-1],
                "gap_end_epoch": now,
                "duration_sec": round(trailing, 1),
                "ongoing": True,
            })
    return gaps


def _market_data_stale(ctx: DashboardContext) -> bool:
    try:
        return bool(native_ws.build(ctx).fields.get("degraded"))
    except Exception:
        # Fail CLOSED (treat as stale), consistent with this package's
        # fail-closed convention elsewhere: an inability to determine market
        # data freshness must never be silently read as "definitely fresh".
        return True


def _shadow_paper_runner_section(ctx: DashboardContext, *, process_alive: bool, market_data_stale: bool) -> dict[str, Any]:
    now = ctx.now if ctx.now is not None else time.time()
    path = ctx.p(*STATUS_REL)
    data, ps, mtime = load_json(path)
    status = data if isinstance(data, dict) else {}
    updated_ts = parse_iso(status.get("updated_at_utc")) or mtime
    age = (now - updated_ts) if updated_ts is not None else None
    threshold = ctx.threshold("shadow_paper_runner")

    open_trades = status.get("open_trades")
    new_trades = status.get("new_trades")
    risk_skipped = status.get("risk_skipped_trades")
    last_event_kind = None
    if isinstance(new_trades, (int, float)) and new_trades > 0:
        last_event_kind = "ENTRY"
    elif isinstance(risk_skipped, (int, float)) and risk_skipped > 0:
        last_event_kind = "REJECT"

    state = classify_bucket_state(
        process_alive=process_alive,
        market_data_stale=market_data_stale,
        artifact_parse_status=ps,
        artifact_age_sec=age,
        artifact_threshold_sec=threshold,
        has_open_position=bool(open_trades),
        last_event_kind=last_event_kind,
        last_event_age_sec=age,
    )
    tail_ticks = _tail_log_timestamps(ctx.p(*SHADOW_PAPER_LOG_REL))
    tail_gaps = detect_evidence_gaps(
        tail_ticks, gap_threshold_sec=(threshold or 180.0) * 3.0, now=now if process_alive else None,
    )
    return {
        "runner": "s34_shadow_paper_runner",
        "granularity": "aggregate_single_strategy",
        "bucket_count": 1,
        "buckets": [{
            "bucket": "regime_gated_default",
            "direction": "BOTH",
            "state": state,
            "enabled": bool((status.get("regime_config") or {}).get("enabled", True)),
            "regime_config": status.get("regime_config"),
            "risk_config": status.get("risk_config"),
            "fill_model": status.get("fill_model"),
            "last_evaluation_utc": status.get("updated_at_utc"),
            "last_evaluation_age_sec": round(age, 1) if age is not None else None,
            "open_trades": open_trades,
            "closed_trades_all_time": status.get("closed_trades"),
            "total_trades_all_time": status.get("total_trades"),
            "new_trades_this_cycle": new_trades,
            "risk_skipped_trades_all_time": risk_skipped,
            "win_loss_net_bps_breakdown": "unavailable_without_unbounded_read",
            "win_loss_net_bps_reason": (
                f"source {STATUS_REL[-1].replace('STATUS', 'TRADES')} exceeds this dashboard's "
                "bounded-JSON-read guard and grows continuously; not parsed here to avoid a "
                "repeated wholesale read of a multi-MB, ever-growing file"
            ),
        }],
        "parse_status": ps.value,
        "source_path": str(path),
        "evidence_gaps": tail_gaps,
        "log_data_available": bool(tail_ticks),
        "evidence_gap_reason": (
            f"gaps in the per-cycle stdout log (one line per evaluation cycle regardless of "
            f"signal outcome; {len(tail_ticks)} ticks in tail read -- 0 means no log data "
            "available, NOT a confirmed zero-gap history); truncated on every process restart, "
            "so only detects stalls within the current process lifetime -- see the incident log "
            "for confirmed cross-restart outages"
        ),
    }


def _v02_mirror_section(ctx: DashboardContext, *, process_alive: bool, market_data_stale: bool) -> dict[str, Any]:
    now = ctx.now if ctx.now is not None else time.time()
    state_path = ctx.p(*V02_STATE_REL)
    ledger_path = ctx.p(*V02_LEDGER_REL)
    data, ps, mtime = load_json(state_path)
    state = data if isinstance(data, dict) else {}
    updated_ts = parse_iso(state.get("updated_at_utc")) or mtime
    age = (now - updated_ts) if updated_ts is not None else None
    interval_sec = state.get("interval_sec")
    threshold = (float(interval_sec) * 3.0) if isinstance(interval_sec, (int, float)) else ctx.threshold("v02_mirror_bucket")

    rows, lps, _lmt = iter_jsonl_tail(ledger_path)
    row_timestamps = [r.get("book_ts_ms", 0) / 1000.0 for r in rows if isinstance(r.get("book_ts_ms"), (int, float))]
    last_row_ts = max(row_timestamps) if row_timestamps else None
    last_event_age = (now - last_row_ts) if last_row_ts is not None else None

    # Gap detection uses the per-cycle stdout log (ticks every interval_sec
    # regardless of whether a candidate row was added) -- NOT the ledger,
    # whose rows only appear when a candidate qualifies and are therefore
    # naturally sparse; using ledger gaps here would misreport quiet markets
    # as outages.
    gap_threshold = (float(interval_sec) * 3.0) if isinstance(interval_sec, (int, float)) else 540.0
    tick_timestamps = _tail_log_timestamps(ctx.p(*V02_MIRROR_LOG_REL))
    evidence_gaps = detect_evidence_gaps(tick_timestamps, gap_threshold_sec=gap_threshold, now=now if process_alive else None)

    bucket_state = classify_bucket_state(
        process_alive=process_alive,
        market_data_stale=market_data_stale,
        artifact_parse_status=ps,
        artifact_age_sec=age,
        artifact_threshold_sec=threshold,
        error_present=bool(state.get("last_error")),
        has_open_position=False,  # observer mirror; no persisted position state
        last_event_kind="ENTRY" if rows else None,
        last_event_age_sec=last_event_age,
    )
    return {
        "runner": "s34_v_engine_v02_shadow_mirror",
        "granularity": "single_protocol_bucket",
        "bucket_count": 1,
        "buckets": [{
            "bucket": state.get("protocol_id"),
            "direction": "inferred_from_protocol_id",
            "state": bucket_state,
            "enabled": True,
            "interval_sec": interval_sec,
            "last_evaluation_utc": state.get("updated_at_utc"),
            "last_evaluation_age_sec": round(age, 1) if age is not None else None,
            "last_success_utc": state.get("last_success_utc"),
            "last_error": state.get("last_error"),
            "ledger_rows_total": state.get("ledger_rows"),
            "rows_added_this_run": state.get("rows_added_this_run"),
            "checkpoint_lag_ms": (state.get("tick_stats") or {}).get("checkpoint_lag_ms"),
            "sample_count_in_ledger_tail": len(rows),
        }],
        "parse_status": ps.value,
        "source_path": str(state_path),
        "evidence_gaps": evidence_gaps,
        "log_data_available": bool(tick_timestamps),
        "evidence_gap_reason": (
            f"gaps > {gap_threshold:.0f}s between consecutive per-cycle stdout-log ticks "
            f"(uniform cadence regardless of signal outcome; {len(tick_timestamps)} ticks in tail read "
            "-- 0 means no log data available, NOT a confirmed zero-gap history); "
            "truncated on restart -- see incident log for confirmed cross-restart outages"
        ),
    }


def _state_machine_shadow_section(ctx: DashboardContext, *, process_alive: bool, market_data_stale: bool) -> dict[str, Any]:
    now = ctx.now if ctx.now is not None else time.time()
    state_path = ctx.p(*SM_STATE_REL)
    ledger_path = ctx.p(*SM_LEDGER_REL)
    data, ps, mtime = load_json(state_path)
    state = data if isinstance(data, dict) else {}
    updated_ts = parse_iso(state.get("updated_utc")) or mtime
    age = (now - updated_ts) if updated_ts is not None else None
    threshold = ctx.threshold("shadow_ledger")

    positions = state.get("positions") if isinstance(state.get("positions"), dict) else {}
    pnl = state.get("pnl") if isinstance(state.get("pnl"), dict) else {}

    rows, lps, _lmt = iter_jsonl_tail(ledger_path)

    # No gap detection here: this runner is genuinely event-driven (some
    # buckets legitimately fire once a day or less), and its ledger/log only
    # record real trading signals -- there is no uniform, signal-independent
    # per-cycle tick to measure gaps against. Using sparse signal timestamps
    # for gap detection would misreport a low-frequency bucket's normal quiet
    # hours as an outage. Evaluation health for this runner comes from
    # process-alive + state-artifact-freshness (EVALUATION_STALE) instead.
    evidence_gaps: list[dict[str, Any]] = []

    # Most-recent close per signal/bucket name, from the bounded ledger tail.
    last_close_by_bucket: dict[str, dict[str, Any]] = {}
    for r in rows:
        if str(r.get("event")) != "CLOSE":
            continue
        bucket = r.get("signal")
        if not bucket:
            continue
        closed_ts = parse_iso(r.get("closed_utc"))
        prev = last_close_by_bucket.get(bucket)
        if prev is None or (closed_ts or 0) >= (prev.get("_ts") or 0):
            last_close_by_bucket[bucket] = {
                "close_reason": r.get("close_reason"),
                "net_bps": r.get("net_bps"),
                "closed_utc": r.get("closed_utc"),
                "_ts": closed_ts or 0,
            }

    open_bucket_names = {
        (p.get("signal") or p.get("rule_name") or pid)
        for pid, p in positions.items() if isinstance(p, dict)
    }

    # "updated_utc" is not a signal bucket -- the producer
    # (tools/s34_realtime_shadow_runner.py) stores its own refresh timestamp
    # as a sibling key inside the same `pnl` dict that holds real per-signal
    # stat rows, and its own print_status() explicitly skips this key for
    # the same reason. Without this filter a phantom "updated_utc" bucket
    # would appear in the operator-facing table.
    pnl_bucket_names = set(pnl.keys()) - {"updated_utc"}
    bucket_names = sorted(pnl_bucket_names | open_bucket_names | set(last_close_by_bucket.keys()))
    buckets = []
    for name in bucket_names:
        pnl_row = pnl.get(name) if isinstance(pnl.get(name), dict) else {}
        last_close = last_close_by_bucket.get(name)
        has_open = name in open_bucket_names
        last_event_kind = "CLOSE" if last_close else None
        last_event_age = (now - last_close["_ts"]) if last_close and last_close.get("_ts") else None
        bucket_state = classify_bucket_state(
            process_alive=process_alive,
            market_data_stale=market_data_stale,
            artifact_parse_status=ps,
            artifact_age_sec=age,
            artifact_threshold_sec=threshold,
            has_open_position=has_open,
            last_event_kind=last_event_kind,
            last_event_age_sec=last_event_age,
        )
        buckets.append({
            "bucket": name,
            "state": bucket_state,
            "currently_open": has_open,
            "sample_count": pnl_row.get("n"),
            "wins": pnl_row.get("wins"),
            "win_rate": pnl_row.get("wr"),
            "avg_net_bps": pnl_row.get("avg_net"),
            "total_net_bps": pnl_row.get("total_net"),
            "last_virtual_close": {k: v for k, v in (last_close or {}).items() if k != "_ts"} or None,
        })

    # Overall panel-level state (worst bucket, else process/artifact-level fallback).
    if buckets:
        # Same precedence order as classify_bucket_state itself (ERROR >
        # PROCESS_DEAD > EVALUATION_STALE > DATA_STALE) so this summary
        # field never disagrees with the per-bucket classifier about which
        # condition is more severe.
        overall_state = next(
            (s for s in (STATE_ERROR, STATE_PROCESS_DEAD, STATE_EVALUATION_STALE, STATE_DATA_STALE)
             if any(b["state"] == s for b in buckets)),
            None,
        ) or buckets[0]["state"]
    else:
        overall_state = classify_bucket_state(
            process_alive=process_alive, market_data_stale=market_data_stale,
            artifact_parse_status=ps, artifact_age_sec=age, artifact_threshold_sec=threshold,
        )

    return {
        "runner": "s34_state_machine_shadow_runner",
        "granularity": "per_signal_bucket",
        "bucket_count": len(buckets),
        "overall_state": overall_state,
        "buckets": buckets,
        "parse_status": ps.value,
        "source_path": str(state_path),
        "evidence_gaps": evidence_gaps,
        "evidence_gap_reason": (
            "not computed for this runner: event-driven, signal-only ledger with no uniform "
            "per-cycle tick -- a low-frequency bucket's normal multi-hour quiet period would be "
            "indistinguishable from an outage using event timestamps alone. See the incident log "
            "for confirmed historical outages, and EVALUATION_STALE bucket states for current health."
        ),
    }


def build(ctx: DashboardContext) -> PanelViewModel:
    now = ctx.now if ctx.now is not None else time.time()
    topo = scan_topology(ctx.proc_snapshot)
    roles = {r["key"]: r for r in topo["roles"]}
    market_stale = _market_data_stale(ctx)

    sections = [
        _shadow_paper_runner_section(
            ctx,
            process_alive=(roles.get("s34_shadow_paper_runner", {}).get("actual_count", 0) > 0),
            market_data_stale=market_stale,
        ),
        _v02_mirror_section(
            ctx,
            process_alive=(roles.get("s34_observer", {}).get("actual_count", 0) > 0),
            market_data_stale=market_stale,
        ),
        _state_machine_shadow_section(
            ctx,
            process_alive=(roles.get("s34_realtime_shadow_runner", {}).get("actual_count", 0) > 0),
            market_data_stale=market_stale,
        ),
    ]

    vm = PanelViewModel(
        key="shadow_paper_activity",
        title="Shadow/Paper Activity",
        source_path="aggregate:" + ",".join(s["source_path"] for s in sections),
        read_timestamp=now,
        # This panel is itself a live scan (like process_topology/
        # executive_header): "freshness" at the panel level means "was this
        # overview just generated", which is always true by construction
        # (source_timestamp == read_timestamp). Per-bucket/per-runner
        # staleness is a SEPARATE, already-reported signal (EVALUATION_STALE/
        # DATA_STALE bucket states + vm.severity below) -- conflating the two
        # by leaving this None previously made the panel permanently render
        # as non-current/degraded even when every bucket was perfectly
        # healthy, which defeated the panel's purpose.
        source_timestamp=now,
        freshness_threshold_seconds=ctx.threshold("shadow_paper_activity"),
        provenance_status=ProvenanceStatus.DERIVED,
    )

    incident_log = read_incident_log(ctx)

    state_counts: dict[str, int] = {s: 0 for s in ALL_STATES}
    current_evidence_gaps = 0
    for sec in sections:
        for b in sec["buckets"]:
            st = b.get("state")
            if st in state_counts:
                state_counts[st] += 1
        current_evidence_gaps += len(sec.get("evidence_gaps") or [])
    # Deliberately NOT summed together: a currently-open gap (something is
    # wrong right now) and a permanently-recorded historical incident
    # (something happened once and was recovered) are different questions --
    # blending them into one count would make the panel read as "something
    # is wrong" forever after the first-ever recorded incident.
    confirmed_historical_incident_count = len(incident_log.get("incidents") or [])

    unhealthy_bucket_count = sum(state_counts[s] for s in _UNHEALTHY_STATES)
    if state_counts[STATE_PROCESS_DEAD] > 0 or state_counts[STATE_ERROR] > 0:
        vm.severity = Severity.HIGH
        vm.add_reason("bucket_process_dead_or_error")
    elif state_counts[STATE_DATA_STALE] > 0 or state_counts[STATE_EVALUATION_STALE] > 0:
        vm.severity = Severity.MEDIUM
        vm.add_reason("bucket_stale")
    else:
        vm.severity = Severity.OK

    if current_evidence_gaps > 0:
        vm.add_reason("current_evidence_gap_present")
    if confirmed_historical_incident_count > 0:
        vm.add_reason("historical_incident_on_record")

    vm.fields = {
        "runners": {s["runner"]: {k: v for k, v in s.items() if k != "buckets"} for s in sections},
        "state_counts": state_counts,
        "unhealthy_bucket_count": unhealthy_bucket_count,
        "current_evidence_gap_count": current_evidence_gaps,
        "confirmed_historical_incident_count": confirmed_historical_incident_count,
        "incident_log": incident_log,
        "market_data_stale": market_stale,
        "no_control_actions_available": True,
    }
    vm.rows = [
        {"runner": s["runner"], **b}
        for s in sections
        for b in s["buckets"]
    ]
    vm.status = "ACTIVE"
    vm.summary = (
        f"buckets={sum(s['bucket_count'] for s in sections)} unhealthy={unhealthy_bucket_count} "
        f"current_gaps={current_evidence_gaps} historical_incidents={confirmed_historical_incident_count} "
        f"market_data_stale={market_stale}"
    )
    return vm
