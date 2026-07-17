"""Panel — Paper Bucket Ledger (read-only, per-rule).

Closes the one deliberate hole in the sibling ``shadow_paper_activity`` panel.
That panel reports the paper runner as a SINGLE aggregate bucket and states
outright:

    "win_loss_net_bps_breakdown": "unavailable_without_unbounded_read"

because its only per-trade source, ``S34_SHADOW_PAPER_TRADES.json``, is ~9MB
and growing -- over ``readers.MAX_JSON_BYTES``. That was the right call: a
wholesale re-read of an ever-growing JSON on every dashboard tick is not a
dashboard, it's a memory incident waiting to happen.

This panel gets the same information WITHOUT breaking that guard, by reading
the runner's OWN SQLite sink instead (``data/s34_intelligence.db``, written by
``tools/s34_shadow_paper_runner.py --intelligence-db``). Aggregation happens in
SQL (``GROUP BY rule_name``) so the cost is an indexed scan, not a full
materialisation -- bounded by the number of RULES (~12), never by the number of
trades. `readers.open_ro` enforces `mode=ro`.

WHAT THIS PANEL IS NOT
----------------------
Read-only, like every panel here: no trade/order/cancel/executor/scheduler/
process control, fail-closed on every unreadable source. It reports the paper
runner's own virtual ledger; nothing here reaches a live order path.

THE NON-POOLING RULE (load-bearing, not cosmetic)
-------------------------------------------------
The paper runner's min-gap semantics CHANGED at ``MIN_GAP_V2_BOUNDARY_MS``
(2026-07-10T20:21:50Z, OD-018: `persistent-v2`). Trades either side of that
boundary were generated under DIFFERENT signal-admission rules, so pooling them
into one win-rate is a category error -- it silently mixes two populations and
manufactures a frequency bias (OD-023 fixes the analysis convention as "apply
the boundary as a time filter"). Every stat row here is therefore reported
THREE ways: `all` (for continuity with the raw ledger), and `pre_v2` / `v2`
separately. The operator should read the split, not the pooled number.

N IS TRADES, NOT INDEPENDENT CYCLES
-----------------------------------
`n` counts CLOSED virtual trades. It is NOT an independent-cycle count and
carries NO significance claim. The same cascade episode can produce several
trades (see `SAME_CLUSTER_LOWER_PRIORITY` in the reject reasons), so these
counts are inflated relative to independent evidence exactly as OD-011 describes
for the Knowledge Objects (measured deflation there: 0.598-0.732). Cycle
adjustment for this ledger has NOT been measured. `win_rate` here is a raw
bookkeeping ratio, not evidence of an edge.

DEPRECATED RULES
----------------
``DEPRECATED_PAPER_RULES`` (contaminated lookahead anchor) still have historical
trades in the ledger but can no longer open new positions. They are flagged so
their historical PnL is never read as live-attainable.
"""
from __future__ import annotations

import sqlite3
import time
from typing import Any

from dashboard.backend.readers import open_ro, stat_mtime
from dashboard.backend.sources import DashboardContext
from dashboard.backend.view_models import (
    PanelViewModel,
    ParseStatus,
    ProvenanceStatus,
    Severity,
)

INTEL_DB_REL = ("data", "s34_intelligence.db")

# OD-018: `persistent-v2` min-gap semantics activation (2026-07-10T20:21:50Z).
# Trades either side are NOT poolable. Sourced from the runner's own
# `min_gap_state_initialized_at_utc`; hardcoded here so the panel still splits
# correctly if the state file is absent.
MIN_GAP_V2_BOUNDARY_MS = 1783110110127
MIN_GAP_V2_BOUNDARY_UTC = "2026-07-10T20:21:50Z"

# Mirrors tools/s34_shadow_paper_runner.py:DEPRECATED_PAPER_RULES.
DEPRECATED_RULES = frozenset({
    "ETH_BUY_LIQ_LONG_50K_TP120_SL40_BE30",
    "ETH_BUY_LIQ_LONG_500K_DAYTREND0_TP60_SL40_BE30",
})
DEPRECATED_REASON = "ARCHIVED_CONTAMINATED_LOOKAHEAD_ANCHOR"

RECENT_TRADE_LIMIT = 25
RECENT_WINDOW_SEC = 3600.0


def _stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Raw bookkeeping stats for a set of closed trades. No significance claim."""
    n = len(rows)
    if not n:
        return {"n": 0, "wins": 0, "win_rate": None, "avg_net_bps": None, "total_net_bps": None}
    nets = [float(r["net_bps"]) for r in rows if r.get("net_bps") is not None]
    if not nets:
        return {"n": n, "wins": 0, "win_rate": None, "avg_net_bps": None, "total_net_bps": None,
                "note": "closed trades present but net_bps missing -- not treated as zero"}
    wins = sum(1 for v in nets if v > 0)
    total = sum(nets)
    return {
        "n": n,
        "wins": wins,
        "win_rate": round(100.0 * wins / len(nets), 1),
        "avg_net_bps": round(total / len(nets), 1),
        "total_net_bps": round(total, 1),
    }


def _fetch_closed(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """Per-trade rows needed for the split. Bounded by closed-trade count and
    projected to 5 small columns -- no blobs, no context_json."""
    cur = conn.execute(
        "SELECT o.rule_name AS rule_name, o.net_bps AS net_bps, o.exit_reason AS exit_reason, "
        "       o.exit_ts_ms AS exit_ts_ms, t.entry_ts_ms AS entry_ts_ms, t.symbol AS symbol, "
        "       t.direction AS direction "
        "FROM s34_outcomes o JOIN s34_trades t ON t.trade_id = o.trade_id"
    )
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def _fetch_open(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    cur = conn.execute(
        "SELECT rule_name, symbol, direction, opened_at_utc, entry_ts_ms, entry_price, "
        "       tp_price, sl_price, be_trigger_price "
        "FROM s34_trades WHERE status = 'OPEN' ORDER BY entry_ts_ms DESC"
    )
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def _fetch_reject_reasons(conn: sqlite3.Connection) -> dict[str, dict[str, int]]:
    """GROUP BY in SQL: bounded by (rules x reasons), not by signal count."""
    out: dict[str, dict[str, int]] = {}
    for rule, reason, n in conn.execute(
        "SELECT rule_name, reason, COUNT(*) FROM s34_rejected_signals GROUP BY rule_name, reason"
    ):
        out.setdefault(rule or "?", {})[reason or "?"] = int(n)
    return out


def _fetch_recent(conn: sqlite3.Connection, limit: int = RECENT_TRADE_LIMIT) -> list[dict[str, Any]]:
    """Newest closed trades — the 'watch it live' tail. LIMIT-bounded."""
    cur = conn.execute(
        "SELECT o.exit_ts_ms, o.rule_name, t.symbol, t.direction, o.exit_reason, o.net_bps, "
        "       o.gross_bps, t.entry_price, t.opened_at_utc, o.outcome_ts_utc "
        "FROM s34_outcomes o JOIN s34_trades t ON t.trade_id = o.trade_id "
        "ORDER BY o.exit_ts_ms DESC LIMIT ?",
        (int(limit),),
    )
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def _split_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    pre = [r for r in rows if (r.get("entry_ts_ms") or 0) < MIN_GAP_V2_BOUNDARY_MS]
    post = [r for r in rows if (r.get("entry_ts_ms") or 0) >= MIN_GAP_V2_BOUNDARY_MS]
    return {"all": _stats(rows), "pre_v2": _stats(pre), "v2": _stats(post)}


def build(ctx: DashboardContext) -> PanelViewModel:
    now = ctx.now if ctx.now is not None else time.time()
    db_path = ctx.p(*INTEL_DB_REL)
    mtime = stat_mtime(db_path)

    vm = PanelViewModel(
        key="paper_bucket_ledger",
        title="Paper Bucket Ledger (per-rule)",
        source_path=str(db_path),
        read_timestamp=now,
        source_timestamp=mtime,
        freshness_threshold_seconds=ctx.threshold("paper_bucket_ledger"),
        provenance_status=ProvenanceStatus.DERIVED,
    )

    if not db_path.exists():
        vm.severity = Severity.MEDIUM
        vm.status = "UNAVAILABLE"
        vm.add_reason("intelligence_db_missing")
        vm.fields = {"parse_status": ParseStatus.MISSING.value,
                     "note": "intelligence DB absent -- NOT evidence of zero trades"}
        vm.rows = []
        vm.summary = "paper bucket ledger unavailable (intelligence DB missing)"
        return vm

    try:
        conn = open_ro(db_path)
        try:
            closed = _fetch_closed(conn)
            open_rows = _fetch_open(conn)
            rejects = _fetch_reject_reasons(conn)
            recent = _fetch_recent(conn)
        finally:
            conn.close()
    except Exception as exc:  # fail-closed: never render a partial ledger as complete
        vm.severity = Severity.HIGH
        vm.status = "ERROR"
        vm.add_reason("intelligence_db_unreadable")
        vm.fields = {"parse_status": ParseStatus.MALFORMED.value, "error": str(exc)[:200],
                     "note": "ledger unreadable -- fail-closed, NOT reported as zero trades"}
        vm.rows = []
        vm.summary = "paper bucket ledger unreadable (fail-closed)"
        return vm

    by_rule: dict[str, list[dict[str, Any]]] = {}
    for r in closed:
        by_rule.setdefault(r.get("rule_name") or "?", []).append(r)
    open_by_rule: dict[str, int] = {}
    for r in open_rows:
        open_by_rule[r.get("rule_name") or "?"] = open_by_rule.get(r.get("rule_name") or "?", 0) + 1

    rows: list[dict[str, Any]] = []
    for rule in sorted(set(by_rule) | set(open_by_rule) | set(rejects)):
        trades = by_rule.get(rule, [])
        split = _split_stats(trades)
        exits: dict[str, int] = {}
        for t in trades:
            exits[t.get("exit_reason") or "?"] = exits.get(t.get("exit_reason") or "?", 0) + 1
        last_exit = max((t.get("exit_ts_ms") or 0) for t in trades) if trades else None
        sample = trades[0] if trades else None
        rows.append({
            "bucket": rule,
            "symbol": (sample or {}).get("symbol"),
            "direction": (sample or {}).get("direction"),
            "deprecated": rule in DEPRECATED_RULES,
            "deprecated_reason": DEPRECATED_REASON if rule in DEPRECATED_RULES else None,
            "open_now": open_by_rule.get(rule, 0),
            **{f"{k}_{sk}": sv for k, v in split.items() for sk, sv in v.items()},
            "stats": split,
            "exit_reasons": exits,
            "reject_reasons": rejects.get(rule, {}),
            "last_exit_ts_ms": last_exit,
            "last_exit_age_sec": round(now - last_exit / 1000.0, 1) if last_exit else None,
        })
    rows.sort(key=lambda r: (-(r["stats"]["all"]["n"] or 0), r["bucket"]))

    overall = _split_stats(closed)
    live_rows = [r for r in rows if not r["deprecated"]]
    live_closed = [t for t in closed if (t.get("rule_name") or "") not in DEPRECATED_RULES]
    recent_closes = sum(
        1 for t in closed
        if t.get("exit_ts_ms") and (now - t["exit_ts_ms"] / 1000.0) <= RECENT_WINDOW_SEC
    )

    if open_rows:
        vm.severity = Severity.INFO
        vm.add_reason("virtual_position_open")
    else:
        vm.severity = Severity.OK

    vm.fields = {
        "parse_status": ParseStatus.OK.value,
        "bucket_count": len(rows),
        "deprecated_bucket_count": sum(1 for r in rows if r["deprecated"]),
        "open_virtual_positions": len(open_rows),
        "open_positions": open_rows,
        "closes_last_hour": recent_closes,
        "recent_closed_trades": recent,
        "overall": overall,
        "overall_excluding_deprecated": _split_stats(live_closed),
        "min_gap_v2_boundary_utc": MIN_GAP_V2_BOUNDARY_UTC,
        "min_gap_v2_boundary_ms": MIN_GAP_V2_BOUNDARY_MS,
        "pooling_warning": (
            "pre_v2 and v2 were generated under DIFFERENT min-gap admission semantics "
            "(OD-018/OD-023) and MUST NOT be pooled; read the split, not `all`"
        ),
        "n_semantics": (
            "n = CLOSED virtual trades, NOT independent cycles; no significance claim. "
            "Same cascade can yield several trades (see SAME_CLUSTER_LOWER_PRIORITY), so n is "
            "inflated vs independent evidence (cf. OD-011 deflation 0.598-0.732 on the KO side, "
            "NOT measured for this ledger)"
        ),
        "deprecated_note": (
            f"{DEPRECATED_REASON}: these buckets keep historical trades but cannot open new "
            "positions; their PnL is NOT live-attainable"
        ),
        "no_control_actions_available": True,
    }
    vm.rows = rows
    vm.status = "ACTIVE"
    a = overall["all"]
    live = vm.fields["overall_excluding_deprecated"]["all"]
    vm.summary = (
        f"buckets={len(rows)} (live={len(live_rows)}) closed={a['n']} wr={a['win_rate']}% "
        f"total={a['total_net_bps']}bps | ex-deprecated: n={live['n']} wr={live['win_rate']}% "
        f"total={live['total_net_bps']}bps | open={len(open_rows)} closes_1h={recent_closes} "
        f"(pre_v2/v2 NOT pooled)"
    )
    return vm
