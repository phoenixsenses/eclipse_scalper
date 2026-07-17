"""Pure analysis of the S34 state machine shadow ledger.

No I/O beyond the caller-supplied rows, no HTTP, no rendering: adapters own
presentation, this module owns the arithmetic. Three invariants drive it, each
of which the raw artifacts violate if read naively:

  1. **Backfilled closes are simulation, not forward evidence.** The runner's
     ``state.json["pnl"]`` block sums both populations into one number, so any
     panel that surfaces it verbatim reports simulated PnL as observed PnL.
     Recompute from the ledger and keep the two apart.
  2. **A win rate over a handful of closes is not a win rate.** Routes under
     ``MIN_EVIDENCE_N`` report INSUFFICIENT_N rather than a number that reads
     as a decision.
  3. **The ledger alone overstates exposure.** A lifecycle with no CLOSE/EXPIRE
     looks live, but the runner is the authority on what it tracks
     (``state.json["positions"]``, keyed by the same id). An id the runner has
     dropped without writing a terminal event is an orphan, not a position.

See SYSTEM_STATE §139/§140. Route keys are never truncated here: the producer
writes state.json with sort_keys=True, so any head-slice of its keys drops
whole SHORT_* routes alphabetically rather than at random.
"""

from __future__ import annotations

import collections
from typing import Any

# Fee the runner applies when deriving net_bps from outcome_bps.
FEE_BPS = 5.0

# Below this many forward closes a route reports INSUFFICIENT_N.
MIN_EVIDENCE_N = 10

TERMINAL_EVENTS = frozenset({"CLOSE", "EXPIRE"})

# Statuses that carry no position: the runner is waiting on a trigger.
NO_EXPOSURE_STATUS_PREFIXES = ("MONITORING",)

UNKNOWN = "UNKNOWN"


def is_backfill(row: dict[str, Any]) -> bool:
    return bool(row.get("backfill"))


def has_exposure(status: Any) -> bool:
    """True when the status carries an actual (virtual) position."""
    if not isinstance(status, str):
        return False
    return not status.startswith(NO_EXPOSURE_STATUS_PREFIXES)


def build_lifecycles(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Fold the event stream into one lifecycle per position id.

    Ledger order is append order, so the last event seen for an id is its
    current truth. Rows without an id cannot be attributed and are skipped.
    """
    lifecycles: dict[str, dict[str, Any]] = {}
    for row in rows:
        pid = row.get("id")
        if not isinstance(pid, str) or not pid:
            continue
        life = lifecycles.get(pid)
        if life is None:
            life = {
                "id": pid,
                "events": [],
                "event_types": set(),
                "first": row,
                "last": row,
                "terminal": None,
            }
            lifecycles[pid] = life
        event = row.get("event")
        life["events"].append(row)
        life["event_types"].add(event)
        life["last"] = row
        if event in TERMINAL_EVENTS:
            life["terminal"] = row
    return lifecycles


def route_stats(nets: list[float]) -> dict[str, Any]:
    """Aggregate one route's net_bps list, withholding WR below MIN_EVIDENCE_N."""
    n = len(nets)
    wins = sum(1 for x in nets if x > 0)
    total = sum(nets)
    stats: dict[str, Any] = {
        "n": n,
        "total_net_bps": round(total, 1),
        "avg_net_bps": round(total / n, 1) if n else None,
        "wins": wins,
        "losses": n - wins,
        "best_bps": round(max(nets), 1) if nets else None,
        "worst_bps": round(min(nets), 1) if nets else None,
    }
    if n >= MIN_EVIDENCE_N:
        stats["wr"] = round(wins / n, 3)
        stats["wr_display"] = f"{wins / n:.1%}"
        stats["sufficient_n"] = True
    else:
        stats["wr"] = None
        stats["wr_display"] = "INSUFFICIENT_N"
        stats["sufficient_n"] = False
    return stats


def _closes_by_route(closes: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    nets: dict[str, list[float]] = collections.defaultdict(list)
    for row in closes:
        net = row.get("net_bps")
        if not isinstance(net, (int, float)):
            continue
        nets[str(row.get("signal") or UNKNOWN)].append(float(net))
    table = {route: route_stats(values) for route, values in nets.items()}
    return dict(sorted(table.items(), key=lambda kv: -kv[1]["n"]))


def _totals(table: dict[str, dict[str, Any]]) -> dict[str, Any]:
    n = sum(v["n"] for v in table.values())
    total = sum(v["total_net_bps"] for v in table.values())
    wins = sum(v["wins"] for v in table.values())
    return {
        "n": n,
        "total_net_bps": round(total, 1),
        "wins": wins,
        "wr": round(wins / n, 3) if n else None,
        "routes": len(table),
    }


def evidence(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-route PnL recomputed from the ledger, forward and backfill kept apart."""
    closes = [r for r in rows if r.get("event") == "CLOSE"]
    forward_closes = [r for r in closes if not is_backfill(r)]
    backfill_closes = [r for r in closes if is_backfill(r)]

    forward_table = _closes_by_route(forward_closes)
    backfill_table = _closes_by_route(backfill_closes)

    span = [r.get("exit_ts_ms") for r in forward_closes if isinstance(r.get("exit_ts_ms"), (int, float))]
    return {
        "forward": {
            "label": "FORWARD_EVIDENCE",
            "by_route": forward_table,
            "totals": _totals(forward_table),
            "first_close_ts_ms": min(span) if span else None,
            "last_close_ts_ms": max(span) if span else None,
            "min_evidence_n": MIN_EVIDENCE_N,
            "fee_bps": FEE_BPS,
        },
        "backfill": {
            "label": "SIMULATED_NOT_EVIDENCE",
            "by_route": backfill_table,
            "totals": _totals(backfill_table),
        },
        "closes_total": len(closes),
    }


def reconcile_liveness(
    lifecycles: dict[str, dict[str, Any]],
    state_positions: Any,
) -> dict[str, Any]:
    """Reconcile ledger-derived liveness against what the runner actually tracks."""
    reconcilable = isinstance(state_positions, dict)
    tracked_ids = set(state_positions) if reconcilable else set()

    exposed: list[dict[str, Any]] = []
    waiting: list[dict[str, Any]] = []
    orphaned: list[dict[str, Any]] = []

    for life in lifecycles.values():
        if life["terminal"] is not None:
            continue
        last = life["last"]
        record = {
            "id": life["id"],
            "signal": last.get("signal"),
            "direction": last.get("direction"),
            "status": last.get("status"),
            "opened_utc": last.get("opened_utc"),
            "entry_price": last.get("entry_price"),
            "runner_tracked": life["id"] in tracked_ids if reconcilable else None,
        }
        if reconcilable and life["id"] not in tracked_ids:
            orphaned.append(record)
        elif has_exposure(last.get("status")):
            exposed.append(record)
        else:
            waiting.append(record)

    ledger_ids = {life["id"] for life in lifecycles.values()}
    untracked_by_ledger = sorted(tracked_ids - ledger_ids) if reconcilable else []

    return {
        "reconciled": reconcilable,
        "runner_tracked_n": len(tracked_ids) if reconcilable else None,
        "exposed": exposed,
        "waiting": waiting,
        "orphaned": orphaned,
        "exposed_n": len(exposed),
        "waiting_n": len(waiting),
        "orphaned_n": len(orphaned),
        "untracked_by_ledger": untracked_by_ledger,
    }


def funnel(lifecycles: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Where each route dies: opened -> silence confirmed -> entered -> closed/expired."""
    routes: dict[str, dict[str, Any]] = collections.defaultdict(
        lambda: {
            "opened": 0,
            "silence_confirmed": 0,
            "entered": 0,
            "closed": 0,
            "expired": 0,
            "still_live": 0,
        }
    )
    expire_reasons: collections.Counter[str] = collections.Counter()
    close_reasons: collections.Counter[str] = collections.Counter()

    for life in lifecycles.values():
        last = life["last"]
        bucket = routes[str(last.get("signal") or UNKNOWN)]
        bucket["opened"] += 1

        if "SILENCE_CONFIRMED" in life["event_types"] or any(
            e.get("silence_confirmed_utc") for e in life["events"]
        ):
            bucket["silence_confirmed"] += 1
        if any(e.get("entry_ts_ms") is not None for e in life["events"]):
            bucket["entered"] += 1

        terminal = life["terminal"]
        if terminal is None:
            bucket["still_live"] += 1
        elif terminal.get("event") == "CLOSE":
            bucket["closed"] += 1
            close_reasons[str(terminal.get("close_reason") or UNKNOWN)] += 1
        else:
            bucket["expired"] += 1
            expire_reasons[str(terminal.get("status") or UNKNOWN)] += 1

    return {
        "by_route": dict(sorted(routes.items(), key=lambda kv: -kv[1]["opened"])),
        "expire_reasons": dict(expire_reasons.most_common()),
        "close_reasons": dict(close_reasons.most_common()),
        "total_positions": len(lifecycles),
    }


def observers(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Observer protocol counters. All are OBSERVE_ONLY: none places an order.

    A triggered observer looks like an action in the raw ledger; it is not one.
    """
    profit_lock = {
        "protocol": "PROFIT_LOCK_200_100_OBSERVER",
        "mode": "OBSERVE_ONLY_NO_ORDER",
        "triggered": 0,
        "shadow_exit": 0,
        "trigger_events": 0,
        "shadow_exit_events": 0,
    }
    for row in rows:
        block = row.get("profit_lock_observer")
        if isinstance(block, dict):
            if block.get("triggered"):
                profit_lock["triggered"] += 1
            if block.get("shadow_exit"):
                profit_lock["shadow_exit"] += 1
        event = row.get("event")
        if event == "PROFIT_LOCK_TRIGGER":
            profit_lock["trigger_events"] += 1
        elif event == "PROFIT_LOCK_SHADOW_EXIT":
            profit_lock["shadow_exit_events"] += 1

    return {
        "profit_lock": profit_lock,
        "bd_first_buy50": {
            "protocol": "BD_FIRST_BUY50_OBSERVER",
            "mode": "OBSERVE_ONLY_NO_ORDER",
            "obs_exit_events": sum(1 for r in rows if r.get("event") == "BD_FIRST_BUY50_OBS_EXIT"),
        },
        "scalein": {
            "protocol": "SCALEIN_OBSERVER",
            "mode": "OBSERVE_ONLY_NO_ORDER",
            "add_events": sum(1 for r in rows if r.get("event") == "SCALEIN_ADD"),
        },
    }
