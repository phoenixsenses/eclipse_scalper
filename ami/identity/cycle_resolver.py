"""BATCH-P3-005: canonical cycle resolver — cycle_definition_version=v1.

OD-003 APPROVED by operator (2026-07-03): A2 (multi-signal resolver per
Observatory §5.4) + B2 (start with the 2-state censoring subset already used
at event level) + C2 (direction conflicts are flagged, never auto-resolved
-- WAIT-equivalent).

Signals used (Observatory §5.4 checklist):
  - same symbol                      -> IMPLEMENTED (grouping key)
  - same event family                -> IMPLEMENTED (grouping key)
  - time-distance threshold           -> IMPLEMENTED: CONTINUITY_GAP_SECONDS
    = 14400s (4h). Grounded in the live system's own established LONG-route
    operational horizon (LONG_HORIZON_MS=4h, SYSTEM_STATE.md §2), not an
    arbitrary pick.
  - same dominant structural state     -> IMPLEMENTED: ami/states/engine.py
    StateEngine.build_bundle() point-in-time query (known-at safe, ts_ms<=?
    filters only) at the "1h" timeframe, a defensible mid-point between the
    system's SHORT (2h) and LONG (4h) route horizons.
  - cascade continuity / shared parent event / overlapping observer horizon
    -> NOT IMPLEMENTED in v1. These require an explicit event-linkage data
    model (parent_event_id chains) that does not exist yet. This is a
    documented limitation, not a fabricated signal -- a future
    cycle_definition_version can add them without disturbing v1's frozen
    rows (immutable versioning: whitepaper §51.1/Obs §5.4 "must not
    silently change previous cycle IDs").

Direction conflict (C2): a cycle whose member events carry both a LONG-ish
and a SHORT-ish signal name is flagged direction_conflict=1 and NOT
auto-resolved into a single direction -- matches Protocol §8.3 exactly.

This module is strictly additive: it does not touch or remove the
non-canonical sensitivity rows written by cooldown_sensitivity.py
(cycle_definition_version="sensitivity-cooldown-*-v1", is_canonical=0).
It adds new event_cycle_membership rows with
cycle_definition_version="canonical-v1", is_canonical=1.
"""
from __future__ import annotations
import hashlib
import time
from collections import Counter
from typing import Callable

CANONICAL_CYCLE_DEFINITION_VERSION = "canonical-v1"
CONTINUITY_GAP_SECONDS = 14400  # 4h -- see module docstring


def default_state_lookup(symbol: str, anchor_ts_ms: int) -> str | None:
    """Real, point-in-time structural-state lookup against microstructure.db (RO)."""
    from ami.states.engine import StateEngine

    eng = StateEngine()
    try:
        bundle = eng.build_bundle(symbol, anchor_ts_ms)
        for s in bundle.states:
            if s.timeframe == "1h" and s.family.value == "STRUCTURE_STATE":
                return s.label
        return None
    finally:
        eng.close()


def _cycle_id(symbol: str, first_anchor_ts_ms: int) -> str:
    key = f"{symbol}|{CANONICAL_CYCLE_DEFINITION_VERSION}|{first_anchor_ts_ms}"
    return "CYC-" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]


def _direction(signal: str) -> str:
    s = signal.upper()
    if "LONG" in s:
        return "LONG"
    if "SHORT" in s:
        return "SHORT"
    return "UNKNOWN"


def resolve_cycles(events: list[dict], state_lookup_fn: Callable[[str, int], str | None] | None = None) -> list[dict]:
    """events: dicts with event_id, symbol, event_family, anchor_ts_ms, censor_status,
    route_version (optional, comma-joined signal names).

    Returns one dict per resolved cycle: cycle_id, symbol, start_ts_ms, end_ts_ms,
    cycle_definition_version, entry_state, peak_state, exit_state, event_count,
    direction_conflict, censored, confidence, member_event_ids.
    """
    lookup = state_lookup_fn or default_state_lookup

    by_key: dict[tuple, list[dict]] = {}
    for e in events:
        by_key.setdefault((e["symbol"], e["event_family"]), []).append(e)

    cycles = []
    for (symbol, family), group in by_key.items():
        group_sorted = sorted(group, key=lambda e: e["anchor_ts_ms"])

        # attach a state label to each event once (avoid duplicate lookups)
        state_cache: dict[int, str | None] = {}
        for e in group_sorted:
            if e["anchor_ts_ms"] not in state_cache:
                state_cache[e["anchor_ts_ms"]] = lookup(symbol, e["anchor_ts_ms"])

        current: list[dict] = []
        prev_ts = None
        prev_state = None
        for e in group_sorted:
            ts = e["anchor_ts_ms"]
            state = state_cache[ts]
            starts_new = (
                prev_ts is None
                or (ts - prev_ts) > CONTINUITY_GAP_SECONDS * 1000
                or (state is not None and prev_state is not None and state != prev_state)
            )
            if starts_new and current:
                cycles.append(_finalize_cycle(symbol, current, state_cache))
                current = []
            current.append(e)
            prev_ts, prev_state = ts, state
        if current:
            cycles.append(_finalize_cycle(symbol, current, state_cache))

    return cycles


def _finalize_cycle(symbol: str, members: list[dict], state_cache: dict[int, str | None]) -> dict:
    first, last = members[0], members[-1]
    states = [state_cache[m["anchor_ts_ms"]] for m in members]
    known_states = [s for s in states if s is not None]

    directions = set()
    for m in members:
        for sig in (m.get("route_version") or "").split(","):
            sig = sig.strip()
            if sig:
                d = _direction(sig)
                if d != "UNKNOWN":
                    directions.add(d)
    direction_conflict = 1 if len(directions) > 1 else 0

    all_completed = all(m.get("censor_status") == "COMPLETED" for m in members)
    censored = "COMPLETED" if all_completed else "RIGHT_CENSORED"

    confidence = round(len(known_states) / len(members), 2) if members else 0.0

    return {
        "cycle_id": _cycle_id(symbol, first["anchor_ts_ms"]),
        "symbol": symbol,
        "start_ts_ms": first["anchor_ts_ms"],
        "end_ts_ms": last["anchor_ts_ms"],
        "cycle_definition_version": CANONICAL_CYCLE_DEFINITION_VERSION,
        "entry_state": states[0],
        "peak_state": Counter(known_states).most_common(1)[0][0] if known_states else None,
        "exit_state": states[-1],
        "event_count": len(members),
        "direction_conflict": direction_conflict,
        "censored": censored,
        "confidence": confidence,
        "member_event_ids": [m["event_id"] for m in members],
    }


def seed(conn, state_lookup_fn: Callable[[str, int], str | None] | None = None,
         provenance: str = "batch-p3-005-cycle-resolver-v1") -> int:
    """Reads ami_events from the warehouse; writes ami_cycles + canonical
    event_cycle_membership rows (is_canonical=1). Does not touch existing
    non-canonical sensitivity membership rows."""
    now = int(time.time() * 1000)
    events = [
        {"event_id": r[0], "symbol": r[1], "event_family": r[2], "anchor_ts_ms": r[3],
         "censor_status": r[4], "route_version": r[5]}
        for r in conn.execute(
            "SELECT event_id, symbol, event_family, anchor_ts_ms, censor_status, route_version FROM ami_events"
        )
    ]
    cycles = resolve_cycles(events, state_lookup_fn)
    for cyc in cycles:
        conn.execute(
            "INSERT INTO ami_cycles (cycle_id, symbol, start_ts_ms, end_ts_ms, cycle_definition_version, "
            "entry_state, peak_state, exit_state, event_count, direction_conflict, censored, confidence, "
            "schema_version, provenance, created_ms, updated_ms) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) "
            "ON CONFLICT(cycle_id) DO UPDATE SET end_ts_ms=excluded.end_ts_ms, exit_state=excluded.exit_state, "
            "event_count=excluded.event_count, direction_conflict=excluded.direction_conflict, "
            "censored=excluded.censored, confidence=excluded.confidence, updated_ms=excluded.updated_ms",
            (cyc["cycle_id"], cyc["symbol"], cyc["start_ts_ms"], cyc["end_ts_ms"],
             cyc["cycle_definition_version"], cyc["entry_state"], cyc["peak_state"], cyc["exit_state"],
             cyc["event_count"], cyc["direction_conflict"], cyc["censored"], cyc["confidence"],
             3, provenance, now, now),
        )
        for eid in cyc["member_event_ids"]:
            conn.execute(
                "INSERT INTO event_cycle_membership (event_id, candidate_cycle_key, cycle_definition_version, "
                "is_canonical, schema_version, provenance, created_ms) VALUES (?,?,?,1,?,?,?) "
                "ON CONFLICT(event_id, cycle_definition_version) DO UPDATE SET "
                "candidate_cycle_key=excluded.candidate_cycle_key",
                (eid, cyc["cycle_id"], cyc["cycle_definition_version"], 3, provenance, now),
            )
    conn.commit()
    return len(cycles)


def main() -> None:
    from ami.warehouse.schema import DEFAULT_PATH, connect, init_schema

    conn = connect(DEFAULT_PATH)
    try:
        init_schema(conn)
        n = seed(conn)
        n_conflict = conn.execute(
            "SELECT COUNT(*) FROM ami_cycles WHERE cycle_definition_version=? AND direction_conflict=1",
            (CANONICAL_CYCLE_DEFINITION_VERSION,),
        ).fetchone()[0]
        print(f"resolved {n} canonical-v1 cycles ({n_conflict} with direction_conflict=1, unresolved/WAIT)")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
