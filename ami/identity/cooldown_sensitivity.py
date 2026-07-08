"""BATCH-P3-002: cooldown sensitivity views (Protocol §8.4).

Produces sensitivity groupings at 1h/2h/4h/6h/12h/24h cooldown windows.
These are explicitly NOT canonical cycle counts -- Protocol §8.4: "label
them as cooldown sensitivity, not true independent cycle counts." No
material cycle-definition decision is made here; ami_cycles remains empty
pending operator approval (OD-003). Every row written to
event_cycle_membership carries is_canonical=0.
"""
from __future__ import annotations
import time

DEFAULT_COOLDOWN_WINDOWS_SECONDS = {
    "1h": 3600,
    "2h": 7200,
    "4h": 14400,
    "6h": 21600,
    "12h": 43200,
    "24h": 86400,
}


def compute_cooldown_groups(events: list[dict], gap_seconds: int, window_label: str) -> list[dict]:
    """events: dicts with event_id, symbol, event_family, anchor_ts_ms.

    Groups by (symbol, event_family), sorted chronologically, split into a
    new episode whenever the gap to the previous anchor exceeds gap_seconds.
    candidate_cycle_key is content-derived (first anchor in the episode) so
    re-running with the same event set reproduces identical keys.
    """
    by_key: dict[tuple, list[dict]] = {}
    for e in events:
        by_key.setdefault((e["symbol"], e["event_family"]), []).append(e)

    memberships = []
    for (symbol, family), group in by_key.items():
        group_sorted = sorted(group, key=lambda e: e["anchor_ts_ms"])
        episode_start_ts = None
        prev_ts = None
        for e in group_sorted:
            ts = e["anchor_ts_ms"]
            if prev_ts is None or (ts - prev_ts) > gap_seconds * 1000:
                episode_start_ts = ts
            prev_ts = ts
            candidate_cycle_key = f"{symbol}|{family}|cooldown-{window_label}|{episode_start_ts}"
            memberships.append({
                "event_id": e["event_id"],
                "candidate_cycle_key": candidate_cycle_key,
                "cycle_definition_version": f"sensitivity-cooldown-{window_label}-v1",
            })
    return memberships


def seed_membership(conn, windows: dict[str, int] = DEFAULT_COOLDOWN_WINDOWS_SECONDS,
                     provenance: str = "batch-p3-002-cooldown-sensitivity") -> int:
    """Reads ami_events from the warehouse itself (already-ingested real events) --
    read-only relative to every other store."""
    now = int(time.time() * 1000)
    events = [
        {"event_id": r[0], "symbol": r[1], "event_family": r[2], "anchor_ts_ms": r[3]}
        for r in conn.execute("SELECT event_id, symbol, event_family, anchor_ts_ms FROM ami_events")
    ]
    n = 0
    for label, gap_seconds in windows.items():
        memberships = compute_cooldown_groups(events, gap_seconds, label)
        for m in memberships:
            conn.execute(
                "INSERT INTO event_cycle_membership (event_id, candidate_cycle_key, cycle_definition_version, "
                "is_canonical, schema_version, provenance, created_ms) VALUES (?,?,?,0,?,?,?) "
                "ON CONFLICT(event_id, cycle_definition_version) DO UPDATE SET "
                "candidate_cycle_key=excluded.candidate_cycle_key",
                (m["event_id"], m["candidate_cycle_key"], m["cycle_definition_version"], 3, provenance, now),
            )
            n += 1
    conn.commit()
    return n


def main() -> None:
    from ami.warehouse.schema import DEFAULT_PATH, connect, init_schema

    conn = connect(DEFAULT_PATH)
    try:
        init_schema(conn)
        n = seed_membership(conn)
        print(f"recorded {n} non-canonical cooldown-sensitivity memberships (6 windows)")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
