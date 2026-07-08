"""BATCH-P3-002: cooldown sensitivity view tests (Protocol §8.4, non-canonical).

Run: pytest tests/test_ami_identity_cooldown_sensitivity.py --basetemp <scratchpad> -p no:cacheprovider
"""
from ami.identity.cooldown_sensitivity import (
    DEFAULT_COOLDOWN_WINDOWS_SECONDS,
    compute_cooldown_groups,
    seed_membership,
)
from ami.warehouse.schema import connect, init_schema

HOUR_MS = 3600_000


def _mk_events(anchors_ms, symbol="ETHUSDT", family="ROUTE_A"):
    return [{"event_id": f"EVT-{i}", "symbol": symbol, "event_family": family, "anchor_ts_ms": ts}
            for i, ts in enumerate(anchors_ms)]


def test_episodes_split_on_gap_exceeding_window():
    events = _mk_events([0, 1000 * 1000, 20 * HOUR_MS])  # last one is a big gap
    memberships = compute_cooldown_groups(events, gap_seconds=3600, window_label="1h")
    keys = {m["event_id"]: m["candidate_cycle_key"] for m in memberships}
    assert keys["EVT-0"] == keys["EVT-1"]  # within 1h gap -> same episode
    assert keys["EVT-2"] != keys["EVT-1"]  # 20h gap -> new episode


def test_episodes_do_not_mix_across_symbol_or_family():
    events = _mk_events([0, 100], symbol="ETHUSDT", family="ROUTE_A") + \
        _mk_events([0, 100], symbol="ETHUSDT", family="ROUTE_B")
    memberships = compute_cooldown_groups(events, gap_seconds=3600, window_label="1h")
    keys_by_family = {}
    for e, m in zip(events, memberships):
        keys_by_family.setdefault(e["event_family"], set()).add(m["candidate_cycle_key"])
    assert keys_by_family["ROUTE_A"].isdisjoint(keys_by_family["ROUTE_B"])


def test_candidate_cycle_key_deterministic_across_calls():
    events = _mk_events([0, 500_000, 5 * HOUR_MS])
    m1 = compute_cooldown_groups(events, gap_seconds=3600, window_label="1h")
    m2 = compute_cooldown_groups(events, gap_seconds=3600, window_label="1h")
    assert m1 == m2


def test_seed_membership_writes_all_windows_non_canonical(tmp_path):
    db = tmp_path / "canonical.sqlite"
    conn = connect(db)
    init_schema(conn)
    now = 0
    for i, ts in enumerate([0, 1000 * 1000, 20 * HOUR_MS]):
        conn.execute(
            "INSERT INTO ami_events (event_id, event_family, symbol, anchor_ts_ms, source_quality, "
            "event_definition_version, schema_version, provenance, created_ms, updated_ms) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            (f"EVT-{i}", "ROUTE_A", "ETHUSDT", ts, "REAL_LIQUIDATION", "test-v1", 3, "test", now, now),
        )
    conn.commit()

    n = seed_membership(conn)
    assert n == 3 * len(DEFAULT_COOLDOWN_WINDOWS_SECONDS)
    is_canonical_values = {r[0] for r in conn.execute("SELECT DISTINCT is_canonical FROM event_cycle_membership")}
    conn.close()
    assert is_canonical_values == {0}


def test_seed_membership_is_idempotent(tmp_path):
    db = tmp_path / "canonical.sqlite"
    conn = connect(db)
    init_schema(conn)
    now = 0
    conn.execute(
        "INSERT INTO ami_events (event_id, event_family, symbol, anchor_ts_ms, source_quality, "
        "event_definition_version, schema_version, provenance, created_ms, updated_ms) "
        "VALUES (?,?,?,?,?,?,?,?,?,?)",
        ("EVT-0", "ROUTE_A", "ETHUSDT", 0, "REAL_LIQUIDATION", "test-v1", 3, "test", now, now),
    )
    conn.commit()
    n1 = seed_membership(conn)
    count1 = conn.execute("SELECT COUNT(*) FROM event_cycle_membership").fetchone()[0]
    n2 = seed_membership(conn)
    count2 = conn.execute("SELECT COUNT(*) FROM event_cycle_membership").fetchone()[0]
    conn.close()
    assert n1 == n2
    assert count1 == count2
