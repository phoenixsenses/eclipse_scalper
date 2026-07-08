"""BATCH-P3-005: canonical cycle resolver tests (OD-003 approved: A2+B2+C2).

Run: pytest tests/test_ami_identity_cycle_resolver.py --basetemp <scratchpad> -p no:cacheprovider
"""
from ami.identity.cycle_resolver import (
    CANONICAL_CYCLE_DEFINITION_VERSION,
    CONTINUITY_GAP_SECONDS,
    resolve_cycles,
    seed,
)
from ami.warehouse.schema import connect, init_schema

HOUR_MS = 3600_000


def _mk_event(idx, ts, symbol="ETHUSDT", family="ROUTE_A", censor="COMPLETED", route_version="LONG_SILENCE"):
    return {"event_id": f"EVT-{idx}", "symbol": symbol, "event_family": family,
            "anchor_ts_ms": ts, "censor_status": censor, "route_version": route_version}


def _fake_lookup_constant(label):
    return lambda symbol, ts: label


def _fake_lookup_by_ts(mapping):
    return lambda symbol, ts: mapping.get(ts)


def test_gap_based_splitting():
    events = [_mk_event(0, 0), _mk_event(1, 1000 * 1000), _mk_event(2, 20 * HOUR_MS)]
    cycles = resolve_cycles(events, state_lookup_fn=_fake_lookup_constant(None))
    assert len(cycles) == 2
    assert cycles[0]["event_count"] == 2  # first two within 4h continuity gap
    assert cycles[1]["event_count"] == 1


def test_structural_state_discontinuity_forces_new_cycle_within_gap():
    ts_a, ts_b = 0, 1000 * 1000  # well within CONTINUITY_GAP_SECONDS
    events = [_mk_event(0, ts_a), _mk_event(1, ts_b)]
    mapping = {ts_a: "RANGE", ts_b: "BREAKDOWN"}
    cycles = resolve_cycles(events, state_lookup_fn=_fake_lookup_by_ts(mapping))
    assert len(cycles) == 2  # state changed -> forced split despite small gap


def test_same_state_within_gap_stays_one_cycle():
    ts_a, ts_b = 0, 1000 * 1000
    events = [_mk_event(0, ts_a), _mk_event(1, ts_b)]
    mapping = {ts_a: "RANGE", ts_b: "RANGE"}
    cycles = resolve_cycles(events, state_lookup_fn=_fake_lookup_by_ts(mapping))
    assert len(cycles) == 1
    assert cycles[0]["event_count"] == 2


def test_direction_conflict_flagged_not_resolved():
    events = [
        _mk_event(0, 0, route_version="LONG_SILENCE"),
        _mk_event(1, 1000, route_version="SHORT_NEITHER"),
    ]
    cycles = resolve_cycles(events, state_lookup_fn=_fake_lookup_constant(None))
    assert len(cycles) == 1
    assert cycles[0]["direction_conflict"] == 1
    # C2: conflict is flagged, not silently collapsed into one direction --
    # the cycle dict carries no "resolved_direction" field at all.
    assert "resolved_direction" not in cycles[0]


def test_no_direction_conflict_when_single_direction():
    events = [_mk_event(0, 0, route_version="LONG_SILENCE"),
              _mk_event(1, 1000, route_version="LONG_SILENCE,LONG_OTHER")]
    cycles = resolve_cycles(events, state_lookup_fn=_fake_lookup_constant(None))
    assert cycles[0]["direction_conflict"] == 0


def test_censored_completed_only_if_all_members_completed():
    events = [_mk_event(0, 0, censor="COMPLETED"), _mk_event(1, 1000, censor="RIGHT_CENSORED")]
    cycles = resolve_cycles(events, state_lookup_fn=_fake_lookup_constant(None))
    assert cycles[0]["censored"] == "RIGHT_CENSORED"

    events2 = [_mk_event(0, 0, censor="COMPLETED"), _mk_event(1, 1000, censor="COMPLETED")]
    cycles2 = resolve_cycles(events2, state_lookup_fn=_fake_lookup_constant(None))
    assert cycles2[0]["censored"] == "COMPLETED"


def test_confidence_reflects_state_availability():
    events = [_mk_event(0, 0), _mk_event(1, 1000)]
    full = resolve_cycles(events, state_lookup_fn=_fake_lookup_constant("RANGE"))
    none_ = resolve_cycles(events, state_lookup_fn=_fake_lookup_constant(None))
    assert full[0]["confidence"] == 1.0
    assert none_[0]["confidence"] == 0.0


def test_cycle_id_deterministic():
    events = [_mk_event(0, 0), _mk_event(1, 1000)]
    c1 = resolve_cycles(events, state_lookup_fn=_fake_lookup_constant(None))
    c2 = resolve_cycles(events, state_lookup_fn=_fake_lookup_constant(None))
    assert c1[0]["cycle_id"] == c2[0]["cycle_id"]
    assert c1[0]["cycle_id"].startswith("CYC-")


def test_symbol_and_family_never_mixed():
    events = [_mk_event(0, 0, symbol="ETHUSDT", family="A"), _mk_event(1, 0, symbol="ETHUSDT", family="B")]
    cycles = resolve_cycles(events, state_lookup_fn=_fake_lookup_constant(None))
    assert len(cycles) == 2


def test_seed_writes_canonical_without_disturbing_sensitivity_rows(tmp_path):
    db = tmp_path / "canonical.sqlite"
    conn = connect(db)
    init_schema(conn)
    now = 0
    for i, ts in enumerate([0, 1000 * 1000]):
        conn.execute(
            "INSERT INTO ami_events (event_id, event_family, symbol, anchor_ts_ms, source_quality, "
            "event_definition_version, censor_status, route_version, schema_version, provenance, "
            "created_ms, updated_ms) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (f"EVT-{i}", "ROUTE_A", "ETHUSDT", ts, "REAL_LIQUIDATION", "test-v1", "COMPLETED",
             "LONG_SILENCE", 3, "test", now, now),
        )
    # pre-existing non-canonical sensitivity row (simulating BATCH-P3-002 output)
    conn.execute(
        "INSERT INTO event_cycle_membership (event_id, candidate_cycle_key, cycle_definition_version, "
        "is_canonical, schema_version, provenance, created_ms) VALUES (?,?,?,0,?,?,?)",
        ("EVT-0", "sensitivity-key-abc", "sensitivity-cooldown-1h-v1", 3, "test", now),
    )
    conn.commit()

    n = seed(conn, state_lookup_fn=lambda symbol, ts: None)
    assert n == 1  # both events in one cycle (within 4h gap, no state signal to split them)

    sensitivity_row = conn.execute(
        "SELECT is_canonical FROM event_cycle_membership WHERE cycle_definition_version=?",
        ("sensitivity-cooldown-1h-v1",),
    ).fetchone()
    assert sensitivity_row == (0,)  # untouched

    canonical_rows = conn.execute(
        "SELECT is_canonical FROM event_cycle_membership WHERE cycle_definition_version=?",
        (CANONICAL_CYCLE_DEFINITION_VERSION,),
    ).fetchall()
    assert canonical_rows == [(1,), (1,)]

    n_cycles = conn.execute("SELECT COUNT(*) FROM ami_cycles").fetchone()[0]
    assert n_cycles == 1
    conn.close()


def test_seed_is_idempotent(tmp_path):
    db = tmp_path / "canonical.sqlite"
    conn = connect(db)
    init_schema(conn)
    now = 0
    conn.execute(
        "INSERT INTO ami_events (event_id, event_family, symbol, anchor_ts_ms, source_quality, "
        "event_definition_version, censor_status, route_version, schema_version, provenance, "
        "created_ms, updated_ms) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        ("EVT-0", "ROUTE_A", "ETHUSDT", 0, "REAL_LIQUIDATION", "test-v1", "COMPLETED",
         "LONG_SILENCE", 3, "test", now, now),
    )
    conn.commit()
    n1 = seed(conn, state_lookup_fn=lambda s, t: None)
    n2 = seed(conn, state_lookup_fn=lambda s, t: None)
    n_cycles = conn.execute("SELECT COUNT(*) FROM ami_cycles").fetchone()[0]
    conn.close()
    assert n1 == n2 == 1
    assert n_cycles == 1
