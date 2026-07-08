"""BATCH-P6-003: all-timestamp candidate universe tests (Protocol §17.8).

Run: pytest tests/test_ami_research_candidate_universe.py --basetemp <scratchpad> -p no:cacheprovider
"""
from ami.research.candidate_universe import EXPERIMENT_ID, build_universe, freeze_and_record, seed
from ami.warehouse.schema import connect, init_schema

MIN_MS = 60_000
NOW = 0


def _candle(ts, dq="AVAILABLE"):
    return {"open_ts_ms": ts, "close_ts_ms": ts + MIN_MS, "data_quality": dq}


def _event(eid, anchor_ts):
    return {"event_id": eid, "anchor_ts_ms": anchor_ts}


def test_universe_is_unconditional_on_events():
    # zero events at all -- every candle must still become a candidate row.
    candles = [_candle(0), _candle(MIN_MS), _candle(2 * MIN_MS)]
    candidates = build_universe(candles, [], "ETHUSDT", "1m")
    assert len(candidates) == 3
    assert all(c["is_event_aligned"] == 0 for c in candidates)
    assert all(c["aligned_event_id"] is None for c in candidates)


def test_event_alignment_marks_only_the_matching_slot():
    candles = [_candle(0), _candle(MIN_MS), _candle(2 * MIN_MS)]
    events = [_event("E1", MIN_MS + 100)]  # falls inside the second candle's window
    candidates = build_universe(candles, events, "ETHUSDT", "1m")
    aligned = [c for c in candidates if c["is_event_aligned"] == 1]
    assert len(aligned) == 1
    assert aligned[0]["slot_ts_ms"] == MIN_MS
    assert aligned[0]["aligned_event_id"] == "E1"


def test_candidate_inherits_gapped_data_quality():
    candles = [_candle(0, dq="GAPPED"), _candle(MIN_MS, dq="AVAILABLE")]
    candidates = build_universe(candles, [], "ETHUSDT", "1m")
    dq_by_slot = {c["slot_ts_ms"]: c["data_quality"] for c in candidates}
    assert dq_by_slot[0] == "GAPPED"
    assert dq_by_slot[MIN_MS] == "AVAILABLE"


def test_known_at_ts_is_candle_close_not_slot_start():
    candles = [_candle(0)]
    candidates = build_universe(candles, [], "ETHUSDT", "1m")
    assert candidates[0]["known_at_ts"] == MIN_MS  # close_ts_ms, strictly after slot_ts_ms=0
    assert candidates[0]["known_at_ts"] > candidates[0]["slot_ts_ms"]


def test_candidate_id_deterministic():
    candles = [_candle(0)]
    c1 = build_universe(candles, [], "ETHUSDT", "1m")
    c2 = build_universe(candles, [], "ETHUSDT", "1m")
    assert c1[0]["candidate_id"] == c2[0]["candidate_id"]
    assert c1[0]["candidate_id"].startswith("CND-")


def test_multiple_anchors_in_same_slot_first_wins_deterministically():
    candles = [_candle(0)]
    events = [_event("E1", 100), _event("E2", 200)]  # both inside the same 1m slot
    candidates = build_universe(candles, events, "ETHUSDT", "1m")
    assert candidates[0]["is_event_aligned"] == 1
    assert candidates[0]["aligned_event_id"] == "E1"  # first-seen wins, not fabricated as both


def _db(tmp_path):
    conn = connect(tmp_path / "canonical.sqlite")
    init_schema(conn)
    return conn


def _insert_candle_row(conn, cid, ts, dq="AVAILABLE"):
    conn.execute(
        "INSERT INTO ami_candles (candle_id, symbol, timeframe, open_ts_ms, close_ts_ms, open, high, low, "
        "close, is_closed, partial_status, known_at_ts, data_quality, candle_definition_version, "
        "schema_version, provenance, created_ms, updated_ms) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (cid, "ETHUSDT", "1m", ts, ts + MIN_MS, 100.0, 101.0, 99.0, 100.0, 1, "CLOSED", ts + MIN_MS, dq,
         "test-v1", 7, "test", NOW, NOW),
    )


def _insert_event_row(conn, eid, anchor_ts):
    conn.execute(
        "INSERT INTO ami_events (event_id, event_family, symbol, anchor_ts_ms, source_quality, "
        "event_definition_version, censor_status, event_count, schema_version, provenance, created_ms, "
        "updated_ms) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        (eid, "FAM_A", "ETHUSDT", anchor_ts, "REAL_LIQUIDATION", "test-v1", "COMPLETED", 1, 7, "test", NOW, NOW),
    )


def test_seed_is_idempotent_against_synthetic_db(tmp_path):
    conn = _db(tmp_path)
    for i in range(5):
        _insert_candle_row(conn, f"CDL-{i}", i * MIN_MS)
    _insert_event_row(conn, "E1", 2 * MIN_MS + 10)
    conn.commit()

    n1 = seed(conn, symbol="ETHUSDT", timeframe="1m")
    n2 = seed(conn, symbol="ETHUSDT", timeframe="1m")
    count = conn.execute("SELECT COUNT(*) FROM ami_candidate_universe").fetchone()[0]
    n_aligned = conn.execute(
        "SELECT COUNT(*) FROM ami_candidate_universe WHERE is_event_aligned=1"
    ).fetchone()[0]
    conn.close()
    assert n1 == n2 == count == 5
    assert n_aligned == 1


def test_freeze_and_record_keeps_four_denominators_separate(tmp_path):
    conn = _db(tmp_path)
    for i in range(5):
        _insert_candle_row(conn, f"CDL-{i}", i * MIN_MS)
    _insert_event_row(conn, "E1", 2 * MIN_MS + 10)
    conn.commit()

    metrics = freeze_and_record(conn, symbol="ETHUSDT", timeframe="1m")
    assert metrics["raw_candidate_n"] == 5
    assert metrics["raw_candidate_event_aligned_n"] == 1
    assert metrics["raw_candidate_no_event_n"] == 4
    assert metrics["anchor_n_in_candidate_window"] == 1  # E1's anchor_ts_ms falls inside the candle window
    assert metrics["event_n_all_history"] == 1
    assert metrics["anchor_n_all_history"] == 1
    assert metrics["independent_cycle_n_all_history"] == 0  # no ami_cycles row inserted in this fixture

    rows = conn.execute(
        "SELECT metric_name, metric_value FROM experiment_results WHERE experiment_id=?",
        (EXPERIMENT_ID,),
    ).fetchall()
    metric_names = {r[0] for r in rows}
    conn.close()
    assert {"raw_candidate_n", "event_n_all_history", "anchor_n_all_history",
            "independent_cycle_n_all_history", "anchor_n_in_candidate_window",
            "cycle_n_in_candidate_window"}.issubset(metric_names)


def test_freeze_and_record_scopes_anchor_ratio_to_candle_window_not_all_history(tmp_path):
    # Regression for the scope-mismatch found while wiring this up:
    # ami_candles has a bounded lookback (Phase 4 design), but ami_events
    # spans full history. An event anchored BEFORE the candle window opened
    # must count toward anchor_n_all_history but NOT toward
    # anchor_n_in_candidate_window / candidate_to_anchor_ratio.
    conn = _db(tmp_path)
    for i in range(3):
        _insert_candle_row(conn, f"CDL-{i}", 100 * MIN_MS + i * MIN_MS)  # window starts at 100*MIN_MS
    _insert_event_row(conn, "E-OLD", anchor_ts=0)  # long before the candle window
    _insert_event_row(conn, "E-IN-WINDOW", anchor_ts=101 * MIN_MS + 10)
    conn.commit()

    metrics = freeze_and_record(conn, symbol="ETHUSDT", timeframe="1m")
    conn.close()
    assert metrics["anchor_n_all_history"] == 2
    assert metrics["anchor_n_in_candidate_window"] == 1  # only E-IN-WINDOW


def test_freeze_and_record_does_not_mutate_w1_experiment(tmp_path):
    # E-CANDIDATE-UNIVERSE-001 must be a distinct experiment row -- it must
    # never overwrite or supersede E-W1-CYCLE-INTEGRITY-001's own frozen
    # registry/results (operator instruction: existing W1 results untouched).
    from ami.research.w1_cycle_integrity import EXPERIMENT_ID as W1_EXPERIMENT_ID
    from ami.research.w1_cycle_integrity import freeze_and_record as w1_freeze_and_record

    conn = _db(tmp_path)
    for i in range(3):
        _insert_candle_row(conn, f"CDL-{i}", i * MIN_MS)
    _insert_event_row(conn, "E1", MIN_MS + 10)
    conn.commit()

    w1_freeze_and_record(conn)
    freeze_and_record(conn)

    n_w1 = conn.execute(
        "SELECT COUNT(*) FROM experiment_registry WHERE experiment_id=?", (W1_EXPERIMENT_ID,)
    ).fetchone()[0]
    n_universe = conn.execute(
        "SELECT COUNT(*) FROM experiment_registry WHERE experiment_id=?", (EXPERIMENT_ID,)
    ).fetchone()[0]
    conn.close()
    assert n_w1 == 1
    assert n_universe == 1


def test_freeze_and_record_is_idempotent(tmp_path):
    conn = _db(tmp_path)
    for i in range(5):
        _insert_candle_row(conn, f"CDL-{i}", i * MIN_MS)
    _insert_event_row(conn, "E1", 2 * MIN_MS + 10)
    conn.commit()

    freeze_and_record(conn)
    freeze_and_record(conn)

    n_registry = conn.execute(
        "SELECT COUNT(*) FROM experiment_registry WHERE experiment_id=?", (EXPERIMENT_ID,)
    ).fetchone()[0]
    n_results = conn.execute(
        "SELECT COUNT(*) FROM experiment_results WHERE experiment_id=? AND metric_name='raw_candidate_n'",
        (EXPERIMENT_ID,),
    ).fetchone()[0]
    conn.close()
    assert n_registry == 1
    assert n_results == 1


def test_seed_real_data_smoke_raw_candidate_n_exceeds_anchor_n():
    # Integration smoke test: the universe must be much larger than the
    # anchor population -- most moments in time are NOT liquidation events.
    from ami.warehouse.schema import DEFAULT_PATH, connect as real_connect, init_schema as real_init

    conn = real_connect(DEFAULT_PATH)
    try:
        real_init(conn)
        n_candidates = seed(conn, symbol="ETHUSDT", timeframe="1m")
        n_anchors = conn.execute(
            "SELECT COUNT(*) FROM ami_events WHERE symbol='ETHUSDT' AND source_quality='REAL_LIQUIDATION'"
        ).fetchone()[0]
        n_aligned = conn.execute(
            "SELECT COUNT(*) FROM ami_candidate_universe WHERE is_event_aligned=1"
        ).fetchone()[0]
    finally:
        conn.close()
    assert n_candidates > 0
    assert n_candidates > n_anchors  # universe is the broadest denominator
    assert n_aligned <= n_anchors  # alignment can only match, never exceed, the anchor population
