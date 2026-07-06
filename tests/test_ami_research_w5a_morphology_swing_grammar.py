"""BATCH-P6-006 (W5a): candle morphology + swing grammar tests.

Run: pytest tests/test_ami_research_w5a_morphology_swing_grammar.py --basetemp <scratchpad> -p no:cacheprovider
"""
from ami.research.w5a_morphology_swing_grammar import (
    EXPERIMENT_ID,
    TRAIN_FRACTION,
    classify_swing_grammar,
    compute_metrics,
    freeze_and_record,
)
from ami.warehouse.schema import connect, init_schema

MIN_MS = 60_000
NOW = 0


def _swing(swing_type, pivot_ts, pivot_price, known_at_ts):
    return {"swing_type": swing_type, "pivot_ts": pivot_ts, "pivot_price": pivot_price, "known_at_ts": known_at_ts}


def test_swing_grammar_uptrend_structure():
    anchor_ts = 100
    swings = [
        _swing("HIGH", 10, 100.0, 15),
        _swing("HIGH", 20, 110.0, 25),   # higher high
        _swing("LOW", 12, 90.0, 17),
        _swing("LOW", 22, 95.0, 27),      # higher low
    ]
    assert classify_swing_grammar(anchor_ts, swings) == "UPTREND_STRUCTURE"


def test_swing_grammar_downtrend_structure():
    anchor_ts = 100
    swings = [
        _swing("HIGH", 10, 110.0, 15),
        _swing("HIGH", 20, 100.0, 25),   # lower high
        _swing("LOW", 12, 95.0, 17),
        _swing("LOW", 22, 90.0, 27),      # lower low
    ]
    assert classify_swing_grammar(anchor_ts, swings) == "DOWNTREND_STRUCTURE"


def test_swing_grammar_mixed_structure():
    anchor_ts = 100
    swings = [
        _swing("HIGH", 10, 100.0, 15),
        _swing("HIGH", 20, 110.0, 25),   # higher high
        _swing("LOW", 12, 95.0, 17),
        _swing("LOW", 22, 90.0, 27),      # lower low -> mixed
    ]
    assert classify_swing_grammar(anchor_ts, swings) == "MIXED_STRUCTURE"


def test_swing_grammar_insufficient_when_fewer_than_two_of_a_type():
    anchor_ts = 100
    swings = [_swing("HIGH", 10, 100.0, 15), _swing("LOW", 12, 90.0, 17)]  # only 1 of each
    assert classify_swing_grammar(anchor_ts, swings) == "INSUFFICIENT_STRUCTURE"


def test_swing_grammar_respects_known_at_point_in_time():
    anchor_ts = 20
    swings = [
        _swing("HIGH", 5, 100.0, 6),
        _swing("HIGH", 10, 110.0, 999),  # known AFTER anchor -- must not count
        _swing("LOW", 7, 90.0, 8),
        _swing("LOW", 12, 85.0, 13),
    ]
    # only 1 known HIGH before anchor -> insufficient, even though 2 HIGH rows exist total
    assert classify_swing_grammar(anchor_ts, swings) == "INSUFFICIENT_STRUCTURE"


def _db(tmp_path):
    conn = connect(tmp_path / "canonical.sqlite")
    init_schema(conn)
    return conn


def _preseed_existing_registration(conn):
    """BATCH-EPISTEMIC-NULLIFIER-LEGACY-BYPASS-CLOSURE-V1: see the identical
    helper in tests/test_ami_research_w4_post_event_path_taxonomy.py -- this
    experiment_id is always already registered in real production; pre-seed
    a matching prior row (strict columns only) so this from-scratch fixture
    exercises the real "refresh" topology, not first-ever registration."""
    now = NOW
    conn.execute(
        "INSERT INTO experiment_registry (experiment_id, question_ids, hypothesis_id, preregistered_at, "
        "frozen_population, frozen_features, frozen_target, frozen_thresholds, frozen_splits, "
        "frozen_economic_gate, frozen_statistical_gate, code_commit, dataset_hash, started_at, "
        "completed_at, software_verdict, scientific_verdict, mutation_test_count, mutation_test_passed, "
        "supersedes_experiment_id, report_artifact_id, schema_version, provenance, created_ms, updated_ms) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (EXPERIMENT_ID, "FAM_CANDLE_MORPHOLOGY_SWING_GRAMMAR", "H-W5A-MORPHOLOGY-GRAMMAR", now,
         "placeholder", "placeholder", "placeholder", "placeholder",
         f"chronological {int(TRAIN_FRACTION*100)}/{int((1-TRAIN_FRACTION)*100)} stability check",
         "placeholder", "placeholder", None, "placeholder-hash", now, now,
         "PASSED", "ANSWERED_SUPPORTED", 0, 1, "E-W4-POST-EVENT-PATH-TAXONOMY-001", None, 7,
         "test-preseed", now, now),
    )
    conn.commit()


def _insert_event(conn, eid, anchor_ts, symbol="ETHUSDT"):
    conn.execute(
        "INSERT INTO ami_events (event_id, event_family, symbol, anchor_ts_ms, source_quality, "
        "event_definition_version, censor_status, event_count, schema_version, provenance, created_ms, "
        "updated_ms) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        (eid, "FAM_A", symbol, anchor_ts, "REAL_LIQUIDATION", "test-v1", "COMPLETED", 1, 7, "test", NOW, NOW),
    )


def _insert_candle(conn, cid, ts, close, symbol="ETHUSDT"):
    conn.execute(
        "INSERT INTO ami_candles (candle_id, symbol, timeframe, open_ts_ms, close_ts_ms, open, high, low, "
        "close, is_closed, partial_status, known_at_ts, data_quality, candle_definition_version, "
        "schema_version, provenance, created_ms, updated_ms) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (cid, symbol, "1m", ts, ts + MIN_MS, close, close, close, close, 1, "CLOSED", ts + MIN_MS,
         "AVAILABLE", "test-v1", 7, "test", NOW, NOW),
    )


def _insert_morphology(conn, cid, label):
    conn.execute(
        "INSERT INTO ami_candle_morphology (candle_id, close_quality_label, morphology_definition_version, "
        "schema_version, provenance, created_ms) VALUES (?,?,?,?,?,?)",
        (cid, label, "test-v1", 7, "test", NOW),
    )


def test_compute_metrics_synthetic_end_to_end(tmp_path):
    conn = _db(tmp_path)
    _insert_event(conn, "E1", anchor_ts=MIN_MS)
    for i in range(0, 1450):
        _insert_candle(conn, f"C{i}", i * MIN_MS, 100.0 - i * 0.01)
        _insert_morphology(conn, f"C{i}", "CLOSE_NEAR_LOW")
    conn.commit()

    metrics = compute_metrics(conn)
    assert metrics["anchor_n"] == 1
    e1 = metrics["per_anchor"][0]
    assert e1["morphology_label"] == "CLOSE_NEAR_LOW"
    assert e1["swing_grammar"] == "INSUFFICIENT_STRUCTURE"  # no ami_swings rows in this fixture


def test_compute_metrics_real_data_smoke():
    from ami.warehouse.schema import DEFAULT_PATH, connect as real_connect

    conn = real_connect(DEFAULT_PATH)
    try:
        metrics = compute_metrics(conn)
    finally:
        conn.close()
    assert metrics["anchor_n"] > 0


def test_freeze_and_record_writes_canonical_sql_and_is_idempotent(tmp_path):
    conn = _db(tmp_path)
    _insert_event(conn, "E1", anchor_ts=MIN_MS)
    _insert_candle(conn, "C0", 0, 100.0)
    _insert_morphology(conn, "C0", "MID_RANGE_CLOSE")
    conn.commit()
    _preseed_existing_registration(conn)

    freeze_and_record(conn)
    freeze_and_record(conn)
    exp_row = conn.execute(
        "SELECT software_verdict, scientific_verdict FROM experiment_registry WHERE experiment_id=?",
        (EXPERIMENT_ID,),
    ).fetchone()
    n_registry = conn.execute(
        "SELECT COUNT(*) FROM experiment_registry WHERE experiment_id=?", (EXPERIMENT_ID,)
    ).fetchone()[0]
    n_results = conn.execute(
        "SELECT COUNT(*) FROM experiment_results WHERE experiment_id=? AND metric_name='anchor_n'",
        (EXPERIMENT_ID,),
    ).fetchone()[0]
    conn.close()
    assert exp_row == ("PASSED", "ANSWERED_SUPPORTED")
    assert n_registry == 1
    assert n_results == 1
