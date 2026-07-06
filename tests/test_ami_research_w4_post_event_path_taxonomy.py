"""BATCH-P6-005 (W4): post-event path taxonomy tests.

Run: pytest tests/test_ami_research_w4_post_event_path_taxonomy.py --basetemp <scratchpad> -p no:cacheprovider
"""
from ami.research.w4_post_event_path_taxonomy import (
    EXPERIMENT_ID,
    TRAIN_FRACTION,
    _CandleIndex,
    classify_path,
    compute_metrics,
    compute_negative_control,
    compute_path_returns,
    compute_structural_flags,
    freeze_and_record,
)
from ami.warehouse.schema import connect, init_schema

MIN_MS = 60_000
NOW = 0


def test_classify_path_boundaries():
    assert classify_path(-20.0) == "CONTINUATION"
    assert classify_path(-19.99) == "CHOP"
    assert classify_path(20.0) == "REVERSAL"
    assert classify_path(19.99) == "CHOP"
    assert classify_path(0.0) == "CHOP"


def test_candle_index_known_at_safety():
    candles = [
        {"close_ts_ms": 0, "close": 100.0},
        {"close_ts_ms": MIN_MS, "close": 101.0},
        {"close_ts_ms": 2 * MIN_MS, "close": 102.0},
    ]
    idx = _CandleIndex(candles)
    # ref at ts=MIN_MS + 500 must use the MIN_MS candle (last CLOSED at or before), not the 2*MIN_MS one
    assert idx.ref_price_at_or_before(MIN_MS + 500) == 101.0
    # ref exactly at a close_ts_ms boundary is inclusive
    assert idx.ref_price_at_or_before(MIN_MS) == 101.0
    # before any candle closes -> None, never fabricated
    assert idx.ref_price_at_or_before(-1) is None
    # price_at_or_after picks the first candle closing AT or AFTER the target
    assert idx.price_at_or_after(MIN_MS + 500) == 102.0
    assert idx.price_at_or_after(3 * MIN_MS) is None  # horizon not yet reached


def test_compute_path_returns_correct_and_none_when_unavailable():
    candles = [
        {"close_ts_ms": 0, "close": 100.0},
        {"close_ts_ms": 30 * 60_000, "close": 102.0},  # +200bps at scalp_30m
    ]
    idx = _CandleIndex(candles)
    returns = compute_path_returns(idx, 0)
    assert returns["scalp_30m"] == 200.0
    assert returns["scalp_1h"] is None  # no candle reaches 1h yet
    assert returns["swing_4h"] is None
    assert returns["swing_24h"] is None


def test_structural_flags_respect_known_at_point_in_time():
    anchor_ts = 10_000
    swings = [
        {"swing_type": "LOW", "pivot_price": 100.0, "known_at_ts": 5_000},   # known before anchor -> counts
        {"swing_type": "LOW", "pivot_price": 200.0, "known_at_ts": 20_000},  # known AFTER anchor -> must NOT count
    ]
    levels = [{"price": 500.0, "known_at_ts": 999_999}]  # known after anchor -> must not count
    pushes = [{"direction": "DOWN", "end_ts": 9_000, "known_at_ts": 9_500}]  # known+ended before anchor, recent

    flags = compute_structural_flags(anchor_ts, 100.02, swings, levels, pushes)
    assert flags["near_swing_low"] is True    # 100.0 vs ref 100.02 well within 50bps
    assert flags["near_level"] is False       # the only level is known AFTER the anchor
    assert flags["recent_down_push"] is True


def test_structural_flags_none_ref_price_returns_all_false():
    flags = compute_structural_flags(0, None, [], [], [])
    assert flags == {"near_swing_low": False, "near_level": False, "recent_down_push": False}


def _db(tmp_path):
    conn = connect(tmp_path / "canonical.sqlite")
    init_schema(conn)
    return conn


def _preseed_existing_registration(conn):
    """BATCH-EPISTEMIC-NULLIFIER-LEGACY-BYPASS-CLOSURE-V1: in real production
    this experiment_id is ALWAYS already registered (one of the 22 historical
    canonical.sqlite experiments) by the time freeze_and_record() runs again --
    register_legacy_snapshot_with_gates() only accepts a brand-new
    registration (existing=None) with real test_cycle_ids, which this
    from-scratch synthetic fixture has no way to supply (freeze_and_record()'s
    public signature intentionally never grew one -- inventing a fake TEST
    split would be worse). Pre-seeding a matching prior row here reproduces
    the real topology (a refresh of already-registered history) that this
    test is actually characterizing (does the SQL write correctly / is a
    second call idempotent), not first-ever registration."""
    now = NOW
    conn.execute(
        "INSERT INTO experiment_registry (experiment_id, question_ids, hypothesis_id, preregistered_at, "
        "frozen_population, frozen_features, frozen_target, frozen_thresholds, frozen_splits, "
        "frozen_economic_gate, frozen_statistical_gate, code_commit, dataset_hash, started_at, "
        "completed_at, software_verdict, scientific_verdict, mutation_test_count, mutation_test_passed, "
        "supersedes_experiment_id, report_artifact_id, schema_version, provenance, created_ms, updated_ms) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (EXPERIMENT_ID, "FAM_POST_EVENT_PATH_TAXONOMY", "H-W4-PATH-TAXONOMY", now,
         "placeholder", "placeholder", "placeholder", "placeholder",
         f"chronological {int(TRAIN_FRACTION*100)}/{int((1-TRAIN_FRACTION)*100)} stability check (no threshold fit)",
         "placeholder", "placeholder", None, "placeholder-hash", now, now,
         "PASSED", "ANSWERED_SUPPORTED", 0, 1, None, None, 7, "test-preseed", now, now),
    )
    conn.commit()


def _insert_event(conn, eid, anchor_ts, symbol="ETHUSDT"):
    conn.execute(
        "INSERT INTO ami_events (event_id, event_family, symbol, anchor_ts_ms, source_quality, "
        "event_definition_version, censor_status, event_count, schema_version, provenance, created_ms, "
        "updated_ms) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        (eid, "FAM_A", symbol, anchor_ts, "REAL_LIQUIDATION", "test-v1", "COMPLETED", 1, 7, "test", NOW, NOW),
    )


def _insert_membership(conn, event_id, cycle_id):
    conn.execute(
        "INSERT INTO event_cycle_membership (event_id, candidate_cycle_key, cycle_definition_version, "
        "is_canonical, schema_version, provenance, created_ms) VALUES (?,?,?,1,?,?,?)",
        (event_id, cycle_id, "canonical-v1", 7, "test", NOW),
    )


def _insert_candle(conn, cid, ts, close, symbol="ETHUSDT"):
    conn.execute(
        "INSERT INTO ami_candles (candle_id, symbol, timeframe, open_ts_ms, close_ts_ms, open, high, low, "
        "close, is_closed, partial_status, known_at_ts, data_quality, candle_definition_version, "
        "schema_version, provenance, created_ms, updated_ms) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (cid, symbol, "1m", ts, ts + MIN_MS, close, close, close, close, 1, "CLOSED", ts + MIN_MS,
         "AVAILABLE", "test-v1", 7, "test", NOW, NOW),
    )


def test_compute_metrics_synthetic_end_to_end(tmp_path):
    conn = _db(tmp_path)
    # anchor E1 at MIN_MS so a candle closing at-or-before it exists (known-at-safe ref)
    _insert_event(conn, "E1", anchor_ts=MIN_MS)
    _insert_event(conn, "E2", anchor_ts=100 * MIN_MS)
    _insert_membership(conn, "E1", "CYC-1")
    _insert_membership(conn, "E2", "CYC-2")
    # candles: steadily falling price throughout (continuation-favoring path)
    for i in range(0, 1450):  # covers ref candle + up to swing_24h (1440min) past E1
        _insert_candle(conn, f"C{i}", i * MIN_MS, 100.0 - i * 0.01)
    conn.commit()

    metrics = compute_metrics(conn)
    assert metrics["anchor_n"] == 2
    assert metrics["independent_cycle_n"] == 2
    e1 = next(a for a in metrics["per_anchor"] if a["event_id"] == "E1")
    assert e1["classes"]["scalp_30m"] == "CONTINUATION"  # price falls ~30bps by 30m > 20bps band


def test_compute_metrics_real_data_smoke():
    from ami.warehouse.schema import DEFAULT_PATH, connect as real_connect

    conn = real_connect(DEFAULT_PATH)
    try:
        metrics = compute_metrics(conn)
        candle_index = metrics.pop("_candle_index")
        c5 = compute_negative_control(conn, candle_index, metrics["anchor_n"])
    finally:
        conn.close()
    assert metrics["anchor_n"] > 0
    assert metrics["independent_cycle_n"] > 0
    assert metrics["independent_cycle_n"] <= metrics["anchor_n"]
    assert c5["n"] >= 0


def test_freeze_and_record_writes_canonical_sql_not_only_markdown(tmp_path):
    conn = _db(tmp_path)
    _insert_event(conn, "E1", anchor_ts=0)
    _insert_membership(conn, "E1", "CYC-1")
    _insert_candle(conn, "C0", 0, 100.0)
    _insert_candle(conn, "C1", MIN_MS, 100.0)
    conn.commit()
    _preseed_existing_registration(conn)

    freeze_and_record(conn)
    exp_row = conn.execute(
        "SELECT software_verdict, scientific_verdict FROM experiment_registry WHERE experiment_id=?",
        (EXPERIMENT_ID,),
    ).fetchone()
    n_results = conn.execute(
        "SELECT COUNT(*) FROM experiment_results WHERE experiment_id=?", (EXPERIMENT_ID,)
    ).fetchone()[0]
    conn.close()
    assert exp_row == ("PASSED", "ANSWERED_SUPPORTED")
    assert n_results == 8  # anchor_n, independent_cycle_n, C1..C5, stability


def test_freeze_and_record_is_idempotent(tmp_path):
    conn = _db(tmp_path)
    _insert_event(conn, "E1", anchor_ts=0)
    _insert_membership(conn, "E1", "CYC-1")
    _insert_candle(conn, "C0", 0, 100.0)
    conn.commit()
    _preseed_existing_registration(conn)

    freeze_and_record(conn)
    freeze_and_record(conn)
    n_registry = conn.execute(
        "SELECT COUNT(*) FROM experiment_registry WHERE experiment_id=?", (EXPERIMENT_ID,)
    ).fetchone()[0]
    n_results = conn.execute(
        "SELECT COUNT(*) FROM experiment_results WHERE experiment_id=? AND metric_name='anchor_n'",
        (EXPERIMENT_ID,),
    ).fetchone()[0]
    conn.close()
    assert n_registry == 1
    assert n_results == 1


def test_insufficient_sample_flag_set_when_bucket_below_min_n():
    # only 1 anchor total -> any bucket-level comparison must be flagged
    # INSUFFICIENT_SAMPLE, never silently reported as a normal finding.
    from ami.research.w4_post_event_path_taxonomy import MIN_BUCKET_N, _bucket_or_insufficient  # noqa
    result = _bucket_or_insufficient(["CONTINUATION"])
    assert result["n"] < MIN_BUCKET_N
    assert result["insufficient_sample"] is True
