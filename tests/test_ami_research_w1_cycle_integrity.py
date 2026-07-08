"""BATCH-P6-002 (W1): Cycle Integrity & Deduplication research wave tests.

Run: pytest tests/test_ami_research_w1_cycle_integrity.py --basetemp <scratchpad> -p no:cacheprovider
"""
from ami.research.w1_cycle_integrity import EXPERIMENT_ID, compute_metrics, freeze_and_record
from ami.warehouse.schema import connect, init_schema

NOW = 0


def _insert_event(conn, eid, event_count=1, censor="COMPLETED"):
    conn.execute(
        "INSERT INTO ami_events (event_id, event_family, symbol, anchor_ts_ms, source_quality, "
        "event_definition_version, censor_status, event_count, schema_version, provenance, created_ms, "
        "updated_ms) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        (eid, "FAM_A", "ETHUSDT", 0, "REAL_LIQUIDATION", "test-v1", censor, event_count, 6, "test", NOW, NOW),
    )


def _insert_cycle(conn, cid, direction_conflict=0, censored="COMPLETED"):
    conn.execute(
        "INSERT INTO ami_cycles (cycle_id, symbol, cycle_definition_version, event_count, "
        "direction_conflict, censored, confidence, schema_version, provenance, created_ms, updated_ms) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (cid, "ETHUSDT", "canonical-v1", 1, direction_conflict, censored, 1.0, 6, "test", NOW, NOW),
    )


def _insert_membership(conn, event_id, key, version):
    conn.execute(
        "INSERT INTO event_cycle_membership (event_id, candidate_cycle_key, cycle_definition_version, "
        "is_canonical, schema_version, provenance, created_ms) VALUES (?,?,?,0,?,?,?)",
        (event_id, key, version, 6, "test", NOW),
    )


def _db(tmp_path):
    conn = connect(tmp_path / "canonical.sqlite")
    init_schema(conn)
    return conn


def test_compute_metrics_raw_anchor_cycle_funnel(tmp_path):
    conn = _db(tmp_path)
    _insert_event(conn, "E1", event_count=1)
    _insert_event(conn, "E2", event_count=3)  # 3 logical trades collapsed into 1 anchor
    _insert_cycle(conn, "C1")
    conn.commit()

    metrics = compute_metrics(conn)
    assert metrics["anchor_n"] == 2
    assert metrics["raw_ledger_trades_n"] == 4  # 1 + 3
    assert metrics["cycle_n"] == 1
    assert metrics["multi_route_anchor_n"] == 1  # only E2 has event_count > 1
    assert metrics["event_count_distribution"] == {1: 1, 3: 1}


def test_compute_metrics_direction_conflict_and_censor_dist(tmp_path):
    conn = _db(tmp_path)
    _insert_event(conn, "E1", censor="COMPLETED")
    _insert_event(conn, "E2", censor="RIGHT_CENSORED")
    _insert_cycle(conn, "C1", direction_conflict=1, censored="COMPLETED")
    _insert_cycle(conn, "C2", direction_conflict=0, censored="RIGHT_CENSORED")
    conn.commit()

    metrics = compute_metrics(conn)
    assert metrics["direction_conflict_n"] == 1
    assert metrics["direction_conflict_rate"] == 0.5
    assert metrics["event_censor_distribution"] == {"COMPLETED": 1, "RIGHT_CENSORED": 1}
    assert metrics["cycle_censor_distribution"] == {"COMPLETED": 1, "RIGHT_CENSORED": 1}


def test_compute_metrics_sensitivity_window_counts(tmp_path):
    conn = _db(tmp_path)
    _insert_event(conn, "E1")
    _insert_event(conn, "E2")
    _insert_membership(conn, "E1", "key-A", "sensitivity-cooldown-1h-v1")
    _insert_membership(conn, "E2", "key-A", "sensitivity-cooldown-1h-v1")  # same episode -> 1 distinct key
    _insert_membership(conn, "E1", "key-B", "sensitivity-cooldown-4h-v1")
    _insert_membership(conn, "E2", "key-C", "sensitivity-cooldown-4h-v1")  # 2 distinct episodes
    conn.commit()

    metrics = compute_metrics(conn)
    assert metrics["sensitivity_window_counts"]["1h"] == 1
    assert metrics["sensitivity_window_counts"]["4h"] == 2
    assert metrics["sensitivity_window_counts"]["24h"] == 0  # no membership rows for this window


def test_freeze_and_record_writes_canonical_sql_not_only_markdown(tmp_path):
    conn = _db(tmp_path)
    _insert_event(conn, "E1")
    _insert_cycle(conn, "C1")
    conn.commit()

    freeze_and_record(conn)
    exp_row = conn.execute(
        "SELECT software_verdict, scientific_verdict FROM experiment_registry WHERE experiment_id=?",
        (EXPERIMENT_ID,),
    ).fetchone()
    n_results = conn.execute(
        "SELECT COUNT(*) FROM experiment_results WHERE experiment_id=?", (EXPERIMENT_ID,)
    ).fetchone()[0]
    conn.close()
    assert exp_row is not None
    assert n_results > 0  # actual metric rows exist in canonical SQL, not just a returned dict


def test_freeze_and_record_is_idempotent_no_duplicate_registry_row(tmp_path):
    conn = _db(tmp_path)
    _insert_event(conn, "E1")
    _insert_cycle(conn, "C1")
    conn.commit()

    freeze_and_record(conn)
    freeze_and_record(conn)  # re-run: same population
    n_registry = conn.execute(
        "SELECT COUNT(*) FROM experiment_registry WHERE experiment_id=?", (EXPERIMENT_ID,)
    ).fetchone()[0]
    n_results_first = conn.execute(
        "SELECT COUNT(*) FROM experiment_results WHERE experiment_id=? AND metric_name='cycle_n'",
        (EXPERIMENT_ID,),
    ).fetchone()[0]
    conn.close()
    assert n_registry == 1
    # each metric must appear exactly once per experiment -- a re-run replaces
    # the previous snapshot's result rows, it does not accumulate duplicates.
    assert n_results_first == 1


def test_compute_metrics_real_data_smoke():
    # Integration smoke test against the real, already-populated canonical.sqlite.
    # Writable (not read_only): fetch_* records a researcher_exposure_ledger
    # row as a side effect by design -- that is the audit trail F-B5/F3 exist for.
    from ami.warehouse.schema import DEFAULT_PATH, connect as real_connect

    conn = real_connect(DEFAULT_PATH)
    try:
        metrics = compute_metrics(conn)
    finally:
        conn.close()
    assert metrics["anchor_n"] > 0
    assert metrics["cycle_n"] > 0
    assert metrics["cycle_n"] <= metrics["anchor_n"]  # cycles can only merge anchors, never invent them
    assert metrics["anchor_n"] <= metrics["raw_ledger_trades_n"]
