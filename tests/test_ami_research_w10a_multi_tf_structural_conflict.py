"""BATCH-P6-011 (W10a): multi-timeframe structural conflict tests.

Run: pytest tests/test_ami_research_w10a_multi_tf_structural_conflict.py --basetemp <scratchpad> -p no:cacheprovider
"""
import sqlite3

from ami.research.w10a_multi_tf_structural_conflict import (
    EXPERIMENT_ID,
    RAW_CELLS,
    TF_FAST,
    TF_SLOW,
    _primary_contrast_test,
    classify_direction,
    classify_tf_cell,
    compute_metrics,
    freeze_and_record,
    primary_bucket_of_cell,
)
from ami.states.engine import StateEngine

HOUR_MS = 3_600_000
NOW = 0


def _mk_micro_db(path, eth_prices):
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE mark_prices (ts_ms INTEGER, symbol TEXT, mark_price REAL)")
    for ts, px in eth_prices:
        conn.execute("INSERT INTO mark_prices VALUES (?,?,?)", (ts, "ETHUSDT", px))
    conn.commit()
    conn.close()


# ---- cell classification (pure) ----

def test_classify_tf_cell_up_up():
    assert classify_tf_cell("UP", "UP") == "UP_UP"


def test_classify_tf_cell_down_down():
    assert classify_tf_cell("DOWN", "DOWN") == "DOWN_DOWN"


def test_classify_tf_cell_up_down_conflict():
    assert classify_tf_cell("UP", "DOWN") == "UP_DOWN"


def test_classify_tf_cell_down_up_conflict():
    assert classify_tf_cell("DOWN", "UP") == "DOWN_UP"


def test_classify_tf_cell_neutral_when_either_flat():
    assert classify_tf_cell("FLAT", "UP") == "NEUTRAL"
    assert classify_tf_cell("UP", "FLAT") == "NEUTRAL"
    assert classify_tf_cell("FLAT", "FLAT") == "NEUTRAL"
    assert classify_tf_cell("DOWN", "FLAT") == "NEUTRAL"


def test_all_five_raw_cells_reachable():
    combos = [("UP", "UP"), ("DOWN", "DOWN"), ("UP", "DOWN"), ("DOWN", "UP"), ("FLAT", "FLAT")]
    cells = {classify_tf_cell(a, b) for a, b in combos}
    assert cells == set(RAW_CELLS)


# ---- primary bucket exclusivity (pure) ----

def test_primary_bucket_agreement_for_up_up_and_down_down():
    assert primary_bucket_of_cell("UP_UP") == "AGREEMENT"
    assert primary_bucket_of_cell("DOWN_DOWN") == "AGREEMENT"


def test_primary_bucket_conflict_for_cross_directions():
    assert primary_bucket_of_cell("UP_DOWN") == "CONFLICT"
    assert primary_bucket_of_cell("DOWN_UP") == "CONFLICT"


def test_primary_bucket_none_for_neutral_never_merged():
    # NEUTRAL must never resolve to AGREEMENT or CONFLICT, at any N.
    assert primary_bucket_of_cell("NEUTRAL") is None


def test_primary_contrast_excludes_neutral_rows_entirely():
    rows = (
        [{"primary_bucket": "CONFLICT", "path_class": "REVERSAL", "cluster_id": f"c{i}"} for i in range(25)]
        + [{"primary_bucket": "AGREEMENT", "path_class": "CONTINUATION", "cluster_id": f"a{i}"} for i in range(25)]
        + [{"primary_bucket": None, "path_class": "REVERSAL", "cluster_id": f"n{i}"} for i in range(25)]
    )
    r = _primary_contrast_test(rows)
    # NEUTRAL rows (primary_bucket=None) must not appear on either side
    assert r["conflict"]["n"] == 25
    assert r["agreement"]["n"] == 25


def test_primary_contrast_insufficient_sample_flagged_not_merged():
    rows = (
        [{"primary_bucket": "CONFLICT", "path_class": "REVERSAL", "cluster_id": f"c{i}"} for i in range(5)]
        + [{"primary_bucket": "AGREEMENT", "path_class": "CONTINUATION", "cluster_id": f"a{i}"} for i in range(25)]
    )
    r = _primary_contrast_test(rows)
    assert r["conflict"]["insufficient_sample"] is True
    assert r["conflict"]["n"] == 5  # never folded into agreement
    assert r["agreement"]["insufficient_sample"] is False


# ---- known-at safety (StateEngine-backed) ----

def test_direction_classification_reads_existing_formula(tmp_path):
    # sanity: direction is one of the 3 allowed labels for both TFs used
    db = tmp_path / "micro.sqlite"
    anchor = 100 * HOUR_MS
    prices = [(anchor - k * HOUR_MS, 100.0 + k * 0.001) for k in range(0, 20)]
    _mk_micro_db(db, prices)
    engine = StateEngine(db_path=db)
    try:
        d_fast = classify_direction(engine, "ETHUSDT", anchor, TF_FAST)
        d_slow = classify_direction(engine, "ETHUSDT", anchor, TF_SLOW)
        assert d_fast in {"UP", "DOWN", "FLAT"}
        assert d_slow in {"UP", "DOWN", "FLAT"}
    finally:
        engine.conn.close()


def test_direction_classification_is_known_at_safe(tmp_path):
    # [CORRECTION 4] inserting a future, wildly-different mark_price row
    # (well after anchor_ts) must NOT change the direction computed at the
    # anchor -- proves no lookahead into an "unclosed" future price.
    db = tmp_path / "micro.sqlite"
    anchor = 100 * HOUR_MS
    prices = [(anchor - k * HOUR_MS, 100.0 + k * 0.5) for k in range(0, 20)]
    _mk_micro_db(db, prices)

    engine_before = StateEngine(db_path=db)
    try:
        d_fast_before = classify_direction(engine_before, "ETHUSDT", anchor, TF_FAST)
        d_slow_before = classify_direction(engine_before, "ETHUSDT", anchor, TF_SLOW)
    finally:
        engine_before.conn.close()

    # insert a future row, far after the anchor, with an extreme price
    conn = sqlite3.connect(db)
    conn.execute("INSERT INTO mark_prices VALUES (?,?,?)", (anchor + 50 * HOUR_MS, "ETHUSDT", 999_999.0))
    conn.commit()
    conn.close()

    engine_after = StateEngine(db_path=db)
    try:
        d_fast_after = classify_direction(engine_after, "ETHUSDT", anchor, TF_FAST)
        d_slow_after = classify_direction(engine_after, "ETHUSDT", anchor, TF_SLOW)
    finally:
        engine_after.conn.close()

    assert d_fast_before == d_fast_after
    assert d_slow_before == d_slow_after


# ---- real-data end-to-end + idempotency ----

def test_compute_metrics_real_data_smoke():
    from ami.warehouse.schema import DEFAULT_PATH, connect as real_connect

    conn = real_connect(DEFAULT_PATH)
    try:
        metrics = compute_metrics(conn)
    finally:
        conn.close()
    assert metrics["analyzed_n"] > 0
    assert metrics["independent_cycle_n"] > 0
    assert set(metrics["raw_cells"].keys()) == set(RAW_CELLS)
    total_cell_n = sum(c["n"] for c in metrics["raw_cells"].values())
    assert total_cell_n == metrics["analyzed_n"]
    # NEUTRAL must never leak into the primary contrast n's
    neutral_n = metrics["raw_cells"]["NEUTRAL"]["n"]
    primary_n = metrics["primary_contrast"]["conflict"]["n"] + metrics["primary_contrast"]["agreement"]["n"]
    assert primary_n == metrics["analyzed_n"] - neutral_n


def test_freeze_and_record_writes_canonical_sql_and_is_idempotent():
    from ami.warehouse.schema import DEFAULT_PATH, connect as real_connect, init_schema as real_init

    conn = real_connect(DEFAULT_PATH)
    try:
        real_init(conn)
        freeze_and_record(conn)
        freeze_and_record(conn)
        exp_row = conn.execute(
            "SELECT software_verdict, scientific_verdict, question_ids FROM experiment_registry WHERE experiment_id=?",
            (EXPERIMENT_ID,),
        ).fetchone()
        n_registry = conn.execute(
            "SELECT COUNT(*) FROM experiment_registry WHERE experiment_id=?", (EXPERIMENT_ID,)
        ).fetchone()[0]
        n_results = conn.execute(
            "SELECT COUNT(*) FROM experiment_results WHERE experiment_id=? AND metric_name='primary_contrast'",
            (EXPERIMENT_ID,),
        ).fetchone()[0]
    finally:
        conn.close()
    assert exp_row == ("PASSED", "ANSWERED_SUPPORTED", "FAM_MULTI_TF_CONFLICT")
    assert n_registry == 1
    assert n_results == 1
