"""Tests for ami/cvd/windowed_taker_flow.py -- sign convention, boundary law,
known-at rejection, deterministic ordering, exact/proxy separation, BUCKET
determinism, matrix construction with no silent row dropping.

Fully synthetic: no network, no real database is opened by any test here.
"""
import json
import sqlite3

import pytest

from ami.cvd import windowed_taker_flow as wtf
from ami.cvd import cvd_rehearsal


def _mk_row(ts, price=100.0, qty=2.0, maker=0, source="LEGACY", key=1):
    return {"ts_ms": ts, "price": price, "quantity": qty, "notional": price * qty,
            "is_buyer_maker": maker, "source": source, "order_key": (source[0], key)}


def _mem():
    conn = sqlite3.connect(":memory:")
    wtf.init_schema(conn)
    return conn


# 1. taker-side sign convention
def test_taker_side_sign_convention():
    assert wtf.taker_sign(0) == 1      # taker BUY
    assert wtf.taker_sign(False) == 1
    assert wtf.taker_sign(1) == -1     # taker SELL
    assert wtf.taker_sign(True) == -1


# 2. signed quantity calculation
def test_signed_quantity_calculation():
    rows = [_mk_row(1000, qty=2.0, maker=0, key=1), _mk_row(1500, qty=5.0, maker=1, key=2)]
    flow = wtf.compute_window_flow(rows, 0, 2000)
    assert flow["cvd_qty"] == pytest.approx(2.0 - 5.0)
    assert flow["taker_buy_qty"] == pytest.approx(2.0)
    assert flow["taker_sell_qty"] == pytest.approx(5.0)


# 3. signed notional calculation
def test_signed_notional_calculation():
    rows = [_mk_row(1000, price=10.0, qty=2.0, maker=0, key=1),
            _mk_row(1500, price=20.0, qty=1.0, maker=1, key=2)]
    flow = wtf.compute_window_flow(rows, 0, 2000)
    assert flow["cvd_notional"] == pytest.approx(20.0 - 20.0)
    assert flow["total_notional"] == pytest.approx(40.0)
    assert flow["taker_buy_notional"] == pytest.approx(20.0)
    assert flow["taker_sell_notional"] == pytest.approx(20.0)


# 4. [T-W, T] boundary inclusivity -- BOTH ends inclusive
def test_window_boundary_inclusive_both_ends():
    T = 100_000
    start, end = wtf.window_bounds(T, "W60")
    assert (start, end) == (T - 60_000, T)
    rows = [_mk_row(start, key=1), _mk_row(end, key=2)]
    flow = wtf.compute_window_flow(rows, start, end)
    assert flow["source_row_count"] == 2  # boundary rows are IN


# 5. rejection of rows after T
def test_rejects_row_after_T():
    with pytest.raises(wtf.KnownAtViolation):
        wtf.compute_window_flow([_mk_row(2001)], 0, 2000)


def test_rejects_row_before_window_start():
    with pytest.raises(wtf.KnownAtViolation):
        wtf.compute_window_flow([_mk_row(999)], 1000, 2000)


# 6. same-timestamp deterministic ordering: manifest is input-order-invariant
def test_same_timestamp_deterministic_ordering():
    a = _mk_row(1000, price=1.0, qty=1.0, maker=0, key=1)
    b = _mk_row(1000, price=2.0, qty=1.0, maker=1, key=2)
    f1 = wtf.compute_window_flow([a, b], 0, 2000)
    f2 = wtf.compute_window_flow([b, a], 0, 2000)
    assert f1["source_row_manifest_sha256"] == f2["source_row_manifest_sha256"]
    assert f1["cvd_notional"] == f2["cvd_notional"]


def test_empty_window_yields_null_not_zero():
    flow = wtf.compute_window_flow([], 0, 2000)
    assert flow["source_row_count"] == 0
    assert flow["cvd_qty"] is None
    assert flow["cvd_notional"] is None
    assert flow["normalized_cvd"] is None


def test_normalized_cvd_defined_only_with_positive_denominator():
    rows = [_mk_row(1000, price=10.0, qty=1.0, maker=1, key=1)]
    flow = wtf.compute_window_flow(rows, 0, 2000)
    assert flow["normalized_cvd"] == pytest.approx(-1.0)


# 16. BUCKET-window determinism
def test_bucket_window_determinism():
    T = 500_000
    s1 = wtf.window_bounds(T, "BUCKET", bucket_start_ts_ms=450_000)
    s2 = wtf.window_bounds(T, "BUCKET", bucket_start_ts_ms=450_000)
    assert s1 == s2 == (450_000, T)


def test_bucket_requires_frozen_start():
    with pytest.raises(ValueError):
        wtf.window_bounds(100, "BUCKET")


def test_bucket_start_after_birth_rejected():
    with pytest.raises(wtf.KnownAtViolation):
        wtf.window_bounds(100_000, "BUCKET", bucket_start_ts_ms=100_001)


def test_unknown_window_id_rejected():
    with pytest.raises(ValueError):
        wtf.window_bounds(100_000, "W999")


# 13. exact/proxy separation
def test_pooling_guard_rejects_mixed_layers():
    with pytest.raises(AssertionError, match="POOLED_EVIDENCE_LAYERS_FORBIDDEN"):
        wtf.assert_not_pooled([{"evidence_layer": "EXACT"}, {"evidence_layer": "PROXY"}])
    wtf.assert_not_pooled([{"evidence_layer": "EXACT"}])  # single layer passes


def test_exact_table_check_rejects_proxy_layer_row():
    conn = _mem()
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO ami_cvd_windowed_flow (feature_id, feature_definition_version,"
            " raw_interpretation_version, quality_contract_version, signal_id, source_event_id,"
            " symbol, signal_birth_ts, window_id, window_start_ts_ms, window_end_ts_ms,"
            " evidence_layer, source_row_count, legacy_row_count, repair_row_count,"
            " source_row_manifest_sha256, source_regime_ids, repair_method,"
            " feature_available_ts_ms, known_at_classification, schema_version, provenance,"
            " created_ms) VALUES ('x','v','r','q','s','e','ETHUSDT',1000,'W60',0,1000,"
            " 'PROXY',0,0,0,'h','[]','NONE',1000,'KNOWN_AT_SAFE',1,'t',0)")


# proxy candle-close discipline: straddling candle excluded (no leakage)
def test_proxy_partial_candle_excluded():
    T = 120_000
    candles = [
        {"open_ts_ms": 0, "close_ts_ms": 59_999, "taker_buy_volume": 3.0,
         "taker_sell_volume": 1.0, "candle_definition_version": "v"},
        {"open_ts_ms": 60_000, "close_ts_ms": 119_999, "taker_buy_volume": 2.0,
         "taker_sell_volume": 2.0, "candle_definition_version": "v"},
        # closes AFTER T -> must never contribute
        {"open_ts_ms": 120_000, "close_ts_ms": 179_999, "taker_buy_volume": 99.0,
         "taker_sell_volume": 0.0, "candle_definition_version": "v"},
    ]
    p = wtf.compute_proxy_window_flow(candles, 0, T)
    assert p["contained_candle_count"] == 2
    assert p["proxy_cvd_qty"] == pytest.approx((3.0 - 1.0) + (2.0 - 2.0))
    assert p["last_contained_close_ts_ms"] == 119_999


def test_proxy_zero_candles_yields_null():
    p = wtf.compute_proxy_window_flow([], 0, 60_000)
    assert p["contained_candle_count"] == 0
    assert p["proxy_cvd_qty"] is None


# 10-adjacent: immutability of feature rows
def _base_exact_row(sig="SIG-A", window="W60", T=1_000_000):
    start, end = wtf.window_bounds(T, window)
    flow = wtf.compute_window_flow([_mk_row(T, key=1)], start, end)
    row = dict(flow)
    row.update({"signal_id": sig, "source_event_id": "EVT-A", "independent_cycle_id": "CYC-A",
                "symbol": "ETHUSDT", "signal_birth_ts": T, "window_id": window,
                "window_start_ts_ms": start, "window_end_ts_ms": end,
                "source_regime_ids": json.dumps(["R0"]), "repair_method": "NONE"})
    return row


def test_immutable_feature_conflict_and_noop():
    conn = _mem()
    row = _base_exact_row()
    assert wtf.insert_exact_feature_row(conn, row, provenance="t") == "INSERTED"
    assert wtf.insert_exact_feature_row(conn, row, provenance="t") == "NOOP_IDENTICAL"
    row2 = dict(row)
    row2["cvd_qty"] = (row["cvd_qty"] or 0) + 1.0
    with pytest.raises(wtf.ImmutableCvdFeatureConflict):
        wtf.insert_exact_feature_row(conn, row2, provenance="t")


# 19. feature_available_ts correctness (schema-level CHECK)
def test_feature_available_ts_must_equal_birth():
    conn = _mem()
    row = _base_exact_row()
    row["window_end_ts_ms"] = row["signal_birth_ts"] - 1  # break the law
    with pytest.raises(wtf.KnownAtViolation):
        wtf.insert_exact_feature_row(conn, row, provenance="t")


def test_schema_rejects_invalid_window_id():
    conn = _mem()
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO ami_cvd_windowed_flow (feature_id, feature_definition_version,"
            " raw_interpretation_version, quality_contract_version, signal_id, source_event_id,"
            " symbol, signal_birth_ts, window_id, window_start_ts_ms, window_end_ts_ms,"
            " evidence_layer, source_row_count, legacy_row_count, repair_row_count,"
            " source_row_manifest_sha256, source_regime_ids, repair_method,"
            " feature_available_ts_ms, known_at_classification, schema_version, provenance,"
            " created_ms) VALUES ('x','v','r','q','s','e','ETHUSDT',1000,'W999',0,1000,"
            " 'EXACT',0,0,0,'h','[]','NONE',1000,'KNOWN_AT_SAFE',1,'t',0)")


# 17+18. matrix construction logic + no silent row dropping (synthetic)
def _synthetic_environment():
    """3 signals: two with frozen bucket starts, one without (SHORT-like).
    Expected: 2*6 + 1*5 = 17 exact rows, 17 proxy rows, 1 explicit exclusion."""
    disp = sqlite3.connect(":memory:")
    micro = sqlite3.connect(":memory:")
    micro.execute("CREATE TABLE agg_trades (id INTEGER PRIMARY KEY, ts_ms INTEGER,"
                  " symbol TEXT, price REAL, quantity REAL, notional REAL,"
                  " is_buyer_maker INTEGER)")
    canon = sqlite3.connect(":memory:")
    canon.execute("CREATE TABLE ami_candles (symbol TEXT, timeframe TEXT, open_ts_ms INTEGER,"
                  " close_ts_ms INTEGER, taker_buy_volume REAL, taker_sell_volume REAL,"
                  " candle_definition_version TEXT)")
    base = 10 * 3600 * 1000  # far from regime boundaries -> single regime R0
    signals = []
    minute_set = set()
    for i, (sid, bucket) in enumerate([("SIG-L1", True), ("SIG-L2", True), ("SIG-S1", False)]):
        T = base + i * 7200_000
        signals.append({
            "signal_id": sid, "setup_id": "X", "direction": "LONG" if bucket else "SHORT",
            "source_event_id": f"EVT-{i}", "independent_cycle_id": f"CYC-{i}",
            "symbol": "ETHUSDT", "signal_birth_ts": T,
            "bucket_start_ts_ms": (T - 120_000) if bucket else None,
        })
        # trades: one per second across the whole hour before T
        rows = [(None, T - 3600_000 + k * 1000, "ETHUSDT", 100.0, 1.0, 100.0, k % 2)
                for k in range(3601)]
        micro.executemany("INSERT INTO agg_trades VALUES (?,?,?,?,?,?,?)", rows)
        for m in range((T - 3600_000) // 60000, T // 60000 + 1):
            minute_set.add(m * 60000)
        # 1m candles across the same hour
        for m in range((T - 3600_000) // 60000, T // 60000):
            canon.execute("INSERT INTO ami_candles VALUES ('ETHUSDT','1m',?,?,?,?,'candle-v1')",
                          (m * 60000, m * 60000 + 59_999, 30.0, 30.0))
    return disp, micro, canon, signals, minute_set, base


def test_matrix_construction_no_silent_row_dropping():
    disp, micro, canon, signals, minute_set, base = _synthetic_environment()
    counts = cvd_rehearsal.build_matrix(
        canon_ro=canon, micro_ro=micro, disp_conn=disp, signals=signals,
        minute_set=minute_set, repaired_minute_set=set(),
        cadence_threshold_ms=5000, cadence_proof_available=True,
        scan_min_ms=0, scan_max_ms=base + 10 * 7200_000,
        provenance="synthetic-test", assessment_version="test-a1")
    assert counts["exact_rows"] == 17
    assert counts["proxy_rows"] == 17
    assert counts["bucket_exclusions"] == 1
    assert counts["quality_rows"] == 17
    # accounting identity: exact rows + exclusions == signals * 6
    assert counts["exact_rows"] + counts["bucket_exclusions"] == len(signals) * 6
    # the excluded signal is explicitly recorded, with the frozen reason
    ex = disp.execute("SELECT signal_id, reason FROM ami_cvd_bucket_exclusions").fetchall()
    assert ex == [("SIG-S1", "BUCKET_WINDOW_NOT_FROZEN_FOR_SIGNAL")]
    # no outcome columns exist on the feature table
    cols = {r[1] for r in disp.execute("PRAGMA table_info(ami_cvd_windowed_flow)")}
    outcome_terms = ("mfe", "mae", "pnl", "win_rate", "outcome")
    assert not {c for c in cols if any(t in c.lower() for t in outcome_terms)}
    # timestamp-violation count must be 0
    assert cvd_rehearsal.timestamp_violation_count(disp) == 0
    # idempotent rerun: NOOP everywhere, content unchanged
    h1 = wtf.content_hash_exact(disp)
    counts2 = cvd_rehearsal.build_matrix(
        canon_ro=canon, micro_ro=micro, disp_conn=disp, signals=signals,
        minute_set=minute_set, repaired_minute_set=set(),
        cadence_threshold_ms=5000, cadence_proof_available=True,
        scan_min_ms=0, scan_max_ms=base + 10 * 7200_000,
        provenance="synthetic-test", assessment_version="test-a1")
    assert counts2["exact_rows"] == 0
    assert counts2["noop_identical"] == 17
    assert wtf.content_hash_exact(disp) == h1


def test_matrix_quality_all_exact_on_complete_synthetic_data():
    disp, micro, canon, signals, minute_set, base = _synthetic_environment()
    counts = cvd_rehearsal.build_matrix(
        canon_ro=canon, micro_ro=micro, disp_conn=disp, signals=signals,
        minute_set=minute_set, repaired_minute_set=set(),
        cadence_threshold_ms=5000, cadence_proof_available=True,
        scan_min_ms=0, scan_max_ms=base + 10 * 7200_000,
        provenance="synthetic-test", assessment_version="test-a1")
    assert counts["status_hist"] == {"EXACT_RECONSTRUCTABLE": 17}


def test_matrix_missing_minutes_fail_closed_without_repair():
    disp, micro, canon, signals, minute_set, base = _synthetic_environment()
    # remove ALL minute-coverage knowledge for signal 1's hour
    T = signals[0]["signal_birth_ts"]
    holed = {m for m in minute_set if not (T - 3600_000 <= m <= T)}
    counts = cvd_rehearsal.build_matrix(
        canon_ro=canon, micro_ro=micro, disp_conn=disp, signals=signals[:1],
        minute_set=holed, repaired_minute_set=set(),
        cadence_threshold_ms=5000, cadence_proof_available=True,
        scan_min_ms=0, scan_max_ms=base + 10 * 7200_000,
        provenance="synthetic-test", assessment_version="test-a1")
    # proxy candles exist -> PROXY_ONLY, never EXACT
    assert counts["status_hist"].get("EXACT_RECONSTRUCTABLE", 0) == 0
    assert set(counts["status_hist"]) <= {"PROXY_ONLY", "SOURCE_GAPPED"}
