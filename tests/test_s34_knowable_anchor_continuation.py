import sqlite3

import pytest

from tools.research_s34_knowable_anchor_continuation import (
    THRESHOLDS_USD,
    evaluate_anchor,
    feature_values,
    load_mark_index,
    reconstruct_anchors,
)
from tools.s34_feature_availability import FeatureClass, FeatureValue, LookaheadViolation, assert_feature_set_available


def test_reconstruct_anchors_uses_first_running_threshold_cross():
    t0 = 1_780_000_000_000
    liqs = [
        {"ts_ms": t0, "price": 100.0, "quantity": 1.0, "notional": 40_000.0, "trade_time_ms": t0},
        {"ts_ms": t0 + 10_000, "price": 100.0, "quantity": 1.0, "notional": 20_000.0, "trade_time_ms": t0 + 10_000},
        {"ts_ms": t0 + 20_000, "price": 100.0, "quantity": 1.0, "notional": 50_000.0, "trade_time_ms": t0 + 20_000},
    ]

    anchors = reconstruct_anchors(
        liqs,
        bucket_sec=300,
        min_gap_sec=0,
        thresholds=(50_000.0, 100_000.0),
        accel_window_sec=30,
    )

    by_threshold = {anchor.threshold_usd: anchor for anchor in anchors}
    assert by_threshold[50_000.0].anchor_ts_ms == t0 + 10_000
    assert by_threshold[50_000.0].running_notional == 60_000.0
    assert by_threshold[100_000.0].anchor_ts_ms == t0 + 20_000
    assert by_threshold[100_000.0].running_liq_count == 3
    assert all(feature.knowable_at_ts <= anchor.anchor_ts_ms for anchor in anchors for feature in feature_values(anchor))


def test_future_known_feature_gate_fails_for_knowable_anchor_context():
    with pytest.raises(LookaheadViolation):
        assert_feature_set_available(
            [
                FeatureValue(
                    name="future_depth",
                    value=1.0,
                    knowable_at_ts=1_780_000_010_000,
                    feature_class=FeatureClass.POINT_IN_TIME,
                )
            ],
            1_780_000_000_000,
            context="knowable_anchor_synthetic",
        )


def test_evaluate_anchor_reports_no_fill_counterfactual(tmp_path):
    db = tmp_path / "micro.db"
    con = sqlite3.connect(db)
    con.executescript(
        """
        CREATE TABLE mark_prices (ts_ms INTEGER, symbol TEXT, mark_price REAL);
        CREATE TABLE book_ticker (
            ts_ms INTEGER, symbol TEXT, bid_price REAL, bid_qty REAL,
            ask_price REAL, ask_qty REAL, mid_price REAL, spread_pct REAL, book_imbalance REAL
        );
        """
    )
    t0 = 1_780_000_000_000
    con.executemany(
        "INSERT INTO mark_prices(ts_ms,symbol,mark_price) VALUES (?,?,?)",
        [(t0, "ETHUSDT", 100.0), (t0 + 30_000, "ETHUSDT", 101.0)],
    )
    con.execute(
        "INSERT INTO book_ticker VALUES (?,?,?,?,?,?,?,?,?)",
        (t0 - 10_000, "ETHUSDT", 99.9, 1.0, 100.1, 1.0, 100.0, 0.2, 0.0),
    )
    con.commit()
    marks = load_mark_index(con, "ETHUSDT")
    anchor = reconstruct_anchors(
        [{"ts_ms": t0, "price": 100.0, "quantity": 1.0, "notional": THRESHOLDS_USD[0], "trade_time_ms": t0}],
        bucket_sec=300,
        min_gap_sec=0,
        thresholds=(THRESHOLDS_USD[0],),
        accel_window_sec=30,
    )[0]

    row = evaluate_anchor(
        con,
        marks,
        anchor,
        symbol="ETHUSDT",
        direction="LONG",
        horizon_sec=30,
        fee_bps_side=3.05,
        max_book_staleness_sec=5,
    )

    assert row["status"] == "NO_ENTRY_FILL"
    assert row["counterfactual_mark_net_bps"] > 0
    con.close()
