import argparse
import sqlite3
from pathlib import Path

import pytest

import tools.s34_shadow_paper_runner as runner
from tools.s34_feature_availability import FeatureClass, FeatureValue, LookaheadViolation
from tools.s34_shadow_paper_runner import (
    DEFAULT_RISK,
    RegimeConfig,
    S34Rule,
    _bucket_events,
    _cluster_shape_label,
    _close_trade,
    _cost_decomposition,
    _evaluate_trade,
    _fill_quote,
    _paper_trade_from_signal,
    _regime_gate,
    _risk_gate,
    _risk_payload,
    run_once,
)


def _make_args(tmp_path: Path, db: Path, **overrides):
    base = {
        "db": str(db),
        "state_json": str(tmp_path / "state.json"),
        "trades_json": str(tmp_path / "trades.json"),
        "events_jsonl": str(tmp_path / "events.jsonl"),
        "summary_json": str(tmp_path / "summary.json"),
        "status_md": str(tmp_path / "status.md"),
        "journal_csv": str(tmp_path / "journal.csv"),
        "daily_report_md": str(tmp_path / "daily.md"),
        "start_utc": "",
        "end_utc": "",
        "backfill_existing": True,
        "backfill_restore_window": False,
        "limit_per_rule": 20,
        "loop": False,
        "interval_sec": 60,
        "dry_run": False,
        "simulated_equity_usdt": 100.0,
        "leverage": 10.0,
        "risk_per_trade_pct": 0.25,
        "max_open_trades": 1,
        "daily_max_loss_pct": 1.0,
        "daily_max_sl": 3,
        "cooldown_after_consecutive_sl": 2,
        "cooldown_hours": 6.0,
        "max_mark_staleness_sec": 30,
        "regime_filter_enabled": False,
        "regime_min_trend_pct": 1.0,
        "regime_min_range_pct": 2.5,
        "regime_min_buy_liq_notional": 5_000_000.0,
        "regime_min_agg_trade_count": 250_000,
        "taker_fee_bps": 4.0,
        "maker_fee_bps": 2.0,
        "tp_fill_mode": "taker",
        "modeled_spread_bps": 0.0,
        "max_book_staleness_sec": 5,
        "allow_modeled_spread_fill": False,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


def _build_db(path: Path, *, close_trade: bool = False) -> None:
    con = sqlite3.connect(path)
    con.executescript(
        """
        CREATE TABLE liquidations (
            id INTEGER PRIMARY KEY,
            ts_ms INTEGER NOT NULL,
            symbol TEXT NOT NULL,
            side TEXT NOT NULL,
            price REAL NOT NULL,
            quantity REAL NOT NULL,
            notional REAL NOT NULL,
            trade_time_ms INTEGER NOT NULL
        );
        CREATE TABLE mark_prices (
            id INTEGER PRIMARY KEY,
            ts_ms INTEGER NOT NULL,
            symbol TEXT NOT NULL,
            mark_price REAL NOT NULL,
            funding_rate REAL,
            next_funding_time_ms INTEGER
        );
        CREATE TABLE agg_trades (
            id INTEGER PRIMARY KEY,
            ts_ms INTEGER NOT NULL,
            symbol TEXT NOT NULL,
            price REAL NOT NULL,
            quantity REAL NOT NULL,
            notional REAL NOT NULL,
            is_buyer_maker INTEGER NOT NULL
        );
        CREATE TABLE book_ticker (
            id INTEGER PRIMARY KEY,
            ts_ms INTEGER NOT NULL,
            symbol TEXT NOT NULL,
            bid_price REAL NOT NULL,
            bid_qty REAL NOT NULL,
            ask_price REAL NOT NULL,
            ask_qty REAL NOT NULL,
            mid_price REAL NOT NULL,
            spread_pct REAL NOT NULL,
            book_imbalance REAL NOT NULL
        );
        """
    )
    t0 = 1_780_000_000_000
    con.execute("INSERT INTO mark_prices(ts_ms,symbol,mark_price) VALUES (?,?,?)", (t0 - 1000, "ETHUSDT", 100.0))
    con.execute(
        "INSERT INTO book_ticker(ts_ms,symbol,bid_price,bid_qty,ask_price,ask_qty,mid_price,spread_pct,book_imbalance) VALUES (?,?,?,?,?,?,?,?,?)",
        (t0 - 1000, "ETHUSDT", 99.99, 1.0, 100.01, 1.0, 100.0, 0.02, 0.0),
    )
    con.execute("INSERT INTO liquidations(ts_ms,symbol,side,price,quantity,notional,trade_time_ms) VALUES (?,?,?,?,?,?,?)", (t0, "ETHUSDT", "BUY", 101.0, 1.0, 250_000.0, t0))
    if close_trade:
        con.execute("INSERT INTO mark_prices(ts_ms,symbol,mark_price) VALUES (?,?,?)", (t0 + 60_000, "ETHUSDT", 101.0))
        con.execute(
            "INSERT INTO book_ticker(ts_ms,symbol,bid_price,bid_qty,ask_price,ask_qty,mid_price,spread_pct,book_imbalance) VALUES (?,?,?,?,?,?,?,?,?)",
            (t0 + 60_000, "ETHUSDT", 100.99, 1.0, 101.01, 1.0, 101.0, 0.02, 0.0),
        )
    else:
        con.execute("INSERT INTO mark_prices(ts_ms,symbol,mark_price) VALUES (?,?,?)", (t0 + 60_000, "ETHUSDT", 100.1))
        con.execute(
            "INSERT INTO book_ticker(ts_ms,symbol,bid_price,bid_qty,ask_price,ask_qty,mid_price,spread_pct,book_imbalance) VALUES (?,?,?,?,?,?,?,?,?)",
            (t0 + 60_000, "ETHUSDT", 100.09, 1.0, 100.11, 1.0, 100.1, 0.02, 0.0),
        )
    con.commit()
    con.close()


def test_risk_payload_uses_bps_stop_and_leverage():
    payload = _risk_payload(S34Rule(name="x", sl_bps=40.0), DEFAULT_RISK)

    assert payload["risk_usdt"] == 0.25
    assert payload["notional_usdt"] == 62.5
    assert payload["margin_required_usdt"] == 6.25
    assert payload["leverage"] == 10.0


def test_runner_dedupes_same_cluster_by_rule_priority(tmp_path):
    db = tmp_path / "microstructure.db"
    _build_db(db, close_trade=False)

    summary = run_once(_make_args(tmp_path, db))

    assert summary["new_signals"] == 2
    assert summary["new_trades"] == 1
    assert summary["skipped_trades"] == 1
    journal = (tmp_path / "journal.csv").read_text(encoding="utf-8")
    assert "P001" in journal
    assert "P002" in journal
    assert "ETH_BUY_LIQ_LONG_200K_TP60_SL40_BE30" in journal
    assert "ARCHIVED_CONTAMINATED_LOOKAHEAD_ANCHOR" in journal
    assert "SYSTEM_GENERATED_PAPER_ONLY_NOT_MANUAL" in journal
    assert "MAX_OPEN_TRADES" not in journal
    assert "notional_usdt" in journal


def test_runner_writes_intelligence_ledger(tmp_path):
    db = tmp_path / "microstructure.db"
    ledger = tmp_path / "s34_intelligence.db"
    _build_db(db, close_trade=False)

    summary = run_once(_make_args(tmp_path, db, intelligence_db=str(ledger)))

    assert summary["new_signals"] == 2
    con = sqlite3.connect(ledger)
    try:
        assert con.execute("SELECT COUNT(*) FROM s34_signals").fetchone()[0] == 2
        assert con.execute("SELECT COUNT(*) FROM s34_decisions WHERE decision='ACCEPT'").fetchone()[0] == 1
        assert con.execute("SELECT COUNT(*) FROM s34_decisions WHERE decision='REJECT'").fetchone()[0] == 1
        assert con.execute("SELECT COUNT(*) FROM s34_trades").fetchone()[0] == 1
        assert (
            con.execute("SELECT reason FROM s34_rejected_signals").fetchone()[0]
            == "ARCHIVED_CONTAMINATED_LOOKAHEAD_ANCHOR"
        )
    finally:
        con.close()


def test_50k_and_500k_buy_continuation_rules_are_archived_contaminated():
    assert runner._deprecated_paper_rule_reason(S34Rule(name="ETH_BUY_LIQ_LONG_50K_TP120_SL40_BE30"))
    assert runner._deprecated_paper_rule_reason(S34Rule(name="ETH_BUY_LIQ_LONG_500K_DAYTREND0_TP60_SL40_BE30"))


def test_paper_trade_feature_gate_raises_on_future_known_feature(monkeypatch):
    ts = 1_780_000_000_000

    def future_feature(_signal):
        return [
            FeatureValue(
                name="cluster_notional",
                value=500_000.0,
                knowable_at_ts=ts + 60_000,
                feature_class=FeatureClass.TERMINAL_CLUSTER,
            )
        ]

    monkeypatch.setattr(runner, "signal_entry_features", future_feature)
    with pytest.raises(LookaheadViolation):
        _paper_trade_from_signal(
            S34Rule(name="TEST_RULE"),
            {
                "bucket": 1,
                "ts_ms": ts,
                "ts_utc": "2026-06-10T00:00:00+00:00",
                "entry_price": 100.0,
            },
            DEFAULT_RISK,
        )


def test_risk_gate_blocks_same_symbol_direction_open_across_rules():
    ts = 1_780_000_000_000
    signal = {"ts_ms": ts, "mark_ts_ms": ts}
    rule_50k = S34Rule(name="ETH_BUY_LIQ_LONG_50K_TP120_SL40_BE30", max_open_trades=1)
    rule_200k = S34Rule(name="ETH_BUY_LIQ_LONG_200K_TP60_SL40_BE30", max_open_trades=1)
    sol_rule = S34Rule(name="SOL_BUY_LIQ_LONG_200K_TP60_SL40_BE30", symbol="SOLUSDT", max_open_trades=1)
    trades = {
        "P001": {
            "status": "OPEN",
            "symbol": "ETHUSDT",
            "direction": "LONG",
            "rule": {"name": "ETH_BUY_LIQ_LONG_200K_TP60_SL40_BE30"},
        }
    }

    assert _risk_gate(trades, signal, DEFAULT_RISK, rule_50k) == (False, "MAX_SYMBOL_DIRECTION_OPEN_TRADES")
    assert _risk_gate(trades, signal, DEFAULT_RISK, rule_200k) == (False, "MAX_OPEN_TRADES")
    assert _risk_gate(trades, signal, DEFAULT_RISK, sol_rule) == (True, "")


def test_daily_max_sl_is_rule_scoped_but_daily_loss_stays_global():
    ts = 1_780_000_000_000
    signal = {"ts_ms": ts, "mark_ts_ms": ts}
    rule_50k = S34Rule(name="ETH_BUY_LIQ_LONG_50K_TP120_SL40_BE30", daily_max_sl=1)
    rule_200k = S34Rule(name="ETH_BUY_LIQ_LONG_200K_TP60_SL40_BE30", daily_max_sl=1)
    trades = {
        "P001": {
            "status": "CLOSED",
            "rule": {"name": "ETH_BUY_LIQ_LONG_200K_TP60_SL40_BE30"},
            "exit_reason": "SL",
            "exit_ts_ms": ts - 1000,
            "net_usdt": -0.1,
        }
    }

    assert _risk_gate(trades, signal, DEFAULT_RISK, rule_50k) == (True, "")
    assert _risk_gate(trades, signal, DEFAULT_RISK, rule_200k) == (False, "DAILY_MAX_SL")


def test_runner_is_idempotent_by_signal_key(tmp_path):
    db = tmp_path / "microstructure.db"
    _build_db(db, close_trade=True)
    args = _make_args(tmp_path, db)

    first = run_once(args)
    second = run_once(args)

    assert first["new_signals"] == 2
    assert second["new_signals"] == 0


def test_regime_filter_skips_when_day_so_far_thresholds_fail(tmp_path):
    db = tmp_path / "microstructure.db"
    _build_db(db, close_trade=True)
    args = _make_args(tmp_path, db, regime_filter_enabled=True)

    summary = run_once(args)

    assert summary["new_signals"] == 2
    assert summary["new_trades"] == 0
    assert summary["skipped_trades"] == 2
    journal = (tmp_path / "journal.csv").read_text(encoding="utf-8")
    assert "REGIME_FILTER" in journal


def test_rule_specific_day_trend_gate_does_not_inherit_global_regime(tmp_path):
    db = tmp_path / "microstructure.db"
    con = sqlite3.connect(db)
    con.executescript(
        """
        CREATE TABLE mark_prices (
            id INTEGER PRIMARY KEY,
            ts_ms INTEGER NOT NULL,
            symbol TEXT NOT NULL,
            mark_price REAL NOT NULL
        );
        CREATE TABLE liquidations (
            id INTEGER PRIMARY KEY,
            ts_ms INTEGER NOT NULL,
            symbol TEXT NOT NULL,
            side TEXT NOT NULL,
            price REAL NOT NULL,
            quantity REAL NOT NULL,
            notional REAL NOT NULL,
            trade_time_ms INTEGER NOT NULL
        );
        CREATE TABLE agg_trades (
            id INTEGER PRIMARY KEY,
            ts_ms INTEGER NOT NULL,
            symbol TEXT NOT NULL,
            price REAL NOT NULL,
            quantity REAL NOT NULL,
            notional REAL NOT NULL,
            is_buyer_maker INTEGER NOT NULL
        );
        """
    )
    t0 = 1_780_000_000_000
    day_start = t0 - (t0 % 86_400_000)
    con.executemany(
        "INSERT INTO mark_prices(ts_ms,symbol,mark_price) VALUES (?,?,?)",
        [
            (day_start + 1_000, "ETHUSDT", 100.0),
            (t0, "ETHUSDT", 100.1),
        ],
    )
    con.execute(
        "INSERT INTO liquidations(ts_ms,symbol,side,price,quantity,notional,trade_time_ms) VALUES (?,?,?,?,?,?,?)",
        (t0, "ETHUSDT", "BUY", 100.1, 1.0, 500_000.0, t0),
    )
    con.execute(
        "INSERT INTO agg_trades(ts_ms,symbol,price,quantity,notional,is_buyer_maker) VALUES (?,?,?,?,?,?)",
        (t0, "ETHUSDT", 100.1, 1.0, 100.1, 0),
    )
    config = RegimeConfig(enabled=True, min_trend_pct=1.0, min_range_pct=2.5, min_buy_liq_notional=5_000_000.0, min_agg_trade_count=250_000)
    signal = {"ts_ms": t0}

    prereg_rule = S34Rule(name="ETH_BUY_LIQ_LONG_50K_TP120_SL40_BE30")
    exploratory_rule = S34Rule(
        name="ETH_BUY_LIQ_LONG_500K_DAYTREND0_TP60_SL40_BE30",
        threshold_usd=500_000.0,
        tp_bps=60.0,
        use_global_regime=False,
        min_day_trend_bps=0.0,
    )

    prereg_ok, prereg_reason, prereg_snap = _regime_gate(con, prereg_rule, signal, config)
    exploratory_ok, exploratory_reason, exploratory_snap = _regime_gate(con, exploratory_rule, signal, config)

    assert prereg_ok is False
    assert prereg_reason == "REGIME_FILTER"
    assert prereg_snap["checks"]["trend_pct_gte"] is False
    assert exploratory_ok is True
    assert exploratory_reason == "REGIME_FILTER"
    assert exploratory_snap["checks"] == {"day_trend_bps_gte": True}
    assert exploratory_snap["thresholds"]["rule_regime"]["min_day_trend_bps"] == 0.0
    con.close()


def test_cluster_shape_label_matches_feature_factory_definition():
    assert _cluster_shape_label(10.0, 85.0) == "single_dominant_80pct"
    assert _cluster_shape_label(120.0, 79.9) == "stretched_120s"
    assert _cluster_shape_label(119.9, 79.9) == "distributed_mid_duration"


def test_bucket_events_signal_time_is_threshold_cross_not_bucket_start(tmp_path):
    db = tmp_path / "microstructure.db"
    con = sqlite3.connect(db)
    con.executescript(
        """
        CREATE TABLE liquidations (
            id INTEGER PRIMARY KEY,
            ts_ms INTEGER NOT NULL,
            symbol TEXT NOT NULL,
            side TEXT NOT NULL,
            price REAL NOT NULL,
            quantity REAL NOT NULL,
            notional REAL NOT NULL,
            trade_time_ms INTEGER NOT NULL
        );
        CREATE TABLE mark_prices (
            id INTEGER PRIMARY KEY,
            ts_ms INTEGER NOT NULL,
            symbol TEXT NOT NULL,
            mark_price REAL NOT NULL
        );
        CREATE TABLE book_ticker (
            id INTEGER PRIMARY KEY,
            ts_ms INTEGER NOT NULL,
            symbol TEXT NOT NULL,
            bid_price REAL NOT NULL,
            bid_qty REAL NOT NULL,
            ask_price REAL NOT NULL,
            ask_qty REAL NOT NULL,
            mid_price REAL NOT NULL,
            spread_pct REAL NOT NULL,
            book_imbalance REAL NOT NULL
        );
        """
    )
    t0 = 1_780_000_000_000
    liqs = [
        (t0, "ETHUSDT", "SELL", 100.0, 1.0, 100_000.0, t0),
        (t0 + 10_000, "ETHUSDT", "SELL", 99.5, 1.0, 150_000.0, t0 + 10_000),
        (t0 + 20_000, "ETHUSDT", "SELL", 99.0, 1.0, 300_000.0, t0 + 20_000),
        (t0 + 30_000, "ETHUSDT", "SELL", 98.0, 1.0, 1_000_000.0, t0 + 30_000),
    ]
    con.executemany(
        "INSERT INTO liquidations(ts_ms,symbol,side,price,quantity,notional,trade_time_ms) VALUES (?,?,?,?,?,?,?)",
        liqs,
    )
    con.executemany(
        "INSERT INTO mark_prices(ts_ms,symbol,mark_price) VALUES (?,?,?)",
        [
            (t0, "ETHUSDT", 100.0),
            (t0 + 20_000, "ETHUSDT", 99.0),
            (t0 + 30_000, "ETHUSDT", 98.0),
        ],
    )
    con.execute(
        "INSERT INTO book_ticker(ts_ms,symbol,bid_price,bid_qty,ask_price,ask_qty,mid_price,spread_pct,book_imbalance) VALUES (?,?,?,?,?,?,?,?,?)",
        (t0 + 20_000, "ETHUSDT", 98.99, 1.0, 99.01, 1.0, 99.0, 0.02, 0.0),
    )
    con.commit()

    rule = S34Rule(
        name="ETH_SELL_LIQ_SHORT_500K_TP60_SL40_BE40",
        liq_side="SELL",
        direction="SHORT",
        threshold_usd=500_000.0,
        tp_bps=60.0,
        be_trigger_bps=40.0,
    )

    signals = _bucket_events(con, rule, t0 - 1, t0 + 300_000, 10)

    assert len(signals) == 1
    signal = signals[0]
    assert signal["ts_ms"] == t0 + 20_000
    assert signal["entry_ts_ms"] == t0 + 20_000
    assert signal["cluster_start_ts_ms"] == t0
    assert signal["cluster_end_ts_ms"] == t0 + 20_000
    assert signal["threshold_cross_ts_ms"] == t0 + 20_000
    assert signal["liq_total_notional"] == 550_000.0
    assert signal["liq_count"] == 3
    assert signal["liq_max_notional"] == 300_000.0
    assert signal["entry_reference_price"] == 99.0
    assert signal["entry_price"] == 98.99
    con.close()


def test_bucket_events_filters_stretched_cluster_shape(tmp_path):
    db = tmp_path / "microstructure.db"
    con = sqlite3.connect(db)
    con.executescript(
        """
        CREATE TABLE liquidations (
            id INTEGER PRIMARY KEY,
            ts_ms INTEGER NOT NULL,
            symbol TEXT NOT NULL,
            side TEXT NOT NULL,
            price REAL NOT NULL,
            quantity REAL NOT NULL,
            notional REAL NOT NULL,
            trade_time_ms INTEGER NOT NULL
        );
        CREATE TABLE mark_prices (
            id INTEGER PRIMARY KEY,
            ts_ms INTEGER NOT NULL,
            symbol TEXT NOT NULL,
            mark_price REAL NOT NULL
        );
        CREATE TABLE book_ticker (
            id INTEGER PRIMARY KEY,
            ts_ms INTEGER NOT NULL,
            symbol TEXT NOT NULL,
            bid_price REAL NOT NULL,
            bid_qty REAL NOT NULL,
            ask_price REAL NOT NULL,
            ask_qty REAL NOT NULL,
            mid_price REAL NOT NULL,
            spread_pct REAL NOT NULL,
            book_imbalance REAL NOT NULL
        );
        """
    )
    t0 = 1_780_000_000_000
    for offset in (0, 60_000, 130_000):
        con.execute(
            "INSERT INTO liquidations(ts_ms,symbol,side,price,quantity,notional,trade_time_ms) VALUES (?,?,?,?,?,?,?)",
            (t0 + offset, "ETHUSDT", "BUY", 100.0, 1.0, 200_000.0, t0 + offset),
        )
    con.executemany(
        "INSERT INTO mark_prices(ts_ms,symbol,mark_price) VALUES (?,?,?)",
        [(t0, "ETHUSDT", 100.0), (t0 + 130_000, "ETHUSDT", 100.0)],
    )
    con.executemany(
        "INSERT INTO book_ticker(ts_ms,symbol,bid_price,bid_qty,ask_price,ask_qty,mid_price,spread_pct,book_imbalance) VALUES (?,?,?,?,?,?,?,?,?)",
        [
            (t0, "ETHUSDT", 99.99, 1.0, 100.01, 1.0, 100.0, 0.02, 0.0),
            (t0 + 130_000, "ETHUSDT", 99.99, 1.0, 100.01, 1.0, 100.0, 0.02, 0.0),
        ],
    )
    con.commit()

    rule = S34Rule(
        name="ETH_BUY_LIQ_LONG_500K_NEGTREND_STRETCHED_TP60_SL40_BE30",
        threshold_usd=500_000.0,
        tp_bps=60.0,
        use_global_regime=False,
        max_day_trend_bps=0.0,
        required_shape_label="stretched_120s",
    )

    signals = _bucket_events(con, rule, t0 - 1, t0 + 300_000, 10)

    assert len(signals) == 1
    assert signals[0]["cluster_shape_label"] == "stretched_120s"
    assert signals[0]["cluster_duration_sec"] == 130.0
    assert round(signals[0]["cluster_max_single_liq_share"], 2) == 33.33
    con.close()


def test_bucket_events_filters_distributed_cluster_by_max_single_share(tmp_path):
    db = tmp_path / "microstructure.db"
    con = sqlite3.connect(db)
    con.executescript(
        """
        CREATE TABLE liquidations (
            id INTEGER PRIMARY KEY,
            ts_ms INTEGER NOT NULL,
            symbol TEXT NOT NULL,
            side TEXT NOT NULL,
            price REAL NOT NULL,
            quantity REAL NOT NULL,
            notional REAL NOT NULL,
            trade_time_ms INTEGER NOT NULL
        );
        CREATE TABLE mark_prices (
            id INTEGER PRIMARY KEY,
            ts_ms INTEGER NOT NULL,
            symbol TEXT NOT NULL,
            mark_price REAL NOT NULL
        );
        CREATE TABLE book_ticker (
            id INTEGER PRIMARY KEY,
            ts_ms INTEGER NOT NULL,
            symbol TEXT NOT NULL,
            bid_price REAL NOT NULL,
            bid_qty REAL NOT NULL,
            ask_price REAL NOT NULL,
            ask_qty REAL NOT NULL,
            mid_price REAL NOT NULL,
            spread_pct REAL NOT NULL,
            book_imbalance REAL NOT NULL
        );
        """
    )
    t0 = 1_780_000_000_000
    for offset in (0, 60_000, 130_000):
        con.execute(
            "INSERT INTO liquidations(ts_ms,symbol,side,price,quantity,notional,trade_time_ms) VALUES (?,?,?,?,?,?,?)",
            (t0 + offset, "BTCUSDT", "BUY", 100_000.0, 4.0, 400_000.0, t0 + offset),
        )
    con.executemany(
        "INSERT INTO mark_prices(ts_ms,symbol,mark_price) VALUES (?,?,?)",
        [(t0, "BTCUSDT", 100_000.0), (t0 + 130_000, "BTCUSDT", 100_000.0)],
    )
    con.executemany(
        "INSERT INTO book_ticker(ts_ms,symbol,bid_price,bid_qty,ask_price,ask_qty,mid_price,spread_pct,book_imbalance) VALUES (?,?,?,?,?,?,?,?,?)",
        [
            (t0, "BTCUSDT", 99_999.0, 1.0, 100_001.0, 1.0, 100_000.0, 0.002, 0.0),
            (t0 + 130_000, "BTCUSDT", 99_999.0, 1.0, 100_001.0, 1.0, 100_000.0, 0.002, 0.0),
        ],
    )
    con.commit()

    distributed = S34Rule(
        name="BTC_BUY_LIQ_LONG_1M_DISTRIBUTED_TP60_SL30_BE30",
        symbol="BTCUSDT",
        threshold_usd=1_000_000.0,
        max_single_liq_share_pct=50.0,
    )
    too_strict = S34Rule(
        name="BTC_BUY_LIQ_LONG_1M_DISTRIBUTED_TP60_SL30_BE30",
        symbol="BTCUSDT",
        threshold_usd=1_000_000.0,
        max_single_liq_share_pct=30.0,
    )

    assert len(_bucket_events(con, distributed, t0 - 1, t0 + 300_000, 10)) == 1
    assert len(_bucket_events(con, too_strict, t0 - 1, t0 + 300_000, 10)) == 0
    con.close()


def test_counter_regime_rule_requires_negative_day_trend(tmp_path):
    db = tmp_path / "microstructure.db"
    con = sqlite3.connect(db)
    con.executescript(
        """
        CREATE TABLE mark_prices (
            id INTEGER PRIMARY KEY,
            ts_ms INTEGER NOT NULL,
            symbol TEXT NOT NULL,
            mark_price REAL NOT NULL
        );
        CREATE TABLE liquidations (
            id INTEGER PRIMARY KEY,
            ts_ms INTEGER NOT NULL,
            symbol TEXT NOT NULL,
            side TEXT NOT NULL,
            price REAL NOT NULL,
            quantity REAL NOT NULL,
            notional REAL NOT NULL,
            trade_time_ms INTEGER NOT NULL
        );
        CREATE TABLE agg_trades (
            id INTEGER PRIMARY KEY,
            ts_ms INTEGER NOT NULL,
            symbol TEXT NOT NULL,
            price REAL NOT NULL,
            quantity REAL NOT NULL,
            notional REAL NOT NULL,
            is_buyer_maker INTEGER NOT NULL
        );
        """
    )
    t0 = 1_780_000_000_000
    day_start = t0 - (t0 % 86_400_000)
    con.executemany(
        "INSERT INTO mark_prices(ts_ms,symbol,mark_price) VALUES (?,?,?)",
        [(day_start + 1_000, "ETHUSDT", 100.0), (t0, "ETHUSDT", 99.5)],
    )
    con.execute(
        "INSERT INTO liquidations(ts_ms,symbol,side,price,quantity,notional,trade_time_ms) VALUES (?,?,?,?,?,?,?)",
        (t0, "ETHUSDT", "BUY", 99.5, 1.0, 500_000.0, t0),
    )
    con.execute(
        "INSERT INTO agg_trades(ts_ms,symbol,price,quantity,notional,is_buyer_maker) VALUES (?,?,?,?,?,?)",
        (t0, "ETHUSDT", 99.5, 1.0, 99.5, 0),
    )
    config = RegimeConfig(enabled=True)
    rule = S34Rule(
        name="ETH_BUY_LIQ_LONG_500K_NEGTREND_STRETCHED_TP60_SL40_BE30",
        threshold_usd=500_000.0,
        tp_bps=60.0,
        use_global_regime=False,
        max_day_trend_bps=0.0,
    )

    ok, reason, snap = _regime_gate(con, rule, {"ts_ms": t0}, config)

    assert ok is True
    assert reason == "REGIME_FILTER"
    assert snap["checks"] == {"day_trend_bps_lte": True}
    assert snap["day_trend_bps"] < 0
    con.close()


def test_btc_pre_filter_and_entry_delay_are_no_lookahead(tmp_path):
    db = tmp_path / "microstructure.db"
    con = sqlite3.connect(db)
    con.executescript(
        """
        CREATE TABLE liquidations (
            id INTEGER PRIMARY KEY,
            ts_ms INTEGER NOT NULL,
            symbol TEXT NOT NULL,
            side TEXT NOT NULL,
            price REAL NOT NULL,
            quantity REAL NOT NULL,
            notional REAL NOT NULL,
            trade_time_ms INTEGER NOT NULL
        );
        CREATE TABLE mark_prices (
            id INTEGER PRIMARY KEY,
            ts_ms INTEGER NOT NULL,
            symbol TEXT NOT NULL,
            mark_price REAL NOT NULL
        );
        CREATE TABLE book_ticker (
            id INTEGER PRIMARY KEY,
            ts_ms INTEGER NOT NULL,
            symbol TEXT NOT NULL,
            bid_price REAL NOT NULL,
            bid_qty REAL NOT NULL,
            ask_price REAL NOT NULL,
            ask_qty REAL NOT NULL,
            mid_price REAL NOT NULL,
            spread_pct REAL NOT NULL,
            book_imbalance REAL NOT NULL
        );
        """
    )
    t0 = 1_780_000_000_000
    con.execute("INSERT INTO liquidations(ts_ms,symbol,side,price,quantity,notional,trade_time_ms) VALUES (?,?,?,?,?,?,?)", (t0, "ETHUSDT", "BUY", 100.0, 1.0, 250_000.0, t0))
    con.executemany(
        "INSERT INTO mark_prices(ts_ms,symbol,mark_price) VALUES (?,?,?)",
        [
            (t0 - 900_000, "BTCUSDT", 1000.0),
            (t0, "BTCUSDT", 1001.0),
            (t0, "ETHUSDT", 100.0),
            (t0 + 60_000, "ETHUSDT", 101.0),
        ],
    )
    con.execute(
        "INSERT INTO book_ticker(ts_ms,symbol,bid_price,bid_qty,ask_price,ask_qty,mid_price,spread_pct,book_imbalance) VALUES (?,?,?,?,?,?,?,?,?)",
        (t0 + 60_000, "ETHUSDT", 100.99, 1.0, 101.01, 1.0, 101.0, 0.02, 0.0),
    )
    rule = S34Rule(
        name="ETH_BUY_LIQ_LONG_200K_BTC_PRE15_TP120_SL40_BE30_DELAY60",
        threshold_usd=200_000.0,
        entry_delay_sec=60,
        btc_pre_window_sec=900,
        btc_pre_min_return_bps=0.0,
    )

    signals = _bucket_events(con, rule, t0 - 1, t0 + 1, 20)

    assert len(signals) == 1
    assert signals[0]["entry_ts_ms"] == t0 + 60_000
    assert signals[0]["entry_reference_price"] == 101.0
    assert signals[0]["entry_price"] == 101.01
    assert signals[0]["btc_pre_return_bps"] > 0

    con.execute("UPDATE mark_prices SET mark_price=? WHERE symbol='BTCUSDT' AND ts_ms=?", (999.0, t0))
    assert _bucket_events(con, rule, t0 - 1, t0 + 1, 20) == []
    con.close()


def test_evaluate_trade_does_not_apply_persisted_be_to_prior_marks(tmp_path):
    db = tmp_path / "microstructure.db"
    con = sqlite3.connect(db)
    con.executescript(
        """
        CREATE TABLE mark_prices (
            id INTEGER PRIMARY KEY,
            ts_ms INTEGER NOT NULL,
            symbol TEXT NOT NULL,
            mark_price REAL NOT NULL
        );
        """
    )
    t0 = 1_780_000_000_000
    marks = [
        (t0 + 1_000, "ETHUSDT", 99.99),
        (t0 + 2_000, "ETHUSDT", 100.31),
        (t0 + 3_000, "ETHUSDT", 99.99),
    ]
    con.executemany("INSERT INTO mark_prices(ts_ms,symbol,mark_price) VALUES (?,?,?)", marks)
    con.commit()

    rule = S34Rule(
        name="test",
        threshold_usd=50_000.0,
        tp_bps=120.0,
        sl_bps=40.0,
        be_trigger_bps=30.0,
        require_book_ticker_fill=False,
    )
    trade = _paper_trade_from_signal(
        rule,
        {
            "bucket": 1,
            "ts_ms": t0,
            "ts_utc": "2026-06-10T00:00:00+00:00",
            "entry_price": 100.0,
        },
        DEFAULT_RISK,
    )
    trade["trade_id"] = "PTEST"

    trade = _evaluate_trade(con, trade, t0 + 2_000)

    assert trade["status"] == "OPEN"
    assert trade["be_active"] is True
    assert trade["be_activated_ts_ms"] == t0 + 2_000
    assert trade["last_evaluated_mark_ts_ms"] == t0 + 2_000

    trade = _evaluate_trade(con, trade, t0 + 3_000)

    assert trade["status"] == "CLOSED"
    assert trade["exit_reason"] == "BE"
    assert trade["exit_ts_ms"] == t0 + 3_000
    assert trade["exit_ts_ms"] >= trade["be_activated_ts_ms"]
    con.close()


def test_fill_quote_long_entry_uses_ask(tmp_path):
    db = tmp_path / "microstructure.db"
    con = sqlite3.connect(db)
    con.executescript(
        """
        CREATE TABLE book_ticker (
            id INTEGER PRIMARY KEY,
            ts_ms INTEGER NOT NULL,
            symbol TEXT NOT NULL,
            bid_price REAL NOT NULL,
            bid_qty REAL NOT NULL,
            ask_price REAL NOT NULL,
            ask_qty REAL NOT NULL,
            mid_price REAL NOT NULL,
            spread_pct REAL NOT NULL,
            book_imbalance REAL NOT NULL
        );
        """
    )
    con.execute(
        "INSERT INTO book_ticker(ts_ms,symbol,bid_price,bid_qty,ask_price,ask_qty,mid_price,spread_pct,book_imbalance) VALUES (?,?,?,?,?,?,?,?,?)",
        (1000, "ETHUSDT", 99.9, 1.0, 100.1, 1.0, 100.0, 0.2, 0.0),
    )
    quote = _fill_quote(con, S34Rule(name="x"), "ETHUSDT", 1000, 100.0, "LONG", "ENTRY", "taker")
    assert quote["fill_price"] == 100.1
    assert quote["source"] == "BOOK_TICKER"
    con.close()


def test_long_taker_exit_uses_bid_and_net_identity(tmp_path):
    db = tmp_path / "microstructure.db"
    con = sqlite3.connect(db)
    con.executescript(
        """
        CREATE TABLE book_ticker (
            id INTEGER PRIMARY KEY,
            ts_ms INTEGER NOT NULL,
            symbol TEXT NOT NULL,
            bid_price REAL NOT NULL,
            bid_qty REAL NOT NULL,
            ask_price REAL NOT NULL,
            ask_qty REAL NOT NULL,
            mid_price REAL NOT NULL,
            spread_pct REAL NOT NULL,
            book_imbalance REAL NOT NULL
        );
        """
    )
    con.execute(
        "INSERT INTO book_ticker(ts_ms,symbol,bid_price,bid_qty,ask_price,ask_qty,mid_price,spread_pct,book_imbalance) VALUES (?,?,?,?,?,?,?,?,?)",
        (1000, "ETHUSDT", 100.9, 1.0, 101.1, 1.0, 101.0, 0.2, 0.0),
    )
    quote = _fill_quote(con, S34Rule(name="x"), "ETHUSDT", 1000, 101.0, "LONG", "EXIT", "taker")
    assert quote["fill_price"] == 100.9
    entry_fill = {"fill_price": 100.1, "mid": 100.0}
    costs = _cost_decomposition("LONG", 100.0, 101.0, entry_fill, quote, 4.0, 4.0)
    identity = (
        costs["gross_bps"]
        - costs["entry_adverse_bps"]
        - costs["exit_adverse_bps"]
        - costs["spread_cost_bps"]
        - costs["fee_cost_bps"]
    )
    assert abs(costs["net_bps"] - identity) < 1e-9
    assert costs["entry_adverse_bps"] == 0.0
    assert costs["exit_adverse_bps"] == 0.0
    assert costs["spread_cost_bps"] > 0.0
    con.close()


def test_tp_fill_mode_branches_apply_expected_fill_and_fee(tmp_path):
    db = tmp_path / "microstructure.db"
    con = sqlite3.connect(db)
    con.executescript(
        """
        CREATE TABLE book_ticker (
            id INTEGER PRIMARY KEY,
            ts_ms INTEGER NOT NULL,
            symbol TEXT NOT NULL,
            bid_price REAL NOT NULL,
            bid_qty REAL NOT NULL,
            ask_price REAL NOT NULL,
            ask_qty REAL NOT NULL,
            mid_price REAL NOT NULL,
            spread_pct REAL NOT NULL,
            book_imbalance REAL NOT NULL
        );
        """
    )
    con.execute(
        "INSERT INTO book_ticker(ts_ms,symbol,bid_price,bid_qty,ask_price,ask_qty,mid_price,spread_pct,book_imbalance) VALUES (?,?,?,?,?,?,?,?,?)",
        (2000, "ETHUSDT", 101.9, 1.0, 102.1, 1.0, 102.0, 0.2, 0.0),
    )
    base = {
        "trade_id": "PTEST",
        "status": "OPEN",
        "signal_ts_ms": 1000,
        "symbol": "ETHUSDT",
        "direction": "LONG",
        "entry_reference_price": 100.0,
        "entry_price": 100.1,
        "entry_fill": {"fill_price": 100.1, "fee_bps": 4.0},
        "tp_price": 101.0,
        "risk": {"notional_usdt": 100.0},
    }
    taker = dict(base)
    taker["rule"] = S34Rule(name="x", tp_fill_mode="taker").__dict__
    taker = _close_trade(con, taker, 2000, 102.0, "TP")
    assert taker["exit_price"] == 101.9
    assert taker["fee_cost_bps"] == 8.0

    maker = dict(base)
    maker["rule"] = S34Rule(name="x", tp_fill_mode="maker").__dict__
    maker = _close_trade(con, maker, 2000, 102.0, "TP")
    assert maker["exit_price"] == 101.0
    assert maker["fee_cost_bps"] == 6.0
    con.close()
