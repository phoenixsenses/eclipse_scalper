import sqlite3

import pytest

from tools.s34_cascade_navigation_dashboard import (
    Lane,
    MarkIndex,
    cascade_features,
    cascade_phase_panel,
    evaluate_lane,
    exploratory_v_fade_panel,
    load_sizing_shadow_paper,
    load_v02_h4_shadow,
    load_v02_shadow_mirror,
    navigation_verdict,
    partial_cascade_state,
    render_md,
    rule_confidence_panel,
)
from tools.s34_feature_availability import LookaheadViolation, assert_feature_set_available

T0 = 1_780_000_000_000


def _rows(*pairs):
    return [{"ts_ms": T0 + off, "price": 100.0, "quantity": 1.0, "notional": notional, "trade_time_ms": T0 + off}
            for off, notional in pairs]


def test_partial_state_excludes_future_liqs():
    rows = _rows((0, 100_000.0), (10_000, 150_000.0), (40_000, 900_000.0))
    as_of = T0 + 20_000  # the 900k liq at +40s is in the FUTURE and must be invisible
    state = partial_cascade_state(rows, as_of, accel_window_sec=30)
    assert state["liq_count"] == 2
    assert state["running_notional"] == 250_000.0
    assert state["gap_since_last_sec"] == 10.0


def test_cascade_features_are_knowable_at_as_of():
    rows = _rows((0, 100_000.0), (5_000, 200_000.0))
    as_of = T0 + 10_000
    state = partial_cascade_state(rows, as_of, accel_window_sec=30)
    # must not raise -- everything stamped at as_of
    assert_feature_set_available(cascade_features(state, as_of), as_of, context="test")
    # a feature stamped in the future must raise
    bad = cascade_features(state, as_of + 1)
    with pytest.raises(LookaheadViolation):
        assert_feature_set_available(bad, as_of, context="test")


def test_phase_dead_when_gap_exceeds_threshold():
    rows = _rows((0, 600_000.0))
    state = partial_cascade_state(rows, T0 + 200_000, accel_window_sec=30)
    panel = cascade_phase_panel(state, dead_gap_sec=120.0)
    assert panel["phase"] == "DEAD"


def test_phase_accelerating_with_recent_growth():
    rows = _rows((0, 60_000.0), (25_000, 80_000.0), (55_000, 300_000.0))
    state = partial_cascade_state(rows, T0 + 56_000, accel_window_sec=30)
    panel = cascade_phase_panel(state, dead_gap_sec=120.0)
    assert panel["phase"] == "ACCELERATING"
    assert panel["continuation_momentum_score"] > 0.0


def test_strong_momentum_without_proven_edge_is_observe_only():
    # The core lesson: live momentum NEVER grants permission without proven edge.
    panels = {
        "_direction": "LONG",
        "cascade_phase": {"phase": "ACCELERATING", "continuation_momentum_score": 0.9, "fade_momentum_score": 0.0},
        "execution_reality": {"book_available": True, "no_fill_risk": "LOW"},
        "rule_confidence": {"family_edge": "NONE"},
    }
    v = navigation_verdict(panels)
    assert v["permission"] == "OBSERVE_ONLY"
    assert v["blocked_by_edge"] is True


def test_proven_edge_plus_momentum_grants_continuation():
    panels = {
        "_direction": "LONG",
        "cascade_phase": {"phase": "ACCELERATING", "continuation_momentum_score": 0.8, "fade_momentum_score": 0.0},
        "execution_reality": {"book_available": True, "no_fill_risk": "LOW"},
        "rule_confidence": {"family_edge": "PROVEN"},
    }
    v = navigation_verdict(panels)
    assert v["permission"] == "CONTINUATION_VIABLE"
    assert v["blocked_by_edge"] is False


def test_rule_confidence_edge_classification():
    conf = {
        "ETH_BUY": [
            {"verdict": "RESEARCH_ONLY", "hold_median_bps": -9.0},
            {"verdict": "RESEARCH_ONLY", "hold_median_bps": -6.0},
        ],
        "X_THIN": [{"verdict": "RESEARCH_ONLY", "hold_median_bps": 5.0}],
        "X_GOOD": [{"verdict": "PAPER_CANDIDATE", "hold_median_bps": 12.0}],
        # a positive holdout median on a BLOCKED (untestable) route must NOT read as edge
        "X_BLOCKED": [
            {"verdict": "RESEARCH_ONLY", "hold_median_bps": -9.0},
            {"verdict": "BLOCKED_THIN_CALIBRATION", "hold_median_bps": 55.0},
        ],
        "X_ALL_BLOCKED": [{"verdict": "BLOCKED_THIN_CALIBRATION", "hold_median_bps": 55.0}],
    }
    assert rule_confidence_panel("ETH_BUY", conf)["family_edge"] == "NONE"
    assert rule_confidence_panel("X_THIN", conf)["family_edge"] == "THIN"
    assert rule_confidence_panel("X_GOOD", conf)["family_edge"] == "PROVEN"
    assert rule_confidence_panel("MISSING", conf)["family_edge"] == "UNKNOWN"
    assert rule_confidence_panel("X_BLOCKED", conf)["family_edge"] == "NONE"
    assert rule_confidence_panel("X_ALL_BLOCKED", conf)["family_edge"] == "UNKNOWN"


def test_exploratory_v_fade_observation_only_panel():
    lane = Lane("ETHUSDT", "SELL", "SHORT", "ETH_SELL")
    state = {
        "liq_count": 3,
        "running_notional": 250_000.0,
        "first_ts_ms": T0 - 60_000,
    }
    marks = MarkIndex(
        [
            (T0 - 4 * 3600 * 1000, 101.0),
            (T0 - 60_000, 100.0),
            (T0, 99.7),
        ]
    )
    panel = exploratory_v_fade_panel(lane, state, marks, T0)
    assert panel["status"] == "ACTIVE"
    assert panel["permission"] == "EXPLORATORY_V_FADE_V0_1"
    assert panel["observation_only"] is True

    verdict = navigation_verdict(
        {
            "_direction": "SHORT",
            "cascade_phase": {"phase": "EXHAUSTION", "continuation_momentum_score": 0.0, "fade_momentum_score": 1.0},
            "execution_reality": {"book_available": True, "no_fill_risk": "LOW"},
            "rule_confidence": {"family_edge": "NONE"},
        }
    )
    assert verdict["permission"] == "OBSERVE_ONLY"


def _build_db(path):
    con = sqlite3.connect(path)
    con.executescript(
        """
        CREATE TABLE liquidations (id INTEGER PRIMARY KEY, ts_ms INTEGER, symbol TEXT, side TEXT,
            price REAL, quantity REAL, notional REAL, trade_time_ms INTEGER);
        CREATE TABLE mark_prices (id INTEGER PRIMARY KEY, ts_ms INTEGER, symbol TEXT, mark_price REAL, funding_rate REAL);
        CREATE TABLE book_ticker (id INTEGER PRIMARY KEY, ts_ms INTEGER, symbol TEXT, bid_price REAL, bid_qty REAL,
            ask_price REAL, ask_qty REAL, mid_price REAL, spread_pct REAL, book_imbalance REAL);
        """
    )
    day0 = T0 - (T0 % 86_400_000)
    con.execute("INSERT INTO liquidations(ts_ms,symbol,side,price,quantity,notional,trade_time_ms) VALUES (?,?,?,?,?,?,?)",
                (T0, "ETHUSDT", "BUY", 100.0, 1.0, 250_000.0, T0))
    con.executemany("INSERT INTO mark_prices(ts_ms,symbol,mark_price,funding_rate) VALUES (?,?,?,?)",
                    [(day0 + 1000, "ETHUSDT", 100.0, 0.0001), (T0, "ETHUSDT", 100.5, 0.0001),
                     (T0 - 950_000, "BTCUSDT", 1000.0, None), (T0, "BTCUSDT", 1002.0, None)])
    con.execute("INSERT INTO book_ticker(ts_ms,symbol,bid_price,bid_qty,ask_price,ask_qty,mid_price,spread_pct,book_imbalance) VALUES (?,?,?,?,?,?,?,?,?)",
                (T0, "ETHUSDT", 100.49, 1.0, 100.51, 1.0, 100.5, 0.02, 0.0))
    con.commit()
    con.close()


def test_evaluate_lane_end_to_end(tmp_path):
    db = tmp_path / "micro.db"
    _build_db(db)
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    from tools.s34_cascade_navigation_dashboard import load_mark_index_range, day_start_ms
    btc = load_mark_index_range(con, "BTCUSDT", T0 - 1_000_000, T0)
    lane = Lane("ETHUSDT", "BUY", "LONG", "ETH_BUY")
    out = evaluate_lane(
        con, lane, {"ETH_BUY": [{"verdict": "RESEARCH_ONLY", "hold_median_bps": -9.0}]}, btc,
        as_of_ts=T0, bucket_sec=300, accel_window_sec=30, dead_gap_sec=120.0,
        btc_window_sec=900, max_book_staleness_sec=5,
    )
    con.close()
    assert out["panels"]["anchor_health"]["running_notional"] == 250_000.0
    assert out["panels"]["execution_reality"]["book_available"] is True
    assert out["panels"]["regime_compass"]["btc_aligned_with_direction"] is True  # BTC up, LONG
    # RESEARCH_ONLY family => never auto-permission
    assert out["verdict"]["permission"] == "OBSERVE_ONLY"


def test_v02_shadow_mirror_fragment_is_observation_only(tmp_path):
    brief = tmp_path / "brief.json"
    brief.write_text(
        """
        {
          "protocol": {
            "id": "S34_V_ENGINE_V0_2_ETH_SELL_MAKER_LONG_H2_O20_W300_O5_DEEPBID",
            "permission": "EXPLORATORY_V_FADE_V0_2_SHADOW_MIRROR",
            "decision": "OBSERVE_ONLY_NO_ORDER",
            "live_rule_match": true
          },
          "ledger": {"rows_total": 11, "rows_added_this_run": 0},
          "overall": {"summary": {"n": 11, "sum_bps": 1081.6, "median_bps": 46.3, "top3_winner_removed_sum_bps": 402.6}},
          "recent": {
            "summary": {"n": 7, "sum_bps": 934.1, "median_bps": 149.9, "top3_winner_removed_sum_bps": 255.1},
            "kill_check": {"triggered": false}
          },
          "latest_observations": [{"signal_utc": "2026-06-26T13:18:54.877000+00:00"}]
        }
        """,
        encoding="utf-8",
    )
    fragment = load_v02_shadow_mirror(brief)
    assert fragment["status"] == "ACTIVE"
    assert fragment["permission"] == "EXPLORATORY_V_FADE_V0_2_SHADOW_MIRROR"
    assert fragment["decision"] == "OBSERVE_ONLY_NO_ORDER"
    assert fragment["live_rule_match"] is True
    assert fragment["recent_n"] == 7

    md = render_md(
        {
            "generated_at_utc": "now",
            "as_of_utc": "now",
            "lanes": [],
            "v02_shadow_mirror": fragment,
        }
    )
    assert "V0.2 Shadow Mirror" in md
    assert "OBSERVE_ONLY_NO_ORDER" in md


def test_sizing_shadow_paper_fragment_is_observation_only(tmp_path):
    payload = tmp_path / "sizing.json"
    payload.write_text(
        """
        {
          "status": "SHADOW_PAPER_SIZING_ONLY_NO_ORDER",
          "rule_id": "S34_V_ENGINE_V0_2_ETH_SELL_MAKER_LONG_H2_O20_W300_O5_DEEPBID",
          "source_shadow_rows": 11,
          "modes": {
            "CURRENT_ENV": {"n": 11, "notional_usdt": 1190.0, "margin_usdt": 29.75, "leverage": 40.0, "sum_bps": 1081.6, "sum_pnl_usdt": 128.711, "ending_equity_usdt": 163.71, "max_drawdown_pct_equity": 0.0},
            "BALANCED": {"n": 11, "notional_usdt": 16.3, "margin_usdt": 0.4, "leverage": 40.0, "sum_bps": 1081.6, "sum_pnl_usdt": 1.763, "ending_equity_usdt": 36.763, "max_drawdown_pct_equity": 0.0},
            "SURVIVAL": {"n": 11, "notional_usdt": 11.0, "margin_usdt": 0.3, "leverage": 40.0, "sum_bps": 1081.6, "sum_pnl_usdt": 1.192, "ending_equity_usdt": 36.19, "max_drawdown_pct_equity": 0.0}
          }
        }
        """,
        encoding="utf-8",
    )
    fragment = load_sizing_shadow_paper(payload)
    assert fragment["status"] == "SHADOW_PAPER_SIZING_ONLY_NO_ORDER"
    assert fragment["balanced"]["notional_usdt"] == 16.3

    md = render_md(
        {
            "generated_at_utc": "now",
            "as_of_utc": "now",
            "lanes": [],
            "sizing_shadow_paper": fragment,
        }
    )
    assert "Sizing Shadow Paper" in md
    assert "BALANCED" in md
    assert "observation only" in md


def test_v02_h4_shadow_fragment_is_observation_only(tmp_path):
    payload = tmp_path / "h4_shadow.json"
    payload.write_text(
        """
        {
          "status": "ACTIVE",
          "protocol_id": "S34_V_ENGINE_V0_2_ETH_SELL_MAKER_LONG_H2_O20_W300_O5_DEEPBID",
          "h2_sum_bps": 1081.6,
          "h4_sum_bps": 1738.6,
          "h4_t3r_bps": 819.2,
          "cross_policy_sum_bps": 1790.7,
          "sl150_touch_count": 0,
          "queue_status": "PROXY_ONLY_TOP_OF_BOOK",
          "decision": "H4_SHADOW_OBSERVATION_ONLY"
        }
        """,
        encoding="utf-8",
    )
    fragment = load_v02_h4_shadow(payload)
    assert fragment["status"] == "ACTIVE"
    assert fragment["decision"] == "H4_SHADOW_OBSERVATION_ONLY"
    assert fragment["h4_sum_bps"] == 1738.6

    md = render_md(
        {
            "generated_at_utc": "now",
            "as_of_utc": "now",
            "lanes": [],
            "v02_h4_shadow": fragment,
        }
    )
    assert "V0.2 H4 Shadow Management" in md
    assert "H4_SHADOW_OBSERVATION_ONLY" in md
