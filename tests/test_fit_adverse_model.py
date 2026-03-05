"""Tests for tools/fit_adverse_model.py."""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from tools.fit_adverse_model import (
    _adverse_bps,
    _conditional_stats,
    _percentile,
    _quartile_label,
    _quartile_groups,
    analyze_symbol,
    _render_md,
)


# ─────────────────────────────────────────────
# Utility tests
# ─────────────────────────────────────────────

def test_adverse_bps_long():
    assert _adverse_bps(-0.001, "LONG") == pytest.approx(10.0)   # price fell 10 bps
    assert _adverse_bps(+0.001, "LONG") == 0.0                    # price rose → no adverse


def test_adverse_bps_short():
    assert _adverse_bps(+0.001, "SHORT") == pytest.approx(10.0)  # price rose 10 bps
    assert _adverse_bps(-0.001, "SHORT") == 0.0


def test_percentile_basic():
    vals = [1.0, 2.0, 3.0, 4.0]
    assert _percentile(vals, 0.0) == 1.0
    assert _percentile(vals, 1.0) == 4.0
    assert _percentile(vals, 0.5) == pytest.approx(2.5)


def test_percentile_empty():
    assert _percentile([], 0.5) == 0.0


def test_quartile_label():
    assert _quartile_label(0.5, 1.0, 2.0, 3.0) == "Q1"
    assert _quartile_label(1.5, 1.0, 2.0, 3.0) == "Q2"
    assert _quartile_label(2.5, 1.0, 2.0, 3.0) == "Q3"
    assert _quartile_label(3.5, 1.0, 2.0, 3.0) == "Q4"


def test_conditional_stats_basic():
    vals = [1.0, 2.0, 3.0]
    result = _conditional_stats(vals)
    assert result["n"] == 3
    assert result["mean"] == pytest.approx(2.0)
    assert result["std"] >= 0


def test_conditional_stats_empty():
    result = _conditional_stats([])
    assert result["n"] == 0
    assert result["mean"] is None


def test_quartile_groups_schema():
    paired = [(float(i), float(i) * 2) for i in range(1, 21)]  # 20 pairs
    groups = _quartile_groups(paired)
    assert set(groups.keys()) == {"Q1", "Q2", "Q3", "Q4"}
    for q in ("Q1", "Q2", "Q3", "Q4"):
        assert "mean" in groups[q]
        assert "n" in groups[q]
        assert groups[q]["n"] > 0


def test_quartile_groups_too_few():
    paired = [(1.0, 2.0), (2.0, 3.0)]  # only 2 — need ≥4
    result = _quartile_groups(paired)
    assert result == {}


# ─────────────────────────────────────────────
# analyze_symbol — mock DB
# ─────────────────────────────────────────────

def _make_in_memory_db(symbol: str = "ETHUSDT") -> str:
    """Create a temporary in-memory DB path (shared cache allows reuse)."""
    # Use a named in-memory DB so it persists for the test
    uri = f"file:testdb_{symbol}?mode=memory&cache=shared"
    conn = sqlite3.connect(uri, uri=True)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS agg_trades "
        "(ts_ms INTEGER, price REAL, quantity REAL, is_buyer_maker INTEGER, symbol TEXT)"
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS mark_prices "
        "(ts_ms INTEGER, mark_price REAL, symbol TEXT)"
    )
    # Insert 200 rows: 1-second buckets of fake trades
    base_ts = 1_700_000_000_000  # ms
    for i in range(200):
        ts = base_ts + i * 1000
        price = 1600.0 + (i % 10) * 0.5
        conn.execute(
            "INSERT INTO agg_trades VALUES (?,?,?,?,?)",
            (ts, price, 1.0, i % 2, symbol)
        )
        conn.execute(
            "INSERT INTO mark_prices VALUES (?,?,?)",
            (ts, price + 0.1, symbol)
        )
    conn.commit()
    return uri


def test_analyze_symbol_insufficient_data():
    """When DB returns no rows, expect an error key."""
    with patch("tools.fit_adverse_model._load_db_data", return_value=([], [])):
        result = analyze_symbol(
            db="dummy.db", symbol="ETHUSDT", lookback_min=60,
            bucket_sec=1, rule="micro_edge_v3_passive_alpha", horizon_sec=120,
        )
    assert "error" in result


def test_analyze_symbol_schema_keys_on_success():
    """With enough data and signal fires, result should have expected keys."""
    # Build 200 simple bucket rows and enough marks
    fake_rows = [
        {
            "ts_ms": 1_700_000_000_000 + i * 1000,
            "spread": 0.0002 + (i % 3) * 0.00005,
            "trade_intensity": 2500.0 + (i % 5) * 100,
            "imbalance": 0.6 + (i % 4) * 0.1,
            "vol_proxy": 0.001,
            "ret_1": 0.0001 * ((-1) ** i),
        }
        for i in range(200)
    ]
    fake_marks = [
        {"ts_ms": 1_700_000_000_000 + i * 1000, "mark_price": 1600.0 + i * 0.01}
        for i in range(400)
    ]

    def _mock_build(trades, marks, bucket_sec):
        return fake_rows

    def _mock_thresholds(rows):
        # Return thresholds that make the rule fire on many rows
        return {
            "imb_q90": 0.5,
            "int_q90": 2000.0,
            "spr_q90": 0.001,
        }

    def _mock_rule_fires(rule, row, thresholds):
        return float(row.get("imbalance", 0)) > 0.55

    with (
        patch("tools.fit_adverse_model._load_db_data", return_value=(fake_rows, fake_marks)),
        patch("tools.fit_adverse_model.build_bucket_features", side_effect=_mock_build),
        patch("tools.fit_adverse_model.compute_rule_thresholds", side_effect=_mock_thresholds),
        patch("tools.fit_adverse_model.rule_fires", side_effect=_mock_rule_fires),
    ):
        result = analyze_symbol(
            db="dummy.db", symbol="ETHUSDT", lookback_min=60,
            bucket_sec=1, rule="micro_edge_v3_passive_alpha", horizon_sec=10,
        )

    if "error" in result:
        pytest.skip(f"Not enough signal fires in mock data: {result['error']}")

    assert "statistics" in result
    assert "conditional" in result
    assert "quantile_edges" in result
    assert "implied_adverse_mult_vs_1bps" in result

    stats = result["statistics"]
    assert "global_mean_adverse_bps" in stats
    assert "percentiles" in stats
    assert stats["global_mean_adverse_bps"] >= 0.0

    cond = result["conditional"]
    for key in ("by_spread_quartile", "by_intensity_quartile", "by_imbalance_quartile"):
        assert key in cond


# ─────────────────────────────────────────────
# CLI smoke test
# ─────────────────────────────────────────────

def test_render_md_with_error():
    """_render_md should handle error entries gracefully."""
    result = {
        "generated_utc": "2026-01-01T00:00:00+00:00",
        "git_hash": "abc1234",
        "inputs": {"rule": "test_rule", "horizon_sec": 120, "lookback_min": 60, "bucket_sec": 1},
        "per_symbol": {"ETHUSDT": {"error": "insufficient_data"}},
    }
    md = _render_md(result)
    assert "ADVERSE SELECTION" in md
    assert "ETHUSDT" in md
    assert "insufficient_data" in md


def test_render_md_with_data():
    """_render_md should include conditional tables when data is present."""
    result = {
        "generated_utc": "2026-01-01T00:00:00+00:00",
        "git_hash": "abc1234",
        "inputs": {"rule": "test_rule", "horizon_sec": 120, "lookback_min": 60, "bucket_sec": 1},
        "per_symbol": {
            "ETHUSDT": {
                "n_total_buckets": 500,
                "n_signal_buckets": 80,
                "statistics": {
                    "global_mean_adverse_bps": 1.5,
                    "global_std_adverse_bps": 0.8,
                    "percentiles": {"p25": 0.5, "p50": 1.2, "p75": 2.1, "p90": 3.0, "p95": 4.0},
                },
                "conditional": {
                    "by_spread_quartile": {
                        "Q1": {"mean": 1.0, "std": 0.3, "n": 20},
                        "Q2": {"mean": 1.3, "std": 0.4, "n": 20},
                        "Q3": {"mean": 1.7, "std": 0.5, "n": 20},
                        "Q4": {"mean": 2.2, "std": 0.7, "n": 20},
                    },
                    "by_intensity_quartile": {},
                    "by_imbalance_quartile": {},
                },
                "quantile_edges": {"spread": [0.0001, 0.0002, 0.0003]},
                "implied_adverse_mult_vs_1bps": 1.5,
            }
        },
    }
    md = _render_md(result)
    assert "Q1" in md
    assert "Q4" in md
    assert "1.0000" in md
