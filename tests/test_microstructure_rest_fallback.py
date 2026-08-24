from __future__ import annotations

import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.microstructure_collector import EventBuffer, MicrostructureCollector, StreamParser


def _buffer(tmp_path: Path) -> EventBuffer:
    db = tmp_path / "micro.db"
    con = sqlite3.connect(str(db))
    con.execute(
        "CREATE TABLE agg_trades (id INTEGER PRIMARY KEY AUTOINCREMENT, ts_ms INTEGER, symbol TEXT, price REAL, quantity REAL, notional REAL, is_buyer_maker INTEGER)"
    )
    con.execute(
        "CREATE TABLE mark_prices (id INTEGER PRIMARY KEY AUTOINCREMENT, ts_ms INTEGER, symbol TEXT, mark_price REAL, funding_rate REAL, next_funding_time_ms INTEGER)"
    )
    con.execute(
        "CREATE TABLE liquidations (id INTEGER PRIMARY KEY AUTOINCREMENT, ts_ms INTEGER, symbol TEXT, side TEXT, price REAL, quantity REAL, notional REAL, trade_time_ms INTEGER)"
    )
    con.commit()
    return EventBuffer(con, flush_interval=999)


def test_rest_agg_trade_maps_to_internal_schema(tmp_path):
    buf = _buffer(tmp_path)
    parser = StreamParser(buf)

    ok = parser.ingest_rest_agg_trade({
        "s": "BTCUSDT",
        "p": "100.50",
        "q": "2",
        "T": 1710000000000,
        "m": True,
    })

    assert ok is True
    assert buf.agg_trades == [(1710000000000, "BTCUSDT", 100.5, 2.0, 201.0, 1)]
    buf.conn.close()


def test_rest_mark_price_maps_to_internal_schema(tmp_path):
    buf = _buffer(tmp_path)
    parser = StreamParser(buf)
    before_ms = int(time.time() * 1000)

    ok = parser.ingest_rest_mark_price({
        "symbol": "ETHUSDT",
        "markPrice": "2500.25",
        "lastFundingRate": "0.0001",
        "nextFundingTime": "1710003600000",
    })

    assert ok is True
    row = buf.mark_prices[0]
    assert row[1:] == ("ETHUSDT", 2500.25, 0.0001, 1710003600000)
    assert row[0] >= before_ms
    buf.conn.close()


def test_force_order_arr_combined_stream_maps_to_liquidation(tmp_path):
    buf = _buffer(tmp_path)
    parser = StreamParser(buf)

    parser.parse(
        """
        {
          "stream": "!forceOrder@arr",
          "data": {
            "e": "forceOrder",
            "E": 1710000000100,
            "o": {
              "s": "ETHUSDT",
              "S": "SELL",
              "q": "0.25",
              "p": "1560.0",
              "T": 1710000000090
            }
          }
        }
        """
    )

    assert buf.liquidations == [(1710000000100, "ETHUSDT", "SELL", 1560.0, 0.25, 390.0, 1710000000090)]
    buf.conn.close()


def test_collector_uses_routed_market_stream_and_arr_liquidations_by_default():
    collector = MicrostructureCollector(symbols=["BTCUSDT", "ETHUSDT"])

    url = collector._build_stream_url()

    assert url.startswith("wss://fstream.binance.com/market/stream?streams=")
    assert "!forceOrder@arr" in url
    assert "btcusdt@forceOrder" not in url
    assert "btcusdt@aggTrade" in url
    assert "ethusdt@markPrice@1s" in url


def test_collector_can_restore_per_symbol_liquidation_strategy():
    collector = MicrostructureCollector(symbols=["BTCUSDT"], liquidation_stream_mode="per_symbol")

    url = collector._build_stream_url()

    assert url.startswith("wss://fstream.binance.com/market/stream?streams=")
    assert "!forceOrder@arr" not in url
    assert "btcusdt@forceOrder" in url


def test_collector_ws_base_is_routed_market_endpoint():
    """Regression guard: 2026-06-06 and 2026-07-03 both saw this endpoint
    flip-flop with the routed /market path silently killing all live data
    (handshake succeeds, zero application frames ever arrive). BINANCE_WS
    must stay on the routed path."""
    assert MicrostructureCollector.BINANCE_WS == "wss://fstream.binance.com/market/stream"


def test_build_stream_url_never_duplicates_market_path_segment():
    for mode in ("all_market_arr", "per_symbol"):
        collector = MicrostructureCollector(symbols=["BTCUSDT", "ETHUSDT", "SOLUSDT"], liquidation_stream_mode=mode)
        url = collector._build_stream_url()
        assert url.count("/market/stream") == 1
        assert "/market/market" not in url


def test_build_stream_url_exact_expected_shape_all_market_arr():
    collector = MicrostructureCollector(symbols=["BTCUSDT", "ETHUSDT", "SOLUSDT"])
    url = collector._build_stream_url()
    assert url == (
        "wss://fstream.binance.com/market/stream?streams="
        "!forceOrder@arr/btcusdt@aggTrade/btcusdt@markPrice@1s/"
        "ethusdt@aggTrade/ethusdt@markPrice@1s/solusdt@aggTrade/solusdt@markPrice@1s"
    )


def test_rest_endpoints_are_unaffected_by_ws_routing():
    """The REST fallback host must stay independent of the WS routing fix."""
    assert MicrostructureCollector.BINANCE_REST == "https://fapi.binance.com"
