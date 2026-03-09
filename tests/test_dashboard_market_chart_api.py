from __future__ import annotations

from pathlib import Path
import sys

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dashboard.backend.app import app


def test_market_chart_endpoint(monkeypatch) -> None:
    monkeypatch.setattr(
        "dashboard.backend.app.read_market_chart",
        lambda symbol="BTCUSDT", interval="5m", limit=240: {
            "source": "binance_spot",
            "symbol": symbol,
            "interval": interval,
            "limit": limit,
            "generated_ts": "2026-03-09T00:00:00Z",
            "candles": [
                {"time": 1700000000, "open": 50000.0, "high": 50100.0, "low": 49900.0, "close": 50050.0, "volume": 100.0},
                {"time": 1700000300, "open": 50050.0, "high": 50120.0, "low": 50020.0, "close": 50080.0, "volume": 120.0},
            ],
            "overlays": [
                {"name": "EMA 20", "values": [None, 50065.0]},
                {"name": "EMA 50", "values": [None, 50055.0]},
            ],
            "oscillator": {"name": "RSI 14", "values": [None, 58.0]},
        },
    )
    client = TestClient(app)
    response = client.get("/api/market/chart?symbol=ETHUSDT&interval=15m&limit=120")
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["symbol"] == "ETHUSDT"
    assert payload["interval"] == "15m"
    assert len(payload["candles"]) == 2
    assert payload["overlays"][0]["name"] == "EMA 20"
    assert payload["oscillator"]["name"] == "RSI 14"
