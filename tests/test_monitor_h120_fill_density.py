from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import monitor_h120_fill_density as mon


def _build_rows(*, count: int, future_touch: bool) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for i in range(count):
        rows.append(
            {
                "ts_ms": float(i * 1000),
                "mid": 100.0,
                "spread": 0.0001,
                "imbalance": 0.90,
                "trade_intensity": 5000.0,
                "ret_1": 0.0,
            }
        )
    for i in range(count, count + 130):
        rows.append(
            {
                "ts_ms": float(i * 1000),
                "mid": (99.98 if future_touch else 100.0),
                "spread": 0.0001,
                "imbalance": 0.10,
                "trade_intensity": 100.0,
                "ret_1": 0.0,
            }
        )
    return rows


def test_ready_to_rank_when_estimated_fills_clear_threshold(monkeypatch) -> None:
    rows = _build_rows(count=80, future_touch=True)
    monkeypatch.setattr(mon, "_tag_book_proxy_pressure", lambda features: [False] * len(features))
    monkeypatch.setattr(mon, "_tag_volatility_burst", lambda features: [False] * len(features))
    payload = mon.analyze_features(rows, symbol="ETHUSDT", lookback_min=20160, bucket_sec=1, min_fills=30)
    assert payload["signals_total"] == 80
    assert payload["signals_filtered"] == 80
    assert payload["touch_rate"] == 1.0
    assert payload["fill_rate"] == 0.5
    assert payload["estimated_fills"] == 40.0
    assert payload["status"] == "READY_TO_RANK"


def test_insufficient_when_estimated_fills_below_threshold(monkeypatch) -> None:
    rows = _build_rows(count=40, future_touch=False)
    monkeypatch.setattr(mon, "_tag_book_proxy_pressure", lambda features: [False] * len(features))
    monkeypatch.setattr(mon, "_tag_volatility_burst", lambda features: [False] * len(features))
    payload = mon.analyze_features(rows, symbol="ETHUSDT", lookback_min=20160, bucket_sec=1, min_fills=30)
    assert payload["signals_total"] == 40
    assert payload["signals_filtered"] == 40
    assert payload["touch_rate"] == 0.0
    assert payload["estimated_fills"] == 0.0
    assert payload["status"] == "INSUFFICIENT"
    assert payload["additional_fills_needed"] == 30


def test_main_json_output(monkeypatch, capsys) -> None:
    rows = _build_rows(count=80, future_touch=True)
    monkeypatch.setattr(mon, "_load_features", lambda db, symbol, lookback_min, bucket_sec: rows)
    monkeypatch.setattr(mon, "_tag_book_proxy_pressure", lambda features: [False] * len(features))
    monkeypatch.setattr(mon, "_tag_volatility_burst", lambda features: [False] * len(features))
    rc = mon.main(["--db", "data/microstructure.db", "--json"])
    assert rc == 0
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload["status"] == "READY_TO_RANK"
