from __future__ import annotations

import pandas as pd
import uuid
from pathlib import Path

from src.microphys.live.config import LiveSettings
import src.microphys.live.daemon as d


def test_publish_lifecycle_for_trades_records_violation_on_bad_price() -> None:
    p = Path("localtests") / f"runtime_hooks_{uuid.uuid4().hex[:8]}"
    p.mkdir(parents=True, exist_ok=True)
    bus = d._init_lifecycle_bus(p / "events.jsonl")
    trades = pd.DataFrame(
        [
            {
                "entry_ts_utc": "2026-03-05T10:00:00Z",
                "side": "buy",
                "entry_price": 0.0,  # invalid LIMIT price -> contract violation
                "fill_price": 0.0,
                "filled": True,
                "order_id": "ord_bad",
            }
        ]
    )
    n, violations = d._publish_lifecycle_for_trades(bus=bus, symbol="ETHUSDT", trades=trades, base_ts_ms=1700000000000)
    assert n >= 1
    assert len(violations) >= 1


def test_run_daemon_supervisor_fail_fast(monkeypatch) -> None:
    def _fake_cycle(cfg: LiveSettings, artifact_snapshot=None):
        return {
            "new_trades": 0,
            "status": {"state": "ok", "data_freshness_sec": 9999.0},
            "execution_model": "simple",
            "execution_params_loaded": False,
        }

    monkeypatch.setattr(d, "run_live_cycle", _fake_cycle)
    cfg = LiveSettings(
        symbol="ETHUSDT",
        exec_runtime_supervisor_enabled=True,
        supervisor_max_feed_age_sec=1.0,
        supervisor_max_order_age_sec=1.0,
        supervisor_max_loop_errors=1,
        refresh_sec=0.01,
    )
    rc = d.run_daemon(cfg, max_cycles=1)
    assert int(rc) == 2
