from __future__ import annotations

import uuid
from pathlib import Path

try:
    from core.trade_logger import TradeLogger
    from monitoring.status_snapshot import collect_data_research_fitness, collect_last_decisions, collect_pnl, render_status_text
except ModuleNotFoundError:  # pragma: no cover
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from core.trade_logger import TradeLogger
    from monitoring.status_snapshot import collect_data_research_fitness, collect_last_decisions, collect_pnl, render_status_text


def test_collect_pnl_and_render(monkeypatch) -> None:
    db = Path("data") / f"test_status_snapshot_{uuid.uuid4().hex}.db"
    try:
        lg = TradeLogger(str(db))
        lg.log_trade(
            {
                "trade_id": "ss1",
                "entry_time": 1_700_000_000.0,
                "exit_time": 1_700_000_100.0,
                "side": "sell",
                "regime": "UP",
                "entry_price": 100.0,
                "exit_price": 99.9,
                "pnl_bps": 10.0,
                "exit_type": "HORIZON_EXIT",
                "exit_reason": "time",
                "elapsed_sec": 100.0,
            }
        )
        p = collect_pnl(str(db))
        assert p["ok"] is True
        assert int(p["total_trades"]) == 1
        monkeypatch.setattr(
            "monitoring.status_snapshot.collect_data_research_fitness",
            lambda *args, **kwargs: {
                "status": "pass",
                "summary": "tier=trade_plus_liq_mark_proxy db_ready=True warnings=0 failures=0",
                "operator_action": "safe to continue",
            },
        )
        txt = render_status_text()
        assert "Eclipse Scalper - Health Check" in txt
        assert "Data research fitness: PASS" in txt
        assert "Fitness action: safe to continue" in txt
    finally:
        db.unlink(missing_ok=True)


def test_collect_last_decisions() -> None:
    jp = Path("logs") / f"test_status_snapshot_{uuid.uuid4().hex}.jsonl"
    jp.parent.mkdir(parents=True, exist_ok=True)
    try:
        jp.write_text(
            "\n".join(
                [
                    '{"ts":"2026-01-01T00:00:00Z","event":"entry.blocked","symbol":"ETHUSDT","data":{"reason":"risk"}}',
                    '{"ts":"2026-01-01T00:00:01Z","event":"entry.submitted","symbol":"ETHUSDT","data":{"reason":"ok"}}',
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        rows = collect_last_decisions(str(jp), limit=2)
        assert len(rows) == 2
        assert rows[0]["event"] in ("entry.submitted", "entry.blocked")
    finally:
        jp.unlink(missing_ok=True)


def test_collect_data_research_fitness_handles_missing_db() -> None:
    payload = collect_data_research_fitness(
        db_path="data/does_not_exist.db",
        csv_path="data/does_not_exist.csv",
        symbols=["ETHUSDT"],
    )
    assert payload["status"] == "unknown"
    assert payload["summary"] == "db_missing"


def test_collect_data_research_fitness_uses_active_symbols_env(monkeypatch) -> None:
    seen: dict[str, object] = {}
    monkeypatch.setenv("ACTIVE_SYMBOLS", "ETHUSDT,DOGEUSDT")
    monkeypatch.setattr(
        "monitoring.status_snapshot.analyze_research_fitness",
        lambda **kwargs: (
            seen.update(kwargs),
            {
                "status": "pass",
                "db_ready": True,
                "warnings": [],
                "failures": [],
                "contract": {"tier": "trade_plus_liq_mark_proxy"},
            },
        )[1],
    )
    payload = collect_data_research_fitness(
        db_path="data/microstructure.db",
        csv_path="data/event_diary.csv",
        symbols=None,
    )
    assert seen["symbols"] == ["ETHUSDT", "DOGEUSDT"]
    assert payload["status"] == "pass"
    assert payload["operator_action"] == "safe to continue"


def test_render_status_text_includes_data_research_fitness(monkeypatch) -> None:
    monkeypatch.setattr(
        "monitoring.status_snapshot.collect_status",
        lambda: {
            "pnl": {"ok": False},
            "diag": {"ok": False},
            "config": {
                "ENTRY_REGIME": "",
                "ENTRY_REGIME_RISK_ENABLED": "",
                "EXIT_SCRATCH_ENABLED": "",
                "NOTIFY_ENABLED": "",
            },
            "positions": {"count": 0, "positions": []},
            "last_decisions": [],
            "kill_switch": {"active": False},
            "data_research_fitness": {
                "status": "warn",
                "summary": "1 warning(s), no failures | tier=trade_plus_liq_mark_proxy db_ready=True",
                "operator_action": "continue with caution; review degraded feature coverage",
            },
        },
    )
    text = render_status_text()
    assert "Data research fitness: WARN" in text
    assert "Fitness action: continue with caution; review degraded feature coverage" in text
