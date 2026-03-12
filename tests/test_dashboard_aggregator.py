"""Tests for execution.dashboard_aggregator."""
import importlib
import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch


def _reload():
    import execution.dashboard_aggregator as m
    importlib.reload(m)
    return m


def _make_bot(risk_state=None, pocket_sched=None):
    bot = MagicMock()
    rc = {}
    if risk_state is not None:
        rm = MagicMock()
        rm.state_dict.return_value = risk_state
        rc["regime_risk_manager"] = rm
    if pocket_sched is not None:
        rc["pocket_scheduler"] = pocket_sched
    bot.state.run_context = rc
    return bot


class TestGetSystemDashboard:
    def test_keys_present(self):
        m = _reload()
        bot = _make_bot()
        d = m.get_system_dashboard(bot)
        for key in ("ts_utc", "ts_epoch", "health", "metrics", "risk_state",
                    "event_gate", "regime_watch", "pocket_scheduler"):
            assert key in d

    def test_risk_state_included(self):
        m = _reload()
        bot = _make_bot(risk_state={"daily_pnl_bps": 5.0, "daily_trades": 3})
        d = m.get_system_dashboard(bot)
        assert d["risk_state"]["daily_pnl_bps"] == 5.0

    def test_no_risk_manager_returns_empty(self):
        m = _reload()
        bot = _make_bot()
        d = m.get_system_dashboard(bot)
        assert d["risk_state"] == {}

    def test_pocket_scheduler_included(self):
        m = _reload()
        sched = MagicMock()
        sched.state_dict.return_value = {"cooldown_sec": 120.0, "pockets": {}}
        bot = _make_bot(pocket_sched=sched)
        d = m.get_system_dashboard(bot)
        assert d["pocket_scheduler"]["cooldown_sec"] == 120.0

    def test_never_raises_on_import_errors(self):
        m = _reload()
        bot = MagicMock()
        bot.state.run_context = {}
        # Should not raise even if all sub-modules fail
        d = m.get_system_dashboard(bot)
        assert "ts_utc" in d


class TestDashboardWrite:
    def _tmpdir(self, name: str):
        import tempfile, pathlib
        d = pathlib.Path(tempfile.gettempdir()) / f"eclipse_test_{name}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _run(self, coro):
        import asyncio
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

    def test_writes_json_file(self, monkeypatch):
        d = self._tmpdir("dash_write")
        out = d / "health" / "dashboard.json"
        monkeypatch.setenv("DASHBOARD_EXPORT_PATH", str(out))
        monkeypatch.setenv("DASHBOARD_EXPORT_INTERVAL_SEC", "0")
        m = _reload()
        m._last_export_ts = 0.0
        bot = _make_bot()

        self._run(m.dashboard_export_tick(bot))

        assert out.exists()
        data = json.loads(out.read_text(encoding="utf-8"))
        assert "ts_utc" in data

    def test_interval_respected(self, monkeypatch):
        import time
        d = self._tmpdir("dash_interval")
        out = d / "health" / "dashboard.json"
        monkeypatch.setenv("DASHBOARD_EXPORT_PATH", str(out))
        monkeypatch.setenv("DASHBOARD_EXPORT_INTERVAL_SEC", "3600")
        m = _reload()
        m._last_export_ts = time.time()  # just exported

        bot = _make_bot()
        self._run(m.dashboard_export_tick(bot))
        # File should NOT be created (interval not elapsed)
        assert not out.exists()
