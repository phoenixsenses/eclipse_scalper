from __future__ import annotations

from types import SimpleNamespace

try:
    from core.regime_risk import RegimeRiskConfig, RegimeRiskManager
except ModuleNotFoundError:  # pragma: no cover
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from core.regime_risk import RegimeRiskConfig, RegimeRiskManager


def test_max_concurrent_blocks() -> None:
    rm = RegimeRiskManager(RegimeRiskConfig(max_concurrent_positions=1))
    d = rm.check_entry("sell", "UP", [SimpleNamespace(symbol="ETHUSDT")])
    assert d.allowed is False
    assert d.reason == "max_concurrent_positions"


def test_daily_loss_limit_blocks() -> None:
    rm = RegimeRiskManager(RegimeRiskConfig(max_daily_loss_bps=50.0))
    rm.on_trade_exit({"reason": "horizon"}, pnl_bps=-60.0)
    d = rm.check_entry("sell", "UP", [])
    assert d.allowed is False
    assert d.reason == "daily_loss_limit"


def test_daily_trade_limit_blocks() -> None:
    rm = RegimeRiskManager(RegimeRiskConfig(max_daily_trades=2))
    rm.on_entry_submitted()
    rm.on_entry_submitted()
    d = rm.check_entry("sell", "UP", [])
    assert d.allowed is False
    assert d.reason == "daily_trade_limit"


def test_max_drawdown_blocks() -> None:
    rm = RegimeRiskManager(RegimeRiskConfig(max_drawdown_bps=100.0, max_daily_loss_bps=0.0))
    rm.on_trade_exit({"reason": "horizon"}, pnl_bps=20.0)
    rm.on_trade_exit({"reason": "horizon"}, pnl_bps=-121.0)  # peak=20, current=-101 => dd=121
    d = rm.check_entry("sell", "UP", [])
    assert d.allowed is False
    assert d.reason == "max_drawdown_limit"


def test_regime_change_hold_close_reduce() -> None:
    rm_hold = RegimeRiskManager(RegimeRiskConfig(regime_change_policy="hold"))
    assert rm_hold.on_regime_change("UP", "DOWN", [SimpleNamespace(symbol="ETHUSDT")]) == []

    rm_close = RegimeRiskManager(RegimeRiskConfig(regime_change_policy="close"))
    acts_close = rm_close.on_regime_change("UP", "DOWN", [SimpleNamespace(symbol="ETHUSDT"), SimpleNamespace(symbol="BTCUSDT")])
    assert len(acts_close) == 2
    assert all(a.action == "close_position" for a in acts_close)

    rm_reduce = RegimeRiskManager(RegimeRiskConfig(regime_change_policy="reduce"))
    acts_reduce = rm_reduce.on_regime_change("UP", "DOWN", [SimpleNamespace(symbol="ETHUSDT")])
    assert len(acts_reduce) == 1
    assert acts_reduce[0].action == "reduce_position"


def test_regime_cooldown_blocks() -> None:
    rm = RegimeRiskManager(RegimeRiskConfig(cooldown_after_regime_change_sec=60.0))
    rm.on_regime_change("UP", "DOWN", [])
    d = rm.check_entry("sell", "DOWN", [])
    assert d.allowed is False
    assert d.reason == "regime_cooldown_active"


def test_consecutive_scratch_pause_and_reset() -> None:
    rm = RegimeRiskManager(RegimeRiskConfig(max_consecutive_scratches=3, scratch_pause_sec=600))
    rm.on_trade_exit({"reason": "scratch"}, pnl_bps=-1.0)
    rm.on_trade_exit({"reason": "scratch"}, pnl_bps=-1.0)
    acts = rm.on_trade_exit({"reason": "scratch"}, pnl_bps=-1.0)
    assert any(a.action == "pause_trading" for a in acts)
    d = rm.check_entry("sell", "UP", [])
    assert d.allowed is False
    assert d.reason == "scratch_pause_active"

    # non-scratch reset path
    rm2 = RegimeRiskManager(RegimeRiskConfig(max_consecutive_scratches=3, scratch_pause_sec=600))
    rm2.on_trade_exit({"reason": "scratch"}, pnl_bps=-1.0)
    rm2.on_trade_exit({"reason": "scratch"}, pnl_bps=-1.0)
    rm2.on_trade_exit({"reason": "horizon"}, pnl_bps=0.5)
    st = rm2.state_dict()
    assert int(st["consecutive_scratches"]) == 0


def test_reset_daily_and_state_dict() -> None:
    rm = RegimeRiskManager(RegimeRiskConfig(max_daily_trades=100))
    rm.on_entry_submitted()
    rm.on_trade_exit({"reason": "horizon"}, pnl_bps=5.0)
    rm.reset_daily()
    st = rm.state_dict()
    assert float(st["daily_pnl_bps"]) == 0.0
    assert int(st["daily_trades"]) == 0
    assert "config" in st
