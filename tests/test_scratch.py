from __future__ import annotations

try:
    from core.scratch import ScratchConfig, ScratchEngine
except ModuleNotFoundError:  # pragma: no cover
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from core.scratch import ScratchConfig, ScratchEngine


def _mk_engine(**kwargs: float) -> ScratchEngine:
    cfg = ScratchConfig(**kwargs)
    e = ScratchEngine(cfg)
    e.reset(entry_price=100.0, side="buy", entry_time=0.0)
    return e


def test_basic_scratch_trigger() -> None:
    e = _mk_engine(max_adverse_bps=5.0, scratch_cooldown_sec=10.0, hard_horizon_sec=120.0)
    assert e.evaluate(99.95, 5.0).action == "HOLD"
    d = e.evaluate(99.94, 11.0)
    assert d.action == "SCRATCH"
    assert d.reason == "max_adverse"


def test_scratch_cooldown_blocks_early_exit() -> None:
    e = _mk_engine(max_adverse_bps=5.0, scratch_cooldown_sec=10.0, hard_horizon_sec=120.0)
    assert e.evaluate(99.90, 9.0).action == "HOLD"
    assert e.evaluate(99.90, 11.0).action == "SCRATCH"


def test_sell_side_adverse() -> None:
    cfg = ScratchConfig(max_adverse_bps=5.0, scratch_cooldown_sec=0.0, hard_horizon_sec=120.0)
    e = ScratchEngine(cfg)
    e.reset(entry_price=100.0, side="sell", entry_time=0.0)
    d = e.evaluate(100.06, 1.0)
    assert d.action == "SCRATCH"


def test_trailing_stop() -> None:
    e = _mk_engine(
        max_adverse_bps=0.0,
        scratch_cooldown_sec=0.0,
        trailing_stop_bps=3.0,
        hard_horizon_sec=120.0,
    )
    assert e.evaluate(100.08, 1.0).action == "HOLD"
    d = e.evaluate(100.04, 2.0)
    assert d.action == "SCRATCH"
    assert d.reason == "trailing_stop"


def test_take_profit() -> None:
    e = _mk_engine(
        max_adverse_bps=0.0,
        scratch_cooldown_sec=10.0,
        take_profit_bps=8.0,
        hard_horizon_sec=120.0,
    )
    d = e.evaluate(100.081, 1.0)
    assert d.action == "TAKE_PROFIT"


def test_horizon_exit() -> None:
    e = _mk_engine(max_adverse_bps=0.0, take_profit_bps=0.0, hard_horizon_sec=10.0)
    d = e.evaluate(100.0, 10.0)
    assert d.action == "HORIZON_EXIT"


def test_all_disabled_only_horizon() -> None:
    e = _mk_engine(
        max_adverse_bps=0.0,
        scratch_cooldown_sec=0.0,
        trailing_stop_bps=0.0,
        take_profit_bps=0.0,
        hard_horizon_sec=5.0,
    )
    assert e.evaluate(99.0, 2.0).action == "HOLD"
    assert e.evaluate(99.0, 5.0).action == "HORIZON_EXIT"


def test_priority_horizon_over_take_profit() -> None:
    e = _mk_engine(
        max_adverse_bps=0.0,
        scratch_cooldown_sec=0.0,
        take_profit_bps=1.0,
        hard_horizon_sec=10.0,
    )
    d = e.evaluate(100.1, 10.0)
    assert d.action == "HORIZON_EXIT"


def test_peak_tracking_and_reset() -> None:
    e = _mk_engine(
        max_adverse_bps=10.0,
        scratch_cooldown_sec=0.0,
        hard_horizon_sec=120.0,
    )
    d1 = e.evaluate(100.05, 1.0)
    d2 = e.evaluate(99.98, 2.0)
    assert d1.peak_favorable_bps >= 4.99
    assert d2.max_adverse_bps >= 1.99
    e.reset(entry_price=100.0, side="buy", entry_time=10.0)
    d3 = e.evaluate(100.0, 11.0)
    assert d3.peak_favorable_bps == 0.0
    assert d3.max_adverse_bps == 0.0
