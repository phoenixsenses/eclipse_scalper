from __future__ import annotations

try:
    from core.scratch import ScratchConfig
    from execution.exit import _get_or_reset_scratch_engine, _scratch_signature
except ModuleNotFoundError:  # pragma: no cover
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from core.scratch import ScratchConfig
    from execution.exit import _get_or_reset_scratch_engine, _scratch_signature


def test_scratch_signature_stable() -> None:
    s1 = _scratch_signature(123.456, 100.0, "long")
    s2 = _scratch_signature(123.456, 100.0, "long")
    assert s1 == s2


def test_get_or_reset_scratch_engine_reuse_and_reset() -> None:
    runtime = {}
    cfg = ScratchConfig(
        max_adverse_bps=5.0,
        scratch_cooldown_sec=2.0,
        trailing_stop_bps=0.0,
        take_profit_bps=0.0,
        hard_horizon_sec=120.0,
    )
    e1 = _get_or_reset_scratch_engine(
        runtime,
        k="ETHUSDT",
        entry_ts=1000.0,
        entry_price=100.0,
        side="long",
        cfg=cfg,
    )
    e2 = _get_or_reset_scratch_engine(
        runtime,
        k="ETHUSDT",
        entry_ts=1000.0,
        entry_price=100.0,
        side="long",
        cfg=cfg,
    )
    assert e1 is e2
    e3 = _get_or_reset_scratch_engine(
        runtime,
        k="ETHUSDT",
        entry_ts=1001.0,
        entry_price=100.0,
        side="long",
        cfg=cfg,
    )
    assert e3 is not e2
