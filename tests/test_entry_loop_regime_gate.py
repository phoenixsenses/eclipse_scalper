from __future__ import annotations

try:
    from execution.entry_loop import _parse_regime_mode, _should_block_regime
except ModuleNotFoundError:  # pragma: no cover
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from execution.entry_loop import _parse_regime_mode, _should_block_regime


def test_parse_regime_mode() -> None:
    assert _parse_regime_mode("up") == "up"
    assert _parse_regime_mode("DOWN") == "down"
    assert _parse_regime_mode("none") == "none"
    assert _parse_regime_mode("invalid") == "none"


def test_should_block_regime_mismatch() -> None:
    blocked, reason = _should_block_regime(
        mode="up",
        current_regime="DOWN",
        block_transition=True,
        block_unknown=True,
    )
    assert blocked is True
    assert reason == "regime_mismatch"


def test_should_block_regime_transition_unknown() -> None:
    blocked_t, reason_t = _should_block_regime(
        mode="none",
        current_regime="TRANSITION",
        block_transition=True,
        block_unknown=True,
    )
    assert blocked_t is True
    assert reason_t == "regime_transition"

    blocked_u, reason_u = _should_block_regime(
        mode="none",
        current_regime="UNKNOWN",
        block_transition=True,
        block_unknown=True,
    )
    assert blocked_u is True
    assert reason_u == "regime_unknown"


def test_should_allow_matching_regime() -> None:
    blocked, reason = _should_block_regime(
        mode="down",
        current_regime="DOWN",
        block_transition=True,
        block_unknown=True,
    )
    assert blocked is False
    assert reason == ""


def test_should_allow_unknown_when_flag_enabled() -> None:
    blocked, reason = _should_block_regime(
        mode="up",
        current_regime="UNKNOWN",
        block_transition=True,
        block_unknown=True,
        allow_unknown=True,
    )
    assert blocked is False
    assert reason == ""


def test_should_allow_unknown_during_warmup() -> None:
    blocked, reason = _should_block_regime(
        mode="up",
        current_regime="UNKNOWN",
        block_transition=True,
        block_unknown=True,
        warmup_active=True,
    )
    assert blocked is False
    assert reason == ""
