from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

try:
    from execution import entry
except ModuleNotFoundError:  # pragma: no cover
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from execution import entry


def test_resolve_confidence_reason_scorer_exception() -> None:
    out = entry._resolve_confidence_reason(
        scorer_reason="scorer_exception",
        confidence=0.0,
        parsed_ok=False,
        micro_enabled=False,
        micro_ready=True,
        data_ready=True,
        regime_ok=True,
    )
    assert out == "scorer_exception"


def test_resolve_confidence_reason_micro_not_ready() -> None:
    out = entry._resolve_confidence_reason(
        scorer_reason="score_none",
        confidence=0.0,
        parsed_ok=True,
        micro_enabled=True,
        micro_ready=False,
        data_ready=True,
        regime_ok=True,
    )
    assert out == "micro_not_ready"


def test_resolve_confidence_reason_score_none_when_parsed_missing() -> None:
    out = entry._resolve_confidence_reason(
        scorer_reason="score_none",
        confidence=0.0,
        parsed_ok=False,
        micro_enabled=False,
        micro_ready=True,
        data_ready=True,
        regime_ok=True,
    )
    assert out == "score_none"


def test_resolve_confidence_reason_keeps_nonzero_score_reason() -> None:
    out = entry._resolve_confidence_reason(
        scorer_reason="signal_not_present",
        confidence=0.31,
        parsed_ok=True,
        micro_enabled=True,
        micro_ready=False,
        data_ready=False,
        regime_ok=False,
    )
    assert out == "signal_not_present"


def test_micro_confidence_mvp_nonzero() -> None:
    feat = SimpleNamespace(trade_intensity=40.0, spread=0.0002, imbalance_signed=0.5)

    class _E:
        def get_features(self, _symbol):
            return feat

    bot = SimpleNamespace(micro_feature_engine=_E())
    out = entry._micro_confidence_mvp(bot, "BTCUSDT")
    assert out > 0.0
    assert out <= 0.5
