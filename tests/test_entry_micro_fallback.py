from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

try:
    from execution import entry as en
except ModuleNotFoundError:  # pragma: no cover
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from execution import entry as en


class _FakeMicroSig:
    def __init__(self, side: str, confidence: float):
        self.side = side
        self.confidence = confidence


class _FakeProvider:
    def __init__(self, sig):
        self._sig = sig

    def evaluate(self, regime_override=None):
        return self._sig


def test_micro_fallback_signal_returns_minimal_signal_on_conf_one(monkeypatch) -> None:
    async def _run() -> None:
        monkeypatch.setenv("ENTRY_MICRO_SIGNAL_ENABLED", "1")
        fake = _FakeProvider(_FakeMicroSig(side="buy", confidence=1.0))
        bot = SimpleNamespace()
        bot._micro_signal_provider = fake
        out = await en._micro_fallback_signal(bot, "ETHUSDT", "long")
        assert isinstance(out, dict)
        assert out.get("action") == "buy"
        assert float(out.get("confidence", 0.0)) == 1.0

    asyncio.run(_run())


def test_micro_fallback_signal_returns_none_on_partial_conf(monkeypatch) -> None:
    async def _run() -> None:
        monkeypatch.setenv("ENTRY_MICRO_SIGNAL_ENABLED", "1")
        fake = _FakeProvider(_FakeMicroSig(side="buy", confidence=0.0))
        bot = SimpleNamespace()
        bot._micro_signal_provider = fake
        out = await en._micro_fallback_signal(bot, "ETHUSDT", "long")
        assert out is None

    asyncio.run(_run())


def test_micro_fallback_signal_wraps_provider_none(monkeypatch) -> None:
    async def _run() -> None:
        monkeypatch.setenv("ENTRY_MICRO_SIGNAL_ENABLED", "1")

        class _NoneProvider:
            def evaluate(self, regime_override=None):
                return None

        bot = SimpleNamespace()
        bot._micro_signal_provider = _NoneProvider()
        out = await en._micro_fallback_signal(bot, "ETHUSDT", "long")
        assert out is None

    asyncio.run(_run())


def test_apply_micro_fallback_sets_side_and_conf(monkeypatch) -> None:
    async def _run() -> None:
        monkeypatch.setenv("ENTRY_MICRO_SIGNAL_ENABLED", "1")
        monkeypatch.setenv("MICRO_SIGNAL_SYMBOL", "ETHUSDT")
        bot = SimpleNamespace()

        async def _fake(bot_obj, k, side):
            return {"source": "micro", "symbol": k, "side": side, "action": "sell", "confidence": 1.0}

        monkeypatch.setattr(en, "_micro_fallback_signal", _fake)
        long_sig, short_sig, conf = await en._apply_micro_fallback_if_missing(
            bot, "ETHUSDT", "short", False, False, 0.0
        )
        assert long_sig is False
        assert short_sig is True
        assert float(conf) == 1.0

    asyncio.run(_run())
