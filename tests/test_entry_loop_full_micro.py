from __future__ import annotations

import asyncio
import sqlite3
from types import SimpleNamespace
from uuid import uuid4
from pathlib import Path


try:
    from execution import entry_loop_full as elf
except ModuleNotFoundError:  # pragma: no cover
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from execution import entry_loop_full as elf


class _FakeProvider:
    def __init__(self, result):
        self.result = result
        self.calls = 0

    def evaluate(self, regime_override=None):
        self.calls += 1
        return self.result


def test_entry_loop_full_micro_path_priority(monkeypatch, tmp_path) -> None:
    async def _run() -> None:
        # bot._shutdown.set() below (used just to stop the loop after one
        # entry) is a direct set() bypassing request_shutdown(), which
        # trips entry_loop_full.py's own bypass tripwire and writes a
        # real, unisolated logs/last_shutdown.json with fatal=True --
        # empirically confirmed to leak a POST-CRASH COOLDOWN into any
        # other test instantiating kill-switch state within 300s. chdir
        # keeps that incidental write off the real repo state.
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("ENTRY_MICRO_SIGNAL_ENABLED", "1")
        monkeypatch.setenv("MICRO_SIGNAL_SYMBOL", "ETHUSDT")
        monkeypatch.setenv("ENTRY_WAIT_FOR_DATA_READY_SEC", "0")
        monkeypatch.setenv("ENTRY_POLL_SEC", "0.01")
        monkeypatch.setenv("ENTRY_PER_SYMBOL_GAP_SEC", "0.01")
        monkeypatch.setenv("ENTRY_LOCAL_COOLDOWN_SEC", "0")
        monkeypatch.setenv("ENTRY_REGIME_LOOKBACK_SEC", "60")
        monkeypatch.setenv("ENTRY_REGIME_DEBOUNCE_SEC", "0")

        provider = _FakeProvider(SimpleNamespace(present=True, confidence=1.0, side="buy", reason="match"))
        monkeypatch.setattr(elf, "_resolve_micro_builder", lambda: (lambda _bot: (None, provider)))
        monkeypatch.setattr(elf, "trade_allowed", None)

        calls = []

        async def _fake_try_enter(bot, symbol, side):
            calls.append((symbol, side))
            bot._shutdown.set()

        monkeypatch.setattr(elf, "try_enter", _fake_try_enter)

        bot = SimpleNamespace()
        bot.cfg = SimpleNamespace()
        bot.active_symbols = ["ETHUSDT"]
        bot._shutdown = asyncio.Event()
        bot.data_ready = asyncio.Event()
        bot.data_ready.set()
        bot.data = SimpleNamespace(price={"ETHUSDT": 100.0})

        await asyncio.wait_for(elf.entry_loop_full(bot), timeout=2.0)

        assert provider.calls >= 1
        assert calls
        assert calls[0] == ("ETHUSDT", "long")

    asyncio.run(_run())


def test_entry_loop_full_disabled_keeps_existing_path(monkeypatch, tmp_path) -> None:
    async def _run() -> None:
        # See test_entry_loop_full_micro_path_priority above: isolate the
        # incidental bypass-tripwire write from bot._shutdown.set().
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("ENTRY_MICRO_SIGNAL_ENABLED", "0")
        monkeypatch.setenv("ENTRY_WAIT_FOR_DATA_READY_SEC", "0")
        monkeypatch.setenv("ENTRY_POLL_SEC", "0.01")
        monkeypatch.setenv("ENTRY_PER_SYMBOL_GAP_SEC", "0.01")
        monkeypatch.setenv("ENTRY_LOCAL_COOLDOWN_SEC", "0")
        monkeypatch.setenv("SPAWN_BOTH_SIDES", "0")

        monkeypatch.setattr(elf, "trade_allowed", None)

        calls = []

        async def _fake_try_enter(bot, symbol, side):
            calls.append((symbol, side))
            bot._shutdown.set()

        monkeypatch.setattr(elf, "try_enter", _fake_try_enter)

        bot = SimpleNamespace()
        bot.cfg = SimpleNamespace()
        bot.active_symbols = ["ETHUSDT"]
        bot._shutdown = asyncio.Event()
        bot.data_ready = asyncio.Event()
        bot.data_ready.set()
        bot.data = SimpleNamespace(price={"ETHUSDT": 100.0})

        await asyncio.wait_for(elf.entry_loop_full(bot), timeout=2.0)

        assert calls
        assert calls[0][0] == "ETHUSDT"
        assert calls[0][1] in ("long", "short")

    asyncio.run(_run())


def test_resolve_regime_price_falls_back_to_micro_mark(monkeypatch) -> None:
    monkeypatch.setattr(elf, "_get_price_with_source", lambda _bot, _sym: (0.0, "none"))
    micro_engine = SimpleNamespace(get_features=lambda _sym: SimpleNamespace(mark_price=1234.5))
    px, src = elf._resolve_regime_price(SimpleNamespace(), "ETHUSDT", micro_engine)
    assert px == 1234.5
    assert src == "micro_mark_price"


def test_regime_warmup_from_mark_prices() -> None:
    base_dir = Path("tmp_tests")
    base_dir.mkdir(parents=True, exist_ok=True)
    dbp = base_dir / f"micro_{uuid4().hex}.db"
    conn = sqlite3.connect(str(dbp))
    try:
        conn.execute("CREATE TABLE mark_prices (ts REAL, symbol TEXT, mark_price REAL)")
        base = float(elf._now()) - 3600.0
        rows = [(base + float(i), "ETHUSDT", 1000.0 + float(i)) for i in range(3601)]
        conn.executemany("INSERT INTO mark_prices(ts, symbol, mark_price) VALUES (?,?,?)", rows)
        conn.commit()
    finally:
        conn.close()

    runtime = elf._RegimeRuntime(lookback_sec=3600, debounce_sec=0)
    out = elf._warmup_regime_from_mark_prices(
        regime_runtime=runtime,
        symbol="ETHUSDT",
        db_path=str(dbp),
        lookback_sec=3600,
    )
    assert int(out.get("fed", 0) or 0) > 1000
    assert str(out.get("regime") or "UNKNOWN") in ("UP", "DOWN")
    try:
        dbp.unlink(missing_ok=True)
    except Exception:
        pass
