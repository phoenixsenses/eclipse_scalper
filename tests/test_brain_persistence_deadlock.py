from __future__ import annotations

import asyncio
import time
from pathlib import Path
from types import SimpleNamespace

try:
    import brain.persistence as bp
except ModuleNotFoundError:  # pragma: no cover
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import brain.persistence as bp


def test_post_load_save_skips_when_lock_held(monkeypatch) -> None:
    called = {"n": 0}

    async def _fake_save(_state, force=False):
        called["n"] += 1

    async def _run() -> None:
        monkeypatch.setattr(bp, "save_brain", _fake_save)
        await bp._IO_LOCK.acquire()
        try:
            await bp._post_load_force_save_best_effort(SimpleNamespace(), timeout_sec=0.1)
        finally:
            if bp._IO_LOCK.locked():
                bp._IO_LOCK.release()

    asyncio.run(_run())
    assert called["n"] == 0


def test_post_load_save_calls_when_unlocked(monkeypatch) -> None:
    called = {"n": 0}

    async def _fake_save(_state, force=False):
        called["n"] += 1

    async def _run() -> None:
        monkeypatch.setattr(bp, "save_brain", _fake_save)
        await bp._post_load_force_save_best_effort(SimpleNamespace(), timeout_sec=0.2)

    asyncio.run(_run())
    assert called["n"] == 1


def test_post_load_save_timeout_is_non_blocking(monkeypatch) -> None:
    async def _slow_save(_state, force=False):
        await asyncio.sleep(5.0)

    async def _run() -> float:
        monkeypatch.setattr(bp, "save_brain", _slow_save)
        t0 = time.perf_counter()
        await bp._post_load_force_save_best_effort(SimpleNamespace(), timeout_sec=0.05)
        return time.perf_counter() - t0

    elapsed = asyncio.run(_run())
    assert elapsed < 1.0
