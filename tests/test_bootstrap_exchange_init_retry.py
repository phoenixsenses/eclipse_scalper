from __future__ import annotations

import asyncio
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

try:
    from execution import bootstrap as bs
except ModuleNotFoundError:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from execution import bootstrap as bs


class _RequestTimeout(Exception):
    pass


class _NetworkError(Exception):
    pass


class _ExchangeNotAvailable(Exception):
    pass


class _DDoSProtection(Exception):
    pass


class _RetryThenOkExchange:
    options = {}
    last_instance = None

    def __init__(self, params):
        self.params = params
        self.load_calls = 0
        self.close_calls = 0
        self.options = dict(params.get("options", {}))
        _RetryThenOkExchange.last_instance = self

    async def load_markets(self):
        self.load_calls += 1
        if self.load_calls < 3:
            raise _RequestTimeout("temporary")
        return {"ok": True}

    async def fetch_time(self):
        return 1_700_000_000_000

    async def close(self):
        self.close_calls += 1


class _AlwaysTimeoutExchange:
    options = {}
    last_instance = None

    def __init__(self, params):
        self.params = params
        self.load_calls = 0
        self.close_calls = 0
        self.options = dict(params.get("options", {}))
        _AlwaysTimeoutExchange.last_instance = self

    async def load_markets(self):
        self.load_calls += 1
        raise _RequestTimeout("always")

    async def close(self):
        self.close_calls += 1


def _install_fake_ccxt(monkeypatch, exchange_cls):
    ccxt_pkg = types.ModuleType("ccxt")
    ccxt_async = types.ModuleType("ccxt.async_support")
    ccxt_async.binance = exchange_cls
    ccxt_async.RequestTimeout = _RequestTimeout
    ccxt_async.NetworkError = _NetworkError
    ccxt_async.ExchangeNotAvailable = _ExchangeNotAvailable
    ccxt_async.DDoSProtection = _DDoSProtection
    monkeypatch.setitem(sys.modules, "ccxt", ccxt_pkg)
    monkeypatch.setitem(sys.modules, "ccxt.async_support", ccxt_async)


def test_init_exchange_retries_then_succeeds(monkeypatch):
    _install_fake_ccxt(monkeypatch, _RetryThenOkExchange)
    monkeypatch.setenv("EXCHANGE", "binance")
    monkeypatch.setenv("BOOT_FAILFAST_PRIVATE_AUTH", "0")
    monkeypatch.setenv("EXCHANGE_INIT_RETRIES", "5")
    monkeypatch.setenv("EXCHANGE_INIT_BACKOFF_SEC", "0")
    monkeypatch.setenv("EXCHANGE_INIT_JITTER_SEC", "0")
    monkeypatch.setenv("EXCHANGE_INIT_FETCH_CURRENCIES", "0")

    sleep_calls = []

    async def _fake_sleep(sec):
        sleep_calls.append(sec)

    monkeypatch.setattr(bs.asyncio, "sleep", _fake_sleep)

    cfg = SimpleNamespace(EXCHANGE="binance", DEFAULT_TYPE="future")
    ex = asyncio.run(bs._init_exchange(cfg))

    assert ex is _RetryThenOkExchange.last_instance
    assert ex.load_calls == 3
    assert ex.options.get("fetchCurrencies") is False
    assert len(sleep_calls) == 2


def test_init_exchange_exhausted_retries_closes_exchange(monkeypatch):
    _install_fake_ccxt(monkeypatch, _AlwaysTimeoutExchange)
    monkeypatch.setenv("EXCHANGE", "binance")
    monkeypatch.setenv("BOOT_FAILFAST_PRIVATE_AUTH", "0")
    monkeypatch.setenv("EXCHANGE_INIT_RETRIES", "3")
    monkeypatch.setenv("EXCHANGE_INIT_BACKOFF_SEC", "0")
    monkeypatch.setenv("EXCHANGE_INIT_JITTER_SEC", "0")

    async def _fake_sleep(_sec):
        return None

    monkeypatch.setattr(bs.asyncio, "sleep", _fake_sleep)

    cfg = SimpleNamespace(EXCHANGE="binance", DEFAULT_TYPE="future")
    with pytest.raises(_RequestTimeout):
        asyncio.run(bs._init_exchange(cfg))

    ex = _AlwaysTimeoutExchange.last_instance
    assert ex is not None
    assert ex.load_calls == 3
    assert ex.close_calls == 1


def test_init_exchange_skip_env_returns_offline_stub(monkeypatch):
    monkeypatch.setenv("BOOTSTRAP_SKIP_EXCHANGE_INIT", "1")
    cfg = SimpleNamespace(EXCHANGE="binance", DEFAULT_TYPE="future")
    ex = asyncio.run(bs._init_exchange(cfg))
    assert type(ex).__name__ == "_OfflineExchangeStub"
    assert isinstance(getattr(ex, "options", {}), dict)
    assert ex.options.get("defaultType") == "stub"
    asyncio.run(ex.close())
