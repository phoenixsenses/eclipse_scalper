from __future__ import annotations

from pathlib import Path
import os
import asyncio
import pytest

try:
    from execution import bootstrap as bs
except ModuleNotFoundError:  # pragma: no cover
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from execution import bootstrap as bs


def test_sanitize_credential_trims_quotes_and_ws() -> None:
    assert bs._sanitize_credential("  'abc123'  ") == "abc123"
    assert bs._sanitize_credential('  "abc123"  ') == "abc123"
    assert bs._sanitize_credential("  abc123  ") == "abc123"


def test_sanitize_credential_strips_inline_comment_tail() -> None:
    assert bs._sanitize_credential("abc123 # from .env") == "abc123"
    assert bs._sanitize_credential("'abc123' # note") == "abc123"


def test_quoted_like_detection() -> None:
    assert bs._is_quoted_like("'x'") is True
    assert bs._is_quoted_like('"x"') is True
    assert bs._is_quoted_like("x") is False


def test_has_trailing_ws_detection() -> None:
    assert bs._has_trailing_ws(" abc") is True
    assert bs._has_trailing_ws("abc ") is True
    assert bs._has_trailing_ws("abc") is False


def test_env_first_present_prefers_first_non_empty() -> None:
    old_a = os.environ.get("X_A")
    old_b = os.environ.get("X_B")
    try:
        os.environ["X_A"] = "   "
        os.environ["X_B"] = "value_b"
        k, v = bs._env_first_present(["X_A", "X_B"])
        assert k == "X_B"
        assert v == "value_b"
    finally:
        if old_a is None:
            os.environ.pop("X_A", None)
        else:
            os.environ["X_A"] = old_a
        if old_b is None:
            os.environ.pop("X_B", None)
        else:
            os.environ["X_B"] = old_b


def test_auth_signature_error_detection() -> None:
    e1 = Exception('binance {"code":-1022,"msg":"Signature for this request is not valid."}')
    e2 = Exception("AuthenticationError: invalid api-key")
    e3 = Exception("timeout while connecting")
    assert bs._looks_auth_signature_error(e1) is True
    assert bs._looks_auth_signature_error(e2) is True
    assert bs._looks_auth_signature_error(e3) is False


class _DummyExchangeAuthErr:
    options = {"defaultType": "future"}

    async def fetch_positions(self):
        raise Exception('binance {"code":-1022,"msg":"Signature for this request is not valid."}')


def test_assert_private_auth_or_fail_raises_signature_error() -> None:
    diag = {
        "key_len": 10,
        "secret_len": 10,
        "key_quoted": False,
        "secret_quoted": False,
        "key_trail_ws": False,
        "secret_trail_ws": False,
        "key_source": "BINANCE_API_KEY",
        "secret_source": "BINANCE_API_SECRET",
        "testnet": False,
        "default_type": "future",
        "recv_window": 10000,
        "server_drift_ms": 12,
    }
    async def _run() -> None:
        with pytest.raises(RuntimeError, match="signature/auth"):
            await bs._assert_private_auth_or_fail(_DummyExchangeAuthErr(), diag=diag)

    asyncio.run(_run())
