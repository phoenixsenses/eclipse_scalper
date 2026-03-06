"""
Tests for tools/validate_env.py

All checks run against monkeypatched os.environ. _load_env_file is
stubbed out so real .env files on disk never override patched env.
No real exchange calls (ccxt.binance is mocked where needed).
Uses tempfile.mkdtemp() instead of tmp_path to avoid Windows permission
issues with pytest's default tmpdir location.
"""

from __future__ import annotations

import os
import sys
import tempfile
import shutil
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ── helpers ───────────────────────────────────────────────────────────────────

def _run_main(monkeypatch, env: dict, argv: list[str] | None = None) -> int:
    """
    Load tools.validate_env with a fresh import (reset module globals),
    patch os.environ to `env`, stub out _load_env_file so no .env file
    on disk can override the patched env, then call main().
    Returns the integer exit code.
    """
    with monkeypatch.context() as m:
        m.setattr(os, "environ", dict(env))

        if argv is not None:
            m.setattr(sys, "argv", ["validate_env"] + argv)
        else:
            m.setattr(sys, "argv", ["validate_env", "--env", ".env.paper"])

        # Fresh import to reset _results / _critical_fail globals
        if "tools.validate_env" in sys.modules:
            del sys.modules["tools.validate_env"]

        import tools.validate_env as ve

        # Prevent _load_env_file from loading the real .env.paper and
        # overriding the monkeypatched environment.
        ve._load_env_file = lambda _path: True

        with patch("builtins.print"):
            return ve.main()


def _minimal_paper_env() -> dict:
    """Minimum valid paper trading environment."""
    return {
        "SCALPER_DRY_RUN": "1",
        "ACTIVE_SYMBOLS": "ETHUSDT",
        "FIXED_NOTIONAL_USDT": "15",
        "ENTRY_REGIME": "up",
        "ENTRY_REGIME_RISK_ENABLED": "1",
        "RISK_MAX_DAILY_LOSS_BPS": "50",
        "RISK_MAX_DRAWDOWN_BPS": "100",
        "RISK_MAX_CONCURRENT_POSITIONS": "1",
        "ENTRY_TRADE_LOGGER_ENABLED": "1",
        "EXIT_TRADE_LOGGER_ENABLED": "1",
        "ENTRY_TRADE_LOG_DB": "data/paper_trades.db",
        "EXIT_TRADE_LOG_DB": "data/paper_trades.db",
        "EXIT_SCRATCH_ENABLED": "0",
        "EXIT_ENABLED": "1",
        "EXIT_MAX_HOLD_SEC": "120",
        "TELEGRAM_TOKEN": "123456:AAAA",
        "TELEGRAM_CHAT_ID": "-1001234567",
    }


# ── load_env_file ─────────────────────────────────────────────────────────────

def test_load_env_file_returns_false_for_missing_file():
    if "tools.validate_env" in sys.modules:
        del sys.modules["tools.validate_env"]
    import tools.validate_env as ve
    result = ve._load_env_file("/nonexistent/path/.env")
    assert result is False


def test_load_env_file_returns_true_for_existing_file():
    tmpdir = Path("data") / f"validate_env_tmp_{uuid.uuid4().hex}"
    tmpdir.mkdir(parents=True, exist_ok=True)
    try:
        env_file = Path(tmpdir) / ".env.test"
        env_file.write_text("TEST_VAR=hello\n")
        if "tools.validate_env" in sys.modules:
            del sys.modules["tools.validate_env"]
        import tools.validate_env as ve
        result = ve._load_env_file(str(env_file))
        assert result is True
    finally:
        shutil.rmtree(str(tmpdir), ignore_errors=True)


# ── paper mode check ──────────────────────────────────────────────────────────

def test_dry_run_must_be_1(monkeypatch):
    """validate_env returns 1 when SCALPER_DRY_RUN is '0'."""
    env = {**_minimal_paper_env(), "SCALPER_DRY_RUN": "0"}
    rc = _run_main(monkeypatch, env)
    assert rc == 2


def test_dry_run_missing_fails(monkeypatch):
    """validate_env returns 1 when SCALPER_DRY_RUN is absent."""
    env = {k: v for k, v in _minimal_paper_env().items() if k != "SCALPER_DRY_RUN"}
    rc = _run_main(monkeypatch, env)
    assert rc == 2


def test_dry_run_live_flag_allows_dry_run_0(monkeypatch):
    """--live flag allows SCALPER_DRY_RUN=0 (no FAIL on paper mode check)."""
    env = {**_minimal_paper_env(), "SCALPER_DRY_RUN": "0"}
    rc = _run_main(monkeypatch, env, argv=["--env", ".env.paper", "--live"])
    # Paper mode check should not cause a FAIL with --live
    # Other checks may warn but shouldn't fail on the minimal env
    assert rc == 0


# ── risk manager checks ───────────────────────────────────────────────────────

def test_negative_daily_loss_bps_fails(monkeypatch):
    """validate_env returns 1 when RISK_MAX_DAILY_LOSS_BPS is negative."""
    env = {**_minimal_paper_env(), "RISK_MAX_DAILY_LOSS_BPS": "-10"}
    rc = _run_main(monkeypatch, env)
    assert rc == 2


def test_zero_drawdown_bps_fails(monkeypatch):
    """validate_env returns 1 when RISK_MAX_DRAWDOWN_BPS is zero."""
    env = {**_minimal_paper_env(), "RISK_MAX_DRAWDOWN_BPS": "0"}
    rc = _run_main(monkeypatch, env)
    assert rc == 2


def test_non_numeric_bps_fails(monkeypatch):
    """validate_env returns 1 when a BPS value is not a valid float."""
    env = {**_minimal_paper_env(), "RISK_MAX_DAILY_LOSS_BPS": "lots"}
    rc = _run_main(monkeypatch, env)
    assert rc == 2


# ── regime gate ───────────────────────────────────────────────────────────────

def test_invalid_regime_mode_warns_not_fails(monkeypatch):
    """Invalid ENTRY_REGIME generates a WARN but not a FAIL (returns 0)."""
    env = {**_minimal_paper_env(), "ENTRY_REGIME": "sideways"}
    rc = _run_main(monkeypatch, env)
    assert rc == 0


def test_regime_none_warns_not_fails(monkeypatch):
    """ENTRY_REGIME=none generates a WARN but not a FAIL."""
    env = {**_minimal_paper_env(), "ENTRY_REGIME": "none"}
    rc = _run_main(monkeypatch, env)
    assert rc == 0


# ── complete valid config ─────────────────────────────────────────────────────

def test_complete_valid_paper_config(monkeypatch):
    """validate_env returns 0 with a fully valid paper config (no exchange keys)."""
    # No exchange keys → check_exchange_api will WARN (not FAIL)
    # check_database will WARN (no microstructure.db) → not FAIL
    # Everything else valid → should return 0
    rc = _run_main(monkeypatch, _minimal_paper_env())
    assert rc == 0


def test_trade_log_dir_created_if_missing(monkeypatch):
    """validate_env creates the trade log directory when it doesn't exist."""
    tmpdir = Path("data") / f"validate_env_tmp_{uuid.uuid4().hex}"
    tmpdir.mkdir(parents=True, exist_ok=True)
    try:
        db_path = str(Path(tmpdir) / "new_subdir" / "paper_trades.db")
        env = {**_minimal_paper_env(), "ENTRY_TRADE_LOG_DB": db_path}
        rc = _run_main(monkeypatch, env)
        assert rc == 0
        assert (Path(tmpdir) / "new_subdir").exists()
    finally:
        shutil.rmtree(str(tmpdir), ignore_errors=True)


# ── exchange API ──────────────────────────────────────────────────────────────

def test_missing_api_key_warns_not_fails(monkeypatch):
    """Missing BINANCE_API_KEY produces a WARN, not a FAIL."""
    env = {k: v for k, v in _minimal_paper_env().items()
           if k not in ("BINANCE_API_KEY", "BINANCE_API_SECRET")}
    rc = _run_main(monkeypatch, env)
    assert rc == 0


def test_placeholder_api_key_warns_not_fails(monkeypatch):
    """Placeholder API key (starting with <) produces a WARN, not a FAIL."""
    env = {**_minimal_paper_env(),
           "BINANCE_API_KEY": "<your_key>",
           "BINANCE_API_SECRET": "<your_secret>"}
    rc = _run_main(monkeypatch, env)
    assert rc == 0


def test_invalid_api_key_fails(monkeypatch):
    """An actually invalid (non-placeholder) key that fails ccxt auth → FAIL."""
    env = {**_minimal_paper_env(),
           "BINANCE_API_KEY": "invalid_key_12345",
           "BINANCE_API_SECRET": "invalid_secret_67890"}

    mock_exchange = MagicMock()
    mock_exchange.options = {}
    mock_exchange.fetch_balance.side_effect = Exception("AuthenticationError: invalid key")
    mock_ccxt = MagicMock()
    mock_ccxt.binance.return_value = mock_exchange

    with patch.dict(sys.modules, {"ccxt": mock_ccxt}):
        rc = _run_main(monkeypatch, env)
    assert rc == 3


def test_exchange_alias_and_quote_strip(monkeypatch):
    """Alias vars + quoted values are sanitized before ccxt client init."""
    env = {**_minimal_paper_env()}
    env.pop("BINANCE_API_KEY", None)
    env.pop("BINANCE_API_SECRET", None)
    env["BINANCE_KEY"] = "  'alias_key_123'  "
    env["BINANCE_SECRET"] = '  "alias_secret_456"  '

    mock_exchange = MagicMock()
    mock_exchange.options = {}
    mock_exchange.fetch_balance.return_value = {"ok": True}
    mock_ccxt = MagicMock()
    mock_ccxt.binance.return_value = mock_exchange

    with patch.dict(sys.modules, {"ccxt": mock_ccxt}):
        rc = _run_main(monkeypatch, env)
    assert rc == 0
    kwargs = mock_ccxt.binance.call_args[0][0]
    assert kwargs["apiKey"] == "alias_key_123"
    assert kwargs["secret"] == "alias_secret_456"


def test_skip_exchange_auth_in_dryrun(monkeypatch):
    """Dry-run can skip strict auth when SKIP_EXCHANGE_AUTH_IN_DRYRUN=1."""
    env = {**_minimal_paper_env(), "BINANCE_API_KEY": "bad", "BINANCE_API_SECRET": "bad", "SKIP_EXCHANGE_AUTH_IN_DRYRUN": "1"}
    rc = _run_main(monkeypatch, env)
    assert rc == 0


# ── trade log ─────────────────────────────────────────────────────────────────

def test_trade_log_writable_dir_passes(monkeypatch):
    """validate_env returns 0 when trade log directory is writable."""
    tmpdir = Path("data") / f"validate_env_tmp_{uuid.uuid4().hex}"
    tmpdir.mkdir(parents=True, exist_ok=True)
    try:
        db_path = str(Path(tmpdir) / "paper_trades.db")
        env = {**_minimal_paper_env(), "ENTRY_TRADE_LOG_DB": db_path}
        rc = _run_main(monkeypatch, env)
        assert rc == 0
    finally:
        shutil.rmtree(str(tmpdir), ignore_errors=True)
