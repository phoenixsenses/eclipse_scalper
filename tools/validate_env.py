#!/usr/bin/env python3
"""
Eclipse Scalper - Environment Validator

Loads a .env file, checks all critical configuration, and prints a
startup checklist. Exits 0 if ready, 1 if any critical check fails.

Usage:
    python -m tools.validate_env                    # uses .env.paper by default
    python -m tools.validate_env --env .env.paper
    python -m tools.validate_env --env .env --live  # allow DRY_RUN=0 (live mode)
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# -- Optional dotenv (not critical - env may already be set) ------------------
def _load_env_file(path: str) -> bool:
    """Load the specified .env file. Returns True if loaded, False if missing."""
    try:
        from dotenv import load_dotenv  # type: ignore
        p = Path(path)
        if not p.exists():
            return False
        load_dotenv(dotenv_path=p, override=True)
        return True
    except Exception:
        return False


# -- Result helpers ------------------------------------------------------------
OK = "OK  "
WARN = "WARN"
FAIL = "FAIL"
SKIP = "--  "

_results: List[Tuple[str, str, str]] = []  # (status, label, detail)
_critical_fail = False
_fail_labels: List[str] = []


def _record(status: str, label: str, detail: str) -> None:
    global _critical_fail, _fail_labels
    _results.append((status, label, detail))
    if status == FAIL:
        _critical_fail = True
        _fail_labels.append(str(label or ""))


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _env_bool(name: str) -> bool:
    return _env(name).lower() in ("1", "true", "yes")


def _validate_debug_enabled() -> bool:
    return _env_bool("VALIDATE_ENV_DEBUG")


def _debug(msg: str) -> None:
    if _validate_debug_enabled():
        print(f"[validate_env.debug] {msg}")


def _sanitize_credential(raw: str) -> str:
    v = str(raw or "").strip()
    if len(v) >= 2 and ((v[0] == "'" and v[-1] == "'") or (v[0] == '"' and v[-1] == '"')):
        v = v[1:-1].strip()
    return v


def _credential_meta(raw: str) -> Dict[str, bool | int]:
    s = str(raw or "")
    return {
        "len": len(s),
        "has_quote": ("'" in s or '"' in s),
        "trail_ws": (s != s.strip()),
    }


def _env_first(*names: str) -> Tuple[str, str]:
    for n in names:
        if n in os.environ and str(os.environ.get(n, "")) != "":
            return n, str(os.environ.get(n, ""))
    return (names[0] if names else "", "")


def _redact_exchange_error(text: str) -> str:
    s = str(text or "")
    s = re.sub(r"(signature=)[^&\\s]+", r"\1<redacted>", s, flags=re.IGNORECASE)
    s = re.sub(r"(timestamp=)[0-9]+", r"\1<redacted>", s, flags=re.IGNORECASE)
    return s


# -- Individual checks --------------------------------------------------------─

def check_paper_mode(allow_live: bool) -> None:
    dry = _env("SCALPER_DRY_RUN")
    if dry == "1":
        _record(OK, "Paper mode", "SCALPER_DRY_RUN=1 (simulation - no real orders)")
    elif allow_live:
        _record(WARN, "Paper mode", "SCALPER_DRY_RUN=0 - LIVE MODE (--live flag acknowledged)")
    else:
        _record(FAIL, "Paper mode",
                "SCALPER_DRY_RUN is not '1'. Pass --live to validate a live config, "
                "or set SCALPER_DRY_RUN=1 in your .env.paper")


def check_exchange_api() -> None:
    key_name, key_raw = _env_first(
        "BINANCE_API_KEY",
        "BINANCE_KEY",
        "BINANCE_APIKEY",
        "BINANCE_FUTURES_API_KEY",
    )
    sec_name, sec_raw = _env_first(
        "BINANCE_API_SECRET",
        "BINANCE_SECRET",
        "BINANCE_APISEC",
        "BINANCE_FUTURES_API_SECRET",
    )
    key = _sanitize_credential(key_raw)
    secret = _sanitize_credential(sec_raw)
    dry_run = (_env("SCALPER_DRY_RUN") == "1")
    skip_auth_in_dryrun = _env_bool("SKIP_EXCHANGE_AUTH_IN_DRYRUN")

    _debug(
        "binance_env "
        f"key_var={key_name} secret_var={sec_name} "
        f"key_len={_credential_meta(key_raw)['len']} secret_len={_credential_meta(sec_raw)['len']} "
        f"key_has_quote={_credential_meta(key_raw)['has_quote']} secret_has_quote={_credential_meta(sec_raw)['has_quote']} "
        f"key_trail_ws={_credential_meta(key_raw)['trail_ws']} secret_trail_ws={_credential_meta(sec_raw)['trail_ws']}"
    )

    if not key or key.startswith("<"):
        _record(WARN, "Exchange API", f"{key_name or 'BINANCE_API_KEY'} not set - bot will run in offline/data-only mode")
        return
    if not secret or secret.startswith("<"):
        _record(WARN, "Exchange API", f"{sec_name or 'BINANCE_API_SECRET'} not set - bot will run in offline/data-only mode")
        return

    if dry_run and skip_auth_in_dryrun:
        _record(SKIP, "Exchange API", "Skipped authenticated check in dry-run (SKIP_EXCHANGE_AUTH_IN_DRYRUN=1)")
        return

    # Optional: connectivity ping (ccxt required)
    try:
        import ccxt  # type: ignore
        exchange = ccxt.binance({"apiKey": key, "secret": secret})
        default_type = _env("DEFAULT_TYPE", "future")
        exchange.options["defaultType"] = default_type
        base_url = ""
        try:
            urls = getattr(exchange, "urls", {}) or {}
            api_urls = urls.get("api", {}) if isinstance(urls, dict) else {}
            if isinstance(api_urls, dict):
                base_url = str(api_urls.get("fapiPublic") or api_urls.get("public") or "")
        except Exception:
            base_url = ""
        recv_window = _env("BINANCE_RECV_WINDOW", "5000")
        ts_ms = int(getattr(exchange, "milliseconds", lambda: 0)())
        _debug(
            "binance_request "
            f"defaultType={default_type} base_url={base_url or '(unknown)'} "
            f"path={( '/fapi/v2/balance' if default_type == 'future' else '/api/v3/account' )} "
            f"timestamp_ms={ts_ms} recvWindow={recv_window}"
        )
        exchange.options["fetchCurrencies"] = False
        exchange.fetch_balance()
        _record(OK, "Exchange API", "Connected (authenticated read-only ping succeeded)")
    except ImportError:
        _record(WARN, "Exchange API",
                "ccxt not importable - skipping connectivity check (keys look set)")
    except Exception as exc:
        _record(FAIL, "Exchange API", f"Authentication failed: {_redact_exchange_error(str(exc))}")


def check_database() -> None:
    db_candidates = [
        Path("data/microstructure.db"),
        Path("data/market_data.db"),
    ]
    found = None
    for p in db_candidates:
        if p.exists():
            size_mb = p.stat().st_size / 1024 / 1024
            found = (p, size_mb)
            break
    if found:
        p, size_mb = found
        _record(OK, "Database", f"{p} ({size_mb:.1f} MB)")
    else:
        _record(WARN, "Database",
                "No microstructure database found at data/microstructure.db or data/market_data.db "
                "- data collection must be running")


def check_trade_log_db() -> None:
    db_path = _env("ENTRY_TRADE_LOG_DB", "data/paper_trades.db")
    p = Path(db_path)
    # Check parent directory is writable
    parent = p.parent
    if not parent.exists():
        try:
            parent.mkdir(parents=True, exist_ok=True)
            _record(OK, "Trade log DB", f"{db_path} (directory created: {parent})")
        except OSError as exc:
            _record(FAIL, "Trade log DB", f"Cannot create directory {parent}: {exc}")
        return
    # Check writability by probing a temp file
    test_file = parent / ".validate_env_write_test"
    try:
        test_file.write_text("ok")
        test_file.unlink()
        status = "exists" if p.exists() else "will be created"
        _record(OK, "Trade log DB", f"{db_path} (writable, {status})")
    except OSError as exc:
        _record(FAIL, "Trade log DB", f"Directory {parent} is not writable: {exc}")


def check_regime_gate() -> None:
    regime = _env("ENTRY_REGIME", "none")
    valid = {"none", "up", "down"}
    if regime not in valid:
        _record(WARN, "Regime gate",
                f"ENTRY_REGIME='{regime}' is not a valid value (none|up|down)")
        return
    if regime == "none":
        _record(WARN, "Regime gate", "ENTRY_REGIME=none - gate disabled (all regimes accepted)")
    else:
        block_trans = _env("ENTRY_REGIME_BLOCK_TRANSITION", "0")
        block_unk = _env("ENTRY_REGIME_BLOCK_UNKNOWN", "0")
        _record(OK, "Regime gate",
                f"ENABLED (ENTRY_REGIME={regime}, block_transition={block_trans}, block_unknown={block_unk})")


def check_scratch_engine() -> None:
    if _env_bool("EXIT_SCRATCH_ENABLED"):
        adverse = _env("EXIT_SCRATCH_ADVERSE_BPS", "5.0")
        horizon = _env("EXIT_SCRATCH_HARD_HORIZON_SEC", "120")
        _record(OK, "Scratch engine",
                f"ENABLED (adverse_bps={adverse}, hard_horizon={horizon}s)")
    else:
        _record(SKIP, "Scratch engine", "DISABLED (EXIT_SCRATCH_ENABLED=0)")


def check_order_placement() -> None:
    mode = _env("MICRO_SIGNAL_ORDER_PLACEMENT_MODE", "best").strip().lower() or "best"
    valid = {"best", "inside_spread", "adaptive"}
    if mode not in valid:
        _record(
            WARN,
            "Order placement",
            f"MICRO_SIGNAL_ORDER_PLACEMENT_MODE={mode!r} invalid (valid: best|inside_spread|adaptive); runtime fallback=best",
        )
        mode = "best"
    timeout = _env("MICRO_SIGNAL_FILL_TIMEOUT_SEC", "10")
    depth = _env("MICRO_SIGNAL_QUEUE_DEPTH_THRESHOLD", "500")
    min_fill_prob = _env("MICRO_SIGNAL_MIN_FILL_PROB", "0.35")
    lookback = _env("MICRO_SIGNAL_FLOW_LOOKBACK_SEC", "30")
    _record(
        OK,
        "Order placement",
        (
            f"mode={mode} fill_timeout={timeout}s queue_depth_threshold={depth} "
            f"min_fill_prob={min_fill_prob} flow_lookback={lookback}s"
        ),
    )


def check_risk_manager() -> None:
    if not _env_bool("ENTRY_REGIME_RISK_ENABLED"):
        _record(WARN, "Risk manager", "DISABLED (ENTRY_REGIME_RISK_ENABLED=0)")
        return

    # Validate bps values are positive floats
    bps_vars = {
        "RISK_MAX_DAILY_LOSS_BPS": _env("RISK_MAX_DAILY_LOSS_BPS"),
        "RISK_MAX_DRAWDOWN_BPS": _env("RISK_MAX_DRAWDOWN_BPS"),
    }
    errors = []
    for name, val in bps_vars.items():
        if not val:
            errors.append(f"{name} not set")
            continue
        try:
            f = float(val)
            if f <= 0:
                errors.append(f"{name}={val} must be > 0")
        except ValueError:
            errors.append(f"{name}={val} is not a valid float")

    if errors:
        _record(FAIL, "Risk manager", "; ".join(errors))
        return

    max_pos = _env("RISK_MAX_CONCURRENT_POSITIONS", "1")
    daily_bps = _env("RISK_MAX_DAILY_LOSS_BPS")
    drawdown = _env("RISK_MAX_DRAWDOWN_BPS")
    _record(OK, "Risk manager",
            f"ENABLED (max_pos={max_pos}, daily_limit={daily_bps}bps, drawdown={drawdown}bps)")


def check_trade_logger() -> None:
    entry_on = _env_bool("ENTRY_TRADE_LOGGER_ENABLED")
    exit_on = _env_bool("EXIT_TRADE_LOGGER_ENABLED")
    db = _env("ENTRY_TRADE_LOG_DB", "data/paper_trades.db")
    if entry_on and exit_on:
        _record(OK, "Trade logger", f"ENABLED (entry+exit -> {db})")
    elif entry_on:
        _record(WARN, "Trade logger", f"entry only (EXIT_TRADE_LOGGER_ENABLED=0) -> {db}")
    elif exit_on:
        _record(WARN, "Trade logger", f"exit only (ENTRY_TRADE_LOGGER_ENABLED=0) -> {db}")
    else:
        _record(WARN, "Trade logger", "DISABLED - no trade records will be written")


def check_telegram() -> None:
    token = _env("TELEGRAM_TOKEN")
    chat_id = _env("TELEGRAM_CHAT_ID")
    if token and not token.startswith("<") and chat_id and not chat_id.startswith("<"):
        _record(OK, "Telegram", "Configured")
    elif not token or token.startswith("<"):
        _record(WARN, "Telegram", "TELEGRAM_TOKEN not set - notifications disabled")
    else:
        _record(WARN, "Telegram", "TELEGRAM_CHAT_ID not set - notifications disabled")


def check_exit_loop() -> None:
    if not _env_bool("EXIT_ENABLED"):
        _record(WARN, "Exit loop", "DISABLED (EXIT_ENABLED=0) - positions will not be closed automatically")
        return
    hold = _env("EXIT_MAX_HOLD_SEC", "120")
    tick = _env("EXIT_TICK_SEC", "2.0")
    _record(OK, "Exit loop", f"ENABLED (max_hold={hold}s, tick={tick}s)")


# -- Main ----------------------------------------------------------------------

def main() -> int:
    try:
        parser = argparse.ArgumentParser(
            description="Eclipse Scalper environment validator"
        )
        parser.add_argument(
            "--env",
            default=".env.paper",
            help="Path to .env file to validate (default: .env.paper)"
        )
        parser.add_argument(
            "--live",
            action="store_true",
            help="Allow SCALPER_DRY_RUN=0 (live trading config validation)"
        )
        args = parser.parse_args()

        # Load .env file
        env_path = Path(args.env)
        env_exists = env_path.exists()
        env_loaded = _load_env_file(args.env)
        if env_loaded:
            env_label = f"Loaded from {args.env}"
        elif env_exists:
            env_label = f"Env file exists but could not be loaded via dotenv: {args.env} (using process env)"
        else:
            env_label = f"File not found: {args.env} (using process env)"

        # Run all checks
        check_paper_mode(allow_live=args.live)
        check_exchange_api()
        check_database()
        check_trade_log_db()
        check_regime_gate()
        check_risk_manager()
        check_trade_logger()
        check_scratch_engine()
        check_order_placement()
        check_exit_loop()
        check_telegram()

        # Print banner
        width = 58
        print()
        print("=" * width)
        print(" Eclipse Scalper - Environment Validation")
        print(f" {env_label}")
        print("=" * width)
        for status, label, detail in _results:
            print(f"  [{status}] {label:<20} {detail}")
        print("=" * width)
        if _critical_fail:
            print("  Status: NOT READY - fix FAIL items before starting")
        else:
            print("  Status: READY FOR PAPER TRADING" if not args.live else "  Status: READY (live mode acknowledged)")
        print("=" * width)
        print()

        if not _critical_fail:
            return 0
        # exit code normalization
        if any(lbl == "Exchange API" for lbl in _fail_labels):
            return 3
        return 2
    except SystemExit:
        raise
    except Exception as exc:
        print(f"validate_env unexpected error: {type(exc).__name__}: {exc}")
        return 4


if __name__ == "__main__":
    raise SystemExit(main())
