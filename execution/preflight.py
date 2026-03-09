# execution/preflight.py -- SCALPER ETERNAL -- PREFLIGHT CHECKS -- 2026 v1.0
# Pre-loop validation: exchange connectivity, API permissions, hedge mode,
# symbol existence, data directory writability.
# Guardian-safe: never raises, never blocks startup.

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any, Dict

from utils.logging import log_core

# ---------------------------------------------------------------------------
# Result constants
# ---------------------------------------------------------------------------
PASS = "pass"
FAIL = "fail"
SKIP = "skip"
WARN = "warn"


def _result(name: str, status: str, detail: str = "") -> Dict[str, str]:
    return {"check": name, "status": status, "detail": detail}


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

async def _check_exchange_connectivity(bot: Any) -> Dict[str, str]:
    """Can we reach the exchange at all?"""
    try:
        ex = getattr(bot, "ex", None)
        if ex is None:
            return _result("exchange_connectivity", SKIP, "no exchange object")
        await ex.fetch_time()
        return _result("exchange_connectivity", PASS)
    except Exception as e:
        return _result("exchange_connectivity", FAIL, str(e))


async def _check_api_permissions(bot: Any) -> Dict[str, str]:
    """Can we create/cancel orders? (checks canTrade flag)"""
    try:
        ex = getattr(bot, "ex", None)
        if ex is None:
            return _result("api_permissions", SKIP, "no exchange object")
        resp = None
        if hasattr(ex, "fapiPrivateV2GetAccount"):
            resp = await ex.fapiPrivateV2GetAccount()
        elif hasattr(ex, "fapiPrivate_get_account"):
            resp = await ex.fapiPrivate_get_account()
        if not isinstance(resp, dict):
            return _result("api_permissions", SKIP, "no account response")
        can_trade = resp.get("canTrade")
        if can_trade is True:
            return _result("api_permissions", PASS)
        if can_trade is False:
            return _result("api_permissions", FAIL, "canTrade=False")
        return _result("api_permissions", SKIP, "canTrade field absent")
    except Exception as e:
        return _result("api_permissions", SKIP, str(e))


async def _check_hedge_mode(bot: Any) -> Dict[str, str]:
    """Binance futures requires hedge (dual-side) mode."""
    try:
        ex = getattr(bot, "ex", None)
        if ex is None:
            return _result("hedge_mode", SKIP, "no exchange object")
        resp = None
        if hasattr(ex, "fapiPrivateGetPositionSideDual"):
            resp = await ex.fapiPrivateGetPositionSideDual()
        elif hasattr(ex, "fapiPrivate_get_positionside_dual"):
            resp = await ex.fapiPrivate_get_positionside_dual()
        if not isinstance(resp, dict):
            return _result("hedge_mode", SKIP, "no response")
        dual = resp.get("dualSidePosition", False)
        if dual:
            return _result("hedge_mode", PASS)
        return _result("hedge_mode", FAIL, "hedge mode OFF -- bot requires dual-side")
    except Exception as e:
        return _result("hedge_mode", SKIP, str(e))


async def _check_symbol_existence(bot: Any) -> Dict[str, str]:
    """Are configured symbols present on exchange?"""
    try:
        ex = getattr(bot, "ex", None)
        if ex is None:
            return _result("symbol_existence", SKIP, "no exchange object")
        markets = getattr(ex, "markets", None) or {}
        if not markets:
            return _result("symbol_existence", SKIP, "markets not loaded")

        cfg = getattr(bot, "cfg", None)
        active = getattr(cfg, "ACTIVE_SYMBOLS", None) or []
        if not active:
            raw = os.getenv("ACTIVE_SYMBOLS", "").strip()
            if raw:
                active = [s.strip().upper() for s in raw.split(",") if s.strip()]
        if not active:
            return _result("symbol_existence", SKIP, "no active symbols configured")

        known = set(markets.keys())
        for m in markets.values():
            if isinstance(m, dict) and m.get("id"):
                known.add(str(m["id"]).upper())

        missing = [s for s in active if s not in known and f"{s}/USDT:USDT" not in known]
        if missing:
            return _result("symbol_existence", FAIL, f"{len(missing)} missing: {missing[:5]}")
        return _result("symbol_existence", PASS, f"all {len(active)} symbols found")
    except Exception as e:
        return _result("symbol_existence", SKIP, str(e))


def _check_data_dir_writable() -> Dict[str, str]:
    """Can we write to logs/ and state/ directories?"""
    try:
        problems = []
        for d in (Path("logs"), Path("state")):
            d.mkdir(parents=True, exist_ok=True)
            probe = d / ".preflight_probe"
            try:
                probe.write_text("ok", encoding="utf-8")
                probe.unlink(missing_ok=True)
            except Exception as e:
                problems.append(f"{d}: {e}")
        if problems:
            return _result("data_dir_writable", FAIL, "; ".join(problems))
        return _result("data_dir_writable", PASS)
    except Exception as e:
        return _result("data_dir_writable", FAIL, str(e))


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

async def run_preflight(bot: Any) -> Dict[str, str]:
    """Run all preflight checks. Returns {check_name: status}.

    Guardian-safe: never raises, never blocks startup.
    Caller decides whether to proceed based on results.
    """
    results: Dict[str, str] = {}
    checks = []
    try:
        checks = [
            _check_exchange_connectivity(bot),
            _check_api_permissions(bot),
            _check_hedge_mode(bot),
            _check_symbol_existence(bot),
        ]
    except Exception as e:
        log_core.warning(f"PREFLIGHT: failed to build check list: {e}")
        return results

    # Run async checks
    for coro in checks:
        try:
            r = await coro
            results[r["check"]] = r["status"]
            _log_result(r)
        except Exception as e:
            log_core.warning(f"PREFLIGHT: check failed unexpectedly: {e}")

    # Sync checks
    try:
        r = _check_data_dir_writable()
        results[r["check"]] = r["status"]
        _log_result(r)
    except Exception as e:
        log_core.warning(f"PREFLIGHT: data_dir check error: {e}")

    passed = sum(1 for v in results.values() if v == PASS)
    total = len(results)
    log_core.info(f"PREFLIGHT COMPLETE: {passed}/{total} passed")
    return results


def _log_result(r: Dict[str, str]) -> None:
    status = r.get("status", "?")
    name = r.get("check", "?")
    detail = r.get("detail", "")
    msg = f"PREFLIGHT [{status.upper()}] {name}"
    if detail:
        msg += f" -- {detail}"
    if status == FAIL:
        log_core.critical(msg)
    elif status == WARN:
        log_core.warning(msg)
    else:
        log_core.info(msg)
