# execution/entry_loop_full.py — SCALPER ETERNAL — ENTRY LOOP (FULL RISK) — 2026 v1.0
# Purpose:
# - Bootstrap-friendly entry loop that calls execution.entry.try_enter()
# - Preserves full risk sizing + ATR stop/TP ladder logic
# - Adds lightweight per-symbol cooldown + cadence controls

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict

from utils.logging import log_core, log_entry
from execution.entry import try_enter

try:
    from risk.kill_switch import trade_allowed  # type: ignore
except Exception:
    trade_allowed = None

def _now() -> float:
    return time.time()


def _cfg(bot, name: str, default: Any) -> Any:
    try:
        return getattr(getattr(bot, "cfg", None), name, default)
    except Exception:
        return default


def _truthy(x) -> bool:
    if x is True:
        return True
    if isinstance(x, (int, float)) and x != 0:
        return True
    if isinstance(x, str) and x.strip().lower() in ("true", "1", "yes", "y", "on"):
        return True
    return False


def _cfg_env_float(bot, name: str, default: float) -> float:
    """
    ENV wins, then cfg, then default.
    """
    try:
        v = str((__import__("os").environ.get(name, "") or "")).strip()
        if v != "":
            return float(v)
    except Exception:
        pass
    try:
        return float(_cfg(bot, name, default) or default)
    except Exception:
        return float(default)


def _cfg_env_bool(bot, name: str, default: Any = False) -> bool:
    try:
        v = str((__import__("os").environ.get(name, "") or "")).strip()
        if v != "":
            return _truthy(v)
    except Exception:
        pass
    return _truthy(_cfg(bot, name, default))


def _symkey(sym: str) -> str:
    s = (sym or "").upper().strip()
    s = s.replace("/USDT:USDT", "USDT").replace("/USDT", "USDT")
    s = s.replace(":USDT", "USDT").replace(":", "")
    s = s.replace("/", "")
    if s.endswith("USDTUSDT"):
        s = s[:-4]
    return s


def _ensure_shutdown_event(bot) -> asyncio.Event:
    ev = getattr(bot, "_shutdown", None)
    if isinstance(ev, asyncio.Event):
        return ev
    ev = asyncio.Event()
    try:
        bot._shutdown = ev  # type: ignore[attr-defined]
    except Exception:
        pass
    return ev


def _pick_symbols(bot) -> list[str]:
    try:
        s = getattr(bot, "active_symbols", None)
        if isinstance(s, set) and s:
            return sorted(list(s))
        if isinstance(s, (list, tuple)) and s:
            return [str(x) for x in s if str(x).strip()]
    except Exception:
        pass

    try:
        s2 = getattr(getattr(bot, "cfg", None), "ACTIVE_SYMBOLS", None)
        if isinstance(s2, (list, tuple)) and s2:
            return [str(x) for x in s2 if str(x).strip()]
    except Exception:
        pass

    return ["BTCUSDT"]


async def entry_loop_full(bot) -> None:
    """
    Bootstrap entry loop that invokes execution.entry.try_enter().
    """
    shutdown_ev = _ensure_shutdown_event(bot)

    poll_sec = _cfg_env_float(bot, "ENTRY_POLL_SEC", 1.0)
    per_symbol_gap_sec = _cfg_env_float(bot, "ENTRY_PER_SYMBOL_GAP_SEC", 2.5)
    local_cooldown_sec = _cfg_env_float(bot, "ENTRY_LOCAL_COOLDOWN_SEC", 8.0)

    spawn_both = _cfg_env_bool(bot, "SPAWN_BOTH_SIDES", False)
    respect_kill = _cfg_env_bool(bot, "ENTRY_RESPECT_KILL_SWITCH", True)
    diag = _cfg_env_bool(bot, "SCALPER_SIGNAL_DIAG", _cfg(bot, "SCALPER_SIGNAL_DIAG", "0"))

    last_attempt_by_sym: Dict[str, float] = {}
    last_tick = 0.0

    log_core.info("ENTRY_LOOP_FULL ONLINE — using execution.entry.try_enter()")

    while not shutdown_ev.is_set():
        try:
            now = _now()
            if poll_sec > 0 and (now - last_tick) < poll_sec:
                await asyncio.sleep(max(0.05, poll_sec - (now - last_tick)))
            last_tick = _now()

            # optional kill-switch gate (mirror basic entry loop behavior)
            if respect_kill and callable(trade_allowed):
                try:
                    ok = await trade_allowed(bot)
                    if not ok:
                        await asyncio.sleep(max(0.25, poll_sec))
                        continue
                except Exception:
                    await asyncio.sleep(max(0.25, poll_sec))
                    continue

            syms = _pick_symbols(bot)
            if not syms:
                await asyncio.sleep(max(0.25, poll_sec))
                continue

            for sym in syms:
                if shutdown_ev.is_set():
                    break

                k = _symkey(sym)
                if not k:
                    continue

                # local cooldown between attempts
                la = float(last_attempt_by_sym.get(k, 0.0) or 0.0)
                if local_cooldown_sec > 0 and (_now() - la) < local_cooldown_sec:
                    continue
                last_attempt_by_sym[k] = _now()

                if diag:
                    log_entry.info(f"ENTRY_LOOP_FULL scan {k}")

                if spawn_both:
                    await try_enter(bot, k, "long")
                    await try_enter(bot, k, "short")
                else:
                    side = "long" if (hash(k) ^ int(_now() // max(1.0, local_cooldown_sec))) % 2 == 0 else "short"
                    await try_enter(bot, k, side)

                await asyncio.sleep(max(0.01, per_symbol_gap_sec))

        except asyncio.CancelledError:
            raise
        except Exception as e:
            log_entry.error(f"ENTRY_LOOP_FULL outer error: {e}")
            await asyncio.sleep(1.0)

    log_core.critical("ENTRY_LOOP_FULL OFFLINE — shutdown flag set")
