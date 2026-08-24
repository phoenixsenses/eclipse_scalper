# execution/diagnostics.py — SCALPER ETERNAL — DIAGNOSTIC PULSE — v0.4 (LOGGER-AWARE)
# Patch vs v0.3:
# - ✅ All print() replaced with log_core.info() (log aggregation compatible)
# - ✅ CLI mode: StreamHandler on utils.logging ensures stdout visibility
# - ✅ Fallback to print() only when logger import fails

import os
import time
import platform
import sys
import inspect
import logging

try:
    from utils.logging import log_core as _log  # type: ignore
except Exception:
    _log = logging.getLogger("diagnostics")

def _out(msg: str) -> None:
    """Write diagnostic line via logger; never raises."""
    try:
        _log.info(msg)
    except Exception:
        print(msg)


def _safe_file_of(obj) -> str:
    try:
        return inspect.getfile(obj)
    except Exception:
        return "unknown"


def we_dont_have_this(thing: str, err: Exception | None = None) -> None:
    """
    Loud, consistent signal for optional module missing.
    - Never raises
    - Does not mutate bot/state
    """
    try:
        if err is not None:
            _log.warning(f"OPTIONAL MISSING — {thing} — {type(err).__name__}: {err}")
        else:
            _log.warning(f"OPTIONAL MISSING — {thing}")
    except Exception:
        return


def print_diagnostics(bot=None) -> None:
    """
    Tiny, safe, non-invasive runtime snapshot.
    Never raises. Never mutates state.
    """
    try:
        _out("=" * 60)
        _out("SCALPER ETERNAL — DIAGNOSTICS")
        _out(f"time_utc      : {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())}")
        _out(f"python        : {sys.version.split()[0]}")
        _out(f"platform      : {platform.system()} {platform.release()}")

        if bot is None:
            _out("bot           : <none>")
            _out("=" * 60)
            return

        # Core identity
        _out(f"bot_class     : {bot.__class__.__name__}")
        _out(f"bot_file      : {_safe_file_of(bot.__class__)}")

        # Config
        cfg = getattr(bot, "cfg", None)
        if cfg is not None:
            _out(f"config        : {cfg.__class__.__name__}")
            _out(f"config_ver    : {getattr(cfg, 'CONFIG_VERSION', 'unknown')}")
            _out(f"config_file   : {_safe_file_of(cfg.__class__)}")
        else:
            _out("config        : <missing>")

        # Entry loop mode (env-driven)
        try:
            entry_mode = os.getenv("ENTRY_LOOP_MODE", "").strip().lower() or "auto"
            _out(f"entry_loop    : {entry_mode}")
        except Exception:
            _out("entry_loop    : <unknown>")

        # State
        state = getattr(bot, "state", None)
        if state is not None:
            pos = getattr(state, "positions", {}) or {}
            _out(f"positions     : {len(pos)}")
            _out(f"total_trades  : {getattr(state, 'total_trades', 'n/a')}")
            _out(f"equity        : {getattr(state, 'current_equity', 'n/a')}")
            _out(f"state_file    : {_safe_file_of(state.__class__)}")
        else:
            _out("state         : <missing>")

        # Shutdown flag
        shutdown_ev = getattr(bot, "_shutdown", None)
        if shutdown_ev is not None and hasattr(shutdown_ev, "is_set"):
            _out(f"shutdown_set  : {shutdown_ev.is_set()}")
        else:
            _out("shutdown_set  : <no event>")

        # Exchange
        ex = getattr(bot, "ex", None)
        if ex is not None:
            _out(f"exchange      : {ex.__class__.__name__}")
            _out(f"exchange_file : {_safe_file_of(ex.__class__)}")
        else:
            _out("exchange      : <missing>")

        _out("=" * 60)

    except Exception as e:
        try:
            _log.error(f"DIAGNOSTICS FAILED: {repr(e)}")
        except Exception:
            pass
