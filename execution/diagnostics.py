# execution/diagnostics.py — SCALPER ETERNAL — DIAGNOSTIC PULSE — v0.3

import time
import platform
import sys
import inspect


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
    - Prefers logger if available, falls back to print
    """
    try:
        # Import inside the function to avoid import-order / circular issues
        try:
            from utils.logging import log_core  # type: ignore
            if err is not None:
                log_core.warning(f"OPTIONAL MISSING — {thing} — {type(err).__name__}: {err}")
            else:
                log_core.warning(f"OPTIONAL MISSING — {thing}")
            return
        except Exception:
            pass

        if err is not None:
            print(f"OPTIONAL MISSING — {thing} — {type(err).__name__}: {err}")
        else:
            print(f"OPTIONAL MISSING — {thing}")

    except Exception:
        # absolute last resort: silence is better than crashing
        return


def print_diagnostics(bot=None) -> None:
    """
    Tiny, safe, non-invasive runtime snapshot.
    Never raises. Never mutates state.
    """
    try:
        print("=" * 60)
        print("SCALPER ETERNAL — DIAGNOSTICS")
        print(f"time_utc      : {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())}")
        print(f"python        : {sys.version.split()[0]}")
        print(f"platform      : {platform.system()} {platform.release()}")

        if bot is None:
            print("bot           : <none>")
            print("=" * 60)
            return

        # Core identity
        print(f"bot_class     : {bot.__class__.__name__}")
        print(f"bot_file      : {_safe_file_of(bot.__class__)}")

        # Config
        cfg = getattr(bot, "cfg", None)
        if cfg is not None:
            print(f"config        : {cfg.__class__.__name__}")
            print(f"config_ver    : {getattr(cfg, 'CONFIG_VERSION', 'unknown')}")
            print(f"config_file   : {_safe_file_of(cfg.__class__)}")
        else:
            print("config        : <missing>")

        # Entry loop mode (env-driven)
        try:
            import os
            entry_mode = os.getenv("ENTRY_LOOP_MODE", "").strip().lower() or "auto"
            print(f"entry_loop    : {entry_mode}")
        except Exception:
            print("entry_loop    : <unknown>")

        # State
        state = getattr(bot, "state", None)
        if state is not None:
            pos = getattr(state, "positions", {}) or {}
            print(f"positions     : {len(pos)}")
            print(f"total_trades  : {getattr(state, 'total_trades', 'n/a')}")
            print(f"equity        : {getattr(state, 'current_equity', 'n/a')}")
            print(f"state_file    : {_safe_file_of(state.__class__)}")
        else:
            print("state         : <missing>")

        # Shutdown flag
        shutdown_ev = getattr(bot, "_shutdown", None)
        if shutdown_ev is not None and hasattr(shutdown_ev, "is_set"):
            print(f"shutdown_set  : {shutdown_ev.is_set()}")
        else:
            print("shutdown_set  : <no event>")

        # Exchange
        ex = getattr(bot, "ex", None)
        if ex is not None:
            print(f"exchange      : {ex.__class__.__name__}")
            print(f"exchange_file : {_safe_file_of(ex.__class__)}")
        else:
            print("exchange      : <missing>")

        print("=" * 60)

    except Exception as e:
        print("DIAGNOSTICS FAILED:", repr(e))
