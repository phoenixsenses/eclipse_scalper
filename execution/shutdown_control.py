from __future__ import annotations

import asyncio
import json
import os
import threading
import time
import traceback
from pathlib import Path
from typing import Any


def _truthy(x: Any) -> bool:
    if x is True:
        return True
    if isinstance(x, (int, float)) and x != 0:
        return True
    if isinstance(x, str) and x.strip().lower() in ("1", "true", "yes", "on", "y"):
        return True
    return False


class TracedShutdownEvent(asyncio.Event):
    def __init__(self, bot):
        super().__init__()
        self._bot = bot

    def set(self) -> None:
        b = self._bot
        try:
            if not bool(getattr(b, "_shutdown_set_recorded", False)):
                setattr(b, "_shutdown_set_recorded", True)
                setattr(b, "_shutdown_set_ts", float(time.time()))
                setattr(b, "_shutdown_set_trace", "".join(traceback.format_stack(limit=50)))
                setattr(b, "_shutdown_set_thread", int(threading.get_ident()))
        except Exception:
            pass
        super().set()


def ensure_traced_shutdown_event(bot) -> asyncio.Event:
    ev = getattr(bot, "_shutdown", None)
    if isinstance(ev, asyncio.Event):
        return ev
    ev = TracedShutdownEvent(bot)
    try:
        bot._shutdown = ev  # type: ignore[attr-defined]
    except Exception:
        pass
    return ev


def _ensure_shutdown_event(bot) -> asyncio.Event:
    # Backward-compatible alias for existing callsites.
    return ensure_traced_shutdown_event(bot)


def _safe_log(bot, level: str, msg: str) -> None:
    try:
        from utils.logging import log_core  # type: ignore
        fn = getattr(log_core, level, None)
        if callable(fn):
            fn(msg)
            return
    except Exception:
        pass
    try:
        lg = getattr(bot, "log", None)
        fn = getattr(lg, level, None)
        if callable(fn):
            fn(msg)
    except Exception:
        pass


def _write_last_shutdown(meta: dict[str, Any]) -> None:
    try:
        from execution.runtime_helpers import atomic_write_json
        atomic_write_json(Path("logs/last_shutdown.json"), meta)
    except Exception:
        pass


def should_continue_nonfatal(bot) -> bool:
    try:
        dry_run = _truthy(os.getenv("SCALPER_DRY_RUN", "0"))
        allow = _truthy(os.getenv("PAPER_ALLOW_NONFATAL_SHUTDOWN", "0"))
        return bool(dry_run and allow)
    except Exception:
        return False


def request_shutdown(bot, reason: str, *, source: str = "", fatal: bool = False, exc: Exception | None = None) -> bool:
    """
    Returns True if shutdown flag/event was set, False if swallowed as non-fatal.
    """
    now = time.time()
    trace = ""
    first = not bool(getattr(bot, "_shutdown_requested", False))
    if first:
        try:
            trace = "".join(traceback.format_stack(limit=25))
            bot._shutdown_requested = True  # type: ignore[attr-defined]
            bot._shutdown_reason = str(reason or "")[:500]  # type: ignore[attr-defined]
            bot._shutdown_source = str(source or "")[:120]  # type: ignore[attr-defined]
            bot._shutdown_fatal = bool(fatal)  # type: ignore[attr-defined]
            bot._shutdown_ts = float(now)  # type: ignore[attr-defined]
            bot._shutdown_trace = trace  # type: ignore[attr-defined]
        except Exception:
            pass
        try:
            st = getattr(bot, "state", None)
            if st is not None:
                if not hasattr(st, "shutdown_reason") or not getattr(st, "shutdown_reason", ""):
                    st.shutdown_reason = str(reason or "")[:500]
                if not hasattr(st, "shutdown_source") or not getattr(st, "shutdown_source", ""):
                    st.shutdown_source = str(source or "")[:120]
                if not hasattr(st, "shutdown_ts") or not float(getattr(st, "shutdown_ts", 0.0) or 0.0):
                    st.shutdown_ts = float(now)
        except Exception:
            pass
        _safe_log(
            bot,
            "critical",
            f"[SHUTDOWN_REQUEST] fatal={int(bool(fatal))} source={source or 'unknown'} "
            f"reason={str(reason or '')[:300]} exc={type(exc).__name__ + ': ' + str(exc) if exc else 'None'}",
        )
        if trace:
            _safe_log(bot, "critical", f"[SHUTDOWN_REQUEST] stack:\n{trace}")
        _write_last_shutdown(
            {
                "ts": now,
                "source": str(source or ""),
                "reason": str(reason or ""),
                "fatal": bool(fatal),
                "exc": (f"{type(exc).__name__}: {exc}" if exc else ""),
                "trace": trace,
            }
        )

    if (not fatal) and should_continue_nonfatal(bot):
        _safe_log(
            bot,
            "warning",
            f"[SHUTDOWN_REQUEST] non-fatal ignored in paper mode "
            f"(PAPER_ALLOW_NONFATAL_SHUTDOWN=1): source={source or 'unknown'} reason={str(reason or '')[:240]}",
        )
        return False

    ev = _ensure_shutdown_event(bot)
    try:
        ev.set()
    except Exception:
        return False
    return True
