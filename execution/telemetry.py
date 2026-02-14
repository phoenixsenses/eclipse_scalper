# execution/telemetry.py — SCALPER ETERNAL — EVENT TELEMETRY — 2026 v1.1 (JSONL + THROTTLED + GUARDIAN-SAFE)
# PURPOSE:
# - Structured event logging (orders, fills, cancels, reconcile actions, kill-switch trips, emergencies, etc.)
# - Writes JSONL to disk (append-only) + optional stdout mirror
# - Provides lightweight counters + last-N ring buffer for quick debugging
#
# Integration patterns:
# - from execution.telemetry import emit, emit_throttled, bump, get_stats
# - await emit(bot, "order.create", {...})
# - await emit_throttled(bot, "price.stale", key=f"{sym}:stale", cooldown_sec=30, data={...})
#
# Notes:
# - Never raises (guardian-safe). Only propagates CancelledError.
# - If file path isn't writable, falls back to in-memory only
# - Designed to work without any other module present

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, List, Tuple
from collections import deque

from utils.logging import log_entry, log_core


# ----------------------------
# Config helpers
# ----------------------------

def _now() -> float:
    return time.time()


def _truthy(x) -> bool:
    if x is True:
        return True
    if isinstance(x, (int, float)) and x != 0:
        return True
    if isinstance(x, str) and x.strip().lower() in ("1", "true", "yes", "y", "on"):
        return True
    return False


def _cfg(bot, name: str, default: Any) -> Any:
    try:
        cfg = getattr(bot, "cfg", None)
        return getattr(cfg, name, default) if cfg is not None else default
    except Exception:
        return default


def _symkey(sym: str) -> str:
    s = (sym or "").upper().strip()
    s = s.replace("/USDT:USDT", "USDT").replace("/USDT", "USDT")
    s = s.replace(":USDT", "USDT").replace(":", "")
    s = s.replace("/", "")
    if s.endswith("USDTUSDT"):
        s = s[:-4]
    return s


def _safe_jsonable(x: Any) -> Any:
    """
    Make best-effort JSON-serializable.
    """
    try:
        if x is None:
            return None
        if isinstance(x, (str, int, float, bool)):
            return x
        if isinstance(x, dict):
            return {str(k): _safe_jsonable(v) for k, v in x.items()}
        if isinstance(x, (list, tuple, set)):
            return [_safe_jsonable(v) for v in list(x)]
        return repr(x)[:2000]
    except Exception:
        return "<?>"

# ----------------------------
# Telemetry state (attached to bot.state)
# ----------------------------

@dataclass
class _TelemetryState:
    last_emit_by_key: Dict[str, float]
    counters: Dict[str, int]
    recent: deque  # deque[dict]
    lock: asyncio.Lock

    # write-failure backoff
    last_write_fail_ts: float = 0.0
    write_fail_streak: int = 0


def _ensure_telemetry_state(bot) -> _TelemetryState:
    st = getattr(bot, "state", None)
    if st is None:
        # create a shim state if missing (rare but safe)
        class _S:
            pass
        st = _S()
        bot.state = st

    t = getattr(st, "telemetry", None)
    if isinstance(t, _TelemetryState):
        return t

    t = _TelemetryState(
        last_emit_by_key={},
        counters={},
        recent=deque(maxlen=int(_cfg(bot, "TELEMETRY_RING_MAX", 250) or 250)),
        lock=asyncio.Lock(),
    )
    try:
        st.telemetry = t
    except Exception:
        pass
    return t


# ----------------------------
# Output path
# ----------------------------

def _default_path() -> str:
    base = os.getenv("SCALPER_LOG_DIR", "") or "logs"
    return os.path.join(base, "telemetry.jsonl")


def _get_path(bot) -> str:
    p = _cfg(bot, "TELEMETRY_PATH", "") or os.getenv("TELEMETRY_PATH", "")
    if not p:
        p = _default_path()
    return str(p)


def _ensure_dir(path: str) -> None:
    try:
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
    except Exception:
        pass


# ----------------------------
# Core emit
# ----------------------------

async def _to_thread(fn):
    """
    asyncio.to_thread exists in 3.9+. This fallback keeps 3.8 alive.
    """
    try:
        to_thread = getattr(asyncio, "to_thread", None)
        if callable(to_thread):
            return await to_thread(fn)
    except Exception:
        pass

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, fn)


async def _append_line(path: str, line: str) -> None:
    """
    Async-ish file append without external deps.
    Uses thread/executor so we don't block the event loop on disk I/O.
    """
    def _write():
        _ensure_dir(path)
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    await _to_thread(_write)


def _base_envelope(bot, event: str) -> Dict[str, Any]:
    env: Dict[str, Any] = {
        "ts": _now(),
        "event": str(event)[:200],
    }

    try:
        env["run_id"] = getattr(getattr(bot, "state", None), "run_id", None)
    except Exception:
        env["run_id"] = None

    try:
        env["exchange"] = type(getattr(bot, "ex", None)).__name__
    except Exception:
        env["exchange"] = None

    return env


async def emit(
    bot,
    event: str,
    data: Optional[Dict[str, Any]] = None,
    *,
    level: str = "info",
    symbol: Optional[str] = None,
    write_file: Optional[bool] = None,
    mirror_stdout: Optional[bool] = None,
) -> None:
    """
    Emit a structured telemetry event.
    Never raises (except CancelledError).
    """
    try:
        # master enable
        if not _truthy(_cfg(bot, "TELEMETRY_ENABLED", True)):
            return

        t = _ensure_telemetry_state(bot)

        if write_file is None:
            write_file = bool(_truthy(_cfg(bot, "TELEMETRY_WRITE_FILE", True)))
        if mirror_stdout is None:
            mirror_stdout = bool(_truthy(_cfg(bot, "TELEMETRY_MIRROR_STDOUT", False)))

        env = _base_envelope(bot, event)

        if symbol:
            env["symbol"] = _symkey(symbol)

        if isinstance(data, dict) and data:
            env["data"] = _safe_jsonable(data)
        else:
            env["data"] = {}

        # ring buffer (in-memory always)
        try:
            t.recent.append(env)
        except Exception:
            pass

        # bump counter
        try:
            t.counters[event] = int(t.counters.get(event, 0) or 0) + 1
        except Exception:
            pass

        # stdout mirror
        try:
            if mirror_stdout:
                log_entry.info(f"TEL {event} {env.get('symbol','')} {env.get('data',{})}")
        except Exception:
            pass

        # file write (locked to avoid interleaving)
        if write_file:
            path = _get_path(bot)

            # ultra-safe dumps: if something slips through, we still don't die
            try:
                line = json.dumps(env, ensure_ascii=False, separators=(",", ":"))
            except Exception:
                line = json.dumps(_safe_jsonable(env), ensure_ascii=False, separators=(",", ":"))

            async with t.lock:
                try:
                    await _append_line(path, line)
                    # reset failure streak on success
                    t.write_fail_streak = 0
                    t.last_write_fail_ts = 0.0
                except Exception as e:
                    # bounded error spam: at most once per ~60s, escalating with streak
                    now = _now()
                    t.write_fail_streak = int(t.write_fail_streak or 0) + 1
                    cool = min(300.0, 30.0 + 10.0 * t.write_fail_streak)  # 30s, 40s, 50s... up to 5m
                    if (now - float(t.last_write_fail_ts or 0.0)) >= cool:
                        t.last_write_fail_ts = now
                        try:
                            log_entry.warning(f"telemetry write failed (streak={t.write_fail_streak}): {type(e).__name__}: {e}")
                        except Exception:
                            pass

        # optional priority log
        try:
            if str(level).lower() == "critical":
                log_core.critical(f"TEL CRITICAL: {event}")
        except Exception:
            pass

    except asyncio.CancelledError:
        raise
    except Exception:
        return


async def emit_throttled(
    bot,
    event: str,
    *,
    key: str,
    cooldown_sec: float,
    data: Optional[Dict[str, Any]] = None,
    level: str = "info",
    symbol: Optional[str] = None,
) -> None:
    """
    Same as emit(), but throttles by (event,key).
    """
    try:
        if not _truthy(_cfg(bot, "TELEMETRY_ENABLED", True)):
            return

        t = _ensure_telemetry_state(bot)
        now = _now()
        cd = max(0.0, float(cooldown_sec or 0.0))
        k = f"{event}|{key}"

        last = float(t.last_emit_by_key.get(k, 0.0) or 0.0)
        if cd > 0 and (now - last) < cd:
            return

        t.last_emit_by_key[k] = now
        await emit(bot, event, data=data, level=level, symbol=symbol)
    except asyncio.CancelledError:
        raise
    except Exception:
        return


def bump(bot, name: str, n: int = 1) -> None:
    """
    Increment a counter without writing an event.
    """
    try:
        if not _truthy(_cfg(bot, "TELEMETRY_ENABLED", True)):
            return
        t = _ensure_telemetry_state(bot)
        t.counters[name] = int(t.counters.get(name, 0) or 0) + int(n or 0)
    except Exception:
        return


def get_stats(bot, top_n: int = 12) -> Dict[str, Any]:
    """
    Returns a snapshot of counters + last events (in-memory).
    """
    try:
        t = _ensure_telemetry_state(bot)
        counters = dict(t.counters)

        items = sorted(counters.items(), key=lambda kv: kv[1], reverse=True)
        top = items[: max(1, int(top_n or 12))]

        recent = list(t.recent)[-50:]
        return {
            "counters_top": top,
            "counters_total_types": len(counters),
            "recent": recent,
            "path": _get_path(bot),
            "write_fail_streak": int(getattr(t, "write_fail_streak", 0) or 0),
        }
    except Exception:
        return {"counters_top": [], "recent": [], "path": _default_path()}


def recent_events(
    bot,
    *,
    event: Optional[str] = None,
    symbol: Optional[str] = None,
    window_sec: Optional[float] = None,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    """
    Return recent events filtered by event/symbol/time window.
    """
    try:
        t = _ensure_telemetry_state(bot)
        now = _now()
        out: List[Dict[str, Any]] = []
        for ev in list(t.recent)[-max(1, int(limit)) * 3:]:
            if event and str(ev.get("event")) != str(event):
                continue
            sym = _symkey(ev.get("symbol") or ev.get("data", {}).get("k") or "")
            if symbol and _symkey(symbol) != sym:
                continue
            if window_sec is not None:
                ts = float(ev.get("ts") or 0.0)
                if ts <= 0 or (now - ts) > float(window_sec):
                    continue
            out.append(ev)
            if len(out) >= int(limit):
                break
        return out
    except Exception:
        return []


def count_recent(
    bot,
    *,
    event: Optional[str] = None,
    symbol: Optional[str] = None,
    window_sec: Optional[float] = None,
) -> int:
    """
    Count recent events filtered by event/symbol/time window.
    """
    try:
        return len(recent_events(bot, event=event, symbol=symbol, window_sec=window_sec, limit=9999))
    except Exception:
        return 0


# ----------------------------
# Convenience helpers (optional)
# ----------------------------

async def emit_order_create(
    bot,
    symbol: str,
    order: Dict[str, Any],
    intent: str = "",
    *,
    correlation_id: Optional[str] = None,
) -> None:
    oid = None
    try:
        oid = order.get("id") or (order.get("info") or {}).get("orderId")
    except Exception:
        oid = None
    await emit(
        bot,
        "order.create",
        data={
            "symbol": _symkey(symbol),
            "order_id": oid,
            "intent": intent,
            "order": order,
            "correlation_id": (str(correlation_id)[:96] if correlation_id else ""),
        },
        symbol=symbol,
    )


async def emit_order_cancel(
    bot,
    symbol: str,
    order_id: str,
    ok: bool,
    why: str = "",
    *,
    correlation_id: Optional[str] = None,
    status: Optional[str] = None,
) -> None:
    await emit(
        bot,
        "order.cancel",
        data={
            "symbol": _symkey(symbol),
            "order_id": str(order_id),
            "ok": bool(ok),
            "why": str(why)[:180],
            "status": (str(status)[:40] if status else ""),
            "correlation_id": (str(correlation_id)[:96] if correlation_id else ""),
        },
        symbol=symbol,
    )


async def emit_fill(bot, symbol: str, fill: Dict[str, Any]) -> None:
    await emit(
        bot,
        "fill",
        data={"symbol": _symkey(symbol), "fill": fill},
        symbol=symbol,
    )


async def emit_reconcile(bot, symbol: str, action: str, details: Dict[str, Any]) -> None:
    await emit(
        bot,
        "reconcile",
        data={"symbol": _symkey(symbol), "action": str(action), "details": details},
        symbol=symbol,
    )


async def emit_kill_trip(bot, reason: str, seconds: float) -> None:
    await emit(
        bot,
        "kill_switch.trip",
        data={"reason": str(reason)[:240], "halt_sec": float(seconds)},
        level="critical",
    )


async def emit_emergency(bot, action: str, details: Dict[str, Any]) -> None:
    await emit(
        bot,
        "emergency",
        data={"action": str(action), "details": details},
        level="critical",
    )
