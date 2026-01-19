# execution/guardian.py — SCALPER ETERNAL — BRAINSTEM OF REALITY — 2026 v1.7 (EXIT WATCHER WIRED)
# Patch vs v1.6:
# - ✅ NEW: Exit watcher inside guardian (best-effort, never fatal)
# - ✅ Polls fetch_my_trades / fetch_closed_orders / fetch_orders (whichever exists)
# - ✅ Dedup via state.known_exit_order_ids + state.known_exit_trade_ids
# - ✅ Calls execution.exit.handle_exit(bot, order) for reduce-only / closePosition intent
# - ✅ Cursor ("since") tracking to avoid re-processing old fills
# - ✅ Fully optional and config-driven; defaults to ON if endpoints exist

from __future__ import annotations

import asyncio
import time
import random
from typing import Any, Awaitable, Callable, Optional, Dict, List

from utils.logging import log_core, log_entry

# Diagnostics helper (best-effort; never fatal)
try:
    from execution.diagnostics import we_dont_have_this  # type: ignore
except Exception:
    def we_dont_have_this(thing: str, err: Exception | None = None) -> None:  # type: ignore
        try:
            if err is not None:
                log_entry.warning(f"OPTIONAL MISSING — {thing} — {type(err).__name__}: {err}")
            else:
                log_entry.warning(f"OPTIONAL MISSING — {thing}")
        except Exception:
            pass


# Kill-switch (state + optional tick)
try:
    from risk.kill_switch import is_halted  # type: ignore
except Exception as e:
    is_halted = None
    we_dont_have_this("risk.kill_switch.is_halted", e)

try:
    from risk.kill_switch import tick_kill_switch  # type: ignore
except Exception as e:
    tick_kill_switch = None
    we_dont_have_this("risk.kill_switch.tick_kill_switch", e)

# Optional entry watch
try:
    from execution.entry_watch import poll_entry_watches  # type: ignore
except Exception as e:
    poll_entry_watches = None
    we_dont_have_this("execution.entry_watch.poll_entry_watches", e)

# Reconcile module (preferred: reconcile_tick; legacy: guardian_loop)
try:
    import execution.reconcile as reconcile_mod  # type: ignore
except Exception as e:
    reconcile_mod = None
    we_dont_have_this("execution.reconcile (module import)", e)

# Optional emergency module
try:
    import execution.emergency as emergency_mod  # type: ignore
except Exception as e:
    emergency_mod = None
    we_dont_have_this("execution.emergency (module import)", e)

# Optional position manager (NEW)
try:
    from execution.position_manager import position_manager_tick  # type: ignore
except Exception as e:
    position_manager_tick = None
    we_dont_have_this("execution.position_manager.position_manager_tick", e)

# ✅ Exit handler (wired)
try:
    from execution.exit import handle_exit  # type: ignore
except Exception as e:
    handle_exit = None
    we_dont_have_this("execution.exit.handle_exit", e)


def _cfg(bot, name: str, default):
    try:
        return getattr(bot.cfg, name, default)
    except Exception:
        return default


def _now() -> float:
    return time.time()


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


def _ensure_shutdown_fields(bot) -> None:
    st = getattr(bot, "state", None)
    if st is None:
        return
    if not hasattr(st, "shutdown_reason"):
        st.shutdown_reason = ""
    if not hasattr(st, "shutdown_source"):
        st.shutdown_source = ""
    if not hasattr(st, "shutdown_ts"):
        st.shutdown_ts = 0.0


def _record_shutdown_reason(bot, reason: str, source: str) -> None:
    try:
        _ensure_shutdown_fields(bot)
        st = bot.state
        if not getattr(st, "shutdown_reason", ""):
            st.shutdown_reason = str(reason or "")[:500]
            st.shutdown_source = str(source or "")[:120]
            st.shutdown_ts = _now()
    except Exception:
        pass


async def _safe_call(name: str, fn: Callable[..., Awaitable[Any]], *args, **kwargs):
    if not callable(fn):
        return
    try:
        await fn(*args, **kwargs)
    except asyncio.CancelledError:
        raise
    except Exception as e:
        log_entry.error(f"GUARDIAN: {name} failed: {e}")


async def _cancel_task_bounded(task: asyncio.Task, *, wait_sec: float = 0.35) -> None:
    if task.done():
        return
    task.cancel()
    try:
        await asyncio.wait_for(task, timeout=max(0.05, float(wait_sec)))
    except asyncio.TimeoutError:
        pass
    except asyncio.CancelledError:
        raise
    except Exception:
        pass


async def _run_legacy_loop_briefly(
    name: str,
    loop_fn: Callable[..., Awaitable[Any]],
    bot,
    brief_sec: float,
):
    if not callable(loop_fn):
        return

    t = asyncio.create_task(loop_fn(bot), name=f"legacy_{name}")
    try:
        await asyncio.wait_for(t, timeout=max(0.05, float(brief_sec)))
    except asyncio.TimeoutError:
        pass
    except asyncio.CancelledError:
        await _cancel_task_bounded(t)
        raise
    except Exception as e:
        log_entry.error(f"GUARDIAN: legacy loop {name} crashed: {e}")
    finally:
        await _cancel_task_bounded(t)


async def _reconcile_tick(bot, brief_sec: float):
    if reconcile_mod is None:
        return

    tick = getattr(reconcile_mod, "reconcile_tick", None)
    if callable(tick):
        await _safe_call("reconcile_tick", tick, bot)
        return

    legacy = getattr(reconcile_mod, "guardian_loop", None)
    if callable(legacy):
        await _run_legacy_loop_briefly("reconcile_guardian_loop", legacy, bot, brief_sec)


async def _emergency_tick(bot, brief_sec: float):
    if emergency_mod is None:
        return

    tick = getattr(emergency_mod, "emergency_tick", None) or getattr(emergency_mod, "run_emergency", None)
    if callable(tick):
        await _safe_call("emergency_tick", tick, bot)
        return

    legacy = getattr(emergency_mod, "emergency_loop", None)
    if callable(legacy):
        await _run_legacy_loop_briefly("emergency_loop", legacy, bot, brief_sec)


async def _position_manager_tick(bot, brief_sec: float):
    if not callable(position_manager_tick):
        return
    await _safe_call("position_manager.position_manager_tick", position_manager_tick, bot)


# ----------------------------
# Exit watcher helpers
# ----------------------------

def _truthy(x) -> bool:
    if x is True:
        return True
    if isinstance(x, (int, float)) and x != 0:
        return True
    if isinstance(x, str) and x.strip().lower() in ("true", "1", "yes", "y", "t"):
        return True
    return False


def _is_reduce_only_like(obj: dict) -> bool:
    """
    Detect reduce-only intent across order/trade shapes.
    """
    try:
        if _truthy(obj.get("reduceOnly")):
            return True
        info = obj.get("info") or {}
        if isinstance(info, dict):
            if _truthy(info.get("reduceOnly")):
                return True
            if _truthy(info.get("closePosition")):
                return True

        params = obj.get("params") or {}
        if isinstance(params, dict):
            if _truthy(params.get("reduceOnly")):
                return True
            if _truthy(params.get("closePosition")):
                return True

        # Some trade feeds only say "side" and "amount" — we don't treat those as reduce-only.
    except Exception:
        pass
    return False


def _ensure_known_exit_sets(state) -> None:
    try:
        s = getattr(state, "known_exit_order_ids", None)
        if not isinstance(s, set):
            state.known_exit_order_ids = set()
    except Exception:
        try:
            state.known_exit_order_ids = set()
        except Exception:
            pass

    try:
        t = getattr(state, "known_exit_trade_ids", None)
        if not isinstance(t, set):
            state.known_exit_trade_ids = set()
    except Exception:
        try:
            state.known_exit_trade_ids = set()
        except Exception:
            pass


def _rc_map(state) -> dict:
    try:
        rc = getattr(state, "run_context", None)
        if not isinstance(rc, dict):
            state.run_context = {}
            rc = state.run_context
        return rc
    except Exception:
        return {}


def _get_exit_since_ms(state) -> int:
    """
    Cursor for trade/order polling.
    Stored in state.run_context["exit_since_ms"].
    """
    rc = _rc_map(state)
    v = rc.get("exit_since_ms")
    try:
        vv = int(v)
        if vv > 0:
            return vv
    except Exception:
        pass
    # default: last ~2 minutes to avoid missing anything right after boot
    return int(time.time() * 1000) - 2 * 60 * 1000


def _set_exit_since_ms(state, since_ms: int) -> None:
    rc = _rc_map(state)
    try:
        rc["exit_since_ms"] = int(since_ms)
    except Exception:
        pass


def _ms_from_ts_any(x) -> int:
    """
    Accept seconds, ms, ISO-ish unsupported => 0.
    """
    if x is None:
        return 0
    try:
        if isinstance(x, (int, float)):
            # heuristic: if it's too large it's already ms
            if float(x) > 10_000_000_000:  # > ~2286-11-20 in seconds
                return int(x)
            return int(float(x) * 1000)
    except Exception:
        pass
    return 0


def _normalize_trade_to_order_like(trade: dict) -> dict:
    """
    Convert a ccxt trade into an "order-like" dict that exit.handle_exit can digest.
    Key goal: provide:
      - id
      - symbol
      - filled
      - info.executedQty
      - info.realizedPnl if present
      - reduceOnly / closePosition if present (usually not)
    """
    out = dict(trade or {})
    try:
        # ensure id exists
        if not out.get("id"):
            oid = out.get("order") or out.get("orderId") or out.get("tradeId")
            if oid:
                out["id"] = str(oid)

        # approximate filled
        amt = out.get("amount")
        if amt is not None and out.get("filled") is None:
            out["filled"] = amt

        info = out.get("info") or {}
        if not isinstance(info, dict):
            info = {"raw": info}
        # set executedQty if missing
        if "executedQty" not in info and out.get("filled") is not None:
            info["executedQty"] = out.get("filled")
        out["info"] = info
    except Exception:
        pass
    return out


async def _exit_watch_tick(bot) -> None:
    """
    Best-effort poller that tries multiple endpoints and feeds reduce-only intent into handle_exit().
    Never raises.
    """
    if not callable(handle_exit):
        return

    enabled = bool(_cfg(bot, "EXIT_WATCH_ENABLED", True))
    if not enabled:
        return

    ex = getattr(bot, "ex", None)
    if ex is None:
        return

    st = getattr(bot, "state", None)
    if st is None:
        return

    _ensure_known_exit_sets(st)

    # Config knobs
    limit = int(_cfg(bot, "EXIT_WATCH_LIMIT", 50))
    if limit <= 0:
        limit = 50

    # cursor in ms
    since_ms = _get_exit_since_ms(st)
    max_seen_ms = since_ms

    # Symbol scope (optional): if you have active symbols, prefer filtering
    symbols_any: List[str] = []
    try:
        syms = getattr(bot, "active_symbols", None) or getattr(getattr(bot, "cfg", None), "ACTIVE_SYMBOLS", None)
        if isinstance(syms, (set, list, tuple)):
            symbols_any = [str(x) for x in list(syms)]
    except Exception:
        symbols_any = []

    # Helper: process an order-like object
    async def _process_order_like(obj: dict) -> None:
        nonlocal max_seen_ms

        if not isinstance(obj, dict):
            return

        oid = obj.get("id") or (obj.get("info") or {}).get("orderId") or obj.get("order")
        if oid:
            oid = str(oid)

        # dedupe by order id (primary)
        if oid and oid in st.known_exit_order_ids:
            return

        # Reduce-only intent filter
        if not _is_reduce_only_like(obj):
            return

        # Update cursor timestamp if we can
        ts_ms = 0
        try:
            ts_ms = _ms_from_ts_any(obj.get("timestamp"))
            if ts_ms <= 0:
                ts_ms = _ms_from_ts_any((obj.get("info") or {}).get("updateTime"))
            if ts_ms <= 0:
                ts_ms = _ms_from_ts_any((obj.get("info") or {}).get("time"))
        except Exception:
            ts_ms = 0

        if ts_ms > max_seen_ms:
            max_seen_ms = ts_ms

        # Mark seen early (avoid re-entrancy duplicates)
        if oid:
            st.known_exit_order_ids.add(oid)

        # Hand off to your exit engine
        try:
            await handle_exit(bot, obj)
        except Exception as e:
            log_entry.error(f"EXIT_WATCH: handle_exit failed: {e}")

    # Helper: process trades (as fallback)
    async def _process_trade(tr: dict) -> None:
        nonlocal max_seen_ms
        if not isinstance(tr, dict):
            return

        tid = tr.get("id") or tr.get("tradeId")
        if tid:
            tid = str(tid)
            if tid in st.known_exit_trade_ids:
                return

        ts_ms = _ms_from_ts_any(tr.get("timestamp"))
        if ts_ms > max_seen_ms:
            max_seen_ms = ts_ms

        # Convert to order-like and attempt reduce-only detection
        order_like = _normalize_trade_to_order_like(tr)

        # If trade has an order id, prefer that for dedupe too
        oid = order_like.get("id")
        if oid:
            oid = str(oid)
            if oid in st.known_exit_order_ids:
                return

        # Trades usually don’t indicate reduceOnly, so we only forward if we can infer:
        # - if it references an order that is reduceOnly in info (rare)
        # - or if exchange provides closePosition/reduceOnly in trade.info (sometimes)
        if not _is_reduce_only_like(order_like):
            # no signal => ignore
            return

        # mark seen
        if tid:
            st.known_exit_trade_ids.add(tid)
        if oid:
            st.known_exit_order_ids.add(oid)

        try:
            await handle_exit(bot, order_like)
        except Exception as e:
            log_entry.error(f"EXIT_WATCH: handle_exit(trade->order_like) failed: {e}")

    # 1) fetch_closed_orders (best if available)
    try:
        fn = getattr(ex, "fetch_closed_orders", None)
        if callable(fn):
            if symbols_any:
                # poll per symbol to avoid huge responses
                for sym in symbols_any[:12]:
                    try:
                        arr = await fn(sym, since_ms, limit)
                        if isinstance(arr, list):
                            for o in arr:
                                await _process_order_like(o)
                    except Exception:
                        continue
            else:
                arr = await fn(None, since_ms, limit)
                if isinstance(arr, list):
                    for o in arr:
                        await _process_order_like(o)
    except Exception:
        pass

    # 2) fetch_orders (sometimes easier than closed_orders)
    try:
        fn = getattr(ex, "fetch_orders", None)
        if callable(fn):
            if symbols_any:
                for sym in symbols_any[:12]:
                    try:
                        arr = await fn(sym, since_ms, limit)
                        if isinstance(arr, list):
                            for o in arr:
                                await _process_order_like(o)
                    except Exception:
                        continue
            else:
                arr = await fn(None, since_ms, limit)
                if isinstance(arr, list):
                    for o in arr:
                        await _process_order_like(o)
    except Exception:
        pass

    # 3) fetch_my_trades (fallback)
    try:
        fn = getattr(ex, "fetch_my_trades", None)
        if callable(fn):
            if symbols_any:
                for sym in symbols_any[:12]:
                    try:
                        arr = await fn(sym, since_ms, limit)
                        if isinstance(arr, list):
                            for tr in arr:
                                await _process_trade(tr)
                    except Exception:
                        continue
            else:
                arr = await fn(None, since_ms, limit)
                if isinstance(arr, list):
                    for tr in arr:
                        await _process_trade(tr)
    except Exception:
        pass

    # Advance cursor slightly (avoid re-reading boundary items forever)
    if max_seen_ms > since_ms:
        _set_exit_since_ms(st, int(max_seen_ms) + 1)


# ----------------------------
# Skip ledger snapshot helpers (unchanged)
# ----------------------------

def _get_skip_ledger(bot) -> Optional[dict]:
    try:
        st = getattr(bot, "state", None)
        rc = getattr(st, "run_context", None) if st is not None else None
        if isinstance(rc, dict):
            led = rc.get("skip_ledger")
            if isinstance(led, dict):
                return led
    except Exception:
        pass
    return None


def _format_skip_ledger_snapshot(ledger: dict, max_items: int = 8) -> str:
    items = []
    try:
        for k, v in ledger.items():
            if isinstance(v, list) and v:
                last = v[-1]
            else:
                last = v
            if isinstance(last, dict):
                reason = str(last.get("reason") or last.get("r") or "unknown")[:120]
                side = str(last.get("side") or "")[:6]
                ts = last.get("ts") or last.get("t") or None
                age = ""
                try:
                    if ts:
                        age_sec = max(0.0, _now() - float(ts))
                        age = f" age={age_sec:.0f}s"
                except Exception:
                    age = ""
                items.append(f"{k}:{side} {reason}{age}".strip())
            else:
                items.append(f"{k}: {str(last)[:140]}")
    except Exception:
        return "SKIPS: (unreadable ledger)"

    if not items:
        return "SKIPS: none"

    items = items[: max(1, int(max_items))]
    return "SKIPS: " + " | ".join(items)


async def guardian_loop(bot):
    """
    The brainstem loop: short, frequent, resilient, monotonic cadence.

    Guardian does NOT shut the bot down by default.
    """
    poll_sec = float(_cfg(bot, "GUARDIAN_POLL_SEC", 5.0))
    respect_kill = bool(_cfg(bot, "GUARDIAN_RESPECT_KILL_SWITCH", True))

    cycle_timeout = float(_cfg(bot, "GUARDIAN_CYCLE_TIMEOUT_SEC", 0.0))  # 0 disables
    shutdown_on_timeout = bool(_cfg(bot, "GUARDIAN_SHUTDOWN_ON_TIMEOUT", False))

    legacy_brief_sec = float(_cfg(bot, "GUARDIAN_LEGACY_BRIEF_SEC", 0.25))
    jitter = float(_cfg(bot, "GUARDIAN_JITTER_SEC", 0.10))

    log_cycle_failures = bool(_cfg(bot, "GUARDIAN_LOG_CYCLE_FAILURES", True))
    max_timeout_streak = int(_cfg(bot, "GUARDIAN_MAX_TIMEOUT_STREAK", 6))
    max_fail_streak = int(_cfg(bot, "GUARDIAN_MAX_FAIL_STREAK", 12))

    skip_snap_enabled = bool(_cfg(bot, "GUARDIAN_SKIP_SNAPSHOT_ENABLED", False))
    skip_snap_cooldown = float(_cfg(bot, "GUARDIAN_SKIP_SNAPSHOT_COOLDOWN_SEC", 30.0))
    skip_snap_max_items = int(_cfg(bot, "GUARDIAN_SKIP_SNAPSHOT_MAX_ITEMS", 8))
    _last_skip_snap = 0.0

    # ✅ Exit watcher cadence (separate from guardian poll)
    exit_watch_enabled = bool(_cfg(bot, "EXIT_WATCH_ENABLED", True))
    exit_watch_every = float(_cfg(bot, "EXIT_WATCH_EVERY_SEC", 1.25))
    if exit_watch_every <= 0:
        exit_watch_every = 1.25
    _last_exit_watch = 0.0

    shutdown_ev = _ensure_shutdown_event(bot)
    _ensure_shutdown_fields(bot)

    log_core.critical("GUARDIAN ONLINE — brainstem loop engaged")

    next_tick = time.monotonic()
    timeout_streak = 0
    fail_streak = 0

    async def _one_cycle():
        nonlocal _last_exit_watch, _last_skip_snap

        # 0) Kill-switch tick (EVALUATE)
        if respect_kill and callable(tick_kill_switch):
            await _safe_call("kill_switch.tick_kill_switch", tick_kill_switch, bot)

        # 0b) Kill-switch state snapshot (for optional hooks only)
        halted = False
        if respect_kill and callable(is_halted):
            try:
                halted = bool(is_halted(bot))
            except Exception:
                halted = False

        # ✅ Exit watch tick (high-frequency, lightweight)
        if exit_watch_enabled and callable(handle_exit):
            now_ts = _now()
            if (now_ts - _last_exit_watch) >= exit_watch_every:
                _last_exit_watch = now_ts
                await _safe_call("exit_watch_tick", _exit_watch_tick, bot)

        # 1) Entry watch: ALWAYS run (safety housekeeping)
        if callable(poll_entry_watches):
            await _safe_call("entry_watch.poll_entry_watches", poll_entry_watches, bot)

        # Optional: entry_watch halt hook
        if halted:
            try:
                import execution.entry_watch as entry_watch_mod  # type: ignore
                ew_hook = getattr(entry_watch_mod, "on_halt", None)
                if callable(ew_hook):
                    await _safe_call("entry_watch.on_halt", ew_hook, bot)
            except Exception as e:
                we_dont_have_this("execution.entry_watch.on_halt (runtime)", e)

        # 2) Reconcile truth: ALWAYS run
        await _reconcile_tick(bot, legacy_brief_sec)

        # Optional: reconcile halt hook
        if halted and reconcile_mod is not None:
            try:
                rh = getattr(reconcile_mod, "on_halt", None)
                if callable(rh):
                    await _safe_call("reconcile.on_halt", rh, bot)
            except Exception as e:
                we_dont_have_this("execution.reconcile.on_halt (runtime)", e)

        # 2b) Position manager tick: ALWAYS run
        await _position_manager_tick(bot, legacy_brief_sec)

        # Optional: position manager halt hook
        if halted:
            try:
                import execution.position_manager as pm_mod  # type: ignore
                pm_hook = getattr(pm_mod, "on_halt", None)
                if callable(pm_hook):
                    await _safe_call("position_manager.on_halt", pm_hook, bot)
            except Exception as e:
                we_dont_have_this("execution.position_manager.on_halt (runtime)", e)

        # 3) Emergency checks: ALWAYS run
        await _emergency_tick(bot, legacy_brief_sec)

        # Optional: emergency halt hook
        if halted and emergency_mod is not None:
            try:
                eh = getattr(emergency_mod, "on_halt", None)
                if callable(eh):
                    await _safe_call("emergency.on_halt", eh, bot)
            except Exception as e:
                we_dont_have_this("execution.emergency.on_halt (runtime)", e)

        # 4) Optional: skip ledger snapshot (observability only)
        if skip_snap_enabled:
            now_ts = _now()
            if (now_ts - _last_skip_snap) >= max(5.0, skip_snap_cooldown):
                led = _get_skip_ledger(bot)
                if isinstance(led, dict) and len(led) > 0:
                    msg = _format_skip_ledger_snapshot(led, max_items=skip_snap_max_items)
                    log_entry.info(f"GUARDIAN SNAPSHOT — {msg}")
                _last_skip_snap = now_ts

    async def _run_cycle_with_timeout() -> None:
        if not cycle_timeout or cycle_timeout <= 0:
            await _one_cycle()
            return

        t = asyncio.create_task(_one_cycle(), name="guardian_one_cycle")
        try:
            await asyncio.wait_for(t, timeout=cycle_timeout)
        except asyncio.TimeoutError:
            await _cancel_task_bounded(t, wait_sec=0.25)
            raise
        except asyncio.CancelledError:
            await _cancel_task_bounded(t, wait_sec=0.25)
            raise
        except Exception:
            await _cancel_task_bounded(t, wait_sec=0.25)
            raise

    while not shutdown_ev.is_set():
        now_m = time.monotonic()
        if now_m < next_tick:
            try:
                await asyncio.sleep(next_tick - now_m)
            except asyncio.CancelledError:
                raise

        next_tick = max(next_tick + poll_sec, time.monotonic() + 0.001)

        if jitter and jitter > 0:
            try:
                j = random.uniform(0.0, min(float(jitter), 0.25))
                if j > 0:
                    await asyncio.sleep(j)
            except asyncio.CancelledError:
                raise

        try:
            await _run_cycle_with_timeout()
            timeout_streak = 0
            fail_streak = 0

        except asyncio.TimeoutError:
            timeout_streak += 1
            fail_streak += 1
            _record_shutdown_reason(
                bot,
                reason=f"GUARDIAN CYCLE TIMEOUT ({timeout_streak} streak) timeout={cycle_timeout:.2f}s",
                source="execution.guardian",
            )
            if log_cycle_failures:
                log_entry.error(f"GUARDIAN: cycle timeout streak={timeout_streak} timeout={cycle_timeout:.2f}s")

            if shutdown_on_timeout and timeout_streak >= max_timeout_streak:
                log_core.critical("GUARDIAN: timeout streak exceeded — setting shutdown (opt-in)")
                try:
                    shutdown_ev.set()
                except Exception:
                    pass

        except asyncio.CancelledError:
            raise

        except Exception as e:
            fail_streak += 1
            _record_shutdown_reason(bot, reason=f"GUARDIAN CYCLE ERROR: {e}", source="execution.guardian")
            if log_cycle_failures:
                log_entry.error(f"GUARDIAN: cycle failed streak={fail_streak}: {e}")

            if shutdown_on_timeout and fail_streak >= max_fail_streak:
                log_core.critical("GUARDIAN: fail streak exceeded — setting shutdown (opt-in)")
                try:
                    shutdown_ev.set()
                except Exception:
                    pass

    # Shutdown detected: print forensic info
    try:
        _ensure_shutdown_fields(bot)
        st = bot.state
        rs = str(getattr(st, "shutdown_reason", "") or "")
        src = str(getattr(st, "shutdown_source", "") or "")
        ts = float(getattr(st, "shutdown_ts", 0.0) or 0.0)
        if rs or src:
            log_core.critical(
                f"GUARDIAN OFFLINE — shutdown detected | source={src or 'unknown'} | reason={rs or 'unknown'} | ts={ts:.0f}"
            )
        else:
            log_core.critical("GUARDIAN OFFLINE — shutdown flag set (no reason recorded)")
    except Exception:
        log_core.critical("GUARDIAN OFFLINE — shutdown flag set")
