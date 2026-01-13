# execution/guardian.py — SCALPER ETERNAL — BRAINSTEM OF REALITY — 2026 v1.6 (POSITION MANAGER TICK WIRED)
# Patch vs v1.5:
# - ✅ Optional-imports execution.position_manager.position_manager_tick with diagnostics.we_dont_have_this()
# - ✅ Runs position_manager_tick after reconcile (truth first), before emergency (so emergency sees latest protection)
# - ✅ Optional on_halt hook for position_manager

import asyncio
import time
import random
from typing import Any, Awaitable, Callable, Optional

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
    """
    Store shutdown forensic info in bot.state so every module can report consistently.
    """
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
    """
    This does NOT set shutdown. It only records why, so when shutdown is set elsewhere
    you can see the cause.
    """
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
    """
    Cancel a task and await it, but never hang forever.
    """
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
    """
    For modules that only expose an infinite loop (legacy).
    Run it briefly with a timeout; then cancel it cleanly.
    """
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
    """
    Tick-based position management (breakeven/stop sanity/trailing rebuild).
    """
    if not callable(position_manager_tick):
        return
    await _safe_call("position_manager.position_manager_tick", position_manager_tick, bot)


def _get_skip_ledger(bot) -> Optional[dict]:
    """
    Optional: entry.py may record skip reasons into bot.state.run_context["skip_ledger"].
    We keep this best-effort and totally non-fatal.
    """
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
    """
    Expect ledger to be {symbol: {...}} or {symbol: [..]} depending on your implementation.
    We try to print something useful regardless.
    """
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

    IMPORTANT:
    - Guardian does NOT shut the bot down by default.
    - It only records reasons and logs them.
    - If you want guardian to force shutdown on repeated timeouts, enable:
        GUARDIAN_SHUTDOWN_ON_TIMEOUT = True
    """
    poll_sec = float(_cfg(bot, "GUARDIAN_POLL_SEC", 5.0))
    respect_kill = bool(_cfg(bot, "GUARDIAN_RESPECT_KILL_SWITCH", True))

    cycle_timeout = float(_cfg(bot, "GUARDIAN_CYCLE_TIMEOUT_SEC", 0.0))  # 0 disables
    shutdown_on_timeout = bool(_cfg(bot, "GUARDIAN_SHUTDOWN_ON_TIMEOUT", False))

    legacy_brief_sec = float(_cfg(bot, "GUARDIAN_LEGACY_BRIEF_SEC", 0.25))
    jitter = float(_cfg(bot, "GUARDIAN_JITTER_SEC", 0.10))

    # Debug telemetry (optional)
    log_cycle_failures = bool(_cfg(bot, "GUARDIAN_LOG_CYCLE_FAILURES", True))
    max_timeout_streak = int(_cfg(bot, "GUARDIAN_MAX_TIMEOUT_STREAK", 6))
    max_fail_streak = int(_cfg(bot, "GUARDIAN_MAX_FAIL_STREAK", 12))

    # Optional skip ledger snapshots
    skip_snap_enabled = bool(_cfg(bot, "GUARDIAN_SKIP_SNAPSHOT_ENABLED", False))
    skip_snap_cooldown = float(_cfg(bot, "GUARDIAN_SKIP_SNAPSHOT_COOLDOWN_SEC", 30.0))
    skip_snap_max_items = int(_cfg(bot, "GUARDIAN_SKIP_SNAPSHOT_MAX_ITEMS", 8))
    _last_skip_snap = 0.0

    shutdown_ev = _ensure_shutdown_event(bot)
    _ensure_shutdown_fields(bot)

    log_core.critical("GUARDIAN ONLINE — brainstem loop engaged")

    next_tick = time.monotonic()
    timeout_streak = 0
    fail_streak = 0

    async def _one_cycle():
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

        # 2b) Position manager tick: ALWAYS run (management independent of fills)
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
        nonlocal _last_skip_snap
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

        # set next tick before running cycle to reduce drift
        next_tick = max(next_tick + poll_sec, time.monotonic() + 0.001)

        # Random jitter: de-sync multiple bots
        if jitter and jitter > 0:
            try:
                j = random.uniform(0.0, min(float(jitter), 0.25))
                if j > 0:
                    await asyncio.sleep(j)
            except asyncio.CancelledError:
                raise

        try:
            await _run_cycle_with_timeout()

            # success resets streaks
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
