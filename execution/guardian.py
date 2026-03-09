# execution/guardian.py — SCALPER ETERNAL — BRAINSTEM OF REALITY — 2026 v1.8
# Patch vs v1.7:
# - ✅ POSMGR GATING: POSMGR_ENABLED / POSMGR_ENSURE_STOP / POSMGR_TICK_ENABLED checks
# - ✅ Cadence hygiene: separate exit-watch cadence, guardian poll cadence, optional posmgr cadence
# - ✅ Optional imports hardened + explicit "OPTIONAL MISSING" logging stays
# - ✅ Never fatal, guardian-safe, monotonic cadence

from __future__ import annotations

import asyncio
import os
import time
import random
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional, Dict, List

from utils.logging import log_core, log_entry
from execution.shutdown_control import request_shutdown
from execution.shutdown_control import ensure_traced_shutdown_event

# ─────────────────────────────────────────────────────────────────────
# Diagnostics helper (best-effort; never fatal)
# ─────────────────────────────────────────────────────────────────────
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


# ─────────────────────────────────────────────────────────────────────
# Optional wiring (never fatal)
# ─────────────────────────────────────────────────────────────────────

# Kill-switch
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

# Entry watch
try:
    from execution.entry_watch import poll_entry_watches  # type: ignore
except Exception as e:
    poll_entry_watches = None
    we_dont_have_this("execution.entry_watch.poll_entry_watches", e)

# Reconcile module
try:
    import execution.reconcile as reconcile_mod  # type: ignore
except Exception as e:
    reconcile_mod = None
    we_dont_have_this("execution.reconcile (module import)", e)

# Emergency module
try:
    import execution.emergency as emergency_mod  # type: ignore
except Exception as e:
    emergency_mod = None
    we_dont_have_this("execution.emergency (module import)", e)

# Position manager tick (POSMGR)
try:
    from execution.position_manager import position_manager_tick  # type: ignore
except Exception as e:
    position_manager_tick = None
    we_dont_have_this("execution.position_manager.position_manager_tick", e)

# Exit handler
try:
    from execution.exit import handle_exit  # type: ignore
except Exception as e:
    handle_exit = None
    we_dont_have_this("execution.exit.handle_exit", e)

# Collection watchdog (inline health check, never fatal)
try:
    from tools.collection_watchdog import run_once as _watchdog_run_once  # type: ignore
    import logging as _logging
    _watchdog_logger = _logging.getLogger("collection_watchdog")
except Exception as e:
    _watchdog_run_once = None
    _watchdog_logger = None
    we_dont_have_this("tools.collection_watchdog.run_once", e)

# Config hot-reload (never fatal)
try:
    from config.hot_reload import check_and_apply as _config_hot_reload  # type: ignore
except Exception as e:
    _config_hot_reload = None
    we_dont_have_this("config.hot_reload.check_and_apply", e)

# Alert rule engine (never fatal)
try:
    from monitoring.alert_rules import get_engine as _get_alert_engine  # type: ignore
except Exception as e:
    _get_alert_engine = None
    we_dont_have_this("monitoring.alert_rules.get_engine", e)


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────

def _cfg(bot, name: str, default):
    try:
        return getattr(bot.cfg, name, default)
    except Exception:
        return default


def _now() -> float:
    return time.time()


def _truthy(x) -> bool:
    if x is True:
        return True
    if isinstance(x, (int, float)) and x != 0:
        return True
    if isinstance(x, str) and x.strip().lower() in ("true", "1", "yes", "y", "t", "on"):
        return True
    return False


def _ensure_shutdown_event(bot) -> asyncio.Event:
    ev = getattr(bot, "_shutdown", None)
    if isinstance(ev, asyncio.Event):
        return ev
    ev = ensure_traced_shutdown_event(bot)
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


_GUARDIAN_STEP_TIMEOUT_SEC = 45.0  # max time any single guardian step may run

# Step profiling: track last N durations per step name
_STEP_TIMINGS: Dict[str, List[float]] = {}  # name -> [duration_sec, ...]
_STEP_TIMINGS_MAX_HISTORY = 20  # keep last N measurements per step
_STEP_TIMINGS_MAX_STEPS = 50  # max distinct step names tracked


def get_step_timings() -> Dict[str, Dict[str, float]]:
    """Return profiling summary: {step_name: {avg_ms, max_ms, last_ms, count}}."""
    result: Dict[str, Dict[str, float]] = {}
    for name, durations in _STEP_TIMINGS.items():
        if not durations:
            continue
        result[name] = {
            "avg_ms": sum(durations) / len(durations) * 1000.0,
            "max_ms": max(durations) * 1000.0,
            "last_ms": durations[-1] * 1000.0,
            "count": float(len(durations)),
        }
    return result


async def _safe_call(name: str, fn: Callable[..., Awaitable[Any]], *args, timeout_sec: float = 0.0, **kwargs):
    if not callable(fn):
        return
    t = float(timeout_sec) if timeout_sec > 0 else _GUARDIAN_STEP_TIMEOUT_SEC
    t0 = time.monotonic()
    try:
        await asyncio.wait_for(fn(*args, **kwargs), timeout=t)
    except asyncio.TimeoutError:
        log_entry.error(f"GUARDIAN: {name} TIMED OUT after {t:.1f}s — skipping")
    except asyncio.CancelledError:
        raise
    except Exception as e:
        log_entry.error(f"GUARDIAN: {name} failed: {e}")
    finally:
        elapsed = time.monotonic() - t0
        try:
            if name not in _STEP_TIMINGS:
                if len(_STEP_TIMINGS) >= _STEP_TIMINGS_MAX_STEPS:
                    oldest = next(iter(_STEP_TIMINGS))
                    _STEP_TIMINGS.pop(oldest, None)
                _STEP_TIMINGS[name] = []
            hist = _STEP_TIMINGS[name]
            hist.append(elapsed)
            if len(hist) > _STEP_TIMINGS_MAX_HISTORY:
                del hist[: len(hist) - _STEP_TIMINGS_MAX_HISTORY]
        except Exception:
            pass


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


# ─────────────────────────────────────────────────────────────────────
# Reconcile / Emergency / POSMGR ticks
# ─────────────────────────────────────────────────────────────────────

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


def _posmgr_allowed(bot) -> bool:
    """
    POSMGR gating policy:
    - If POSMGR_ENABLED is explicitly false -> no tick.
    - If POSMGR_TICK_ENABLED is explicitly false -> no tick.
    - If POSMGR_ENSURE_STOP is explicitly false -> still allow tick IF you want trailing/BE,
      but in YOUR current debugging sessions you usually want POSMGR silent.
      So default policy here: when POSMGR_ENSURE_STOP is false, we treat it as "disable posmgr" unless overridden.
    """
    if not callable(position_manager_tick):
        return False

    # hard master switch
    if not _truthy(_cfg(bot, "POSMGR_ENABLED", True)):
        return False

    # tick switch
    if not _truthy(_cfg(bot, "POSMGR_TICK_ENABLED", True)):
        return False

    # If ENSURE_STOP=0, still allow tick by default so trailing/BE can run.
    # You can explicitly disable via POSMGR_TICK_ENABLED=0.

    return True


async def _position_manager_tick(bot):
    if not _posmgr_allowed(bot):
        return
    await _safe_call("position_manager.position_manager_tick", position_manager_tick, bot)


_DEGRADED_CONSECUTIVE_FAILURES = 0
_DEGRADED_THRESHOLD = 3  # consecutive probe failures to enter degraded mode
_DEGRADED_NOTIFIED = False


def _set_exchange_degraded(bot, degraded: bool) -> None:
    """Set exchange degraded flag on bot.state.run_context."""
    try:
        rc = _rc_map(getattr(bot, "state", None))
        rc["exchange_degraded"] = bool(degraded)
        rc["exchange_degraded_ts"] = _now() if degraded else 0.0
    except Exception:
        pass


def is_exchange_degraded(bot) -> bool:
    """Check if exchange is in degraded mode (entries blocked, exits allowed)."""
    try:
        rc = getattr(getattr(bot, "state", None), "run_context", None)
        if isinstance(rc, dict):
            return bool(rc.get("exchange_degraded", False))
    except Exception:
        pass
    return False


async def _exchange_connectivity_tick(bot) -> None:
    """Probe exchange health — enter degraded mode on repeated failures."""
    global _DEGRADED_CONSECUTIVE_FAILURES, _DEGRADED_NOTIFIED
    try:
        ex = getattr(bot, "ex", None)
        if ex is None:
            return
        hc = getattr(ex, "_health_check", None)
        if not callable(hc):
            return
        last = float(getattr(ex, "last_health_check", 0.0) or 0.0)
        now = _now()
        # If last health check is older than 90s, force a probe
        if last > 0 and (now - last) < 90:
            return
        try:
            await hc()
            # Success: clear degraded state
            if _DEGRADED_CONSECUTIVE_FAILURES > 0:
                _DEGRADED_CONSECUTIVE_FAILURES = 0
                if is_exchange_degraded(bot):
                    _set_exchange_degraded(bot, False)
                    _DEGRADED_NOTIFIED = False
                    log_entry.warning("GUARDIAN: exchange connectivity restored — exiting DEGRADED mode")
                    notify = getattr(bot, "notify", None)
                    if notify is not None:
                        try:
                            await notify.speak(
                                "EXCHANGE CONNECTIVITY RESTORED — degraded mode OFF, entries re-enabled",
                                "critical",
                            )
                        except Exception:
                            pass
        except Exception as e:
            _DEGRADED_CONSECUTIVE_FAILURES += 1
            log_entry.error(f"GUARDIAN: exchange connectivity probe failed ({_DEGRADED_CONSECUTIVE_FAILURES}x): {e}")
            if _DEGRADED_CONSECUTIVE_FAILURES >= _DEGRADED_THRESHOLD and not is_exchange_degraded(bot):
                _set_exchange_degraded(bot, True)
                log_entry.critical(
                    f"GUARDIAN: ENTERING DEGRADED MODE — {_DEGRADED_CONSECUTIVE_FAILURES} consecutive "
                    f"exchange failures. New entries BLOCKED, exits still allowed."
                )
            notify = getattr(bot, "notify", None)
            if notify is not None and not _DEGRADED_NOTIFIED:
                _DEGRADED_NOTIFIED = True
                try:
                    await notify.speak(
                        f"EXCHANGE DEGRADED — {_DEGRADED_CONSECUTIVE_FAILURES}x failures. "
                        f"Entries blocked, exits allowed. Will auto-recover on next successful probe.",
                        "critical",
                    )
                except Exception:
                    pass
    except Exception:
        pass


_STUCK_ALERTED: dict = {}  # sym -> alert_ts (tracks symbols we already alerted for stuck positions)


async def _position_stuck_check(bot) -> None:
    """Alert if any position has been open > TTL without exit activity."""
    try:
        st = getattr(bot, "state", None)
        if st is None:
            return
        positions = getattr(st, "positions", None)
        if not isinstance(positions, dict):
            return
        ttl = float(_cfg(bot, "POSITION_STUCK_TTL_SEC", 3600.0) or 3600.0)
        if ttl <= 0:
            return
        now = _now()
        for sym, pos in positions.items():
            if not isinstance(pos, dict):
                continue
            qty = float(pos.get("size") or pos.get("qty") or 0.0)
            if abs(qty) <= 0:
                _STUCK_ALERTED.pop(sym, None)
                continue
            entry_ts = float(pos.get("entry_ts") or pos.get("open_ts") or 0.0)
            if entry_ts <= 0:
                continue
            age = now - entry_ts
            if age > ttl and sym not in _STUCK_ALERTED:
                _STUCK_ALERTED[sym] = now
                # Evict oldest if dict grows too large
                if len(_STUCK_ALERTED) > 500:
                    oldest = min(_STUCK_ALERTED, key=_STUCK_ALERTED.get)  # type: ignore[arg-type]
                    _STUCK_ALERTED.pop(oldest, None)
                msg = (
                    f"POSITION STUCK — {sym} open for {age / 3600:.1f}h "
                    f"(TTL={ttl / 3600:.1f}h) qty={qty}"
                )
                log_entry.warning(msg)
                notify = getattr(bot, "notify", None)
                if notify is not None:
                    try:
                        await notify.speak(msg, "warning")
                    except Exception:
                        pass
    except Exception:
        pass


_LAST_LOG_ROTATION_TS: float = 0.0
_LOG_ROTATION_INTERVAL_SEC: float = 86400.0  # once per day


_HEARTBEAT_PATH = Path("logs/health/heartbeat.json")


async def _write_heartbeat(bot) -> None:
    """Write a heartbeat file every guardian cycle for external dead-bot detection.

    External monitors (cron, Prometheus node-exporter, uptime-kuma) can check
    if this file's mtime is stale (> 30s) to detect a dead bot process.
    """
    try:
        from execution.runtime_helpers import atomic_write_json
        now = _now()
        data = {
            "ts": now,
            "pid": os.getpid(),
            "uptime_sec": now - float(getattr(bot, "STARTUP_TIMESTAMP", now) or now),
        }
        try:
            if callable(is_halted):
                data["kill_switch_active"] = bool(is_halted(bot))
        except Exception:
            pass
        try:
            positions = getattr(getattr(bot, "state", None), "positions", None)
            if isinstance(positions, dict):
                data["open_positions"] = sum(
                    1 for v in positions.values()
                    if isinstance(v, dict) and abs(float(v.get("size") or v.get("qty") or 0)) > 0
                )
        except Exception:
            pass
        data["exchange_degraded"] = is_exchange_degraded(bot)
        # Include step profiling summary (top 5 slowest by avg)
        try:
            timings = get_step_timings()
            if timings:
                top5 = sorted(timings.items(), key=lambda x: x[1]["avg_ms"], reverse=True)[:5]
                data["step_profiling"] = {k: v for k, v in top5}
        except Exception:
            pass
        atomic_write_json(_HEARTBEAT_PATH, data)
    except Exception:
        pass


async def _log_rotation_tick(bot) -> None:
    """Run log rotation at most once per day."""
    global _LAST_LOG_ROTATION_TS
    now = _now()
    interval = float(_cfg(bot, "LOG_ROTATION_INTERVAL_SEC", _LOG_ROTATION_INTERVAL_SEC) or _LOG_ROTATION_INTERVAL_SEC)
    if (now - _LAST_LOG_ROTATION_TS) < interval:
        return
    _LAST_LOG_ROTATION_TS = now
    try:
        from monitoring.log_rotation import run_rotation
        max_size = float(_cfg(bot, "LOG_ROTATION_MAX_SIZE_MB", 50.0) or 50.0)
        max_age = float(_cfg(bot, "LOG_ROTATION_MAX_AGE_DAYS", 180.0) or 180.0)
        await asyncio.get_event_loop().run_in_executor(
            None, lambda: run_rotation(max_size_mb=max_size, max_age_days=max_age)
        )
    except Exception:
        pass


async def _config_hot_reload_tick(bot) -> None:
    """Check for config override file changes and apply them."""
    try:
        if not callable(_config_hot_reload):
            return
        cfg = getattr(bot, "cfg", None)
        if cfg is None:
            return
        changes = _config_hot_reload(cfg)
        if changes:
            # Notify operator about config changes
            notify = getattr(bot, "notify", None)
            if notify is not None:
                changed_keys = ", ".join(changes.keys())
                try:
                    await notify.speak(
                        f"CONFIG HOT-RELOAD: {len(changes)} field(s) updated: {changed_keys}",
                        "normal",
                    )
                except Exception:
                    pass
    except Exception:
        pass


async def _alert_rules_tick(bot) -> None:
    """Evaluate structured alert rules and send notifications for triggered alerts."""
    try:
        if not callable(_get_alert_engine):
            return
        engine = _get_alert_engine()
        metrics = engine.collect_metrics(bot)
        if not metrics:
            return
        alerts = engine.evaluate(metrics)
        if not alerts:
            return
        notify = getattr(bot, "notify", None)
        for alert in alerts:
            try:
                prefix = {"CRITICAL": "\U0001f6a8", "WARNING": "\u26a0\ufe0f", "INFO": "\u2139\ufe0f"}.get(
                    alert.severity, "\u2139\ufe0f"
                )
                msg = f"{prefix} ALERT [{alert.rule_id}]: {alert.message}"
                priority = "critical" if alert.severity == "CRITICAL" else "normal"
                if notify is not None:
                    await notify.speak(msg, priority)
                log_entry.warning(f"ALERT RULE FIRED: {alert.rule_id} — {alert.message}")
            except Exception:
                pass
    except Exception:
        pass


async def _margin_ratio_refresh(bot) -> None:
    """Fetch margin ratio from Binance and cache on bot.state for kill switch."""
    try:
        ex = getattr(bot, "ex", None)
        if ex is None:
            return
        resp = None
        if hasattr(ex, "fapiPrivateV2GetAccount"):
            resp = await ex.fapiPrivateV2GetAccount()
        elif hasattr(ex, "fapiPrivate_get_account"):
            resp = await ex.fapiPrivate_get_account()
        if isinstance(resp, dict):
            raw = resp.get("totalMarginRatio") or resp.get("marginRatio")
            if raw is not None:
                ratio = float(raw)
                st = getattr(bot, "state", None)
                if st is not None:
                    st.margin_ratio = ratio
                    st.margin_ratio_ts = _now()
    except Exception:
        pass


async def _collection_watchdog_tick(bot) -> None:
    """Run collection watchdog inline — checks DB freshness, writes health state."""
    if not callable(_watchdog_run_once):
        return
    try:
        from pathlib import Path as _Path
        import os as _os
        db_env = _os.environ.get("MICROSTRUCTURE_DB_PATH", "data/microstructure.db")
        db_path = _Path(str(getattr(getattr(bot, "cfg", None), "MICROSTRUCTURE_DB_PATH", db_env) or db_env))
        symbols_raw = str(getattr(getattr(bot, "cfg", None), "WATCHDOG_SYMBOLS", "BTCUSDT,ETHUSDT") or "BTCUSDT,ETHUSDT")
        symbols = [s.strip().upper() for s in symbols_raw.split(",") if s.strip()]
        threshold = int(getattr(getattr(bot, "cfg", None), "WATCHDOG_STALE_THRESHOLD_SEC", 300) or 300)
        dry_run = bool(getattr(getattr(bot, "cfg", None), "SCALPER_DRY_RUN", False))
        await _watchdog_run_once(
            db_path=db_path,
            symbols=symbols,
            table="",
            stale_threshold_sec=threshold,
            logger=_watchdog_logger or __import__("logging").getLogger("collection_watchdog"),
            dry_run=dry_run,
        )
    except Exception as e:
        log_entry.error(f"GUARDIAN: collection_watchdog_tick failed: {e}")


# ─────────────────────────────────────────────────────────────────────
# Exit watcher helpers
# ─────────────────────────────────────────────────────────────────────

def _is_reduce_only_like(obj: dict) -> bool:
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
    rc = _rc_map(state)
    v = rc.get("exit_since_ms")
    try:
        vv = int(v)
        if vv > 0:
            return vv
    except Exception:
        pass
    return int(time.time() * 1000) - 2 * 60 * 1000


def _set_exit_since_ms(state, since_ms: int) -> None:
    rc = _rc_map(state)
    try:
        rc["exit_since_ms"] = int(since_ms)
    except Exception:
        pass


def _ms_from_ts_any(x) -> int:
    if x is None:
        return 0
    try:
        if isinstance(x, (int, float)):
            if float(x) > 10_000_000_000:
                return int(x)
            return int(float(x) * 1000)
    except Exception:
        pass
    return 0


def _normalize_trade_to_order_like(trade: dict) -> dict:
    out = dict(trade or {})
    try:
        if not out.get("id"):
            oid = out.get("order") or out.get("orderId") or out.get("tradeId")
            if oid:
                out["id"] = str(oid)

        amt = out.get("amount")
        if amt is not None and out.get("filled") is None:
            out["filled"] = amt

        info = out.get("info") or {}
        if not isinstance(info, dict):
            info = {"raw": info}
        if "executedQty" not in info and out.get("filled") is not None:
            info["executedQty"] = out.get("filled")
        out["info"] = info
    except Exception:
        pass
    return out


async def _exit_watch_tick(bot) -> None:
    if not callable(handle_exit):
        return

    enabled = bool(_cfg(bot, "EXIT_WATCH_ENABLED", True))
    if not enabled:
        return

    ex = getattr(bot, "ex", None)
    st = getattr(bot, "state", None)
    if ex is None or st is None:
        return

    _ensure_known_exit_sets(st)

    limit = int(_cfg(bot, "EXIT_WATCH_LIMIT", 50))
    if limit <= 0:
        limit = 50

    since_ms = _get_exit_since_ms(st)
    max_seen_ms = since_ms

    symbols_any: List[str] = []
    try:
        syms = getattr(bot, "active_symbols", None) or getattr(getattr(bot, "cfg", None), "ACTIVE_SYMBOLS", None)
        if isinstance(syms, (set, list, tuple)):
            symbols_any = [str(x) for x in list(syms)]
    except Exception:
        symbols_any = []

    def _looks_like_exit_by_side(obj: dict) -> bool:
        try:
            sym = obj.get("symbol")
            if not sym:
                return False
            k = _symkey(sym)
            pos = (st.positions or {}).get(k)
            if not pos:
                return False
            pos_side = str(getattr(pos, "side", "") or "").lower().strip()
            order_side = str(obj.get("side") or (obj.get("info") or {}).get("side") or "").lower().strip()
            if pos_side == "long" and order_side == "sell":
                return True
            if pos_side == "short" and order_side == "buy":
                return True
        except Exception:
            pass
        return False

    async def _process_order_like(obj: dict) -> None:
        nonlocal max_seen_ms
        if not isinstance(obj, dict):
            return

        oid = obj.get("id") or (obj.get("info") or {}).get("orderId") or obj.get("order")
        if oid:
            oid = str(oid)
            if oid in st.known_exit_order_ids:
                return

        if not _is_reduce_only_like(obj) and not _looks_like_exit_by_side(obj):
            return

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

        if oid:
            st.known_exit_order_ids.add(oid)

        await _safe_call("exit.handle_exit", handle_exit, bot, obj)

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

        order_like = _normalize_trade_to_order_like(tr)
        oid = order_like.get("id")
        if oid:
            oid = str(oid)
            if oid in st.known_exit_order_ids:
                return

        if not _is_reduce_only_like(order_like) and not _looks_like_exit_by_side(order_like):
            return

        if tid:
            st.known_exit_trade_ids.add(tid)
        if oid:
            st.known_exit_order_ids.add(oid)

        await _safe_call("exit.handle_exit(trade->order_like)", handle_exit, bot, order_like)

    # 1) closed orders
    try:
        fn = getattr(ex, "fetch_closed_orders", None)
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

    # 2) orders
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

    # 3) trades fallback
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

    if max_seen_ms > since_ms:
        _set_exit_since_ms(st, int(max_seen_ms) + 1)


# ─────────────────────────────────────────────────────────────────────
# Guardian loop
# ─────────────────────────────────────────────────────────────────────

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

    # Exit watcher cadence (separate from guardian poll)
    exit_watch_enabled = bool(_cfg(bot, "EXIT_WATCH_ENABLED", True))
    exit_watch_every = float(_cfg(bot, "EXIT_WATCH_EVERY_SEC", 1.25))
    if exit_watch_every <= 0:
        exit_watch_every = 1.25
    _last_exit_watch = 0.0

    # POSMGR cadence (optional extra throttle)
    posmgr_every = float(_cfg(bot, "POSMGR_EVERY_SEC", 0.0))  # 0 -> every guardian cycle (if allowed)
    _last_posmgr = 0.0

    # Collection watchdog cadence (default: every 60s)
    watchdog_every = float(_cfg(bot, "WATCHDOG_EVERY_SEC", 60.0))
    if watchdog_every <= 0:
        watchdog_every = 60.0
    _last_watchdog = 0.0

    # Exchange connectivity probe cadence (default: every 60s)
    ex_probe_every = float(_cfg(bot, "EX_PROBE_EVERY_SEC", 60.0))
    if ex_probe_every <= 0:
        ex_probe_every = 60.0
    _last_ex_probe = 0.0

    shutdown_ev = _ensure_shutdown_event(bot)
    _ensure_shutdown_fields(bot)

    log_core.critical("GUARDIAN ONLINE — brainstem loop engaged")

    next_tick = time.monotonic()
    timeout_streak = 0
    fail_streak = 0

    async def _one_cycle():
        nonlocal _last_exit_watch, _last_posmgr, _last_watchdog, _last_ex_probe

        # 0) Heartbeat for external monitoring (every cycle, cheap)
        await _safe_call("heartbeat", _write_heartbeat, bot)

        # 0a) Kill-switch tick (evaluate)
        if respect_kill and callable(tick_kill_switch):
            await _safe_call("kill_switch.tick_kill_switch", tick_kill_switch, bot)

        # 0b) Kill-switch state snapshot (for optional hooks)
        halted = False
        if respect_kill and callable(is_halted):
            try:
                halted = bool(is_halted(bot))
            except Exception:
                halted = False

        # Exit watch tick (high-frequency, lightweight)
        if exit_watch_enabled and callable(handle_exit):
            now_ts = _now()
            if (now_ts - _last_exit_watch) >= exit_watch_every:
                _last_exit_watch = now_ts
                await _safe_call("exit_watch_tick", _exit_watch_tick, bot)

        # 1) Entry watch housekeeping
        if callable(poll_entry_watches):
            await _safe_call("entry_watch.poll_entry_watches", poll_entry_watches, bot)

        # optional entry_watch halt hook
        if halted:
            try:
                import execution.entry_watch as entry_watch_mod  # type: ignore
                ew_hook = getattr(entry_watch_mod, "on_halt", None)
                if callable(ew_hook):
                    await _safe_call("entry_watch.on_halt", ew_hook, bot)
            except Exception as e:
                we_dont_have_this("execution.entry_watch.on_halt (runtime)", e)

        # 2) Reconcile truth
        await _reconcile_tick(bot, legacy_brief_sec)

        # optional reconcile halt hook
        if halted and reconcile_mod is not None:
            try:
                rh = getattr(reconcile_mod, "on_halt", None)
                if callable(rh):
                    await _safe_call("reconcile.on_halt", rh, bot)
            except Exception as e:
                we_dont_have_this("execution.reconcile.on_halt (runtime)", e)

        # 2b) Position manager tick (guarded + optionally throttled)
        if _posmgr_allowed(bot):
            now_ts = _now()
            if (posmgr_every <= 0) or ((now_ts - _last_posmgr) >= posmgr_every):
                _last_posmgr = now_ts
                await _position_manager_tick(bot)

        # optional posmgr halt hook
        if halted:
            try:
                import execution.position_manager as pm_mod  # type: ignore
                pm_hook = getattr(pm_mod, "on_halt", None)
                if callable(pm_hook):
                    await _safe_call("position_manager.on_halt", pm_hook, bot)
            except Exception as e:
                we_dont_have_this("execution.position_manager.on_halt (runtime)", e)

        # 3) Emergency checks
        await _emergency_tick(bot, legacy_brief_sec)

        # 4) Collection watchdog (DB freshness + health state write)
        if callable(_watchdog_run_once):
            now_ts = _now()
            if (now_ts - _last_watchdog) >= watchdog_every:
                _last_watchdog = now_ts
                await _safe_call("collection_watchdog_tick", _collection_watchdog_tick, bot)

        # 5) Exchange connectivity probe
        now_ts = _now()
        if (now_ts - _last_ex_probe) >= ex_probe_every:
            _last_ex_probe = now_ts
            await _safe_call("exchange_connectivity_tick", _exchange_connectivity_tick, bot)

        # 6) Margin ratio refresh (piggybacks on exchange probe cadence)
        if (now_ts - _last_ex_probe) < 2.0:  # just probed exchange
            await _safe_call("margin_ratio_refresh", _margin_ratio_refresh, bot)

        # 7) Position stuck detection (on watchdog cadence)
        if (now_ts - _last_watchdog) < 2.0:  # just ran watchdog
            await _safe_call("position_stuck_check", _position_stuck_check, bot)

        # 8) Log rotation (once per day, non-blocking)
        await _safe_call("log_rotation_tick", _log_rotation_tick, bot)

        # 9) Config hot-reload (check override file every ~10s)
        await _safe_call("config_hot_reload_tick", _config_hot_reload_tick, bot)

        # 10) Structured alert rules evaluation (on watchdog cadence)
        if (now_ts - _last_watchdog) < 2.0:
            await _safe_call("alert_rules_tick", _alert_rules_tick, bot)

        # optional emergency halt hook
        if halted and emergency_mod is not None:
            try:
                eh = getattr(emergency_mod, "on_halt", None)
                if callable(eh):
                    await _safe_call("emergency.on_halt", eh, bot)
            except Exception as e:
                we_dont_have_this("execution.emergency.on_halt (runtime)", e)

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
                log_core.critical("GUARDIAN: timeout streak exceeded — requesting shutdown (opt-in)")
                request_shutdown(
                    bot,
                    reason=f"GUARDIAN timeout streak exceeded: {timeout_streak} (max={max_timeout_streak})",
                    source="execution.guardian",
                    fatal=False,
                )

        except asyncio.CancelledError:
            raise

        except Exception as e:
            fail_streak += 1
            _record_shutdown_reason(bot, reason=f"GUARDIAN CYCLE ERROR: {e}", source="execution.guardian")
            if log_cycle_failures:
                log_entry.error(f"GUARDIAN: cycle failed streak={fail_streak}: {e}")

            if shutdown_on_timeout and fail_streak >= max_fail_streak:
                log_core.critical("GUARDIAN: fail streak exceeded — requesting shutdown (opt-in)")
                request_shutdown(
                    bot,
                    reason=f"GUARDIAN fail streak exceeded: {fail_streak} (max={max_fail_streak})",
                    source="execution.guardian",
                    fatal=False,
                    exc=e,
                )

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
