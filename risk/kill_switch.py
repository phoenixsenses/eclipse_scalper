# risk/kill_switch.py — SCALPER ETERNAL — GLOBAL CIRCUIT BREAKER — 2026 v1.6 (TELEMETRY WIRED)
# Patch vs v1.5:
# - ✅ Optional telemetry emits:
#     - kill_switch.halt / clear
#     - kill_switch.escalate_flat / escalate_shutdown
#     - kill_switch.evaluate_error
# - ✅ Never fatal if telemetry missing
# - ✅ No logic changes to halt conditions / escalation behavior

import time
from typing import Tuple, Optional, Any

from utils.logging import log_core

# Optional telemetry (never fatal)
try:
    from execution.telemetry import emit  # type: ignore
except Exception:
    emit = None


def _now() -> float:
    return time.time()


def _safe_float(x, default: float = 0.0) -> float:
    try:
        v = float(x)
        if v != v:  # NaN
            return default
        return v
    except Exception:
        return default


def _truthy(x) -> bool:
    if x is True:
        return True
    if isinstance(x, (int, float)) and x != 0:
        return True
    if isinstance(x, str) and x.strip().lower() in ("1", "true", "yes", "y", "on"):
        return True
    return False


def _is_finite(x: Any) -> bool:
    try:
        v = float(x)
        if v != v:
            return False
        if v == float("inf") or v == float("-inf"):
            return False
        return True
    except Exception:
        return False


def _ensure_state_fields(state) -> None:
    # Halt / quarantine
    if not hasattr(state, "halt_until_ts"):
        state.halt_until_ts = 0.0
    if not hasattr(state, "halt_reason"):
        state.halt_reason = ""
    if not hasattr(state, "halt_count"):
        state.halt_count = 0

    # Metrics for rate-based checks
    if not hasattr(state, "kill_metrics") or not isinstance(getattr(state, "kill_metrics", None), dict):
        state.kill_metrics = {}

    km = state.kill_metrics
    km.setdefault("boot_ts", _now())  # used for boot grace

    km.setdefault("last_check_ts", 0.0)
    km.setdefault("last_ex_error_count", 0)
    km.setdefault("last_ex_request_count", 0)
    km.setdefault("last_trip_ts", 0.0)
    km.setdefault("trip_streak", 0)

    # Last evaluation record (cheap gating)
    km.setdefault("last_eval_ok", True)
    km.setdefault("last_eval_reason", "")

    # Optional: track data-sample readiness
    km.setdefault("data_samples_ok", 0)

    # Trip history (ring buffer)
    km.setdefault("trip_history", [])  # list of dicts


def _cfg(bot, name: str, default):
    cfg = getattr(bot, "cfg", None)
    return getattr(cfg, name, default) if cfg is not None else default


async def _safe_speak(bot, text: str, priority: str = "critical"):
    notify = getattr(bot, "notify", None)
    if notify is None:
        return
    try:
        await notify.speak(text, priority)
    except Exception:
        pass


async def _telemetry(bot, event: str, *, data: Optional[dict] = None, level: str = "info") -> None:
    if not callable(emit):
        return
    try:
        await emit(bot, event, data=(data or {}), symbol=None, level=level)
    except Exception:
        pass


def _record_shutdown_reason(bot, reason: str, source: str) -> None:
    """
    Guardian prints these fields when shutdown happens.
    Kill-switch can set them too, best-effort.
    """
    try:
        st = getattr(bot, "state", None)
        if st is None:
            return
        if not hasattr(st, "shutdown_reason"):
            st.shutdown_reason = ""
        if not hasattr(st, "shutdown_source"):
            st.shutdown_source = ""
        if not hasattr(st, "shutdown_ts"):
            st.shutdown_ts = 0.0

        if not getattr(st, "shutdown_reason", ""):
            st.shutdown_reason = str(reason or "")[:500]
            st.shutdown_source = str(source or "")[:120]
            st.shutdown_ts = _now()
    except Exception:
        pass


def _set_shutdown_flag(bot) -> None:
    """
    Matches guardian's shutdown event usage (bot._shutdown is asyncio.Event).
    If not present, we set state.halt_until_ts huge (best-effort fallback).
    """
    try:
        ev = getattr(bot, "_shutdown", None)
        if ev is not None and hasattr(ev, "set"):
            ev.set()
            return
    except Exception:
        pass

    # fallback: extended halt if we can't access the event
    try:
        _ensure_state_fields(bot.state)
        bot.state.halt_until_ts = max(float(getattr(bot.state, "halt_until_ts", 0.0) or 0.0), _now() + 3600.0)
    except Exception:
        pass


def is_halted(bot) -> bool:
    """
    Sync-safe: core loop may call this without await.
    """
    try:
        _ensure_state_fields(bot.state)
        return _now() < float(bot.state.halt_until_ts or 0.0)
    except Exception:
        return False


def remaining_halt_seconds(bot) -> float:
    try:
        _ensure_state_fields(bot.state)
        return max(0.0, float(bot.state.halt_until_ts or 0.0) - _now())
    except Exception:
        return 0.0


def should_evaluate(bot) -> bool:
    """
    Optional throttle: if watcher calls evaluate too frequently, you can skip.
    Default: always True.
    """
    try:
        _ensure_state_fields(bot.state)
        min_gap = float(_cfg(bot, "KILL_SWITCH_MIN_EVAL_GAP_SEC", 0.0) or 0.0)
        if min_gap <= 0:
            return True
        last = float(bot.state.kill_metrics.get("last_check_ts", 0.0) or 0.0)
        return (_now() - last) >= min_gap
    except Exception:
        return True


def _push_trip_history(bot, reason: str, seconds_eff: float) -> None:
    try:
        _ensure_state_fields(bot.state)
        km = bot.state.kill_metrics
        hist = km.get("trip_history")
        if not isinstance(hist, list):
            hist = []
            km["trip_history"] = hist

        hist.append(
            {
                "ts": _now(),
                "reason": str(reason or "")[:240],
                "halt_sec": float(seconds_eff),
            }
        )

        max_hist = int(_cfg(bot, "KILL_SWITCH_TRIP_HISTORY_MAX", 12) or 12)
        if max_hist <= 0:
            max_hist = 12
        if len(hist) > max_hist:
            del hist[:-max_hist]
    except Exception:
        pass


async def request_halt(bot, seconds: float, reason: str, severity: str = "critical") -> None:
    """
    Enter quarantine mode. Entries should stop while halted.
    Adds escalation if the bot keeps tripping repeatedly.
    """
    try:
        _ensure_state_fields(bot.state)
        km = bot.state.kill_metrics
        now = _now()

        base = max(5.0, float(seconds or 0.0))

        streak_window = float(_cfg(bot, "KILL_SWITCH_TRIP_STREAK_WINDOW_SEC", 900.0))  # 15m
        max_mult = float(_cfg(bot, "KILL_SWITCH_MAX_BACKOFF_MULT", 4.0))
        last_trip = float(km.get("last_trip_ts", 0.0) or 0.0)

        if now - last_trip <= streak_window:
            km["trip_streak"] = int(km.get("trip_streak", 0) or 0) + 1
        else:
            km["trip_streak"] = 1

        km["last_trip_ts"] = now

        mult = min(max_mult, 1.0 + 0.5 * max(0, int(km["trip_streak"]) - 1))
        seconds_eff = base * mult

        until = now + seconds_eff
        if until > float(bot.state.halt_until_ts or 0.0):
            bot.state.halt_until_ts = until

        # Store FIRST halt reason (most important forensic)
        if not str(getattr(bot.state, "halt_reason", "") or ""):
            bot.state.halt_reason = str(reason or "").strip()[:500]

        bot.state.halt_count = int(getattr(bot.state, "halt_count", 0) or 0) + 1

        km["last_eval_ok"] = False
        km["last_eval_reason"] = str(reason or "").strip()[:500]

        _push_trip_history(bot, reason=str(reason or ""), seconds_eff=float(seconds_eff))

        msg = f"KILL SWITCH HALT — {int(seconds_eff)}s | {str(reason or '')[:220]}"
        log_core.critical(msg)
        await _safe_speak(bot, msg, severity)

        # Telemetry: halt
        await _telemetry(
            bot,
            "kill_switch.halt",
            level="critical",
            data={
                "reason": str(reason or "")[:500],
                "severity": str(severity or "")[:32],
                "seconds_req": float(seconds or 0.0),
                "seconds_eff": float(seconds_eff),
                "trip_streak": int(km.get("trip_streak", 0) or 0),
                "halt_until_ts": float(getattr(bot.state, "halt_until_ts", 0.0) or 0.0),
                "halt_count": int(getattr(bot.state, "halt_count", 0) or 0),
            },
        )

        # ----------------------------
        # Escalation ladder
        # ----------------------------
        esc_flat_trips = int(_cfg(bot, "KILL_ESCALATE_FLAT_AFTER_TRIPS", 0) or 0)
        esc_shutdown_trips = int(_cfg(bot, "KILL_ESCALATE_SHUTDOWN_AFTER_TRIPS", 0) or 0)
        esc_window = float(_cfg(bot, "KILL_ESCALATE_WINDOW_SEC", streak_window) or streak_window)

        # Count trips in window using history (more robust than streak)
        trips_in_window = 0
        try:
            hist = bot.state.kill_metrics.get("trip_history", [])
            if isinstance(hist, list) and hist:
                cutoff = now - max(30.0, esc_window)
                trips_in_window = sum(1 for x in hist if isinstance(x, dict) and _safe_float(x.get("ts"), 0.0) >= cutoff)
        except Exception:
            trips_in_window = int(km.get("trip_streak", 0) or 0)

        do_flatten = _truthy(_cfg(bot, "KILL_SWITCH_EMERGENCY_FLAT", False))

        # Escalate: emergency flat
        if esc_flat_trips > 0 and do_flatten and trips_in_window >= esc_flat_trips:
            why = f"ESCALATION: {trips_in_window} trips/{esc_window:.0f}s — {reason}"
            await _telemetry(
                bot,
                "kill_switch.escalate_flat",
                level="critical",
                data={
                    "trips_in_window": int(trips_in_window),
                    "window_sec": float(esc_window),
                    "reason": str(reason or "")[:500],
                },
            )
            try:
                await _try_emergency_flat(bot, why)
            except Exception:
                pass

        # Escalate: shutdown
        if esc_shutdown_trips > 0 and trips_in_window >= esc_shutdown_trips:
            shutdown_reason = f"KILL_SWITCH ESCALATED SHUTDOWN — {trips_in_window} trips/{esc_window:.0f}s | {reason}"
            _record_shutdown_reason(
                bot,
                reason=shutdown_reason,
                source="risk.kill_switch",
            )
            log_core.critical("KILL SWITCH → SHUTDOWN FLAG SET (escalation)")

            await _telemetry(
                bot,
                "kill_switch.escalate_shutdown",
                level="critical",
                data={
                    "trips_in_window": int(trips_in_window),
                    "window_sec": float(esc_window),
                    "reason": str(reason or "")[:500],
                    "shutdown_reason": shutdown_reason[:500],
                },
            )

            _set_shutdown_flag(bot)

    except Exception:
        pass


async def clear_halt(bot, note: str = "") -> None:
    try:
        _ensure_state_fields(bot.state)
        bot.state.halt_until_ts = 0.0
        bot.state.halt_reason = ""

        km = bot.state.kill_metrics
        km["last_eval_ok"] = True
        km["last_eval_reason"] = ""

        if note:
            log_core.info(f"KILL SWITCH CLEAR — {note}")

        await _telemetry(
            bot,
            "kill_switch.clear",
            level="info",
            data={
                "note": str(note or "")[:300],
            },
        )
    except Exception:
        pass


def _pick_probe_symbol(bot) -> str:
    k = "BTCUSDT"
    try:
        active = getattr(bot, "active_symbols", None)
        if isinstance(active, set) and active:
            if k in active:
                return k
            return next(iter(active))
    except Exception:
        pass
    return k


def _age_from_cached_ohlcv(bot, k: str) -> Optional[float]:
    """
    If last_poll isn't set yet, but OHLCV cache is loaded from disk,
    derive staleness from the last candle timestamp.
    Expects rows like [ts_ms, o, h, l, c, v].
    """
    data = getattr(bot, "data", None)
    if data is None:
        return None

    try:
        ohlcv = getattr(data, "ohlcv", None)
        if not isinstance(ohlcv, dict):
            return None
        rows = ohlcv.get(k) or []
        if not isinstance(rows, list) or not rows:
            return None
        last = rows[-1]
        if not isinstance(last, (list, tuple)) or not last:
            return None
        ts_ms = _safe_float(last[0], 0.0)
        if ts_ms <= 0:
            return None
        return max(0.0, _now() - (ts_ms / 1000.0))
    except Exception:
        return None


def _data_age(bot) -> float:
    """
    Best effort:
      1) Prefer DataCache.get_cache_age(k, "1m")
      2) If that returns inf / non-finite (common right after cache load), derive from cached OHLCV last candle
      3) Else huge sentinel
    """
    data = getattr(bot, "data", None)
    if data is None:
        return 999999.0

    k = _pick_probe_symbol(bot)

    fn = getattr(data, "get_cache_age", None)
    if callable(fn):
        try:
            age = fn(k, "1m")
            if _is_finite(age):
                return float(age)
        except Exception:
            pass

    derived = _age_from_cached_ohlcv(bot, k)
    if derived is not None and _is_finite(derived):
        return float(derived)

    return 999999.0


def _equity(bot) -> float:
    return _safe_float(getattr(bot.state, "current_equity", 0.0), 0.0)


def _peak_equity(bot) -> float:
    pe = _safe_float(getattr(bot.state, "peak_equity", 0.0), 0.0)
    if pe > 0:
        return pe
    return _safe_float(getattr(bot, "session_peak_equity", 0.0), 0.0)


def _start_of_day_equity(bot) -> float:
    return _safe_float(getattr(bot.state, "start_of_day_equity", 0.0), 0.0)


def _daily_pnl(bot) -> float:
    return _safe_float(getattr(bot.state, "daily_pnl", 0.0), 0.0)


def _ex_counts(bot) -> Tuple[int, int]:
    ex = getattr(bot, "ex", None)
    req = int(getattr(ex, "request_count", 0) or 0) if ex is not None else 0
    err = int(getattr(ex, "error_count", 0) or 0) if ex is not None else 0
    return req, err


def _exchange_health_stale(bot) -> bool:
    ex = getattr(bot, "ex", None)
    if ex is None:
        return False

    max_stale = float(_cfg(bot, "KILL_MAX_EX_HEALTH_STALE_SEC", 0.0) or 0.0)
    if max_stale <= 0:
        return False

    last = _safe_float(getattr(ex, "last_health_check", 0.0), 0.0)
    if last <= 0:
        return False

    return (_now() - last) > max_stale


async def _try_emergency_flat(bot, reason: str):
    try:
        fn = None
        try:
            from execution.emergency import emergency_flat as fn  # type: ignore
        except Exception:
            fn = None

        if callable(fn):
            log_core.critical(f"KILL SWITCH → EMERGENCY FLAT: {reason}")
            await fn(bot)
    except Exception:
        pass


async def evaluate(bot) -> Tuple[bool, Optional[str]]:
    """
    Returns (trade_allowed, trip_reason_if_any).
    If tripped, sets halt state and (optionally) escalates.
    """
    try:
        _ensure_state_fields(bot.state)

        if not _truthy(_cfg(bot, "KILL_SWITCH_ENABLED", True)):
            bot.state.kill_metrics["last_eval_ok"] = True
            bot.state.kill_metrics["last_eval_reason"] = ""
            return True, None

        # Optional throttle
        if not should_evaluate(bot):
            if is_halted(bot):
                return False, str(getattr(bot.state, "halt_reason", "") or "HALTED")
            ok = bool(bot.state.kill_metrics.get("last_eval_ok", True))
            rs = str(bot.state.kill_metrics.get("last_eval_reason", "") or "")
            return ok, (rs if not ok else None)

        # If already halted: still not allowed
        if is_halted(bot):
            bot.state.kill_metrics["last_eval_ok"] = False
            bot.state.kill_metrics["last_eval_reason"] = str(getattr(bot.state, "halt_reason", "") or "HALTED")
            bot.state.kill_metrics["last_check_ts"] = _now()
            return False, str(getattr(bot.state, "halt_reason", "") or "HALTED")

        now = _now()
        km = bot.state.kill_metrics

        # ---------- thresholds ----------
        cooldown = float(_cfg(bot, "KILL_SWITCH_COOLDOWN_SEC", 300.0))

        # boot grace for data staleness only
        data_boot_grace = float(_cfg(bot, "KILL_DATA_BOOT_GRACE_SEC", 120.0) or 0.0)
        boot_ts = _safe_float(km.get("boot_ts", now), now)
        uptime = max(0.0, now - boot_ts)

        min_data_samples = int(_cfg(bot, "KILL_MIN_DATA_SAMPLES_BEFORE_ENFORCE", 1) or 1)

        max_daily_loss_pct = float(_cfg(bot, "MAX_DAILY_LOSS_PCT", 0.0) or 0.0)
        max_drawdown_pct = float(_cfg(bot, "MAX_DRAWDOWN_PCT", 0.0) or 0.0)

        max_data_stale = float(_cfg(bot, "KILL_MAX_DATA_STALENESS_SEC", 150.0))
        max_error_rate = float(_cfg(bot, "KILL_MAX_API_ERROR_RATE", 0.35))
        max_error_burst = int(_cfg(bot, "KILL_MAX_API_ERROR_BURST", 12))
        min_req_window = int(_cfg(bot, "KILL_MIN_REQ_WINDOW", 10))

        min_equity = float(_cfg(bot, "KILL_MIN_EQUITY", 0.0) or 0.0)

        # ---------- basic equity sanity ----------
        eq = _equity(bot)
        if min_equity > 0 and eq > 0 and eq < min_equity:
            reason = f"EQUITY BELOW MIN — ${eq:,.0f} < ${min_equity:,.0f}"
            await request_halt(bot, cooldown, reason, "critical")
            km["last_check_ts"] = now
            return False, reason

        # ---------- daily loss ----------
        sod = _start_of_day_equity(bot)
        dpnl = _daily_pnl(bot)
        if max_daily_loss_pct > 0 and sod > 0:
            if dpnl < -max_daily_loss_pct * sod:
                reason = f"DAILY LOSS LIMIT — PnL ${dpnl:,.0f} (limit {max_daily_loss_pct:.1%})"
                await request_halt(bot, cooldown, reason, "critical")
                km["last_check_ts"] = now
                return False, reason

        # ---------- drawdown from peak ----------
        peak = _peak_equity(bot)
        if max_drawdown_pct > 0 and peak > 0 and eq > 0:
            dd = (peak - eq) / peak
            if dd > max_drawdown_pct:
                reason = f"DRAWDOWN LIMIT — DD {dd:.1%} (limit {max_drawdown_pct:.1%})"
                await request_halt(bot, cooldown, reason, "critical")
                km["last_check_ts"] = now
                return False, reason

        # ---------- data staleness ----------
        age = _data_age(bot)

        if _is_finite(age) and age < 999998.0:
            km["data_samples_ok"] = int(km.get("data_samples_ok", 0) or 0) + 1

        if data_boot_grace > 0 and uptime < data_boot_grace:
            km["last_check_ts"] = now
        else:
            if int(km.get("data_samples_ok", 0) or 0) >= max(1, min_data_samples):
                if max_data_stale > 0 and age > max_data_stale:
                    reason = f"DATA STALE — age {age:.0f}s > {max_data_stale:.0f}s"
                    await request_halt(bot, min(cooldown, 120.0), reason, "critical")
                    km["last_check_ts"] = now
                    return False, reason

        # ---------- exchange health stale (optional hook) ----------
        if _exchange_health_stale(bot):
            max_stale = float(_cfg(bot, "KILL_MAX_EX_HEALTH_STALE_SEC", 0.0) or 0.0)
            reason = f"EXCHANGE HEALTH STALE — no ping in > {max_stale:.0f}s"
            await request_halt(bot, min(cooldown, 180.0), reason, "critical")
            km["last_check_ts"] = now
            return False, reason

        # ---------- API error spiral (rate + burst) ----------
        req, err = _ex_counts(bot)

        last_req = int(km.get("last_ex_request_count", 0) or 0)
        last_err = int(km.get("last_ex_error_count", 0) or 0)

        dreq = max(0, req - last_req)
        derr = max(0, err - last_err)

        km["last_ex_request_count"] = req
        km["last_ex_error_count"] = err
        km["last_check_ts"] = now

        if derr >= max_error_burst:
            reason = f"API ERROR BURST — {derr} errors since last check"
            await request_halt(bot, min(cooldown, 180.0), reason, "critical")
            return False, reason

        if dreq >= min_req_window:
            rate = (derr / max(1, dreq))
            if rate >= max_error_rate:
                reason = f"API ERROR RATE — {rate:.0%} ({derr}/{dreq})"
                await request_halt(bot, min(cooldown, 180.0), reason, "critical")
                return False, reason

        # PASS
        km["last_eval_ok"] = True
        km["last_eval_reason"] = ""
        return True, None

    except Exception as e:
        fail_closed = _truthy(_cfg(bot, "KILL_SWITCH_FAIL_CLOSED_ON_ERROR", True))
        log_core.error(f"KILL SWITCH EVALUATE ERROR: {e} | fail_closed={fail_closed}")

        await _telemetry(
            bot,
            "kill_switch.evaluate_error",
            level="critical",
            data={
                "err": repr(e)[:400],
                "fail_closed": bool(fail_closed),
            },
        )

        latch_sec = float(_cfg(bot, "KILL_SWITCH_ERROR_LATCH_SEC", 30.0) or 0.0)

        try:
            _ensure_state_fields(bot.state)
            bot.state.kill_metrics["last_eval_ok"] = (not fail_closed)
            bot.state.kill_metrics["last_eval_reason"] = f"EVALUATE ERROR: {e}"
            bot.state.kill_metrics["last_check_ts"] = _now()
        except Exception:
            pass

        if fail_closed and latch_sec > 0:
            try:
                await request_halt(bot, float(latch_sec), f"EVALUATE ERROR LATCH — {e}", "critical")
            except Exception:
                pass

        return (False, f"EVALUATE ERROR: {e}") if fail_closed else (True, None)


async def tick_kill_switch(bot) -> None:
    """
    Optional watchdog tick. Safe to call from guardian every loop.
    It evaluates if needed and does nothing else unless it trips.
    """
    try:
        ok, _reason = await evaluate(bot)
        if not ok:
            return
    except Exception:
        return


async def trade_allowed(bot) -> bool:
    """
    CHEAP GATE — does NOT call evaluate().
    Your guardian loop should call tick_kill_switch() periodically.
    """
    try:
        _ensure_state_fields(bot.state)
        if not _truthy(_cfg(bot, "KILL_SWITCH_ENABLED", True)):
            return True
        return not is_halted(bot)
    except Exception:
        return False
