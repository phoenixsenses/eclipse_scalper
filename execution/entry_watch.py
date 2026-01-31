# execution/entry_watch.py — SCALPER ETERNAL — ENTRY WATCH TOWER — 2026 v1.2 (DIAGNOSTIC-AWARE + NO SILENT FALLBACK)
# Patch vs v1.1:
# - ✅ Wires diagnostics helper into optional imports (logs what's missing instead of silent None)
# - ✅ Adds tiny helper we_dont_have_this() (per your request) + optional diag dump hook
# - ✅ Adds one-time “ENTRY_WATCH ONLINE” banner (no spam)
# - ✅ Zero behavior change to watch logic (only observability)

import time
import asyncio
from typing import Optional, Any

from utils.logging import log_entry
from execution.order_router import create_order, cancel_order

# ─────────────────────────────────────────────────────────────────────
# Diagnostics wiring (never fatal, never changes behavior)
# ─────────────────────────────────────────────────────────────────────

_DIAG_ONCE = False


def we_dont_have_this() -> None:
    # Tiny function you asked for (literal string)
    print("wedon't have this")


def _optional_import(module: str, attr: Optional[str] = None):
    """
    Best-effort optional import that logs explicitly what's missing.
    Returns:
      - module object if attr is None
      - attribute (callable or value) if attr is provided
      - None if missing/unavailable
    """
    try:
        mod = __import__(module, fromlist=[attr] if attr else [])
        if attr is None:
            return mod
        return getattr(mod, attr)
    except Exception:
        what = f"{module}.{attr}" if attr else module
        log_entry.warning(f"OPTIONAL MISSING — {what}")
        return None


# Optional: diagnostics dump (if available)
_print_diagnostics = _optional_import("execution.diagnostics", "print_diagnostics")


def _diag_dump(bot, note: str) -> None:
    """
    Never raises. Never mutates.
    Call only when you *really* need it (error path / debug), to avoid spam.
    """
    try:
        if callable(_print_diagnostics):
            log_entry.warning(f"DIAG DUMP — {note}")
            _print_diagnostics(bot)
    except Exception:
        pass


def _banner_once() -> None:
    global _DIAG_ONCE
    if _DIAG_ONCE:
        return
    _DIAG_ONCE = True
    log_entry.info("ENTRY_WATCH ONLINE — watching pending entries (idempotent)")

# ─────────────────────────────────────────────────────────────────────
# Core helpers
# ─────────────────────────────────────────────────────────────────────


def _now() -> float:
    return time.time()


def _symkey(sym: str) -> str:
    s = (sym or "").upper().strip()
    s = s.replace("/USDT:USDT", "USDT").replace("/USDT", "USDT")
    s = s.replace(":USDT", "USDT").replace(":", "")
    s = s.replace("/", "")
    if s.endswith("USDTUSDT"):
        s = s[:-4]
    return s


def _safe_float(x, default=0.0) -> float:
    try:
        v = float(x)
        if v != v:
            return default
        return v
    except Exception:
        return default


def _truthy(x) -> bool:
    if x is True:
        return True
    if isinstance(x, (int, float)) and x != 0:
        return True
    if isinstance(x, str) and x.strip().lower() in ("1", "true", "yes", "y", "on", "t"):
        return True
    return False


def _cfg(bot, name: str, default):
    cfg = getattr(bot, "cfg", None)
    return getattr(cfg, name, default) if cfg is not None else default


def _ensure_state_fields(state) -> None:
    if not hasattr(state, "entry_watches") or not isinstance(getattr(state, "entry_watches", None), dict):
        state.entry_watches = {}  # k -> watch dict
    # optional: global lock to prevent concurrent polls
    if not hasattr(state, "_entry_watch_lock") or not isinstance(getattr(state, "_entry_watch_lock", None), asyncio.Lock):
        state._entry_watch_lock = asyncio.Lock()


def _resolve_raw_symbol(bot, k: str, fallback: str) -> str:
    try:
        data = getattr(bot, "data", None)
        raw_map = getattr(data, "raw_symbol", {}) if data is not None else {}
        if isinstance(raw_map, dict) and raw_map.get(k):
            return str(raw_map[k])
    except Exception:
        pass
    return fallback


def _order_filled(order: dict) -> float:
    if not isinstance(order, dict):
        return 0.0
    filled = _safe_float(order.get("filled"), 0.0)
    if filled > 0:
        return filled
    info = order.get("info") or {}
    return _safe_float(info.get("executedQty"), 0.0)


def _order_status(order: dict) -> str:
    if not isinstance(order, dict):
        return ""
    st = str(order.get("status") or "")
    if st:
        return st.lower()
    info = order.get("info") or {}
    st2 = info.get("status") or info.get("orderStatus") or ""
    return str(st2 or "").lower()


def _bad_order_id(oid: Any) -> bool:
    if oid is None:
        return True
    s = str(oid).strip()
    if not s:
        return True
    if s.upper() in {"DRY_RUN_ORDER", "NONE", "NULL"}:
        return True
    return False


async def _fetch_order_best_effort(bot, order_id: str, sym_raw: str) -> Optional[dict]:
    """
    Wrapper-friendly:
    - try fetch_order(id, symbol)
    - fallback fetch_open_orders(symbol) + scan
    - fallback fetch_closed_orders(symbol) + scan (if available)
    """
    ex = getattr(bot, "ex", None)
    if ex is None or _bad_order_id(order_id):
        return None

    fo = getattr(ex, "fetch_order", None)
    if callable(fo):
        try:
            return await fo(order_id, sym_raw)
        except Exception:
            pass

    foo = getattr(ex, "fetch_open_orders", None)
    if callable(foo):
        try:
            oo = await foo(sym_raw)
            if isinstance(oo, list):
                for o in oo:
                    if isinstance(o, dict) and str(o.get("id") or "") == str(order_id):
                        return o
        except Exception:
            pass

    fco = getattr(ex, "fetch_closed_orders", None)
    if callable(fco):
        try:
            co = await fco(sym_raw)
            if isinstance(co, list):
                for o in co:
                    if isinstance(o, dict) and str(o.get("id") or "") == str(order_id):
                        return o
        except Exception:
            pass

    return None


def _has_position_best_effort(bot, k: str) -> bool:
    """
    Prefer exchange truth if available; fallback to state.
    """
    try:
        state_pos = getattr(bot.state, "positions", {}) or {}
        if isinstance(state_pos, dict) and k in state_pos:
            p = state_pos.get(k)
            try:
                sz = float(getattr(p, "size", 0.0))
                if abs(sz) > 0.0:
                    return True
            except Exception:
                return True
    except Exception:
        pass

    try:
        ex = getattr(bot, "ex", None)
        pos_cache = getattr(ex, "positions", None)
        if isinstance(pos_cache, dict) and k in pos_cache:
            p = pos_cache.get(k)
            try:
                sz = p.get("contracts") if isinstance(p, dict) else getattr(p, "contracts", None)
                if sz is None:
                    sz = p.get("size") if isinstance(p, dict) else getattr(p, "size", None)
                if sz is None:
                    return True
                return abs(float(sz)) > 0.0
            except Exception:
                return True
    except Exception:
        pass

    return False


async def register_entry_watch(
    bot,
    *,
    symbol: str,
    order_id: str,
    side: str,
    requested_qty: float,
    created_ts: Optional[float] = None,
    replace_price: Optional[float] = None,
    order_type: str = "LIMIT",
    time_in_force: str = "GTC",
) -> None:
    """
    Call this right after you place an ENTRY order you want watched.
    symbol may be raw or canonical.
    """
    try:
        _banner_once()
        _ensure_state_fields(bot.state)
        k = _symkey(symbol)

        if _bad_order_id(order_id) or float(requested_qty or 0.0) <= 0:
            return

        sym_raw = _resolve_raw_symbol(bot, k, str(symbol))

        bot.state.entry_watches[k] = {
            "k": k,
            "symbol_any": str(symbol),
            "symbol_raw": str(sym_raw),
            "order_id": str(order_id),
            "side": str(side or "").lower(),
            "requested_qty": float(requested_qty or 0.0),
            "filled_qty": 0.0,
            "created_ts": float(created_ts or _now()),
            "last_check_ts": 0.0,
            "replace_price": float(replace_price) if replace_price is not None else None,
            "replaced_count": 0,
            "market_converted": False,
            "done": False,
            "reason": "",
            "order_type": str(order_type or "LIMIT").upper(),
            "time_in_force": str(time_in_force or "GTC").upper(),
            "busy": False,
        }
    except Exception:
        pass


async def clear_entry_watch(bot, symbol: str, note: str = "") -> None:
    try:
        _ensure_state_fields(bot.state)
        k = _symkey(symbol)
        bot.state.entry_watches.pop(k, None)
        if note:
            log_entry.info(f"ENTRY_WATCH CLEAR {k} — {note}")
    except Exception:
        pass


async def on_halt(bot) -> None:
    """
    Optional hook: when halted, we DO NOT place new market conversions or replaces.
    We still *may* clear aged watches safely.
    """
    return


async def poll_entry_watches(bot) -> None:
    """
    Call this every ~5-15 seconds from guardian_loop.
    Must be idempotent and safe even if called twice.
    """
    _banner_once()
    _ensure_state_fields(bot.state)

    lock = getattr(bot.state, "_entry_watch_lock", None)
    if not isinstance(lock, asyncio.Lock):
        lock = asyncio.Lock()
        try:
            bot.state._entry_watch_lock = lock
        except Exception:
            pass

    watches = getattr(bot.state, "entry_watches", {}) or {}
    if not isinstance(watches, dict) or not watches:
        return

    async with lock:
        watches = getattr(bot.state, "entry_watches", {}) or {}
        if not isinstance(watches, dict) or not watches:
            return

        # Tunables
        check_every = float(_cfg(bot, "ENTRY_WATCH_POLL_SEC", 10.0))
        max_age = float(_cfg(bot, "ENTRY_WATCH_MAX_AGE_SEC", 90.0))
        replace_after = float(_cfg(bot, "ENTRY_WATCH_REPLACE_AFTER_SEC", 25.0))
        max_replaces = int(_cfg(bot, "ENTRY_WATCH_MAX_REPLACES", 2))
        convert_to_market = _truthy(_cfg(bot, "ENTRY_WATCH_CONVERT_TO_MARKET", True))
        min_fill_ratio = float(_cfg(bot, "MIN_FILL_RATIO", 0.80))
        cancel_on_timeout = _truthy(_cfg(bot, "ENTRY_WATCH_CANCEL_ON_TIMEOUT", True))

        min_remaining_qty = float(_cfg(bot, "ENTRY_WATCH_MIN_REMAINING_QTY", 0.0))  # 0 disables

        halted = False
        try:
            from risk.kill_switch import is_halted as _ih  # type: ignore
            halted = bool(_ih(bot))
        except Exception:
            # Explicitly log missing kill switch (observability only)
            log_entry.warning("OPTIONAL MISSING — risk.kill_switch.is_halted")
            halted = False

        now = _now()

        for k, w in list(watches.items()):
            try:
                if not isinstance(w, dict):
                    continue
                if _truthy(w.get("done")):
                    continue
                if _truthy(w.get("busy")):
                    continue

                last_check = float(w.get("last_check_ts") or 0.0)
                if now - last_check < check_every:
                    continue

                w["last_check_ts"] = now
                w["busy"] = True

                order_id = w.get("order_id")
                if _bad_order_id(order_id):
                    await clear_entry_watch(bot, k, "bad order id (dry-run/none)")
                    continue

                requested = float(w.get("requested_qty") or 0.0)
                if requested <= 0:
                    await clear_entry_watch(bot, k, "invalid requested_qty")
                    continue

                side = str(w.get("side") or "").lower()
                created = float(w.get("created_ts") or now)
                watch_type = str(w.get("order_type") or "LIMIT").upper()
                tif = str(w.get("time_in_force") or "GTC").upper()

                sym_raw = str(w.get("symbol_raw") or _resolve_raw_symbol(bot, k, str(w.get("symbol_any") or k)))
                w["symbol_raw"] = sym_raw

                order = await _fetch_order_best_effort(bot, str(order_id), sym_raw)
                if not isinstance(order, dict):
                    if now - created > max_age:
                        await clear_entry_watch(bot, k, "order not fetchable, aged out")
                    continue

                filled = _order_filled(order)
                status = _order_status(order)
                w["filled_qty"] = float(filled)

                if _has_position_best_effort(bot, k):
                    await clear_entry_watch(bot, k, "position exists (entry succeeded)")
                    continue

                if status in ("closed", "canceled", "cancelled", "rejected", "expired"):
                    await clear_entry_watch(bot, k, f"order ended ({status})")
                    continue

                age = now - created
                remaining_qty = float(max(0.0, requested - filled))

                if min_remaining_qty > 0 and remaining_qty < min_remaining_qty:
                    await clear_entry_watch(bot, k, f"remaining {remaining_qty:.6f} < min_remaining")
                    continue

                if filled > 0 and requested > 0:
                    ratio = filled / max(1e-12, requested)
                    if ratio < min_fill_ratio and age > replace_after:
                        await cancel_order(bot, str(order_id), sym_raw)
                        await clear_entry_watch(bot, k, f"partial too low ({ratio:.1%}) remainder canceled")
                        continue

                if age > max_age:
                    if cancel_on_timeout:
                        await cancel_order(bot, str(order_id), sym_raw)
                        await clear_entry_watch(bot, k, f"stuck > {max_age:.0f}s (canceled)")
                    else:
                        await clear_entry_watch(bot, k, f"stuck > {max_age:.0f}s (ignored)")
                    continue

                if halted:
                    continue

                if watch_type == "LIMIT" and age > replace_after and int(w.get("replaced_count") or 0) < max_replaces:
                    rp = w.get("replace_price", None)
                    if rp is not None and remaining_qty > 0:
                        await cancel_order(bot, str(order_id), sym_raw)

                        entry_side = "buy" if side == "long" else "sell"
                        new_order = await create_order(
                            bot,
                            symbol=sym_raw,
                            type="LIMIT",
                            side=entry_side,
                            amount=float(remaining_qty),
                            price=float(rp),
                            params={"timeInForce": tif},
                            intent_reduce_only=False,
                            retries=6,
                        )
                        if isinstance(new_order, dict) and new_order.get("id"):
                            w["order_id"] = str(new_order["id"])
                            w["replaced_count"] = int(w.get("replaced_count") or 0) + 1
                            w["created_ts"] = now
                            log_entry.info(f"ENTRY_WATCH {k} — cancel/replace LIMIT @ {rp} (#{w['replaced_count']})")
                        else:
                            log_entry.warning(f"ENTRY_WATCH {k} — replace failed, will keep watching")
                        continue

                if (
                    convert_to_market
                    and watch_type == "LIMIT"
                    and (not _truthy(w.get("market_converted")))
                    and age > (max_age * 0.8)
                ):
                    await cancel_order(bot, str(order_id), sym_raw)

                    if remaining_qty > 0 and (min_remaining_qty <= 0 or remaining_qty >= min_remaining_qty):
                        entry_side = "buy" if side == "long" else "sell"
                        mo = await create_order(
                            bot,
                            symbol=sym_raw,
                            type="MARKET",
                            side=entry_side,
                            amount=float(remaining_qty),
                            price=None,
                            params={},
                            intent_reduce_only=False,
                            retries=6,
                        )
                        if isinstance(mo, dict) and mo.get("id"):
                            w["market_converted"] = True
                            log_entry.warning(
                                f"ENTRY_WATCH {k} — converted remainder to MARKET (remaining={remaining_qty:.6f})"
                            )

                    await clear_entry_watch(bot, k, "converted-to-market or canceled")
                    continue

            except asyncio.CancelledError:
                raise
            except Exception as e:
                log_entry.error(f"ENTRY_WATCH error {k}: {e}")
                # Optional: uncomment if you want auto diag dumps on entry_watch exceptions
                # _diag_dump(bot, f"entry_watch exception {k}: {e}")
                continue
            finally:
                try:
                    if isinstance(w, dict):
                        w["busy"] = False
                except Exception:
                    pass
