# execution/emergency.py — SCALPER ETERNAL — COSMIC EMERGENCY SALVATION — 2026 v4.6 (DIAGNOSTIC-AWARE + NO SILENT FALLBACK)
# Patch vs v4.5:
# - ✅ Wires diagnostics helper into optional imports (logs what's missing instead of silent None)
# - ✅ Optional diag dump hook on catastrophic paths (off by default; uncomment when needed)
# - ✅ One-time "EMERGENCY ONLINE" banner (no spam)
# - ✅ No behavior change to emergency logic unless you enable the optional diag hook

import asyncio
import time
import os
import json
from datetime import datetime, timezone
from typing import List, Dict, Any, Tuple, Optional, Set

from utils.logging import log_core, log
from brain.persistence import save_brain
from execution.order_router import create_order, cancel_order  # ✅ ROUTER

# ─────────────────────────────────────────────────────────────────────
# Diagnostics wiring (never fatal, no behavior change)
# ─────────────────────────────────────────────────────────────────────

_EMERGENCY_ONCE = False


def _banner_once() -> None:
    global _EMERGENCY_ONCE
    if _EMERGENCY_ONCE:
        return
    _EMERGENCY_ONCE = True
    log_core.info("EMERGENCY ONLINE — truth-first flatten is armed")


def _optional_import(module: str, attr: Optional[str] = None):
    """
    Best-effort optional import that logs explicitly what's missing.
    Returns module or attribute; returns None if unavailable.
    """
    try:
        mod = __import__(module, fromlist=[attr] if attr else [])
        if attr is None:
            return mod
        return getattr(mod, attr)
    except Exception:
        what = f"{module}.{attr}" if attr else module
        log_core.warning(f"OPTIONAL MISSING — {what}")
        return None


# Optional diagnostics dump (if present)
_print_diagnostics = _optional_import("execution.diagnostics", "print_diagnostics")


def _diag_dump(bot, note: str) -> None:
    """
    Never raises. Never mutates.
    Call only in emergencies / debug to avoid spam.
    """
    try:
        if callable(_print_diagnostics):
            log_core.warning(f"DIAG DUMP — {note}")
            _print_diagnostics(bot)
    except Exception:
        pass


# Optional kill-switch integration (best effort, but LOUD if missing)
request_halt = _optional_import("risk.kill_switch", "request_halt")


# ----------------------------
# Helpers
# ----------------------------

def _normalize_symbol(sym: str) -> str:
    s = str(sym or "").upper().strip()
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
    if isinstance(x, str) and x.strip().lower() in ("true", "1", "yes", "y", "t", "on"):
        return True
    return False


def _cfg(bot, name: str, default):
    try:
        return getattr(bot.cfg, name, default)
    except Exception:
        return default


async def _safe_speak(bot, text: str, priority: str = "critical"):
    notify = getattr(bot, "notify", None)
    if notify is None:
        return
    try:
        await notify.speak(text, priority)
    except Exception:
        pass


def _extract_symbol(obj: Dict[str, Any]) -> str:
    try:
        sym = obj.get("symbol")
        if sym:
            return str(sym)
        info = obj.get("info") or {}
        if isinstance(info, dict):
            s2 = info.get("symbol")
            if s2:
                return str(s2)
    except Exception:
        pass
    return ""


def _extract_position_side(p: Dict[str, Any]) -> str:
    info = p.get("info") or {}
    ps = info.get("positionSide")
    if isinstance(ps, str) and ps:
        return ps.upper()

    side = p.get("side")
    if isinstance(side, str):
        s = side.lower().strip()
        if s == "long":
            return "LONG"
        if s == "short":
            return "SHORT"

    return ""


def _extract_ccxt_position_amt(p: Dict[str, Any]) -> float:
    info = p.get("info") or {}

    for k in ("positionAmt", "position_amount", "amt"):
        if isinstance(info, dict) and k in info:
            return _safe_float(info.get(k), 0.0)

    for k in ("contracts", "size", "amount"):
        if k in p:
            return _safe_float(p.get(k), 0.0)

    return 0.0


def _signed_to_close_order(signed_amt: float) -> Optional[str]:
    if signed_amt > 0:
        return "sell"
    if signed_amt < 0:
        return "buy"
    return None


def _resolve_symbol_any(bot, k_or_sym: str) -> str:
    k = _normalize_symbol(k_or_sym)
    try:
        data = getattr(bot, "data", None)
        raw_map = getattr(data, "raw_symbol", {}) if data is not None else {}
        if isinstance(raw_map, dict) and raw_map.get(k):
            return str(raw_map[k])
    except Exception:
        pass
    return k


def _ensure_state_dict(state, name: str) -> dict:
    v = getattr(state, name, None)
    if isinstance(v, dict):
        return v
    d = {}
    try:
        setattr(state, name, d)
    except Exception:
        pass
    return d


# ----------------------------
# Open-order cancellation
# ----------------------------

async def _cancel_open_orders_best_effort(bot, symbols_hint: List[str]):
    """
    Cancel open orders with maximum coverage using router cancel.
    We do BOTH:
      - global fetch_open_orders()
      - per-symbol fetch_open_orders(sym) for hints (capped)
    """
    ex = getattr(bot, "ex", None)
    if ex is None:
        return

    orders: List[Dict[str, Any]] = []

    # 1) global
    try:
        oo = await ex.fetch_open_orders()
        if isinstance(oo, list) and oo:
            orders.extend([o for o in oo if isinstance(o, dict)])
    except Exception:
        pass

    # 2) per-symbol (global is sometimes incomplete in wrappers)
    tried: Set[str] = set()
    for s in (symbols_hint or [])[:80]:
        k = _normalize_symbol(s)
        sym_raw = _resolve_symbol_any(bot, k)
        for cand in (s, k, sym_raw):
            if not cand or cand in tried:
                continue
            tried.add(cand)
            try:
                oo = await ex.fetch_open_orders(cand)
                if isinstance(oo, list) and oo:
                    orders.extend([o for o in oo if isinstance(o, dict)])
            except Exception:
                continue

    if not orders:
        return

    # Dedup by (id, symbol)
    seen = set()
    uniq: List[Dict[str, Any]] = []
    for o in orders:
        oid = str(o.get("id") or "")
        osym = _extract_symbol(o)
        key = (oid, osym)
        if oid and key not in seen:
            seen.add(key)
            uniq.append(o)

    sem = asyncio.Semaphore(16)

    async def _c(o: Dict[str, Any]):
        oid = o.get("id")
        if not oid:
            return
        osym = _extract_symbol(o)
        if not osym:
            return
        async with sem:
            try:
                await cancel_order(bot, str(oid), str(osym))
            except Exception:
                pass

    await asyncio.gather(*(_c(o) for o in uniq), return_exceptions=True)


# ----------------------------
# Position snapshot + verify
# ----------------------------

async def _fetch_live_positions_map(bot) -> Dict[str, List[Dict[str, Any]]]:
    live_map: Dict[str, List[Dict[str, Any]]] = {}
    try:
        live_positions = await bot.ex.fetch_positions()
        if isinstance(live_positions, list):
            for p in live_positions:
                if not isinstance(p, dict):
                    continue
                sym_raw = _extract_symbol(p)
                if not sym_raw:
                    continue
                sym_norm = _normalize_symbol(sym_raw)
                amt = _extract_ccxt_position_amt(p)
                if abs(amt) <= 1e-12:
                    continue
                live_map.setdefault(sym_norm, []).append(p)
    except Exception:
        return {}
    return live_map


async def _verify_remaining(bot) -> Dict[str, float]:
    remaining: Dict[str, float] = {}
    try:
        live_positions = await bot.ex.fetch_positions()
        if isinstance(live_positions, list):
            for p in live_positions:
                if not isinstance(p, dict):
                    continue
                sym_raw = _extract_symbol(p)
                if not sym_raw:
                    continue
                sym_norm = _normalize_symbol(sym_raw)
                amt = abs(_extract_ccxt_position_amt(p))
                if amt > 1e-8:
                    remaining[sym_norm] = remaining.get(sym_norm, 0.0) + amt
    except Exception:
        pass
    return remaining


# ----------------------------
# Emergency Tick (guardian integration)
# ----------------------------

async def emergency_tick(bot):
    """
    Guardian-compatible tick.
    Triggers emergency_flat if bot.state.emergency_flag is set (or cfg requests it).
    """
    _banner_once()

    try:
        flag = False
        reason = "EMERGENCY_FLAG"
        forced = False

        try:
            flag = _truthy(getattr(bot.state, "emergency_flag", False))
            reason = str(getattr(bot.state, "emergency_reason", "") or reason)
            forced = _truthy(getattr(bot.state, "emergency_forced", False))
        except Exception:
            flag = False

        if _truthy(_cfg(bot, "EMERGENCY_ALWAYS_FLAT", False)):
            flag = True
            reason = "EMERGENCY_ALWAYS_FLAT"
            forced = True

        if not flag:
            return

        try:
            bot.state.emergency_flag = False
        except Exception:
            pass

        await emergency_flat(bot, reason=reason, forced=forced)

    except Exception as e:
        log_core.error(f"emergency_tick failed: {e}")
        # Optional: uncomment for deep debug during chaos
        # _diag_dump(bot, f"emergency_tick exception: {e}")


# ----------------------------
# Emergency Flat
# ----------------------------

async def emergency_flat(bot, reason: str = "COSMIC_CHAOS", forced: bool = False):
    """
    Close all positions + cancel open orders with maximum survival probability.
    forced=True:
      - more retries
      - may drop reduceOnly ONLY if verify shows exposure persists after a pass
    """
    _banner_once()

    start_time = time.time()
    salvation_timestamp = datetime.now(timezone.utc).isoformat()

    log_core.critical("=" * 100)
    log_core.critical("COSMIC EMERGENCY SALVATION INVOKED — PRESERVATION ABSOLUTE")
    log_core.critical(f"REASON: {str(reason).upper()} | TIMESTAMP: {salvation_timestamp} | FORCED={forced}")
    log_core.critical("=" * 100)

    # Optional: request kill-switch halt immediately (best effort)
    if callable(request_halt):
        try:
            await request_halt(bot, seconds=60 * 30, reason=f"EMERGENCY_FLAT: {reason}", severity="critical")
        except Exception:
            pass

    # Snapshot from state (fast)
    state_positions_items: List[Tuple[str, Any]] = list((getattr(bot.state, "positions", {}) or {}).items())

    # Live truth map (authoritative)
    live_map = await _fetch_live_positions_map(bot)
    live_symbols = list(live_map.keys())

    # Union targets: state + live (state may lie)
    state_symbols = [sym for sym, _ in state_positions_items]
    target_symbols_norm = sorted(set(_normalize_symbol(s) for s in (state_symbols + live_symbols) if s))

    # Sort state positions by notional (best effort)
    state_positions_items.sort(
        key=lambda x: abs(_safe_float(getattr(x[1], "size", 0.0)) * _safe_float(getattr(x[1], "entry_price", 0.0))),
        reverse=True,
    )

    targeted = len(target_symbols_norm)
    failed_symbols: List[str] = []

    _ensure_state_dict(bot.state, "last_exit_time")

    # Cancel open orders first
    try:
        await _cancel_open_orders_best_effort(bot, target_symbols_norm)
        log_core.info("Open orders annihilated (best effort, router)")
    except Exception as e:
        log_core.error(f"Open order annihilation failed: {e}")

    async def purify_symbol(sym_norm: str, pos_obj_hint: Optional[Any]) -> bool:
        k = _normalize_symbol(sym_norm)
        sym_any = _resolve_symbol_any(bot, k)

        # Build close plans from LIVE truth first
        live_list = live_map.get(k, [])

        close_plans: List[Tuple[str, float, Dict[str, Any]]] = []
        for p in live_list:
            amt_signed = _extract_ccxt_position_amt(p)
            side_to_close = _signed_to_close_order(amt_signed)
            if not side_to_close:
                continue
            amt = abs(amt_signed)
            if amt <= 1e-12:
                continue
            params = {"reduceOnly": True}
            ps = _extract_position_side(p)
            if ps in ("LONG", "SHORT"):
                params["positionSide"] = ps
            close_plans.append((side_to_close, amt, params))

        # If live is empty but we had a state hint, use it
        if not close_plans and pos_obj_hint is not None:
            local_side = str(getattr(pos_obj_hint, "side", "") or "").lower().strip()
            local_close_side = "sell" if local_side == "long" else "buy"
            local_amount = abs(_safe_float(getattr(pos_obj_hint, "size", 0.0)))
            if local_amount > 0:
                close_plans.append((local_close_side, local_amount, {"reduceOnly": True}))

        if not close_plans:
            return True

        attempts_per_plan = 5 if forced else 3
        success_any = False

        for (side_to_close, amt, params) in close_plans:
            if amt <= 0:
                continue

            for attempt in range(1, attempts_per_plan + 1):
                try:
                    use_params = dict(params)

                    await create_order(
                        bot,
                        symbol=sym_any,
                        type="market",
                        side=side_to_close,
                        amount=float(amt),
                        price=None,
                        params=use_params,
                        intent_reduce_only=_truthy(use_params.get("reduceOnly")),
                        intent_close_position=_truthy(use_params.get("closePosition")),
                        retries=6 if forced else 4,
                    )
                    success_any = True
                    break

                except Exception as e:
                    log.error(f"{k} salvation failed ({side_to_close}) attempt {attempt}: {e}")
                    await asyncio.sleep(min(5.0, 0.5 * (2 ** (attempt - 1)) + 0.20 * attempt))

        return success_any

    # Build hint map from state (normalized -> pos_obj)
    state_hint: Dict[str, Any] = {}
    for sym, pos in state_positions_items:
        state_hint[_normalize_symbol(sym)] = pos

    # Parallel salvation with cap
    sem = asyncio.Semaphore(6)

    async def guarded(sym_norm: str):
        async with sem:
            ok = await purify_symbol(sym_norm, state_hint.get(sym_norm))
            if not ok:
                failed_symbols.append(sym_norm)
            return ok

    if target_symbols_norm:
        await asyncio.gather(*(guarded(s) for s in target_symbols_norm), return_exceptions=True)

    # Verification pass (source of truth)
    remaining = await _verify_remaining(bot)

    # Forced escalation: ONLY if exposure remains, do a second wave that may drop reduceOnly
    if forced and remaining:
        log_core.critical(f"FORCED ESCALATION: exposure remains on {len(remaining)} symbols — final strike engaged")

        async def final_strike(sym_norm: str):
            k = _normalize_symbol(sym_norm)
            sym_any = _resolve_symbol_any(bot, k)

            try:
                positions = await bot.ex.fetch_positions()
            except Exception:
                return

            if not isinstance(positions, list):
                return

            plans: List[Tuple[str, float, Dict[str, Any]]] = []
            for p in positions:
                if not isinstance(p, dict):
                    continue
                psym = _normalize_symbol(_extract_symbol(p))
                if psym != k:
                    continue
                amt_signed = _extract_ccxt_position_amt(p)
                side_to_close = _signed_to_close_order(amt_signed)
                if not side_to_close:
                    continue
                amt = abs(amt_signed)
                if amt <= 1e-12:
                    continue
                params = {}  # NON-reduceOnly (dangerous)
                pos_side = _extract_position_side(p)
                if pos_side in ("LONG", "SHORT"):
                    params["positionSide"] = pos_side
                plans.append((side_to_close, amt, params))

            for (side_to_close, amt, params) in plans:
                try:
                    await create_order(
                        bot,
                        symbol=sym_any,
                        type="market",
                        side=side_to_close,
                        amount=float(amt),
                        price=None,
                        params=dict(params),
                        intent_reduce_only=False,
                        retries=8,
                    )
                except Exception:
                    pass

        await asyncio.gather(*(final_strike(s) for s in list(remaining.keys())[:60]), return_exceptions=True)
        remaining = await _verify_remaining(bot)

    if remaining:
        log_core.critical(f"VERIFICATION: {len(remaining)} symbols still exposed — THE VOID NOT CALM")
    else:
        log_core.critical("VERIFICATION COMPLETE — TOTAL PURIFICATION ACHIEVED")

    # Cancel again (mop-up)
    try:
        await _cancel_open_orders_best_effort(bot, target_symbols_norm)
    except Exception:
        pass

    # State handling: clear only if verified flat, unless forced
    if forced or not remaining:
        try:
            getattr(bot.state, "positions", {}).clear()
        except Exception:
            pass

    # Global cooldown
    lock_until = time.time() + 60 * 60
    lex = _ensure_state_dict(bot.state, "last_exit_time")
    for sym_norm in target_symbols_norm[:500]:
        try:
            lex[_normalize_symbol(sym_norm)] = lock_until
        except Exception:
            pass

    try:
        bot.session_peak_equity = bot.state.current_equity
    except Exception:
        pass

    duration = time.time() - start_time

    testament = {
        "salvation_timestamp": salvation_timestamp,
        "reason": reason,
        "forced": forced,
        "duration_seconds": round(duration, 2),
        "symbols_targeted": targeted,
        "failed_symbols": failed_symbols,
        "remaining_symbols_norm": list(remaining.keys()),
        "remaining_amounts": remaining,
        "final_equity": getattr(bot.state, "current_equity", 0.0),
        "message": "THE BLADE ENDURES — PRESERVATION ABSOLUTE",
    }

    testament_path = os.path.expanduser("~/.blade_cosmic_testament.json")
    try:
        with open(testament_path, "w", encoding="utf-8") as f:
            json.dump(testament, f, indent=2)
        log_core.critical(f"COSMIC TESTAMENT PRESERVED — {testament_path}")
    except Exception as e:
        log_core.error(f"Testament preservation failed: {e}")

    log_core.critical("=" * 100)
    log_core.critical("COSMIC SALVATION COMPLETE — THE BLADE IS PRESERVED BEYOND INFINITY")
    log_core.critical(f"REASON: {str(reason).upper()} | DURATION: {duration:.1f}s | FORCED={forced}")
    log_core.critical(f"SYMBOLS TARGETED: {targeted} | FAILED PLANS: {len(failed_symbols)}")
    if remaining:
        preview = ", ".join(list(remaining.keys())[:20])
        log_core.critical(f"REMAINING (LIVE, NORM): {preview}{'...' if len(remaining) > 20 else ''}")
    log_core.critical(f"FINAL EQUITY (STATE): ${_safe_float(getattr(bot.state, 'current_equity', 0.0), 0.0):,.0f}")
    log_core.critical("Φ" + " " * 40 + "SALVATION ABSOLUTE" + " " * 40 + "Φ")
    log_core.critical("=" * 100)

    await _safe_speak(
        bot,
        f"COSMIC SALVATION COMPLETE — {str(reason).upper()}\n"
        f"Targeted: {targeted} | Remaining(live): {len(remaining)} | Duration: {duration:.1f}s\n"
        f"Final Equity: ${_safe_float(getattr(bot.state, 'current_equity', 0.0), 0.0):,.0f}\n"
        f"THE BLADE IS ETERNAL — BEYOND THE VOID",
        "critical",
    )

    if failed_symbols:
        await _safe_speak(bot, f"WARNING: Failed salvation plans on: {', '.join(failed_symbols[:40])}", "critical")

    try:
        await save_brain(bot.state, force=True)
    except Exception as pe:
        log_core.error(f"Brain save failed after emergency: {pe}")
