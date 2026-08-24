# execution/bot_factory.py — SCALPER ETERNAL — BOT FACTORY — 2026 v1.0
# Purpose:
# - Allows execution/bootstrap.py to swap its SimpleNamespace bot into your real bot.core.EclipseEternal instance.
# - Keeps bootstrap as the loop orchestrator (guardian/data_loop/entry_loop), while providing a richer bot object.
#
# Guarantees:
# - Never fatal: if import or wiring fails, returns the original bot unchanged.
# - Preserves bootstrap critical fields: cfg, ex/exchange, state, _shutdown, data_ready, active_symbols.
# - Avoids starting core internal tasks (watcher/cache_saver) by default.

from __future__ import annotations

from typing import Any


def _safe_set(obj: Any, name: str, value: Any) -> None:
    try:
        setattr(obj, name, value)
    except Exception:
        pass


def _safe_get(obj: Any, name: str, default: Any = None) -> Any:
    try:
        return getattr(obj, name, default)
    except Exception:
        return default


def build_bot(seed_bot: Any) -> Any:
    """
    seed_bot is the lightweight bot created by execution/bootstrap.py.
    We return an EclipseEternal instance with seed fields grafted in.
    If anything goes wrong, return seed_bot unchanged.
    """
    try:
        from bot.core import EclipseEternal  # your real bot class
    except Exception:
        return seed_bot

    try:
        core = EclipseEternal()
    except Exception:
        return seed_bot

    # ----------------------------
    # Preserve critical bootstrap wiring
    # ----------------------------
    cfg = _safe_get(seed_bot, "cfg", None)
    state = _safe_get(seed_bot, "state", None)

    ex = _safe_get(seed_bot, "ex", None) or _safe_get(seed_bot, "exchange", None)
    exchange = _safe_get(seed_bot, "exchange", None) or ex

    shutdown_ev = _safe_get(seed_bot, "_shutdown", None)
    data_ready_ev = _safe_get(seed_bot, "data_ready", None)

    # In bootstrap, ACTIVE_SYMBOLS are bridged into bot.active_symbols (set)
    active_symbols = _safe_get(seed_bot, "active_symbols", None)

    # Data + notify may already be built by bootstrap; keep them if present
    data_obj = _safe_get(seed_bot, "data", None)
    notify_obj = _safe_get(seed_bot, "notify", None)

    # ----------------------------
    # Apply onto core
    # ----------------------------
    if cfg is not None:
        _safe_set(core, "cfg", cfg)

    if state is not None:
        _safe_set(core, "state", state)

    if ex is not None:
        _safe_set(core, "ex", ex)
        _safe_set(core, "exchange", exchange if exchange is not None else ex)

    # Ensure events used by guardian/data_loop/entry_loop exist and match bootstrap
    if shutdown_ev is not None:
        _safe_set(core, "_shutdown", shutdown_ev)
    else:
        # core already has one; leave it
        pass

    if data_ready_ev is not None:
        _safe_set(core, "data_ready", data_ready_ev)
    else:
        # bootstrap expects it; create if missing
        if _safe_get(core, "data_ready", None) is None:
            try:
                import asyncio
                _safe_set(core, "data_ready", asyncio.Event())
            except Exception:
                pass

    # Preserve bootstrap data/notify if they exist (bootstrap may have built DataCache/Notify)
    if data_obj is not None:
        _safe_set(core, "data", data_obj)
    if notify_obj is not None:
        _safe_set(core, "notify", notify_obj)

    # Preserve active symbols if bootstrap provided them
    if active_symbols is not None:
        try:
            # bootstrap sets a set(); accept list/tuple too
            if isinstance(active_symbols, (set, list, tuple)):
                _safe_set(core, "active_symbols", set(active_symbols))
        except Exception:
            pass

    # Preserve min_amounts if bootstrap/other modules populated it
    min_amounts = _safe_get(seed_bot, "min_amounts", None)
    if isinstance(min_amounts, dict) and min_amounts:
        _safe_set(core, "min_amounts", min_amounts)

    # ----------------------------
    # Important compatibility shims
    # ----------------------------
    # Some of your execution modules call bot.notify.speak(...)
    # core already has speak(), but keep notify object for that path too.

    # Many modules expect bot.state.run_context dict exists
    try:
        rc = _safe_get(core.state, "run_context", None)
        if not isinstance(rc, dict):
            _safe_set(core.state, "run_context", {})
    except Exception:
        pass

    # Do NOT autostart core internal loops here.
    # bootstrap controls tasks. If you later want core-managed loops,
    # that belongs in runner path (bot/runner.py), not bootstrap.

    return core
