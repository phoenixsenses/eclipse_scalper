from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import time
import traceback
from pathlib import Path
from typing import Any, Dict

from execution.entry import try_enter
from execution.runtime_helpers import (
    cfg_env_bool as _cfg_env_bool,
    cfg_env_float as _cfg_env_float,
    cfg_value as _cfg,
    safe_float as _safe_float,
    symkey as _symkey,
    truthy as _truthy,
)
from execution.shutdown_control import ensure_traced_shutdown_event
from utils.logging import log_core, log_entry
try:
    from core.latency_profiler import LatencyProfiler  # type: ignore
except Exception:
    LatencyProfiler = None  # type: ignore

try:
    from risk.kill_switch import trade_allowed  # type: ignore
except Exception:
    trade_allowed = None
try:
    from core.regime import RegimeClassifier  # type: ignore
except Exception:
    RegimeClassifier = None  # type: ignore
try:
    from execution.entry_loop import _build_micro_signal_provider as _build_micro_signal_provider_ref  # type: ignore
except Exception:
    _build_micro_signal_provider_ref = None  # type: ignore


def _now() -> float:
    return time.time()


_THROTTLE_LAST: Dict[str, float] = {}


def _throttled_log(key: str, every_sec: float, fn, msg: str) -> None:
    now = _now()
    last = float(_THROTTLE_LAST.get(key, 0.0) or 0.0)
    if (now - last) >= max(0.0, every_sec):
        _THROTTLE_LAST[key] = now
        try:
            fn(msg)
        except Exception:
            pass


def _get_price(bot, symbol: str) -> float:
    px, _ = _get_price_with_source(bot, symbol)
    return px


def _get_price_with_source(bot, symbol: str) -> tuple[float, str]:
    k = _symkey(symbol)
    px = 0.0
    source = "none"
    try:
        data = getattr(bot, "data", None)
        gp = getattr(data, "get_price", None) if data is not None else None
        if callable(gp):
            try:
                px = float(gp(k, in_position=False) or 0.0)
                if px > 0:
                    source = "data.get_price(in_position=False)"
            except TypeError:
                px = float(gp(k) or 0.0)
                if px > 0:
                    source = "data.get_price"
    except Exception:
        px = 0.0
    if px <= 0:
        try:
            data = getattr(bot, "data", None)
            gd = getattr(data, "get", None) if data is not None else None
            if callable(gd):
                candidate = gd(k, "last")
                if isinstance(candidate, dict):
                    px = float(candidate.get("last", 0.0) or 0.0)
                else:
                    px = float(candidate or 0.0)
                if px > 0:
                    source = "data.get(last)"
        except Exception:
            px = 0.0
    if px <= 0:
        try:
            price_map = getattr(getattr(bot, "data", None), "price", {}) or {}
            if isinstance(price_map, dict):
                px = float(price_map.get(k, 0.0) or 0.0)
                if px <= 0:
                    px = float(price_map.get(str(symbol), 0.0) or 0.0)
                if px > 0:
                    source = "data.price"
        except Exception:
            px = 0.0
    if px <= 0:
        try:
            tickers = getattr(getattr(bot, "exchange", None), "tickers", None)
            if isinstance(tickers, dict):
                for key in (str(symbol), k, f"{k[:-4]}/USDT:USDT", f"{k[:-4]}/USDT"):
                    tv = tickers.get(key)
                    if isinstance(tv, dict):
                        v = float(tv.get("last", 0.0) or 0.0)
                        if v > 0:
                            px = v
                            source = "exchange.tickers"
                            break
        except Exception:
            px = 0.0
    return (float(px), source) if px > 0 else (0.0, source)


def _micro_mark_price(micro_engine, symbol: str) -> float:
    if micro_engine is None:
        return 0.0
    try:
        getter = getattr(micro_engine, "get_features", None)
        if callable(getter):
            feat = getter(symbol)
            if feat is None:
                feat = getter(_symkey(symbol))
            if feat is not None:
                v = float(getattr(feat, "mark_price", 0.0) or 0.0)
                if v > 0:
                    return v
    except Exception:
        pass
    try:
        feat = getattr(micro_engine, "current_features", None)
        v = float(getattr(feat, "mark_price", 0.0) or 0.0)
        return v if v > 0 else 0.0
    except Exception:
        return 0.0


def _resolve_regime_price(bot, symbol: str, micro_engine) -> tuple[float, str]:
    px_now, source = _get_price_with_source(bot, symbol)
    if px_now > 0:
        return float(px_now), source
    mpx = _micro_mark_price(micro_engine, symbol)
    if mpx > 0:
        return float(mpx), "micro_mark_price"
    return 0.0, (source or "none")


def _parse_ts_any(v: Any) -> float:
    if v is None:
        return 0.0
    if isinstance(v, (int, float)):
        x = float(v)
        if x > 1e12:
            return x / 1000.0
        return x if x > 1e9 else x
    s = str(v).strip()
    if not s:
        return 0.0
    try:
        return _parse_ts_any(float(s))
    except Exception:
        pass
    try:
        s2 = s.replace("Z", "+00:00")
        return float(__import__("datetime").datetime.fromisoformat(s2).timestamp())
    except Exception:
        return 0.0


def _warmup_regime_from_mark_prices(
    *,
    regime_runtime: "_RegimeRuntime",
    symbol: str,
    db_path: str,
    lookback_sec: int,
) -> dict[str, Any]:
    out: dict[str, Any] = {"symbol": _symkey(symbol), "fed": 0, "span_sec": 0.0, "regime": "UNKNOWN"}
    db = Path(str(db_path or "").strip() or "data/microstructure.db")
    if not db.exists():
        out["reason"] = "db_missing"
        return out
    conn = sqlite3.connect(str(db), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        tables = {
            str(r["name"]) for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        }
        if "mark_prices" not in tables:
            out["reason"] = "mark_prices_missing"
            return out
        cols = [str(r["name"]) for r in conn.execute("PRAGMA table_info(mark_prices)").fetchall()]
        low_map = {c.lower(): c for c in cols}

        def _pick(*names: str) -> str:
            for n in names:
                if n in low_map:
                    return low_map[n]
            return ""

        ts_col = _pick("ts_ms", "ts_utc", "ts", "timestamp", "event_ts", "time", "t")
        px_col = _pick("mark_price", "price", "p", "close", "c")
        sym_col = _pick("symbol", "s", "pair", "instrument")
        if not ts_col or not px_col:
            out["reason"] = "missing_ts_or_price_col"
            return out

        horizon = int(max(60, lookback_sec))
        lim = int(max(2000, horizon * 4))
        params: list[Any] = []
        where = ""
        if sym_col:
            where = f"WHERE {sym_col} = ?"
            params.append(_symkey(symbol))
        q = (
            f"SELECT {ts_col} AS tsv, {px_col} AS pxv "
            f"FROM mark_prices {where} ORDER BY {ts_col} DESC LIMIT {lim}"
        )
        rows = conn.execute(q, params).fetchall()
        if not rows:
            out["reason"] = "no_rows"
            return out
        pts: list[tuple[float, float]] = []
        now = _now()
        min_ts = now - float(horizon)
        for r in rows:
            # _parse_ts_any normalizes seconds/milliseconds/ISO timestamps.
            ts = _parse_ts_any(r["tsv"])
            px = float(r["pxv"] or 0.0)
            if ts <= 0 or px <= 0:
                continue
            if ts < min_ts:
                continue
            pts.append((ts, px))
        if not pts:
            out["reason"] = "no_recent_rows"
            return out
        pts.sort(key=lambda x: x[0])
        state = None
        for ts, px in pts:
            state = regime_runtime.update(symbol, float(ts), float(px))
        out["fed"] = len(pts)
        out["span_sec"] = max(0.0, float(pts[-1][0] - pts[0][0]))
        if isinstance(state, dict):
            out["regime"] = str(state.get("current_regime") or "UNKNOWN")
            out["rolling_return"] = float(state.get("rolling_return") or 0.0)
            out["regime_age_sec"] = float(state.get("regime_age_sec") or 0.0)
        return out
    except Exception as e:
        out["reason"] = f"{type(e).__name__}:{e}"
        return out
    finally:
        try:
            conn.close()
        except Exception:
            pass


class _RegimeRuntime:
    def __init__(self, lookback_sec: int, debounce_sec: int):
        self.lookback_sec = int(max(1, lookback_sec))
        self.debounce_sec = int(max(0, debounce_sec))
        self._by_symbol: Dict[str, Any] = {}

    def update(self, symbol: str, ts: float, price: float) -> Dict[str, Any]:
        k = _symkey(symbol)
        cls = self._by_symbol.get(k)
        if cls is None and callable(RegimeClassifier):
            cls = RegimeClassifier(lookback_sec=self.lookback_sec, debounce_sec=self.debounce_sec)
            self._by_symbol[k] = cls
        if cls is None:
            return {"current_regime": "UNKNOWN", "rolling_return": 0.0, "regime_age_sec": 0.0}
        try:
            cls.update(float(ts), float(price))
            return {
                "current_regime": str(getattr(cls, "current_regime", "UNKNOWN") or "UNKNOWN"),
                "rolling_return": float(getattr(cls, "rolling_return", 0.0) or 0.0),
                "regime_age_sec": float(getattr(cls, "regime_age_sec", 0.0) or 0.0),
            }
        except Exception:
            return {"current_regime": "UNKNOWN", "rolling_return": 0.0, "regime_age_sec": 0.0}


def _parse_micro_eval_result(res: Any) -> tuple[bool, str, float, str]:
    try:
        if res is None:
            return False, "", 0.0, "provider_returned_none"
        if hasattr(res, "present") and hasattr(res, "reason"):
            present = bool(getattr(res, "present", False))
            conf = float(getattr(res, "confidence", 0.0) or 0.0)
            side = str(getattr(res, "side", "") or "").strip().lower()
            reason = str(getattr(res, "reason", "unknown") or "unknown")
            if side not in ("buy", "sell"):
                side = ""
            return present, side, conf, reason
        side = str(getattr(res, "side", "") or "").strip().lower()
        conf = float(getattr(res, "confidence", 0.0) or 0.0)
        if side not in ("buy", "sell"):
            side = ""
        return True, side, conf, "legacy_signal"
    except Exception as e:
        return False, "", 0.0, f"error:{type(e).__name__}"


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


def _ensure_data_ready_event(bot) -> asyncio.Event:
    ev = getattr(bot, "data_ready", None)
    if isinstance(ev, asyncio.Event):
        return ev
    ev = asyncio.Event()
    try:
        bot.data_ready = ev  # type: ignore[attr-defined]
    except Exception:
        pass
    return ev


def _pick_symbols(bot) -> list[str]:
    try:
        s = getattr(bot, "active_symbols", None)
        if isinstance(s, set) and s:
            return sorted(list(s))
        if isinstance(s, (list, tuple)) and s:
            return [str(x) for x in s if str(x).strip()]
    except Exception:
        pass
    try:
        s2 = getattr(getattr(bot, "cfg", None), "ACTIVE_SYMBOLS", None)
        if isinstance(s2, (list, tuple)) and s2:
            return [str(x) for x in s2 if str(x).strip()]
    except Exception:
        pass
    return ["BTCUSDT"]


def _resolve_micro_builder():
    if callable(_build_micro_signal_provider_ref):
        return _build_micro_signal_provider_ref
    try:
        from execution.entry_loop import _build_micro_signal_provider as _lazy_builder  # type: ignore
        return _lazy_builder
    except Exception:
        return None


def _write_last_shutdown_json(payload: Dict[str, Any]) -> None:
    try:
        from execution.runtime_helpers import atomic_write_json
        atomic_write_json(Path("logs/last_shutdown.json"), payload)
    except Exception:
        pass


def _append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=True, sort_keys=True) + "\n")
    except Exception:
        pass


def _tripwire_shutdown_bypass(bot) -> tuple[str, str, int, float]:
    rs = str(getattr(bot, "_shutdown_reason", "") or "")
    src = str(getattr(bot, "_shutdown_source", "") or "")
    fatal = int(bool(getattr(bot, "_shutdown_fatal", False)))
    ts = float(getattr(bot, "_shutdown_ts", 0.0) or 0.0)
    if rs or src:
        return rs, src, fatal, ts
    tr = str(getattr(bot, "_shutdown_set_trace", "") or "")
    setter_ts = float(getattr(bot, "_shutdown_set_ts", 0.0) or 0.0)
    teardown_flag = bool(getattr(bot, "_teardown_in_progress", False))
    if teardown_flag or ("execution\\bootstrap.py" in tr or "execution/bootstrap.py" in tr):
        return "teardown_cleanup", "execution.bootstrap", 0, (_now() if setter_ts <= 0 else setter_ts)
    if not tr:
        tr = "".join(traceback.format_stack(limit=25))
        setter_ts = 0.0
    rs = "bypass_detected"
    src = "unknown_setter"
    fatal = 1
    ts = _now()
    try:
        bot._shutdown_reason = rs  # type: ignore[attr-defined]
        bot._shutdown_source = src  # type: ignore[attr-defined]
        bot._shutdown_fatal = bool(fatal)  # type: ignore[attr-defined]
        bot._shutdown_ts = float(ts)  # type: ignore[attr-defined]
        bot._shutdown_trace = tr  # type: ignore[attr-defined]
    except Exception:
        pass
    _write_last_shutdown_json(
        {
            "ts": float(ts),
            "setter_ts": float(setter_ts) if setter_ts > 0 else None,
            "source": src,
            "reason": rs,
            "fatal": bool(fatal),
            "exc": "",
            "trace": tr,
        }
    )
    log_core.critical("[SHUTDOWN_BYPASS_DETECTED] shutdown set without metadata")
    log_core.critical(f"[SHUTDOWN_BYPASS_DETECTED] stack:\n{tr}")
    return rs, src, fatal, ts


async def entry_loop_full(bot) -> None:
    shutdown_ev = _ensure_shutdown_event(bot)
    data_ready_ev = _ensure_data_ready_event(bot)

    poll_sec = _cfg_env_float(bot, "ENTRY_POLL_SEC", 1.0)
    per_symbol_gap_sec = _cfg_env_float(bot, "ENTRY_PER_SYMBOL_GAP_SEC", 2.5)
    local_cooldown_sec = _cfg_env_float(bot, "ENTRY_LOCAL_COOLDOWN_SEC", 8.0)
    wait_data_sec = _cfg_env_float(bot, "ENTRY_WAIT_FOR_DATA_READY_SEC", 8.0)

    spawn_both = _cfg_env_bool(bot, "SPAWN_BOTH_SIDES", False)
    respect_kill = _cfg_env_bool(bot, "ENTRY_RESPECT_KILL_SWITCH", True)
    diag = _cfg_env_bool(bot, "SCALPER_SIGNAL_DIAG", _cfg(bot, "SCALPER_SIGNAL_DIAG", "0"))
    micro_enabled = _cfg_env_bool(bot, "ENTRY_MICRO_SIGNAL_ENABLED", False)
    micro_symbol = _symkey(str(os.getenv("MICRO_SIGNAL_SYMBOL", "") or ""))

    regime_runtime = _RegimeRuntime(
        int(_cfg_env_float(bot, "ENTRY_REGIME_LOOKBACK_SEC", 3600.0) or 3600),
        int(_cfg_env_float(bot, "ENTRY_REGIME_DEBOUNCE_SEC", 60.0) or 60),
    )
    regime_lookback_sec = int(_cfg_env_float(bot, "ENTRY_REGIME_LOOKBACK_SEC", 3600.0) or 3600)

    last_attempt_by_sym: Dict[str, float] = {}
    last_tick = 0.0
    micro_engine = None
    micro_provider = None
    latency = LatencyProfiler() if LatencyProfiler is not None else None
    latency_last_log = 0.0
    latency_log_sec = float(_cfg_env_float(bot, "ENTRY_LATENCY_LOG_SEC", 60.0) or 60.0)
    latency_jsonl = Path(str(os.getenv("ENTRY_LATENCY_LOG_PATH", "logs/latency_profile.jsonl") or "logs/latency_profile.jsonl"))
    staleness_count = 0
    staleness_age_sum = 0.0
    staleness_db_age_sum = 0.0

    log_core.info("ENTRY_LOOP_FULL ONLINE - using execution.entry.try_enter()")

    micro_builder = _resolve_micro_builder() if micro_enabled else None
    if micro_enabled and callable(micro_builder):
        try:
            micro_engine, micro_provider = micro_builder(bot)
            if micro_engine is not None and callable(getattr(micro_engine, "start", None)):
                await micro_engine.start()
            if micro_provider is not None:
                log_core.info(f"ENTRY_LOOP_FULL: micro path enabled symbol={micro_symbol or '(auto)'}")
        except Exception as e:
            micro_engine, micro_provider = None, None
            log_core.warning(f"ENTRY_LOOP_FULL: micro provider init failed: {type(e).__name__}: {e}")
    elif micro_enabled:
        log_core.warning("ENTRY_LOOP_FULL: micro enabled but provider builder unavailable")

    warmup_enabled = _cfg_env_bool(bot, "ENTRY_REGIME_WARMUP_ENABLED", True)
    if warmup_enabled and callable(RegimeClassifier):
        warmup_db = str(
            os.getenv("ENTRY_REGIME_WARMUP_DB")
            or os.getenv("MICRO_SIGNAL_DB")
            or "data/microstructure.db"
        ).strip()
        for sym in _pick_symbols(bot):
            k = _symkey(sym)
            if not k:
                continue
            ws = _warmup_regime_from_mark_prices(
                regime_runtime=regime_runtime,
                symbol=k,
                db_path=warmup_db,
                lookback_sec=regime_lookback_sec,
            )
            log_core.info(
                f"[REGIME_WARMUP] {k} fed {int(ws.get('fed', 0) or 0)} prices "
                f"spanning {int(float(ws.get('span_sec', 0.0) or 0.0))}s "
                f"regime={str(ws.get('regime') or 'UNKNOWN')}"
            )

    if wait_data_sec > 0 and not data_ready_ev.is_set():
        try:
            await asyncio.wait_for(data_ready_ev.wait(), timeout=wait_data_sec)
        except Exception:
            pass

    try:
        while not shutdown_ev.is_set():
            try:
                t_loop0 = time.perf_counter()
                now = _now()
                if poll_sec > 0 and (now - last_tick) < poll_sec:
                    await asyncio.sleep(max(0.05, poll_sec - (now - last_tick)))
                last_tick = _now()

                if wait_data_sec > 0 and not data_ready_ev.is_set():
                    await asyncio.sleep(max(0.25, poll_sec))
                    continue

                if respect_kill and callable(trade_allowed):
                    t_risk0 = time.perf_counter()
                    try:
                        ok = await trade_allowed(bot)
                        if latency is not None:
                            latency.add("risk_ms", (time.perf_counter() - t_risk0) * 1000.0)
                        if not ok:
                            await asyncio.sleep(max(0.25, poll_sec))
                            continue
                    except Exception:
                        if latency is not None:
                            latency.add("risk_ms", (time.perf_counter() - t_risk0) * 1000.0)
                        await asyncio.sleep(max(0.25, poll_sec))
                        continue

                syms = _pick_symbols(bot)
                if not syms:
                    await asyncio.sleep(max(0.25, poll_sec))
                    continue

                for sym in syms:
                    if shutdown_ev.is_set():
                        break

                    k = _symkey(sym)
                    if not k:
                        continue

                    t_feature0 = time.perf_counter()
                    px_now, px_src = _resolve_regime_price(bot, k, micro_engine if micro_enabled else None)
                    feature_ms = (time.perf_counter() - t_feature0) * 1000.0
                    if latency is not None:
                        latency.add("feature_ms", feature_ms)
                    _throttled_log(
                        key=f"price_diag_full:{k}",
                        every_sec=5.0,
                        fn=log_core.info,
                        msg=f"[PRICE_DIAG] {k} px_now={px_now} type={type(px_now).__name__} src={px_src}",
                    )
                    if px_src == "micro_mark_price":
                        _throttled_log(
                            key=f"regime_px_fallback:{k}",
                            every_sec=5.0,
                            fn=log_core.info,
                            msg=f"[REGIME_PRICE_FALLBACK] {k} using micro mark_price={px_now}",
                        )
                    if micro_engine is not None:
                        try:
                            diag_fn = getattr(micro_engine, "diagnostics", None)
                            diag = diag_fn(k) if callable(diag_fn) else None
                            if isinstance(diag, dict):
                                age_sec = _safe_float(diag.get("age_sec"), -1.0)
                                db_age_sec = _safe_float(diag.get("last_db_age_sec"), -1.0)
                                if age_sec >= 0:
                                    staleness_count += 1
                                    staleness_age_sum += float(age_sec)
                                if db_age_sec >= 0:
                                    staleness_db_age_sum += float(db_age_sec)
                        except Exception:
                            pass
                    t_regime0 = time.perf_counter()
                    regime_state = regime_runtime.update(k, _now(), px_now) if px_now > 0 else None
                    regime_ms = (time.perf_counter() - t_regime0) * 1000.0
                    if latency is not None:
                        latency.add("regime_ms", regime_ms)
                    if isinstance(regime_state, dict):
                        _throttled_log(
                            key=f"regime_full:{k}",
                            every_sec=60.0,
                            fn=log_core.info,
                            msg=(
                                f"[REGIME] {k} regime={str(regime_state.get('current_regime') or 'UNKNOWN')} "
                                f"return_1h={float(regime_state.get('rolling_return') or 0.0)*100.0:+.2f}% "
                                f"age={int(float(regime_state.get('regime_age_sec') or 0.0))}s"
                            ),
                        )

                    la = float(last_attempt_by_sym.get(k, 0.0) or 0.0)
                    if local_cooldown_sec > 0 and (_now() - la) < local_cooldown_sec:
                        continue
                    last_attempt_by_sym[k] = _now()

                    if diag:
                        log_entry.info(f"ENTRY_LOOP_FULL scan {k}")

                    micro_taken = False
                    if micro_provider is not None:
                        sym_match = (not micro_symbol) or (_symkey(k) == _symkey(micro_symbol))
                        if sym_match:
                            try:
                                t_signal0 = time.perf_counter()
                                eval_res = micro_provider.evaluate(
                                    regime_override=regime_state if isinstance(regime_state, dict) else None
                                )
                                if asyncio.iscoroutine(eval_res) or hasattr(eval_res, "__await__"):
                                    eval_res = await eval_res
                                signal_ms = (time.perf_counter() - t_signal0) * 1000.0
                                if latency is not None:
                                    latency.add("signal_ms", signal_ms)
                                present, action, conf, reason = _parse_micro_eval_result(eval_res)
                                _throttled_log(
                                    key=f"entry_decision_full_micro:{k}:{reason}:{action or 'none'}",
                                    every_sec=5.0,
                                    fn=log_core.info,
                                    msg=f"ENTRY_DECISION {k}: conf={conf:.2f} gates={{'micro': 1}} reason={reason}",
                                )
                                if str(reason).strip().lower() == "no_match":
                                    try:
                                        meta = dict(getattr(eval_res, "meta", {}) or {})
                                        if meta:
                                            _throttled_log(
                                                key=f"entry_decision_full_micro_detail:{k}",
                                                every_sec=5.0,
                                                fn=log_core.info,
                                                msg=(
                                                    "[MICRO_SIGNAL] no_match_detail "
                                                    f"{k} target_side={meta.get('target_side')} pocket={meta.get('pocket')} "
                                                    f"imb_ok={int(bool(meta.get('imb_ok', False)))} "
                                                    f"int_ok={int(bool(meta.get('int_ok', False)))} spr_ok={int(bool(meta.get('spr_ok', False)))} "
                                                    f"missing={meta.get('missing')} "
                                                    f"imb={_safe_float(meta.get('imbalance_signed'), 0.0):+.4f}/{_safe_float(meta.get('min_imbalance'), 0.0):.4f} "
                                                    f"int={_safe_float(meta.get('trade_intensity'), 0.0):.0f}/{_safe_float(meta.get('min_intensity'), 0.0):.0f} "
                                                    f"spr={_safe_float(meta.get('spread'), 0.0):.6f}/{_safe_float(meta.get('max_spread'), 0.0):.6f}"
                                                ),
                                            )
                                    except Exception:
                                        pass
                                if present and conf >= 1.0 and action in ("buy", "sell"):
                                    m_side = "long" if action == "buy" else "short"
                                    t_submit0 = time.perf_counter()
                                    await try_enter(bot, k, m_side)
                                    submit_ms = (time.perf_counter() - t_submit0) * 1000.0
                                    if latency is not None:
                                        latency.add("submit_ms", submit_ms)
                                        rec = getattr(bot, "_last_entry_latency", {}) if hasattr(bot, "_last_entry_latency") else {}
                                        fill_ack_ms = _safe_float((rec.get(k) or {}).get("fill_ack_ms"), submit_ms) if isinstance(rec, dict) else submit_ms
                                        latency.add("fill_ack_ms", fill_ack_ms)
                                    micro_taken = True
                            except Exception as e:
                                _throttled_log(
                                    key=f"micro_full_err:{k}",
                                    every_sec=30.0,
                                    fn=log_core.warning,
                                    msg=f"ENTRY_LOOP_FULL micro eval failed {k}: {type(e).__name__}: {e}",
                                )
                    if micro_taken:
                        await asyncio.sleep(max(0.01, per_symbol_gap_sec))
                        continue

                    if spawn_both:
                        t_submit0 = time.perf_counter()
                        await try_enter(bot, k, "long")
                        await try_enter(bot, k, "short")
                        submit_ms = (time.perf_counter() - t_submit0) * 1000.0
                        if latency is not None:
                            latency.add("submit_ms", submit_ms)
                            latency.add("fill_ack_ms", submit_ms)
                    else:
                        side = "long" if (hash(k) ^ int(_now() // max(1.0, local_cooldown_sec))) % 2 == 0 else "short"
                        t_submit0 = time.perf_counter()
                        await try_enter(bot, k, side)
                        submit_ms = (time.perf_counter() - t_submit0) * 1000.0
                        if latency is not None:
                            latency.add("submit_ms", submit_ms)
                            rec = getattr(bot, "_last_entry_latency", {}) if hasattr(bot, "_last_entry_latency") else {}
                            fill_ack_ms = _safe_float((rec.get(k) or {}).get("fill_ack_ms"), submit_ms) if isinstance(rec, dict) else submit_ms
                            latency.add("fill_ack_ms", fill_ack_ms)

                    await asyncio.sleep(max(0.01, per_symbol_gap_sec))
                if latency is not None:
                    latency.add("e2e_ms", (time.perf_counter() - t_loop0) * 1000.0)
                    if (_now() - latency_last_log) >= max(5.0, latency_log_sec):
                        latency_last_log = _now()
                        snap = latency.snapshot()
                        def _m(k: str) -> float:
                            return float((snap.get(k) or {}).get("mean_ms", 0.0))
                        avg_feature_age = (staleness_age_sum / staleness_count) if staleness_count > 0 else 0.0
                        avg_db_age = (staleness_db_age_sum / staleness_count) if staleness_count > 0 else 0.0
                        msg = (
                            f"[LATENCY] e2e={_m('e2e_ms'):.0f}ms feature={_m('feature_ms'):.0f}ms "
                            f"signal={_m('signal_ms'):.0f}ms regime={_m('regime_ms'):.0f}ms risk={_m('risk_ms'):.0f}ms "
                            f"submit={_m('submit_ms'):.0f}ms fill_ack={_m('fill_ack_ms'):.0f}ms "
                            f"feature_age={avg_feature_age:.2f}s db_lag={avg_db_age:.2f}s"
                        )
                        log_core.info(msg)
                        _append_jsonl(
                            latency_jsonl,
                            {
                                "ts": float(_now()),
                                "loop": "entry_loop_full",
                                "metrics": snap,
                                "feature_age_sec_avg": float(avg_feature_age),
                                "db_lag_sec_avg": float(avg_db_age),
                            },
                        )
                        latency.reset()
                        staleness_count = 0
                        staleness_age_sum = 0.0
                        staleness_db_age_sum = 0.0

            except asyncio.CancelledError:
                raise
            except Exception as e:
                log_entry.error(f"ENTRY_LOOP_FULL outer error: {e}")
                await asyncio.sleep(1.0)
    finally:
        try:
            if micro_engine is not None and callable(getattr(micro_engine, "stop", None)):
                await micro_engine.stop()
        except Exception:
            pass
        rs, src, fatal, ts = _tripwire_shutdown_bypass(bot)
        age = max(0.0, _now() - ts) if ts > 0 else 0.0
        if rs or src:
            log_core.critical(
                f"ENTRY_LOOP_FULL OFFLINE - shutdown flag set reason={rs or 'unknown'} "
                f"source={src or 'unknown'} fatal={fatal} age_sec={age:.1f}"
            )
        else:
            log_core.critical("ENTRY_LOOP_FULL OFFLINE - shutdown flag set")
