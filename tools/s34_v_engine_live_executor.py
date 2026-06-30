"""Live executor for the S34 V Engine alpha.

This executor is intentionally separate from the legacy S34 paper-trade mirror.
It reads the local microstructure DB, detects the knowable V Engine anchor, and
submits a single ETHUSDT maker LONG lifecycle:

    ETH SELL running_notional >= 200K
    28 <= V-depth < 40
    prior 4h ETH trend < -50 bps
    initial maker limit O20
    after 300s, cancel/replace to O5 if still unfilled
    close by market reduce-only after 2h from detected fill

It places real orders only when armed with --live --confirm-live-orders and the
existing S34 live env switches are live-enabled.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import re
import sqlite3
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from exchanges import get_exchange  # noqa: E402
from tools.research_s34_knowable_anchor_continuation import (  # noqa: E402
    load_liquidations,
    load_mark_index,
    reconstruct_anchors,
)
from tools.research_s34_maker_fade import anchor_vdepth_bps, maker_limit_price  # noqa: E402
from tools.research_s34_wave_absorption import book_features_at  # noqa: E402
from tools.s34_live_order_executor import (  # noqa: E402
    load_env,
    set_leverage_if_possible,
    truthy,
)


ENV_PATH = ROOT / ".env"
DEFAULT_DB = ROOT / "data" / "microstructure.db"
DEFAULT_STATE_PATH = ROOT / "runtime" / "s34_v_engine_live_state.json"
PID_PATH = ROOT / "logs" / "pids" / "s34_v_engine_live_executor.pid"

RULE_NAME = "S34_V_ENGINE_V0_2_ETH_SELL_MAKER_LONG_H2_O20_W300_O5_DEEPBID"
SYMBOL = "ETHUSDT"
LIQ_SIDE = "SELL"
DIRECTION = "LONG"
THRESHOLD_USD = 200_000.0
VDEPTH_MIN_BPS = 28.0
VDEPTH_MAX_BPS = 40.0
PRIOR4H_LT_BPS = -50.0
MIN_BID_DEPTH_USD = 135_423.8
BUCKET_SEC = 300
MIN_GAP_SEC = 900
ACCEL_WINDOW_SEC = 30
INITIAL_OFFSET_BPS = 20.0
REPLACE_OFFSET_BPS = 5.0
REPLACE_WAIT_SEC = 300
HORIZON_SEC = 2 * 3600
DEFAULT_STOP_BPS = 150.0
ORDER_CLIENT_PREFIX = "S34VE"
DEFAULT_SIGNAL_FRESH_SEC = 60
MISSED_SIGNAL_GRACE_SEC = 10 * 60

# Silence gate: noisy early-exit for live position management (validated in research)
# Any follow-on ETH SELL cascade ≥50K within 1-30min of anchor → noisy event → exit early.
# If 30min passes with no cascade → silence confirmed → extend hold from 2h to 4h.
SIL_GATE_LO_MS      = 60_000        # ultra-early window start (60s)
SIL_GATE_HI_MS      = 30 * 60_000   # silence confirmation at 30min
SIL_PROP_THRESH_USD = 50_000.0       # follow-on cascade detection threshold
SIL_HORIZON_SEC     = 4 * 3600      # silence events hold 4h from anchor (not fill)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def fenv(env: dict[str, str], key: str, default: float) -> float:
    try:
        return float(env.get(key, default))
    except (TypeError, ValueError):
        return default


def allowed_rules(env: dict[str, str]) -> set[str]:
    raw = env.get("S34_LIVE_ALLOWED_RULES", "")
    return {x.strip() for x in raw.split(",") if x.strip()}


def assert_live_allowed(env: dict[str, str], args: argparse.Namespace) -> bool:
    if not args.live:
        os.environ["SCALPER_DRY_RUN"] = "1"
        return False
    if not args.confirm_live_orders:
        raise RuntimeError("live mode requires --confirm-live-orders")
    if not truthy(env.get("S34_LIVE_TRADING_ENABLED")) or truthy(env.get("S34_LIVE_DRY_RUN")) or truthy(env.get("SCALPER_DRY_RUN")):
        raise RuntimeError(
            "live mode requires S34_LIVE_TRADING_ENABLED=1, S34_LIVE_DRY_RUN=0, and SCALPER_DRY_RUN=0"
        )
    return True


def load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"processed_event_ids": {}, "active": None, "orders": [], "status": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"processed_event_ids": {}, "active": None, "orders": [], "status": {}}
    if not isinstance(data, dict):
        return {"processed_event_ids": {}, "active": None, "orders": [], "status": {}}
    data.setdefault("processed_event_ids", {})
    data.setdefault("active", None)
    data.setdefault("orders", [])
    data.setdefault("status", {})
    return data


def save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")


def recent_anchor(
    conn: sqlite3.Connection,
    *,
    lookback_sec: int,
    signal_fresh_sec: int,
    max_book_staleness_sec: int,
    diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    now_ms = int(time.time() * 1000)
    start_ms = now_ms - int(lookback_sec) * 1000
    diag = diagnostics if diagnostics is not None else {}
    diag.clear()
    diag.update(
        {
            "scan_ts_utc": utc_now(),
            "lookback_sec": int(lookback_sec),
            "signal_fresh_sec": int(signal_fresh_sec),
            "max_book_staleness_sec": int(max_book_staleness_sec),
            "anchors_reconstructed": 0,
            "eligible_fresh_candidates": 0,
            "reject_counts": {
                "too_old": 0,
                "vdepth": 0,
                "prior4h": 0,
                "book": 0,
                "mark": 0,
            },
        }
    )
    liqs = load_liquidations(conn, SYMBOL, LIQ_SIDE, start_ms, now_ms)
    anchors = reconstruct_anchors(
        liqs,
        bucket_sec=BUCKET_SEC,
        min_gap_sec=MIN_GAP_SEC,
        thresholds=(THRESHOLD_USD,),
        accel_window_sec=ACCEL_WINDOW_SEC,
    )
    diag["anchors_reconstructed"] = len(anchors)
    if not anchors:
        return None
    marks = load_mark_index(conn, SYMBOL)
    candidates = []
    stale_eligible: list[dict[str, Any]] = []
    for anchor in anchors:
        age_sec = (now_ms - int(anchor.anchor_ts_ms)) / 1000.0
        depth = anchor_vdepth_bps(marks, anchor, LIQ_SIDE)
        if depth is None or not (VDEPTH_MIN_BPS <= float(depth) < VDEPTH_MAX_BPS):
            diag["reject_counts"]["vdepth"] += 1
            continue
        prior4h = marks.ret_bps(int(anchor.anchor_ts_ms) - 4 * 3600 * 1000, int(anchor.anchor_ts_ms))
        if prior4h is None or not math.isfinite(float(prior4h)) or not (float(prior4h) < PRIOR4H_LT_BPS):
            diag["reject_counts"]["prior4h"] += 1
            continue
        book = book_features_at(conn, SYMBOL, int(anchor.anchor_ts_ms), int(max_book_staleness_sec))
        if not book or float(book["bid_depth_usd"]) < MIN_BID_DEPTH_USD:
            diag["reject_counts"]["book"] += 1
            continue
        mark = marks.at_or_after(int(anchor.anchor_ts_ms))
        if not mark:
            diag["reject_counts"]["mark"] += 1
            continue
        row = {
            "event_id": f"{RULE_NAME}:{anchor.bucket}:{int(anchor.anchor_ts_ms)}",
            "bucket": int(anchor.bucket),
            "anchor_ts_ms": int(anchor.anchor_ts_ms),
            "anchor_utc": datetime.fromtimestamp(int(anchor.anchor_ts_ms) / 1000.0, tz=timezone.utc).isoformat(),
            "anchor_age_sec": round(float(age_sec), 1),
            "anchor_mark_ts_ms": int(mark[0]),
            "anchor_mark_price": float(mark[1]),
            "vdepth_bps": round(float(depth), 1),
            "prior_4h_bps": round(float(prior4h), 1),
            "book_ts_ms": int(book["book_ts_ms"]),
            "book_staleness_ms": int(book["book_staleness_ms"]),
            "book_imbalance": round(float(book["book_imbalance"]), 4),
            "bid_depth_usd": round(float(book["bid_depth_usd"]), 1),
            "ask_depth_usd": round(float(book["ask_depth_usd"]), 1),
            "spread_bps": round(float(book["spread_bps"]), 4),
            "absorption_gate": "deep_bid",
            "min_bid_depth_usd": MIN_BID_DEPTH_USD,
            "running_notional": round(float(anchor.running_notional), 1),
            "running_liq_count": int(anchor.running_liq_count),
            "running_accel_usd_per_sec": round(float(anchor.running_accel), 1),
            "elapsed_since_first_sec": round(float(anchor.elapsed_since_first_sec), 1),
            "single_liq_dominance_pct": round(float(anchor.running_single_liq_dominance), 1),
        }
        if age_sec < 0 or age_sec > float(signal_fresh_sec):
            diag["reject_counts"]["too_old"] += 1
            if 0 <= age_sec <= float(signal_fresh_sec) + MISSED_SIGNAL_GRACE_SEC:
                stale_eligible.append(row)
            continue
        candidates.append(row)
    candidates.sort(key=lambda r: int(r["anchor_ts_ms"]), reverse=True)
    stale_eligible.sort(key=lambda r: int(r["anchor_ts_ms"]), reverse=True)
    diag["eligible_fresh_candidates"] = len(candidates)
    if stale_eligible:
        missed = stale_eligible[0]
        diag["latest_missed_signal"] = {
            "event_id": missed["event_id"],
            "anchor_utc": missed["anchor_utc"],
            "anchor_age_sec": missed["anchor_age_sec"],
            "missed_by_sec": round(float(missed["anchor_age_sec"]) - float(signal_fresh_sec), 1),
            "vdepth_bps": missed["vdepth_bps"],
            "prior_4h_bps": missed["prior_4h_bps"],
            "bid_depth_usd": missed["bid_depth_usd"],
        }
    return candidates[0] if candidates else None


def client_id(event_id: str, leg: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9]", "", event_id)[-18:]
    return f"{ORDER_CLIENT_PREFIX}{safe}{leg}"[:32]


def order_client_id_value(order: dict[str, Any]) -> str:
    info = order.get("info") if isinstance(order.get("info"), dict) else {}
    return str(
        order.get("clientOrderId")
        or order.get("client_id")
        or order.get("clientId")
        or info.get("clientOrderId")
        or info.get("origClientOrderId")
        or ""
    )


def is_s34ve_order(order: dict[str, Any]) -> bool:
    return order_client_id_value(order).startswith(ORDER_CLIENT_PREFIX)


def order_is_reduce_only(order: dict[str, Any]) -> bool:
    info = order.get("info") if isinstance(order.get("info"), dict) else {}
    raw = order.get("reduceOnly")
    if raw is None:
        raw = info.get("reduceOnly") or info.get("reduce_only")
    return truthy(raw) or str(raw).lower() == "true"


def order_is_stop(order: dict[str, Any]) -> bool:
    typ = str(order.get("type") or (order.get("info") or {}).get("type") or "").upper()
    cid = order_client_id_value(order)
    return "STOP" in typ or cid.endswith("SL") or cid.endswith("ES")


def s34ve_open_orders(open_orders: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [o for o in open_orders if isinstance(o, dict) and is_s34ve_order(o)]


def has_s34ve_protective_stop(open_orders: list[dict[str, Any]]) -> bool:
    return any(is_s34ve_order(o) and order_is_stop(o) and order_is_reduce_only(o) for o in open_orders)


def order_id_or_client(order: dict[str, Any]) -> str | None:
    oid = order.get("id") or order.get("orderId") or (order.get("info") or {}).get("orderId")
    if oid is not None:
        return str(oid)
    cid = order_client_id_value(order)
    return cid or None


def available_usdt(balance: dict[str, Any]) -> float | None:
    usdt = balance.get("USDT") or balance.get("free", {}).get("USDT") or balance.get("total", {}).get("USDT")
    if isinstance(usdt, dict):
        usdt = usdt.get("free") or usdt.get("total")
    try:
        return float(usdt)
    except (TypeError, ValueError):
        return None


async def fetch_balance_usdt(ex: Any) -> float | None:
    try:
        return available_usdt(await ex.fetch_balance())
    except Exception as exc:  # noqa: BLE001
        print(f"{utc_now()} balance fetch failed: {exc}")
        return None


def position_amount(positions: Any, *, symbol: str, direction: str) -> float:
    target_side = direction.upper()
    for pos in positions or []:
        if not isinstance(pos, dict):
            continue
        raw_symbol = str(pos.get("symbol") or pos.get("info", {}).get("symbol") or "")
        if symbol.replace("USDT", "") not in raw_symbol and symbol not in raw_symbol:
            continue
        info = pos.get("info") or {}
        side = str(pos.get("side") or info.get("positionSide") or "").upper()
        contracts = pos.get("contracts")
        if contracts is None:
            contracts = info.get("positionAmt") or info.get("positionAmt".lower())
        try:
            amt = float(contracts)
        except (TypeError, ValueError):
            continue
        if target_side == "LONG" and (side in {"", "LONG"} or amt > 0) and abs(amt) > 0:
            return abs(amt)
        if target_side == "SHORT" and (side in {"", "SHORT"} or amt < 0) and abs(amt) > 0:
            return abs(amt)
    return 0.0


async def fetch_position_amount(ex: Any) -> float:
    try:
        positions = await ex.fetch_positions([SYMBOL])
    except Exception:
        positions = await ex.fetch_positions()
    return position_amount(positions, symbol=SYMBOL, direction=DIRECTION)


async def fetch_open_orders_safe(ex: Any) -> list[dict[str, Any]]:
    try:
        orders = await ex.fetch_open_orders(SYMBOL)
    except Exception as exc:  # noqa: BLE001
        print(f"{utc_now()} open-order reconciliation failed: {exc}", file=sys.stderr)
        return []
    return [o for o in orders or [] if isinstance(o, dict)]


async def fetch_reference_price(ex: Any) -> float | None:
    try:
        ticker = await ex.fetch_ticker(SYMBOL)
    except Exception as exc:  # noqa: BLE001
        print(f"{utc_now()} ticker fetch failed for emergency stop: {exc}", file=sys.stderr)
        return None
    for key in ("mark", "last", "close", "bid", "ask"):
        value = ticker.get(key) if isinstance(ticker, dict) else None
        try:
            px = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(px) and px > 0:
            return px
    info = ticker.get("info") if isinstance(ticker, dict) and isinstance(ticker.get("info"), dict) else {}
    for key in ("markPrice", "lastPrice", "price"):
        try:
            px = float(info.get(key))
        except (TypeError, ValueError):
            continue
        if math.isfinite(px) and px > 0:
            return px
    return None


async def precision_amount_price(ex: Any, *, amount: float, price: float) -> tuple[str, str]:
    await ex.fetch_markets()
    return ex.amount_to_precision(SYMBOL, float(amount)), ex.price_to_precision(SYMBOL, float(price))


async def place_limit(ex: Any, *, event: dict[str, Any], offset_bps: float, amount: float, dry_run: bool, leg: str) -> dict[str, Any]:
    price = maker_limit_price(float(event["anchor_mark_price"]), DIRECTION, float(offset_bps))
    amount_p, price_p = await precision_amount_price(ex, amount=amount, price=price)
    if not dry_run:
        await set_leverage_if_possible(ex, SYMBOL, int(event["leverage"]), dry_run)
    params = {
        "newClientOrderId": client_id(str(event["event_id"]), leg),
        "positionSide": "LONG",
        "timeInForce": "GTX",
    }
    order = await ex.create_order(SYMBOL, "limit", "buy", amount_p, price_p, params=params)
    return {
        "order": order,
        "client_id": params["newClientOrderId"],
        "limit_price": float(price),
        "limit_price_precise": price_p,
        "amount": amount_p,
        "offset_bps": float(offset_bps),
        "placed_at_utc": utc_now(),
    }


async def cancel_order_best_effort(ex: Any, order_id: str | None, symbol: str) -> dict[str, Any] | None:
    if not order_id:
        return None
    try:
        return await ex.cancel_order(str(order_id), symbol)
    except Exception as exc:  # noqa: BLE001
        print(f"{utc_now()} cancel best-effort failed order={order_id}: {exc}")
        return {"error": str(exc)}


async def cancel_stale_s34ve_orders(ex: Any, open_orders: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cancelled = []
    for order in s34ve_open_orders(open_orders):
        oid = order_id_or_client(order)
        result = await cancel_order_best_effort(ex, oid, SYMBOL)
        cancelled.append({"order_id": oid, "client_id": order_client_id_value(order), "result": result})
    return cancelled


def order_id(result: dict[str, Any] | None) -> str | None:
    if not isinstance(result, dict):
        return None
    order = result.get("order") if "order" in result else result
    if not isinstance(order, dict):
        return None
    oid = order.get("id") or order.get("orderId") or (order.get("info") or {}).get("orderId")
    return str(oid) if oid is not None else None


async def close_position_market(ex: Any, *, amount: float, dry_run: bool) -> dict[str, Any]:
    amount_p = ex.amount_to_precision(SYMBOL, float(amount))
    return await ex.create_order(
        SYMBOL,
        "market",
        "sell",
        amount_p,
        params={
            "reduceOnly": True,
            "positionSide": "LONG",
            "newClientOrderId": client_id(f"{RULE_NAME}{int(time.time())}", "X"),
        },
    )


async def place_emergency_stop_market(
    ex: Any,
    *,
    amount: float,
    reference_price: float,
    stop_bps: float,
    event_id: str,
) -> dict[str, Any]:
    if float(reference_price) <= 0:
        raise RuntimeError("cannot place emergency stop without reference price")
    stop_px = stop_price_long(float(reference_price), float(stop_bps))
    amount_p, stop_p = await precision_amount_price(ex, amount=float(amount), price=stop_px)
    result = await ex.create_order(
        SYMBOL,
        "STOP_MARKET",
        "sell",
        amount_p,
        params={
            "stopPrice": stop_p,
            "reduceOnly": True,
            "workingType": "MARK_PRICE",
            "positionSide": "LONG",
            "newClientOrderId": client_id(event_id, "ES"),
        },
    )
    return {
        "order": result,
        "stop_bps": float(stop_bps),
        "reference_price": float(reference_price),
        "stop_price": float(stop_px),
        "stop_price_precise": stop_p,
        "amount": amount_p,
        "placed_at_utc": utc_now(),
        "emergency": True,
    }


def stop_price_long(entry_price: float, stop_bps: float) -> float:
    return float(entry_price) * (1.0 - float(stop_bps) / 10_000.0)


def query_follow_on_cascade(
    db_path: Path, anchor_ts_ms: int, lo_ms: int, hi_ms: int
) -> dict[str, Any] | None:
    """Return first ETH SELL cascade ≥SIL_PROP_THRESH_USD in (anchor+lo, anchor+hi], or None."""
    abs_lo = anchor_ts_ms + lo_ms
    abs_hi = anchor_ts_ms + hi_ms
    if abs_lo >= abs_hi:
        return None
    try:
        with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
            row = conn.execute(
                "SELECT ts_ms, notional FROM liquidations"
                " WHERE symbol='ETHUSDT' AND side='SELL'"
                " AND ts_ms>=? AND ts_ms<? AND notional>=?"
                " ORDER BY ts_ms ASC LIMIT 1",
                (abs_lo, abs_hi, SIL_PROP_THRESH_USD),
            ).fetchone()
    except Exception:
        return None
    return {"ts_ms": int(row[0]), "notional": float(row[1])} if row else None


def active_entry_ref(active: dict[str, Any]) -> float | None:
    if active.get("status") == "ENTRY_REPLACED_OPEN" or active.get("fill_leg") == "replacement":
        repl = active.get("replace_order") or {}
        px = repl.get("limit_price")
        if px is not None:
            return float(px)
    initial = active.get("initial_order") or {}
    px = initial.get("limit_price")
    if px is not None:
        return float(px)
    anchor = active.get("anchor_mark_price")
    return float(anchor) if anchor is not None else None


async def place_stop_market(
    ex: Any,
    *,
    active: dict[str, Any],
    amount: float,
    stop_bps: float,
    dry_run: bool,
) -> dict[str, Any]:
    entry_ref = active_entry_ref(active)
    if entry_ref is None or float(entry_ref) <= 0:
        raise RuntimeError("cannot place V Engine stop without entry reference")
    stop_px = stop_price_long(float(entry_ref), float(stop_bps))
    amount_p, stop_p = await precision_amount_price(ex, amount=float(amount), price=stop_px)
    result = await ex.create_order(
        SYMBOL,
        "STOP_MARKET",
        "sell",
        amount_p,
        params={
            "stopPrice": stop_p,
            "reduceOnly": True,
            "workingType": "MARK_PRICE",
            "positionSide": "LONG",
            "newClientOrderId": client_id(str(active.get("event_id") or RULE_NAME), "SL"),
        },
    )
    return {
        "order": result,
        "stop_bps": float(stop_bps),
        "entry_ref_price": float(entry_ref),
        "stop_price": float(stop_px),
        "stop_price_precise": stop_p,
        "amount": amount_p,
        "placed_at_utc": utc_now(),
        "dry_run": bool(dry_run),
    }


def planned_margin(env: dict[str, str], balance: float | None) -> tuple[float, int]:
    leverage = int(fenv(env, "S34_LIVE_MAX_LEVERAGE", 1.0))
    if balance is not None and balance > 0:
        pct = fenv(env, "S34_LIVE_MARGIN_PCT_ETH", fenv(env, "S34_LIVE_MARGIN_PCT", 10.0))
        margin = round(float(balance) * pct / 100.0, 2)
    else:
        margin = fenv(env, "S34_LIVE_MARGIN_USDT", 30.0)
    return float(margin), int(leverage)


def has_active_order(state: dict[str, Any]) -> bool:
    return isinstance(state.get("active"), dict) and str(state["active"].get("status") or "") in {
        "ENTRY_INITIAL_OPEN",
        "ENTRY_REPLACED_OPEN",
        "POSITION_OPEN",
        "EXIT_SUBMITTED",
    }


async def reconcile_exchange_safety(env: dict[str, str], state: dict[str, Any], ex: Any, dry_run: bool) -> dict[str, Any]:
    open_orders = await fetch_open_orders_safe(ex)
    s34_orders = s34ve_open_orders(open_orders)
    pos_amt = await fetch_position_amount(ex)
    state["reconciliation"] = {
        "updated_at_utc": utc_now(),
        "position_amount": pos_amt,
        "s34ve_open_order_count": len(s34_orders),
        "s34ve_open_client_ids": [order_client_id_value(o) for o in s34_orders],
    }

    if not has_active_order(state) and pos_amt <= 0 and s34_orders:
        cancelled = (
            [{"order_id": order_id_or_client(o), "client_id": order_client_id_value(o), "dry_run": True} for o in s34_orders]
            if dry_run
            else await cancel_stale_s34ve_orders(ex, s34_orders)
        )
        state["orders"].append({"event": "RECONCILE", "action": "cancel_stale_s34ve_orders", "result": cancelled})
        state["reconciliation"]["stale_orders_cancelled"] = len(cancelled)
        state["status"]["duplicate_guard"] = "cancelled_stale_s34ve_orders"
        return {"allow_new_entry": False, "reason": "cancelled_stale_s34ve_orders"}

    if not has_active_order(state) and pos_amt > 0:
        event_id = f"{RULE_NAME}:ORPHAN:{int(time.time())}"
        stop_bps = fenv(env, "S34_V_ENGINE_EMERGENCY_STOP_BPS", fenv(env, "S34_V_ENGINE_LIVE_STOP_BPS", DEFAULT_STOP_BPS))
        existing_stop = has_s34ve_protective_stop(open_orders)
        ref_price = await fetch_reference_price(ex)
        active: dict[str, Any] = {
            "event_id": event_id,
            "rule": RULE_NAME,
            "status": "POSITION_OPEN",
            "orphan_reconciled": True,
            "reconciled_at_utc": utc_now(),
            "position_amount": pos_amt,
            "raw_amount": pos_amt,
            "anchor_mark_price": ref_price,
            "exit_due_ts_ms": int(time.time() * 1000) + HORIZON_SEC * 1000,
            "horizon_sec": HORIZON_SEC,
        }
        if existing_stop:
            active["stop_order"] = {"reconciled_existing": True}
            active["stop_order_id"] = None
            state["reconciliation"]["orphan_position_stop"] = "existing_stop_detected"
        elif not dry_run:
            if ref_price is None:
                raise RuntimeError("orphan position detected but no reference price for emergency stop")
            stop = await place_emergency_stop_market(
                ex,
                amount=pos_amt,
                reference_price=float(ref_price),
                stop_bps=float(stop_bps),
                event_id=event_id,
            )
            active["stop_order"] = stop
            active["stop_order_id"] = order_id(stop)
            state["orders"].append({"event": event_id, "action": "place_orphan_emergency_stop", "result": stop})
            state["reconciliation"]["orphan_position_stop"] = "placed"
        state["active"] = active
        state["processed_event_ids"][event_id] = {"created_at_utc": utc_now(), "status": "ORPHAN_RECONCILED"}
        state["status"]["duplicate_guard"] = "orphan_position_reconciled"
        print(f"{utc_now()} orphan ETH LONG reconciled amount={pos_amt}; new entries blocked this cycle")
        return {"allow_new_entry": False, "reason": "orphan_position_reconciled"}

    if s34_orders and not has_active_order(state):
        state["status"]["duplicate_guard"] = "s34ve_open_orders_present"
        return {"allow_new_entry": False, "reason": "s34ve_open_orders_present"}

    return {"allow_new_entry": True, "reason": "ok"}


async def maybe_open_new_signal(args: argparse.Namespace, env: dict[str, str], state: dict[str, Any], ex: Any, dry_run: bool) -> None:
    if has_active_order(state):
        state["status"]["new_entry_blocked_by"] = "local_active_order"
        return
    if RULE_NAME not in allowed_rules(env):
        state["status"]["allowed"] = False
        state["status"]["new_entry_blocked_by"] = "rule_not_allowed"
        return
    safety = state.get("reconciliation") if isinstance(state.get("reconciliation"), dict) else {}
    if float(safety.get("position_amount") or 0.0) > 0:
        state["status"]["duplicate_guard"] = "exchange_position_present"
        state["status"]["new_entry_blocked_by"] = "exchange_position_present"
        return
    if int(safety.get("s34ve_open_order_count") or 0) > 0:
        state["status"]["duplicate_guard"] = "exchange_s34ve_orders_present"
        state["status"]["new_entry_blocked_by"] = "exchange_s34ve_orders_present"
        return
    scan_diag: dict[str, Any] = {}
    with sqlite3.connect(f"file:{args.db}?mode=ro", uri=True) as conn:
        signal = recent_anchor(
            conn,
            lookback_sec=int(args.lookback_sec),
            signal_fresh_sec=int(args.signal_fresh_sec),
            max_book_staleness_sec=int(fenv(env, "S34_LIVE_REQUIRE_BOOK_FRESH_SEC", 10.0)),
            diagnostics=scan_diag,
        )
    state["status"]["last_signal_scan"] = scan_diag
    if not signal:
        state["status"]["new_entry_blocked_by"] = "no_fresh_eligible_signal"
        missed = scan_diag.get("latest_missed_signal")
        if isinstance(missed, dict):
            state["status"]["last_missed_signal"] = missed
            if state["status"].get("last_missed_signal_event_id_logged") != missed.get("event_id"):
                state["status"]["last_missed_signal_event_id_logged"] = missed.get("event_id")
                print(json.dumps({"event": "MISSED_SIGNAL_FRESHNESS", **missed}, sort_keys=True))
        return
    eid = str(signal["event_id"])
    if eid in (state.get("processed_event_ids") or {}):
        state["status"]["new_entry_blocked_by"] = "duplicate_processed_event"
        state["status"]["duplicate_event_id"] = eid
        return
    balance = await fetch_balance_usdt(ex)
    min_bal = fenv(env, "S34_LIVE_MIN_BALANCE_USDT", 0.0)
    if min_bal > 0 and balance is not None and balance < min_bal:
        print(f"{utc_now()} min balance guard: {balance:.2f} < {min_bal:.2f}")
        state["status"]["new_entry_blocked_by"] = "min_balance_guard"
        state["status"]["balance_usdt"] = balance
        state["status"]["min_balance_usdt"] = min_bal
        return
    margin, leverage = planned_margin(env, balance)
    notional = margin * leverage
    amount = notional / float(signal["anchor_mark_price"])
    signal.update({"margin_usdt": margin, "leverage": leverage, "notional_usdt": notional, "raw_amount": amount})
    placed = await place_limit(ex, event=signal, offset_bps=INITIAL_OFFSET_BPS, amount=amount, dry_run=dry_run, leg="E1")
    active = {
        **signal,
        "rule": RULE_NAME,
        "status": "ENTRY_INITIAL_OPEN",
        "opened_at_utc": utc_now(),
        "initial_order": placed,
        "initial_order_id": order_id(placed),
        "replace_at_ms": int(time.time() * 1000) + REPLACE_WAIT_SEC * 1000,
        "horizon_sec": HORIZON_SEC,
    }
    state["active"] = active
    state["processed_event_ids"][eid] = {"created_at_utc": utc_now(), "status": "ENTRY_INITIAL_OPEN"}
    state["orders"].append({"event": eid, "action": "place_initial_limit", "result": placed})
    state["status"]["new_entry_blocked_by"] = None
    state["status"]["last_opened_event_id"] = eid
    state["status"]["last_opened_at_utc"] = utc_now()
    print(json.dumps({"mode": "DRY_RUN" if dry_run else "LIVE", "v_engine_signal": signal, "entry": placed}, indent=2))


async def manage_active(
    env: dict[str, str],
    state: dict[str, Any],
    ex: Any,
    dry_run: bool,
    *,
    db_path: Path | None = None,
) -> None:
    active = state.get("active")
    if not isinstance(active, dict):
        return
    status = str(active.get("status") or "")
    pos_amt = await fetch_position_amount(ex)
    now_ms = int(time.time() * 1000)
    if status in {"ENTRY_INITIAL_OPEN", "ENTRY_REPLACED_OPEN"} and pos_amt > 0:
        active["status"] = "POSITION_OPEN"
        active["fill_leg"] = "replacement" if status == "ENTRY_REPLACED_OPEN" else "initial"
        active["fill_detected_at_utc"] = utc_now()
        active["fill_detected_ts_ms"] = now_ms
        active["position_amount"] = pos_amt
        active["exit_due_ts_ms"] = now_ms + HORIZON_SEC * 1000
        stop_bps = fenv(env, "S34_V_ENGINE_LIVE_STOP_BPS", DEFAULT_STOP_BPS)
        if stop_bps > 0:
            stop = await place_stop_market(ex, active=active, amount=pos_amt, stop_bps=stop_bps, dry_run=dry_run)
            active["stop_order"] = stop
            active["stop_order_id"] = order_id(stop)
            state["orders"].append({"event": active.get("event_id"), "action": "place_protective_stop", "result": stop})
        print(f"{utc_now()} V Engine position detected amount={pos_amt}")
        return
    if status == "ENTRY_INITIAL_OPEN" and now_ms >= int(active.get("replace_at_ms") or 0):
        cancel = await cancel_order_best_effort(ex, active.get("initial_order_id"), SYMBOL)
        placed = await place_limit(
            ex,
            event=active,
            offset_bps=REPLACE_OFFSET_BPS,
            amount=float(active["raw_amount"]),
            dry_run=dry_run,
            leg="E2",
        )
        active["status"] = "ENTRY_REPLACED_OPEN"
        active["initial_cancel_result"] = cancel
        active["replace_order"] = placed
        active["replace_order_id"] = order_id(placed)
        state["orders"].append({"event": active.get("event_id"), "action": "replace_limit", "cancel": cancel, "result": placed})
        print(json.dumps({"mode": "DRY_RUN" if dry_run else "LIVE", "replace": placed}, indent=2))
        return
    # ── Silence gate: check for noisy cascade and exit early, or confirm silence ──
    # Runs every poll cycle while POSITION_OPEN and gate not yet resolved.
    # Ultra-early (<60s): exit immediately (flash crash trap, WR=45.8%).
    # Noisy (60s-30min): exit early to cut the losing LONG.
    # Silence (30min no cascade): extend hold from 2h (fill) to 4h (anchor), WR=70%.
    if (
        status == "POSITION_OPEN"
        and pos_amt > 0
        and db_path is not None
        and not active.get("silence_gate_resolved")
    ):
        anchor_ts_ms = int(active.get("anchor_ts_ms") or active.get("fill_detected_ts_ms") or 0)
        if anchor_ts_ms > 0:
            since_anchor_ms = now_ms - anchor_ts_ms
            if since_anchor_ms >= SIL_GATE_LO_MS:
                # Check for follow-on cascade in [anchor+60s, now] capped at 30min
                follow_on = query_follow_on_cascade(db_path, anchor_ts_ms, SIL_GATE_LO_MS, since_anchor_ms)
                if follow_on is not None:
                    delay_ms = follow_on["ts_ms"] - anchor_ts_ms
                    active["silence_gate_resolved"] = "NOISY"
                    active["noisy_cascade_ts_ms"] = follow_on["ts_ms"]
                    active["noisy_cascade_delay_ms"] = delay_ms
                    amount = pos_amt or float(active.get("position_amount") or 0.0)
                    if amount > 0:
                        cancel_stop = await cancel_order_best_effort(ex, active.get("stop_order_id"), SYMBOL)
                        result = await close_position_market(ex, amount=amount, dry_run=dry_run)
                        active["status"] = "EXIT_SUBMITTED"
                        active["exit_reason"] = "NOISY_EARLY_EXIT"
                        active["exit_submitted_at_utc"] = utc_now()
                        active["exit_order"] = result
                        state["orders"].append({
                            "event": active.get("event_id"),
                            "action": "noisy_early_exit",
                            "noisy_cascade_delay_ms": delay_ms,
                            "cancel_stop": cancel_stop,
                            "result": result,
                        })
                        print(f"{utc_now()} V Engine NOISY early exit cascade_delay={delay_ms/1000:.0f}s notional={follow_on['notional']:.0f}")
                        return
            if since_anchor_ms >= SIL_GATE_HI_MS and not active.get("silence_gate_resolved"):
                # 30min passed with no cascade → silence confirmed, extend to 4h from anchor
                active["silence_gate_resolved"] = "SILENCE"
                new_exit = anchor_ts_ms + SIL_HORIZON_SEC * 1000
                active["exit_due_ts_ms"] = new_exit
                active["silence_confirmed_at_utc"] = utc_now()
                print(f"{utc_now()} V Engine SILENCE confirmed - hold extended to 4h from anchor")
    if status == "POSITION_OPEN" and now_ms >= int(active.get("exit_due_ts_ms") or 0):
        amount = pos_amt or float(active.get("position_amount") or 0.0)
        if amount <= 0:
            active["status"] = "CLOSED_NO_POSITION"
            state["active"] = None
            return
        cancel_stop = await cancel_order_best_effort(ex, active.get("stop_order_id"), SYMBOL)
        active["stop_cancel_result"] = cancel_stop
        result = await close_position_market(ex, amount=amount, dry_run=dry_run)
        active["status"] = "EXIT_SUBMITTED"
        active["exit_submitted_at_utc"] = utc_now()
        active["exit_order"] = result
        state["orders"].append({"event": active.get("event_id"), "action": "time_exit_market", "cancel_stop": cancel_stop, "result": result})
        print(json.dumps({"mode": "DRY_RUN" if dry_run else "LIVE", "time_exit": result}, indent=2))
        return
    if status == "POSITION_OPEN" and pos_amt > 0 and not active.get("stop_order"):
        stop_bps = fenv(env, "S34_V_ENGINE_LIVE_STOP_BPS", DEFAULT_STOP_BPS)
        if stop_bps > 0:
            stop = await place_stop_market(ex, active=active, amount=pos_amt, stop_bps=stop_bps, dry_run=dry_run)
            active["stop_order"] = stop
            active["stop_order_id"] = order_id(stop)
            state["orders"].append({"event": active.get("event_id"), "action": "repair_missing_protective_stop", "result": stop})
            print(f"{utc_now()} V Engine protective stop repaired")
        return
    if status == "EXIT_SUBMITTED":
        if pos_amt <= 0:
            eid = str(active.get("event_id") or "")
            if eid:
                state["processed_event_ids"][eid] = {"closed_at_utc": utc_now(), "status": "CLOSED"}
            state["active"] = None
            print(f"{utc_now()} V Engine position closed")


async def run_once(args: argparse.Namespace, env: dict[str, str], state: dict[str, Any], dry_run: bool) -> None:
    ks_file = env.get("S34_LIVE_KILL_SWITCH_FILE", "")
    ex = get_exchange()
    try:
        await manage_active(env, state, ex, dry_run, db_path=args.db)
        reconciliation = await reconcile_exchange_safety(env, state, ex, dry_run)
        if ks_file and Path(ks_file).exists():
            state["status"]["kill_switch"] = "active"
            print(f"{utc_now()} KILL SWITCH active ({ks_file}) - new entries blocked")
        elif reconciliation.get("allow_new_entry"):
            await maybe_open_new_signal(args, env, state, ex, dry_run)
        else:
            state["status"]["new_entry_blocked_by"] = reconciliation.get("reason")
        state["status"] = {
            **(state.get("status") if isinstance(state.get("status"), dict) else {}),
            "updated_at_utc": utc_now(),
            "mode": "DRY_RUN" if dry_run else "LIVE",
            "rule": RULE_NAME,
            "active_status": None if not isinstance(state.get("active"), dict) else state["active"].get("status"),
            "allowed_rules": sorted(allowed_rules(env)),
        }
        save_state(args.state_path, state)
    finally:
        await ex.close()


async def async_main(args: argparse.Namespace) -> int:
    env = load_env(args.env)
    dry_run = not assert_live_allowed(env, args)
    state = load_state(args.state_path)
    PID_PATH.parent.mkdir(parents=True, exist_ok=True)
    PID_PATH.write_text(str(os.getpid()), encoding="utf-8")
    if args.once:
        await run_once(args, env, state, dry_run)
        return 0
    poll = fenv(env, "S34_LIVE_POLL_SEC", 2.0)
    print(f"{utc_now()} S34 V Engine live executor started mode={'DRY_RUN' if dry_run else 'LIVE'} poll={poll}s")
    consecutive_errors = 0
    while True:
        try:
            await run_once(args, env, state, dry_run)
            consecutive_errors = 0
            await asyncio.sleep(poll)
        except RuntimeError as exc:
            if "live mode requires" in str(exc):
                print(f"{utc_now()} fatal config error: {exc}", file=sys.stderr)
                raise
            consecutive_errors += 1
            backoff = min(poll * (2 ** consecutive_errors), 60.0)
            print(f"{utc_now()} error {consecutive_errors}/10 - retrying in {backoff:.0f}s: {type(exc).__name__}: {exc!r}", file=sys.stderr)
            print(traceback.format_exc(), file=sys.stderr)
            if consecutive_errors >= 10:
                raise
            await asyncio.sleep(backoff)
        except Exception as exc:  # noqa: BLE001
            consecutive_errors += 1
            backoff = min(poll * (2 ** consecutive_errors), 60.0)
            print(f"{utc_now()} error {consecutive_errors}/10 - retrying in {backoff:.0f}s: {type(exc).__name__}: {exc!r}", file=sys.stderr)
            print(traceback.format_exc(), file=sys.stderr)
            if consecutive_errors >= 10:
                raise
            await asyncio.sleep(backoff)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the S34 V Engine live maker executor.")
    parser.add_argument("--env", type=Path, default=ENV_PATH)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--state-path", type=Path, default=DEFAULT_STATE_PATH)
    parser.add_argument("--lookback-sec", type=int, default=1800)
    parser.add_argument("--signal-fresh-sec", type=int, default=DEFAULT_SIGNAL_FRESH_SEC)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--confirm-live-orders", action="store_true")
    try:
        return asyncio.run(async_main(parser.parse_args()))
    except KeyboardInterrupt:
        return 130
    except Exception as exc:  # noqa: BLE001
        print(f"error: {type(exc).__name__}: {exc!r}", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
