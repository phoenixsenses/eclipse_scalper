from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        if v != v:  # nan
            return float(default)
        return v
    except Exception:
        return float(default)


def init_portfolio_state(starting_equity: float) -> Dict[str, Any]:
    return {
        "starting_equity": float(starting_equity),
        "realized_pnl": 0.0,
        "peak_equity": float(starting_equity),
        "kill_until_ts_ms": 0,
        "symbols": {},
    }


def apply_fill(
    state: Dict[str, Any],
    *,
    symbol: str,
    side: str,
    fill_price: float,
    notional: float,
    fee_notional: float = 0.0,
) -> None:
    sym = str(symbol)
    px = max(1e-12, _safe_float(fill_price))
    notional = max(0.0, _safe_float(notional))
    if notional <= 0:
        return
    dq = (notional / px) * (1.0 if str(side).lower() == "buy" else -1.0)
    slots = dict(state.get("symbols", {}) or {})
    cur = dict(slots.get(sym, {"qty": 0.0, "avg_entry_price": px}))
    old_qty = _safe_float(cur.get("qty"))
    old_avg = max(1e-12, _safe_float(cur.get("avg_entry_price"), px))
    realized = _safe_float(state.get("realized_pnl"))
    if old_qty == 0.0 or (old_qty > 0 and dq > 0) or (old_qty < 0 and dq < 0):
        new_qty = old_qty + dq
        new_avg = (abs(old_qty) * old_avg + abs(dq) * px) / max(1e-12, (abs(old_qty) + abs(dq)))
    else:
        closed = min(abs(old_qty), abs(dq))
        realized += (1.0 if old_qty > 0 else -1.0) * (px - old_avg) * closed
        new_qty = old_qty + dq
        if new_qty == 0.0:
            new_avg = 0.0
        elif (old_qty > 0 and new_qty > 0) or (old_qty < 0 and new_qty < 0):
            new_avg = old_avg
        else:
            new_avg = px
    realized -= abs(_safe_float(fee_notional))
    cur["qty"] = float(new_qty)
    cur["avg_entry_price"] = float(new_avg)
    slots[sym] = cur
    state["symbols"] = slots
    state["realized_pnl"] = float(realized)


def mark_to_market(state: Dict[str, Any], mid_by_symbol: Dict[str, float]) -> Dict[str, Any]:
    start_eq = _safe_float(state.get("starting_equity"), 0.0)
    realized = _safe_float(state.get("realized_pnl"), 0.0)
    unrealized = 0.0
    gross_notional = 0.0
    by_symbol: Dict[str, Dict[str, float]] = {}
    for sym, row in dict(state.get("symbols", {}) or {}).items():
        qty = _safe_float(row.get("qty"))
        avg = _safe_float(row.get("avg_entry_price"))
        mid = _safe_float(mid_by_symbol.get(sym), avg if avg > 0 else 0.0)
        u = qty * (mid - avg)
        n = abs(qty * mid)
        unrealized += u
        gross_notional += n
        by_symbol[str(sym)] = {
            "qty": qty,
            "avg_entry_price": avg,
            "mid": mid,
            "unrealized_pnl": u,
            "notional": n,
        }
    equity = start_eq + realized + unrealized
    peak = max(_safe_float(state.get("peak_equity"), equity), equity)
    state["peak_equity"] = float(peak)
    drawdown_pct = 0.0 if peak <= 0 else max(0.0, (peak - equity) / peak)
    return {
        "equity": float(equity),
        "realized_pnl": float(realized),
        "unrealized_pnl": float(unrealized),
        "gross_notional": float(gross_notional),
        "drawdown_pct": float(drawdown_pct),
        "by_symbol": by_symbol,
    }

