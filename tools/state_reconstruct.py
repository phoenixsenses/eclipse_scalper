from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List

from tools.replay_strategy import extract_events


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


def _ts_iso_to_sec(s: str) -> float:
    # ISO format in repo is deterministic UTC with Z.
    txt = str(s or "").strip()
    if txt.endswith("Z"):
        txt = txt[:-1] + "+00:00"
    from datetime import datetime

    return datetime.fromisoformat(txt).timestamp()


def _get_trade_side(payload: Dict[str, Any]) -> float:
    side = str(payload.get("side", "")).strip().lower()
    if side in ("buy", "b"):
        return 1.0
    if side in ("sell", "s"):
        return -1.0
    # Binance agg trade convention: is_buyer_maker=True => taker sell.
    bm = payload.get("is_buyer_maker")
    if bm is None:
        return 0.0
    try:
        return -1.0 if int(bm) == 1 else 1.0
    except Exception:
        return 0.0


def reconstruct_state_vectors_from_events(
    events: List[Dict[str, Any]],
    window_sec: float = 30.0,
) -> List[Dict[str, Any]]:
    by_symbol_trades: Dict[str, Deque[Dict[str, Any]]] = {}
    by_symbol_marks: Dict[str, Deque[Dict[str, Any]]] = {}
    by_symbol_liqs: Dict[str, Deque[Dict[str, Any]]] = {}
    out: List[Dict[str, Any]] = []

    w = float(max(1.0, window_sec))

    for ev in events:
        sym = str(ev.get("symbol") or "ALL").upper()
        ts_utc = str(ev.get("ts_utc") or "")
        ts = _ts_iso_to_sec(ts_utc)
        table = str(ev.get("source_table") or "")
        payload = ev.get("payload") if isinstance(ev.get("payload"), dict) else {}

        tq = by_symbol_trades.setdefault(sym, deque())
        mq = by_symbol_marks.setdefault(sym, deque())
        lq = by_symbol_liqs.setdefault(sym, deque())

        if table == "agg_trades":
            qty = _safe_float(payload.get("quantity", payload.get("qty", payload.get("size", 0.0))), 0.0)
            px = _safe_float(payload.get("price", payload.get("p", 0.0)), 0.0)
            side = _get_trade_side(payload)
            tq.append({"ts": ts, "qty": qty, "px": px, "side": side})
        elif table == "mark_prices":
            mark = _safe_float(payload.get("mark_price", payload.get("price", payload.get("p", 0.0))), 0.0)
            mq.append({"ts": ts, "mark": mark})
        elif table == "liquidations":
            qty = _safe_float(payload.get("quantity", payload.get("qty", payload.get("size", 0.0))), 0.0)
            lq.append({"ts": ts, "qty": qty})

        # prune rolling window
        cutoff = ts - w
        while tq and float(tq[0]["ts"]) < cutoff:
            tq.popleft()
        while mq and float(mq[0]["ts"]) < cutoff:
            mq.popleft()
        while lq and float(lq[0]["ts"]) < cutoff:
            lq.popleft()

        # build state vector components
        buy_qty = 0.0
        sell_qty = 0.0
        trade_qty_sum = 0.0
        trade_n = 0
        last_trade_px = 0.0
        for tr in tq:
            q = float(tr["qty"])
            s = float(tr["side"])
            trade_qty_sum += abs(q)
            trade_n += 1
            if q > 0:
                if s >= 0:
                    buy_qty += q
                else:
                    sell_qty += q
            if float(tr["px"]) > 0:
                last_trade_px = float(tr["px"])
        ofi = (buy_qty - sell_qty) / max(1e-9, (buy_qty + sell_qty))
        trade_rate = trade_n / w
        liquidity_pressure = ofi

        mark_last = 0.0
        mark_prev = 0.0
        absrets: List[float] = []
        for m in mq:
            val = float(m["mark"])
            if val <= 0:
                continue
            if mark_last <= 0:
                mark_last = val
            else:
                prev = mark_last
                if prev > 0:
                    absrets.append(abs((val / prev) - 1.0))
                mark_last = val
        if mq:
            first = next((float(x["mark"]) for x in mq if float(x["mark"]) > 0), 0.0)
            mark_prev = first
        spread_proxy = 0.0
        if mark_last > 0 and last_trade_px > 0:
            spread_proxy = abs(last_trade_px - mark_last) / mark_last
        vol_proxy = (sum(absrets) / len(absrets)) if absrets else 0.0
        ret_window = ((mark_last / mark_prev) - 1.0) if (mark_last > 0 and mark_prev > 0) else 0.0

        liq_n = len(lq)
        liq_qty_sum = sum(abs(float(x["qty"])) for x in lq)

        out.append(
            {
                "ts_utc": ts_utc,
                "symbol": sym,
                "source_table": table,
                "state": {
                    "order_flow_imbalance": float(round(ofi, 12)),
                    "spread_proxy": float(round(spread_proxy, 12)),
                    "liquidity_pressure": float(round(liquidity_pressure, 12)),
                    "trade_rate": float(round(trade_rate, 12)),
                    "vol_proxy": float(round(vol_proxy, 12)),
                    "ret_window": float(round(ret_window, 12)),
                    "mark_last": float(round(mark_last, 12)),
                    "trade_n_window": int(trade_n),
                    "trade_qty_window": float(round(trade_qty_sum, 12)),
                    "liq_n_window": int(liq_n),
                    "liq_qty_window": float(round(liq_qty_sum, 12)),
                },
            }
        )
    return out


def reconstruct_state_vectors(
    db: Path,
    symbols: Iterable[str],
    start_iso: str,
    end_iso: str,
    window_sec: float = 30.0,
) -> List[Dict[str, Any]]:
    events = extract_events(
        db=db,
        symbols=list(symbols),
        start_iso=start_iso,
        end_iso=end_iso,
    )
    return reconstruct_state_vectors_from_events(events, window_sec=window_sec)


def write_state_vectors_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=True, sort_keys=True, separators=(",", ":")))
            f.write("\n")


def _parse_symbols(raw: str) -> List[str]:
    return [s.strip().upper() for s in str(raw or "").replace(";", ",").split(",") if s.strip()]


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Deterministic market state reconstruction from replay events.")
    p.add_argument("--db", default="data/microstructure.db")
    p.add_argument("--symbols", default="ETHUSDT")
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--window-sec", type=float, default=30.0)
    p.add_argument("--out", required=True)
    return p


def main() -> int:
    args = _parser().parse_args()
    try:
        rows = reconstruct_state_vectors(
            db=Path(str(args.db)),
            symbols=_parse_symbols(args.symbols),
            start_iso=str(args.start),
            end_iso=str(args.end),
            window_sec=float(args.window_sec),
        )
        out = Path(str(args.out))
        write_state_vectors_jsonl(out, rows)
        print(f"state_reconstruct ok rows={len(rows)} out={out}")
        return 0
    except Exception as e:
        print(f"state_reconstruct error runtime={type(e).__name__}:{e}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

