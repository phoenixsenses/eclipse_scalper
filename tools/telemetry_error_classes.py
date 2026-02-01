#!/usr/bin/env python3
"""
Telemetry error-class dashboard (JSONL).
"""

from __future__ import annotations

import argparse
import json
import time
import re
from pathlib import Path


DEFAULT_EVENTS = {
    "order.create_failed",
    "order.create.retry_alert",
    "order.blocked",
    "entry.blocked",
    "exit.blocked",
    "entry.error",
    "exit.error",
    "data.quality.roll_alert",
}


def _load_jsonl(path: Path, limit: int = 5000):
    if not path.exists():
        return []
    out = []
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except Exception:
                    continue
                if len(out) >= limit:
                    break
    except Exception:
        return out
    return out


def _parse_thresholds(raw: str) -> dict:
    if not raw:
        return {}
    out = {}
    parts = [p.strip() for p in raw.replace(";", ",").split(",") if p.strip()]
    for p in parts:
        if "=" not in p:
            continue
        k, v = p.split("=", 1)
        k = k.strip().lower()
        try:
            out[k] = int(float(v))
        except Exception:
            continue
    return out


def _get_symbol(ev: dict) -> str:
    if not isinstance(ev, dict):
        return ""
    sym = ev.get("symbol") or (ev.get("data") or {}).get("symbol") or (ev.get("data") or {}).get("k")
    return str(sym or "").upper()


def _error_text(ev: dict) -> str:
    try:
        data = ev.get("data") or {}
        err = data.get("err") or ""
        reason = data.get("reason") or ""
        code = data.get("code") or ""
        return f"{err} {reason} {code}".lower()
    except Exception:
        return ""


def _extract_binance_code(msg: str) -> int | None:
    try:
        hits = re.findall(r"-\d{4,5}", msg)
        if not hits:
            return None
        return int(hits[0])
    except Exception:
        return None


def _classify_from_code(code: str) -> str | None:
    c = str(code or "").upper()
    if not c:
        return None
    if "MARGIN" in c:
        return "margin"
    if "NOTIONAL" in c:
        return "min_notional"
    if "SPREAD" in c:
        return "spread"
    if "IMPACT" in c:
        return "market_impact"
    if "PRICE" in c:
        return "price_filter"
    if "STOP" in c:
        return "stop_price"
    if "SYMBOL" in c:
        return "invalid_symbol"
    if "REDUCEONLY" in c:
        return "reduceonly"
    if "POSITION" in c or "HEDGE" in c:
        return "position_side"
    if "STALE" in c:
        return "stale_data"
    if "QUALITY" in c:
        return "data_quality"
    if "NETWORK" in c:
        return "network"
    if "AUTH" in c:
        return "auth"
    if "TIMESTAMP" in c:
        return "timestamp"
    if "EXCHANGE_BUSY" in c:
        return "exchange_busy"
    if "ROUTER_BLOCK" in c:
        return "router_block"
    if "UNKNOWN" in c:
        return "unknown"
    return None


def _classify_event(ev: dict) -> str:
    data = ev.get("data") or {}
    code = data.get("code")
    reason = str(data.get("reason") or "").lower()
    msg = _error_text(ev)

    code_class = _classify_from_code(code)
    if code_class:
        return code_class

    if "filter failure" in msg and "price_filter" in msg:
        return "price_filter"
    if "filter failure" in msg and "min_notional" in msg:
        return "min_notional"
    if "filter failure" in msg and "lot_size" in msg:
        return "lot_size"

    if "position side" in msg or "dual side" in msg or "hedge mode" in msg:
        return "position_side"
    if "reduceonly" in msg and ("rejected" in msg or "not required" in msg):
        return "reduceonly"

    bin_code = _extract_binance_code(msg)
    if bin_code is not None:
        if bin_code in (-1000, -1001, -1003, -1006, -1007, -1008):
            return "exchange_busy"
        if bin_code in (-1021,):
            return "timestamp"
        if bin_code in (-1022,):
            return "auth"
        if bin_code in (-1100, -1101, -1102, -1103):
            return "invalid_params"
        if bin_code in (-1111, -1112):
            return "price_filter"
        if bin_code in (-1121,):
            return "invalid_symbol"
        if bin_code in (-2019,):
            return "margin"
        if bin_code in (-2021,):
            return "stop_price"

    if "network" in msg or "timeout" in msg or "timed out" in msg or "connection" in msg:
        return "network"
    if "margin" in msg or "insufficient" in msg or "balance" in msg:
        return "margin"
    if "notional" in msg or "min notional" in msg:
        return "min_notional"
    if "price" in msg and "filter" in msg:
        return "price_filter"
    if "stop price" in msg or "order would trigger immediately" in msg:
        return "stop_price"
    if "symbol not found" in msg or "invalid symbol" in msg:
        return "invalid_symbol"
    if "spread" in msg:
        return "spread"
    if "impact" in msg:
        return "market_impact"
    if reason:
        return reason.replace(" ", "_")
    return "unknown"


def main() -> int:
    parser = argparse.ArgumentParser(description="Telemetry error class dashboard")
    parser.add_argument("--path", default="logs/telemetry.jsonl", help="Path to telemetry.jsonl")
    parser.add_argument("--limit", type=int, default=5000, help="Max events to load from JSONL")
    parser.add_argument("--since-min", type=int, default=60, help="Only include events from last N minutes (0 = all)")
    parser.add_argument("--top", type=int, default=10, help="Top classes to show")
    parser.add_argument("--min", type=int, default=1, help="Minimum count to display")
    parser.add_argument("--thresholds", default="", help="Thresholds, e.g. network=5,margin=1")
    parser.add_argument("--events", default="", help="Comma-separated events to include (default: common error events)")
    args = parser.parse_args()

    path = Path(args.path)
    events = _load_jsonl(path, limit=max(1, int(args.limit)))
    since_min = int(args.since_min)
    if since_min > 0:
        cutoff = time.time() - (since_min * 60)
        events = [ev for ev in events if float(ev.get("ts") or 0.0) >= cutoff]

    if args.events.strip():
        wanted = {e.strip() for e in args.events.split(",") if e.strip()}
    else:
        wanted = set(DEFAULT_EVENTS)

    events = [ev for ev in events if str(ev.get("event")) in wanted]

    print("Telemetry Error Classes (Snapshot)")
    print(f"- File events loaded: {len(events)} from {path}")
    print(f"- Window: last {since_min} min" if since_min > 0 else "- Window: all")
    print(f"- Events: {', '.join(sorted(wanted))}")

    if not events:
        return 0

    class_counts = {}
    class_symbols = {}
    for ev in events:
        cls = _classify_event(ev)
        class_counts[cls] = class_counts.get(cls, 0) + 1
        sym = _get_symbol(ev)
        if sym:
            class_symbols.setdefault(cls, {})
            class_symbols[cls][sym] = class_symbols[cls].get(sym, 0) + 1

    print("- Top classes:")
    shown = 0
    for cls, cnt in sorted(class_counts.items(), key=lambda kv: kv[1], reverse=True):
        if cnt < int(args.min):
            continue
        syms = class_symbols.get(cls, {})
        top_syms = sorted(syms.items(), key=lambda kv: kv[1], reverse=True)[:3]
        sym_str = ", ".join([f"{s}={n}" for s, n in top_syms]) if top_syms else "-"
        print(f"  {cls}: {cnt} (top symbols: {sym_str})")
        shown += 1
        if shown >= int(args.top):
            break

    thresholds = _parse_thresholds(args.thresholds)
    if thresholds:
        print("- Threshold alerts:")
        for cls, th in thresholds.items():
            cnt = class_counts.get(cls, 0)
            if cnt >= th:
                print(f"  ALERT {cls}: {cnt} >= {th}")
            else:
                print(f"  OK {cls}: {cnt} < {th}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
