from __future__ import annotations

import argparse
import json
import math
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "microstructure.db"
WINDOW_PATH = ROOT / "reports" / "runtime_validation" / "s34_liq_restore_window.json"
OUT_JSON = ROOT / "reports" / "research" / "s34" / "S34_RESTORED_WINDOW_REPLAY_2026-06-07.json"
OUT_MD = ROOT / "reports" / "research" / "s34" / "S34_RESTORED_WINDOW_REPLAY_2026-06-07.md"


@dataclass
class Event:
    ts_ms: int
    symbol: str
    side: str
    notional_5m: float
    count_5m: int
    entry_price: float


def utc_ms(raw: str) -> int:
    return int(datetime.fromisoformat(raw.replace("Z", "+00:00")).timestamp() * 1000)


def iso(ts_ms: int) -> str:
    return datetime.fromtimestamp(ts_ms / 1000, timezone.utc).isoformat()


def load_window() -> tuple[int, int]:
    payload = json.loads(WINDOW_PATH.read_text(encoding="utf-8"))
    start = utc_ms(payload["window_start_utc"])
    # Analyze only mature data, leave a 60m tail for forward horizons.
    end = int(datetime.now(timezone.utc).timestamp() * 1000) - 60 * 60 * 1000
    return start, max(start, end)


def scalar(con: sqlite3.Connection, sql: str, params: tuple[Any, ...]) -> Any:
    row = con.execute(sql, params).fetchone()
    return row[0] if row else None


def price_at(con: sqlite3.Connection, symbol: str, ts_ms: int) -> float | None:
    row = con.execute(
        "SELECT mark_price FROM mark_prices WHERE symbol=? AND ts_ms<=? ORDER BY ts_ms DESC LIMIT 1",
        (symbol, ts_ms),
    ).fetchone()
    return None if row is None else float(row[0])


def min_max_price(con: sqlite3.Connection, symbol: str, start_ms: int, end_ms: int) -> tuple[float | None, float | None]:
    row = con.execute(
        "SELECT MIN(mark_price), MAX(mark_price) FROM mark_prices WHERE symbol=? AND ts_ms BETWEEN ? AND ?",
        (symbol, start_ms, end_ms),
    ).fetchone()
    if row is None:
        return None, None
    return (None if row[0] is None else float(row[0]), None if row[1] is None else float(row[1]))


def detect_events(
    con: sqlite3.Connection,
    symbol: str,
    side: str,
    start_ms: int,
    end_ms: int,
    threshold: float,
    min_gap_sec: int,
) -> list[Event]:
    rows = con.execute(
        """
        SELECT ts_ms, SUM(notional) AS notional_5m, COUNT(*) AS count_5m
        FROM liquidations
        WHERE symbol=? AND side=? AND ts_ms BETWEEN ? AND ?
        GROUP BY ts_ms / 300000
        HAVING notional_5m >= ?
        ORDER BY ts_ms
        """,
        (symbol, side, start_ms, end_ms, threshold),
    ).fetchall()
    events: list[Event] = []
    last = 0
    for ts_ms, notional, count in rows:
        ts = int(ts_ms)
        if ts - last < min_gap_sec * 1000:
            continue
        px = price_at(con, symbol, ts)
        if px is None:
            continue
        events.append(Event(ts, symbol, side, float(notional or 0.0), int(count or 0), float(px)))
        last = ts
    return events


def signed_return_bps(entry: float, exit_price: float, direction: str) -> float:
    if direction == "SHORT":
        return (entry - exit_price) / entry * 10000.0
    return (exit_price - entry) / entry * 10000.0


def evaluate_fixed(con: sqlite3.Connection, events: Iterable[Event], direction: str, horizon_sec: int, cost_bps: float) -> dict[str, Any]:
    returns = []
    for ev in events:
        xp = price_at(con, ev.symbol, ev.ts_ms + horizon_sec * 1000)
        if xp is None:
            continue
        returns.append(signed_return_bps(ev.entry_price, xp, direction) - cost_bps)
    return summarize_returns(returns)


def evaluate_stop_route(
    con: sqlite3.Connection,
    events: Iterable[Event],
    direction: str,
    horizon_sec: int,
    tp_bps: float,
    sl_bps: float,
    be_trigger_bps: float | None,
    cost_bps: float,
) -> dict[str, Any]:
    returns = []
    exits = {"tp": 0, "sl": 0, "be": 0, "time": 0}
    for ev in events:
        start = ev.ts_ms
        end = ev.ts_ms + horizon_sec * 1000
        rows = con.execute(
            "SELECT ts_ms, mark_price FROM mark_prices WHERE symbol=? AND ts_ms BETWEEN ? AND ? ORDER BY ts_ms",
            (ev.symbol, start, end),
        ).fetchall()
        if not rows:
            continue
        entry = ev.entry_price
        be_active = False
        exit_ret = None
        for _, px_raw in rows:
            px = float(px_raw)
            raw = signed_return_bps(entry, px, direction)
            if be_trigger_bps is not None and raw >= be_trigger_bps:
                be_active = True
            if raw >= tp_bps:
                exit_ret = tp_bps - cost_bps
                exits["tp"] += 1
                break
            if raw <= -sl_bps:
                if be_active:
                    exit_ret = 0.0 - cost_bps
                    exits["be"] += 1
                else:
                    exit_ret = -sl_bps - cost_bps
                    exits["sl"] += 1
                break
        if exit_ret is None:
            xp = float(rows[-1][1])
            exit_ret = signed_return_bps(entry, xp, direction) - cost_bps
            exits["time"] += 1
        returns.append(exit_ret)
    out = summarize_returns(returns)
    out["exits"] = exits
    out["tp_bps"] = tp_bps
    out["sl_bps"] = sl_bps
    out["be_trigger_bps"] = be_trigger_bps
    return out


def summarize_returns(returns: list[float]) -> dict[str, Any]:
    if not returns:
        return {"n": 0, "wr": None, "mean_net_bps": None, "median_net_bps": None, "total_net_bps": 0.0}
    rs = sorted(returns)
    return {
        "n": len(returns),
        "wr": sum(1 for r in returns if r > 0) / len(returns),
        "mean_net_bps": mean(returns),
        "median_net_bps": rs[len(rs) // 2],
        "total_net_bps": sum(returns),
        "best_bps": max(returns),
        "worst_bps": min(returns),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(DB_PATH))
    ap.add_argument("--symbols", default="ETHUSDT")
    ap.add_argument("--thresholds", default="25000,50000,100000,200000")
    ap.add_argument("--cost-bps", type=float, default=8.0)
    args = ap.parse_args()

    start_ms, end_ms = load_window()
    con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True, timeout=30)
    payload: dict[str, Any] = {
        "window": {"start_utc": iso(start_ms), "end_utc": iso(end_ms)},
        "cost_bps": float(args.cost_bps),
        "results": [],
    }
    try:
        for symbol in [s.strip().upper() for s in args.symbols.split(",") if s.strip()]:
            for threshold in [float(x) for x in args.thresholds.split(",") if x.strip()]:
                for liq_side in ("BUY", "SELL"):
                    events = detect_events(con, symbol, liq_side, start_ms, end_ms, threshold, min_gap_sec=900)
                    for direction in ("SHORT", "LONG"):
                        item = {
                            "symbol": symbol,
                            "liq_side": liq_side,
                            "direction": direction,
                            "threshold": threshold,
                            "n_events": len(events),
                            "sample_events": [asdict(e) | {"ts_utc": iso(e.ts_ms)} for e in events[:10]],
                            "fixed_horizon": {},
                            "stop_routes": [],
                        }
                        for h in (300, 900, 1800, 3600):
                            item["fixed_horizon"][str(h)] = evaluate_fixed(con, events, direction, h, float(args.cost_bps))
                        for tp in (40.0, 60.0, 80.0, 120.0):
                            for sl in (40.0, 60.0, 80.0):
                                item["stop_routes"].append(
                                    evaluate_stop_route(con, events, direction, 3600, tp, sl, None, float(args.cost_bps))
                                )
                                item["stop_routes"].append(
                                    evaluate_stop_route(con, events, direction, 3600, tp, sl, 30.0, float(args.cost_bps))
                                )
                        item["best_route"] = max(
                            item["stop_routes"],
                            key=lambda r: (float(r.get("mean_net_bps") or -1e9), float(r.get("wr") or 0.0)),
                            default=None,
                        )
                        payload["results"].append(item)
    finally:
        con.close()

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_md(payload)
    print(f"wrote {OUT_JSON}")
    print(f"wrote {OUT_MD}")
    return 0


def fmt(v: Any, spec: str = ".2f") -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return "n/a"
    if isinstance(v, float):
        return format(v, spec)
    return str(v)


def write_md(payload: dict[str, Any]) -> None:
    lines = [
        "# S34 Restored-Window Replay - 2026-06-07",
        "",
        "This is a post-restore forensic replay over live liquidation data collected after the WebSocket route fix.",
        "",
        f"- window_start: `{payload['window']['start_utc']}`",
        f"- window_end_for_analysis: `{payload['window']['end_utc']}`",
        f"- cost_model: `{payload['cost_bps']} bps round trip`",
        "",
        "## Fixed Horizon Results",
        "",
        "| symbol | liq side | direction | threshold | n | 5m mean | 15m mean | 30m mean | 60m mean |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in payload["results"]:
        fh = r["fixed_horizon"]
        lines.append(
            f"| {r['symbol']} | {r['liq_side']} | {r['direction']} | {r['threshold']:.0f} | {r['n_events']} | "
            f"{fmt(fh['300'].get('mean_net_bps'))} | {fmt(fh['900'].get('mean_net_bps'))} | "
            f"{fmt(fh['1800'].get('mean_net_bps'))} | {fmt(fh['3600'].get('mean_net_bps'))} |"
        )
    lines += [
        "",
        "## Best Stop Route Per Threshold",
        "",
        "| symbol | threshold | n | tp bps | sl bps | BE trigger | WR | mean net bps | exits |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for r in payload["results"]:
        b = r.get("best_route") or {}
        lines.append(
            f"| {r['symbol']} | {r['threshold']:.0f} | {b.get('n', 0)} | {fmt(b.get('tp_bps'))} | "
            f"{fmt(b.get('sl_bps'))} | {fmt(b.get('be_trigger_bps'))} | "
            f"{fmt((b.get('wr') or 0.0) * 100)}% | {fmt(b.get('mean_net_bps'))} | "
            f"`{json.dumps(b.get('exits') or {}, separators=(',', ':'))}` |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "- This is not a final alpha verdict; the window is still short.",
        "- The goal is to identify whether restored liquidation data can explain stop/TP management better than the incomplete-sensor trade.",
        "- Any result with small `n` should be treated as directional evidence only.",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
