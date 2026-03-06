from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any

from tools.run_summary import build_run_summary

def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Estimate cumulative funding impact from paper trades.")
    p.add_argument("--trades-db", default="data/paper_trades.db")
    p.add_argument("--micro-db", default="data/microstructure.db")
    p.add_argument("--symbol", default="ETHUSDT")
    p.add_argument("--out-md", default="reports/FUNDING_RATE_ANALYSIS.md")
    p.add_argument("--out-json", default="reports/FUNDING_RATE_ANALYSIS.json")
    return p.parse_args()


def _has_col(conn: sqlite3.Connection, table: str, col: str) -> bool:
    try:
        rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
        return any(str(r[1]).lower() == str(col).lower() for r in rows)
    except Exception:
        return False


def _avg_funding_rate(conn: sqlite3.Connection, symbol: str, start_ms: int, end_ms: int) -> float:
    row = conn.execute(
        "SELECT AVG(funding_rate) FROM mark_prices WHERE symbol=? AND ts_ms>=? AND ts_ms<=?",
        (symbol, int(start_ms), int(end_ms)),
    ).fetchone()
    return float((row[0] if row else 0.0) or 0.0)


def _funding_bps_for_trade(avg_rate_8h: float, side: str, hold_sec: float) -> float:
    # funding rate is per 8h period
    periods = max(0.0, float(hold_sec)) / 28800.0
    rate = float(avg_rate_8h)
    # Long pays when funding positive; short receives.
    side_u = str(side or "").upper()
    sign = 1.0 if side_u in ("BUY", "LONG") else -1.0
    # positive => cost for long; negative for short
    cost_frac = sign * rate * periods
    return float(cost_frac * 10000.0)


def main() -> int:
    args = _args()
    tdb = Path(args.trades_db)
    mdb = Path(args.micro_db)
    if not tdb.exists():
        print(f"funding_rate_analysis: missing trades db {tdb}")
        return 2
    if not mdb.exists():
        print(f"funding_rate_analysis: missing micro db {mdb}")
        return 2

    tconn = sqlite3.connect(str(tdb), check_same_thread=False)
    tconn.row_factory = sqlite3.Row
    mconn = sqlite3.connect(str(mdb), check_same_thread=False)
    try:
        has_symbol = _has_col(tconn, "trades", "symbol")
        if has_symbol:
            rows = tconn.execute(
                "SELECT entry_time, exit_time, side, symbol FROM trades WHERE entry_time>0 AND exit_time>entry_time"
            ).fetchall()
        else:
            rows = tconn.execute(
                "SELECT entry_time, exit_time, side FROM trades WHERE entry_time>0 AND exit_time>entry_time"
            ).fetchall()
    finally:
        tconn.close()

    total_bps = 0.0
    n = 0
    long_sec = 0.0
    short_sec = 0.0
    per_trade: list[dict[str, Any]] = []
    for r in rows:
        entry = float(r["entry_time"] or 0.0)
        exit_ = float(r["exit_time"] or 0.0)
        if exit_ <= entry:
            continue
        hold_sec = float(exit_ - entry)
        side = str(r["side"] or "")
        sym = str(r["symbol"] or args.symbol) if "symbol" in r.keys() else str(args.symbol)
        avg_rate = _avg_funding_rate(mconn, sym, int(entry * 1000), int(exit_ * 1000))
        fbps = _funding_bps_for_trade(avg_rate, side, hold_sec)
        total_bps += fbps
        n += 1
        if str(side).upper() in ("BUY", "LONG"):
            long_sec += hold_sec
        else:
            short_sec += hold_sec
        per_trade.append({"symbol": sym, "side": side, "hold_sec": hold_sec, "avg_rate_8h": avg_rate, "funding_bps": fbps})
    mconn.close()

    avg_bps = (total_bps / n) if n > 0 else 0.0
    bias = long_sec - short_sec
    out = {
        "trades": int(n),
        "total_funding_bps": float(total_bps),
        "avg_funding_bps_per_trade": float(avg_bps),
        "long_hold_hours": float(long_sec / 3600.0),
        "short_hold_hours": float(short_sec / 3600.0),
        "directional_bias_hours": float(bias / 3600.0),
    }

    md = "\n".join(
        [
            "# Funding Rate Analysis",
            "",
            f"- trades: {int(out['trades'])}",
            f"- total_funding_bps: {float(out['total_funding_bps']):+.4f}",
            f"- avg_funding_bps_per_trade: {float(out['avg_funding_bps_per_trade']):+.5f}",
            f"- long_hold_hours: {float(out['long_hold_hours']):.2f}",
            f"- short_hold_hours: {float(out['short_hold_hours']):.2f}",
            f"- directional_bias_hours: {float(out['directional_bias_hours']):+.2f}",
            "",
            "Funding for 120s horizons is usually tiny per trade, but directional bias can accumulate over long runs.",
            "",
        ]
    )
    out_md = Path(args.out_md)
    out_json = Path(args.out_json)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {"summary": out, "sample": per_trade[:20]}
    payload["run_summary"] = build_run_summary(
        run_type="funding_rate_analysis",
        inputs={"trades_db": str(args.trades_db), "micro_db": str(args.micro_db), "symbol": str(args.symbol)},
        metrics={"trades": int(out["trades"]), "total_funding_bps": float(out["total_funding_bps"])},
        artifacts={"json": str(out_json), "md": str(out_md)},
    )
    md = md + "## Run Summary\n- `" + str(payload["run_summary"]) + "`\n"
    out_md.write_text(md, encoding="utf-8")
    out_json.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    print(f"funding_rate_analysis: wrote {out_md}")
    print(f"funding_rate_analysis: wrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
