from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.run_summary import build_run_summary

from ami.storage import production as PR
from ami.storage import research_reader as RR

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
    """Direct-SQL oracle -- kept unchanged as the parity reference for
    `_avg_funding_rate_v2` (BATCH-STORAGE-ROTATION-RETENTION-RANGE-READ-
    CONSUMER-MIGRATION-V1). No longer called by main(); the reader-backed
    path is used instead."""
    row = conn.execute(
        "SELECT AVG(funding_rate) FROM mark_prices WHERE symbol=? AND ts_ms>=? AND ts_ms<=?",
        (symbol, int(start_ms), int(end_ms)),
    ).fetchone()
    return float((row[0] if row else 0.0) or 0.0)


def _avg_funding_rate_v2(root, symbol: str, start_ms: int, end_ms: int, source_db_path=None) -> float:
    """Reader-backed replacement for `_avg_funding_rate`, via
    `plan_read`/`execute_read`. `symbol` is a genuine runtime parameter,
    sourced per-trade from `trades.symbol` in the paper-trades DB
    (falling back to `--symbol` only when that column is absent).

    Range semantics: the oracle's SQL uses an INCLUSIVE upper bound
    (`ts_ms<=end_ms`); `plan_read`/`execute_read` use the reader's
    half-open `[start_ms, end_ms)` convention. Since `ts_ms` is always an
    integer column, `ts_ms<=end_ms` is exactly equivalent to
    `ts_ms<end_ms+1` -- so `end_ms+1` is passed here to reproduce the
    original inclusive boundary bit-for-bit, not approximately.

    Replicates `AVG(funding_rate)`'s SQL semantics exactly: `funding_rate`
    is a nullable column, and SQL's `AVG` skips NULL rows from both the
    sum and the count -- so this sums/counts only the non-NULL values,
    matching SQL precisely (a plain `statistics.mean` over all fetched
    values would be wrong if any row's `funding_rate` were NULL)."""
    plan = RR.plan_read(root, table="mark_prices", symbol=symbol, start_ms=int(start_ms), end_ms=int(end_ms) + 1)
    result = RR.execute_read(plan, columns=("funding_rate",), source_db_path=source_db_path)
    total = 0.0
    count = 0
    for (funding_rate,) in result.iter_rows():
        if funding_rate is not None:
            total += float(funding_rate)
            count += 1
    return (total / count) if count > 0 else 0.0


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
    # Every mark_prices range read now goes through the reader (via
    # SOURCE_DB_PATH), so the old direct `mconn` connection to the source
    # DB is fully dead here and is not opened (mirrors the ASOF
    # dead-connection-drop precedent). This also closes a pre-existing gap:
    # `mconn` was never opened with `mode=ro`, unlike the reader's
    # internal SQLite fallback, which always is.
    root, _ = PR.resolve_production_root()
    source_db_path = str(mdb)
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
        avg_rate = _avg_funding_rate_v2(root, sym, int(entry * 1000), int(exit_ * 1000), source_db_path=source_db_path)
        fbps = _funding_bps_for_trade(avg_rate, side, hold_sec)
        total_bps += fbps
        n += 1
        if str(side).upper() in ("BUY", "LONG"):
            long_sec += hold_sec
        else:
            short_sec += hold_sec
        per_trade.append({"symbol": sym, "side": side, "hold_sec": hold_sec, "avg_rate_8h": avg_rate, "funding_bps": fbps})

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
