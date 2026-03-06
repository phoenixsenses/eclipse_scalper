from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Tuple

from core.regime import RegimeClassifier
from tools.validate_passive_pocket_forward import _add_regime_labels


def _detect_price_column(conn: sqlite3.Connection, table: str) -> str:
    cols = [str(r[1]) for r in conn.execute(f"PRAGMA table_info({table})").fetchall()]
    for cand in ("price", "mark_price", "mid"):
        if cand in cols:
            return cand
    raise RuntimeError(f"Could not detect price column in table={table}; columns={cols}")


def _load_marks(conn: sqlite3.Connection, symbol: str, table: str) -> List[Dict[str, Any]]:
    price_col = _detect_price_column(conn, table)
    q = (
        f"SELECT ts_ms, {price_col} AS price "
        f"FROM {table} WHERE symbol=? AND ts_ms IS NOT NULL ORDER BY ts_ms ASC"
    )
    out: List[Dict[str, Any]] = []
    for ts_ms, price in conn.execute(q, (symbol,)):
        if ts_ms is None or price is None:
            continue
        try:
            out.append({"ts_ms": int(ts_ms), "mid": float(price)})
        except Exception:
            continue
    return out


def verify_consistency(db: str, symbol: str, lookback_sec: int = 3600, table: str = "mark_prices") -> Dict[str, Any]:
    conn = sqlite3.connect(str(db), check_same_thread=False)
    try:
        rows = _load_marks(conn, symbol=symbol, table=table)
    finally:
        conn.close()
    if not rows:
        return {
            "status": "no_data",
            "db": str(db),
            "symbol": symbol,
            "table": table,
            "lookback_sec": int(lookback_sec),
            "comparisons": 0,
            "matches": 0,
            "match_rate": 0.0,
            "mismatches": [],
        }

    # Backtest-style labels (row-offset horizon, as in existing evaluator).
    bt_rows = [dict(r) for r in rows]
    _add_regime_labels(bt_rows, window_sec=int(lookback_sec))

    cls = RegimeClassifier(lookback_sec=int(lookback_sec), debounce_sec=0)
    mismatches: List[Dict[str, Any]] = []
    comparisons = 0
    matches = 0
    for i, row in enumerate(rows):
        cls.update(timestamp=float(row["ts_ms"]) / 1000.0, price=float(row["mid"]))
        bt = str(bt_rows[i].get("_regime_label") or "")
        live = cls.current_regime
        # Match empty backtest labels to UNKNOWN during warmup.
        if bt == "":
            bt = "UNKNOWN"
        if live == "TRANSITION":
            live = "UNKNOWN"
        comparisons += 1
        if bt == live:
            matches += 1
        elif len(mismatches) < 20:
            mismatches.append(
                {
                    "idx": i,
                    "ts_ms": int(row["ts_ms"]),
                    "backtest": bt,
                    "classifier": live,
                    "rolling_return": cls.rolling_return,
                }
            )

    match_rate = (matches / comparisons) if comparisons else 0.0
    status = "ok" if match_rate >= 0.95 else "warn"
    return {
        "status": status,
        "db": str(db),
        "symbol": symbol,
        "table": table,
        "lookback_sec": int(lookback_sec),
        "rows": len(rows),
        "comparisons": comparisons,
        "matches": matches,
        "match_rate": match_rate,
        "mismatch_count": comparisons - matches,
        "mismatches": mismatches,
    }


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Verify regime classifier consistency against backtest labels.")
    p.add_argument("--db", default="data/microstructure.db")
    p.add_argument("--symbol", default="ETHUSDT")
    p.add_argument("--lookback-sec", type=int, default=3600)
    p.add_argument("--table", default="mark_prices")
    return p


def main() -> int:
    args = _build_parser().parse_args()
    res = verify_consistency(
        db=str(args.db),
        symbol=str(args.symbol),
        lookback_sec=int(args.lookback_sec),
        table=str(args.table),
    )
    print(
        f"regime_consistency symbol={res.get('symbol')} table={res.get('table')} "
        f"rows={res.get('rows', 0)} comparisons={res.get('comparisons', 0)} "
        f"match_rate={float(res.get('match_rate', 0.0)):.4f} "
        f"status={res.get('status')}"
    )
    for m in res.get("mismatches", []):
        print(
            f"  mismatch idx={m.get('idx')} ts_ms={m.get('ts_ms')} "
            f"backtest={m.get('backtest')} classifier={m.get('classifier')} "
            f"ret={m.get('rolling_return')}"
        )
    return 0 if res.get("status") != "warn" else 2


if __name__ == "__main__":
    raise SystemExit(main())

