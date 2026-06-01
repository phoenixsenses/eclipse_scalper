"""SOL candidate conflict/correlation check.

Checks whether SOL BUY-liquidation short candidates overlap with ETH S34
detector signals or major ETH/BTC liquidation stress windows.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path
from statistics import mean

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DB = "data/microstructure.db"
OUT_MD = Path("reports/SOL_ETH_CONFLICT_CHECK.md")
OUT_JSON = Path("reports/SOL_ETH_CONFLICT_CHECK.json")


def _wr(vals: list[float]) -> float | None:
    return 100.0 * sum(1 for x in vals if x > 0) / len(vals) if vals else None


def _mark(conn: sqlite3.Connection, symbol: str, ts_ms: int, before: bool) -> float | None:
    op = "<=" if before else ">="
    order = "DESC" if before else "ASC"
    row = conn.execute(
        f"SELECT mark_price FROM mark_prices WHERE symbol=? AND ts_ms {op} ? ORDER BY ts_ms {order} LIMIT 1",
        (symbol, ts_ms),
    ).fetchone()
    return float(row[0]) if row and row[0] is not None else None


def _short_ret_bps(conn: sqlite3.Connection, symbol: str, ts_ms: int, horizon_sec: int) -> float | None:
    ep = _mark(conn, symbol, ts_ms, True)
    xp = _mark(conn, symbol, ts_ms + horizon_sec * 1000, False)
    if ep is None or xp is None or ep <= 0:
        return None
    return (ep - xp) / ep * 1e4


def main() -> None:
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    candidates = conn.execute(
        """
        SELECT ts_ms, notional
        FROM liquidations
        WHERE symbol='SOLUSDT' AND side='BUY' AND notional>=50000
        ORDER BY ts_ms ASC
        """
    ).fetchall()

    rows = []
    for ts_ms, notional in candidates:
        ts_ms = int(ts_ms)
        eth_detector = conn.execute(
            "SELECT COUNT(*) FROM detector_signals WHERE symbol='ETHUSDT' AND ABS(signal_ts_ms - ?) <= ?",
            (ts_ms, 15 * 60 * 1000),
        ).fetchone()[0]
        eth_big_buy = conn.execute(
            """
            SELECT COUNT(*) FROM liquidations
            WHERE symbol='ETHUSDT' AND side='BUY' AND notional>=200000 AND ABS(ts_ms - ?) <= ?
            """,
            (ts_ms, 15 * 60 * 1000),
        ).fetchone()[0]
        btc_big_buy = conn.execute(
            """
            SELECT COUNT(*) FROM liquidations
            WHERE symbol='BTCUSDT' AND side='BUY' AND notional>=500000 AND ABS(ts_ms - ?) <= ?
            """,
            (ts_ms, 15 * 60 * 1000),
        ).fetchone()[0]
        sol_ret = _short_ret_bps(conn, "SOLUSDT", ts_ms, 900)
        eth_ret = _short_ret_bps(conn, "ETHUSDT", ts_ms, 900)
        btc_ret = _short_ret_bps(conn, "BTCUSDT", ts_ms, 900)
        rows.append(
            {
                "ts_ms": ts_ms,
                "notional": float(notional),
                "eth_detector_overlap": int(eth_detector),
                "eth_big_buy_overlap": int(eth_big_buy),
                "btc_big_buy_overlap": int(btc_big_buy),
                "sol_short_900_bps": sol_ret,
                "eth_short_900_bps": eth_ret,
                "btc_short_900_bps": btc_ret,
            }
        )
    conn.close()

    def stats(label: str, subset: list[dict]) -> dict:
        vals = [float(r["sol_short_900_bps"]) for r in subset if r["sol_short_900_bps"] is not None]
        return {
            "label": label,
            "n": len(vals),
            "wr": _wr(vals),
            "mean_bps": mean(vals) if vals else None,
        }

    overlap_eth_detector = [r for r in rows if r["eth_detector_overlap"] > 0]
    no_eth_detector = [r for r in rows if r["eth_detector_overlap"] == 0]
    overlap_eth_big = [r for r in rows if r["eth_big_buy_overlap"] > 0]
    no_eth_big = [r for r in rows if r["eth_big_buy_overlap"] == 0]
    overlap_btc_big = [r for r in rows if r["btc_big_buy_overlap"] > 0]
    no_btc_big = [r for r in rows if r["btc_big_buy_overlap"] == 0]
    summary = [
        stats("all", rows),
        stats("eth_detector_overlap", overlap_eth_detector),
        stats("no_eth_detector_overlap", no_eth_detector),
        stats("eth_big_buy_overlap", overlap_eth_big),
        stats("no_eth_big_buy_overlap", no_eth_big),
        stats("btc_big_buy_overlap", overlap_btc_big),
        stats("no_btc_big_buy_overlap", no_btc_big),
    ]
    payload = {
        "candidate": "SOLUSDT BUY>=50000 SHORT 900s",
        "rows": rows,
        "summary": summary,
        "overlap_rates": {
            "eth_detector": len(overlap_eth_detector) / len(rows) if rows else None,
            "eth_big_buy": len(overlap_eth_big) / len(rows) if rows else None,
            "btc_big_buy": len(overlap_btc_big) / len(rows) if rows else None,
        },
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = ["# SOL/ETH Conflict Check", "", "- candidate: `SOLUSDT BUY>=50000 SHORT 900s`", ""]
    lines.append("## Overlap Rates")
    lines.append("")
    for k, v in payload["overlap_rates"].items():
        lines.append(f"- {k}: `{(v * 100.0) if v is not None else None:.2f}%`")
    lines.append("")
    lines.append("## SOL Candidate Performance By Overlap")
    lines.append("")
    lines.append("| group | N | WR | mean_bps |")
    lines.append("|---|---:|---:|---:|")
    for r in summary:
        wr = "n/a" if r["wr"] is None else f"{r['wr']:.2f}%"
        mb = "n/a" if r["mean_bps"] is None else f"{r['mean_bps']:.2f}"
        lines.append(f"| {r['label']} | {r['n']} | {wr} | {mb} |")
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUT_MD}")
    print(f"Wrote {OUT_JSON}")
    print(json.dumps(payload["overlap_rates"], indent=2))


if __name__ == "__main__":
    main()
