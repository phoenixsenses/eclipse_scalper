from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path


DB_PATH = Path("data/microstructure.db")
KEY_TABLES = {"agg_trades", "book_ticker", "liquidations", "mark_prices"}


def main() -> None:
    con = sqlite3.connect(f"file:{DB_PATH.as_posix()}?mode=ro", uri=True, timeout=30)
    cur = con.cursor()
    tables = [
        row[0]
        for row in cur.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name NOT LIKE 'sqlite_%' "
            "ORDER BY name"
        )
    ]
    out = []
    for table in tables:
        columns = [row[1] for row in cur.execute(f"PRAGMA table_info({table})")]
        ts_col = "ts_ms" if "ts_ms" in columns else ("timestamp_ms" if "timestamp_ms" in columns else None)
        sym_col = "symbol" if "symbol" in columns else None
        started = time.time()
        item: dict[str, object] = {
            "table": table,
            "columns": columns,
            "ts_col": ts_col,
        }
        try:
            if ts_col:
                count, mn, mx = cur.execute(
                    f"SELECT COUNT(*), MIN({ts_col}), MAX({ts_col}) FROM {table}"
                ).fetchone()
            else:
                count = cur.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                mn = mx = None
            item.update({"rows": count, "min_ts_ms": mn, "max_ts_ms": mx})
            if sym_col and ts_col and table in KEY_TABLES:
                item["symbols"] = cur.execute(
                    f"SELECT symbol, COUNT(*), MIN({ts_col}), MAX({ts_col}) "
                    f"FROM {table} GROUP BY symbol ORDER BY symbol"
                ).fetchall()
        except Exception as exc:  # noqa: BLE001 - diagnostics should report all failures.
            item["error"] = repr(exc)
        item["seconds"] = round(time.time() - started, 2)
        out.append(item)
    con.close()
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
