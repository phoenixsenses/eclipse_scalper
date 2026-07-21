from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path


DB_PATH = Path("data/microstructure.db")


def scalar(cur: sqlite3.Cursor, sql: str) -> object:
    return cur.execute(sql).fetchone()[0]


def main() -> None:
    con = sqlite3.connect(f"file:{DB_PATH.as_posix()}?mode=ro", uri=True, timeout=10)
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
        started = time.time()
        columns = [row[1] for row in cur.execute(f"PRAGMA table_info({table})")]
        ts_col = "ts_ms" if "ts_ms" in columns else ("timestamp_ms" if "timestamp_ms" in columns else None)
        item: dict[str, object] = {
            "table": table,
            "columns": columns,
            "ts_col": ts_col,
            "indexes": [row[1] for row in cur.execute(f"PRAGMA index_list({table})")],
        }
        for label, sql in {
            "rowid_min": f"SELECT MIN(rowid) FROM {table}",
            "rowid_max": f"SELECT MAX(rowid) FROM {table}",
        }.items():
            try:
                item[label] = scalar(cur, sql)
            except Exception as exc:  # noqa: BLE001
                item[label + "_error"] = repr(exc)
        if ts_col:
            for label, sql in {
                "min_ts_ms": f"SELECT {ts_col} FROM {table} ORDER BY {ts_col} ASC LIMIT 1",
                "max_ts_ms": f"SELECT {ts_col} FROM {table} ORDER BY {ts_col} DESC LIMIT 1",
            }.items():
                try:
                    item[label] = scalar(cur, sql)
                except Exception as exc:  # noqa: BLE001
                    item[label + "_error"] = repr(exc)
        item["seconds"] = round(time.time() - started, 3)
        out.append(item)
    con.close()
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
