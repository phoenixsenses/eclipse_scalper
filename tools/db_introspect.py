from __future__ import annotations

import argparse
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tools.run_summary import build_run_summary
TS_CANDIDATES = [
    "ts",
    "ts_ms",
    "timestamp",
    "ts_utc",
    "event_ts",
    "event_time",
    "time",
    "time_ms",
    "trade_time_ms",
    "created_at",
    "updated_at",
]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Inspect sqlite schema and write markdown/json reports.")
    p.add_argument("--db", default="data/microstructure.db")
    p.add_argument("--out-md", default="reports/db_schema.md")
    p.add_argument("--out-json", default="reports/db_tables.json")
    return p.parse_args()


def _safe_float(v: Any) -> float | None:
    try:
        return float(v)
    except Exception:
        return None


def _norm_ts(v: Any) -> float | None:
    x = _safe_float(v)
    if x is None:
        return None
    if x > 1e12:
        return x / 1000.0
    return x


def _is_text_ts(conn: sqlite3.Connection, table: str, col: str) -> bool:
    for row in conn.execute(f"PRAGMA table_info({table})").fetchall():
        if str(row[1]) == col:
            t = str(row[2] or "").upper()
            return ("TEXT" in t) or ("CHAR" in t) or ("CLOB" in t)
    return False


def _detect_ts_col(cols: list[str]) -> str | None:
    lm = {c.lower(): c for c in cols}
    for cand in TS_CANDIDATES:
        if cand.lower() in lm:
            return lm[cand.lower()]
    return None


def _list_tables(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name").fetchall()
    return [str(r[0]) for r in rows if r and str(r[0]) and not str(r[0]).startswith("sqlite_")]


@dataclass(frozen=True)
class TableInfo:
    name: str
    row_count: int
    columns: list[dict[str, Any]]
    indexes: list[dict[str, Any]]
    ts_candidates: list[str]
    ts_col: str | None
    min_ts: float | None
    max_ts: float | None


def _table_info(conn: sqlite3.Connection, table: str) -> TableInfo:
    cols_rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    columns = [
        {
            "name": str(r[1]),
            "type": str(r[2] or ""),
            "notnull": int(r[3]),
            "default": r[4],
            "pk": int(r[5]),
        }
        for r in cols_rows
    ]
    cols = [c["name"] for c in columns]
    ts_col = _detect_ts_col(cols)
    ts_candidates = [c for c in cols if c.lower() in {x.lower() for x in TS_CANDIDATES}]

    row_count = int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])

    min_ts = None
    max_ts = None
    if ts_col:
        min_raw, max_raw = conn.execute(f"SELECT MIN({ts_col}), MAX({ts_col}) FROM {table}").fetchone()
        if _is_text_ts(conn, table, ts_col):
            # best-effort for text timestamps; leave raw as None in numeric fields
            min_ts = None
            max_ts = None
        else:
            min_ts = _norm_ts(min_raw)
            max_ts = _norm_ts(max_raw)

    indexes: list[dict[str, Any]] = []
    for idx in conn.execute(f"PRAGMA index_list({table})").fetchall():
        idx_name = str(idx[1])
        cols_i = [str(r[2]) for r in conn.execute(f"PRAGMA index_info({idx_name})").fetchall()]
        indexes.append(
            {
                "name": idx_name,
                "unique": int(idx[2]),
                "origin": str(idx[3]) if len(idx) > 3 else "",
                "partial": int(idx[4]) if len(idx) > 4 else 0,
                "columns": cols_i,
            }
        )

    return TableInfo(
        name=table,
        row_count=row_count,
        columns=columns,
        indexes=indexes,
        ts_candidates=ts_candidates,
        ts_col=ts_col,
        min_ts=min_ts,
        max_ts=max_ts,
    )


def _likely_core_tables(tables: list[TableInfo]) -> dict[str, str | None]:
    def pick(kind: str, names: tuple[str, ...]) -> str | None:
        matched = [t for t in tables if any(n in t.name.lower() for n in names)]
        if not matched:
            return None
        matched.sort(key=lambda x: x.row_count, reverse=True)
        return matched[0].name

    return {
        "trades": pick("trades", ("trade", "agg_trade", "aggtrades")),
        "book": pick("book", ("book", "depth", "mark")),
        "liquidations": pick("liquidations", ("liq", "liquid")),
    }


def _write_markdown(path: Path, db_path: Path, tables: list[TableInfo], core: dict[str, str | None]) -> None:
    lines: list[str] = []
    lines.append("# DB Schema Report")
    lines.append("")
    lines.append(f"- DB: `{db_path}`")
    lines.append(f"- Tables: {len(tables)}")
    lines.append("")
    lines.append("## Likely Core Tables")
    lines.append("")
    for k, v in core.items():
        lines.append(f"- {k}: `{v}`")
    lines.append("")
    lines.append("## Table Summary")
    lines.append("")
    lines.append("| table | rows | ts_col | min_ts | max_ts |")
    lines.append("|---|---:|---|---:|---:|")
    for t in tables:
        lines.append(
            f"| `{t.name}` | {t.row_count} | `{t.ts_col or ''}` | {t.min_ts if t.min_ts is not None else ''} | {t.max_ts if t.max_ts is not None else ''} |"
        )
    lines.append("")
    for t in tables:
        lines.append(f"## {t.name}")
        lines.append("")
        lines.append(f"- rows: {t.row_count}")
        lines.append(f"- timestamp candidates: `{', '.join(t.ts_candidates) if t.ts_candidates else '-'}`")
        lines.append(f"- chosen timestamp: `{t.ts_col or '-'}`")
        lines.append("")
        lines.append("### Columns")
        lines.append("")
        lines.append("| name | type | notnull | pk |")
        lines.append("|---|---|---:|---:|")
        for c in t.columns:
            lines.append(f"| `{c['name']}` | `{c['type']}` | {c['notnull']} | {c['pk']} |")
        lines.append("")
        lines.append("### Indexes")
        lines.append("")
        if not t.indexes:
            lines.append("- none")
        else:
            for idx in t.indexes:
                cols = ", ".join(idx["columns"])
                lines.append(f"- `{idx['name']}` unique={idx['unique']} columns=[{cols}]")
        lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def introspect(db_path: Path) -> dict[str, Any]:
    conn = sqlite3.connect(str(db_path))
    try:
        tables = [_table_info(conn, t) for t in _list_tables(conn)]
    finally:
        conn.close()

    core = _likely_core_tables(tables)
    payload = {
        "db": str(db_path),
        "tables": [
            {
                "name": t.name,
                "row_count": t.row_count,
                "columns": t.columns,
                "indexes": t.indexes,
                "timestamp_candidates": t.ts_candidates,
                "timestamp_column": t.ts_col,
                "min_ts": t.min_ts,
                "max_ts": t.max_ts,
            }
            for t in tables
        ],
        "likely_core_tables": core,
    }
    payload["run_summary"] = build_run_summary(
        run_type="db_introspect",
        inputs={"db": str(db_path)},
        metrics={"table_count": len(tables)},
        artifacts={"json": "reports/db_tables.json", "md": "reports/db_schema.md"},
    )
    return payload


def main() -> int:
    args = _parse_args()
    db_path = Path(str(args.db))
    if not db_path.exists():
        print(f"db_introspect error missing_db={db_path}")
        return 2

    payload = introspect(db_path)
    out_json = Path(str(args.out_json))
    out_md = Path(str(args.out_md))
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    tables = [
        TableInfo(
            name=t["name"],
            row_count=int(t["row_count"]),
            columns=list(t["columns"]),
            indexes=list(t["indexes"]),
            ts_candidates=list(t["timestamp_candidates"]),
            ts_col=t.get("timestamp_column"),
            min_ts=t.get("min_ts"),
            max_ts=t.get("max_ts"),
        )
        for t in payload["tables"]
    ]
    _write_markdown(out_md, db_path, tables, payload["likely_core_tables"])
    print(f"db_introspect ok tables={len(tables)} out_json={out_json} out_md={out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
