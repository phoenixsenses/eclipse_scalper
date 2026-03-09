from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List

from src.microphys.io.sqlite_reader import discover_mappings
from tools.check_data_ready import detect_ts_col, list_tables, table_columns
from tools.run_summary import build_run_summary


REQUIRED_TABLES = {
    "agg_trades": ("ts_ms", "symbol", "price", "quantity", "notional", "is_buyer_maker"),
    "mark_prices": ("ts_ms", "symbol", "mark_price"),
    "liquidations": ("ts_ms", "symbol", "side", "price", "quantity", "notional"),
}
BOOK_FIELD_CANDIDATES = ("bid_px", "ask_px", "bid_qty", "ask_qty", "best_bid", "best_ask", "bid", "ask")


def _parse_symbols(raw: str) -> List[str]:
    return [s.strip().upper() for s in str(raw or "").replace(";", ",").split(",") if s.strip()]


def _safe_count(conn: sqlite3.Connection, sql: str, args: tuple[Any, ...] = ()) -> int:
    row = conn.execute(sql, args).fetchone()
    return int(row[0] or 0) if row else 0


def _detect_symbol_col(cols: List[str]) -> str | None:
    lower = {c.lower(): c for c in cols}
    for cand in ("symbol", "s", "pair", "instrument", "ticker"):
        if cand in lower:
            return lower[cand]
    return None


def _feature_capability(
    table_contracts: Dict[str, Dict[str, Any]],
    mappings: Dict[str, Any],
    discovered_tables: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    trades_ok = bool(table_contracts.get("agg_trades", {}).get("present"))
    marks_ok = bool(table_contracts.get("mark_prices", {}).get("present"))
    liq_ok = bool(table_contracts.get("liquidations", {}).get("present"))

    book_mapping = mappings.get("book")
    true_book = False
    book_source = None
    for table, info in discovered_tables.items():
        cols = {str(c).lower() for c in info.get("columns", [])}
        if ({"bid_px", "ask_px"} <= cols) or ({"best_bid", "best_ask"} <= cols) or ({"bid", "ask"} <= cols):
            true_book = True
            book_source = table
            break
    if (not true_book) and book_mapping is not None:
        book_source = str(book_mapping.table)
        cols = book_mapping.cols
        true_book = bool(cols.get("bid_px") and cols.get("ask_px"))

    if trades_ok and marks_ok and liq_ok and true_book:
        tier = "full_book"
    elif trades_ok and marks_ok and liq_ok:
        tier = "trade_plus_liq_mark_proxy"
    elif trades_ok and marks_ok:
        tier = "trade_plus_mark"
    elif marks_ok:
        tier = "mark_only"
    else:
        tier = "insufficient"

    return {
        "tier": tier,
        "mark_only": bool(marks_ok),
        "trade_flow": bool(trades_ok and marks_ok),
        "trade_plus_liq": bool(trades_ok and marks_ok and liq_ok),
        "requires_book": bool(true_book),
        "book_source_table": book_source,
        "reason": (
            "true_top_of_book_available"
            if true_book
            else "true_top_of_book_missing_mark_prices_or_proxy_used"
        ),
    }


def analyze_contract(db_path: Path, symbols: List[str], require_true_book: bool = False) -> Dict[str, Any]:
    conn = sqlite3.connect(str(db_path))
    try:
        existing_tables = set(list_tables(conn))
        table_contracts: Dict[str, Dict[str, Any]] = {}
        failures: List[str] = []
        warnings: List[str] = []
        symbol_coverage: Dict[str, Dict[str, bool]] = {sym: {} for sym in symbols}

        for table, required_cols in REQUIRED_TABLES.items():
            present = table in existing_tables
            cols = table_columns(conn, table) if present else []
            missing = [c for c in required_cols if c not in cols]
            ts_col = detect_ts_col(cols) if present else None
            sym_col = _detect_symbol_col(cols) if present else None
            row_count = _safe_count(conn, f"SELECT COUNT(*) FROM {table}") if present else 0
            available_book_fields = [c for c in cols if c.lower() in {x.lower() for x in BOOK_FIELD_CANDIDATES}]
            table_contracts[table] = {
                "present": present,
                "row_count": row_count,
                "timestamp_column": ts_col,
                "symbol_column": sym_col,
                "required_columns_missing": missing,
                "available_book_fields": available_book_fields,
            }
            if not present:
                failures.append(f"missing_table:{table}")
                for sym in symbols:
                    symbol_coverage[sym][table] = False
                continue
            if missing:
                failures.append(f"missing_columns:{table}:{','.join(missing)}")
            if ts_col is None:
                failures.append(f"missing_timestamp_column:{table}")
            if sym_col is None:
                failures.append(f"missing_symbol_column:{table}")
            if sym_col:
                for sym in symbols:
                    symbol_coverage[sym][table] = bool(
                        _safe_count(conn, f"SELECT COUNT(*) FROM {table} WHERE {sym_col} = ?", (sym,)) > 0
                    )
            else:
                for sym in symbols:
                    symbol_coverage[sym][table] = False

        for sym, coverage in symbol_coverage.items():
            missing_tables = [table for table, ok in coverage.items() if not ok]
            if missing_tables:
                warnings.append(f"symbol_coverage_gap:{sym}:{','.join(missing_tables)}")

        mappings = discover_mappings(db_path)
        discovered_tables = {
            table: {"columns": table_columns(conn, table)}
            for table in existing_tables
        }
        capability = _feature_capability(table_contracts, mappings, discovered_tables)
        if not capability["requires_book"]:
            warnings.append("true_top_of_book_missing")
            if require_true_book:
                failures.append("true_top_of_book_required")

        status = "pass"
        if failures:
            status = "fail"
        elif warnings:
            status = "warn"

        payload: Dict[str, Any] = {
            "db": str(db_path),
            "symbols": list(symbols),
            "required_tables": list(REQUIRED_TABLES),
            "status": status,
            "table_contracts": table_contracts,
            "symbol_coverage": symbol_coverage,
            "feature_capability": capability,
            "warnings": warnings,
            "failures": failures,
        }
        payload["run_summary"] = build_run_summary(
            run_type="validate_microstructure_contract",
            inputs={"db": str(db_path), "symbols": list(symbols), "require_true_book": bool(require_true_book)},
            metrics={
                "status": status,
                "table_count": len(table_contracts),
                "warning_count": len(warnings),
                "failure_count": len(failures),
                "requires_book": bool(capability["requires_book"]),
            },
            artifacts={"json": "reports/MICROSTRUCTURE_CONTRACT.json", "md": "reports/MICROSTRUCTURE_CONTRACT.md"},
        )
        return payload
    finally:
        conn.close()


def _write_markdown(path: Path, payload: Dict[str, Any]) -> None:
    lines: List[str] = []
    lines.append("# Microstructure Contract Report")
    lines.append("")
    lines.append(f"- DB: `{payload['db']}`")
    lines.append(f"- Status: `{payload['status']}`")
    lines.append(f"- Symbols: `{', '.join(payload['symbols'])}`")
    lines.append(f"- Capability tier: `{payload['feature_capability']['tier']}`")
    lines.append(f"- True book available: `{payload['feature_capability']['requires_book']}`")
    lines.append("")
    lines.append("## Table Contracts")
    lines.append("")
    lines.append("| table | present | rows | ts_col | symbol_col | missing_required |")
    lines.append("|---|---|---:|---|---|---|")
    for table in payload["required_tables"]:
        info = payload["table_contracts"][table]
        lines.append(
            f"| `{table}` | {info['present']} | {info['row_count']} | `{info['timestamp_column'] or ''}` | "
            f"`{info['symbol_column'] or ''}` | `{','.join(info['required_columns_missing'])}` |"
        )
    lines.append("")
    lines.append("## Symbol Coverage")
    lines.append("")
    for sym, coverage in payload["symbol_coverage"].items():
        parts = ", ".join(f"{table}={ok}" for table, ok in coverage.items())
        lines.append(f"- `{sym}`: {parts}")
    lines.append("")
    lines.append("## Findings")
    lines.append("")
    if payload["failures"]:
        for item in payload["failures"]:
            lines.append(f"- fail: `{item}`")
    if payload["warnings"]:
        for item in payload["warnings"]:
            lines.append(f"- warn: `{item}`")
    if not payload["failures"] and not payload["warnings"]:
        lines.append("- none")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate research-side microstructure input contract.")
    p.add_argument("--db", default="data/microstructure.db")
    p.add_argument("--symbols", default="BTCUSDT,ETHUSDT")
    p.add_argument("--out-json", default="reports/MICROSTRUCTURE_CONTRACT.json")
    p.add_argument("--out-md", default="reports/MICROSTRUCTURE_CONTRACT.md")
    p.add_argument("--require-true-book", action="store_true")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    db_path = Path(str(args.db))
    if not db_path.exists():
        print(f"validate_microstructure_contract error missing_db={db_path}")
        return 2
    payload = analyze_contract(
        db_path=db_path,
        symbols=_parse_symbols(args.symbols),
        require_true_book=bool(args.require_true_book),
    )
    out_json = Path(str(args.out_json))
    out_md = Path(str(args.out_md))
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_markdown(out_md, payload)
    print(
        f"validate_microstructure_contract status={payload['status']} "
        f"warnings={len(payload['warnings'])} failures={len(payload['failures'])}"
    )
    return 0 if payload["status"] != "fail" else 1


if __name__ == "__main__":
    raise SystemExit(main())
