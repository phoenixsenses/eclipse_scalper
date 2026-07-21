"""Audit S34 V Engine DATA_INCOMPLETE observation rows.

The v0.1 observer labels filled events as DATA_INCOMPLETE when the simulated
2h exit cannot be priced from book_ticker. This report classifies whether those
rows are true book-history gaps, stale quotes, or malformed ledger rows.
"""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.research_s34_knowable_anchor_continuation import file_fingerprint, iso_ms, r1
from tools.s34_v_engine_shadow_observer import (
    DEFAULT_LEDGER_JSONL,
    HORIZON_SEC,
    PROTOCOL_ID,
    SYMBOL,
    load_jsonl,
    utc_now,
)


DEFAULT_DB = ROOT / "data" / "microstructure.db"
OUT_DIR = ROOT / "reports" / "research" / "s34"
OUT_JSON = OUT_DIR / "S34_V_ENGINE_DATA_INCOMPLETE_AUDIT.json"
OUT_MD = OUT_DIR / "S34_V_ENGINE_DATA_INCOMPLETE_AUDIT.md"
OUT_CSV = OUT_DIR / "S34_V_ENGINE_DATA_INCOMPLETE_AUDIT_ROWS.csv"

AUDIT_FIELDS = (
    "observation_id",
    "signal_utc",
    "sim_status",
    "reason",
    "maker_fill_utc",
    "expected_exit_utc",
    "nearest_book_utc",
    "book_staleness_sec",
    "book_gap_sec",
    "book_bid",
    "book_ask",
    "book_mid",
    "vdepth_bps",
    "prior_4h_bps",
    "fill_delay_sec",
)


def nearest_book_before(conn: sqlite3.Connection, symbol: str, ts_ms: int) -> dict[str, Any] | None:
    row = conn.execute(
        """
        SELECT ts_ms, bid_price, ask_price, mid_price
        FROM book_ticker
        WHERE symbol=? AND ts_ms<=?
        ORDER BY ts_ms DESC
        LIMIT 1
        """,
        (symbol, int(ts_ms)),
    ).fetchone()
    if not row:
        return None
    return {
        "ts_ms": int(row[0]),
        "bid": float(row[1]),
        "ask": float(row[2]),
        "mid": float(row[3]),
        "staleness_sec": (int(ts_ms) - int(row[0])) / 1000.0,
    }


def nearest_book_after(conn: sqlite3.Connection, symbol: str, ts_ms: int) -> dict[str, Any] | None:
    row = conn.execute(
        """
        SELECT ts_ms, bid_price, ask_price, mid_price
        FROM book_ticker
        WHERE symbol=? AND ts_ms>=?
        ORDER BY ts_ms ASC
        LIMIT 1
        """,
        (symbol, int(ts_ms)),
    ).fetchone()
    if not row:
        return None
    return {
        "ts_ms": int(row[0]),
        "bid": float(row[1]),
        "ask": float(row[2]),
        "mid": float(row[3]),
        "gap_sec": (int(row[0]) - int(ts_ms)) / 1000.0,
    }


def classify_incomplete(
    row: dict[str, Any],
    *,
    before: dict[str, Any] | None,
    after: dict[str, Any] | None,
    max_staleness_sec: int,
) -> str:
    if row.get("maker_fill_ts_ms") is None:
        return "missing_fill_ts"
    if before is None and after is None:
        return "no_book_history_for_symbol"
    if before is None:
        return "exit_before_book_history"
    if float(before["staleness_sec"]) <= float(max_staleness_sec):
        return "unexpected_complete_book_available"
    if after is None:
        return "book_history_ends_before_exit"
    return "stale_exit_book_gap"


def audit_rows(conn: sqlite3.Connection, rows: list[dict[str, Any]], *, max_staleness_sec: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        if row.get("observation_status") != "DATA_INCOMPLETE":
            continue
        fill_ts = row.get("maker_fill_ts_ms")
        expected_exit_ts = int(fill_ts) + HORIZON_SEC * 1000 if fill_ts is not None else None
        before = nearest_book_before(conn, SYMBOL, expected_exit_ts) if expected_exit_ts is not None else None
        after = nearest_book_after(conn, SYMBOL, expected_exit_ts) if expected_exit_ts is not None else None
        reason = classify_incomplete(row, before=before, after=after, max_staleness_sec=int(max_staleness_sec))
        out.append(
            {
                "observation_id": row.get("observation_id"),
                "signal_ts_ms": row.get("signal_ts_ms"),
                "signal_utc": row.get("signal_utc"),
                "sim_status": row.get("sim_status"),
                "reason": reason,
                "maker_fill_ts_ms": fill_ts,
                "maker_fill_utc": iso_ms(fill_ts) if fill_ts is not None else None,
                "expected_exit_ts_ms": expected_exit_ts,
                "expected_exit_utc": iso_ms(expected_exit_ts) if expected_exit_ts is not None else None,
                "nearest_book_ts_ms": None if before is None else before["ts_ms"],
                "nearest_book_utc": None if before is None else iso_ms(before["ts_ms"]),
                "book_staleness_sec": None if before is None else r1(before["staleness_sec"]),
                "next_book_ts_ms": None if after is None else after["ts_ms"],
                "next_book_utc": None if after is None else iso_ms(after["ts_ms"]),
                "book_gap_sec": None if after is None else r1(after["gap_sec"]),
                "book_bid": None if before is None else before["bid"],
                "book_ask": None if before is None else before["ask"],
                "book_mid": None if before is None else before["mid"],
                "vdepth_bps": row.get("vdepth_bps"),
                "prior_4h_bps": row.get("prior_4h_bps"),
                "fill_delay_sec": row.get("fill_delay_sec"),
            }
        )
    out.sort(key=lambda r: int(r.get("signal_ts_ms") or 0))
    return out


def distribution(values: list[float]) -> dict[str, Any]:
    xs = sorted(float(v) for v in values if v is not None)
    if not xs:
        return {"n": 0, "min": None, "median": None, "max": None}
    return {
        "n": len(xs),
        "min": r1(xs[0]),
        "median": r1(xs[len(xs) // 2]),
        "max": r1(xs[-1]),
    }


def build_report(
    *,
    ledger_rows: list[dict[str, Any]],
    audit: list[dict[str, Any]],
    source_db: dict[str, Any],
    source_ledger: dict[str, Any],
    max_staleness_sec: int,
) -> dict[str, Any]:
    reason_counts = dict(sorted(Counter(str(r["reason"]) for r in audit).items()))
    sim_counts = dict(sorted(Counter(str(r.get("sim_status") or "UNKNOWN") for r in audit).items()))
    by_reason: dict[str, dict[str, Any]] = {}
    for reason in sorted(reason_counts):
        items = [r for r in audit if r["reason"] == reason]
        staleness = [float(r["book_staleness_sec"]) for r in items if r.get("book_staleness_sec") is not None]
        gaps = [float(r["book_gap_sec"]) for r in items if r.get("book_gap_sec") is not None]
        by_reason[reason] = {
            "n": len(items),
            "first_signal_utc": items[0].get("signal_utc") if items else None,
            "last_signal_utc": items[-1].get("signal_utc") if items else None,
            "book_staleness_sec": distribution(staleness),
            "next_book_gap_sec": distribution(gaps),
        }
    return {
        "generated_at_utc": utc_now(),
        "source_db": source_db,
        "source_ledger": source_ledger,
        "protocol_id": PROTOCOL_ID,
        "symbol": SYMBOL,
        "max_staleness_sec": int(max_staleness_sec),
        "ledger_rows": len(ledger_rows),
        "data_incomplete_rows": len(audit),
        "reason_counts": reason_counts,
        "sim_status_counts": sim_counts,
        "by_reason": by_reason,
        "rows": audit,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(AUDIT_FIELDS), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def render_md(report: dict[str, Any]) -> str:
    lines = [
        "# S34 V Engine Data-Incomplete Audit",
        "",
        f"Generated: `{report['generated_at_utc']}`",
        "",
        f"Protocol: `{report['protocol_id']}`",
        "",
        "This audits observation rows that could not be closed because the simulated exit could not be priced from `book_ticker`.",
        "",
        "## Summary",
        "",
        f"- ledger rows: `{report['ledger_rows']}`",
        f"- data incomplete rows: `{report['data_incomplete_rows']}`",
        f"- max allowed book staleness: `{report['max_staleness_sec']}s`",
        f"- reason counts: `{report['reason_counts']}`",
        f"- sim status counts: `{report['sim_status_counts']}`",
        "",
        "## Reasons",
        "",
        "| Reason | N | First signal | Last signal | Staleness sec | Next book gap sec |",
        "| --- | ---: | --- | --- | --- | --- |",
    ]
    for reason, data in report["by_reason"].items():
        lines.append(
            f"| `{reason}` | {data['n']} | {data['first_signal_utc']} | {data['last_signal_utc']} | "
            f"{data['book_staleness_sec']} | {data['next_book_gap_sec']} |"
        )
    lines.extend(["", "## Incomplete Rows", ""])
    lines.append("| Signal UTC | Sim | Reason | Fill UTC | Expected exit | Nearest book | Stale sec | Next gap sec |")
    lines.append("| --- | --- | --- | --- | --- | --- | ---: | ---: |")
    for row in report["rows"]:
        lines.append(
            f"| {row.get('signal_utc')} | {row.get('sim_status')} | `{row.get('reason')}` | {row.get('maker_fill_utc')} | "
            f"{row.get('expected_exit_utc')} | {row.get('nearest_book_utc')} | {row.get('book_staleness_sec')} | {row.get('book_gap_sec')} |"
        )
    lines.append("")
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Audit S34 V Engine DATA_INCOMPLETE rows.")
    p.add_argument("--db", type=Path, default=DEFAULT_DB)
    p.add_argument("--ledger-jsonl", type=Path, default=DEFAULT_LEDGER_JSONL)
    p.add_argument("--json-out", type=Path, default=OUT_JSON)
    p.add_argument("--md-out", type=Path, default=OUT_MD)
    p.add_argument("--csv-out", type=Path, default=OUT_CSV)
    p.add_argument("--max-book-staleness-sec", type=int, default=10)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    ledger_rows = load_jsonl(args.ledger_jsonl)
    with sqlite3.connect(f"file:{args.db}?mode=ro", uri=True) as conn:
        audit = audit_rows(conn, ledger_rows, max_staleness_sec=int(args.max_book_staleness_sec))
    report = build_report(
        ledger_rows=ledger_rows,
        audit=audit,
        source_db=file_fingerprint(args.db),
        source_ledger=file_fingerprint(args.ledger_jsonl),
        max_staleness_sec=int(args.max_book_staleness_sec),
    )
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    args.md_out.write_text(render_md(report), encoding="utf-8")
    write_csv(args.csv_out, audit)
    print(render_md(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
