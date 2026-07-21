from __future__ import annotations

import argparse
import csv
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import s34_intelligence_ledger as ledger


DEFAULT_DB = Path("data/s34_intelligence.db")
DEFAULT_CSV = Path("reports/research/s34/V4_FORWARD_TRACKER.csv")
DEFAULT_FORWARD_FROM = "2026-06-24T00:00:00+00:00"


def _parse_iso(value: str) -> datetime:
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    if "+" not in text and text.count(":") >= 2:
        text += "+00:00"
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _connect(path: Path) -> sqlite3.Connection:
    if not path.exists():
        raise FileNotFoundError(f"intelligence db not found: {path}")
    con = ledger.connect(path)
    con.row_factory = sqlite3.Row
    return con


def _stats(rows: list[sqlite3.Row]) -> dict[str, Any]:
    total = [float(row["net_bps"]) for row in rows if row["net_bps"] is not None]
    blocked = [float(row["net_bps"]) for row in rows if row["net_bps"] is not None and str(row["action"]) == "would_block"]
    kept = [float(row["net_bps"]) for row in rows if row["net_bps"] is not None and str(row["action"]) != "would_block"]
    return {
        "total_n": len(total),
        "total_cum_net": sum(total),
        "would_block_n": len(blocked),
        "would_block_cum_net": sum(blocked),
        "would_block_win_rate": mean([1 if v > 0 else 0 for v in blocked]) if blocked else None,
        "kept_n": len(kept),
        "kept_cum_net": sum(kept),
        "kept_win_rate": mean([1 if v > 0 else 0 for v in kept]) if kept else None,
        "delta_bps": sum(kept) - sum(total),
    }


def build_tracker(db_path: Path, forward_from_utc: str) -> dict[str, Any]:
    forward_from = _parse_iso(forward_from_utc)
    with _connect(db_path) as con:
        rows = con.execute(
            """
            SELECT
              sg.signal_id,
              sg.guardrail_name,
              sg.action,
              sg.level,
              o.trade_id,
              o.outcome_ts_utc,
              o.exit_ts_ms,
              o.net_bps
            FROM s34_shadow_guardrails sg
            JOIN s34_outcomes o ON o.signal_id=sg.signal_id
            WHERE sg.guardrail_name LIKE '%v4%'
              AND o.net_bps IS NOT NULL
            """
        ).fetchall()
    in_sample = []
    forward = []
    for row in rows:
        outcome_ts = datetime.fromtimestamp(int(row["exit_ts_ms"]) / 1000.0, tz=timezone.utc)
        if outcome_ts < forward_from:
            in_sample.append(row)
        else:
            forward.append(row)
    combined = list(rows)
    forward_stats = _stats(forward)
    if forward_stats["would_block_n"] < 5:
        status = "MIXED"
    elif forward_stats["would_block_cum_net"] < 0:
        status = "CONFIRMING"
    else:
        status = "REVERSING"
    return {
        "run_at_utc": datetime.now(timezone.utc).isoformat(),
        "forward_from_utc": forward_from.isoformat(),
        "in_sample": _stats(in_sample),
        "forward": forward_stats,
        "combined": _stats(combined),
        "status": status,
    }


def append_csv(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    fwd = payload["forward"]
    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "run_at_utc",
                "forward_from_utc",
                "forward_total_n",
                "forward_would_block_n",
                "forward_would_block_cum",
                "forward_kept_n",
                "forward_kept_cum",
                "forward_delta",
                "status",
            ],
        )
        if not exists:
            writer.writeheader()
        writer.writerow(
            {
                "run_at_utc": payload["run_at_utc"],
                "forward_from_utc": payload["forward_from_utc"],
                "forward_total_n": fwd["total_n"],
                "forward_would_block_n": fwd["would_block_n"],
                "forward_would_block_cum": f"{fwd['would_block_cum_net']:.6f}",
                "forward_kept_n": fwd["kept_n"],
                "forward_kept_cum": f"{fwd['kept_cum_net']:.6f}",
                "forward_delta": f"{fwd['delta_bps']:.6f}",
                "status": payload["status"],
            }
        )


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, (int, float)):
        return f"{value:+.1f}" if abs(float(value)) >= 0.01 else "0.0"
    return str(value)


def format_tracker(payload: dict[str, Any]) -> str:
    labels = [("IN-SAMPLE", payload["in_sample"]), ("FORWARD", payload["forward"]), ("COMBINED", payload["combined"])]
    lines = [
        "=== V4 SHADOW GUARDRAIL FORWARD TRACKER ===",
        f"Forward from: {payload['forward_from_utc']}",
        "",
        f"{'':<18} {'IN-SAMPLE':>12} {'FORWARD':>10} {'COMBINED':>10}",
    ]
    fields = [
        ("Total N", "total_n"),
        ("Total cum net", "total_cum_net"),
        ("Would-block N", "would_block_n"),
        ("Would-block cum", "would_block_cum_net"),
        ("Kept N", "kept_n"),
        ("Kept cum net", "kept_cum_net"),
        ("Delta (bps)", "delta_bps"),
    ]
    for label, key in fields:
        lines.append(f"{label:<18} { _fmt(labels[0][1][key]):>12} { _fmt(labels[1][1][key]):>10} { _fmt(labels[2][1][key]):>10}")
    lines.extend(
        [
            "",
            f"FORWARD STATUS: {payload['status']}",
            "  CONFIRMING  : forward would-block cum net < 0",
            "  MIXED       : forward would-block N < 5",
            "  REVERSING   : forward would-block cum net > 0",
            "===========================================",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Track V4 shadow guardrail forward performance.")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--forward_from_utc", "--forward-from-utc", dest="forward_from_utc", default=DEFAULT_FORWARD_FROM)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    args = parser.parse_args()
    try:
        payload = build_tracker(args.db, args.forward_from_utc)
        append_csv(args.csv, payload)
    except Exception as exc:  # noqa: BLE001
        print(f"error: {exc}")
        return 1
    print(format_tracker(payload))
    print(f"CSV appended: {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
