from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import s34_intelligence_ledger as ledger


DEFAULT_DB = Path("data/s34_intelligence.db")


def _connect(path: Path) -> sqlite3.Connection:
    if not path.exists():
        raise FileNotFoundError(f"intelligence db not found: {path}")
    con = ledger.connect(path)
    con.row_factory = sqlite3.Row
    return con


def tag_all(db_path: Path = DEFAULT_DB, max_gap_sec: int = 900) -> dict[str, Any]:
    with _connect(db_path) as con:
        signals = con.execute(
            """
            SELECT signal_id, signal_ts_ms, symbol, direction, cluster_notional
            FROM s34_signals
            ORDER BY symbol, direction, signal_ts_ms
            """
        ).fetchall()
        same = 0
        adjacent = 0
        written = 0
        for i, left in enumerate(signals):
            for right in signals[i + 1 :]:
                if str(left["symbol"]) != str(right["symbol"]) or str(left["direction"]) != str(right["direction"]):
                    continue
                gap_sec = abs(int(left["signal_ts_ms"]) - int(right["signal_ts_ms"])) / 1000.0
                if gap_sec > max_gap_sec:
                    if str(left["symbol"]) == str(right["symbol"]) and str(left["direction"]) == str(right["direction"]):
                        break
                    continue
                corr_type = ledger.classify_signal_gap(gap_sec)
                if corr_type is None:
                    continue
                if corr_type == "SAME_CASCADE":
                    same += 1
                elif corr_type == "ADJACENT_CASCADE":
                    adjacent += 1
                if ledger.record_cascade_correlation(con, left, right):
                    written += 1
        con.commit()
        total_rows = con.execute("SELECT COUNT(*) FROM s34_cascade_correlations").fetchone()[0]
    return {
        "signals_scanned": len(signals),
        "same_cascade_pairs": same,
        "adjacent_pairs": adjacent,
        "attempted_writes": written,
        "total_rows": total_rows,
    }


def get_correlated_signals(signal_id: str, db_path: str | Path = DEFAULT_DB) -> list[dict[str, Any]]:
    return ledger.get_correlated_signals(signal_id, db_path)


def format_summary(payload: dict[str, Any]) -> str:
    return "\n".join(
        [
            "=== CASCADE CORRELATION TAGGER ===",
            f"Signals scanned    : {payload['signals_scanned']}",
            f"SAME_CASCADE pairs : {payload['same_cascade_pairs']}",
            f"ADJACENT pairs     : {payload['adjacent_pairs']}",
            f"Rows in table      : {payload['total_rows']}",
            "Written to s34_cascade_correlations",
            "==================================",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Tag correlated S34 liquidation cascade signals.")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--max_gap_sec", "--max-gap-sec", dest="max_gap_sec", type=int, default=900)
    args = parser.parse_args()
    try:
        payload = tag_all(args.db, args.max_gap_sec)
    except Exception as exc:  # noqa: BLE001
        print(f"error: {exc}")
        return 1
    print(format_summary(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
