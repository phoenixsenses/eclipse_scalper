from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Paper vs backtest reconciliation summary.")
    p.add_argument("--paper-db", default="data/paper_trades.db")
    p.add_argument("--rank-json", default="reports/PASSIVE_POCKET_RANKING.json")
    p.add_argument("--out", default="reports/RECONCILIATION.md")
    return p.parse_args()


def main() -> int:
    args = _args()
    paper_db = Path(args.paper_db)
    rank_json = Path(args.rank_json)
    if not paper_db.exists():
        print(f"reconcile: missing paper db {paper_db}")
        return 2
    conn = sqlite3.connect(str(paper_db), check_same_thread=False)
    try:
        row = conn.execute(
            "SELECT COUNT(*) n, COALESCE(SUM(pnl_bps),0) pnl, "
            "COALESCE(AVG(CASE WHEN pnl_bps>0 THEN 1.0 ELSE 0.0 END),0) wr "
            "FROM trades"
        ).fetchone()
    finally:
        conn.close()
    paper_n = int(row[0] or 0)
    paper_pnl = float(row[1] or 0.0)
    paper_wr = float(row[2] or 0.0)
    bt_n = 0
    bt_npa = 0.0
    if rank_json.exists():
        try:
            obj = json.loads(rank_json.read_text(encoding="utf-8", errors="replace"))
            ranking = obj.get("ranking") if isinstance(obj, dict) else None
            if isinstance(ranking, list):
                bt_n = len(ranking)
                if ranking:
                    bt_npa = float(ranking[0].get("npa_core", 0.0) or 0.0)
        except Exception:
            pass
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "\n".join(
            [
                "# RECONCILIATION",
                "",
                "## Paper",
                f"- trades: {paper_n}",
                f"- pnl_bps: {paper_pnl:+.2f}",
                f"- win_rate: {paper_wr*100.0:.1f}%",
                "",
                "## Backtest (rank snapshot)",
                f"- pockets: {bt_n}",
                f"- top_npa_core: {bt_npa:+.6f}",
                "",
                "## Note",
                "Use this report as a quick consistency check. Trade-level matching can be extended later.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"reconcile: wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

