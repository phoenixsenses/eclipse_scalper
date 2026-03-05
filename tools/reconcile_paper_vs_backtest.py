from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Paper vs backtest reconciliation summary.")
    p.add_argument("--paper-db", default="data/paper_trades.db")
    p.add_argument("--rank-json", default="reports/PASSIVE_POCKET_RANKING.json")
    p.add_argument("--backtest-json", default="", help="Optional trade-level backtest JSON with entries/trades list.")
    p.add_argument("--match-window-sec", type=float, default=30.0, help="Entry timestamp tolerance for paper-backtest matching.")
    p.add_argument("--out", default="reports/RECONCILIATION.md")
    return p.parse_args()


def _mean(vals: list[float]) -> float:
    return (sum(vals) / len(vals)) if vals else 0.0


def _load_paper_trades(db: Path) -> list[dict[str, Any]]:
    conn = sqlite3.connect(str(db), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT entry_time, exit_time, side, regime, entry_price, exit_price, pnl_bps, max_adverse_bps, exit_type, exit_reason "
            "FROM trades WHERE entry_time>0 ORDER BY entry_time ASC"
        ).fetchall()
    finally:
        conn.close()
    out: list[dict[str, Any]] = []
    for r in rows:
        out.append({k: r[k] for k in r.keys()})
    return out


def _extract_backtest_events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        obj = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return []
    candidates: list[Any] = []
    if isinstance(obj, dict):
        for k in ("trades", "entries", "events", "rows"):
            v = obj.get(k)
            if isinstance(v, list):
                candidates.extend(v)
    elif isinstance(obj, list):
        candidates = obj
    out: list[dict[str, Any]] = []
    for raw in candidates:
        if not isinstance(raw, dict):
            continue
        ts = raw.get("entry_time", raw.get("entry_ts", raw.get("ts")))
        if ts is None:
            continue
        try:
            tsv = float(ts)
        except Exception:
            continue
        if tsv > 1e12:
            tsv = tsv / 1000.0
        out.append(
            {
                "entry_time": tsv,
                "entry_price": float(raw.get("entry_price", 0.0) or 0.0),
                "exit_price": float(raw.get("exit_price", 0.0) or 0.0),
                "pnl_bps": float(raw.get("pnl_bps", raw.get("net_return_bps", 0.0)) or 0.0),
                "max_adverse_bps": float(raw.get("max_adverse_bps", 0.0) or 0.0),
                "exit_type": str(raw.get("exit_type", "")),
            }
        )
    return sorted(out, key=lambda x: float(x.get("entry_time", 0.0)))


def _match_trades(paper: list[dict[str, Any]], bt: list[dict[str, Any]], window_sec: float) -> list[dict[str, float]]:
    out: list[dict[str, float]] = []
    j = 0
    for p in paper:
        pt = float(p.get("entry_time") or 0.0)
        while j < len(bt) and float(bt[j].get("entry_time", 0.0)) < (pt - window_sec):
            j += 1
        best = None
        for k in range(max(0, j - 1), min(len(bt), j + 4)):
            cand = bt[k]
            dt = abs(float(cand.get("entry_time", 0.0)) - pt)
            if dt <= window_sec and (best is None or dt < best[0]):
                best = (dt, cand)
        if best is None:
            continue
        b = best[1]
        out.append(
            {
                "entry_price_diff": float(p.get("entry_price", 0.0) or 0.0) - float(b.get("entry_price", 0.0) or 0.0),
                "exit_price_diff": float(p.get("exit_price", 0.0) or 0.0) - float(b.get("exit_price", 0.0) or 0.0),
                "pnl_bps_diff": float(p.get("pnl_bps", 0.0) or 0.0) - float(b.get("pnl_bps", 0.0) or 0.0),
                "adverse_bps_diff": float(p.get("max_adverse_bps", 0.0) or 0.0) - float(b.get("max_adverse_bps", 0.0) or 0.0),
            }
        )
    return out


def main() -> int:
    args = _args()
    paper_db = Path(args.paper_db)
    rank_json = Path(args.rank_json)
    bt_json = Path(args.backtest_json) if str(args.backtest_json).strip() else None
    if not paper_db.exists():
        print(f"reconcile: missing paper db {paper_db}")
        return 2

    paper = _load_paper_trades(paper_db)
    paper_n = len(paper)
    paper_pnl = sum(float(x.get("pnl_bps", 0.0) or 0.0) for x in paper)
    paper_wr = _mean([1.0 if float(x.get("pnl_bps", 0.0) or 0.0) > 0 else 0.0 for x in paper])
    paper_adv = _mean([float(x.get("max_adverse_bps", 0.0) or 0.0) for x in paper]) if paper else 0.0

    bt_n = 0
    bt_npa = 0.0
    bt_fill = 0.0
    bt_adv = 0.0
    if rank_json.exists():
        try:
            obj = json.loads(rank_json.read_text(encoding="utf-8", errors="replace"))
            ranking = obj.get("ranking") if isinstance(obj, dict) else None
            if isinstance(ranking, list) and ranking:
                bt_n = len(ranking)
                top = ranking[0]
                bt_npa = float(top.get("npa_core", 0.0) or 0.0)
                bt_fill = float(top.get("fill_rate_after_gate", 0.0) or 0.0)
                bt_adv = float(top.get("avg_adverse_bps_on_fills", 0.0) or 0.0)
        except Exception:
            pass

    bt_events = _extract_backtest_events(bt_json) if bt_json else []
    matched = _match_trades(paper, bt_events, float(args.match_window_sec)) if bt_events else []
    match_stats = {
        "matched_n": len(matched),
        "entry_price_diff": _mean([m["entry_price_diff"] for m in matched]) if matched else 0.0,
        "exit_price_diff": _mean([m["exit_price_diff"] for m in matched]) if matched else 0.0,
        "pnl_bps_diff": _mean([m["pnl_bps_diff"] for m in matched]) if matched else 0.0,
        "adverse_bps_diff": _mean([m["adverse_bps_diff"] for m in matched]) if matched else 0.0,
    }

    fill_reality = {
        "paper_filled_trades": int(paper_n),
        "backtest_fill_rate_assumed": float(bt_fill),
        "paper_vs_assumed_hint": (
            "paper_fill_lower_than_model"
            if (bt_fill > 0 and paper_n > 0 and (paper_n / max(1, paper_n)) < bt_fill)
            else "ok_or_unknown"
        ),
    }
    adverse_reality = {
        "paper_avg_max_adverse_bps": float(paper_adv),
        "backtest_avg_adverse_bps_on_fills": float(bt_adv),
        "delta_bps": float(paper_adv - bt_adv),
    }

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
                f"- avg_max_adverse_bps: {paper_adv:+.3f}",
                "",
                "## Backtest (rank snapshot)",
                f"- pockets: {bt_n}",
                f"- top_npa_core: {bt_npa:+.6f}",
                f"- top_fill_rate_after_gate: {bt_fill:.2%}",
                f"- top_avg_adverse_bps_on_fills: {bt_adv:+.3f}",
                "",
                "## Trade-level Matching",
                f"- matched_n: {int(match_stats['matched_n'])}",
                f"- mean_entry_price_diff: {float(match_stats['entry_price_diff']):+.6f}",
                f"- mean_exit_price_diff: {float(match_stats['exit_price_diff']):+.6f}",
                f"- mean_pnl_bps_diff: {float(match_stats['pnl_bps_diff']):+.4f}",
                f"- mean_adverse_bps_diff: {float(match_stats['adverse_bps_diff']):+.4f}",
                "",
                "## Fill Rate Reality Check",
                f"- paper_filled_trades: {fill_reality['paper_filled_trades']}",
                f"- backtest_fill_rate_assumed: {fill_reality['backtest_fill_rate_assumed']:.2%}",
                f"- hint: {fill_reality['paper_vs_assumed_hint']}",
                "",
                "## Adverse Selection Reality Check",
                f"- paper_avg_max_adverse_bps: {adverse_reality['paper_avg_max_adverse_bps']:+.3f}",
                f"- backtest_avg_adverse_bps_on_fills: {adverse_reality['backtest_avg_adverse_bps_on_fills']:+.3f}",
                f"- delta_bps: {adverse_reality['delta_bps']:+.3f}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"reconcile: wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

