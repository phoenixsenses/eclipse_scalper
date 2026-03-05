from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.microphys.replay import compute_replay_parity, load_live_fill_rows, load_simulated_fill_rows


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate replay parity report from simulated and live fills.")
    p.add_argument("--sim", default="logs/micro_edge_debug_trades.jsonl", help="Simulated fill rows (jsonl/json).")
    p.add_argument("--live-db", default="data/paper_trades.db", help="Live/paper SQLite DB path.")
    p.add_argument("--live-table", default="trades")
    p.add_argument("--match-window-sec", type=float, default=30.0)
    p.add_argument("--out-md", default="reports/REPLAY_PARITY_REPORT.md")
    p.add_argument("--out-json", default="reports/REPLAY_PARITY_REPORT.json")
    return p.parse_args()


def _render_md(d: dict) -> str:
    lines = [
        "# REPLAY PARITY REPORT",
        "",
        "## Summary",
        f"- sim_count: {int(d.get('sim_count', 0))}",
        f"- live_count: {int(d.get('live_count', 0))}",
        f"- matched_count: {int(d.get('matched_count', 0))}",
        f"- match_rate_vs_sim: {float(d.get('match_rate_vs_sim', 0.0)):.2%}",
        f"- sim_fill_rate: {float(d.get('sim_fill_rate', 0.0)):.2%}",
        f"- live_fill_rate: {float(d.get('live_fill_rate', 0.0)):.2%}",
        f"- fill_rate_delta: {float(d.get('fill_rate_delta', 0.0)):+.4f}",
        f"- mean_abs_dt_sec: {float(d.get('mean_abs_dt_sec', 0.0)):.3f}",
        f"- mean_fill_delay_delta_sec: {float(d.get('mean_fill_delay_delta_sec', 0.0)):+.3f}",
        f"- mean_pnl_bps_delta: {float(d.get('mean_pnl_bps_delta', 0.0)):+.4f}",
        f"- mean_adverse_bps_delta: {float(d.get('mean_adverse_bps_delta', 0.0)):+.4f}",
        "",
        "## Matched Sample (first 20)",
        "| symbol | side | dt_sec | sim_pnl_bps | live_pnl_bps | pnl_delta | sim_adv | live_adv | adv_delta |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for m in list(d.get("matches") or [])[:20]:
        lines.append(
            f"| {m.get('symbol','')} | {m.get('side','')} | {float(m.get('dt_sec',0.0)):+.3f} | "
            f"{float(m.get('sim_pnl_bps',0.0)):+.4f} | {float(m.get('live_pnl_bps',0.0)):+.4f} | "
            f"{float(m.get('pnl_bps_delta',0.0)):+.4f} | {float(m.get('sim_max_adverse_bps',0.0)):+.4f} | "
            f"{float(m.get('live_max_adverse_bps',0.0)):+.4f} | {float(m.get('adverse_bps_delta',0.0)):+.4f} |"
        )
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    args = _args()
    sim_rows = load_simulated_fill_rows(args.sim)
    live_rows = load_live_fill_rows(args.live_db, table=str(args.live_table))
    result = compute_replay_parity(sim_rows, live_rows, match_window_sec=float(args.match_window_sec))
    d = result.to_dict()

    out_json = Path(str(args.out_json))
    out_md = Path(str(args.out_md))
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(d, ensure_ascii=True, indent=2), encoding="utf-8")
    out_md.write_text(_render_md(d), encoding="utf-8")

    print(
        f"replay_parity_report sim={int(d.get('sim_count',0))} "
        f"live={int(d.get('live_count',0))} matched={int(d.get('matched_count',0))} "
        f"match_rate={float(d.get('match_rate_vs_sim',0.0)):.2%} out_md={out_md} out_json={out_json}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

