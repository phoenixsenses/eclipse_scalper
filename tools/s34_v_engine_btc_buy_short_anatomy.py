"""Anatomy for the BTC BUY-liq -> maker SHORT weak lead.

Research-only follow-up for the portfolio-map lane:
BTCUSDT_BUY_FADE_SHORT_T250K_V28_40_H4.
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.research_s34_knowable_anchor_continuation import file_fingerprint, iso_ms, load_mark_index, r1
from tools.research_s34_maker_fade import NO_TP_OR_SL, collect_events, simulate_event, split_rows, summarize
from tools.s34_v_engine_portfolio_map import prior_return_bps, utc_now


DEFAULT_DB = ROOT / "data" / "microstructure.db"
OUT_DIR = ROOT / "reports" / "research" / "s34"
OUT_JSON = OUT_DIR / "S34_BTC_BUY_SHORT_WEAK_LEAD_ANATOMY.json"
OUT_MD = OUT_DIR / "S34_BTC_BUY_SHORT_WEAK_LEAD_ANATOMY.md"

SYMBOL = "BTCUSDT"
LIQ_SIDE = "BUY"
THRESHOLD_USD = 250_000.0
V_MIN = 28.0
V_MAX = 40.0
PRIOR4H_GT = 50.0
HORIZON_SEC = 4 * 3600
OFFSET_BPS = 20.0
CROSS_MARGIN_BPS = 1.0
BUCKET_SEC = 300
MIN_GAP_SEC = 900
ACCEL_WINDOW_SEC = 30


def session(hour: int) -> str:
    if 0 <= hour < 8:
        return "asia_00_08"
    if 8 <= hour < 13:
        return "eu_08_13"
    if 13 <= hour < 20:
        return "us_13_20"
    return "late_20_24"


def bin_label(value: float | None, cuts: tuple[float, ...], labels: tuple[str, ...]) -> str:
    if value is None or not math.isfinite(float(value)):
        return "na"
    x = float(value)
    for cut, label in zip(cuts, labels):
        if x < cut:
            return label
    return labels[-1]


def summarize_group(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    groups: dict[str, list[float]] = {}
    for row in rows:
        val = str(row.get(key) or "na")
        net = row.get("net_bps")
        if net is None:
            continue
        groups.setdefault(val, []).append(float(net))
    out = [{"group": k, "summary": summarize(v)} for k, v in sorted(groups.items())]
    out.sort(key=lambda r: (int(r["summary"]["n"] or 0), float(r["summary"]["sum_bps"] or 0.0)), reverse=True)
    return out


def cards(rows: list[dict[str, Any]], *, reverse: bool, n: int) -> list[dict[str, Any]]:
    filled = [r for r in rows if r.get("net_bps") is not None]
    filled.sort(key=lambda r: float(r["net_bps"]), reverse=reverse)
    keep = []
    for row in filled[:n]:
        keep.append(
            {
                "anchor_utc": row.get("anchor_utc"),
                "split": row.get("split"),
                "net_bps": r1(row.get("net_bps")),
                "vdepth_bps": r1(row.get("vdepth_bps")),
                "prior_4h_bps": r1(row.get("prior_4h_bps")),
                "fill_delay_sec": r1(row.get("fill_delay_sec")),
                "running_notional": r1(row.get("running_notional")),
                "running_liq_count": row.get("running_liq_count"),
                "elapsed_since_first_sec": r1(row.get("elapsed_since_first_sec")),
                "single_liq_dominance": r1(row.get("running_single_liq_dominance")),
                "session": row.get("session"),
            }
        )
    return keep


def build_report(conn: sqlite3.Connection, *, db_path: Path, max_book_staleness_sec: int) -> dict[str, Any]:
    marks = load_mark_index(conn, SYMBOL)
    events = collect_events(
        conn,
        symbol=SYMBOL,
        threshold=THRESHOLD_USD,
        sides=(LIQ_SIDE,),
        min_vdepth_bps=V_MIN,
        bucket_sec=BUCKET_SEC,
        min_gap_sec=MIN_GAP_SEC,
        accel_window_sec=ACCEL_WINDOW_SEC,
        max_horizon_sec=HORIZON_SEC,
    )
    rows: list[dict[str, Any]] = []
    for ev in events:
        if not (V_MIN <= float(ev.vdepth_bps) < V_MAX):
            continue
        prior4h = prior_return_bps(marks, int(ev.anchor.anchor_ts_ms), 4 * 3600)
        if prior4h is None or not math.isfinite(float(prior4h)) or not (float(prior4h) > PRIOR4H_GT):
            continue
        sim = simulate_event(
            conn,
            ev,
            offset_bps=OFFSET_BPS,
            cross_margin_bps=CROSS_MARGIN_BPS,
            horizon_sec=HORIZON_SEC,
            maker_fee_bps=2.0,
            taker_fee_bps=3.05,
            max_book_staleness_sec=int(max_book_staleness_sec),
            horizon_from="fill",
            tp_bps=NO_TP_OR_SL,
            sl_bps=NO_TP_OR_SL,
        )
        sim["prior_4h_bps"] = r1(prior4h)
        sim["anchor_utc"] = iso_ms(ev.anchor.anchor_ts_ms)
        hour = int(str(sim["anchor_utc"])[11:13])
        sim["session"] = session(hour)
        sim["vdepth_bin"] = bin_label(float(ev.vdepth_bps), (32.0, 36.0, 40.0), ("v28_32", "v32_36", "v36_40"))
        sim["prior_bin"] = bin_label(float(prior4h), (100.0, 200.0, 400.0), ("p50_100", "p100_200", "p200_400"))
        sim["fill_delay_bin"] = bin_label(sim.get("fill_delay_sec"), (60.0, 300.0, 900.0), ("fill_lt60s", "fill_1_5m", "fill_5_15m"))
        if sim.get("status") == "FILLED" and sim.get("net_bps") is not None:
            rows.append(sim)
    rows.sort(key=lambda r: int(r["bucket"]))
    cal, hold, split = split_rows(rows, 0.30)
    groups = {
        "session": summarize_group(rows, "session"),
        "vdepth_bin": summarize_group(rows, "vdepth_bin"),
        "prior_bin": summarize_group(rows, "prior_bin"),
        "fill_delay_bin": summarize_group(rows, "fill_delay_bin"),
    }
    return {
        "generated_at_utc": utc_now(),
        "source_db": file_fingerprint(db_path),
        "route_id": "BTCUSDT_BUY_FADE_SHORT_T250K_V28_40_H4",
        "config": {
            "symbol": SYMBOL,
            "liq_side": LIQ_SIDE,
            "direction": "SHORT",
            "threshold_usd": THRESHOLD_USD,
            "vdepth": [V_MIN, V_MAX],
            "prior4h_gt_bps": PRIOR4H_GT,
            "horizon_hr": 4,
            "offset_bps": OFFSET_BPS,
            "cross_margin_bps": CROSS_MARGIN_BPS,
        },
        "split": split,
        "summary": {
            "calibration": summarize([float(r["net_bps"]) for r in cal]),
            "holdout": summarize([float(r["net_bps"]) for r in hold]),
            "overall": summarize([float(r["net_bps"]) for r in rows]),
        },
        "groups": groups,
        "top_winners": cards(rows, reverse=True, n=8),
        "top_losers": cards(rows, reverse=False, n=8),
        "rows": rows,
    }


def cell(summary: dict[str, Any]) -> str:
    return f"N={summary['n']} sum={summary['sum_bps']} med={summary['median_bps']} T3R={summary['top3_winner_removed_sum_bps']} max_loss={summary['max_loss_bps']}"


def render_md(report: dict[str, Any]) -> str:
    lines = [
        "# S34 BTC BUY -> Maker SHORT Weak-Lead Anatomy",
        "",
        f"Generated: `{report['generated_at_utc']}`",
        "",
        "Research-only anatomy for `BTCUSDT_BUY_FADE_SHORT_T250K_V28_40_H4`; no live/paper state changed.",
        "",
        "## Summary",
        "",
        f"- Calibration: {cell(report['summary']['calibration'])}",
        f"- Holdout: {cell(report['summary']['holdout'])}",
        f"- Overall: {cell(report['summary']['overall'])}",
        "",
        "## Group Breakdowns",
        "",
    ]
    for group_name, group_rows in report["groups"].items():
        lines.append(f"### {group_name}")
        lines.append("")
        lines.append("| Group | Summary |")
        lines.append("| --- | --- |")
        for row in group_rows:
            lines.append(f"| `{row['group']}` | {cell(row['summary'])} |")
        lines.append("")
    lines.extend(["## Top Winners", "", "| UTC | Split | Net | Vdepth | Prior4h | Fill delay | Session |"])
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | --- |")
    for row in report["top_winners"]:
        lines.append(
            f"| {row['anchor_utc']} | {row['split']} | {row['net_bps']} | {row['vdepth_bps']} | "
            f"{row['prior_4h_bps']} | {row['fill_delay_sec']} | {row['session']} |"
        )
    lines.extend(["", "## Top Losers", "", "| UTC | Split | Net | Vdepth | Prior4h | Fill delay | Session |"])
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | --- |")
    for row in report["top_losers"]:
        lines.append(
            f"| {row['anchor_utc']} | {row['split']} | {row['net_bps']} | {row['vdepth_bps']} | "
            f"{row['prior_4h_bps']} | {row['fill_delay_sec']} | {row['session']} |"
        )
    lines.extend(["", "## Read", ""])
    hold = report["summary"]["holdout"]
    overall = report["summary"]["overall"]
    if float(hold.get("top3_winner_removed_sum_bps") or 0.0) <= 0.0:
        lines.append("- Holdout T3R is not positive enough for a freeze. Keep as observation lane only.")
    if int(overall.get("n") or 0) < 20:
        lines.append("- N is still thin; anatomy is hypothesis-generating, not confirmation.")
    lines.append("")
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run BTC BUY->SHORT weak lead anatomy.")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--max-book-staleness-sec", type=int, default=10)
    parser.add_argument("--json-out", type=Path, default=OUT_JSON)
    parser.add_argument("--md-out", type=Path, default=OUT_MD)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    with sqlite3.connect(f"file:{args.db}?mode=ro", uri=True) as conn:
        report = build_report(conn, db_path=args.db, max_book_staleness_sec=int(args.max_book_staleness_sec))
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    args.md_out.write_text(render_md(report), encoding="utf-8")
    print(render_md(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
