"""S34 real-fill synchronization x absorption screen.

Combines the standing cross-asset synchronization gate with the new wave
absorption proxy on the real book_ticker subset. Research-only; no live/paper
state changes.
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.research_s34_knowable_anchor_continuation import (
    book_at,
    file_fingerprint,
    load_liquidations,
    load_mark_index,
    mean,
    pctile,
    r1,
    r3,
    reconstruct_anchors,
    signed_return_bps,
)
from tools.research_s34_wave_absorption import book_features_at


DEFAULT_DB = ROOT / "data" / "microstructure.db"
OUT_DIR = ROOT / "reports" / "research" / "s34"
OUT_JSON = OUT_DIR / "S34_SYNC_ABSORPTION_REALFILL.json"
OUT_MD = OUT_DIR / "S34_SYNC_ABSORPTION_REALFILL.md"

SYMBOL = "ETHUSDT"
THRESHOLD_USD = 200_000.0
MIN_VDEPTH_BPS = 28.0
HORIZON_SEC = 4 * 3600
SYNC_WINDOW_SEC = 600
BUCKET_SEC = 300
MIN_GAP_SEC = 900
ACCEL_WINDOW_SEC = 30


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def month_of(ts_ms: int) -> str:
    return datetime.fromtimestamp(int(ts_ms) / 1000.0, tz=timezone.utc).strftime("%Y-%m")


def metrics(vals: list[float]) -> dict[str, Any]:
    xs = [float(v) for v in vals if math.isfinite(float(v))]
    if not xs:
        return {
            "n": 0,
            "sum_bps": 0.0,
            "mean_bps": None,
            "median_bps": None,
            "win_rate": None,
            "max_loss_bps": None,
            "t3r_bps": 0.0,
            "tail_n_lt_-100": 0,
            "tail_n_lt_-200": 0,
        }
    ordered = sorted(xs, reverse=True)
    return {
        "n": len(xs),
        "sum_bps": r1(sum(xs)),
        "mean_bps": r1(mean(xs)),
        "median_bps": r1(pctile(xs, 0.5)),
        "win_rate": r3(sum(1 for v in xs if v > 0.0) / len(xs)),
        "max_loss_bps": r1(min(xs)),
        "t3r_bps": r1(sum(ordered[3:]) if len(ordered) > 3 else sum(ordered)),
        "tail_n_lt_-100": sum(1 for v in xs if v < -100.0),
        "tail_n_lt_-200": sum(1 for v in xs if v < -200.0),
    }


def window_liq(liq_ts: list[int], liq_rows: list[dict[str, Any]], start_ms: int, end_ms: int) -> float:
    lo = bisect.bisect_right(liq_ts, int(start_ms))
    hi = bisect.bisect_right(liq_ts, int(end_ms))
    return sum(float(liq_rows[i]["notional"]) for i in range(lo, hi))


def quantile(rows: list[dict[str, Any]], key: str, q: float) -> float | None:
    vals = [float(r[key]) for r in rows if r.get(key) is not None and math.isfinite(float(r[key]))]
    return pctile(vals, q) if vals else None


def label_absorption(rows: list[dict[str, Any]]) -> dict[str, Any]:
    cuts = {
        "imbalance_med": quantile(rows, "book_imbalance", 0.5),
        "bid_depth_med": quantile(rows, "bid_depth_usd", 0.5),
        "imbalance_p25": quantile(rows, "book_imbalance", 0.25),
        "bid_depth_p25": quantile(rows, "bid_depth_usd", 0.25),
    }
    for row in rows:
        row["imbalance_gate"] = "bid_support" if cuts["imbalance_med"] is not None and float(row["book_imbalance"]) >= float(cuts["imbalance_med"]) else "ask_heavy"
        row["bid_depth_gate"] = "deep_bid" if cuts["bid_depth_med"] is not None and float(row["bid_depth_usd"]) >= float(cuts["bid_depth_med"]) else "shallow_bid"
        row["absorption_gate"] = (
            "absorbed"
            if row["imbalance_gate"] == "bid_support" and row["bid_depth_gate"] == "deep_bid"
            else "vacuum_like"
            if cuts["imbalance_p25"] is not None
            and cuts["bid_depth_p25"] is not None
            and float(row["book_imbalance"]) <= float(cuts["imbalance_p25"])
            and float(row["bid_depth_usd"]) <= float(cuts["bid_depth_p25"])
            else "mixed"
        )
    return {k: r1(v) if isinstance(v, float) else v for k, v in cuts.items()}


def split_months(rows: list[dict[str, Any]], holdout_frac: float) -> tuple[set[str], dict[str, Any]]:
    months = sorted({str(r["month"]) for r in rows})
    hold_n = max(1, int(round(len(months) * float(holdout_frac)))) if months else 0
    hold = set(months[-hold_n:]) if hold_n else set()
    return hold, {"method": "chronological_month_tail", "months": months, "holdout_months": sorted(hold)}


def summarize_rows(rows: list[dict[str, Any]], hold_months: set[str]) -> dict[str, Any]:
    return {
        "all": metrics([float(r["net_bps"]) for r in rows]),
        "cal": metrics([float(r["net_bps"]) for r in rows if r["month"] not in hold_months]),
        "hold": metrics([float(r["net_bps"]) for r in rows if r["month"] in hold_months]),
    }


def group(rows: list[dict[str, Any]], hold_months: set[str], key: str, value: str | None = None) -> dict[str, Any]:
    sub = [r for r in rows if str(r.get(key)) == str(value)] if value is not None else rows
    return summarize_rows(sub, hold_months)


def group_table(rows: list[dict[str, Any]], hold_months: set[str], key: str) -> list[dict[str, Any]]:
    vals = sorted({str(r.get(key)) for r in rows})
    out = [{"group": v, "summary": group(rows, hold_months, key, v)} for v in vals]
    out.sort(key=lambda r: (float(r["summary"]["all"]["t3r_bps"] or -1e18), float(r["summary"]["all"]["sum_bps"] or -1e18)), reverse=True)
    return out


def build_rows(
    conn: sqlite3.Connection,
    *,
    max_book_staleness_sec: int,
    fee_bps_side: float,
    sync_threshold_k: float,
    max_vdepth_bps: float | None,
) -> list[dict[str, Any]]:
    eth_marks = load_mark_index(conn, SYMBOL)
    eth_liq = load_liquidations(conn, SYMBOL, "SELL", None, None)
    btc_liq = load_liquidations(conn, "BTCUSDT", "SELL", None, None)
    sol_liq = load_liquidations(conn, "SOLUSDT", "SELL", None, None)
    btc_ts = [int(r["ts_ms"]) for r in btc_liq]
    sol_ts = [int(r["ts_ms"]) for r in sol_liq]
    anchors = reconstruct_anchors(
        eth_liq,
        bucket_sec=BUCKET_SEC,
        min_gap_sec=MIN_GAP_SEC,
        thresholds=(THRESHOLD_USD,),
        accel_window_sec=ACCEL_WINDOW_SEC,
    )
    rows: list[dict[str, Any]] = []
    for anchor in anchors:
        start = eth_marks.at_or_after(int(anchor.first_ts_ms))
        mark_entry = eth_marks.at_or_after(int(anchor.anchor_ts_ms))
        if not start or not mark_entry or float(start[1]) <= 0.0:
            continue
        vdepth = (float(start[1]) - float(mark_entry[1])) / float(start[1]) * 10_000.0
        if vdepth < MIN_VDEPTH_BPS:
            continue
        if max_vdepth_bps is not None and vdepth >= float(max_vdepth_bps):
            continue
        entry_ts = int(anchor.anchor_ts_ms)
        exit_ts = entry_ts + HORIZON_SEC * 1000
        eb = book_at(conn, SYMBOL, entry_ts, int(max_book_staleness_sec))
        xb = book_at(conn, SYMBOL, exit_ts, int(max_book_staleness_sec))
        bf = book_features_at(conn, SYMBOL, entry_ts, int(max_book_staleness_sec))
        if not eb or not xb or not bf:
            continue
        gross = signed_return_bps("LONG", float(eb.ask), float(xb.bid))
        net = gross - 2.0 * float(fee_bps_side)
        start_sync = entry_ts - SYNC_WINDOW_SEC * 1000
        btc_k = window_liq(btc_ts, btc_liq, start_sync, entry_ts) / 1000.0
        sol_k = window_liq(sol_ts, sol_liq, start_sync, entry_ts) / 1000.0
        market_k = btc_k + sol_k
        rows.append(
            {
                "entry_ts_ms": entry_ts,
                "month": month_of(entry_ts),
                "net_bps": float(net),
                "gross_bps": r1(gross),
                "vdepth_bps": r1(vdepth),
                "btc_sell_liq_k": r1(btc_k),
                "sol_sell_liq_k": r1(sol_k),
                "market_concurrent_k": r1(market_k),
                "sync_gate": "sync" if market_k >= float(sync_threshold_k) else "idio",
                "asset_count_200k": int(btc_k >= 200.0) + int(sol_k >= 200.0) + 1,
                **bf,
            }
        )
    rows.sort(key=lambda r: int(r["entry_ts_ms"]))
    return rows


def build_report(
    conn: sqlite3.Connection,
    *,
    db_path: Path,
    max_book_staleness_sec: int,
    fee_bps_side: float,
    sync_threshold_k: float,
    holdout_frac: float,
    max_vdepth_bps: float | None,
) -> dict[str, Any]:
    rows = build_rows(
        conn,
        max_book_staleness_sec=max_book_staleness_sec,
        fee_bps_side=fee_bps_side,
        sync_threshold_k=sync_threshold_k,
        max_vdepth_bps=max_vdepth_bps,
    )
    cuts = label_absorption(rows)
    hold_months, split = split_months(rows, holdout_frac)
    combos = []
    for sync in ("sync", "idio"):
        for imb in ("bid_support", "ask_heavy"):
            sub = [r for r in rows if r["sync_gate"] == sync and r["imbalance_gate"] == imb]
            combos.append({"combo": f"{sync}+{imb}", "summary": summarize_rows(sub, hold_months)})
    combos.sort(key=lambda r: (float(r["summary"]["all"]["t3r_bps"] or -1e18), float(r["summary"]["all"]["sum_bps"] or -1e18)), reverse=True)
    return {
        "generated_at_utc": utc_now(),
        "source_db": file_fingerprint(db_path),
        "config": {
            "symbol": SYMBOL,
            "side": "SELL",
            "threshold_usd": THRESHOLD_USD,
            "min_vdepth_bps": MIN_VDEPTH_BPS,
            "max_vdepth_bps": max_vdepth_bps,
            "horizon_hr": 4,
            "fee_bps_side": float(fee_bps_side),
            "sync_window_min": SYNC_WINDOW_SEC // 60,
            "sync_threshold_k": float(sync_threshold_k),
            "max_book_staleness_sec": int(max_book_staleness_sec),
        },
        "split": split,
        "cuts": cuts,
        "event_n": len(rows),
        "overall": summarize_rows(rows, hold_months),
        "groups": {
            "sync_gate": group_table(rows, hold_months, "sync_gate"),
            "asset_count_200k": group_table(rows, hold_months, "asset_count_200k"),
            "imbalance_gate": group_table(rows, hold_months, "imbalance_gate"),
            "bid_depth_gate": group_table(rows, hold_months, "bid_depth_gate"),
            "absorption_gate": group_table(rows, hold_months, "absorption_gate"),
        },
        "combos": combos,
        "rows": rows,
    }


def cell(s: dict[str, Any]) -> str:
    return (
        f"N={s['n']} sum={s['sum_bps']} mean={s['mean_bps']} med={s['median_bps']} "
        f"win={None if s['win_rate'] is None else r1(s['win_rate'] * 100.0)} "
        f"T3R={s['t3r_bps']} max_loss={s['max_loss_bps']} tail<-100={s['tail_n_lt_-100']}"
    )


def render_md(report: dict[str, Any]) -> str:
    cfg = report["config"]
    lines = [
        "# S34 Sync x Absorption Real-Fill",
        "",
        f"Generated: `{report['generated_at_utc']}`",
        "",
        "Research-only. Real bid/ask fills on book_ticker subset; no live/paper state changed.",
        "",
        f"Route: `{cfg['symbol']} SELL deep-V {cfg['min_vdepth_bps']}bps-{cfg['max_vdepth_bps'] or 'inf'}bps, {int(cfg['threshold_usd']/1000)}K, {cfg['horizon_hr']}h LONG fade`",
        f"Events: `{report['event_n']}`; holdout months `{report['split']['holdout_months']}`; sync threshold `{cfg['sync_threshold_k']}K`",
        "",
        "## Overall",
        "",
        f"- All: {cell(report['overall']['all'])}",
        f"- Cal: {cell(report['overall']['cal'])}",
        f"- Hold: {cell(report['overall']['hold'])}",
        "",
        "## Cuts",
        "",
    ]
    for key, value in report["cuts"].items():
        lines.append(f"- `{key}`: `{value}`")
    for name, rows in report["groups"].items():
        lines.extend(["", f"## {name}", ""])
        lines.append("| Group | All | Cal | Hold |")
        lines.append("| --- | --- | --- | --- |")
        for row in rows:
            s = row["summary"]
            lines.append(f"| `{row['group']}` | {cell(s['all'])} | {cell(s['cal'])} | {cell(s['hold'])} |")
    lines.extend(["", "## Sync x Imbalance Combos", ""])
    lines.append("| Combo | All | Cal | Hold |")
    lines.append("| --- | --- | --- | --- |")
    for row in report["combos"]:
        s = row["summary"]
        lines.append(f"| `{row['combo']}` | {cell(s['all'])} | {cell(s['cal'])} | {cell(s['hold'])} |")
    lines.extend(["", "## Read", ""])
    combo_map = {r["combo"]: r["summary"]["all"] for r in report["combos"]}
    best = report["combos"][0] if report["combos"] else None
    if best:
        lines.append(f"- Best combo by T3R: `{best['combo']}` -> {cell(best['summary']['all'])}.")
    if "sync+bid_support" in combo_map and "sync+ask_heavy" in combo_map:
        a = combo_map["sync+bid_support"]
        b = combo_map["sync+ask_heavy"]
        lines.append(
            f"- Within sync, bid_support vs ask_heavy delta T3R `{r1(float(a['t3r_bps'] or 0.0) - float(b['t3r_bps'] or 0.0))}`, "
            f"delta max_loss `{r1(float(a['max_loss_bps'] or 0.0) - float(b['max_loss_bps'] or 0.0))}`."
        )
    lines.append("- A valid overlay must improve tail/T3R in holdout, not only the all-sample mean.")
    lines.append("")
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run S34 sync x absorption real-fill screen.")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--max-book-staleness-sec", type=int, default=10)
    parser.add_argument("--fee-bps-side", type=float, default=3.05)
    parser.add_argument("--sync-threshold-k", type=float, default=200.0)
    parser.add_argument("--holdout-frac", type=float, default=0.30)
    parser.add_argument("--max-vdepth-bps", type=float, default=None)
    parser.add_argument("--json-out", type=Path, default=OUT_JSON)
    parser.add_argument("--md-out", type=Path, default=OUT_MD)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    with sqlite3.connect(f"file:{args.db}?mode=ro", uri=True) as conn:
        report = build_report(
            conn,
            db_path=args.db,
            max_book_staleness_sec=int(args.max_book_staleness_sec),
            fee_bps_side=float(args.fee_bps_side),
            sync_threshold_k=float(args.sync_threshold_k),
            holdout_frac=float(args.holdout_frac),
            max_vdepth_bps=args.max_vdepth_bps,
        )
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    args.md_out.write_text(render_md(report), encoding="utf-8")
    print(render_md(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
