"""Cross-asset absorption pooling for S34 V2 expansion.

Pools BTC/ETH/SOL SELL deep-V fade events with top-of-book absorption features
to increase statistical power. Research-only; no live/paper changes.
"""

from __future__ import annotations

import argparse
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
OUT_JSON = OUT_DIR / "S34_CROSS_ASSET_ABSORPTION_POOL.json"
OUT_MD = OUT_DIR / "S34_CROSS_ASSET_ABSORPTION_POOL.md"

SYMBOL_THRESHOLDS = {
    "BTCUSDT": (250_000.0, 500_000.0, 1_000_000.0),
    "ETHUSDT": (100_000.0, 150_000.0, 200_000.0),
    "SOLUSDT": (25_000.0, 50_000.0, 100_000.0),
}
VDEPTH_BANDS = ((20.0, 28.0), (28.0, 40.0), (40.0, 60.0), (60.0, None))
HORIZON_SEC = 4 * 3600
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


def band_label(depth: float) -> str:
    for lo, hi in VDEPTH_BANDS:
        if float(depth) >= lo and (hi is None or float(depth) < hi):
            return f"v{int(lo)}_{'inf' if hi is None else int(hi)}"
    return "v_unknown"


def collect_symbol_rows(
    conn: sqlite3.Connection,
    *,
    symbol: str,
    threshold: float,
    max_book_staleness_sec: int,
    fee_bps_side: float,
) -> list[dict[str, Any]]:
    marks = load_mark_index(conn, symbol)
    liqs = load_liquidations(conn, symbol, "SELL", None, None)
    anchors = reconstruct_anchors(
        liqs,
        bucket_sec=BUCKET_SEC,
        min_gap_sec=MIN_GAP_SEC,
        thresholds=(float(threshold),),
        accel_window_sec=ACCEL_WINDOW_SEC,
    )
    rows: list[dict[str, Any]] = []
    for anchor in anchors:
        start = marks.at_or_after(int(anchor.first_ts_ms))
        mark_entry = marks.at_or_after(int(anchor.anchor_ts_ms))
        if not start or not mark_entry or float(start[1]) <= 0.0:
            continue
        depth = (float(start[1]) - float(mark_entry[1])) / float(start[1]) * 10_000.0
        if depth < min(lo for lo, _ in VDEPTH_BANDS):
            continue
        entry_ts = int(anchor.anchor_ts_ms)
        exit_ts = entry_ts + HORIZON_SEC * 1000
        eb = book_at(conn, symbol, entry_ts, int(max_book_staleness_sec))
        xb = book_at(conn, symbol, exit_ts, int(max_book_staleness_sec))
        bf = book_features_at(conn, symbol, entry_ts, int(max_book_staleness_sec))
        if not eb or not xb or not bf:
            continue
        gross = signed_return_bps("LONG", float(eb.ask), float(xb.bid))
        net = gross - 2.0 * float(fee_bps_side)
        rows.append(
            {
                "symbol": symbol,
                "threshold_usd": float(threshold),
                "route_id": f"{symbol}_SELL_FADE_LONG_T{int(float(threshold)/1000)}K_{band_label(depth)}_H4",
                "entry_ts_ms": entry_ts,
                "month": month_of(entry_ts),
                "vdepth_bps": r1(depth),
                "vdepth_band": band_label(depth),
                "net_bps": float(net),
                "gross_bps": r1(gross),
                "running_notional": r1(anchor.running_notional),
                "running_liq_count": int(anchor.running_liq_count),
                "running_accel": r1(anchor.running_accel),
                **bf,
            }
        )
    return rows


def quantile(rows: list[dict[str, Any]], key: str, q: float, *, symbol: str | None = None) -> float | None:
    vals = [
        float(r[key])
        for r in rows
        if (symbol is None or r.get("symbol") == symbol) and r.get(key) is not None and math.isfinite(float(r[key]))
    ]
    return pctile(vals, q) if vals else None


def label_rows(rows: list[dict[str, Any]], *, per_symbol_cuts: bool) -> dict[str, Any]:
    cuts: dict[str, Any] = {}
    symbols = sorted({str(r["symbol"]) for r in rows}) if per_symbol_cuts else ["POOL"]
    for sym in symbols:
        subset_symbol = None if sym == "POOL" else sym
        c = {
            "imbalance_med": quantile(rows, "book_imbalance", 0.5, symbol=subset_symbol),
            "bid_depth_med": quantile(rows, "bid_depth_usd", 0.5, symbol=subset_symbol),
            "imbalance_p25": quantile(rows, "book_imbalance", 0.25, symbol=subset_symbol),
            "bid_depth_p25": quantile(rows, "bid_depth_usd", 0.25, symbol=subset_symbol),
        }
        cuts[sym] = {k: r1(v) if isinstance(v, float) else v for k, v in c.items()}
        for row in rows:
            if subset_symbol is not None and row.get("symbol") != subset_symbol:
                continue
            imb = float(row["book_imbalance"])
            bid = float(row["bid_depth_usd"])
            row["imbalance_gate"] = "bid_support" if c["imbalance_med"] is not None and imb >= float(c["imbalance_med"]) else "ask_heavy"
            row["bid_depth_gate"] = "deep_bid" if c["bid_depth_med"] is not None and bid >= float(c["bid_depth_med"]) else "shallow_bid"
            if row["imbalance_gate"] == "bid_support" and row["bid_depth_gate"] == "deep_bid":
                row["absorption_gate"] = "absorbed"
            elif (
                c["imbalance_p25"] is not None
                and c["bid_depth_p25"] is not None
                and imb <= float(c["imbalance_p25"])
                and bid <= float(c["bid_depth_p25"])
            ):
                row["absorption_gate"] = "vacuum_like"
            else:
                row["absorption_gate"] = "mixed"
    return cuts


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


def group_table(rows: list[dict[str, Any]], hold_months: set[str], key: str) -> list[dict[str, Any]]:
    groups = sorted({str(r.get(key)) for r in rows})
    out = []
    for group in groups:
        sub = [r for r in rows if str(r.get(key)) == group]
        out.append({"group": group, "summary": summarize_rows(sub, hold_months)})
    out.sort(key=lambda r: (float(r["summary"]["all"]["t3r_bps"] or -1e18), float(r["summary"]["all"]["sum_bps"] or -1e18)), reverse=True)
    return out


def best_routes(rows: list[dict[str, Any]], hold_months: set[str], min_n: int) -> list[dict[str, Any]]:
    route_ids = sorted({str(r["route_id"]) for r in rows})
    out = []
    for rid in route_ids:
        sub = [r for r in rows if r["route_id"] == rid]
        if len(sub) < int(min_n):
            continue
        out.append({"route_id": rid, "summary": summarize_rows(sub, hold_months)})
    out.sort(key=lambda r: (float(r["summary"]["all"]["t3r_bps"] or -1e18), float(r["summary"]["hold"]["t3r_bps"] or -1e18)), reverse=True)
    return out


def build_report(
    conn: sqlite3.Connection,
    *,
    db_path: Path,
    symbols: tuple[str, ...],
    max_book_staleness_sec: int,
    fee_bps_side: float,
    holdout_frac: float,
    per_symbol_cuts: bool,
    min_route_n: int,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for symbol in symbols:
        for threshold in SYMBOL_THRESHOLDS[symbol]:
            rows.extend(
                collect_symbol_rows(
                    conn,
                    symbol=symbol,
                    threshold=float(threshold),
                    max_book_staleness_sec=max_book_staleness_sec,
                    fee_bps_side=fee_bps_side,
                )
            )
    rows.sort(key=lambda r: int(r["entry_ts_ms"]))
    cuts = label_rows(rows, per_symbol_cuts=per_symbol_cuts)
    hold_months, split = split_months(rows, holdout_frac)
    return {
        "generated_at_utc": utc_now(),
        "source_db": file_fingerprint(db_path),
        "config": {
            "symbols": list(symbols),
            "symbol_thresholds": {k: list(v) for k, v in SYMBOL_THRESHOLDS.items() if k in symbols},
            "vdepth_bands": [[lo, hi] for lo, hi in VDEPTH_BANDS],
            "horizon_hr": 4,
            "fee_bps_side": float(fee_bps_side),
            "per_symbol_cuts": bool(per_symbol_cuts),
            "max_book_staleness_sec": int(max_book_staleness_sec),
        },
        "split": split,
        "cuts": cuts,
        "event_n": len(rows),
        "overall": summarize_rows(rows, hold_months),
        "groups": {
            "symbol": group_table(rows, hold_months, "symbol"),
            "vdepth_band": group_table(rows, hold_months, "vdepth_band"),
            "imbalance_gate": group_table(rows, hold_months, "imbalance_gate"),
            "bid_depth_gate": group_table(rows, hold_months, "bid_depth_gate"),
            "absorption_gate": group_table(rows, hold_months, "absorption_gate"),
            "symbol_x_absorption": group_table([{**r, "symbol_x_absorption": f"{r['symbol']}:{r['absorption_gate']}"} for r in rows], hold_months, "symbol_x_absorption"),
        },
        "routes": best_routes(rows, hold_months, min_route_n),
        "rows": rows,
    }


def cell(s: dict[str, Any]) -> str:
    return (
        f"N={s['n']} sum={s['sum_bps']} mean={s['mean_bps']} med={s['median_bps']} "
        f"win={None if s['win_rate'] is None else r1(s['win_rate'] * 100.0)} "
        f"T3R={s['t3r_bps']} max_loss={s['max_loss_bps']} tail<-100={s['tail_n_lt_-100']}"
    )


def render_md(report: dict[str, Any]) -> str:
    lines = [
        "# S34 Cross-Asset Absorption Pool",
        "",
        f"Generated: `{report['generated_at_utc']}`",
        "",
        "Research-only. Pools BTC/ETH/SOL SELL deep-V real-fill events with T=0 book absorption; no live/paper state changed.",
        "",
        f"Events: `{report['event_n']}`; holdout months `{report['split']['holdout_months']}`; per-symbol cuts `{report['config']['per_symbol_cuts']}`",
        "",
        "## Overall",
        "",
        f"- All: {cell(report['overall']['all'])}",
        f"- Calibration: {cell(report['overall']['cal'])}",
        f"- Holdout: {cell(report['overall']['hold'])}",
        "",
    ]
    for name, rows in report["groups"].items():
        lines.extend([f"## {name}", "", "| Group | All | Cal | Hold |", "| --- | --- | --- | --- |"])
        for row in rows:
            s = row["summary"]
            lines.append(f"| `{row['group']}` | {cell(s['all'])} | {cell(s['cal'])} | {cell(s['hold'])} |")
        lines.append("")
    lines.extend(["## Route Candidates", "", "| Route | All | Cal | Hold |", "| --- | --- | --- | --- |"])
    for row in report["routes"][:30]:
        s = row["summary"]
        lines.append(f"| `{row['route_id']}` | {cell(s['all'])} | {cell(s['cal'])} | {cell(s['hold'])} |")
    lines.extend(["", "## Read", ""])
    gate_map = {r["group"]: r["summary"]["all"] for r in report["groups"]["bid_depth_gate"]}
    deep = gate_map.get("deep_bid")
    shallow = gate_map.get("shallow_bid")
    if deep and shallow:
        lines.append(
            f"- Pooled deep_bid vs shallow_bid delta T3R `{r1(float(deep['t3r_bps'] or 0.0) - float(shallow['t3r_bps'] or 0.0))}`, "
            f"delta max_loss `{r1(float(deep['max_loss_bps'] or 0.0) - float(shallow['max_loss_bps'] or 0.0))}`."
        )
    lines.append("- Cross-asset pooling only helps if the absorption relation is directionally stable across symbols and survives holdout.")
    lines.append("")
    return "\n".join(lines)


def parse_symbols(text: str) -> tuple[str, ...]:
    vals = tuple(part.strip().upper() for part in str(text).split(",") if part.strip())
    if not vals:
        raise ValueError("empty symbols")
    unknown = [v for v in vals if v not in SYMBOL_THRESHOLDS]
    if unknown:
        raise ValueError(f"unknown symbols: {unknown}")
    return vals


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run S34 cross-asset absorption pool.")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--symbols", default="BTCUSDT,ETHUSDT,SOLUSDT")
    parser.add_argument("--max-book-staleness-sec", type=int, default=10)
    parser.add_argument("--fee-bps-side", type=float, default=3.05)
    parser.add_argument("--holdout-frac", type=float, default=0.30)
    parser.add_argument("--pool-cuts", action="store_true", help="Use pooled feature medians instead of per-symbol medians.")
    parser.add_argument("--min-route-n", type=int, default=5)
    parser.add_argument("--json-out", type=Path, default=OUT_JSON)
    parser.add_argument("--md-out", type=Path, default=OUT_MD)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    symbols = parse_symbols(args.symbols)
    with sqlite3.connect(f"file:{args.db}?mode=ro", uri=True) as conn:
        report = build_report(
            conn,
            db_path=args.db,
            symbols=symbols,
            max_book_staleness_sec=int(args.max_book_staleness_sec),
            fee_bps_side=float(args.fee_bps_side),
            holdout_frac=float(args.holdout_frac),
            per_symbol_cuts=not bool(args.pool_cuts),
            min_route_n=int(args.min_route_n),
        )
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    args.md_out.write_text(render_md(report), encoding="utf-8")
    print(render_md(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
