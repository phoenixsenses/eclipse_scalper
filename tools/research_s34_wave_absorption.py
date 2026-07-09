"""S34 Wave Absorption research.

Tests whether top-of-book liquidity at the ETH SELL deep-V threshold cross
separates clean reverts from runaways. Research-only; no live/paper state
changes.
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

from ami.storage import production as PR
from ami.storage import research_reader as RR

DEFAULT_DB = ROOT / "data" / "microstructure.db"
SOURCE_DB_PATH = (ROOT / "data" / "microstructure.db").as_posix()
OUT_DIR = ROOT / "reports" / "research" / "s34"
OUT_JSON = OUT_DIR / "S34_WAVE_ABSORPTION.json"
OUT_MD = OUT_DIR / "S34_WAVE_ABSORPTION.md"

SYMBOL = "ETHUSDT"
THRESHOLD_USD = 200_000.0
MIN_VDEPTH_BPS = 28.0
HORIZON_SEC = 4 * 3600
BUCKET_SEC = 300
MIN_GAP_SEC = 900
ACCEL_WINDOW_SEC = 30


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def metrics(vals: list[float]) -> dict[str, Any]:
    xs = [float(v) for v in vals if math.isfinite(float(v))]
    if not xs:
        return {
            "n": 0,
            "sum_bps": 0.0,
            "mean_bps": None,
            "median_bps": None,
            "win_rate": None,
            "max_win_bps": None,
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
        "max_win_bps": r1(max(xs)),
        "max_loss_bps": r1(min(xs)),
        "t3r_bps": r1(sum(ordered[3:]) if len(ordered) > 3 else sum(ordered)),
        "tail_n_lt_-100": sum(1 for v in xs if v < -100.0),
        "tail_n_lt_-200": sum(1 for v in xs if v < -200.0),
    }


def month_of(ts_ms: int) -> str:
    return datetime.fromtimestamp(int(ts_ms) / 1000.0, tz=timezone.utc).strftime("%Y-%m")


_BOOK_COLS = ("ts_ms", "bid_price", "bid_qty", "ask_price", "ask_qty", "mid_price", "spread_pct",
              "book_imbalance", "bid_depth_usd")


def book_features_at(conn: sqlite3.Connection, symbol: str, ts_ms: int, max_staleness_sec: int) -> dict[str, Any] | None:
    """Direct-SQL oracle -- kept unchanged as the parity reference for
    `book_features_at_v2` (BATCH-STORAGE-ROTATION-RETENTION-ASOF-LOOKUP-
    CONSUMER-MIGRATION-V7). No longer called by build_rows(); the reader-
    backed path is used instead."""
    row = conn.execute(
        """
        SELECT ts_ms, bid_price, bid_qty, ask_price, ask_qty, mid_price, spread_pct, book_imbalance, bid_depth_usd
        FROM book_ticker
        WHERE symbol=? AND ts_ms<=?
        ORDER BY ts_ms DESC
        LIMIT 1
        """,
        (symbol, int(ts_ms)),
    ).fetchone()
    if not row:
        return None
    staleness_ms = int(ts_ms) - int(row[0])
    if staleness_ms > int(max_staleness_sec) * 1000:
        return None
    bid = float(row[1])
    bid_qty = float(row[2])
    ask = float(row[3])
    ask_qty = float(row[4])
    mid = float(row[5])
    spread_pct = float(row[6])
    imbalance = float(row[7])
    bid_depth = float(row[8]) if row[8] is not None else bid * bid_qty
    ask_depth = ask * ask_qty
    return {
        "book_ts_ms": int(row[0]),
        "book_staleness_ms": staleness_ms,
        "bid": bid,
        "ask": ask,
        "mid": mid,
        "spread_bps": spread_pct * 10_000.0,
        "book_imbalance": imbalance,
        "bid_depth_usd": bid_depth,
        "ask_depth_usd": ask_depth,
        "total_top_depth_usd": bid_depth + ask_depth,
    }


def book_features_at_v2(root, symbol: str, ts_ms: int, max_staleness_sec: int, source_db_path=None) -> dict[str, Any] | None:
    """Reader-backed replacement for `book_features_at`, via
    lookup_latest_at_or_before. `symbol` is a genuine function parameter,
    but this file's sole call site (in `build_rows`) always passes the
    hardcoded module constant `SYMBOL="ETHUSDT"` -- confirmed by source
    inspection, never overridden. book_ticker has no ETHUSDT archive
    partition (only SOLUSDT/2026-04 is archived), so real production use
    of this file resolves SQLITE_ONLY -- confirmed, not assumed. The
    helper itself stays symbol-generic and never assumes SOLUSDT's
    archive coverage applies to a different symbol."""
    result = RR.lookup_latest_at_or_before(
        root, table="book_ticker", symbol=symbol, ts_ms=int(ts_ms),
        columns=_BOOK_COLS, source_db_path=source_db_path)
    if not result.found:
        return None
    row = dict(zip(_BOOK_COLS, result.row))
    staleness_ms = int(ts_ms) - int(row["ts_ms"])
    if staleness_ms > int(max_staleness_sec) * 1000:
        return None
    bid = float(row["bid_price"])
    bid_qty = float(row["bid_qty"])
    ask = float(row["ask_price"])
    ask_qty = float(row["ask_qty"])
    mid = float(row["mid_price"])
    spread_pct = float(row["spread_pct"])
    imbalance = float(row["book_imbalance"])
    bid_depth = float(row["bid_depth_usd"]) if row["bid_depth_usd"] is not None else bid * bid_qty
    ask_depth = ask * ask_qty
    return {
        "book_ts_ms": int(row["ts_ms"]),
        "book_staleness_ms": staleness_ms,
        "bid": bid,
        "ask": ask,
        "mid": mid,
        "spread_bps": spread_pct * 10_000.0,
        "book_imbalance": imbalance,
        "bid_depth_usd": bid_depth,
        "ask_depth_usd": ask_depth,
        "total_top_depth_usd": bid_depth + ask_depth,
    }


def split_by_time(rows: list[dict[str, Any]], holdout_frac: float) -> tuple[set[int], dict[str, Any]]:
    months = sorted({str(r["month"]) for r in rows})
    hold_n = max(1, int(round(len(months) * float(holdout_frac)))) if months else 0
    hold_months = set(months[-hold_n:]) if hold_n else set()
    return hold_months, {"method": "chronological_month_tail", "months": months, "holdout_months": sorted(hold_months)}


def summarize_subset(rows: list[dict[str, Any]], hold_months: set[str]) -> dict[str, Any]:
    all_vals = [float(r["net_bps"]) for r in rows]
    cal_vals = [float(r["net_bps"]) for r in rows if r["month"] not in hold_months]
    hold_vals = [float(r["net_bps"]) for r in rows if r["month"] in hold_months]
    return {"all": metrics(all_vals), "cal": metrics(cal_vals), "hold": metrics(hold_vals)}


def median(rows: list[dict[str, Any]], key: str) -> float | None:
    vals = [float(r[key]) for r in rows if r.get(key) is not None and math.isfinite(float(r[key]))]
    return pctile(vals, 0.5) if vals else None


def percentile(rows: list[dict[str, Any]], key: str, p: float) -> float | None:
    vals = [float(r[key]) for r in rows if r.get(key) is not None and math.isfinite(float(r[key]))]
    return pctile(vals, p) if vals else None


def label_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    cuts = {
        "spread_med": median(rows, "spread_bps"),
        "bid_depth_med": median(rows, "bid_depth_usd"),
        "total_depth_med": median(rows, "total_top_depth_usd"),
        "imbalance_med": median(rows, "book_imbalance"),
        "spread_p75": percentile(rows, "spread_bps", 0.75),
        "bid_depth_p25": percentile(rows, "bid_depth_usd", 0.25),
        "imbalance_p25": percentile(rows, "book_imbalance", 0.25),
    }
    for row in rows:
        row["spread_bucket"] = "tight_spread" if cuts["spread_med"] is not None and float(row["spread_bps"]) <= float(cuts["spread_med"]) else "wide_spread"
        row["bid_depth_bucket"] = "deep_bid" if cuts["bid_depth_med"] is not None and float(row["bid_depth_usd"]) >= float(cuts["bid_depth_med"]) else "shallow_bid"
        row["total_depth_bucket"] = "deep_top" if cuts["total_depth_med"] is not None and float(row["total_top_depth_usd"]) >= float(cuts["total_depth_med"]) else "shallow_top"
        row["imbalance_bucket"] = "bid_support" if cuts["imbalance_med"] is not None and float(row["book_imbalance"]) >= float(cuts["imbalance_med"]) else "ask_heavy"
        absorbed = (
            cuts["spread_med"] is not None
            and cuts["bid_depth_med"] is not None
            and cuts["imbalance_med"] is not None
            and float(row["spread_bps"]) <= float(cuts["spread_med"])
            and float(row["bid_depth_usd"]) >= float(cuts["bid_depth_med"])
            and float(row["book_imbalance"]) >= float(cuts["imbalance_med"])
        )
        vacuum = (
            cuts["spread_p75"] is not None
            and cuts["bid_depth_p25"] is not None
            and cuts["imbalance_p25"] is not None
            and float(row["spread_bps"]) >= float(cuts["spread_p75"])
            and float(row["bid_depth_usd"]) <= float(cuts["bid_depth_p25"])
            and float(row["book_imbalance"]) <= float(cuts["imbalance_p25"])
        )
        if absorbed:
            row["absorption_state"] = "absorbed"
        elif vacuum:
            row["absorption_state"] = "vacuum"
        else:
            row["absorption_state"] = "mixed"
    return {k: r1(v) if isinstance(v, float) else v for k, v in cuts.items()}


def group_report(rows: list[dict[str, Any]], hold_months: set[str], key: str) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row.get(key) or "na"), []).append(row)
    out = []
    for name, sub in sorted(groups.items()):
        out.append({"group": name, "n": len(sub), "summary": summarize_subset(sub, hold_months)})
    out.sort(key=lambda r: (float(r["summary"]["all"]["t3r_bps"] or -1e18), float(r["summary"]["all"]["sum_bps"] or -1e18)), reverse=True)
    return out


def build_rows(conn: sqlite3.Connection, *, root, source_db_path, max_book_staleness_sec: int, cost_bps: float, max_vdepth_bps: float | None) -> list[dict[str, Any]]:
    marks = load_mark_index(conn, SYMBOL)
    liqs = load_liquidations(conn, SYMBOL, "SELL", None, None)
    anchors = reconstruct_anchors(
        liqs,
        bucket_sec=BUCKET_SEC,
        min_gap_sec=MIN_GAP_SEC,
        thresholds=(THRESHOLD_USD,),
        accel_window_sec=ACCEL_WINDOW_SEC,
    )
    rows: list[dict[str, Any]] = []
    for anchor in anchors:
        start = marks.at_or_after(int(anchor.first_ts_ms))
        entry = marks.at_or_after(int(anchor.anchor_ts_ms))
        exit_ = marks.at_or_after(int(anchor.anchor_ts_ms) + HORIZON_SEC * 1000)
        if not start or not entry or not exit_ or float(start[1]) <= 0.0:
            continue
        vdepth = (float(start[1]) - float(entry[1])) / float(start[1]) * 10_000.0
        if float(vdepth) < MIN_VDEPTH_BPS:
            continue
        if max_vdepth_bps is not None and float(vdepth) >= float(max_vdepth_bps):
            continue
        book = book_features_at_v2(root, SYMBOL, int(anchor.anchor_ts_ms), int(max_book_staleness_sec), source_db_path=source_db_path)
        if not book:
            continue
        net = signed_return_bps("LONG", float(entry[1]), float(exit_[1])) - float(cost_bps)
        rows.append(
            {
                "entry_ts_ms": int(anchor.anchor_ts_ms),
                "month": month_of(int(anchor.anchor_ts_ms)),
                "net_bps": float(net),
                "vdepth_bps": r1(vdepth),
                "running_notional": r1(anchor.running_notional),
                "running_liq_count": int(anchor.running_liq_count),
                "running_accel": r1(anchor.running_accel),
                **book,
            }
        )
    rows.sort(key=lambda r: int(r["entry_ts_ms"]))
    return rows


def cards(rows: list[dict[str, Any]], *, worst: bool, n: int = 8) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda r: float(r["net_bps"]), reverse=not worst)
    keep = []
    for row in ordered[:n]:
        keep.append(
            {
                "entry_utc": datetime.fromtimestamp(int(row["entry_ts_ms"]) / 1000.0, tz=timezone.utc).isoformat(),
                "net_bps": r1(row["net_bps"]),
                "absorption_state": row.get("absorption_state"),
                "spread_bps": r1(row.get("spread_bps")),
                "bid_depth_usd": r1(row.get("bid_depth_usd")),
                "book_imbalance": r1(row.get("book_imbalance")),
                "vdepth_bps": r1(row.get("vdepth_bps")),
            }
        )
    return keep


def build_report(conn: sqlite3.Connection, *, root, source_db_path, db_path: Path, max_book_staleness_sec: int, cost_bps: float, holdout_frac: float, max_vdepth_bps: float | None) -> dict[str, Any]:
    rows = build_rows(conn, root=root, source_db_path=source_db_path, max_book_staleness_sec=max_book_staleness_sec, cost_bps=cost_bps, max_vdepth_bps=max_vdepth_bps)
    cuts = label_rows(rows)
    hold_months, split = split_by_time(rows, holdout_frac)
    groups = {
        "absorption_state": group_report(rows, hold_months, "absorption_state"),
        "spread_bucket": group_report(rows, hold_months, "spread_bucket"),
        "bid_depth_bucket": group_report(rows, hold_months, "bid_depth_bucket"),
        "total_depth_bucket": group_report(rows, hold_months, "total_depth_bucket"),
        "imbalance_bucket": group_report(rows, hold_months, "imbalance_bucket"),
    }
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
            "cost_bps": r1(cost_bps),
            "max_book_staleness_sec": int(max_book_staleness_sec),
            "holdout_frac": float(holdout_frac),
        },
        "split": split,
        "cuts": cuts,
        "event_n": len(rows),
        "overall": summarize_subset(rows, hold_months),
        "groups": groups,
        "worst_cards": cards(rows, worst=True),
        "best_cards": cards(rows, worst=False),
        "rows": rows,
    }


def metric_cell(summary: dict[str, Any]) -> str:
    return (
        f"N={summary['n']} sum={summary['sum_bps']} mean={summary['mean_bps']} med={summary['median_bps']} "
        f"win={None if summary['win_rate'] is None else r1(summary['win_rate'] * 100.0)} "
        f"T3R={summary['t3r_bps']} max_loss={summary['max_loss_bps']} tail<-100={summary['tail_n_lt_-100']}"
    )


def render_md(report: dict[str, Any]) -> str:
    cfg = report["config"]
    lines = [
        "# S34 Wave Absorption",
        "",
        f"Generated: `{report['generated_at_utc']}`",
        "",
        "Research-only. Tests whether top-of-book liquidity at the knowable threshold cross separates revert vs runaway. No live/paper state changed.",
        "",
        f"Route: `{cfg['symbol']} SELL deep-V {cfg['min_vdepth_bps']}bps-{cfg['max_vdepth_bps'] or 'inf'}bps, {int(cfg['threshold_usd']/1000)}K, {cfg['horizon_hr']}h LONG fade, cost {cfg['cost_bps']}bps`",
        f"Book coverage rows: `{report['event_n']}`; split `{report['split']['method']}`, holdout months `{report['split']['holdout_months']}`",
        "",
        "## Overall",
        "",
        f"- All: {metric_cell(report['overall']['all'])}",
        f"- Calibration: {metric_cell(report['overall']['cal'])}",
        f"- Holdout: {metric_cell(report['overall']['hold'])}",
        "",
        "## Cuts",
        "",
    ]
    for key, value in report["cuts"].items():
        lines.append(f"- `{key}`: `{value}`")
    for group_name, rows in report["groups"].items():
        lines.extend(["", f"## {group_name}", ""])
        lines.append("| Group | All | Cal | Hold |")
        lines.append("| --- | --- | --- | --- |")
        for row in rows:
            s = row["summary"]
            lines.append(f"| `{row['group']}` | {metric_cell(s['all'])} | {metric_cell(s['cal'])} | {metric_cell(s['hold'])} |")
    lines.extend(["", "## Worst Cards", "", "| UTC | Net | State | Spread | Bid depth | Imbalance | V-depth |"])
    lines.append("| --- | ---: | --- | ---: | ---: | ---: | ---: |")
    for row in report["worst_cards"]:
        lines.append(
            f"| {row['entry_utc']} | {row['net_bps']} | `{row['absorption_state']}` | {row['spread_bps']} | "
            f"{row['bid_depth_usd']} | {row['book_imbalance']} | {row['vdepth_bps']} |"
        )
    lines.extend(["", "## Best Cards", "", "| UTC | Net | State | Spread | Bid depth | Imbalance | V-depth |"])
    lines.append("| --- | ---: | --- | ---: | ---: | ---: | ---: |")
    for row in report["best_cards"]:
        lines.append(
            f"| {row['entry_utc']} | {row['net_bps']} | `{row['absorption_state']}` | {row['spread_bps']} | "
            f"{row['bid_depth_usd']} | {row['book_imbalance']} | {row['vdepth_bps']} |"
        )
    lines.extend(["", "## Read", ""])
    abs_groups = {r["group"]: r["summary"]["all"] for r in report["groups"]["absorption_state"]}
    absorbed = abs_groups.get("absorbed")
    vacuum = abs_groups.get("vacuum")
    if absorbed and vacuum:
        lines.append(
            f"- Absorbed vs vacuum delta max_loss: `{r1(float(absorbed['max_loss_bps'] or 0.0) - float(vacuum['max_loss_bps'] or 0.0))}` bps; "
            f"delta T3R: `{r1(float(absorbed['t3r_bps'] or 0.0) - float(vacuum['t3r_bps'] or 0.0))}` bps."
        )
    lines.append("- Treat this as a separator screen. A useful absorption feature must reduce tails and hold up in the later-month holdout, not just improve median.")
    lines.append("")
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run S34 Wave Absorption research.")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--max-book-staleness-sec", type=int, default=10)
    parser.add_argument("--fee-bps-side", type=float, default=3.05)
    parser.add_argument("--modeled-spread-bps", type=float, default=2.0)
    parser.add_argument("--holdout-frac", type=float, default=0.30)
    parser.add_argument("--max-vdepth-bps", type=float, default=None)
    parser.add_argument("--json-out", type=Path, default=OUT_JSON)
    parser.add_argument("--md-out", type=Path, default=OUT_MD)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cost = 2.0 * float(args.fee_bps_side) + float(args.modeled_spread_bps)
    root, _ = PR.resolve_production_root()
    source_db_path = str(args.db)
    with sqlite3.connect(f"file:{args.db}?mode=ro", uri=True) as conn:
        report = build_report(
            conn,
            root=root,
            source_db_path=source_db_path,
            db_path=args.db,
            max_book_staleness_sec=int(args.max_book_staleness_sec),
            cost_bps=cost,
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
