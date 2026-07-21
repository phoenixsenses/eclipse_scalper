"""S34 absorption timing test.

Checks whether the book-absorption separator is visible before/at the threshold
cross, or only after the rebound starts. Research-only; no live/paper changes.
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

from tools.research_s34_sync_absorption_realfill import metrics
from tools.research_s34_wave_absorption import book_features_at
from tools.research_s34_knowable_anchor_continuation import file_fingerprint, pctile, r1


DEFAULT_DB = ROOT / "data" / "microstructure.db"
DEFAULT_SOURCE = ROOT / "reports" / "research" / "s34" / "S34_SYNC_ABSORPTION_REALFILL_V28_40.json"
OUT_DIR = ROOT / "reports" / "research" / "s34"
OUT_JSON = OUT_DIR / "S34_ABSORPTION_TIMING.json"
OUT_MD = OUT_DIR / "S34_ABSORPTION_TIMING.md"

OFFSETS_SEC = (-60, -30, -10, 0, 5, 30)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def quantile(vals: list[float], q: float) -> float | None:
    xs = [float(v) for v in vals if math.isfinite(float(v))]
    return pctile(xs, q) if xs else None


def classify_offset(rows: list[dict[str, Any]], offset_sec: int) -> dict[str, Any]:
    prefix = f"o{offset_sec:+d}s"
    imb_vals = [float(r[f"{prefix}_book_imbalance"]) for r in rows if r.get(f"{prefix}_book_imbalance") is not None]
    bid_vals = [float(r[f"{prefix}_bid_depth_usd"]) for r in rows if r.get(f"{prefix}_bid_depth_usd") is not None]
    cuts = {
        "offset_sec": int(offset_sec),
        "imbalance_med": quantile(imb_vals, 0.5),
        "bid_depth_med": quantile(bid_vals, 0.5),
        "imbalance_p25": quantile(imb_vals, 0.25),
        "bid_depth_p25": quantile(bid_vals, 0.25),
    }
    for row in rows:
        if row.get(f"{prefix}_book_imbalance") is None or row.get(f"{prefix}_bid_depth_usd") is None:
            row[f"{prefix}_imbalance_gate"] = "no_book"
            row[f"{prefix}_bid_depth_gate"] = "no_book"
            row[f"{prefix}_absorption_gate"] = "no_book"
            continue
        imb = float(row[f"{prefix}_book_imbalance"])
        bid = float(row[f"{prefix}_bid_depth_usd"])
        row[f"{prefix}_imbalance_gate"] = "bid_support" if cuts["imbalance_med"] is not None and imb >= float(cuts["imbalance_med"]) else "ask_heavy"
        row[f"{prefix}_bid_depth_gate"] = "deep_bid" if cuts["bid_depth_med"] is not None and bid >= float(cuts["bid_depth_med"]) else "shallow_bid"
        if row[f"{prefix}_imbalance_gate"] == "bid_support" and row[f"{prefix}_bid_depth_gate"] == "deep_bid":
            row[f"{prefix}_absorption_gate"] = "absorbed"
        elif (
            cuts["imbalance_p25"] is not None
            and cuts["bid_depth_p25"] is not None
            and imb <= float(cuts["imbalance_p25"])
            and bid <= float(cuts["bid_depth_p25"])
        ):
            row[f"{prefix}_absorption_gate"] = "vacuum_like"
        else:
            row[f"{prefix}_absorption_gate"] = "mixed"
    return {k: r1(v) if isinstance(v, float) else v for k, v in cuts.items()}


def summarize_group(rows: list[dict[str, Any]], key: str, value: str) -> dict[str, Any]:
    vals = [float(r["net_bps"]) for r in rows if r.get(key) == value and r.get("net_bps") is not None]
    return metrics(vals)


def offset_report(rows: list[dict[str, Any]], offset_sec: int) -> dict[str, Any]:
    prefix = f"o{offset_sec:+d}s"
    return {
        "offset_sec": int(offset_sec),
        "book_n": sum(1 for r in rows if r.get(f"{prefix}_book_imbalance") is not None),
        "imbalance_gate": {
            "bid_support": summarize_group(rows, f"{prefix}_imbalance_gate", "bid_support"),
            "ask_heavy": summarize_group(rows, f"{prefix}_imbalance_gate", "ask_heavy"),
        },
        "bid_depth_gate": {
            "deep_bid": summarize_group(rows, f"{prefix}_bid_depth_gate", "deep_bid"),
            "shallow_bid": summarize_group(rows, f"{prefix}_bid_depth_gate", "shallow_bid"),
        },
        "absorption_gate": {
            "absorbed": summarize_group(rows, f"{prefix}_absorption_gate", "absorbed"),
            "mixed": summarize_group(rows, f"{prefix}_absorption_gate", "mixed"),
            "vacuum_like": summarize_group(rows, f"{prefix}_absorption_gate", "vacuum_like"),
        },
    }


def transition_report(rows: list[dict[str, Any]], a_sec: int, b_sec: int) -> list[dict[str, Any]]:
    a = f"o{a_sec:+d}s_imbalance_gate"
    b = f"o{b_sec:+d}s_imbalance_gate"
    combos = sorted({(str(r.get(a)), str(r.get(b))) for r in rows if r.get(a) != "no_book" and r.get(b) != "no_book"})
    out = []
    for av, bv in combos:
        vals = [float(r["net_bps"]) for r in rows if str(r.get(a)) == av and str(r.get(b)) == bv]
        out.append({"transition": f"{av}->{bv}", "summary": metrics(vals)})
    out.sort(key=lambda r: (float(r["summary"]["t3r_bps"] or -1e18), float(r["summary"]["sum_bps"] or -1e18)), reverse=True)
    return out


def build_report(conn: sqlite3.Connection, *, source_path: Path, db_path: Path, max_book_staleness_sec: int) -> dict[str, Any]:
    source = json.loads(source_path.read_text(encoding="utf-8"))
    rows = []
    for src in source.get("rows", []):
        row = {
            "entry_ts_ms": int(src["entry_ts_ms"]),
            "month": src.get("month"),
            "net_bps": float(src["net_bps"]),
            "source_sync_gate": src.get("sync_gate"),
            "source_imbalance_gate": src.get("imbalance_gate"),
            "source_bid_depth_gate": src.get("bid_depth_gate"),
        }
        for off in OFFSETS_SEC:
            prefix = f"o{off:+d}s"
            feat = book_features_at(conn, "ETHUSDT", int(row["entry_ts_ms"]) + int(off) * 1000, int(max_book_staleness_sec))
            if feat is None:
                row[f"{prefix}_book_imbalance"] = None
                row[f"{prefix}_bid_depth_usd"] = None
                row[f"{prefix}_spread_bps"] = None
                row[f"{prefix}_book_staleness_ms"] = None
            else:
                row[f"{prefix}_book_imbalance"] = feat["book_imbalance"]
                row[f"{prefix}_bid_depth_usd"] = feat["bid_depth_usd"]
                row[f"{prefix}_spread_bps"] = feat["spread_bps"]
                row[f"{prefix}_book_staleness_ms"] = feat["book_staleness_ms"]
        rows.append(row)
    cuts = [classify_offset(rows, off) for off in OFFSETS_SEC]
    offsets = [offset_report(rows, off) for off in OFFSETS_SEC]
    transitions = {
        "minus30_to_0": transition_report(rows, -30, 0),
        "0_to_plus5": transition_report(rows, 0, 5),
        "minus10_to_0": transition_report(rows, -10, 0),
    }
    return {
        "generated_at_utc": utc_now(),
        "source_report": str(source_path),
        "source_config": source.get("config", {}),
        "source_db": file_fingerprint(db_path),
        "offsets_sec": list(OFFSETS_SEC),
        "event_n": len(rows),
        "overall": metrics([float(r["net_bps"]) for r in rows]),
        "cuts": cuts,
        "offset_reports": offsets,
        "transitions": transitions,
        "rows": rows,
    }


def cell(s: dict[str, Any]) -> str:
    return (
        f"N={s['n']} sum={s['sum_bps']} mean={s['mean_bps']} med={s['median_bps']} "
        f"win={None if s['win_rate'] is None else r1(s['win_rate'] * 100.0)} "
        f"T3R={s['t3r_bps']} max_loss={s['max_loss_bps']} tail<-100={s['tail_n_lt_-100']}"
    )


def render_md(report: dict[str, Any]) -> str:
    cfg = report["source_config"]
    lines = [
        "# S34 Absorption Timing",
        "",
        f"Generated: `{report['generated_at_utc']}`",
        "",
        "Research-only. Tests whether absorption is visible before/at threshold cross or only after entry. No live/paper state changed.",
        "",
        f"Source route: `{cfg.get('symbol')} SELL deep-V {cfg.get('min_vdepth_bps')}bps-{cfg.get('max_vdepth_bps') or 'inf'}bps, {cfg.get('horizon_hr')}h real-fill`",
        f"Rows: `{report['event_n']}`",
        "",
        f"Overall: {cell(report['overall'])}",
        "",
        "## Offset Summary",
        "",
        "| Offset | Book N | Bid Support | Ask Heavy | Deep Bid | Shallow Bid | Absorbed | Vacuum-like |",
        "| ---: | ---: | --- | --- | --- | --- | --- | --- |",
    ]
    for row in report["offset_reports"]:
        lines.append(
            f"| {row['offset_sec']}s | {row['book_n']} | "
            f"{cell(row['imbalance_gate']['bid_support'])} | {cell(row['imbalance_gate']['ask_heavy'])} | "
            f"{cell(row['bid_depth_gate']['deep_bid'])} | {cell(row['bid_depth_gate']['shallow_bid'])} | "
            f"{cell(row['absorption_gate']['absorbed'])} | {cell(row['absorption_gate']['vacuum_like'])} |"
        )
    lines.extend(["", "## Transitions", ""])
    for name, rows in report["transitions"].items():
        lines.append(f"### {name}")
        lines.append("")
        lines.append("| Transition | Summary |")
        lines.append("| --- | --- |")
        for row in rows:
            lines.append(f"| `{row['transition']}` | {cell(row['summary'])} |")
        lines.append("")
    lines.extend(["## Read", ""])
    by_offset = {r["offset_sec"]: r for r in report["offset_reports"]}
    for off in (-30, 0, 5):
        r = by_offset.get(off)
        if not r:
            continue
        bid = r["imbalance_gate"]["bid_support"]
        ask = r["imbalance_gate"]["ask_heavy"]
        lines.append(
            f"- {off}s bid_support vs ask_heavy: delta T3R `{r1(float(bid['t3r_bps'] or 0.0) - float(ask['t3r_bps'] or 0.0))}`, "
            f"delta max_loss `{r1(float(bid['max_loss_bps'] or 0.0) - float(ask['max_loss_bps'] or 0.0))}`."
        )
    lines.append("- If T-30/T-10/T already separates tails, absorption is a legal permission feature. If only T+5/T+30 separates, it is management/confirmation.")
    lines.append("")
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run S34 absorption timing test.")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--source-json", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--max-book-staleness-sec", type=int, default=10)
    parser.add_argument("--json-out", type=Path, default=OUT_JSON)
    parser.add_argument("--md-out", type=Path, default=OUT_MD)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    with sqlite3.connect(f"file:{args.db}?mode=ro", uri=True) as conn:
        report = build_report(
            conn,
            source_path=args.source_json,
            db_path=args.db,
            max_book_staleness_sec=int(args.max_book_staleness_sec),
        )
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    args.md_out.write_text(render_md(report), encoding="utf-8")
    print(render_md(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
