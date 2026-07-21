"""S34 V Engine multi-offset shadow ledger.

Research-only observer for the frozen V Engine v0.1 state. It records the same
eligible liquidation events across multiple maker offsets/cross margins so
execution can be compared without changing paper or live trading state.
"""

from __future__ import annotations

import argparse
import csv
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

from tools.research_s34_knowable_anchor_continuation import file_fingerprint, iso_ms, load_mark_index, r1, r3, sha256_text
from tools.research_s34_maker_fade import NO_TP_OR_SL, simulate_event, summarize
from tools.s34_v_engine_execution_frontier import anchor_mark_counterfactual, collect_v01_events, parse_float_tuple, prior_return_bps
from tools.s34_v_engine_shadow_observer import (
    FADE_DIRECTION,
    HORIZON_SEC,
    LIQ_SIDE,
    PERMISSION,
    PROTOCOL_ID,
    PROTOCOL_STATUS,
    SYMBOL,
    THRESHOLD_USD,
    VDEPTH_MAX_BPS,
    VDEPTH_MIN_BPS,
    merge_rows,
    observation_status,
    status_counts,
    utc_now,
)


DEFAULT_DB = ROOT / "data" / "microstructure.db"
OUT_DIR = ROOT / "reports" / "research" / "s34"
DEFAULT_LEDGER_JSONL = OUT_DIR / "S34_V_ENGINE_MULTI_OFFSET_SHADOW.jsonl"
DEFAULT_LEDGER_CSV = OUT_DIR / "S34_V_ENGINE_MULTI_OFFSET_SHADOW.csv"
DEFAULT_BRIEF_JSON = OUT_DIR / "S34_V_ENGINE_MULTI_OFFSET_SHADOW_BRIEF.json"
DEFAULT_BRIEF_MD = OUT_DIR / "S34_V_ENGINE_MULTI_OFFSET_SHADOW_BRIEF.md"

SHADOW_ID = "S34_V_ENGINE_MULTI_OFFSET_SHADOW"
SHADOW_PERMISSION = "EXPLORATORY_V_FADE_MULTI_OFFSET"

LEDGER_FIELDS = (
    "observation_id",
    "shadow_id",
    "protocol_id",
    "protocol_status",
    "permission",
    "config_id",
    "symbol",
    "liq_side",
    "fade_direction",
    "signal_ts_ms",
    "signal_utc",
    "bucket",
    "threshold_usd",
    "vdepth_bps",
    "prior_4h_bps",
    "running_notional",
    "running_liq_count",
    "running_rate_usd_per_sec",
    "running_accel_usd_per_sec",
    "elapsed_since_first_sec",
    "single_liq_dominance_pct",
    "anchor_mark_price",
    "offset_bps",
    "cross_margin_bps",
    "limit_price",
    "maker_fill_ts_ms",
    "maker_fill_utc",
    "fill_delay_sec",
    "entry_price",
    "exit_ts_ms",
    "exit_utc",
    "exit_reason",
    "exit_price",
    "gross_bps",
    "fee_bps",
    "net_bps",
    "sim_status",
    "observation_status",
    "counterfactual_anchor_mark_net_bps",
    "notes",
)


def max_mark_ts(conn: sqlite3.Connection, symbol: str) -> int | None:
    row = conn.execute("SELECT MAX(ts_ms) FROM mark_prices WHERE symbol=?", (symbol,)).fetchone()
    return int(row[0]) if row and row[0] is not None else None


def config_id(offset_bps: float, cross_margin_bps: float) -> str:
    return f"O{float(offset_bps):g}_C{float(cross_margin_bps):g}"


def observation_id(*, signal_ts_ms: int, bucket: int, offset_bps: float, cross_margin_bps: float) -> str:
    raw = f"{SHADOW_ID}|{PROTOCOL_ID}|{SYMBOL}|{LIQ_SIDE}|{bucket}|{signal_ts_ms}|{offset_bps:.6f}|{cross_margin_bps:.6f}"
    return sha256_text(raw)[:24]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(LEDGER_FIELDS), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_rows(
    conn: sqlite3.Connection,
    *,
    offsets: tuple[float, ...],
    cross_margins: tuple[float, ...],
    maker_fee_bps: float,
    taker_fee_bps: float,
    max_book_staleness_sec: int,
) -> list[dict[str, Any]]:
    events = collect_v01_events(conn)
    marks = load_mark_index(conn, SYMBOL)
    data_end = max_mark_ts(conn, SYMBOL)
    fee_bps = float(maker_fee_bps) + float(taker_fee_bps)
    rows: list[dict[str, Any]] = []
    for event in events:
        ts = int(event.anchor.anchor_ts_ms)
        prior4h = prior_return_bps(marks, ts, 4 * 3600)
        anchor_cf = r1(anchor_mark_counterfactual(marks, ts, fee_bps=fee_bps))
        for cross in cross_margins:
            for offset in offsets:
                sim = simulate_event(
                    conn,
                    event,
                    offset_bps=float(offset),
                    cross_margin_bps=float(cross),
                    horizon_sec=HORIZON_SEC,
                    maker_fee_bps=float(maker_fee_bps),
                    taker_fee_bps=float(taker_fee_bps),
                    max_book_staleness_sec=int(max_book_staleness_sec),
                    horizon_from="fill",
                    tp_bps=NO_TP_OR_SL,
                    sl_bps=NO_TP_OR_SL,
                )
                cid = config_id(offset, cross)
                rows.append(
                    {
                        "observation_id": observation_id(
                            signal_ts_ms=ts,
                            bucket=int(event.anchor.bucket),
                            offset_bps=float(offset),
                            cross_margin_bps=float(cross),
                        ),
                        "shadow_id": SHADOW_ID,
                        "protocol_id": PROTOCOL_ID,
                        "protocol_status": PROTOCOL_STATUS,
                        "permission": SHADOW_PERMISSION,
                        "config_id": cid,
                        "symbol": SYMBOL,
                        "liq_side": LIQ_SIDE,
                        "fade_direction": FADE_DIRECTION,
                        "signal_ts_ms": ts,
                        "signal_utc": iso_ms(ts),
                        "bucket": int(event.anchor.bucket),
                        "threshold_usd": THRESHOLD_USD,
                        "vdepth_bps": r1(event.vdepth_bps),
                        "prior_4h_bps": r1(prior4h),
                        "running_notional": r1(event.anchor.running_notional),
                        "running_liq_count": int(event.anchor.running_liq_count),
                        "running_rate_usd_per_sec": r1(event.anchor.running_rate),
                        "running_accel_usd_per_sec": r1(event.anchor.running_accel),
                        "elapsed_since_first_sec": r1(event.anchor.elapsed_since_first_sec),
                        "single_liq_dominance_pct": r1(event.anchor.running_single_liq_dominance),
                        "anchor_mark_price": sim.get("anchor_mark_price"),
                        "offset_bps": float(offset),
                        "cross_margin_bps": float(cross),
                        "limit_price": sim.get("limit_price"),
                        "maker_fill_ts_ms": sim.get("maker_fill_ts_ms"),
                        "maker_fill_utc": sim.get("maker_fill_utc"),
                        "fill_delay_sec": r1(sim.get("fill_delay_sec")),
                        "entry_price": sim.get("entry_price"),
                        "exit_ts_ms": sim.get("exit_ts_ms"),
                        "exit_utc": sim.get("exit_utc"),
                        "exit_reason": sim.get("exit_reason"),
                        "exit_price": sim.get("exit_price"),
                        "gross_bps": r1(sim.get("gross_bps")),
                        "fee_bps": r1(sim.get("fee_bps")),
                        "net_bps": r1(sim.get("net_bps")),
                        "sim_status": sim.get("status"),
                        "observation_status": observation_status(sim, data_end),
                        "counterfactual_anchor_mark_net_bps": anchor_cf,
                        "notes": "research_only_parallel_offset_shadow_no_order",
                    }
                )
    rows.sort(key=lambda r: (int(r["signal_ts_ms"]), str(r["config_id"])))
    return rows


def closed_filled(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        r
        for r in rows
        if r.get("observation_status") == "CLOSED"
        and r.get("sim_status") == "FILLED"
        and r.get("net_bps") is not None
    ]


def by_config(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        out.setdefault(str(row.get("config_id")), []).append(row)
    return dict(sorted(out.items()))


def config_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    fills = closed_filled(rows)
    no_fill_closed = [
        r for r in rows if r.get("observation_status") == "CLOSED" and r.get("sim_status") == "NO_MAKER_FILL"
    ]
    no_fill_cf = [
        float(r["counterfactual_anchor_mark_net_bps"])
        for r in no_fill_closed
        if r.get("counterfactual_anchor_mark_net_bps") is not None
        and math.isfinite(float(r["counterfactual_anchor_mark_net_bps"]))
    ]
    nets = [float(r["net_bps"]) for r in fills]
    delays = [float(r["fill_delay_sec"]) for r in fills if r.get("fill_delay_sec") is not None]
    return {
        "signals": len(rows),
        "closed_filled": len(fills),
        "closed_no_fill": len(no_fill_closed),
        "fill_rate": r3(len(fills) / len(rows)) if rows else None,
        "summary": summarize(nets),
        "no_fill_counterfactual_summary": summarize(no_fill_cf),
        "missed_cf_sum_bps": r1(sum(no_fill_cf)),
        "median_fill_delay_sec": r1(sorted(delays)[len(delays) // 2]) if delays else None,
        "status_counts": status_counts(rows),
    }


def build_brief(rows: list[dict[str, Any]], *, source_db: dict[str, Any], added_n: int) -> dict[str, Any]:
    configs = []
    for cid, items in by_config(rows).items():
        parts = cid.replace("O", "").split("_C", 1)
        offset = float(parts[0])
        cross = float(parts[1]) if len(parts) > 1 else None
        configs.append(
            {
                "config_id": cid,
                "offset_bps": offset,
                "cross_margin_bps": cross,
                **config_summary(items),
            }
        )
    configs.sort(
        key=lambda r: (
            float(r["summary"].get("top3_winner_removed_sum_bps") or -1e18),
            float(r["summary"].get("sum_bps") or -1e18),
            float(r.get("fill_rate") or 0.0),
        ),
        reverse=True,
    )
    return {
        "generated_at_utc": utc_now(),
        "source_db": source_db,
        "shadow_id": SHADOW_ID,
        "protocol_id": PROTOCOL_ID,
        "decision": "OBSERVE_ONLY",
        "ledger": {
            "rows_total": len(rows),
            "rows_added_this_run": int(added_n),
            "status_counts": status_counts(rows),
        },
        "configs": configs,
        "latest_observations": rows[-18:],
    }


def summary_cell(summary: dict[str, Any]) -> str:
    return f"N={summary['n']} sum={summary['sum_bps']} med={summary['median_bps']} T3R={summary['top3_winner_removed_sum_bps']}"


def render_md(brief: dict[str, Any]) -> str:
    lines = [
        "# S34 V Engine Multi-Offset Shadow Brief",
        "",
        f"Generated: `{brief['generated_at_utc']}`",
        "",
        f"Protocol: `{brief['protocol_id']}`",
        "",
        "Research-only parallel shadow ledger. It compares maker offsets on the same eligible V Engine events; no order is authorized.",
        "",
        "## Ledger",
        "",
        f"- rows total: `{brief['ledger']['rows_total']}`",
        f"- rows added this run: `{brief['ledger']['rows_added_this_run']}`",
        f"- status counts: `{brief['ledger']['status_counts']}`",
        "",
        "## Offset Configs",
        "",
        "| Rank | Config | Fill% | Filled | No-fill CF | Missed CF sum | Median fill delay |",
        "| ---: | --- | ---: | --- | --- | ---: | ---: |",
    ]
    for idx, row in enumerate(brief["configs"], start=1):
        fill_pct = None if row["fill_rate"] is None else r1(row["fill_rate"] * 100.0)
        lines.append(
            f"| {idx} | `{row['config_id']}` | {fill_pct} | {summary_cell(row['summary'])} | "
            f"{summary_cell(row['no_fill_counterfactual_summary'])} | {row['missed_cf_sum_bps']} | {row['median_fill_delay_sec']} |"
        )
    lines.extend(["", "## Read", ""])
    if brief["configs"]:
        best = brief["configs"][0]
        lines.append(
            f"- Best current T3R-ranked config: `{best['config_id']}` with {summary_cell(best['summary'])}."
        )
        lines.append(
            "- Treat this as execution observation, not a new frozen rule. The decision remains observe-only until new forward rows accumulate."
        )
    lines.extend(["", "## Latest Rows", ""])
    lines.append("| UTC | Config | Status | Sim | Fill delay | Net | CF mark net |")
    lines.append("| --- | --- | --- | --- | ---: | ---: | ---: |")
    for row in brief["latest_observations"]:
        lines.append(
            f"| {row.get('signal_utc')} | `{row.get('config_id')}` | {row.get('observation_status')} | {row.get('sim_status')} | "
            f"{row.get('fill_delay_sec')} | {row.get('net_bps')} | {row.get('counterfactual_anchor_mark_net_bps')} |"
        )
    lines.append("")
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Update S34 V Engine multi-offset shadow ledger.")
    p.add_argument("--db", type=Path, default=DEFAULT_DB)
    p.add_argument("--ledger-jsonl", type=Path, default=DEFAULT_LEDGER_JSONL)
    p.add_argument("--ledger-csv", type=Path, default=DEFAULT_LEDGER_CSV)
    p.add_argument("--brief-json", type=Path, default=DEFAULT_BRIEF_JSON)
    p.add_argument("--brief-md", type=Path, default=DEFAULT_BRIEF_MD)
    p.add_argument("--offsets-bps", default="15,20,25")
    p.add_argument("--cross-margins-bps", default="1,2")
    p.add_argument("--maker-fee-bps", type=float, default=2.0)
    p.add_argument("--taker-fee-bps", type=float, default=3.05)
    p.add_argument("--max-book-staleness-sec", type=int, default=10)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    offsets = parse_float_tuple(args.offsets_bps)
    crosses = parse_float_tuple(args.cross_margins_bps)
    with sqlite3.connect(f"file:{args.db}?mode=ro", uri=True) as conn:
        observed = build_rows(
            conn,
            offsets=offsets,
            cross_margins=crosses,
            maker_fee_bps=float(args.maker_fee_bps),
            taker_fee_bps=float(args.taker_fee_bps),
            max_book_staleness_sec=int(args.max_book_staleness_sec),
        )
    existing = load_jsonl(args.ledger_jsonl)
    merged, added = merge_rows(existing, observed)
    brief = build_brief(merged, source_db=file_fingerprint(args.db), added_n=added)
    write_jsonl(args.ledger_jsonl, merged)
    write_csv(args.ledger_csv, merged)
    args.brief_json.parent.mkdir(parents=True, exist_ok=True)
    args.brief_json.write_text(json.dumps(brief, indent=2, ensure_ascii=True), encoding="utf-8")
    args.brief_md.write_text(render_md(brief), encoding="utf-8")
    print(render_md(brief))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
