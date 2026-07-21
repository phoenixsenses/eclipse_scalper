"""S34 V Engine v0.1 shadow observation ledger.

Logs the frozen exploratory V-engine candidate without creating paper/live
orders. Signal fields are knowable at the threshold-cross anchor. Outcome fields
are labels computed after enough market data exists.
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

from tools.research_s34_knowable_anchor_continuation import (
    file_fingerprint,
    iso_ms,
    load_mark_index,
    r1,
    r3,
    sha256_text,
    signed_return_bps,
)
from tools.research_s34_maker_fade import (
    NO_TP_OR_SL,
    collect_events,
    simulate_event,
    summarize,
)


DEFAULT_DB = ROOT / "data" / "microstructure.db"
OUT_DIR = ROOT / "reports" / "research" / "s34"
DEFAULT_LEDGER_JSONL = OUT_DIR / "S34_V_ENGINE_V0_1_OBSERVATION_LEDGER.jsonl"
DEFAULT_LEDGER_CSV = OUT_DIR / "S34_V_ENGINE_V0_1_OBSERVATION_LEDGER.csv"
DEFAULT_BRIEF_JSON = OUT_DIR / "S34_V_ENGINE_V0_1_WEEKLY_BRIEF.json"
DEFAULT_BRIEF_MD = OUT_DIR / "S34_V_ENGINE_V0_1_WEEKLY_BRIEF.md"

PROTOCOL_ID = "S34_V_ENGINE_V0_1_ETH_SELL_MAKER_LONG_H2_O20_V28_40_P4D"
PROTOCOL_STATUS = "EXPLORATORY_FROZEN"
PERMISSION = "EXPLORATORY_V_FADE_V0_1"

SYMBOL = "ETHUSDT"
LIQ_SIDE = "SELL"
FADE_DIRECTION = "LONG"
THRESHOLD_USD = 200_000.0
VDEPTH_MIN_BPS = 28.0
VDEPTH_MAX_BPS = 40.0
PRIOR4H_LT_BPS = -50.0
OFFSET_BPS = 20.0
CROSS_MARGIN_BPS = 2.0
HORIZON_SEC = 2 * 3600
BUCKET_SEC = 300
MIN_GAP_SEC = 900
ACCEL_WINDOW_SEC = 30

LEDGER_FIELDS = (
    "observation_id",
    "protocol_id",
    "protocol_status",
    "permission",
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


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def max_mark_ts(conn: sqlite3.Connection, symbol: str) -> int | None:
    row = conn.execute("SELECT MAX(ts_ms) FROM mark_prices WHERE symbol=?", (symbol,)).fetchone()
    return int(row[0]) if row and row[0] is not None else None


def prior_return_bps(marks: Any, ts_ms: int, window_sec: int) -> float | None:
    return marks.ret_bps(int(ts_ms) - int(window_sec) * 1000, int(ts_ms))


def counterfactual_anchor_mark_net_bps(marks: Any, anchor_ts_ms: int, fee_bps: float) -> float | None:
    entry = marks.at_or_after(int(anchor_ts_ms))
    exit_ = marks.at_or_after(int(anchor_ts_ms) + HORIZON_SEC * 1000)
    if not entry or not exit_:
        return None
    return signed_return_bps(FADE_DIRECTION, float(entry[1]), float(exit_[1])) - float(fee_bps)


def observation_id(*, signal_ts_ms: int, bucket: int, vdepth_bps: float, prior_4h_bps: float) -> str:
    raw = f"{PROTOCOL_ID}|{SYMBOL}|{LIQ_SIDE}|{bucket}|{signal_ts_ms}|{vdepth_bps:.6f}|{prior_4h_bps:.6f}"
    return sha256_text(raw)[:24]


def observation_status(sim: dict[str, Any], data_end_ms: int | None) -> str:
    if data_end_ms is None:
        return "PENDING"
    if sim.get("status") in {"NO_EXIT_BOOK", "NO_EXIT_FILL"}:
        return "DATA_INCOMPLETE"
    if sim.get("status") == "FILLED" and sim.get("exit_ts_ms") is not None:
        return "CLOSED" if int(sim["exit_ts_ms"]) <= int(data_end_ms) else "PENDING"
    if sim.get("status") == "NO_MAKER_FILL":
        close_ms = int(sim["anchor_ts_ms"]) + HORIZON_SEC * 1000
        return "CLOSED" if close_ms <= int(data_end_ms) else "PENDING"
    return "PENDING"


def build_rows(
    conn: sqlite3.Connection,
    *,
    maker_fee_bps: float,
    taker_fee_bps: float,
    max_book_staleness_sec: int,
) -> list[dict[str, Any]]:
    marks = load_mark_index(conn, SYMBOL)
    data_end = max_mark_ts(conn, SYMBOL)
    fee_bps = float(maker_fee_bps) + float(taker_fee_bps)
    events = collect_events(
        conn,
        symbol=SYMBOL,
        threshold=THRESHOLD_USD,
        sides=(LIQ_SIDE,),
        min_vdepth_bps=VDEPTH_MIN_BPS,
        bucket_sec=BUCKET_SEC,
        min_gap_sec=MIN_GAP_SEC,
        accel_window_sec=ACCEL_WINDOW_SEC,
        max_horizon_sec=HORIZON_SEC,
    )
    rows: list[dict[str, Any]] = []
    for event in events:
        if not (VDEPTH_MIN_BPS <= float(event.vdepth_bps) < VDEPTH_MAX_BPS):
            continue
        ts = int(event.anchor.anchor_ts_ms)
        prior4h = prior_return_bps(marks, ts, 4 * 3600)
        if prior4h is None or not math.isfinite(float(prior4h)) or not (float(prior4h) < PRIOR4H_LT_BPS):
            continue
        sim = simulate_event(
            conn,
            event,
            offset_bps=OFFSET_BPS,
            cross_margin_bps=CROSS_MARGIN_BPS,
            horizon_sec=HORIZON_SEC,
            maker_fee_bps=float(maker_fee_bps),
            taker_fee_bps=float(taker_fee_bps),
            max_book_staleness_sec=int(max_book_staleness_sec),
            horizon_from="fill",
            tp_bps=NO_TP_OR_SL,
            sl_bps=NO_TP_OR_SL,
        )
        status = observation_status(sim, data_end)
        oid = observation_id(
            signal_ts_ms=ts,
            bucket=int(event.anchor.bucket),
            vdepth_bps=float(event.vdepth_bps),
            prior_4h_bps=float(prior4h),
        )
        rows.append(
            {
                "observation_id": oid,
                "protocol_id": PROTOCOL_ID,
                "protocol_status": PROTOCOL_STATUS,
                "permission": PERMISSION,
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
                "offset_bps": OFFSET_BPS,
                "cross_margin_bps": CROSS_MARGIN_BPS,
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
                "observation_status": status,
                "counterfactual_anchor_mark_net_bps": r1(counterfactual_anchor_mark_net_bps(marks, ts, fee_bps)),
                "notes": "research_only_observation_no_order",
            }
        )
    rows.sort(key=lambda r: int(r["signal_ts_ms"]))
    return rows


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


def merge_rows(existing: list[dict[str, Any]], observed: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    by_id = {str(row["observation_id"]): row for row in existing if row.get("observation_id")}
    added = 0
    for row in observed:
        oid = str(row["observation_id"])
        if oid not in by_id:
            added += 1
        by_id[oid] = row
    merged = list(by_id.values())
    merged.sort(key=lambda r: (int(r.get("signal_ts_ms") or 0), str(r.get("observation_id") or "")))
    return merged, added


def closed_filled(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        r
        for r in rows
        if r.get("observation_status") == "CLOSED"
        and r.get("sim_status") == "FILLED"
        and r.get("net_bps") is not None
    ]


def status_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for row in rows:
        key = str(row.get("observation_status") or "UNKNOWN")
        out[key] = out.get(key, 0) + 1
    return dict(sorted(out.items()))


def weekly_groups(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        ts = row.get("signal_ts_ms")
        if ts is None:
            continue
        dt = datetime.fromtimestamp(int(ts) / 1000.0, tz=timezone.utc)
        iso = dt.isocalendar()
        key = f"{iso.year}-W{iso.week:02d}"
        groups.setdefault(key, []).append(row)
    out = []
    for key, items in sorted(groups.items()):
        fills = closed_filled(items)
        vals = [float(r["net_bps"]) for r in fills]
        out.append(
            {
                "week": key,
                "signals": len(items),
                "closed": sum(1 for r in items if r.get("observation_status") == "CLOSED"),
                "pending": sum(1 for r in items if r.get("observation_status") == "PENDING"),
                "data_incomplete": sum(1 for r in items if r.get("observation_status") == "DATA_INCOMPLETE"),
                "filled": len(fills),
                "fill_rate": r3(len(fills) / len(items)) if items else None,
                "summary": summarize(vals),
            }
        )
    return out


def recent_rows(rows: list[dict[str, Any]], days: int) -> list[dict[str, Any]]:
    if not rows:
        return []
    end_ms = max(int(r["signal_ts_ms"]) for r in rows if r.get("signal_ts_ms") is not None)
    start_ms = end_ms - int(days) * 24 * 3600 * 1000
    return [r for r in rows if int(r.get("signal_ts_ms") or 0) >= start_ms]


def build_brief(rows: list[dict[str, Any]], *, brief_days: int, source_db: dict[str, Any], added_n: int) -> dict[str, Any]:
    recent = recent_rows(rows, brief_days)
    all_fills = closed_filled(rows)
    recent_fills = closed_filled(recent)
    no_fill_closed = [
        r for r in rows if r.get("observation_status") == "CLOSED" and r.get("sim_status") == "NO_MAKER_FILL"
    ]
    no_fill_cf = [
        float(r["counterfactual_anchor_mark_net_bps"])
        for r in no_fill_closed
        if r.get("counterfactual_anchor_mark_net_bps") is not None
    ]
    recent_vals = [float(r["net_bps"]) for r in recent_fills]
    all_vals = [float(r["net_bps"]) for r in all_fills]
    recent_summary = summarize(recent_vals)
    kill_triggered = (
        int(recent_summary["n"] or 0) >= 3
        and float(recent_summary["top3_winner_removed_sum_bps"] or 0.0) < 0.0
    )
    return {
        "generated_at_utc": utc_now(),
        "source_db": source_db,
        "protocol": {
            "id": PROTOCOL_ID,
            "status": PROTOCOL_STATUS,
            "permission": PERMISSION,
            "decision": "OBSERVE_ONLY",
        },
        "ledger": {
            "rows_total": len(rows),
            "rows_added_this_run": int(added_n),
            "status_counts": status_counts(rows),
        },
        "overall": {
            "signals": len(rows),
            "closed_filled": len(all_fills),
            "fill_rate": r3(len(all_fills) / len(rows)) if rows else None,
            "summary": summarize(all_vals),
            "closed_no_fill_n": len(no_fill_closed),
            "closed_no_fill_counterfactual_summary": summarize(no_fill_cf),
        },
        "recent": {
            "days": int(brief_days),
            "signals": len(recent),
            "closed_filled": len(recent_fills),
            "fill_rate": r3(len(recent_fills) / len(recent)) if recent else None,
            "summary": recent_summary,
            "kill_check": {
                "rule": "60-day forward T3R < 0 after at least 3 closed fills",
                "triggered": bool(kill_triggered),
            },
        },
        "weekly": weekly_groups(rows),
        "latest_observations": rows[-10:],
    }


def summary_cell(summary: dict[str, Any]) -> str:
    return (
        f"N={summary['n']} sum={summary['sum_bps']} med={summary['median_bps']} "
        f"T3R={summary['top3_winner_removed_sum_bps']}"
    )


def render_md(brief: dict[str, Any]) -> str:
    p = brief["protocol"]
    overall = brief["overall"]
    recent = brief["recent"]
    lines = [
        "# S34 V Engine v0.1 Shadow Observation Brief",
        "",
        f"Generated: `{brief['generated_at_utc']}`",
        "",
        f"Protocol: `{p['id']}`",
        "",
        f"Status: `{p['status']}` / `{p['decision']}`. This is observation only; no live or paper order is authorized.",
        "",
        "## Ledger",
        "",
        f"- rows total: `{brief['ledger']['rows_total']}`",
        f"- rows added this run: `{brief['ledger']['rows_added_this_run']}`",
        f"- status counts: `{brief['ledger']['status_counts']}`",
        "",
        "## Performance Labels",
        "",
        f"- overall: signals `{overall['signals']}`, closed fills `{overall['closed_filled']}`, fill rate `{overall['fill_rate']}`, {summary_cell(overall['summary'])}",
        f"- recent {recent['days']}d: signals `{recent['signals']}`, closed fills `{recent['closed_filled']}`, fill rate `{recent['fill_rate']}`, {summary_cell(recent['summary'])}",
        f"- no-fill counterfactual: closed no-fill `{overall['closed_no_fill_n']}`, {summary_cell(overall['closed_no_fill_counterfactual_summary'])}",
        f"- kill check: `{'TRIGGERED' if recent['kill_check']['triggered'] else 'not triggered'}` ({recent['kill_check']['rule']})",
        "",
        "## Weekly",
        "",
        "| Week | Signals | Closed | Pending | Data incomplete | Filled | Fill% | Summary |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in brief["weekly"]:
        fill_pct = None if row["fill_rate"] is None else r1(row["fill_rate"] * 100.0)
        lines.append(
            f"| `{row['week']}` | {row['signals']} | {row['closed']} | {row['pending']} | {row['data_incomplete']} | "
            f"{row['filled']} | {fill_pct} | {summary_cell(row['summary'])} |"
        )
    lines.extend(["", "## Latest Observations", ""])
    lines.append("| UTC | Status | Sim | V-depth | Prior4h | Fill delay | Net | CF mark net |")
    lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for row in brief["latest_observations"]:
        lines.append(
            f"| {row.get('signal_utc')} | {row.get('observation_status')} | {row.get('sim_status')} | "
            f"{row.get('vdepth_bps')} | {row.get('prior_4h_bps')} | {row.get('fill_delay_sec')} | "
            f"{row.get('net_bps')} | {row.get('counterfactual_anchor_mark_net_bps')} |"
        )
    lines.append("")
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Update the S34 V Engine v0.1 observation ledger and brief.")
    p.add_argument("--db", type=Path, default=DEFAULT_DB)
    p.add_argument("--ledger-jsonl", type=Path, default=DEFAULT_LEDGER_JSONL)
    p.add_argument("--ledger-csv", type=Path, default=DEFAULT_LEDGER_CSV)
    p.add_argument("--brief-json", type=Path, default=DEFAULT_BRIEF_JSON)
    p.add_argument("--brief-md", type=Path, default=DEFAULT_BRIEF_MD)
    p.add_argument("--maker-fee-bps", type=float, default=2.0)
    p.add_argument("--taker-fee-bps", type=float, default=3.05)
    p.add_argument("--max-book-staleness-sec", type=int, default=10)
    p.add_argument("--brief-days", type=int, default=60)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    with sqlite3.connect(f"file:{args.db}?mode=ro", uri=True) as conn:
        observed = build_rows(
            conn,
            maker_fee_bps=float(args.maker_fee_bps),
            taker_fee_bps=float(args.taker_fee_bps),
            max_book_staleness_sec=int(args.max_book_staleness_sec),
        )
    existing = load_jsonl(args.ledger_jsonl)
    merged, added = merge_rows(existing, observed)
    source_db = file_fingerprint(args.db)
    brief = build_brief(merged, brief_days=int(args.brief_days), source_db=source_db, added_n=added)
    write_jsonl(args.ledger_jsonl, merged)
    write_csv(args.ledger_csv, merged)
    args.brief_json.parent.mkdir(parents=True, exist_ok=True)
    args.brief_json.write_text(json.dumps(brief, indent=2, ensure_ascii=True), encoding="utf-8")
    args.brief_md.write_text(render_md(brief), encoding="utf-8")
    print(render_md(brief))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
