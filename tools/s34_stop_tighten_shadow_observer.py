"""Shadow observer for S34 Stop-Tighten v0.1.

Logs what the frozen stop-tighten overlay would have done on the current V
Engine route. Research-only: no exchange, no live/paper state writes.
"""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.research_s34_knowable_anchor_continuation import file_fingerprint, iso_ms, load_mark_index, r1, sha256_text
from tools.research_s34_maker_fade import summarize
from tools.s34_v_engine_confirmation_cost_current import CONFIG_ID, load_json
from tools.s34_v_engine_failure_anatomy import finite_float
from tools.s34_v_engine_position_management import condition_map, tighten_stop
from tools.s34_v_engine_shadow_observer import SYMBOL, utc_now


DEFAULT_DB = ROOT / "data" / "microstructure.db"
DEFAULT_CANCEL_REPLACE_JSON = ROOT / "reports" / "research" / "s34" / "S34_V_ENGINE_CANCEL_REPLACE.json"
OUT_DIR = ROOT / "reports" / "research" / "s34"
DEFAULT_LEDGER_JSONL = OUT_DIR / "S34_STOP_TIGHTEN_V0_1_LEDGER.jsonl"
DEFAULT_LEDGER_CSV = OUT_DIR / "S34_STOP_TIGHTEN_V0_1_LEDGER.csv"
DEFAULT_BRIEF_JSON = OUT_DIR / "S34_STOP_TIGHTEN_V0_1_BRIEF.json"
DEFAULT_BRIEF_MD = OUT_DIR / "S34_STOP_TIGHTEN_V0_1_BRIEF.md"

PROTOCOL_ID = "S34_STOP_TIGHTEN_V0_1_ETH_SELL_MAKER_LONG_5M_BTC_DOWN_TRIGSL80"
PERMISSION = "SHADOW_STOP_TIGHTEN_V0_1"
PARENT_RULE = "S34_V_ENGINE_V0_1_ETH_SELL_MAKER_LONG_H2_O20_W300_O5"
DELAY_MIN = 5
STOP_BPS = 80.0
CONDITION = "no_reclaim_btc_down"
STOP_REFERENCE = "trigger"

FIELDS = (
    "observation_id",
    "protocol_id",
    "permission",
    "parent_rule",
    "symbol",
    "signal_ts_ms",
    "signal_utc",
    "maker_fill_ts_ms",
    "maker_fill_utc",
    "entry_price",
    "baseline_exit_ts_ms",
    "baseline_net_bps",
    "trigger_check_ts_ms",
    "trigger_check_utc",
    "triggered",
    "btc_context_bucket",
    "anchor_reclaimed",
    "ret_delay_bps",
    "managed_net_bps",
    "delta_bps",
    "action_net_bps",
    "action_source",
    "observation_status",
)


def observation_id(row: dict[str, Any]) -> str:
    raw = f"{PROTOCOL_ID}|{row.get('anchor_ts_ms')}|{row.get('maker_fill_ts_ms')}|{row.get('entry_price')}"
    return sha256_text(raw)[:24]


def status_for(row: dict[str, Any], data_end_ms: int | None) -> str:
    if data_end_ms is None:
        return "PENDING"
    exit_ts = row.get("exit_ts_ms")
    if exit_ts is None:
        return "DATA_INCOMPLETE"
    return "CLOSED" if int(exit_ts) <= int(data_end_ms) else "PENDING"


def build_rows(conn: sqlite3.Connection, *, cancel_replace_path: Path, max_book_staleness_sec: int) -> list[dict[str, Any]]:
    payload = load_json(cancel_replace_path)
    base_rows = [
        r
        for r in payload.get("rows", [])
        if r.get("config_id") == CONFIG_ID and r.get("status") == "FILLED" and finite_float(r.get("net_bps")) is not None
    ]
    eth_marks = load_mark_index(conn, SYMBOL)
    btc_marks = load_mark_index(conn, "BTCUSDT")
    data_end = eth_marks.ts[-1] if eth_marks.ts else None
    fn = condition_map()[CONDITION]
    rows = []
    for row in base_rows:
        managed = tighten_stop(
            conn,
            row,
            delay_min=DELAY_MIN,
            stop_bps=STOP_BPS,
            condition=CONDITION,
            fn=fn,
            stop_reference=STOP_REFERENCE,
            eth_marks=eth_marks,
            btc_marks=btc_marks,
            max_book_staleness_sec=max_book_staleness_sec,
        )
        base = finite_float(row.get("net_bps"))
        managed_net = finite_float(managed.get("managed_net_bps"))
        fill_ts = int(row["maker_fill_ts_ms"])
        trigger_ts = fill_ts + DELAY_MIN * 60_000
        rows.append(
            {
                "observation_id": observation_id(row),
                "protocol_id": PROTOCOL_ID,
                "permission": PERMISSION,
                "parent_rule": PARENT_RULE,
                "symbol": SYMBOL,
                "signal_ts_ms": int(row["anchor_ts_ms"]),
                "signal_utc": row.get("anchor_utc") or iso_ms(int(row["anchor_ts_ms"])),
                "maker_fill_ts_ms": fill_ts,
                "maker_fill_utc": row.get("maker_fill_utc") or iso_ms(fill_ts),
                "entry_price": row.get("entry_price"),
                "baseline_exit_ts_ms": row.get("exit_ts_ms"),
                "baseline_net_bps": r1(base),
                "trigger_check_ts_ms": trigger_ts,
                "trigger_check_utc": iso_ms(trigger_ts),
                "triggered": bool(managed.get("triggered")),
                "btc_context_bucket": managed.get("btc_context_bucket"),
                "anchor_reclaimed": bool(managed.get("anchor_reclaimed")),
                "ret_delay_bps": managed.get("ret_delay_bps"),
                "managed_net_bps": r1(managed_net),
                "delta_bps": r1(None if base is None or managed_net is None else managed_net - base),
                "action_net_bps": managed.get("action_net_bps"),
                "action_source": managed.get("source"),
                "observation_status": status_for(row, data_end),
            }
        )
    rows.sort(key=lambda r: int(r["signal_ts_ms"]))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(FIELDS), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_brief(rows: list[dict[str, Any]], *, db_path: Path) -> dict[str, Any]:
    closed = [r for r in rows if r.get("observation_status") == "CLOSED"]
    baseline = [float(r["baseline_net_bps"]) for r in closed if finite_float(r.get("baseline_net_bps")) is not None]
    managed = [float(r["managed_net_bps"]) for r in closed if finite_float(r.get("managed_net_bps")) is not None]
    deltas = [float(r["delta_bps"]) for r in closed if finite_float(r.get("delta_bps")) is not None]
    triggered = [r for r in closed if r.get("triggered")]
    return {
        "generated_at_utc": utc_now(),
        "source_db": file_fingerprint(db_path),
        "protocol_id": PROTOCOL_ID,
        "parent_rule": PARENT_RULE,
        "config": {
            "delay_min": DELAY_MIN,
            "condition": CONDITION,
            "stop_reference": STOP_REFERENCE,
            "stop_bps": STOP_BPS,
        },
        "counts": {
            "rows": len(rows),
            "closed": len(closed),
            "triggered": len(triggered),
        },
        "baseline": summarize(baseline),
        "managed": summarize(managed),
        "delta": summarize(deltas),
    }


def cell(summary: dict[str, Any]) -> str:
    return f"N={summary['n']} sum={summary['sum_bps']} med={summary['median_bps']} T3R={summary['top3_winner_removed_sum_bps']} max_loss={summary['max_loss_bps']}"


def render_md(brief: dict[str, Any]) -> str:
    base_t3r = float(brief["baseline"].get("top3_winner_removed_sum_bps") or 0.0)
    managed_t3r = float(brief["managed"].get("top3_winner_removed_sum_bps") or 0.0)
    return "\n".join(
        [
            "# S34 Stop-Tighten v0.1 Shadow Brief",
            "",
            f"Generated: `{brief['generated_at_utc']}`",
            "",
            f"Protocol: `{brief['protocol_id']}`",
            "",
            f"Parent: `{brief['parent_rule']}`",
            "",
            f"Rows: `{brief['counts']['rows']}` closed `{brief['counts']['closed']}` triggered `{brief['counts']['triggered']}`",
            "",
            f"- Baseline: {cell(brief['baseline'])}",
            f"- Managed: {cell(brief['managed'])}",
            f"- Delta: {cell(brief['delta'])}",
            f"- Delta T3R: `{r1(managed_t3r - base_t3r)}` bps",
            "",
            "Research-only. No live or paper state changed.",
            "",
        ]
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refresh S34 stop-tighten shadow ledger.")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--cancel-replace-json", type=Path, default=DEFAULT_CANCEL_REPLACE_JSON)
    parser.add_argument("--max-book-staleness-sec", type=int, default=10)
    parser.add_argument("--ledger-jsonl", type=Path, default=DEFAULT_LEDGER_JSONL)
    parser.add_argument("--ledger-csv", type=Path, default=DEFAULT_LEDGER_CSV)
    parser.add_argument("--brief-json", type=Path, default=DEFAULT_BRIEF_JSON)
    parser.add_argument("--brief-md", type=Path, default=DEFAULT_BRIEF_MD)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    with sqlite3.connect(f"file:{args.db}?mode=ro", uri=True) as conn:
        rows = build_rows(conn, cancel_replace_path=args.cancel_replace_json, max_book_staleness_sec=int(args.max_book_staleness_sec))
    brief = build_brief(rows, db_path=args.db)
    write_jsonl(args.ledger_jsonl, rows)
    write_csv(args.ledger_csv, rows)
    args.brief_json.write_text(json.dumps(brief, indent=2, ensure_ascii=True), encoding="utf-8")
    args.brief_md.write_text(render_md(brief), encoding="utf-8")
    print(render_md(brief))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
