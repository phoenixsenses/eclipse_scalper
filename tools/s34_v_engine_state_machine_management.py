"""Combined state-machine management test for the current S34 V Engine route.

Research-only. Combines the two strongest management ideas:

- 5m danger -> tighten stop to trigger price -80 bps.
- 30m recovery -> extend fixed exit from 2h to 4h.
- otherwise keep baseline 2h.

No live/paper state changes.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.research_s34_knowable_anchor_continuation import book_at, file_fingerprint, r1, signed_return_bps
from tools.research_s34_maker_fade import summarize
from tools.s34_v_engine_position_management import (
    CONFIG_ID,
    DEFAULT_CANCEL_REPLACE_JSON,
    DEFAULT_DB,
    SYMBOL,
    annotate,
    condition_map,
    finite_float,
    load_json,
    load_mark_index,
    summarize_rows,
    tighten_stop,
    utc_now,
)


OUT_DIR = ROOT / "reports" / "research" / "s34"
OUT_JSON = OUT_DIR / "S34_V_ENGINE_STATE_MACHINE_MANAGEMENT.json"
OUT_MD = OUT_DIR / "S34_V_ENGINE_STATE_MACHINE_MANAGEMENT.md"

FADE_DIRECTION = "LONG"


def h4_exit_net(
    conn: sqlite3.Connection,
    row: dict[str, Any],
    *,
    max_book_staleness_sec: int,
) -> tuple[float | None, str]:
    entry = finite_float(row.get("entry_price"))
    fill_ts = row.get("maker_fill_ts_ms")
    fee = finite_float(row.get("fee_bps")) or 5.05
    if entry is None or fill_ts is None:
        return None, "missing_entry"
    exit_ts = int(fill_ts) + 4 * 3600 * 1000
    book = book_at(conn, SYMBOL, exit_ts, int(max_book_staleness_sec))
    if not book:
        return None, "no_h4_book"
    gross = signed_return_bps(FADE_DIRECTION, float(entry), float(book.bid))
    return gross - float(fee), "h4_book_ticker"


def recovery_pass(row: dict[str, Any], *, eth_marks: Any, btc_marks: Any) -> tuple[bool, dict[str, Any]]:
    ann = annotate(row, delay_min=30, eth_marks=eth_marks, btc_marks=btc_marks)
    passed = bool(ann.get("anchor_reclaimed")) and ann.get("btc_context_bucket") != "btc_down_continues"
    return passed, ann


def build_row(
    conn: sqlite3.Connection,
    row: dict[str, Any],
    *,
    eth_marks: Any,
    btc_marks: Any,
    danger_fn: Any,
    max_book_staleness_sec: int,
) -> dict[str, Any]:
    danger_row = tighten_stop(
        conn,
        row,
        delay_min=5,
        stop_bps=80.0,
        condition="no_reclaim_btc_down",
        fn=danger_fn,
        stop_reference="trigger",
        eth_marks=eth_marks,
        btc_marks=btc_marks,
        max_book_staleness_sec=int(max_book_staleness_sec),
    )
    recovery, recovery_ann = recovery_pass(row, eth_marks=eth_marks, btc_marks=btc_marks)
    h4_net, h4_source = h4_exit_net(conn, row, max_book_staleness_sec=int(max_book_staleness_sec))
    baseline = finite_float(row.get("net_bps"))
    danger_net = finite_float(danger_row.get("managed_net_bps"))
    danger = bool(danger_row.get("triggered"))

    def choose(priority: str) -> tuple[float | None, str]:
        if priority == "danger_priority":
            if danger:
                return danger_net, "danger_stop_tighten"
            if recovery and h4_net is not None:
                return h4_net, "recovery_extend_h4"
            return baseline, "baseline_h2"
        if priority == "recovery_priority":
            if recovery and h4_net is not None:
                return h4_net, "recovery_extend_h4"
            if danger:
                return danger_net, "danger_stop_tighten"
            return baseline, "baseline_h2"
        raise ValueError(priority)

    danger_priority_net, danger_priority_action = choose("danger_priority")
    recovery_priority_net, recovery_priority_action = choose("recovery_priority")
    return {
        **row,
        "danger_triggered": danger,
        "danger_net_bps": r1(danger_net),
        "danger_source": danger_row.get("source"),
        "recovery_triggered": bool(recovery),
        "recovery_h4_net_bps": r1(h4_net),
        "recovery_h4_source": h4_source,
        "recovery_anchor_reclaimed_30m": recovery_ann.get("anchor_reclaimed"),
        "recovery_btc_context_30m": recovery_ann.get("btc_context_bucket"),
        "danger_priority_net_bps": r1(danger_priority_net),
        "danger_priority_action": danger_priority_action,
        "recovery_priority_net_bps": r1(recovery_priority_net),
        "recovery_priority_action": recovery_priority_action,
    }


def count_actions(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for row in rows:
        val = str(row.get(key) or "none")
        out[val] = out.get(val, 0) + 1
    return dict(sorted(out.items()))


def summarize_key(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    return summarize([float(v) for r in rows if (v := finite_float(r.get(key))) is not None])


def build_report(
    conn: sqlite3.Connection,
    *,
    cancel_replace_path: Path,
    db_path: Path,
    max_book_staleness_sec: int,
) -> dict[str, Any]:
    payload = load_json(cancel_replace_path)
    base_rows = [
        r
        for r in payload.get("rows", [])
        if r.get("config_id") == CONFIG_ID and r.get("status") == "FILLED" and finite_float(r.get("net_bps")) is not None
    ]
    eth_marks = load_mark_index(conn, SYMBOL)
    btc_marks = load_mark_index(conn, "BTCUSDT")
    danger_fn = condition_map()["no_reclaim_btc_down"]
    rows = [
        build_row(
            conn,
            row,
            eth_marks=eth_marks,
            btc_marks=btc_marks,
            danger_fn=danger_fn,
            max_book_staleness_sec=int(max_book_staleness_sec),
        )
        for row in base_rows
    ]
    baseline = summarize_key(rows, "net_bps")
    stop_only = summarize_rows(
        [
            {
                **r,
                "triggered": r["danger_triggered"],
                "managed_net_bps": r["danger_net_bps"] if r["danger_triggered"] else r.get("net_bps"),
            }
            for r in rows
        ]
    )["summary"]
    h4_recovery_only = summarize_key(
        [
            {
                **r,
                "recovery_only_net_bps": r["recovery_h4_net_bps"] if r["recovery_triggered"] else r.get("net_bps"),
            }
            for r in rows
        ],
        "recovery_only_net_bps",
    )
    variants = [
        {
            "variant": "stop_only_5m_no_reclaim_btc_down_trigger_sl80",
            "summary": stop_only,
            "actions": {"danger_stop_tighten": sum(1 for r in rows if r["danger_triggered"]), "baseline_h2": sum(1 for r in rows if not r["danger_triggered"])},
        },
        {
            "variant": "winner_extension_only_30m_anchor_and_btc_h4",
            "summary": h4_recovery_only,
            "actions": {"recovery_extend_h4": sum(1 for r in rows if r["recovery_triggered"]), "baseline_h2": sum(1 for r in rows if not r["recovery_triggered"])},
        },
        {
            "variant": "state_machine_danger_priority",
            "summary": summarize_key(rows, "danger_priority_net_bps"),
            "actions": count_actions(rows, "danger_priority_action"),
        },
        {
            "variant": "state_machine_recovery_priority",
            "summary": summarize_key(rows, "recovery_priority_net_bps"),
            "actions": count_actions(rows, "recovery_priority_action"),
        },
    ]
    base_sum = float(baseline.get("sum_bps") or 0.0)
    base_t3r = float(baseline.get("top3_winner_removed_sum_bps") or 0.0)
    for variant in variants:
        variant["delta_sum_bps"] = r1(float(variant["summary"].get("sum_bps") or 0.0) - base_sum)
        variant["delta_t3r_bps"] = r1(float(variant["summary"].get("top3_winner_removed_sum_bps") or 0.0) - base_t3r)
        variant["delta_max_loss_bps"] = r1(float(variant["summary"].get("max_loss_bps") or 0.0) - float(baseline.get("max_loss_bps") or 0.0))
    variants.sort(key=lambda r: (float(r["delta_t3r_bps"] or -1e18), float(r["delta_sum_bps"] or -1e18)), reverse=True)
    return {
        "generated_at_utc": utc_now(),
        "source_db": file_fingerprint(db_path),
        "config_id": CONFIG_ID,
        "baseline": baseline,
        "counts": {
            "rows": len(rows),
            "danger_trigger_n": sum(1 for r in rows if r["danger_triggered"]),
            "recovery_trigger_n": sum(1 for r in rows if r["recovery_triggered"]),
            "overlap_n": sum(1 for r in rows if r["danger_triggered"] and r["recovery_triggered"]),
        },
        "variants": variants,
        "rows": rows,
    }


def cell(summary: dict[str, Any]) -> str:
    return f"N={summary['n']} sum={summary['sum_bps']} med={summary['median_bps']} T3R={summary['top3_winner_removed_sum_bps']} max_loss={summary['max_loss_bps']}"


def render_md(report: dict[str, Any]) -> str:
    lines = [
        "# S34 V Engine State-Machine Management",
        "",
        f"Generated: `{report['generated_at_utc']}`",
        "",
        f"Config: `{report['config_id']}`",
        "",
        "Research-only. Combines stop-tighten and winner-extension overlays; no live/paper state changed.",
        "",
        f"Baseline H2: {cell(report['baseline'])}",
        "",
        "## State Counts",
        "",
        f"- Rows: `{report['counts']['rows']}`",
        f"- 5m danger triggers: `{report['counts']['danger_trigger_n']}`",
        f"- 30m recovery triggers: `{report['counts']['recovery_trigger_n']}`",
        f"- Danger/recovery overlap: `{report['counts']['overlap_n']}`",
        "",
        "## Variants",
        "",
        "| Rank | Variant | Actions | Summary | Delta sum | Delta T3R | Delta max loss |",
        "| ---: | --- | --- | --- | ---: | ---: | ---: |",
    ]
    for idx, row in enumerate(report["variants"], start=1):
        actions = ", ".join(f"{k}:{v}" for k, v in row["actions"].items())
        lines.append(
            f"| {idx} | `{row['variant']}` | `{actions}` | {cell(row['summary'])} | "
            f"{row['delta_sum_bps']} | {row['delta_t3r_bps']} | {row['delta_max_loss_bps']} |"
        )
    best = report["variants"][0] if report["variants"] else None
    lines.extend(["", "## Read", ""])
    if best:
        lines.append(f"- Best combined path by T3R: `{best['variant']}` -> {cell(best['summary'])}.")
    lines.append("- If combined state-machine underperforms winner-extension-only, keep stop-tighten as a separate safety shadow rather than coupling it into the exit engine.")
    lines.append("")
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test combined S34 V Engine state-machine management.")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--cancel-replace-json", type=Path, default=DEFAULT_CANCEL_REPLACE_JSON)
    parser.add_argument("--max-book-staleness-sec", type=int, default=10)
    parser.add_argument("--json-out", type=Path, default=OUT_JSON)
    parser.add_argument("--md-out", type=Path, default=OUT_MD)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    with sqlite3.connect(f"file:{args.db}?mode=ro", uri=True) as conn:
        report = build_report(
            conn,
            cancel_replace_path=args.cancel_replace_json,
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
