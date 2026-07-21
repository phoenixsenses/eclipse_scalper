"""S34 maker-lifecycle absorption permission test.

Applies the T=0 book absorption gate to the current maker lifecycle
O20_W300_O5_C1, then combines it with the state-machine managed outcome.
Research-only; no live/paper changes.
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

from tools.research_s34_maker_fade import summarize
from tools.research_s34_wave_absorption import book_features_at
from tools.research_s34_knowable_anchor_continuation import file_fingerprint, pctile, r1


DEFAULT_DB = ROOT / "data" / "microstructure.db"
DEFAULT_CANCEL_REPLACE_JSON = ROOT / "reports" / "research" / "s34" / "S34_V_ENGINE_CANCEL_REPLACE.json"
DEFAULT_STATE_MACHINE_JSON = ROOT / "reports" / "research" / "s34" / "S34_V_ENGINE_STATE_MACHINE_MANAGEMENT.json"
OUT_DIR = ROOT / "reports" / "research" / "s34"
OUT_JSON = OUT_DIR / "S34_MAKER_ABSORPTION_PERMISSION.json"
OUT_MD = OUT_DIR / "S34_MAKER_ABSORPTION_PERMISSION.md"

CONFIG_ID = "O20_W300_O5_C1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def finite(value: Any) -> float | None:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def quantile(rows: list[dict[str, Any]], key: str, q: float) -> float | None:
    vals = [float(v) for r in rows if (v := finite(r.get(key))) is not None]
    return pctile(vals, q) if vals else None


def classify_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    cuts = {
        "imbalance_med": quantile(rows, "book_imbalance", 0.5),
        "bid_depth_med": quantile(rows, "bid_depth_usd", 0.5),
        "imbalance_p25": quantile(rows, "book_imbalance", 0.25),
        "bid_depth_p25": quantile(rows, "bid_depth_usd", 0.25),
    }
    for row in rows:
        imb = finite(row.get("book_imbalance"))
        bid = finite(row.get("bid_depth_usd"))
        if imb is None or bid is None:
            row["imbalance_gate"] = "no_book"
            row["bid_depth_gate"] = "no_book"
            row["absorption_gate"] = "no_book"
            continue
        row["imbalance_gate"] = "bid_support" if cuts["imbalance_med"] is not None and imb >= float(cuts["imbalance_med"]) else "ask_heavy"
        row["bid_depth_gate"] = "deep_bid" if cuts["bid_depth_med"] is not None and bid >= float(cuts["bid_depth_med"]) else "shallow_bid"
        if row["imbalance_gate"] == "bid_support" and row["bid_depth_gate"] == "deep_bid":
            row["absorption_gate"] = "absorbed"
        elif (
            cuts["imbalance_p25"] is not None
            and cuts["bid_depth_p25"] is not None
            and imb <= float(cuts["imbalance_p25"])
            and bid <= float(cuts["bid_depth_p25"])
        ):
            row["absorption_gate"] = "vacuum_like"
        else:
            row["absorption_gate"] = "mixed"
    return {k: r1(v) if isinstance(v, float) else v for k, v in cuts.items()}


def annotate_book(conn: sqlite3.Connection, row: dict[str, Any], *, max_book_staleness_sec: int) -> dict[str, Any]:
    ts = row.get("anchor_ts_ms")
    if ts is None:
        ts = row.get("signal_ts_ms")
    out = dict(row)
    feat = book_features_at(conn, str(row.get("symbol") or "ETHUSDT"), int(ts), int(max_book_staleness_sec)) if ts is not None else None
    if feat:
        out.update(
            {
                "book_imbalance": feat["book_imbalance"],
                "bid_depth_usd": feat["bid_depth_usd"],
                "spread_bps": feat["spread_bps"],
                "book_staleness_ms": feat["book_staleness_ms"],
            }
        )
    else:
        out.update({"book_imbalance": None, "bid_depth_usd": None, "spread_bps": None, "book_staleness_ms": None})
    return out


def gate_summary(rows: list[dict[str, Any]], gate_key: str, gate_value: str, pnl_key: str = "net_bps") -> dict[str, Any]:
    eligible = [r for r in rows if r.get(gate_key) == gate_value]
    filled = [r for r in eligible if r.get("status") == "FILLED" and finite(r.get(pnl_key)) is not None]
    no_fill = [r for r in eligible if r.get("status") != "FILLED"]
    return {
        "gate": f"{gate_key}={gate_value}",
        "eligible_n": len(eligible),
        "filled_n": len(filled),
        "no_fill_n": len(no_fill),
        "fill_rate": r1(len(filled) / len(eligible) * 100.0) if eligible else None,
        "filled_summary": summarize([float(r[pnl_key]) for r in filled]),
        "no_fill_anchor_cf": summarize([float(v) for r in no_fill if (v := finite(r.get("anchor_cf_net_bps"))) is not None]),
    }


def pnl_summary(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    return summarize([float(v) for r in rows if (v := finite(r.get(key))) is not None])


def state_by_anchor(rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    out = {}
    for row in rows:
        ts = row.get("anchor_ts_ms")
        if ts is not None:
            out[int(ts)] = row
    return out


def build_report(
    conn: sqlite3.Connection,
    *,
    db_path: Path,
    cancel_replace_path: Path,
    state_machine_path: Path,
    max_book_staleness_sec: int,
) -> dict[str, Any]:
    cr = load_json(cancel_replace_path)
    result = next(r for r in cr.get("results", []) if r.get("config_id") == CONFIG_ID)
    eligible_rows = [
        annotate_book(conn, dict(r), max_book_staleness_sec=max_book_staleness_sec)
        for r in result.get("rows", [])
    ]
    cuts = classify_rows(eligible_rows)
    filled_rows = [r for r in eligible_rows if r.get("status") == "FILLED" and finite(r.get("net_bps")) is not None]

    sm = load_json(state_machine_path)
    sm_idx = state_by_anchor(sm.get("rows", []))
    managed_rows = []
    for row in filled_rows:
        ts = row.get("anchor_ts_ms")
        sm_row = sm_idx.get(int(ts)) if ts is not None else None
        if not sm_row:
            continue
        managed_rows.append(
            {
                **row,
                "state_machine_net_bps": sm_row.get("recovery_priority_net_bps"),
                "state_machine_action": sm_row.get("recovery_priority_action"),
                "danger_triggered": sm_row.get("danger_triggered"),
                "recovery_triggered": sm_row.get("recovery_triggered"),
            }
        )

    gates = [
        gate_summary(eligible_rows, "imbalance_gate", "bid_support"),
        gate_summary(eligible_rows, "imbalance_gate", "ask_heavy"),
        gate_summary(eligible_rows, "imbalance_gate", "no_book"),
        gate_summary(eligible_rows, "bid_depth_gate", "deep_bid"),
        gate_summary(eligible_rows, "bid_depth_gate", "shallow_bid"),
        gate_summary(eligible_rows, "bid_depth_gate", "no_book"),
        gate_summary(eligible_rows, "absorption_gate", "absorbed"),
        gate_summary(eligible_rows, "absorption_gate", "mixed"),
        gate_summary(eligible_rows, "absorption_gate", "vacuum_like"),
        gate_summary(eligible_rows, "absorption_gate", "no_book"),
    ]
    sm_gates = []
    for key, value in (
        ("imbalance_gate", "bid_support"),
        ("imbalance_gate", "ask_heavy"),
        ("imbalance_gate", "no_book"),
        ("bid_depth_gate", "deep_bid"),
        ("bid_depth_gate", "shallow_bid"),
        ("bid_depth_gate", "no_book"),
        ("absorption_gate", "absorbed"),
        ("absorption_gate", "mixed"),
        ("absorption_gate", "vacuum_like"),
        ("absorption_gate", "no_book"),
    ):
        sub = [r for r in managed_rows if r.get(key) == value]
        sm_gates.append(
            {
                "gate": f"{key}={value}",
                "filled_n": len(sub),
                "baseline_h2": pnl_summary(sub, "net_bps"),
                "state_machine": pnl_summary(sub, "state_machine_net_bps"),
                "actions": action_counts(sub),
            }
        )

    return {
        "generated_at_utc": utc_now(),
        "source_db": file_fingerprint(db_path),
        "config_id": CONFIG_ID,
        "cuts": cuts,
        "eligible_n": len(eligible_rows),
        "book_covered_n": sum(1 for r in eligible_rows if r.get("imbalance_gate") != "no_book"),
        "filled_n": len(filled_rows),
        "baseline": pnl_summary(filled_rows, "net_bps"),
        "permission_gates": gates,
        "state_machine_gates": sm_gates,
        "rows": managed_rows,
    }


def action_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for row in rows:
        key = str(row.get("state_machine_action") or "none")
        out[key] = out.get(key, 0) + 1
    return dict(sorted(out.items()))


def cell(summary: dict[str, Any]) -> str:
    return (
        f"N={summary['n']} sum={summary['sum_bps']} med={summary['median_bps']} "
        f"T3R={summary['top3_winner_removed_sum_bps']} max_loss={summary['max_loss_bps']}"
    )


def render_md(report: dict[str, Any]) -> str:
    lines = [
        "# S34 Maker Absorption Permission",
        "",
        f"Generated: `{report['generated_at_utc']}`",
        "",
        "Research-only. Applies T=0 book absorption to current maker lifecycle and state-machine outcomes. No live/paper state changed.",
        "",
        f"Config: `{report['config_id']}`",
        f"Eligible: `{report['eligible_n']}`; book-covered eligible: `{report['book_covered_n']}`; filled: `{report['filled_n']}`",
        "",
        f"Baseline filled H2: {cell(report['baseline'])}",
        "",
        "## Cuts",
        "",
    ]
    for key, value in report["cuts"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Permission Gates On Maker Lifecycle", ""])
    lines.append("| Gate | Eligible | Filled | Fill% | Filled summary | No-fill anchor CF |")
    lines.append("| --- | ---: | ---: | ---: | --- | --- |")
    for row in report["permission_gates"]:
        lines.append(
            f"| `{row['gate']}` | {row['eligible_n']} | {row['filled_n']} | {row['fill_rate']} | "
            f"{cell(row['filled_summary'])} | {cell(row['no_fill_anchor_cf'])} |"
        )
    lines.extend(["", "## Permission + State Machine", ""])
    lines.append("| Gate | Filled | Baseline H2 | State machine | Actions |")
    lines.append("| --- | ---: | --- | --- | --- |")
    for row in report["state_machine_gates"]:
        actions = ", ".join(f"{k}:{v}" for k, v in row["actions"].items())
        lines.append(
            f"| `{row['gate']}` | {row['filled_n']} | {cell(row['baseline_h2'])} | "
            f"{cell(row['state_machine'])} | `{actions}` |"
        )
    lines.extend(["", "## Read", ""])
    gate_map = {row["gate"]: row for row in report["permission_gates"]}
    bid = gate_map.get("imbalance_gate=bid_support")
    ask = gate_map.get("imbalance_gate=ask_heavy")
    if bid and ask:
        lines.append(
            f"- Maker filled bid_support vs ask_heavy delta T3R: "
            f"`{r1(float(bid['filled_summary']['top3_winner_removed_sum_bps'] or 0.0) - float(ask['filled_summary']['top3_winner_removed_sum_bps'] or 0.0))}`; "
            f"delta max_loss `{r1(float(bid['filled_summary']['max_loss_bps'] or 0.0) - float(ask['filled_summary']['max_loss_bps'] or 0.0))}`."
        )
    sm_map = {row["gate"]: row for row in report["state_machine_gates"]}
    sm_bid = sm_map.get("imbalance_gate=bid_support")
    if sm_bid:
        lines.append(f"- Bid_support + state-machine: {cell(sm_bid['state_machine'])}.")
    lines.append("- A live-adjacent permission gate must improve filled P&L without just deleting the fills that carry expectancy.")
    lines.append("")
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run maker absorption permission test.")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--cancel-replace-json", type=Path, default=DEFAULT_CANCEL_REPLACE_JSON)
    parser.add_argument("--state-machine-json", type=Path, default=DEFAULT_STATE_MACHINE_JSON)
    parser.add_argument("--max-book-staleness-sec", type=int, default=10)
    parser.add_argument("--json-out", type=Path, default=OUT_JSON)
    parser.add_argument("--md-out", type=Path, default=OUT_MD)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    with sqlite3.connect(f"file:{args.db}?mode=ro", uri=True) as conn:
        report = build_report(
            conn,
            db_path=args.db,
            cancel_replace_path=args.cancel_replace_json,
            state_machine_path=args.state_machine_json,
            max_book_staleness_sec=int(args.max_book_staleness_sec),
        )
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    args.md_out.write_text(render_md(report), encoding="utf-8")
    print(render_md(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
