from __future__ import annotations

import argparse
import csv
import json
import math
import sqlite3
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.research_s34_knowable_anchor_continuation import (
    AnchorSnapshot,
    MarkIndex,
    file_fingerprint,
    iso_ms,
    load_liquidations,
    load_mark_index,
    mean,
    pctile,
    r1,
    r3,
    reconstruct_anchors,
    sha256_text,
)
from tools.research_s34_knowable_anchor_route_recheck import (
    ROUTES,
    RouteSpec,
    route_filters_pass,
    simulate_route,
)


DEFAULT_DB = ROOT / "data" / "microstructure.db"
DEFAULT_TRADES = ROOT / "reports" / "research" / "s34" / "S34_SHADOW_PAPER_TRADES.json"
OUT_DIR = ROOT / "reports" / "research" / "s34"
OUT_MD = OUT_DIR / "S34_PAPER_VS_KNOWABLE_ANCHOR_AUDIT.md"
OUT_JSON = OUT_DIR / "S34_PAPER_VS_KNOWABLE_ANCHOR_AUDIT.json"
OUT_CSV = OUT_DIR / "S34_PAPER_VS_KNOWABLE_ANCHOR_AUDIT.csv"


ROUTE_BY_NAME = {spec.rule_name: spec for spec in ROUTES}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def summarize_values(values: list[float]) -> dict[str, Any]:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return {
        "n": len(vals),
        "mean": r1(mean(vals)),
        "median": r1(pctile(vals, 0.5)),
        "p25": r1(pctile(vals, 0.25)),
        "p75": r1(pctile(vals, 0.75)),
        "win_rate": r3(sum(v > 0.0 for v in vals) / len(vals)) if vals else None,
        "cum": r1(sum(vals)) if vals else 0.0,
    }


def load_closed_paper_trades(path: Path, route_names: set[str]) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    trades = payload.get("trades", payload)
    out: list[dict[str, Any]] = []
    for trade in trades:
        rule = trade.get("rule") or {}
        if trade.get("status") != "CLOSED":
            continue
        if str(rule.get("name") or "") not in route_names:
            continue
        out.append(trade)
    return out


def route_from_trade(trade: dict[str, Any]) -> RouteSpec:
    rule = trade.get("rule") or {}
    name = str(rule.get("name") or "")
    base = ROUTE_BY_NAME.get(name)
    if not base:
        raise KeyError(f"unsupported rule: {name}")
    return RouteSpec(
        family=base.family,
        rule_name=base.rule_name,
        symbol=str(rule.get("symbol") or base.symbol),
        liq_side=str(rule.get("liq_side") or base.liq_side),
        direction=str(rule.get("direction") or base.direction),
        threshold_usd=float(rule.get("threshold_usd") or base.threshold_usd),
        tp_bps=float(rule.get("tp_bps") or base.tp_bps),
        sl_bps=float(rule.get("sl_bps") or base.sl_bps),
        be_bps=float(rule.get("be_trigger_bps") or rule.get("be_bps") or base.be_bps),
        entry_delay_sec=base.entry_delay_sec,
        max_horizon_sec=int(rule.get("max_horizon_sec") or base.max_horizon_sec),
        min_day_trend_bps=base.min_day_trend_bps,
        btc_pre_window_sec=base.btc_pre_window_sec,
        btc_pre_min_return_bps=base.btc_pre_min_return_bps,
        max_single_dominance_pct=base.max_single_dominance_pct,
    )


def load_context(
    conn: sqlite3.Connection,
    spec: RouteSpec,
    trades: list[dict[str, Any]],
    *,
    bucket_sec: int,
    min_gap_sec: int,
    accel_window_sec: int,
    warmup_sec: int,
) -> dict[str, Any]:
    signal_times = [int(t["signal_ts_ms"]) for t in trades]
    start_ms = min(signal_times) - int(warmup_sec) * 1000
    end_ms = max(signal_times) + (int(spec.max_horizon_sec) + int(bucket_sec) + int(warmup_sec)) * 1000
    liqs = load_liquidations(conn, spec.symbol, spec.liq_side, start_ms, end_ms)
    anchors = reconstruct_anchors(
        liqs,
        bucket_sec=bucket_sec,
        min_gap_sec=min_gap_sec,
        thresholds=(float(spec.threshold_usd),),
        accel_window_sec=accel_window_sec,
    )
    anchors_by_bucket = {int(a.bucket): a for a in anchors}
    return {
        "liqs": liqs,
        "anchors_by_bucket": anchors_by_bucket,
        "marks": load_mark_index(conn, spec.symbol),
    }


def bucket_liqs(liqs: list[dict[str, Any]], bucket: int, bucket_sec: int) -> list[dict[str, Any]]:
    bucket_ms = int(bucket_sec) * 1000
    start = int(bucket) * bucket_ms
    end = start + bucket_ms
    return [row for row in liqs if start <= int(row["ts_ms"]) < end]


def cumulative_at(rows: list[dict[str, Any]], ts_ms: int) -> float:
    return sum(float(row["notional"]) for row in rows if int(row["ts_ms"]) <= int(ts_ms))


def classify_timing(signal_ts_ms: int, anchor: AnchorSnapshot | None, threshold: float, cum_at_signal: float) -> str:
    if anchor is None:
        return "NO_RECONSTRUCTED_ANCHOR"
    delta_ms = int(signal_ts_ms) - int(anchor.anchor_ts_ms)
    if float(cum_at_signal) < float(threshold):
        return "SIGNAL_BEFORE_KNOWABLE_THRESHOLD"
    if delta_ms < -1000:
        return "SIGNAL_TS_BEFORE_ANCHOR"
    if abs(delta_ms) <= 1000:
        return "SIGNAL_TS_MATCHES_ANCHOR"
    return "SIGNAL_TS_AFTER_ANCHOR"


def audit_trade(
    conn: sqlite3.Connection,
    trade: dict[str, Any],
    context: dict[str, Any],
    btc_marks: MarkIndex,
    *,
    bucket_sec: int,
) -> dict[str, Any]:
    spec = route_from_trade(trade)
    rule = trade.get("rule") or {}
    signal = trade.get("signal") or {}
    bucket = int(signal.get("bucket") or int(trade["signal_ts_ms"]) // (int(bucket_sec) * 1000))
    signal_ts_ms = int(trade["signal_ts_ms"])
    rows = bucket_liqs(context["liqs"], bucket, bucket_sec)
    anchor = context["anchors_by_bucket"].get(bucket)
    total_bucket = sum(float(row["notional"]) for row in rows)
    cum_signal = cumulative_at(rows, signal_ts_ms)
    row: dict[str, Any] = {
        "trade_id": trade.get("trade_id"),
        "rule_name": spec.rule_name,
        "symbol": spec.symbol,
        "direction": spec.direction,
        "bucket": bucket,
        "paper_signal_ts_ms": signal_ts_ms,
        "paper_signal_utc": iso_ms(signal_ts_ms),
        "paper_entry_ts_ms": (trade.get("entry_fill") or {}).get("ts_ms") or (signal.get("entry_fill") or {}).get("ts_ms"),
        "paper_exit_ts_ms": trade.get("exit_ts_ms"),
        "paper_exit_reason": trade.get("exit_reason"),
        "paper_net_bps": r1(float(trade["net_bps"])) if finite(trade.get("net_bps")) else None,
        "paper_signal_liq_total_notional": r1(float(signal.get("liq_total_notional"))) if finite(signal.get("liq_total_notional")) else None,
        "reconstructed_bucket_notional": r1(total_bucket),
        "cum_notional_at_paper_signal": r1(cum_signal),
        "threshold_usd": float(spec.threshold_usd),
        "bucket_liq_count": len(rows),
        "paper_signal_liq_count": signal.get("liq_count"),
    }
    if anchor is None:
        row.update(
            {
                "anchor_status": "MISSING",
                "timing_class": classify_timing(signal_ts_ms, None, spec.threshold_usd, cum_signal),
                "reconstructed_net_bps": None,
                "paper_minus_reconstructed_bps": None,
            }
        )
        return row

    marks: MarkIndex = context["marks"]
    filter_ok, filter_reason, filter_diag = route_filters_pass(spec, anchor, marks, btc_marks)
    row.update(
        {
            "anchor_status": "FILTER_PASS" if filter_ok else "FILTER_REJECT",
            "filter_reason": filter_reason,
            "filter_diagnostics": filter_diag,
            "reconstructed_anchor_ts_ms": int(anchor.anchor_ts_ms),
            "reconstructed_anchor_utc": iso_ms(anchor.anchor_ts_ms),
            "anchor_first_ts_ms": int(anchor.first_ts_ms),
            "anchor_elapsed_since_first_sec": r1(float(anchor.elapsed_since_first_sec)),
            "signal_minus_anchor_sec": r1((signal_ts_ms - int(anchor.anchor_ts_ms)) / 1000.0),
            "running_notional_at_anchor": r1(float(anchor.running_notional)),
            "running_liq_count_at_anchor": int(anchor.running_liq_count),
            "running_single_liq_dominance": r1(float(anchor.running_single_liq_dominance)),
            "timing_class": classify_timing(signal_ts_ms, anchor, spec.threshold_usd, cum_signal),
        }
    )
    if not filter_ok:
        row.update({"reconstructed_net_bps": None, "paper_minus_reconstructed_bps": None})
        return row

    fee_bps_side = float(rule.get("taker_fee_bps") or rule.get("round_trip_cost_bps", 8.0) / 2.0)
    max_book_staleness_sec = int(rule.get("max_book_staleness_sec") or 5)
    sim = simulate_route(
        conn,
        marks,
        spec,
        anchor,
        fee_bps_side=fee_bps_side,
        max_book_staleness_sec=max_book_staleness_sec,
    )
    rec_net = sim.get("net_bps")
    row.update(
        {
            "reconstructed_status": sim.get("status"),
            "reconstructed_entry_target_ts_ms": sim.get("entry_target_ts_ms"),
            "reconstructed_entry_book_ts_ms": sim.get("entry_book_ts_ms"),
            "reconstructed_entry_reference_price": sim.get("entry_reference_price"),
            "reconstructed_entry_price": sim.get("entry_price"),
            "reconstructed_exit_ts_ms": sim.get("exit_ts_ms"),
            "reconstructed_exit_reason": sim.get("exit_reason"),
            "reconstructed_exit_price": sim.get("exit_price"),
            "reconstructed_net_bps": r1(float(rec_net)) if finite(rec_net) else None,
            "reconstructed_counterfactual_mark_net_bps": r1(float(sim.get("counterfactual_mark_net_bps")))
            if finite(sim.get("counterfactual_mark_net_bps"))
            else None,
            "paper_minus_reconstructed_bps": r1(float(trade["net_bps"]) - float(rec_net))
            if finite(trade.get("net_bps")) and finite(rec_net)
            else None,
        }
    )
    return row


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    timing: dict[str, int] = {}
    statuses: dict[str, int] = {}
    by_rule: dict[str, dict[str, Any]] = {}
    for row in rows:
        timing[str(row.get("timing_class") or "NA")] = timing.get(str(row.get("timing_class") or "NA"), 0) + 1
        statuses[str(row.get("anchor_status") or "NA")] = statuses.get(str(row.get("anchor_status") or "NA"), 0) + 1
    for rule in sorted({str(r["rule_name"]) for r in rows}):
        subset = [r for r in rows if str(r["rule_name"]) == rule]
        by_rule[rule] = {
            "n": len(subset),
            "paper": summarize_values([float(r["paper_net_bps"]) for r in subset if finite(r.get("paper_net_bps"))]),
            "reconstructed": summarize_values(
                [float(r["reconstructed_net_bps"]) for r in subset if finite(r.get("reconstructed_net_bps"))]
            ),
            "delta": summarize_values(
                [float(r["paper_minus_reconstructed_bps"]) for r in subset if finite(r.get("paper_minus_reconstructed_bps"))]
            ),
            "signal_minus_anchor_sec": summarize_values(
                [float(r["signal_minus_anchor_sec"]) for r in subset if finite(r.get("signal_minus_anchor_sec"))]
            ),
            "timing": {
                key: sum(1 for r in subset if r.get("timing_class") == key)
                for key in sorted({str(r.get("timing_class") or "NA") for r in subset})
            },
        }
    return {
        "n": len(rows),
        "timing_class_counts": dict(sorted(timing.items())),
        "anchor_status_counts": dict(sorted(statuses.items())),
        "paper": summarize_values([float(r["paper_net_bps"]) for r in rows if finite(r.get("paper_net_bps"))]),
        "reconstructed": summarize_values([float(r["reconstructed_net_bps"]) for r in rows if finite(r.get("reconstructed_net_bps"))]),
        "delta": summarize_values(
            [float(r["paper_minus_reconstructed_bps"]) for r in rows if finite(r.get("paper_minus_reconstructed_bps"))]
        ),
        "signal_minus_anchor_sec": summarize_values(
            [float(r["signal_minus_anchor_sec"]) for r in rows if finite(r.get("signal_minus_anchor_sec"))]
        ),
        "by_rule": by_rule,
    }


def write_outputs(payload: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    fieldnames = sorted({key for row in rows for key in row.keys() if key != "filter_diagnostics"})
    with OUT_CSV.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})

    lines = [
        "# S34 Paper vs Knowable-Anchor Audit",
        "",
        f"Generated: `{payload['generated_at_utc']}`",
        "",
        "RESEARCH_ONLY. Closed shadow-paper trades are compared against reconstructed real-time-knowable threshold-cross anchors in the same bucket.",
        "",
        "## Overall",
        "",
        "| N | Paper Median | Paper Mean | Recon Median | Recon Mean | Delta Median | Signal-Anchor Median Sec |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    s = payload["summary"]
    lines.append(
        f"| {s['n']} | {s['paper']['median']} | {s['paper']['mean']} | {s['reconstructed']['median']} | "
        f"{s['reconstructed']['mean']} | {s['delta']['median']} | {s['signal_minus_anchor_sec']['median']} |"
    )
    lines.extend(["", "## Timing Classes", "", "| Class | Count |", "| --- | ---: |"])
    for key, value in s["timing_class_counts"].items():
        lines.append(f"| {key} | {value} |")
    lines.extend(["", "## By Rule", "", "| Rule | N | Paper Med/Mean | Recon Med/Mean | Delta Med | Signal-Anchor Med Sec | Timing |", "| --- | ---: | --- | --- | ---: | ---: | --- |"])
    for rule, item in s["by_rule"].items():
        timing = ", ".join(f"{k}={v}" for k, v in item["timing"].items())
        lines.append(
            f"| `{rule}` | {item['n']} | {item['paper']['median']} / {item['paper']['mean']} | "
            f"{item['reconstructed']['median']} / {item['reconstructed']['mean']} | {item['delta']['median']} | "
            f"{item['signal_minus_anchor_sec']['median']} | {timing} |"
        )
    lines.extend(["", "## Largest Paper-Reconstruction Deltas", "", "| Trade | Rule | Paper | Recon | Delta | Timing | Signal-Anchor Sec |", "| --- | --- | ---: | ---: | ---: | --- | ---: |"])
    deltas = [r for r in rows if finite(r.get("paper_minus_reconstructed_bps"))]
    for row in sorted(deltas, key=lambda r: abs(float(r["paper_minus_reconstructed_bps"])), reverse=True)[:20]:
        lines.append(
            f"| {row.get('trade_id')} | `{row.get('rule_name')}` | {row.get('paper_net_bps')} | "
            f"{row.get('reconstructed_net_bps')} | {row.get('paper_minus_reconstructed_bps')} | "
            f"{row.get('timing_class')} | {row.get('signal_minus_anchor_sec')} |"
        )
    lines.append("")
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit shadow paper closed trades against knowable threshold-cross anchors.")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--trades", type=Path, default=DEFAULT_TRADES)
    parser.add_argument("--bucket-sec", type=int, default=300)
    parser.add_argument("--min-gap-sec", type=int, default=900)
    parser.add_argument("--accel-window-sec", type=int, default=30)
    parser.add_argument("--warmup-sec", type=int, default=21600)
    parser.add_argument("--rules", default=",".join(sorted(ROUTE_BY_NAME)))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    route_names = {name.strip() for name in str(args.rules).split(",") if name.strip()}
    trades = load_closed_paper_trades(args.trades, route_names)
    with sqlite3.connect(f"file:{args.db}?mode=ro", uri=True) as conn:
        btc_marks = load_mark_index(conn, "BTCUSDT")
        grouped: dict[str, list[dict[str, Any]]] = {}
        for trade in trades:
            grouped.setdefault(str((trade.get("rule") or {}).get("name")), []).append(trade)
        contexts: dict[str, dict[str, Any]] = {}
        for name, group in grouped.items():
            spec = route_from_trade(group[0])
            contexts[name] = load_context(
                conn,
                spec,
                group,
                bucket_sec=int(args.bucket_sec),
                min_gap_sec=int(args.min_gap_sec),
                accel_window_sec=int(args.accel_window_sec),
                warmup_sec=int(args.warmup_sec),
            )
        rows = [
            audit_trade(conn, trade, contexts[str((trade.get("rule") or {}).get("name"))], btc_marks, bucket_sec=int(args.bucket_sec))
            for trade in trades
        ]
    payload = {
        "generated_at_utc": utc_now(),
        "source_db": file_fingerprint(args.db),
        "trades_file": file_fingerprint(args.trades),
        "config": {
            "bucket_sec": int(args.bucket_sec),
            "min_gap_sec": int(args.min_gap_sec),
            "accel_window_sec": int(args.accel_window_sec),
            "warmup_sec": int(args.warmup_sec),
            "rules": sorted(route_names),
        },
        "routes_sha256": sha256_text(json.dumps([asdict(ROUTE_BY_NAME[n]) for n in sorted(route_names) if n in ROUTE_BY_NAME], sort_keys=True)),
        "summary": summarize_rows(rows),
        "rows": rows,
    }
    write_outputs(payload, rows)
    print(json.dumps({"n": len(rows), "summary": payload["summary"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
