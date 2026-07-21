"""S34 resonance dose-response from real-fill sync x absorption rows.

Reads a S34_SYNC_ABSORPTION_REALFILL*.json report and sweeps synchronization
thresholds / asset-count buckets. Research-only.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.research_s34_sync_absorption_realfill import metrics, r1


DEFAULT_IN = ROOT / "reports" / "research" / "s34" / "S34_SYNC_ABSORPTION_REALFILL_V28_40.json"
OUT_DIR = ROOT / "reports" / "research" / "s34"
OUT_JSON = OUT_DIR / "S34_RESONANCE_DOSE_RESPONSE.json"
OUT_MD = OUT_DIR / "S34_RESONANCE_DOSE_RESPONSE.md"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return metrics([float(r["net_bps"]) for r in rows if r.get("net_bps") is not None and math.isfinite(float(r["net_bps"]))])


def threshold_sweep(rows: list[dict[str, Any]], thresholds: tuple[float, ...], *, require_bid_support: bool) -> list[dict[str, Any]]:
    out = []
    for thr in thresholds:
        sub = [r for r in rows if float(r.get("market_concurrent_k") or 0.0) >= float(thr)]
        if require_bid_support:
            sub = [r for r in sub if r.get("imbalance_gate") == "bid_support"]
        out.append({"threshold_k": float(thr), "require_bid_support": bool(require_bid_support), "summary": summarize(sub)})
    out.sort(key=lambda r: (float(r["summary"]["t3r_bps"] or -1e18), float(r["summary"]["sum_bps"] or -1e18)), reverse=True)
    return out


def asset_count_sweep(rows: list[dict[str, Any]], *, require_bid_support: bool) -> list[dict[str, Any]]:
    out = []
    for count in sorted({int(r.get("asset_count_200k") or 0) for r in rows}):
        sub = [r for r in rows if int(r.get("asset_count_200k") or 0) == count]
        if require_bid_support:
            sub = [r for r in sub if r.get("imbalance_gate") == "bid_support"]
        out.append({"asset_count_200k": count, "require_bid_support": bool(require_bid_support), "summary": summarize(sub)})
    out.sort(key=lambda r: int(r["asset_count_200k"]))
    return out


def build_report(payload: dict[str, Any], *, source_path: Path) -> dict[str, Any]:
    rows = list(payload.get("rows", []))
    thresholds = (0.0, 50.0, 100.0, 200.0, 300.0, 500.0, 1000.0)
    return {
        "generated_at_utc": utc_now(),
        "source_report": str(source_path),
        "source_config": payload.get("config", {}),
        "event_n": len(rows),
        "threshold_sweep_all": threshold_sweep(rows, thresholds, require_bid_support=False),
        "threshold_sweep_bid_support": threshold_sweep(rows, thresholds, require_bid_support=True),
        "asset_count_all": asset_count_sweep(rows, require_bid_support=False),
        "asset_count_bid_support": asset_count_sweep(rows, require_bid_support=True),
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
        "# S34 Resonance Dose Response",
        "",
        f"Generated: `{report['generated_at_utc']}`",
        "",
        f"Source: `{report['source_report']}`",
        "",
        f"Route: `{cfg.get('symbol')} SELL deep-V {cfg.get('min_vdepth_bps')}bps-{cfg.get('max_vdepth_bps') or 'inf'}bps, {cfg.get('horizon_hr')}h real-fill`",
        f"Rows: `{report['event_n']}`",
        "",
        "## Sync Threshold Sweep",
        "",
        "| Rank | Threshold K | Summary |",
        "| ---: | ---: | --- |",
    ]
    for idx, row in enumerate(report["threshold_sweep_all"], start=1):
        lines.append(f"| {idx} | {row['threshold_k']} | {cell(row['summary'])} |")
    lines.extend(["", "## Sync Threshold + Bid Support", "", "| Rank | Threshold K | Summary |", "| ---: | ---: | --- |"])
    for idx, row in enumerate(report["threshold_sweep_bid_support"], start=1):
        lines.append(f"| {idx} | {row['threshold_k']} | {cell(row['summary'])} |")
    lines.extend(["", "## Asset Count", "", "| Asset Count | All | Bid Support Only |", "| ---: | --- | --- |"])
    bid_by_count = {r["asset_count_200k"]: r for r in report["asset_count_bid_support"]}
    for row in report["asset_count_all"]:
        bid = bid_by_count.get(row["asset_count_200k"], {"summary": {}})
        lines.append(f"| {row['asset_count_200k']} | {cell(row['summary'])} | {cell(bid['summary'])} |")
    best_all = report["threshold_sweep_all"][0] if report["threshold_sweep_all"] else None
    best_bid = report["threshold_sweep_bid_support"][0] if report["threshold_sweep_bid_support"] else None
    lines.extend(["", "## Read", ""])
    if best_all:
        lines.append(f"- Best threshold without absorption: `{best_all['threshold_k']}K` -> {cell(best_all['summary'])}.")
    if best_bid:
        lines.append(f"- Best threshold with bid_support: `{best_bid['threshold_k']}K` -> {cell(best_bid['summary'])}.")
    lines.append("- This is a dose-response screen; a threshold is only believable if it improves T3R/tails without collapsing N.")
    lines.append("")
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run S34 resonance dose-response from real-fill rows.")
    parser.add_argument("--input-json", type=Path, default=DEFAULT_IN)
    parser.add_argument("--json-out", type=Path, default=OUT_JSON)
    parser.add_argument("--md-out", type=Path, default=OUT_MD)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload = json.loads(args.input_json.read_text(encoding="utf-8"))
    report = build_report(payload, source_path=args.input_json)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    args.md_out.write_text(render_md(report), encoding="utf-8")
    print(render_md(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
