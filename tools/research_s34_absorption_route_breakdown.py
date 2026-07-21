"""Route-level breakdown for cross-asset absorption pool results."""

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

from tools.research_s34_cross_asset_absorption_pool import metrics, r1


DEFAULT_IN = ROOT / "reports" / "research" / "s34" / "S34_CROSS_ASSET_ABSORPTION_POOL.json"
OUT_DIR = ROOT / "reports" / "research" / "s34"
OUT_JSON = OUT_DIR / "S34_ABSORPTION_ROUTE_BREAKDOWN.json"
OUT_MD = OUT_DIR / "S34_ABSORPTION_ROUTE_BREAKDOWN.md"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return metrics([float(r["net_bps"]) for r in rows if r.get("net_bps") is not None and math.isfinite(float(r["net_bps"]))])


def build(payload: dict[str, Any], *, min_n: int) -> dict[str, Any]:
    rows = list(payload.get("rows", []))
    route_ids = sorted({str(r["route_id"]) for r in rows})
    out = []
    for rid in route_ids:
        route_rows = [r for r in rows if str(r["route_id"]) == rid]
        if len(route_rows) < int(min_n):
            continue
        deep = [r for r in route_rows if r.get("bid_depth_gate") == "deep_bid"]
        shallow = [r for r in route_rows if r.get("bid_depth_gate") == "shallow_bid"]
        bid_support = [r for r in route_rows if r.get("imbalance_gate") == "bid_support"]
        ask_heavy = [r for r in route_rows if r.get("imbalance_gate") == "ask_heavy"]
        deep_s = summarize(deep)
        shallow_s = summarize(shallow)
        bid_s = summarize(bid_support)
        ask_s = summarize(ask_heavy)
        out.append(
            {
                "route_id": rid,
                "n": len(route_rows),
                "overall": summarize(route_rows),
                "deep_bid": deep_s,
                "shallow_bid": shallow_s,
                "deep_minus_shallow_t3r": r1(float(deep_s["t3r_bps"] or 0.0) - float(shallow_s["t3r_bps"] or 0.0)),
                "deep_minus_shallow_sum": r1(float(deep_s["sum_bps"] or 0.0) - float(shallow_s["sum_bps"] or 0.0)),
                "bid_support": bid_s,
                "ask_heavy": ask_s,
                "bid_minus_ask_t3r": r1(float(bid_s["t3r_bps"] or 0.0) - float(ask_s["t3r_bps"] or 0.0)),
                "bid_minus_ask_sum": r1(float(bid_s["sum_bps"] or 0.0) - float(ask_s["sum_bps"] or 0.0)),
            }
        )
    out.sort(key=lambda r: (float(r["deep_minus_shallow_t3r"] or -1e18), float(r["deep_minus_shallow_sum"] or -1e18)), reverse=True)
    return {
        "generated_at_utc": utc_now(),
        "source_config": payload.get("config", {}),
        "min_route_n": int(min_n),
        "route_n": len(out),
        "routes": out,
    }


def cell(s: dict[str, Any]) -> str:
    return (
        f"N={s['n']} sum={s['sum_bps']} med={s['median_bps']} "
        f"T3R={s['t3r_bps']} max_loss={s['max_loss_bps']} tail<-100={s['tail_n_lt_-100']}"
    )


def render(report: dict[str, Any]) -> str:
    lines = [
        "# S34 Absorption Route Breakdown",
        "",
        f"Generated: `{report['generated_at_utc']}`",
        "",
        f"Routes with N >= {report['min_route_n']}: `{report['route_n']}`",
        "",
        "## Deep Bid vs Shallow Bid",
        "",
        "| Rank | Route | Overall | Deep bid | Shallow bid | Delta T3R | Delta sum |",
        "| ---: | --- | --- | --- | --- | ---: | ---: |",
    ]
    for i, row in enumerate(report["routes"][:40], start=1):
        lines.append(
            f"| {i} | `{row['route_id']}` | {cell(row['overall'])} | {cell(row['deep_bid'])} | "
            f"{cell(row['shallow_bid'])} | {row['deep_minus_shallow_t3r']} | {row['deep_minus_shallow_sum']} |"
        )
    lines.extend(["", "## Bid Support vs Ask Heavy", ""])
    lines.append("| Rank | Route | Bid support | Ask heavy | Delta T3R | Delta sum |")
    lines.append("| ---: | --- | --- | --- | ---: | ---: |")
    bid_ranked = sorted(report["routes"], key=lambda r: (float(r["bid_minus_ask_t3r"] or -1e18), float(r["bid_minus_ask_sum"] or -1e18)), reverse=True)
    for i, row in enumerate(bid_ranked[:40], start=1):
        lines.append(
            f"| {i} | `{row['route_id']}` | {cell(row['bid_support'])} | {cell(row['ask_heavy'])} | "
            f"{row['bid_minus_ask_t3r']} | {row['bid_minus_ask_sum']} |"
        )
    lines.extend(["", "## Read", ""])
    positive_deep = [r for r in report["routes"] if float(r["deep_minus_shallow_t3r"] or 0.0) > 0.0 and int(r["deep_bid"]["n"] or 0) >= 3 and int(r["shallow_bid"]["n"] or 0) >= 3]
    lines.append(f"- Routes where deep_bid beats shallow_bid on T3R with both sides N>=3: `{len(positive_deep)}`.")
    if positive_deep:
        lines.append(f"- Best route-level deep_bid lead: `{positive_deep[0]['route_id']}` delta T3R `{positive_deep[0]['deep_minus_shallow_t3r']}`.")
    lines.append("")
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Route-level absorption breakdown.")
    parser.add_argument("--input-json", type=Path, default=DEFAULT_IN)
    parser.add_argument("--min-route-n", type=int, default=10)
    parser.add_argument("--json-out", type=Path, default=OUT_JSON)
    parser.add_argument("--md-out", type=Path, default=OUT_MD)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload = json.loads(args.input_json.read_text(encoding="utf-8"))
    report = build(payload, min_n=int(args.min_route_n))
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    args.md_out.write_text(render(report), encoding="utf-8")
    print(render(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
