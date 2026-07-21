from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
IN_JSON = ROOT / "reports" / "research" / "s34" / "S34_ABSORPTION_SYNC_2X2_POOL.json"
FALLBACK_JSON = ROOT / "reports" / "research" / "s34" / "S34_CROSS_ASSET_ABSORPTION_POOL.json"
OUT_JSON = ROOT / "reports" / "research" / "s34" / "S34_V3_ROUTE_NODE_MAP.json"
OUT_MD = ROOT / "reports" / "research" / "s34" / "S34_V3_ROUTE_NODE_MAP.md"

MIN_SPLIT_N = 40


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def finite(value: Any) -> float | None:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    vals = [finite(r.get("net_bps")) for r in rows]
    vals = [v for v in vals if v is not None]
    if not vals:
        return {
            "n": 0,
            "sum_bps": 0.0,
            "mean_bps": None,
            "median_bps": None,
            "win_rate_pct": None,
            "t3r_bps": 0.0,
            "max_loss_bps": None,
            "tail_lt_100": 0,
            "tail_lt_200": 0,
        }
    ordered = sorted(vals, reverse=True)
    return {
        "n": len(vals),
        "sum_bps": round(sum(vals), 1),
        "mean_bps": round(sum(vals) / len(vals), 1),
        "median_bps": round(median(vals), 1),
        "win_rate_pct": round(100.0 * sum(1 for v in vals if v > 0) / len(vals), 1),
        "t3r_bps": round(sum(ordered[3:]) if len(ordered) > 3 else sum(vals), 1),
        "max_loss_bps": round(min(vals), 1),
        "tail_lt_100": sum(1 for v in vals if v < -100.0),
        "tail_lt_200": sum(1 for v in vals if v < -200.0),
    }


def split_sets(payload: dict[str, Any]) -> set[str]:
    return set(payload.get("split", {}).get("holdout_months", []))


def split_metrics(rows: list[dict[str, Any]], hold_months: set[str]) -> dict[str, Any]:
    return {
        "all": metrics(rows),
        "cal": metrics([r for r in rows if str(r.get("month")) not in hold_months]),
        "hold": metrics([r for r in rows if str(r.get("month")) in hold_months]),
    }


def route_components(route_id: str) -> dict[str, Any]:
    parts = route_id.split("_")
    return {
        "symbol": parts[0] if parts else None,
        "threshold": next((p for p in parts if p.startswith("T") and p.endswith("K")), None),
        "vdepth_band": next((p for p in parts if p.startswith("v")), None),
        "horizon": parts[-1] if parts else None,
    }


def score_node(summary: dict[str, Any]) -> dict[str, Any]:
    cal = summary["cal"]
    hold = summary["hold"]
    strong_n = cal["n"] >= MIN_SPLIT_N and hold["n"] >= MIN_SPLIT_N
    hold_positive = hold["sum_bps"] > 0 and hold["t3r_bps"] > 0 and (hold["median_bps"] or 0) > 0
    cal_positive = cal["sum_bps"] > 0 and cal["t3r_bps"] > 0
    tail_ok = hold["tail_lt_100"] == 0 if strong_n else False
    status = "HYPOTHESIS"
    if strong_n and cal_positive and hold_positive and tail_ok:
        status = "STRONG_NODE"
    elif strong_n and hold_positive and cal_positive:
        status = "WATCH_NODE"
    elif hold["n"] == 0:
        status = "NO_HOLDOUT"
    elif hold["sum_bps"] <= 0 or hold["t3r_bps"] <= 0:
        status = "WEAK_OR_DEAD"
    score = (
        float(hold["t3r_bps"] or 0.0)
        + 0.25 * float(hold["sum_bps"] or 0.0)
        - 100.0 * float(hold["tail_lt_100"] or 0)
        - 25.0 * max(0, MIN_SPLIT_N - int(hold["n"]))
    )
    return {
        "status": status,
        "passes_min_split_n": strong_n,
        "passes_cal_positive": cal_positive,
        "passes_hold_positive": hold_positive,
        "passes_tail_zero": tail_ok,
        "node_score": round(score, 1),
    }


def gate_delta(rows: list[dict[str, Any]], hold_months: set[str], key: str, good: str, bad: str) -> dict[str, Any]:
    hold = [r for r in rows if str(r.get("month")) in hold_months]
    good_m = metrics([r for r in hold if str(r.get(key)) == good])
    bad_m = metrics([r for r in hold if str(r.get(key)) == bad])
    return {
        "gate": key,
        "good": good,
        "bad": bad,
        "good_hold": good_m,
        "bad_hold": bad_m,
        "delta_t3r_bps": round(float(good_m["t3r_bps"] or 0.0) - float(bad_m["t3r_bps"] or 0.0), 1),
        "delta_sum_bps": round(float(good_m["sum_bps"] or 0.0) - float(bad_m["sum_bps"] or 0.0), 1),
    }


def build_report(payload: dict[str, Any], source: Path) -> dict[str, Any]:
    rows = list(payload["rows"])
    hold_months = split_sets(payload)
    by_route = {}
    for route in sorted({str(r["route_id"]) for r in rows}):
        rrows = [r for r in rows if str(r["route_id"]) == route]
        summary = split_metrics(rrows, hold_months)
        node = {
            "route_id": route,
            **route_components(route),
            "summary": summary,
            "score": score_node(summary),
            "leverage_points_holdout": [
                gate_delta(rrows, hold_months, "sync_gate", "sync", "idio") if "sync_gate" in rrows[0] else None,
                gate_delta(rrows, hold_months, "bid_depth_gate", "deep_bid", "shallow_bid"),
                gate_delta(rrows, hold_months, "imbalance_gate", "bid_support", "ask_heavy"),
                gate_delta(rrows, hold_months, "absorption_gate", "mixed", "absorbed"),
            ],
        }
        node["leverage_points_holdout"] = [x for x in node["leverage_points_holdout"] if x is not None]
        by_route[route] = node
    nodes = list(by_route.values())
    nodes.sort(key=lambda n: (n["score"]["node_score"], n["summary"]["hold"]["t3r_bps"], n["summary"]["hold"]["n"]), reverse=True)
    return {
        "generated_at_utc": utc_now(),
        "source": str(source),
        "discipline": {
            "min_split_n": MIN_SPLIT_N,
            "verdict_rule": "N<40 in either calibration or holdout is HYPOTHESIS only, never edge.",
        },
        "split": payload.get("split", {}),
        "overall": split_metrics(rows, hold_months),
        "node_counts": {
            status: sum(1 for n in nodes if n["score"]["status"] == status)
            for status in sorted({n["score"]["status"] for n in nodes})
        },
        "routes": nodes,
        "strong_nodes": [n for n in nodes if n["score"]["status"] == "STRONG_NODE"],
        "watch_nodes": [n for n in nodes if n["score"]["status"] == "WATCH_NODE"],
        "hypothesis_nodes": [n for n in nodes if n["score"]["status"] == "HYPOTHESIS"],
    }


def fmt(s: dict[str, Any]) -> str:
    return (
        f"N={s['n']} sum={s['sum_bps']} med={s['median_bps']} "
        f"T3R={s['t3r_bps']} max_loss={s['max_loss_bps']} tail<-100={s['tail_lt_100']}"
    )


def render(report: dict[str, Any]) -> str:
    lines = [
        "# S34 v3 Route Node Map",
        "",
        f"Generated: `{report['generated_at_utc']}`",
        "",
        "Research-only. No live/paper/executor changes.",
        "",
        "## Discipline",
        "",
        f"- Min split N: `{report['discipline']['min_split_n']}` calibration AND holdout.",
        f"- Rule: {report['discipline']['verdict_rule']}",
        "",
        "## Overall",
        "",
        f"- All: {fmt(report['overall']['all'])}",
        f"- Calibration: {fmt(report['overall']['cal'])}",
        f"- Holdout: {fmt(report['overall']['hold'])}",
        f"- Node counts: `{report['node_counts']}`",
        "",
        "## Ranked Routes",
        "",
        "| Rank | Status | Route | Cal | Hold | Score | Best holdout leverage point |",
        "| ---: | --- | --- | --- | --- | ---: | --- |",
    ]
    for idx, node in enumerate(report["routes"], start=1):
        leverage = sorted(node["leverage_points_holdout"], key=lambda x: x["delta_t3r_bps"], reverse=True)
        best = "NA"
        if leverage:
            b = leverage[0]
            best = f"`{b['gate']}:{b['good']}>{b['bad']}` dT3R={b['delta_t3r_bps']} dSum={b['delta_sum_bps']}"
        lines.append(
            f"| {idx} | `{node['score']['status']}` | `{node['route_id']}` | "
            f"{fmt(node['summary']['cal'])} | {fmt(node['summary']['hold'])} | "
            f"{node['score']['node_score']} | {best} |"
        )
    lines += [
        "",
        "## Read",
        "",
    ]
    if report["strong_nodes"]:
        lines.append(f"- Strong nodes found: `{len(report['strong_nodes'])}`.")
    else:
        lines.append("- No route clears the N>=40 per-split + positive holdout + tail gate. There is no validated network node yet.")
    if report["watch_nodes"]:
        lines.append(f"- Watch nodes found: `{len(report['watch_nodes'])}`.")
    else:
        lines.append("- No watch node clears the full N gate either; small-N winners remain hypotheses.")
    top = report["routes"][0] if report["routes"] else None
    if top:
        lines.append(f"- Top ranked node by holdout-aware score: `{top['route_id']}` -> status `{top['score']['status']}`.")
    lines.append("- Keep route families separate; do not create a single pooled live rule from this map.")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    source = IN_JSON if IN_JSON.exists() else FALLBACK_JSON
    payload = json.loads(source.read_text(encoding="utf-8"))
    report = build_report(payload, source)
    OUT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    OUT_MD.write_text(render(report), encoding="utf-8")
    print(render(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
