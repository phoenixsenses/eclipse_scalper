"""S34 next-question research results.

Research-only answers:
1. Fine sizing frontier from 10x to 20x.
2. Exit x sizing matrix using per-trade exit variant labels.
3. Tail-neighbor analysis for v0.2-like but filter-out nearby events.

No live executor, order logic, leverage, size, or .env changes.
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.research_s34_knowable_anchor_continuation import load_mark_index, signed_return_bps
from tools.research_s34_maker_fade import collect_events
from tools.research_s34_wave_absorption import book_features_at
from tools.s34_v_engine_execution_frontier import prior_return_bps
from tools.s34_v_engine_shadow_observer import ACCEL_WINDOW_SEC, BUCKET_SEC, HORIZON_SEC, MIN_GAP_SEC

DEFAULT_DB = ROOT / "data" / "microstructure.db"
MIRROR_LEDGER = ROOT / "reports" / "research" / "s34" / "S34_V_ENGINE_V0_2_SHADOW_MIRROR_LEDGER.jsonl"
FORWARD_PACK = ROOT / "reports" / "research" / "s34" / "S34_V_ENGINE_FORWARD_RESEARCH_PACK.json"
OUT_JSON = ROOT / "reports" / "research" / "s34" / "S34_V_ENGINE_NEXT_QUESTIONS_RESULTS.json"
OUT_MD = ROOT / "reports" / "research" / "s34" / "S34_V_ENGINE_NEXT_QUESTIONS_RESULTS.md"

START_EQUITY = 35.0
SYMBOL = "ETHUSDT"
LIQ_SIDE = "SELL"
THRESHOLD_USD = 200_000.0
V02_MIN_VDEPTH = 28.0
V02_MAX_VDEPTH = 40.0
V02_PRIOR4H_LT = -50.0
V02_MIN_BID_DEPTH = 135_423.8
FEE_BPS = 5.0


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def iso_ms(ts_ms: int) -> str:
    return datetime.fromtimestamp(int(ts_ms) / 1000.0, tz=timezone.utc).isoformat()


def r1(v: float | None) -> float | None:
    if v is None or not math.isfinite(float(v)):
        return None
    return round(float(v), 1)


def r3(v: float | None) -> float | None:
    if v is None or not math.isfinite(float(v)):
        return None
    return round(float(v), 3)


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if text:
            rows.append(json.loads(text))
    return rows


def summary(vals: list[float]) -> dict[str, Any]:
    vals = [float(v) for v in vals if math.isfinite(float(v))]
    if not vals:
        return {"n": 0, "sum_bps": 0.0, "median_bps": None, "win_rate": None, "max_loss_bps": None, "t3r_bps": 0.0}
    t3r = sum(sorted(vals, reverse=True)[3:]) if len(vals) > 3 else sum(vals)
    return {
        "n": len(vals),
        "sum_bps": r1(sum(vals)),
        "mean_bps": r1(sum(vals) / len(vals)),
        "median_bps": r1(median(vals)),
        "win_rate": r3(sum(1 for v in vals if v > 0) / len(vals)),
        "max_loss_bps": r1(min(vals)),
        "tail_lte_minus100_n": sum(1 for v in vals if v <= -100.0),
        "tail_lte_minus150_n": sum(1 for v in vals if v <= -150.0),
        "tail_lte_minus300_n": sum(1 for v in vals if v <= -300.0),
        "t3r_bps": r1(t3r),
    }


def compound(vals: list[float], ratio: float) -> dict[str, Any]:
    equity = START_EQUITY
    peak = equity
    max_dd = 0.0
    ruined_at = None
    for i, bps in enumerate(vals, start=1):
        equity *= 1.0 + float(ratio) * float(bps) / 10_000.0
        peak = max(peak, equity)
        max_dd = min(max_dd, equity - peak)
        if ruined_at is None and equity <= 0:
            ruined_at = i
    return {
        "end_equity": r3(equity),
        "multiple": r3(equity / START_EQUITY),
        "max_drawdown_pct": r3(abs(max_dd) / START_EQUITY * 100.0),
        "ruined_at": ruined_at,
    }


def closed_net_bps() -> list[float]:
    rows = [
        r for r in load_jsonl(MIRROR_LEDGER)
        if r.get("observation_status") == "CLOSED" and r.get("sim_status") == "FILLED" and r.get("net_bps") is not None
    ]
    rows.sort(key=lambda r: str(r.get("signal_utc") or ""))
    return [float(r["net_bps"]) for r in rows]


def fine_sizing_frontier(vals: list[float]) -> dict[str, Any]:
    rows = []
    for ratio in range(10, 21):
        base = compound(vals, ratio)
        m300 = compound(vals + [-300.0], ratio)
        m507 = compound(vals + [-507.0], ratio)
        rows.append(
            {
                "ratio": ratio,
                "observed_end": base["end_equity"],
                "observed_multiple": base["multiple"],
                "minus300_end": m300["end_equity"],
                "minus507_end": m507["end_equity"],
                "survives_minus300": m300["ruined_at"] is None and float(m300["end_equity"]) > 0.0,
                "survives_minus507": m507["ruined_at"] is None and float(m507["end_equity"]) > 0.0,
                "ruin_tail_bps": r1(-10_000.0 / float(ratio)),
            }
        )
    survive_507 = [r for r in rows if r["survives_minus507"]]
    return {
        "rows": rows,
        "max_ratio_survives_minus507": max((r["ratio"] for r in survive_507), default=None),
        "read": "Ratio is notional/equity. 34x is current env; this zooms 10x-20x.",
    }


def exit_variant_values() -> dict[str, list[float]]:
    pack = load_json(FORWARD_PACK, {})
    rows = (pack.get("exit_management") or {}).get("rows") or []
    by_variant: dict[str, list[float]] = {}
    for row in rows:
        for variant, sim in (row.get("variants") or {}).items():
            if sim.get("net_bps") is not None:
                by_variant.setdefault(variant, []).append(float(sim["net_bps"]))
    return by_variant


def exit_sizing_matrix() -> dict[str, Any]:
    ratios = [10, 12, 15, 18, 20, 34]
    variants = exit_variant_values()
    rows = []
    for variant, vals in sorted(variants.items()):
        for ratio in ratios:
            base = compound(vals, ratio)
            m300 = compound(vals + [-300.0], ratio)
            m507 = compound(vals + [-507.0], ratio)
            rows.append(
                {
                    "variant": variant,
                    "ratio": ratio,
                    "summary": summary(vals),
                    "observed_end": base["end_equity"],
                    "observed_multiple": base["multiple"],
                    "minus300_end": m300["end_equity"],
                    "minus507_end": m507["end_equity"],
                    "survives_minus300": m300["ruined_at"] is None and float(m300["end_equity"]) > 0.0,
                    "survives_minus507": m507["ruined_at"] is None and float(m507["end_equity"]) > 0.0,
                }
            )
    robust = [r for r in rows if r["survives_minus507"]]
    robust.sort(key=lambda r: (float(r["observed_end"]), float(r["minus507_end"])), reverse=True)
    all_rows = sorted(rows, key=lambda r: (float(r["observed_end"]), float(r["minus507_end"])), reverse=True)
    return {
        "rows": rows,
        "top_robust_minus507": robust[:10],
        "top_growth": all_rows[:10],
        "read": "Per-trade exit variants from forward research pack; tails are appended synthetic shocks.",
    }


def anchor_forward_bps(marks: Any, ts_ms: int) -> float | None:
    entry = marks.at_or_after(int(ts_ms))
    exit_ = marks.at_or_after(int(ts_ms) + HORIZON_SEC * 1000)
    if not entry or not exit_:
        return None
    return signed_return_bps("LONG", float(entry[1]), float(exit_[1])) - FEE_BPS


def tail_neighbor_analysis(conn: sqlite3.Connection) -> dict[str, Any]:
    marks = load_mark_index(conn, SYMBOL)
    events = collect_events(
        conn,
        symbol=SYMBOL,
        threshold=THRESHOLD_USD,
        sides=(LIQ_SIDE,),
        min_vdepth_bps=20.0,
        bucket_sec=BUCKET_SEC,
        min_gap_sec=MIN_GAP_SEC,
        accel_window_sec=ACCEL_WINDOW_SEC,
        max_horizon_sec=HORIZON_SEC,
    )
    rows = []
    for event in events:
        ts = int(event.anchor.anchor_ts_ms)
        vdepth = float(event.vdepth_bps)
        if not (20.0 <= vdepth < 50.0):
            continue
        prior4h = prior_return_bps(marks, ts, 4 * 3600)
        if prior4h is None or not math.isfinite(float(prior4h)) or not (float(prior4h) < -20.0):
            continue
        book = book_features_at(conn, SYMBOL, ts, 5)
        bid_depth = float(book.get("bid_depth_usd") or 0.0) if book else None
        net = anchor_forward_bps(marks, ts)
        if net is None:
            continue
        in_v02 = (
            V02_MIN_VDEPTH <= vdepth < V02_MAX_VDEPTH
            and float(prior4h) < V02_PRIOR4H_LT
            and bid_depth is not None
            and bid_depth >= V02_MIN_BID_DEPTH
        )
        if in_v02:
            bucket = "IN_V02_ANCHOR_CF"
        elif not (V02_MIN_VDEPTH <= vdepth < V02_MAX_VDEPTH):
            bucket = "NEAR_MISS_VDEPTH"
        elif not (float(prior4h) < V02_PRIOR4H_LT):
            bucket = "NEAR_MISS_PRIOR4H"
        elif bid_depth is None or bid_depth < V02_MIN_BID_DEPTH:
            bucket = "NEAR_MISS_BID_DEPTH"
        else:
            bucket = "NEAR_MISS_OTHER"
        rows.append(
            {
                "signal_utc": iso_ms(ts),
                "bucket": bucket,
                "vdepth_bps": r1(vdepth),
                "prior4h_bps": r1(prior4h),
                "bid_depth_usd": r1(bid_depth),
                "anchor_cf_net_bps": r1(net),
            }
        )
    by_bucket: dict[str, list[float]] = {}
    for row in rows:
        by_bucket.setdefault(str(row["bucket"]), []).append(float(row["anchor_cf_net_bps"]))
    return {
        "status": "TAIL_NEIGHBOR_ANCHOR_CF",
        "definition": "ETH SELL 200K, vdepth 20-50, prior4h<-20; classify v0.2 pass vs near-miss filters. Outcome is anchor mark 2h net bps, not maker fill.",
        "rows_n": len(rows),
        "by_bucket": {k: summary(v) for k, v in sorted(by_bucket.items())},
        "sample_rows": rows[:50],
        "read": "This is neighbor tail-risk context, not a live filter. It uses anchor counterfactual because near-misses do not all have maker lifecycle fills.",
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    vals = closed_net_bps()
    with sqlite3.connect(f"file:{args.db}?mode=ro", uri=True) as conn:
        neighbors = tail_neighbor_analysis(conn)
    return {
        "generated_at_utc": utc_now(),
        "status": "RESEARCH_ONLY_NO_LIVE_CHANGE",
        "source_n": len(vals),
        "fine_sizing_frontier": fine_sizing_frontier(vals),
        "exit_sizing_matrix": exit_sizing_matrix(),
        "tail_neighbor_analysis": neighbors,
        "read": "No live executor, leverage, size, order logic, or .env changes.",
    }


def render_md(report: dict[str, Any]) -> str:
    lines = [
        "# S34 V Engine Next Questions Results",
        "",
        f"Generated: `{report['generated_at_utc']}`",
        "",
        f"Status: `{report['status']}`. {report['read']}",
        "",
        "## Fine Sizing Frontier",
        "",
        f"Max ratio that survives appended -507 bps: `{report['fine_sizing_frontier']['max_ratio_survives_minus507']}`",
        "",
        "| Ratio | Observed End | -300 End | -507 End | Ruin Tail | Survive -507 |",
        "| ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in report["fine_sizing_frontier"]["rows"]:
        lines.append(
            f"| {row['ratio']} | {row['observed_end']} | {row['minus300_end']} | "
            f"{row['minus507_end']} | {row['ruin_tail_bps']} | {row['survives_minus507']} |"
        )
    lines.extend([
        "",
        "## Exit x Sizing Matrix: Top Robust (-507 survives)",
        "",
        "| Variant | Ratio | Observed End | -300 End | -507 End | Sum bps | T3R |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in report["exit_sizing_matrix"]["top_robust_minus507"]:
        s = row["summary"]
        lines.append(
            f"| {row['variant']} | {row['ratio']} | {row['observed_end']} | {row['minus300_end']} | "
            f"{row['minus507_end']} | {s.get('sum_bps')} | {s.get('t3r_bps')} |"
        )
    lines.extend([
        "",
        "## Exit x Sizing Matrix: Top Growth",
        "",
        "| Variant | Ratio | Observed End | -300 End | -507 End | Survive -507 |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ])
    for row in report["exit_sizing_matrix"]["top_growth"]:
        lines.append(
            f"| {row['variant']} | {row['ratio']} | {row['observed_end']} | {row['minus300_end']} | "
            f"{row['minus507_end']} | {row['survives_minus507']} |"
        )
    lines.extend([
        "",
        "## Tail Neighbor Analysis",
        "",
        report["tail_neighbor_analysis"]["definition"],
        "",
        "| Bucket | N | Sum bps | Median | Win | <=-100 | <=-150 | <=-300 | Max loss |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for bucket, row in report["tail_neighbor_analysis"]["by_bucket"].items():
        lines.append(
            f"| {bucket} | {row.get('n')} | {row.get('sum_bps')} | {row.get('median_bps')} | "
            f"{row.get('win_rate')} | {row.get('tail_lte_minus100_n')} | {row.get('tail_lte_minus150_n')} | "
            f"{row.get('tail_lte_minus300_n')} | {row.get('max_loss_bps')} |"
        )
    lines.append("")
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run S34 next-question research results.")
    p.add_argument("--db", type=Path, default=DEFAULT_DB)
    p.add_argument("--out-json", type=Path, default=OUT_JSON)
    p.add_argument("--out-md", type=Path, default=OUT_MD)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(args)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    md = render_md(report)
    args.out_md.write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
