"""S34 V Engine v0.3 research results.

Research-only:
1. Define/evaluate v0.3 shadow candidate: same v0.2 entry, TP300/SL150/4h,
   sizing ratios 15x and 18x.
2. Ablate vdepth and bid-depth bands around the v0.2 route.
3. Multi-tail stress for 15x vs 18x.

No live executor, order logic, size, leverage, or .env changes.
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
FORWARD_PACK = ROOT / "reports" / "research" / "s34" / "S34_V_ENGINE_FORWARD_RESEARCH_PACK.json"
OUT_JSON = ROOT / "reports" / "research" / "s34" / "S34_V_ENGINE_V03_RESEARCH_RESULTS.json"
OUT_MD = ROOT / "reports" / "research" / "s34" / "S34_V_ENGINE_V03_RESEARCH_RESULTS.md"

START_EQUITY = 35.0
SYMBOL = "ETHUSDT"
LIQ_SIDE = "SELL"
THRESHOLD_USD = 200_000.0
FEE_BPS = 5.0
V02_VDEPTH = (28.0, 40.0)
V02_PRIOR4H_LT = -50.0
V02_BID_DEPTH_MIN = 135_423.8


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


def exit_values(variant: str) -> list[float]:
    pack = load_json(FORWARD_PACK, {})
    rows = (pack.get("exit_management") or {}).get("rows") or []
    vals = []
    for row in rows:
        sim = (row.get("variants") or {}).get(variant) or {}
        if sim.get("net_bps") is not None:
            vals.append(float(sim["net_bps"]))
    return vals


def v03_shadow_definition() -> dict[str, Any]:
    vals = exit_values("tp300_sl150_4h")
    return {
        "protocol_id": "S34_V_ENGINE_V0_3_SHADOW_TP300_SL150_4H",
        "status": "PROPOSED_SHADOW_ONLY_NOT_LIVE",
        "entry": "same v0.2 entry: ETH SELL 200K, vdepth 28-40, prior4h<-50, bid_depth>=135423.8, maker lifecycle O20/W300/O5/C1",
        "exit": "TP300 / SL150 / max 4h",
        "sizing_modes": {
            "V03_15X": {"ratio": 15, **compound(vals, 15), "minus300": compound(vals + [-300.0], 15), "minus507": compound(vals + [-507.0], 15)},
            "V03_18X": {"ratio": 18, **compound(vals, 18), "minus300": compound(vals + [-300.0], 18), "minus507": compound(vals + [-507.0], 18)},
        },
        "exit_summary": summary(vals),
        "acceptance_draft": {
            "forward_n_min": 20,
            "must_have": ["sum_bps>0", "t3r_bps>0", "minus300_survival", "minus507_survival", "queue_stress_minus40_positive"],
            "permission": "shadow only; no live authorization",
        },
    }


def anchor_cf(marks: Any, ts_ms: int) -> float | None:
    entry = marks.at_or_after(int(ts_ms))
    exit_ = marks.at_or_after(int(ts_ms) + HORIZON_SEC * 1000)
    if not entry or not exit_:
        return None
    return signed_return_bps("LONG", float(entry[1]), float(exit_[1])) - FEE_BPS


def collect_ablation_rows(conn: sqlite3.Connection) -> list[dict[str, Any]]:
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
        bid = float(book.get("bid_depth_usd") or 0.0) if book else 0.0
        net = anchor_cf(marks, ts)
        if net is None:
            continue
        rows.append(
            {
                "ts_ms": ts,
                "signal_utc": iso_ms(ts),
                "vdepth_bps": vdepth,
                "prior4h_bps": float(prior4h),
                "bid_depth_usd": bid,
                "net_bps": float(net),
            }
        )
    return rows


def bin_label(value: float, bins: list[tuple[float, float]]) -> str | None:
    for lo, hi in bins:
        if lo <= value < hi:
            return f"{lo:g}_{hi:g}"
    return None


def vdepth_bid_ablation(conn: sqlite3.Connection) -> dict[str, Any]:
    rows = collect_ablation_rows(conn)
    vbins = [(20, 24), (24, 28), (28, 32), (32, 36), (36, 40), (40, 45), (45, 50)]
    dbins = [(0, 100_000), (100_000, 135_423.8), (135_423.8, 200_000), (200_000, 400_000), (400_000, 1_000_000_000)]
    by_v: dict[str, list[float]] = {}
    by_d: dict[str, list[float]] = {}
    by_cross: dict[str, list[float]] = {}
    for row in rows:
        vlab = bin_label(float(row["vdepth_bps"]), vbins)
        dlab = bin_label(float(row["bid_depth_usd"]), dbins)
        if vlab:
            by_v.setdefault(vlab, []).append(float(row["net_bps"]))
        if dlab:
            by_d.setdefault(dlab, []).append(float(row["net_bps"]))
        if vlab and dlab:
            by_cross.setdefault(f"v{vlab}|d{dlab}", []).append(float(row["net_bps"]))
    return {
        "status": "ANCHOR_CF_ABLATION",
        "definition": "ETH SELL 200K neighbors, vdepth 20-50, prior4h<-20. Outcome is anchor mark 2h net bps.",
        "rows_n": len(rows),
        "by_vdepth_bin": {k: summary(v) for k, v in sorted(by_v.items())},
        "by_bid_depth_bin": {k: summary(v) for k, v in sorted(by_d.items())},
        "best_cross_bins": sorted(
            [{"cell": k, **summary(v)} for k, v in by_cross.items()],
            key=lambda r: (float(r.get("t3r_bps") or -1e9), float(r.get("sum_bps") or -1e9)),
            reverse=True,
        )[:15],
        "worst_cross_bins": sorted(
            [{"cell": k, **summary(v)} for k, v in by_cross.items()],
            key=lambda r: (float(r.get("sum_bps") or 1e9), float(r.get("max_loss_bps") or 1e9)),
        )[:10],
    }


def multi_tail_stress() -> dict[str, Any]:
    vals = exit_values("tp300_sl150_4h")
    scenarios = {
        "observed": vals,
        "two_minus300_end": vals + [-300.0, -300.0],
        "minus507_then_minus150": vals + [-507.0, -150.0],
        "minus300_then_minus507": vals + [-300.0, -507.0],
        "every5_minus150": vals[:5] + [-150.0] + vals[5:10] + [-150.0] + vals[10:],
        "every5_minus300": vals[:5] + [-300.0] + vals[5:10] + [-300.0] + vals[10:],
    }
    return {
        scenario: {
            "15x": compound(seq, 15),
            "18x": compound(seq, 18),
            "19x": compound(seq, 19),
        }
        for scenario, seq in scenarios.items()
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    with sqlite3.connect(f"file:{args.db}?mode=ro", uri=True) as conn:
        ablation = vdepth_bid_ablation(conn)
    return {
        "generated_at_utc": utc_now(),
        "status": "RESEARCH_ONLY_NO_LIVE_CHANGE",
        "v03_shadow_definition": v03_shadow_definition(),
        "vdepth_bid_depth_ablation": ablation,
        "multi_tail_stress": multi_tail_stress(),
        "read": "No live executor, leverage, size, order logic, or .env changes.",
    }


def render_md(report: dict[str, Any]) -> str:
    v03 = report["v03_shadow_definition"]
    lines = [
        "# S34 V Engine v0.3 Research Results",
        "",
        f"Generated: `{report['generated_at_utc']}`",
        "",
        f"Status: `{report['status']}`. {report['read']}",
        "",
        "## v0.3 Shadow Candidate",
        "",
        f"- Protocol: `{v03['protocol_id']}`",
        f"- Status: `{v03['status']}`",
        f"- Entry: {v03['entry']}",
        f"- Exit: {v03['exit']}",
        f"- Exit summary: N={v03['exit_summary']['n']} sum={v03['exit_summary']['sum_bps']} med={v03['exit_summary']['median_bps']} T3R={v03['exit_summary']['t3r_bps']}",
        "",
        "| Mode | Ratio | Observed End | -300 End | -507 End |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, row in v03["sizing_modes"].items():
        lines.append(
            f"| {name} | {row['ratio']} | {row['end_equity']} | {row['minus300']['end_equity']} | {row['minus507']['end_equity']} |"
        )
    lines.extend([
        "",
        "## vDepth Ablation",
        "",
        "| vDepth bin | N | Sum | Median | Win | <=-100 | <=-150 | <=-300 | Max loss | T3R |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for cell, row in report["vdepth_bid_depth_ablation"]["by_vdepth_bin"].items():
        lines.append(
            f"| {cell} | {row['n']} | {row['sum_bps']} | {row['median_bps']} | {row['win_rate']} | "
            f"{row['tail_lte_minus100_n']} | {row['tail_lte_minus150_n']} | {row['tail_lte_minus300_n']} | {row['max_loss_bps']} | {row['t3r_bps']} |"
        )
    lines.extend([
        "",
        "## Bid-Depth Ablation",
        "",
        "| Bid-depth bin | N | Sum | Median | Win | <=-100 | <=-150 | <=-300 | Max loss | T3R |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for cell, row in report["vdepth_bid_depth_ablation"]["by_bid_depth_bin"].items():
        lines.append(
            f"| {cell} | {row['n']} | {row['sum_bps']} | {row['median_bps']} | {row['win_rate']} | "
            f"{row['tail_lte_minus100_n']} | {row['tail_lte_minus150_n']} | {row['tail_lte_minus300_n']} | {row['max_loss_bps']} | {row['t3r_bps']} |"
        )
    lines.extend([
        "",
        "## Best Cross Bins",
        "",
        "| Cell | N | Sum | Median | Win | Max loss | T3R |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in report["vdepth_bid_depth_ablation"]["best_cross_bins"][:8]:
        lines.append(
            f"| {row['cell']} | {row['n']} | {row['sum_bps']} | {row['median_bps']} | {row['win_rate']} | {row['max_loss_bps']} | {row['t3r_bps']} |"
        )
    lines.extend([
        "",
        "## Multi-Tail Stress: TP300/SL150/4h",
        "",
        "| Scenario | 15x End | 18x End | 19x End | 18x Ruined |",
        "| --- | ---: | ---: | ---: | --- |",
    ])
    for scenario, row in report["multi_tail_stress"].items():
        lines.append(
            f"| {scenario} | {row['15x']['end_equity']} | {row['18x']['end_equity']} | {row['19x']['end_equity']} | {row['18x']['ruined_at']} |"
        )
    lines.append("")
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run S34 v0.3 research results.")
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
