"""S34 Conditional Edge Screen.

The unconditional route recheck says every S34 route loses. This asks the next
question: is there any *knowable-at-cross* feature that isolates a positive,
holdout-stable sub-population? i.e. not "do 500K crosses make money" but "do
500K crosses *with property X* make money, and does that survive out-of-sample".

Method (no lookahead, no peeking):
  1. Reconstruct route anchors (running threshold cross) and simulate each with
     the same frozen TP/SL/BE exits as the recheck (reused `simulate_route`).
  2. Attach only features knowable at the cross (anchor running values + day/BTC
     context at-or-before the cross + entry spread).
  3. Chronological split: calibration = earlier buckets, holdout = later buckets.
  4. For each numeric feature, derive tercile cut points FROM CALIBRATION ONLY,
     then score every bin on calibration AND apply the same cuts to holdout.
     A bin is a CANDIDATE only if it is positive on calibration AND still
     positive on holdout with enough filled trades in both.

Screening many features inflates false positives -- a candidate here is a lead
to re-test in a fresh registered route recheck, not a green light.
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.research_s34_knowable_anchor_continuation import (
    AnchorSnapshot,
    book_at,
    load_liquidations,
    load_mark_index,
    mean,
    pctile,
    r1,
    r3,
    reconstruct_anchors,
    split_anchor_ids,
)
from tools.research_s34_knowable_anchor_route_recheck import (
    ROUTES,
    RouteSpec,
    anchor_shape_label,
    day_trend_bps,
    route_filters_pass,
    simulate_route,
)
from tools.s34_cascade_navigation_dashboard import session_label

DEFAULT_DB = ROOT / "data" / "microstructure.db"
OUT_DIR = ROOT / "reports" / "research" / "s34"
OUT_JSON = OUT_DIR / "S34_CONDITIONAL_EDGE_SCREEN.json"
OUT_MD = OUT_DIR / "S34_CONDITIONAL_EDGE_SCREEN.md"

DEFAULT_RULES = (
    "ETH_BUY_LIQ_LONG_50K_TP120_SL40_BE30",
    "ETH_BUY_LIQ_LONG_200K_TP60_SL40_BE30",
    "ETH_SELL_LIQ_SHORT_500K_TP60_SL40_BE40",
    "SOL_BUY_LIQ_LONG_200K_TP60_SL40_BE30",
)
ROUTE_BY_NAME = {spec.rule_name: spec for spec in ROUTES}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def summ(vals: list[float]) -> dict[str, Any]:
    vals = [v for v in vals if math.isfinite(v)]
    if not vals:
        return {"n": 0, "median": None, "mean": None, "cum": 0.0, "win_rate": None}
    return {
        "n": len(vals),
        "median": r1(pctile(vals, 0.5)),
        "mean": r1(mean(vals)),
        "cum": r1(sum(vals)),
        "win_rate": r3(sum(1 for v in vals if v > 0) / len(vals)),
    }


def numeric_features() -> dict[str, Callable[[dict[str, Any]], float | None]]:
    return {
        "size_notional": lambda f: f["running_notional"],
        "liq_count": lambda f: f["running_liq_count"],
        "elapsed_sec": lambda f: f["elapsed_since_first_sec"],
        "dominance_pct": lambda f: f["running_single_liq_dominance"],
        "rate_usd_per_sec": lambda f: f["running_rate"],
        "accel_usd_per_sec": lambda f: f["running_accel"],
        "day_trend_bps": lambda f: f["day_trend_bps"],
        "btc_ret_bps": lambda f: f["btc_ret_bps"],
        "btc_ret_abs_bps": lambda f: (abs(f["btc_ret_bps"]) if f["btc_ret_bps"] is not None else None),
        "entry_spread_bps": lambda f: f["entry_spread_bps"],
        "hour_utc": lambda f: f["hour_utc"],
    }


def categorical_features() -> dict[str, Callable[[dict[str, Any]], str]]:
    return {"session": lambda f: f["session"], "shape": lambda f: f["shape"]}


def anchor_knowable_features(
    conn: sqlite3.Connection, spec: RouteSpec, anchor: AnchorSnapshot, marks, btc_marks, *, max_book_staleness_sec: int
) -> dict[str, Any]:
    ts = int(anchor.anchor_ts_ms)
    entry_target = ts + int(spec.entry_delay_sec) * 1000
    book = book_at(conn, spec.symbol, entry_target, max_book_staleness_sec)
    spread = None
    if book and book.mid > 0:
        spread = (book.ask - book.bid) / book.mid * 10_000.0
    return {
        "running_notional": float(anchor.running_notional),
        "running_liq_count": int(anchor.running_liq_count),
        "elapsed_since_first_sec": float(anchor.elapsed_since_first_sec),
        "running_single_liq_dominance": float(anchor.running_single_liq_dominance),
        "running_rate": float(anchor.running_rate),
        "running_accel": float(anchor.running_accel),
        "day_trend_bps": day_trend_bps(marks, ts),
        "btc_ret_bps": btc_marks.ret_bps(ts - int(spec.btc_pre_window_sec or 900) * 1000, ts),
        "entry_spread_bps": spread,
        "hour_utc": float(datetime.fromtimestamp(ts / 1000.0, tz=timezone.utc).hour),
        "session": session_label(ts),
        "shape": anchor_shape_label(anchor),
    }


def screen_numeric(
    name: str,
    getter: Callable[[dict[str, Any]], float | None],
    cal: list[dict[str, Any]],
    hold: list[dict[str, Any]],
    *,
    min_cal: int,
    min_hold: int,
) -> dict[str, Any]:
    cal_pairs = [(getter(r["features"]), r["net_bps"]) for r in cal if getter(r["features"]) is not None]
    cal_vals = sorted(v for v, _ in cal_pairs)
    if len(cal_vals) < min_cal:
        return {"feature": name, "status": "THIN", "bins": []}
    q33, q66 = pctile(cal_vals, 1 / 3), pctile(cal_vals, 2 / 3)

    def assign(v: float) -> str:
        if v <= q33:
            return "low"
        if v <= q66:
            return "mid"
        return "high"

    def bin_rows(rows: list[dict[str, Any]], label: str) -> list[float]:
        out = []
        for r in rows:
            v = getter(r["features"])
            if v is not None and assign(v) == label:
                out.append(float(r["net_bps"]))
        return out

    bins = []
    for label in ("low", "mid", "high"):
        cs = summ(bin_rows(cal, label))
        hs = summ(bin_rows(hold, label))
        candidate = (
            cs["n"] >= min_cal // 3
            and hs["n"] >= min_hold // 3
            and (cs["median"] or -1) > 0
            and (cs["mean"] or -1) > 0
            and (hs["median"] or -1) > 0
            and (hs["mean"] or -1) > 0
        )
        bins.append({"bin": label, "cal": cs, "hold": hs, "candidate": candidate})
    return {
        "feature": name,
        "status": "SCREENED",
        "cut_low": r1(q33),
        "cut_high": r1(q66),
        "bins": bins,
        "has_candidate": any(b["candidate"] for b in bins),
    }


def screen_categorical(
    name: str, getter: Callable[[dict[str, Any]], str], cal, hold, *, min_cal: int, min_hold: int
) -> dict[str, Any]:
    cats = sorted({getter(r["features"]) for r in cal} | {getter(r["features"]) for r in hold})
    bins = []
    for cat in cats:
        cs = summ([float(r["net_bps"]) for r in cal if getter(r["features"]) == cat])
        hs = summ([float(r["net_bps"]) for r in hold if getter(r["features"]) == cat])
        candidate = (
            cs["n"] >= min_cal // 3
            and hs["n"] >= min_hold // 3
            and (cs["median"] or -1) > 0
            and (cs["mean"] or -1) > 0
            and (hs["median"] or -1) > 0
            and (hs["mean"] or -1) > 0
        )
        bins.append({"bin": cat, "cal": cs, "hold": hs, "candidate": candidate})
    return {"feature": name, "status": "SCREENED", "bins": bins, "has_candidate": any(b["candidate"] for b in bins)}


def screen_route(
    conn: sqlite3.Connection,
    spec: RouteSpec,
    *,
    bucket_sec: int,
    min_gap_sec: int,
    accel_window_sec: int,
    holdout_frac: float,
    fee_bps_side: float,
    max_book_staleness_sec: int,
    min_cal: int,
    min_hold: int,
) -> dict[str, Any]:
    liqs = load_liquidations(conn, spec.symbol, spec.liq_side, None, None)
    marks = load_mark_index(conn, spec.symbol)
    btc_marks = load_mark_index(conn, "BTCUSDT")
    anchors = reconstruct_anchors(
        liqs, bucket_sec=bucket_sec, min_gap_sec=min_gap_sec,
        thresholds=(float(spec.threshold_usd),), accel_window_sec=accel_window_sec,
    )
    filtered = [a for a in anchors if route_filters_pass(spec, a, marks, btc_marks)[0]]
    _, holdout_ids, split = split_anchor_ids(filtered, holdout_frac)

    rows: list[dict[str, Any]] = []
    for anchor in filtered:
        sim = simulate_route(conn, marks, spec, anchor, fee_bps_side=fee_bps_side, max_book_staleness_sec=max_book_staleness_sec)
        if sim.get("status") != "FILLED" or sim.get("net_bps") is None:
            continue
        rows.append({
            "split": "holdout" if str(anchor.bucket) in holdout_ids else "calibration",
            "net_bps": float(sim["net_bps"]),
            "features": anchor_knowable_features(conn, spec, anchor, marks, btc_marks, max_book_staleness_sec=max_book_staleness_sec),
        })
    cal = [r for r in rows if r["split"] == "calibration"]
    hold = [r for r in rows if r["split"] == "holdout"]

    feature_results = []
    for name, getter in numeric_features().items():
        feature_results.append(screen_numeric(name, getter, cal, hold, min_cal=min_cal, min_hold=min_hold))
    for name, getter in categorical_features().items():
        feature_results.append(screen_categorical(name, getter, cal, hold, min_cal=min_cal, min_hold=min_hold))

    candidates = [
        {"feature": fr["feature"], "bin": b["bin"], "cal": b["cal"], "hold": b["hold"]}
        for fr in feature_results for b in fr["bins"] if b.get("candidate")
    ]
    return {
        "rule_name": spec.rule_name,
        "family": spec.family,
        "filled_cal_n": len(cal),
        "filled_hold_n": len(hold),
        "split": split,
        "baseline_cal": summ([r["net_bps"] for r in cal]),
        "baseline_hold": summ([r["net_bps"] for r in hold]),
        "features": feature_results,
        "candidates": candidates,
    }


def render_md(report: dict[str, Any]) -> str:
    lines = [
        "# S34 Conditional Edge Screen",
        "",
        f"Generated: `{report['generated_at_utc']}`",
        "",
        "Knowable-at-cross feature conditioning on clean per-anchor outcomes. Tercile cuts derived on "
        "calibration only, then applied to the chronological holdout. A CANDIDATE bin is positive on "
        "BOTH splits with enough filled trades. Many features are screened -> treat any candidate as a "
        "lead to re-test as a fresh registered route, not a green light.",
        "",
    ]
    for route in report["routes"]:
        lines.append(f"## `{route['rule_name']}`  (cal filled={route['filled_cal_n']}, hold filled={route['filled_hold_n']})")
        bc, bh = route["baseline_cal"], route["baseline_hold"]
        lines.append(f"- baseline net_bps: cal median={bc['median']} mean={bc['mean']} | hold median={bh['median']} mean={bh['mean']}")
        if route["candidates"]:
            lines.append("- **CANDIDATES (positive on both splits):**")
            lines.append("")
            lines.append("| Feature | Bin | Cal n/med/mean | Hold n/med/mean |")
            lines.append("| --- | --- | --- | --- |")
            for c in route["candidates"]:
                lines.append(
                    f"| `{c['feature']}` | {c['bin']} | {c['cal']['n']}/{c['cal']['median']}/{c['cal']['mean']} | "
                    f"{c['hold']['n']}/{c['hold']['median']}/{c['hold']['mean']} |"
                )
        else:
            lines.append("- **No conditioning is positive on both calibration and holdout.**")
        lines.append("")
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Screen knowable-at-cross features for a holdout-stable positive sub-population.")
    p.add_argument("--db", type=Path, default=DEFAULT_DB)
    p.add_argument("--rules", default=",".join(DEFAULT_RULES))
    p.add_argument("--bucket-sec", type=int, default=300)
    p.add_argument("--min-gap-sec", type=int, default=900)
    p.add_argument("--accel-window-sec", type=int, default=30)
    p.add_argument("--holdout-frac", type=float, default=0.30)
    p.add_argument("--fee-bps-side", type=float, default=3.05)
    p.add_argument("--max-book-staleness-sec", type=int, default=5)
    p.add_argument("--min-cal", type=int, default=30)
    p.add_argument("--min-hold", type=int, default=15)
    p.add_argument("--json-out", type=Path, default=OUT_JSON)
    p.add_argument("--md-out", type=Path, default=OUT_MD)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    names = [n.strip() for n in str(args.rules).split(",") if n.strip()]
    specs = [ROUTE_BY_NAME[n] for n in names if n in ROUTE_BY_NAME]
    with sqlite3.connect(f"file:{args.db}?mode=ro", uri=True) as conn:
        routes = [
            screen_route(
                conn, spec,
                bucket_sec=int(args.bucket_sec), min_gap_sec=int(args.min_gap_sec),
                accel_window_sec=int(args.accel_window_sec), holdout_frac=float(args.holdout_frac),
                fee_bps_side=float(args.fee_bps_side), max_book_staleness_sec=int(args.max_book_staleness_sec),
                min_cal=int(args.min_cal), min_hold=int(args.min_hold),
            )
            for spec in specs
        ]
    report = {"generated_at_utc": utc_now(), "config": {"rules": names, "fee_bps_side": float(args.fee_bps_side)}, "routes": routes}
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    args.md_out.write_text(render_md(report), encoding="utf-8")
    print(render_md(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
