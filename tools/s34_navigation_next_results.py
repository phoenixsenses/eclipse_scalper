"""S34 navigation next results.

Research-only follow-up for the navigation bridge:

1. Route-specific map by threshold.
2. Bull thin-depth pattern anatomy.
3. KNN neighbor navigation score.
4. ETH BUY-side navigation map.

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
from tools.s34_v_engine_shadow_observer import ACCEL_WINDOW_SEC, BUCKET_SEC, MIN_GAP_SEC

DEFAULT_DB = ROOT / "data" / "microstructure.db"
NAV_EVENTS = ROOT / "reports" / "research" / "s34" / "S34_NAVIGATION_EVENTS.jsonl"
OUT_JSON = ROOT / "reports" / "research" / "s34" / "S34_NAVIGATION_NEXT_RESULTS.json"
OUT_MD = ROOT / "reports" / "research" / "s34" / "S34_NAVIGATION_NEXT_RESULTS.md"

SYMBOL = "ETHUSDT"
FEE_BPS = 5.0
MAX_HORIZON_SEC = 4 * 3600


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


def route_specific_map(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_route: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_route.setdefault(f"{row.get('symbol')}_{row.get('liq_side')}_{int(float(row.get('threshold_usd') or 0))}", []).append(row)
    out = {}
    for route, items in sorted(by_route.items()):
        out[route] = {
            "all_2h": summary([float(r["net_2h_bps"]) for r in items if r.get("net_2h_bps") is not None]),
            "tail_low_2h": summary([float(r["net_2h_bps"]) for r in items if "TAIL_LOW_CONTEXT" in (r.get("tags") or [])]),
            "vdepth_core_bid_ok_2h": summary(
                [
                    float(r["net_2h_bps"])
                    for r in items
                    if "VDEPTH_CORE" in (r.get("tags") or []) and "BID_DEPTH_OK" in (r.get("tags") or [])
                ]
            ),
            "tp300_sl150_4h": summary([float(r["net_tp300_sl150_4h_bps"]) for r in items if r.get("net_tp300_sl150_4h_bps") is not None]),
        }
    return out


def bull_thin_depth_anatomy(rows: list[dict[str, Any]]) -> dict[str, Any]:
    target = [
        r for r in rows
        if {"BULL_PULLBACK", "VDEPTH_CORE", "BID_DEPTH_THIN"}.issubset(set(r.get("tags") or []))
    ]
    def bucket(key: str, fn) -> dict[str, Any]:
        groups: dict[str, list[float]] = {}
        for row in target:
            groups.setdefault(str(fn(row)), []).append(float(row[key]))
        return {k: summary(v) for k, v in sorted(groups.items())}

    return {
        "status": "BULL_THIN_DEPTH_ANATOMY",
        "n": len(target),
        "overall_2h": summary([float(r["net_2h_bps"]) for r in target]),
        "overall_tp300_sl150_4h": summary([float(r["net_tp300_sl150_4h_bps"]) for r in target]),
        "by_threshold_2h": bucket("net_2h_bps", lambda r: f"thr{int(float(r['threshold_usd']))}"),
        "by_vdepth_subbin_2h": bucket("net_2h_bps", lambda r: "v28_32" if float(r["vdepth_bps"]) < 32 else "v32_40"),
        "by_book_imbalance_2h": bucket(
            "net_2h_bps",
            lambda r: "imb_pos" if float(r.get("book_imbalance") or 0.0) >= 0 else "imb_neg",
        ),
        "tail_rows": [
            {
                "signal_utc": r.get("signal_utc"),
                "threshold_usd": r.get("threshold_usd"),
                "vdepth_bps": r.get("vdepth_bps"),
                "bid_depth_usd": r.get("bid_depth_usd"),
                "book_imbalance": r.get("book_imbalance"),
                "net_2h_bps": r.get("net_2h_bps"),
                "net_tp300_sl150_4h_bps": r.get("net_tp300_sl150_4h_bps"),
            }
            for r in target
            if float(r.get("net_2h_bps") or 0.0) <= -100.0
        ],
        "read": "Promising but tail-bearing context; not a candidate until tail rows are explained out-of-sample.",
    }


def feature_vector(row: dict[str, Any]) -> list[float]:
    return [
        float(row.get("threshold_usd") or 0.0) / 200_000.0,
        float(row.get("vdepth_bps") or 0.0) / 40.0,
        float(row.get("prior4h_bps") or 0.0) / 200.0,
        math.log1p(max(0.0, float(row.get("bid_depth_usd") or 0.0))) / 13.0,
        float(row.get("book_imbalance") or 0.0),
        float(row.get("eth1h_bps") or 0.0) / 100.0,
        float(row.get("btc4h_bps") or 0.0) / 100.0,
    ]


def dist(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def knn_navigation(rows: list[dict[str, Any]], k: int = 20) -> dict[str, Any]:
    usable = [r for r in rows if r.get("net_2h_bps") is not None]
    vecs = [feature_vector(r) for r in usable]
    cards = []
    for i, row in enumerate(usable):
        ds = []
        for j, other in enumerate(usable):
            if i == j:
                continue
            ds.append((dist(vecs[i], vecs[j]), other))
        nn = [r for _, r in sorted(ds, key=lambda x: x[0])[:k]]
        vals = [float(r["net_2h_bps"]) for r in nn]
        s = summary(vals)
        actual = float(row["net_2h_bps"])
        if s["n"] < k:
            pred = "UNKNOWN"
        elif s["tail_lte_minus150_n"] == 0 and float(s.get("t3r_bps") or -1e9) > 0:
            pred = "CLEAN"
        elif s["tail_lte_minus150_n"] >= 2 or float(s.get("max_loss_bps") or 0.0) <= -250:
            pred = "DANGER"
        else:
            pred = "MIXED"
        cards.append({"prediction": pred, "actual_bps": actual, "neighbor_summary": s})
    by_pred: dict[str, list[float]] = {}
    for c in cards:
        by_pred.setdefault(c["prediction"], []).append(float(c["actual_bps"]))
    return {
        "status": "LEAVE_ONE_OUT_KNN",
        "k": k,
        "pred_counts": {k: len(v) for k, v in sorted(by_pred.items())},
        "actual_by_prediction": {k: summary(v) for k, v in sorted(by_pred.items())},
        "read": "Leave-one-out KNN over navigation event features. It is a map quality test, not a trade rule.",
    }


def mark_exit(path: tuple[tuple[int, float], ...], entry: float, horizon_sec: int) -> tuple[int, float] | None:
    if not path:
        return None
    target = int(path[0][0]) + int(horizon_sec) * 1000
    last = None
    for ts, px in path:
        if int(ts) <= target:
            last = (int(ts), float(px))
        else:
            break
    return last or (int(path[-1][0]), float(path[-1][1]))


def collect_buy_side_map(conn: sqlite3.Connection) -> dict[str, Any]:
    marks = load_mark_index(conn, SYMBOL)
    rows = []
    for threshold in (50_000.0, 100_000.0, 200_000.0, 500_000.0):
        events = collect_events(
            conn,
            symbol=SYMBOL,
            threshold=threshold,
            sides=("BUY",),
            min_vdepth_bps=5.0,
            bucket_sec=BUCKET_SEC,
            min_gap_sec=MIN_GAP_SEC,
            accel_window_sec=ACCEL_WINDOW_SEC,
            max_horizon_sec=MAX_HORIZON_SEC,
        )
        for ev in events:
            ts = int(ev.anchor.anchor_ts_ms)
            entry = float(ev.anchor_mark_price)
            exit_2h = mark_exit(ev.path, entry, 2 * 3600)
            if not exit_2h:
                continue
            # BUY-liq fade means SHORT after an upside liquidation shock.
            fade_short = signed_return_bps("SHORT", entry, float(exit_2h[1])) - FEE_BPS
            continuation_long = signed_return_bps("LONG", entry, float(exit_2h[1])) - FEE_BPS
            prior4h = prior_return_bps(marks, ts, 4 * 3600)
            rows.append(
                {
                    "threshold": threshold,
                    "signal_utc": iso_ms(ts),
                    "vdepth_bps": r1(ev.vdepth_bps),
                    "prior4h_bps": r1(prior4h),
                    "fade_short_2h_bps": r1(fade_short),
                    "continuation_long_2h_bps": r1(continuation_long),
                }
            )
    by_thr_short: dict[str, list[float]] = {}
    by_thr_long: dict[str, list[float]] = {}
    for row in rows:
        key = f"thr{int(row['threshold'])}"
        by_thr_short.setdefault(key, []).append(float(row["fade_short_2h_bps"]))
        by_thr_long.setdefault(key, []).append(float(row["continuation_long_2h_bps"]))
    return {
        "status": "ETH_BUY_SIDE_NAVIGATION",
        "rows_n": len(rows),
        "fade_short_by_threshold": {k: summary(v) for k, v in sorted(by_thr_short.items())},
        "continuation_long_by_threshold": {k: summary(v) for k, v in sorted(by_thr_long.items())},
        "read": "BUY-side map reports both reversal SHORT and continuation LONG labels. Mark labels only.",
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    rows = load_jsonl(args.nav_events)
    with sqlite3.connect(f"file:{args.db}?mode=ro", uri=True) as conn:
        buy = collect_buy_side_map(conn)
    return {
        "generated_at_utc": utc_now(),
        "status": "RESEARCH_ONLY_NO_LIVE_CHANGE",
        "route_specific_map": route_specific_map(rows),
        "bull_thin_depth_anatomy": bull_thin_depth_anatomy(rows),
        "knn_navigation": knn_navigation(rows, k=int(args.knn_k)),
        "buy_side_navigation": buy,
        "read": "Navigation tests only. No live order/config changes.",
    }


def render_md(report: dict[str, Any]) -> str:
    lines = [
        "# S34 Navigation Next Results",
        "",
        f"Generated: `{report['generated_at_utc']}`",
        "",
        f"Status: `{report['status']}`. {report['read']}",
        "",
        "## Route-Specific Map",
        "",
        "| Route | All 2h N | All Sum | TailLow N | TailLow Sum | TailLow T3R | TailLow <=150 | 4hTP Sum |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for route, row in report["route_specific_map"].items():
        all2 = row["all_2h"]
        tl = row["tail_low_2h"]
        tp = row["tp300_sl150_4h"]
        lines.append(
            f"| {route} | {all2['n']} | {all2['sum_bps']} | {tl['n']} | {tl['sum_bps']} | {tl['t3r_bps']} | {tl['tail_lte_minus150_n']} | {tp['sum_bps']} |"
        )
    b = report["bull_thin_depth_anatomy"]
    lines.extend([
        "",
        "## Bull Thin-Depth Anatomy",
        "",
        f"- N: `{b['n']}`",
        f"- 2h: N={b['overall_2h']['n']} sum={b['overall_2h']['sum_bps']} med={b['overall_2h']['median_bps']} T3R={b['overall_2h']['t3r_bps']} tails<=150={b['overall_2h']['tail_lte_minus150_n']}",
        f"- TP300/SL150/4h: sum={b['overall_tp300_sl150_4h']['sum_bps']} med={b['overall_tp300_sl150_4h']['median_bps']} T3R={b['overall_tp300_sl150_4h']['t3r_bps']}",
        "",
        "| Threshold bucket | N | Sum | Median | Tail<=150 | T3R |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ])
    for key, row in b["by_threshold_2h"].items():
        lines.append(f"| {key} | {row['n']} | {row['sum_bps']} | {row['median_bps']} | {row['tail_lte_minus150_n']} | {row['t3r_bps']} |")
    lines.extend([
        "",
        "## KNN Navigation",
        "",
        f"- k: `{report['knn_navigation']['k']}`",
        f"- prediction counts: `{report['knn_navigation']['pred_counts']}`",
        "",
        "| Prediction | N | Sum | Median | Win | Tail<=150 | Max loss | T3R |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for pred, row in report["knn_navigation"]["actual_by_prediction"].items():
        lines.append(f"| {pred} | {row['n']} | {row['sum_bps']} | {row['median_bps']} | {row['win_rate']} | {row['tail_lte_minus150_n']} | {row['max_loss_bps']} | {row['t3r_bps']} |")
    lines.extend([
        "",
        "## BUY-Side Navigation",
        "",
        "| Side/Threshold | N | Sum | Median | Win | Tail<=150 | Max loss | T3R |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for key, row in report["buy_side_navigation"]["fade_short_by_threshold"].items():
        lines.append(f"| fade_SHORT_{key} | {row['n']} | {row['sum_bps']} | {row['median_bps']} | {row['win_rate']} | {row['tail_lte_minus150_n']} | {row['max_loss_bps']} | {row['t3r_bps']} |")
    for key, row in report["buy_side_navigation"]["continuation_long_by_threshold"].items():
        lines.append(f"| continuation_LONG_{key} | {row['n']} | {row['sum_bps']} | {row['median_bps']} | {row['win_rate']} | {row['tail_lte_minus150_n']} | {row['max_loss_bps']} | {row['t3r_bps']} |")
    lines.append("")
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run S34 navigation next tests.")
    p.add_argument("--db", type=Path, default=DEFAULT_DB)
    p.add_argument("--nav-events", type=Path, default=NAV_EVENTS)
    p.add_argument("--knn-k", type=int, default=20)
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
