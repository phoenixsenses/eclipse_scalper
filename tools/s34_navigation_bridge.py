"""S34 Navigation Bridge.

Uses historical S34-style events as a map, not as a trade trigger. It tags each
event with market-state labels, checks whether those labels separate tail risk,
compares 2h vs 4h/TP exits, and surfaces pattern candidates for shadow-only
follow-up.

Research/navigation only. No live executor, order logic, size, leverage, or .env
changes.
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
OUT_JSON = ROOT / "reports" / "research" / "s34" / "S34_NAVIGATION_BRIDGE.json"
OUT_MD = ROOT / "reports" / "research" / "s34" / "S34_NAVIGATION_BRIDGE.md"
OUT_JSONL = ROOT / "reports" / "research" / "s34" / "S34_NAVIGATION_EVENTS.jsonl"

SYMBOL = "ETHUSDT"
SIDE = "SELL"
THRESHOLDS = (50_000.0, 100_000.0, 200_000.0)
MAX_HORIZON_SEC = 4 * 3600
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


def mark_exit(path: tuple[tuple[int, float], ...], entry: float, horizon_sec: int) -> tuple[int, float] | None:
    target = int(path[0][0]) + int(horizon_sec) * 1000 if path else None
    if target is None:
        return None
    last = None
    for ts, px in path:
        if int(ts) <= target:
            last = (int(ts), float(px))
        else:
            break
    return last or (int(path[-1][0]), float(path[-1][1]))


def tp_sl_4h(path: tuple[tuple[int, float], ...], entry: float) -> tuple[str, int, float] | None:
    tp = float(entry) * 1.03
    sl = float(entry) * 0.985
    end_ts = int(path[0][0]) + 4 * 3600 * 1000 if path else 0
    last = None
    for ts, px in path:
        if int(ts) > end_ts:
            break
        last = (int(ts), float(px))
        if float(px) <= sl:
            return ("SL", int(ts), sl)
        if float(px) >= tp:
            return ("TP", int(ts), tp)
    if last:
        return ("TIME", int(last[0]), float(last[1]))
    return None


def net_bps(entry: float, exit_px: float) -> float:
    return (float(exit_px) - float(entry)) / float(entry) * 10_000.0 - FEE_BPS


def event_tags(row: dict[str, Any]) -> list[str]:
    tags: list[str] = []
    v = float(row["vdepth_bps"])
    prior = float(row["prior4h_bps"])
    bid = float(row.get("bid_depth_usd") or 0.0)
    btc4h = row.get("btc4h_bps")
    eth1h = row.get("eth1h_bps")
    if prior < -50.0:
        tags.append("RISK_OFF_REBOUND")
    elif eth1h is not None and btc4h is not None and float(eth1h) > 20.0 and float(btc4h) > 50.0:
        tags.append("BULL_PULLBACK")
    else:
        tags.append("NEUTRAL_CONTEXT")
    if 28.0 <= v < 40.0:
        tags.append("VDEPTH_CORE")
    elif v < 28.0:
        tags.append("VDEPTH_DANGER_LOW")
    else:
        tags.append("VDEPTH_DANGER_HIGH")
    if bid >= 135_423.8:
        tags.append("BID_DEPTH_OK")
    else:
        tags.append("BID_DEPTH_THIN")
    if 135_423.8 <= bid < 200_000.0:
        tags.append("BID_DEPTH_CORE")
    if bid >= 400_000.0:
        tags.append("BID_DEPTH_HEAVY")
    if float(row.get("net_2h_bps") or 0.0) <= -100.0:
        tags.append("TAIL_REALIZED")
    if float(row.get("net_4h_bps") or 0.0) > float(row.get("net_2h_bps") or 0.0):
        tags.append("EXIT_4H_ACTUAL_BETTER")
    else:
        tags.append("EXIT_2H_ACTUAL_BETTER")
    if set(["RISK_OFF_REBOUND", "VDEPTH_CORE", "BID_DEPTH_OK"]).issubset(tags):
        tags.append("TAIL_LOW_CONTEXT")
        tags.append("SIZE_15X_STABLE")
    else:
        tags.append("TAIL_HIGH_OR_UNKNOWN")
        tags.append("SIZE_34X_FRAGILE")
    return tags


def collect_navigation_events(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    eth_marks = load_mark_index(conn, SYMBOL)
    btc_marks = load_mark_index(conn, "BTCUSDT")
    rows: list[dict[str, Any]] = []
    seen: set[tuple[float, int]] = set()
    for threshold in THRESHOLDS:
        events = collect_events(
            conn,
            symbol=SYMBOL,
            threshold=threshold,
            sides=(SIDE,),
            min_vdepth_bps=5.0,
            bucket_sec=BUCKET_SEC,
            min_gap_sec=MIN_GAP_SEC,
            accel_window_sec=ACCEL_WINDOW_SEC,
            max_horizon_sec=MAX_HORIZON_SEC,
        )
        for ev in events:
            key = (float(threshold), int(ev.anchor.anchor_ts_ms))
            if key in seen:
                continue
            seen.add(key)
            entry = float(ev.anchor_mark_price)
            exit_2h = mark_exit(ev.path, entry, 2 * 3600)
            exit_4h = mark_exit(ev.path, entry, 4 * 3600)
            tp4h = tp_sl_4h(ev.path, entry)
            if not exit_2h or not exit_4h or not tp4h:
                continue
            ts = int(ev.anchor.anchor_ts_ms)
            prior4h = prior_return_bps(eth_marks, ts, 4 * 3600)
            if prior4h is None:
                continue
            book = book_features_at(conn, SYMBOL, ts, 5)
            bid = float(book.get("bid_depth_usd") or 0.0) if book else 0.0
            btc4h = btc_marks.ret_bps(ts - 4 * 3600 * 1000, ts)
            eth1h = eth_marks.ret_bps(ts - 3600 * 1000, ts)
            row = {
                "event_id": f"{SYMBOL}_{SIDE}_{int(threshold)}_{ts}",
                "symbol": SYMBOL,
                "liq_side": SIDE,
                "direction": "LONG",
                "threshold_usd": float(threshold),
                "signal_ts_ms": ts,
                "signal_utc": iso_ms(ts),
                "vdepth_bps": r1(ev.vdepth_bps),
                "prior4h_bps": r1(prior4h),
                "eth1h_bps": r1(eth1h),
                "btc4h_bps": r1(btc4h),
                "bid_depth_usd": r1(bid),
                "book_imbalance": r3(float(book.get("book_imbalance") or 0.0)) if book else None,
                "net_2h_bps": r1(net_bps(entry, exit_2h[1])),
                "net_4h_bps": r1(net_bps(entry, exit_4h[1])),
                "net_tp300_sl150_4h_bps": r1(net_bps(entry, tp4h[2])),
                "tp300_sl150_4h_exit": tp4h[0],
            }
            row["tags"] = event_tags(row)
            row["tag_combo"] = "+".join(t for t in row["tags"] if t in {"RISK_OFF_REBOUND", "BULL_PULLBACK", "NEUTRAL_CONTEXT", "VDEPTH_CORE", "VDEPTH_DANGER_LOW", "VDEPTH_DANGER_HIGH", "BID_DEPTH_OK", "BID_DEPTH_THIN", "BID_DEPTH_CORE", "BID_DEPTH_HEAVY"})
            rows.append(row)
    rows.sort(key=lambda r: (int(r["signal_ts_ms"]), float(r["threshold_usd"])))
    return rows


def group_summary(rows: list[dict[str, Any]], key_fn, value_key: str = "net_2h_bps") -> dict[str, Any]:
    groups: dict[str, list[float]] = {}
    for row in rows:
        key = str(key_fn(row))
        if row.get(value_key) is not None:
            groups.setdefault(key, []).append(float(row[value_key]))
    return {k: summary(v) for k, v in sorted(groups.items())}


def tag_distribution(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        for tag in row["tags"]:
            counts[tag] = int(counts.get(tag, 0)) + 1
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def exit_preference_validation(rows: list[dict[str, Any]]) -> dict[str, Any]:
    actual = {
        "EXIT_4H_ACTUAL_BETTER": [float(r["net_4h_bps"]) - float(r["net_2h_bps"]) for r in rows if "EXIT_4H_ACTUAL_BETTER" in r["tags"]],
        "EXIT_2H_ACTUAL_BETTER": [float(r["net_4h_bps"]) - float(r["net_2h_bps"]) for r in rows if "EXIT_2H_ACTUAL_BETTER" in r["tags"]],
    }
    by_exit = {
        "2h": summary([float(r["net_2h_bps"]) for r in rows]),
        "4h": summary([float(r["net_4h_bps"]) for r in rows]),
        "tp300_sl150_4h": summary([float(r["net_tp300_sl150_4h_bps"]) for r in rows]),
    }
    return {
        "actual_preference_counts": {k: len(v) for k, v in actual.items()},
        "actual_4h_minus_2h_delta": {k: summary(v) for k, v in actual.items()},
        "by_exit": by_exit,
    }


def pattern_candidates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row["tag_combo"]), []).append(row)
    out = []
    for combo, items in groups.items():
        vals = [float(r["net_2h_bps"]) for r in items]
        vals4 = [float(r["net_tp300_sl150_4h_bps"]) for r in items]
        s2 = summary(vals)
        s4 = summary(vals4)
        if s2["n"] < 5:
            continue
        out.append(
            {
                "combo": combo,
                "n": s2["n"],
                "net_2h": s2,
                "tp300_sl150_4h": s4,
                "pattern_verdict": (
                    "SHADOW_LEAD"
                    if s2["t3r_bps"] > 0
                    and s4["t3r_bps"] > 0
                    and s2["tail_lte_minus150_n"] == 0
                    and s2["n"] >= 10
                    else "CONTEXT_ONLY"
                ),
            }
        )
    out.sort(key=lambda r: (float(r["tp300_sl150_4h"].get("t3r_bps") or -1e9), float(r["net_2h"].get("t3r_bps") or -1e9)), reverse=True)
    return out


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    with sqlite3.connect(f"file:{args.db}?mode=ro", uri=True) as conn:
        rows = collect_navigation_events(conn)
    report = {
        "generated_at_utc": utc_now(),
        "status": "NAVIGATION_RESEARCH_ONLY_NO_LIVE_CHANGE",
        "definition": "ETH SELL liquidation navigation universe: thresholds 50K/100K/200K, vdepth>=5, deterministic tags, mark-based 2h/4h labels.",
        "events_n": len(rows),
        "tag_distribution": tag_distribution(rows),
        "tail_low_validation": {
            "TAIL_LOW_CONTEXT": summary([float(r["net_2h_bps"]) for r in rows if "TAIL_LOW_CONTEXT" in r["tags"]]),
            "TAIL_HIGH_OR_UNKNOWN": summary([float(r["net_2h_bps"]) for r in rows if "TAIL_HIGH_OR_UNKNOWN" in r["tags"]]),
        },
        "tail_low_by_threshold": group_summary(
            [r for r in rows if "TAIL_LOW_CONTEXT" in r["tags"]],
            lambda r: f"TAIL_LOW_thr{int(float(r['threshold_usd']))}",
        ),
        "exact_v02_route": {
            "2h": summary(
                [
                    float(r["net_2h_bps"])
                    for r in rows
                    if float(r["threshold_usd"]) == 200_000.0 and "TAIL_LOW_CONTEXT" in r["tags"]
                ]
            ),
            "4h": summary(
                [
                    float(r["net_4h_bps"])
                    for r in rows
                    if float(r["threshold_usd"]) == 200_000.0 and "TAIL_LOW_CONTEXT" in r["tags"]
                ]
            ),
            "tp300_sl150_4h": summary(
                [
                    float(r["net_tp300_sl150_4h_bps"])
                    for r in rows
                    if float(r["threshold_usd"]) == 200_000.0 and "TAIL_LOW_CONTEXT" in r["tags"]
                ]
            ),
            "read": "Exact navigation approximation of current v0.2 route: threshold 200K plus RISK_OFF_REBOUND + VDEPTH_CORE + BID_DEPTH_OK. Mark-label, not maker lifecycle.",
        },
        "exit_preference_validation": exit_preference_validation(rows),
        "pattern_candidates": pattern_candidates(rows),
        "events": rows,
        "events_sample": rows[-50:],
        "read": "Navigation bridge labels context only. It does not authorize trades.",
    }
    return report


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n")


def render_md(report: dict[str, Any]) -> str:
    lines = [
        "# S34 Navigation Bridge",
        "",
        f"Generated: `{report['generated_at_utc']}`",
        "",
        f"Status: `{report['status']}`. {report['read']}",
        "",
        f"Definition: {report['definition']}",
        "",
        f"Events: `{report['events_n']}`",
        "",
        "## Tag Distribution",
        "",
        "| Tag | Count |",
        "| --- | ---: |",
    ]
    for tag, count in report["tag_distribution"].items():
        lines.append(f"| {tag} | {count} |")
    lines.extend([
        "",
        "## Tail-Low Validation",
        "",
        "| Bucket | N | Sum | Median | Win | <=-100 | <=-150 | <=-300 | Max loss | T3R |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for bucket, row in report["tail_low_validation"].items():
        lines.append(
            f"| {bucket} | {row['n']} | {row['sum_bps']} | {row['median_bps']} | {row['win_rate']} | "
            f"{row['tail_lte_minus100_n']} | {row['tail_lte_minus150_n']} | {row['tail_lte_minus300_n']} | {row['max_loss_bps']} | {row['t3r_bps']} |"
        )
    lines.extend([
        "",
        "## Tail-Low By Threshold",
        "",
        "| Bucket | N | Sum | Median | Win | <=-100 | <=-150 | <=-300 | Max loss | T3R |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for bucket, row in report["tail_low_by_threshold"].items():
        lines.append(
            f"| {bucket} | {row['n']} | {row['sum_bps']} | {row['median_bps']} | {row['win_rate']} | "
            f"{row['tail_lte_minus100_n']} | {row['tail_lte_minus150_n']} | {row['tail_lte_minus300_n']} | {row['max_loss_bps']} | {row['t3r_bps']} |"
        )
    lines.extend([
        "",
        "## Exact v0.2 Route Approximation",
        "",
        "| Exit | N | Sum | Median | Win | <=-100 | <=-150 | <=-300 | Max loss | T3R |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for label, row in report["exact_v02_route"].items():
        if not isinstance(row, dict) or "n" not in row:
            continue
        lines.append(
            f"| {label} | {row['n']} | {row['sum_bps']} | {row['median_bps']} | {row['win_rate']} | "
            f"{row['tail_lte_minus100_n']} | {row['tail_lte_minus150_n']} | {row['tail_lte_minus300_n']} | {row['max_loss_bps']} | {row['t3r_bps']} |"
        )
    lines.extend([
        "",
        "## Exit Preference",
        "",
        "| Exit | N | Sum | Median | Win | Max loss | T3R |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for label, row in report["exit_preference_validation"]["by_exit"].items():
        lines.append(f"| {label} | {row['n']} | {row['sum_bps']} | {row['median_bps']} | {row['win_rate']} | {row['max_loss_bps']} | {row['t3r_bps']} |")
    lines.extend([
        "",
        "## Pattern Candidates",
        "",
        "| Verdict | Combo | N | 2h Sum | 2h T3R | 4hTP Sum | 4hTP T3R | Tail<=150 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in report["pattern_candidates"][:20]:
        s2 = row["net_2h"]
        s4 = row["tp300_sl150_4h"]
        lines.append(
            f"| {row['pattern_verdict']} | {row['combo']} | {row['n']} | {s2['sum_bps']} | {s2['t3r_bps']} | {s4['sum_bps']} | {s4['t3r_bps']} | {s2['tail_lte_minus150_n']} |"
        )
    lines.append("")
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build S34 navigation bridge report.")
    p.add_argument("--db", type=Path, default=DEFAULT_DB)
    p.add_argument("--out-json", type=Path, default=OUT_JSON)
    p.add_argument("--out-md", type=Path, default=OUT_MD)
    p.add_argument("--out-jsonl", type=Path, default=OUT_JSONL)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(args)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    write_jsonl(args.out_jsonl, report.get("events") or [])
    md = render_md(report)
    args.out_md.write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
