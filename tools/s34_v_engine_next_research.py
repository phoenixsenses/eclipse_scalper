"""Next-step S34 V Engine research bundle.

Research-only bundle for the roadmap after the portfolio map:

1. BTC BUY-liq -> maker SHORT weak-lead anatomy.
2. ETH SELL-liq threshold redundancy across 150K/200K/300K.
3. ETH core pattern/state feature screen using the failure-anatomy layer.

No live, paper, or executor state is read or modified.
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

from tools.research_s34_knowable_anchor_continuation import file_fingerprint, load_mark_index, r1, r3, sha256_text
from tools.research_s34_maker_fade import NO_TP_OR_SL, collect_events, simulate_event, summarize
from tools.s34_v_engine_failure_anatomy import build_anatomy_rows, finite_float, group_by, load_ledger
from tools.s34_v_engine_shadow_observer import DEFAULT_LEDGER_JSONL


DEFAULT_DB = ROOT / "data" / "microstructure.db"
OUT_DIR = ROOT / "reports" / "research" / "s34"
OUT_JSON = OUT_DIR / "S34_V_ENGINE_NEXT_RESEARCH.json"
OUT_MD = OUT_DIR / "S34_V_ENGINE_NEXT_RESEARCH.md"

BUCKET_SEC = 300
MIN_GAP_SEC = 900
ACCEL_WINDOW_SEC = 30
OFFSET_BPS = 20.0
CROSS_MARGIN_BPS = 1.0
MAKER_FEE_BPS = 2.0
TAKER_FEE_BPS = 3.05
MAX_BOOK_STALENESS_SEC = 10


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def prior_return_bps(marks: Any, ts_ms: int, window_sec: int) -> float | None:
    return marks.ret_bps(int(ts_ms) - int(window_sec) * 1000, int(ts_ms))


def split_rows(rows: list[dict[str, Any]], holdout_frac: float) -> dict[str, Any]:
    buckets = sorted({int(r["bucket"]) for r in rows})
    holdout_n = max(1, int(round(len(buckets) * float(holdout_frac)))) if buckets else 0
    holdout = set(buckets[-holdout_n:]) if holdout_n else set()
    for row in rows:
        row["split"] = "holdout" if int(row["bucket"]) in holdout else "calibration"
    return {
        "method": "chronological_bucket_tail_holdout",
        "holdout_frac": float(holdout_frac),
        "bucket_n": len(buckets),
        "holdout_bucket_n": len(holdout),
        "holdout_bucket_ids_sha256": sha256_text("\n".join(str(x) for x in sorted(holdout))),
    }


def split_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out = {}
    for split in ("calibration", "holdout", "overall"):
        subset = rows if split == "overall" else [r for r in rows if r.get("split") == split]
        out[split] = summarize([float(r["net_bps"]) for r in subset if finite_float(r.get("net_bps")) is not None])
    return out


def sim_rows(
    conn: sqlite3.Connection,
    events: list[Any],
    *,
    horizon_hr: float,
) -> list[dict[str, Any]]:
    rows = []
    for ev in events:
        row = simulate_event(
            conn,
            ev,
            offset_bps=OFFSET_BPS,
            cross_margin_bps=CROSS_MARGIN_BPS,
            horizon_sec=int(float(horizon_hr) * 3600),
            maker_fee_bps=MAKER_FEE_BPS,
            taker_fee_bps=TAKER_FEE_BPS,
            max_book_staleness_sec=MAX_BOOK_STALENESS_SEC,
            horizon_from="fill",
            tp_bps=NO_TP_OR_SL,
            sl_bps=NO_TP_OR_SL,
        )
        if row.get("status") == "FILLED" and row.get("net_bps") is not None:
            rows.append(row)
    return rows


def btc_weak_lead(conn: sqlite3.Connection, *, holdout_frac: float) -> dict[str, Any]:
    symbol = "BTCUSDT"
    marks = load_mark_index(conn, symbol)
    events_all = collect_events(
        conn,
        symbol=symbol,
        threshold=250_000.0,
        sides=("BUY",),
        min_vdepth_bps=28.0,
        bucket_sec=BUCKET_SEC,
        min_gap_sec=MIN_GAP_SEC,
        accel_window_sec=ACCEL_WINDOW_SEC,
        max_horizon_sec=4 * 3600,
    )
    events = [ev for ev in events_all if 28.0 <= float(ev.vdepth_bps) < 40.0]
    screens = []
    for prior_th in (0.0, 50.0, 100.0, 150.0, 200.0):
        filt = []
        for ev in events:
            prior4h = prior_return_bps(marks, int(ev.anchor.anchor_ts_ms), 4 * 3600)
            if prior4h is None or not math.isfinite(float(prior4h)):
                continue
            if float(prior4h) <= float(prior_th):
                continue
            filt.append(ev)
        for horizon_hr in (1.0, 2.0, 4.0):
            rows = sim_rows(conn, filt, horizon_hr=horizon_hr)
            split = split_rows(rows, holdout_frac)
            screens.append(
                {
                    "route_id": f"BTC_BUY_FADE_SHORT_T250K_V28_40_P4GT{int(prior_th)}_H{horizon_hr:g}",
                    "prior4h_gt_bps": float(prior_th),
                    "horizon_hr": float(horizon_hr),
                    "eligible_event_n": len(filt),
                    "filled_n": len(rows),
                    "split": split,
                    **split_summary(rows),
                }
            )
    screens.sort(
        key=lambda r: (
            float(r["holdout"]["top3_winner_removed_sum_bps"] or -1e18),
            float(r["overall"]["top3_winner_removed_sum_bps"] or -1e18),
            float(r["overall"]["sum_bps"] or -1e18),
        ),
        reverse=True,
    )
    return {"event_n": len(events), "screens": screens[:20]}


def route_events(
    conn: sqlite3.Connection,
    *,
    threshold: float,
    horizon_hr: float,
) -> list[dict[str, Any]]:
    marks = load_mark_index(conn, "ETHUSDT")
    events = collect_events(
        conn,
        symbol="ETHUSDT",
        threshold=float(threshold),
        sides=("SELL",),
        min_vdepth_bps=28.0,
        bucket_sec=BUCKET_SEC,
        min_gap_sec=MIN_GAP_SEC,
        accel_window_sec=ACCEL_WINDOW_SEC,
        max_horizon_sec=int(float(horizon_hr) * 3600),
    )
    out = []
    for ev in events:
        if not (28.0 <= float(ev.vdepth_bps) < 40.0):
            continue
        prior4h = prior_return_bps(marks, int(ev.anchor.anchor_ts_ms), 4 * 3600)
        if prior4h is None or not math.isfinite(float(prior4h)) or float(prior4h) >= -50.0:
            continue
        row = sim_rows(conn, [ev], horizon_hr=horizon_hr)
        net = row[0]["net_bps"] if row else None
        out.append(
            {
                "threshold": float(threshold),
                "route_id": f"ETH_SELL_FADE_LONG_T{int(threshold/1000)}K_V28_40_H{horizon_hr:g}",
                "bucket": int(ev.anchor.bucket),
                "first_ts_ms": int(ev.anchor.first_ts_ms),
                "anchor_ts_ms": int(ev.anchor.anchor_ts_ms),
                "cluster_key": f"{int(ev.anchor.bucket)}:{int(ev.anchor.first_ts_ms)}",
                "vdepth_bps": r1(ev.vdepth_bps),
                "prior4h_bps": r1(prior4h),
                "filled": bool(row),
                "net_bps": r1(net),
            }
        )
    return out


def eth_threshold_redundancy(conn: sqlite3.Connection) -> dict[str, Any]:
    thresholds = (150_000.0, 200_000.0, 300_000.0)
    routes = {int(t): route_events(conn, threshold=t, horizon_hr=2.0) for t in thresholds}
    by_threshold = {}
    for t, rows in routes.items():
        filled = [r for r in rows if r["filled"] and finite_float(r.get("net_bps")) is not None]
        by_threshold[str(t)] = {
            "event_n": len(rows),
            "filled_n": len(filled),
            "unique_cluster_n": len({r["cluster_key"] for r in rows}),
            "summary": summarize([float(r["net_bps"]) for r in filled]),
        }
    pairs = []
    for a, b in ((150000, 200000), (200000, 300000), (150000, 300000)):
        a_keys = {r["cluster_key"] for r in routes[a]}
        b_keys = {r["cluster_key"] for r in routes[b]}
        inter = a_keys & b_keys
        union = a_keys | b_keys
        pairs.append(
            {
                "a": a,
                "b": b,
                "a_cluster_n": len(a_keys),
                "b_cluster_n": len(b_keys),
                "shared_cluster_n": len(inter),
                "jaccard": r3(len(inter) / len(union)) if union else None,
                "b_subset_of_a": r3(len(inter) / len(b_keys)) if b_keys else None,
            }
        )
    shared_all = set.intersection(*(set(r["cluster_key"] for r in rows) for rows in routes.values())) if routes else set()
    return {
        "by_threshold": by_threshold,
        "pairs": pairs,
        "shared_all_cluster_n": len(shared_all),
        "read": "High b_subset_of_a means higher thresholds are mostly the same cascades, not independent frequency.",
    }


def condition_screen(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    return [
        {
            "feature": key,
            "value": row["value"],
            "n": row["n"],
            "loser_rate": row["loser_rate"],
            "summary": row["summary"],
        }
        for row in group_by(rows, key)
        if int(row["n"]) >= 2
    ]


def eth_pattern_layer(conn: sqlite3.Connection, *, ledger_path: Path) -> dict[str, Any]:
    ledger = load_ledger(ledger_path)
    eth_marks = load_mark_index(conn, "ETHUSDT")
    btc_marks = load_mark_index(conn, "BTCUSDT")
    rows = build_anatomy_rows(ledger, eth_marks=eth_marks, btc_marks=btc_marks, rebreak_bps=10.0)
    features = [
        "anchor_reclaimed_15m",
        "first_15m_bucket",
        "btc_context_bucket",
        "candle15_pattern",
        "low_rebreak_15m",
        "prior4h_intensity_bucket",
    ]
    screens = []
    for key in features:
        screens.extend(condition_screen(rows, key))
    screens.sort(
        key=lambda r: (
            float(r["summary"]["top3_winner_removed_sum_bps"] or -1e18),
            int(r["n"]),
            float(r["summary"]["sum_bps"] or -1e18),
        ),
        reverse=True,
    )
    return {
        "closed_filled_n": len(rows),
        "overall": summarize([float(r["net_bps"]) for r in rows if finite_float(r.get("net_bps")) is not None]),
        "top_feature_states": screens[:20],
    }


def cell(s: dict[str, Any]) -> str:
    return f"N={s['n']} sum={s['sum_bps']} med={s['median_bps']} T3R={s['top3_winner_removed_sum_bps']}"


def render_md(payload: dict[str, Any]) -> str:
    lines = [
        "# S34 V Engine Next Research",
        "",
        f"Generated: `{payload['generated_at_utc']}`",
        "",
        "Research-only continuation of the portfolio roadmap. No live/paper state changed.",
        "",
        "## 1. BTC BUY -> Maker SHORT Weak Lead",
        "",
        "| Rank | Route | Eligible | Filled | Cal | Hold | Overall |",
        "| ---: | --- | ---: | ---: | --- | --- | --- |",
    ]
    for i, row in enumerate(payload["btc_weak_lead"]["screens"][:10], start=1):
        lines.append(
            f"| {i} | `{row['route_id']}` | {row['eligible_event_n']} | {row['filled_n']} | "
            f"{cell(row['calibration'])} | {cell(row['holdout'])} | {cell(row['overall'])} |"
        )
    lines.extend(["", "## 2. ETH Threshold Redundancy", ""])
    lines.append("| Threshold | Events | Filled | Unique clusters | Summary |")
    lines.append("| ---: | ---: | ---: | ---: | --- |")
    for threshold, row in payload["eth_threshold_redundancy"]["by_threshold"].items():
        lines.append(
            f"| {int(threshold)//1000}K | {row['event_n']} | {row['filled_n']} | {row['unique_cluster_n']} | {cell(row['summary'])} |"
        )
    lines.extend(["", "| Pair | Shared clusters | Jaccard | Higher subset of lower |", "| --- | ---: | ---: | ---: |"])
    for row in payload["eth_threshold_redundancy"]["pairs"]:
        lines.append(f"| {row['a']//1000}K vs {row['b']//1000}K | {row['shared_cluster_n']} | {row['jaccard']} | {row['b_subset_of_a']} |")
    lines.append(f"\nShared by all three: `{payload['eth_threshold_redundancy']['shared_all_cluster_n']}` clusters.")
    lines.extend(["", "## 3. ETH Pattern / State Layer", ""])
    lines.append(f"Overall core filled sample: {cell(payload['eth_pattern_layer']['overall'])}")
    lines.append("")
    lines.append("| Rank | Feature | Value | N | Loser% | Summary |")
    lines.append("| ---: | --- | --- | ---: | ---: | --- |")
    for i, row in enumerate(payload["eth_pattern_layer"]["top_feature_states"][:15], start=1):
        loser_pct = None if row["loser_rate"] is None else r1(float(row["loser_rate"]) * 100.0)
        lines.append(f"| {i} | `{row['feature']}` | `{row['value']}` | {row['n']} | {loser_pct} | {cell(row['summary'])} |")
    lines.extend(["", "## Read", ""])
    lines.append("- If BTC rows only work in holdout or fail T3R, keep them as observation lane, not candidate.")
    lines.append("- If 150/200/300K share the same clusters, they should be a threshold-response curve, not separate portfolio engines.")
    lines.append("- Pattern states are confirmation/permission candidates only; any waited entry must pay price deterioration in a separate test.")
    lines.append("")
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run next S34 V Engine research bundle.")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--ledger-jsonl", type=Path, default=DEFAULT_LEDGER_JSONL)
    parser.add_argument("--holdout-frac", type=float, default=0.30)
    parser.add_argument("--json-out", type=Path, default=OUT_JSON)
    parser.add_argument("--md-out", type=Path, default=OUT_MD)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    with sqlite3.connect(f"file:{args.db}?mode=ro", uri=True) as conn:
        payload = {
            "generated_at_utc": utc_now(),
            "source_db": file_fingerprint(args.db),
            "config": {
                "offset_bps": OFFSET_BPS,
                "cross_margin_bps": CROSS_MARGIN_BPS,
                "maker_fee_bps": MAKER_FEE_BPS,
                "taker_fee_bps": TAKER_FEE_BPS,
                "holdout_frac": float(args.holdout_frac),
            },
            "btc_weak_lead": btc_weak_lead(conn, holdout_frac=float(args.holdout_frac)),
            "eth_threshold_redundancy": eth_threshold_redundancy(conn),
            "eth_pattern_layer": eth_pattern_layer(conn, ledger_path=args.ledger_jsonl),
        }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    args.md_out.write_text(render_md(payload), encoding="utf-8")
    print(render_md(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
