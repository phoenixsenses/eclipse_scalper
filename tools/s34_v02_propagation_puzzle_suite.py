"""S34 V02 propagation puzzle suite.

Research-only. Tests whether the strongly negative same-side transition cells
are actually a propagation/momentum state, and whether silence/reclaim is the
fade state. No live/paper/executor state is changed.
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
from bisect import bisect_left
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.research_s34_knowable_anchor_continuation import iso_ms, pctile, r1, r3, signed_return_bps  # noqa: E402
from tools.s34_v02_event_chain_puzzle_tests import (  # noqa: E402
    ASSET_THRESHOLDS,
    ETH_THRESHOLD,
    FEE_BPS,
    H4_LEDGER_JSONL,
    build_events,
    load_jsonl,
    mark_price_at,
    metrics,
    neighbor_events,
    raw_ret,
)


DEFAULT_DB = ROOT / "data" / "microstructure.db"
OUT_DIR = ROOT / "reports" / "research" / "s34"
OUT_JSON = OUT_DIR / "S34_V02_PROPAGATION_PUZZLE_SUITE.json"
OUT_MD = OUT_DIR / "S34_V02_PROPAGATION_PUZZLE_SUITE.md"

SYMBOL = "ETHUSDT"
SAME_SIDE_WINDOW_SEC = 3600
CROSS_WINDOWS_SEC = (300, 900, 1800, 3600)
CHAIN_GAP_SEC = 4 * 3600


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def month_of(ts_ms: int) -> str:
    return datetime.fromtimestamp(int(ts_ms) / 1000.0, tz=timezone.utc).strftime("%Y-%m")


def idx(events: list[dict[str, Any]]) -> list[int]:
    return [int(e["anchor_ts_ms"]) for e in events]


def opposite(side: str) -> str:
    return "BUY" if side == "SELL" else "SELL"


def momentum_direction(side: str) -> str:
    return "SHORT" if side == "SELL" else "LONG"


def fade_direction(side: str) -> str:
    return "LONG" if side == "SELL" else "SHORT"


def direction_return(marks: Any, direction: str, start_ms: int, horizon_sec: int, fee_bps: float = FEE_BPS) -> float | None:
    a = mark_price_at(marks, int(start_ms), before=False)
    b = mark_price_at(marks, int(start_ms) + int(horizon_sec) * 1000, before=False)
    if not a or not b:
        return None
    return r1(signed_return_bps(direction, float(a[1]), float(b[1])) - fee_bps)


def feature_bucket(value: Any, cuts: tuple[float, float], labels: tuple[str, str, str]) -> str:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not math.isfinite(x):
        return "NA"
    if x < cuts[0]:
        return labels[0]
    if x < cuts[1]:
        return labels[1]
    return labels[2]


def quantile_cuts(rows: list[dict[str, Any]], key: str) -> tuple[float, float] | None:
    vals = []
    for row in rows:
        try:
            x = float(row.get(key))
        except (TypeError, ValueError):
            continue
        if math.isfinite(x):
            vals.append(x)
    if len(vals) < 5:
        return None
    return float(pctile(vals, 0.33)), float(pctile(vals, 0.66))


def enrich_events(
    conn: sqlite3.Connection,
    eth_events: list[dict[str, Any]],
    asset_events: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    marks = {sym: __import__("tools.research_s34_knowable_anchor_continuation", fromlist=["load_mark_index"]).load_mark_index(conn, sym) for sym in ASSET_THRESHOLDS}
    eth_idx = idx(eth_events)
    asset_idx = {sym: idx(rows) for sym, rows in asset_events.items()}
    out = []
    for i, e in enumerate(eth_events):
        ts = int(e["anchor_ts_ms"])
        side = str(e["side"])
        next_same = [
            x for x in neighbor_events(eth_idx, eth_events, ts_ms=ts, before_sec=0, after_sec=SAME_SIDE_WINDOW_SEC, side=side)
            if int(x["anchor_ts_ms"]) > ts
        ]
        next_opp = [
            x for x in neighbor_events(eth_idx, eth_events, ts_ms=ts, before_sec=0, after_sec=SAME_SIDE_WINDOW_SEC, side=opposite(side))
            if int(x["anchor_ts_ms"]) > ts
        ]
        prev_same = [
            x for x in neighbor_events(eth_idx, eth_events, ts_ms=ts, before_sec=SAME_SIDE_WINDOW_SEC, after_sec=0, side=side)
            if int(x["anchor_ts_ms"]) < ts
        ]
        cross_next_same_by_window = {}
        cross_prev_same_by_window = {}
        for win in CROSS_WINDOWS_SEC:
            n_next = 0
            n_prev = 0
            for sym in ("BTCUSDT", "SOLUSDT"):
                rows = asset_events.get(sym, [])
                ids = asset_idx.get(sym, [])
                n_next += len([
                    x for x in neighbor_events(ids, rows, ts_ms=ts, before_sec=0, after_sec=win, side=side)
                    if int(x["anchor_ts_ms"]) > ts
                ])
                n_prev += len([
                    x for x in neighbor_events(ids, rows, ts_ms=ts, before_sec=win, after_sec=0, side=side)
                    if int(x["anchor_ts_ms"]) < ts
                ])
            cross_next_same_by_window[f"cross_next_same_{win}s"] = n_next
            cross_prev_same_by_window[f"cross_prev_same_{win}s"] = n_prev
        # Consecutive same-side chain rank around this event.
        rank = 1
        j = i - 1
        while j >= 0 and eth_events[j]["side"] == side and (ts - int(eth_events[j]["anchor_ts_ms"])) <= CHAIN_GAP_SEC * rank:
            rank += 1
            j -= 1
        chain_len_forward = 1
        j = i + 1
        prev_ts = ts
        while j < len(eth_events) and eth_events[j]["side"] == side and (int(eth_events[j]["anchor_ts_ms"]) - prev_ts) <= CHAIN_GAP_SEC:
            chain_len_forward += 1
            prev_ts = int(eth_events[j]["anchor_ts_ms"])
            j += 1
        row = {
            **e,
            "has_next_same_60m": bool(next_same),
            "has_next_opp_60m": bool(next_opp),
            "has_prev_same_60m": bool(prev_same),
            "next_same_gap_sec": r1((int(next_same[0]["anchor_ts_ms"]) - ts) / 1000.0) if next_same else None,
            "next_opp_gap_sec": r1((int(next_opp[0]["anchor_ts_ms"]) - ts) / 1000.0) if next_opp else None,
            "same_side_chain_rank": rank,
            "same_side_chain_len_forward": chain_len_forward,
            "momentum_h15_bps": direction_return(marks[SYMBOL], momentum_direction(side), ts, 900),
            "momentum_h1_bps": direction_return(marks[SYMBOL], momentum_direction(side), ts, 3600),
            "momentum_h2_bps": direction_return(marks[SYMBOL], momentum_direction(side), ts, 7200),
            "momentum_h4_bps": direction_return(marks[SYMBOL], momentum_direction(side), ts, 14400),
            "fade_h15_bps": direction_return(marks[SYMBOL], fade_direction(side), ts, 900),
            "fade_h1_bps": direction_return(marks[SYMBOL], fade_direction(side), ts, 3600),
            "fade_h2_bps": direction_return(marks[SYMBOL], fade_direction(side), ts, 7200),
            "fade_h4_bps": direction_return(marks[SYMBOL], fade_direction(side), ts, 14400),
            **cross_next_same_by_window,
            **cross_prev_same_by_window,
        }
        row["propagation_pressure_score"] = (
            int(row["has_next_same_60m"]) * 3
            + int(row["cross_next_same_900s"] > 0) * 2
            + int(row["cross_next_same_1800s"] > 0) * 1
            + int(float(row.get("post_anchor_liq_notional") or 0.0) > 0.0)
            + int(float(row.get("event_duration_sec") or 0.0) > 60.0)
        )
        row["silence_after_shock"] = (
            not row["has_next_same_60m"]
            and row["cross_next_same_1800s"] == 0
            and row.get("reclaim_ts_ms") is not None
        )
        out.append(row)
    return out


def group_metrics(rows: list[dict[str, Any]], key: str, value: str) -> dict[str, Any]:
    groups = sorted({str(r.get(key)) for r in rows})
    return {g: metrics([r.get(value) for r in rows if str(r.get(key)) == g]) for g in groups}


def propagation_predictor(rows: list[dict[str, Any]]) -> dict[str, Any]:
    candidates = ["running_accel", "post_anchor_liq_notional", "event_duration_sec", "running_liq_count", "single_dominance_pct", "bid_depth_usd", "ask_depth_usd", "book_imbalance"]
    labelled = []
    for key in candidates:
        cuts = quantile_cuts(rows, key)
        if not cuts:
            continue
        labels = ("LOW", "MID", "HIGH")
        by_bucket = {}
        for bucket in labels:
            subset = [r for r in rows if feature_bucket(r.get(key), cuts, labels) == bucket]
            n = len(subset)
            by_bucket[bucket] = {
                "n": n,
                "next_same_rate": r3(sum(1 for r in subset if r["has_next_same_60m"]) / n) if n else None,
                "fade_h4": metrics([r.get("fade_h4_bps") for r in subset]),
                "momentum_h1": metrics([r.get("momentum_h1_bps") for r in subset]),
            }
        labelled.append({"feature": key, "cuts": [r1(cuts[0]), r1(cuts[1])], "by_bucket": by_bucket})
    return {
        "overall_next_same_rate": r3(sum(1 for r in rows if r["has_next_same_60m"]) / len(rows)) if rows else None,
        "by_side": {
            side: {
                "n": len([r for r in rows if r["side"] == side]),
                "next_same_rate": r3(sum(1 for r in rows if r["side"] == side and r["has_next_same_60m"]) / max(1, len([r for r in rows if r["side"] == side]))),
            }
            for side in ("SELL", "BUY")
        },
        "feature_buckets": labelled,
    }


def momentum_alpha(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out = {}
    for side in ("SELL", "BUY"):
        side_rows = [r for r in rows if r["side"] == side]
        prop = [r for r in side_rows if r["has_next_same_60m"] or r["cross_next_same_900s"] > 0]
        no_prop = [r for r in side_rows if r not in prop]
        out[side] = {
            "all_momentum_h1": metrics([r.get("momentum_h1_bps") for r in side_rows]),
            "all_momentum_h4": metrics([r.get("momentum_h4_bps") for r in side_rows]),
            "propagation_momentum_h1": metrics([r.get("momentum_h1_bps") for r in prop]),
            "propagation_momentum_h4": metrics([r.get("momentum_h4_bps") for r in prop]),
            "no_propagation_fade_h4": metrics([r.get("fade_h4_bps") for r in no_prop]),
            "propagation_fade_h4": metrics([r.get("fade_h4_bps") for r in prop]),
            "propagation_n": len(prop),
            "no_propagation_n": len(no_prop),
        }
    return out


def chain_rank_tests(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out = {}
    for side in ("SELL", "BUY"):
        side_rows = [r for r in rows if r["side"] == side]
        by_rank = {}
        for rank in ("1", "2", "3+"):
            if rank == "1":
                subset = [r for r in side_rows if int(r["same_side_chain_rank"]) == 1]
            elif rank == "2":
                subset = [r for r in side_rows if int(r["same_side_chain_rank"]) == 2]
            else:
                subset = [r for r in side_rows if int(r["same_side_chain_rank"]) >= 3]
            by_rank[rank] = {
                "n": len(subset),
                "momentum_h1": metrics([r.get("momentum_h1_bps") for r in subset]),
                "momentum_h4": metrics([r.get("momentum_h4_bps") for r in subset]),
                "fade_h4": metrics([r.get("fade_h4_bps") for r in subset]),
                "next_same_rate": r3(sum(1 for r in subset if r["has_next_same_60m"]) / len(subset)) if subset else None,
            }
        out[side] = by_rank
    return out


def cross_asset_timing(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out = {}
    for side in ("SELL", "BUY"):
        side_rows = [r for r in rows if r["side"] == side]
        out[side] = {}
        for win in CROSS_WINDOWS_SEC:
            key = f"cross_next_same_{win}s"
            out[side][key] = {
                "true_fade_h4": metrics([r.get("fade_h4_bps") for r in side_rows if int(r.get(key) or 0) > 0]),
                "false_fade_h4": metrics([r.get("fade_h4_bps") for r in side_rows if int(r.get(key) or 0) == 0]),
                "true_momentum_h1": metrics([r.get("momentum_h1_bps") for r in side_rows if int(r.get(key) or 0) > 0]),
                "false_momentum_h1": metrics([r.get("momentum_h1_bps") for r in side_rows if int(r.get(key) or 0) == 0]),
            }
    return out


def silence_state(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out = {}
    for side in ("SELL", "BUY"):
        side_rows = [r for r in rows if r["side"] == side]
        silent = [r for r in side_rows if r["silence_after_shock"]]
        noisy = [r for r in side_rows if not r["silence_after_shock"]]
        out[side] = {
            "silence_n": len(silent),
            "noisy_n": len(noisy),
            "silence_fade_h4": metrics([r.get("fade_h4_bps") for r in silent]),
            "noisy_fade_h4": metrics([r.get("fade_h4_bps") for r in noisy]),
            "silence_momentum_h1": metrics([r.get("momentum_h1_bps") for r in silent]),
            "noisy_momentum_h1": metrics([r.get("momentum_h1_bps") for r in noisy]),
        }
    return out


def buy_side_diagnosis(rows: list[dict[str, Any]]) -> dict[str, Any]:
    buy = [r for r in rows if r["side"] == "BUY"]
    return {
        "buy_fade_short_h4": metrics([r.get("fade_h4_bps") for r in buy]),
        "buy_continuation_long_h1": metrics([r.get("momentum_h1_bps") for r in buy]),
        "buy_continuation_long_h4": metrics([r.get("momentum_h4_bps") for r in buy]),
        "buy_buy_propagation_long_h1": metrics([r.get("momentum_h1_bps") for r in buy if r["has_next_same_60m"]]),
        "buy_buy_propagation_short_fade_h4": metrics([r.get("fade_h4_bps") for r in buy if r["has_next_same_60m"]]),
        "buy_silence_short_fade_h4": metrics([r.get("fade_h4_bps") for r in buy if r["silence_after_shock"]]),
    }


def composite_indicator(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out = {}
    for side in ("SELL", "BUY"):
        side_rows = [r for r in rows if r["side"] == side]
        by_score = {}
        for label, subset in {
            "LOW_0_1": [r for r in side_rows if int(r["propagation_pressure_score"]) <= 1],
            "MID_2_3": [r for r in side_rows if 2 <= int(r["propagation_pressure_score"]) <= 3],
            "HIGH_4_PLUS": [r for r in side_rows if int(r["propagation_pressure_score"]) >= 4],
        }.items():
            by_score[label] = {
                "n": len(subset),
                "fade_h4": metrics([r.get("fade_h4_bps") for r in subset]),
                "momentum_h1": metrics([r.get("momentum_h1_bps") for r in subset]),
                "next_same_rate": r3(sum(1 for r in subset if r["has_next_same_60m"]) / len(subset)) if subset else None,
            }
        out[side] = by_score
    return out


def transition_matrix(rows: list[dict[str, Any]]) -> dict[str, Any]:
    # Rebuild only adjacent transitions from enriched ETH rows.
    trans = []
    for i, r in enumerate(rows[:-1]):
        nxt = rows[i + 1]
        gap = (int(nxt["anchor_ts_ms"]) - int(r["anchor_ts_ms"])) / 1000.0
        if gap > CHAIN_GAP_SEC:
            continue
        key = f"{r['side']}->{nxt['side']}"
        trans.append({**r, "transition": key, "gap_sec": gap})
    return {
        key: {
            "n": len([r for r in trans if r["transition"] == key]),
            "fade_h4": metrics([r.get("fade_h4_bps") for r in trans if r["transition"] == key]),
            "momentum_h1": metrics([r.get("momentum_h1_bps") for r in trans if r["transition"] == key]),
            "pressure_score": metrics([r.get("propagation_pressure_score") for r in trans if r["transition"] == key]),
        }
        for key in sorted({r["transition"] for r in trans})
    }


def v02_hold_decision(enriched: list[dict[str, Any]]) -> dict[str, Any]:
    h4_rows = [
        r for r in load_jsonl(H4_LEDGER_JSONL)
        if r.get("bucket") == "H4_SHADOW" and r.get("observation_status") == "CLOSED" and r.get("net_bps") is not None
    ]
    ts_to_event = {int(r["anchor_ts_ms"]): r for r in enriched if r["side"] == "SELL"}
    # Match nearest same timestamp within 2s because ledgers may stringify/source from same anchor.
    e_ts = sorted(ts_to_event)
    rows = []
    for trade in h4_rows:
        ts = int(trade["signal_ts_ms"])
        pos = bisect_left(e_ts, ts)
        candidates = []
        for j in (pos - 1, pos, pos + 1):
            if 0 <= j < len(e_ts) and abs(e_ts[j] - ts) <= 2_000:
                candidates.append(ts_to_event[e_ts[j]])
        ev = candidates[0] if candidates else None
        h2 = float(trade.get("h2_net_bps"))
        h4 = float(trade.get("net_bps"))
        rows.append({
            "signal_utc": trade.get("signal_utc"),
            "h2": h2,
            "h4": h4,
            "h4_minus_h2": r1(h4 - h2),
            "matched": bool(ev),
            "same_side_next_60m": bool(ev and ev["has_next_same_60m"]),
            "cross_next_same_1800s": int(ev.get("cross_next_same_1800s") or 0) if ev else None,
            "silence_after_shock": bool(ev and ev["silence_after_shock"]),
            "pressure_score": int(ev["propagation_pressure_score"]) if ev else None,
        })
    return {
        "matched_n": sum(1 for r in rows if r["matched"]),
        "all_h4_minus_h2": metrics([r["h4_minus_h2"] for r in rows]),
        "same_side_next_true": metrics([r["h4_minus_h2"] for r in rows if r["same_side_next_60m"]]),
        "same_side_next_false": metrics([r["h4_minus_h2"] for r in rows if r["matched"] and not r["same_side_next_60m"]]),
        "silence_true": metrics([r["h4_minus_h2"] for r in rows if r["silence_after_shock"]]),
        "silence_false": metrics([r["h4_minus_h2"] for r in rows if r["matched"] and not r["silence_after_shock"]]),
        "pressure_high": metrics([r["h4_minus_h2"] for r in rows if r["pressure_score"] is not None and r["pressure_score"] >= 4]),
        "pressure_low_mid": metrics([r["h4_minus_h2"] for r in rows if r["pressure_score"] is not None and r["pressure_score"] < 4]),
        "rows": rows,
    }


def render_md(report: dict[str, Any]) -> str:
    lines = [
        "# S34 V02 Propagation Puzzle Suite",
        "",
        f"Generated: `{report['generated_at_utc']}`",
        "",
        "Research-only. No live executor/config/order logic is touched.",
        "",
        "## Verdict",
        "",
        f"- `{report['verdict']}`",
        "",
        "## 1. Propagation Predictor",
        "",
        f"Overall next-same-side rate: `{report['propagation_predictor']['overall_next_same_rate']}`",
        "",
        "## 2. Momentum vs Fade Alpha",
        "",
        "```json",
        json.dumps(report["momentum_alpha"], indent=2, sort_keys=True),
        "```",
        "",
        "## 3. First / Second / Third Chain Rank",
        "",
        "```json",
        json.dumps(report["chain_rank_tests"], indent=2, sort_keys=True),
        "```",
        "",
        "## 4. Cross-Asset Propagation Timing",
        "",
        "```json",
        json.dumps(report["cross_asset_timing"], indent=2, sort_keys=True),
        "```",
        "",
        "## 5. Silence After Shock",
        "",
        "```json",
        json.dumps(report["silence_state"], indent=2, sort_keys=True),
        "```",
        "",
        "## 6. BUY Side Diagnosis",
        "",
        "```json",
        json.dumps(report["buy_side_diagnosis"], indent=2, sort_keys=True),
        "```",
        "",
        "## 7. Composite Propagation Indicator",
        "",
        "```json",
        json.dumps(report["composite_indicator"], indent=2, sort_keys=True),
        "```",
        "",
        "## 8. Transition Matrix Navigation",
        "",
        "```json",
        json.dumps(report["transition_matrix"], indent=2, sort_keys=True),
        "```",
        "",
        "## 9. V02 H4 Hold Decision",
        "",
        "```json",
        json.dumps({k: v for k, v in report["v02_hold_decision"].items() if k != "rows"}, indent=2, sort_keys=True),
        "```",
        "",
        "## Read",
        "",
    ]
    lines.extend([f"- {x}" for x in report["read"]])
    lines.append("")
    return "\n".join(lines)


def run(db: Path) -> dict[str, Any]:
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=30)
    try:
        asset_events = {sym: build_events(conn, symbol=sym, threshold=thr) for sym, thr in ASSET_THRESHOLDS.items()}
        enriched = enrich_events(conn, asset_events["ETHUSDT"], asset_events)
    finally:
        conn.close()
    report = {
        "generated_at_utc": utc_now(),
        "research_only": True,
        "live_executor_touched": False,
        "event_counts": {sym: len(rows) for sym, rows in asset_events.items()},
        "propagation_predictor": propagation_predictor(enriched),
        "momentum_alpha": momentum_alpha(enriched),
        "chain_rank_tests": chain_rank_tests(enriched),
        "cross_asset_timing": cross_asset_timing(enriched),
        "silence_state": silence_state(enriched),
        "buy_side_diagnosis": buy_side_diagnosis(enriched),
        "composite_indicator": composite_indicator(enriched),
        "transition_matrix": transition_matrix(enriched),
        "v02_hold_decision": v02_hold_decision(enriched),
        "sample_rows": enriched[:50],
        "verdict": "PROPAGATION_STATE_IS_REAL_NAVIGATION_NOT_LIVE_ALPHA_YET",
        "read": [
            "The negative SELL->SELL and BUY->BUY fade cells behave like same-side propagation/runaway states.",
            "The key practical split is not BUY vs SELL alone; it is propagation pressure versus silence/reclaim.",
            "Momentum tests here are mark-entry broad-event tests, not live-ready maker execution.",
            "For current V02, use these as navigation/management observers until forward filled N is large enough.",
        ],
    }
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run S34 V02 propagation puzzle suite.")
    p.add_argument("--db", type=Path, default=DEFAULT_DB)
    p.add_argument("--out-json", type=Path, default=OUT_JSON)
    p.add_argument("--out-md", type=Path, default=OUT_MD)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = run(args.db)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    args.out_md.write_text(render_md(report), encoding="utf-8")
    print(render_md(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
