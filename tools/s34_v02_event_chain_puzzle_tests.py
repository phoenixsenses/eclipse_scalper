"""S34 V02 event-chain puzzle tests.

Research-only. Treats liquidation cascades as connected lifecycle events instead
of isolated anchors: pre-event, start, anchor, end, post-event, next-event, and
cross-asset chains.
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

from tools.research_s34_knowable_anchor_continuation import (  # noqa: E402
    iso_ms,
    load_liquidations,
    load_mark_index,
    pctile,
    r1,
    r3,
    reconstruct_anchors,
    signed_return_bps,
)
from tools.research_s34_wave_absorption import book_features_at  # noqa: E402


DEFAULT_DB = ROOT / "data" / "microstructure.db"
OUT_DIR = ROOT / "reports" / "research" / "s34"
FOUR_ARM_JSON = OUT_DIR / "S34_V02_FOUR_ARM_SYMMETRY_TESTS.json"
H4_LEDGER_JSONL = OUT_DIR / "S34_V02_H4_FORWARD_SHADOW_LEDGER.jsonl"
OUT_JSON = OUT_DIR / "S34_V02_EVENT_CHAIN_PUZZLE_TESTS.json"
OUT_MD = OUT_DIR / "S34_V02_EVENT_CHAIN_PUZZLE_TESTS.md"

BUCKET_SEC = 300
MIN_GAP_SEC = 900
ACCEL_WINDOW_SEC = 30
ETH_THRESHOLD = 200_000.0
ASSET_THRESHOLDS = {"BTCUSDT": 500_000.0, "ETHUSDT": 200_000.0, "SOLUSDT": 100_000.0}
FEE_BPS = 8.0
HORIZONS_SEC = {"M15": 900, "H1": 3600, "H2": 7200, "H4": 14400}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def finite(v: Any) -> float | None:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def metrics(vals: list[Any]) -> dict[str, Any]:
    xs = [float(x) for x in (finite(v) for v in vals) if x is not None]
    if not xs:
        return {
            "n": 0,
            "sum_bps": 0.0,
            "mean_bps": None,
            "median_bps": None,
            "win_rate": None,
            "t3r_bps": 0.0,
            "min_bps": None,
            "max_bps": None,
            "tail_lt_-100_n": 0,
        }
    ordered = sorted(xs, reverse=True)
    return {
        "n": len(xs),
        "sum_bps": r1(sum(xs)),
        "mean_bps": r1(mean(xs)),
        "median_bps": r1(pctile(xs, 0.5)),
        "win_rate": r3(sum(1 for x in xs if x > 0.0) / len(xs)),
        "t3r_bps": r1(sum(ordered[3:]) if len(ordered) > 3 else sum(ordered)),
        "min_bps": r1(min(xs)),
        "max_bps": r1(max(xs)),
        "tail_lt_-100_n": sum(1 for x in xs if x < -100.0),
    }


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def month_of(ts_ms: int) -> str:
    return datetime.fromtimestamp(int(ts_ms) / 1000.0, tz=timezone.utc).strftime("%Y-%m")


def side_to_fade(side: str) -> str:
    return "LONG" if side.upper() == "SELL" else "SHORT"


def mark_price_at(marks: Any, ts_ms: int, *, before: bool = False) -> tuple[int, float] | None:
    row = marks.at_or_before(int(ts_ms)) if before else marks.at_or_after(int(ts_ms))
    return (int(row[0]), float(row[1])) if row else None


def mark_ret(
    marks: Any,
    *,
    direction: str,
    entry_ms: int,
    horizon_sec: int,
    fee_bps: float = FEE_BPS,
) -> float | None:
    entry = mark_price_at(marks, int(entry_ms), before=False)
    exit_ = mark_price_at(marks, int(entry_ms) + int(horizon_sec) * 1000, before=False)
    if not entry or not exit_:
        return None
    return r1(signed_return_bps(direction, float(entry[1]), float(exit_[1])) - float(fee_bps))


def raw_ret(marks: Any, start_ms: int, end_ms: int) -> float | None:
    a = mark_price_at(marks, int(start_ms), before=False)
    b = mark_price_at(marks, int(end_ms), before=False)
    if not a or not b or a[1] <= 0:
        return None
    return r1((b[1] - a[1]) / a[1] * 10_000.0)


def bucket_liq_stats(
    conn: sqlite3.Connection,
    *,
    symbol: str,
    side: str,
    bucket: int,
) -> dict[str, Any]:
    start = int(bucket) * BUCKET_SEC * 1000
    end = start + BUCKET_SEC * 1000
    rows = conn.execute(
        """
        SELECT ts_ms, notional FROM liquidations
        WHERE symbol=? AND side=? AND ts_ms>=? AND ts_ms<?
        ORDER BY ts_ms
        """,
        (symbol, side, start, end),
    ).fetchall()
    if not rows:
        return {}
    notionals = [float(r[1]) for r in rows]
    first_ts = int(rows[0][0])
    last_ts = int(rows[-1][0])
    total = sum(notionals)
    return {
        "event_start_ts_ms": first_ts,
        "event_end_ts_ms": last_ts,
        "event_duration_sec": r1((last_ts - first_ts) / 1000.0),
        "event_total_notional": r1(total),
        "event_liq_count": len(rows),
        "event_single_dominance_pct": r1(max(notionals) / total * 100.0) if total > 0 else None,
        "post_anchor_liq_notional": None,
    }


def first_reclaim_ts(
    marks: Any,
    *,
    side: str,
    anchor_price: float,
    start_ms: int,
    horizon_sec: int = 3600,
) -> int | None:
    end_ms = int(start_ms) + int(horizon_sec) * 1000
    for ts, px in marks.slice_range(int(start_ms), end_ms):
        if side.upper() == "SELL" and float(px) >= float(anchor_price):
            return int(ts)
        if side.upper() == "BUY" and float(px) <= float(anchor_price):
            return int(ts)
    return None


def build_events(conn: sqlite3.Connection, *, symbol: str, threshold: float) -> list[dict[str, Any]]:
    marks = load_mark_index(conn, symbol)
    out = []
    for side in ("SELL", "BUY"):
        liqs = load_liquidations(conn, symbol, side, None, None)
        anchors = reconstruct_anchors(
            liqs,
            bucket_sec=BUCKET_SEC,
            min_gap_sec=MIN_GAP_SEC,
            thresholds=(float(threshold),),
            accel_window_sec=ACCEL_WINDOW_SEC,
        )
        for anchor in anchors:
            stats = bucket_liq_stats(conn, symbol=symbol, side=side, bucket=int(anchor.bucket))
            anchor_mark = mark_price_at(marks, int(anchor.anchor_ts_ms), before=False)
            if not stats or not anchor_mark:
                continue
            stats["post_anchor_liq_notional"] = r1(
                conn.execute(
                    """
                    SELECT COALESCE(SUM(notional),0.0) FROM liquidations
                    WHERE symbol=? AND side=? AND ts_ms>? AND ts_ms<?
                    """,
                    (
                        symbol,
                        side,
                        int(anchor.anchor_ts_ms),
                        int(anchor.bucket) * BUCKET_SEC * 1000 + BUCKET_SEC * 1000,
                    ),
                ).fetchone()[0]
            )
            direction = side_to_fade(side)
            reclaim = first_reclaim_ts(
                marks,
                side=side,
                anchor_price=float(anchor_mark[1]),
                start_ms=int(stats["event_end_ts_ms"]),
                horizon_sec=3600,
            )
            book = book_features_at(conn, symbol, int(anchor.anchor_ts_ms), 10)
            row = {
                "symbol": symbol,
                "side": side,
                "fade_direction": direction,
                "bucket": int(anchor.bucket),
                "event_id": f"{symbol}:{side}:{anchor.bucket}:{int(threshold)}",
                "first_ts_ms": int(anchor.first_ts_ms),
                "first_utc": iso_ms(anchor.first_ts_ms),
                "anchor_ts_ms": int(anchor.anchor_ts_ms),
                "anchor_utc": iso_ms(anchor.anchor_ts_ms),
                "anchor_mark_price": float(anchor_mark[1]),
                "month": month_of(anchor.anchor_ts_ms),
                "threshold_usd": float(threshold),
                "elapsed_to_anchor_sec": r1(anchor.elapsed_since_first_sec),
                "running_notional": r1(anchor.running_notional),
                "running_liq_count": int(anchor.running_liq_count),
                "running_rate": r1(anchor.running_rate),
                "running_accel": r1(anchor.running_accel),
                "single_dominance_pct": r1(anchor.running_single_liq_dominance),
                **stats,
                "event_end_utc": iso_ms(stats["event_end_ts_ms"]),
                "reclaim_ts_ms": reclaim,
                "reclaim_utc": iso_ms(reclaim),
                "reclaim_delay_sec": r1((reclaim - int(stats["event_end_ts_ms"])) / 1000.0) if reclaim else None,
                "pre15_bps": raw_ret(marks, int(anchor.anchor_ts_ms) - 900_000, int(anchor.anchor_ts_ms)),
                "post15_fade_bps": mark_ret(marks, direction=direction, entry_ms=int(anchor.anchor_ts_ms), horizon_sec=900),
                "anchor_h1_fade_bps": mark_ret(marks, direction=direction, entry_ms=int(anchor.anchor_ts_ms), horizon_sec=3600),
                "anchor_h2_fade_bps": mark_ret(marks, direction=direction, entry_ms=int(anchor.anchor_ts_ms), horizon_sec=7200),
                "anchor_h4_fade_bps": mark_ret(marks, direction=direction, entry_ms=int(anchor.anchor_ts_ms), horizon_sec=14400),
                "end_h1_fade_bps": mark_ret(marks, direction=direction, entry_ms=int(stats["event_end_ts_ms"]), horizon_sec=3600),
                "end_h2_fade_bps": mark_ret(marks, direction=direction, entry_ms=int(stats["event_end_ts_ms"]), horizon_sec=7200),
                "end_h4_fade_bps": mark_ret(marks, direction=direction, entry_ms=int(stats["event_end_ts_ms"]), horizon_sec=14400),
                "reclaim_h2_fade_bps": mark_ret(marks, direction=direction, entry_ms=reclaim, horizon_sec=7200) if reclaim else None,
                "reclaim_h4_fade_bps": mark_ret(marks, direction=direction, entry_ms=reclaim, horizon_sec=14400) if reclaim else None,
                "bid_depth_usd": r1(book["bid_depth_usd"]) if book else None,
                "ask_depth_usd": r1(book["ask_depth_usd"]) if book else None,
                "book_imbalance": r3(book["book_imbalance"]) if book else None,
                "spread_bps": r1(book["spread_bps"]) if book else None,
            }
            out.append(row)
    return sorted(out, key=lambda r: int(r["anchor_ts_ms"]))


def index_events(events: list[dict[str, Any]]) -> tuple[list[int], list[dict[str, Any]]]:
    return [int(e["anchor_ts_ms"]) for e in events], events


def neighbor_events(
    ts_index: list[int],
    events: list[dict[str, Any]],
    *,
    ts_ms: int,
    before_sec: int,
    after_sec: int,
    exclude_id: str | None = None,
    side: str | None = None,
) -> list[dict[str, Any]]:
    lo = bisect_left(ts_index, int(ts_ms) - before_sec * 1000)
    hi = bisect_left(ts_index, int(ts_ms) + after_sec * 1000 + 1)
    rows = events[lo:hi]
    if exclude_id:
        rows = [r for r in rows if r.get("event_id") != exclude_id]
    if side:
        rows = [r for r in rows if r.get("side") == side]
    return rows


def same_symbol_transition(events: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for i, e in enumerate(events):
        nxt = events[i + 1] if i + 1 < len(events) else None
        if not nxt:
            continue
        gap = (int(nxt["anchor_ts_ms"]) - int(e["anchor_ts_ms"])) / 1000.0
        if gap > 4 * 3600:
            continue
        rows.append({
            "transition": f"{e['side']}->{nxt['side']}",
            "gap_sec": r1(gap),
            "current_side": e["side"],
            "next_side": nxt["side"],
            "current_h4_fade_bps": e.get("anchor_h4_fade_bps"),
            "next_h4_fade_bps": nxt.get("anchor_h4_fade_bps"),
            "current_event_id": e["event_id"],
            "next_event_id": nxt["event_id"],
        })
    return {
        "transition_count": {k: sum(1 for r in rows if r["transition"] == k) for k in sorted({r["transition"] for r in rows})},
        "current_outcome_by_transition": {k: metrics([r["current_h4_fade_bps"] for r in rows if r["transition"] == k]) for k in sorted({r["transition"] for r in rows})},
        "next_outcome_by_transition": {k: metrics([r["next_h4_fade_bps"] for r in rows if r["transition"] == k]) for k in sorted({r["transition"] for r in rows})},
        "gap_by_transition": {k: metrics([r["gap_sec"] for r in rows if r["transition"] == k]) for k in sorted({r["transition"] for r in rows})},
        "sample": rows[:20],
    }


def event_end_vs_anchor(events: list[dict[str, Any]]) -> dict[str, Any]:
    by_side = {}
    for side in ("SELL", "BUY"):
        rows = [e for e in events if e["side"] == side]
        by_side[side] = {
            "anchor_h2": metrics([e.get("anchor_h2_fade_bps") for e in rows]),
            "anchor_h4": metrics([e.get("anchor_h4_fade_bps") for e in rows]),
            "event_end_h2": metrics([e.get("end_h2_fade_bps") for e in rows]),
            "event_end_h4": metrics([e.get("end_h4_fade_bps") for e in rows]),
            "reclaim_h2": metrics([e.get("reclaim_h2_fade_bps") for e in rows]),
            "reclaim_h4": metrics([e.get("reclaim_h4_fade_bps") for e in rows]),
            "anchor_to_end_delta_h4": metrics([
                (float(e["end_h4_fade_bps"]) - float(e["anchor_h4_fade_bps"]))
                for e in rows
                if e.get("end_h4_fade_bps") is not None and e.get("anchor_h4_fade_bps") is not None
            ]),
            "reclaim_delay_sec": metrics([e.get("reclaim_delay_sec") for e in rows]),
        }
    return by_side


def load_h4_filled_rows() -> list[dict[str, Any]]:
    rows = []
    for row in load_jsonl(H4_LEDGER_JSONL):
        if row.get("bucket") == "H4_SHADOW" and row.get("observation_status") == "CLOSED" and row.get("net_bps") is not None:
            rows.append(row)
    return rows


def v02_runner_anatomy(eth_events: list[dict[str, Any]]) -> dict[str, Any]:
    h4_rows = load_h4_filled_rows()
    idx, events = index_events(eth_events)
    out = []
    for row in h4_rows:
        ts = int(row["signal_ts_ms"])
        prev_60 = [e for e in neighbor_events(idx, events, ts_ms=ts, before_sec=3600, after_sec=0) if int(e["anchor_ts_ms"]) < ts]
        next_60 = [e for e in neighbor_events(idx, events, ts_ms=ts, before_sec=0, after_sec=3600) if int(e["anchor_ts_ms"]) > ts]
        opposite_next_60 = [e for e in next_60 if e["side"] != "SELL"]
        same_next_60 = [e for e in next_60 if e["side"] == "SELL"]
        h4 = finite(row.get("net_bps"))
        h2 = finite(row.get("h2_net_bps"))
        out.append({
            "signal_utc": row.get("signal_utc"),
            "runner_h4": bool(h4 is not None and h2 is not None and h4 > h2),
            "h4_net_bps": h4,
            "h2_net_bps": h2,
            "h4_minus_h2_bps": r1(h4 - h2) if h4 is not None and h2 is not None else None,
            "state_path_v2": row.get("state_path_v2"),
            "prev_event_n_60m": len(prev_60),
            "prev_sides_60m": "".join(e["side"][0] for e in prev_60[-5:]),
            "next_event_n_60m": len(next_60),
            "next_sides_60m": "".join(e["side"][0] for e in next_60[:5]),
            "same_side_next_60m": len(same_next_60),
            "opposite_next_60m": len(opposite_next_60),
            "first_next_gap_sec": r1((int(next_60[0]["anchor_ts_ms"]) - ts) / 1000.0) if next_60 else None,
        })
    groups = {
        "runner": [r for r in out if r["runner_h4"]],
        "non_runner": [r for r in out if not r["runner_h4"]],
    }
    return {
        "n": len(out),
        "by_runner": {
            name: {
                "n": len(rows),
                "h4_minus_h2": metrics([r["h4_minus_h2_bps"] for r in rows]),
                "prev_event_n_60m": metrics([r["prev_event_n_60m"] for r in rows]),
                "next_event_n_60m": metrics([r["next_event_n_60m"] for r in rows]),
                "same_side_next_60m": metrics([r["same_side_next_60m"] for r in rows]),
                "opposite_next_60m": metrics([r["opposite_next_60m"] for r in rows]),
                "first_next_gap_sec": metrics([r["first_next_gap_sec"] for r in rows]),
            }
            for name, rows in groups.items()
        },
        "rows": out,
    }


def cross_asset_chain(eth_events: list[dict[str, Any]], asset_events: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    rows = [e for e in eth_events if e["side"] == "SELL" and e.get("anchor_h4_fade_bps") is not None]
    asset_indices = {sym: index_events(evts) for sym, evts in asset_events.items() if sym != "ETHUSDT"}
    out = []
    for e in rows:
        item = {
            "event_id": e["event_id"],
            "anchor_utc": e["anchor_utc"],
            "h4": e["anchor_h4_fade_bps"],
        }
        for sym, (idx, evts) in asset_indices.items():
            prev_same = [x for x in neighbor_events(idx, evts, ts_ms=e["anchor_ts_ms"], before_sec=1800, after_sec=0, side="SELL") if int(x["anchor_ts_ms"]) < int(e["anchor_ts_ms"])]
            next_same = [x for x in neighbor_events(idx, evts, ts_ms=e["anchor_ts_ms"], before_sec=0, after_sec=1800, side="SELL") if int(x["anchor_ts_ms"]) > int(e["anchor_ts_ms"])]
            prev_opp = [x for x in neighbor_events(idx, evts, ts_ms=e["anchor_ts_ms"], before_sec=1800, after_sec=0, side="BUY") if int(x["anchor_ts_ms"]) < int(e["anchor_ts_ms"])]
            next_opp = [x for x in neighbor_events(idx, evts, ts_ms=e["anchor_ts_ms"], before_sec=0, after_sec=1800, side="BUY") if int(x["anchor_ts_ms"]) > int(e["anchor_ts_ms"])]
            item[f"{sym}_prev_sell_30m"] = len(prev_same)
            item[f"{sym}_next_sell_30m"] = len(next_same)
            item[f"{sym}_prev_buy_30m"] = len(prev_opp)
            item[f"{sym}_next_buy_30m"] = len(next_opp)
        item["sync_prev_sell"] = bool(item.get("BTCUSDT_prev_sell_30m", 0) or item.get("SOLUSDT_prev_sell_30m", 0))
        item["propagation_next_sell"] = bool(item.get("BTCUSDT_next_sell_30m", 0) or item.get("SOLUSDT_next_sell_30m", 0))
        item["counter_next_buy"] = bool(item.get("BTCUSDT_next_buy_30m", 0) or item.get("SOLUSDT_next_buy_30m", 0))
        out.append(item)
    return {
        "sync_prev_sell": {
            "true": metrics([r["h4"] for r in out if r["sync_prev_sell"]]),
            "false": metrics([r["h4"] for r in out if not r["sync_prev_sell"]]),
        },
        "propagation_next_sell": {
            "true": metrics([r["h4"] for r in out if r["propagation_next_sell"]]),
            "false": metrics([r["h4"] for r in out if not r["propagation_next_sell"]]),
        },
        "counter_next_buy": {
            "true": metrics([r["h4"] for r in out if r["counter_next_buy"]]),
            "false": metrics([r["h4"] for r in out if not r["counter_next_buy"]]),
        },
        "rows": out[:50],
    }


def four_arm_chain_read() -> dict[str, Any]:
    four = load_json(FOUR_ARM_JSON, {})
    arms = four.get("arms", {})
    out = {}
    for arm_id in ("SELL_LONG_BASELINE", "BUY_SHORT_MIRROR"):
        rows = [
            r for r in arms.get(arm_id, {}).get("rows", [])
            if r.get("status") == "FILLED" and r.get("h4_net_bps") is not None
        ]
        out[arm_id] = {
            "h4": metrics([r.get("h4_net_bps") for r in rows]),
            "h4_minus_h2": metrics([r.get("h4_minus_h2_bps") for r in rows]),
            "by_cross_support": {
                "true": metrics([r.get("h4_net_bps") for r in rows if r.get("cross_support_ok") is True]),
                "false": metrics([r.get("h4_net_bps") for r in rows if r.get("cross_support_ok") is False]),
            },
            "by_fill_leg": {
                leg: metrics([r.get("h4_net_bps") for r in rows if r.get("fill_leg") == leg])
                for leg in sorted({str(r.get("fill_leg")) for r in rows})
            },
        }
    return out


def render_md(report: dict[str, Any]) -> str:
    lines = [
        "# S34 V02 Event-Chain Puzzle Tests",
        "",
        f"Generated: `{report['generated_at_utc']}`",
        "",
        "Research-only. No live executor/config/order logic is touched.",
        "",
        "## Verdict",
        "",
        f"- Overall: `{report['verdict']}`",
        "",
        "## 1. Same-Symbol Transition Graph",
        "",
        "Current-event H4 fade outcome by transition:",
        "",
        "| Transition | Count | Current H4 | Next H4 | Gap sec |",
        "| --- | ---: | --- | --- | --- |",
    ]
    trans = report["same_symbol_transition"]
    for key, count in trans["transition_count"].items():
        cur = trans["current_outcome_by_transition"][key]
        nxt = trans["next_outcome_by_transition"][key]
        gap = trans["gap_by_transition"][key]
        lines.append(
            f"| `{key}` | {count} | N={cur['n']} sum={cur['sum_bps']} med={cur['median_bps']} T3R={cur['t3r_bps']} | "
            f"N={nxt['n']} sum={nxt['sum_bps']} med={nxt['median_bps']} T3R={nxt['t3r_bps']} | med={gap['median_bps']} |"
        )
    lines += [
        "",
        "## 2. Anchor vs Event-End vs Reclaim",
        "",
        "| Side | Anchor H4 | Event-end H4 | Reclaim H4 | Anchor->End delta H4 | Reclaim delay |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for side, row in report["event_end_vs_anchor"].items():
        lines.append(
            f"| `{side}` | N={row['anchor_h4']['n']} sum={row['anchor_h4']['sum_bps']} med={row['anchor_h4']['median_bps']} T3R={row['anchor_h4']['t3r_bps']} | "
            f"N={row['event_end_h4']['n']} sum={row['event_end_h4']['sum_bps']} med={row['event_end_h4']['median_bps']} T3R={row['event_end_h4']['t3r_bps']} | "
            f"N={row['reclaim_h4']['n']} sum={row['reclaim_h4']['sum_bps']} med={row['reclaim_h4']['median_bps']} T3R={row['reclaim_h4']['t3r_bps']} | "
            f"N={row['anchor_to_end_delta_h4']['n']} sum={row['anchor_to_end_delta_h4']['sum_bps']} med={row['anchor_to_end_delta_h4']['median_bps']} | "
            f"med={row['reclaim_delay_sec']['median_bps']}s |"
        )
    lines += [
        "",
        "## 3. V02 Runner Chain Anatomy",
        "",
        "```json",
        json.dumps(report["v02_runner_anatomy"]["by_runner"], indent=2, sort_keys=True),
        "```",
        "",
        "## 4. Cross-Asset Chain",
        "",
        "```json",
        json.dumps({k: v for k, v in report["cross_asset_chain"].items() if k != "rows"}, indent=2, sort_keys=True),
        "```",
        "",
        "## 5. Four-Arm Chain Read",
        "",
        "```json",
        json.dumps(report["four_arm_chain_read"], indent=2, sort_keys=True),
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
    finally:
        conn.close()
    eth_events = asset_events["ETHUSDT"]
    transition = same_symbol_transition(eth_events)
    end_vs_anchor = event_end_vs_anchor(eth_events)
    runner = v02_runner_anatomy(eth_events)
    cross = cross_asset_chain(eth_events, asset_events)
    four = four_arm_chain_read()

    read = [
        "This reframes the V02 alpha as an event lifecycle problem, not a single anchor problem.",
        "Event-end/reclaim can be better for diagnosis, but if it sacrifices too much entry price it should stay navigation/management-only.",
        "Same-symbol transition tells whether the next cascade is a continuation, counter-cascade, or silence state.",
        "Cross-asset chain tests whether BTC/SOL are leading/propagating the ETH event or merely co-moving.",
        "These are research/navigation outputs only; no live or paper bucket is changed.",
    ]
    sell = end_vs_anchor.get("SELL", {})
    verdict = "CHAIN_NAVIGATION_HYPOTHESES_FOUND_NOT_LIVE_RULE"
    if sell and sell.get("event_end_h4", {}).get("t3r_bps", 0) > sell.get("anchor_h4", {}).get("t3r_bps", 0):
        read.append("SELL event_end H4 T3R beats anchor H4 T3R in this broad anchor-mark test; this is a management/navigation lead to validate on V02 fills.")
    return {
        "generated_at_utc": utc_now(),
        "research_only": True,
        "live_executor_touched": False,
        "config": {
            "bucket_sec": BUCKET_SEC,
            "min_gap_sec": MIN_GAP_SEC,
            "thresholds": ASSET_THRESHOLDS,
            "fee_bps": FEE_BPS,
        },
        "event_counts": {sym: len(rows) for sym, rows in asset_events.items()},
        "same_symbol_transition": transition,
        "event_end_vs_anchor": end_vs_anchor,
        "v02_runner_anatomy": runner,
        "cross_asset_chain": cross,
        "four_arm_chain_read": four,
        "verdict": verdict,
        "read": read,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run S34 V02 event-chain puzzle tests.")
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
