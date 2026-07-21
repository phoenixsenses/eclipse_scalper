"""S34 v0.2 momentum precursor tests.

Question: before rebound momentum starts, do microstructure metrics change
systematically 2-5 seconds earlier?

Research-only. This script reads historical DB windows around the frozen
S34_V_ENGINE_V0_2_ETH_SELL_MAKER_LONG_H2_O20_W300_O5_DEEPBID signal and writes
reports only. It does not touch live executor, order logic, size, leverage,
config, or environment files.
"""

from __future__ import annotations

import json
import math
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.research_s34_knowable_anchor_continuation import load_mark_index, r1, r3  # noqa: E402
from tools.research_s34_wave_absorption import book_features_at  # noqa: E402
from tools.s34_navigation_full_followup import DEFAULT_DB, mark_at_or_after, summary  # noqa: E402
from tools.s34_stress_reaction_deep_tests import mark_series  # noqa: E402
from tools.s34_v_engine_execution_frontier import collect_v01_events, prior_return_bps  # noqa: E402
from tools.s34_v_engine_v02_shadow_mirror import MIN_BID_DEPTH_USD  # noqa: E402

OUT_JSON = ROOT / "reports" / "research" / "s34" / "S34_V02_MOMENTUM_PRECURSOR_TESTS.json"
OUT_MD = ROOT / "reports" / "research" / "s34" / "S34_V02_MOMENTUM_PRECURSOR_TESTS.md"

SYMBOL = "ETHUSDT"
BTC = "BTCUSDT"
FEE_BPS = 5.0
MOMENTUM_THRESHOLDS = (20.0, 40.0)
MAX_ONSET_SEC = 600


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ts_iso(ts_ms: int) -> str:
    return datetime.fromtimestamp(int(ts_ms) / 1000.0, tz=timezone.utc).isoformat()


def t3r(vals: list[float]) -> float:
    vals = [float(v) for v in vals if math.isfinite(float(v))]
    return float(sum(sorted(vals, reverse=True)[3:])) if len(vals) > 3 else float(sum(vals))


def safe_mean(vals: list[float]) -> float | None:
    vals = [float(v) for v in vals if math.isfinite(float(v))]
    return sum(vals) / len(vals) if vals else None


def effect(a: list[float], b: list[float]) -> dict[str, Any]:
    a = [float(v) for v in a if math.isfinite(float(v))]
    b = [float(v) for v in b if math.isfinite(float(v))]
    if not a or not b:
        return {"a_n": len(a), "b_n": len(b), "delta_median": None, "delta_mean": None, "auc": None}
    # Probability a random a is greater than b, with ties half-weighted.
    wins = 0.0
    total = 0
    for x in a:
        for y in b:
            total += 1
            if x > y:
                wins += 1.0
            elif x == y:
                wins += 0.5
    return {
        "a_n": len(a),
        "b_n": len(b),
        "a_median": r1(median(a)),
        "b_median": r1(median(b)),
        "delta_median": r1(median(a) - median(b)),
        "a_mean": r1(safe_mean(a)),
        "b_mean": r1(safe_mean(b)),
        "delta_mean": r1((safe_mean(a) or 0.0) - (safe_mean(b) or 0.0)),
        "auc_a_gt_b": r3(wins / total) if total else None,
    }


def mark_ret(conn: sqlite3.Connection, symbol: str, start_ms: int, end_ms: int) -> float | None:
    a = mark_at_or_after(conn, symbol, int(start_ms))
    b = mark_at_or_after(conn, symbol, int(end_ms))
    if not a or not b or float(a[1]) <= 0:
        return None
    return (float(b[1]) - float(a[1])) / float(a[1]) * 10_000.0


def book_at_or_before(conn: sqlite3.Connection, ts_ms: int, max_stale_ms: int = 2000) -> dict[str, Any] | None:
    row = conn.execute(
        """
        SELECT ts_ms, bid_price, bid_qty, ask_price, ask_qty, mid_price, book_imbalance
        FROM book_ticker
        WHERE symbol=? AND ts_ms<=?
        ORDER BY ts_ms DESC
        LIMIT 1
        """,
        (SYMBOL, int(ts_ms)),
    ).fetchone()
    if not row:
        return None
    stale = int(ts_ms) - int(row[0])
    if stale > max_stale_ms:
        return None
    bid = float(row[1])
    bid_qty = float(row[2])
    ask = float(row[3])
    ask_qty = float(row[4])
    mid = float(row[5])
    if mid <= 0 or bid <= 0 or ask <= 0:
        return None
    denom = bid_qty + ask_qty
    micro = (ask * bid_qty + bid * ask_qty) / denom if denom > 0 else mid
    return {
        "book_ts_ms": int(row[0]),
        "staleness_ms": stale,
        "bid_qty": bid_qty,
        "ask_qty": ask_qty,
        "bid_notional": bid * bid_qty,
        "ask_notional": ask * ask_qty,
        "mid": mid,
        "spread_bps": (ask - bid) / mid * 10_000.0,
        "book_imbalance": float(row[6]),
        "micro_minus_mid_bps": (micro - mid) / mid * 10_000.0,
    }


def book_delta(conn: sqlite3.Connection, t0: int, t1: int, prefix: str) -> dict[str, float | None]:
    a = book_at_or_before(conn, t0)
    b = book_at_or_before(conn, t1)
    keys = ["bid_notional", "ask_notional", "spread_bps", "book_imbalance", "micro_minus_mid_bps", "bid_qty", "ask_qty"]
    out: dict[str, float | None] = {}
    for key in keys:
        out[f"{prefix}_{key}_start"] = r1(a.get(key)) if a else None
        out[f"{prefix}_{key}_end"] = r1(b.get(key)) if b else None
        out[f"{prefix}_{key}_delta"] = r1(float(b[key]) - float(a[key])) if a and b else None
    return out


def trade_flow(conn: sqlite3.Connection, start_ms: int, end_ms: int, prefix: str) -> dict[str, Any]:
    rows = conn.execute(
        """
        SELECT is_buyer_maker, COALESCE(SUM(notional),0.0), COUNT(*)
        FROM agg_trades
        WHERE symbol=? AND ts_ms>=? AND ts_ms<?
        GROUP BY is_buyer_maker
        """,
        (SYMBOL, int(start_ms), int(end_ms)),
    ).fetchall()
    taker_buy = 0.0
    taker_sell = 0.0
    count = 0
    for is_buyer_maker, notion, c in rows:
        count += int(c or 0)
        if int(is_buyer_maker) == 0:
            taker_buy += float(notion or 0.0)
        else:
            taker_sell += float(notion or 0.0)
    total = taker_buy + taker_sell
    return {
        f"{prefix}_taker_buy_notional": r1(taker_buy),
        f"{prefix}_taker_sell_notional": r1(taker_sell),
        f"{prefix}_taker_flow_imbalance": r3((taker_buy - taker_sell) / total) if total > 0 else None,
        f"{prefix}_agg_trade_count": count,
    }


def liq_flow(conn: sqlite3.Connection, start_ms: int, end_ms: int, prefix: str) -> dict[str, Any]:
    rows = conn.execute(
        """
        SELECT side, COALESCE(SUM(notional),0.0), COUNT(*)
        FROM liquidations
        WHERE symbol=? AND ts_ms>=? AND ts_ms<?
        GROUP BY side
        """,
        (SYMBOL, int(start_ms), int(end_ms)),
    ).fetchall()
    buy = 0.0
    sell = 0.0
    count = 0
    for side, notion, c in rows:
        count += int(c or 0)
        if str(side).upper() == "BUY":
            buy += float(notion or 0.0)
        elif str(side).upper() == "SELL":
            sell += float(notion or 0.0)
    return {
        f"{prefix}_sell_liq_notional": r1(sell),
        f"{prefix}_buy_liq_notional": r1(buy),
        f"{prefix}_liq_count": count,
    }


def window_features(conn: sqlite3.Connection, anchor_ms: int, ref_ms: int, prefix: str) -> dict[str, Any]:
    # Three nested pre-onset windows.
    windows = {
        "w10_5": (-10_000, -5_000),
        "w5_2": (-5_000, -2_000),
        "w2_0": (-2_000, 0),
        "w5_0": (-5_000, 0),
    }
    out: dict[str, Any] = {}
    for name, (a, b) in windows.items():
        s = ref_ms + a
        e = ref_ms + b
        p = f"{prefix}_{name}"
        out.update(book_delta(conn, s, e, p))
        out.update(trade_flow(conn, s, e, p))
        out.update(liq_flow(conn, s, e, p))
        out[f"{p}_eth_ret_bps"] = r1(mark_ret(conn, SYMBOL, s, e))
        out[f"{p}_btc_ret_bps"] = r1(mark_ret(conn, BTC, s, e))
    # Changes across the whole pre-onset horizon.
    out.update(book_delta(conn, ref_ms - 5_000, ref_ms, f"{prefix}_pre5"))
    out[f"{prefix}_anchor_to_ref_sec"] = r1((ref_ms - anchor_ms) / 1000.0)
    return out


def find_onset(conn: sqlite3.Connection, anchor_ms: int, threshold_bps: float) -> dict[str, Any]:
    entry = mark_at_or_after(conn, SYMBOL, anchor_ms)
    if not entry:
        return {"onset_ts_ms": None, "onset_delay_sec": None, "local_low_bps": None}
    entry_ts, entry_px = int(entry[0]), float(entry[1])
    series = mark_series(conn, entry_ts, entry_ts + MAX_ONSET_SEC * 1000)
    if not series:
        return {"onset_ts_ms": None, "onset_delay_sec": None, "local_low_bps": None}
    low_px = entry_px
    low_ts = entry_ts
    for t, px in series:
        px = float(px)
        if px < low_px:
            low_px = px
            low_ts = int(t)
        if low_px > 0 and (px - low_px) / low_px * 10_000.0 >= threshold_bps:
            return {
                "onset_ts_ms": int(t),
                "onset_utc": ts_iso(int(t)),
                "onset_delay_sec": r1((int(t) - entry_ts) / 1000.0),
                "local_low_ts_ms": low_ts,
                "local_low_delay_sec": r1((low_ts - entry_ts) / 1000.0),
                "local_low_bps": r1((low_px - entry_px) / entry_px * 10_000.0),
            }
    return {
        "onset_ts_ms": None,
        "onset_utc": None,
        "onset_delay_sec": None,
        "local_low_ts_ms": low_ts,
        "local_low_delay_sec": r1((low_ts - entry_ts) / 1000.0),
        "local_low_bps": r1((low_px - entry_px) / entry_px * 10_000.0),
    }


def net_long(conn: sqlite3.Connection, anchor_ms: int, sec: int) -> float | None:
    a = mark_at_or_after(conn, SYMBOL, anchor_ms)
    b = mark_at_or_after(conn, SYMBOL, anchor_ms + sec * 1000)
    if not a or not b or float(a[1]) <= 0:
        return None
    return (float(b[1]) - float(a[1])) / float(a[1]) * 10_000.0 - FEE_BPS


def build_v02_events(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    marks = load_mark_index(conn, SYMBOL)
    rows = []
    for event in collect_v01_events(conn):
        anchor_ms = int(event.anchor.anchor_ts_ms)
        prior4h = prior_return_bps(marks, anchor_ms, 4 * 3600)
        if prior4h is None or not math.isfinite(float(prior4h)):
            continue
        book = book_features_at(conn, SYMBOL, anchor_ms, 10)
        if not book or float(book.get("bid_depth_usd") or 0.0) < MIN_BID_DEPTH_USD:
            continue
        row = {
            "event_id": f"V02:{event.anchor.bucket}:{anchor_ms}",
            "anchor_ts_ms": anchor_ms,
            "anchor_utc": ts_iso(anchor_ms),
            "bucket": int(event.anchor.bucket),
            "anchor_price": float(event.anchor_mark_price),
            "vdepth_bps": r1(float(event.vdepth_bps)),
            "prior4h_bps": r1(float(prior4h)),
            "bid_depth_usd": r1(float(book.get("bid_depth_usd") or 0.0)),
            "book_imbalance": r3(float(book.get("book_imbalance") or 0.0)),
            "net_15m_bps": r1(net_long(conn, anchor_ms, 900)),
            "net_30m_bps": r1(net_long(conn, anchor_ms, 1800)),
            "net_2h_bps": r1(net_long(conn, anchor_ms, 7200)),
        }
        row["outcome_group_2h"] = "GOOD" if (row["net_2h_bps"] is not None and float(row["net_2h_bps"]) > 0) else "BAD"
        row["tail_2h"] = bool(row["net_2h_bps"] is not None and float(row["net_2h_bps"]) <= -100.0)
        for th in MOMENTUM_THRESHOLDS:
            onset = find_onset(conn, anchor_ms, th)
            row[f"onset{int(th)}"] = onset
            if onset.get("onset_ts_ms") is not None:
                row.update(window_features(conn, anchor_ms, int(onset["onset_ts_ms"]), f"onset{int(th)}"))
            else:
                # Use anchor+60s as a negative-control reference for no-onset rows.
                row.update(window_features(conn, anchor_ms, anchor_ms + 60_000, f"no_onset{int(th)}_ref60"))
        rows.append(row)
    rows.sort(key=lambda r: int(r["anchor_ts_ms"]))
    return rows


def profile(rows: list[dict[str, Any]], keys: list[str]) -> dict[str, Any]:
    out = {"n": len(rows)}
    for key in keys:
        vals = [float(r[key]) for r in rows if r.get(key) is not None and math.isfinite(float(r[key]))]
        out[key] = {
            "n": len(vals),
            "median": r1(median(vals)) if vals else None,
            "mean": r1(safe_mean(vals)) if vals else None,
        }
    return out


def compare_groups(rows: list[dict[str, Any]], prefix: str) -> dict[str, Any]:
    keys = [
        f"{prefix}_pre5_bid_notional_delta",
        f"{prefix}_pre5_ask_notional_delta",
        f"{prefix}_pre5_spread_bps_delta",
        f"{prefix}_pre5_book_imbalance_delta",
        f"{prefix}_pre5_micro_minus_mid_bps_delta",
        f"{prefix}_w5_0_taker_flow_imbalance",
        f"{prefix}_w5_0_taker_buy_notional",
        f"{prefix}_w5_0_taker_sell_notional",
        f"{prefix}_w5_0_sell_liq_notional",
        f"{prefix}_w5_0_eth_ret_bps",
        f"{prefix}_w5_0_btc_ret_bps",
        f"{prefix}_w2_0_taker_flow_imbalance",
        f"{prefix}_w2_0_sell_liq_notional",
        f"{prefix}_w2_0_eth_ret_bps",
        f"{prefix}_anchor_to_ref_sec",
    ]
    onset_rows = [r for r in rows if r.get(f"{prefix}_anchor_to_ref_sec") is not None and r.get(prefix, {}).get("onset_ts_ms") is not None]
    good = [r for r in onset_rows if r.get("outcome_group_2h") == "GOOD"]
    bad = [r for r in onset_rows if r.get("outcome_group_2h") == "BAD"]
    out = {
        "onset_n": len(onset_rows),
        "good_n": len(good),
        "bad_n": len(bad),
        "good_profile": profile(good, keys),
        "bad_profile": profile(bad, keys),
        "effects_good_minus_bad": {},
    }
    for key in keys:
        a = [float(r[key]) for r in good if r.get(key) is not None and math.isfinite(float(r[key]))]
        b = [float(r[key]) for r in bad if r.get(key) is not None and math.isfinite(float(r[key]))]
        out["effects_good_minus_bad"][key] = effect(a, b)
    return out


def onset_coverage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out = {}
    for th in MOMENTUM_THRESHOLDS:
        key = f"onset{int(th)}"
        all_delays = [float(r[key]["onset_delay_sec"]) for r in rows if r.get(key, {}).get("onset_delay_sec") is not None]
        out[key] = {
            "n": len(rows),
            "onset_n": len(all_delays),
            "coverage": r3(len(all_delays) / len(rows)) if rows else None,
            "delay_median_sec": r1(median(all_delays)) if all_delays else None,
            "delay_p25_sec": r1(sorted(all_delays)[int(0.25 * (len(all_delays) - 1))]) if all_delays else None,
            "delay_p75_sec": r1(sorted(all_delays)[int(0.75 * (len(all_delays) - 1))]) if all_delays else None,
        }
    return out


def feature_rank(effects: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for key, row in effects.items():
        auc = row.get("auc_a_gt_b")
        if auc is None:
            continue
        rows.append(
            {
                "feature": key,
                "auc_distance": r3(abs(float(auc) - 0.5)),
                "auc_good_gt_bad": auc,
                "delta_median": row.get("delta_median"),
                "good_median": row.get("a_median"),
                "bad_median": row.get("b_median"),
            }
        )
    rows.sort(key=lambda r: float(r["auc_distance"] or 0.0), reverse=True)
    return rows


def simple_indicator_screens(rows: list[dict[str, Any]], prefix: str) -> dict[str, Any]:
    # Designed as dashboard indicators, not entry filters. Summaries use 2h LONG outcome.
    screens: dict[str, Any] = {
        "BID_REPLENISH_5S": lambda r: float(r.get(f"{prefix}_pre5_bid_notional_delta") or 0.0) > 0,
        "SPREAD_COMPRESS_5S": lambda r: float(r.get(f"{prefix}_pre5_spread_bps_delta") or 0.0) < 0,
        "MICROPRICE_UP_5S": lambda r: float(r.get(f"{prefix}_pre5_micro_minus_mid_bps_delta") or 0.0) > 0,
        "TAKER_BUY_DOMINANT_5S": lambda r: float(r.get(f"{prefix}_w5_0_taker_flow_imbalance") or 0.0) > 0,
        "SELL_LIQ_QUIET_5S": lambda r: float(r.get(f"{prefix}_w5_0_sell_liq_notional") or 0.0) <= 1.0,
        "BTC_UP_5S": lambda r: float(r.get(f"{prefix}_w5_0_btc_ret_bps") or 0.0) > 0,
        "ETH_UP_5S": lambda r: float(r.get(f"{prefix}_w5_0_eth_ret_bps") or 0.0) > 0,
    }
    out = {}
    onset_rows = [r for r in rows if r.get(f"{prefix}_anchor_to_ref_sec") is not None and r.get(prefix, {}).get("onset_ts_ms") is not None]
    for name, fn in screens.items():
        yes = [r for r in onset_rows if fn(r)]
        no = [r for r in onset_rows if not fn(r)]
        out[name] = {
            "yes_n": len(yes),
            "yes_2h": summary([float(r["net_2h_bps"]) for r in yes if r.get("net_2h_bps") is not None]),
            "no_n": len(no),
            "no_2h": summary([float(r["net_2h_bps"]) for r in no if r.get("net_2h_bps") is not None]),
        }
    return out


def event_cards(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scored = sorted([r for r in rows if r.get("net_2h_bps") is not None], key=lambda r: float(r["net_2h_bps"]))
    def card(r: dict[str, Any]) -> dict[str, Any]:
        return {
            "event_id": r["event_id"],
            "anchor_utc": r["anchor_utc"],
            "net_2h_bps": r["net_2h_bps"],
            "vdepth_bps": r["vdepth_bps"],
            "bid_depth_usd": r["bid_depth_usd"],
            "onset20_delay": r.get("onset20", {}).get("onset_delay_sec"),
            "onset40_delay": r.get("onset40", {}).get("onset_delay_sec"),
            "pre5_bid_delta": r.get("onset20_pre5_bid_notional_delta"),
            "pre5_micro_delta": r.get("onset20_pre5_micro_minus_mid_bps_delta"),
            "w5_flow_imb": r.get("onset20_w5_0_taker_flow_imbalance"),
            "w5_sell_liq": r.get("onset20_w5_0_sell_liq_notional"),
            "w5_btc_ret": r.get("onset20_w5_0_btc_ret_bps"),
        }
    return {"worst10": [card(r) for r in scored[:10]], "best10": [card(r) for r in reversed(scored[-10:])]}


def run() -> dict[str, Any]:
    with sqlite3.connect(f"file:{DEFAULT_DB}?mode=ro", uri=True) as conn:
        rows = build_v02_events(conn)
    comparisons = {f"onset{int(th)}": compare_groups(rows, f"onset{int(th)}") for th in MOMENTUM_THRESHOLDS}
    return {
        "generated_at_utc": utc_now(),
        "status": "RESEARCH_ONLY_NO_LIVE_CHANGE",
        "rule": "S34_V_ENGINE_V0_2_ETH_SELL_MAKER_LONG_H2_O20_W300_O5_DEEPBID",
        "event_n": len(rows),
        "outcome_2h": summary([float(r["net_2h_bps"]) for r in rows if r.get("net_2h_bps") is not None]),
        "onset_coverage": onset_coverage(rows),
        "comparisons": comparisons,
        "feature_rank_onset20": feature_rank(comparisons["onset20"]["effects_good_minus_bad"]),
        "feature_rank_onset40": feature_rank(comparisons["onset40"]["effects_good_minus_bad"]),
        "indicator_screens_onset20": simple_indicator_screens(rows, "onset20"),
        "indicator_screens_onset40": simple_indicator_screens(rows, "onset40"),
        "event_cards": event_cards(rows),
    }


def fmt(s: dict[str, Any]) -> str:
    return (
        f"N={s.get('n')} sum={s.get('sum_bps')} med={s.get('median_bps')} "
        f"T3R={s.get('t3r_bps') or s.get('top3_winner_removed_sum_bps')} maxLoss={s.get('max_loss_bps')}"
    )


def write_report(result: dict[str, Any]) -> None:
    lines = [
        "# S34 v0.2 Momentum Precursor Tests",
        "",
        f"Generated: `{result['generated_at_utc']}`",
        "",
        f"Status: `{result['status']}`",
        "",
        f"Rule: `{result['rule']}`",
        "",
        f"Events: `{result['event_n']}`",
        "",
        f"2h anchor-mark outcome: {fmt(result['outcome_2h'])}",
        "",
        "## 1. Momentum Onset Coverage",
        "",
    ]
    for name, row in result["onset_coverage"].items():
        lines.append(f"- `{name}`: `{row}`")

    lines.extend(["", "## 2. Winner vs Loser Pre-Onset Effects", ""])
    for name, comp in result["comparisons"].items():
        lines.append(f"### `{name}`")
        lines.append(f"- onset N `{comp['onset_n']}`, good N `{comp['good_n']}`, bad N `{comp['bad_n']}`")
        lines.append("| Feature | Good median | Bad median | Delta median | AUC good>bad |")
        lines.append("| --- | ---: | ---: | ---: | ---: |")
        for row in feature_rank(comp["effects_good_minus_bad"])[:15]:
            lines.append(
                f"| `{row['feature']}` | {row['good_median']} | {row['bad_median']} | {row['delta_median']} | {row['auc_good_gt_bad']} |"
            )
        lines.append("")

    lines.extend(["", "## 3. Simple Indicator Screens (onset20)", ""])
    lines.append("| Indicator | Yes N | Yes 2h | No N | No 2h |")
    lines.append("| --- | ---: | --- | ---: | --- |")
    for name, row in result["indicator_screens_onset20"].items():
        lines.append(f"| `{name}` | {row['yes_n']} | {fmt(row['yes_2h'])} | {row['no_n']} | {fmt(row['no_2h'])} |")

    lines.extend(["", "## 4. Simple Indicator Screens (onset40)", ""])
    lines.append("| Indicator | Yes N | Yes 2h | No N | No 2h |")
    lines.append("| --- | ---: | --- | ---: | --- |")
    for name, row in result["indicator_screens_onset40"].items():
        lines.append(f"| `{name}` | {row['yes_n']} | {fmt(row['yes_2h'])} | {row['no_n']} | {fmt(row['no_2h'])} |")

    lines.extend(["", "## 5. Event Cards", ""])
    lines.append("Worst 10:")
    for row in result["event_cards"]["worst10"]:
        lines.append(f"- `{row}`")
    lines.append("Best 10:")
    for row in result["event_cards"]["best10"]:
        lines.append(f"- `{row}`")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    result = run()
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    write_report(result)
    print(OUT_MD.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
