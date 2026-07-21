"""S34 v0.2 entry-quality navigation tests.

Research-only. Profiles the current frozen v0.2 lifecycle fills:

S34_V_ENGINE_V0_2_ETH_SELL_MAKER_LONG_H2_O20_W300_O5_DEEPBID

The goal is not to change the alpha. It is to ask whether the current fills
occur during a healthier navigation state: pre/post arming, healthy retest,
BTC/ETH confirmation, sell-liq quiet, taker-buy impulse, depth still present,
and spread clean.

No live executor, order logic, size, leverage, config, runtime state, or .env
changes.
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

from tools.research_s34_wave_absorption import book_features_at  # noqa: E402
from tools.s34_navigation_full_followup import DEFAULT_DB, mark_at_or_after, summary  # noqa: E402
from tools.s34_stress_reaction_deep_tests import mark_series  # noqa: E402
from tools.s34_v02_momentum_arming_live_like import (  # noqa: E402
    ARMING_CONFIGS,
    BTC,
    CROSS_MARGIN_BPS,
    HORIZONS,
    MAKER_FEE_BPS,
    MAX_BOOK_STALENESS_SEC,
    REPLACE_OFFSET_BPS,
    SYMBOL,
    TAKER_FEE_BPS,
    WAIT_SEC,
    arming_snapshot,
    build_v02_events,
    mark_ret_bps,
    scan_first_arming,
    sell_liq_5s,
    trade_flow_5s,
)
from tools.s34_v_engine_cancel_replace import simulate_cancel_replace  # noqa: E402
from tools.s34_v_engine_v02_shadow_mirror import INITIAL_OFFSET_BPS  # noqa: E402

OUT_JSON = ROOT / "reports" / "research" / "s34" / "S34_V02_ENTRY_QUALITY_NAVIGATION.json"
OUT_MD = ROOT / "reports" / "research" / "s34" / "S34_V02_ENTRY_QUALITY_NAVIGATION.md"


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


def mark_net_from_price(conn: sqlite3.Connection, start_ms: int, entry_px: float, horizon_sec: int) -> float | None:
    exit_ = mark_at_or_after(conn, SYMBOL, int(start_ms) + int(horizon_sec) * 1000)
    if not exit_ or float(entry_px) <= 0:
        return None
    return (float(exit_[1]) - float(entry_px)) / float(entry_px) * 10_000.0 - (MAKER_FEE_BPS + TAKER_FEE_BPS)


def flow_window(conn: sqlite3.Connection, end_ms: int, sec: int) -> dict[str, Any]:
    start_ms = int(end_ms) - int(sec) * 1000
    out = trade_flow_5s(conn, start_ms, int(end_ms))
    out["sell_liq_usd"] = sell_liq_5s(conn, start_ms, int(end_ms))
    out["eth_ret_bps"] = mark_ret_bps(conn, SYMBOL, start_ms, int(end_ms))
    out["btc_ret_bps"] = mark_ret_bps(conn, BTC, start_ms, int(end_ms))
    return out


def rel_change(a: float | None, b: float | None) -> float | None:
    if a is None or b is None or not math.isfinite(float(a)) or float(a) == 0:
        return None
    return (float(b) - float(a)) / float(a)


def price_ret_bps(a: float | None, b: float | None) -> float | None:
    if a is None or b is None or not math.isfinite(float(a)) or float(a) <= 0:
        return None
    return (float(b) - float(a)) / float(a) * 10_000.0


def score_row(row: dict[str, Any]) -> tuple[int, list[str], list[str], int, list[str], list[str]]:
    score = 0
    pos: list[str] = []
    neg: list[str] = []
    retest_score = 0
    retest_pos: list[str] = []
    retest_neg: list[str] = []

    def add(cond: bool, tag: str, points: int = 1) -> None:
        nonlocal score
        if cond:
            score += points
            pos.append(tag)

    def sub(cond: bool, tag: str, points: int = 1) -> None:
        nonlocal score
        if cond:
            score -= points
            neg.append(tag)

    add(row.get("phase") == "POST_ARM_FILL", "POST_ARM_FILL")
    add(float(row.get("btc_ret_15s_bps") or -1e9) > 0, "BTC_15S_UP")
    add(float(row.get("eth_ret_15s_bps") or -1e9) > 0, "ETH_15S_UP")
    add(float(row.get("sell_liq_15s_usd") or 0.0) <= 1.0, "SELL_LIQ_QUIET_15S")
    add(float(row.get("taker_imbalance_15s") or -1e9) > 0, "TAKER_BUY_IMPULSE_15S")
    add(float(row.get("bid_depth_fill_usd") or 0.0) >= 135_423.8, "BID_STILL_THERE")
    add(float(row.get("bid_depth_fill_vs_anchor") or -1e9) >= 0.8, "BID_DEPTH_RETAINED")
    add(float(row.get("spread_fill_bps") or 1e9) <= float(row.get("spread_anchor_bps") or -1e9) + 0.05, "SPREAD_CLEAN")
    add(float(row.get("entry_vs_arm_bps") or 1e9) <= 0.0, "RETEST_NOT_CHASE")

    sub(float(row.get("sell_liq_15s_usd") or 0.0) > 1.0, "SELL_LIQ_RESTART_15S")
    sub(float(row.get("btc_ret_15s_bps") or 0.0) < -2.0, "BTC_NOT_CONFIRMING")
    sub(float(row.get("spread_fill_bps") or 0.0) > float(row.get("spread_anchor_bps") or 0.0) + 0.2, "SPREAD_EXPANDING")
    sub(float(row.get("bid_depth_fill_vs_anchor") or 1.0) < 0.5, "BID_VANISHED")
    sub(float(row.get("arm_to_fill_min_bps") or 0.0) < -20.0, "SECOND_FLUSH")

    def radd(cond: bool, tag: str, points: int = 1) -> None:
        nonlocal retest_score
        if cond:
            retest_score += points
            retest_pos.append(tag)

    def rsub(cond: bool, tag: str, points: int = 1) -> None:
        nonlocal retest_score
        if cond:
            retest_score -= points
            retest_neg.append(tag)

    # Retest-specific score: after arming, a good entry can have short-term red
    # returns. Penalize uncontrolled liquidation/depth/spread, not the fact that
    # the passive limit filled on a pullback.
    radd(row.get("phase") == "POST_ARM_FILL", "POST_ARM_FILL")
    radd(float(row.get("entry_vs_arm_bps") or 1e9) <= 0.0, "PULLBACK_FILL")
    radd(-25.0 <= float(row.get("entry_vs_arm_bps") or -1e9) <= -2.0, "RETEST_BAND_2_25")
    radd(float(row.get("arm_to_fill_min_bps") or 0.0) >= -25.0, "NO_DEEP_SECOND_FLUSH")
    radd(float(row.get("bid_depth_fill_usd") or 0.0) >= 135_423.8, "BID_STILL_THERE")
    radd(float(row.get("bid_depth_fill_vs_anchor") or -1e9) >= 0.8, "BID_DEPTH_RETAINED")
    radd(float(row.get("spread_fill_bps") or 1e9) <= float(row.get("spread_anchor_bps") or -1e9) + 0.05, "SPREAD_CLEAN")
    radd(float(row.get("fill_minus_arm_sec") or 1e9) <= 300.0, "FAST_RETEST_FILL")
    radd(float(row.get("sell_liq_15s_usd") or 0.0) <= 250_000.0, "NO_LARGE_SELL_LIQ_RESTART")

    rsub(float(row.get("arm_to_fill_min_bps") or 0.0) < -25.0, "DEEP_SECOND_FLUSH")
    rsub(float(row.get("bid_depth_fill_vs_anchor") or 1.0) < 0.5, "BID_VANISHED")
    rsub(float(row.get("spread_fill_bps") or 0.0) > float(row.get("spread_anchor_bps") or 0.0) + 0.2, "SPREAD_EXPANDING")
    rsub(float(row.get("fill_minus_arm_sec") or 0.0) > 900.0, "LATE_RETEST_FILL")
    rsub(float(row.get("sell_liq_15s_usd") or 0.0) > 250_000.0, "LARGE_SELL_LIQ_RESTART")

    return score, pos, neg, retest_score, retest_pos, retest_neg


def score_bucket(score: int) -> str:
    if score >= 7:
        return "ENTRY_QUALITY_HIGH"
    if score >= 5:
        return "ENTRY_QUALITY_MID"
    return "ENTRY_QUALITY_LOW"


def retest_score_bucket(score: int) -> str:
    if score >= 7:
        return "RETEST_QUALITY_HIGH"
    if score >= 5:
        return "RETEST_QUALITY_MID"
    return "RETEST_QUALITY_LOW"


def retest_depth_bucket(v: float | None) -> str:
    if v is None:
        return "UNKNOWN"
    x = float(v)
    if x <= -20.0:
        return "DEEP_RETEST_GE20"
    if x <= -10.0:
        return "MID_RETEST_10_20"
    if x <= -2.0:
        return "LIGHT_RETEST_2_10"
    if x <= 0.0:
        return "TOUCH_RETEST_0_2"
    return "CHASE_ABOVE_ARM"


def fill_delay_bucket(v: float | None) -> str:
    if v is None:
        return "UNKNOWN"
    x = float(v)
    if x <= 60.0:
        return "FAST_0_60S"
    if x <= 300.0:
        return "NORMAL_60_300S"
    if x <= 900.0:
        return "SLOW_300_900S"
    return "LATE_GT900S"


def group(rows: list[dict[str, Any]], key: str, metric: str = "net_2h_bps") -> dict[str, Any]:
    out: dict[str, Any] = {}
    buckets = sorted({"UNKNOWN" if r.get(key) is None else str(r.get(key)) for r in rows})
    for bucket in buckets:
        vals = [
            float(r[metric])
            for r in rows
            if ("UNKNOWN" if r.get(key) is None else str(r.get(key))) == bucket and r.get(metric) is not None
        ]
        out[bucket] = summary(vals)
    return out


def tag_group(
    rows: list[dict[str, Any]],
    tag: str,
    metric: str = "net_2h_bps",
    fields: tuple[str, ...] = ("positive_tags", "negative_tags"),
) -> dict[str, Any]:
    yes = [r for r in rows if any(tag in set(r.get(field) or []) for field in fields)]
    no = [r for r in rows if r not in yes]
    return {
        "yes_n": len(yes),
        "yes": summary([float(r[metric]) for r in yes if r.get(metric) is not None]),
        "no_n": len(no),
        "no": summary([float(r[metric]) for r in no if r.get(metric) is not None]),
    }


def chronological_split(rows: list[dict[str, Any]], metric: str = "net_2h_bps") -> dict[str, Any]:
    vals = [(int(r["anchor_ts_ms"]), float(r[metric])) for r in rows if r.get(metric) is not None]
    vals.sort(key=lambda x: x[0])
    if not vals:
        return {"cal": summary([]), "hold": summary([])}
    cut = max(1, int(len(vals) * 0.6))
    return {
        "cal": summary([v for _, v in vals[:cut]]),
        "hold": summary([v for _, v in vals[cut:]]),
    }


def build_rows(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    cfg_by_name = {cfg.name: cfg for cfg in ARMING_CONFIGS}
    arming_cfg = cfg_by_name["FLOW_POSITIVE_ONLY"]
    rows: list[dict[str, Any]] = []
    for event in build_v02_events(conn):
        anchor_ts = int(event.anchor.anchor_ts_ms)
        anchor_book = book_features_at(conn, SYMBOL, anchor_ts, MAX_BOOK_STALENESS_SEC)
        arm = scan_first_arming(conn, anchor_ts, arming_cfg)
        sim = simulate_cancel_replace(
            conn,
            event,
            initial_offset_bps=INITIAL_OFFSET_BPS,
            replace_offset_bps=REPLACE_OFFSET_BPS,
            wait_sec=WAIT_SEC,
            cross_margin_bps=CROSS_MARGIN_BPS,
            maker_fee_bps=MAKER_FEE_BPS,
            taker_fee_bps=TAKER_FEE_BPS,
            max_book_staleness_sec=MAX_BOOK_STALENESS_SEC,
        )
        if sim.get("status") != "FILLED" or sim.get("maker_fill_ts_ms") is None:
            continue
        fill_ts = int(sim["maker_fill_ts_ms"])
        entry_px = float(sim["entry_price"])
        fill_book = book_features_at(conn, SYMBOL, fill_ts, MAX_BOOK_STALENESS_SEC)
        arm_ts = int(arm["ts_ms"]) if arm else None
        arm_px = None
        arm_snapshot = None
        if arm_ts is not None:
            arm_mark = mark_at_or_after(conn, SYMBOL, arm_ts)
            arm_px = float(arm_mark[1]) if arm_mark else None
            arm_snapshot = arming_snapshot(conn, arm_ts)

        if arm_ts is None:
            phase = "NO_ARM_FILL"
        elif fill_ts < arm_ts:
            phase = "PRE_ARM_FILL"
        else:
            phase = "POST_ARM_FILL"

        series = mark_series(conn, arm_ts, fill_ts) if arm_ts is not None and fill_ts >= arm_ts else []
        arm_to_fill_min_bps = None
        if series and arm_px:
            arm_to_fill_min_bps = min(price_ret_bps(arm_px, px) for _, px in series if price_ret_bps(arm_px, px) is not None)

        row: dict[str, Any] = {
            "event_id": f"V02:{event.anchor.bucket}:{anchor_ts}",
            "anchor_ts_ms": anchor_ts,
            "anchor_utc": iso_ms(anchor_ts),
            "vdepth_bps": r1(float(event.vdepth_bps)),
            "fill_leg": sim.get("fill_leg"),
            "fill_ts_ms": fill_ts,
            "fill_utc": iso_ms(fill_ts),
            "fill_delay_sec": r1(float(sim.get("fill_delay_sec") or 0.0)),
            "entry_price": r1(entry_px),
            "net_2h_bps": r1(float(sim.get("net_bps"))),
            "phase": phase,
            "arming_ts_ms": arm_ts,
            "arming_utc": iso_ms(arm_ts) if arm_ts is not None else None,
            "arming_delay_sec": int(arm["delay_sec"]) if arm else None,
            "fill_minus_arm_sec": r1((fill_ts - arm_ts) / 1000.0) if arm_ts is not None else None,
            "entry_vs_arm_bps": r1(price_ret_bps(arm_px, entry_px)) if arm_px else None,
            "arm_to_fill_min_bps": r1(arm_to_fill_min_bps),
            "arm_eth_5s_bps": r1(arm_snapshot.get("eth_ret_bps")) if arm_snapshot else None,
            "arm_btc_5s_bps": r1(arm_snapshot.get("btc_ret_bps")) if arm_snapshot else None,
            "arm_taker_imbalance": r3(arm_snapshot.get("taker_imbalance")) if arm_snapshot else None,
            "arm_sell_liq_usd": r1(arm_snapshot.get("sell_liq_usd")) if arm_snapshot else None,
        }
        for sec in (5, 15, 30):
            fw = flow_window(conn, fill_ts, sec)
            row[f"eth_ret_{sec}s_bps"] = r1(fw.get("eth_ret_bps"))
            row[f"btc_ret_{sec}s_bps"] = r1(fw.get("btc_ret_bps"))
            row[f"sell_liq_{sec}s_usd"] = r1(fw.get("sell_liq_usd"))
            row[f"taker_buy_{sec}s_usd"] = r1(fw.get("taker_buy_usd"))
            row[f"taker_imbalance_{sec}s"] = r3(fw.get("taker_imbalance"))
        if anchor_book:
            row["bid_depth_anchor_usd"] = r1(float(anchor_book.get("bid_depth_usd") or 0.0))
            row["spread_anchor_bps"] = r1(float(anchor_book.get("spread_bps") or 0.0))
            row["imbalance_anchor"] = r3(float(anchor_book.get("book_imbalance") or 0.0))
        if fill_book:
            row["bid_depth_fill_usd"] = r1(float(fill_book.get("bid_depth_usd") or 0.0))
            row["spread_fill_bps"] = r1(float(fill_book.get("spread_bps") or 0.0))
            row["imbalance_fill"] = r3(float(fill_book.get("book_imbalance") or 0.0))
        row["bid_depth_fill_vs_anchor"] = r3(rel_change(row.get("bid_depth_anchor_usd"), row.get("bid_depth_fill_usd")) + 1.0) if row.get("bid_depth_anchor_usd") and row.get("bid_depth_fill_usd") else None
        row["spread_fill_minus_anchor_bps"] = r1(float(row.get("spread_fill_bps") or 0.0) - float(row.get("spread_anchor_bps") or 0.0))

        for label, sec in HORIZONS.items():
            row[f"net_{label}_bps"] = r1(mark_net_from_price(conn, fill_ts, entry_px, sec))

        score, pos, neg, retest_score, retest_pos, retest_neg = score_row(row)
        row["entry_quality_score"] = score
        row["entry_quality_bucket"] = score_bucket(score)
        row["positive_tags"] = pos
        row["negative_tags"] = neg
        row["retest_quality_score"] = retest_score
        row["retest_quality_bucket"] = retest_score_bucket(retest_score)
        row["retest_positive_tags"] = retest_pos
        row["retest_negative_tags"] = retest_neg
        row["retest_depth_bucket"] = retest_depth_bucket(row.get("entry_vs_arm_bps"))
        row["fill_minus_arm_bucket"] = fill_delay_bucket(row.get("fill_minus_arm_sec"))
        row["healthy_retest"] = (
            row["phase"] == "POST_ARM_FILL"
            and float(row.get("entry_vs_arm_bps") or 1e9) <= 0.0
            and float(row.get("sell_liq_15s_usd") or 0.0) <= 250_000.0
            and float(row.get("bid_depth_fill_usd") or 0.0) >= 135_423.8
            and float(row.get("arm_to_fill_min_bps") or 0.0) >= -25.0
        )
        row["panic_retest"] = (
            row["phase"] == "POST_ARM_FILL"
            and (
                float(row.get("sell_liq_15s_usd") or 0.0) > 1.0
                or float(row.get("btc_ret_15s_bps") or 0.0) < -2.0
                or float(row.get("bid_depth_fill_vs_anchor") or 1.0) < 0.5
            )
        )
        rows.append(row)
    rows.sort(key=lambda r: int(r["anchor_ts_ms"]))
    return rows


def run() -> dict[str, Any]:
    with sqlite3.connect(f"file:{DEFAULT_DB}?mode=ro", uri=True) as conn:
        rows = build_rows(conn)

    metrics = ("net_30s_bps", "net_60s_bps", "net_5m_bps", "net_15m_bps", "net_2h_bps")
    by_metric = {m: summary([float(r[m]) for r in rows if r.get(m) is not None]) for m in metrics}
    tag_names = sorted({tag for r in rows for tag in list(r.get("positive_tags") or []) + list(r.get("negative_tags") or [])})
    retest_tag_names = sorted(
        {tag for r in rows for tag in list(r.get("retest_positive_tags") or []) + list(r.get("retest_negative_tags") or [])}
    )
    tag_tables = {tag: tag_group(rows, tag, "net_2h_bps") for tag in tag_names}
    retest_tag_tables = {
        tag: tag_group(rows, tag, "net_2h_bps", fields=("retest_positive_tags", "retest_negative_tags"))
        for tag in retest_tag_names
    }
    return {
        "generated_at_utc": utc_now(),
        "status": "RESEARCH_ONLY_NO_LIVE_CHANGE",
        "rule": "S34_V_ENGINE_V0_2_ETH_SELL_MAKER_LONG_H2_O20_W300_O5_DEEPBID",
        "filled_n": len(rows),
        "overall": by_metric,
        "phase_2h": group(rows, "phase", "net_2h_bps"),
        "fill_leg_2h": group(rows, "fill_leg", "net_2h_bps"),
        "quality_bucket_2h": group(rows, "entry_quality_bucket", "net_2h_bps"),
        "retest_quality_bucket_2h": group(rows, "retest_quality_bucket", "net_2h_bps"),
        "retest_depth_bucket_2h": group(rows, "retest_depth_bucket", "net_2h_bps"),
        "fill_minus_arm_bucket_2h": group(rows, "fill_minus_arm_bucket", "net_2h_bps"),
        "healthy_retest_2h": group(rows, "healthy_retest", "net_2h_bps"),
        "panic_retest_2h": group(rows, "panic_retest", "net_2h_bps"),
        "quality_bucket_60s": group(rows, "entry_quality_bucket", "net_60s_bps"),
        "quality_bucket_15m": group(rows, "entry_quality_bucket", "net_15m_bps"),
        "retest_quality_bucket_60s": group(rows, "retest_quality_bucket", "net_60s_bps"),
        "retest_quality_bucket_15m": group(rows, "retest_quality_bucket", "net_15m_bps"),
        "chronological_2h": chronological_split(rows, "net_2h_bps"),
        "tag_tables_2h": tag_tables,
        "retest_tag_tables_2h": retest_tag_tables,
        "rows": rows,
    }


def fmt(s: dict[str, Any]) -> str:
    return (
        f"N={s.get('n')} sum={s.get('sum_bps')} med={s.get('median_bps')} "
        f"WR={s.get('win_rate')} T3R={s.get('t3r_bps')} maxLoss={s.get('max_loss_bps')}"
    )


def write_group(lines: list[str], title: str, table: dict[str, Any]) -> None:
    lines.extend(["", f"## {title}", ""])
    lines.append("| Bucket | 2h result |")
    lines.append("| --- | --- |")
    for key, val in table.items():
        lines.append(f"| `{key}` | {fmt(val)} |")


def write_report(result: dict[str, Any]) -> None:
    lines = [
        "# S34 v0.2 Entry Quality Navigation",
        "",
        f"Generated: `{result['generated_at_utc']}`",
        "",
        f"Status: `{result['status']}`",
        "",
        f"Rule: `{result['rule']}`",
        "",
        f"Closed filled rows: `{result['filled_n']}`",
        "",
        "## 1. Overall Current Lifecycle Outcomes",
        "",
        "| Horizon | Result |",
        "| --- | --- |",
    ]
    for metric, val in result["overall"].items():
        lines.append(f"| `{metric}` | {fmt(val)} |")

    write_group(lines, "2. Fill Phase vs Arming", result["phase_2h"])
    write_group(lines, "3. Fill Leg", result["fill_leg_2h"])
    write_group(lines, "4. Entry Quality Score Buckets", result["quality_bucket_2h"])
    write_group(lines, "5. Retest Quality Score Buckets", result["retest_quality_bucket_2h"])
    write_group(lines, "6. Retest Depth Buckets", result["retest_depth_bucket_2h"])
    write_group(lines, "7. Fill Delay After Arming", result["fill_minus_arm_bucket_2h"])
    write_group(lines, "8. Healthy Retest", result["healthy_retest_2h"])
    write_group(lines, "9. Panic Retest", result["panic_retest_2h"])

    lines.extend(["", "## 10. Retest Quality Short Horizon Diagnostics", ""])
    lines.append("| Bucket | 60s | 15m |")
    lines.append("| --- | --- | --- |")
    for bucket in sorted(set(result["retest_quality_bucket_60s"]) | set(result["retest_quality_bucket_15m"])):
        lines.append(
            f"| `{bucket}` | {fmt(result['retest_quality_bucket_60s'].get(bucket, {}))} | "
            f"{fmt(result['retest_quality_bucket_15m'].get(bucket, {}))} |"
        )

    lines.extend(["", "## 11. Original Momentum-Oriented Tag Separators (2h)", ""])
    lines.append("| Tag | Yes | No |")
    lines.append("| --- | --- | --- |")
    for tag, row in result["tag_tables_2h"].items():
        lines.append(f"| `{tag}` | {fmt(row['yes'])} | {fmt(row['no'])} |")

    lines.extend(["", "## 12. Retest-Oriented Tag Separators (2h)", ""])
    lines.append("| Tag | Yes | No |")
    lines.append("| --- | --- | --- |")
    for tag, row in result["retest_tag_tables_2h"].items():
        lines.append(f"| `{tag}` | {fmt(row['yes'])} | {fmt(row['no'])} |")

    lines.extend(["", "## 13. Event Cards", ""])
    for row in result["rows"]:
        compact = {
            k: row.get(k)
            for k in [
                "event_id",
                "anchor_utc",
                "fill_leg",
                "phase",
                "fill_delay_sec",
                "fill_minus_arm_sec",
                "entry_vs_arm_bps",
                "entry_quality_score",
                "entry_quality_bucket",
                "retest_quality_score",
                "retest_quality_bucket",
                "retest_depth_bucket",
                "fill_minus_arm_bucket",
                "healthy_retest",
                "panic_retest",
                "btc_ret_15s_bps",
                "eth_ret_15s_bps",
                "sell_liq_15s_usd",
                "taker_imbalance_15s",
                "bid_depth_fill_vs_anchor",
                "spread_fill_minus_anchor_bps",
                "positive_tags",
                "negative_tags",
                "retest_positive_tags",
                "retest_negative_tags",
                "net_60s_bps",
                "net_15m_bps",
                "net_2h_bps",
            ]
        }
        lines.append(f"- `{compact}`")

    lines.extend(
        [
            "",
            "## 14. Interpretation",
            "",
            "- These are navigation labels for the current v0.2 alpha, not new entry filters.",
            "- Because N is 11, tags are for dashboard/shadow observation only.",
            "- A tag becomes actionable only after forward OOS confirms it on new fills.",
        ]
    )
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
