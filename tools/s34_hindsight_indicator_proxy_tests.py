"""S34 hindsight-pattern indicator proxy tests.

Tests whether the non-tradeable near-window hindsight pattern can be converted
into live-knowable navigation/exit indicators:
- precursor features for near3-only vs causal3 controls;
- sequence completion using only past/current thresholds;
- post-completion deceleration/reclaim proxies;
- v0.2 exit/tighten overlay.

Research/navigation only. No live executor, order logic, size, leverage, config,
or env changes.
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
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.s34_navigation_full_followup import DEFAULT_DB, mark_at_or_after, r1, r3, summary  # noqa: E402
from tools.s34_navigation_scalp_and_stress import route_v02  # noqa: E402
from tools.s34_stress_reaction_deep_tests import BASE_FEE_BPS, bracket_outcome, mark_series  # noqa: E402
from tools.s34_stress_scalp_live_readiness_tests import SELECTORS, build_live_like_rows, ts  # noqa: E402

OUT_JSON = ROOT / "reports" / "research" / "s34" / "S34_HINDSIGHT_INDICATOR_PROXY_TESTS.json"
OUT_MD = ROOT / "reports" / "research" / "s34" / "S34_HINDSIGHT_INDICATOR_PROXY_TESTS.md"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def t3r(vals: list[float]) -> float:
    vals = [float(v) for v in vals if math.isfinite(float(v))]
    return float(sum(sorted(vals, reverse=True)[3:])) if len(vals) > 3 else float(sum(vals))


def price_ret(conn: sqlite3.Connection, start_ms: int, end_ms: int) -> float | None:
    a = mark_at_or_after(conn, "ETHUSDT", int(start_ms))
    b = mark_at_or_after(conn, "ETHUSDT", int(end_ms))
    if not a or not b or float(a[1]) <= 0:
        return None
    return (float(b[1]) - float(a[1])) / float(a[1]) * 10_000.0


def liq_notional(conn: sqlite3.Connection, start_ms: int, end_ms: int, *, side: str | None = None, symbol: str = "ETHUSDT") -> float:
    clauses = ["symbol=?", "ts_ms>=?", "ts_ms<=?"]
    params: list[Any] = [symbol, int(start_ms), int(end_ms)]
    if side:
        clauses.append("side=?")
        params.append(side)
    row = conn.execute(
        f"SELECT COALESCE(SUM(notional),0.0) FROM liquidations WHERE {' AND '.join(clauses)}",
        params,
    ).fetchone()
    return float(row[0] or 0.0)


def liq_count(conn: sqlite3.Connection, start_ms: int, end_ms: int, *, side: str | None = None, symbol: str = "ETHUSDT") -> int:
    clauses = ["symbol=?", "ts_ms>=?", "ts_ms<=?"]
    params: list[Any] = [symbol, int(start_ms), int(end_ms)]
    if side:
        clauses.append("side=?")
        params.append(side)
    row = conn.execute(
        f"SELECT COUNT(*) FROM liquidations WHERE {' AND '.join(clauses)}",
        params,
    ).fetchone()
    return int(row[0] or 0)


def threshold_cross_times(conn: sqlite3.Connection, end_ms: int, lookback_sec: int = 900) -> dict[str, int | None]:
    rows = conn.execute(
        """
        SELECT ts_ms, notional
        FROM liquidations
        WHERE symbol='ETHUSDT' AND side='SELL' AND ts_ms>=? AND ts_ms<=?
        ORDER BY ts_ms ASC
        """,
        (int(end_ms) - lookback_sec * 1000, int(end_ms)),
    ).fetchall()
    total = 0.0
    out: dict[str, int | None] = {"50k": None, "100k": None, "200k": None}
    for t, n in rows:
        total += float(n or 0.0)
        if out["50k"] is None and total >= 50_000:
            out["50k"] = int(t)
        if out["100k"] is None and total >= 100_000:
            out["100k"] = int(t)
        if out["200k"] is None and total >= 200_000:
            out["200k"] = int(t)
    return out


def proxy_features(conn: sqlite3.Connection, row: dict[str, Any]) -> dict[str, Any]:
    t = ts(row)
    crosses = threshold_cross_times(conn, t, 900)
    known_times = [v for v in crosses.values() if v is not None]
    span = (max(known_times) - min(known_times)) / 1000.0 if len(known_times) >= 2 else None
    pre60 = liq_notional(conn, t - 60_000, t, side="SELL")
    post60 = liq_notional(conn, t, t + 60_000, side="SELL")
    post120 = liq_notional(conn, t, t + 120_000, side="SELL")
    decel60 = (post60 - pre60) / max(pre60, 1.0)
    ret60 = price_ret(conn, t, t + 60_000)
    ret120 = price_ret(conn, t, t + 120_000)
    series = mark_series(conn, t, t + 300_000)
    reclaim60 = None
    if series:
        entry = mark_at_or_after(conn, "ETHUSDT", t)
        if entry:
            px0 = float(entry[1])
            min_px = min(float(px) for _, px in series[: max(1, min(len(series), 60))])
            px60 = mark_at_or_after(conn, "ETHUSDT", t + 60_000)
            if px60:
                reclaim60 = (float(px60[1]) - min_px) / px0 * 10_000.0
    return {
        "seq_complete": all(crosses[k] is not None for k in ("50k", "100k", "200k")),
        "seq_span_sec": r1(span),
        "pre60_sell_notional": r1(pre60),
        "post60_sell_notional": r1(post60),
        "post120_sell_notional": r1(post120),
        "decel60_ratio": r3(decel60),
        "ret60_bps": r1(ret60),
        "ret120_bps": r1(ret120),
        "reclaim60_bps": r1(reclaim60),
        "prior_chain_thresholds": row.get("chain_prior_15m_thresholds"),
        "causal_chain_thresholds": row.get("chain_causal_15m_thresholds"),
        "near_chain_thresholds": row.get("chain_near_15m_thresholds"),
    }


def outcome_bracket(conn: sqlite3.Connection, row: dict[str, Any], *, direction: str, tp: float, sl: float, horizon_sec: int) -> tuple[float | None, str]:
    val, ex, _ = bracket_outcome(conn, row, horizon_sec=horizon_sec, direction="REVERSE" if direction == "SHORT" else "NORMAL", tp=tp, sl=sl, fee_bps=BASE_FEE_BPS)
    return val, ex


def eval_subset(conn: sqlite3.Connection, rows: list[dict[str, Any]], selector: Callable[[dict[str, Any]], bool], *, direction: str = "SHORT", tp: float = 200, sl: float = 40, horizon_sec: int = 1200) -> dict[str, Any]:
    vals = []
    exits: dict[str, int] = defaultdict(int)
    for row in rows:
        if not selector(row):
            continue
        val, ex = outcome_bracket(conn, row, direction=direction, tp=tp, sl=sl, horizon_sec=horizon_sec)
        if val is not None:
            vals.append(float(val))
            exits[str(ex)] += 1
    return {"matched_n": len([r for r in rows if selector(r)]), "summary": summary(vals), "exits": dict(exits)}


def build_feature_table(conn: sqlite3.Connection, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        item = dict(row)
        item["is_near3_only"] = SELECTORS["original_holdstate_near3"](row) and not SELECTORS["live_like_causal3"](row)
        item["is_causal3_only"] = SELECTORS["live_like_causal3"](row) and not SELECTORS["original_holdstate_near3"](row)
        item["is_both"] = SELECTORS["live_like_causal3"](row) and SELECTORS["original_holdstate_near3"](row)
        item.update(proxy_features(conn, row))
        out.append(item)
    return out


def feature_profile(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys = [
        "seq_span_sec",
        "pre60_sell_notional",
        "post60_sell_notional",
        "post120_sell_notional",
        "decel60_ratio",
        "ret60_bps",
        "ret120_bps",
        "reclaim60_bps",
        "prior_chain_thresholds",
        "causal_chain_thresholds",
    ]
    out: dict[str, Any] = {"n": len(rows), "seq_complete_rate": r3(sum(1 for r in rows if r.get("seq_complete")) / len(rows)) if rows else None}
    for key in keys:
        vals = [float(r[key]) for r in rows if r.get(key) is not None and math.isfinite(float(r[key]))]
        out[f"{key}_median"] = r1(median(vals)) if vals else None
        out[f"{key}_mean"] = r1(sum(vals) / len(vals)) if vals else None
    return out


def precursor_profile(conn: sqlite3.Connection, rows: list[dict[str, Any]]) -> dict[str, Any]:
    ft = build_feature_table(conn, rows)
    groups = {
        "near3_only": [r for r in ft if r["is_near3_only"]],
        "causal3_only": [r for r in ft if r["is_causal3_only"]],
        "both": [r for r in ft if r["is_both"]],
        "other": [r for r in ft if not (r["is_near3_only"] or r["is_causal3_only"] or r["is_both"])],
    }
    return {name: feature_profile(items) for name, items in groups.items()}


def proxy_screens(conn: sqlite3.Connection, rows: list[dict[str, Any]]) -> dict[str, Any]:
    ft = build_feature_table(conn, rows)
    screens: dict[str, Callable[[dict[str, Any]], bool]] = {
        "seq_complete": lambda r: bool(r.get("seq_complete")),
        "seq_complete_fast_60s": lambda r: bool(r.get("seq_complete")) and float(r.get("seq_span_sec") or 1e9) <= 60,
        "seq_complete_fast_180s": lambda r: bool(r.get("seq_complete")) and float(r.get("seq_span_sec") or 1e9) <= 180,
        "decel60_negative": lambda r: float(r.get("decel60_ratio") or 0.0) < 0,
        "decel60_lt_-50pct": lambda r: float(r.get("decel60_ratio") or 0.0) < -0.5,
        "reclaim60_positive": lambda r: float(r.get("reclaim60_bps") or 0.0) > 0,
        "reclaim60_gt_20": lambda r: float(r.get("reclaim60_bps") or 0.0) > 20,
        "seq_fast_and_decel": lambda r: bool(r.get("seq_complete")) and float(r.get("seq_span_sec") or 1e9) <= 180 and float(r.get("decel60_ratio") or 0.0) < 0,
        "seq_fast_and_reclaim": lambda r: bool(r.get("seq_complete")) and float(r.get("seq_span_sec") or 1e9) <= 180 and float(r.get("reclaim60_bps") or 0.0) > 0,
    }
    out = {}
    base = lambda r: int(r.get("stress_score_live_like") or 0) >= 3 and float(r.get("btc4h_bps") or 0.0) < -75 and float(r.get("vdepth_bps") or 0.0) < 50
    for name, fn in screens.items():
        sel = lambda r, fn=fn: base(r) and fn(r)
        out[name] = {
            "SHORT_TP200_SL40_20M": eval_subset(conn, ft, sel, direction="SHORT", tp=200, sl=40, horizon_sec=1200),
            "LONG_TP80_SL80_20M": eval_subset(conn, ft, sel, direction="LONG", tp=80, sl=80, horizon_sec=1200),
            "SHORT_TP150_SL30_15M": eval_subset(conn, ft, sel, direction="SHORT", tp=150, sl=30, horizon_sec=900),
        }
    return out


def v02_overlay_tests(conn: sqlite3.Connection, rows: list[dict[str, Any]]) -> dict[str, Any]:
    ft = build_feature_table(conn, rows)
    v02 = [r for r in ft if route_v02(r)]
    overlays: dict[str, Callable[[dict[str, Any]], bool]] = {
        "no_overlay": lambda r: False,
        "seq_complete_fast_180": lambda r: bool(r.get("seq_complete")) and float(r.get("seq_span_sec") or 1e9) <= 180,
        "decel60_lt_-50pct": lambda r: float(r.get("decel60_ratio") or 0.0) < -0.5,
        "reclaim60_positive": lambda r: float(r.get("reclaim60_bps") or 0.0) > 0,
        "near3_hindsight": lambda r: bool(r.get("is_near3_only") or r.get("is_both")),
    }
    out = {}
    for name, trigger in overlays.items():
        hold_vals = []
        exit_vals = []
        tighten_vals = []
        trigger_n = 0
        for row in v02:
            hold = fixed_long(conn, row, 7200)
            if hold is not None:
                hold_vals.append(hold)
            if name == "no_overlay" or not trigger(row):
                if hold is not None:
                    exit_vals.append(hold)
                    tighten_vals.append(hold)
                continue
            trigger_n += 1
            ex60 = fixed_long(conn, row, 60)
            if ex60 is not None:
                exit_vals.append(ex60)
            tight = long_tighten(conn, row, tp=120, sl=40, horizon_sec=7200)
            if tight is not None:
                tighten_vals.append(tight)
        out[name] = {
            "v02_n": len(v02),
            "trigger_n": trigger_n,
            "baseline_hold_2h": summary(hold_vals),
            "exit_on_indicator": summary(exit_vals),
            "tighten_tp120_sl40": summary(tighten_vals),
        }
    return out


def fixed_long(conn: sqlite3.Connection, row: dict[str, Any], sec: int) -> float | None:
    a = mark_at_or_after(conn, "ETHUSDT", ts(row))
    b = mark_at_or_after(conn, "ETHUSDT", ts(row) + int(sec) * 1000)
    if not a or not b or float(a[1]) <= 0:
        return None
    return (float(b[1]) - float(a[1])) / float(a[1]) * 10_000.0 - BASE_FEE_BPS


def long_tighten(conn: sqlite3.Connection, row: dict[str, Any], *, tp: float, sl: float, horizon_sec: int) -> float | None:
    val, _, _ = bracket_outcome(conn, row, horizon_sec=horizon_sec, direction="NORMAL", tp=tp, sl=sl, fee_bps=BASE_FEE_BPS)
    return val


def indicator_labels(conn: sqlite3.Connection, rows: list[dict[str, Any]]) -> dict[str, Any]:
    ft = build_feature_table(conn, rows)
    labels = {
        "CHAIN_BUILDING": lambda r: int(r.get("chain_causal_15m_thresholds") or 0) >= 2 and not bool(r.get("seq_complete")),
        "CHAIN_COMPLETE": lambda r: bool(r.get("seq_complete")),
        "EXHAUSTION_PROXY": lambda r: bool(r.get("seq_complete")) and float(r.get("decel60_ratio") or 0.0) < -0.5,
        "PANIC_CONTINUES": lambda r: float(r.get("decel60_ratio") or 0.0) >= 0.0 and float(r.get("ret60_bps") or 0.0) < 0,
        "RECLAIM_CONFIRMED": lambda r: float(r.get("reclaim60_bps") or 0.0) > 20,
        "NO_TRADE_HINDSIGHT_ZONE": lambda r: bool(r.get("is_near3_only")),
    }
    out = {}
    for name, fn in labels.items():
        subset = [r for r in ft if fn(r)]
        out[name] = {
            "n": len(subset),
            "profile": feature_profile(subset),
            "short": eval_subset(conn, ft, fn, direction="SHORT", tp=200, sl=40, horizon_sec=1200),
            "long": eval_subset(conn, ft, fn, direction="LONG", tp=80, sl=80, horizon_sec=1200),
        }
    return out


def run() -> dict[str, Any]:
    rows = build_live_like_rows()
    with sqlite3.connect(DEFAULT_DB) as conn:
        return {
            "generated_at_utc": utc_now(),
            "status": "RESEARCH_ONLY_NO_LIVE_CHANGE",
            "precursor_profile": precursor_profile(conn, rows),
            "proxy_screens": proxy_screens(conn, rows),
            "v02_overlay_tests": v02_overlay_tests(conn, rows),
            "indicator_labels": indicator_labels(conn, rows),
        }


def fmt(s: dict[str, Any]) -> str:
    return (
        f"N={s.get('n')} sum={s.get('sum_bps')} med={s.get('median_bps')} "
        f"T3R={s.get('t3r_bps')} tail150={s.get('tail_lte_minus150_n')} maxLoss={s.get('max_loss_bps')}"
    )


def write_report(result: dict[str, Any]) -> None:
    lines = [
        "# S34 Hindsight Indicator Proxy Tests",
        "",
        f"Generated: `{result['generated_at_utc']}`",
        "",
        f"Status: `{result['status']}`",
        "",
        "## 1. Precursor Profile",
        "",
    ]
    for name, row in result["precursor_profile"].items():
        lines.append(f"- `{name}`: `{row}`")

    lines.extend(["", "## 2. Live-Knowable Proxy Screens", ""])
    lines.append("| Proxy | SHORT TP200/SL40/20m | LONG TP80/SL80/20m | SHORT TP150/SL30/15m |")
    lines.append("| --- | --- | --- | --- |")
    for name, row in result["proxy_screens"].items():
        lines.append(
            f"| `{name}` | {fmt(row['SHORT_TP200_SL40_20M']['summary'])} | "
            f"{fmt(row['LONG_TP80_SL80_20M']['summary'])} | {fmt(row['SHORT_TP150_SL30_15M']['summary'])} |"
        )

    lines.extend(["", "## 3. v0.2 Exit Overlay", ""])
    lines.append("| Overlay | Trigger N | Baseline hold 2h | Exit on indicator | Tighten TP120/SL40 |")
    lines.append("| --- | ---: | --- | --- | --- |")
    for name, row in result["v02_overlay_tests"].items():
        lines.append(
            f"| `{name}` | {row['trigger_n']} | {fmt(row['baseline_hold_2h'])} | "
            f"{fmt(row['exit_on_indicator'])} | {fmt(row['tighten_tp120_sl40'])} |"
        )

    lines.extend(["", "## 4. Indicator Labels", ""])
    lines.append("| Label | N | SHORT | LONG | Profile |")
    lines.append("| --- | ---: | --- | --- | --- |")
    for name, row in result["indicator_labels"].items():
        lines.append(f"| `{name}` | {row['n']} | {fmt(row['short']['summary'])} | {fmt(row['long']['summary'])} | `{row['profile']}` |")

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
