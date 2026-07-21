from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.research_s34_knowable_anchor_continuation import (
    AnchorSnapshot,
    MarkIndex,
    book_at,
    file_fingerprint,
    iso_ms,
    load_liquidations,
    load_mark_index,
    mean,
    pctile,
    r1,
    r3,
    reconstruct_anchors,
    sha256_text,
    signed_return_bps,
    split_anchor_ids,
)
from tools.s34_shadow_paper_runner import _cost_decomposition


DEFAULT_DB = ROOT / "data" / "microstructure.db"
OUT_DIR = ROOT / "reports" / "research" / "s34"
OUT_MD = OUT_DIR / "S34_KNOWABLE_ANCHOR_ROUTE_RECHECK.md"
OUT_JSON = OUT_DIR / "S34_KNOWABLE_ANCHOR_ROUTE_RECHECK.json"


@dataclass(frozen=True)
class RouteSpec:
    family: str
    rule_name: str
    symbol: str
    liq_side: str
    direction: str
    threshold_usd: float
    tp_bps: float
    sl_bps: float
    be_bps: float
    entry_delay_sec: int = 0
    max_horizon_sec: int = 3600
    min_day_trend_bps: float | None = None
    max_day_trend_bps: float | None = None
    required_shape_label: str | None = None
    btc_pre_window_sec: int = 0
    btc_pre_min_return_bps: float | None = None
    max_single_dominance_pct: float | None = None


ROUTES: tuple[RouteSpec, ...] = (
    RouteSpec("ETH_BUY", "ETH_BUY_LIQ_LONG_50K_TP120_SL40_BE30", "ETHUSDT", "BUY", "LONG", 50_000.0, 120.0, 40.0, 30.0),
    RouteSpec("ETH_BUY", "ETH_BUY_LIQ_LONG_200K_TP60_SL40_BE30", "ETHUSDT", "BUY", "LONG", 200_000.0, 60.0, 40.0, 30.0),
    RouteSpec(
        "ETH_BUY",
        "ETH_BUY_LIQ_LONG_200K_BTC_PRE15_TP120_SL40_BE30_DELAY60",
        "ETHUSDT",
        "BUY",
        "LONG",
        200_000.0,
        120.0,
        40.0,
        30.0,
        entry_delay_sec=60,
        btc_pre_window_sec=900,
        btc_pre_min_return_bps=0.0,
    ),
    RouteSpec(
        "ETH_BUY",
        "ETH_BUY_LIQ_LONG_500K_DAYTREND0_TP60_SL40_BE30",
        "ETHUSDT",
        "BUY",
        "LONG",
        500_000.0,
        60.0,
        40.0,
        30.0,
        min_day_trend_bps=0.0,
    ),
    RouteSpec(
        "ETH_BUY",
        "ETH_BUY_LIQ_LONG_500K_NEGTREND_STRETCHED_TP60_SL40_BE30",
        "ETHUSDT",
        "BUY",
        "LONG",
        500_000.0,
        60.0,
        40.0,
        30.0,
        max_day_trend_bps=0.0,
        required_shape_label="stretched_120s",
    ),
    RouteSpec("SOL_BUY", "SOL_BUY_LIQ_LONG_100K_TP60_SL40_BE30", "SOLUSDT", "BUY", "LONG", 100_000.0, 60.0, 40.0, 30.0),
    RouteSpec("SOL_BUY", "SOL_BUY_LIQ_LONG_200K_TP60_SL40_BE30", "SOLUSDT", "BUY", "LONG", 200_000.0, 60.0, 40.0, 30.0),
    RouteSpec(
        "BTC_BUY_DISTRIBUTED",
        "BTC_BUY_LIQ_LONG_1M_DISTRIBUTED_TP60_SL30_BE30",
        "BTCUSDT",
        "BUY",
        "LONG",
        1_000_000.0,
        60.0,
        30.0,
        30.0,
        max_single_dominance_pct=50.0,
    ),
    RouteSpec("ETH_SELL", "ETH_SELL_LIQ_SHORT_500K_TP60_SL40_BE40", "ETHUSDT", "SELL", "SHORT", 500_000.0, 60.0, 40.0, 40.0),
    RouteSpec("ETH_SELL", "ETH_SELL_LIQ_SHORT_1M_TP80_SL40_BE40", "ETHUSDT", "SELL", "SHORT", 1_000_000.0, 80.0, 40.0, 40.0),
    RouteSpec("SOL_SELL", "SOL_SELL_LIQ_SHORT_100K_TP60_SL30_BE40", "SOLUSDT", "SELL", "SHORT", 100_000.0, 60.0, 30.0, 40.0),
    RouteSpec("SOL_SELL", "SOL_SELL_LIQ_SHORT_200K_TP60_SL30_BE30", "SOLUSDT", "SELL", "SHORT", 200_000.0, 60.0, 30.0, 30.0),
)


def day_start_ms(ts_ms: int) -> int:
    dt = datetime.fromtimestamp(int(ts_ms) / 1000.0, tz=timezone.utc)
    start = datetime(dt.year, dt.month, dt.day, tzinfo=timezone.utc)
    return int(start.timestamp() * 1000)


def day_trend_bps(marks: MarkIndex, ts_ms: int) -> float | None:
    start = marks.at_or_after(day_start_ms(ts_ms))
    cur = marks.at_or_before(ts_ms)
    if not start or not cur or float(start[1]) <= 0.0:
        return None
    return (float(cur[1]) - float(start[1])) / float(start[1]) * 10_000.0


def route_key(spec: RouteSpec) -> str:
    return spec.rule_name


def summarize(rows: list[dict[str, Any]], key: str = "net_bps") -> dict[str, Any]:
    vals = [float(r[key]) for r in rows if r.get(key) is not None and math.isfinite(float(r[key]))]
    exits: dict[str, int] = {}
    for row in rows:
        if row.get(key) is None:
            continue
        reason = str(row.get("exit_reason") or "NA")
        exits[reason] = exits.get(reason, 0) + 1
    return {
        "n": len(vals),
        "mean_bps": r1(mean(vals)),
        "median_bps": r1(pctile(vals, 0.5)),
        "p25_bps": r1(pctile(vals, 0.25)),
        "p75_bps": r1(pctile(vals, 0.75)),
        "win_rate": r3(sum(v > 0 for v in vals) / len(vals)) if vals else None,
        "cum_bps": r1(sum(vals)) if vals else 0.0,
        "top3_winner_removed_cum_bps": r1(sum(sorted(vals, reverse=True)[3:])) if len(vals) > 3 else r1(sum(vals)),
        "exits": dict(sorted(exits.items())),
    }


def anchor_shape_label(anchor: AnchorSnapshot) -> str:
    """Knowable cluster-shape label at the threshold cross (mirrors the runner's
    _cluster_shape_label but uses only running values known at the anchor, so the
    'stretched_120s' classification carries no lookahead)."""
    if float(anchor.running_single_liq_dominance) >= 80.0:
        return "single_dominant_80pct"
    if float(anchor.elapsed_since_first_sec) >= 120.0:
        return "stretched_120s"
    return "distributed_mid_duration"


def route_filters_pass(spec: RouteSpec, anchor: AnchorSnapshot, marks: MarkIndex, btc_marks: MarkIndex) -> tuple[bool, str, dict[str, Any]]:
    diagnostics: dict[str, Any] = {}
    if spec.max_single_dominance_pct is not None:
        diagnostics["running_single_liq_dominance"] = float(anchor.running_single_liq_dominance)
        if float(anchor.running_single_liq_dominance) > float(spec.max_single_dominance_pct):
            return False, "DOMINANCE_FILTER", diagnostics
    if spec.required_shape_label is not None:
        label = anchor_shape_label(anchor)
        diagnostics["shape_label"] = label
        if label != str(spec.required_shape_label):
            return False, "SHAPE_FILTER", diagnostics
    if spec.min_day_trend_bps is not None:
        trend = day_trend_bps(marks, anchor.anchor_ts_ms)
        diagnostics["day_trend_bps"] = trend
        if trend is None or float(trend) < float(spec.min_day_trend_bps):
            return False, "DAY_TREND_FILTER", diagnostics
    if spec.max_day_trend_bps is not None:
        trend = day_trend_bps(marks, anchor.anchor_ts_ms)
        diagnostics["day_trend_bps"] = trend
        if trend is None or float(trend) > float(spec.max_day_trend_bps):
            return False, "MAX_DAY_TREND_FILTER", diagnostics
    if spec.btc_pre_min_return_bps is not None and int(spec.btc_pre_window_sec) > 0:
        ret = btc_marks.ret_bps(anchor.anchor_ts_ms - int(spec.btc_pre_window_sec) * 1000, anchor.anchor_ts_ms)
        diagnostics["btc_pre_return_bps"] = ret
        if ret is None or float(ret) < float(spec.btc_pre_min_return_bps):
            return False, "BTC_PRE_FILTER", diagnostics
    return True, "", diagnostics


def simulate_route(
    conn: sqlite3.Connection,
    marks: MarkIndex,
    spec: RouteSpec,
    anchor: AnchorSnapshot,
    *,
    fee_bps_side: float,
    max_book_staleness_sec: int,
) -> dict[str, Any]:
    entry_target_ms = int(anchor.anchor_ts_ms) + int(spec.entry_delay_sec) * 1000
    entry_book = book_at(conn, spec.symbol, entry_target_ms, max_book_staleness_sec)
    entry_mark = marks.at_or_after(entry_target_ms)
    base = {
        "rule_name": spec.rule_name,
        "family": spec.family,
        "symbol": spec.symbol,
        "liq_side": spec.liq_side,
        "direction": spec.direction,
        "threshold_usd": spec.threshold_usd,
        "anchor_ts_ms": int(anchor.anchor_ts_ms),
        "anchor_utc": iso_ms(anchor.anchor_ts_ms),
        "entry_target_ts_ms": int(entry_target_ms),
        "entry_target_utc": iso_ms(entry_target_ms),
        "elapsed_since_first_sec": float(anchor.elapsed_since_first_sec),
        "running_notional": float(anchor.running_notional),
        "running_liq_count": int(anchor.running_liq_count),
        "running_accel": float(anchor.running_accel),
        "acceleration_bucket": anchor.acceleration_bucket,
        "running_single_liq_dominance": float(anchor.running_single_liq_dominance),
    }
    if not entry_book:
        base.update({"status": "NO_ENTRY_FILL", "net_bps": None})
        return add_mark_counterfactual(base, marks, spec, entry_target_ms, fee_bps_side)
    if not entry_mark:
        base.update({"status": "NO_ENTRY_MARK", "net_bps": None})
        return add_mark_counterfactual(base, marks, spec, entry_target_ms, fee_bps_side)

    entry_px = float(entry_book.ask if spec.direction == "LONG" else entry_book.bid)
    entry_reference = float(entry_mark[1])
    if spec.direction == "LONG":
        tp_price = entry_reference * (1.0 + float(spec.tp_bps) / 10_000.0)
        sl_price = entry_reference * (1.0 - float(spec.sl_bps) / 10_000.0)
        be_trigger_price = entry_reference * (1.0 + float(spec.be_bps) / 10_000.0)
    else:
        tp_price = entry_reference * (1.0 - float(spec.tp_bps) / 10_000.0)
        sl_price = entry_reference * (1.0 + float(spec.sl_bps) / 10_000.0)
        be_trigger_price = entry_reference * (1.0 - float(spec.be_bps) / 10_000.0)
    path = [
        (ts, px)
        for ts, px in marks.slice_range(int(entry_mark[0]), int(entry_mark[0]) + int(spec.max_horizon_sec) * 1000)
        if int(ts) > int(entry_mark[0])
    ]
    if not path:
        base.update({"status": "NO_MARK_PATH", "net_bps": None})
        return add_mark_counterfactual(base, marks, spec, entry_target_ms, fee_bps_side)

    exit_reason = "TIME"
    exit_ts_ms = int(path[-1][0])
    exit_reference = float(path[-1][1])
    mfe = -1e18
    mae = 1e18
    be_active = False
    for ts_ms, mark in path:
        ret = signed_return_bps(spec.direction, entry_reference, float(mark))
        mfe = max(mfe, ret)
        mae = min(mae, ret)
        if spec.direction == "LONG":
            if not be_active and float(mark) >= be_trigger_price:
                be_active = True
            if float(mark) >= tp_price:
                exit_reason = "TP"
                exit_ts_ms = int(ts_ms)
                exit_reference = float(mark)
                break
            stop_price = entry_reference if be_active else sl_price
            if float(mark) <= stop_price:
                exit_reason = "BE" if be_active else "SL"
                exit_ts_ms = int(ts_ms)
                exit_reference = float(mark)
                break
        else:
            if not be_active and float(mark) <= be_trigger_price:
                be_active = True
            if float(mark) <= tp_price:
                exit_reason = "TP"
                exit_ts_ms = int(ts_ms)
                exit_reference = float(mark)
                break
            stop_price = entry_reference if be_active else sl_price
            if float(mark) >= stop_price:
                exit_reason = "BE" if be_active else "SL"
                exit_ts_ms = int(ts_ms)
                exit_reference = float(mark)
                break

    exit_book = book_at(conn, spec.symbol, exit_ts_ms, max_book_staleness_sec)
    if not exit_book:
        base.update(
            {
                "status": "NO_EXIT_FILL",
                "exit_reason": exit_reason,
                "exit_ts_ms": exit_ts_ms,
                "exit_reference_price": exit_reference,
                "net_bps": None,
                "mfe_bps": mfe,
                "mae_bps": mae,
            }
        )
        return add_mark_counterfactual(base, marks, spec, entry_target_ms, fee_bps_side)

    exit_px = float(exit_book.bid if spec.direction == "LONG" else exit_book.ask)
    entry_fill = {
        "fill_price": entry_px,
        "mid": float(entry_book.mid),
        "fee_bps": float(fee_bps_side),
    }
    exit_fill = {
        "fill_price": exit_px,
        "mid": float(exit_book.mid),
        "fee_bps": float(fee_bps_side),
    }
    cost = _cost_decomposition(
        spec.direction,
        entry_reference,
        exit_reference,
        entry_fill,
        exit_fill,
        float(fee_bps_side),
        float(fee_bps_side),
    )
    base.update(
        {
            "status": "FILLED",
            "entry_book_ts_ms": int(entry_book.ts_ms),
            "entry_staleness_ms": int(entry_book.staleness_ms),
            "entry_price": entry_px,
            "entry_reference_mark_ts_ms": int(entry_mark[0]),
            "entry_reference_price": entry_reference,
            "exit_book_ts_ms": int(exit_book.ts_ms),
            "exit_staleness_ms": int(exit_book.staleness_ms),
            "exit_ts_ms": int(exit_ts_ms),
            "exit_utc": iso_ms(exit_ts_ms),
            "exit_reason": exit_reason,
            "exit_price": exit_px,
            "exit_reference_price": exit_reference,
            "gross_bps": cost["gross_bps"],
            "executable_gross_bps": cost["executable_gross_bps"],
            "entry_adverse_bps": cost["entry_adverse_bps"],
            "exit_adverse_bps": cost["exit_adverse_bps"],
            "spread_cost_bps": cost["spread_cost_bps"],
            "fee_bps": 2.0 * float(fee_bps_side),
            "net_bps": cost["net_bps"],
            "mfe_bps": mfe,
            "mae_bps": mae,
            "hold_sec": (int(exit_ts_ms) - int(entry_mark[0])) / 1000.0,
        }
    )
    return add_mark_counterfactual(base, marks, spec, entry_target_ms, fee_bps_side)


def add_mark_counterfactual(
    row: dict[str, Any],
    marks: MarkIndex,
    spec: RouteSpec,
    entry_target_ms: int,
    fee_bps_side: float,
) -> dict[str, Any]:
    entry = marks.at_or_after(entry_target_ms)
    if not entry:
        row["counterfactual_mark_net_bps"] = None
        return row
    entry_ts_ms, entry_px = int(entry[0]), float(entry[1])
    if spec.direction == "LONG":
        tp_price = entry_px * (1.0 + float(spec.tp_bps) / 10_000.0)
        sl_price = entry_px * (1.0 - float(spec.sl_bps) / 10_000.0)
        be_trigger_price = entry_px * (1.0 + float(spec.be_bps) / 10_000.0)
    else:
        tp_price = entry_px * (1.0 - float(spec.tp_bps) / 10_000.0)
        sl_price = entry_px * (1.0 + float(spec.sl_bps) / 10_000.0)
        be_trigger_price = entry_px * (1.0 - float(spec.be_bps) / 10_000.0)
    path = [(ts, px) for ts, px in marks.slice_range(entry_ts_ms, entry_ts_ms + int(spec.max_horizon_sec) * 1000) if int(ts) > entry_ts_ms]
    if not path:
        row["counterfactual_mark_net_bps"] = None
        return row
    exit_reason = "TIME"
    exit_ts_ms = int(path[-1][0])
    exit_px = float(path[-1][1])
    be_active = False
    for ts_ms, mark in path:
        if spec.direction == "LONG":
            if not be_active and float(mark) >= be_trigger_price:
                be_active = True
            if float(mark) >= tp_price:
                exit_reason = "TP"
                exit_ts_ms = int(ts_ms)
                exit_px = float(mark)
                break
            stop_price = entry_px if be_active else sl_price
            if float(mark) <= stop_price:
                exit_reason = "BE" if be_active else "SL"
                exit_ts_ms = int(ts_ms)
                exit_px = float(mark)
                break
        else:
            if not be_active and float(mark) <= be_trigger_price:
                be_active = True
            if float(mark) <= tp_price:
                exit_reason = "TP"
                exit_ts_ms = int(ts_ms)
                exit_px = float(mark)
                break
            stop_price = entry_px if be_active else sl_price
            if float(mark) >= stop_price:
                exit_reason = "BE" if be_active else "SL"
                exit_ts_ms = int(ts_ms)
                exit_px = float(mark)
                break
    gross = signed_return_bps(spec.direction, entry_px, exit_px)
    row["counterfactual_mark_exit_reason"] = exit_reason
    row["counterfactual_mark_exit_ts_ms"] = exit_ts_ms
    row["counterfactual_mark_gross_bps"] = gross
    row["counterfactual_mark_net_bps"] = gross - 2.0 * float(fee_bps_side)
    return row


def run_spec(
    conn: sqlite3.Connection,
    spec: RouteSpec,
    *,
    bucket_sec: int,
    min_gap_sec: int,
    accel_window_sec: int,
    holdout_frac: float,
    fee_bps_side: float,
    max_book_staleness_sec: int,
    start_ms: int | None,
    end_ms: int | None,
) -> dict[str, Any]:
    liqs = load_liquidations(conn, spec.symbol, spec.liq_side, start_ms, end_ms)
    marks = load_mark_index(conn, spec.symbol)
    btc_marks = load_mark_index(conn, "BTCUSDT")
    anchors = reconstruct_anchors(
        liqs,
        bucket_sec=bucket_sec,
        min_gap_sec=min_gap_sec,
        thresholds=(float(spec.threshold_usd),),
        accel_window_sec=accel_window_sec,
    )
    filtered: list[tuple[AnchorSnapshot, dict[str, Any]]] = []
    rejected: dict[str, int] = {}
    for anchor in anchors:
        ok, reason, diagnostics = route_filters_pass(spec, anchor, marks, btc_marks)
        if not ok:
            rejected[reason] = rejected.get(reason, 0) + 1
            continue
        filtered.append((anchor, diagnostics))
    split_anchors = [item[0] for item in filtered]
    calibration_ids, holdout_ids, split = split_anchor_ids(split_anchors, holdout_frac)

    rows: list[dict[str, Any]] = []
    for anchor, diagnostics in filtered:
        row = simulate_route(
            conn,
            marks,
            spec,
            anchor,
            fee_bps_side=fee_bps_side,
            max_book_staleness_sec=max_book_staleness_sec,
        )
        row["filter_diagnostics"] = diagnostics
        row["split"] = "holdout" if str(anchor.bucket) in holdout_ids else "calibration"
        rows.append(row)

    calibration = [r for r in rows if r["split"] == "calibration"]
    holdout = [r for r in rows if r["split"] == "holdout"]
    cal_filled = [r for r in calibration if r.get("status") == "FILLED"]
    hold_filled = [r for r in holdout if r.get("status") == "FILLED"]
    cal_no_fill = [r for r in calibration if r.get("status") != "FILLED"]
    hold_no_fill = [r for r in holdout if r.get("status") != "FILLED"]
    cal_summary = summarize(cal_filled)
    hold_summary = summarize(hold_filled)
    cal_cf = summarize(calibration, "counterfactual_mark_net_bps")
    hold_cf = summarize(holdout, "counterfactual_mark_net_bps")
    verdict = "RESEARCH_ONLY"
    if cal_summary["n"] < 20:
        verdict = "BLOCKED_THIN_CALIBRATION"
    elif (
        (cal_summary["median_bps"] is not None and float(cal_summary["median_bps"]) > 0.0)
        and (cal_summary["mean_bps"] is not None and float(cal_summary["mean_bps"]) > 0.0)
        and float(cal_summary["top3_winner_removed_cum_bps"] or 0.0) > 0.0
        and hold_summary["n"] >= 10
        and (hold_summary["median_bps"] is not None and float(hold_summary["median_bps"]) > 0.0)
        and (hold_summary["mean_bps"] is not None and float(hold_summary["mean_bps"]) > 0.0)
        and float(hold_summary["top3_winner_removed_cum_bps"] or 0.0) > 0.0
    ):
        verdict = "PAPER_CANDIDATE"
    return {
        "spec": asdict(spec),
        "rule_name": spec.rule_name,
        "family": spec.family,
        "all_anchor_n": len(anchors),
        "filtered_anchor_n": len(filtered),
        "filter_rejections": rejected,
        "split": split,
        "calibration": {
            "total_n": len(calibration),
            "filled_n": len(cal_filled),
            "no_fill_n": len(cal_no_fill),
            "no_fill_rate": r3(len(cal_no_fill) / len(calibration)) if calibration else None,
            "summary": cal_summary,
            "mark_counterfactual": cal_cf,
        },
        "holdout": {
            "total_n": len(holdout),
            "filled_n": len(hold_filled),
            "no_fill_n": len(hold_no_fill),
            "no_fill_rate": r3(len(hold_no_fill) / len(holdout)) if holdout else None,
            "summary": hold_summary,
            "mark_counterfactual": hold_cf,
        },
        "verdict": verdict,
    }


def write_report(payload: dict[str, Any]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        "# S34 Knowable-Anchor TP/SL/BE Route Recheck",
        "",
        f"Generated: `{payload['generated_at_utc']}`",
        "",
        "RESEARCH_ONLY. Exact frozen route exits rerun on real-time-knowable running-notional anchors. No live/executor changes.",
        "",
        "## Summary",
        "",
        "| Family | Rule | Anchors | Filtered | Cal N | Cal Median | Cal Mean | Cal T3R | Hold N | Hold Median | Hold Mean | Hold T3R | No-fill Cal/Hold | Mark CF Cal/Hold | Verdict |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for result in payload["results"]:
        cal = result["calibration"]
        hold = result["holdout"]
        cs = cal["summary"]
        hs = hold["summary"]
        lines.append(
            f"| {result['family']} | `{result['rule_name']}` | {result['all_anchor_n']} | {result['filtered_anchor_n']} "
            f"| {cal['filled_n']} | {cs['median_bps']} | {cs['mean_bps']} | {cs['top3_winner_removed_cum_bps']} "
            f"| {hold['filled_n']} | {hs['median_bps']} | {hs['mean_bps']} | {hs['top3_winner_removed_cum_bps']} "
            f"| {cal['no_fill_rate']} / {hold['no_fill_rate']} "
            f"| {cal['mark_counterfactual']['median_bps']} / {hold['mark_counterfactual']['median_bps']} "
            f"| {result['verdict']} |"
        )
    lines += [
        "",
        "## Read",
        "",
        "- `Cal/Hold T3R` is top-3-winner-removed cumulative net bps.",
        "- `Mark CF` is mark-price counterfactual median on all accepted anchors, useful for separating directional signal from executable-fill coverage.",
        "- `PAPER_CANDIDATE` requires positive calibration and holdout median/mean/T3R with minimum filled N; otherwise the route remains `RESEARCH_ONLY` or `BLOCKED`.",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Rerun exact S34 TP/SL/BE routes on knowable running-notional anchors.")
    p.add_argument("--db", default=str(DEFAULT_DB))
    p.add_argument("--bucket-sec", type=int, default=300)
    p.add_argument("--min-gap-sec", type=int, default=900)
    p.add_argument("--accel-window-sec", type=int, default=30)
    p.add_argument("--holdout-frac", type=float, default=0.30)
    p.add_argument("--fee-bps-side", type=float, default=3.05)
    p.add_argument("--max-book-staleness-sec", type=int, default=5)
    p.add_argument("--start-ms", type=int, default=None)
    p.add_argument("--end-ms", type=int, default=None)
    p.add_argument("--out-prefix", default="")
    p.add_argument("--invert-direction", action="store_true",
                   help="Flip LONG<->SHORT (fade test): trade against the liquidation cascade instead of with it.")
    return p.parse_args()


def invert_spec(spec: RouteSpec) -> RouteSpec:
    flipped = "SHORT" if spec.direction == "LONG" else "LONG"
    return replace(spec, direction=flipped, rule_name=spec.rule_name + "_FADE", family=spec.family + "_FADE")


def main() -> None:
    args = parse_args()
    db_path = Path(str(args.db))
    routes = tuple(invert_spec(s) for s in ROUTES) if args.invert_direction else ROUTES
    conn = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)
    conn.execute("PRAGMA query_only=1")
    try:
        results = [
            run_spec(
                conn,
                spec,
                bucket_sec=int(args.bucket_sec),
                min_gap_sec=int(args.min_gap_sec),
                accel_window_sec=int(args.accel_window_sec),
                holdout_frac=float(args.holdout_frac),
                fee_bps_side=float(args.fee_bps_side),
                max_book_staleness_sec=int(args.max_book_staleness_sec),
                start_ms=args.start_ms,
                end_ms=args.end_ms,
            )
            for spec in routes
        ]
    finally:
        conn.close()
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_db": file_fingerprint(db_path),
        "routes_sha256": sha256_text(json.dumps([asdict(r) for r in routes], sort_keys=True)),
        "config": {
            "bucket_sec": int(args.bucket_sec),
            "min_gap_sec": int(args.min_gap_sec),
            "accel_window_sec": int(args.accel_window_sec),
            "holdout_frac": float(args.holdout_frac),
            "fee_bps_side": float(args.fee_bps_side),
            "max_book_staleness_sec": int(args.max_book_staleness_sec),
            "start_ms": args.start_ms,
            "end_ms": args.end_ms,
        },
        "results": results,
    }
    if args.out_prefix:
        global OUT_MD, OUT_JSON
        safe = str(args.out_prefix).replace("/", "_").replace("\\", "_")
        OUT_MD = OUT_DIR / f"S34_KNOWABLE_ANCHOR_ROUTE_RECHECK_{safe}.md"
        OUT_JSON = OUT_DIR / f"S34_KNOWABLE_ANCHOR_ROUTE_RECHECK_{safe}.json"
    write_report(payload)
    print(json.dumps({"routes": len(results), "out_md": str(OUT_MD), "out_json": str(OUT_JSON)}, indent=2))


if __name__ == "__main__":
    main()
