"""S34 V Engine v0.1 failure anatomy.

Explains why the frozen SELL-liq -> maker LONG V candidate wins or fails.
This is diagnostic only: it reads the observation ledger and historical marks,
then writes reports. It does not create signals, orders, or parameter changes.
"""

from __future__ import annotations

import argparse
import csv
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

from tools.research_s34_knowable_anchor_continuation import (
    MarkIndex,
    file_fingerprint,
    load_mark_index,
    mean,
    pctile,
    r1,
    r3,
    signed_return_bps,
)
from tools.research_s34_maker_fade import summarize
from tools.s34_v_engine_shadow_observer import (
    DEFAULT_LEDGER_JSONL,
    FADE_DIRECTION,
    PROTOCOL_ID,
    SYMBOL,
    utc_now,
)


DEFAULT_DB = ROOT / "data" / "microstructure.db"
OUT_DIR = ROOT / "reports" / "research" / "s34"
OUT_JSON = OUT_DIR / "S34_V_ENGINE_FAILURE_ANATOMY.json"
OUT_MD = OUT_DIR / "S34_V_ENGINE_FAILURE_ANATOMY.md"
OUT_CSV = OUT_DIR / "S34_V_ENGINE_FAILURE_ANATOMY_ROWS.csv"

ANATOMY_FIELDS = (
    "observation_id",
    "signal_utc",
    "maker_fill_utc",
    "net_bps",
    "outcome_class",
    "trap_tags",
    "fill_delay_sec",
    "fill_delay_bucket",
    "vdepth_bps",
    "prior_4h_bps",
    "prior4h_intensity_bucket",
    "running_accel_usd_per_sec",
    "elapsed_since_first_sec",
    "single_liq_dominance_pct",
    "entry_price",
    "anchor_mark_price",
    "mfe_15m_bps",
    "mae_15m_bps",
    "ret_15m_bps",
    "mfe_30m_bps",
    "mae_30m_bps",
    "ret_30m_bps",
    "mfe_60m_bps",
    "mae_60m_bps",
    "ret_60m_bps",
    "mfe_120m_bps",
    "mae_120m_bps",
    "ret_120m_bps",
    "low_rebreak_15m",
    "low_rebreak_30m",
    "anchor_reclaimed_15m",
    "anchor_reclaimed_30m",
    "first_15m_bucket",
    "btc_prior_4h_bps",
    "btc_after_15m_bps",
    "btc_after_60m_bps",
    "btc_context_bucket",
    "candle5_body_bps",
    "candle5_lower_wick_bps",
    "candle5_upper_wick_bps",
    "candle5_close_ret_bps",
    "candle5_pattern",
    "candle15_body_bps",
    "candle15_lower_wick_bps",
    "candle15_upper_wick_bps",
    "candle15_close_ret_bps",
    "candle15_pattern",
)


def load_ledger(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def finite_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def ret_at(marks: MarkIndex, direction: str, entry_px: float, ts_ms: int) -> float | None:
    row = marks.at_or_after(int(ts_ms))
    if not row:
        return None
    return signed_return_bps(direction, float(entry_px), float(row[1]))


def path_stats(
    marks: MarkIndex,
    *,
    direction: str,
    entry_px: float,
    start_ms: int,
    horizon_sec: int,
) -> dict[str, Any]:
    end_ms = int(start_ms) + int(horizon_sec) * 1000
    rows = [(int(ts), float(px)) for ts, px in marks.slice_range(int(start_ms), end_ms)]
    vals = [signed_return_bps(direction, float(entry_px), px) for _, px in rows]
    if not vals:
        return {"ret_bps": None, "mfe_bps": None, "mae_bps": None, "min_price": None, "max_price": None}
    ret = ret_at(marks, direction, entry_px, end_ms)
    prices = [px for _, px in rows]
    return {
        "ret_bps": r1(ret),
        "mfe_bps": r1(max(vals)),
        "mae_bps": r1(min(vals)),
        "min_price": min(prices),
        "max_price": max(prices),
    }


def ohlc_after(marks: MarkIndex, start_ms: int, minutes: int) -> dict[str, float] | None:
    rows = [(int(ts), float(px)) for ts, px in marks.slice_range(int(start_ms), int(start_ms) + int(minutes) * 60_000)]
    if not rows:
        return None
    prices = [px for _, px in rows]
    return {"open": prices[0], "high": max(prices), "low": min(prices), "close": prices[-1]}


def candle_features(ohlc: dict[str, float] | None, *, ref_price: float) -> dict[str, Any]:
    if not ohlc or float(ref_price) <= 0.0:
        return {
            "body_bps": None,
            "lower_wick_bps": None,
            "upper_wick_bps": None,
            "close_ret_bps": None,
            "pattern": "no_candle",
        }
    open_px = float(ohlc["open"])
    high = float(ohlc["high"])
    low = float(ohlc["low"])
    close = float(ohlc["close"])
    body = abs(close - open_px) / float(ref_price) * 10_000.0
    lower = (min(open_px, close) - low) / float(ref_price) * 10_000.0
    upper = (high - max(open_px, close)) / float(ref_price) * 10_000.0
    close_ret = signed_return_bps(FADE_DIRECTION, float(ref_price), close)
    bullish = close >= open_px
    if bullish and lower >= max(2.0 * body, 5.0):
        pattern = "hammer_reversal"
    elif bullish and close_ret > 10.0:
        pattern = "bull_reclaim"
    elif not bullish and lower >= max(2.0 * body, 5.0):
        pattern = "failed_hammer"
    elif close_ret < -10.0:
        pattern = "bear_followthrough"
    else:
        pattern = "neutral"
    return {
        "body_bps": r1(body),
        "lower_wick_bps": r1(lower),
        "upper_wick_bps": r1(upper),
        "close_ret_bps": r1(close_ret),
        "pattern": pattern,
    }


def fill_delay_bucket(sec: float | None) -> str:
    if sec is None:
        return "no_fill_delay"
    if sec <= 30.0:
        return "fill_0_30s"
    if sec <= 120.0:
        return "fill_30_120s"
    if sec <= 600.0:
        return "fill_2_10m"
    if sec <= 1800.0:
        return "fill_10_30m"
    return "fill_30m_plus"


def prior4h_intensity_bucket(v: float | None) -> str:
    if v is None:
        return "prior4h_na"
    if v > -100.0:
        return "prior4h_mild_down"
    if v > -250.0:
        return "prior4h_medium_down"
    if v > -400.0:
        return "prior4h_hard_down"
    return "prior4h_extreme_down"


def first_ret_bucket(v: float | None) -> str:
    if v is None:
        return "ret15_na"
    if v < -25.0:
        return "ret15_dump"
    if v < 0.0:
        return "ret15_soft_red"
    if v < 25.0:
        return "ret15_soft_green"
    return "ret15_rebound"


def btc_context_bucket(prior4h: float | None, after15: float | None) -> str:
    if prior4h is None or after15 is None:
        return "btc_na"
    if prior4h < -50.0 and after15 >= 0.0:
        return "btc_down_then_stable"
    if prior4h < -50.0 and after15 < 0.0:
        return "btc_down_continues"
    if prior4h >= -50.0 and after15 >= 0.0:
        return "btc_supportive"
    return "btc_softening"


def trap_tags(row: dict[str, Any]) -> list[str]:
    tags = []
    if row.get("low_rebreak_15m"):
        tags.append("low_rebreak_15m")
    elif row.get("low_rebreak_30m"):
        tags.append("low_rebreak_30m")
    fd = finite_float(row.get("fill_delay_sec"))
    if fd is not None and fd > 600.0:
        tags.append("late_fill_gt10m")
    if row.get("first_15m_bucket") in {"ret15_dump", "ret15_soft_red"}:
        tags.append("weak_first_15m")
    if row.get("candle5_pattern") in {"failed_hammer", "bear_followthrough"}:
        tags.append(f"candle5_{row.get('candle5_pattern')}")
    if row.get("btc_context_bucket") == "btc_down_continues":
        tags.append("btc_down_continues")
    if not tags:
        tags.append("clean_or_unclassified")
    return tags


def build_anatomy_rows(
    ledger: list[dict[str, Any]],
    *,
    eth_marks: MarkIndex,
    btc_marks: MarkIndex,
    rebreak_bps: float,
) -> list[dict[str, Any]]:
    out = []
    for row in ledger:
        if row.get("observation_status") != "CLOSED" or row.get("sim_status") != "FILLED":
            continue
        entry_px = finite_float(row.get("entry_price"))
        fill_ts = row.get("maker_fill_ts_ms")
        signal_ts = row.get("signal_ts_ms")
        if entry_px is None or fill_ts is None or signal_ts is None:
            continue
        fill_ts = int(fill_ts)
        signal_ts = int(signal_ts)
        horizons = {m: path_stats(eth_marks, direction=FADE_DIRECTION, entry_px=entry_px, start_ms=fill_ts, horizon_sec=m * 60) for m in (15, 30, 60, 120)}
        anchor_mark = finite_float(row.get("anchor_mark_price"))
        ret15 = horizons[15]["ret_bps"]
        btc_prior4h = btc_marks.ret_bps(signal_ts - 4 * 3600 * 1000, signal_ts)
        btc_after15 = btc_marks.ret_bps(fill_ts, fill_ts + 15 * 60_000)
        btc_after60 = btc_marks.ret_bps(fill_ts, fill_ts + 60 * 60_000)
        candle5 = candle_features(ohlc_after(eth_marks, fill_ts, 5), ref_price=entry_px)
        candle15 = candle_features(ohlc_after(eth_marks, fill_ts, 15), ref_price=entry_px)
        low_rebreak_px = entry_px * (1.0 - float(rebreak_bps) / 10_000.0)
        anchor_reclaim_px = anchor_mark if anchor_mark is not None else entry_px
        built = {
            "observation_id": row.get("observation_id"),
            "signal_utc": row.get("signal_utc"),
            "maker_fill_utc": row.get("maker_fill_utc"),
            "net_bps": r1(row.get("net_bps")),
            "outcome_class": "winner" if float(row.get("net_bps") or 0.0) > 0.0 else "loser",
            "fill_delay_sec": r1(row.get("fill_delay_sec")),
            "fill_delay_bucket": fill_delay_bucket(finite_float(row.get("fill_delay_sec"))),
            "vdepth_bps": r1(row.get("vdepth_bps")),
            "prior_4h_bps": r1(row.get("prior_4h_bps")),
            "prior4h_intensity_bucket": prior4h_intensity_bucket(finite_float(row.get("prior_4h_bps"))),
            "running_accel_usd_per_sec": r1(row.get("running_accel_usd_per_sec")),
            "elapsed_since_first_sec": r1(row.get("elapsed_since_first_sec")),
            "single_liq_dominance_pct": r1(row.get("single_liq_dominance_pct")),
            "entry_price": entry_px,
            "anchor_mark_price": anchor_mark,
            "low_rebreak_15m": horizons[15]["min_price"] is not None and float(horizons[15]["min_price"]) <= low_rebreak_px,
            "low_rebreak_30m": horizons[30]["min_price"] is not None and float(horizons[30]["min_price"]) <= low_rebreak_px,
            "anchor_reclaimed_15m": horizons[15]["max_price"] is not None and float(horizons[15]["max_price"]) >= anchor_reclaim_px,
            "anchor_reclaimed_30m": horizons[30]["max_price"] is not None and float(horizons[30]["max_price"]) >= anchor_reclaim_px,
            "first_15m_bucket": first_ret_bucket(ret15),
            "btc_prior_4h_bps": r1(btc_prior4h),
            "btc_after_15m_bps": r1(btc_after15),
            "btc_after_60m_bps": r1(btc_after60),
            "btc_context_bucket": btc_context_bucket(btc_prior4h, btc_after15),
            "candle5_body_bps": candle5["body_bps"],
            "candle5_lower_wick_bps": candle5["lower_wick_bps"],
            "candle5_upper_wick_bps": candle5["upper_wick_bps"],
            "candle5_close_ret_bps": candle5["close_ret_bps"],
            "candle5_pattern": candle5["pattern"],
            "candle15_body_bps": candle15["body_bps"],
            "candle15_lower_wick_bps": candle15["lower_wick_bps"],
            "candle15_upper_wick_bps": candle15["upper_wick_bps"],
            "candle15_close_ret_bps": candle15["close_ret_bps"],
            "candle15_pattern": candle15["pattern"],
        }
        for minutes, stats in horizons.items():
            built[f"mfe_{minutes}m_bps"] = stats["mfe_bps"]
            built[f"mae_{minutes}m_bps"] = stats["mae_bps"]
            built[f"ret_{minutes}m_bps"] = stats["ret_bps"]
        built["trap_tags"] = ",".join(trap_tags(built))
        out.append(built)
    out.sort(key=lambda r: int(datetime.fromisoformat(str(r["signal_utc"])).timestamp() * 1000))
    return out


def numeric_profile(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> dict[str, Any]:
    out: dict[str, Any] = {"n": len(rows)}
    for key in keys:
        vals = [float(v) for r in rows if (v := finite_float(r.get(key))) is not None]
        out[key] = {
            "median": r1(pctile(vals, 0.5)),
            "mean": r1(mean(vals)),
            "p25": r1(pctile(vals, 0.25)),
            "p75": r1(pctile(vals, 0.75)),
        }
    return out


def group_by(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    out = []
    vals = sorted({str(r.get(key)) for r in rows})
    for val in vals:
        subset = [r for r in rows if str(r.get(key)) == val]
        nets = [float(r["net_bps"]) for r in subset if finite_float(r.get("net_bps")) is not None]
        out.append(
            {
                "key": key,
                "value": val,
                "n": len(subset),
                "summary": summarize(nets),
                "loser_rate": r3(sum(1 for r in subset if r.get("outcome_class") == "loser") / len(subset)) if subset else None,
            }
        )
    return sorted(out, key=lambda r: (int(r["n"]), float(r["summary"].get("sum_bps") or -1e18)), reverse=True)


def top_cards(rows: list[dict[str, Any]], *, n: int) -> dict[str, list[dict[str, Any]]]:
    def card(r: dict[str, Any]) -> dict[str, Any]:
        return {
            "signal_utc": r.get("signal_utc"),
            "net_bps": r.get("net_bps"),
            "trap_tags": r.get("trap_tags"),
            "fill_delay_sec": r.get("fill_delay_sec"),
            "vdepth_bps": r.get("vdepth_bps"),
            "prior_4h_bps": r.get("prior_4h_bps"),
            "ret_15m_bps": r.get("ret_15m_bps"),
            "mae_30m_bps": r.get("mae_30m_bps"),
            "mfe_30m_bps": r.get("mfe_30m_bps"),
            "btc_context_bucket": r.get("btc_context_bucket"),
            "candle5_pattern": r.get("candle5_pattern"),
            "candle15_pattern": r.get("candle15_pattern"),
        }

    return {
        "winners": [card(r) for r in sorted([x for x in rows if x.get("outcome_class") == "winner"], key=lambda r: float(r["net_bps"]), reverse=True)[:n]],
        "losers": [card(r) for r in sorted([x for x in rows if x.get("outcome_class") == "loser"], key=lambda r: float(r["net_bps"]))[:n]],
    }


def no_fill_cards(ledger: list[dict[str, Any]], *, n: int) -> list[dict[str, Any]]:
    rows = [
        r
        for r in ledger
        if r.get("observation_status") == "CLOSED"
        and r.get("sim_status") == "NO_MAKER_FILL"
        and finite_float(r.get("counterfactual_anchor_mark_net_bps")) is not None
    ]
    rows.sort(key=lambda r: float(r["counterfactual_anchor_mark_net_bps"]), reverse=True)
    return [
        {
            "signal_utc": r.get("signal_utc"),
            "counterfactual_anchor_mark_net_bps": r.get("counterfactual_anchor_mark_net_bps"),
            "vdepth_bps": r.get("vdepth_bps"),
            "prior_4h_bps": r.get("prior_4h_bps"),
            "running_accel_usd_per_sec": r.get("running_accel_usd_per_sec"),
            "single_liq_dominance_pct": r.get("single_liq_dominance_pct"),
        }
        for r in rows[:n]
    ]


def build_report(rows: list[dict[str, Any]], *, ledger: list[dict[str, Any]], db_path: Path, rebreak_bps: float, top_n: int) -> dict[str, Any]:
    winners = [r for r in rows if r.get("outcome_class") == "winner"]
    losers = [r for r in rows if r.get("outcome_class") == "loser"]
    no_fill_cf = [
        float(r["counterfactual_anchor_mark_net_bps"])
        for r in ledger
        if r.get("observation_status") == "CLOSED"
        and r.get("sim_status") == "NO_MAKER_FILL"
        and finite_float(r.get("counterfactual_anchor_mark_net_bps")) is not None
    ]
    keys = (
        "net_bps",
        "fill_delay_sec",
        "vdepth_bps",
        "prior_4h_bps",
        "ret_15m_bps",
        "mae_30m_bps",
        "mfe_30m_bps",
        "btc_after_15m_bps",
        "candle5_lower_wick_bps",
        "candle5_close_ret_bps",
    )
    screens = []
    for key in (
        "fill_delay_bucket",
        "first_15m_bucket",
        "low_rebreak_15m",
        "low_rebreak_30m",
        "anchor_reclaimed_15m",
        "prior4h_intensity_bucket",
        "btc_context_bucket",
        "candle5_pattern",
        "candle15_pattern",
    ):
        screens.extend(group_by(rows, key))
    return {
        "generated_at_utc": utc_now(),
        "source_db": file_fingerprint(db_path),
        "protocol_id": PROTOCOL_ID,
        "scope": "closed FILLED observations from S34 V Engine v0.1 ledger",
        "config": {
            "symbol": SYMBOL,
            "direction": FADE_DIRECTION,
            "low_rebreak_bps": float(rebreak_bps),
        },
        "ledger_counts": {
            "total_rows": len(ledger),
            "closed_filled_rows": len(rows),
            "closed_no_fill_rows": sum(1 for r in ledger if r.get("observation_status") == "CLOSED" and r.get("sim_status") == "NO_MAKER_FILL"),
            "data_incomplete_rows": sum(1 for r in ledger if r.get("observation_status") == "DATA_INCOMPLETE"),
        },
        "overall": summarize([float(r["net_bps"]) for r in rows]),
        "winner_profile": numeric_profile(winners, keys),
        "loser_profile": numeric_profile(losers, keys),
        "group_screens": screens,
        "cards": top_cards(rows, n=int(top_n)),
        "no_fill_counterfactual": {
            "summary": summarize(no_fill_cf),
            "cards": no_fill_cards(ledger, n=int(top_n)),
        },
        "rows": rows,
    }


def cell(summary: dict[str, Any]) -> str:
    return f"N={summary['n']} sum={summary['sum_bps']} med={summary['median_bps']} T3R={summary['top3_winner_removed_sum_bps']}"


def render_md(report: dict[str, Any]) -> str:
    lines = [
        "# S34 V Engine Failure Anatomy",
        "",
        f"Generated: `{report['generated_at_utc']}`",
        "",
        f"Protocol: `{report['protocol_id']}`",
        "",
        "Diagnostic only. This report explains closed maker-fill observations; it does not change v0.1.",
        "",
        "## Sample",
        "",
        f"- ledger rows: `{report['ledger_counts']['total_rows']}`",
        f"- closed filled rows: `{report['ledger_counts']['closed_filled_rows']}`",
        f"- closed no-fill rows: `{report['ledger_counts']['closed_no_fill_rows']}`",
        f"- data incomplete rows: `{report['ledger_counts']['data_incomplete_rows']}`",
        f"- overall closed-fill labels: {cell(report['overall'])}",
        "",
        "## Winner vs Loser Profile",
        "",
        "```json",
        json.dumps({"winners": report["winner_profile"], "losers": report["loser_profile"]}, indent=2),
        "```",
        "",
        "## Trap / Leading Area Screens",
        "",
        "| Feature | Value | N | Loser% | Summary |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    for row in report["group_screens"]:
        if int(row["n"]) < 2:
            continue
        loser_pct = None if row["loser_rate"] is None else r1(row["loser_rate"] * 100.0)
        lines.append(f"| `{row['key']}` | `{row['value']}` | {row['n']} | {loser_pct} | {cell(row['summary'])} |")
    lines.extend(["", "## Top Losers", ""])
    lines.append("| Net | UTC | Tags | Fill delay | V-depth | Prior4h | Ret15 | MAE30 | BTC | C5 | C15 |")
    lines.append("| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |")
    for row in report["cards"]["losers"]:
        lines.append(
            f"| {row['net_bps']} | {row['signal_utc']} | `{row['trap_tags']}` | {row['fill_delay_sec']} | "
            f"{row['vdepth_bps']} | {row['prior_4h_bps']} | {row['ret_15m_bps']} | {row['mae_30m_bps']} | "
            f"{row['btc_context_bucket']} | {row['candle5_pattern']} | {row['candle15_pattern']} |"
        )
    lines.extend(["", "## Top Winners", ""])
    lines.append("| Net | UTC | Tags | Fill delay | V-depth | Prior4h | Ret15 | MFE30 | BTC | C5 | C15 |")
    lines.append("| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |")
    for row in report["cards"]["winners"]:
        lines.append(
            f"| {row['net_bps']} | {row['signal_utc']} | `{row['trap_tags']}` | {row['fill_delay_sec']} | "
            f"{row['vdepth_bps']} | {row['prior_4h_bps']} | {row['ret_15m_bps']} | {row['mfe_30m_bps']} | "
            f"{row['btc_context_bucket']} | {row['candle5_pattern']} | {row['candle15_pattern']} |"
        )
    lines.extend(["", "## No-Fill Counterfactual", ""])
    lines.append(f"Closed no-fill mark counterfactual: {cell(report['no_fill_counterfactual']['summary'])}")
    lines.extend(["", "| CF mark net | UTC | V-depth | Prior4h | Accel | Dominance |", "| ---: | --- | ---: | ---: | ---: | ---: |"])
    for row in report["no_fill_counterfactual"]["cards"]:
        lines.append(
            f"| {row['counterfactual_anchor_mark_net_bps']} | {row['signal_utc']} | {row['vdepth_bps']} | "
            f"{row['prior_4h_bps']} | {row['running_accel_usd_per_sec']} | {row['single_liq_dominance_pct']} |"
        )
    lines.append("")
    return "\n".join(lines)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(ANATOMY_FIELDS), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze failures in the S34 V Engine v0.1 observation ledger.")
    p.add_argument("--db", type=Path, default=DEFAULT_DB)
    p.add_argument("--ledger-jsonl", type=Path, default=DEFAULT_LEDGER_JSONL)
    p.add_argument("--low-rebreak-bps", type=float, default=10.0)
    p.add_argument("--top-n", type=int, default=10)
    p.add_argument("--json-out", type=Path, default=OUT_JSON)
    p.add_argument("--md-out", type=Path, default=OUT_MD)
    p.add_argument("--csv-out", type=Path, default=OUT_CSV)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    ledger = load_ledger(args.ledger_jsonl)
    with sqlite3.connect(f"file:{args.db}?mode=ro", uri=True) as conn:
        eth_marks = load_mark_index(conn, SYMBOL)
        btc_marks = load_mark_index(conn, "BTCUSDT")
    rows = build_anatomy_rows(ledger, eth_marks=eth_marks, btc_marks=btc_marks, rebreak_bps=float(args.low_rebreak_bps))
    report = build_report(rows, ledger=ledger, db_path=args.db, rebreak_bps=float(args.low_rebreak_bps), top_n=int(args.top_n))
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    args.md_out.write_text(render_md(report), encoding="utf-8")
    write_csv(args.csv_out, rows)
    print(render_md(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
