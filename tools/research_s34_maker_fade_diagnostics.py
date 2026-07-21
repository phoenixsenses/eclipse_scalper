"""S34 Maker Fade Diagnostics.

Deepens the SELL-liq -> maker LONG deep-V fade branch by explaining skew:
top winners/losers, V-depth bins, horizon sweep, and single-variable
conditioning. This is RESEARCH_ONLY and writes reports only.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.research_s34_knowable_anchor_continuation import (
    MarkIndex,
    file_fingerprint,
    iso_ms,
    load_mark_index,
    r1,
    r3,
    sha256_text,
)
from tools.research_s34_maker_fade import (
    NO_TP_OR_SL,
    collect_events,
    parse_float_tuple,
    simulate_event,
    summarize,
)


DEFAULT_DB = ROOT / "data" / "microstructure.db"
OUT_DIR = ROOT / "reports" / "research" / "s34"
OUT_JSON = OUT_DIR / "S34_MAKER_FADE_DIAGNOSTICS.json"
OUT_MD = OUT_DIR / "S34_MAKER_FADE_DIAGNOSTICS.md"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_int_tuple(text: str) -> tuple[int, ...]:
    vals = []
    for part in str(text).split(","):
        part = part.strip()
        if part:
            vals.append(int(part))
    if not vals:
        raise ValueError("empty int tuple")
    return tuple(vals)


def session_bucket(ts_ms: int) -> str:
    hour = datetime.fromtimestamp(int(ts_ms) / 1000.0, tz=timezone.utc).hour
    if 0 <= hour < 8:
        return "asia_00_08"
    if 8 <= hour < 13:
        return "eu_08_13"
    if 13 <= hour < 20:
        return "us_overlap_13_20"
    return "late_us_20_24"


def vdepth_bucket(v: float) -> str:
    if v < 40.0:
        return "v28_40"
    if v < 60.0:
        return "v40_60"
    if v < 100.0:
        return "v60_100"
    return "v100_plus"


def fill_delay_bucket(sec: float | None) -> str:
    if sec is None or not math.isfinite(float(sec)):
        return "no_fill"
    if sec <= 60.0:
        return "fill_0_60s"
    if sec <= 300.0:
        return "fill_1_5m"
    if sec <= 1800.0:
        return "fill_5_30m"
    return "fill_30m_plus"


def elapsed_bucket(sec: float) -> str:
    if sec <= 10.0:
        return "elapsed_0_10s"
    if sec <= 60.0:
        return "elapsed_10_60s"
    if sec <= 180.0:
        return "elapsed_1_3m"
    return "elapsed_3m_plus"


def dominance_bucket(pct: float) -> str:
    if pct <= 40.0:
        return "dominance_0_40"
    if pct <= 70.0:
        return "dominance_40_70"
    return "dominance_70_plus"


def trend_bucket(v: float | None, label: str) -> str:
    if v is None or not math.isfinite(float(v)):
        return f"{label}_na"
    if v < -50.0:
        return f"{label}_down"
    if v <= 50.0:
        return f"{label}_flat"
    return f"{label}_up"


def range_bucket(v: float | None, label: str) -> str:
    if v is None or not math.isfinite(float(v)):
        return f"{label}_na"
    if v < 80.0:
        return f"{label}_low"
    if v < 180.0:
        return f"{label}_mid"
    return f"{label}_high"


def ret_bps(marks: MarkIndex, start_ms: int, end_ms: int) -> float | None:
    return marks.ret_bps(int(start_ms), int(end_ms))


def range_bps(marks: MarkIndex, start_ms: int, end_ms: int) -> float | None:
    rows = marks.slice_range(int(start_ms), int(end_ms))
    if not rows:
        return None
    vals = [float(px) for _, px in rows if float(px) > 0.0]
    if not vals:
        return None
    ref = vals[-1]
    if ref <= 0.0:
        return None
    return (max(vals) - min(vals)) / ref * 10_000.0


def split_bucket_ids(buckets: list[int], holdout_frac: float) -> tuple[set[int], dict[str, Any]]:
    ids = sorted(set(int(b) for b in buckets))
    holdout_n = max(1, int(round(len(ids) * float(holdout_frac)))) if ids else 0
    holdout_ids = set(ids[-holdout_n:]) if holdout_n else set()
    return holdout_ids, {
        "method": "chronological_bucket_tail_holdout",
        "holdout_frac": float(holdout_frac),
        "bucket_n": len(ids),
        "holdout_bucket_n": len(holdout_ids),
        "holdout_bucket_ids_sha256": sha256_text("\n".join(str(x) for x in sorted(holdout_ids))),
    }


def add_metadata(row: dict[str, Any], *, marks: MarkIndex, btc_marks: MarkIndex, holdout_ids: set[int]) -> dict[str, Any]:
    ts = int(row["anchor_ts_ms"])
    bucket = int(row["bucket"])
    fill_delay = row.get("fill_delay_sec")
    prior_4h = ret_bps(marks, ts - 4 * 3600 * 1000, ts)
    prior_8h = ret_bps(marks, ts - 8 * 3600 * 1000, ts)
    prior_1h = ret_bps(marks, ts - 3600 * 1000, ts)
    prior_60m = prior_1h
    prior_30m = ret_bps(marks, ts - 1800 * 1000, ts)
    pre_1h_range = range_bps(marks, ts - 3600 * 1000, ts)
    pre_4h_range = range_bps(marks, ts - 4 * 3600 * 1000, ts)
    btc_4h = ret_bps(btc_marks, ts - 4 * 3600 * 1000, ts)
    row.update(
        {
            "split": "holdout" if bucket in holdout_ids else "calibration",
            "session": session_bucket(ts),
            "vdepth_bucket": vdepth_bucket(float(row["vdepth_bps"])),
            "fill_delay_bucket": fill_delay_bucket(float(fill_delay) if fill_delay is not None else None),
            "elapsed_bucket": elapsed_bucket(float(row["elapsed_since_first_sec"])),
            "dominance_bucket": dominance_bucket(float(row["running_single_liq_dominance"])),
            "prior_4h_bps": r1(prior_4h),
            "prior_8h_bps": r1(prior_8h),
            "prior_1h_bps": r1(prior_1h),
            "prior_60m_bps": r1(prior_60m),
            "prior_30m_bps": r1(prior_30m),
            "pre_1h_range_bps": r1(pre_1h_range),
            "pre_4h_range_bps": r1(pre_4h_range),
            "btc_4h_bps": r1(btc_4h),
            "prior_4h_bucket": trend_bucket(prior_4h, "prior4h"),
            "prior_8h_bucket": trend_bucket(prior_8h, "prior8h"),
            "prior_1h_bucket": trend_bucket(prior_1h, "prior1h"),
            "prior_60m_bucket": trend_bucket(prior_60m, "prior60m"),
            "prior_30m_bucket": trend_bucket(prior_30m, "prior30m"),
            "pre_1h_range_bucket": range_bucket(pre_1h_range, "range1h"),
            "pre_4h_range_bucket": range_bucket(pre_4h_range, "range4h"),
            "btc_4h_bucket": trend_bucket(btc_4h, "btc4h"),
        }
    )
    return row


def run_rows(
    conn: sqlite3.Connection,
    *,
    events: list[Any],
    marks: MarkIndex,
    btc_marks: MarkIndex,
    holdout_ids: set[int],
    offsets: tuple[float, ...],
    margins: tuple[float, ...],
    horizons_hr: tuple[int, ...],
    maker_fee_bps: float,
    taker_fee_bps: float,
    max_book_staleness_sec: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for horizon_hr in horizons_hr:
        horizon_sec = int(horizon_hr) * 3600
        for offset in offsets:
            for margin in margins:
                for event in events:
                    row = simulate_event(
                        conn,
                        event,
                        offset_bps=float(offset),
                        cross_margin_bps=float(margin),
                        horizon_sec=horizon_sec,
                        maker_fee_bps=float(maker_fee_bps),
                        taker_fee_bps=float(taker_fee_bps),
                        max_book_staleness_sec=int(max_book_staleness_sec),
                        horizon_from="fill",
                        tp_bps=NO_TP_OR_SL,
                        sl_bps=NO_TP_OR_SL,
                    )
                    row["horizon_hr"] = int(horizon_hr)
                    row["config_id"] = f"H{int(horizon_hr)}_O{float(offset):g}_C{float(margin):g}"
                    rows.append(add_metadata(row, marks=marks, btc_marks=btc_marks, holdout_ids=holdout_ids))
    return rows


def filled(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in rows if r.get("status") == "FILLED" and r.get("net_bps") is not None]


def split_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    cal = [float(r["net_bps"]) for r in rows if r.get("split") == "calibration"]
    hold = [float(r["net_bps"]) for r in rows if r.get("split") == "holdout"]
    return {"calibration": summarize(cal), "holdout": summarize(hold), "overall": summarize(cal + hold)}


def passes(summary: dict[str, Any], min_n: int) -> bool:
    cal = summary["calibration"]
    hold = summary["holdout"]
    return (
        int(cal["n"] or 0) >= int(min_n)
        and int(hold["n"] or 0) >= int(min_n)
        and float(cal["sum_bps"] or 0.0) > 0.0
        and float(hold["sum_bps"] or 0.0) > 0.0
        and float(cal["top3_winner_removed_sum_bps"] or 0.0) > 0.0
        and float(hold["top3_winner_removed_sum_bps"] or 0.0) > 0.0
    )


def base_grid(rows: list[dict[str, Any]], min_n: int) -> list[dict[str, Any]]:
    out = []
    for config_id in sorted({r["config_id"] for r in rows}):
        subset = filled([r for r in rows if r["config_id"] == config_id])
        summary = split_summary(subset)
        example = next(r for r in rows if r["config_id"] == config_id)
        out.append(
            {
                "config_id": config_id,
                "horizon_hr": int(example["horizon_hr"]),
                "offset_bps": float(example["offset_bps"]),
                "cross_margin_bps": float(example["cross_margin_bps"]),
                "filled_n": len(subset),
                "fill_rate": r3(len(subset) / sum(1 for r in rows if r["config_id"] == config_id)),
                "summary": summary,
                "pass": passes(summary, min_n),
            }
        )
    return sorted(
        out,
        key=lambda r: (
            bool(r["pass"]),
            float(r["summary"]["holdout"].get("sum_bps") or -1e18),
            float(r["summary"]["calibration"].get("sum_bps") or -1e18),
        ),
        reverse=True,
    )


def vdepth_horizon_table(rows: list[dict[str, Any]], *, focus_offset: float, focus_cross: float) -> list[dict[str, Any]]:
    out = []
    focus = [
        r
        for r in rows
        if float(r["offset_bps"]) == float(focus_offset)
        and float(r["cross_margin_bps"]) == float(focus_cross)
        and r.get("status") == "FILLED"
    ]
    for horizon in sorted({int(r["horizon_hr"]) for r in focus}):
        for bucket in ("v28_40", "v40_60", "v60_100", "v100_plus"):
            subset = [r for r in focus if int(r["horizon_hr"]) == horizon and r["vdepth_bucket"] == bucket]
            out.append({"horizon_hr": horizon, "vdepth_bucket": bucket, "summary": split_summary(subset)})
    return out


def conditioning_leaderboard(rows: list[dict[str, Any]], *, min_n: int) -> list[dict[str, Any]]:
    condition_keys = (
        "vdepth_bucket",
        "session",
        "fill_delay_bucket",
        "elapsed_bucket",
        "dominance_bucket",
        "prior_30m_bucket",
        "prior_1h_bucket",
        "prior_4h_bucket",
        "prior_8h_bucket",
        "pre_1h_range_bucket",
        "pre_4h_range_bucket",
        "btc_4h_bucket",
    )
    out = []
    for config_id in sorted({r["config_id"] for r in rows}):
        config_rows = filled([r for r in rows if r["config_id"] == config_id])
        if not config_rows:
            continue
        example = config_rows[0]
        for key in condition_keys:
            for value in sorted({str(r.get(key)) for r in config_rows}):
                subset = [r for r in config_rows if str(r.get(key)) == value]
                summary = split_summary(subset)
                out.append(
                    {
                        "config_id": config_id,
                        "horizon_hr": int(example["horizon_hr"]),
                        "offset_bps": float(example["offset_bps"]),
                        "cross_margin_bps": float(example["cross_margin_bps"]),
                        "condition": key,
                        "value": value,
                        "summary": summary,
                        "pass": passes(summary, min_n),
                    }
                )
    return sorted(
        out,
        key=lambda r: (
            bool(r["pass"]),
            float(r["summary"]["holdout"].get("top3_winner_removed_sum_bps") or -1e18),
            float(r["summary"]["holdout"].get("sum_bps") or -1e18),
        ),
        reverse=True,
    )


def multi_feature_leaderboard(
    rows: list[dict[str, Any]],
    *,
    min_n: int,
    max_combo_size: int,
) -> list[dict[str, Any]]:
    condition_keys = (
        "vdepth_bucket",
        "prior_4h_bucket",
        "prior_8h_bucket",
        "prior_1h_bucket",
        "prior_30m_bucket",
        "elapsed_bucket",
        "session",
        "dominance_bucket",
        "fill_delay_bucket",
        "pre_1h_range_bucket",
        "pre_4h_range_bucket",
        "btc_4h_bucket",
    )
    out = []
    for config_id in sorted({r["config_id"] for r in rows}):
        config_rows = filled([r for r in rows if r["config_id"] == config_id])
        if not config_rows:
            continue
        example = config_rows[0]
        for size in range(2, int(max_combo_size) + 1):
            for keys in itertools.combinations(condition_keys, size):
                grouped: dict[tuple[str, ...], list[dict[str, Any]]] = {}
                for row in config_rows:
                    vals = tuple(str(row.get(key)) for key in keys)
                    if any(v.endswith("_na") or v == "None" for v in vals):
                        continue
                    grouped.setdefault(vals, []).append(row)
                for vals, subset in grouped.items():
                    summary = split_summary(subset)
                    if int(summary["calibration"]["n"] or 0) < int(min_n) or int(summary["holdout"]["n"] or 0) < int(min_n):
                        continue
                    out.append(
                        {
                            "config_id": config_id,
                            "horizon_hr": int(example["horizon_hr"]),
                            "offset_bps": float(example["offset_bps"]),
                            "cross_margin_bps": float(example["cross_margin_bps"]),
                            "conditions": [{"key": k, "value": v} for k, v in zip(keys, vals)],
                            "condition_label": " & ".join(f"{k}={v}" for k, v in zip(keys, vals)),
                            "summary": summary,
                            "pass": passes(summary, min_n),
                        }
                    )
    return sorted(
        out,
        key=lambda r: (
            bool(r["pass"]),
            float(r["summary"]["calibration"].get("top3_winner_removed_sum_bps") or -1e18),
            float(r["summary"]["holdout"].get("top3_winner_removed_sum_bps") or -1e18),
            float(r["summary"]["holdout"].get("sum_bps") or -1e18),
        ),
        reverse=True,
    )


def trade_cards(rows: list[dict[str, Any]], *, focus_horizon: int, focus_offset: float, focus_cross: float, n: int) -> dict[str, list[dict[str, Any]]]:
    focus = [
        r
        for r in filled(rows)
        if int(r["horizon_hr"]) == int(focus_horizon)
        and float(r["offset_bps"]) == float(focus_offset)
        and float(r["cross_margin_bps"]) == float(focus_cross)
    ]

    def card(r: dict[str, Any]) -> dict[str, Any]:
        return {
            "anchor_utc": r.get("anchor_utc"),
            "split": r.get("split"),
            "net_bps": r1(float(r["net_bps"])),
            "vdepth_bps": r1(float(r["vdepth_bps"])),
            "vdepth_bucket": r.get("vdepth_bucket"),
            "session": r.get("session"),
            "fill_delay_sec": r1(float(r.get("fill_delay_sec") or 0.0)),
            "elapsed_since_first_sec": r1(float(r.get("elapsed_since_first_sec") or 0.0)),
            "dominance_pct": r1(float(r.get("running_single_liq_dominance") or 0.0)),
            "prior_4h_bps": r.get("prior_4h_bps"),
            "prior_8h_bps": r.get("prior_8h_bps"),
            "prior_1h_bps": r.get("prior_1h_bps"),
            "prior_30m_bps": r.get("prior_30m_bps"),
            "pre_1h_range_bps": r.get("pre_1h_range_bps"),
            "btc_4h_bps": r.get("btc_4h_bps"),
            "entry_price": r1(float(r.get("entry_price") or 0.0)),
            "exit_price": r1(float(r.get("exit_price") or 0.0)),
        }

    winners = [card(r) for r in sorted(focus, key=lambda r: float(r["net_bps"]), reverse=True)[:n]]
    losers = [card(r) for r in sorted(focus, key=lambda r: float(r["net_bps"]))[:n]]
    return {"winners": winners, "losers": losers}


def winner_loser_profile(cards: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    def profile(items: list[dict[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {"n": len(items)}
        for key in ("vdepth_bps", "fill_delay_sec", "elapsed_since_first_sec", "dominance_pct", "prior_4h_bps", "prior_8h_bps", "btc_4h_bps"):
            vals = [float(x[key]) for x in items if x.get(key) is not None and math.isfinite(float(x[key]))]
            out[f"{key}_median"] = r1(sorted(vals)[len(vals) // 2]) if vals else None
        for key in ("vdepth_bucket", "session"):
            counts: dict[str, int] = {}
            for item in items:
                counts[str(item.get(key))] = counts.get(str(item.get(key)), 0) + 1
            out[key] = dict(sorted(counts.items()))
        return out

    return {"winners": profile(cards["winners"]), "losers": profile(cards["losers"])}


def render_summary_cell(summary: dict[str, Any], split: str) -> str:
    s = summary[split]
    return f"N={s['n']} sum={s['sum_bps']} med={s['median_bps']} T3R={s['top3_winner_removed_sum_bps']}"


def render_md(report: dict[str, Any]) -> str:
    cfg = report["config"]
    lines = [
        "# S34 Maker Fade Diagnostics",
        "",
        f"Generated: `{report['generated_at_utc']}`",
        "",
        f"Scope: `{cfg['symbol']}` SELL-liq -> maker LONG, threshold `{int(cfg['threshold']/1000)}K`, min V-depth `{cfg['min_vdepth_bps']}`bps.",
        "",
        f"Events: `{report['event_n']}`. Focus config: H{cfg['focus_horizon_hr']} O{cfg['focus_offset_bps']} C{cfg['focus_cross_margin_bps']}.",
        "",
        "## Base Grid",
        "",
        "| Config | Fill% | Cal | Hold | Pass |",
        "| --- | ---: | --- | --- | --- |",
    ]
    for row in report["base_grid"][:30]:
        lines.append(
            f"| `{row['config_id']}` | {None if row['fill_rate'] is None else r1(row['fill_rate'] * 100.0)} "
            f"| {render_summary_cell(row['summary'], 'calibration')} | {render_summary_cell(row['summary'], 'holdout')} | {'YES' if row['pass'] else ''} |"
        )
    lines.extend(["", "## V-Depth x Horizon", ""])
    lines.append("| Horizon | V-depth | Cal | Hold |")
    lines.append("| ---: | --- | --- | --- |")
    for row in report["vdepth_horizon"]:
        lines.append(
            f"| {row['horizon_hr']}h | {row['vdepth_bucket']} | {render_summary_cell(row['summary'], 'calibration')} | {render_summary_cell(row['summary'], 'holdout')} |"
        )
    lines.extend(["", "## Conditioning Leaderboard", ""])
    lines.append("| Config | Condition | Cal | Hold | Pass |")
    lines.append("| --- | --- | --- | --- | --- |")
    for row in report["conditioning"][:40]:
        lines.append(
            f"| `{row['config_id']}` | `{row['condition']}={row['value']}` | "
            f"{render_summary_cell(row['summary'], 'calibration')} | {render_summary_cell(row['summary'], 'holdout')} | {'YES' if row['pass'] else ''} |"
        )
    lines.extend(["", "## Multi-Feature Conditioning", ""])
    lines.append("| Config | Conditions | Cal | Hold | Pass |")
    lines.append("| --- | --- | --- | --- | --- |")
    for row in report["multi_feature_conditioning"][:60]:
        lines.append(
            f"| `{row['config_id']}` | `{row['condition_label']}` | "
            f"{render_summary_cell(row['summary'], 'calibration')} | {render_summary_cell(row['summary'], 'holdout')} | {'YES' if row['pass'] else ''} |"
        )
    lines.extend(["", "## Top Winners", ""])
    lines.append("| Net | Split | UTC | V-depth | Session | Fill delay | Elapsed | Dom | Prior4h | BTC4h |")
    lines.append("| ---: | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |")
    for row in report["trade_cards"]["winners"]:
        lines.append(
            f"| {row['net_bps']} | {row['split']} | {row['anchor_utc']} | {row['vdepth_bps']} | {row['session']} | "
            f"{row['fill_delay_sec']} | {row['elapsed_since_first_sec']} | {row['dominance_pct']} | {row['prior_4h_bps']} | {row['btc_4h_bps']} |"
        )
    lines.extend(["", "## Top Losers", ""])
    lines.append("| Net | Split | UTC | V-depth | Session | Fill delay | Elapsed | Dom | Prior4h | BTC4h |")
    lines.append("| ---: | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |")
    for row in report["trade_cards"]["losers"]:
        lines.append(
            f"| {row['net_bps']} | {row['split']} | {row['anchor_utc']} | {row['vdepth_bps']} | {row['session']} | "
            f"{row['fill_delay_sec']} | {row['elapsed_since_first_sec']} | {row['dominance_pct']} | {row['prior_4h_bps']} | {row['btc_4h_bps']} |"
        )
    lines.extend(["", "## Winner/Loser Profile", "", "```json", json.dumps(report["winner_loser_profile"], indent=2), "```", ""])
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose skew in SELL-liq maker LONG deep-V fade.")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--symbol", default="ETHUSDT")
    parser.add_argument("--threshold", type=float, default=200_000.0)
    parser.add_argument("--min-vdepth-bps", type=float, default=28.0)
    parser.add_argument("--horizons-hr", default="1,2,4,8,12,24")
    parser.add_argument("--offset-bps", default="5,10,20")
    parser.add_argument("--cross-margin-bps", default="1,2,5")
    parser.add_argument("--maker-fee-bps", type=float, default=2.0)
    parser.add_argument("--taker-fee-bps", type=float, default=3.05)
    parser.add_argument("--max-book-staleness-sec", type=int, default=10)
    parser.add_argument("--holdout-frac", type=float, default=0.30)
    parser.add_argument("--min-n", type=int, default=10)
    parser.add_argument("--combo-min-n", type=int, default=5)
    parser.add_argument("--max-combo-size", type=int, default=3)
    parser.add_argument("--focus-horizon-hr", type=int, default=4)
    parser.add_argument("--focus-offset-bps", type=float, default=20.0)
    parser.add_argument("--focus-cross-margin-bps", type=float, default=1.0)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--json-out", type=Path, default=OUT_JSON)
    parser.add_argument("--md-out", type=Path, default=OUT_MD)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    horizons = parse_int_tuple(args.horizons_hr)
    offsets = parse_float_tuple(args.offset_bps)
    margins = parse_float_tuple(args.cross_margin_bps)
    max_horizon_sec = max(horizons) * 3600
    with sqlite3.connect(f"file:{args.db}?mode=ro", uri=True) as conn:
        marks = load_mark_index(conn, args.symbol)
        btc_marks = load_mark_index(conn, "BTCUSDT")
        events = collect_events(
            conn,
            symbol=args.symbol,
            threshold=float(args.threshold),
            sides=("SELL",),
            min_vdepth_bps=float(args.min_vdepth_bps),
            bucket_sec=300,
            min_gap_sec=900,
            accel_window_sec=30,
            max_horizon_sec=max_horizon_sec,
        )
        holdout_ids, split = split_bucket_ids([int(ev.anchor.bucket) for ev in events], float(args.holdout_frac))
        rows = run_rows(
            conn,
            events=events,
            marks=marks,
            btc_marks=btc_marks,
            holdout_ids=holdout_ids,
            offsets=offsets,
            margins=margins,
            horizons_hr=horizons,
            maker_fee_bps=float(args.maker_fee_bps),
            taker_fee_bps=float(args.taker_fee_bps),
            max_book_staleness_sec=int(args.max_book_staleness_sec),
        )
    cards = trade_cards(
        rows,
        focus_horizon=int(args.focus_horizon_hr),
        focus_offset=float(args.focus_offset_bps),
        focus_cross=float(args.focus_cross_margin_bps),
        n=int(args.top_n),
    )
    payload = {
        "generated_at_utc": utc_now(),
        "source_db": file_fingerprint(args.db),
        "config": {
            "symbol": args.symbol,
            "threshold": float(args.threshold),
            "min_vdepth_bps": float(args.min_vdepth_bps),
            "horizons_hr": list(horizons),
            "offset_bps": list(offsets),
            "cross_margin_bps": list(margins),
            "maker_fee_bps": float(args.maker_fee_bps),
            "taker_fee_bps": float(args.taker_fee_bps),
            "max_book_staleness_sec": int(args.max_book_staleness_sec),
            "holdout_frac": float(args.holdout_frac),
            "min_n": int(args.min_n),
            "combo_min_n": int(args.combo_min_n),
            "max_combo_size": int(args.max_combo_size),
            "focus_horizon_hr": int(args.focus_horizon_hr),
            "focus_offset_bps": float(args.focus_offset_bps),
            "focus_cross_margin_bps": float(args.focus_cross_margin_bps),
        },
        "split": split,
        "event_n": len(events),
        "events_sha256": sha256_text("\n".join(f"{ev.side}:{ev.anchor.anchor_ts_ms}:{ev.vdepth_bps:.6f}" for ev in events)),
        "base_grid": base_grid(rows, int(args.min_n)),
        "vdepth_horizon": vdepth_horizon_table(rows, focus_offset=float(args.focus_offset_bps), focus_cross=float(args.focus_cross_margin_bps)),
        "conditioning": conditioning_leaderboard(rows, min_n=int(args.min_n)),
        "multi_feature_conditioning": multi_feature_leaderboard(
            rows,
            min_n=int(args.combo_min_n),
            max_combo_size=int(args.max_combo_size),
        ),
        "trade_cards": cards,
        "winner_loser_profile": winner_loser_profile(cards),
        "rows": rows,
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    args.md_out.write_text(render_md(payload), encoding="utf-8")
    print(render_md(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
