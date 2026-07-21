"""Winner-extension research for the current S34 V Engine route.

Tests whether recovery-confirmed winners should be held longer than the current
2h fixed exit. Same entry lifecycle as live/current research:
O20 -> wait 300s -> O5, cross margin C1.

Research-only. No live/paper state changes.
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.research_s34_knowable_anchor_continuation import book_at, file_fingerprint, load_mark_index, r1, r3, signed_return_bps
from tools.research_s34_maker_fade import collect_events, maker_limit_price, summarize
from tools.s34_v_engine_cancel_replace import find_fill_between
from tools.s34_v_engine_confirmation_cost_current import btc_context
from tools.s34_v_engine_failure_anatomy import candle_features, finite_float, ohlc_after
from tools.s34_v_engine_shadow_observer import ACCEL_WINDOW_SEC, BUCKET_SEC, MIN_GAP_SEC, PRIOR4H_LT_BPS, PROTOCOL_ID, SYMBOL, THRESHOLD_USD, VDEPTH_MAX_BPS, VDEPTH_MIN_BPS, utc_now


DEFAULT_DB = ROOT / "data" / "microstructure.db"
OUT_DIR = ROOT / "reports" / "research" / "s34"
OUT_JSON = OUT_DIR / "S34_V_ENGINE_WINNER_EXTENSION.json"
OUT_MD = OUT_DIR / "S34_V_ENGINE_WINNER_EXTENSION.md"

INITIAL_OFFSET_BPS = 20.0
REPLACE_OFFSET_BPS = 5.0
WAIT_SEC = 300
CROSS_MARGIN_BPS = 1.0
MAKER_FEE_BPS = 2.0
TAKER_FEE_BPS = 3.05
FADE_DIRECTION = "LONG"
MAX_HORIZON_HR = 8.0


def prior_return_bps(marks: Any, ts_ms: int, window_sec: int) -> float | None:
    return marks.ret_bps(int(ts_ms) - int(window_sec) * 1000, int(ts_ms))


def collect_current_events(conn: sqlite3.Connection) -> list[Any]:
    marks = load_mark_index(conn, SYMBOL)
    events = collect_events(
        conn,
        symbol=SYMBOL,
        threshold=THRESHOLD_USD,
        sides=("SELL",),
        min_vdepth_bps=VDEPTH_MIN_BPS,
        bucket_sec=BUCKET_SEC,
        min_gap_sec=MIN_GAP_SEC,
        accel_window_sec=ACCEL_WINDOW_SEC,
        max_horizon_sec=int(MAX_HORIZON_HR * 3600),
    )
    out = []
    for ev in events:
        if not (VDEPTH_MIN_BPS <= float(ev.vdepth_bps) < VDEPTH_MAX_BPS):
            continue
        prior = prior_return_bps(marks, int(ev.anchor.anchor_ts_ms), 4 * 3600)
        if prior is None or not math.isfinite(float(prior)) or not (float(prior) < PRIOR4H_LT_BPS):
            continue
        out.append(ev)
    return out


def current_fill(event: Any) -> tuple[int, float, str] | None:
    anchor_ts = int(event.anchor_mark_ts_ms)
    cancel_ts = anchor_ts + WAIT_SEC * 1000
    initial_limit = maker_limit_price(event.anchor_mark_price, event.fade_direction, INITIAL_OFFSET_BPS)
    initial = find_fill_between(
        event,
        limit_px=initial_limit,
        cross_margin_bps=CROSS_MARGIN_BPS,
        start_ts_ms=anchor_ts,
        end_ts_ms=cancel_ts,
    )
    if initial:
        return int(initial[0]), float(initial[1]), "initial"
    replacement_limit = maker_limit_price(event.anchor_mark_price, event.fade_direction, REPLACE_OFFSET_BPS)
    repl = find_fill_between(
        event,
        limit_px=replacement_limit,
        cross_margin_bps=CROSS_MARGIN_BPS,
        start_ts_ms=cancel_ts,
        end_ts_ms=None,
    )
    if repl:
        return int(repl[0]), float(repl[1]), "replacement"
    return None


def exit_net(conn: sqlite3.Connection, event: Any, *, fill_ts_ms: int, entry_px: float, horizon_hr: float, max_book_staleness_sec: int) -> tuple[float | None, int, str]:
    exit_ts = int(fill_ts_ms) + int(float(horizon_hr) * 3600) * 1000
    quote = book_at(conn, event.symbol, exit_ts, max_book_staleness_sec)
    if not quote:
        return None, exit_ts, "no_book"
    gross = signed_return_bps(event.fade_direction, float(entry_px), float(quote.bid))
    return gross - MAKER_FEE_BPS - TAKER_FEE_BPS, exit_ts, "book_ticker"


def path_ret(marks: Any, *, entry_px: float, fill_ts_ms: int, minutes: int) -> dict[str, Any]:
    rows = [(int(ts), float(px)) for ts, px in marks.slice_range(int(fill_ts_ms), int(fill_ts_ms) + int(minutes) * 60_000)]
    if not rows:
        return {"ret": None, "max_price": None}
    end = marks.at_or_after(int(fill_ts_ms) + int(minutes) * 60_000)
    ret = None if not end else signed_return_bps(FADE_DIRECTION, float(entry_px), float(end[1]))
    return {"ret": r1(ret), "max_price": max(px for _, px in rows)}


def annotate(event: Any, *, fill_ts_ms: int, entry_px: float, eth_marks: Any, btc_marks: Any, minutes: int) -> dict[str, Any]:
    path = path_ret(eth_marks, entry_px=entry_px, fill_ts_ms=fill_ts_ms, minutes=minutes)
    btc_prior = btc_marks.ret_bps(int(event.anchor.anchor_ts_ms) - 4 * 3600 * 1000, int(event.anchor.anchor_ts_ms))
    btc_after = btc_marks.ret_bps(int(fill_ts_ms), int(fill_ts_ms) + int(minutes) * 60_000)
    candle = candle_features(ohlc_after(eth_marks, int(fill_ts_ms), minutes), ref_price=entry_px)
    return {
        "confirm_min": int(minutes),
        "anchor_reclaimed": bool(path["max_price"] is not None and float(path["max_price"]) >= float(event.anchor_mark_price)),
        "ret_confirm_bps": path["ret"],
        "btc_context_bucket": btc_context(btc_prior, btc_after),
        "candle_pattern": candle["pattern"],
    }


def condition_pass(row: dict[str, Any], condition: str) -> bool:
    if condition == "all":
        return True
    if condition == "anchor_reclaimed":
        return bool(row.get("anchor_reclaimed"))
    if condition == "btc_not_down_continues":
        return row.get("btc_context_bucket") != "btc_down_continues"
    if condition == "anchor_and_btc":
        return bool(row.get("anchor_reclaimed")) and row.get("btc_context_bucket") != "btc_down_continues"
    if condition == "strong_rebound":
        return finite_float(row.get("ret_confirm_bps")) is not None and float(row["ret_confirm_bps"]) >= 25.0
    if condition == "bull_reclaim":
        return row.get("candle_pattern") == "bull_reclaim"
    raise ValueError(condition)


def build_rows(conn: sqlite3.Connection, *, max_book_staleness_sec: int) -> list[dict[str, Any]]:
    eth_marks = load_mark_index(conn, SYMBOL)
    btc_marks = load_mark_index(conn, "BTCUSDT")
    rows = []
    for event in collect_current_events(conn):
        fill = current_fill(event)
        if not fill:
            continue
        fill_ts, entry_px, fill_leg = fill
        ann15 = annotate(event, fill_ts_ms=fill_ts, entry_px=entry_px, eth_marks=eth_marks, btc_marks=btc_marks, minutes=15)
        ann30 = annotate(event, fill_ts_ms=fill_ts, entry_px=entry_px, eth_marks=eth_marks, btc_marks=btc_marks, minutes=30)
        horizon_nets = {}
        horizon_status = {}
        for horizon in (1.0, 2.0, 4.0, 6.0, 8.0):
            net, exit_ts, source = exit_net(conn, event, fill_ts_ms=fill_ts, entry_px=entry_px, horizon_hr=horizon, max_book_staleness_sec=max_book_staleness_sec)
            horizon_nets[f"h{horizon:g}_net_bps"] = r1(net)
            horizon_status[f"h{horizon:g}_source"] = source
            horizon_status[f"h{horizon:g}_exit_ts_ms"] = exit_ts
        rows.append(
            {
                "event_id": event.anchor.event_id,
                "bucket": int(event.anchor.bucket),
                "anchor_ts_ms": int(event.anchor.anchor_ts_ms),
                "fill_ts_ms": fill_ts,
                "fill_leg": fill_leg,
                "entry_price": entry_px,
                "vdepth_bps": r1(event.vdepth_bps),
                **{f"m15_{k}": v for k, v in ann15.items()},
                **{f"m30_{k}": v for k, v in ann30.items()},
                **horizon_nets,
                **horizon_status,
            }
        )
    rows.sort(key=lambda r: int(r["anchor_ts_ms"]))
    return rows


def eval_cells(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cells = []
    for confirm_min in (15, 30):
        for condition in ("all", "anchor_reclaimed", "btc_not_down_continues", "anchor_and_btc", "strong_rebound", "bull_reclaim"):
            subset = [r for r in rows if condition_pass({k.removeprefix(f"m{confirm_min}_"): v for k, v in r.items() if k.startswith(f"m{confirm_min}_")}, condition)]
            for horizon in (1.0, 2.0, 4.0, 6.0, 8.0):
                key = f"h{horizon:g}_net_bps"
                vals = [float(v) for r in subset if (v := finite_float(r.get(key))) is not None]
                cells.append(
                    {
                        "confirm_min": confirm_min,
                        "condition": condition,
                        "horizon_hr": horizon,
                        "n": len(vals),
                        "summary": summarize(vals),
                    }
                )
    cells.sort(
        key=lambda r: (
            float(r["summary"]["top3_winner_removed_sum_bps"] or -1e18),
            float(r["summary"]["sum_bps"] or -1e18),
        ),
        reverse=True,
    )
    return cells


def build_report(conn: sqlite3.Connection, *, db_path: Path, max_book_staleness_sec: int) -> dict[str, Any]:
    rows = build_rows(conn, max_book_staleness_sec=max_book_staleness_sec)
    cells = eval_cells(rows)
    baseline = summarize([float(v) for r in rows if (v := finite_float(r.get("h2_net_bps"))) is not None])
    for cell in cells:
        cell["delta_vs_baseline_sum_bps"] = r1(float(cell["summary"]["sum_bps"] or 0.0) - float(baseline["sum_bps"] or 0.0))
        cell["delta_vs_baseline_t3r_bps"] = r1(float(cell["summary"]["top3_winner_removed_sum_bps"] or 0.0) - float(baseline["top3_winner_removed_sum_bps"] or 0.0))
    return {
        "generated_at_utc": utc_now(),
        "source_db": file_fingerprint(db_path),
        "protocol_id": PROTOCOL_ID,
        "config": {
            "entry": "O20_W300_O5_C1",
            "horizons_hr": [1, 2, 4, 6, 8],
            "confirm_minutes": [15, 30],
        },
        "filled_n": len(rows),
        "baseline_h2": baseline,
        "cells": cells,
        "rows": rows,
    }


def cell_str(summary: dict[str, Any]) -> str:
    return f"N={summary['n']} sum={summary['sum_bps']} med={summary['median_bps']} T3R={summary['top3_winner_removed_sum_bps']} max_loss={summary['max_loss_bps']}"


def render_md(report: dict[str, Any]) -> str:
    lines = [
        "# S34 V Engine Winner Extension",
        "",
        f"Generated: `{report['generated_at_utc']}`",
        "",
        f"Filled rows: `{report['filled_n']}`",
        "",
        f"Baseline H2: {cell_str(report['baseline_h2'])}",
        "",
        "| Rank | Confirm | Condition | Horizon | Summary | Delta sum | Delta T3R |",
        "| ---: | ---: | --- | ---: | --- | ---: | ---: |",
    ]
    for idx, row in enumerate(report["cells"][:40], start=1):
        lines.append(
            f"| {idx} | {row['confirm_min']}m | `{row['condition']}` | {row['horizon_hr']}h | "
            f"{cell_str(row['summary'])} | {row['delta_vs_baseline_sum_bps']} | {row['delta_vs_baseline_t3r_bps']} |"
        )
    positives = [r for r in report["cells"] if r["horizon_hr"] > 2.0 and float(r["delta_vs_baseline_t3r_bps"] or 0.0) > 0.0]
    lines.extend(["", "## Read", ""])
    lines.append(f"- Longer-hold positive T3R cells: `{len(positives)}`.")
    if positives:
        best = positives[0]
        lines.append(
            f"- Best longer hold: {best['confirm_min']}m `{best['condition']}` {best['horizon_hr']}h -> {cell_str(best['summary'])}."
        )
    lines.append("")
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run S34 winner-extension research.")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--max-book-staleness-sec", type=int, default=10)
    parser.add_argument("--json-out", type=Path, default=OUT_JSON)
    parser.add_argument("--md-out", type=Path, default=OUT_MD)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    with sqlite3.connect(f"file:{args.db}?mode=ro", uri=True) as conn:
        report = build_report(conn, db_path=args.db, max_book_staleness_sec=int(args.max_book_staleness_sec))
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    args.md_out.write_text(render_md(report), encoding="utf-8")
    print(render_md(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
