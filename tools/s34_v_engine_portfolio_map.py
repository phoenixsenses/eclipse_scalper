"""S34 V Engine portfolio roadmap screen.

Research-only batch screen for the next alpha-family questions:

1. Mirror: BUY liquidation -> maker SHORT.
2. Multi-asset: apply the same V-fade methodology to BTC/ETH/SOL.
3. Pattern families: classify whether a route is a candidate, a weak lead, or
   only a data-building lane.

This does not touch live, paper, or microstructure DB state.
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

from tools.research_s34_knowable_anchor_continuation import (  # noqa: E402
    file_fingerprint,
    load_mark_index,
    r1,
    r3,
    sha256_text,
)
from tools.research_s34_maker_fade import (  # noqa: E402
    NO_TP_OR_SL,
    collect_events,
    simulate_event,
    summarize,
)


DEFAULT_DB = ROOT / "data" / "microstructure.db"
OUT_DIR = ROOT / "reports" / "research" / "s34"
OUT_JSON = OUT_DIR / "S34_V_ENGINE_PORTFOLIO_MAP.json"
OUT_MD = OUT_DIR / "S34_V_ENGINE_PORTFOLIO_MAP.md"

SYMBOL_THRESHOLDS = {
    "ETHUSDT": (100_000.0, 150_000.0, 200_000.0, 300_000.0),
    "BTCUSDT": (250_000.0, 500_000.0, 1_000_000.0, 2_000_000.0),
    "SOLUSDT": (25_000.0, 50_000.0, 100_000.0, 200_000.0),
}
VDEPTH_BANDS = ((20.0, 28.0), (28.0, 40.0), (40.0, 60.0), (60.0, 100_000.0))
HORIZON_HR = (1.0, 2.0, 4.0)
OFFSET_BPS = 20.0
CROSS_MARGIN_BPS = 1.0
BUCKET_SEC = 300
MIN_GAP_SEC = 900
ACCEL_WINDOW_SEC = 30


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def prior_return_bps(marks: Any, ts_ms: int, window_sec: int) -> float | None:
    return marks.ret_bps(int(ts_ms) - int(window_sec) * 1000, int(ts_ms))


def side_prior_ok(side: str, prior4h: float | None, threshold_bps: float) -> bool:
    if prior4h is None or not math.isfinite(float(prior4h)):
        return False
    if side == "SELL":
        return float(prior4h) < -abs(float(threshold_bps))
    if side == "BUY":
        return float(prior4h) > abs(float(threshold_bps))
    return False


def split_by_time(rows: list[dict[str, Any]], holdout_frac: float) -> tuple[set[int], dict[str, Any]]:
    buckets = sorted({int(r["bucket"]) for r in rows})
    holdout_n = max(1, int(round(len(buckets) * float(holdout_frac)))) if buckets else 0
    holdout_ids = set(buckets[-holdout_n:]) if holdout_n else set()
    return holdout_ids, {
        "method": "chronological_bucket_tail_holdout",
        "holdout_frac": float(holdout_frac),
        "bucket_n": len(buckets),
        "holdout_bucket_n": len(holdout_ids),
        "holdout_bucket_ids_sha256": sha256_text("\n".join(str(x) for x in sorted(holdout_ids))),
    }


def score_cell(rows: list[dict[str, Any]], holdout_frac: float, min_cal_n: int, min_hold_n: int) -> dict[str, Any]:
    holdout_ids, split = split_by_time(rows, holdout_frac)
    for row in rows:
        row["split"] = "holdout" if int(row["bucket"]) in holdout_ids else "calibration"
    cal = [float(r["net_bps"]) for r in rows if r.get("split") == "calibration"]
    hold = [float(r["net_bps"]) for r in rows if r.get("split") == "holdout"]
    cal_s = summarize(cal)
    hold_s = summarize(hold)
    all_s = summarize(cal + hold)
    hard_pass = (
        int(cal_s["n"] or 0) >= int(min_cal_n)
        and int(hold_s["n"] or 0) >= int(min_hold_n)
        and float(cal_s["sum_bps"] or 0.0) > 0.0
        and float(hold_s["sum_bps"] or 0.0) > 0.0
        and float(cal_s["top3_winner_removed_sum_bps"] or 0.0) > 0.0
        and float(hold_s["top3_winner_removed_sum_bps"] or 0.0) > 0.0
    )
    weak_lead = (
        int(all_s["n"] or 0) >= int(min_cal_n + min_hold_n)
        and float(all_s["sum_bps"] or 0.0) > 0.0
        and float(all_s["top3_winner_removed_sum_bps"] or 0.0) > 0.0
        and float(all_s["median_bps"] or 0.0) > 0.0
    )
    return {
        "split": split,
        "calibration": cal_s,
        "holdout": hold_s,
        "overall": all_s,
        "hard_pass": hard_pass,
        "weak_lead": bool(weak_lead and not hard_pass),
    }


def verdict(cell: dict[str, Any]) -> str:
    if cell["hard_pass"]:
        return "HARD_PASS"
    if cell["weak_lead"]:
        return "WEAK_LEAD"
    if int(cell["overall"]["n"] or 0) < 5:
        return "DATA_BUILD"
    return "REJECT"


def run_screen(
    conn: sqlite3.Connection,
    *,
    symbols: tuple[str, ...],
    holdout_frac: float,
    prior_threshold_bps: float,
    maker_fee_bps: float,
    taker_fee_bps: float,
    max_book_staleness_sec: int,
    min_cal_n: int,
    min_hold_n: int,
) -> dict[str, Any]:
    cells: list[dict[str, Any]] = []
    coverage: dict[str, Any] = {}
    max_horizon_sec = int(max(HORIZON_HR) * 3600)
    for symbol in symbols:
        marks = load_mark_index(conn, symbol)
        thresholds = SYMBOL_THRESHOLDS.get(symbol, SYMBOL_THRESHOLDS["ETHUSDT"])
        coverage[symbol] = {"thresholds": list(thresholds)}
        for threshold in thresholds:
            events = collect_events(
                conn,
                symbol=symbol,
                threshold=float(threshold),
                sides=("SELL", "BUY"),
                min_vdepth_bps=min(lo for lo, _ in VDEPTH_BANDS),
                bucket_sec=BUCKET_SEC,
                min_gap_sec=MIN_GAP_SEC,
                accel_window_sec=ACCEL_WINDOW_SEC,
                max_horizon_sec=max_horizon_sec,
            )
            for side in ("SELL", "BUY"):
                side_events = [ev for ev in events if ev.side == side]
                for v_lo, v_hi in VDEPTH_BANDS:
                    band_events = [ev for ev in side_events if float(v_lo) <= float(ev.vdepth_bps) < float(v_hi)]
                    for horizon_hr in HORIZON_HR:
                        rows = []
                        for event in band_events:
                            prior4h = prior_return_bps(marks, int(event.anchor.anchor_ts_ms), 4 * 3600)
                            if not side_prior_ok(side, prior4h, prior_threshold_bps):
                                continue
                            sim = simulate_event(
                                conn,
                                event,
                                offset_bps=OFFSET_BPS,
                                cross_margin_bps=CROSS_MARGIN_BPS,
                                horizon_sec=int(float(horizon_hr) * 3600),
                                maker_fee_bps=float(maker_fee_bps),
                                taker_fee_bps=float(taker_fee_bps),
                                max_book_staleness_sec=int(max_book_staleness_sec),
                                horizon_from="fill",
                                tp_bps=NO_TP_OR_SL,
                                sl_bps=NO_TP_OR_SL,
                            )
                            if sim.get("status") == "FILLED" and sim.get("net_bps") is not None:
                                sim["prior_4h_bps"] = r1(prior4h)
                                rows.append(sim)
                        scored = score_cell(rows, holdout_frac, min_cal_n, min_hold_n)
                        route = {
                            "symbol": symbol,
                            "liq_side": side,
                            "fade_direction": "LONG" if side == "SELL" else "SHORT",
                            "threshold_usd": float(threshold),
                            "vdepth_min_bps": float(v_lo),
                            "vdepth_max_bps": None if v_hi >= 100_000.0 else float(v_hi),
                            "horizon_hr": float(horizon_hr),
                            "offset_bps": OFFSET_BPS,
                            "cross_margin_bps": CROSS_MARGIN_BPS,
                            "filled_rows": rows,
                            **scored,
                        }
                        route["verdict"] = verdict(route)
                        route["route_id"] = (
                            f"{symbol}_{side}_FADE_{route['fade_direction']}_T{int(threshold/1000)}K_"
                            f"V{int(v_lo)}_{'INF' if v_hi >= 100_000.0 else int(v_hi)}_H{float(horizon_hr):g}"
                        )
                        cells.append(route)
    cells.sort(
        key=lambda r: (
            {"HARD_PASS": 3, "WEAK_LEAD": 2, "DATA_BUILD": 1, "REJECT": 0}.get(r["verdict"], 0),
            float(r["holdout"]["top3_winner_removed_sum_bps"] or -1e18),
            float(r["overall"]["top3_winner_removed_sum_bps"] or -1e18),
            float(r["overall"]["sum_bps"] or -1e18),
        ),
        reverse=True,
    )
    return {"coverage": coverage, "cells": cells}


def cell_str(s: dict[str, Any]) -> str:
    return f"N={s['n']} sum={s['sum_bps']} med={s['median_bps']} T3R={s['top3_winner_removed_sum_bps']}"


def compact_cell(row: dict[str, Any]) -> dict[str, Any]:
    return {
        k: v
        for k, v in row.items()
        if k not in {"filled_rows"}
    }


def render_md(payload: dict[str, Any]) -> str:
    cells = payload["cells"]
    counts: dict[str, int] = {}
    for row in cells:
        counts[row["verdict"]] = counts.get(row["verdict"], 0) + 1
    lines = [
        "# S34 V Engine Portfolio Map",
        "",
        f"Generated: `{payload['generated_at_utc']}`",
        "",
        "Research-only screen for mirror, multi-asset, and portfolio-lane expansion. No live/paper state changed.",
        "",
        "## Verdict Counts",
        "",
        f"- HARD_PASS: `{counts.get('HARD_PASS', 0)}`",
        f"- WEAK_LEAD: `{counts.get('WEAK_LEAD', 0)}`",
        f"- DATA_BUILD: `{counts.get('DATA_BUILD', 0)}`",
        f"- REJECT: `{counts.get('REJECT', 0)}`",
        "",
        "## Top Routes",
        "",
        "| Rank | Verdict | Route | Cal | Hold | Overall |",
        "| ---: | --- | --- | --- | --- | --- |",
    ]
    for idx, row in enumerate(cells[:30], start=1):
        lines.append(
            f"| {idx} | `{row['verdict']}` | `{row['route_id']}` | {cell_str(row['calibration'])} | "
            f"{cell_str(row['holdout'])} | {cell_str(row['overall'])} |"
        )
    lines.extend(["", "## Mirror Read", ""])
    eth_sell = [r for r in cells if r["symbol"] == "ETHUSDT" and r["liq_side"] == "SELL" and r["threshold_usd"] == 200_000.0]
    eth_buy = [r for r in cells if r["symbol"] == "ETHUSDT" and r["liq_side"] == "BUY" and r["threshold_usd"] == 200_000.0]
    best_sell = eth_sell[0] if eth_sell else None
    best_buy = eth_buy[0] if eth_buy else None
    if best_sell:
        lines.append(f"- ETH SELL->LONG best: `{best_sell['route_id']}` {best_sell['verdict']} overall {cell_str(best_sell['overall'])}.")
    if best_buy:
        lines.append(f"- ETH BUY->SHORT best: `{best_buy['route_id']}` {best_buy['verdict']} overall {cell_str(best_buy['overall'])}.")
    lines.extend(["", "## Asset Read", ""])
    for symbol in payload["config"]["symbols"]:
        symbol_rows = [r for r in cells if r["symbol"] == symbol]
        best = symbol_rows[0] if symbol_rows else None
        if best:
            lines.append(f"- {symbol}: best `{best['route_id']}` {best['verdict']} overall {cell_str(best['overall'])}.")
        else:
            lines.append(f"- {symbol}: no rows.")
    lines.extend(["", "## Next Link", ""])
    lines.append("- Treat HARD_PASS as freeze-candidate only after manual anatomy; this is still exploratory batch screening.")
    lines.append("- Treat WEAK_LEAD as a new observation lane, not live/paper.")
    lines.append("- DATA_BUILD means the pattern may be structurally plausible but current N is too thin.")
    lines.append("")
    return "\n".join(lines)


def parse_symbols(text: str) -> tuple[str, ...]:
    out = tuple(part.strip().upper() for part in str(text).split(",") if part.strip())
    if not out:
        raise ValueError("empty symbols")
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch screen S34 V Engine portfolio expansion lanes.")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--symbols", default="ETHUSDT,BTCUSDT,SOLUSDT")
    parser.add_argument("--holdout-frac", type=float, default=0.30)
    parser.add_argument("--prior-threshold-bps", type=float, default=50.0)
    parser.add_argument("--maker-fee-bps", type=float, default=2.0)
    parser.add_argument("--taker-fee-bps", type=float, default=3.05)
    parser.add_argument("--max-book-staleness-sec", type=int, default=10)
    parser.add_argument("--min-cal-n", type=int, default=5)
    parser.add_argument("--min-hold-n", type=int, default=5)
    parser.add_argument("--json-out", type=Path, default=OUT_JSON)
    parser.add_argument("--md-out", type=Path, default=OUT_MD)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    symbols = parse_symbols(args.symbols)
    with sqlite3.connect(f"file:{args.db}?mode=ro", uri=True) as conn:
        result = run_screen(
            conn,
            symbols=symbols,
            holdout_frac=float(args.holdout_frac),
            prior_threshold_bps=float(args.prior_threshold_bps),
            maker_fee_bps=float(args.maker_fee_bps),
            taker_fee_bps=float(args.taker_fee_bps),
            max_book_staleness_sec=int(args.max_book_staleness_sec),
            min_cal_n=int(args.min_cal_n),
            min_hold_n=int(args.min_hold_n),
        )
    payload = {
        "generated_at_utc": utc_now(),
        "source_db": file_fingerprint(args.db),
        "config": {
            "symbols": list(symbols),
            "symbol_thresholds": {k: list(v) for k, v in SYMBOL_THRESHOLDS.items() if k in symbols},
            "vdepth_bands": [[lo, None if hi >= 100_000.0 else hi] for lo, hi in VDEPTH_BANDS],
            "horizon_hr": list(HORIZON_HR),
            "offset_bps": OFFSET_BPS,
            "cross_margin_bps": CROSS_MARGIN_BPS,
            "holdout_frac": float(args.holdout_frac),
            "prior_threshold_bps": float(args.prior_threshold_bps),
            "min_cal_n": int(args.min_cal_n),
            "min_hold_n": int(args.min_hold_n),
        },
        "coverage": result["coverage"],
        "cells": [compact_cell(row) for row in result["cells"]],
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    args.md_out.write_text(render_md(payload), encoding="utf-8")
    print(render_md(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
