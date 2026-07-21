"""S34 V Engine v0.2 Sensitivity.

Exploratory-only sensitivity around the frozen v0.1 candidate:
ETH SELL-liq -> maker LONG, 2h horizon, prior 4h down, moderate V-depth.

This script does not change the frozen v0.1 protocol. It searches nearby
thresholds for possible v0.2 candidates and reports cal/hold/T3R discipline.
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

from tools.research_s34_knowable_anchor_continuation import (
    file_fingerprint,
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
OUT_JSON = OUT_DIR / "S34_V_ENGINE_V0_2_SENSITIVITY.json"
OUT_MD = OUT_DIR / "S34_V_ENGINE_V0_2_SENSITIVITY.md"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_range_tuple(text: str) -> tuple[tuple[float, float], ...]:
    ranges = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        lo, hi = part.split("-", 1)
        ranges.append((float(lo), float(hi)))
    if not ranges:
        raise ValueError("empty range list")
    return tuple(ranges)


def ret_bps(marks, start_ms: int, end_ms: int) -> float | None:
    return marks.ret_bps(int(start_ms), int(end_ms))


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


def passes(summary: dict[str, Any], min_cal_n: int, min_hold_n: int) -> bool:
    cal = summary["calibration"]
    hold = summary["holdout"]
    return (
        int(cal["n"] or 0) >= int(min_cal_n)
        and int(hold["n"] or 0) >= int(min_hold_n)
        and float(cal["sum_bps"] or 0.0) > 0.0
        and float(hold["sum_bps"] or 0.0) > 0.0
        and float(cal["top3_winner_removed_sum_bps"] or 0.0) > 0.0
        and float(hold["top3_winner_removed_sum_bps"] or 0.0) > 0.0
    )


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    cal = [float(r["net_bps"]) for r in rows if r.get("split") == "calibration"]
    hold = [float(r["net_bps"]) for r in rows if r.get("split") == "holdout"]
    return {
        "calibration": summarize(cal),
        "holdout": summarize(hold),
        "overall": summarize(cal + hold),
    }


def run_sensitivity(
    conn: sqlite3.Connection,
    *,
    symbol: str,
    threshold: float,
    vdepth_ranges: tuple[tuple[float, float], ...],
    prior4h_thresholds: tuple[float, ...],
    offsets: tuple[float, ...],
    cross_margin_bps: float,
    horizon_hr: float,
    holdout_frac: float,
    maker_fee_bps: float,
    taker_fee_bps: float,
    max_book_staleness_sec: int,
    min_cal_n: int,
    min_hold_n: int,
) -> dict[str, Any]:
    horizon_sec = int(float(horizon_hr) * 3600)
    marks = load_mark_index(conn, symbol)
    max_hi = max(hi for _, hi in vdepth_ranges)
    # Collect broadly, then filter exact v-depth ranges below.
    events = collect_events(
        conn,
        symbol=symbol,
        threshold=float(threshold),
        sides=("SELL",),
        min_vdepth_bps=min(lo for lo, _ in vdepth_ranges),
        bucket_sec=300,
        min_gap_sec=900,
        accel_window_sec=30,
        max_horizon_sec=horizon_sec,
    )
    holdout_ids, split = split_bucket_ids([int(ev.anchor.bucket) for ev in events], holdout_frac)
    event_meta: dict[int, dict[str, Any]] = {}
    for idx, event in enumerate(events):
        ts = int(event.anchor.anchor_ts_ms)
        event_meta[idx] = {
            "event": event,
            "vdepth_bps": float(event.vdepth_bps),
            "prior_4h_bps": ret_bps(marks, ts - 4 * 3600 * 1000, ts),
            "split": "holdout" if int(event.anchor.bucket) in holdout_ids else "calibration",
        }
    results = []
    for v_lo, v_hi in vdepth_ranges:
        for prior_th in prior4h_thresholds:
            for offset in offsets:
                rows = []
                total_events = 0
                for meta in event_meta.values():
                    prior = meta["prior_4h_bps"]
                    if prior is None or not math.isfinite(float(prior)):
                        continue
                    if not (float(v_lo) <= float(meta["vdepth_bps"]) < float(v_hi)):
                        continue
                    if not (float(prior) < float(prior_th)):
                        continue
                    total_events += 1
                    sim = simulate_event(
                        conn,
                        meta["event"],
                        offset_bps=float(offset),
                        cross_margin_bps=float(cross_margin_bps),
                        horizon_sec=horizon_sec,
                        maker_fee_bps=float(maker_fee_bps),
                        taker_fee_bps=float(taker_fee_bps),
                        max_book_staleness_sec=int(max_book_staleness_sec),
                        horizon_from="fill",
                        tp_bps=NO_TP_OR_SL,
                        sl_bps=NO_TP_OR_SL,
                    )
                    sim["split"] = meta["split"]
                    sim["vdepth_bps"] = r1(meta["vdepth_bps"])
                    sim["prior_4h_bps"] = r1(prior)
                    if sim.get("status") == "FILLED" and sim.get("net_bps") is not None:
                        rows.append(sim)
                summary = summarize_rows(rows)
                results.append(
                    {
                        "rule_label": f"V{int(v_lo)}_{int(v_hi)}_P4LT{int(abs(prior_th))}_O{int(offset)}",
                        "vdepth_min_bps": float(v_lo),
                        "vdepth_max_bps": float(v_hi),
                        "prior4h_lt_bps": float(prior_th),
                        "offset_bps": float(offset),
                        "cross_margin_bps": float(cross_margin_bps),
                        "horizon_hr": float(horizon_hr),
                        "eligible_event_n": total_events,
                        "filled_n": len(rows),
                        "fill_rate": r3(len(rows) / total_events) if total_events else None,
                        "summary": summary,
                        "v02_candidate": passes(summary, min_cal_n, min_hold_n),
                    }
                )
    ranked = sorted(
        results,
        key=lambda r: (
            bool(r["v02_candidate"]),
            float(r["summary"]["holdout"]["top3_winner_removed_sum_bps"] or -1e18),
            float(r["summary"]["calibration"]["top3_winner_removed_sum_bps"] or -1e18),
            float(r["summary"]["holdout"]["sum_bps"] or -1e18),
        ),
        reverse=True,
    )
    return {"split": split, "event_n": len(events), "events_sha256": sha256_text("\n".join(f"{ev.anchor.anchor_ts_ms}:{ev.vdepth_bps:.6f}" for ev in events)), "results": ranked}


def cell(summary: dict[str, Any], split: str) -> str:
    s = summary[split]
    return f"N={s['n']} sum={s['sum_bps']} med={s['median_bps']} T3R={s['top3_winner_removed_sum_bps']}"


def render_md(payload: dict[str, Any]) -> str:
    cfg = payload["config"]
    lines = [
        "# S34 V Engine v0.2 Sensitivity",
        "",
        f"Generated: `{payload['generated_at_utc']}`",
        "",
        "EXPLORATORY_ONLY. This does not modify the frozen v0.1 mini-protocol.",
        "",
        f"Base: `{cfg['symbol']}` SELL-liq maker LONG, threshold `{int(cfg['threshold']/1000)}K`, horizon `{cfg['horizon_hr']}h`, cross margin `{cfg['cross_margin_bps']}bps`.",
        "",
        f"Events considered: `{payload['event_n']}`",
        "",
        "## Ranked Sensitivity",
        "",
        "| Rank | Label | V-depth | Prior4h | Offset | Fill% | Cal | Hold | v0.2? |",
        "| ---: | --- | --- | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for idx, row in enumerate(payload["results"], start=1):
        lines.append(
            f"| {idx} | `{row['rule_label']}` | {row['vdepth_min_bps']}..{row['vdepth_max_bps']} | "
            f"< {row['prior4h_lt_bps']} | {row['offset_bps']} | {None if row['fill_rate'] is None else r1(row['fill_rate'] * 100.0)} "
            f"| {cell(row['summary'], 'calibration')} | {cell(row['summary'], 'holdout')} | {'YES' if row['v02_candidate'] else ''} |"
        )
    candidates = [r for r in payload["results"] if r["v02_candidate"]]
    lines.extend(["", f"## v0.2 Candidate Count: {len(candidates)}", ""])
    if candidates:
        for row in candidates[:5]:
            lines.append(
                f"- `{row['rule_label']}`: cal {cell(row['summary'], 'calibration')}; hold {cell(row['summary'], 'holdout')}"
            )
    else:
        lines.append("- none")
    lines.append("")
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run v0.2 sensitivity around the S34 V Engine v0.1 frozen candidate.")
    p.add_argument("--db", type=Path, default=DEFAULT_DB)
    p.add_argument("--symbol", default="ETHUSDT")
    p.add_argument("--threshold", type=float, default=200_000.0)
    p.add_argument("--vdepth-ranges", default="25-45,28-40,30-45,25-50")
    p.add_argument("--prior4h-thresholds", default="-50,-80,-120,-150")
    p.add_argument("--offset-bps", default="10,20,30")
    p.add_argument("--cross-margin-bps", type=float, default=2.0)
    p.add_argument("--horizon-hr", type=float, default=2.0)
    p.add_argument("--maker-fee-bps", type=float, default=2.0)
    p.add_argument("--taker-fee-bps", type=float, default=3.05)
    p.add_argument("--max-book-staleness-sec", type=int, default=10)
    p.add_argument("--holdout-frac", type=float, default=0.30)
    p.add_argument("--min-cal-n", type=int, default=8)
    p.add_argument("--min-hold-n", type=int, default=12)
    p.add_argument("--json-out", type=Path, default=OUT_JSON)
    p.add_argument("--md-out", type=Path, default=OUT_MD)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    vdepth_ranges = parse_range_tuple(args.vdepth_ranges)
    prior_thresholds = parse_float_tuple(args.prior4h_thresholds)
    offsets = parse_float_tuple(args.offset_bps)
    with sqlite3.connect(f"file:{args.db}?mode=ro", uri=True) as conn:
        report = run_sensitivity(
            conn,
            symbol=args.symbol,
            threshold=float(args.threshold),
            vdepth_ranges=vdepth_ranges,
            prior4h_thresholds=prior_thresholds,
            offsets=offsets,
            cross_margin_bps=float(args.cross_margin_bps),
            horizon_hr=float(args.horizon_hr),
            holdout_frac=float(args.holdout_frac),
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
            "symbol": args.symbol,
            "threshold": float(args.threshold),
            "vdepth_ranges": [[lo, hi] for lo, hi in vdepth_ranges],
            "prior4h_thresholds": list(prior_thresholds),
            "offset_bps": list(offsets),
            "cross_margin_bps": float(args.cross_margin_bps),
            "horizon_hr": float(args.horizon_hr),
            "maker_fee_bps": float(args.maker_fee_bps),
            "taker_fee_bps": float(args.taker_fee_bps),
            "max_book_staleness_sec": int(args.max_book_staleness_sec),
            "holdout_frac": float(args.holdout_frac),
            "min_cal_n": int(args.min_cal_n),
            "min_hold_n": int(args.min_hold_n),
        },
        **report,
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    args.md_out.write_text(render_md(payload), encoding="utf-8")
    print(render_md(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
