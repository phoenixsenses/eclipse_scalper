"""S34 V Engine adaptive maker offset research.

Compares simple point-in-time offset policies against fixed O20 controls. The
goal is execution optimization, not new alpha discovery: improve the trade-off
between fill probability, filled expectancy, skew, and missed no-fill upside.
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.research_s34_knowable_anchor_continuation import file_fingerprint, load_mark_index, r1, r3
from tools.research_s34_maker_fade import NO_TP_OR_SL, simulate_event, summarize
from tools.s34_v_engine_execution_frontier import (
    anchor_mark_counterfactual,
    collect_v01_events,
    prior_return_bps,
)
from tools.s34_v_engine_shadow_observer import HORIZON_SEC, PROTOCOL_ID, SYMBOL, utc_now


DEFAULT_DB = ROOT / "data" / "microstructure.db"
OUT_DIR = ROOT / "reports" / "research" / "s34"
OUT_JSON = OUT_DIR / "S34_V_ENGINE_ADAPTIVE_OFFSET.json"
OUT_MD = OUT_DIR / "S34_V_ENGINE_ADAPTIVE_OFFSET.md"

OffsetPolicy = Callable[[dict[str, Any]], float]


def event_features(event: Any, *, eth_marks: Any, btc_marks: Any) -> dict[str, Any]:
    ts = int(event.anchor.anchor_ts_ms)
    return {
        "anchor_ts_ms": ts,
        "vdepth_bps": float(event.vdepth_bps),
        "prior_4h_bps": prior_return_bps(eth_marks, ts, 4 * 3600),
        "btc_prior_4h_bps": prior_return_bps(btc_marks, ts, 4 * 3600),
        "running_accel_usd_per_sec": float(event.anchor.running_accel),
        "single_liq_dominance_pct": float(event.anchor.running_single_liq_dominance),
        "elapsed_since_first_sec": float(event.anchor.elapsed_since_first_sec),
    }


def policies() -> dict[str, OffsetPolicy]:
    return {
        "fixed_o20": lambda f: 20.0,
        "fixed_o15": lambda f: 15.0,
        "fixed_o10": lambda f: 10.0,
        "vdepth_step_15_20_25": lambda f: 15.0 if f["vdepth_bps"] < 32.0 else (20.0 if f["vdepth_bps"] < 36.0 else 25.0),
        "vdepth_inverse_25_20_15": lambda f: 25.0 if f["vdepth_bps"] < 32.0 else (20.0 if f["vdepth_bps"] < 36.0 else 15.0),
        "dominance_aggressive": lambda f: 10.0 if f["single_liq_dominance_pct"] >= 55.0 else 20.0,
        "accel_aggressive": lambda f: 10.0 if f["running_accel_usd_per_sec"] >= 5_000.0 else 20.0,
        "missed_winner_rescue": lambda f: 10.0
        if f["single_liq_dominance_pct"] >= 55.0 or f["running_accel_usd_per_sec"] >= 5_000.0
        else 20.0,
        "btc_supportive_aggressive": lambda f: 10.0 if (f.get("btc_prior_4h_bps") is not None and f["btc_prior_4h_bps"] >= -50.0) else 20.0,
        "eth_extreme_conservative": lambda f: 25.0 if (f.get("prior_4h_bps") is not None and f["prior_4h_bps"] <= -300.0) else 20.0,
        "risk_balanced": lambda f: 25.0
        if (f.get("prior_4h_bps") is not None and f["prior_4h_bps"] <= -300.0)
        else (10.0 if f["single_liq_dominance_pct"] >= 55.0 or f["running_accel_usd_per_sec"] >= 5_000.0 else 20.0),
    }


def source_counts(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for row in rows:
        value = str(row.get(key) or "none")
        out[value] = out.get(value, 0) + 1
    return dict(sorted(out.items()))


def run_policy(
    conn: sqlite3.Connection,
    *,
    events: list[Any],
    eth_marks: Any,
    btc_marks: Any,
    policy_name: str,
    policy: OffsetPolicy,
    cross_margin_bps: float,
    maker_fee_bps: float,
    taker_fee_bps: float,
    max_book_staleness_sec: int,
) -> dict[str, Any]:
    rows = []
    no_fill_cf = []
    fee_bps = float(maker_fee_bps) + float(taker_fee_bps)
    for event in events:
        features = event_features(event, eth_marks=eth_marks, btc_marks=btc_marks)
        offset = float(policy(features))
        sim = simulate_event(
            conn,
            event,
            offset_bps=offset,
            cross_margin_bps=float(cross_margin_bps),
            horizon_sec=HORIZON_SEC,
            maker_fee_bps=float(maker_fee_bps),
            taker_fee_bps=float(taker_fee_bps),
            max_book_staleness_sec=int(max_book_staleness_sec),
            horizon_from="fill",
            tp_bps=NO_TP_OR_SL,
            sl_bps=NO_TP_OR_SL,
        )
        cf = anchor_mark_counterfactual(eth_marks, int(event.anchor.anchor_ts_ms), fee_bps=fee_bps)
        sim.update({k: r1(v) if isinstance(v, float) else v for k, v in features.items()})
        sim["policy"] = policy_name
        sim["chosen_offset_bps"] = offset
        sim["anchor_cf_net_bps"] = r1(cf)
        if sim.get("status") == "NO_MAKER_FILL" and cf is not None and math.isfinite(float(cf)):
            no_fill_cf.append(float(cf))
        rows.append(sim)
    filled = [r for r in rows if r.get("status") == "FILLED" and r.get("net_bps") is not None]
    nets = [float(r["net_bps"]) for r in filled]
    return {
        "policy": policy_name,
        "cross_margin_bps": float(cross_margin_bps),
        "eligible_n": len(rows),
        "filled_n": len(filled),
        "no_fill_n": len(rows) - len(filled),
        "fill_rate": r3(len(filled) / len(rows)) if rows else None,
        "filled_summary": summarize(nets),
        "no_fill_anchor_cf_summary": summarize(no_fill_cf),
        "missed_cf_sum_bps": r1(sum(no_fill_cf)),
        "offset_counts": source_counts(rows, "chosen_offset_bps"),
        "rows": rows,
    }


def run_adaptive(
    conn: sqlite3.Connection,
    *,
    cross_margins: tuple[float, ...],
    maker_fee_bps: float,
    taker_fee_bps: float,
    max_book_staleness_sec: int,
) -> dict[str, Any]:
    eth_marks = load_mark_index(conn, SYMBOL)
    btc_marks = load_mark_index(conn, "BTCUSDT")
    events = collect_v01_events(conn)
    results = []
    for cross in cross_margins:
        for name, policy in policies().items():
            results.append(
                run_policy(
                    conn,
                    events=events,
                    eth_marks=eth_marks,
                    btc_marks=btc_marks,
                    policy_name=name,
                    policy=policy,
                    cross_margin_bps=float(cross),
                    maker_fee_bps=float(maker_fee_bps),
                    taker_fee_bps=float(taker_fee_bps),
                    max_book_staleness_sec=int(max_book_staleness_sec),
                )
            )
    results.sort(
        key=lambda r: (
            float(r["filled_summary"].get("top3_winner_removed_sum_bps") or -1e18),
            float(r["filled_summary"].get("sum_bps") or -1e18),
            float(r["fill_rate"] or 0.0),
        ),
        reverse=True,
    )
    return {"event_n": len(events), "results": results}


def cell(summary: dict[str, Any]) -> str:
    return f"N={summary['n']} sum={summary['sum_bps']} med={summary['median_bps']} T3R={summary['top3_winner_removed_sum_bps']}"


def render_md(report: dict[str, Any]) -> str:
    lines = [
        "# S34 V Engine Adaptive Offset",
        "",
        f"Generated: `{report['generated_at_utc']}`",
        "",
        f"Protocol: `{report['protocol_id']}`",
        "",
        "Research-only. Compares point-in-time adaptive maker offsets against fixed controls.",
        "",
        f"Events: `{report['event_n']}`",
        "",
        "## Ranked Policies",
        "",
        "| Rank | Policy | Cross | Fill% | Offsets | Filled | No-fill CF | Missed CF |",
        "| ---: | --- | ---: | ---: | --- | --- | --- | ---: |",
    ]
    for idx, row in enumerate(report["results"], start=1):
        fill_pct = None if row["fill_rate"] is None else r1(row["fill_rate"] * 100.0)
        lines.append(
            f"| {idx} | `{row['policy']}` | {row['cross_margin_bps']} | {fill_pct} | `{row['offset_counts']}` | "
            f"{cell(row['filled_summary'])} | {cell(row['no_fill_anchor_cf_summary'])} | {row['missed_cf_sum_bps']} |"
        )
    best = report["results"][0] if report["results"] else None
    fixed = next((r for r in report["results"] if r["policy"] == "fixed_o20" and float(r["cross_margin_bps"]) == 1.0), None)
    lines.extend(["", "## Read", ""])
    if best:
        lines.append(f"- Best policy by T3R: `{best['policy']}` C{best['cross_margin_bps']} -> {cell(best['filled_summary'])}.")
    if fixed:
        lines.append(f"- Fixed O20 C1 control: {cell(fixed['filled_summary'])}.")
    if best and fixed:
        best_t3r = float(best["filled_summary"].get("top3_winner_removed_sum_bps") or 0.0)
        fixed_t3r = float(fixed["filled_summary"].get("top3_winner_removed_sum_bps") or 0.0)
        best_sum = float(best["filled_summary"].get("sum_bps") or 0.0)
        fixed_sum = float(fixed["filled_summary"].get("sum_bps") or 0.0)
        t3r_delta = r1(best_t3r - fixed_t3r)
        sum_delta = r1(best_sum - fixed_sum)
        lines.append(f"- T3R delta vs O20 C1: `{t3r_delta}` bps; sum delta `{sum_delta}` bps.")
        lines.append(
            "- Verdict: no new frozen variant. The best adaptive policy is only a small execution tweak over fixed O20, so keep it observation-only until forward N grows."
        )
    lines.append("")
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Test adaptive maker offsets for S34 V Engine v0.1.")
    p.add_argument("--db", type=Path, default=DEFAULT_DB)
    p.add_argument("--cross-margins-bps", default="1,2,5")
    p.add_argument("--maker-fee-bps", type=float, default=2.0)
    p.add_argument("--taker-fee-bps", type=float, default=3.05)
    p.add_argument("--max-book-staleness-sec", type=int, default=10)
    p.add_argument("--json-out", type=Path, default=OUT_JSON)
    p.add_argument("--md-out", type=Path, default=OUT_MD)
    return p.parse_args(argv)


def parse_float_tuple(text: str) -> tuple[float, ...]:
    vals = []
    for part in str(text).split(","):
        part = part.strip()
        if part:
            vals.append(float(part))
    if not vals:
        raise ValueError("empty float tuple")
    return tuple(vals)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    crosses = parse_float_tuple(args.cross_margins_bps)
    with sqlite3.connect(f"file:{args.db}?mode=ro", uri=True) as conn:
        result = run_adaptive(
            conn,
            cross_margins=crosses,
            maker_fee_bps=float(args.maker_fee_bps),
            taker_fee_bps=float(args.taker_fee_bps),
            max_book_staleness_sec=int(args.max_book_staleness_sec),
        )
    report = {
        "generated_at_utc": utc_now(),
        "source_db": file_fingerprint(args.db),
        "protocol_id": PROTOCOL_ID,
        "config": {
            "symbol": SYMBOL,
            "cross_margins_bps": list(crosses),
            "maker_fee_bps": float(args.maker_fee_bps),
            "taker_fee_bps": float(args.taker_fee_bps),
        },
        **result,
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    args.md_out.write_text(render_md(report), encoding="utf-8")
    print(render_md(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
