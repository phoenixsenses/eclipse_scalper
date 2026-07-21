"""S34 V02 propagation candidate gauntlet and tag export.

Research-only. Converts the propagation puzzle into testable candidate states
and exports navigation tags. It does not touch live/paper/executor state.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sqlite3
import sys
from bisect import bisect_left
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.research_s34_knowable_anchor_continuation import iso_ms, pctile, r1, r3, signed_return_bps  # noqa: E402
from tools.s34_v02_event_chain_puzzle_tests import (  # noqa: E402
    ASSET_THRESHOLDS,
    FEE_BPS,
    H4_LEDGER_JSONL,
    build_events,
    load_jsonl,
    mark_price_at,
    metrics,
    neighbor_events,
)
from tools.s34_v02_propagation_puzzle_suite import enrich_events  # noqa: E402


DEFAULT_DB = ROOT / "data" / "microstructure.db"
OUT_DIR = ROOT / "reports" / "research" / "s34"
OUT_JSON = OUT_DIR / "S34_V02_PROPAGATION_CANDIDATE_GAUNTLET.json"
OUT_MD = OUT_DIR / "S34_V02_PROPAGATION_CANDIDATE_GAUNTLET.md"
OUT_TAGS_JSONL = OUT_DIR / "S34_V02_PROPAGATION_NAV_TAGS.jsonl"
OUT_TAGS_FRAGMENT = OUT_DIR / "S34_V02_PROPAGATION_NAV_TAGS_FRAGMENT.json"

SYMBOL = "ETHUSDT"
TAUS_SEC = (300, 900, 1800, 3600)
PERMUTATIONS = 500


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def side_fade_direction(side: str) -> str:
    return "LONG" if side == "SELL" else "SHORT"


def side_momentum_direction(side: str) -> str:
    return "SHORT" if side == "SELL" else "LONG"


def direction_return(marks: Any, direction: str, start_ms: int, horizon_sec: int, fee_bps: float = FEE_BPS) -> float | None:
    a = mark_price_at(marks, int(start_ms), before=False)
    b = mark_price_at(marks, int(start_ms) + int(horizon_sec) * 1000, before=False)
    if not a or not b:
        return None
    return r1(signed_return_bps(direction, float(a[1]), float(b[1])) - fee_bps)


def month_of(ts_ms: int) -> str:
    return datetime.fromtimestamp(int(ts_ms) / 1000.0, tz=timezone.utc).strftime("%Y-%m")


def split_months(rows: list[dict[str, Any]]) -> tuple[set[str], dict[str, Any]]:
    months = sorted({str(r.get("month")) for r in rows if r.get("month")})
    hold_n = max(1, round(len(months) * 0.35)) if months else 0
    hold = set(months[-hold_n:]) if hold_n else set()
    return hold, {"method": "chronological_month_tail_35pct", "months": months, "holdout_months": sorted(hold)}


def finite(v: Any) -> float | None:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def top3_removed_dependency(summary: dict[str, Any]) -> str:
    if summary.get("n", 0) < 4:
        return "N_TOO_SMALL"
    if float(summary.get("sum_bps") or 0.0) > 0 and float(summary.get("t3r_bps") or 0.0) <= 0:
        return "TOP3_DEPENDENT"
    return "OK"


def event_index(events: list[dict[str, Any]]) -> list[int]:
    return [int(e["anchor_ts_ms"]) for e in events]


def count_same_between(events: list[dict[str, Any]], idx: list[int], *, side: str, ts_ms: int, tau_sec: int) -> int:
    rows = neighbor_events(idx, events, ts_ms=int(ts_ms), before_sec=0, after_sec=int(tau_sec), side=side)
    return sum(1 for r in rows if int(r["anchor_ts_ms"]) > int(ts_ms))


def causal_counts(
    row: dict[str, Any],
    *,
    eth_events: list[dict[str, Any]],
    eth_idx: list[int],
    asset_events: dict[str, list[dict[str, Any]]],
    asset_idx: dict[str, list[int]],
    tau_sec: int,
) -> dict[str, Any]:
    ts = int(row["anchor_ts_ms"])
    side = str(row["side"])
    same = count_same_between(eth_events, eth_idx, side=side, ts_ms=ts, tau_sec=tau_sec)
    cross = 0
    for sym in ("BTCUSDT", "SOLUSDT"):
        cross += count_same_between(asset_events[sym], asset_idx[sym], side=side, ts_ms=ts, tau_sec=tau_sec)
    post_notional = float(row.get("post_anchor_liq_notional") or 0.0)
    # Event-end is knowable only after the current bucket has finished. For short
    # taus we do not use post-anchor notional unless tau has reached event end.
    event_end_offset = max(0.0, (int(row.get("event_end_ts_ms") or ts) - ts) / 1000.0)
    post_liq_known = tau_sec >= event_end_offset
    score = int(same > 0) * 3 + int(cross > 0) * 2 + int(post_liq_known and post_notional > 0.0)
    return {
        "tau_sec": int(tau_sec),
        "same_restart_n": int(same),
        "cross_same_n": int(cross),
        "post_liq_known": bool(post_liq_known),
        "pressure_score": int(score),
        "pressure_high": bool(score >= 4),
        "pressure_mid": bool(2 <= score <= 3),
        "pressure_low": bool(score <= 1),
        "silence_after_shock": bool(score <= 1 and row.get("reclaim_ts_ms") is not None),
    }


def add_candidate_outcomes(conn: sqlite3.Connection, rows: list[dict[str, Any]]) -> None:
    from tools.research_s34_knowable_anchor_continuation import load_mark_index

    marks = load_mark_index(conn, SYMBOL)
    for row in rows:
        side = str(row["side"])
        ts = int(row["anchor_ts_ms"])
        fade = side_fade_direction(side)
        mom = side_momentum_direction(side)
        row["fade_h1_bps"] = direction_return(marks, fade, ts, 3600)
        row["fade_h4_bps"] = direction_return(marks, fade, ts, 14400)
        row["momentum_h1_bps"] = direction_return(marks, mom, ts, 3600)
        row["momentum_h4_bps"] = direction_return(marks, mom, ts, 14400)
        row["opposite_fade_h4_bps"] = row["momentum_h4_bps"]
        row["opposite_momentum_h1_bps"] = row["fade_h1_bps"]


CANDIDATES = {
    "SELL_SILENCE_FADE_LONG_H4": {
        "side": "SELL",
        "mode": "silence",
        "outcome": "fade_h4_bps",
        "negative": "opposite_fade_h4_bps",
        "description": "SELL shock, no same/cross propagation by tau, reclaim present -> LONG fade H4.",
    },
    "SELL_PROPAGATION_MOMENTUM_SHORT_H1": {
        "side": "SELL",
        "mode": "pressure",
        "outcome": "momentum_h1_bps",
        "negative": "opposite_momentum_h1_bps",
        "description": "SELL shock with propagation pressure by tau -> SHORT momentum H1.",
    },
    "BUY_PROPAGATION_MOMENTUM_LONG_H1": {
        "side": "BUY",
        "mode": "pressure",
        "outcome": "momentum_h1_bps",
        "negative": "opposite_momentum_h1_bps",
        "description": "BUY shock with propagation pressure by tau -> LONG momentum H1.",
    },
    "BUY_SILENCE_FADE_SHORT_H4": {
        "side": "BUY",
        "mode": "silence",
        "outcome": "fade_h4_bps",
        "negative": "opposite_fade_h4_bps",
        "description": "BUY shock, no same/cross propagation by tau, reclaim present -> SHORT fade H4.",
    },
}


def select_candidate_rows(rows: list[dict[str, Any]], candidate: dict[str, Any], tau: int) -> list[dict[str, Any]]:
    key = f"tau_{tau}"
    out = []
    for row in rows:
        if row.get("side") != candidate["side"]:
            continue
        c = row.get(key, {})
        if candidate["mode"] == "pressure" and c.get("pressure_high"):
            out.append(row)
        if candidate["mode"] == "silence" and c.get("silence_after_shock"):
            out.append(row)
    return out


def eval_candidate(rows: list[dict[str, Any]], candidate_id: str, candidate: dict[str, Any], tau: int, hold_months: set[str]) -> dict[str, Any]:
    selected = select_candidate_rows(rows, candidate, tau)
    outcome = candidate["outcome"]
    negative = candidate["negative"]
    cal = [r for r in selected if r.get("month") not in hold_months]
    hold = [r for r in selected if r.get("month") in hold_months]
    main = metrics([r.get(outcome) for r in selected])
    neg = metrics([r.get(negative) for r in selected])
    hold_m = metrics([r.get(outcome) for r in hold])
    pass_flags = {
        "n_ge_40": main["n"] >= 40,
        "all_sum_gt_0": float(main["sum_bps"] or 0.0) > 0,
        "all_t3r_gt_0": float(main["t3r_bps"] or 0.0) > 0,
        "hold_sum_gt_0": float(hold_m["sum_bps"] or 0.0) > 0,
        "hold_t3r_gt_0": float(hold_m["t3r_bps"] or 0.0) > 0,
        "negative_control_worse": float(main["t3r_bps"] or -1e18) > float(neg["t3r_bps"] or -1e18),
        "not_top3_dependent": top3_removed_dependency(main) == "OK",
    }
    return {
        "candidate_id": candidate_id,
        "tau_sec": int(tau),
        "description": candidate["description"],
        "selected_n": len(selected),
        "all": main,
        "cal": metrics([r.get(outcome) for r in cal]),
        "hold": hold_m,
        "negative_control": neg,
        "execution_proxy": {
            "median_spread_bps": r1(pctile([float(r["spread_bps"]) for r in selected if finite(r.get("spread_bps")) is not None], 0.5)) if selected else None,
            "median_bid_depth_usd": r1(pctile([float(r["bid_depth_usd"]) for r in selected if finite(r.get("bid_depth_usd")) is not None], 0.5)) if selected else None,
            "median_ask_depth_usd": r1(pctile([float(r["ask_depth_usd"]) for r in selected if finite(r.get("ask_depth_usd")) is not None], 0.5)) if selected else None,
            "model": "MARK_TAKER_PROXY_FOR_CANDIDATE_GAUNTLET; maker lifecycle not yet proven",
        },
        "pass_flags": pass_flags,
        "gauntlet_pass": bool(all(pass_flags.values())),
        "sample": [
            {
                "anchor_utc": r.get("anchor_utc"),
                "side": r.get("side"),
                "month": r.get("month"),
                "outcome_bps": r.get(outcome),
                "negative_bps": r.get(negative),
                "tags": r.get("tags_1800", []),
            }
            for r in selected[:10]
        ],
    }


def permutation_maxstat(results: list[dict[str, Any]], rows: list[dict[str, Any]], *, iterations: int, seed: int) -> dict[str, Any]:
    observed = [float(r["all"]["t3r_bps"] or 0.0) for r in results]
    observed_max = max(observed) if observed else 0.0
    cells = []
    for res in results:
        cand = CANDIDATES[res["candidate_id"]]
        selected = select_candidate_rows(rows, cand, int(res["tau_sec"]))
        vals = [finite(r.get(cand["outcome"])) for r in selected]
        vals = [float(v) for v in vals if v is not None]
        cells.append(vals)
    all_vals = [v for vals in cells for v in vals]
    if not all_vals or not cells:
        return {"status": "INSUFFICIENT_DATA"}
    rng = random.Random(seed)
    max_stats = []
    sizes = [len(vals) for vals in cells]
    for _ in range(iterations):
        vals = all_vals[:]
        rng.shuffle(vals)
        pos = 0
        t3rs = []
        for n in sizes:
            sample = vals[pos:pos + n]
            pos += n
            t3rs.append(float(metrics(sample)["t3r_bps"] or 0.0))
        max_stats.append(max(t3rs) if t3rs else 0.0)
    p_right = (1 + sum(1 for x in max_stats if x >= observed_max)) / (len(max_stats) + 1)
    return {
        "status": "OK",
        "iterations": int(iterations),
        "seed": int(seed),
        "observed_max_t3r": r1(observed_max),
        "null_p95_max_t3r": r1(pctile(max_stats, 0.95)),
        "mc_corrected_p_right": r3(p_right),
        "read": "Coarse max-stat correction across all candidate/tau cells. It is a guardrail, not final proof.",
    }


def build_tags(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    tagged = []
    for row in rows:
        tag_row = {
            "event_id": row.get("event_id"),
            "symbol": SYMBOL,
            "side": row.get("side"),
            "anchor_ts_ms": int(row["anchor_ts_ms"]),
            "anchor_utc": row.get("anchor_utc"),
            "month": row.get("month"),
            "tags_by_tau": {},
            "primary_tags": [],
            "fade_h4_bps": row.get("fade_h4_bps"),
            "momentum_h1_bps": row.get("momentum_h1_bps"),
        }
        for tau in TAUS_SEC:
            c = row[f"tau_{tau}"]
            tags = []
            if c["pressure_high"]:
                tags.append("PROPAGATION_PRESSURE_HIGH")
                tags.append("MOMENTUM_MODE")
                tags.append(f"{row['side']}_PROPAGATION")
            elif c["silence_after_shock"]:
                tags.append("SILENCE_AFTER_SHOCK")
                tags.append("FADE_MODE")
                tags.append(f"{row['side']}_SILENCE_RECLAIM")
            elif c["pressure_mid"]:
                tags.append("PROPAGATION_PRESSURE_MID")
            else:
                tags.append("PROPAGATION_PRESSURE_LOW")
            if row["side"] == "SELL" and c["pressure_high"]:
                tags.append("SELL_FADE_DANGER")
                tags.append("SELL_SHORT_MOMENTUM_WATCH")
            if row["side"] == "SELL" and c["silence_after_shock"]:
                tags.append("SELL_LONG_FADE_NAV_OK")
            if row["side"] == "BUY" and c["pressure_high"]:
                tags.append("BUY_SHORT_FADE_DANGER")
                tags.append("BUY_LONG_MOMENTUM_WATCH")
            if row["side"] == "BUY" and c["silence_after_shock"]:
                tags.append("BUY_SHORT_FADE_NAV_WATCH")
            tag_row["tags_by_tau"][str(tau)] = tags
            if tau == 1800:
                tag_row["primary_tags"] = tags
                row["tags_1800"] = tags
        tagged.append(tag_row)
    return tagged


def v02_compatibility(rows: list[dict[str, Any]]) -> dict[str, Any]:
    h4_rows = [
        r for r in load_jsonl(H4_LEDGER_JSONL)
        if r.get("bucket") == "H4_SHADOW" and r.get("observation_status") == "CLOSED" and r.get("net_bps") is not None
    ]
    sell_rows = [r for r in rows if r.get("side") == "SELL"]
    sell_ts = sorted(int(r["anchor_ts_ms"]) for r in sell_rows)
    by_ts = {int(r["anchor_ts_ms"]): r for r in sell_rows}
    matched = []
    for trade in h4_rows:
        ts = int(trade["signal_ts_ms"])
        pos = bisect_left(sell_ts, ts)
        ev = None
        for j in (pos - 1, pos, pos + 1):
            if 0 <= j < len(sell_ts) and abs(sell_ts[j] - ts) <= 2_000:
                ev = by_ts[sell_ts[j]]
                break
        h2 = finite(trade.get("h2_net_bps"))
        h4 = finite(trade.get("net_bps"))
        row = {
            "signal_utc": trade.get("signal_utc"),
            "matched": ev is not None,
            "h2_bps": h2,
            "h4_bps": h4,
            "h4_minus_h2_bps": r1(h4 - h2) if h2 is not None and h4 is not None else None,
            "tags_1800": ev.get("tags_1800", []) if ev else [],
            "pressure_high_1800": bool(ev and ev["tau_1800"]["pressure_high"]),
            "silence_1800": bool(ev and ev["tau_1800"]["silence_after_shock"]),
        }
        matched.append(row)
    return {
        "n": len(matched),
        "matched_n": sum(1 for r in matched if r["matched"]),
        "all_h4_minus_h2": metrics([r["h4_minus_h2_bps"] for r in matched]),
        "pressure_high_1800": metrics([r["h4_minus_h2_bps"] for r in matched if r["pressure_high_1800"]]),
        "not_pressure_high_1800": metrics([r["h4_minus_h2_bps"] for r in matched if r["matched"] and not r["pressure_high_1800"]]),
        "silence_1800": metrics([r["h4_minus_h2_bps"] for r in matched if r["silence_1800"]]),
        "not_silence_1800": metrics([r["h4_minus_h2_bps"] for r in matched if r["matched"] and not r["silence_1800"]]),
        "rows": matched,
        "read": "V02 compatibility remains small-N. Tags are management/navigation context, not order logic.",
    }


def render_md(report: dict[str, Any]) -> str:
    lines = [
        "# S34 V02 Propagation Candidate Gauntlet",
        "",
        f"Generated: `{report['generated_at_utc']}`",
        "",
        "Research-only. No live executor/config/order logic is touched.",
        "",
        "## Verdict",
        "",
        f"- Overall: `{report['verdict']}`",
        f"- Tags exported: `{report['tags']['jsonl']}`",
        "",
        "## Candidate Leaderboard",
        "",
        "| Rank | Candidate | Tau | N | All | Hold | Neg Ctrl | Pass |",
        "| ---: | --- | ---: | ---: | --- | --- | --- | --- |",
    ]
    ranked = sorted(
        report["candidate_results"],
        key=lambda r: (float(r["all"].get("t3r_bps") or -1e18), float(r["all"].get("sum_bps") or -1e18)),
        reverse=True,
    )
    for i, row in enumerate(ranked, start=1):
        lines.append(
            f"| {i} | `{row['candidate_id']}` | {row['tau_sec']} | {row['selected_n']} | "
            f"sum={row['all']['sum_bps']} med={row['all']['median_bps']} T3R={row['all']['t3r_bps']} | "
            f"sum={row['hold']['sum_bps']} T3R={row['hold']['t3r_bps']} | "
            f"sum={row['negative_control']['sum_bps']} T3R={row['negative_control']['t3r_bps']} | "
            f"{row['gauntlet_pass']} |"
        )
    lines += [
        "",
        "## Permutation Max-Stat",
        "",
        "```json",
        json.dumps(report["permutation"], indent=2, sort_keys=True),
        "```",
        "",
        "## V02 Compatibility",
        "",
        "```json",
        json.dumps({k: v for k, v in report["v02_compatibility"].items() if k != "rows"}, indent=2, sort_keys=True),
        "```",
        "",
        "## Tag Counts @ 1800s",
        "",
        "```json",
        json.dumps(report["tag_counts_1800"], indent=2, sort_keys=True),
        "```",
        "",
        "## Read",
        "",
    ]
    lines.extend([f"- {x}" for x in report["read"]])
    lines.append("")
    return "\n".join(lines)


def run(db: Path, *, permutations: int) -> dict[str, Any]:
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=30)
    try:
        asset_events = {sym: build_events(conn, symbol=sym, threshold=thr) for sym, thr in ASSET_THRESHOLDS.items()}
        rows = enrich_events(conn, asset_events[SYMBOL], asset_events)
        add_candidate_outcomes(conn, rows)
    finally:
        conn.close()
    eth_idx = event_index(asset_events[SYMBOL])
    asset_idx = {sym: event_index(evts) for sym, evts in asset_events.items()}
    for row in rows:
        for tau in TAUS_SEC:
            row[f"tau_{tau}"] = causal_counts(
                row,
                eth_events=asset_events[SYMBOL],
                eth_idx=eth_idx,
                asset_events=asset_events,
                asset_idx=asset_idx,
                tau_sec=tau,
            )
    tags = build_tags(rows)
    hold_months, split_meta = split_months(rows)
    results = []
    for cand_id, cand in CANDIDATES.items():
        for tau in TAUS_SEC:
            results.append(eval_candidate(rows, cand_id, cand, tau, hold_months))
    perm = permutation_maxstat(results, rows, iterations=permutations, seed=3403)
    tag_counts: dict[str, int] = {}
    for row in tags:
        for tag in row.get("primary_tags", []):
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
    passers = [r for r in results if r["gauntlet_pass"]]
    report = {
        "generated_at_utc": utc_now(),
        "research_only": True,
        "live_executor_touched": False,
        "event_counts": {sym: len(evts) for sym, evts in asset_events.items()},
        "split": split_meta,
        "candidate_results": results,
        "permutation": perm,
        "v02_compatibility": v02_compatibility(rows),
        "tags": {
            "jsonl": str(OUT_TAGS_JSONL),
            "fragment": str(OUT_TAGS_FRAGMENT),
            "n": len(tags),
        },
        "tag_counts_1800": dict(sorted(tag_counts.items())),
        "verdict": "SHADOW_CANDIDATES_FOUND_BUT_NO_LIVE_PROMOTION" if passers else "NO_FULL_GAUNTLET_PASS_NAVIGATION_ONLY",
        "read": [
            "The best broad candidates are state labels, not live-ready strategies yet.",
            "A full pass requires N>=40, positive all/hold sum, positive all/hold T3R, worse negative control, and no top-3 dependency.",
            "The tag export is meant for chart/navigation: PROPAGATION_PRESSURE_HIGH, SILENCE_AFTER_SHOCK, MOMENTUM_MODE, FADE_MODE.",
            "Execution is still mark/taker proxy here. Any candidate that looks good must next pass maker/taker live-like fill and forward shadow.",
        ],
    }
    if passers:
        report["read"].append(f"{len(passers)} candidate/tau cells passed the mechanical gauntlet, but they still need execution realism and forward shadow before paper/live.")
    else:
        report["read"].append("No candidate/tau cell passed all gates. Tags should be used as navigation/danger context only.")
    return report, tags


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run S34 V02 propagation candidate gauntlet and export nav tags.")
    p.add_argument("--db", type=Path, default=DEFAULT_DB)
    p.add_argument("--permutations", type=int, default=PERMUTATIONS)
    p.add_argument("--out-json", type=Path, default=OUT_JSON)
    p.add_argument("--out-md", type=Path, default=OUT_MD)
    p.add_argument("--out-tags-jsonl", type=Path, default=OUT_TAGS_JSONL)
    p.add_argument("--out-tags-fragment", type=Path, default=OUT_TAGS_FRAGMENT)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report, tags = run(args.db, permutations=int(args.permutations))
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    args.out_md.write_text(render_md(report), encoding="utf-8")
    with args.out_tags_jsonl.open("w", encoding="utf-8", newline="\n") as fh:
        for row in tags:
            fh.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
    latest = sorted(tags, key=lambda r: int(r["anchor_ts_ms"]))[-100:]
    args.out_tags_fragment.write_text(
        json.dumps({"generated_at_utc": utc_now(), "latest": latest, "count": len(tags)}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(render_md(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
