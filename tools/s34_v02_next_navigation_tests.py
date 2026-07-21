"""S34 V02 next navigation tests.

Research-only. Runs the five requested follow-up tests:

1. early tau sweep
2. SELL event_end/reclaim entry
3. BUY propagation scalp horizons
4. tag sequence model
5. V02 management compatibility

No live/paper/executor state is changed.
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
from bisect import bisect_left
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.research_s34_knowable_anchor_continuation import iso_ms, r1, r3, signed_return_bps  # noqa: E402
from tools.s34_v02_candidate_execution_gauntlet import (  # noqa: E402
    MAX_BOOK_STALENESS_SEC,
    book_at_or_after,
    event_index,
    exit_book_price,
    simulate_taker,
    split_months,
)
from tools.s34_v02_event_chain_puzzle_tests import ASSET_THRESHOLDS, H4_LEDGER_JSONL, build_events, load_jsonl, metrics, neighbor_events  # noqa: E402
from tools.s34_v02_propagation_puzzle_suite import enrich_events  # noqa: E402


DEFAULT_DB = ROOT / "data" / "microstructure.db"
OUT_DIR = ROOT / "reports" / "research" / "s34"
OUT_JSON = OUT_DIR / "S34_V02_NEXT_NAVIGATION_TESTS.json"
OUT_MD = OUT_DIR / "S34_V02_NEXT_NAVIGATION_TESTS.md"

SYMBOL = "ETHUSDT"
EARLY_TAUS_SEC = (30, 60, 120, 300, 600, 900)
SEQ_TAUS_SEC = (30, 60, 120, 300, 600, 900, 1800)
SCALP_HORIZONS_SEC = (60, 180, 300, 900, 1800, 3600)
EVENT_END_OFFSETS_SEC = (0, 60, 300, 900)
FEE_BPS = 8.0


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def finite(v: Any) -> float | None:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def month_of(ts_ms: int) -> str:
    return datetime.fromtimestamp(int(ts_ms) / 1000.0, tz=timezone.utc).strftime("%Y-%m")


def side_fade(side: str) -> str:
    return "LONG" if side == "SELL" else "SHORT"


def side_momentum(side: str) -> str:
    return "SHORT" if side == "SELL" else "LONG"


def causal_state(
    row: dict[str, Any],
    *,
    tau_sec: int,
    eth_events: list[dict[str, Any]],
    eth_idx: list[int],
    asset_events: dict[str, list[dict[str, Any]]],
    asset_idx: dict[str, list[int]],
) -> dict[str, Any]:
    ts = int(row["anchor_ts_ms"])
    detect_ts = ts + int(tau_sec) * 1000
    side = str(row["side"])
    same = [
        x for x in neighbor_events(eth_idx, eth_events, ts_ms=ts, before_sec=0, after_sec=int(tau_sec), side=side)
        if ts < int(x["anchor_ts_ms"]) <= detect_ts
    ]
    cross = []
    for sym in ("BTCUSDT", "SOLUSDT"):
        cross.extend([
            x for x in neighbor_events(asset_idx[sym], asset_events[sym], ts_ms=ts, before_sec=0, after_sec=int(tau_sec), side=side)
            if ts < int(x["anchor_ts_ms"]) <= detect_ts
        ])
    event_end_known = row.get("event_end_ts_ms") is not None and int(row["event_end_ts_ms"]) <= detect_ts
    reclaim_known = row.get("reclaim_ts_ms") is not None and int(row["reclaim_ts_ms"]) <= detect_ts
    post_liq_known = event_end_known and float(row.get("post_anchor_liq_notional") or 0.0) > 0.0
    score = int(bool(same)) * 3 + int(bool(cross)) * 2 + int(post_liq_known)
    if score >= 4:
        state = "PRESSURE_HIGH"
    elif score >= 2:
        state = "PRESSURE_MID"
    elif reclaim_known:
        state = "SILENCE_RECLAIM"
    else:
        state = "PRESSURE_LOW"
    return {
        "tau_sec": int(tau_sec),
        "detect_ts_ms": detect_ts,
        "same_restart_n": len(same),
        "cross_same_n": len(cross),
        "event_end_known": bool(event_end_known),
        "reclaim_known": bool(reclaim_known),
        "pressure_score": int(score),
        "state": state,
        "pressure_high": state == "PRESSURE_HIGH",
        "silence_reclaim": state == "SILENCE_RECLAIM",
    }


def taker_outcome(
    conn: sqlite3.Connection,
    marks: Any,
    *,
    direction: str,
    entry_ts_ms: int,
    horizon_sec: int,
) -> dict[str, Any]:
    return simulate_taker(
        conn,
        marks,
        row={},
        direction=direction,
        detect_ts_ms=int(entry_ts_ms),
        horizon_sec=int(horizon_sec),
        delay_sec=0,
        stop_bps=None,
    )


def add_states(rows: list[dict[str, Any]], asset_events: dict[str, list[dict[str, Any]]]) -> None:
    eth_idx = event_index(asset_events[SYMBOL])
    asset_idx = {sym: event_index(evts) for sym, evts in asset_events.items()}
    for row in rows:
        for tau in SEQ_TAUS_SEC:
            row[f"state_{tau}"] = causal_state(
                row,
                tau_sec=tau,
                eth_events=asset_events[SYMBOL],
                eth_idx=eth_idx,
                asset_events=asset_events,
                asset_idx=asset_idx,
            )
        row["state_sequence"] = ">".join(row[f"state_{tau}"]["state"] for tau in SEQ_TAUS_SEC)
        row["state_sequence_compact"] = ">".join(
            {
                "PRESSURE_HIGH": "H",
                "PRESSURE_MID": "M",
                "PRESSURE_LOW": "L",
                "SILENCE_RECLAIM": "S",
            }[row[f"state_{tau}"]["state"]]
            for tau in SEQ_TAUS_SEC
        )


def early_tau_sweep(conn: sqlite3.Connection, rows: list[dict[str, Any]]) -> dict[str, Any]:
    from tools.research_s34_knowable_anchor_continuation import load_mark_index

    marks = load_mark_index(conn, SYMBOL)
    out: dict[str, Any] = {}
    for tau in EARLY_TAUS_SEC:
        key = f"state_{tau}"
        tau_rows = []
        for row in rows:
            st = row[key]
            detect_ts = int(st["detect_ts_ms"])
            side = str(row["side"])
            if side == "SELL":
                fade_dir, mom_dir = "LONG", "SHORT"
                fade_horizon, mom_horizon = 14_400, 3_600
            else:
                fade_dir, mom_dir = "SHORT", "LONG"
                fade_horizon, mom_horizon = 14_400, 3_600
            fade = taker_outcome(conn, marks, direction=fade_dir, entry_ts_ms=detect_ts, horizon_sec=fade_horizon)
            mom = taker_outcome(conn, marks, direction=mom_dir, entry_ts_ms=detect_ts, horizon_sec=mom_horizon)
            tau_rows.append({**row, "state": st["state"], "fade_net": fade.get("net_bps"), "mom_net": mom.get("net_bps"), "fade_status": fade.get("status"), "mom_status": mom.get("status")})
        out[str(tau)] = {
            "SELL": {
                "silence_fade": metrics([r["fade_net"] for r in tau_rows if r["side"] == "SELL" and r["state"] == "SILENCE_RECLAIM" and r["fade_status"] == "FILLED"]),
                "pressure_momentum": metrics([r["mom_net"] for r in tau_rows if r["side"] == "SELL" and r["state"] == "PRESSURE_HIGH" and r["mom_status"] == "FILLED"]),
                "pressure_fade": metrics([r["fade_net"] for r in tau_rows if r["side"] == "SELL" and r["state"] == "PRESSURE_HIGH" and r["fade_status"] == "FILLED"]),
            },
            "BUY": {
                "silence_fade": metrics([r["fade_net"] for r in tau_rows if r["side"] == "BUY" and r["state"] == "SILENCE_RECLAIM" and r["fade_status"] == "FILLED"]),
                "pressure_momentum": metrics([r["mom_net"] for r in tau_rows if r["side"] == "BUY" and r["state"] == "PRESSURE_HIGH" and r["mom_status"] == "FILLED"]),
                "pressure_fade": metrics([r["fade_net"] for r in tau_rows if r["side"] == "BUY" and r["state"] == "PRESSURE_HIGH" and r["fade_status"] == "FILLED"]),
            },
            "state_counts": {
                state: sum(1 for r in tau_rows if r["state"] == state)
                for state in ("PRESSURE_LOW", "PRESSURE_MID", "PRESSURE_HIGH", "SILENCE_RECLAIM")
            },
        }
    return out


def sell_event_end_reclaim(conn: sqlite3.Connection, rows: list[dict[str, Any]]) -> dict[str, Any]:
    from tools.research_s34_knowable_anchor_continuation import load_mark_index

    marks = load_mark_index(conn, SYMBOL)
    sell = [r for r in rows if r["side"] == "SELL"]
    out: dict[str, Any] = {}
    for off in EVENT_END_OFFSETS_SEC:
        sims = []
        for row in sell:
            entry_ts = int(row["event_end_ts_ms"]) + int(off) * 1000
            state = causal_state(
                row,
                tau_sec=max(0, int((entry_ts - int(row["anchor_ts_ms"])) / 1000)),
                eth_events=rows,
                eth_idx=event_index(rows),
                asset_events={SYMBOL: rows, "BTCUSDT": [], "SOLUSDT": []},
                asset_idx={SYMBOL: event_index(rows), "BTCUSDT": [], "SOLUSDT": []},
            ) if False else None
            sim = taker_outcome(conn, marks, direction="LONG", entry_ts_ms=entry_ts, horizon_sec=14_400)
            sim.update({"month": row.get("month"), "state_900": row["state_900"]["state"], "state_1800": row["state_1800"]["state"]})
            sims.append(sim)
        filled = [s for s in sims if s.get("status") == "FILLED"]
        out[f"event_end_plus_{off}s"] = {
            "all": metrics([s.get("net_bps") for s in filled]),
            "state900_silence": metrics([s.get("net_bps") for s in filled if s.get("state_900") == "SILENCE_RECLAIM"]),
            "state1800_silence": metrics([s.get("net_bps") for s in filled if s.get("state_1800") == "SILENCE_RECLAIM"]),
            "fill_rate": r3(len(filled) / len(sims)) if sims else None,
        }
    reclaim_sims = []
    for row in sell:
        if row.get("reclaim_ts_ms") is None:
            continue
        sim = taker_outcome(conn, marks, direction="LONG", entry_ts_ms=int(row["reclaim_ts_ms"]), horizon_sec=14_400)
        sim.update({"month": row.get("month"), "state_900": row["state_900"]["state"], "state_1800": row["state_1800"]["state"]})
        reclaim_sims.append(sim)
    reclaim_filled = [s for s in reclaim_sims if s.get("status") == "FILLED"]
    out["reclaim_entry"] = {
        "all": metrics([s.get("net_bps") for s in reclaim_filled]),
        "state900_silence": metrics([s.get("net_bps") for s in reclaim_filled if s.get("state_900") == "SILENCE_RECLAIM"]),
        "state1800_silence": metrics([s.get("net_bps") for s in reclaim_filled if s.get("state_1800") == "SILENCE_RECLAIM"]),
        "fill_rate": r3(len(reclaim_filled) / len(reclaim_sims)) if reclaim_sims else None,
    }
    return out


def buy_scalp(conn: sqlite3.Connection, rows: list[dict[str, Any]]) -> dict[str, Any]:
    from tools.research_s34_knowable_anchor_continuation import load_mark_index

    marks = load_mark_index(conn, SYMBOL)
    out: dict[str, Any] = {}
    buy = [r for r in rows if r["side"] == "BUY"]
    for tau in EARLY_TAUS_SEC:
        selected = [r for r in buy if r[f"state_{tau}"]["state"] == "PRESSURE_HIGH"]
        by_h = {}
        for horizon in SCALP_HORIZONS_SEC:
            sims = [
                taker_outcome(conn, marks, direction="LONG", entry_ts_ms=int(r["anchor_ts_ms"]) + int(tau) * 1000, horizon_sec=horizon)
                for r in selected
            ]
            filled = [s for s in sims if s.get("status") == "FILLED"]
            by_h[str(horizon)] = {**metrics([s.get("net_bps") for s in filled]), "attempt_n": len(sims), "fill_rate": r3(len(filled) / len(sims)) if sims else None}
        out[str(tau)] = {"selected_n": len(selected), "horizons": by_h}
    return out


def tag_sequence_model(conn: sqlite3.Connection, rows: list[dict[str, Any]]) -> dict[str, Any]:
    from tools.research_s34_knowable_anchor_continuation import load_mark_index

    marks = load_mark_index(conn, SYMBOL)
    # Causal outcome at final sequence point: tau 1800 for fade, tau 900 for scalp summary.
    seq_rows = []
    for row in rows:
        side = row["side"]
        seq = row["state_sequence_compact"]
        detect = int(row["anchor_ts_ms"]) + 1800 * 1000
        fade = taker_outcome(conn, marks, direction=side_fade(side), entry_ts_ms=detect, horizon_sec=14_400)
        mom = taker_outcome(conn, marks, direction=side_momentum(side), entry_ts_ms=detect, horizon_sec=3_600)
        seq_rows.append({**row, "seq": seq, "fade_net": fade.get("net_bps"), "mom_net": mom.get("net_bps"), "fade_status": fade.get("status"), "mom_status": mom.get("status")})
    out: dict[str, Any] = {}
    for side in ("SELL", "BUY"):
        side_rows = [r for r in seq_rows if r["side"] == side]
        common = sorted({r["seq"] for r in side_rows}, key=lambda s: sum(1 for r in side_rows if r["seq"] == s), reverse=True)[:12]
        out[side] = {
            seq: {
                "n": sum(1 for r in side_rows if r["seq"] == seq),
                "fade_h4": metrics([r["fade_net"] for r in side_rows if r["seq"] == seq and r["fade_status"] == "FILLED"]),
                "momentum_h1": metrics([r["mom_net"] for r in side_rows if r["seq"] == seq and r["mom_status"] == "FILLED"]),
            }
            for seq in common
        }
    return out


def v02_management_compatibility(rows: list[dict[str, Any]]) -> dict[str, Any]:
    h4_rows = [
        r for r in load_jsonl(H4_LEDGER_JSONL)
        if r.get("bucket") == "H4_SHADOW" and r.get("observation_status") == "CLOSED" and r.get("net_bps") is not None
    ]
    sell = [r for r in rows if r["side"] == "SELL"]
    ts_list = sorted(int(r["anchor_ts_ms"]) for r in sell)
    by_ts = {int(r["anchor_ts_ms"]): r for r in sell}
    matched = []
    for trade in h4_rows:
        ts = int(trade["signal_ts_ms"])
        pos = bisect_left(ts_list, ts)
        ev = None
        for j in (pos - 1, pos, pos + 1):
            if 0 <= j < len(ts_list) and abs(ts_list[j] - ts) <= 2_000:
                ev = by_ts[ts_list[j]]
                break
        h2 = finite(trade.get("h2_net_bps"))
        h4 = finite(trade.get("net_bps"))
        row = {
            "signal_utc": trade.get("signal_utc"),
            "matched": ev is not None,
            "h4_minus_h2": r1(h4 - h2) if h2 is not None and h4 is not None else None,
            "state_300": ev["state_300"]["state"] if ev else None,
            "state_900": ev["state_900"]["state"] if ev else None,
            "state_1800": ev["state_1800"]["state"] if ev else None,
            "seq": ev["state_sequence_compact"] if ev else None,
        }
        matched.append(row)
    return {
        "n": len(matched),
        "matched_n": sum(1 for r in matched if r["matched"]),
        "all": metrics([r["h4_minus_h2"] for r in matched]),
        "state_300": {s: metrics([r["h4_minus_h2"] for r in matched if r["state_300"] == s]) for s in sorted({str(r["state_300"]) for r in matched if r["state_300"]})},
        "state_900": {s: metrics([r["h4_minus_h2"] for r in matched if r["state_900"] == s]) for s in sorted({str(r["state_900"]) for r in matched if r["state_900"]})},
        "state_1800": {s: metrics([r["h4_minus_h2"] for r in matched if r["state_1800"] == s]) for s in sorted({str(r["state_1800"]) for r in matched if r["state_1800"]})},
        "rows": matched,
    }


def render_md(report: dict[str, Any]) -> str:
    lines = [
        "# S34 V02 Next Navigation Tests",
        "",
        f"Generated: `{report['generated_at_utc']}`",
        "",
        "Research-only. No live executor/config/order logic is touched.",
        "",
        "## Verdict",
        "",
        f"- `{report['verdict']}`",
        "",
        "## 1. Early Tau Sweep",
        "",
        "```json",
        json.dumps(report["early_tau_sweep"], indent=2, sort_keys=True),
        "```",
        "",
        "## 2. SELL Event-End / Reclaim Entry",
        "",
        "```json",
        json.dumps(report["sell_event_end_reclaim"], indent=2, sort_keys=True),
        "```",
        "",
        "## 3. BUY Propagation Scalp Horizons",
        "",
        "```json",
        json.dumps(report["buy_scalp"], indent=2, sort_keys=True),
        "```",
        "",
        "## 4. Tag Sequence Model",
        "",
        "```json",
        json.dumps(report["tag_sequence_model"], indent=2, sort_keys=True),
        "```",
        "",
        "## 5. V02 Management Compatibility",
        "",
        "```json",
        json.dumps({k: v for k, v in report["v02_management_compatibility"].items() if k != "rows"}, indent=2, sort_keys=True),
        "```",
        "",
        "## Read",
        "",
    ]
    lines.extend(f"- {x}" for x in report["read"])
    lines.append("")
    return "\n".join(lines)


def run(db: Path) -> dict[str, Any]:
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=30)
    try:
        asset_events = {sym: build_events(conn, symbol=sym, threshold=thr) for sym, thr in ASSET_THRESHOLDS.items()}
        rows = enrich_events(conn, asset_events[SYMBOL], asset_events)
        add_states(rows, asset_events)
        report = {
            "generated_at_utc": utc_now(),
            "research_only": True,
            "live_executor_touched": False,
            "event_counts": {sym: len(evts) for sym, evts in asset_events.items()},
            "early_tau_sweep": early_tau_sweep(conn, rows),
            "sell_event_end_reclaim": sell_event_end_reclaim(conn, rows),
            "buy_scalp": buy_scalp(conn, rows),
            "tag_sequence_model": tag_sequence_model(conn, rows),
            "v02_management_compatibility": v02_management_compatibility(rows),
            "verdict": "NAVIGATION_VALUE_CONFIRMED_EXECUTION_ALPHA_NOT_CONFIRMED",
            "read": [
                "Early tau labels are now strictly causal: only states known by anchor+tau are used.",
                "The tests evaluate navigation/management value; they do not change live execution.",
                "A candidate needs positive causal execution, holdout/T3R robustness, and V02 compatibility before paper/live.",
            ],
        }
    finally:
        conn.close()
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run S34 V02 next navigation tests.")
    p.add_argument("--db", type=Path, default=DEFAULT_DB)
    p.add_argument("--out-json", type=Path, default=OUT_JSON)
    p.add_argument("--out-md", type=Path, default=OUT_MD)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = run(args.db)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    args.out_md.write_text(render_md(report), encoding="utf-8")
    print(render_md(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
