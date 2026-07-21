"""S34 stress-reaction deep tests.

Research-only follow-up for the 15m stress-reaction scalp lead:
- BTC stress threshold dose-response.
- V-depth tail cleaner.
- TP/SL/horizon exit sweep.
- MFE/MAE and time-to-hit anatomy.
- Event-chain filters.
- v0.2 live-route guard tags.
- Fee sensitivity.

No live executor, order logic, size, leverage, or .env changes.
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
from tools.s34_navigation_scalp_tail_tests import prepare_rows, profile  # noqa: E402

OUT_JSON = ROOT / "reports" / "research" / "s34" / "S34_STRESS_REACTION_DEEP_TESTS.json"
OUT_MD = ROOT / "reports" / "research" / "s34" / "S34_STRESS_REACTION_DEEP_TESTS.md"

BASE_FEE_BPS = 5.0
BTC_THRESHOLDS = [-50.0, -75.0, -100.0, -150.0, -200.0, -250.0]
VDEPTH_FILTERS: dict[str, Callable[[dict[str, Any]], bool]] = {
    "all": lambda r: True,
    "v_lt_25": lambda r: float(r.get("vdepth_bps") or 0.0) < 25.0,
    "v_25_40": lambda r: 25.0 <= float(r.get("vdepth_bps") or 0.0) < 40.0,
    "v_lt_40": lambda r: float(r.get("vdepth_bps") or 0.0) < 40.0,
    "v_lt_50": lambda r: float(r.get("vdepth_bps") or 0.0) < 50.0,
    "v_lt_60": lambda r: float(r.get("vdepth_bps") or 0.0) < 60.0,
    "exclude_danger_high": lambda r: "VDEPTH_DANGER_HIGH" not in set(r.get("tags") or []),
}
TPS = [75.0, 100.0, 125.0, 150.0, 200.0]
SLS = [20.0, 25.0, 30.0, 40.0, 50.0]
HORIZONS = {"10m": 600, "12m": 720, "15m": 900, "20m": 1200}
FEES = [0.0, 2.5, 5.0, 8.0]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ts(row: dict[str, Any]) -> int:
    return int(row.get("signal_ts_ms") or 0)


def stress3(row: dict[str, Any]) -> bool:
    return int(row.get("stress_score") or 0) >= 3


def btc_filter(threshold: float) -> Callable[[dict[str, Any]], bool]:
    return lambda r: stress3(r) and float(r.get("btc4h_bps") or 0.0) < threshold


def mark_series(conn: sqlite3.Connection, start_ms: int, end_ms: int) -> list[tuple[int, float]]:
    rows = conn.execute(
        "SELECT ts_ms, mark_price FROM mark_prices WHERE symbol='ETHUSDT' AND ts_ms>=? AND ts_ms<=? ORDER BY ts_ms ASC",
        (int(start_ms), int(end_ms)),
    ).fetchall()
    return [(int(t), float(p)) for t, p in rows if p is not None]


def fixed_horizon(conn: sqlite3.Connection, row: dict[str, Any], sec: int, direction: str, fee_bps: float = BASE_FEE_BPS) -> float | None:
    entry = mark_at_or_after(conn, "ETHUSDT", ts(row))
    exit_ = mark_at_or_after(conn, "ETHUSDT", ts(row) + sec * 1000)
    if not entry or not exit_ or entry[1] <= 0:
        return None
    raw = (exit_[1] - entry[1]) / entry[1] * 10_000.0
    pnl = raw if direction == "NORMAL" else -raw
    return pnl - fee_bps


def bracket_outcome(
    conn: sqlite3.Connection,
    row: dict[str, Any],
    *,
    horizon_sec: int,
    direction: str,
    tp: float,
    sl: float,
    fee_bps: float = BASE_FEE_BPS,
) -> tuple[float | None, str, int | None]:
    entry = mark_at_or_after(conn, "ETHUSDT", ts(row))
    if not entry or entry[1] <= 0:
        return None, "NO_ENTRY", None
    entry_ts, entry_px = entry
    series = mark_series(conn, entry_ts, ts(row) + horizon_sec * 1000)
    if not series:
        return None, "NO_SERIES", None
    for t, px in series:
        raw = (px - entry_px) / entry_px * 10_000.0
        pnl = raw if direction == "NORMAL" else -raw
        if pnl >= tp:
            return tp - fee_bps, "TP", int((t - entry_ts) / 1000)
        if pnl <= -sl:
            return -sl - fee_bps, "SL", int((t - entry_ts) / 1000)
    end_px = series[-1][1]
    raw = (end_px - entry_px) / entry_px * 10_000.0
    pnl = raw if direction == "NORMAL" else -raw
    return pnl - fee_bps, "TIME", int((series[-1][0] - entry_ts) / 1000)


def target_rows(rows: list[dict[str, Any]], filt: Callable[[dict[str, Any]], bool]) -> list[dict[str, Any]]:
    return [r for r in rows if filt(r)]


def fixed_summary(conn: sqlite3.Connection, rows: list[dict[str, Any]], filt: Callable[[dict[str, Any]], bool], *, sec: int, direction: str) -> dict[str, Any]:
    vals = [v for r in target_rows(rows, filt) if (v := fixed_horizon(conn, r, sec, direction)) is not None]
    return summary(vals)


def bracket_summary(
    conn: sqlite3.Connection,
    rows: list[dict[str, Any]],
    filt: Callable[[dict[str, Any]], bool],
    *,
    sec: int,
    direction: str,
    tp: float,
    sl: float,
    fee_bps: float = BASE_FEE_BPS,
) -> dict[str, Any]:
    vals = []
    exits = defaultdict(int)
    times = []
    for r in target_rows(rows, filt):
        v, ex, hit_sec = bracket_outcome(conn, r, horizon_sec=sec, direction=direction, tp=tp, sl=sl, fee_bps=fee_bps)
        if v is None:
            continue
        vals.append(v)
        exits[ex] += 1
        if hit_sec is not None:
            times.append(hit_sec)
    s = summary(vals)
    return {"summary": s, "exits": dict(exits), "median_exit_sec": r1(median(times)) if times else None}


def btc_sweep(conn: sqlite3.Connection, rows: list[dict[str, Any]]) -> dict[str, Any]:
    out = {}
    for th in BTC_THRESHOLDS:
        filt = btc_filter(th)
        out[f"btc4h_lt_{int(abs(th))}"] = {
            "n": len(target_rows(rows, filt)),
            "fixed_15m_reverse": fixed_summary(conn, rows, filt, sec=900, direction="REVERSE"),
            "bracket_tp150_sl30_15m": bracket_summary(conn, rows, filt, sec=900, direction="REVERSE", tp=150.0, sl=30.0),
            "bracket_tp150_sl50_15m": bracket_summary(conn, rows, filt, sec=900, direction="REVERSE", tp=150.0, sl=50.0),
        }
    return out


def vdepth_sweep(conn: sqlite3.Connection, rows: list[dict[str, Any]]) -> dict[str, Any]:
    out = {}
    for name, vf in VDEPTH_FILTERS.items():
        filt = lambda r, vf=vf: stress3(r) and float(r.get("btc4h_bps") or 0.0) < -75.0 and vf(r)
        out[name] = {
            "n": len(target_rows(rows, filt)),
            "fixed_15m_reverse": fixed_summary(conn, rows, filt, sec=900, direction="REVERSE"),
            "bracket_tp150_sl30_15m": bracket_summary(conn, rows, filt, sec=900, direction="REVERSE", tp=150.0, sl=30.0),
        }
    return out


def exit_sweep(conn: sqlite3.Connection, rows: list[dict[str, Any]]) -> dict[str, Any]:
    filt = lambda r: stress3(r) and float(r.get("btc4h_bps") or 0.0) < -75.0
    cells = []
    for hname, sec in HORIZONS.items():
        for tp in TPS:
            for sl in SLS:
                b = bracket_summary(conn, rows, filt, sec=sec, direction="REVERSE", tp=tp, sl=sl)
                cells.append({"horizon": hname, "tp": tp, "sl": sl, **b})
    cells.sort(key=lambda c: (float(c["summary"].get("t3r_bps") or -1e9), float(c["summary"].get("sum_bps") or -1e9)), reverse=True)
    return {"n": len(target_rows(rows, filt)), "top": cells[:20]}


def mfe_mae(conn: sqlite3.Connection, rows: list[dict[str, Any]]) -> dict[str, Any]:
    filt = lambda r: stress3(r) and float(r.get("btc4h_bps") or 0.0) < -75.0
    entries = target_rows(rows, filt)
    records = []
    for r in entries:
        entry = mark_at_or_after(conn, "ETHUSDT", ts(r))
        if not entry or entry[1] <= 0:
            continue
        entry_ts, entry_px = entry
        series = mark_series(conn, entry_ts, ts(r) + 900_000)
        if not series:
            continue
        path = []
        for t, px in series:
            raw = (px - entry_px) / entry_px * 10_000.0
            pnl = -raw
            path.append((int((t - entry_ts) / 1000), pnl))
        mfe_t, mfe = max(path, key=lambda x: x[1])
        mae_t, mae = min(path, key=lambda x: x[1])
        hit_tp150 = next((t for t, pnl in path if pnl >= 150.0), None)
        hit_sl30 = next((t for t, pnl in path if pnl <= -30.0), None)
        final = path[-1][1] - BASE_FEE_BPS
        records.append({"row": r, "mfe": mfe, "mfe_sec": mfe_t, "mae": mae, "mae_sec": mae_t, "tp150_sec": hit_tp150, "sl30_sec": hit_sl30, "final_net": final})
    winners = [rec for rec in records if rec["final_net"] > 0]
    losers = [rec for rec in records if rec["final_net"] <= -50.0]
    return {
        "n": len(records),
        "all": record_summary(records),
        "winners": record_summary(winners),
        "losers": record_summary(losers),
    }


def record_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    def vals(key: str) -> list[float]:
        return [float(r[key]) for r in records if r.get(key) is not None and math.isfinite(float(r[key]))]
    return {
        "n": len(records),
        "mfe_median": r1(median(vals("mfe"))) if vals("mfe") else None,
        "mae_median": r1(median(vals("mae"))) if vals("mae") else None,
        "mfe_sec_median": r1(median(vals("mfe_sec"))) if vals("mfe_sec") else None,
        "mae_sec_median": r1(median(vals("mae_sec"))) if vals("mae_sec") else None,
        "tp150_hit_n": len(vals("tp150_sec")),
        "tp150_sec_median": r1(median(vals("tp150_sec"))) if vals("tp150_sec") else None,
        "sl30_hit_n": len(vals("sl30_sec")),
        "sl30_sec_median": r1(median(vals("sl30_sec"))) if vals("sl30_sec") else None,
        "final": summary(vals("final_net")),
    }


def sequence_counts(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    ordered = sorted(rows, key=ts)
    out = {}
    for row in ordered:
        t = ts(row)
        prior_15m = [r for r in ordered if 0 < t - ts(r) <= 900_000]
        prior_1h = [r for r in ordered if 0 < t - ts(r) <= 3_600_000]
        near_15m = [r for r in ordered if 0 <= abs(ts(r) - t) <= 900_000]
        out[str(row.get("event_id"))] = {
            "prior_15m_n": len(prior_15m),
            "prior_1h_n": len(prior_1h),
            "near_15m_n": len(near_15m),
            "near_15m_thresholds": len({int(float(r.get("threshold_usd") or 0)) for r in near_15m}),
        }
    return out


def event_chain_filters(conn: sqlite3.Connection, rows: list[dict[str, Any]]) -> dict[str, Any]:
    seq = sequence_counts(rows)
    base = lambda r: stress3(r) and float(r.get("btc4h_bps") or 0.0) < -75.0
    filters = {
        "all": base,
        "prior_15m_0": lambda r: base(r) and seq[str(r.get("event_id"))]["prior_15m_n"] == 0,
        "prior_15m_ge1": lambda r: base(r) and seq[str(r.get("event_id"))]["prior_15m_n"] >= 1,
        "prior_1h_ge3": lambda r: base(r) and seq[str(r.get("event_id"))]["prior_1h_n"] >= 3,
        "near_15m_thresholds_ge2": lambda r: base(r) and seq[str(r.get("event_id"))]["near_15m_thresholds"] >= 2,
        "near_15m_thresholds_ge3": lambda r: base(r) and seq[str(r.get("event_id"))]["near_15m_thresholds"] >= 3,
    }
    out = {}
    for name, filt in filters.items():
        out[name] = {
            "n": len(target_rows(rows, filt)),
            "fixed_15m_reverse": fixed_summary(conn, rows, filt, sec=900, direction="REVERSE"),
            "bracket_tp150_sl30_15m": bracket_summary(conn, rows, filt, sec=900, direction="REVERSE", tp=150.0, sl=30.0),
        }
    return out


def v02_guard(conn: sqlite3.Connection, rows: list[dict[str, Any]]) -> dict[str, Any]:
    filters = {
        "all_v02": route_v02,
        "tail_low": lambda r: route_v02(r) and "TAIL_LOW_CONTEXT" in set(r.get("tags") or []),
        "tail_high_unknown": lambda r: route_v02(r) and "TAIL_HIGH_OR_UNKNOWN" in set(r.get("tags") or []),
        "bid_ok": lambda r: route_v02(r) and "BID_DEPTH_OK" in set(r.get("tags") or []),
        "bid_thin": lambda r: route_v02(r) and "BID_DEPTH_THIN" in set(r.get("tags") or []),
        "bid_heavy": lambda r: route_v02(r) and "BID_DEPTH_HEAVY" in set(r.get("tags") or []),
        "stress_ge2": lambda r: route_v02(r) and int(r.get("stress_score") or 0) >= 2,
        "stress_ge3": lambda r: route_v02(r) and int(r.get("stress_score") or 0) >= 3,
    }
    out = {}
    for name, filt in filters.items():
        vals2h = [v for r in target_rows(rows, filt) if (v := fixed_horizon(conn, r, 7200, "NORMAL")) is not None]
        vals15 = [v for r in target_rows(rows, filt) if (v := fixed_horizon(conn, r, 900, "NORMAL")) is not None]
        out[name] = {"n": len(target_rows(rows, filt)), "normal_15m": summary(vals15), "normal_2h": summary(vals2h), "profile": profile(target_rows(rows, filt))}
    return out


def fee_sensitivity(conn: sqlite3.Connection, rows: list[dict[str, Any]]) -> dict[str, Any]:
    filt = lambda r: stress3(r) and float(r.get("btc4h_bps") or 0.0) < -75.0
    out = {}
    for fee in FEES:
        out[f"fee_{fee:g}bps"] = bracket_summary(conn, rows, filt, sec=900, direction="REVERSE", tp=150.0, sl=30.0, fee_bps=fee)
    return out


def big_cards(conn: sqlite3.Connection, rows: list[dict[str, Any]]) -> dict[str, Any]:
    filt = lambda r: stress3(r) and float(r.get("btc4h_bps") or 0.0) < -75.0
    scored = []
    for r in target_rows(rows, filt):
        v = fixed_horizon(conn, r, 900, "REVERSE")
        if v is not None:
            scored.append((v, r))
    scored.sort(key=lambda x: x[0])
    return {
        "summary": summary([v for v, _ in scored]),
        "worst10": [card(r, v) for v, r in scored[:10]],
        "best10": [card(r, v) for v, r in reversed(scored[-10:])],
        "winner_profile": profile([r for v, r in scored if v > 0]),
        "loser_profile": profile([r for v, r in scored if v <= -50]),
        "tail_profile": profile([r for v, r in scored if v <= -150]),
    }


def card(row: dict[str, Any], value: float) -> dict[str, Any]:
    return {
        "event_id": row.get("event_id"),
        "fold": row.get("fold"),
        "signal_utc": row.get("signal_utc"),
        "value_bps": r1(value),
        "stress_score": row.get("stress_score"),
        "threshold": row.get("threshold_usd"),
        "vdepth": row.get("vdepth_bps"),
        "prior4h": row.get("prior4h_bps"),
        "eth1h": row.get("eth1h_bps"),
        "btc4h": row.get("btc4h_bps"),
        "bid_depth": row.get("bid_depth_usd"),
        "book_imbalance": row.get("book_imbalance"),
        "tags": row.get("tags"),
    }


def run() -> dict[str, Any]:
    rows = prepare_rows()
    with sqlite3.connect(DEFAULT_DB) as conn:
        return {
            "generated_at_utc": utc_now(),
            "status": "RESEARCH_ONLY_NO_LIVE_CHANGE",
            "btc_sweep": btc_sweep(conn, rows),
            "vdepth_sweep_btc_lt_75": vdepth_sweep(conn, rows),
            "exit_sweep_btc_lt_75": exit_sweep(conn, rows),
            "mfe_mae_btc_lt_75": mfe_mae(conn, rows),
            "event_chain_filters_btc_lt_75": event_chain_filters(conn, rows),
            "v02_guard": v02_guard(conn, rows),
            "fee_sensitivity_btc_lt_75_tp150_sl30_15m": fee_sensitivity(conn, rows),
            "big_winner_loser_btc_lt_75_reverse_15m": big_cards(conn, rows),
        }


def fmt(s: dict[str, Any]) -> str:
    return (
        f"N={s.get('n')} sum={s.get('sum_bps')} med={s.get('median_bps')} "
        f"T3R={s.get('t3r_bps')} tail150={s.get('tail_lte_minus150_n')} maxLoss={s.get('max_loss_bps')}"
    )


def write_report(result: dict[str, Any]) -> None:
    lines = [
        "# S34 Stress Reaction Deep Tests",
        "",
        f"Generated: `{result['generated_at_utc']}`",
        "",
        f"Status: `{result['status']}`",
        "",
        "## BTC Stress Sweep",
        "",
        "| Filter | N | Fixed 15m Reverse | TP150/SL30/15m | TP150/SL50/15m |",
        "| --- | ---: | --- | --- | --- |",
    ]
    for name, row in result["btc_sweep"].items():
        lines.append(
            f"| {name} | {row['n']} | {fmt(row['fixed_15m_reverse'])} | "
            f"{fmt(row['bracket_tp150_sl30_15m']['summary'])} | {fmt(row['bracket_tp150_sl50_15m']['summary'])} |"
        )

    lines.extend(["", "## VDepth Cleaner (score>=3 & btc4h<-75)", ""])
    lines.append("| Filter | N | Fixed 15m Reverse | TP150/SL30/15m |")
    lines.append("| --- | ---: | --- | --- |")
    for name, row in result["vdepth_sweep_btc_lt_75"].items():
        lines.append(f"| {name} | {row['n']} | {fmt(row['fixed_15m_reverse'])} | {fmt(row['bracket_tp150_sl30_15m']['summary'])} |")

    lines.extend(["", "## Exit Sweep Top 20 (score>=3 & btc4h<-75)", ""])
    lines.append(f"N: `{result['exit_sweep_btc_lt_75']['n']}`")
    lines.append("")
    lines.append("| Horizon | TP | SL | Summary | Exits | Median exit sec |")
    lines.append("| --- | ---: | ---: | --- | --- | ---: |")
    for row in result["exit_sweep_btc_lt_75"]["top"][:20]:
        lines.append(
            f"| {row['horizon']} | {row['tp']} | {row['sl']} | {fmt(row['summary'])} | "
            f"`{row['exits']}` | {row['median_exit_sec']} |"
        )

    lines.extend(["", "## MFE / MAE Timing (score>=3 & btc4h<-75, reverse 15m)", ""])
    for name, row in result["mfe_mae_btc_lt_75"].items():
        if isinstance(row, dict):
            lines.append(f"- `{name}`: `{row}`")

    lines.extend(["", "## Event Chain Filters", ""])
    lines.append("| Filter | N | Fixed 15m Reverse | TP150/SL30/15m |")
    lines.append("| --- | ---: | --- | --- |")
    for name, row in result["event_chain_filters_btc_lt_75"].items():
        lines.append(f"| {name} | {row['n']} | {fmt(row['fixed_15m_reverse'])} | {fmt(row['bracket_tp150_sl30_15m']['summary'])} |")

    lines.extend(["", "## v0.2 Guard", ""])
    lines.append("| Filter | N | Normal 15m | Normal 2h |")
    lines.append("| --- | ---: | --- | --- |")
    for name, row in result["v02_guard"].items():
        lines.append(f"| {name} | {row['n']} | {fmt(row['normal_15m'])} | {fmt(row['normal_2h'])} |")

    lines.extend(["", "## Fee Sensitivity (score>=3 & btc4h<-75, reverse TP150/SL30/15m)", ""])
    lines.append("| Fee | Summary | Exits |")
    lines.append("| --- | --- | --- |")
    for name, row in result["fee_sensitivity_btc_lt_75_tp150_sl30_15m"].items():
        lines.append(f"| {name} | {fmt(row['summary'])} | `{row['exits']}` |")

    lines.extend(["", "## Big Winner / Loser Anatomy (score>=3 & btc4h<-75, reverse 15m)", ""])
    block = result["big_winner_loser_btc_lt_75_reverse_15m"]
    lines.append(f"Summary: {fmt(block['summary'])}")
    lines.append(f"Winner profile: `{block['winner_profile']}`")
    lines.append(f"Loser profile: `{block['loser_profile']}`")
    lines.append(f"Tail profile: `{block['tail_profile']}`")
    lines.append("")
    lines.append("Worst 10:")
    for row in block["worst10"]:
        lines.append(f"- `{row}`")
    lines.append("Best 10:")
    for row in block["best10"]:
        lines.append(f"- `{row}`")

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
