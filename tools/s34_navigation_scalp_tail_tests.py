"""S34 stress scalp tail tests.

Research-only follow-up:
- TP/SL grid for the score>=3 stress reverse reaction at short horizons.
- Big winner / big loser anatomy for stress reverse and v0.2 stress guard.
- Tail-cleaner screen for score>=3 reverse reaction.

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
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.s34_navigation_causal_gauntlet import normal_value, value_from_normal  # noqa: E402
from tools.s34_navigation_full_followup import DEFAULT_DB, NAV_EVENTS, load_jsonl, mark_at_or_after, r1, r3, summary  # noqa: E402
from tools.s34_navigation_regime_inversion_walkforward import attach_preds, build_cells, cell_stats, make_folds, neighbors  # noqa: E402
from tools.s34_navigation_scalp_and_stress import route_v02, stress_score  # noqa: E402
from tools.s34_navigation_branch_anatomy import fold_state, top_counts  # noqa: E402
from tools.s34_navigation_causal_gauntlet import causal_preds  # noqa: E402

OUT_JSON = ROOT / "reports" / "research" / "s34" / "S34_NAVIGATION_SCALP_TAIL_TESTS.json"
OUT_MD = ROOT / "reports" / "research" / "s34" / "S34_NAVIGATION_SCALP_TAIL_TESTS.md"

FEE_BPS = 5.0
HORIZONS = {"5m": 300, "15m": 900, "30m": 1800}
TPS = [30.0, 50.0, 75.0, 100.0, 150.0]
SLS = [30.0, 50.0, 75.0, 100.0, 150.0]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ts(row: dict[str, Any]) -> int:
    return int(row.get("signal_ts_ms") or 0)


def top_cells(train: list[dict[str, Any]], by_name: dict[str, Any], n: int = 5) -> list[Any]:
    stats = cell_stats(train, list(by_name.values()))
    eligible = [v for v in stats.values() if float(v.get("t3r") or 0.0) > 0]
    eligible.sort(key=lambda v: float(v.get("t3r") or 0.0), reverse=True)
    return [by_name[v["cell"].name] for v in eligible[:n] if v["cell"].name in by_name]


def prepare_rows() -> list[dict[str, Any]]:
    raw = load_jsonl(NAV_EVENTS)
    folds = make_folds(raw, folds=5, min_train_frac=0.4)
    out = []
    for fold_idx, (train_raw, hold_raw) in enumerate(folds, start=1):
        train_normals = [normal_value(r) for r in train_raw]
        train_pred = causal_preds(neighbors(train_raw, train_raw, leave_one_out=True), train_normals)
        hold_pred = causal_preds(neighbors(train_raw, hold_raw, leave_one_out=False), train_normals)
        train = attach_preds(train_raw, train_pred)
        hold = attach_preds(hold_raw, hold_pred)
        by_name = {c.name: c for c in build_cells(train + hold)}
        selected = top_cells(train, by_name, 5)
        state = fold_state(hold)
        fold_meta = {"state": state}
        for row in hold:
            item = dict(row)
            top_hit = any(c.selector(row) for c in selected)
            item["fold"] = fold_idx
            item["stress_score"] = stress_score(fold_meta, row, top_hit)
            item["fold_density_per_day"] = state.get("event_density_per_day")
            item["fold_tail150_rate"] = state.get("tail150_rate")
            out.append(item)
    return out


def mark_series(conn: sqlite3.Connection, start_ms: int, end_ms: int) -> list[tuple[int, float]]:
    rows = conn.execute(
        "SELECT ts_ms, mark_price FROM mark_prices WHERE symbol='ETHUSDT' AND ts_ms>=? AND ts_ms<=? ORDER BY ts_ms ASC",
        (int(start_ms), int(end_ms)),
    ).fetchall()
    return [(int(t), float(p)) for t, p in rows if p is not None]


def fixed_horizon(conn: sqlite3.Connection, row: dict[str, Any], sec: int, direction: str) -> float | None:
    entry = mark_at_or_after(conn, "ETHUSDT", ts(row))
    exit_ = mark_at_or_after(conn, "ETHUSDT", ts(row) + sec * 1000)
    if not entry or not exit_ or entry[1] <= 0:
        return None
    raw = (exit_[1] - entry[1]) / entry[1] * 10_000.0
    return raw - FEE_BPS if direction == "NORMAL" else -raw - FEE_BPS


def bracket_outcome(conn: sqlite3.Connection, row: dict[str, Any], *, horizon_sec: int, direction: str, tp: float, sl: float) -> tuple[float | None, str]:
    entry = mark_at_or_after(conn, "ETHUSDT", ts(row))
    if not entry or entry[1] <= 0:
        return None, "NO_ENTRY"
    entry_ts, entry_px = entry
    series = mark_series(conn, entry_ts, ts(row) + horizon_sec * 1000)
    if not series:
        return None, "NO_SERIES"
    for _, px in series:
        raw = (px - entry_px) / entry_px * 10_000.0
        pnl = raw if direction == "NORMAL" else -raw
        if pnl >= tp:
            return tp - FEE_BPS, "TP"
        if pnl <= -sl:
            return -sl - FEE_BPS, "SL"
    end_px = series[-1][1]
    raw = (end_px - entry_px) / entry_px * 10_000.0
    pnl = raw if direction == "NORMAL" else -raw
    return pnl - FEE_BPS, "TIME"


def grid(rows: list[dict[str, Any]], conn: sqlite3.Connection, *, direction: str, group_filter) -> dict[str, Any]:
    target = [r for r in rows if group_filter(r)]
    out = {}
    for hname, sec in HORIZONS.items():
        fixed_vals = [v for r in target if (v := fixed_horizon(conn, r, sec, direction)) is not None]
        cells = []
        for tp in TPS:
            for sl in SLS:
                vals = []
                exits = defaultdict(int)
                for r in target:
                    v, exit_ = bracket_outcome(conn, r, horizon_sec=sec, direction=direction, tp=tp, sl=sl)
                    if v is None:
                        continue
                    vals.append(v)
                    exits[exit_] += 1
                s = summary(vals)
                cells.append({"tp": tp, "sl": sl, "summary": s, "exits": dict(exits)})
        cells.sort(key=lambda c: (float(c["summary"].get("t3r_bps") or -1e9), float(c["summary"].get("sum_bps") or -1e9)), reverse=True)
        out[hname] = {"fixed": summary(fixed_vals), "top_brackets": cells[:10]}
    return {"n": len(target), "horizons": out}


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


def winner_loser_anatomy(rows: list[dict[str, Any]], conn: sqlite3.Connection, *, direction: str, horizon_sec: int, group_filter) -> dict[str, Any]:
    scored = []
    for r in rows:
        if not group_filter(r):
            continue
        v = fixed_horizon(conn, r, horizon_sec, direction)
        if v is not None:
            scored.append((v, r))
    scored.sort(key=lambda x: x[0])
    vals = [v for v, _ in scored]
    return {
        "summary": summary(vals),
        "worst10": [card(r, v) for v, r in scored[:10]],
        "best10": [card(r, v) for v, r in reversed(scored[-10:])],
        "winner_profile": profile([r for v, r in scored if v > 0]),
        "loser_profile": profile([r for v, r in scored if v <= -50]),
        "tail_profile": profile([r for v, r in scored if v <= -150]),
    }


def profile(items: list[dict[str, Any]]) -> dict[str, Any]:
    def avg(key: str) -> float | None:
        vals = [float(r[key]) for r in items if r.get(key) is not None and math.isfinite(float(r[key]))]
        return r1(sum(vals) / len(vals)) if vals else None
    return {
        "n": len(items),
        "avg_threshold": avg("threshold_usd"),
        "avg_vdepth": avg("vdepth_bps"),
        "avg_prior4h": avg("prior4h_bps"),
        "avg_eth1h": avg("eth1h_bps"),
        "avg_btc4h": avg("btc4h_bps"),
        "avg_bid_depth": avg("bid_depth_usd"),
        "avg_book_imbalance": avg("book_imbalance"),
        "tag_mix": top_counts([str(t) for r in items for t in (r.get("tags") or [])], 8),
        "threshold_mix": top_counts([f"thr{int(float(r.get('threshold_usd') or 0))}" for r in items], 5),
    }


def tail_cleaner(rows: list[dict[str, Any]], conn: sqlite3.Connection) -> dict[str, Any]:
    base_filter = lambda r: int(r.get("stress_score") or 0) >= 3
    filters = {
        "all_score3": base_filter,
        "btc4h_lt_-75": lambda r: base_filter(r) and float(r.get("btc4h_bps") or 0) < -75,
        "btc4h_ge_-75": lambda r: base_filter(r) and float(r.get("btc4h_bps") or 0) >= -75,
        "vdepth_lt_25": lambda r: base_filter(r) and float(r.get("vdepth_bps") or 0) < 25,
        "vdepth_25_40": lambda r: base_filter(r) and 25 <= float(r.get("vdepth_bps") or 0) < 40,
        "vdepth_ge_40": lambda r: base_filter(r) and float(r.get("vdepth_bps") or 0) >= 40,
        "bid_depth_ok": lambda r: base_filter(r) and float(r.get("bid_depth_usd") or 0) > 0,
        "risk_off": lambda r: base_filter(r) and "RISK_OFF_REBOUND" in set(r.get("tags") or []),
        "neutral": lambda r: base_filter(r) and "NEUTRAL_CONTEXT" in set(r.get("tags") or []),
    }
    out = {}
    for name, fn in filters.items():
        vals15 = [v for r in rows if fn(r) and (v := fixed_horizon(conn, r, 900, "REVERSE")) is not None]
        vals30 = [v for r in rows if fn(r) and (v := fixed_horizon(conn, r, 1800, "REVERSE")) is not None]
        out[name] = {"n": len(vals15), "reverse_15m": summary(vals15), "reverse_30m": summary(vals30)}
    return out


def run() -> dict[str, Any]:
    rows = prepare_rows()
    with sqlite3.connect(DEFAULT_DB) as conn:
        result = {
            "generated_at_utc": utc_now(),
            "status": "RESEARCH_ONLY_NO_LIVE_CHANGE",
            "stress_score3_reverse_grid": grid(rows, conn, direction="REVERSE", group_filter=lambda r: int(r.get("stress_score") or 0) >= 3),
            "stress_score3_reverse_anatomy_15m": winner_loser_anatomy(rows, conn, direction="REVERSE", horizon_sec=900, group_filter=lambda r: int(r.get("stress_score") or 0) >= 3),
            "stress_score3_reverse_anatomy_30m": winner_loser_anatomy(rows, conn, direction="REVERSE", horizon_sec=1800, group_filter=lambda r: int(r.get("stress_score") or 0) >= 3),
            "v02_stress_guard_anatomy_2h": {
                "baseline": winner_loser_anatomy(rows, conn, direction="NORMAL", horizon_sec=7200, group_filter=route_v02),
                "score_ge_1": winner_loser_anatomy(rows, conn, direction="NORMAL", horizon_sec=7200, group_filter=lambda r: route_v02(r) and int(r.get("stress_score") or 0) >= 1),
                "score_ge_2": winner_loser_anatomy(rows, conn, direction="NORMAL", horizon_sec=7200, group_filter=lambda r: route_v02(r) and int(r.get("stress_score") or 0) >= 2),
                "score_ge_3": winner_loser_anatomy(rows, conn, direction="NORMAL", horizon_sec=7200, group_filter=lambda r: route_v02(r) and int(r.get("stress_score") or 0) >= 3),
            },
            "tail_cleaner": tail_cleaner(rows, conn),
        }
    return result


def fmt(s: dict[str, Any]) -> str:
    return (
        f"N={s.get('n')} sum={s.get('sum_bps')} med={s.get('median_bps')} "
        f"T3R={s.get('t3r_bps')} tail150={s.get('tail_lte_minus150_n')} maxLoss={s.get('max_loss_bps')}"
    )


def write_report(result: dict[str, Any]) -> None:
    lines = [
        "# S34 Navigation Scalp Tail Tests",
        "",
        f"Generated: `{result['generated_at_utc']}`",
        "",
        f"Status: `{result['status']}`",
        "",
        "## Stress Score>=3 Reverse TP/SL Grid",
        "",
        f"N: `{result['stress_score3_reverse_grid']['n']}`",
    ]
    for h, cell in result["stress_score3_reverse_grid"]["horizons"].items():
        lines.extend(["", f"### Horizon {h}", "", f"Fixed: {fmt(cell['fixed'])}", ""])
        lines.append("| TP | SL | Summary | Exits |")
        lines.append("| ---: | ---: | --- | --- |")
        for row in cell["top_brackets"][:8]:
            lines.append(f"| {row['tp']} | {row['sl']} | {fmt(row['summary'])} | `{row['exits']}` |")

    lines.extend(["", "## Tail Cleaner - Score>=3 Reverse", ""])
    lines.append("| Filter | N | Reverse 15m | Reverse 30m |")
    lines.append("| --- | ---: | --- | --- |")
    for name, row in result["tail_cleaner"].items():
        lines.append(f"| {name} | {row['n']} | {fmt(row['reverse_15m'])} | {fmt(row['reverse_30m'])} |")

    lines.extend(["", "## Stress Reverse Big Winner / Loser Anatomy", ""])
    for key in ("stress_score3_reverse_anatomy_15m", "stress_score3_reverse_anatomy_30m"):
        block = result[key]
        lines.extend(
            [
                f"### {key}",
                "",
                f"Summary: {fmt(block['summary'])}",
                f"Winner profile: `{block['winner_profile']}`",
                f"Loser profile: `{block['loser_profile']}`",
                f"Tail profile: `{block['tail_profile']}`",
                "",
                "Worst 5:",
            ]
        )
        for row in block["worst10"][:5]:
            lines.append(f"- `{row}`")
        lines.append("Best 5:")
        for row in block["best10"][:5]:
            lines.append(f"- `{row}`")

    lines.extend(["", "## v0.2 Stress Guard Big Winner / Loser Anatomy", ""])
    for name, block in result["v02_stress_guard_anatomy_2h"].items():
        lines.extend(
            [
                f"### {name}",
                "",
                f"Summary: {fmt(block['summary'])}",
                f"Winner profile: `{block['winner_profile']}`",
                f"Loser profile: `{block['loser_profile']}`",
                f"Tail profile: `{block['tail_profile']}`",
                "",
            ]
        )
        lines.append("Worst 5:")
        for row in block["worst10"][:5]:
            lines.append(f"- `{row}`")
        lines.append("Best 5:")
        for row in block["best10"][:5]:
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
