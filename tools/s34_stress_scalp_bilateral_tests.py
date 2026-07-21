"""S34 stress-scalp bilateral / two-phase tests.

Research-only follow-up for the observation that the causal SHORT/reverse state
is consistently negative. Tests whether that negative result contains usable
opposite-side information or a later exhaustion/flip phase.

No live executor, order logic, size, leverage, config, or env changes.
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
from tools.s34_stress_reaction_deep_tests import BASE_FEE_BPS, mark_series  # noqa: E402
from tools.s34_stress_scalp_live_readiness_tests import SELECTORS, build_live_like_rows, ts  # noqa: E402

OUT_JSON = ROOT / "reports" / "research" / "s34" / "S34_STRESS_SCALP_BILATERAL_TESTS.json"
OUT_MD = ROOT / "reports" / "research" / "s34" / "S34_STRESS_SCALP_BILATERAL_TESTS.md"

PRIMARY = "live_like_causal3"
COMPARATORS = ("original_holdstate_near3", "live_like_near3", "live_like_causal3", "live_like_causal2")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def t3r(vals: list[float]) -> float:
    vals = [float(v) for v in vals if math.isfinite(float(v))]
    return float(sum(sorted(vals, reverse=True)[3:])) if len(vals) > 3 else float(sum(vals))


def bracket_from_entry(
    conn: sqlite3.Connection,
    *,
    entry_ts_ms: int,
    entry_px: float,
    horizon_sec: int,
    direction: str,
    tp: float,
    sl: float,
    fee_bps: float = BASE_FEE_BPS,
) -> tuple[float | None, str, int | None]:
    series = mark_series(conn, entry_ts_ms, entry_ts_ms + horizon_sec * 1000)
    if not series or entry_px <= 0:
        return None, "NO_SERIES", None
    for t, px in series:
        raw = (float(px) - entry_px) / entry_px * 10_000.0
        pnl = raw if direction == "LONG" else -raw
        if pnl >= tp:
            return tp - fee_bps, "TP", int((int(t) - entry_ts_ms) / 1000)
        if pnl <= -sl:
            return -sl - fee_bps, "SL", int((int(t) - entry_ts_ms) / 1000)
    end_ts, end_px = series[-1]
    raw = (float(end_px) - entry_px) / entry_px * 10_000.0
    pnl = raw if direction == "LONG" else -raw
    return pnl - fee_bps, "TIME", int((int(end_ts) - entry_ts_ms) / 1000)


def fixed_from_entry(
    conn: sqlite3.Connection,
    *,
    entry_ts_ms: int,
    entry_px: float,
    horizon_sec: int,
    direction: str,
    fee_bps: float = BASE_FEE_BPS,
) -> float | None:
    exit_ = mark_at_or_after(conn, "ETHUSDT", entry_ts_ms + horizon_sec * 1000)
    if not exit_ or entry_px <= 0:
        return None
    raw = (float(exit_[1]) - entry_px) / entry_px * 10_000.0
    pnl = raw if direction == "LONG" else -raw
    return pnl - fee_bps


def entry_at(conn: sqlite3.Connection, row: dict[str, Any], delay_sec: int = 0) -> tuple[int, float] | None:
    entry = mark_at_or_after(conn, "ETHUSDT", ts(row) + int(delay_sec) * 1000)
    if not entry:
        return None
    return int(entry[0]), float(entry[1])


def eval_bracket(
    conn: sqlite3.Connection,
    rows: list[dict[str, Any]],
    selector: Callable[[dict[str, Any]], bool],
    *,
    direction: str,
    tp: float,
    sl: float,
    horizon_sec: int,
    delay_sec: int = 0,
) -> dict[str, Any]:
    vals = []
    exits: dict[str, int] = defaultdict(int)
    hit_secs = []
    for row in rows:
        if not selector(row):
            continue
        ent = entry_at(conn, row, delay_sec)
        if not ent:
            continue
        val, exit_, hit = bracket_from_entry(
            conn,
            entry_ts_ms=ent[0],
            entry_px=ent[1],
            horizon_sec=horizon_sec,
            direction=direction,
            tp=tp,
            sl=sl,
        )
        if val is None:
            continue
        vals.append(float(val))
        exits[str(exit_)] += 1
        if hit is not None:
            hit_secs.append(hit)
    return {
        "matched_n": len([r for r in rows if selector(r)]),
        "summary": summary(vals),
        "exits": dict(exits),
        "median_exit_sec": r1(median(hit_secs)) if hit_secs else None,
    }


def eval_fixed(
    conn: sqlite3.Connection,
    rows: list[dict[str, Any]],
    selector: Callable[[dict[str, Any]], bool],
    *,
    direction: str,
    horizon_sec: int,
    delay_sec: int = 0,
) -> dict[str, Any]:
    vals = []
    for row in rows:
        if not selector(row):
            continue
        ent = entry_at(conn, row, delay_sec)
        if not ent:
            continue
        val = fixed_from_entry(conn, entry_ts_ms=ent[0], entry_px=ent[1], horizon_sec=horizon_sec, direction=direction)
        if val is not None:
            vals.append(float(val))
    return {"matched_n": len([r for r in rows if selector(r)]), "summary": summary(vals)}


def mirror_long_tests(conn: sqlite3.Connection, rows: list[dict[str, Any]]) -> dict[str, Any]:
    out = {}
    for selector_name in COMPARATORS:
        sel = SELECTORS[selector_name]
        cells = {}
        for sec in (300, 900, 1200, 1800):
            cells[f"fixed_{int(sec/60)}m_LONG"] = eval_fixed(conn, rows, sel, direction="LONG", horizon_sec=sec)
            cells[f"fixed_{int(sec/60)}m_SHORT"] = eval_fixed(conn, rows, sel, direction="SHORT", horizon_sec=sec)
        for name, tp, sl, sec in (
            ("LONG_TP40_SL200_20M", 40.0, 200.0, 1200),
            ("LONG_TP80_SL80_20M", 80.0, 80.0, 1200),
            ("LONG_TP200_SL40_20M", 200.0, 40.0, 1200),
            ("LONG_TP150_SL30_15M", 150.0, 30.0, 900),
            ("SHORT_TP200_SL40_20M", 200.0, 40.0, 1200),
        ):
            cells[name] = eval_bracket(conn, rows, sel, direction="LONG" if name.startswith("LONG") else "SHORT", tp=tp, sl=sl, horizon_sec=sec)
        out[selector_name] = cells
    return out


def two_phase_tests(conn: sqlite3.Connection, rows: list[dict[str, Any]]) -> dict[str, Any]:
    sel = SELECTORS[PRIMARY]
    out = {}
    # Phase A = causal signal now. Phase B = delayed flip to SHORT after the
    # chain had time to complete.
    for delay in (60, 180, 300, 600, 900):
        out[f"phaseA_LONG_20m_then_phaseB_SHORT_delay{delay}s"] = {
            "phaseA_LONG_TP80_SL80_20M": eval_bracket(conn, rows, sel, direction="LONG", tp=80.0, sl=80.0, horizon_sec=1200),
            "phaseB_SHORT_TP200_SL40_20M": eval_bracket(conn, rows, sel, direction="SHORT", tp=200.0, sl=40.0, horizon_sec=1200, delay_sec=delay),
            "phaseB_SHORT_fixed15m": eval_fixed(conn, rows, sel, direction="SHORT", horizon_sec=900, delay_sec=delay),
            "phaseB_LONG_fixed15m": eval_fixed(conn, rows, sel, direction="LONG", horizon_sec=900, delay_sec=delay),
        }
    return out


def reverse_failure_anatomy(conn: sqlite3.Connection, rows: list[dict[str, Any]]) -> dict[str, Any]:
    sel = SELECTORS[PRIMARY]
    records = []
    for row in rows:
        if not sel(row):
            continue
        ent = entry_at(conn, row)
        if not ent:
            continue
        entry_ts, entry_px = ent
        series = mark_series(conn, entry_ts, entry_ts + 1200 * 1000)
        if not series:
            continue
        short_path = []
        long_path = []
        for t, px in series:
            raw = (float(px) - entry_px) / entry_px * 10_000.0
            short_path.append((int((int(t) - entry_ts) / 1000), -raw))
            long_path.append((int((int(t) - entry_ts) / 1000), raw))
        short_mfe_t, short_mfe = max(short_path, key=lambda x: x[1])
        short_mae_t, short_mae = min(short_path, key=lambda x: x[1])
        long_mfe_t, long_mfe = max(long_path, key=lambda x: x[1])
        long_mae_t, long_mae = min(long_path, key=lambda x: x[1])
        short_sl40 = next((t for t, pnl in short_path if pnl <= -40.0), None)
        long_tp40 = next((t for t, pnl in long_path if pnl >= 40.0), None)
        final_short = short_path[-1][1] - BASE_FEE_BPS
        records.append(
            {
                "row": row,
                "short_final": final_short,
                "short_mfe": short_mfe,
                "short_mfe_sec": short_mfe_t,
                "short_mae": short_mae,
                "short_mae_sec": short_mae_t,
                "short_sl40_sec": short_sl40,
                "long_mfe": long_mfe,
                "long_mfe_sec": long_mfe_t,
                "long_mae": long_mae,
                "long_mae_sec": long_mae_t,
                "long_tp40_sec": long_tp40,
            }
        )

    def vals(key: str, subset: list[dict[str, Any]]) -> list[float]:
        return [float(r[key]) for r in subset if r.get(key) is not None and math.isfinite(float(r[key]))]

    short_losers = [r for r in records if r["short_final"] <= -40.0]
    short_winners = [r for r in records if r["short_final"] > 0.0]
    def rec_summary(subset: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "n": len(subset),
            "short_mfe_med": r1(median(vals("short_mfe", subset))) if vals("short_mfe", subset) else None,
            "short_mae_med": r1(median(vals("short_mae", subset))) if vals("short_mae", subset) else None,
            "short_mfe_sec_med": r1(median(vals("short_mfe_sec", subset))) if vals("short_mfe_sec", subset) else None,
            "short_mae_sec_med": r1(median(vals("short_mae_sec", subset))) if vals("short_mae_sec", subset) else None,
            "short_sl40_hit_n": len(vals("short_sl40_sec", subset)),
            "short_sl40_sec_med": r1(median(vals("short_sl40_sec", subset))) if vals("short_sl40_sec", subset) else None,
            "long_mfe_med": r1(median(vals("long_mfe", subset))) if vals("long_mfe", subset) else None,
            "long_tp40_hit_n": len(vals("long_tp40_sec", subset)),
            "long_tp40_sec_med": r1(median(vals("long_tp40_sec", subset))) if vals("long_tp40_sec", subset) else None,
        }
    return {
        "all": rec_summary(records),
        "short_losers": rec_summary(short_losers),
        "short_winners": rec_summary(short_winners),
        "short_final": summary(vals("short_final", records)),
    }


def conflict_inversion(conn: sqlite3.Connection, rows: list[dict[str, Any]]) -> dict[str, Any]:
    sel = SELECTORS[PRIMARY]
    conflicts = []
    v02_rows = [r for r in rows if route_v02_local(r)]
    for row in rows:
        if not sel(row):
            continue
        if any(abs(ts(row) - ts(v)) <= 900_000 for v in v02_rows):
            conflicts.append(row)
    return {
        "conflict_n": len(conflicts),
        "SHORT_TP200_SL40_20M": eval_bracket(conn, conflicts, lambda r: True, direction="SHORT", tp=200, sl=40, horizon_sec=1200),
        "LONG_TP200_SL40_20M": eval_bracket(conn, conflicts, lambda r: True, direction="LONG", tp=200, sl=40, horizon_sec=1200),
        "LONG_fixed2h": eval_fixed(conn, conflicts, lambda r: True, direction="LONG", horizon_sec=7200),
        "SHORT_fixed20m": eval_fixed(conn, conflicts, lambda r: True, direction="SHORT", horizon_sec=1200),
    }


def route_v02_local(row: dict[str, Any]) -> bool:
    return (
        str(row.get("symbol")) == "ETHUSDT"
        and str(row.get("liq_side")) == "SELL"
        and int(float(row.get("threshold_usd") or 0)) == 200_000
        and 28.0 <= float(row.get("vdepth_bps") or 0.0) < 40.0
        and float(row.get("prior4h_bps") or 0.0) < -50.0
    )


def chain_liq_features(conn: sqlite3.Connection, row: dict[str, Any], window_sec: int = 900) -> dict[str, Any]:
    t = ts(row)
    start = t - window_sec * 1000
    end = t
    q = """
        SELECT symbol, side, COUNT(*), COALESCE(SUM(notional),0.0)
        FROM liquidations
        WHERE ts_ms>=? AND ts_ms<=?
          AND symbol IN ('ETHUSDT','BTCUSDT','SOLUSDT')
        GROUP BY symbol, side
    """
    rows = conn.execute(q, (start, end)).fetchall()
    out: dict[str, Any] = {"sell_notional": 0.0, "buy_notional": 0.0, "eth_sell_notional": 0.0, "eth_buy_notional": 0.0, "symbols": set()}
    for sym, side, count, notion in rows:
        side_u = str(side).upper()
        notion_f = float(notion or 0.0)
        out["symbols"].add(str(sym))
        if side_u == "SELL":
            out["sell_notional"] += notion_f
            if sym == "ETHUSDT":
                out["eth_sell_notional"] += notion_f
        elif side_u == "BUY":
            out["buy_notional"] += notion_f
            if sym == "ETHUSDT":
                out["eth_buy_notional"] += notion_f
    sell = float(out["sell_notional"])
    buy = float(out["buy_notional"])
    out["symbols_n"] = len(out["symbols"])
    out["symbols"] = sorted(out["symbols"])
    out["sell_share"] = sell / (sell + buy) if sell + buy > 0 else None
    if sell > 0 and buy / max(sell, 1.0) >= 0.35:
        out["chain_direction"] = "COUNTER_BUY_PRESENT"
    elif sell / max(buy, 1.0) >= 3.0:
        out["chain_direction"] = "SELL_DOMINANT"
    elif buy > sell:
        out["chain_direction"] = "BUY_DOMINANT"
    else:
        out["chain_direction"] = "MIXED"
    return out


def chain_direction_tests(conn: sqlite3.Connection, rows: list[dict[str, Any]]) -> dict[str, Any]:
    enriched = []
    for row in rows:
        if not SELECTORS[PRIMARY](row):
            continue
        item = dict(row)
        item.update(chain_liq_features(conn, row))
        enriched.append(item)
    out = {}
    for label in sorted({str(r.get("chain_direction")) for r in enriched}):
        subset = [r for r in enriched if r.get("chain_direction") == label]
        out[label] = {
            "n": len(subset),
            "LONG_TP80_SL80_20M": eval_bracket(conn, subset, lambda r: True, direction="LONG", tp=80, sl=80, horizon_sec=1200),
            "SHORT_TP200_SL40_20M": eval_bracket(conn, subset, lambda r: True, direction="SHORT", tp=200, sl=40, horizon_sec=1200),
            "LONG_fixed15m": eval_fixed(conn, subset, lambda r: True, direction="LONG", horizon_sec=900),
            "SHORT_fixed15m": eval_fixed(conn, subset, lambda r: True, direction="SHORT", horizon_sec=900),
            "symbols_avg": r1(sum(int(r.get("symbols_n") or 0) for r in subset) / len(subset)) if subset else None,
            "sell_share_avg": r3(sum(float(r.get("sell_share") or 0.0) for r in subset) / len(subset)) if subset else None,
        }
    return out


def causal_vs_near_spread(conn: sqlite3.Connection, rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups = {
        "causal3_only": lambda r: SELECTORS["live_like_causal3"](r) and not SELECTORS["original_holdstate_near3"](r),
        "near3_only": lambda r: SELECTORS["original_holdstate_near3"](r) and not SELECTORS["live_like_causal3"](r),
        "both_causal3_and_near3": lambda r: SELECTORS["live_like_causal3"](r) and SELECTORS["original_holdstate_near3"](r),
    }
    out = {}
    for name, sel in groups.items():
        out[name] = {
            "SHORT_TP200_SL40_20M": eval_bracket(conn, rows, sel, direction="SHORT", tp=200, sl=40, horizon_sec=1200),
            "LONG_TP80_SL80_20M": eval_bracket(conn, rows, sel, direction="LONG", tp=80, sl=80, horizon_sec=1200),
            "LONG_fixed15m": eval_fixed(conn, rows, sel, direction="LONG", horizon_sec=900),
            "SHORT_fixed15m": eval_fixed(conn, rows, sel, direction="SHORT", horizon_sec=900),
        }
    return out


def run() -> dict[str, Any]:
    rows = build_live_like_rows()
    with sqlite3.connect(DEFAULT_DB) as conn:
        return {
            "generated_at_utc": utc_now(),
            "status": "RESEARCH_ONLY_NO_LIVE_CHANGE",
            "primary_state": PRIMARY,
            "mirror_long_tests": mirror_long_tests(conn, rows),
            "two_phase_tests": two_phase_tests(conn, rows),
            "reverse_failure_anatomy": reverse_failure_anatomy(conn, rows),
            "conflict_inversion": conflict_inversion(conn, rows),
            "causal_vs_near_spread": causal_vs_near_spread(conn, rows),
            "chain_direction_tests": chain_direction_tests(conn, rows),
        }


def fmt(s: dict[str, Any]) -> str:
    return (
        f"N={s.get('n')} sum={s.get('sum_bps')} med={s.get('median_bps')} "
        f"T3R={s.get('t3r_bps')} tail150={s.get('tail_lte_minus150_n')} maxLoss={s.get('max_loss_bps')}"
    )


def write_report(result: dict[str, Any]) -> None:
    lines = [
        "# S34 Stress Scalp Bilateral Tests",
        "",
        f"Generated: `{result['generated_at_utc']}`",
        "",
        f"Status: `{result['status']}`",
        "",
        f"Primary state: `{result['primary_state']}`",
        "",
        "## 1. Exact Mirror LONG / SHORT",
        "",
    ]
    for selector, cells in result["mirror_long_tests"].items():
        lines.append(f"### `{selector}`")
        lines.append("")
        lines.append("| Test | Summary | Exits |")
        lines.append("| --- | --- | --- |")
        for name, row in cells.items():
            lines.append(f"| `{name}` | {fmt(row['summary'])} | `{row.get('exits', {})}` |")
        lines.append("")

    lines.extend(["", "## 2. Two-Phase Causal LONG -> Later SHORT", ""])
    for name, block in result["two_phase_tests"].items():
        lines.append(f"### `{name}`")
        for sub, row in block.items():
            lines.append(f"- `{sub}`: {fmt(row['summary'])}; exits `{row.get('exits', {})}`")

    lines.extend(["", "## 3. Reverse Failure Anatomy", ""])
    for name, row in result["reverse_failure_anatomy"].items():
        lines.append(f"- `{name}`: `{row}`")

    lines.extend(["", "## 4. Conflict Inversion", ""])
    ci = result["conflict_inversion"]
    lines.append(f"Conflict N: `{ci['conflict_n']}`")
    for name, row in ci.items():
        if isinstance(row, dict):
            lines.append(f"- `{name}`: {fmt(row['summary'])}; exits `{row.get('exits', {})}`")

    lines.extend(["", "## 5. Causal vs Near Spread", ""])
    for name, block in result["causal_vs_near_spread"].items():
        lines.append(f"### `{name}`")
        for sub, row in block.items():
            lines.append(f"- `{sub}`: {fmt(row['summary'])}; exits `{row.get('exits', {})}`")

    lines.extend(["", "## 6. Chain Direction", ""])
    for name, block in result["chain_direction_tests"].items():
        lines.append(f"### `{name}`")
        lines.append(f"- N `{block['n']}`, symbols_avg `{block['symbols_avg']}`, sell_share_avg `{block['sell_share_avg']}`")
        for sub, row in block.items():
            if isinstance(row, dict):
                lines.append(f"- `{sub}`: {fmt(row['summary'])}; exits `{row.get('exits', {})}`")

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
