"""S34 Order-Flow Lead.

The reactive liquidation family is exhausted because the move happens faster than
a cascade can be confirmed. The only remaining hope is a LEADING signal: knowable
order flow that moves *before* the liquidations. This tests whether aggressive
trade order-flow imbalance (OFI) predicts the next-seconds directional move, net
of cost -- i.e. an order-flow momentum bet, entered on flow, not on liquidations.

OFI is built from agg_trades: is_buyer_maker=0 means the taker was a buyer
(+notional), =1 means the taker was a seller (-notional). Rolling OFI over a
window W is fully knowable at the bin end. When |rolling OFI| exceeds an adaptive
high quantile, we bet in the OFI direction (net buying -> LONG) and measure the
forward mark return. Sweeps (W, quantile); every combo is split chronologically
into calibration/holdout. A combo is a lead only if net-positive AND win>50% on
BOTH splits. (If momentum is negative, the fade is just the sign flip.)

Bounded to a recent window (default 30 days) so the agg_trades scan stays cheap.
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.research_s34_knowable_anchor_continuation import (
    MarkIndex,
    mean,
    pctile,
    r1,
    r3,
    signed_return_bps,
)

from ami.storage import production as PR
from ami.storage import research_reader as RR

DEFAULT_DB = ROOT / "data" / "microstructure.db"
SOURCE_DB_PATH = (ROOT / "data" / "microstructure.db").as_posix()
OUT_DIR = ROOT / "reports" / "research" / "s34"
OUT_JSON = OUT_DIR / "S34_ORDERFLOW_LEAD.json"
OUT_MD = OUT_DIR / "S34_ORDERFLOW_LEAD.md"

SYMBOLS = ("ETHUSDT", "SOLUSDT", "BTCUSDT")
BIN_SEC = 5
HORIZONS_SEC = (10, 20, 30, 60)
SWEEP_W_SEC = (15, 30, 60)
SWEEP_Q = (0.95, 0.99)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def stat(vals: list[float], cost_bps: float) -> dict[str, Any]:
    vals = [v for v in vals if math.isfinite(v)]
    if not vals:
        return {"n": 0, "gross_median": None, "net_median": None, "win_rate": None, "gross_mean": None}
    med = pctile(vals, 0.5)
    return {
        "n": len(vals),
        "gross_median": r1(med),
        "net_median": r1(med - cost_bps),
        "gross_mean": r1(mean(vals)),
        "win_rate": r3(sum(1 for v in vals if v > 0) / len(vals)),
    }


def load_ofi_bins(conn, symbol, start_ms, end_ms) -> list[tuple[int, float]]:
    """Direct-SQL oracle -- kept unchanged as the parity reference for
    `load_ofi_bins_v2` (BATCH-STORAGE-ROTATION-RETENTION-RANGE-READ-
    CONSUMER-MIGRATION-V2). No longer called by main(); the reader-backed
    path is used instead.

    (bin_end_ts_ms, signed_notional) at BIN_SEC granularity, knowable at bin end."""
    bin_ms = BIN_SEC * 1000
    rows = conn.execute(
        f"""
        SELECT ts_ms/{bin_ms} AS b,
               SUM(CASE WHEN is_buyer_maker=0 THEN notional ELSE -notional END) AS ofi
        FROM agg_trades
        WHERE symbol=? AND ts_ms>=? AND ts_ms<=?
        GROUP BY ts_ms/{bin_ms}
        ORDER BY b
        """,
        (symbol, int(start_ms), int(end_ms)),
    ).fetchall()
    return [((int(b) + 1) * bin_ms, float(ofi or 0.0)) for b, ofi in rows]


def load_ofi_bins_v2(root, symbol, start_ms, end_ms, source_db_path=None) -> list[tuple[int, float]]:
    """Reader-backed replacement for `load_ofi_bins`, via
    `plan_read`/`execute_read`. Fetches raw (ts_ms, notional, is_buyer_maker)
    rows over the window and reproduces the oracle's `GROUP BY ts_ms/bin_ms`
    + signed-`SUM` aggregate in Python, bin-for-bin.

    `symbol` is a genuine runtime parameter (SYMBOLS: ETHUSDT/SOLUSDT/
    BTCUSDT). Range semantics: the oracle's SQL uses an INCLUSIVE upper
    bound (`ts_ms<=end_ms`); the reader uses half-open `[start, end)`, so
    `end_ms+1` is passed -- exact for integer ts_ms. Integer bin division
    (`ts_ms//bin_ms`) matches SQLite's integer `ts_ms/bin_ms` exactly. The
    oracle's `float(ofi or 0.0)` coalesces a NULL SUM to 0.0, but SQL's SUM
    is only NULL for an empty group, which GROUP BY never emits -- so every
    real bin has at least one row and a non-NULL sum; the Python side sums
    per bin identically."""
    bin_ms = BIN_SEC * 1000
    plan = RR.plan_read(root, table="agg_trades", symbol=symbol, start_ms=int(start_ms), end_ms=int(end_ms) + 1)
    result = RR.execute_read(plan, columns=("ts_ms", "notional", "is_buyer_maker"), source_db_path=source_db_path)
    bins: dict[int, float] = {}
    for ts_ms, notional, is_buyer_maker in result.iter_rows():
        b = int(ts_ms) // bin_ms
        signed = float(notional) if is_buyer_maker == 0 else -float(notional)
        bins[b] = bins.get(b, 0.0) + signed
    return [((b + 1) * bin_ms, bins[b]) for b in sorted(bins)]


def load_marks_range(conn, symbol, start_ms, end_ms) -> MarkIndex:
    """Direct-SQL oracle -- kept unchanged as the parity reference for
    `load_marks_range_v2` (same gate). No longer called by main()."""
    rows = conn.execute(
        "SELECT ts_ms, mark_price FROM mark_prices WHERE symbol=? AND ts_ms>=? AND ts_ms<=? ORDER BY ts_ms",
        (symbol, int(start_ms), int(end_ms)),
    ).fetchall()
    return MarkIndex(rows)


def load_marks_range_v2(root, symbol, start_ms, end_ms, source_db_path=None) -> MarkIndex:
    """Reader-backed replacement for `load_marks_range`, via
    `plan_read`/`execute_read`. Streams (ts_ms, mark_price) over the window
    in canonical `(ts_ms ASC, id ASC)` order -- matching the oracle's
    `ORDER BY ts_ms` -- and materializes into the same `MarkIndex`. Inclusive
    upper bound reproduced with `end_ms+1` (exact for integer ts_ms)."""
    plan = RR.plan_read(root, table="mark_prices", symbol=symbol, start_ms=int(start_ms), end_ms=int(end_ms) + 1)
    result = RR.execute_read(plan, columns=("ts_ms", "mark_price"), source_db_path=source_db_path)
    rows = [(int(ts_ms), float(mark_price)) for ts_ms, mark_price in result.iter_rows()]
    return MarkIndex(rows)


def rolling_ofi(bins: list[tuple[int, float]], w_sec: int) -> list[tuple[int, float]]:
    n = max(1, int(w_sec) // BIN_SEC)
    win: deque[float] = deque(maxlen=n)
    out = []
    for ts, ofi in bins:
        win.append(ofi)
        out.append((ts, sum(win)))
    return out


def eval_symbol(symbol, *, root, source_db_path=None, start_ms, end_ms, cost_bps_rt, holdout_frac, cooldown_sec, min_n) -> dict[str, Any]:
    bins = load_ofi_bins_v2(root, symbol, start_ms, end_ms, source_db_path=source_db_path)
    marks = load_marks_range_v2(root, symbol, start_ms - 120_000, end_ms + 120_000, source_db_path=source_db_path)
    combos = []
    for w in SWEEP_W_SEC:
        roll = rolling_ofi(bins, w)
        absvals = sorted(abs(v) for _, v in roll if v != 0.0)
        if len(absvals) < 100:
            continue
        for q in SWEEP_Q:
            cut = pctile(absvals, q)
            if cut is None or cut <= 0:
                continue
            triggers = []
            last = None
            cd_ms = int(cooldown_sec) * 1000
            for ts, v in roll:
                if abs(v) >= cut and (last is None or ts - last >= cd_ms):
                    triggers.append((ts, 1 if v > 0 else -1))
                    last = ts
            if len(triggers) < 2 * min_n:
                combos.append({"w_sec": w, "quantile": q, "trigger_n": len(triggers), "status": "THIN"})
                continue
            cut_ts = triggers[int(len(triggers) * (1.0 - holdout_frac))][0]
            cal_h: dict[int, list[float]] = {h: [] for h in HORIZONS_SEC}
            hold_h: dict[int, list[float]] = {h: [] for h in HORIZONS_SEC}
            for ts, sgn in triggers:
                entry = marks.at_or_after(ts)
                if not entry:
                    continue
                direction = "LONG" if sgn > 0 else "SHORT"
                target = cal_h if ts < cut_ts else hold_h
                for h in HORIZONS_SEC:
                    ex = marks.at_or_after(ts + h * 1000)
                    if not ex:
                        continue
                    target[h].append(signed_return_bps(direction, float(entry[1]), float(ex[1])))
            combos.append({
                "w_sec": w, "quantile": q, "trigger_n": len(triggers), "status": "SCREENED",
                "calibration": {str(h): stat(cal_h[h], cost_bps_rt) for h in HORIZONS_SEC},
                "holdout": {str(h): stat(hold_h[h], cost_bps_rt) for h in HORIZONS_SEC},
            })
    return {"symbol": symbol, "combos": combos}


def both_positive(combo, horizon, min_n) -> bool:
    if combo.get("status") != "SCREENED":
        return False
    c = combo["calibration"][str(horizon)]
    h = combo["holdout"][str(horizon)]
    return (
        c["n"] >= min_n and h["n"] >= min_n
        and (c["net_median"] or -1) > 0 and (h["net_median"] or -1) > 0
        and (c["win_rate"] or 0) > 0.5 and (h["win_rate"] or 0) > 0.5
    )


def render_md(report: dict[str, Any]) -> str:
    cfg = report["config"]
    lines = [
        "# S34 Order-Flow Lead (agg-trade OFI momentum)",
        "",
        f"Generated: `{report['generated_at_utc']}`  |  window `{cfg['days']}`d, bin `{BIN_SEC}`s, cost `{cfg['cost_bps_rt']}`bps, "
        f"cooldown `{cfg['cooldown_sec']}`s, holdout `{cfg['holdout_frac']}`, min_n `{cfg['min_n']}`",
        "",
        "Bet in the rolling-OFI direction when |OFI| exceeds the adaptive quantile. Net = gross median - round-trip cost. "
        "`**` = net-positive AND win>50% on BOTH splits at the 30s horizon. (Fade = sign flip of these numbers.)",
        "",
    ]
    leads = []
    for sym in report["symbols"]:
        lines.append(f"## {sym['symbol']}")
        lines.append("")
        lines.append("| W | q | Trig N | cal net@20 | cal net@30 | cal win@30 | hold net@20 | hold net@30 | hold win@30 | |")
        lines.append("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
        for c in sym["combos"]:
            if c.get("status") != "SCREENED":
                lines.append(f"| {c['w_sec']} | {c['quantile']} | {c['trigger_n']} | _thin_ | | | | | | |")
                continue
            c20, c30 = c["calibration"]["20"], c["calibration"]["30"]
            h20, h30 = c["holdout"]["20"], c["holdout"]["30"]
            flag = "**" if both_positive(c, 30, cfg["min_n"]) else ""
            if flag:
                leads.append((sym["symbol"], c))
            wr = lambda s: None if s["win_rate"] is None else r1(s["win_rate"] * 100.0)
            lines.append(
                f"| {c['w_sec']} | {c['quantile']} | {c['trigger_n']} | {c20['net_median']} | {c30['net_median']} | "
                f"{wr(c30)} | {h20['net_median']} | {h30['net_median']} | {wr(h30)} | {flag} |"
            )
        lines.append("")
    lines.append("## Leads (both-split positive @30s)")
    lines.extend(
        [f"- **{s} W={c['w_sec']} q={c['quantile']}**: cal net@30={c['calibration']['30']['net_median']}, "
         f"hold net@30={c['holdout']['30']['net_median']}" for s, c in leads] or ["- none"]
    )
    lines.append("")
    return "\n".join(lines)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Order-flow (agg-trade OFI) leading-signal momentum test, holdout-split.")
    p.add_argument("--db", type=Path, default=DEFAULT_DB)
    p.add_argument("--days", type=int, default=30)
    p.add_argument("--cost-bps-rt", type=float, default=6.1)
    p.add_argument("--cooldown-sec", type=int, default=60)
    p.add_argument("--holdout-frac", type=float, default=0.30)
    p.add_argument("--min-n", type=int, default=40)
    p.add_argument("--symbols", default=",".join(SYMBOLS))
    p.add_argument("--json-out", type=Path, default=OUT_JSON)
    p.add_argument("--md-out", type=Path, default=OUT_MD)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    syms = [s.strip() for s in str(args.symbols).split(",") if s.strip()]
    root, _ = PR.resolve_production_root()
    source_db_path = str(args.db)
    # `conn` stays only for the MAX(ts_ms) bootstrap bound; every windowed
    # range read moved to the reader (via source_db_path).
    with sqlite3.connect(f"file:{args.db}?mode=ro", uri=True) as conn:
        end_ms = conn.execute("SELECT MAX(ts_ms) FROM agg_trades WHERE symbol=?", (syms[0],)).fetchone()[0]
        if end_ms is None:
            print("no agg_trades", file=sys.stderr)
            return 1
        start_ms = int(end_ms) - int(args.days) * 86_400_000
        symbols = [
            eval_symbol(sym, root=root, source_db_path=source_db_path, start_ms=start_ms, end_ms=int(end_ms),
                        cost_bps_rt=float(args.cost_bps_rt), holdout_frac=float(args.holdout_frac),
                        cooldown_sec=int(args.cooldown_sec), min_n=int(args.min_n))
            for sym in syms
        ]
    report = {
        "generated_at_utc": utc_now(),
        "config": {"days": int(args.days), "cost_bps_rt": float(args.cost_bps_rt), "cooldown_sec": int(args.cooldown_sec),
                   "holdout_frac": float(args.holdout_frac), "min_n": int(args.min_n),
                   "start_ms": start_ms, "end_ms": int(end_ms)},
        "symbols": symbols,
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    args.md_out.write_text(render_md(report), encoding="utf-8")
    print(render_md(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
