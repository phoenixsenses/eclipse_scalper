"""
S34 Exact Route-Change Validation
Runner-parity comparison for two route-change candidates.

Candidate 1: ETH_SELL_LIQ_SHORT_1M — TP80 (current) vs TP60/TP70/TP60_BE30
Candidate 2: SOL_SELL_LIQ_SHORT_100K — TP60 (current) vs TP40/TP40_SL40/TP50

Uses runner._bucket_events, _paper_trade_from_signal, _evaluate_trade directly.
No risk gates, no regime filter (rules have use_global_regime=False).
No runner/config/pre-reg changes.
"""
from __future__ import annotations
import json
import math
import sqlite3
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import s34_shadow_paper_runner as runner

from ami.storage import production as PR
from ami.storage import research_reader as RR

DB_PATH        = ROOT / "data" / "microstructure.db"
SOURCE_DB_PATH = DB_PATH.as_posix()
OUT_DIR   = ROOT / "reports" / "research" / "s34"
OUT_MD    = OUT_DIR / "S34_EXACT_ROUTE_CHANGE_VALIDATION.md"
OUT_JSON  = OUT_DIR / "S34_EXACT_ROUTE_CHANGE_VALIDATION.json"

PRELIMINARY_N = 30
NO_FILL_THRESHOLD = 0.40

# ── Candidate definitions ────────────────────────────────────────────────────

CANDIDATES = [
    {
        "name": "ETH_1M_SELL",
        "base_rule": runner.S34Rule(
            name="ETH_SELL_1M_BASE",
            symbol="ETHUSDT", liq_side="SELL", direction="SHORT",
            threshold_usd=1_000_000.0, sl_bps=40.0, be_trigger_bps=40.0,
            use_global_regime=False, priority=8,
        ),
        "variants": [
            {"label": "TP80/SL40/BE40 (current)", "tp": 80.0, "sl": 40.0, "be": 40.0, "is_current": True},
            {"label": "TP60/SL40/BE40",            "tp": 60.0, "sl": 40.0, "be": 40.0},
            {"label": "TP70/SL40/BE40",            "tp": 70.0, "sl": 40.0, "be": 40.0},
            {"label": "TP60/SL40/BE30",            "tp": 60.0, "sl": 40.0, "be": 30.0},
        ],
    },
    {
        "name": "SOL_100K_SELL",
        "base_rule": runner.S34Rule(
            name="SOL_SELL_100K_BASE",
            symbol="SOLUSDT", liq_side="SELL", direction="SHORT",
            threshold_usd=100_000.0, sl_bps=30.0, be_trigger_bps=40.0,
            use_global_regime=False, priority=15,
        ),
        "variants": [
            {"label": "TP60/SL30/BE40 (current)", "tp": 60.0, "sl": 30.0, "be": 40.0, "is_current": True},
            {"label": "TP40/SL30/BE40",            "tp": 40.0, "sl": 30.0, "be": 40.0},
            {"label": "TP40/SL40/BE40",            "tp": 40.0, "sl": 40.0, "be": 40.0},
            {"label": "TP50/SL30/BE40",            "tp": 50.0, "sl": 30.0, "be": 40.0},
        ],
    },
]


# ── Statistics helpers ────────────────────────────────────────────────────────

def _pctile(vals: list[float], q: float) -> float | None:
    c = sorted(v for v in vals if v is not None and math.isfinite(v))
    if not c: return None
    pos = (len(c) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return c[lo] if lo == hi else c[lo] + (c[hi] - c[lo]) * (pos - lo)

def _median(v): return _pctile(v, 0.5)
def _mean(v):
    c = [x for x in v if x is not None and math.isfinite(x)]
    return sum(c) / len(c) if c else None

def _top3_removed_cum(vals: list[float]) -> float:
    s = sorted(vals, reverse=True)
    trimmed = s[3:] if len(s) > 3 else []
    return sum(trimmed)

def _r1(v): return round(float(v), 1) if v is not None and math.isfinite(float(v)) else None
def _r3(v): return round(float(v), 3) if v is not None and math.isfinite(float(v)) else None


# ── MFE computation ───────────────────────────────────────────────────────────

def _mfe_marks_range(conn: sqlite3.Connection, symbol: str, start_ms: int, end_ms: int) -> list[tuple]:
    """Direct-SQL oracle -- kept unchanged as the parity reference for
    `_mfe_marks_range_v2` (BATCH-STORAGE-ROTATION-RETENTION-RANGE-READ-
    CONSUMER-MIGRATION-V6). No longer called by `_mfe_bps`; the
    reader-backed path is used instead."""
    return conn.execute(
        "SELECT mark_price FROM mark_prices WHERE symbol=? AND ts_ms>=? AND ts_ms<=? ORDER BY ts_ms ASC",
        (symbol, start_ms, end_ms),
    ).fetchall()


def _mfe_marks_range_v2(root, symbol: str, start_ms: int, end_ms: int, source_db_path=None) -> list[tuple]:
    """Reader-backed replacement for `_mfe_marks_range`, via `plan_read`/
    `execute_read`. `symbol` is a genuine runtime parameter (`trade["symbol"]`
    spans ETHUSDT/SOLUSDT across the two route-change candidates). Inclusive
    upper bound reproduced with `end_ms+1` (exact for integer ts_ms). Streams
    in canonical (ts_ms ASC, id ASC) order -- a refinement of the oracle's
    `ORDER BY ts_ms ASC` that yields an identical ts_ms sequence."""
    plan = RR.plan_read(root, table="mark_prices", symbol=symbol, start_ms=int(start_ms), end_ms=int(end_ms) + 1)
    result = RR.execute_read(plan, columns=("mark_price",), source_db_path=source_db_path)
    return [(row[0],) for row in result.iter_rows()]


def _mfe_bps(conn: sqlite3.Connection, trade: dict, root, source_db_path=None) -> float | None:
    entry_ts  = int(trade.get("entry_ts_ms") or 0)
    exit_ts   = int(trade.get("exit_ts_ms")  or 0)
    entry_ref = float(trade.get("entry_reference_price") or 0)
    if not entry_ts or not exit_ts or entry_ref <= 0:
        return None
    direction = str(trade.get("direction", "LONG")).upper()
    rows = _mfe_marks_range_v2(root, trade["symbol"], entry_ts, exit_ts, source_db_path=source_db_path)
    if not rows:
        return None
    prices = [float(r[0]) for r in rows]
    mfe = max(entry_ref - p for p in prices) if direction == "SHORT" else max(p - entry_ref for p in prices)
    return mfe / entry_ref * 10_000


# ── Metrics computation ───────────────────────────────────────────────────────

def _compute_metrics(conn: sqlite3.Connection, trades: list[dict], tp_bps: float, root, source_db_path=None) -> dict:
    closed = [t for t in trades if t.get("status") == "CLOSED" and t.get("net_bps") is not None]
    if not closed:
        return {"n_closed": 0, "insufficient_data": True}

    nets = [float(t["net_bps"]) for t in closed]
    wins = [n for n in nets if n > 0]
    wr   = len(wins) / len(nets) if nets else None

    # Exit mix
    em = {"TP": 0, "SL": 0, "BE": 0, "TIME": 0}
    for t in closed:
        r = str(t.get("exit_reason") or "")
        if r in em: em[r] += 1

    # Avg hold sec
    holds = []
    for t in closed:
        e, x = t.get("entry_ts_ms"), t.get("exit_ts_ms")
        if e and x: holds.append((int(x) - int(e)) / 1000)
    avg_hold = _mean(holds)

    # Positive days
    day_map: dict[str, list[float]] = {}
    for t in closed:
        day = str(t.get("signal_ts_utc") or "")[:10]
        day_map.setdefault(day, []).append(float(t["net_bps"]))
    pos_days   = sum(1 for v in day_map.values() if sum(v) > 0)
    total_days = len(day_map)

    # Temporal halves
    half = len(closed) // 2
    h1 = [float(t["net_bps"]) for t in closed[:half]]
    h2 = [float(t["net_bps"]) for t in closed[half:]]

    # Giveback: MFE >= 50% of TP but net < 0
    giveback_n = 0
    mfe_denominator = len([t for t in closed if float(t["net_bps"]) < 0])
    for t in closed:
        if float(t["net_bps"]) < 0:
            mfe = _mfe_bps(conn, t, root, source_db_path=source_db_path)
            if mfe is not None and mfe >= 0.5 * tp_bps:
                giveback_n += 1
    giveback_rate = giveback_n / len(closed) if closed else None

    return {
        "n_closed": len(closed),
        "median_net": _r1(_median(nets)),
        "mean_net":   _r1(_mean(nets)),
        "cum_net":    _r1(sum(nets)),
        "top3_removed_cum": _r1(_top3_removed_cum(nets)),
        "win_rate":   _r3(wr),
        "exit_mix":   em,
        "avg_hold_sec": _r1(avg_hold),
        "pos_days":   pos_days,
        "total_days": total_days,
        "h1_median":  _r1(_median(h1)),
        "h1_cum":     _r1(sum(h1)),
        "h2_median":  _r1(_median(h2)),
        "h2_cum":     _r1(sum(h2)),
        "giveback_rate": _r3(giveback_rate),
        "giveback_n":    giveback_n,
    }


# ── Verdict ───────────────────────────────────────────────────────────────────

def _verdict(cand_label: str, current: dict, alt: dict, n: int, no_fill_rate: float) -> str:
    if n < PRELIMINARY_N:
        return "INCONCLUSIVE_N_SMALL"
    if no_fill_rate > NO_FILL_THRESHOLD:
        return "INCONCLUSIVE_NO_FILL_HIGH"
    if current.get("insufficient_data") or alt.get("insufficient_data"):
        return "INCONCLUSIVE_NO_DATA"
    curr_med = current.get("median_net") or 0.0
    alt_med  = alt.get("median_net")    or 0.0
    curr_cum = current.get("cum_net")   or 0.0
    alt_cum  = alt.get("cum_net")       or 0.0
    curr_wr  = current.get("win_rate")  or 0.0
    alt_wr   = alt.get("win_rate")      or 0.0
    curr_gb  = current.get("giveback_rate") or 0.0
    alt_gb   = alt.get("giveback_rate")     or 0.0
    med_delta = alt_med - curr_med
    cum_delta = alt_cum - curr_cum
    wr_delta  = alt_wr  - curr_wr
    gb_delta  = alt_gb  - curr_gb  # negative = fewer givebacks = better
    # Alt is better if: higher median AND (higher cum or lower giveback)
    if alt_med > curr_med and cum_delta > 0 and alt_wr >= curr_wr - 0.05:
        return "SWITCH_RECOMMENDED"
    if alt_med > curr_med and gb_delta < -0.03:
        return "SWITCH_RECOMMENDED"
    if alt_med < curr_med - 5 and cum_delta < -50:
        return "KEEP_CURRENT"
    if abs(med_delta) < 3 and abs(cum_delta) < 30:
        return "INCONCLUSIVE_NO_MEANINGFUL_DIFFERENCE"
    if alt_med > curr_med:
        return "ALT_MARGINALLY_BETTER"
    return "KEEP_CURRENT"


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(f"file:{DB_PATH.as_posix()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    root, _ = PR.resolve_production_root()

    # OUT-OF-SCOPE for RANGE-READ V6: unbounded, no symbol filter (full-table
    # MAX scan across all symbols) -- not an execute_read/plan_read target.
    end_ms = conn.execute("SELECT MAX(ts_ms) FROM mark_prices").fetchone()[0]
    if not end_ms:
        print("ERROR: no mark prices in DB")
        return
    start_ms = 0

    all_results = []
    summary_rows = []

    for cand in CANDIDATES:
        base_rule = cand["base_rule"]
        variants  = cand["variants"]
        cname     = cand["name"]

        print(f"\n=== {cname} ===")
        print(f"Scanning signals: {base_rule.symbol} {base_rule.liq_side} threshold>={base_rule.threshold_usd:,.0f} ...")
        # OUT-OF-SCOPE for RANGE-READ V6: runner._bucket_events/_paper_trade_from_signal/
        # _evaluate_trade own their own direct-SQL internals in the untouched
        # s34_shadow_paper_runner module; not migrated here.
        signals = runner._bucket_events(conn, base_rule, start_ms, int(end_ms), 100_000)
        n_signals  = len(signals)
        n_no_fill  = sum(1 for s in signals if s.get("fill_error"))
        n_fillable = n_signals - n_no_fill
        no_fill_rate = n_no_fill / n_signals if n_signals else 0.0
        print(f"  signals={n_signals}  no_fill={n_no_fill} ({no_fill_rate:.0%})  fillable={n_fillable}")

        variant_results = []
        current_metrics = None

        for v in variants:
            vname  = v["label"]
            tp_bps = float(v["tp"])
            sl_bps = float(v["sl"])
            be_bps = float(v["be"])
            is_cur = bool(v.get("is_current", False))

            vrule = replace(
                base_rule,
                name=f"{cname}_{vname.replace('/','_').replace(' ','').replace('(current)','')}",
                tp_bps=tp_bps, sl_bps=sl_bps, be_trigger_bps=be_bps,
            )

            trades = []
            for sig in signals:
                if sig.get("fill_error"):
                    continue
                trade = runner._paper_trade_from_signal(vrule, sig, runner.DEFAULT_RISK)
                try:
                    trade = runner._evaluate_trade(conn, trade, int(end_ms))
                except RuntimeError as exc:
                    trade["status"] = "EXIT_FILL_ERROR"
                    trade["exit_fill_error"] = str(exc)
                trades.append(trade)

            n_closed   = sum(1 for t in trades if t.get("status") == "CLOSED")
            n_open_end = sum(1 for t in trades if t.get("status") == "OPEN")
            metrics    = _compute_metrics(conn, trades, tp_bps, root, source_db_path=SOURCE_DB_PATH)

            if is_cur:
                current_metrics = metrics

            wr_str = f"{metrics['win_rate']*100:.0f}%" if metrics.get('win_rate') is not None else "N/A"
            gb_str = f"{metrics['giveback_rate']*100:.0f}%" if metrics.get('giveback_rate') is not None else "N/A"
            print(f"  [{vname}] closed={n_closed} open_at_end={n_open_end} "
                  f"median={metrics.get('median_net')} cum={metrics.get('cum_net')} "
                  f"WR={wr_str} giveback={gb_str}")

            variant_results.append({
                "label": vname,
                "is_current": is_cur,
                "tp_bps": tp_bps, "sl_bps": sl_bps, "be_trigger_bps": be_bps,
                "n_signals": n_signals,
                "n_no_fill": n_no_fill,
                "n_fillable": n_fillable,
                "no_fill_rate": round(no_fill_rate, 3),
                "n_closed": n_closed,
                "n_open_at_end": n_open_end,
                **metrics,
            })

        # Build verdicts vs current for each non-current variant
        for vr in variant_results:
            if vr["is_current"] or current_metrics is None:
                vr["verdict"] = "CURRENT"
            else:
                vr["verdict"] = _verdict(
                    vr["label"], current_metrics, vr,
                    vr["n_closed"], vr["no_fill_rate"],
                )

        all_results.append({
            "candidate": cname,
            "symbol": base_rule.symbol,
            "liq_side": base_rule.liq_side,
            "threshold_usd": base_rule.threshold_usd,
            "n_signals": n_signals,
            "n_no_fill": n_no_fill,
            "no_fill_rate": round(no_fill_rate, 3),
            "variants": variant_results,
        })

        # Find best alternative for summary table
        alts = [vr for vr in variant_results if not vr["is_current"]]
        cur  = next((vr for vr in variant_results if vr["is_current"]), None)
        if cur and alts:
            best_alt = max(alts, key=lambda v: (v.get("median_net") or -999))
            summary_rows.append({
                "candidate": cname,
                "current_label":  cur["label"],
                "alt_label":      best_alt["label"],
                "n": cur["n_closed"],
                "current_median": cur.get("median_net"),
                "alt_median":     best_alt.get("median_net"),
                "current_cum":    cur.get("cum_net"),
                "alt_cum":        best_alt.get("cum_net"),
                "current_gb":     cur.get("giveback_rate"),
                "alt_gb":         best_alt.get("giveback_rate"),
                "gb_delta":       _r3((best_alt.get("giveback_rate") or 0) - (cur.get("giveback_rate") or 0)),
                "verdict":        best_alt["verdict"],
            })

    conn.close()

    # ── Write JSON ────────────────────────────────────────────────────────────
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "methodology": "runner-parity via direct _bucket_events/_paper_trade_from_signal/_evaluate_trade import",
        "note": "No risk gates applied. All fillable signals evaluated independently per variant.",
        "summary": summary_rows,
        "candidates": all_results,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    # ── Write MD ─────────────────────────────────────────────────────────────
    lines = _build_md(payload)
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"\nMD  : {OUT_MD}")
    print(f"JSON: {OUT_JSON}")


def _exit_str(em: dict) -> str:
    parts = []
    for k in ("TP", "BE", "SL", "TIME"):
        if em.get(k): parts.append(f"{k}={em[k]}")
    return " ".join(parts) if parts else "-"

def _pct(v) -> str:
    return f"{v*100:.0f}%" if v is not None else "-"

def _bps(v) -> str:
    if v is None: return "-"
    return f"{float(v):+.1f}"

def _build_md(payload: dict) -> list[str]:
    lines = [
        "# S34 Exact Route-Change Validation",
        "",
        f"Generated: `{payload['generated_at_utc']}`",
        "",
        f"**Methodology**: {payload['methodology']}",
        "",
        f"**Note**: {payload['note']}",
        "",
        "---",
        "",
        "## Summary Table",
        "",
        "| Candidate | Current | Alternative | Exact N | Current median | Alt median | Current cum | Alt cum | Giveback delta | Verdict |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in payload["summary"]:
        gb_d = row.get("gb_delta")
        gb_str = f"{gb_d*100:+.0f} pp" if gb_d is not None else "-"
        lines.append(
            f"| {row['candidate']} | {row['current_label']} | {row['alt_label']} "
            f"| {row['n']} | {_bps(row['current_median'])} | {_bps(row['alt_median'])} "
            f"| {_bps(row['current_cum'])} | {_bps(row['alt_cum'])} "
            f"| {gb_str} | **{row['verdict']}** |"
        )
    lines += ["", "---", ""]

    for cand in payload["candidates"]:
        lines += [
            f"## {cand['candidate']} — {cand['symbol']} {cand['liq_side']} >= ${cand['threshold_usd']:,.0f}",
            "",
            f"Signals: {cand['n_signals']}  |  No-fill: {cand['n_no_fill']} ({cand['no_fill_rate']*100:.0f}%)  |  Fillable: {cand['n_signals']-cand['n_no_fill']}",
            "",
            "| Variant | N | Median | Mean | Cum | Top3-rem | WR | Exit mix | Hold sec | GivebackN | Giveback% | H1 med | H2 med | Verdict |",
            "|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---|",
        ]
        for v in cand["variants"]:
            gb_r = v.get("giveback_rate")
            gb_str = _pct(gb_r)
            em = v.get("exit_mix") or {}
            preliminary = " ⚠" if v["n_closed"] < PRELIMINARY_N else ""
            verdict_str = v.get("verdict", "-")
            lines.append(
                f"| {'**' if v['is_current'] else ''}{v['label']}{'**' if v['is_current'] else ''} "
                f"| {v['n_closed']}{preliminary} "
                f"| {_bps(v.get('median_net'))} | {_bps(v.get('mean_net'))} "
                f"| {_bps(v.get('cum_net'))} | {_bps(v.get('top3_removed_cum'))} "
                f"| {_pct(v.get('win_rate'))} "
                f"| {_exit_str(em)} "
                f"| {v.get('avg_hold_sec') or '-'} "
                f"| {v.get('giveback_n') or 0} | {gb_str} "
                f"| {_bps(v.get('h1_median'))} | {_bps(v.get('h2_median'))} "
                f"| {verdict_str} |"
            )
        lines += [
            "",
            f"*H1 = first 50% of closed trades chronologically. H2 = second 50%.*",
            f"*Giveback = MFE >= 50% of TP but net_bps < 0. ⚠ = N < {PRELIMINARY_N} (preliminary).*",
            "",
            "---",
            "",
        ]

    return lines


if __name__ == "__main__":
    main()
