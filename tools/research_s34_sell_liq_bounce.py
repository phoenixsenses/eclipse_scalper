"""
S34 Sell-Liq Bounce Research
Runner-parity simulation: LONG entry triggered by SELL liquidation cascade.

Hypothesis: after a SELL liq cascade threshold is crossed, price bounces
back +28 bps (median, 600s window) 61% of the time. Can a LONG entry
at threshold cross capture this reversal?

Sweeps ETH thresholds (250K / 500K / 1M) x TP/SL combos.
All temporal OOS (full-period with H1/H2 split reported).
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

DB_PATH  = ROOT / "data" / "microstructure.db"
SOURCE_DB_PATH = (ROOT / "data" / "microstructure.db").as_posix()
OUT_DIR  = ROOT / "reports" / "research" / "s34"
OUT_MD   = OUT_DIR / "S34_SELL_LIQ_BOUNCE.md"
OUT_JSON = OUT_DIR / "S34_SELL_LIQ_BOUNCE.json"

PRELIMINARY_N     = 30
NO_FILL_THRESHOLD = 0.40

# ── Sweep definitions ─────────────────────────────────────────────────────────

# (threshold_usd, label)
THRESHOLDS = [
    (250_000.0,  "ETH_SELL_250K"),
    (500_000.0,  "ETH_SELL_500K"),
    (1_000_000.0,"ETH_SELL_1M"),
]

# (tp, sl, be)
VARIANTS = [
    (20.0, 20.0, 15.0),
    (25.0, 25.0, 15.0),
    (30.0, 30.0, 15.0),
    (30.0, 40.0, 20.0),
    (40.0, 30.0, 20.0),
    (40.0, 40.0, 20.0),
    (60.0, 40.0, 30.0),
]


# ── Stat helpers ──────────────────────────────────────────────────────────────

def _pctile(vals: list[float], q: float) -> float | None:
    c = sorted(v for v in vals if v is not None and math.isfinite(v))
    if not c:
        return None
    pos = (len(c) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return c[lo] if lo == hi else c[lo] + (c[hi] - c[lo]) * (pos - lo)

def _median(v): return _pctile(v, 0.5)
def _mean(v):
    c = [x for x in v if x is not None and math.isfinite(x)]
    return sum(c) / len(c) if c else None

def _top3_removed_cum(vals: list[float]) -> float:
    s = sorted(vals, reverse=True)
    return sum(s[3:]) if len(s) > 3 else sum(s)

def _r1(v): return round(float(v), 1) if v is not None and math.isfinite(float(v)) else None


# ── MFE ──────────────────────────────────────────────────────────────────────

def _mfe_marks_range(conn: sqlite3.Connection, symbol: str, start_ms: int, end_ms: int) -> list[tuple]:
    """Direct-SQL oracle -- kept unchanged as the parity reference for
    `_mfe_marks_range_v2` (BATCH-STORAGE-ROTATION-RETENTION-RANGE-READ-
    CONSUMER-MIGRATION-V5). No longer called by `_mfe_bps`; the reader-
    backed path is used instead."""
    return conn.execute(
        "SELECT mark_price FROM mark_prices WHERE symbol=? AND ts_ms>=? AND ts_ms<=? ORDER BY ts_ms",
        (symbol, start_ms, end_ms),
    ).fetchall()


def _mfe_marks_range_v2(root, symbol: str, start_ms: int, end_ms: int, source_db_path=None) -> list[tuple]:
    """Reader-backed replacement for `_mfe_marks_range`, via `plan_read`/
    `execute_read`. `symbol` is a genuine runtime parameter (`trade["symbol"]`;
    always ETHUSDT in this file's actual call site, per `_run_config`'s
    hardcoded `symbol="ETHUSDT"` S34Rule, but the helper itself stays
    symbol-generic). Inclusive upper bound reproduced with `end_ms+1`
    (exact for integer ts_ms). Ordering does not affect the oracle's
    result (only max() over the value set matters), so the reader's
    canonical `(ts_ms ASC, id ASC)` ordering is a superset guarantee, not
    a behavior change."""
    plan = RR.plan_read(root, table="mark_prices", symbol=symbol, start_ms=int(start_ms), end_ms=int(end_ms) + 1)
    result = RR.execute_read(plan, columns=("mark_price",), source_db_path=source_db_path)
    return list(result.iter_rows())


def _mfe_bps(conn: sqlite3.Connection, trade: dict, *, root, source_db_path=None) -> float | None:
    entry_ts  = int(trade.get("entry_ts_ms") or 0)
    exit_ts   = int(trade.get("exit_ts_ms")  or 0)
    entry_ref = float(trade.get("entry_reference_price") or 0)
    if not entry_ts or not exit_ts or entry_ref <= 0:
        return None
    rows = _mfe_marks_range_v2(root, trade["symbol"], entry_ts, exit_ts, source_db_path=source_db_path)
    if not rows:
        return None
    prices = [float(r[0]) for r in rows]
    mfe = max(p - entry_ref for p in prices)   # LONG: favorable = price rises
    return mfe / entry_ref * 10_000


# ── Metrics ───────────────────────────────────────────────────────────────────

def _compute_metrics(conn: sqlite3.Connection, trades: list[dict], tp_bps: float, *, root, source_db_path=None) -> dict:
    closed = [t for t in trades if t.get("status") == "CLOSED" and t.get("net_bps") is not None]
    if not closed:
        return {"n_closed": 0, "insufficient_data": True}

    nets  = [float(t["net_bps"]) for t in closed]
    wins  = [n for n in nets if n > 0]
    wr    = len(wins) / len(nets) if nets else None

    em = {"TP": 0, "SL": 0, "BE": 0, "TIME": 0}
    for t in closed:
        r = str(t.get("exit_reason") or "")
        if r in em:
            em[r] += 1

    holds = []
    for t in closed:
        e, x = t.get("entry_ts_ms"), t.get("exit_ts_ms")
        if e and x:
            holds.append((int(x) - int(e)) / 1000)
    avg_hold = _mean(holds)

    day_map: dict[str, list[float]] = {}
    for t in closed:
        day = str(t.get("signal_ts_utc") or "")[:10]
        day_map.setdefault(day, []).append(float(t["net_bps"]))
    pos_days   = sum(1 for v in day_map.values() if sum(v) > 0)
    total_days = len(day_map)

    half = len(closed) // 2
    h1 = [float(t["net_bps"]) for t in closed[:half]]
    h2 = [float(t["net_bps"]) for t in closed[half:]]

    giveback_n = 0
    for t in closed:
        if float(t["net_bps"]) < 0:
            mfe = _mfe_bps(conn, t, root=root, source_db_path=source_db_path)
            if mfe is not None and mfe >= 0.5 * tp_bps:
                giveback_n += 1

    return {
        "n_closed":          len(closed),
        "median_net":        _r1(_median(nets)),
        "mean_net":          _r1(_mean(nets)),
        "cum_net":           _r1(sum(nets)),
        "top3_removed_cum":  _r1(_top3_removed_cum(nets)),
        "win_rate":          round(wr, 3) if wr is not None else None,
        "exit_mix":          em,
        "avg_hold_sec":      _r1(avg_hold),
        "pos_days":          pos_days,
        "total_days":        total_days,
        "h1_median":         _r1(_median(h1)),
        "h1_cum":            _r1(sum(h1)),
        "h2_median":         _r1(_median(h2)),
        "h2_cum":            _r1(sum(h2)),
        "giveback_n":        giveback_n,
        "giveback_rate":     round(giveback_n / len(closed), 3) if closed else None,
    }


# ── Simulate one config ───────────────────────────────────────────────────────

def _run_config(
    conn: sqlite3.Connection,
    threshold: float,
    tp: float, sl: float, be: float,
    *,
    root,
    source_db_path=None,
) -> dict:
    vrule = runner.S34Rule(
        name=f"ETH_BOUNCE_{int(threshold/1000)}K_TP{int(tp)}_SL{int(sl)}_BE{int(be)}",
        symbol="ETHUSDT",
        liq_side="SELL",
        direction="LONG",       # <-- bounce direction
        threshold_usd=threshold,
        tp_bps=tp,
        sl_bps=sl,
        be_trigger_bps=be,
        max_horizon_sec=1800,
        entry_delay_sec=0,
        min_gap_sec=900,
        use_global_regime=False,
        taker_fee_bps=4.0,
        require_book_ticker_fill=True,
    )

    # OUT-OF-SCOPE for RANGE-READ V5: an unbounded MIN/MAX scan over the
    # symbol's entire `liquidations` history (no ts_ms>=?/<=? window) --
    # not a bounded range read, and `liquidations` is out-of-allowlist
    # regardless. Left on direct SQL.
    ts_range = conn.execute("SELECT MIN(ts_ms), MAX(ts_ms) FROM liquidations WHERE symbol='ETHUSDT'").fetchone()
    start_ms, end_ms = int(ts_range[0]), int(ts_range[1])

    signals = runner._bucket_events(conn, vrule, start_ms, end_ms, limit=10_000)
    total_signals = len(signals)
    if total_signals == 0:
        return {"total_signals": 0, "no_fill": 0, "metrics": {"n_closed": 0, "insufficient_data": True}}

    trades: list[dict] = []
    no_fill = 0
    for sig in signals:
        if sig.get("fill_error"):
            no_fill += 1
            continue
        trade = runner._paper_trade_from_signal(vrule, sig, runner.DEFAULT_RISK)
        try:
            trade = runner._evaluate_trade(conn, trade, end_ms)
        except RuntimeError as exc:
            trade["status"] = "EXIT_FILL_ERROR"
            trade["exit_fill_error"] = str(exc)
        trades.append(trade)

    no_fill_rate = no_fill / total_signals if total_signals else 0.0
    preliminary  = total_signals < PRELIMINARY_N or no_fill_rate > NO_FILL_THRESHOLD

    metrics = _compute_metrics(conn, trades, tp, root=root, source_db_path=source_db_path)
    return {
        "total_signals": total_signals,
        "no_fill":       no_fill,
        "no_fill_rate":  round(no_fill_rate, 3),
        "preliminary":   preliminary,
        "metrics":       metrics,
    }


# ── Score for ranking ─────────────────────────────────────────────────────────

def _score(res: dict) -> float:
    m = res.get("metrics", {})
    if m.get("insufficient_data"):
        return -9999.0
    med = m.get("median_net") or 0.0
    cum = m.get("top3_removed_cum") or 0.0
    wr  = (m.get("win_rate") or 0.0) * 100
    return med * 2 + cum * 0.05 + wr * 0.5


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # `conn` stays for the two still-direct unbounded MIN/MAX scans and the
    # external s34_shadow_paper_runner calls (_bucket_events/
    # _paper_trade_from_signal/_evaluate_trade, untouched here); the
    # per-trade MFE mark_prices window moved to the reader (via `root`/
    # SOURCE_DB_PATH).
    conn = sqlite3.connect(f"file:{DB_PATH.as_posix()}?mode=ro", uri=True)
    root, _ = PR.resolve_production_root()

    # OUT-OF-SCOPE for RANGE-READ V5: unbounded MIN/MAX scan (no ts_ms
    # window), not a bounded range read. Left on direct SQL.
    ts_range = conn.execute("SELECT MIN(ts_ms), MAX(ts_ms) FROM mark_prices WHERE symbol='ETHUSDT'").fetchone()
    data_start = datetime.fromtimestamp(ts_range[0] / 1000, tz=timezone.utc).isoformat()
    data_end   = datetime.fromtimestamp(ts_range[1] / 1000, tz=timezone.utc).isoformat()

    all_results = []
    print(f"Data window: {data_start} -> {data_end}\n")

    for threshold, thresh_label in THRESHOLDS:
        print(f"=== {thresh_label} (threshold={threshold/1000:.0f}K) ===")
        thresh_results = []
        for tp, sl, be in VARIANTS:
            label = f"TP{int(tp)}/SL{int(sl)}/BE{int(be)}"
            res = _run_config(conn, threshold, tp, sl, be, root=root, source_db_path=SOURCE_DB_PATH)
            m   = res["metrics"]
            if m.get("insufficient_data"):
                print(f"  [{label}] no data")
                continue
            prelim = " [PRELIM]" if res["preliminary"] else ""
            print(
                f"  [{label}]{prelim}  "
                f"sigs={res['total_signals']}  nf={res['no_fill']}({res['no_fill_rate']*100:.0f}%)  "
                f"closed={m['n_closed']}  "
                f"median={m['median_net']:+.1f}  cum={m['cum_net']:+.1f}  "
                f"WR={m['win_rate']*100:.0f}%  "
                f"h1/h2={m['h1_median']:+.1f}/{m['h2_median']:+.1f}  "
                f"exits=TP:{m['exit_mix']['TP']} SL:{m['exit_mix']['SL']} BE:{m['exit_mix']['BE']} T:{m['exit_mix']['TIME']}"
            )
            thresh_results.append({
                "threshold_label": thresh_label,
                "threshold_usd": threshold,
                "variant":        label,
                "tp_bps":         tp,
                "sl_bps":         sl,
                "be_bps":         be,
                **res,
                "score":          _score(res),
            })
        all_results.extend(thresh_results)
        print()

    conn.close()

    # ── Rank top combos ───────────────────────────────────────────────────────
    ranked = sorted(
        [r for r in all_results if not r["metrics"].get("insufficient_data")],
        key=lambda r: -r["score"],
    )

    print("=== TOP 10 BY SCORE ===")
    for r in ranked[:10]:
        m = r["metrics"]
        prelim = " [PRELIM]" if r["preliminary"] else ""
        print(
            f"  {r['threshold_label']} {r['variant']}{prelim}  "
            f"median={m['median_net']:+.1f}  cum={m['top3_removed_cum']:+.1f}(t3r)  "
            f"WR={m['win_rate']*100:.0f}%  h1={m['h1_median']:+.1f} h2={m['h2_median']:+.1f}  "
            f"score={r['score']:.1f}"
        )

    # ── Identify viable combos ────────────────────────────────────────────────
    viable = [
        r for r in ranked
        if not r["preliminary"]
        and (r["metrics"].get("median_net") or 0) > 0
        and (r["metrics"].get("top3_removed_cum") or 0) > 0
        and (r["metrics"].get("h2_median") or 0) > 0
    ]
    print(f"\nViable (N>={PRELIMINARY_N}, NF<{NO_FILL_THRESHOLD*100:.0f}%, median>0, t3r>0, h2>0): {len(viable)}")
    for r in viable[:5]:
        m = r["metrics"]
        print(f"  {r['threshold_label']} {r['variant']}: median={m['median_net']:+.1f} h1={m['h1_median']:+.1f} h2={m['h2_median']:+.1f}")

    # ── Write MD report ───────────────────────────────────────────────────────
    now = datetime.now(timezone.utc).isoformat()
    lines = [
        "# S34 Sell-Liq Bounce Research",
        "",
        f"Generated: `{now}`",
        "",
        "**Hypothesis**: LONG entry at SELL liq cascade threshold cross captures post-cascade reversal.",
        f"Data window: `{data_start}` to `{data_end}`",
        "",
        "No runner/config/pre-reg changes. Research only.",
        "",
    ]

    for thresh, thresh_label in THRESHOLDS:
        rows = [r for r in all_results if r["threshold_usd"] == thresh]
        if not rows:
            continue
        lines += [f"## {thresh_label}", ""]
        lines += [
            "| Variant | Sigs | NF% | Closed | Median | Cum | T3R | WR | H1 | H2 | Prelim |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
        for r in rows:
            m = r["metrics"]
            if m.get("insufficient_data"):
                continue
            p = "yes" if r["preliminary"] else ""
            lines.append(
                f"| {r['variant']} "
                f"| {r['total_signals']} "
                f"| {r['no_fill_rate']*100:.0f}% "
                f"| {m['n_closed']} "
                f"| {m['median_net']:+.1f} "
                f"| {m['cum_net']:+.1f} "
                f"| {m['top3_removed_cum']:+.1f} "
                f"| {m['win_rate']*100:.0f}% "
                f"| {m['h1_median']:+.1f} "
                f"| {m['h2_median']:+.1f} "
                f"| {p} |"
            )
        lines.append("")

    lines += [
        "## Top 10 by Score",
        "",
        "| Threshold | Variant | Median | T3R Cum | WR | H1 | H2 | Prelim |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for r in ranked[:10]:
        m = r["metrics"]
        p = "yes" if r["preliminary"] else ""
        lines.append(
            f"| {r['threshold_label']} | {r['variant']} "
            f"| {m['median_net']:+.1f} | {m['top3_removed_cum']:+.1f} "
            f"| {m['win_rate']*100:.0f}% | {m['h1_median']:+.1f} | {m['h2_median']:+.1f} | {p} |"
        )
    lines.append("")

    if viable:
        lines += ["## Viable Combos", ""]
        for r in viable:
            m = r["metrics"]
            lines += [
                f"### {r['threshold_label']} {r['variant']}",
                "",
                f"- N={m['n_closed']}, median={m['median_net']:+.1f} bps, WR={m['win_rate']*100:.0f}%",
                f"- H1 median={m['h1_median']:+.1f}, H2 median={m['h2_median']:+.1f}",
                f"- Cum={m['cum_net']:+.1f}, T3R cum={m['top3_removed_cum']:+.1f}",
                f"- Exit mix: TP={m['exit_mix']['TP']} SL={m['exit_mix']['SL']} BE={m['exit_mix']['BE']} TIME={m['exit_mix']['TIME']}",
                f"- Giveback rate={m['giveback_rate']}",
                "",
            ]
    else:
        lines += ["## Viable Combos", "", "None found in this sweep.", ""]

    lines += [
        "## Interpretation Notes",
        "",
        "- Bounce hypothesis: 61% post-cascade reversal at avg +33.8 bps (600s window, ETH 500K SELL N=222).",
        "- Current SELL SHORT rules enter after cascade drops ~37-42 bps; TP requires 22-38 more bps. Runner shows median -12 to -38 bps.",
        "- This sweep tests LONG direction on same SELL liq signal. Viable = median>0, T3R>0, H2>0, N>=30.",
        "- If viable combos found: queue for runner-parity deep-dive before any pre-reg amendment.",
        "- Research only. No runner/config changes.",
    ]

    OUT_MD.write_text("\n".join(lines), encoding="utf-8")

    # ── Write JSON ────────────────────────────────────────────────────────────
    OUT_JSON.write_text(
        json.dumps(
            {
                "generated_at": now,
                "data_start": data_start,
                "data_end":   data_end,
                "preliminary_n": PRELIMINARY_N,
                "no_fill_threshold": NO_FILL_THRESHOLD,
                "results": all_results,
                "ranked_top10": [
                    {
                        "threshold_label": r["threshold_label"],
                        "variant": r["variant"],
                        "score": r["score"],
                        "metrics": r["metrics"],
                        "preliminary": r["preliminary"],
                    }
                    for r in ranked[:10]
                ],
                "viable": [
                    {
                        "threshold_label": r["threshold_label"],
                        "variant": r["variant"],
                        "metrics": r["metrics"],
                    }
                    for r in viable
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"\nMD  : {OUT_MD}")
    print(f"JSON: {OUT_JSON}")


if __name__ == "__main__":
    main()
