"""
SOL 200K SELL bear-day filter sweep — read-only research.

Finding from regime analysis: SOL 200K SELL is strongly bear-day dependent.
  Bull days (trend >= 0): N=6, median=-10.1, WR=33%
  Bear days (trend < 0):  N=40, median=+50.7, WR=70%

Questions:
  1. At what day_trend threshold does the edge turn consistently positive?
  2. What fraction of signals are filtered out at each threshold?
  3. Does bear-day filter also remove the buy-heavy liq problem?
  4. Is filtered edge stable (first/second half)?
  5. Net: is a bear-day gate worth it, or does signal frequency drop too much?

Output: reports/research/s34/S34_SOL200K_SELL_DAYFILTER.md
Read-only. No runner/config/pre-reg changes.
"""
from __future__ import annotations

import bisect
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.s34_shadow_paper_runner import (
    RiskConfig,
    S34Rule,
    _bucket_events,
    _evaluate_trade,
    _paper_trade_from_signal,
)

from ami.storage import production as PR
from ami.storage import research_reader as RR

SOURCE_DB      = f"file:{(ROOT / 'data' / 'microstructure.db').as_posix()}?mode=ro"
SOURCE_DB_PATH = (ROOT / "data" / "microstructure.db").as_posix()
OUT_MD        = ROOT / "reports" / "research" / "s34" / "S34_SOL200K_SELL_DAYFILTER.md"

LOOKBACK_DAYS = 120
MAX_HORIZON   = 3600
BUCKET_SEC    = 300
MIN_GAP_SEC   = 900

RULE = S34Rule(
    name="SOL_SELL_LIQ_SHORT_200K_TP60_SL30_BE30_RESEARCH",
    symbol="SOLUSDT", liq_side="SELL", direction="SHORT",
    threshold_usd=200_000.0, tp_bps=60.0, sl_bps=30.0,
    be_trigger_bps=30.0, use_global_regime=False,
    bucket_sec=BUCKET_SEC, min_gap_sec=MIN_GAP_SEC, max_horizon_sec=MAX_HORIZON,
)

# Trend thresholds to sweep: only fire when day_trend <= threshold
TREND_GATES = [
    ("no filter",   None),
    ("trend < 0",   0.0),
    ("trend < -25", -25.0),
    ("trend < -50", -50.0),
    ("trend < -100",-100.0),
    ("trend < -150",-150.0),
]

# Liq imbalance gate (combined with best trend gate)
LIQ_GATES = [
    ("no liq filter",       None),
    ("sell-heavy < -0.05",  -0.05),
    ("sell-heavy < -0.10",  -0.10),
    ("sell-heavy < -0.25",  -0.25),
]


def _utc_day(ts_ms: int) -> str:
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).date().isoformat()


def _mark_prices_range(conn: sqlite3.Connection, symbol: str, start_ms: int, end_ms: int) -> list[tuple]:
    """Direct-SQL oracle -- kept unchanged as the parity reference for
    `_mark_prices_range_v2` (BATCH-STORAGE-ROTATION-RETENTION-RANGE-READ-
    CONSUMER-MIGRATION-V6). No longer called by `build_caches`; the
    reader-backed path is used instead."""
    return conn.execute(
        "SELECT ts_ms, mark_price FROM mark_prices "
        "WHERE symbol=? AND ts_ms>=? AND ts_ms<=? ORDER BY ts_ms",
        (symbol, start_ms, end_ms),
    ).fetchall()


def _mark_prices_range_v2(root, symbol: str, start_ms: int, end_ms: int, source_db_path=None) -> list[tuple]:
    """Reader-backed replacement for `_mark_prices_range`, via `plan_read`/
    `execute_read`. `symbol` is a genuine parameter of `build_caches`, but
    the sole real call site hardcodes "SOLUSDT", which has no mark_prices
    archive partition -- real production smoke resolves SQLITE_ONLY;
    ARCHIVE_ONLY/HYBRID coverage for this table/symbol is via synthetic
    fixtures. Inclusive upper bound reproduced with `end_ms+1`. Streams in
    canonical (ts_ms ASC, id ASC) order -- a refinement of the oracle's
    `ORDER BY ts_ms` that yields an identical ts_ms sequence."""
    plan = RR.plan_read(root, table="mark_prices", symbol=symbol, start_ms=int(start_ms), end_ms=int(end_ms) + 1)
    result = RR.execute_read(plan, columns=("ts_ms", "mark_price"), source_db_path=source_db_path)
    return list(result.iter_rows())


def build_caches(conn: sqlite3.Connection, sym: str,
                 start_ms: int, end_ms: int, root, source_db_path=None) -> tuple[dict, dict]:
    print(f"  Loading mark_prices for {sym}...", flush=True)
    rows = _mark_prices_range_v2(root, sym, start_ms - 86_400_000, end_ms, source_db_path=source_db_path)
    by_day: dict[str, tuple[list, list]] = defaultdict(lambda: ([], []))
    for ts_ms, price in rows:
        d = _utc_day(int(ts_ms))
        by_day[d][0].append(int(ts_ms))
        by_day[d][1].append(float(price))
    print(f"    {len(rows):,} marks, {len(by_day)} days", flush=True)

    print(f"  Loading liquidations for {sym}...", flush=True)
    # OUT-OF-SCOPE for RANGE-READ V6: liquidations is an out-of-allowlist
    # table (no archive partition / reader support). Left on direct SQL.
    liq_rows = conn.execute(
        "SELECT ts_ms, side, notional FROM liquidations "
        "WHERE symbol=? AND ts_ms>=? AND ts_ms<=? ORDER BY ts_ms",
        (sym, start_ms - 86_400_000, end_ms),
    ).fetchall()
    by_day_side: dict[tuple, tuple[list, list]] = defaultdict(lambda: ([], []))
    for ts_ms, side, notional in liq_rows:
        d = _utc_day(int(ts_ms))
        by_day_side[(d, side)][0].append(int(ts_ms))
        by_day_side[(d, side)][1].append(float(notional))
    print(f"    {len(liq_rows):,} liquidations", flush=True)

    return dict(by_day), dict(by_day_side)


def day_context(ts_ms: int, mark_cache: dict, liq_cache: dict) -> dict | None:
    day = _utc_day(ts_ms)
    entry = mark_cache.get(day)
    if not entry:
        return None
    ts_list, p_list = entry
    idx = bisect.bisect_right(ts_list, ts_ms)
    if idx == 0:
        return None
    day_open = p_list[0]
    mark_now = p_list[idx - 1]
    day_trend_bps = (mark_now - day_open) / day_open * 10_000
    day_range_bps = (max(p_list[:idx]) - min(p_list[:idx])) / day_open * 10_000

    def cum_liq(side: str) -> float:
        key = (day, side)
        e2 = liq_cache.get(key)
        if not e2:
            return 0.0
        ts2, n2 = e2
        idx2 = bisect.bisect_right(ts2, ts_ms)
        return sum(n2[:idx2])

    buy_liq  = cum_liq("BUY")
    sell_liq = cum_liq("SELL")
    total    = buy_liq + sell_liq
    return {
        "day_trend_bps":    day_trend_bps,
        "day_range_bps":    day_range_bps,
        "day_liq_imbalance": (buy_liq - sell_liq) / total if total > 0 else 0.0,
    }


def summarize(trades: list[dict]) -> dict:
    vals = [float(t["net_bps"]) for t in trades if t.get("net_bps") is not None]
    if not vals:
        return {"n": 0, "median": None, "mean": None, "wr": None,
                "top3_removed_cum": None, "pos_days": None, "n_days": None}
    days: dict[str, float] = defaultdict(float)
    for t, v in zip(trades, vals):
        days[_utc_day(int(t["signal_ts_ms"]))] += v
    return {
        "n":                len(vals),
        "median":           median(vals),
        "mean":             mean(vals),
        "wr":               sum(v > 0 for v in vals) / len(vals),
        "top3_removed_cum": sum(sorted(vals, reverse=True)[3:]) if len(vals) > 3 else None,
        "pos_days":         sum(v > 0 for v in days.values()),
        "n_days":           len(days),
    }


def half_split(trades: list[dict]) -> tuple[list, list]:
    s = sorted(trades, key=lambda t: int(t["signal_ts_ms"]))
    mid = len(s) // 2
    return s[:mid], s[mid:]


def _f(v, d: int = 1) -> str:
    if v is None:
        return "—"
    return f"{float(v):+.{d}f}"


def _pct(v) -> str:
    return f"{float(v)*100:.0f}%" if v is not None else "—"


def main() -> None:
    conn = sqlite3.connect(SOURCE_DB, uri=True, timeout=120)
    conn.execute("PRAGMA query_only=1")
    root, _ = PR.resolve_production_root()

    # OUT-OF-SCOPE for RANGE-READ V6: unbounded, no symbol filter, AND
    # out-of-allowlist table (liquidations) -- not an execute_read target.
    max_ts   = conn.execute("SELECT MAX(ts_ms) FROM liquidations").fetchone()[0]
    end_ms   = int(max_ts)
    start_ms = end_ms - LOOKBACK_DAYS * 86_400_000

    mark_cache, liq_cache = build_caches(conn, "SOLUSDT", start_ms, end_ms, root, source_db_path=SOURCE_DB_PATH)

    print("  Bucketing SOLUSDT SELL 200K events...", flush=True)
    # OUT-OF-SCOPE for RANGE-READ V6: _bucket_events/_paper_trade_from_signal/
    # _evaluate_trade own their own direct-SQL internals in the untouched
    # s34_shadow_paper_runner module; not migrated here.
    signals = _bucket_events(conn, RULE, start_ms, end_ms, 100_000)
    print(f"  {len(signals)} signals", flush=True)

    # Collect all enriched trades
    all_trades: list[dict] = []
    no_fill = 0
    for i, sig in enumerate(signals):
        if sig.get("fill_error"):
            no_fill += 1
            continue
        trade = _paper_trade_from_signal(RULE, sig, RiskConfig())
        try:
            ev = _evaluate_trade(
                conn, trade,
                min(end_ms, int(sig["entry_ts_ms"]) + MAX_HORIZON * 1000))
        except RuntimeError as exc:
            if "no_fill_data" in str(exc):
                no_fill += 1
                continue
            raise
        if ev.get("status") != "CLOSED":
            continue
        ctx = day_context(int(sig["ts_ms"]), mark_cache, liq_cache)
        if ctx is None:
            continue
        ev.update(ctx)
        all_trades.append(ev)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(signals)} done...", flush=True, end="\r")

    conn.close()
    nf_pct = no_fill / len(signals) * 100 if signals else 0
    print(f"  closed={len(all_trades)}  no_fill={no_fill} ({nf_pct:.0f}%)    ", flush=True)

    # ── Trend gate sweep ──────────────────────────────────────────────────────
    print("\n  Running trend gate sweep...", flush=True)
    trend_results: list[dict] = []
    for label, threshold in TREND_GATES:
        if threshold is None:
            filtered = all_trades
        else:
            filtered = [t for t in all_trades if t.get("day_trend_bps", 0) < threshold]
        s = summarize(filtered)
        h1, h2 = half_split(filtered)
        s1, s2 = summarize(h1), summarize(h2)
        trend_results.append({
            "label":     label,
            "threshold": threshold,
            "n_total":   len(all_trades),
            "n_filter":  len(filtered),
            "filter_pct": len(filtered) / len(all_trades) if all_trades else 0,
            "stats":     s,
            "h1":        s1,
            "h2":        s2,
        })
        print(f"    {label:20s}  N={len(filtered):3d}  med={_f(s.get('median'))}  "
              f"WR={_pct(s.get('wr'))}  1H={_f(s1.get('median'))}  2H={_f(s2.get('median'))}", flush=True)

    # ── Liq gate combined with best trend gate ────────────────────────────────
    # Use trend < 0 (most practical gate) as base
    bear_trades = [t for t in all_trades if t.get("day_trend_bps", 0) < 0]
    print(f"\n  Liq gate sweep on bear-day subset (N={len(bear_trades)})...", flush=True)
    liq_results: list[dict] = []
    for label, threshold in LIQ_GATES:
        if threshold is None:
            filtered = bear_trades
        else:
            filtered = [t for t in bear_trades
                        if t.get("day_liq_imbalance", 0) < threshold]
        s = summarize(filtered)
        h1, h2 = half_split(filtered)
        s1, s2 = summarize(h1), summarize(h2)
        liq_results.append({
            "label":    label,
            "n_filter": len(filtered),
            "filter_pct": len(filtered) / len(all_trades) if all_trades else 0,
            "stats":    s,
            "h1":       s1,
            "h2":       s2,
        })
        print(f"    {label:30s}  N={len(filtered):3d}  med={_f(s.get('median'))}  "
              f"WR={_pct(s.get('wr'))}  1H={_f(s1.get('median'))}  2H={_f(s2.get('median'))}", flush=True)

    # ── Cross-check: what do bull-day trades look like? ───────────────────────
    bull_trades = [t for t in all_trades if t.get("day_trend_bps", 0) >= 0]
    s_bull = summarize(bull_trades)
    s_bear = summarize(bear_trades)

    # Range split within bear days
    bear_range_lt250 = [t for t in bear_trades if t.get("day_range_bps", 0) < 250]
    bear_range_250_500 = [t for t in bear_trades if 250 <= t.get("day_range_bps", 0) < 500]
    bear_range_ge500 = [t for t in bear_trades if t.get("day_range_bps", 0) >= 500]

    # ── Write report ──────────────────────────────────────────────────────────
    lines: list[str] = [
        "# SOL 200K SELL Bear-Day Filter Sweep",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "**Motivation:** Regime analysis found SOL 200K SELL is strongly bear-day dependent.",
        "Bull days: N=6, median=-10.1 bps, WR=33%. Bear days: N=40, median=+50.7, WR=70%.",
        "This sweep finds the optimal day_trend gate and checks if it's stable.",
        "",
        f"Lookback: {LOOKBACK_DAYS} days. Total closed trades (no filter): {len(all_trades)}. "
        f"No-fill: {no_fill} ({nf_pct:.0f}%).",
        "",
        "---",
        "",
        "## 1. Day Trend Gate Sweep",
        "",
        "| Gate | N | % kept | Median | Mean | WR | Top3-Rmv | Pos Days | 1H Median | 2H Median |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in trend_results:
        s = r["stats"]
        if s["n"] == 0:
            lines.append(f"| {r['label']} | 0 | {r['filter_pct']*100:.0f}% | — | — | — | — | — | — | — |")
            continue
        pd = f"{s['pos_days']}/{s['n_days']}" if s.get("n_days") else "—"
        lines.append(
            f"| {r['label']} | {s['n']} | {r['filter_pct']*100:.0f}% "
            f"| {_f(s['median'])} | {_f(s['mean'])} | {_pct(s['wr'])} "
            f"| {_f(s['top3_removed_cum'], 0)} | {pd} "
            f"| {_f(r['h1'].get('median'))} | {_f(r['h2'].get('median'))} |"
        )

    lines += [
        "",
        "---",
        "",
        "## 2. Combined Gate: Bear Day + Liq Imbalance",
        "",
        "Base filter: `day_trend < 0`. Adding liq imbalance gate on top.",
        "",
        "| Gate | N | % of all | Median | Mean | WR | Top3-Rmv | 1H Median | 2H Median |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in liq_results:
        s = r["stats"]
        if s["n"] == 0:
            lines.append(f"| {r['label']} | 0 | — | — | — | — | — | — | — |")
            continue
        lines.append(
            f"| {r['label']} | {s['n']} | {r['filter_pct']*100:.0f}% "
            f"| {_f(s['median'])} | {_f(s['mean'])} | {_pct(s['wr'])} "
            f"| {_f(s['top3_removed_cum'], 0)} "
            f"| {_f(r['h1'].get('median'))} | {_f(r['h2'].get('median'))} |"
        )

    lines += [
        "",
        "---",
        "",
        "## 3. Bear Day Range Split",
        "",
        "Within bear-day trades only — does range matter?",
        "",
        "| Slice | N | Median | Mean | WR | Top3-Rmv |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label, sub in [
        ("Bear, range < 250",    bear_range_lt250),
        ("Bear, range 250-500",  bear_range_250_500),
        ("Bear, range >= 500",   bear_range_ge500),
    ]:
        s = summarize(sub)
        if s["n"] == 0:
            lines.append(f"| {label} | 0 | — | — | — | — |")
        else:
            lines.append(f"| {label} | {s['n']} | {_f(s['median'])} | {_f(s['mean'])} "
                         f"| {_pct(s['wr'])} | {_f(s['top3_removed_cum'], 0)} |")

    # ── Conclusion ────────────────────────────────────────────────────────────
    # Find the best stable gate (highest median with N >= 15 and both halves positive)
    best = None
    for r in trend_results:
        s = r["stats"]
        h1_med = r["h1"].get("median") or -999
        h2_med = r["h2"].get("median") or -999
        if s["n"] >= 15 and h1_med > 0 and h2_med > 0:
            if best is None or (s["median"] or 0) > (best["stats"]["median"] or 0):
                best = r

    lines += [
        "",
        "---",
        "",
        "## 4. Conclusion",
        "",
    ]

    if best:
        s = best["stats"]
        lines += [
            f"**Best stable gate: `{best['label']}`**",
            "",
            f"- N={s['n']} ({best['filter_pct']*100:.0f}% of all signals kept)",
            f"- Median={_f(s['median'])}  Mean={_f(s['mean'])}  WR={_pct(s['wr'])}",
            f"- Half-split: 1H={_f(best['h1'].get('median'))}  2H={_f(best['h2'].get('median'))}  (both positive)",
            f"- Top3-removed cum={_f(s.get('top3_removed_cum'), 0)}",
            f"- Positive days: {s['pos_days']}/{s['n_days']}",
            "",
        ]

        # Is N still enough for exploratory criteria?
        n_ok = s["n"] >= 30
        lines += [
            "**Gate assessment:**",
            "",
            f"- Enough N for exploratory (N>=30): {'YES' if n_ok else 'NO — N too small for pre-reg criteria'}",
            f"- Median improvement vs no filter: "
            f"{_f((s.get('median') or 0) - (trend_results[0]['stats'].get('median') or 0))} bps",
            "",
        ]

        if n_ok:
            lines.append("**This gate materially improves the signal and keeps enough N. "
                         "Candidate for future runner filter, but requires new pre-registration amendment "
                         "— do not add to runner without filing amendment first.**")
        else:
            lines.append("**Gate improves quality but N too small for independent pre-registration. "
                         "Watchlist. More lookback data needed before committing.**")
    else:
        lines.append("No gate found that is both stable (both halves positive) and has N >= 15. "
                     "SOL 200K SELL bear-day hypothesis holds directionally but N is insufficient "
                     "to pre-register with a gate. Continue collecting data.")

    lines += [
        "",
        "_Read-only. No runner, config, or pre-reg changes made._",
    ]

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nReport: {OUT_MD}")

    # Console summary
    print("\n" + "="*68)
    print("SOL 200K SELL BEAR-DAY FILTER — KEY NUMBERS")
    print("="*68)
    print(f"  All N={len(all_trades)}  med={_f(summarize(all_trades)['median'])}  WR={_pct(summarize(all_trades)['wr'])}")
    print(f"  Bull (trend>=0):  N={s_bull['n']:2d}  med={_f(s_bull['median'])}  WR={_pct(s_bull['wr'])}")
    print(f"  Bear (trend<0):   N={s_bear['n']:2d}  med={_f(s_bear['median'])}  WR={_pct(s_bear['wr'])}")
    print()
    for r in trend_results[1:]:
        s = r["stats"]
        print(f"  {r['label']:20s}  N={s.get('n',0):2d}  med={_f(s.get('median'))}  "
              f"WR={_pct(s.get('wr'))}  1H={_f(r['h1'].get('median'))}  2H={_f(r['h2'].get('median'))}")
    if best:
        print(f"\n  Best stable gate: {best['label']}")


if __name__ == "__main__":
    main()
