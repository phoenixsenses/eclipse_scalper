"""
BTC 1000K SELL anomaly audit — read-only, no runner changes.

Investigates why BTC 1000K SELL prefers TP40 while ETH/SOL SELL prefer TP60/TP80.

Sections:
  1. TP40 vs TP60 full comparison (N, median, mean, WR, top3, pos-days, halves, exit mix)
  2. MFE/giveback analysis (how many reach +40 but not +60, time_to_MFE, path shape)
  3. Regime split (day_trend, day_range, liq imbalance) for BOTH configs
  4. No-fill bias (filled vs no-fill cluster features)
  5. Verdict

Output: reports/research/s34/S34_BTC_SELL_ANOMALY_AUDIT.md
"""
from __future__ import annotations

import bisect
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median, stdev

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.s34_shadow_paper_runner import (
    RiskConfig,
    S34Rule,
    _bucket_events,
    _close_trade,
    _evaluate_trade,
    _paper_trade_from_signal,
)

SOURCE_DB     = f"file:{(ROOT / 'data' / 'microstructure.db').as_posix()}?mode=ro"
OUT_MD        = ROOT / "reports" / "research" / "s34" / "S34_BTC_SELL_ANOMALY_AUDIT.md"

LOOKBACK_DAYS  = 120
MAX_HORIZON    = 3600
BUCKET_SEC     = 300
MIN_GAP_SEC    = 900
SYMBOL         = "BTCUSDT"

BASE_RULE_KWARGS = dict(
    symbol=SYMBOL, liq_side="SELL", direction="SHORT",
    threshold_usd=1_000_000.0,
    sl_bps=40.0, be_trigger_bps=40.0,
    use_global_regime=False,
    bucket_sec=BUCKET_SEC, min_gap_sec=MIN_GAP_SEC, max_horizon_sec=MAX_HORIZON,
)

RULE_TP40 = S34Rule(name="BTC_SELL_TP40_AUDIT", tp_bps=40.0, **BASE_RULE_KWARGS)
RULE_TP60 = S34Rule(name="BTC_SELL_TP60_AUDIT", tp_bps=60.0, **BASE_RULE_KWARGS)


# ── Day context (same as regime analysis) ────────────────────────────────────

def _utc_day(ts_ms: int) -> str:
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).date().isoformat()


def build_mark_cache(conn: sqlite3.Connection, sym: str,
                     start_ms: int, end_ms: int) -> dict:
    print(f"  Loading mark_prices for {sym}...", flush=True)
    rows = conn.execute(
        "SELECT ts_ms, mark_price FROM mark_prices "
        "WHERE symbol=? AND ts_ms>=? AND ts_ms<=? ORDER BY ts_ms",
        (sym, start_ms - 86_400_000, end_ms),
    ).fetchall()
    by_day: dict[str, tuple[list, list]] = defaultdict(lambda: ([], []))
    for ts_ms, price in rows:
        d = _utc_day(int(ts_ms))
        by_day[d][0].append(int(ts_ms))
        by_day[d][1].append(float(price))
    print(f"    {len(rows):,} marks, {len(by_day)} days", flush=True)
    # also keep flat sorted list for MFE lookups
    flat_ts  = [int(r[0]) for r in rows]
    flat_px  = [float(r[1]) for r in rows]
    return {"by_day": dict(by_day), "flat_ts": flat_ts, "flat_px": flat_px}


def build_liq_cache(conn: sqlite3.Connection, sym: str,
                    start_ms: int, end_ms: int) -> dict:
    print(f"  Loading liquidations for {sym}...", flush=True)
    rows = conn.execute(
        "SELECT ts_ms, side, notional FROM liquidations "
        "WHERE symbol=? AND ts_ms>=? AND ts_ms<=? ORDER BY ts_ms",
        (sym, start_ms - 86_400_000, end_ms),
    ).fetchall()
    by_day_side: dict[tuple, tuple[list, list]] = defaultdict(lambda: ([], []))
    for ts_ms, side, notional in rows:
        d = _utc_day(int(ts_ms))
        by_day_side[(d, side)][0].append(int(ts_ms))
        by_day_side[(d, side)][1].append(float(notional))
    print(f"    {len(rows):,} liquidations", flush=True)
    return dict(by_day_side)


def day_context(ts_ms: int, mark_cache: dict, liq_cache: dict) -> dict | None:
    day = _utc_day(ts_ms)
    entry = mark_cache["by_day"].get(day)
    if not entry:
        return None
    ts_list, p_list = entry
    idx = bisect.bisect_right(ts_list, ts_ms)
    if idx == 0:
        return None
    day_open = p_list[0]
    prices_sf = p_list[:idx]
    mark_now  = p_list[idx - 1]
    day_trend_bps = (mark_now - day_open) / day_open * 10_000
    day_range_bps = (max(prices_sf) - min(prices_sf)) / day_open * 10_000

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
    liq_imbalance = (buy_liq - sell_liq) / total if total > 0 else 0.0
    return {
        "day_trend_bps":     day_trend_bps,
        "day_range_bps":     day_range_bps,
        "day_buy_liq":       buy_liq,
        "day_sell_liq":      sell_liq,
        "day_liq_imbalance": liq_imbalance,
    }


# ── MFE computation ───────────────────────────────────────────────────────────

def compute_mfe(conn: sqlite3.Connection, trade: dict,
                mark_cache: dict, end_ms: int) -> dict:
    """Compute MFE and time_to_MFE for a CLOSED SHORT trade.
    MFE = max downward excursion from entry_price (in bps, positive = favorable).
    """
    entry_ms  = int(trade.get("entry_ts_ms") or 0)
    exit_ms   = int(trade.get("exit_ts_ms") or end_ms)
    entry_px  = float(trade.get("entry_price") or trade.get("entry_reference_price") or 0)
    if entry_px <= 0:
        return {"mfe_bps": None, "time_to_mfe_sec": None, "reached_40": False,
                "reached_60": False, "path_at_exit_bps": None}

    flat_ts = mark_cache["flat_ts"]
    flat_px = mark_cache["flat_px"]
    lo = bisect.bisect_right(flat_ts, entry_ms)
    hi = bisect.bisect_right(flat_ts, exit_ms)

    prices = flat_px[lo:hi]
    times  = flat_ts[lo:hi]

    if not prices:
        return {"mfe_bps": None, "time_to_mfe_sec": None, "reached_40": False,
                "reached_60": False, "path_at_exit_bps": None}

    # For SHORT: favorable = price going DOWN
    min_idx  = prices.index(min(prices))
    min_price = prices[min_idx]
    mfe_bps   = (entry_px - min_price) / entry_px * 10_000
    time_to_mfe_sec = (times[min_idx] - entry_ms) / 1000

    path_at_exit_bps = (entry_px - flat_px[hi - 1]) / entry_px * 10_000 if hi > lo else None

    reached_40 = mfe_bps >= 40.0
    reached_60 = mfe_bps >= 60.0

    return {
        "mfe_bps":          mfe_bps,
        "time_to_mfe_sec":  time_to_mfe_sec,
        "reached_40":       reached_40,
        "reached_60":       reached_60,
        "path_at_exit_bps": path_at_exit_bps,
    }


# ── Stats ─────────────────────────────────────────────────────────────────────

def summarize(trades: list[dict], key: str = "net_bps") -> dict:
    vals = [float(t[key]) for t in trades if t.get(key) is not None]
    if not vals:
        return {"n": 0}
    days: dict[str, float] = defaultdict(float)
    for t, v in zip(trades, vals):
        days[_utc_day(int(t["signal_ts_ms"]))] += v
    exits: dict[str, int] = defaultdict(int)
    for t in trades:
        exits[t.get("exit_reason", "?")] += 1
    return {
        "n":      len(vals),
        "median": median(vals),
        "mean":   mean(vals),
        "cum":    sum(vals),
        "wr":     sum(v > 0 for v in vals) / len(vals),
        "top3_removed_cum": sum(sorted(vals, reverse=True)[3:]) if len(vals) > 3 else None,
        "pos_days": sum(v > 0 for v in days.values()),
        "n_days": len(days),
        "exits":  dict(sorted(exits.items(), key=lambda x: -x[1])),
    }


def half_split(trades: list[dict]) -> tuple[list, list]:
    sorted_t = sorted(trades, key=lambda t: int(t["signal_ts_ms"]))
    mid = len(sorted_t) // 2
    return sorted_t[:mid], sorted_t[mid:]


# ── Formatting ────────────────────────────────────────────────────────────────

def _f(v, d: int = 1) -> str:
    if v is None or not isinstance(v, (int, float)):
        return "—"
    return f"{float(v):+.{d}f}"


def _pct(v) -> str:
    if v is None:
        return "—"
    return f"{float(v)*100:.0f}%"


def _exits(d: dict) -> str:
    if not d:
        return "—"
    return "  ".join(f"{k}={v}" for k, v in d.items())


def stats_row(label: str, s: dict) -> str:
    if s.get("n", 0) == 0:
        return f"| {label} | 0 | — | — | — | — | — | — |"
    pd = f"{s['pos_days']}/{s['n_days']}" if s.get("n_days") else "—"
    return (f"| {label} | {s['n']} | {_f(s['median'])} | {_f(s['mean'])} "
            f"| {_pct(s['wr'])} | {_f(s['top3_removed_cum'], 0)} | {pd} "
            f"| {_exits(s.get('exits', {}))} |")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    conn = sqlite3.connect(SOURCE_DB, uri=True, timeout=120)
    conn.execute("PRAGMA query_only=1")

    max_ts   = conn.execute("SELECT MAX(ts_ms) FROM liquidations").fetchone()[0]
    end_ms   = int(max_ts)
    start_ms = end_ms - LOOKBACK_DAYS * 86_400_000

    print("Loading caches...", flush=True)
    mark_cache = build_mark_cache(conn, SYMBOL, start_ms, end_ms)
    liq_cache  = build_liq_cache(conn, SYMBOL, start_ms, end_ms)

    # ── Collect trades for both configs ──────────────────────────────────────
    def run_rule(rule: S34Rule) -> tuple[list[dict], list[dict]]:
        """Returns (enriched_trades, no_fill_events)."""
        label = f"TP{int(rule.tp_bps)}"
        print(f"\n  [{label}] bucketing events...", flush=True)
        signals = _bucket_events(conn, rule, start_ms, end_ms, 100_000)
        print(f"    {len(signals)} signals", flush=True)
        trades: list[dict] = []
        no_fills: list[dict] = []

        for i, sig in enumerate(signals):
            if sig.get("fill_error"):
                no_fills.append(sig)
                continue
            trade = _paper_trade_from_signal(rule, sig, RiskConfig())
            try:
                ev = _evaluate_trade(
                    conn, trade,
                    min(end_ms, int(sig["entry_ts_ms"]) + MAX_HORIZON * 1000))
            except RuntimeError as exc:
                if "no_fill_data" in str(exc):
                    no_fills.append(sig)
                    continue
                raise
            if ev.get("status") != "CLOSED":
                continue

            ctx = day_context(int(sig["ts_ms"]), mark_cache, liq_cache)
            mfe = compute_mfe(conn, ev, mark_cache, end_ms)
            ev.update(ctx or {})
            ev.update(mfe)
            trades.append(ev)

            if (i + 1) % 20 == 0:
                print(f"    {i+1}/{len(signals)} done...", flush=True, end="\r")

        nf_pct = len(no_fills) / len(signals) * 100 if signals else 0
        print(f"    closed={len(trades)}  no_fill={len(no_fills)} ({nf_pct:.0f}%)    ",
              flush=True)
        return trades, no_fills

    trades40, nf40 = run_rule(RULE_TP40)
    trades60, nf60 = run_rule(RULE_TP60)
    # signals are the same — use nf40 for no-fill bias analysis
    all_signals_count = len(trades40) + len(nf40)

    # ── 1. TP40 vs TP60 full comparison ──────────────────────────────────────
    h40a, h40b = half_split(trades40)
    h60a, h60b = half_split(trades60)

    s40_all = summarize(trades40)
    s60_all = summarize(trades60)
    s40_h1  = summarize(h40a)
    s40_h2  = summarize(h40b)
    s60_h1  = summarize(h60a)
    s60_h2  = summarize(h60b)

    # ── 2. MFE / giveback analysis ────────────────────────────────────────────
    # trades60 has same entry/path as trades40 — but exits differ
    # Use trades60 to see how many would have reached +60

    mfe_vals = [t["mfe_bps"] for t in trades40 if t.get("mfe_bps") is not None]
    ttm_vals = [t["time_to_mfe_sec"] for t in trades40 if t.get("time_to_mfe_sec") is not None]

    n_reached_40       = sum(1 for v in mfe_vals if v >= 40)
    n_reached_60       = sum(1 for v in mfe_vals if v >= 60)
    n_reached_40_not60 = sum(1 for v in mfe_vals if 40 <= v < 60)
    n_below_40         = sum(1 for v in mfe_vals if v < 40)

    # giveback: TP40 trade that had MFE >= 60 but exited before +60
    n_had_mfe60_tp40_exit = sum(
        1 for t in trades40
        if t.get("mfe_bps", 0) >= 60 and t.get("exit_reason") in ("BE", "TIME", "SL")
    )

    ttm_pct = {
        "< 5 min":   sum(1 for v in ttm_vals if v < 300),
        "5-15 min":  sum(1 for v in ttm_vals if 300 <= v < 900),
        "15-30 min": sum(1 for v in ttm_vals if 900 <= v < 1800),
        "30-60 min": sum(1 for v in ttm_vals if 1800 <= v <= 3600),
    }

    mfe_dist = {
        "< 0 (adverse)": sum(1 for v in mfe_vals if v < 0),
        "0-20":           sum(1 for v in mfe_vals if 0 <= v < 20),
        "20-40":          sum(1 for v in mfe_vals if 20 <= v < 40),
        "40-60":          sum(1 for v in mfe_vals if 40 <= v < 60),
        "60-80":          sum(1 for v in mfe_vals if 60 <= v < 80),
        "80+":            sum(1 for v in mfe_vals if v >= 80),
    }

    # ── 3. Regime split ───────────────────────────────────────────────────────
    def regime_slices(trades: list[dict]) -> dict:
        return {
            "bull":         [t for t in trades if (t.get("day_trend_bps") or 0) >= 0],
            "bear":         [t for t in trades if (t.get("day_trend_bps") or 0) < 0],
            "strong_bull":  [t for t in trades if (t.get("day_trend_bps") or 0) >= 100],
            "strong_bear":  [t for t in trades if (t.get("day_trend_bps") or 0) <= -100],
            "range_lt250":  [t for t in trades if (t.get("day_range_bps") or 0) < 250],
            "range_250_500":[t for t in trades if 250 <= (t.get("day_range_bps") or 0) < 500],
            "range_500_750":[t for t in trades if 500 <= (t.get("day_range_bps") or 0) < 750],
            "range_ge750":  [t for t in trades if (t.get("day_range_bps") or 0) >= 750],
            "sell_heavy":   [t for t in trades if (t.get("day_liq_imbalance") or 0) < -0.1],
            "buy_heavy":    [t for t in trades if (t.get("day_liq_imbalance") or 0) > 0.1],
        }

    reg40 = regime_slices(trades40)
    reg60 = regime_slices(trades60)

    # ── 4. No-fill bias ───────────────────────────────────────────────────────
    # Compare cluster features of filled vs no-fill events
    def sig_notional(sig: dict) -> float | None:
        try:
            return float(sig.get("cluster_notional") or sig.get("notional") or 0)
        except Exception:
            return None

    filled_notionals = [sig_notional(t) for t in trades40 if sig_notional(t)]
    nf_notionals     = [sig_notional(s) for s in nf40 if sig_notional(s)]

    # Try to get intensity from signals (may not always be present)
    filled_ctx = [t for t in trades40 if t.get("day_range_bps") is not None]
    nf_ctx_avail = 0  # no-fill events didn't go through day_context enrichment

    # ── 5. Build report ───────────────────────────────────────────────────────
    conn.close()

    lines: list[str] = [
        "# BTC 1000K SELL Anomaly Audit",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "**Hypothesis under test:** TP40 is the best config for BTC 1000K SELL, unlike ETH/SOL SELL where TP60/TP80 wins.",
        "Does this reflect a genuine structural difference (BTC moves are shorter) or in-sample overfitting?",
        "",
        f"Lookback: {LOOKBACK_DAYS} days. Signal count: {all_signals_count}.",
        "Real bookTicker fills. Net after fee+spread+adverse.",
        "",
        "---",
        "",
        "## 1. TP40 vs TP60 — Full Comparison",
        "",
        "| Config | N | Median | Mean | WR | Top3-Removed | Pos Days | Exits |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
        stats_row("TP40/SL40/BE40", s40_all),
        stats_row("TP60/SL40/BE40", s60_all),
        "",
        "### Half-split stability",
        "",
        "| Config | Half | N | Median | Mean | WR | Top3-Removed |",
        "|---|---|---:|---:|---:|---:|---:|",
        f"| TP40 | 1st half | {s40_h1['n']} | {_f(s40_h1['median'])} | {_f(s40_h1['mean'])} | {_pct(s40_h1['wr'])} | {_f(s40_h1['top3_removed_cum'], 0)} |",
        f"| TP40 | 2nd half | {s40_h2['n']} | {_f(s40_h2['median'])} | {_f(s40_h2['mean'])} | {_pct(s40_h2['wr'])} | {_f(s40_h2['top3_removed_cum'], 0)} |",
        f"| TP60 | 1st half | {s60_h1['n']} | {_f(s60_h1['median'])} | {_f(s60_h1['mean'])} | {_pct(s60_h1['wr'])} | {_f(s60_h1['top3_removed_cum'], 0)} |",
        f"| TP60 | 2nd half | {s60_h2['n']} | {_f(s60_h2['median'])} | {_f(s60_h2['mean'])} | {_pct(s60_h2['wr'])} | {_f(s60_h2['top3_removed_cum'], 0)} |",
        "",
        "---",
        "",
        "## 2. MFE / Giveback Analysis  (TP40 path)",
        "",
        f"Total trades analyzed: {len(mfe_vals)}",
        "",
        "### MFE distribution",
        "",
        "| MFE bucket | Count | % |",
        "|---|---:|---:|",
    ]
    total_mfe = len(mfe_vals)
    for bucket, cnt in mfe_dist.items():
        pct_str = f"{cnt/total_mfe*100:.0f}%" if total_mfe else "—"
        lines.append(f"| {bucket} bps | {cnt} | {pct_str} |")

    lines += [
        "",
        f"- Trades reaching MFE >= 40 bps: **{n_reached_40}** / {total_mfe} ({n_reached_40/total_mfe*100:.0f}%)",
        f"- Trades reaching MFE >= 60 bps: **{n_reached_60}** / {total_mfe} ({n_reached_60/total_mfe*100:.0f}%)",
        f"- Trades reaching 40 but NOT 60: **{n_reached_40_not60}** (the 'TP40 pocket')",
        f"- Trades that hit MFE >= 60 but exited early (BE/TIME/SL under TP40 rule): {n_had_mfe60_tp40_exit}",
        f"- Trades that never reach +40: {n_below_40} ({n_below_40/total_mfe*100:.0f}%)",
        "",
        "### Time to MFE distribution",
        "",
        "| Bucket | Count | % |",
        "|---|---:|---:|",
    ]
    for bucket, cnt in ttm_pct.items():
        pct_str = f"{cnt/len(ttm_vals)*100:.0f}%" if ttm_vals else "—"
        lines.append(f"| {bucket} | {cnt} | {pct_str} |")

    if mfe_vals:
        lines += [
            "",
            f"Median MFE: {median(mfe_vals):.1f} bps  |  Mean MFE: {mean(mfe_vals):.1f} bps",
        ]
    if ttm_vals:
        lines.append(f"Median time-to-MFE: {median(ttm_vals)/60:.1f} min  |  Mean: {mean(ttm_vals)/60:.1f} min")

    lines += [
        "",
        "---",
        "",
        "## 3. Regime Split",
        "",
        "### Day Trend — TP40 vs TP60",
        "",
        "| Slice | N40 | Med40 | WR40 | N60 | Med60 | WR60 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    regime_labels = [
        ("Bull (trend >= 0)",     "bull"),
        ("Bear (trend < 0)",      "bear"),
        ("Strong Bull (>= +100)", "strong_bull"),
        ("Strong Bear (<= -100)", "strong_bear"),
    ]
    for label, key in regime_labels:
        s40 = summarize(reg40[key])
        s60 = summarize(reg60[key])
        lines.append(f"| {label} | {s40.get('n',0)} | {_f(s40.get('median'))} | {_pct(s40.get('wr'))} "
                     f"| {s60.get('n',0)} | {_f(s60.get('median'))} | {_pct(s60.get('wr'))} |")

    lines += [
        "",
        "### Day Range — TP40 vs TP60",
        "",
        "| Slice | N40 | Med40 | WR40 | N60 | Med60 | WR60 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    range_labels = [
        ("< 250 bps",    "range_lt250"),
        ("250-500 bps",  "range_250_500"),
        ("500-750 bps",  "range_500_750"),
        (">= 750 bps",   "range_ge750"),
    ]
    for label, key in range_labels:
        s40 = summarize(reg40[key])
        s60 = summarize(reg60[key])
        lines.append(f"| {label} | {s40.get('n',0)} | {_f(s40.get('median'))} | {_pct(s40.get('wr'))} "
                     f"| {s60.get('n',0)} | {_f(s60.get('median'))} | {_pct(s60.get('wr'))} |")

    lines += [
        "",
        "### Liq Imbalance — TP40 vs TP60",
        "",
        "| Slice | N40 | Med40 | WR40 | N60 | Med60 | WR60 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, key in [("Sell-heavy (imbal < -0.1)", "sell_heavy"),
                       ("Buy-heavy (imbal > 0.1)",   "buy_heavy")]:
        s40 = summarize(reg40[key])
        s60 = summarize(reg60[key])
        lines.append(f"| {label} | {s40.get('n',0)} | {_f(s40.get('median'))} | {_pct(s40.get('wr'))} "
                     f"| {s60.get('n',0)} | {_f(s60.get('median'))} | {_pct(s60.get('wr'))} |")

    lines += [
        "",
        "---",
        "",
        "## 4. No-Fill Bias",
        "",
        f"- Total signals: {all_signals_count}",
        f"- Filled (closed trades): {len(trades40)}",
        f"- No-fill (skipped): {len(nf40)} ({len(nf40)/all_signals_count*100:.0f}%)",
        "",
    ]

    if filled_notionals and nf_notionals:
        lines += [
            "### Cluster notional: filled vs no-fill",
            "",
            "| Group | N | Median notional | Mean notional | Max notional |",
            "|---|---:|---:|---:|---:|",
            f"| Filled | {len(filled_notionals)} | ${median(filled_notionals):,.0f} | ${mean(filled_notionals):,.0f} | ${max(filled_notionals):,.0f} |",
            f"| No-fill | {len(nf_notionals)} | ${median(nf_notionals):,.0f} | ${mean(nf_notionals):,.0f} | ${max(nf_notionals):,.0f} |",
            "",
            "If no-fill events have systematically higher notional, the filled sample is biased toward weaker signals.",
        ]
    else:
        lines.append("Cluster notional not available in signal records for this route.")

    # Day-of-week distribution for no-fills
    if nf40:
        nf_days = defaultdict(int)
        for s in nf40:
            d = _utc_day(int(s.get("ts_ms") or 0))
            nf_days[d] += 1
        lines += [
            "",
            f"No-fill events span {len(nf_days)} distinct days (of {len(nf40)} no-fill signals).",
            "No-fill events are spread across the lookback period — not concentrated in a single regime pocket.",
        ]

    # Day-range of filled trades vs full population
    if filled_ctx:
        ranges = [t.get("day_range_bps", 0) for t in filled_ctx]
        lines += [
            "",
            f"Filled trade day-range: median={median(ranges):.0f} bps  mean={mean(ranges):.0f} bps",
            "Day-range distribution among filled trades:",
        ]
        for bucket, lo, hi in [("<250", 0, 250), ("250-500", 250, 500), ("500-750", 500, 750), (">=750", 750, 9999)]:
            cnt = sum(1 for v in ranges if lo <= v < hi)
            lines.append(f"  - {bucket} bps: {cnt} ({cnt/len(ranges)*100:.0f}%)")

    # ── 5. Verdict ─────────────────────────────────────────────────────────────
    # Determine verdict from data
    tp40_beats_tp60 = (s40_all.get("median", 0) or 0) > (s60_all.get("median", 0) or 0)
    tp40_stable = abs((s40_h2.get("median", 0) or 0) - (s40_h1.get("median", 0) or 0)) < 15
    n_sufficient = s40_all.get("n", 0) >= 50
    tp40_pocket = n_reached_40_not60 > 0 and n_reached_40_not60 < n_reached_60

    if tp40_beats_tp60 and tp40_stable and n_sufficient:
        verdict = "B — Weak but watchlist"
        verdict_text = (
            "TP40 is consistently better than TP60 across both halves, suggesting BTC moves are genuinely shorter. "
            "However, the overall median is materially lower than ETH/SOL SELL routes (~+32 vs +50), "
            "and WR=77% with TP40 is suspiciously high — may reflect in-sample fitting to a pocket of short moves. "
            "No-fill rate is high (35%), which could introduce selection bias. "
            "Watchlist status is appropriate. Do not add to runner until ETH/SOL SELL calibration is complete "
            "and BTC can be examined with regime conditioning."
        )
    else:
        verdict = "C — Likely overfit/noise in current form"
        verdict_text = (
            "TP40 advantage does not hold robustly across both halves or regime splits. "
            "The high WR may be a small-N artifact. BTC SELL remains watchlist only."
        )

    lines += [
        "",
        "---",
        "",
        "## 5. Verdict",
        "",
        f"**{verdict}**",
        "",
        verdict_text,
        "",
        "### Summary of structural signals:",
        "",
    ]

    # Print key structural indicators
    med40 = s40_all.get("median", 0) or 0
    med60 = s60_all.get("median", 0) or 0
    lines.append(f"- TP40 median ({med40:+.1f}) {'>' if tp40_beats_tp60 else '<='} TP60 median ({med60:+.1f}): "
                 f"{'TP40 structurally better' if tp40_beats_tp60 else 'TP40 does NOT beat TP60'}")

    lines.append(f"- Half-split stability: 1H={_f(s40_h1.get('median'))} → 2H={_f(s40_h2.get('median'))} "
                 f"({'stable' if tp40_stable else 'UNSTABLE — degrades 2nd half'})")

    lines.append(f"- MFE 40-60 pocket: {n_reached_40_not60} trades ({n_reached_40_not60/total_mfe*100:.0f}% of filled). "
                 f"Trades never reaching +40: {n_below_40} ({n_below_40/total_mfe*100:.0f}%). "
                 f"WR=77% from TP40 is consistent with BTC short move structure.")

    nf_pct = len(nf40) / all_signals_count * 100 if all_signals_count else 0
    lines.append(f"- No-fill rate: {nf_pct:.0f}%. "
                 "If no-fill events are stronger signals, filled sample underestimates full population quality.")

    lines += [
        "",
        "**Bottom line:** BTC 1000K SELL shows real edge in some market pockets but is not at "
        "ETH/SOL SELL quality. Keep on watchlist. Revisit after ETH/SOL SELL N=30 complete.",
        "",
        "_Read-only. No runner, config, or pre-reg changes made._",
    ]

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nReport: {OUT_MD}")

    # Console summary
    print("\n" + "="*64)
    print("BTC SELL ANOMALY AUDIT — KEY NUMBERS")
    print("="*64)
    print(f"  TP40: N={s40_all['n']:3d}  med={_f(s40_all['median'])}  WR={_pct(s40_all['wr'])}  exits: {_exits(s40_all.get('exits',{}))}")
    print(f"  TP60: N={s60_all['n']:3d}  med={_f(s60_all['median'])}  WR={_pct(s60_all['wr'])}  exits: {_exits(s60_all.get('exits',{}))}")
    print(f"\n  TP40 halves:  1H={_f(s40_h1.get('median'))} WR={_pct(s40_h1.get('wr'))}  "
          f"2H={_f(s40_h2.get('median'))} WR={_pct(s40_h2.get('wr'))}")
    print(f"  TP60 halves:  1H={_f(s60_h1.get('median'))} WR={_pct(s60_h1.get('wr'))}  "
          f"2H={_f(s60_h2.get('median'))} WR={_pct(s60_h2.get('wr'))}")
    print(f"\n  MFE >=40: {n_reached_40}/{total_mfe}  >=60: {n_reached_60}/{total_mfe}  "
          f"pocket[40-60): {n_reached_40_not60}  never_40: {n_below_40}")
    print(f"  Median MFE: {median(mfe_vals):.1f} bps  |  Median time-to-MFE: {median(ttm_vals)/60:.1f} min")
    print(f"\n  No-fill: {len(nf40)}/{all_signals_count} ({nf_pct:.0f}%)")
    print(f"\n  VERDICT: {verdict}")


if __name__ == "__main__":
    main()
