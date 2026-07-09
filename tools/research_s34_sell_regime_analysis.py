"""
SELL regime analysis — read-only research.

For each SELL_EXP route, splits outcomes by:
  - day_trend_bps (bullish >= 0 vs bearish < 0)
  - day_range_bps buckets (<250, 250-500, 500-750, >750)
  - day liq imbalance (buy-heavy vs sell-heavy)

Day context computed no-lookahead from raw microstructure:
  - day_trend  = (mark_at_event - day_open) / day_open * 10000
  - day_range  = (day_high - day_low) / day_open * 10000  [so far at event]
  - liq_imbal  = (day_buy_liq - day_sell_liq) / (day_buy_liq + day_sell_liq)

Routes:
  ETH 500K SELL  TP60/SL40/BE40
  ETH 1M  SELL   TP80/SL40/BE40
  SOL 100K SELL  TP60/SL30/BE40
  SOL 200K SELL  TP60/SL30/BE30

Read-only. No runner/config/pre-reg changes.
"""
from __future__ import annotations

import bisect
import json
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

SOURCE_DB     = f"file:{(ROOT / 'data' / 'microstructure.db').as_posix()}?mode=ro"
SOURCE_DB_PATH = (ROOT / "data" / "microstructure.db").as_posix()
OUT_MD        = ROOT / "reports" / "research" / "s34" / "S34_SELL_REGIME_ANALYSIS.md"

LOOKBACK_DAYS  = 120
MAX_HORIZON    = 3600
BUCKET_SEC     = 300
MIN_GAP_SEC    = 900

ROUTES = [
    S34Rule(
        name="ETH_SELL_LIQ_SHORT_500K_TP60_SL40_BE40_RESEARCH",
        symbol="ETHUSDT", liq_side="SELL", direction="SHORT",
        threshold_usd=500_000.0, tp_bps=60.0, sl_bps=40.0,
        be_trigger_bps=40.0, use_global_regime=False,
        bucket_sec=BUCKET_SEC, min_gap_sec=MIN_GAP_SEC, max_horizon_sec=MAX_HORIZON,
    ),
    S34Rule(
        name="ETH_SELL_LIQ_SHORT_1M_TP80_SL40_BE40_RESEARCH",
        symbol="ETHUSDT", liq_side="SELL", direction="SHORT",
        threshold_usd=1_000_000.0, tp_bps=80.0, sl_bps=40.0,
        be_trigger_bps=40.0, use_global_regime=False,
        bucket_sec=BUCKET_SEC, min_gap_sec=MIN_GAP_SEC, max_horizon_sec=MAX_HORIZON,
    ),
    S34Rule(
        name="SOL_SELL_LIQ_SHORT_100K_TP60_SL30_BE40_RESEARCH",
        symbol="SOLUSDT", liq_side="SELL", direction="SHORT",
        threshold_usd=100_000.0, tp_bps=60.0, sl_bps=30.0,
        be_trigger_bps=40.0, use_global_regime=False,
        bucket_sec=BUCKET_SEC, min_gap_sec=MIN_GAP_SEC, max_horizon_sec=MAX_HORIZON,
    ),
    S34Rule(
        name="SOL_SELL_LIQ_SHORT_200K_TP60_SL30_BE30_RESEARCH",
        symbol="SOLUSDT", liq_side="SELL", direction="SHORT",
        threshold_usd=200_000.0, tp_bps=60.0, sl_bps=30.0,
        be_trigger_bps=30.0, use_global_regime=False,
        bucket_sec=BUCKET_SEC, min_gap_sec=MIN_GAP_SEC, max_horizon_sec=MAX_HORIZON,
    ),
]

ROUTE_LABELS = {
    "ETH_SELL_LIQ_SHORT_500K_TP60_SL40_BE40_RESEARCH":   "ETH 500K SELL  TP60/SL40/BE40",
    "ETH_SELL_LIQ_SHORT_1M_TP80_SL40_BE40_RESEARCH":     "ETH 1M  SELL   TP80/SL40/BE40",
    "SOL_SELL_LIQ_SHORT_100K_TP60_SL30_BE40_RESEARCH":   "SOL 100K SELL  TP60/SL30/BE40",
    "SOL_SELL_LIQ_SHORT_200K_TP60_SL30_BE30_RESEARCH":   "SOL 200K SELL  TP60/SL30/BE30",
}


# ── Day context cache ──────────────────────────────────────────────────────────

def _utc_day(ts_ms: int) -> str:
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).date().isoformat()


def _mark_prices_range(conn: sqlite3.Connection, symbol: str, start_ms: int, end_ms: int) -> list[tuple]:
    """Direct-SQL oracle -- kept unchanged as the parity reference for
    `_mark_prices_range_v2` (BATCH-STORAGE-ROTATION-RETENTION-RANGE-READ-
    CONSUMER-MIGRATION-V5). No longer called by `build_mark_cache`; the
    reader-backed path is used instead."""
    return conn.execute(
        "SELECT ts_ms, mark_price FROM mark_prices "
        "WHERE symbol=? AND ts_ms>=? AND ts_ms<=? ORDER BY ts_ms",
        (symbol, start_ms, end_ms),
    ).fetchall()


def _mark_prices_range_v2(root, symbol: str, start_ms: int, end_ms: int, source_db_path=None) -> list[tuple]:
    """Reader-backed replacement for `_mark_prices_range`, via `plan_read`/
    `execute_read`. `symbol` is a genuine runtime parameter (iterated over
    the routes' distinct symbols -- ETHUSDT/SOLUSDT). Inclusive upper
    bound reproduced with `end_ms+1` (exact for integer ts_ms). Streams in
    canonical (ts_ms ASC, id ASC) order -- a refinement of the oracle's
    `ORDER BY ts_ms` that yields an identical ts_ms sequence."""
    plan = RR.plan_read(root, table="mark_prices", symbol=symbol, start_ms=int(start_ms), end_ms=int(end_ms) + 1)
    result = RR.execute_read(plan, columns=("ts_ms", "mark_price"), source_db_path=source_db_path)
    return list(result.iter_rows())


def build_mark_cache(conn: sqlite3.Connection, symbols: list[str],
                     start_ms: int, end_ms: int, root, source_db_path=None) -> dict[tuple, tuple]:
    """Returns {(symbol, day_str): (ts_list, price_list)} sorted by ts."""
    print("  Building mark_prices cache...", flush=True)
    cache: dict[tuple, tuple] = {}
    for sym in symbols:
        # `start_ms - 86_400_000`: extra day for open lookback (unchanged
        # from the oracle's window semantics; migrated to the reader below).
        rows = _mark_prices_range_v2(root, sym, start_ms - 86_400_000, end_ms, source_db_path=source_db_path)
        by_day: dict[str, tuple[list, list]] = defaultdict(lambda: ([], []))
        for ts_ms, price in rows:
            day = _utc_day(int(ts_ms))
            by_day[day][0].append(int(ts_ms))
            by_day[day][1].append(float(price))
        for day, (ts_list, p_list) in by_day.items():
            cache[(sym, day)] = (ts_list, p_list)
        print(f"    {sym}: {len(rows):,} marks, {len(by_day)} days", flush=True)
    return cache


def build_liq_cache(conn: sqlite3.Connection, symbols: list[str],
                    start_ms: int, end_ms: int) -> dict[tuple, tuple]:
    """Returns {(symbol, day_str, side): (ts_list, notional_list)} sorted by ts.

    OUT-OF-SCOPE for RANGE-READ V5: `liquidations` is an out-of-allowlist
    table (no archive partition / reader support). Left on direct SQL."""
    print("  Building liquidations cache...", flush=True)
    cache: dict[tuple, tuple] = {}
    for sym in symbols:
        rows = conn.execute(
            "SELECT ts_ms, side, notional FROM liquidations "
            "WHERE symbol=? AND ts_ms>=? AND ts_ms<=? ORDER BY ts_ms",
            (sym, start_ms - 86_400_000, end_ms),
        ).fetchall()
        by_day_side: dict[tuple, tuple[list, list]] = defaultdict(lambda: ([], []))
        for ts_ms, side, notional in rows:
            day = _utc_day(int(ts_ms))
            key = (sym, day, side)
            by_day_side[key][0].append(int(ts_ms))
            by_day_side[key][1].append(float(notional))
        for k, (ts_list, n_list) in by_day_side.items():
            cache[k] = (ts_list, n_list)
        print(f"    {sym}: {len(rows):,} liquidations", flush=True)
    return cache


def day_context(
    ts_ms: int, symbol: str,
    mark_cache: dict, liq_cache: dict,
) -> dict | None:
    day = _utc_day(ts_ms)
    entry = mark_cache.get((symbol, day))
    if not entry:
        return None
    ts_list, p_list = entry
    # find marks up to and including ts_ms
    idx = bisect.bisect_right(ts_list, ts_ms)
    if idx == 0:
        return None
    day_open  = p_list[0]
    prices_so_far = p_list[:idx]
    day_high  = max(prices_so_far)
    day_low   = min(prices_so_far)
    mark_now  = p_list[idx - 1]
    day_trend_bps = (mark_now - day_open) / day_open * 10_000
    day_range_bps = (day_high - day_low) / day_open * 10_000

    # liq imbalance — cumulative BUY/SELL liq for the day up to ts_ms
    def cum_liq(side: str) -> float:
        key = (symbol, day, side)
        entry2 = liq_cache.get(key)
        if not entry2:
            return 0.0
        ts2, n2 = entry2
        idx2 = bisect.bisect_right(ts2, ts_ms)
        return sum(n2[:idx2])

    buy_liq  = cum_liq("BUY")
    sell_liq = cum_liq("SELL")
    total    = buy_liq + sell_liq
    liq_imbalance = (buy_liq - sell_liq) / total if total > 0 else 0.0

    return {
        "day_trend_bps":      day_trend_bps,
        "day_range_bps":      day_range_bps,
        "day_buy_liq":        buy_liq,
        "day_sell_liq":       sell_liq,
        "day_liq_imbalance":  liq_imbalance,
    }


# ── Stats ──────────────────────────────────────────────────────────────────────

def summarize(trades: list[dict]) -> dict:
    vals = [float(t["net_bps"]) for t in trades if t.get("net_bps") is not None]
    if not vals:
        return {"n": 0, "median": None, "mean": None, "wr": None,
                "top3_removed_cum": None, "positive_days": None, "n_days": None}
    days: dict[str, float] = defaultdict(float)
    for t, v in zip(trades, vals):
        days[_utc_day(int(t["signal_ts_ms"]))] += v
    return {
        "n":               len(vals),
        "median":          median(vals),
        "mean":            mean(vals),
        "wr":              sum(v > 0 for v in vals) / len(vals),
        "top3_removed_cum": sum(sorted(vals, reverse=True)[3:]) if len(vals) > 3 else None,
        "positive_days":   sum(v > 0 for v in days.values()),
        "n_days":          len(days),
    }


# ── Report formatting ──────────────────────────────────────────────────────────

def _f(v, d: int = 1) -> str:
    if v is None:
        return "—"
    return f"{float(v):+.{d}f}"


def _pct(v) -> str:
    if v is None:
        return "—"
    return f"{float(v)*100:.0f}%"


def section_table(title: str, rows: list[tuple]) -> list[str]:
    lines = [f"\n### {title}", ""]
    lines.append("| Slice | N | Median | Mean | WR | Top3-Removed | Pos Days |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for label, s in rows:
        if s["n"] == 0:
            lines.append(f"| {label} | 0 | — | — | — | — | — |")
        else:
            pos_d = f"{s['positive_days']}/{s['n_days']}" if s['n_days'] else "—"
            lines.append(
                f"| {label} | {s['n']} | {_f(s['median'])} | {_f(s['mean'])} | "
                f"{_pct(s['wr'])} | {_f(s['top3_removed_cum'], 0)} | {pos_d} |"
            )
    return lines


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    # `conn` stays for the still-direct out-of-allowlist (liquidations) read
    # and the external s34_shadow_paper_runner calls (_bucket_events /
    # _evaluate_trade, which own their own direct-SQL internals in that
    # module, untouched here); the mark_prices cache build moved to the
    # reader (via `root`/SOURCE_DB_PATH).
    conn = sqlite3.connect(SOURCE_DB, uri=True, timeout=120)
    conn.execute("PRAGMA query_only=1")
    root, _ = PR.resolve_production_root()

    max_ts   = conn.execute("SELECT MAX(ts_ms) FROM liquidations").fetchone()[0]
    end_ms   = int(max_ts)
    start_ms = end_ms - LOOKBACK_DAYS * 86_400_000

    symbols = list({r.symbol for r in ROUTES})
    mark_cache = build_mark_cache(conn, symbols, start_ms, end_ms, root, source_db_path=SOURCE_DB_PATH)
    liq_cache  = build_liq_cache(conn, symbols, start_ms, end_ms)

    report_lines: list[str] = [
        "# S34 SELL Regime Analysis",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        f"Lookback: {LOOKBACK_DAYS} days. Routes: {len(ROUTES)}.",
        "Day context: no-lookahead (day-so-far at event time).",
        "All outcomes: real bookTicker fills, net after fee+spread+adverse.",
        "",
        "---",
    ]

    all_results: dict[str, dict] = {}

    for rule in ROUTES:
        label = ROUTE_LABELS[rule.name]
        print(f"\n[{label}]", flush=True)
        print("  Bucketing events...", flush=True)

        signals = _bucket_events(conn, rule, start_ms, end_ms, 100_000)
        print(f"  {len(signals)} signals", flush=True)

        enriched: list[dict] = []
        no_fill = 0
        no_ctx  = 0

        for i, sig in enumerate(signals):
            if sig.get("fill_error"):
                no_fill += 1
                continue
            trade = _paper_trade_from_signal(rule, sig, RiskConfig())
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

            ctx = day_context(int(sig["ts_ms"]), rule.symbol, mark_cache, liq_cache)
            if ctx is None:
                no_ctx += 1
                continue
            ev.update(ctx)
            enriched.append(ev)

            if (i + 1) % 20 == 0:
                print(f"  {i+1}/{len(signals)} done...", flush=True, end="\r")

        print(f"  closed={len(enriched)}  no_fill={no_fill}  no_ctx={no_ctx}    ", flush=True)

        # ── Trend split ────────────────────────────────────────────────────────
        trend_bull = [t for t in enriched if t["day_trend_bps"] >= 0]
        trend_bear = [t for t in enriched if t["day_trend_bps"] < 0]
        # finer: strong bull/bear
        trend_strong_bull = [t for t in enriched if t["day_trend_bps"] >= 100]
        trend_strong_bear = [t for t in enriched if t["day_trend_bps"] <= -100]

        # ── Range buckets ──────────────────────────────────────────────────────
        range_low  = [t for t in enriched if t["day_range_bps"] < 250]
        range_mid  = [t for t in enriched if 250 <= t["day_range_bps"] < 500]
        range_high = [t for t in enriched if 500 <= t["day_range_bps"] < 750]
        range_vhigh = [t for t in enriched if t["day_range_bps"] >= 750]

        # ── Liq imbalance split ────────────────────────────────────────────────
        # imbalance > 0 = more BUY liquidations (long liquidations, bearish day)
        # imbalance < 0 = more SELL liquidations (short squeezes, bullish day)
        liq_buy_heavy  = [t for t in enriched if t["day_liq_imbalance"] > 0.1]
        liq_sell_heavy = [t for t in enriched if t["day_liq_imbalance"] < -0.1]
        liq_balanced   = [t for t in enriched if abs(t["day_liq_imbalance"]) <= 0.1]
        # extreme imbalance
        liq_sell_dom   = [t for t in enriched if t["day_liq_imbalance"] < -0.25]

        all_s = summarize(enriched)
        all_results[rule.name] = {
            "label": label, "n_signals": len(signals),
            "n_closed": len(enriched), "no_fill": no_fill,
            "all": all_s,
            "trend_bull": summarize(trend_bull),
            "trend_bear": summarize(trend_bear),
            "trend_strong_bull": summarize(trend_strong_bull),
            "trend_strong_bear": summarize(trend_strong_bear),
            "range_low": summarize(range_low),
            "range_mid": summarize(range_mid),
            "range_high": summarize(range_high),
            "range_vhigh": summarize(range_vhigh),
            "liq_buy_heavy": summarize(liq_buy_heavy),
            "liq_sell_heavy": summarize(liq_sell_heavy),
            "liq_balanced": summarize(liq_balanced),
            "liq_sell_dom": summarize(liq_sell_dom),
        }

    # ── Write report ───────────────────────────────────────────────────────────
    for rname, res in all_results.items():
        lbl = res["label"]
        nf_pct = res["no_fill"] / res["n_signals"] * 100 if res["n_signals"] else 0
        report_lines += [
            f"\n## {lbl}",
            "",
            f"Signals: {res['n_signals']} | Closed: {res['n_closed']} | "
            f"No-fill: {res['no_fill']} ({nf_pct:.0f}%)",
            "",
            "**Overall:**",
            f"  N={res['all']['n']}  median={_f(res['all']['median'])}  "
            f"mean={_f(res['all']['mean'])}  WR={_pct(res['all']['wr'])}",
        ]

        report_lines += section_table("Day Trend (bullish vs bearish)", [
            ("All",                     res["all"]),
            ("Bull (trend >= 0 bps)",   res["trend_bull"]),
            ("Bear (trend < 0 bps)",    res["trend_bear"]),
            ("Strong Bull (>= +100)",   res["trend_strong_bull"]),
            ("Strong Bear (<= -100)",   res["trend_strong_bear"]),
        ])

        report_lines += section_table("Day Range Buckets", [
            ("< 250 bps  (low vol)",     res["range_low"]),
            ("250-500 bps (medium)",     res["range_mid"]),
            ("500-750 bps (high)",       res["range_high"]),
            (">= 750 bps (very high)",   res["range_vhigh"]),
        ])

        report_lines += section_table("Liq Imbalance (forced-flow direction)", [
            ("Buy-heavy (BUY liq > SELL liq, imbal > 0.1)",  res["liq_buy_heavy"]),
            ("Balanced   (|imbal| <= 0.1)",                   res["liq_balanced"]),
            ("Sell-heavy (SELL liq > BUY liq, imbal < -0.1)", res["liq_sell_heavy"]),
            ("SELL dominant (imbal < -0.25)",                  res["liq_sell_dom"]),
        ])

        report_lines.append("\n---")

    # Interpretation note
    report_lines += [
        "",
        "## Interpretation notes",
        "",
        "- SELL signals fire on forced SHORT COVERING (short squeeze = sell liquidation = SELL liq side).",
        "- `liq_imbalance > 0` = more LONG liquidations (crash day) — BUY-heavy.",
        "- `liq_imbalance < 0` = more SHORT liquidations (squeeze day) — SELL-heavy.",
        "- If SELL signals perform better on sell-heavy days: the signal is riding same-day squeeze momentum.",
        "- If SELL signals perform better on buy-heavy days: the signal is a counter-trend fade of a bounce inside a dump.",
        "- `day_range_bps` is the key vol metric — if the edge is concentrated in high-range days, it's a vol-of-vol bet.",
        "",
        "**This is a research read-only scan. No runner or config changes made.**",
    ]

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"\nReport written to {OUT_MD}")

    # Also print summary to console
    print("\n" + "="*72)
    print("SELL REGIME ANALYSIS — SUMMARY")
    print("="*72)
    for rname, res in all_results.items():
        lbl = res["label"]
        print(f"\n{lbl}")
        print(f"  All:         N={res['all']['n']:3d}  med={_f(res['all']['median'])}  WR={_pct(res['all']['wr'])}")
        print(f"  Bull day:    N={res['trend_bull']['n']:3d}  med={_f(res['trend_bull']['median'])}  WR={_pct(res['trend_bull']['wr'])}")
        print(f"  Bear day:    N={res['trend_bear']['n']:3d}  med={_f(res['trend_bear']['median'])}  WR={_pct(res['trend_bear']['wr'])}")
        print(f"  Range<250:   N={res['range_low']['n']:3d}  med={_f(res['range_low']['median'])}  WR={_pct(res['range_low']['wr'])}")
        print(f"  Range 250-5: N={res['range_mid']['n']:3d}  med={_f(res['range_mid']['median'])}  WR={_pct(res['range_mid']['wr'])}")
        print(f"  Range 500-7: N={res['range_high']['n']:3d}  med={_f(res['range_high']['median'])}  WR={_pct(res['range_high']['wr'])}")
        print(f"  Range>=750:  N={res['range_vhigh']['n']:3d}  med={_f(res['range_vhigh']['median'])}  WR={_pct(res['range_vhigh']['wr'])}")
        print(f"  Liq sell-hvy:N={res['liq_sell_heavy']['n']:3d}  med={_f(res['liq_sell_heavy']['median'])}  WR={_pct(res['liq_sell_heavy']['wr'])}")
        print(f"  Liq buy-hvy: N={res['liq_buy_heavy']['n']:3d}  med={_f(res['liq_buy_heavy']['median'])}  WR={_pct(res['liq_buy_heavy']['wr'])}")

    conn.close()


if __name__ == "__main__":
    main()
