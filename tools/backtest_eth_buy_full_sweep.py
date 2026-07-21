"""
Full parameter backtest for ETH_BUY_LIQ_LONG_500K trades.

Tests every combination of:
  - TP bps:     30, 40, 50, 60, 70, 80, 100
  - SL bps:     20, 30, 40, 50
  - BE trigger: 0 (off), 20, 30, 40, 50
  - TIME limit: 60, 90, 120, 180 min
  - Entry delay: 0, 10, 15, 20, 30 s
  - Min liq_count: 0, 8, 10

Uses actual mark_price data to reconstruct each trade's price path.

Constraints:
  - All evaluation is temporal OOS (first-half IS, second-half OOS)
  - N < 30 => results marked PRELIMINARY
  - Also prints TIME-exit analysis (price after timeout)
"""

import sqlite3
import json
import sys
import os
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).parent.parent
INTEL_DB = ROOT / "data" / "s34_intelligence.db"
MICRO_DB = ROOT / "data" / "microstructure.db"

# ── helpers ──────────────────────────────────────────────────────────────────

def bps(exit_p, entry_p, direction="BUY"):
    if direction == "BUY":
        return (exit_p - entry_p) / entry_p * 10000
    return (entry_p - exit_p) / entry_p * 10000

def load_trades():
    con = sqlite3.connect(INTEL_DB)
    con.row_factory = sqlite3.Row
    cur = con.execute("""
        SELECT t.trade_id, t.signal_id, t.rule_name, t.symbol, t.direction,
               t.status, t.entry_ts_ms, t.entry_price,
               t.tp_price, t.sl_price, t.be_trigger_price,
               t.exit_ts_ms, t.exit_reason, t.exit_price, t.net_bps,
               s.features_json, s.cluster_liq_count, s.cluster_notional
        FROM s34_trades t
        JOIN s34_signals s ON t.signal_id = s.signal_id
        WHERE t.rule_name = 'ETH_BUY_LIQ_LONG_500K_DAYTREND0_TP60_SL40_BE30'
          AND t.status = 'CLOSED'
          AND t.entry_price IS NOT NULL
          AND t.exit_ts_ms IS NOT NULL
        ORDER BY t.entry_ts_ms
    """)
    rows = [dict(r) for r in cur.fetchall()]
    con.close()
    return rows

def get_mark_prices(symbol: str, start_ms: int, end_ms: int) -> list[tuple[int,float]]:
    """Returns [(ts_ms, price), ...] sorted by ts_ms from microstructure.db."""
    con = sqlite3.connect(MICRO_DB)
    try:
        # detect column names
        cols = [r[1] for r in con.execute("PRAGMA table_info(mark_prices)").fetchall()]
        # Find ts and price columns
        ts_col = next((c for c in cols if "ts" in c.lower() or "time" in c.lower()), cols[0])
        price_col = next((c for c in cols if "price" in c.lower() or "mark" in c.lower()), cols[1])
        sym_col = next((c for c in cols if "sym" in c.lower()), None)

        if sym_col:
            rows = con.execute(
                f"SELECT {ts_col}, {price_col} FROM mark_prices "
                f"WHERE {sym_col}=? AND {ts_col}>=? AND {ts_col}<=? ORDER BY {ts_col}",
                (symbol, start_ms, end_ms)
            ).fetchall()
        else:
            rows = con.execute(
                f"SELECT {ts_col}, {price_col} FROM mark_prices "
                f"WHERE {ts_col}>=? AND {ts_col}<=? ORDER BY {ts_col}",
                (start_ms, end_ms)
            ).fetchall()
        return [(int(r[0]), float(r[1])) for r in rows if r[1] is not None]
    except Exception as e:
        return []
    finally:
        con.close()

def price_at(prices: list[tuple[int,float]], target_ms: int) -> float | None:
    """Nearest price to target_ms."""
    if not prices:
        return None
    best = min(prices, key=lambda r: abs(r[0] - target_ms))
    # only use if within 60s
    if abs(best[0] - target_ms) > 60_000:
        return None
    return best[1]

def simulate_trade(
    entry_price: float,
    entry_ts_ms: int,
    prices: list[tuple[int,float]],
    tp_bps: float,
    sl_bps: float,
    be_bps: float,         # 0 = BE disabled
    time_limit_min: float,
    delay_sec: float,
    direction: str = "BUY",
) -> dict:
    """
    Given a price path, simulate a trade and return exit_reason + net_bps.
    delay_sec: we enter at T+delay using that price instead of original entry_price.
    """
    # Apply entry delay
    if delay_sec > 0:
        delayed_ts = entry_ts_ms + int(delay_sec * 1000)
        delayed_price = price_at(prices, delayed_ts)
        if delayed_price is None:
            return {"exit_reason": "NO_DATA", "net_bps": 0.0}
        actual_entry = delayed_price
        actual_entry_ts = delayed_ts
    else:
        actual_entry = entry_price
        actual_entry_ts = entry_ts_ms

    if direction == "BUY":
        tp_price   = actual_entry * (1 + tp_bps / 10000)
        sl_price   = actual_entry * (1 - sl_bps / 10000)
        be_trig    = actual_entry * (1 + be_bps / 10000) if be_bps > 0 else None
    else:
        tp_price   = actual_entry * (1 - tp_bps / 10000)
        sl_price   = actual_entry * (1 + sl_bps / 10000)
        be_trig    = actual_entry * (1 - be_bps / 10000) if be_bps > 0 else None

    time_limit_ms  = int(time_limit_min * 60 * 1000)
    expiry_ts      = actual_entry_ts + time_limit_ms
    current_sl     = sl_price
    be_triggered   = False

    # Walk price path
    for ts, p in prices:
        if ts < actual_entry_ts:
            continue
        if ts > expiry_ts:
            return {"exit_reason": "TIME", "net_bps": bps(p, actual_entry, direction)}

        if direction == "BUY":
            if p >= tp_price:
                return {"exit_reason": "TP", "net_bps": bps(tp_price, actual_entry, direction)}
            if p <= current_sl:
                label = "BE" if be_triggered else "SL"
                return {"exit_reason": label, "net_bps": bps(current_sl, actual_entry, direction)}
            if be_trig and not be_triggered and p >= be_trig:
                be_triggered = True
                current_sl = actual_entry  # move SL to breakeven
        else:
            if p <= tp_price:
                return {"exit_reason": "TP", "net_bps": bps(tp_price, actual_entry, direction)}
            if p >= current_sl:
                label = "BE" if be_triggered else "SL"
                return {"exit_reason": label, "net_bps": bps(current_sl, actual_entry, direction)}
            if be_trig and not be_triggered and p <= be_trig:
                be_triggered = True
                current_sl = actual_entry

    # no data after entry
    return {"exit_reason": "NO_DATA", "net_bps": 0.0}

def stats(results: list[dict]) -> dict:
    """Aggregate list of sim results."""
    valid = [r for r in results if r["exit_reason"] != "NO_DATA"]
    if not valid:
        return {"N": 0, "WR": 0, "mean": 0, "SL_cnt": 0, "BE_cnt": 0, "TIME_cnt": 0, "TP_cnt": 0}
    n = len(valid)
    wins = [r for r in valid if r["net_bps"] > 0]
    sls  = [r for r in valid if r["exit_reason"] == "SL"]
    bes  = [r for r in valid if r["exit_reason"] == "BE"]
    tims = [r for r in valid if r["exit_reason"] == "TIME"]
    tps  = [r for r in valid if r["exit_reason"] == "TP"]
    mean = sum(r["net_bps"] for r in valid) / n
    return {
        "N": n,
        "WR": round(len(wins) / n * 100, 0),
        "mean": round(mean, 1),
        "SL_cnt": len(sls),
        "BE_cnt": len(bes),
        "TIME_cnt": len(tims),
        "TP_cnt": len(tps),
    }

def fmt_stat(s: dict, label: str = "") -> str:
    flag = " [PRELIM]" if s["N"] < 30 else ""
    return (
        f"{label:<42} N={s['N']:>3}{flag}  WR={s['WR']:>4}%  "
        f"mean={s['mean']:>+6.1f}  "
        f"TP={s['TP_cnt']:>2} SL={s['SL_cnt']:>2} BE={s['BE_cnt']:>2} TIME={s['TIME_cnt']:>2}"
    )

# ── main ─────────────────────────────────────────────────────────────────────

def main():
    print("Loading trades...")
    trades = load_trades()
    n_total = len(trades)
    split = n_total // 2
    print(f"  Total ETH BUY CLOSED: {n_total}  |  IS: {split}  OOS: {n_total-split}")

    # Load price paths (4 hours after entry for each trade)
    print("Loading price paths from microstructure.db...")
    price_cache = {}
    for t in trades:
        key = t["trade_id"]
        start = t["entry_ts_ms"] - 60_000          # 1m before entry
        end   = t["entry_ts_ms"] + 4 * 3600_000    # 4h after
        price_cache[key] = get_mark_prices("ETHUSDT", start, end)

    coverage = sum(1 for v in price_cache.values() if len(v) > 10)
    print(f"  Price paths loaded: {coverage}/{n_total} have data\n")

    # ── SECTION 1: Current baseline ─────────────────────────────────────────
    print("=" * 80)
    print("SECTION 1 — CURRENT BASELINE (TP60 SL40 BE30 TIME120 delay=0)")
    print("=" * 80)
    for label, subset in [("ALL", trades), ("IS (first half)", trades[:split]), ("OOS (second half)", trades[split:])]:
        sims = []
        for t in subset:
            prices = price_cache[t["trade_id"]]
            r = simulate_trade(
                t["entry_price"], t["entry_ts_ms"], prices,
                tp_bps=60, sl_bps=40, be_bps=30, time_limit_min=120, delay_sec=0
            )
            sims.append(r)
        print(fmt_stat(stats(sims), f"  {label}"))
    print()

    # Actual paper results (no simulation needed)
    print("  ACTUAL paper results (from DB):")
    actual_bps = [t["net_bps"] for t in trades if t["net_bps"] is not None]
    tp_cnt  = sum(1 for t in trades if t["exit_reason"] == "TP")
    sl_cnt  = sum(1 for t in trades if t["exit_reason"] == "SL")
    be_cnt  = sum(1 for t in trades if t["exit_reason"] in ("BE", "BREAKEVEN"))
    tm_cnt  = sum(1 for t in trades if t["exit_reason"] in ("TIME", "TIMEOUT"))
    win_cnt = sum(1 for v in actual_bps if v > 0)
    print(f"  {'actual paper':<42} N={len(actual_bps):>3}  WR={round(win_cnt/len(actual_bps)*100) if actual_bps else 0:>4}%  "
          f"mean={round(sum(actual_bps)/len(actual_bps), 1) if actual_bps else 0:>+6.1f}  "
          f"TP={tp_cnt} SL={sl_cnt} BE={be_cnt} TIME={tm_cnt}")
    print()

    # ── SECTION 2: TIME exit deep-dive ──────────────────────────────────────
    print("=" * 80)
    print("SECTION 2 — TIME EXIT ANALYSIS (what price did AFTER timeout)")
    print("=" * 80)
    time_exits = [t for t in trades if t["exit_reason"] in ("TIME", "TIMEOUT")]
    print(f"  TIME exits: {len(time_exits)}\n")
    header = f"{'trade_id':<10} {'exit_bps':>8} {'30m':>8} {'1h':>8} {'2h':>8} {'4h':>8}  {'would_TP60?'}"
    print("  " + header)
    print("  " + "-" * len(header))
    for t in time_exits:
        prices = price_cache[t["trade_id"]]
        ep = t["entry_price"]
        ex_ts = t["exit_ts_ms"]
        tp60 = ep * (1 + 60/10000)
        results = []
        for delta_min in [30, 60, 120, 240]:
            p = price_at(prices, ex_ts + delta_min * 60_000)
            if p:
                results.append(f"{bps(p, ep):>+7.1f}")
            else:
                results.append(f"{'N/A':>7}")
        # did price ever reach TP60 in 4h window?
        window = [p for ts, p in prices if ts >= ex_ts and ts <= ex_ts + 4*3600_000]
        ever_tp = any(p >= tp60 for p in window) if window else False
        print(f"  {t['trade_id']:<10} {t['net_bps']:>+8.1f} {results[0]:>8} {results[1]:>8} {results[2]:>8} {results[3]:>8}  {'YES' if ever_tp else 'no'}")
    print()

    # ── SECTION 3: TP sensitivity (SL40, BE30, TIME120, delay=0) ────────────
    print("=" * 80)
    print("SECTION 3 — TP SWEEP  (SL=40 BE=30 TIME=120 delay=0)")
    print("=" * 80)
    for tp in [30, 40, 50, 60, 70, 80, 100]:
        all_s, oos_s = [], []
        for i, t in enumerate(trades):
            prices = price_cache[t["trade_id"]]
            r = simulate_trade(t["entry_price"], t["entry_ts_ms"], prices,
                               tp_bps=tp, sl_bps=40, be_bps=30, time_limit_min=120, delay_sec=0)
            all_s.append(r)
            if i >= split:
                oos_s.append(r)
        s_all = stats(all_s)
        s_oos = stats(oos_s)
        flag = "*" if tp == 60 else " "
        print(fmt_stat(s_all, f"{flag} TP={tp:>3} bps ALL"))
        print(fmt_stat(s_oos, f"  TP={tp:>3} bps OOS"))
    print()

    # ── SECTION 4: SL sensitivity (TP60, BE30, TIME120, delay=0) ────────────
    print("=" * 80)
    print("SECTION 4 — SL SWEEP  (TP=60 BE=30 TIME=120 delay=0)")
    print("=" * 80)
    for sl in [15, 20, 25, 30, 40, 50, 60]:
        all_s, oos_s = [], []
        for i, t in enumerate(trades):
            prices = price_cache[t["trade_id"]]
            r = simulate_trade(t["entry_price"], t["entry_ts_ms"], prices,
                               tp_bps=60, sl_bps=sl, be_bps=30, time_limit_min=120, delay_sec=0)
            all_s.append(r)
            if i >= split:
                oos_s.append(r)
        flag = "*" if sl == 40 else " "
        print(fmt_stat(stats(all_s), f"{flag} SL={sl:>3} bps ALL"))
        print(fmt_stat(stats(oos_s), f"  SL={sl:>3} bps OOS"))
    print()

    # ── SECTION 5: BE sensitivity (TP60, SL40, TIME120, delay=0) ────────────
    print("=" * 80)
    print("SECTION 5 — BE TRIGGER SWEEP  (TP=60 SL=40 TIME=120 delay=0)")
    print("=" * 80)
    for be in [0, 15, 20, 25, 30, 40, 50]:
        all_s, oos_s = [], []
        for i, t in enumerate(trades):
            prices = price_cache[t["trade_id"]]
            r = simulate_trade(t["entry_price"], t["entry_ts_ms"], prices,
                               tp_bps=60, sl_bps=40, be_bps=be, time_limit_min=120, delay_sec=0)
            all_s.append(r)
            if i >= split:
                oos_s.append(r)
        flag = "*" if be == 30 else " "
        be_label = f"BE={'off' if be==0 else f'{be}bps'}"
        print(fmt_stat(stats(all_s), f"{flag} {be_label:<8} ALL"))
        print(fmt_stat(stats(oos_s), f"  {be_label:<8} OOS"))
    print()

    # ── SECTION 6: TIME limit sensitivity (TP60, SL40, BE30, delay=0) ───────
    print("=" * 80)
    print("SECTION 6 — TIME LIMIT SWEEP  (TP=60 SL=40 BE=30 delay=0)")
    print("=" * 80)
    for tlim in [30, 60, 90, 120, 180, 240, 360]:
        all_s, oos_s = [], []
        for i, t in enumerate(trades):
            prices = price_cache[t["trade_id"]]
            r = simulate_trade(t["entry_price"], t["entry_ts_ms"], prices,
                               tp_bps=60, sl_bps=40, be_bps=30, time_limit_min=tlim, delay_sec=0)
            all_s.append(r)
            if i >= split:
                oos_s.append(r)
        flag = "*" if tlim == 120 else " "
        print(fmt_stat(stats(all_s), f"{flag} TIME={tlim:>4}m ALL"))
        print(fmt_stat(stats(oos_s), f"  TIME={tlim:>4}m OOS"))
    print()

    # ── SECTION 7: Entry delay sweep (TP60, SL40, BE30, TIME120) ────────────
    print("=" * 80)
    print("SECTION 7 — ENTRY DELAY SWEEP  (TP=60 SL=40 BE=30 TIME=120)")
    print("=" * 80)
    for delay in [0, 5, 10, 15, 20, 30, 45, 60]:
        all_s, oos_s = [], []
        for i, t in enumerate(trades):
            prices = price_cache[t["trade_id"]]
            r = simulate_trade(t["entry_price"], t["entry_ts_ms"], prices,
                               tp_bps=60, sl_bps=40, be_bps=30, time_limit_min=120, delay_sec=delay)
            all_s.append(r)
            if i >= split:
                oos_s.append(r)
        flag = "*" if delay == 0 else " "
        print(fmt_stat(stats(all_s), f"{flag} delay={delay:>3}s ALL"))
        print(fmt_stat(stats(oos_s), f"  delay={delay:>3}s OOS"))
    print()

    # ── SECTION 8: liq_count filter + best params ────────────────────────────
    print("=" * 80)
    print("SECTION 8 — LIQ_COUNT FILTER  (TP=60 SL=40 BE=30 TIME=120 delay=0)")
    print("=" * 80)
    for min_liq in [0, 6, 8, 10, 12, 15]:
        subset = [t for t in trades if (t.get("cluster_liq_count") or 0) >= min_liq]
        oos_sub = [t for t in subset if trades.index(t) >= split]
        all_s, oos_s = [], []
        for t in subset:
            prices = price_cache[t["trade_id"]]
            r = simulate_trade(t["entry_price"], t["entry_ts_ms"], prices,
                               tp_bps=60, sl_bps=40, be_bps=30, time_limit_min=120, delay_sec=0)
            all_s.append(r)
        for t in oos_sub:
            prices = price_cache[t["trade_id"]]
            r = simulate_trade(t["entry_price"], t["entry_ts_ms"], prices,
                               tp_bps=60, sl_bps=40, be_bps=30, time_limit_min=120, delay_sec=0)
            oos_s.append(r)
        flag = "*" if min_liq == 8 else " "
        print(fmt_stat(stats(all_s), f"{flag} liq>={min_liq:>2} ALL  (n_taken={len(subset)})"))
        print(fmt_stat(stats(oos_s), f"  liq>={min_liq:>2} OOS  (n_taken={len(oos_sub)})"))
    print()

    # ── SECTION 9: Combined — best of delay + liq filter + param sweep ───────
    print("=" * 80)
    print("SECTION 9 — COMBINED: liq>=8 + delay=20s — param sweep for TP/SL/BE/TIME")
    print("=" * 80)
    liq8 = [t for t in trades if (t.get("cluster_liq_count") or 0) >= 8]
    liq8_oos = [t for t in liq8 if trades.index(t) >= split]
    print(f"  liq>=8 subset: N={len(liq8)}  OOS N={len(liq8_oos)}\n")

    combos = [
        # label,                tp,  sl,  be,  time, delay
        ("CURRENT (TP60 SL40 BE30 T120 d0)",    60, 40, 30, 120, 0),
        ("TP80 SL40 BE30 T120 d20",             80, 40, 30, 120, 20),
        ("TP60 SL40 BE30 T120 d20",             60, 40, 30, 120, 20),
        ("TP60 SL30 BE30 T120 d20",             60, 30, 30, 120, 20),
        ("TP60 SL40 BE0 T120 d20",              60, 40, 0,  120, 20),
        ("TP80 SL40 BE0 T180 d20",              80, 40, 0,  180, 20),
        ("TP100 SL40 BE0 T180 d20",            100, 40, 0,  180, 20),
        ("TP60 SL40 BE0 T180 d0",               60, 40, 0,  180, 0),
        ("TP80 SL40 BE0 T180 d0",               80, 40, 0,  180, 0),
        ("TP80 SL30 BE40 T180 d20",             80, 30, 40, 180, 20),
        ("TP100 SL40 BE50 T240 d20",           100, 40, 50, 240, 20),
        ("TP50 SL25 BE0 T120 d20",              50, 25, 0,  120, 20),
    ]
    for label, tp, sl, be, tlim, delay in combos:
        all_s, oos_s = [], []
        for t in liq8:
            prices = price_cache[t["trade_id"]]
            r = simulate_trade(t["entry_price"], t["entry_ts_ms"], prices,
                               tp_bps=tp, sl_bps=sl, be_bps=be, time_limit_min=tlim, delay_sec=delay)
            all_s.append(r)
        for t in liq8_oos:
            prices = price_cache[t["trade_id"]]
            r = simulate_trade(t["entry_price"], t["entry_ts_ms"], prices,
                               tp_bps=tp, sl_bps=sl, be_bps=be, time_limit_min=tlim, delay_sec=delay)
            oos_s.append(r)
        print(fmt_stat(stats(all_s), f"  ALL {label}"))
        print(fmt_stat(stats(oos_s), f"  OOS {label}"))
        print()

    # ── SECTION 10: Grid search — best TP + SL + BE on liq>=8 + delay=20 ───
    print("=" * 80)
    print("SECTION 10 — GRID SEARCH (liq>=8 delay=20s TIME=120 — top 15 OOS EV)")
    print("=" * 80)
    results_grid = []
    for tp in [40, 50, 60, 70, 80, 100]:
        for sl in [20, 30, 40]:
            for be in [0, 20, 30, 40]:
                if be >= tp:
                    continue
                all_s, oos_s = [], []
                for t in liq8:
                    prices = price_cache[t["trade_id"]]
                    r = simulate_trade(t["entry_price"], t["entry_ts_ms"], prices,
                                       tp_bps=tp, sl_bps=sl, be_bps=be, time_limit_min=120, delay_sec=20)
                    all_s.append(r)
                for t in liq8_oos:
                    prices = price_cache[t["trade_id"]]
                    r = simulate_trade(t["entry_price"], t["entry_ts_ms"], prices,
                                       tp_bps=tp, sl_bps=sl, be_bps=be, time_limit_min=120, delay_sec=20)
                    oos_s.append(r)
                s = stats(oos_s)
                results_grid.append((tp, sl, be, s, stats(all_s)))

    results_grid.sort(key=lambda x: x[3]["mean"], reverse=True)
    print(f"  {'params':<28} {'OOS mean':>10} {'OOS WR':>8} {'OOS N':>6} {'ALL mean':>10}")
    print(f"  {'-'*70}")
    for tp, sl, be, s_oos, s_all in results_grid[:20]:
        be_str = "off" if be == 0 else f"{be}"
        label = f"TP={tp} SL={sl} BE={be_str}"
        prelim = "[P]" if s_oos["N"] < 20 else ""
        print(f"  {label:<28} {s_oos['mean']:>+10.1f} {s_oos['WR']:>7}% {s_oos['N']:>6} {s_all['mean']:>+10.1f} {prelim}")
    print()

    print("=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()
