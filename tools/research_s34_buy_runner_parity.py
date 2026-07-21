"""
S34 BUY Rule Runner Parity
Exact runner-parity simulation for all active BUY rules.

Tests whether BUY rules also suffer from the timing gap found in SELL rules
(runner enters LONG after BUY liq cascade has already pushed price up).

Rules tested:
  PRE-REG MAIN (4):
    ETH_BUY_500K_DAYTREND0, SOL_BUY_200K, SOL_BUY_100K, BTC_BUY_1M_DISTRIBUTED
  EXPLORATORY (3):
    ETH_BUY_200K, ETH_BUY_200K_BTC_PRE15, ETH_BUY_500K_NEGTREND_STRETCHED

No runner/config/pre-reg changes. Research only.
"""
from __future__ import annotations
import json
import math
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import s34_shadow_paper_runner as runner

_REGIME_CFG = runner.RegimeConfig(enabled=True)

DB_PATH  = ROOT / "data" / "microstructure.db"
FF_PATH  = ROOT / "data" / "s34_feature_factory.db"
OUT_DIR  = ROOT / "reports" / "research" / "s34"
OUT_MD   = OUT_DIR / "S34_BUY_RUNNER_PARITY.md"
OUT_JSON = OUT_DIR / "S34_BUY_RUNNER_PARITY.json"

PRELIMINARY_N     = 30
NO_FILL_THRESHOLD = 0.40

# ── Rule definitions (exact live config) ─────────────────────────────────────

RULES = [
    # --- PRE-REG MAIN ---
    {
        "label": "ETH_BUY_500K_DAYTREND0",
        "group": "MAIN",
        "ff_route": "LONG_DELAY0_TP60",
        "ff_threshold": 500_000,
        "rule": runner.S34Rule(
            name="ETH_BUY_LIQ_LONG_500K_DAYTREND0_TP60_SL40_BE30",
            symbol="ETHUSDT", liq_side="BUY", direction="LONG",
            threshold_usd=500_000.0,
            tp_bps=60.0, sl_bps=40.0, be_trigger_bps=30.0,
            max_horizon_sec=3600, entry_delay_sec=0, min_gap_sec=900,
            min_day_trend_bps=0.0,
            use_global_regime=False, taker_fee_bps=4.0,
            require_book_ticker_fill=True,
        ),
    },
    {
        "label": "SOL_BUY_200K",
        "group": "MAIN",
        "ff_route": "LONG_DELAY0_TP60",
        "ff_threshold": 200_000,
        "rule": runner.S34Rule(
            name="SOL_BUY_LIQ_LONG_200K_TP60_SL40_BE30",
            symbol="SOLUSDT", liq_side="BUY", direction="LONG",
            threshold_usd=200_000.0,
            tp_bps=60.0, sl_bps=40.0, be_trigger_bps=30.0,
            max_horizon_sec=3600, entry_delay_sec=0, min_gap_sec=900,
            use_global_regime=False, taker_fee_bps=4.0,
            require_book_ticker_fill=True,
        ),
    },
    {
        "label": "SOL_BUY_100K",
        "group": "MAIN",
        "ff_route": "LONG_DELAY0_TP60",
        "ff_threshold": 100_000,
        "rule": runner.S34Rule(
            name="SOL_BUY_LIQ_LONG_100K_TP60_SL40_BE30",
            symbol="SOLUSDT", liq_side="BUY", direction="LONG",
            threshold_usd=100_000.0,
            tp_bps=60.0, sl_bps=40.0, be_trigger_bps=30.0,
            max_horizon_sec=3600, entry_delay_sec=0, min_gap_sec=900,
            use_global_regime=False, taker_fee_bps=4.0,
            require_book_ticker_fill=True,
        ),
    },
    {
        "label": "BTC_BUY_1M_DISTRIBUTED",
        "group": "MAIN",
        "ff_route": "LONG_DELAY0_TP60",
        "ff_threshold": 1_000_000,
        "rule": runner.S34Rule(
            name="BTC_BUY_LIQ_LONG_1M_DISTRIBUTED_TP60_SL30_BE30",
            symbol="BTCUSDT", liq_side="BUY", direction="LONG",
            threshold_usd=1_000_000.0,
            tp_bps=60.0, sl_bps=30.0, be_trigger_bps=30.0,
            max_horizon_sec=3600, entry_delay_sec=0, min_gap_sec=900,
            max_single_liq_share_pct=50.0,
            use_global_regime=False, taker_fee_bps=4.0,
            require_book_ticker_fill=True,
        ),
    },
    # --- EXPLORATORY ---
    {
        "label": "ETH_BUY_200K",
        "group": "EXPLORATORY",
        "ff_route": "LONG_DELAY0_TP60",
        "ff_threshold": 200_000,
        "rule": runner.S34Rule(
            name="ETH_BUY_LIQ_LONG_200K_TP60_SL40_BE30",
            symbol="ETHUSDT", liq_side="BUY", direction="LONG",
            threshold_usd=200_000.0,
            tp_bps=60.0, sl_bps=40.0, be_trigger_bps=30.0,
            max_horizon_sec=3600, entry_delay_sec=0, min_gap_sec=900,
            use_global_regime=False, taker_fee_bps=4.0,
            require_book_ticker_fill=True,
        ),
    },
    {
        "label": "ETH_BUY_200K_BTC_PRE15_DELAY60",
        "group": "EXPLORATORY",
        "ff_route": "LONG_DELAY60_TP120",
        "ff_threshold": 200_000,
        "rule": runner.S34Rule(
            name="ETH_BUY_LIQ_LONG_200K_BTC_PRE15_TP120_SL40_BE30_DELAY60",
            symbol="ETHUSDT", liq_side="BUY", direction="LONG",
            threshold_usd=200_000.0,
            tp_bps=120.0, sl_bps=40.0, be_trigger_bps=30.0,
            max_horizon_sec=3600, entry_delay_sec=60, min_gap_sec=900,
            btc_pre_window_sec=900, btc_pre_min_return_bps=0.0,
            use_global_regime=False, taker_fee_bps=4.0,
            require_book_ticker_fill=True,
        ),
    },
    {
        "label": "ETH_BUY_500K_NEGTREND_STRETCHED",
        "group": "EXPLORATORY",
        "ff_route": "LONG_DELAY0_TP60",
        "ff_threshold": 500_000,
        "rule": runner.S34Rule(
            name="ETH_BUY_LIQ_LONG_500K_NEGTREND_STRETCHED_TP60_SL40_BE30",
            symbol="ETHUSDT", liq_side="BUY", direction="LONG",
            threshold_usd=500_000.0,
            tp_bps=60.0, sl_bps=40.0, be_trigger_bps=30.0,
            max_horizon_sec=3600, entry_delay_sec=0, min_gap_sec=900,
            max_day_trend_bps=0.0,
            required_shape_label="stretched_120s",
            use_global_regime=False, taker_fee_bps=4.0,
            require_book_ticker_fill=True,
        ),
    },
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

def _r1(v): return round(float(v), 1) if v is not None and math.isfinite(float(v)) else None


# ── Cascade timing gap ────────────────────────────────────────────────────────

def _cascade_timing_gap(ff_conn, symbol: str, liq_side: str, threshold: float, ff_route: str) -> dict | None:
    """Measure price move from first_ts to cluster_end_ts for FF events."""
    rows = ff_conn.execute("""
        SELECT e.event_ts_ms, e.cluster_end_ts_ms, l.entry_price
        FROM liq_event_features e
        JOIN liq_event_outcome_labels l ON e.event_id = l.event_id
        WHERE e.symbol=? AND e.liq_side=? AND e.cluster_notional>=? AND l.route_id=?
        ORDER BY e.event_ts_ms
    """, (symbol, liq_side, threshold, ff_route)).fetchall()

    micro = sqlite3.connect(f"file:{DB_PATH.as_posix()}?mode=ro", uri=True)
    moves = []
    for evt_ts, end_ts, ff_entry in rows:
        m = micro.execute(
            "SELECT mark_price FROM mark_prices WHERE symbol=? AND ts_ms<=? ORDER BY ts_ms DESC LIMIT 1",
            (symbol, end_ts)
        ).fetchone()
        if m and ff_entry:
            # BUY liq -> price drops during cascade (SHORT positions force-closed, but wait...)
            # BUY liq = SHORT positions being liquidated = forced BUYING = price RISES
            # So: mark_at_end > ff_entry means price rose during cascade
            move = (m[0] - ff_entry) / ff_entry * 10000  # + = price rose
            moves.append(move)
    micro.close()

    if not moves:
        return None
    return {
        "n": len(moves),
        "median_move": _r1(_median(moves)),
        "mean_move": _r1(_mean(moves)),
    }


# ── Feature factory median ────────────────────────────────────────────────────

def _ff_median(ff_conn, symbol: str, liq_side: str, threshold: float, ff_route: str) -> float | None:
    rows = ff_conn.execute("""
        SELECT l.net_bps
        FROM liq_event_features e
        JOIN liq_event_outcome_labels l ON e.event_id = l.event_id
        WHERE e.symbol=? AND e.liq_side=? AND e.cluster_notional>=? AND l.route_id=?
    """, (symbol, liq_side, threshold, ff_route)).fetchall()
    nets = [float(r[0]) for r in rows if r[0] is not None]
    return _r1(_median(nets)) if nets else None


# ── Simulate rule ─────────────────────────────────────────────────────────────

def _run_rule(conn, rule_def: dict) -> dict:
    rule = rule_def["rule"]
    ts = conn.execute("SELECT MIN(ts_ms), MAX(ts_ms) FROM liquidations WHERE symbol=?", (rule.symbol,)).fetchone()
    start_ms, end_ms = int(ts[0]), int(ts[1])

    signals = runner._bucket_events(conn, rule, start_ms, end_ms, limit=10_000)
    total = len(signals)
    if total == 0:
        return {"total": 0, "no_fill": 0, "regime_filtered": 0, "risk_filtered": 0, "trades": []}

    # all_trades accumulates open+closed trades; passed to _risk_gate for state checks
    all_trades: dict[str, dict] = {}
    trade_counter = 0
    no_fill, regime_filtered, risk_filtered = 0, 0, 0

    for sig in signals:
        sig_ts = int(sig["ts_ms"])

        # Advance open trades to current signal time (may close them)
        for tid in list(all_trades):
            if all_trades[tid].get("status") == "OPEN":
                try:
                    all_trades[tid] = runner._evaluate_trade(conn, all_trades[tid], sig_ts)
                except RuntimeError as exc:
                    all_trades[tid]["status"] = "EXIT_FILL_ERROR"
                    all_trades[tid]["exit_fill_error"] = str(exc)

        if sig.get("fill_error"):
            no_fill += 1
            continue

        regime_ok, _, _ = runner._regime_gate(conn, rule, sig, _REGIME_CFG)
        if not regime_ok:
            regime_filtered += 1
            continue

        risk_ok, _ = runner._risk_gate(all_trades, sig, runner.DEFAULT_RISK, rule)
        if not risk_ok:
            risk_filtered += 1
            continue

        trade = runner._paper_trade_from_signal(rule, sig, runner.DEFAULT_RISK)
        trade_counter += 1
        all_trades[str(trade_counter)] = trade

    # Close any still-open trades at end of data
    for tid in list(all_trades):
        if all_trades[tid].get("status") == "OPEN":
            try:
                all_trades[tid] = runner._evaluate_trade(conn, all_trades[tid], end_ms)
            except RuntimeError as exc:
                all_trades[tid]["status"] = "EXIT_FILL_ERROR"
                all_trades[tid]["exit_fill_error"] = str(exc)

    return {
        "total": total,
        "no_fill": no_fill,
        "regime_filtered": regime_filtered,
        "risk_filtered": risk_filtered,
        "trades": list(all_trades.values()),
    }


# ── Metrics ───────────────────────────────────────────────────────────────────

def _metrics(trades: list[dict]) -> dict:
    closed = [t for t in trades if t.get("status") == "CLOSED" and t.get("net_bps") is not None]
    if not closed:
        return {"n": 0, "insufficient": True}
    nets = [float(t["net_bps"]) for t in closed]
    em = {"TP": 0, "SL": 0, "BE": 0, "TIME": 0}
    for t in closed:
        r = str(t.get("exit_reason") or "")
        if r in em:
            em[r] += 1
    half = len(closed) // 2
    h1 = [float(t["net_bps"]) for t in closed[:half]]
    h2 = [float(t["net_bps"]) for t in closed[half:]]

    holds = []
    for t in closed:
        e, x = t.get("entry_ts_ms"), t.get("exit_ts_ms")
        if e and x:
            holds.append((int(x) - int(e)) / 1000)

    return {
        "n":         len(closed),
        "median":    _r1(_median(nets)),
        "mean":      _r1(_mean(nets)),
        "cum":       _r1(sum(nets)),
        "top3r_cum": _r1(sum(sorted(nets)[3:]) if len(nets) > 3 else sum(nets)),
        "wr":        round(sum(1 for n in nets if n > 0) / len(nets), 3),
        "exits":     em,
        "avg_hold":  _r1(_mean(holds)),
        "h1_median": _r1(_median(h1)),
        "h1_cum":    _r1(sum(h1)),
        "h2_median": _r1(_median(h2)),
        "h2_cum":    _r1(sum(h2)),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    conn    = sqlite3.connect(f"file:{DB_PATH.as_posix()}?mode=ro", uri=True)
    ff_conn = sqlite3.connect(f"file:{FF_PATH.as_posix()}?mode=ro", uri=True)

    now = datetime.now(timezone.utc).isoformat()
    all_rows = []

    print(f"S34 BUY Runner Parity — {now}\n")

    for rd in RULES:
        label    = rd["label"]
        group    = rd["group"]
        rule     = rd["rule"]
        ff_route = rd["ff_route"]
        ff_thr   = rd["ff_threshold"]

        ff_med = _ff_median(ff_conn, rule.symbol, rule.liq_side, ff_thr, ff_route)
        timing = _cascade_timing_gap(ff_conn, rule.symbol, rule.liq_side, ff_thr, ff_route)

        res = _run_rule(conn, rd)
        m   = _metrics(res["trades"])

        nf_rate  = res["no_fill"] / res["total"] if res["total"] else 0
        prelim   = res["total"] < PRELIMINARY_N or nf_rate > NO_FILL_THRESHOLD

        if m.get("insufficient"):
            print(f"[{group}] {label}: NO DATA (signals={res['total']}, no_fill={res['no_fill']})")
            continue

        timing_str = f"cascade_move={timing['median_move']:+.1f}bps" if timing else "cascade_move=N/A"
        prelim_str = " [PRELIM]" if prelim else ""
        delta_str  = f"delta={m['median'] - (ff_med or 0):+.1f}" if ff_med is not None else ""

        print(
            f"[{group}] {label}{prelim_str}\n"
            f"  sigs={res['total']}  nf={res['no_fill']}({nf_rate*100:.0f}%)  regime_filtered={res['regime_filtered']}  risk_filtered={res['risk_filtered']}  closed={m['n']}\n"
            f"  FF median={ff_med:+.1f}  runner median={m['median']:+.1f}  {delta_str}\n"
            f"  {timing_str}  WR={m['wr']*100:.0f}%  cum={m['cum']:+.1f}  top3r={m['top3r_cum']:+.1f}\n"
            f"  h1={m['h1_median']:+.1f}/{m['h1_cum']:+.1f}  h2={m['h2_median']:+.1f}/{m['h2_cum']:+.1f}\n"
            f"  exits: TP={m['exits']['TP']} SL={m['exits']['SL']} BE={m['exits']['BE']} TIME={m['exits']['TIME']}\n"
        )

        all_rows.append({
            "label":          label,
            "group":          group,
            "ff_median":      ff_med,
            "runner_median":  m["median"],
            "delta":          _r1(m["median"] - (ff_med or 0)) if ff_med is not None else None,
            "cascade_timing": timing,
            "total_signals":    res["total"],
            "no_fill":          res["no_fill"],
            "regime_filtered":  res["regime_filtered"],
            "risk_filtered":    res["risk_filtered"],
            "no_fill_rate":     round(nf_rate, 3),
            "preliminary":    prelim,
            "metrics":        m,
        })

    conn.close()
    ff_conn.close()

    # ── Summary ───────────────────────────────────────────────────────────────
    print("=== SUMMARY ===")
    viable = [r for r in all_rows if not r["preliminary"] and (r["metrics"].get("median") or 0) > 0]
    print(f"Viable (N>={PRELIMINARY_N}, NF<{NO_FILL_THRESHOLD*100:.0f}%, median>0): {len(viable)}")
    for r in all_rows:
        m = r["metrics"]
        flag = "VIABLE" if not r["preliminary"] and (m.get("median") or 0) > 0 else ("PRELIM" if r["preliminary"] else "NEG")
        print(f"  [{flag}] {r['label']}: FF={r['ff_median']:+.1f} runner={m.get('median'):+.1f} h1={m.get('h1_median'):+.1f} h2={m.get('h2_median'):+.1f}")

    # ── Write outputs ─────────────────────────────────────────────────────────
    lines = [
        "# S34 BUY Rule Runner Parity",
        "",
        f"Generated: `{now}`",
        "",
        "Exact runner-parity for all active BUY rules. Tests timing-gap hypothesis:",
        "does runner's late entry (at threshold cross vs feature factory's first_ts) hurt BUY rules?",
        "",
        "| Rule | Group | FF Median | Runner Median | Delta | Cascade Move | WR | H1 | H2 | Prelim |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for r in all_rows:
        m  = r["metrics"]
        tm = r["cascade_timing"]
        p  = "yes" if r["preliminary"] else ""
        cm = f"{tm['median_move']:+.1f}" if tm else "N/A"
        lines.append(
            f"| {r['label']} | {r['group']} "
            f"| {r['ff_median']:+.1f} | {m.get('median'):+.1f} | {r['delta']:+.1f} "
            f"| {cm} | {m['wr']*100:.0f}% "
            f"| {m.get('h1_median'):+.1f} | {m.get('h2_median'):+.1f} | {p} |"
        )

    lines += [
        "",
        "## Interpretation",
        "",
        "- **Delta** = runner median − FF median. Negative delta = runner underperforms FF (timing gap penalty).",
        "- **Cascade Move** = median price move from first_ts to cluster_end_ts during cascade. "
          "For BUY liq (SHORT liquidations → forced buying), price should RISE during cascade.",
        "- If cascade_move is large AND delta is similarly large negative, timing gap is the cause.",
        "- VIABLE = N>=30, NF<40%, runner median>0.",
        "",
        "## Viable Rules",
        "",
    ]
    if viable:
        for r in viable:
            m = r["metrics"]
            lines += [
                f"### {r['label']}",
                f"- Runner median={m['median']:+.1f}, WR={m['wr']*100:.0f}%, N={m['n']}",
                f"- H1={m['h1_median']:+.1f}, H2={m['h2_median']:+.1f}",
                f"- Exits: TP={m['exits']['TP']} SL={m['exits']['SL']} BE={m['exits']['BE']} TIME={m['exits']['TIME']}",
                "",
            ]
    else:
        lines.append("None.")

    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    OUT_JSON.write_text(json.dumps({"generated_at": now, "results": all_rows}, indent=2), encoding="utf-8")
    print(f"\nMD  : {OUT_MD}")
    print(f"JSON: {OUT_JSON}")


if __name__ == "__main__":
    main()
