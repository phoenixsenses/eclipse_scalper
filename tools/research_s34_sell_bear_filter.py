"""
S34 SELL Bear-Day Filter Test
Runner-parity simulation: SELL rules with and without max_day_trend_bps=0.0.

BUY rules work because min_day_trend_bps=0.0 filters to bullish days (64% of
signals rejected). Testing whether the parallel filter for SELL (bearish days
only) rescues the negative SELL performance.

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

DB_PATH  = ROOT / "data" / "microstructure.db"
OUT_DIR  = ROOT / "reports" / "research" / "s34"
OUT_MD   = OUT_DIR / "S34_SELL_BEAR_FILTER.md"
OUT_JSON = OUT_DIR / "S34_SELL_BEAR_FILTER.json"

PRELIMINARY_N     = 30
NO_FILL_THRESHOLD = 0.40

SELL_RULES = [
    {
        "label": "ETH_SELL_500K",
        "base": runner.S34Rule(
            name="ETH_SELL_LIQ_SHORT_500K_TP60_SL40_BE40",
            liq_side="SELL", direction="SHORT",
            threshold_usd=500_000.0, tp_bps=60.0, sl_bps=40.0, be_trigger_bps=40.0,
            max_horizon_sec=1800, use_global_regime=False, taker_fee_bps=4.0,
            require_book_ticker_fill=True,
        ),
    },
    {
        "label": "ETH_SELL_1M",
        "base": runner.S34Rule(
            name="ETH_SELL_LIQ_SHORT_1M_TP80_SL40_BE40",
            liq_side="SELL", direction="SHORT",
            threshold_usd=1_000_000.0, tp_bps=80.0, sl_bps=40.0, be_trigger_bps=40.0,
            max_horizon_sec=1800, use_global_regime=False, taker_fee_bps=4.0,
            require_book_ticker_fill=True,
        ),
    },
    {
        "label": "SOL_SELL_200K",
        "base": runner.S34Rule(
            name="SOL_SELL_LIQ_SHORT_200K_TP60_SL30_BE30",
            symbol="SOLUSDT", liq_side="SELL", direction="SHORT",
            threshold_usd=200_000.0, tp_bps=60.0, sl_bps=30.0, be_trigger_bps=30.0,
            max_horizon_sec=1800, use_global_regime=False, taker_fee_bps=4.0,
            require_book_ticker_fill=True,
        ),
    },
    {
        "label": "SOL_SELL_100K",
        "base": runner.S34Rule(
            name="SOL_SELL_LIQ_SHORT_100K_TP60_SL30_BE40",
            symbol="SOLUSDT", liq_side="SELL", direction="SHORT",
            threshold_usd=100_000.0, tp_bps=60.0, sl_bps=30.0, be_trigger_bps=40.0,
            max_horizon_sec=1800, use_global_regime=False, taker_fee_bps=4.0,
            require_book_ticker_fill=True,
        ),
    },
]


def _pctile(vals, q):
    c = sorted(v for v in vals if v is not None and math.isfinite(v))
    if not c: return None
    pos = (len(c) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return c[lo] if lo == hi else c[lo] + (c[hi] - c[lo]) * (pos - lo)

def _median(v): return _pctile(v, 0.5)
def _mean(v):
    c = [x for x in v if x is not None and math.isfinite(x)]
    return sum(c) / len(c) if c else None
def _r1(v): return round(float(v), 1) if v is not None and math.isfinite(float(v)) else None


_REGIME_CFG = runner.RegimeConfig(enabled=True)


def _simulate(conn, rule) -> dict:
    ts = conn.execute(
        "SELECT MIN(ts_ms), MAX(ts_ms) FROM liquidations WHERE symbol=?", (rule.symbol,)
    ).fetchone()
    start_ms, end_ms = int(ts[0]), int(ts[1])
    signals = runner._bucket_events(conn, rule, start_ms, end_ms, limit=10_000)
    total = len(signals)
    if total == 0:
        return {"total": 0, "no_fill": 0, "regime_filtered": 0, "trades": []}
    trades, no_fill, regime_filtered = [], 0, 0
    for sig in signals:
        if sig.get("fill_error"):
            no_fill += 1
            continue
        regime_ok, _, _ = runner._regime_gate(conn, rule, sig, _REGIME_CFG)
        if not regime_ok:
            regime_filtered += 1
            continue
        trade = runner._paper_trade_from_signal(rule, sig, runner.DEFAULT_RISK)
        try:
            trade = runner._evaluate_trade(conn, trade, end_ms)
        except RuntimeError as exc:
            trade["status"] = "EXIT_FILL_ERROR"
            trade["exit_fill_error"] = str(exc)
        trades.append(trade)
    return {"total": total, "no_fill": no_fill, "regime_filtered": regime_filtered, "trades": trades}


def _metrics(trades):
    closed = [t for t in trades if t.get("status") == "CLOSED" and t.get("net_bps") is not None]
    if not closed:
        return None
    nets = [float(t["net_bps"]) for t in closed]
    em = {"TP": 0, "SL": 0, "BE": 0, "TIME": 0}
    for t in closed:
        r = str(t.get("exit_reason") or "")
        if r in em: em[r] += 1
    half = len(closed) // 2
    h1 = [float(t["net_bps"]) for t in closed[:half]]
    h2 = [float(t["net_bps"]) for t in closed[half:]]
    return {
        "n":         len(closed),
        "median":    _r1(_median(nets)),
        "cum":       _r1(sum(nets)),
        "top3r":     _r1(sum(sorted(nets)[3:]) if len(nets) > 3 else sum(nets)),
        "wr":        round(sum(1 for n in nets if n > 0) / len(nets), 3),
        "exits":     em,
        "h1_median": _r1(_median(h1)),
        "h2_median": _r1(_median(h2)),
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(f"file:{DB_PATH.as_posix()}?mode=ro", uri=True)
    now  = datetime.now(timezone.utc).isoformat()

    results = []
    print(f"S34 SELL Bear-Day Filter — {now}\n")

    for rd in SELL_RULES:
        label = rd["label"]
        base  = rd["base"]

        # Unfiltered (current)
        res_raw  = _simulate(conn, base)
        m_raw    = _metrics(res_raw["trades"])
        nf_raw   = res_raw["no_fill"] / res_raw["total"] if res_raw["total"] else 0

        # Bear-day filter: max_day_trend_bps=0.0
        from dataclasses import replace
        bear_rule = replace(base,
            name=base.name + "_BEAR",
            max_day_trend_bps=0.0,
        )
        res_bear = _simulate(conn, bear_rule)
        m_bear   = _metrics(res_bear["trades"])
        nf_bear  = res_bear["no_fill"] / res_bear["total"] if res_bear["total"] else 0

        # Strong-bear filter: max_day_trend_bps=-100.0 (day down >1%)
        strong_rule = replace(base,
            name=base.name + "_STRONGBEAR",
            max_day_trend_bps=-100.0,
        )
        res_strong = _simulate(conn, strong_rule)
        m_strong   = _metrics(res_strong["trades"])
        nf_strong  = res_strong["no_fill"] / res_strong["total"] if res_strong["total"] else 0

        print(f"=== {label} ===")
        for tag, res, m, nf in [
            ("RAW (current)",         res_raw,    m_raw,    nf_raw),
            ("BEAR (trend<=0)",       res_bear,   m_bear,   nf_bear),
            ("STRONGBEAR (trend<-1%)",res_strong, m_strong, nf_strong),
        ]:
            if m:
                prelim = " [PRELIM]" if res["total"] < PRELIMINARY_N or nf > NO_FILL_THRESHOLD else ""
                print(
                    f"  [{tag}]{prelim}  sigs={res['total']} nf={res['no_fill']}({nf*100:.0f}%)  "
                    f"closed={m['n']}  median={m['median']:+.1f}  cum={m['cum']:+.1f}  "
                    f"top3r={m['top3r']:+.1f}  WR={m['wr']*100:.0f}%  "
                    f"h1={m['h1_median']:+.1f} h2={m['h2_median']:+.1f}  "
                    f"exits=TP:{m['exits']['TP']} SL:{m['exits']['SL']} BE:{m['exits']['BE']} T:{m['exits']['TIME']}"
                )
            else:
                print(f"  [{tag}]  no data")

        results.append({
            "label": label,
            "raw":        {"total": res_raw["total"],    "no_fill": res_raw["no_fill"],    "metrics": m_raw},
            "bear":       {"total": res_bear["total"],   "no_fill": res_bear["no_fill"],   "metrics": m_bear},
            "strongbear": {"total": res_strong["total"], "no_fill": res_strong["no_fill"], "metrics": m_strong},
        })
        print()

    conn.close()

    # MD report
    lines = [
        "# S34 SELL Bear-Day Filter Research",
        "", f"Generated: `{now}`", "",
        "Tests whether adding `max_day_trend_bps=0.0` (bearish days only) to SELL rules",
        "produces the same improvement that `min_day_trend_bps=0.0` gives BUY rules.",
        "",
        "| Rule | Filter | Sigs | NF% | N | Median | Top3R | WR | H1 | H2 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in results:
        for tag, key in [("RAW","raw"), ("BEAR<=0","bear"), ("STRONGBEAR<-1%","strongbear")]:
            d = r[key]
            m = d["metrics"]
            if m:
                nf = d["no_fill"] / d["total"] * 100 if d["total"] else 0
                lines.append(
                    f"| {r['label']} | {tag} | {d['total']} | {nf:.0f}% "
                    f"| {m['n']} | {m['median']:+.1f} | {m['top3r']:+.1f} "
                    f"| {m['wr']*100:.0f}% | {m['h1_median']:+.1f} | {m['h2_median']:+.1f} |"
                )

    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    OUT_JSON.write_text(json.dumps({"generated_at": now, "results": results}, indent=2), encoding="utf-8")
    print(f"MD  : {OUT_MD}")
    print(f"JSON: {OUT_JSON}")


if __name__ == "__main__":
    main()
