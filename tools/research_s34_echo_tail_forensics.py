"""
research_s34_echo_tail_forensics.py — descriptive forensics of the 14 echo-set tails (read-only).

The causal echo core has ~14 tails (net_4h < -100) we cannot filter at T0 (see SEPARABILITY_STATS,
tail T0-irreducible). "Cannot filter" != "ignore" — WHEN/under-what-regime they appear can still
inform risk management (regime awareness, sizing, reactive management). This refines §162/§163
(project_tail_forensics / tail_management: general-fade tail 21.7% irreducible, mechanical price-stop
helps magnitude but isn't edge, reactive-60s cut = whipsaw) to the ECHO causal set specifically.

Reads S34_ECHO_SEPARABILITY_STATS_records.jsonl (the 118 causal events + features + labels + net;
NO rebuild) and adds targeted causal DB context per event: BTC pre-1h/2h move, ETH book spread/depth
at ts, pre-30m volume spike, post-hold recovery (6h/8h). Contrasts tail(14) vs non-tail(104).

HARD LINE: DESCRIPTIVE. N=14 => every contrast is fragile, medians not inference. This maps a
"higher-risk regime" HYPOTHESIS for FORWARD sizing/management; it selects NO threshold and claims NO
edge on the burned sample. Any sizing/stop rule = forward prereg (OD-028/029).
"""
from __future__ import annotations
import json, math, sqlite3, sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DB_PATH = ROOT / "data" / "microstructure.db"
OUT_DIR = ROOT / "reports" / "research" / "s34"
RECORDS = OUT_DIR / "S34_ECHO_SEPARABILITY_STATS_records.jsonl"
OUT_JSON = OUT_DIR / "S34_ECHO_TAIL_FORENSICS.json"
OUT_MD = OUT_DIR / "S34_ECHO_TAIL_FORENSICS.md"
PROP_THRESH = 50_000.0


def mark_at(cur, sym, ts):
    r = cur.execute("SELECT mark_price FROM mark_prices WHERE symbol=? AND ts_ms<=? ORDER BY ts_ms DESC LIMIT 1",
                    (sym, ts)).fetchone()
    return float(r[0]) if r and r[0] else None


def mark_bps(cur, sym, ts, lookback_ms):
    a = mark_at(cur, sym, ts - lookback_ms); b = mark_at(cur, sym, ts)
    return (b - a) / a * 1e4 if (a and b and a > 0) else None


def fwd_bps(cur, sym, ts, ahead_ms):
    a = mark_at(cur, sym, ts); b = mark_at(cur, sym, ts + ahead_ms)
    return (b - a) / a * 1e4 if (a and b and a > 0) else None


def book_at(cur, ts):
    r = cur.execute("SELECT spread_pct, bid_depth_usd FROM book_ticker WHERE symbol='ETHUSDT' "
                    "AND ts_ms<=? ORDER BY ts_ms DESC LIMIT 1", (ts,)).fetchone()
    return (r[0], r[1]) if r else (None, None)


def vol_sum(cur, lo, hi):
    r = cur.execute("SELECT COALESCE(SUM(notional),0) FROM agg_trades WHERE symbol='ETHUSDT' "
                    "AND ts_ms>=? AND ts_ms<?", (lo, hi)).fetchone()
    return float(r[0]) if r else 0.0


def med(vs):
    vs = sorted(v for v in vs if v is not None)
    if not vs:
        return None
    n = len(vs)
    return vs[n // 2] if n % 2 else (vs[n // 2 - 1] + vs[n // 2]) / 2.0


def rank_auc(values, labels):
    pairs = [(v, y) for v, y in zip(values, labels) if v is not None]
    n1 = sum(1 for _, y in pairs if y); n0 = sum(1 for _, y in pairs if not y)
    if n1 == 0 or n0 == 0:
        return None
    order = sorted(range(len(pairs)), key=lambda i: pairs[i][0])
    ranks = [0.0] * len(pairs); i = 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[order[j + 1]][0] == pairs[order[i]][0]:
            j += 1
        mid = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = mid
        i = j + 1
    R1 = sum(ranks[idx] for idx, (_, y) in enumerate(pairs) if y)
    return (R1 - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def main():
    if not RECORDS.exists():
        print("ERR: run research_s34_echo_separability_stats.py first (%s missing)" % RECORDS)
        return
    rows = [json.loads(l) for l in RECORDS.read_text(encoding="utf-8").splitlines() if l.strip()]

    with sqlite3.connect("file:%s?mode=ro" % DB_PATH, uri=True) as conn:
        conn.execute("PRAGMA query_only=1")
        cur = conn.cursor()
        for r in rows:
            ts = int(r["ts"])
            r["btc1h_pre"] = mark_bps(cur, "BTCUSDT", ts, 3600_000)
            r["btc2h_pre"] = mark_bps(cur, "BTCUSDT", ts, 2 * 3600_000)
            r["eth1h_pre"] = mark_bps(cur, "ETHUSDT", ts, 3600_000)
            sp, dep = book_at(cur, ts)
            r["spread_pct"] = sp
            r["bid_depth_usd"] = dep
            v30 = vol_sum(cur, ts - 30 * 60_000, ts)
            v_prev = vol_sum(cur, ts - 60 * 60_000, ts - 30 * 60_000)
            r["vol_spike_30m"] = (v30 / v_prev) if v_prev > 0 else None
            r["net_6h"] = round(fwd_bps(cur, "ETHUSDT", ts, 6 * 3600_000) - 5.0, 1) if fwd_bps(cur, "ETHUSDT", ts, 6 * 3600_000) is not None else None
            r["net_8h"] = round(fwd_bps(cur, "ETHUSDT", ts, 8 * 3600_000) - 5.0, 1) if fwd_bps(cur, "ETHUSDT", ts, 8 * 3600_000) is not None else None

    tails = [r for r in rows if r["_tail_4h"] == 1]
    nons = [r for r in rows if r["_tail_4h"] == 0]
    ty = [r["_tail_4h"] for r in rows]

    # A) time / regime
    def dist(g, key):
        return sorted([r[key] for r in g if r.get(key) is not None])
    contrasts = {}
    for key in ["hour", "dow", "btc4h", "btc7d", "btc1h_pre", "btc2h_pre", "eth1h_pre",
                "rv_bps", "sync_k", "rn", "spread_pct", "bid_depth_usd", "vol_spike_30m"]:
        contrasts[key] = {"tail_med": med([r.get(key) for r in tails]),
                          "non_med": med([r.get(key) for r in nons]),
                          "auc_tail": round(rank_auc([r.get(key) for r in rows], ty), 3)}

    # B) clustering — inter-tail gaps within the causal set
    tail_ts = sorted(int(r["ts"]) for r in tails)
    gaps_h = [round((tail_ts[i] - tail_ts[i - 1]) / 3600_000, 1) for i in range(1, len(tail_ts))]
    within24 = sum(1 for g in gaps_h if g <= 24)
    within48 = sum(1 for g in gaps_h if g <= 48)

    # C) post-tail recovery: do the 14 tails recover after the 4h loss?
    recov = {"net_4h_med": med([r["_net_4h"] for r in tails]),
             "net_6h_med": med([r.get("net_6h") for r in tails]),
             "net_8h_med": med([r.get("net_8h") for r in tails]),
             "n_worse_at8h": sum(1 for r in tails if r.get("net_8h") is not None and r["net_8h"] < r["_net_4h"]),
             "n_better_at8h": sum(1 for r in tails if r.get("net_8h") is not None and r["net_8h"] > r["_net_4h"])}

    out = {"tool": "echo_tail_forensics", "generated_utc": datetime.now(timezone.utc).isoformat(),
           "n_causal": len(rows), "n_tail": len(tails),
           "frame": "DESCRIPTIVE, N=14 fragile. Maps a higher-risk-regime HYPOTHESIS for FORWARD "
                    "sizing/management. NO threshold, NO edge claim. Refines §162/§163 to the echo set.",
           "regime_contrasts": contrasts,
           "tail_hours": dist(tails, "hour"), "tail_dows": dist(tails, "dow"),
           "clustering": {"tail_ts_utc": [datetime.fromtimestamp(t/1000, timezone.utc).strftime("%Y-%m-%d %H:%M") for t in tail_ts],
                          "inter_tail_gaps_h": gaps_h, "median_gap_h": med(gaps_h),
                          "n_within_24h": within24, "n_within_48h": within48},
           "post_tail_recovery": recov}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")

    L = ["# Echo-set Tail Forensics (14 tails, descriptive)", "",
         "_%s · READ-ONLY · causal N=%d · tails=%d_" % (out["generated_utc"], len(rows), len(tails)),
         "", "> " + out["frame"], "",
         "## A) Regime / time contrast (tail median vs non-tail median; AUC=separation, ~0.5=none)", "",
         "| feature | tail med | non-tail med | AUC→tail |", "|---|---:|---:|---:|"]
    for k, v in contrasts.items():
        def g(x): return "—" if x is None else ("%.4g" % x)
        L.append("| %s | %s | %s | %.3f |" % (k, g(v["tail_med"]), g(v["non_med"]), v["auc_tail"]))
    L += ["", "Tail hours (UTC): %s" % out["tail_hours"],
          "Tail dows (0=Mon): %s" % out["tail_dows"], "",
          "## B) Clustering (are tails serial?)", "",
          "- Tail timestamps: %s" % ", ".join(out["clustering"]["tail_ts_utc"]),
          "- Inter-tail gaps (h): %s" % out["clustering"]["inter_tail_gaps_h"],
          "- Median gap: %s h · within 24h: %d · within 48h: %d (of %d gaps)" % (
              out["clustering"]["median_gap_h"], out["clustering"]["n_within_24h"],
              out["clustering"]["n_within_48h"], len(gaps_h)), "",
          "## C) Post-tail recovery (do the 4h losses mean-revert?)", "",
          "- median net: 4h=%s · 6h=%s · 8h=%s bps" % (
              recov["net_4h_med"], recov["net_6h_med"], recov["net_8h_med"]),
          "- of 14 tails: %d worse at 8h, %d better at 8h" % (recov["n_worse_at8h"], recov["n_better_at8h"]), "",
          "## Read",
          "- Any feature with AUC clearly >0.6 or <0.4 = a FORWARD risk-regime hypothesis (size down /",
          "  tighten stop in that regime), NOT a T0 filter (separability already showed none survive).",
          "- Clustering: if tails bunch within 24-48h => a 'tail-density' risk-scaler is worth a FORWARD",
          "  arm (reduce size after a recent tail). Post-tail recovery: if 8h≈4h, holding longer doesn't",
          "  save them (mechanical stop territory, §163). All descriptive, N=14 — forward is the judge.", ""]
    OUT_MD.write_text("\n".join(L), encoding="utf-8")

    print(json.dumps({"n_tail": len(tails), "tail_hours": out["tail_hours"],
                      "top_regime_contrasts": {k: v for k, v in sorted(
                          contrasts.items(), key=lambda kv: -abs(kv[1]["auc_tail"] - 0.5))[:6]},
                      "clustering": out["clustering"], "recovery": recov}, indent=2, default=str))
    print("MD:", OUT_MD)


if __name__ == "__main__":
    main()
