"""
research_s34_echo_reactive_arm_spec.py — concrete forward SPEC for the reactive overlay (Arm 2) and
mechanical stop (Arm 3), + multi-feature info-curve + late-flush deep dive (read-only, OD-029 safe).

Four blocks over the causal echo set (cand_causal, N~118), one build_events pass:
  A) MECH-STOP SIM (item 2): pre-declared stop grid; net/WR/tail/worst/no-overlap per stop, 4h & 6h.
  B) REACTIVE TRIGGER TRADEOFF (item 1): at T+k (k=4,5,6,7), pre-declared be_ratio theta grid; for each
     theta: tails caught vs winners whipsawed + reactive-net (exit@T+k if be>=theta else hold 4h).
  C) MULTI-FEATURE INFO-CURVE (item 3): incremental AUC vs tail for window-evolving features
     (be_ratio, new BTC/ETH SELL liq, BTC/ETH fwd return) — incl ETH own-position early P&L as the
     price-stop BENCHMARK the external be_ratio signal must beat.
  D) LATE-FLUSH DEEP DIVE (item 4): what happens in BTC during [ts,ts+10m] for the late-flush
     disagreement set vs stays-low (return, new liq count/notional, cross-coin SOL).

HARD LINE: grids are PRE-DECLARED (constants below), results are TRADEOFF CURVES not a chosen threshold.
Nothing is selected/adopted on the burned sample; forward locks the threshold (OD-028/029). No edge claim.
§163 precedent: mechanical stop caps tail magnitude but is not edge; reactive cut whipsaws. Descriptive.

Reuses gauntlet build_events/gross/stats/no_overlap. Read-only, deterministic.
"""
from __future__ import annotations
import json, math, sqlite3, sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.research_s34_knowable_anchor_continuation import (  # noqa: E402
    load_liquidations, load_mark_index, reconstruct_anchors,
)
from tools.research_s34_echo_live_gauntlet import (  # noqa: E402
    build_events, load_vol_state, regime, gross, mark_at, liq_cnt, liq_max,
    ETH_THRESH, PROP_THRESH, FEE_BPS, LOOKBACK_MS, HOLD_MS,
)

DB_PATH = ROOT / "data" / "microstructure.db"
OUT_DIR = ROOT / "reports" / "research" / "s34"
OUT_JSON = OUT_DIR / "S34_ECHO_REACTIVE_ARM_SPEC.json"
OUT_MD = OUT_DIR / "S34_ECHO_REACTIVE_ARM_SPEC.md"

# --- PRE-DECLARED grids (fixed BEFORE seeing results; forward locks the choice) ---
STOP_GRID = [None, 100.0, 120.0, 150.0, 200.0, 250.0]   # bps below entry (gross() expects POSITIVE)
THETA_GRID = [0.5, 1.0, 1.5, 2.0, 3.0]                        # be_ratio = BTC maxliq / ETH anchor notional
REACT_K = [4, 5, 6, 7]                                        # minutes after ts to resolve reactive trigger
INFO_K = [1, 2, 3, 4, 5, 6, 7, 8, 10]


def cand_causal(ev):
    return (not ev["bull"] and ev["sess"] != "EUROPE"
            and ev["dow"] not in {0, 2} and ev["echo_30_90"] and regime(ev))


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


def summ(nets):
    if not nets:
        return {"n": 0}
    n = len(nets); wins = sum(1 for v in nets if v > 0)
    return {"n": n, "wr": round(100 * wins / n, 1), "mean": round(sum(nets) / n, 1),
            "sum": round(sum(nets), 1), "worst": round(min(nets), 1),
            "tail_n": sum(1 for v in nets if v < -100)}


def med(vs):
    vs = sorted(v for v in vs if v is not None)
    if not vs:
        return None
    n = len(vs)
    return vs[n // 2] if n % 2 else (vs[n // 2 - 1] + vs[n // 2]) / 2.0


def main():
    with sqlite3.connect("file:%s?mode=ro" % DB_PATH, uri=True) as conn:
        conn.execute("PRAGMA query_only=1")
        conn.execute("PRAGMA cache_size=-128000"); conn.execute("PRAGMA temp_store=MEMORY")
        now_ms = int(datetime.now(tz=timezone.utc).timestamp() * 1000)
        liqs = load_liquidations(conn, "ETHUSDT", "SELL", now_ms - LOOKBACK_MS, now_ms)
        anchors = reconstruct_anchors(liqs, bucket_sec=300, min_gap_sec=900,
                                      thresholds=(ETH_THRESH,), accel_window_sec=30)
        marks_eth = load_mark_index(conn, "ETHUSDT")
        vol_rows = load_vol_state(conn)
        events = build_events(conn, anchors, marks_eth, vol_rows)
        marks_btc = load_mark_index(conn, "BTCUSDT")

        rows = []
        for ev in events:
            if not cand_causal(ev) or ev.get("g_t0_4h") is None:
                continue
            ts, rn = ev["ts"], ev["rn"]
            entry = mark_at(marks_eth, ts)
            r = {"ts": ts, "rn": rn, "entry": entry,
                 "net_4h": ev["g_t0_4h"] - FEE_BPS, "g_4h": ev["g_t0_4h"], "g_6h": ev.get("g_t0_6h"),
                 "tail": 1 if (ev["g_t0_4h"] - FEE_BPS) < -100 else 0,
                 "stop": {}, "be_k": {}, "ethret_k": {}, "btcret_k": {},
                 "new_btc_sell_cnt": {}, "new_btc_sell_not": {}, "new_eth_sell_cnt": {}, "exit_k": {}}
            for s in STOP_GRID:
                g4 = gross(marks_eth, ts, ts + HOLD_MS, stop_bps=s)
                r["stop"][str(s)] = (g4 - FEE_BPS) if g4 is not None else None
            for k in INFO_K:
                hi = ts + k * 60_000
                r["be_k"][k] = (liq_max(conn, "BTCUSDT", "SELL", ts - 10 * 60_000, hi) / rn) if rn > 0 else 0.0
                mk = mark_at(marks_eth, hi); mb = mark_at(marks_btc, hi)
                r["exit_k"][k] = mk
                r["ethret_k"][k] = ((mk - entry) / entry * 1e4) if (mk and entry) else None
                mb0 = mark_at(marks_btc, ts)
                r["btcret_k"][k] = ((mb - mb0) / mb0 * 1e4) if (mb and mb0) else None
                r["new_btc_sell_cnt"][k] = liq_cnt(conn, "BTCUSDT", "SELL", ts, hi, PROP_THRESH)
                r["new_btc_sell_not"][k] = float(conn.execute(
                    "SELECT COALESCE(SUM(notional),0) FROM liquidations WHERE symbol='BTCUSDT' AND side='SELL' "
                    "AND ts_ms>=? AND ts_ms<?", (ts, hi)).fetchone()[0])
                r["new_eth_sell_cnt"][k] = liq_cnt(conn, "ETHUSDT", "SELL", ts, hi, PROP_THRESH)
            rows.append(r)

    ty = [r["tail"] for r in rows]
    n_tail = sum(ty)
    base_nets = [r["net_4h"] for r in rows]

    # ---- Block A: mechanical stop sim ----
    stop_block = {}
    for s in STOP_GRID:
        nets = [r["stop"][str(s)] for r in rows if r["stop"][str(s)] is not None]
        tail_nets = [r["stop"][str(s)] for r in rows if r["tail"] == 1 and r["stop"][str(s)] is not None]
        d = summ(nets); d["tail_only"] = summ(tail_nets)
        stop_block[str(s)] = d

    # ---- Block B: reactive trigger tradeoff ----
    react_block = {}
    for k in REACT_K:
        for theta in THETA_GRID:
            react_nets = []; cut = 0; tails_cut = 0; winners_cut = 0
            for r in rows:
                be = r["be_k"][k]
                if be is not None and be >= theta:
                    cut += 1
                    mk = r["exit_k"][k]
                    rn_net = ((mk - r["entry"]) / r["entry"] * 1e4 - FEE_BPS) if (mk and r["entry"]) else r["net_4h"]
                    react_nets.append(rn_net)
                    if r["tail"] == 1:
                        tails_cut += 1
                    elif r["net_4h"] > 0:
                        winners_cut += 1
                else:
                    react_nets.append(r["net_4h"])
            react_block["k%d_theta%s" % (k, theta)] = {
                "k": k, "theta": theta, "n_cut": cut,
                "tails_cut": tails_cut, "tail_catch_rate": round(tails_cut / n_tail, 3) if n_tail else None,
                "winners_whipsawed": winners_cut,
                "reactive_sum": round(sum(react_nets), 1), "reactive_mean": round(sum(react_nets) / len(react_nets), 1),
                "reactive_tail_n": sum(1 for v in react_nets if v < -100),
                "delta_vs_hold": round(sum(react_nets) - sum(base_nets), 1)}

    # ---- Block C: multi-feature info-curve (AUC vs tail per k) ----
    info_curve = []
    for k in INFO_K:
        info_curve.append({
            "k": k,
            "be_ratio": round(rank_auc([r["be_k"][k] for r in rows], ty), 3),
            "eth_own_pnl": round(rank_auc([(-(r["ethret_k"][k] or 0.0)) for r in rows], ty), 3),  # more negative = tail; flip sign so higher=tail
            "btc_ret": round(rank_auc([(-(r["btcret_k"][k] or 0.0)) for r in rows], ty), 3),
            "new_btc_sell_cnt": round(rank_auc([r["new_btc_sell_cnt"][k] for r in rows], ty), 3),
            "new_btc_sell_not": round(rank_auc([r["new_btc_sell_not"][k] for r in rows], ty), 3),
            "new_eth_sell_cnt": round(rank_auc([r["new_eth_sell_cnt"][k] for r in rows], ty), 3)})

    # ---- Block D: late-flush deep dive (be causal-low @T0-proxy via be_k[1] low, resolved-high @k10) ----
    be1 = [r["be_k"][1] for r in rows]; be10 = [r["be_k"][10] for r in rows]
    m1, m10 = med(be1), med(be10)
    late = [r for r in rows if (r["be_k"][1] or 0) <= m1 and (r["be_k"][10] or 0) > m10]
    low = [r for r in rows if (r["be_k"][1] or 0) <= m1 and (r["be_k"][10] or 0) <= m10]
    def deep(g):
        return {"n": len(g),
                "tail_rate": round(sum(x["tail"] for x in g) / len(g), 3) if g else None,
                "mean_net_4h": round(sum(x["net_4h"] for x in g) / len(g), 1) if g else None,
                "med_btc_ret_10m": round(med([x["btcret_k"][10] for x in g]), 1) if g else None,
                "med_new_btc_sell_cnt_10m": med([x["new_btc_sell_cnt"][10] for x in g]) if g else None,
                "med_new_btc_sell_not_10m_$M": round((med([x["new_btc_sell_not"][10] for x in g]) or 0) / 1e6, 3) if g else None,
                "med_new_eth_sell_cnt_10m": med([x["new_eth_sell_cnt"][10] for x in g]) if g else None}

    out = {"tool": "echo_reactive_arm_spec", "generated_utc": datetime.now(timezone.utc).isoformat(),
           "n_causal": len(rows), "n_tail": n_tail,
           "frame": "PRE-DECLARED grids; TRADEOFF curves, NOT a chosen threshold. Forward locks the "
                    "choice (OD-028/029). No edge claim. §163: stop caps magnitude not edge, cut whipsaws.",
           "baseline_hold": summ(base_nets),
           "grids": {"stop_bps": STOP_GRID, "theta": THETA_GRID, "react_k": REACT_K},
           "A_mech_stop": stop_block, "B_reactive_tradeoff": react_block,
           "C_info_curve": info_curve,
           "D_late_flush": {"medians": {"be_k1": round(m1, 3), "be_k10": round(m10, 3)},
                            "late_flush": deep(late), "stays_low": deep(low)}}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")

    b = out["baseline_hold"]
    L = ["# Echo Reactive/Stop Arm Spec (forward, pre-declared grids)", "",
         "_%s · READ-ONLY · causal N=%d · tails=%d_" % (out["generated_utc"], len(rows), n_tail),
         "", "> " + out["frame"], "",
         "Baseline hold-4h: N=%d WR=%s mean=%+.1f sum=%+.1f worst=%.1f tail=%d" % (
             b["n"], b["wr"], b["mean"], b["sum"], b["worst"], b["tail_n"]), "",
         "## A) Mechanical stop grid (hold 4h)", "",
         "| stop bps | N | WR | mean | sum | worst | tail_n | tail-only mean | tail-only worst |",
         "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for s in STOP_GRID:
        d = stop_block[str(s)]; t = d.get("tail_only", {})
        L.append("| %s | %d | %s | %+.1f | %+.1f | %.1f | %d | %s | %s |" % (
            "none" if s is None else int(s), d["n"], d["wr"], d["mean"], d["sum"], d["worst"],
            d["tail_n"], t.get("mean", "—"), t.get("worst", "—")))
    L += ["", "## B) Reactive trigger tradeoff (exit@T+k if be_ratio>=theta, else hold 4h)", "",
          "| k | theta | n_cut | tails_cut | tail-catch | winners_whipsawed | reactive_sum | Δ vs hold |",
          "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for key, v in react_block.items():
        L.append("| %d | %s | %d | %d | %s | %d | %+.1f | %+.1f |" % (
            v["k"], v["theta"], v["n_cut"], v["tails_cut"], str(v["tail_catch_rate"]),
            v["winners_whipsawed"], v["reactive_sum"], v["delta_vs_hold"]))
    L += ["", "_Δ vs hold > 0 means the reactive cut improved total net IN-SAMPLE (fragile, forward-only)._", "",
          "## C) Multi-feature info-curve (AUC vs tail; sign-oriented so higher=more tail)", "",
          "| k(min) | be_ratio | eth_own_pnl(benchmark) | btc_ret | new_btc_sell_cnt | new_btc_sell_$ | new_eth_sell_cnt |",
          "|---:|---:|---:|---:|---:|---:|---:|"]
    for c in info_curve:
        L.append("| %d | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f |" % (
            c["k"], c["be_ratio"], c["eth_own_pnl"], c["btc_ret"], c["new_btc_sell_cnt"],
            c["new_btc_sell_not"], c["new_eth_sell_cnt"]))
    lf, lo = out["D_late_flush"]["late_flush"], out["D_late_flush"]["stays_low"]
    L += ["", "_If eth_own_pnl (watching your own position) matches/beats be_ratio, a plain price-stop "
          "already captures it and the external BTC-flush signal adds nothing._", "",
          "## D) Late-flush deep dive (BTC in [ts,ts+10m])", "",
          "| group | n | tail rate | mean net 4h | BTC ret@10m | new BTC sell cnt | new BTC sell $M | new ETH sell cnt |",
          "|---|---:|---:|---:|---:|---:|---:|---:|",
          "| LATE-FLUSH | %d | %s | %s | %s | %s | %s | %s |" % (
              lf["n"], lf["tail_rate"], lf["mean_net_4h"], lf["med_btc_ret_10m"],
              lf["med_new_btc_sell_cnt_10m"], lf["med_new_btc_sell_not_10m_$M"], lf["med_new_eth_sell_cnt_10m"]),
          "| STAYS-LOW | %d | %s | %s | %s | %s | %s | %s |" % (
              lo["n"], lo["tail_rate"], lo["mean_net_4h"], lo["med_btc_ret_10m"],
              lo["med_new_btc_sell_cnt_10m"], lo["med_new_btc_sell_not_10m_$M"], lo["med_new_eth_sell_cnt_10m"]), "",
          "## Read (forward spec, not adoption)",
          "- A: pick the stop that caps tail worst without gutting mean — but §163 says it's a magnitude",
          "  cap, not edge; forward confirms. B: any theta with Δ>0 AND low winners_whipsawed is a forward",
          "  reactive candidate; the whole table is the tradeoff surface, forward locks (k,theta).",
          "- C: compare be_ratio vs eth_own_pnl per k — external signal only earns an arm if it beats the",
          "  self-position price-stop benchmark. D: names the mechanism (BTC liq buildup) behind late-flush.", ""]
    OUT_MD.write_text("\n".join(L), encoding="utf-8")

    # viz
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ks = [c["k"] for c in info_curve]
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        for f, mk in [("be_ratio", "-o"), ("eth_own_pnl", "-s"), ("new_btc_sell_cnt", "-^"), ("btc_ret", "-d")]:
            ax.plot(ks, [c[f] for c in info_curve], mk, label=f)
        ax.axhline(0.5, color="gray", ls=":"); ax.axhline(0.62, color="red", ls="--", alpha=0.5)
        ax.set_xlabel("k (min after ts)"); ax.set_ylabel("AUC vs tail"); ax.legend(fontsize=8)
        ax.set_title("Multi-feature info-curve — causal echo N=%d" % len(rows)); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(OUT_DIR / "S34_ECHO_REACTIVE_INFOCURVE.png", dpi=110); plt.close(fig)
        viz = "written"
    except Exception as e:
        viz = "SKIPPED: %s" % e

    print(json.dumps({"n_causal": len(rows), "n_tail": n_tail, "viz": viz,
                      "baseline_sum": b["sum"],
                      "stop_summary": {s: {"sum": stop_block[s]["sum"], "worst": stop_block[s]["worst"],
                                           "tail_n": stop_block[s]["tail_n"]} for s in stop_block},
                      "reactive_best_delta": max(react_block.items(), key=lambda kv: kv[1]["delta_vs_hold"]),
                      "late_flush": out["D_late_flush"]}, indent=2, default=str))
    print("MD:", OUT_MD)


if __name__ == "__main__":
    main()
