"""
research_s34_echo_be_ratio_infocurve.py — WHEN does the lookahead's information arrive?
(read-only, DESCRIPTIVE dissection of the contaminant window, OD-029 safe)

The be_ratio signal is 78% inside [ts, ts+10m] (see S34_ECHO_SEPARABILITY_STATS). This tool
dissects that window minute-by-minute to answer: is the signal resolvable EARLY (T+2-3, a usable
reactive overlay) or only LATE (T+9-10, too late to act)?

For the causal echo set (cand_causal, N~118), per minute k in 0..10:
  - be_ratio_k = liq_max(BTC SELL, [ts-10m, ts+k*60s]) / rn      (cumulative resolved be_ratio)
  - btc_ret_k, eth_ret_k = forward mark return ts -> ts+k        (what price does in the window)
  - new_eth_sell_k = count of new >=50K ETH SELL liqs in (ts, ts+k]  (liquidity consumption arriving)
then AUC(be_ratio_k) vs noisy and vs tail_4h => the INFORMATION-ARRIVAL CURVE.
Plus DISAGREEMENT SET: events where causal(k=0) is low but resolved(k=10) is high (the flush develops
AFTER ts) — their noisy/tail/net profile = what "late-developing" flushes look like.

HARD LINE: descriptive only. This characterizes WHERE/WHEN the contaminant's information sits; it does
NOT select a reactive threshold or gate on the burned sample. Any reactive overlay = forward prereg
(OD-028/029). noisy is defined over (ts+60s, ts+30m) so small-k be_ratio_k partly co-measures the same
continuation — read the curve as "how early the continuation reveals itself", not independent alpha.

Reuses build_events from the gauntlet. Read-only, deterministic.
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
    build_events, load_vol_state, regime, liq_max, liq_cnt,
    ETH_THRESH, PROP_THRESH, FEE_BPS, LOOKBACK_MS,
)

DB_PATH = ROOT / "data" / "microstructure.db"
OUT_DIR = ROOT / "reports" / "research" / "s34"
OUT_JSON = OUT_DIR / "S34_ECHO_BE_RATIO_INFOCURVE.json"
OUT_MD = OUT_DIR / "S34_ECHO_BE_RATIO_INFOCURVE.md"

K_LIST = list(range(0, 11))   # minutes 0..10


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


def fwd_bps(marks, t0, t1):
    r0 = marks.at_or_after(t0); r1 = marks.at_or_before(t1)
    if r0 and r1 and float(r0[1]) > 0:
        return (float(r1[1]) - float(r0[1])) / float(r0[1]) * 1e4
    return None


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
            rec = {"ts": ts, "_noisy": 1 if ev["noisy"] else 0,
                   "_tail": 1 if (ev["g_t0_4h"] - FEE_BPS) < -100 else 0,
                   "_net": round(ev["g_t0_4h"] - FEE_BPS, 1), "be": {}, "btc_ret": {},
                   "eth_ret": {}, "new_eth_sell": {}}
            for k in K_LIST:
                hi = ts + k * 60_000
                bc = liq_max(conn, "BTCUSDT", "SELL", ts - 10 * 60_000, hi)
                rec["be"][k] = (bc / rn) if rn > 0 else 0.0
                rec["btc_ret"][k] = fwd_bps(marks_btc, ts, hi) if k > 0 else 0.0
                rec["eth_ret"][k] = fwd_bps(marks_eth, ts, hi) if k > 0 else 0.0
                rec["new_eth_sell"][k] = liq_cnt(conn, "ETHUSDT", "SELL", ts, hi, PROP_THRESH) if k > 0 else 0
            rows.append(rec)

    ny = [r["_noisy"] for r in rows]
    ty = [r["_tail"] for r in rows]
    curve = []
    for k in K_LIST:
        be_k = [r["be"][k] for r in rows]
        curve.append({
            "k_min": k,
            "auc_be_noisy": round(rank_auc(be_k, ny), 3),
            "auc_be_tail": round(rank_auc(be_k, ty), 3),
            "auc_btcret_tail": round(rank_auc([r["btc_ret"][k] for r in rows], ty), 3) if k > 0 else None,
            "auc_neweth_noisy": round(rank_auc([r["new_eth_sell"][k] for r in rows], ny), 3) if k > 0 else None,
            "auc_neweth_tail": round(rank_auc([r["new_eth_sell"][k] for r in rows], ty), 3) if k > 0 else None,
            "med_btc_ret": round(med([r["btc_ret"][k] for r in rows]), 1) if k > 0 else 0.0,
            "med_eth_ret": round(med([r["eth_ret"][k] for r in rows]), 1) if k > 0 else 0.0,
            "med_new_eth_sell": med([r["new_eth_sell"][k] for r in rows]) if k > 0 else 0,
        })

    # disagreement set: causal(k=0) below median AND resolved(k=10) above median
    be0 = [r["be"][0] for r in rows]; be10 = [r["be"][10] for r in rows]
    m0, m10 = med(be0), med(be10)
    dis = [r for r in rows if r["be"][0] <= m0 and r["be"][10] > m10]  # late-developing flush
    agree_low = [r for r in rows if r["be"][0] <= m0 and r["be"][10] <= m10]
    def prof(g):
        return {"n": len(g),
                "noisy_rate": round(sum(x["_noisy"] for x in g) / len(g), 3) if g else None,
                "tail_rate": round(sum(x["_tail"] for x in g) / len(g), 3) if g else None,
                "mean_net": round(sum(x["_net"] for x in g) / len(g), 1) if g else None,
                "med_btc_ret_10": round(med([x["btc_ret"][10] for x in g]), 1) if g else None}

    out = {"tool": "echo_be_ratio_infocurve", "generated_utc": datetime.now(timezone.utc).isoformat(),
           "n_causal": len(rows), "n_noisy": sum(ny), "n_tail": sum(ty),
           "frame": "Descriptive dissection of the [ts, ts+10m] contaminant window. Curve = how early "
                    "the flush-continuation reveals itself. NOT a reactive threshold/gate; any overlay "
                    "is FORWARD prereg. noisy defined over (ts+60s,ts+30m) so small-k co-measures it.",
           "info_arrival_curve": curve,
           "disagreement_late_flush": prof(dis), "agree_low": prof(agree_low),
           "medians": {"be_causal_k0": round(m0, 3), "be_resolved_k10": round(m10, 3)}}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")

    L = ["# Echo be_ratio — Information-Arrival Curve ([ts, ts+10m] dissection)", "",
         "_%s · READ-ONLY · causal N=%d · noisy=%d · tail=%d_" % (
             out["generated_utc"], len(rows), sum(ny), sum(ty)), "", "> " + out["frame"], "",
         "## When does the signal arrive? (cumulative be_ratio over [ts-10m, ts+k])", "",
         "| k(min) | AUC be→noisy | AUC be→tail | AUC BTCret→tail | AUC newETHsell→tail | med BTCret bps | med newETHsell |",
         "|---:|---:|---:|---:|---:|---:|---:|"]
    for c in curve:
        def g(x): return "—" if x is None else ("%.3f" % x if isinstance(x, float) and abs(x) <= 1.5 else str(x))
        L.append("| %d | %.3f | %.3f | %s | %s | %s | %s |" % (
            c["k_min"], c["auc_be_noisy"], c["auc_be_tail"],
            g(c["auc_btcret_tail"]), g(c["auc_neweth_tail"]),
            str(c["med_btc_ret"]), str(c["med_new_eth_sell"])))
    d, a = out["disagreement_late_flush"], out["agree_low"]
    L += ["", "## Disagreement set — late-developing flush (causal-low @k0, resolved-high @k10)", "",
          "| group | n | noisy rate | tail rate | mean net | med BTCret@10m |",
          "|---|---:|---:|---:|---:|---:|",
          "| LATE-FLUSH (disagreement) | %s | %s | %s | %s | %s |" % (
              d["n"], d["noisy_rate"], d["tail_rate"], d["mean_net"], d["med_btc_ret_10"]),
          "| stays-low (agreement) | %s | %s | %s | %s | %s |" % (
              a["n"], a["noisy_rate"], a["tail_rate"], a["mean_net"], a["med_btc_ret_10"]), "",
          "## Read",
          "- AUC be→tail rising from ~0.5 (k=0) toward ~0.73 (k=10): the k where it crosses ~0.62 is the",
          "  minimum reactive delay a be_ratio overlay would need. Early cross => fast overlay feasible;",
          "  only-late cross => too late to act, overlay dead.",
          "- LATE-FLUSH vs stays-low tail/net gap = the loss a reactive cut could (forward) address —",
          "  but at the cost of 10m delay + whipsaw on the winners in that same set. Forward-only.", ""]
    OUT_MD.write_text("\n".join(L), encoding="utf-8")

    # viz: AUC vs k
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ks = [c["k_min"] for c in curve]
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(ks, [c["auc_be_noisy"] for c in curve], "-o", label="be→noisy")
        ax.plot(ks, [c["auc_be_tail"] for c in curve], "-s", label="be→tail")
        ax.plot([c["k_min"] for c in curve if c["auc_neweth_tail"] is not None],
                [c["auc_neweth_tail"] for c in curve if c["auc_neweth_tail"] is not None],
                "-^", label="newETHsell→tail")
        ax.axhline(0.5, color="gray", ls=":"); ax.axhline(0.62, color="red", ls="--", alpha=0.5, label="useful~0.62")
        ax.set_xlabel("k = minutes after ts"); ax.set_ylabel("AUC")
        ax.set_title("Information-arrival curve — causal echo set N=%d" % len(rows))
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(OUT_DIR / "S34_ECHO_BE_RATIO_INFOCURVE.png", dpi=110)
        plt.close(fig); viz = "written"
    except Exception as e:
        viz = "SKIPPED: %s" % e

    print(json.dumps({"n_causal": len(rows), "n_noisy": sum(ny), "n_tail": sum(ty), "viz": viz,
                      "curve": [{"k": c["k_min"], "be_noisy": c["auc_be_noisy"],
                                 "be_tail": c["auc_be_tail"]} for c in curve],
                      "late_flush": out["disagreement_late_flush"], "agree_low": out["agree_low"]},
                     indent=2))
    print("MD:", OUT_MD)


if __name__ == "__main__":
    main()
