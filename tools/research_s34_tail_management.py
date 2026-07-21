"""
research_s34_tail_management.py — does CUTTING the faded-flush tail net help after CHOP?

Follow-up to §162 tail forensics. §162 showed the 4h faded SELL-flush tail (MAE<=-150bps,
~21.7% of fades) is T0-irreducible and the reactive-60s CVD detector only helps near-term
(tail-AUC 0.636@30m -> 0.505@4h). Operator: "hem ona bak, trade chop-chop'ları araştıralım" —
i.e. quantify the WHIPSAW/CHOP cost: a cut that tames the tail also chops winners out; does the
net PnL improve, or does the chop cost more than the tail it saves?

Rules simulated per fade (ETHUSDT SELL-flush >=200K -> LONG, causal, mark-path fills):
  - HOLD_<h>           : baseline, exit at horizon
  - TIME_STOP_<m>      : shorter hold, exit at m minutes
  - PRICE_STOP_<b>     : exit at first path point where signed return <= -b bps (stop fill = -b)
  - REACTIVE_CUT       : if first-60s CVD < thr (selling continues) -> exit at T0+delay; else hold
  - REACTIVE+PRICE     : reactive cut AND price stop, whichever first

For each rule: N, mean/median NET bps, tail-rate, MAX_LOSS, T3R (top-3-winner-removed mean —
tail_cut.py's robustness metric), and the CHOP LEDGER: among trades the rule exited early, how
many would-be WINNERS (full-hold net >= 0) got chopped, and total bps given up. A real cut must
improve MAX_LOSS *and* keep mean/T3R on BOTH train and test WITHOUT a chop bill that eats it.
Read-only (DB mode=ro), causal. NO edge claim — a management diagnostic.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.research_s34_knowable_anchor_continuation import (  # noqa: E402
    MarkIndex,
    load_liquidations,
    load_mark_index_range,
    mean,
    pctile,
    reconstruct_anchors,
    signed_return_bps,
)

DEFAULT_DB = "file:data/microstructure.db?mode=ro"
OUT_MD = ROOT / "reports/research/s34/S34_TAIL_MANAGEMENT.md"
OUT_JSON = ROOT / "reports/research/s34/S34_TAIL_MANAGEMENT.json"
FEE_SIDE = 3.05


def cvd_first_60s_musd(conn, symbol, t0):
    r = conn.execute(
        "SELECT SUM(CASE WHEN is_buyer_maker=0 THEN notional ELSE 0 END),"
        "SUM(CASE WHEN is_buyer_maker=1 THEN notional ELSE 0 END) FROM agg_trades "
        "WHERE symbol=? AND ts_ms BETWEEN ? AND ?", (symbol, t0, t0 + 60_000)).fetchone()
    return ((r[0] or 0.0) - (r[1] or 0.0)) / 1e6


def net_bps(gross: float) -> float:
    return gross - 2.0 * FEE_SIDE


def exit_hold(path, entry_px, horizon_ms, t0, direction):
    """Exit at horizon end."""
    if not path:
        return None
    _, px = path[-1]
    return net_bps(signed_return_bps(direction, entry_px, px))


def exit_time_stop(path, entry_px, stop_ms, t0, direction):
    """Exit at first path point at/after t0+stop_ms."""
    target = t0 + stop_ms
    chosen = None
    for ts, px in path:
        if ts >= target:
            chosen = px
            break
    if chosen is None:
        chosen = path[-1][1]
    return net_bps(signed_return_bps(direction, entry_px, chosen))


def exit_price_stop(path, entry_px, stop_bps, t0, direction):
    """Exit at first path point where signed return <= -stop_bps (fill at -stop_bps);
    else hold to end. Returns (net, stopped_bool, stop_ts)."""
    for ts, px in path:
        r = signed_return_bps(direction, entry_px, px)
        if r <= -stop_bps:
            return net_bps(-stop_bps), True, ts
    return net_bps(signed_return_bps(direction, entry_px, path[-1][1])), False, None


def t3r(nets: list[float]) -> float | None:
    """Top-3-winner-removed mean (robustness: does the rule survive without its 3 luckiest)."""
    if len(nets) <= 3:
        return None
    trimmed = sorted(nets)[:-3]
    return mean(trimmed)


def summ(nets: list[float]) -> dict:
    if not nets:
        return {"n": 0}
    return {
        "n": len(nets),
        "mean_net": round(mean(nets), 1),
        "median_net": round(pctile(nets, 0.5), 1),
        "max_loss": round(min(nets), 1),
        "win_rate": round(sum(1 for x in nets if x > 0) / len(nets), 3),
        "t3r": round(t3r(nets), 1) if t3r(nets) is not None else None,
        "sum_net": round(sum(nets), 1),
    }


def build(conn, args):
    liqs = load_liquidations(conn, args.symbol, args.side, None, None)
    anchors = reconstruct_anchors(liqs, bucket_sec=args.bucket_sec, min_gap_sec=args.min_gap_sec,
                                  thresholds=(args.threshold,), accel_window_sec=args.accel_window_sec)
    t_lo = min(a.anchor_ts_ms for a in anchors) - 3_600_000
    t_hi = max(a.anchor_ts_ms for a in anchors) + args.horizon_sec * 1000 + 3_600_000
    marks = load_mark_index_range(conn, args.symbol, t_lo, t_hi)
    direction = "LONG" if args.side == "SELL" else "SHORT"
    horizon_ms = args.horizon_sec * 1000

    recs = []
    for a in anchors:
        em = marks.at_or_after(a.anchor_ts_ms)
        if not em:
            continue
        entry_px = float(em[1])
        path = marks.slice_range(a.anchor_ts_ms, a.anchor_ts_ms + horizon_ms)
        if not path or path[-1][0] < a.anchor_ts_ms + horizon_ms - 120_000:
            continue  # require path reaching ~horizon
        cvd = cvd_first_60s_musd(conn, args.symbol, a.anchor_ts_ms)
        recs.append({"t0": a.anchor_ts_ms, "entry_px": entry_px, "path": path, "cvd": cvd, "dir": direction})
    return recs, horizon_ms


def run_rules(recs, horizon_ms, args):
    direction = recs[0]["dir"] if recs else "LONG"

    def hold_net(rec):
        return exit_hold(rec["path"], rec["entry_px"], horizon_ms, rec["t0"], direction)

    rules: dict[str, Any] = {}
    # baseline
    rules[f"HOLD_{args.horizon_sec//3600}h"] = {"nets": [hold_net(r) for r in recs], "chop": None}

    # time stops
    for m in (30, 60, 120):
        nets = [exit_time_stop(r["path"], r["entry_px"], m * 60_000, r["t0"], direction) for r in recs]
        rules[f"TIME_STOP_{m}m"] = {"nets": nets, "chop": None}

    # price stops (+ chop ledger vs full hold)
    for b in (80, 120, 150):
        nets, chop_winners, chop_bps, stopped_n = [], 0, 0.0, 0
        for r in recs:
            n, stopped, _ = exit_price_stop(r["path"], r["entry_px"], b, r["t0"], direction)
            full = hold_net(r)
            nets.append(n)
            if stopped:
                stopped_n += 1
                if full >= 0:  # would-be winner chopped out
                    chop_winners += 1
                chop_bps += (full - n)  # bps given up by stopping (can be + or -)
        rules[f"PRICE_STOP_{b}bps"] = {"nets": nets,
                                       "chop": {"stopped": stopped_n, "winners_chopped": chop_winners,
                                                "net_bps_given_up_vs_hold": round(chop_bps, 1)}}

    # reactive cut: cvd<thr -> exit at t0+delay; else hold
    for thr in (0.0, -1.0):
        for delay_m in (5, 15):
            nets, cut_n, chop_winners, chop_bps = [], 0, 0, 0.0
            for r in recs:
                full = hold_net(r)
                if r["cvd"] < thr:
                    n = exit_time_stop(r["path"], r["entry_px"], delay_m * 60_000, r["t0"], direction)
                    cut_n += 1
                    if full >= 0:
                        chop_winners += 1
                    chop_bps += (full - n)
                else:
                    n = full
                nets.append(n)
            rules[f"REACTIVE_cvd<{thr}_exit{delay_m}m"] = {
                "nets": nets, "chop": {"cut": cut_n, "winners_chopped": chop_winners,
                                       "net_bps_given_up_vs_hold": round(chop_bps, 1)}}

    # combined: reactive cut OR price stop -150
    nets, acted, chop_winners, chop_bps = [], 0, 0, 0.0
    for r in recs:
        full = hold_net(r)
        n_ps, stopped, _ = exit_price_stop(r["path"], r["entry_px"], 150, r["t0"], direction)
        if r["cvd"] < 0.0:
            n = min(n_ps, exit_time_stop(r["path"], r["entry_px"], 15 * 60_000, r["t0"], direction))
            acted += 1
        elif stopped:
            n = n_ps
            acted += 1
        else:
            n = full
        if n < full:
            if full >= 0:
                chop_winners += 1
            chop_bps += (full - n)
        nets.append(n)
    rules["COMBINED_reactive+pstop150"] = {"nets": nets,
                                           "chop": {"acted": acted, "winners_chopped": chop_winners,
                                                    "net_bps_given_up_vs_hold": round(chop_bps, 1)}}
    return rules


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--db", default=DEFAULT_DB)
    p.add_argument("--symbol", default="ETHUSDT")
    p.add_argument("--side", default="SELL")
    p.add_argument("--threshold", type=float, default=200_000.0)
    p.add_argument("--bucket-sec", type=int, default=300)
    p.add_argument("--min-gap-sec", type=int, default=900)
    p.add_argument("--accel-window-sec", type=int, default=30)
    p.add_argument("--horizon-sec", type=int, default=4 * 3600)
    p.add_argument("--holdout-frac", type=float, default=0.30)
    p.add_argument("--json-out", default=str(OUT_JSON))
    p.add_argument("--md-out", default=str(OUT_MD))
    args = p.parse_args()

    conn = sqlite3.connect(args.db, uri=True)
    conn.execute("PRAGMA query_only=1")
    recs, horizon_ms = build(conn, args)
    conn.close()
    if not recs:
        print(json.dumps({"error": "no recs"}))
        return
    recs.sort(key=lambda r: r["t0"])
    cut = int(len(recs) * (1 - args.holdout_frac))
    train, test = recs[:cut], recs[cut:]

    all_rules = run_rules(recs, horizon_ms, args)
    tr_rules = run_rules(train, horizon_ms, args)
    te_rules = run_rules(test, horizon_ms, args)

    report = {"population": {"symbol": args.symbol, "side": args.side, "n": len(recs),
                            "horizon_h": args.horizon_sec / 3600, "fee_side": FEE_SIDE,
                            "date_range": [datetime.fromtimestamp(recs[0]["t0"]/1000, timezone.utc).isoformat(),
                                           datetime.fromtimestamp(recs[-1]["t0"]/1000, timezone.utc).isoformat()]},
              "rules": {}}
    for name in all_rules:
        report["rules"][name] = {
            "all": {**summ(all_rules[name]["nets"]), "chop": all_rules[name]["chop"]},
            "train": summ(tr_rules[name]["nets"]),
            "test": summ(te_rules[name]["nets"]),
        }

    Path(args.json_out).write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    base = f"HOLD_{args.horizon_sec//3600}h"
    L = [f"# S34 Tail MANAGEMENT — does cutting the tail beat the chop?\n",
         f"_{args.symbol} {args.side}-flush -> {'LONG' if args.side=='SELL' else 'SHORT'} fade, "
         f">= {int(args.threshold/1000)}K, {args.horizon_sec//3600}h, n={len(recs)}, fee {FEE_SIDE}/side._\n",
         "## Rule comparison (ALL / TRAIN / TEST net bps)\n",
         "| rule | n | mean | median | WR | MAX_LOSS | T3R | tr.mean | te.mean | tr.maxL | te.maxL | chop |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, r in report["rules"].items():
        a, t, e = r["all"], r["train"], r["test"]
        chop = a.get("chop")
        chops = ""
        if chop:
            wc = chop.get("winners_chopped")
            gu = chop.get("net_bps_given_up_vs_hold")
            chops = f"w{wc}/give{gu}"
        star = " ⭐" if (a["max_loss"] > report["rules"][base]["all"]["max_loss"]
                        and a["mean_net"] >= report["rules"][base]["all"]["mean_net"]
                        and t["mean_net"] >= 0 and e["mean_net"] >= 0) else ""
        L.append(f"| {name}{star} | {a['n']} | {a['mean_net']} | {a['median_net']} | {a['win_rate']} | "
                 f"{a['max_loss']} | {a['t3r']} | {t['mean_net']} | {e['mean_net']} | {t['max_loss']} | "
                 f"{e['max_loss']} | {chops} |")
    L.append("\n⭐ = improves MAX_LOSS vs baseline AND keeps mean>=baseline AND train&test mean>=0.")
    L.append("\n**Chop ledger** `wN` = would-be winners (full-hold net>=0) chopped out; `giveX` = total net bps "
             "given up vs holding. A cut that saves the tail but has a large chop bill is net-negative.\n")
    L.append("_Read-only management diagnostic. Price-stop fills assume exit AT the stop level (optimistic; real "
             "stops slip). No edge claim._")
    Path(args.md_out).write_text("\n".join(L) + "\n", encoding="utf-8")

    print(json.dumps({"n": len(recs), "baseline": report["rules"][base]["all"],
                      "best_by_mean": max(report["rules"].items(), key=lambda kv: kv[1]["all"]["mean_net"])[0],
                      "md": args.md_out}, indent=2, default=str))


if __name__ == "__main__":
    main()
