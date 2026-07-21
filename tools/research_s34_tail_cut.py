"""S34 Tail-Cut Conditioning.

The deep-V fade (and its maker form) keeps the same wall: positive median but a
-330 bps tail and negative T3R, because the minority of cascades that do NOT
revert run away on a 4h hold. The central question: can KNOWABLE information
separate reverters from the runaway continuers, cutting the tail without
whipsawing out of the winners?

Two interventions, both knowable / causal (not lookahead):
  1. Event-stop: after fading, if the same-side liquidation RESUMES (cumulative
     same-side notional after entry reaches `resume_notional`), the exhaustion
     thesis is invalidated -> exit immediately at that moment. This is a real-time
     event rule, not a price stop, so it should not whipsaw on noise.
  2. BTC-veto: only fade if BTC is NOT trending in the cascade direction at entry
     (a SELL/down-spike faded LONG is skipped if BTC fell hard in the prior window
     -- then the move may be a real trend, not forced liquidation).

Baseline = 4h hold, taker fills (buy ask / sell bid), fee per side. Reports N,
sum, median, win, MAX_LOSS and T3R (top-3-winner-removed) on calibration and
holdout -- a real cut shows much better max_loss AND positive T3R on both splits.
Default cell: ETHUSDT SELL deep-V (>=28 bps overshoot), 200K.
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.research_s34_knowable_anchor_continuation import (
    book_at,
    load_liquidations,
    load_mark_index,
    mean,
    pctile,
    r1,
    r3,
    reconstruct_anchors,
    signed_return_bps,
)

DEFAULT_DB = ROOT / "data" / "microstructure.db"
OUT_DIR = ROOT / "reports" / "research" / "s34"
OUT_JSON = OUT_DIR / "S34_TAIL_CUT.json"
OUT_MD = OUT_DIR / "S34_TAIL_CUT.md"

HORIZON_SEC = 4 * 3600


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def metrics(vals: list[float]) -> dict[str, Any]:
    vals = [v for v in vals if math.isfinite(v)]
    if not vals:
        return {"n": 0, "sum": 0.0, "median": None, "win_rate": None, "max_loss": None, "t3r": None}
    s = sorted(vals, reverse=True)
    return {
        "n": len(vals),
        "sum": r1(sum(vals)),
        "median": r1(pctile(vals, 0.5)),
        "win_rate": r3(sum(1 for v in vals if v > 0) / len(vals)),
        "max_loss": r1(min(vals)),
        "t3r": r1(sum(s[3:])) if len(s) > 3 else r1(sum(s)),
    }


def split_metrics(rows: list[dict[str, Any]], key: str, holdout_frac: float) -> dict[str, Any]:
    rows = [r for r in rows if r.get(key) is not None]
    if not rows:
        return {"all": metrics([]), "cal": metrics([]), "hold": metrics([])}
    rows = sorted(rows, key=lambda r: r["entry_ts_ms"])
    cut = rows[int(len(rows) * (1.0 - holdout_frac))]["entry_ts_ms"]
    return {
        "all": metrics([r[key] for r in rows]),
        "cal": metrics([r[key] for r in rows if r["entry_ts_ms"] < cut]),
        "hold": metrics([r[key] for r in rows if r["entry_ts_ms"] >= cut]),
    }


def build_events(conn, symbol, side, threshold, *, bucket_sec, min_gap_sec, accel_window_sec, min_vdepth_bps,
                 fee_bps_side, max_book_staleness_sec, resume_notional, btc_veto_bps):
    fade_dir = "LONG" if side == "SELL" else "SHORT"
    marks = load_mark_index(conn, symbol)
    btc = load_mark_index(conn, "BTCUSDT")
    liqs = load_liquidations(conn, symbol, side, None, None)
    liq_ts = [int(r["ts_ms"]) for r in liqs]
    anchors = reconstruct_anchors(liqs, bucket_sec=bucket_sec, min_gap_sec=min_gap_sec,
                                  thresholds=(float(threshold),), accel_window_sec=accel_window_sec)
    rows = []
    for a in anchors:
        if float(a.threshold_usd) != float(threshold):
            continue
        start = marks.at_or_after(int(a.first_ts_ms))
        anc = marks.at_or_after(int(a.anchor_ts_ms))
        if not start or not anc or float(start[1]) <= 0:
            continue
        depth = ((float(start[1]) - float(anc[1])) if side == "SELL" else (float(anc[1]) - float(start[1]))) / float(start[1]) * 10_000.0
        if depth < float(min_vdepth_bps):
            continue
        entry_ts = int(a.anchor_ts_ms)
        eb = book_at(conn, symbol, entry_ts, max_book_staleness_sec)
        if not eb:
            continue
        entry_px = eb.ask if fade_dir == "LONG" else eb.bid

        def exit_net(exit_ts):
            xb = book_at(conn, symbol, int(exit_ts), max_book_staleness_sec)
            if not xb:
                return None
            xpx = xb.bid if fade_dir == "LONG" else xb.ask
            return signed_return_bps(fade_dir, float(entry_px), float(xpx)) - 2.0 * float(fee_bps_side)

        baseline = exit_net(entry_ts + HORIZON_SEC * 1000)

        # event-stop: same-side cumulative notional after entry reaches resume_notional
        lo = bisect.bisect_right(liq_ts, entry_ts)
        hi = bisect.bisect_right(liq_ts, entry_ts + HORIZON_SEC * 1000)
        cum = 0.0
        resume_ts = None
        for i in range(lo, hi):
            cum += float(liqs[i]["notional"])
            if cum >= float(resume_notional):
                resume_ts = int(liqs[i]["ts_ms"])
                break
        event_stop = exit_net(resume_ts) if resume_ts is not None else baseline

        # BTC state at entry (prior 15m); veto LONG fade if BTC fell hard, SHORT fade if BTC rose hard
        btc_ret = btc.ret_bps(entry_ts - 900_000, entry_ts)
        if btc_ret is None:
            btc_pass = True
        elif fade_dir == "LONG":
            btc_pass = btc_ret >= -float(btc_veto_bps)
        else:
            btc_pass = btc_ret <= float(btc_veto_bps)

        rows.append({
            "entry_ts_ms": entry_ts, "depth_bps": r1(depth), "btc_ret_bps": r1(btc_ret) if btc_ret is not None else None,
            "btc_pass": bool(btc_pass), "resumed": resume_ts is not None,
            "baseline_net": baseline, "event_stop_net": event_stop,
        })
    return rows


def variant_rows(rows, *, net_key, btc_filter):
    out = []
    for r in rows:
        if btc_filter and not r["btc_pass"]:
            continue
        if r.get(net_key) is None:
            continue
        out.append({"entry_ts_ms": r["entry_ts_ms"], "net": r[net_key]})
    return out


def render_md(report):
    cfg = report["config"]
    lines = [
        "# S34 Tail-Cut Conditioning",
        "",
        f"Generated: `{report['generated_at_utc']}`  |  {cfg['symbol']} {cfg['side']} deep-V>= {cfg['min_vdepth_bps']}bps, "
        f"{int(cfg['threshold']/1000)}K, 4h, fee {cfg['fee_bps_side']}/side, resume {int(cfg['resume_notional']/1000)}K, btc_veto {cfg['btc_veto_bps']}bps",
        "",
        f"Events: {report['event_n']}  |  resumed-after-entry: {report['resumed_n']}  |  btc-vetoed: {report['btc_vetoed_n']}",
        "",
        "Knowable interventions vs the -330 tail. A real cut: much better MAX_LOSS and positive T3R on BOTH splits.",
        "",
        "| Variant | split | N | sum | med | win% | max_loss | T3R |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, sm in report["variants"].items():
        for split in ("cal", "hold"):
            m = sm[split]
            wr = None if m["win_rate"] is None else r1(m["win_rate"] * 100.0)
            lines.append(f"| {name} | {split} | {m['n']} | {m['sum']} | {m['median']} | {wr} | {m['max_loss']} | {m['t3r']} |")
    lines.append("")
    lines.append("## Reverter vs continuer separation (big losers, baseline net < -100)")
    bl = report["big_losers"]
    lines.append(f"- big losers N={bl['n']}; of them resumed={bl['resumed']}, btc-failed={bl['btc_failed']}, "
                 f"either-flag={bl['either']} ({bl['either_pct']}% caught by a knowable flag)")
    lines.append("")
    return "\n".join(lines)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Tail-cut conditioning: event-stop + BTC-veto on the deep-V fade.")
    p.add_argument("--db", type=Path, default=DEFAULT_DB)
    p.add_argument("--symbol", default="ETHUSDT")
    p.add_argument("--side", default="SELL")
    p.add_argument("--threshold", type=float, default=200_000.0)
    p.add_argument("--min-vdepth-bps", type=float, default=28.0)
    p.add_argument("--bucket-sec", type=int, default=300)
    p.add_argument("--min-gap-sec", type=int, default=900)
    p.add_argument("--accel-window-sec", type=int, default=30)
    p.add_argument("--fee-bps-side", type=float, default=3.05)
    p.add_argument("--max-book-staleness-sec", type=int, default=10)
    p.add_argument("--holdout-frac", type=float, default=0.30)
    p.add_argument("--resume-notional", type=float, default=200_000.0)
    p.add_argument("--btc-veto-bps", type=float, default=30.0)
    p.add_argument("--json-out", type=Path, default=OUT_JSON)
    p.add_argument("--md-out", type=Path, default=OUT_MD)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    with sqlite3.connect(f"file:{args.db}?mode=ro", uri=True) as conn:
        rows = build_events(conn, args.symbol, args.side, float(args.threshold),
                            bucket_sec=int(args.bucket_sec), min_gap_sec=int(args.min_gap_sec),
                            accel_window_sec=int(args.accel_window_sec), min_vdepth_bps=float(args.min_vdepth_bps),
                            fee_bps_side=float(args.fee_bps_side), max_book_staleness_sec=int(args.max_book_staleness_sec),
                            resume_notional=float(args.resume_notional), btc_veto_bps=float(args.btc_veto_bps))
    hf = float(args.holdout_frac)
    variants = {
        "baseline_4h": split_metrics(variant_rows(rows, net_key="baseline_net", btc_filter=False), "net", hf),
        "event_stop": split_metrics(variant_rows(rows, net_key="event_stop_net", btc_filter=False), "net", hf),
        "btc_filter_4h": split_metrics(variant_rows(rows, net_key="baseline_net", btc_filter=True), "net", hf),
        "event_stop+btc_filter": split_metrics(variant_rows(rows, net_key="event_stop_net", btc_filter=True), "net", hf),
    }
    big = [r for r in rows if r.get("baseline_net") is not None and r["baseline_net"] < -100.0]
    resumed = sum(1 for r in big if r["resumed"])
    btc_failed = sum(1 for r in big if not r["btc_pass"])
    either = sum(1 for r in big if r["resumed"] or not r["btc_pass"])
    big_losers = {"n": len(big), "resumed": resumed, "btc_failed": btc_failed, "either": either,
                  "either_pct": r1(either / len(big) * 100.0) if big else None}
    report = {
        "generated_at_utc": utc_now(),
        "config": {"symbol": args.symbol, "side": args.side, "threshold": float(args.threshold),
                   "min_vdepth_bps": float(args.min_vdepth_bps), "fee_bps_side": float(args.fee_bps_side),
                   "resume_notional": float(args.resume_notional), "btc_veto_bps": float(args.btc_veto_bps)},
        "event_n": len(rows), "resumed_n": sum(1 for r in rows if r["resumed"]),
        "btc_vetoed_n": sum(1 for r in rows if not r["btc_pass"]),
        "variants": variants, "big_losers": big_losers, "rows": rows,
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    args.md_out.write_text(render_md(report), encoding="utf-8")
    print(render_md(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
