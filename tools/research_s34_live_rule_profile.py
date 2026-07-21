"""S34 Live-Rule Behaviour Profile (monitoring / risk -- NOT a new-edge search).

The permutation-null settled that there is no validated edge. This does something
different and honest: it CHARACTERIZES the live ETH-SELL deep-V fade rule so the
operator understands what is actually running -- when it fires (the "ripple
directions"), its recurring setup, its outcome/tail distribution, and -- most
useful for live risk -- what the WORST trades (-100+ losers) have in common, so
you know when to be cautious. This is descriptive monitoring, not prediction.
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
import sqlite3
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.research_s34_knowable_anchor_continuation import (
    load_liquidations, load_mark_index, mean, pctile, r1, r3, reconstruct_anchors, signed_return_bps,
)

DEFAULT_DB = ROOT / "data" / "microstructure.db"
OUT_DIR = ROOT / "reports" / "research" / "s34"
OUT_JSON = OUT_DIR / "S34_LIVE_RULE_PROFILE.json"
OUT_MD = OUT_DIR / "S34_LIVE_RULE_PROFILE.md"
HORIZON_SEC = 4 * 3600


def utc_now(): return datetime.now(timezone.utc).isoformat()
def month_of(ts): return datetime.fromtimestamp(ts / 1000.0, tz=timezone.utc).strftime("%Y-%m")
def hour_of(ts): return datetime.fromtimestamp(ts / 1000.0, tz=timezone.utc).hour


def session_of(ts):
    h = hour_of(ts)
    return "ASIA" if h < 7 else "EUROPE" if h < 13 else "US" if h < 21 else "OFF"


def med(v):
    v = [x for x in v if x is not None and math.isfinite(x)]
    return r1(pctile(v, 0.5)) if v else None


def build(conn, threshold, min_vdepth, cost, bucket_sec, min_gap_sec, accel_window_sec):
    eth = load_mark_index(conn, "ETHUSDT")
    btc = load_mark_index(conn, "BTCUSDT")
    eth_liq = load_liquidations(conn, "ETHUSDT", "SELL", None, None)
    btc_liq = load_liquidations(conn, "BTCUSDT", "SELL", None, None)
    sol_liq = load_liquidations(conn, "SOLUSDT", "SELL", None, None)
    bts = [int(r["ts_ms"]) for r in btc_liq]; sts = [int(r["ts_ms"]) for r in sol_liq]

    def conc(ts_list, rows, a, b):
        lo = bisect.bisect_right(ts_list, a); hi = bisect.bisect_right(ts_list, b)
        return sum(float(rows[i]["notional"]) for i in range(lo, hi)) / 1000.0

    anchors = reconstruct_anchors(eth_liq, bucket_sec=bucket_sec, min_gap_sec=min_gap_sec,
                                  thresholds=(float(threshold),), accel_window_sec=accel_window_sec)
    out = []
    for a in anchors:
        if float(a.threshold_usd) != float(threshold): continue
        s = eth.at_or_after(int(a.first_ts_ms)); anc = eth.at_or_after(int(a.anchor_ts_ms))
        if not s or not anc or float(s[1]) <= 0: continue
        depth = (float(s[1]) - float(anc[1])) / float(s[1]) * 10_000.0
        if depth < min_vdepth: continue
        t = int(a.anchor_ts_ms); ex = eth.at_or_after(t + HORIZON_SEC * 1000)
        if not ex: continue
        net = signed_return_bps("LONG", float(anc[1]), float(ex[1])) - cost
        out.append({"ts_ms": t, "month": month_of(t), "session": session_of(t), "net": net,
                    "depth_bps": r1(depth), "dominance": r1(float(a.running_single_liq_dominance)),
                    "sync_k": r1(conc(bts, btc_liq, t - 600000, t) + conc(sts, sol_liq, t - 600000, t)),
                    "btc_ret_10m": (r1(btc.ret_bps(t - 600000, t)) if btc.ret_bps(t - 600000, t) is not None else None)})
    return out


def dist(rows, key):
    c = Counter(r[key] for r in rows)
    return {k: {"n": v, "win": r3(sum(1 for r in rows if r[key] == k and r["net"] > 0) / v),
                "median": med([r["net"] for r in rows if r[key] == k])} for k, v in sorted(c.items())}


def render_md(rep):
    cfg = rep["config"]
    o = rep["overall"]
    lines = [
        "# S34 Live-Rule Behaviour Profile (monitoring / risk, NOT a new-edge search)",
        "",
        f"Generated: `{rep['generated_at_utc']}`  |  ETH-SELL deep-V>= {cfg['min_vdepth']}bps {int(cfg['threshold']/1000)}K 4h fade (live rule family), cost {cfg['cost']}bps",
        "",
        f"This DESCRIBES the live rule; it does not claim an edge (permutation-null already showed none).",
        "",
        f"## Overall: N={o['n']}, win {o['win']}, median {o['median']}, max_loss {o['max_loss']}, losers(<-100) {o['tail_n']} ({o['tail_pct']}%)",
        "",
        "## Ripple directions -- when it fires",
        "",
        "| Session | N | win | median |",
        "| --- | ---: | ---: | ---: |",
    ]
    for k, d in rep["by_session"].items():
        lines.append(f"| {k} | {d['n']} | {d['win']} | {d['median']} |")
    lines.append("")
    lines.append("| Month | N | win | median |")
    lines.append("| --- | ---: | ---: | ---: |")
    for k, d in rep["by_month"].items():
        lines.append(f"| {k} | {d['n']} | {d['win']} | {d['median']} |")
    lines.append("")
    w = rep["worst_vs_winners"]
    lines.append("## Failure geometry -- what the WORST trades (<-100) share (live risk watch-out)")
    lines.append(f"- worst N={w['worst_n']} | winners N={w['win_n']}")
    lines.append(f"- sync_k (concurrent cross-asset sell-liq): worst {w['worst_sync']} vs winners {w['win_sync']}")
    lines.append(f"- depth_bps: worst {w['worst_depth']} vs winners {w['win_depth']}")
    lines.append(f"- btc_ret_10m: worst {w['worst_btc']} vs winners {w['win_btc']}")
    lines.append(f"- worst-trade session mode: {w['worst_session']}")
    lines.append("")
    return "\n".join(lines)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Descriptive behaviour profile of the live ETH-SELL deep-V fade rule.")
    p.add_argument("--db", type=Path, default=DEFAULT_DB)
    p.add_argument("--threshold", type=float, default=200_000.0)
    p.add_argument("--min-vdepth-bps", type=float, default=28.0)
    p.add_argument("--bucket-sec", type=int, default=300)
    p.add_argument("--min-gap-sec", type=int, default=900)
    p.add_argument("--accel-window-sec", type=int, default=30)
    p.add_argument("--fee-bps-side", type=float, default=3.05)
    p.add_argument("--modeled-spread-bps", type=float, default=2.0)
    return p.parse_args(argv)


def main(argv=None) -> int:
    a = parse_args(argv)
    cost = 2.0 * float(a.fee_bps_side) + float(a.modeled_spread_bps)
    with sqlite3.connect(f"file:{a.db}?mode=ro", uri=True) as conn:
        rows = build(conn, float(a.threshold), float(a.min_vdepth_bps), cost, int(a.bucket_sec), int(a.min_gap_sec), int(a.accel_window_sec))
    nets = [r["net"] for r in rows]
    tail = [r for r in rows if r["net"] < -100.0]; wins = [r for r in rows if r["net"] > 0]
    overall = {"n": len(rows), "win": r3(sum(1 for x in nets if x > 0) / len(nets)) if nets else None,
               "median": med(nets), "max_loss": r1(min(nets)) if nets else None,
               "tail_n": len(tail), "tail_pct": r1(len(tail) / len(rows) * 100.0) if rows else None}
    worst = {"worst_n": len(tail), "win_n": len(wins),
             "worst_sync": med([r["sync_k"] for r in tail]), "win_sync": med([r["sync_k"] for r in wins]),
             "worst_depth": med([r["depth_bps"] for r in tail]), "win_depth": med([r["depth_bps"] for r in wins]),
             "worst_btc": med([r["btc_ret_10m"] for r in tail]), "win_btc": med([r["btc_ret_10m"] for r in wins]),
             "worst_session": (Counter(r["session"] for r in tail).most_common(1)[0][0] if tail else None)}
    rep = {"generated_at_utc": utc_now(),
           "config": {"threshold": float(a.threshold), "min_vdepth": float(a.min_vdepth_bps), "cost": r1(cost)},
           "overall": overall, "by_session": dist(rows, "session"), "by_month": dist(rows, "month"),
           "worst_vs_winners": worst, "rows": rows}
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(rep, indent=2, ensure_ascii=True), encoding="utf-8")
    OUT_MD.write_text(render_md(rep), encoding="utf-8")
    print(render_md(rep))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
