"""S34 Cross-Asset Lead-Lag.

We searched for the precursor inside each asset's own flow (onset/velocity) and it
was dead. But the precursor may live in ANOTHER asset: if BTC leads ETH leads SOL,
then the leader's move is a knowable precursor to the follower's cascade -- you act
on the earlier signal, before the follower's move is spent. This measures the
lead-lag / correlation structure (the user's "correlation + before + after" chain).

Two parts:
  A. Global lead-lag: 1-minute log returns sampled over a recent window; Pearson
     corr( lead[t], follow[t+lag] ) for lag in -10..+10 min. Peak at positive lag
     => the lead asset leads the follow asset by that many minutes.
  B. Cascade precursor: around each ETH deep-V SELL cascade, the BTC return in the
     preceding window ([-5m,0], [-1m,0]) -- is there a consistent BTC precursor,
     and does its size predict the ETH fade outcome (reversal vs runaway)?
"""

from __future__ import annotations

import argparse
import math
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.research_s34_knowable_anchor_continuation import (
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
OUT_JSON = OUT_DIR / "S34_CROSS_ASSET_LEADLAG.json"
OUT_MD = OUT_DIR / "S34_CROSS_ASSET_LEADLAG.md"

HORIZON_SEC = 4 * 3600
LAGS_MIN = list(range(-10, 11))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sample_minute_returns(marks, start_ms, end_ms):
    """1-min log returns sampled from the mark index."""
    ts, ret = [], []
    prev = None
    step = 60_000
    t = int(start_ms)
    while t <= int(end_ms):
        mk = marks.at_or_before(t)
        if mk and prev is not None and float(prev) > 0 and float(mk[1]) > 0:
            ts.append(t)
            ret.append(math.log(float(mk[1]) / float(prev)))
        if mk:
            prev = float(mk[1])
        t += step
    return ts, ret


def pearson(x, y):
    n = len(x)
    if n < 30:
        return None
    mx, my = sum(x) / n, sum(y) / n
    sx = math.sqrt(sum((a - mx) ** 2 for a in x))
    sy = math.sqrt(sum((b - my) ** 2 for b in y))
    if sx == 0 or sy == 0:
        return None
    return sum((x[i] - mx) * (y[i] - my) for i in range(n)) / (sx * sy)


def lead_lag(lead_ret, follow_ret):
    """corr(lead[t], follow[t+lag]); lead and follow are aligned by index (same minute grid)."""
    out = {}
    n = min(len(lead_ret), len(follow_ret))
    lead_ret, follow_ret = lead_ret[:n], follow_ret[:n]
    for lag in LAGS_MIN:
        if lag >= 0:
            a = lead_ret[: n - lag]
            b = follow_ret[lag:]
        else:
            a = lead_ret[-lag:]
            b = follow_ret[: n + lag]
        c = pearson(a, b)
        out[lag] = r3(c) if c is not None else None
    return out


def med(vals):
    vals = [v for v in vals if v is not None and math.isfinite(v)]
    return r1(pctile(vals, 0.5)) if vals else None


def cascade_precursor(conn, symbol, threshold, *, bucket_sec, min_gap_sec, accel_window_sec, min_vdepth_bps, cost):
    marks = load_mark_index(conn, symbol)
    btc = load_mark_index(conn, "BTCUSDT")
    liqs = load_liquidations(conn, symbol, "SELL", None, None)
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
        depth = (float(start[1]) - float(anc[1])) / float(start[1]) * 10_000.0
        if depth < float(min_vdepth_bps):
            continue
        t = int(a.anchor_ts_ms)
        ex = marks.at_or_after(t + HORIZON_SEC * 1000)
        if not ex:
            continue
        net = signed_return_bps("LONG", float(anc[1]), float(ex[1])) - cost
        rows.append({"net": net,
                     "btc_pre_5m": btc.ret_bps(t - 5 * 60 * 1000, t),
                     "btc_pre_1m": btc.ret_bps(t - 60 * 1000, t)})
    win = [r for r in rows if r["net"] > 0]
    runaway = [r for r in rows if r["net"] < -100.0]
    return {
        "n": len(rows),
        "btc_pre_5m_all": med([r["btc_pre_5m"] for r in rows]),
        "btc_pre_1m_all": med([r["btc_pre_1m"] for r in rows]),
        "btc_pre_5m_win": med([r["btc_pre_5m"] for r in win]),
        "btc_pre_5m_runaway": med([r["btc_pre_5m"] for r in runaway]),
    }


def render_md(report):
    lines = [
        "# S34 Cross-Asset Lead-Lag (is the precursor in another asset?)",
        "",
        f"Generated: `{report['generated_at_utc']}`  |  window {report['window_days']}d, 1-min returns",
        "",
        "## A. Lead-lag cross-correlation (corr lead[t] vs follow[t+lag]); peak at +lag => lead leads follow",
        "",
        "| Pair (lead->follow) | peak lag (min) | peak corr | corr@0 | corr@+1 | corr@+2 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for pair, ll in report["lead_lag"].items():
        vals = {int(k): v for k, v in ll.items() if v is not None}
        if vals:
            peak_lag = max(vals, key=lambda k: abs(vals[k]))
            lines.append(f"| {pair} | {peak_lag} | {vals[peak_lag]} | {ll.get('0')} | {ll.get('1')} | {ll.get('2')} |")
    lines.append("")
    lines.append("## B. BTC precursor around ETH deep-V SELL cascades")
    pc = report["precursor"]
    lines.append(f"- N={pc['n']}; BTC return before cross: [-5m]={pc['btc_pre_5m_all']}bps, [-1m]={pc['btc_pre_1m_all']}bps")
    lines.append(f"- BTC[-5m] median: winners={pc['btc_pre_5m_win']} vs runaways={pc['btc_pre_5m_runaway']}")
    lines.append("")
    return "\n".join(lines)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Cross-asset lead-lag / precursor analysis.")
    p.add_argument("--db", type=Path, default=DEFAULT_DB)
    p.add_argument("--window-days", type=int, default=45)
    p.add_argument("--threshold", type=float, default=200_000.0)
    p.add_argument("--min-vdepth-bps", type=float, default=28.0)
    p.add_argument("--bucket-sec", type=int, default=300)
    p.add_argument("--min-gap-sec", type=int, default=900)
    p.add_argument("--accel-window-sec", type=int, default=30)
    p.add_argument("--fee-bps-side", type=float, default=3.05)
    p.add_argument("--modeled-spread-bps", type=float, default=2.0)
    p.add_argument("--json-out", type=Path, default=OUT_JSON)
    p.add_argument("--md-out", type=Path, default=OUT_MD)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    cost = 2.0 * float(args.fee_bps_side) + float(args.modeled_spread_bps)
    with sqlite3.connect(f"file:{args.db}?mode=ro", uri=True) as conn:
        eth = load_mark_index(conn, "ETHUSDT")
        btc = load_mark_index(conn, "BTCUSDT")
        sol = load_mark_index(conn, "SOLUSDT")
        end = max(eth.ts[-1] if eth.ts else 0, btc.ts[-1] if btc.ts else 0)
        start = end - int(args.window_days) * 86_400_000
        _, eth_r = sample_minute_returns(eth, start, end)
        _, btc_r = sample_minute_returns(btc, start, end)
        _, sol_r = sample_minute_returns(sol, start, end)
        lead_lag_out = {
            "BTC->ETH": lead_lag(btc_r, eth_r),
            "BTC->SOL": lead_lag(btc_r, sol_r),
            "ETH->SOL": lead_lag(eth_r, sol_r),
            "ETH->BTC": lead_lag(eth_r, btc_r),
        }
        precursor = cascade_precursor(conn, "ETHUSDT", float(args.threshold), bucket_sec=int(args.bucket_sec),
                                      min_gap_sec=int(args.min_gap_sec), accel_window_sec=int(args.accel_window_sec),
                                      min_vdepth_bps=float(args.min_vdepth_bps), cost=cost)
    # JSON keys must be strings
    ll_json = {pair: {str(k): v for k, v in d.items()} for pair, d in lead_lag_out.items()}
    report = {"generated_at_utc": utc_now(), "window_days": int(args.window_days),
              "lead_lag": ll_json, "precursor": precursor}
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    args.md_out.write_text(render_md(report), encoding="utf-8")
    print(render_md(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
