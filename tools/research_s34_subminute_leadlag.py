"""S34 Sub-Minute Lead-Lag.

At 1-minute resolution BTC/ETH/SOL are contemporaneous (corr 0.86 @ lag 0, ~0 @
+1min) -- no tradeable cross-asset precursor. The precursor's last refuge is the
SECOND scale: does BTC lead ETH by a few seconds? This samples marks at 5s
resolution over a recent dense window and measures corr(lead[t], follow[t+lag])
for lags -60s..+60s. A peak at a positive lag (seconds) means the leader leads --
a precursor exists, though at HFT/latency scale. A peak at lag 0 closes the
precursor question at every measurable resolution.
"""

from __future__ import annotations

import argparse
import math
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.research_s34_knowable_anchor_continuation import load_mark_index, r3

DEFAULT_DB = ROOT / "data" / "microstructure.db"
OUT_DIR = ROOT / "reports" / "research" / "s34"
OUT_JSON = OUT_DIR / "S34_SUBMINUTE_LEADLAG.json"
OUT_MD = OUT_DIR / "S34_SUBMINUTE_LEADLAG.md"

STEP_MS = 5000
LAGS_STEPS = list(range(-12, 13))  # -60s .. +60s in 5s steps


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sample_returns(marks, start_ms, end_ms, step_ms):
    ret = []
    prev = None
    t = int(start_ms)
    while t <= int(end_ms):
        mk = marks.at_or_before(t)
        if mk and prev is not None and float(prev) > 0 and float(mk[1]) > 0:
            ret.append(math.log(float(mk[1]) / float(prev)))
        elif prev is not None:
            ret.append(0.0)
        if mk:
            prev = float(mk[1])
        t += step_ms
    return ret


def pearson(x, y):
    n = len(x)
    if n < 100:
        return None
    mx, my = sum(x) / n, sum(y) / n
    sx = math.sqrt(sum((a - mx) ** 2 for a in x))
    sy = math.sqrt(sum((b - my) ** 2 for b in y))
    if sx == 0 or sy == 0:
        return None
    return sum((x[i] - mx) * (y[i] - my) for i in range(n)) / (sx * sy)


def lead_lag(lead_ret, follow_ret):
    out = {}
    n = min(len(lead_ret), len(follow_ret))
    lead_ret, follow_ret = lead_ret[:n], follow_ret[:n]
    for s in LAGS_STEPS:
        if s >= 0:
            a, b = lead_ret[: n - s], follow_ret[s:]
        else:
            a, b = lead_ret[-s:], follow_ret[: n + s]
        c = pearson(a, b)
        out[s * STEP_MS // 1000] = r3(c) if c is not None else None
    return out


def peak(ll):
    vals = {int(k): v for k, v in ll.items() if v is not None}
    if not vals:
        return None, None
    pk = max(vals, key=lambda k: abs(vals[k]))
    return pk, vals[pk]


def render_md(report):
    lines = [
        "# S34 Sub-Minute Lead-Lag (precursor's last refuge: the second scale)",
        "",
        f"Generated: `{report['generated_at_utc']}`  |  {report['window_days']}d window, {STEP_MS//1000}s returns, samples={report['n_samples']}",
        "",
        "Peak at a positive lag (sec) => lead leads follow at that horizon (HFT precursor). Peak at 0 => contemporaneous.",
        "",
        "| Pair (lead->follow) | peak lag (s) | peak corr | corr@-5s | corr@0 | corr@+5s | corr@+10s |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for pair, ll in report["lead_lag"].items():
        pk, pv = peak(ll)
        lines.append(f"| {pair} | {pk} | {pv} | {ll.get('-5')} | {ll.get('0')} | {ll.get('5')} | {ll.get('10')} |")
    lines.append("")
    return "\n".join(lines)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Second-scale cross-asset lead-lag (precursor at HFT scale?).")
    p.add_argument("--db", type=Path, default=DEFAULT_DB)
    p.add_argument("--window-days", type=float, default=7.0)
    p.add_argument("--json-out", type=Path, default=OUT_JSON)
    p.add_argument("--md-out", type=Path, default=OUT_MD)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    with sqlite3.connect(f"file:{args.db}?mode=ro", uri=True) as conn:
        eth = load_mark_index(conn, "ETHUSDT")
        btc = load_mark_index(conn, "BTCUSDT")
        sol = load_mark_index(conn, "SOLUSDT")
        end = max(eth.ts[-1] if eth.ts else 0, btc.ts[-1] if btc.ts else 0)
        start = end - int(float(args.window_days) * 86_400_000)
        eth_r = sample_returns(eth, start, end, STEP_MS)
        btc_r = sample_returns(btc, start, end, STEP_MS)
        sol_r = sample_returns(sol, start, end, STEP_MS)
    lead_lag_out = {
        "BTC->ETH": lead_lag(btc_r, eth_r),
        "BTC->SOL": lead_lag(btc_r, sol_r),
        "ETH->SOL": lead_lag(eth_r, sol_r),
        "ETH->BTC": lead_lag(eth_r, btc_r),
    }
    ll_json = {pair: {str(k): v for k, v in d.items()} for pair, d in lead_lag_out.items()}
    report = {"generated_at_utc": utc_now(), "window_days": float(args.window_days),
              "n_samples": min(len(eth_r), len(btc_r), len(sol_r)), "lead_lag": ll_json}
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    args.md_out.write_text(render_md(report), encoding="utf-8")
    print(render_md(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
