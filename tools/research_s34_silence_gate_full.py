"""S34 Silence Gate Full Suite.

Status: RESEARCH_ONLY_NO_LIVE_CHANGE

Following the puzzle suite that found:
  - Silence gate (no cascade in next 30min): Hold T3R=+7733, WR=70.1%  <- REAL
  - BULL_PULLBACK: Hold N=22, WR=90.9%, permutation PASS (p=0.03)

This suite tests:
  A. Silence gate permutation null (cal + hold) — is +7733 statistically real?
  B. Silence + BULL_PULLBACK combo — overlap and combined stats
  C. sync_k as predictor of silence — can we predict silence from entry-time sync_k?
     (proxy for live applicability without entry delay)
  D. 200K threshold specific silence — matching the live rule exactly
  E. Silence breakdown by sync_k level — does silence work better in low-sync regimes?
  F. prior4h trend + silence combo — simplest gate combination

SAF-02: research-only. DAT-01: no lookahead (silence uses post-entry window,
flagged as management signal, not prediction). DAT-03: seed=42.
"""

from __future__ import annotations

import bisect
import json
import math
import random
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.s34_navigation_full_followup import (
    load_jsonl, r1, r3, NAV_EVENTS, FEE_BPS,
)

DEFAULT_DB   = ROOT / "data" / "microstructure.db"
OUT_JSON     = ROOT / "reports" / "research" / "s34" / "S34_SILENCE_GATE_FULL.json"
OUT_MD       = ROOT / "reports" / "research" / "s34" / "S34_SILENCE_GATE_FULL.md"

HOLDOUT_FRAC    = 0.30
SEED            = 42
N_PERM          = 1000
MIN_N           = 15
SILENCE_LO_MS   = 60  * 1000        # skip 1st 60s (same bucket)
SILENCE_HI_MS   = 30  * 60 * 1000   # 30-min silence window
PROP_THRESH     = 50_000.0           # next cascade size for silence check
SYNC_WINDOW_MS  = 10  * 60 * 1000   # 10-min prior window for sync_k
SYNC_BINS       = (0, 50_000, 100_000, 200_000, 300_000, 500_000, 1_000_000)
PRIOR4H_CUTS    = (-50, 0, 25, 50, 100)
LIVE_THRESHOLD  = 200_000.0         # live rule: 200K


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()

def ts_utc(ts: int) -> str:
    return datetime.fromtimestamp(int(ts)/1000, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def t3r(vals: list[float]) -> float:
    if len(vals) <= 3:
        return sum(vals)
    return sum(sorted(vals, reverse=True)[3:])

def qs(vals: list[float]) -> dict:
    if not vals:
        return {"n": 0, "t3r": None, "sum": None, "med": None, "win": None, "maxL": None}
    return {
        "n":    len(vals),
        "t3r":  r1(t3r(vals)) if len(vals) >= MIN_N else None,
        "sum":  r1(sum(vals)),
        "med":  r1(median(vals)),
        "win":  r3(sum(1 for v in vals if v > 0)/len(vals)),
        "maxL": r1(min(vals)),
    }

def pctile(vals: list[float], p: float) -> float:
    v = sorted(x for x in vals if math.isfinite(x))
    if not v:
        return float("nan")
    i = p * (len(v)-1); lo = int(i)
    return v[lo] + (i-lo)*(v[min(lo+1, len(v)-1)]-v[lo])


# ---------------------------------------------------------------------------
# DB loaders
# ---------------------------------------------------------------------------

def load_liq(conn: sqlite3.Connection, symbol: str, side: str) -> tuple[list[int], list[float]]:
    rows = conn.execute(
        "SELECT ts_ms, notional FROM liquidations WHERE symbol=? AND side=? ORDER BY ts_ms",
        (symbol, side),
    ).fetchall()
    return [int(r[0]) for r in rows], [float(r[1]) for r in rows]

def win_sum(ts_list: list[int], vals: list[float], lo: int, hi: int) -> float:
    a = bisect.bisect_left(ts_list, lo)
    b = bisect.bisect_right(ts_list, hi)
    return sum(vals[i] for i in range(a, b))

def win_count_thresh(ts_list: list[int], vals: list[float],
                     lo: int, hi: int, thr: float) -> int:
    a = bisect.bisect_left(ts_list, lo)
    b = bisect.bisect_right(ts_list, hi)
    return sum(1 for i in range(a, b) if vals[i] >= thr)


# ---------------------------------------------------------------------------
# Annotate rows
# ---------------------------------------------------------------------------

def annotate(rows: list[dict],
             eth_sell_ts, eth_sell_not,
             btc_sell_ts, btc_sell_not,
             sol_sell_ts, sol_sell_not) -> list[dict]:
    out = []
    for r in rows:
        ts = int(r["signal_ts_ms"])
        # silence: no ETH SELL cascade >=50K in (ts+60s, ts+30min)
        cnt = win_count_thresh(eth_sell_ts, eth_sell_not,
                               ts + SILENCE_LO_MS, ts + SILENCE_HI_MS, PROP_THRESH)
        silence = cnt == 0
        # sync_k: BTC+SOL SELL in prior 10 min
        b = win_sum(btc_sell_ts, btc_sell_not, ts - SYNC_WINDOW_MS, ts)
        s = win_sum(sol_sell_ts, sol_sell_not, ts - SYNC_WINDOW_MS, ts)
        sync_k = b + s
        # flags
        is_bull = "BULL_PULLBACK" in (r.get("tags") or [])
        item = dict(r)
        item["silence"]   = silence
        item["sync_k"]    = sync_k
        item["is_bull"]   = is_bull
        item["is_live"]   = float(r.get("threshold_usd") or 0) >= LIVE_THRESHOLD
        item["prior4h"]   = float(r.get("prior4h_bps") or 0)
        out.append(item)
    return out


# ---------------------------------------------------------------------------
# A — Permutation null (silence gate)
# ---------------------------------------------------------------------------

def test_a_permutation(rows: list[dict], label: str) -> dict:
    """Shuffle net_2h_bps, compute T3R for silence subset, repeat N_PERM times."""
    rng = random.Random(SEED)
    sil = [r for r in rows if r["silence"]]
    all_vals = [float(r["net_2h_bps"]) for r in rows if r.get("net_2h_bps") is not None]
    sil_vals = [float(r["net_2h_bps"]) for r in sil if r.get("net_2h_bps") is not None]
    real_t3r  = t3r(sil_vals) if len(sil_vals) >= MIN_N else float("nan")

    null_t3rs = []
    n_sil = len(sil_vals)
    for _ in range(N_PERM):
        shuf = rng.sample(all_vals, min(n_sil, len(all_vals)))
        null_t3rs.append(t3r(shuf))
    null_p95 = pctile(null_t3rs, 0.95)
    null_p99 = pctile(null_t3rs, 0.99)
    p_right  = sum(1 for v in null_t3rs if math.isfinite(v) and v >= real_t3r)/len(null_t3rs)

    return {
        "split":     label,
        "n_silence": len(sil_vals),
        "n_noisy":   len(rows) - len(sil),
        "real_t3r":  r1(real_t3r),
        "null_p95":  r1(null_p95),
        "null_p99":  r1(null_p99),
        "p_right":   r3(p_right),
        "verdict":   "PASS" if p_right < 0.05 else "ARTIFACT",
        "silence_stats": qs(sil_vals),
        "noisy_stats":   qs([float(r["net_2h_bps"]) for r in rows
                             if not r["silence"] and r.get("net_2h_bps") is not None]),
    }


# ---------------------------------------------------------------------------
# B — Silence + BULL_PULLBACK combo
# ---------------------------------------------------------------------------

def test_b_silence_bull(cal: list[dict], hold: list[dict]) -> dict:
    groups = {
        "silence_AND_bull":      lambda r: r["silence"] and r["is_bull"],
        "silence_AND_NOT_bull":  lambda r: r["silence"] and not r["is_bull"],
        "noisy_AND_bull":        lambda r: not r["silence"] and r["is_bull"],
        "noisy_AND_NOT_bull":    lambda r: not r["silence"] and not r["is_bull"],
        "silence_only":          lambda r: r["silence"],
        "bull_only":             lambda r: r["is_bull"],
    }
    result = {}
    for name, fn in groups.items():
        cal_v  = [float(r["net_2h_bps"]) for r in cal  if fn(r) and r.get("net_2h_bps") is not None]
        hold_v = [float(r["net_2h_bps"]) for r in hold if fn(r) and r.get("net_2h_bps") is not None]
        result[name] = {"cal": qs(cal_v), "hold": qs(hold_v)}
    return result


# ---------------------------------------------------------------------------
# C — sync_k as predictor of silence (live applicability proxy)
# ---------------------------------------------------------------------------

def test_c_sync_predicts_silence(cal: list[dict], hold: list[dict]) -> dict:
    """For each sync_k bin, compute:
       - silence_rate (how often is the next 30min silent?)
       - fade outcome for silent trades in that bin
    This answers: 'can we use sync_k at entry to predict silence?'
    """
    result = {}
    bins = list(zip(SYNC_BINS[:-1], SYNC_BINS[1:])) + [(SYNC_BINS[-1], float("inf"))]
    for lo, hi in bins:
        label = f"sync_{int(lo/1000)}K_to_{int(hi/1000) if hi != float('inf') else 'inf'}K"
        for split_label, rows in [("cal", cal), ("hold", hold)]:
            sub = [r for r in rows if lo <= r["sync_k"] < hi]
            if not sub:
                continue
            n_sil  = sum(1 for r in sub if r["silence"])
            sil_v  = [float(r["net_2h_bps"]) for r in sub if r["silence"] and r.get("net_2h_bps") is not None]
            noisy_v= [float(r["net_2h_bps"]) for r in sub if not r["silence"] and r.get("net_2h_bps") is not None]
            all_v  = [float(r["net_2h_bps"]) for r in sub if r.get("net_2h_bps") is not None]
            result.setdefault(label, {})[split_label] = {
                "n":           len(sub),
                "silence_rate":r3(n_sil/len(sub)),
                "silence":     qs(sil_v),
                "noisy":       qs(noisy_v),
                "all":         qs(all_v),
            }
    return result


# ---------------------------------------------------------------------------
# D — 200K threshold specific (live rule matching)
# ---------------------------------------------------------------------------

def test_d_live_rule(cal: list[dict], hold: list[dict]) -> dict:
    result = {}
    for split_label, rows in [("cal", cal), ("hold", hold)]:
        live = [r for r in rows if r["is_live"]]
        sil_v   = [float(r["net_2h_bps"]) for r in live if r["silence"] and r.get("net_2h_bps") is not None]
        noisy_v = [float(r["net_2h_bps"]) for r in live if not r["silence"] and r.get("net_2h_bps") is not None]
        all_v   = [float(r["net_2h_bps"]) for r in live if r.get("net_2h_bps") is not None]
        n_sil   = sum(1 for r in live if r["silence"])
        result[split_label] = {
            "n_live":        len(live),
            "silence_n":     n_sil,
            "silence_rate":  r3(n_sil/len(live)) if live else None,
            "silence":       qs(sil_v),
            "noisy":         qs(noisy_v),
            "all":           qs(all_v),
        }
    return result


# ---------------------------------------------------------------------------
# E — Silence by sync_k level (does silence work better in low-sync?)
# ---------------------------------------------------------------------------

def test_e_silence_by_sync(cal: list[dict], hold: list[dict]) -> dict:
    """For SILENT events only, break by sync_k level."""
    thresholds = [50_000, 100_000, 200_000, 300_000, 500_000]
    result = {}
    for thr in thresholds:
        label = f"silence_AND_sync_lt_{int(thr/1000)}K"
        fn = lambda r, t=thr: r["silence"] and r["sync_k"] < t
        cal_v  = [float(r["net_2h_bps"]) for r in cal  if fn(r) and r.get("net_2h_bps") is not None]
        hold_v = [float(r["net_2h_bps"]) for r in hold if fn(r) and r.get("net_2h_bps") is not None]
        result[label] = {"cal": qs(cal_v), "hold": qs(hold_v)}

    # Also: silence AND high sync
    for thr in [200_000, 300_000]:
        label = f"silence_AND_sync_gte_{int(thr/1000)}K"
        fn = lambda r, t=thr: r["silence"] and r["sync_k"] >= t
        cal_v  = [float(r["net_2h_bps"]) for r in cal  if fn(r) and r.get("net_2h_bps") is not None]
        hold_v = [float(r["net_2h_bps"]) for r in hold if fn(r) and r.get("net_2h_bps") is not None]
        result[label] = {"cal": qs(cal_v), "hold": qs(hold_v)}

    return result


# ---------------------------------------------------------------------------
# F — prior4h trend + silence combo
# ---------------------------------------------------------------------------

def test_f_prior4h_silence(cal: list[dict], hold: list[dict]) -> dict:
    result = {}
    for cut in PRIOR4H_CUTS:
        label = f"silence_AND_prior4h_gt_{cut}"
        fn = lambda r, c=cut: r["silence"] and r["prior4h"] > c
        cal_v  = [float(r["net_2h_bps"]) for r in cal  if fn(r) and r.get("net_2h_bps") is not None]
        hold_v = [float(r["net_2h_bps"]) for r in hold if fn(r) and r.get("net_2h_bps") is not None]
        result[label] = {"cal": qs(cal_v), "hold": qs(hold_v)}

    # Triple combo: silence + prior4h > 0 + sync_k < 200K
    for p_cut, s_cut in [(0, 200_000), (0, 300_000), (25, 200_000)]:
        label = f"silence_prior4h_gt_{p_cut}_sync_lt_{int(s_cut/1000)}K"
        fn = lambda r, pc=p_cut, sc=s_cut: (
            r["silence"] and r["prior4h"] > pc and r["sync_k"] < sc
        )
        cal_v  = [float(r["net_2h_bps"]) for r in cal  if fn(r) and r.get("net_2h_bps") is not None]
        hold_v = [float(r["net_2h_bps"]) for r in hold if fn(r) and r.get("net_2h_bps") is not None]
        result[label] = {"cal": qs(cal_v), "hold": qs(hold_v)}

    return result


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------

def fmt(d: dict) -> str:
    return f"N={d['n']} T3R={d.get('t3r')} med={d['med']} win={d['win']} maxL={d['maxL']}"

def render_md(res: dict) -> str:
    sp = res["split"]
    lines = [
        "# S34 Silence Gate Full Suite",
        "",
        f"Generated: `{res['generated_at_utc']}`  |  Status: `RESEARCH_ONLY_NO_LIVE_CHANGE`",
        "",
        f"Cal: {sp['cal_n']} events ({sp['cal_start']} to {sp['cal_end']})",
        f"Hold: {sp['hold_n']} events ({sp['hold_start']} to {sp['hold_end']})",
        "",
    ]

    # A
    lines += ["## A. Silence Gate Permutation Null", ""]
    for pm in res["test_a"]:
        v = pm["verdict"]
        lines += [
            f"### {pm['split']} split (N_silence={pm['n_silence']}, N_noisy={pm['n_noisy']})",
            "",
            f"- Real T3R={pm['real_t3r']}  |  Null p95={pm['null_p95']}  |  Null p99={pm['null_p99']}",
            f"- p-right={pm['p_right']}  ->  **{v}**",
            f"- Silence: {fmt(pm['silence_stats'])}",
            f"- Noisy:   {fmt(pm['noisy_stats'])}",
            "",
        ]

    # B
    lines += ["## B. Silence + BULL_PULLBACK Combo", ""]
    lines += [
        "| Group | Cal N | Cal T3R | Cal med | Cal win | Hold N | Hold T3R | Hold med | Hold win | Hold maxL |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, d in res["test_b"].items():
        c = d["cal"]; h = d["hold"]
        lines.append(
            f"| {name} | {c['n']} | {c.get('t3r')} | {c['med']} | {c['win']} |"
            f" {h['n']} | {h.get('t3r')} | {h['med']} | {h['win']} | {h['maxL']} |"
        )
    lines.append("")

    # C
    lines += ["## C. sync_k as Predictor of Silence (Live Proxy)", ""]
    lines += [
        "| sync_k bin | Split | N | Silence rate | Silence T3R | Silence med | Silence win | All T3R |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for bin_label, splits in res["test_c"].items():
        for split_label, d in splits.items():
            s = d["silence"]; a = d["all"]
            lines.append(
                f"| {bin_label} | {split_label} | {d['n']} | {d['silence_rate']} |"
                f" {s.get('t3r')} | {s['med']} | {s['win']} | {a.get('t3r')} |"
            )
    lines.append("")

    # D
    lines += ["## D. Live Rule (200K threshold) Silence Analysis", ""]
    lines += [
        "| Split | N live | Silence N | Silence rate | Silence T3R | Silence med | Silence win | Noisy T3R | Noisy med | All T3R |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for split_label, d in res["test_d"].items():
        s = d["silence"]; no = d["noisy"]; a = d["all"]
        lines.append(
            f"| {split_label} | {d['n_live']} | {d['silence_n']} | {d['silence_rate']} |"
            f" {s.get('t3r')} | {s['med']} | {s['win']} |"
            f" {no.get('t3r')} | {no['med']} | {a.get('t3r')} |"
        )
    lines.append("")

    # E
    lines += ["## E. Silence by sync_k Level", ""]
    lines += [
        "| Gate | Cal N | Cal T3R | Cal med | Cal win | Hold N | Hold T3R | Hold med | Hold win | Hold maxL |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, d in res["test_e"].items():
        c = d["cal"]; h = d["hold"]
        lines.append(
            f"| {label} | {c['n']} | {c.get('t3r')} | {c['med']} | {c['win']} |"
            f" {h['n']} | {h.get('t3r')} | {h['med']} | {h['win']} | {h['maxL']} |"
        )
    lines.append("")

    # F
    lines += ["## F. prior4h Trend + Silence Combo", ""]
    lines += [
        "| Gate | Cal N | Cal T3R | Cal med | Cal win | Hold N | Hold T3R | Hold med | Hold win | Hold maxL |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, d in res["test_f"].items():
        c = d["cal"]; h = d["hold"]
        lines.append(
            f"| {label} | {c['n']} | {c.get('t3r')} | {c['med']} | {c['win']} |"
            f" {h['n']} | {h.get('t3r')} | {h['med']} | {h['win']} | {h['maxL']} |"
        )
    lines.append("")

    lines += [
        "## Key Questions Answered",
        "",
        "- A: Is silence gate statistically real on holdout? (p-right < 0.05 = YES)",
        "- B: Does silence+BULL_PULLBACK combo beat silence alone?",
        "- C: Can sync_k at entry predict whether silence will occur? (live proxy)",
        "- D: Does the 200K live-rule specific silence gate work in holdout?",
        "- E: Does silence work better when sync_k is low? (regime interaction)",
        "- F: Does prior4h trend + silence combo improve hold signal?",
        "",
        "RESEARCH_ONLY. No live change without operator sign-off.",
    ]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    print("Loading nav events...")
    all_rows = load_jsonl(NAV_EVENTS)
    all_rows = [r for r in all_rows if r.get("net_2h_bps") is not None]
    all_rows.sort(key=lambda r: int(r["signal_ts_ms"]))
    n_total = len(all_rows)
    n_cal   = int(n_total * (1.0 - HOLDOUT_FRAC))
    cal_raw = all_rows[:n_cal]
    hold_raw= all_rows[n_cal:]
    print(f"Total={n_total}  Cal={len(cal_raw)}  Hold={len(hold_raw)}")

    print("Loading liquidation data from DB...")
    with sqlite3.connect(f"file:{DEFAULT_DB}?mode=ro", uri=True) as conn:
        eth_sell_ts, eth_sell_not = load_liq(conn, "ETHUSDT", "SELL")
        btc_sell_ts, btc_sell_not = load_liq(conn, "BTCUSDT", "SELL")
        sol_sell_ts, sol_sell_not = load_liq(conn, "SOLUSDT", "SELL")

    print("Annotating rows with silence / sync_k / flags...")
    cal  = annotate(cal_raw,  eth_sell_ts, eth_sell_not,
                    btc_sell_ts, btc_sell_not, sol_sell_ts, sol_sell_not)
    hold = annotate(hold_raw, eth_sell_ts, eth_sell_not,
                    btc_sell_ts, btc_sell_not, sol_sell_ts, sol_sell_not)

    print("Test A: permutation null (silence gate, cal + hold)...")
    test_a = [test_a_permutation(cal, "cal"), test_a_permutation(hold, "hold")]

    print("Test B: silence + BULL_PULLBACK combo...")
    test_b = test_b_silence_bull(cal, hold)

    print("Test C: sync_k as predictor of silence...")
    test_c = test_c_sync_predicts_silence(cal, hold)

    print("Test D: live rule (200K) silence analysis...")
    test_d = test_d_live_rule(cal, hold)

    print("Test E: silence by sync_k level...")
    test_e = test_e_silence_by_sync(cal, hold)

    print("Test F: prior4h + silence combo...")
    test_f = test_f_prior4h_silence(cal, hold)

    split_info = {
        "cal_n":      len(cal),
        "hold_n":     len(hold),
        "cal_start":  ts_utc(cal_raw[0]["signal_ts_ms"]),
        "cal_end":    ts_utc(cal_raw[-1]["signal_ts_ms"]),
        "hold_start": ts_utc(hold_raw[0]["signal_ts_ms"]),
        "hold_end":   ts_utc(hold_raw[-1]["signal_ts_ms"]),
    }

    result = {
        "generated_at_utc": utc_now(),
        "status": "RESEARCH_ONLY_NO_LIVE_CHANGE",
        "split": split_info,
        "test_a": test_a,
        "test_b": test_b,
        "test_c": test_c,
        "test_d": test_d,
        "test_e": test_e,
        "test_f": test_f,
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(result, indent=2, ensure_ascii=True), encoding="utf-8")
    md = render_md(result)
    OUT_MD.write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
