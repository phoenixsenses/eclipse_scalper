"""S34 Puzzle Full Suite — 6 tests post-regime-probe.

Status: RESEARCH_ONLY_NO_LIVE_CHANGE

Tests:
  1. Silence gate holdout calibration — does SELL silence in next 30min
     still fade well in Jun holdout, or has even silence broken?
  2. sync_k threshold scan (50K-500K) — find the continuous gating level
     where holdout T3R flips sign.
  3. BULL_PULLBACK permutation null + anatomy — is N=22, WR 90.9% real?
     What features define it?
  4. prior4h_bps trend gate holdout — does a simple trend filter
     (prior4h > X) survive in holdout?
  5. KNN + sync_k augmented feature vector — retraining KNN with sync_k
     as 8th feature: does holdout T3R improve?
  6. Weekly breakdown — which exact week did the structure break?

SAF-02 / DAT-01 / DAT-03: research-only, no lookahead,
holdout labels from cal neighbors only, seed=42.
"""

from __future__ import annotations

import bisect
import json
import math
import random
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from statistics import median
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.s34_navigation_full_followup import (
    classify_neighbor,
    feature_vector,
    load_jsonl,
    summary,
    r1,
    r3,
    NAV_EVENTS,
    FEE_BPS,
)

DEFAULT_DB = ROOT / "data" / "microstructure.db"
OUT_JSON = ROOT / "reports" / "research" / "s34" / "S34_PUZZLE_FULL_SUITE.json"
OUT_MD  = ROOT / "reports" / "research" / "s34" / "S34_PUZZLE_FULL_SUITE.md"

HOLDOUT_FRAC    = 0.30
KS_BASE         = (5, 8, 10, 20)
KS_AUG          = (5, 8)          # augmented KNN uses fewer k to save time
SEED            = 42
MIN_N           = 15
SYNC_WINDOW_MS  = 10 * 60 * 1000  # 10-min prior window for sync_k
SILENCE_LO_MS   = 60 * 1000       # skip first 60 s to avoid same bucket
SILENCE_HI_MS   = 30 * 60 * 1000  # 30 min lookahead for silence
PROP_THRESHOLD  = 50_000.0        # cascade threshold for propagation check
SYNC_THRESHOLDS = (0, 50_000, 100_000, 150_000, 200_000, 300_000, 500_000)
PRIOR4H_CUTS    = (-100, -50, 0, 25, 50, 100)
N_PERM          = 1000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()

def ts_utc(ts: int) -> str:
    return datetime.fromtimestamp(int(ts)/1000, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def week_of(ts: int) -> str:
    dt = datetime.fromtimestamp(int(ts)/1000, tz=timezone.utc)
    mon = dt - timedelta(days=dt.weekday())
    return mon.strftime("%Y-%m-%d")

def month_of(ts: int) -> str:
    return datetime.fromtimestamp(int(ts)/1000, tz=timezone.utc).strftime("%Y-%m")

def dist(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((x-y)**2 for x, y in zip(a, b)))

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
    idx = p * (len(v)-1)
    lo = int(idx)
    return v[lo] + (idx-lo)*(v[min(lo+1, len(v)-1)]-v[lo])


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def load_liq(conn: sqlite3.Connection, symbol: str, side: str) -> tuple[list[int], list[float]]:
    rows = conn.execute(
        "SELECT ts_ms, notional FROM liquidations WHERE symbol=? AND side=? ORDER BY ts_ms",
        (symbol, side),
    ).fetchall()
    return [int(r[0]) for r in rows], [float(r[1]) for r in rows]

def window_sum(ts_list: list[int], vals: list[float], lo: int, hi: int) -> float:
    a = bisect.bisect_left(ts_list, lo)
    b = bisect.bisect_right(ts_list, hi)
    return sum(vals[i] for i in range(a, b))

def window_count(ts_list: list[int], lo: int, hi: int, threshold: float,
                 vals: list[float]) -> int:
    a = bisect.bisect_left(ts_list, lo)
    b = bisect.bisect_right(ts_list, hi)
    return sum(1 for i in range(a, b) if vals[i] >= threshold)


# ---------------------------------------------------------------------------
# KNN label builder
# ---------------------------------------------------------------------------

def knn_label(qv: list[float], ref_rows: list[dict], ref_vecs: list[list[float]],
              k: int) -> str:
    dists = [(dist(qv, ref_vecs[j]), ref_rows[j]) for j in range(len(ref_rows))]
    nn = [r for _, r in sorted(dists, key=lambda x: x[0])[:k]]
    vals = [float(r["net_2h_bps"]) for r in nn if r.get("net_2h_bps") is not None]
    s = summary(vals)
    return "UNKNOWN" if s["n"] < k else classify_neighbor(s, "base")

def build_labels(rows: list[dict], ref_rows: list[dict], ref_vecs: list[list[float]],
                 ks: tuple[int, ...], fvec_fn=None) -> list[dict]:
    if fvec_fn is None:
        fvec_fn = feature_vector
    qvecs = [fvec_fn(r) for r in rows]
    out = []
    for row, qv in zip(rows, qvecs):
        rid = row.get("event_id") or row.get("signal_ts_ms")
        excl = [(rv, rr) for rv, rr in zip(ref_vecs, ref_rows)
                if (rr.get("event_id") or rr.get("signal_ts_ms")) != rid]
        if not excl:
            out.append({**row, "labels": {f"k{k}": "UNKNOWN" for k in ks}})
            continue
        evecs, erows = zip(*excl)
        item = dict(row)
        item["labels"] = {f"k{k}": knn_label(qv, list(erows), list(evecs), k) for k in ks}
        out.append(item)
    return out


# ---------------------------------------------------------------------------
# Test 1 — Silence gate holdout calibration
# ---------------------------------------------------------------------------

def test1_silence_gate(eth_sell_ts: list[int], eth_sell_not: list[float],
                       cal_rows: list[dict], hold_rows: list[dict]) -> dict:
    def classify(row: dict) -> str:
        ts = int(row["signal_ts_ms"])
        cnt = window_count(eth_sell_ts, ts+SILENCE_LO_MS, ts+SILENCE_HI_MS,
                           PROP_THRESHOLD, eth_sell_not)
        return "SILENCE" if cnt == 0 else "NOISY"

    def seg(rows: list[dict], state: str) -> list[float]:
        return [float(r["net_2h_bps"]) for r in rows
                if classify(r) == state and r.get("net_2h_bps") is not None]

    result = {}
    for split, rows in [("cal", cal_rows), ("hold", hold_rows)]:
        silence_v = seg(rows, "SILENCE")
        noisy_v   = seg(rows, "NOISY")
        n_sil = len([r for r in rows if classify(r) == "SILENCE"])
        result[split] = {
            "silence_n":    n_sil,
            "silence_rate": r3(n_sil / len(rows)) if rows else None,
            "silence":      qs(silence_v),
            "noisy":        qs(noisy_v),
        }
    return result


# ---------------------------------------------------------------------------
# Test 2 — sync_k threshold scan
# ---------------------------------------------------------------------------

def test2_sync_threshold(btc_sell_ts: list[int], btc_sell_not: list[float],
                         sol_sell_ts: list[int], sol_sell_not: list[float],
                         cal_rows: list[dict], hold_rows: list[dict]) -> dict:
    def sync_k(row: dict) -> float:
        ts = int(row["signal_ts_ms"])
        b = window_sum(btc_sell_ts, btc_sell_not, ts-SYNC_WINDOW_MS, ts)
        s = window_sum(sol_sell_ts, sol_sell_not, ts-SYNC_WINDOW_MS, ts)
        return b + s

    # Pre-compute sync_k for all rows
    for r in cal_rows + hold_rows:
        r["_sync_k"] = sync_k(r)

    result = {}
    for thr in SYNC_THRESHOLDS:
        label = f"sync_lt_{int(thr/1000)}K" if thr > 0 else "all"
        filter_fn = (lambda r, t=thr: r["_sync_k"] < t) if thr > 0 else (lambda r: True)
        cal_v  = [float(r["net_2h_bps"]) for r in cal_rows  if filter_fn(r) and r.get("net_2h_bps") is not None]
        hold_v = [float(r["net_2h_bps"]) for r in hold_rows if filter_fn(r) and r.get("net_2h_bps") is not None]
        result[label] = {
            "threshold_K": thr/1000,
            "cal":  qs(cal_v),
            "hold": qs(hold_v),
        }
    # Also annotate rows with sync_k bucket for later tests
    return result


# ---------------------------------------------------------------------------
# Test 3 — BULL_PULLBACK permutation null + anatomy
# ---------------------------------------------------------------------------

def test3_bull_pullback(cal_rows: list[dict], hold_rows: list[dict]) -> dict:
    def is_bp(r: dict) -> bool:
        return "BULL_PULLBACK" in (r.get("tags") or [])

    cal_bp  = [r for r in cal_rows  if is_bp(r)]
    hold_bp = [r for r in hold_rows if is_bp(r)]
    cal_non = [r for r in cal_rows  if not is_bp(r)]
    hold_non= [r for r in hold_rows if not is_bp(r)]

    cal_bp_v  = [float(r["net_2h_bps"]) for r in cal_bp  if r.get("net_2h_bps") is not None]
    hold_bp_v = [float(r["net_2h_bps"]) for r in hold_bp if r.get("net_2h_bps") is not None]
    cal_non_v = [float(r["net_2h_bps"]) for r in cal_non if r.get("net_2h_bps") is not None]

    # Anatomy — feature medians
    def feat_med(rows: list[dict], key: str) -> float | None:
        vals = [float(r[key]) for r in rows if r.get(key) is not None]
        return r1(median(vals)) if vals else None

    anatomy_keys = ["prior4h_bps", "vdepth_bps", "bid_depth_usd", "book_imbalance",
                    "eth1h_bps", "btc4h_bps", "threshold_usd"]
    anatomy = {
        "bull_pullback": {k: feat_med(cal_bp,  k) for k in anatomy_keys},
        "non_bull":      {k: feat_med(cal_non, k) for k in anatomy_keys},
    }

    # Add sync_k anatomy if precomputed
    if cal_bp and "_sync_k" in cal_bp[0]:
        anatomy["bull_pullback"]["sync_k"] = r1(median([r["_sync_k"] for r in cal_bp  if "_sync_k" in r])/1000)
        anatomy["non_bull"]["sync_k"]      = r1(median([r["_sync_k"] for r in cal_non if "_sync_k" in r])/1000)

    # Permutation null on cal BULL_PULLBACK (single test, no MC needed)
    rng = random.Random(SEED)
    all_cal_v = [float(r["net_2h_bps"]) for r in cal_rows if r.get("net_2h_bps") is not None]
    real_t3r  = t3r(cal_bp_v) if len(cal_bp_v) >= MIN_N else float("nan")
    null_t3rs = []
    n_bp = len(cal_bp_v)
    for _ in range(N_PERM):
        shuf = rng.sample(all_cal_v, min(n_bp, len(all_cal_v)))
        null_t3rs.append(t3r(shuf))
    null_p95  = pctile(null_t3rs, 0.95)
    p_right   = sum(1 for v in null_t3rs if math.isfinite(v) and v >= real_t3r)/len(null_t3rs)

    # prior4h gate inside BULL_PULLBACK hold
    prior4h_gate = {}
    for cut in PRIOR4H_CUTS:
        sub_h = [float(r["net_2h_bps"]) for r in hold_bp
                 if r.get("net_2h_bps") is not None and (r.get("prior4h_bps") or 0) > cut]
        prior4h_gate[f"prior4h_gt_{cut}"] = qs(sub_h)

    return {
        "cal_bp":     qs(cal_bp_v),
        "hold_bp":    qs(hold_bp_v),
        "cal_non_bp": qs(cal_non_v),
        "anatomy":    anatomy,
        "permutation": {
            "real_t3r": r1(real_t3r),
            "null_p95": r1(null_p95),
            "p_right":  r3(p_right),
            "verdict":  "PASS" if p_right < 0.05 else "ARTIFACT",
        },
        "prior4h_gate_in_hold_bp": prior4h_gate,
    }


# ---------------------------------------------------------------------------
# Test 4 — prior4h_bps trend gate holdout
# ---------------------------------------------------------------------------

def test4_prior4h_gate(cal_rows: list[dict], hold_rows: list[dict]) -> dict:
    result = {}
    for cut in PRIOR4H_CUTS:
        label = f"prior4h_gt_{cut}"
        cal_v  = [float(r["net_2h_bps"]) for r in cal_rows
                  if (r.get("prior4h_bps") or 0) > cut and r.get("net_2h_bps") is not None]
        hold_v = [float(r["net_2h_bps"]) for r in hold_rows
                  if (r.get("prior4h_bps") or 0) > cut and r.get("net_2h_bps") is not None]
        result[label] = {"cal": qs(cal_v), "hold": qs(hold_v)}

    # Combo: prior4h > 0 AND sync_k < 200K (if precomputed)
    if cal_rows and "_sync_k" in cal_rows[0]:
        for (p_cut, s_cut) in [(0, 200_000), (0, 100_000), (25, 200_000)]:
            lbl = f"prior4h_gt_{p_cut}_AND_sync_lt_{int(s_cut/1000)}K"
            fn = lambda r, pc=p_cut, sc=s_cut: (r.get("prior4h_bps") or 0) > pc and r.get("_sync_k", 1e9) < sc
            cal_v  = [float(r["net_2h_bps"]) for r in cal_rows  if fn(r) and r.get("net_2h_bps") is not None]
            hold_v = [float(r["net_2h_bps"]) for r in hold_rows if fn(r) and r.get("net_2h_bps") is not None]
            result[lbl] = {"cal": qs(cal_v), "hold": qs(hold_v)}

    return result


# ---------------------------------------------------------------------------
# Test 5 — KNN with augmented feature vector (+ sync_k)
# ---------------------------------------------------------------------------

def augmented_feature(row: dict) -> list[float]:
    base = feature_vector(row)
    sync = float(row.get("_sync_k", 0.0)) / 500_000.0
    return base + [sync]

def test5_knn_augmented(cal_rows_raw: list[dict], hold_rows_raw: list[dict]) -> dict:
    print("  Test 5: building augmented cal vectors...")
    aug_cal_vecs = [augmented_feature(r) for r in cal_rows_raw]

    print("  Test 5: cal KNN (augmented, leave-one-out)...")
    cal_aug = build_labels(cal_rows_raw, cal_rows_raw, aug_cal_vecs, KS_AUG, augmented_feature)

    print("  Test 5: hold KNN (augmented, cal-pool)...")
    hold_aug = build_labels(hold_rows_raw, cal_rows_raw, aug_cal_vecs, KS_AUG, augmented_feature)

    result = {}
    for k in KS_AUG:
        for label, direction in [("CLEAN", "NORMAL"), ("DANGER", "REVERSE")]:
            name = f"k{k}_{label}_{direction}"
            def flt(r, k=k, label=label): return r.get("labels", {}).get(f"k{k}") == label
            if direction == "NORMAL":
                cal_v  = [float(r["net_2h_bps"]) for r in cal_aug  if flt(r) and r.get("net_2h_bps") is not None]
                hold_v = [float(r["net_2h_bps"]) for r in hold_aug if flt(r) and r.get("net_2h_bps") is not None]
            else:
                cal_v  = [-float(r["net_2h_bps"])-2*FEE_BPS for r in cal_aug  if flt(r) and r.get("net_2h_bps") is not None]
                hold_v = [-float(r["net_2h_bps"])-2*FEE_BPS for r in hold_aug if flt(r) and r.get("net_2h_bps") is not None]
            result[name] = {"cal": qs(cal_v), "hold": qs(hold_v)}

    return result


# ---------------------------------------------------------------------------
# Test 6 — Weekly breakdown
# ---------------------------------------------------------------------------

def test6_weekly(cal_rows: list[dict], hold_rows: list[dict]) -> dict:
    all_rows = cal_rows + hold_rows
    by_week: dict[str, list[dict]] = defaultdict(list)
    for r in all_rows:
        by_week[week_of(int(r["signal_ts_ms"]))].append(r)

    result = {}
    for wk, rows in sorted(by_week.items()):
        is_hold = any(r in hold_rows for r in rows[:1])  # approximation
        k5c = [r for r in rows if r.get("labels", {}).get("k5") == "CLEAN"]
        all_v  = [float(r["net_2h_bps"]) for r in rows  if r.get("net_2h_bps") is not None]
        k5c_v  = [float(r["net_2h_bps"]) for r in k5c  if r.get("net_2h_bps") is not None]
        sync_k_vals = [r["_sync_k"]/1000 for r in rows if "_sync_k" in r]
        result[wk] = {
            "holdout": is_hold,
            "n_all": len(rows),
            "all_med": r1(median(all_v)) if all_v else None,
            "all_win": r3(sum(1 for v in all_v if v > 0)/len(all_v)) if all_v else None,
            "k5c_n": len(k5c_v),
            "k5c_med": r1(median(k5c_v)) if k5c_v else None,
            "k5c_win": r3(sum(1 for v in k5c_v if v > 0)/len(k5c_v)) if k5c_v else None,
            "mean_sync_k": r1(sum(sync_k_vals)/len(sync_k_vals)) if sync_k_vals else None,
        }
    return result


# ---------------------------------------------------------------------------
# Markdown render
# ---------------------------------------------------------------------------

def render_md(res: dict) -> str:
    sp = res["split"]
    lines = [
        "# S34 Puzzle Full Suite — 6 Tests",
        "",
        f"Generated: `{res['generated_at_utc']}`  |  Status: `RESEARCH_ONLY_NO_LIVE_CHANGE`",
        "",
        f"Cal: {sp['cal_n']} events ({sp['cal_start']} to {sp['cal_end']})",
        f"Hold: {sp['hold_n']} events ({sp['hold_start']} to {sp['hold_end']})",
        "",
    ]

    # Test 1
    lines += ["## Test 1: Silence Gate Holdout", ""]
    t1 = res["test1_silence"]
    lines += [
        "| Split | Silence N | Silence rate | Silence T3R | Silence med | Silence win | Noisy T3R | Noisy med | Noisy win |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for sp_label in ("cal", "hold"):
        d = t1[sp_label]
        s = d["silence"]; no = d["noisy"]
        lines.append(
            f"| {sp_label} | {d['silence_n']} | {d['silence_rate']} |"
            f" {s.get('t3r')} | {s['med']} | {s['win']} |"
            f" {no.get('t3r')} | {no['med']} | {no['win']} |"
        )
    lines.append("")

    # Test 2
    lines += ["## Test 2: sync_k Threshold Scan", ""]
    lines += [
        "| Gate | Cal N | Cal T3R | Cal med | Cal win | Hold N | Hold T3R | Hold med | Hold win |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    t2 = res["test2_sync"]
    for label, d in t2.items():
        c = d["cal"]; h = d["hold"]
        lines.append(
            f"| {label} | {c['n']} | {c.get('t3r')} | {c['med']} | {c['win']} |"
            f" {h['n']} | {h.get('t3r')} | {h['med']} | {h['win']} |"
        )
    lines.append("")

    # Test 3
    lines += ["## Test 3: BULL_PULLBACK Permutation Null + Anatomy", ""]
    t3 = res["test3_bull_pullback"]
    pm = t3["permutation"]
    lines += [
        f"- Cal BULL_PULLBACK: N={t3['cal_bp']['n']}, T3R={t3['cal_bp'].get('t3r')}, med={t3['cal_bp']['med']}, win={t3['cal_bp']['win']}",
        f"- Hold BULL_PULLBACK: N={t3['hold_bp']['n']}, T3R={t3['hold_bp'].get('t3r')}, med={t3['hold_bp']['med']}, win={t3['hold_bp']['win']}",
        f"- Cal NON-BULL_PULLBACK: N={t3['cal_non_bp']['n']}, T3R={t3['cal_non_bp'].get('t3r')}, med={t3['cal_non_bp']['med']}",
        "",
        f"**Permutation null (cal, {N_PERM} shuffles):** real T3R={pm['real_t3r']}, null p95={pm['null_p95']}, p-right={pm['p_right']} -> **{pm['verdict']}**",
        "",
        "### Anatomy — cal feature medians",
        "",
        "| Feature | BULL_PULLBACK | NON-BULL |",
        "| --- | ---: | ---: |",
    ]
    for k in ["prior4h_bps", "vdepth_bps", "bid_depth_usd", "book_imbalance",
              "eth1h_bps", "btc4h_bps", "threshold_usd", "sync_k"]:
        bp_v  = t3["anatomy"]["bull_pullback"].get(k)
        non_v = t3["anatomy"]["non_bull"].get(k)
        lines.append(f"| {k} | {bp_v} | {non_v} |")
    lines += [
        "",
        "### prior4h gate within hold BULL_PULLBACK subset",
        "",
        "| Gate | Hold N | Hold T3R | Hold med | Hold win |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for gate, d in t3["prior4h_gate_in_hold_bp"].items():
        lines.append(f"| {gate} | {d['n']} | {d.get('t3r')} | {d['med']} | {d['win']} |")
    lines.append("")

    # Test 4
    lines += ["## Test 4: prior4h_bps Trend Gate Holdout", ""]
    lines += [
        "| Gate | Cal N | Cal T3R | Cal med | Cal win | Hold N | Hold T3R | Hold med | Hold win |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    t4 = res["test4_prior4h"]
    for label, d in t4.items():
        c = d["cal"]; h = d["hold"]
        lines.append(
            f"| {label} | {c['n']} | {c.get('t3r')} | {c['med']} | {c['win']} |"
            f" {h['n']} | {h.get('t3r')} | {h['med']} | {h['win']} |"
        )
    lines.append("")

    # Test 5
    lines += ["## Test 5: KNN Augmented (+ sync_k feature)", ""]
    lines += [
        "| Pattern | Cal N | Cal T3R | Cal med | Hold N | Hold T3R | Hold med | Hold win |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    t5 = res["test5_knn_aug"]
    for name, d in t5.items():
        c = d["cal"]; h = d["hold"]
        lines.append(
            f"| {name} | {c['n']} | {c.get('t3r')} | {c['med']} |"
            f" {h['n']} | {h.get('t3r')} | {h['med']} | {h['win']} |"
        )
    lines.append("")

    # Test 6
    lines += ["## Test 6: Weekly Breakdown", ""]
    lines += [
        "(*) = holdout",
        "",
        "| Week | Hold? | All N | All med | All win | k5=CLEAN N | CLEAN med | CLEAN win | Mean sync_k (K) |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    t6 = res["test6_weekly"]
    for wk, d in sorted(t6.items()):
        star = "Y" if d["holdout"] else ""
        lines.append(
            f"| {wk} | {star} | {d['n_all']} | {d['all_med']} | {d['all_win']} |"
            f" {d['k5c_n']} | {d['k5c_med']} | {d['k5c_win']} | {d['mean_sync_k']} |"
        )
    lines.append("")

    lines += [
        "## Summary Verdict",
        "",
        "- Test 1: silence gate — does no-propagation still predict fade in hold?",
        "- Test 2: at what sync_k threshold does holdout T3R flip positive?",
        "- Test 3: BULL_PULLBACK — permutation result + anatomy reveals the knowable gate",
        "- Test 4: prior4h trend filter + sync_k combo — simplest possible gate",
        "- Test 5: augmented KNN — does adding sync_k as feature rescue holdout?",
        "- Test 6: exact week of regime break visible in sync_k spike",
        "",
        "RESEARCH_ONLY. No live/paper promotion without OOS permutation null.",
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

    print("Opening DB...")
    with sqlite3.connect(f"file:{DEFAULT_DB}?mode=ro", uri=True) as conn:
        eth_sell_ts, eth_sell_not = load_liq(conn, "ETHUSDT", "SELL")
        btc_sell_ts, btc_sell_not = load_liq(conn, "BTCUSDT", "SELL")
        sol_sell_ts, sol_sell_not = load_liq(conn, "SOLUSDT", "SELL")

    print("Pre-computing sync_k for all rows (needed for Tests 2,4,5,6)...")
    for r in all_rows:
        ts = int(r["signal_ts_ms"])
        b = window_sum(btc_sell_ts, btc_sell_not, ts-SYNC_WINDOW_MS, ts)
        s = window_sum(sol_sell_ts, sol_sell_not, ts-SYNC_WINDOW_MS, ts)
        r["_sync_k"] = b + s

    print("Building base cal KNN labels (leave-one-out)...")
    cal_vecs = [feature_vector(r) for r in cal_raw]
    cal_rows = build_labels(cal_raw, cal_raw, cal_vecs, KS_BASE)

    print("Building base hold KNN labels (cal-pool)...")
    hold_rows = build_labels(hold_raw, cal_raw, cal_vecs, KS_BASE)

    # Propagate _sync_k to labeled rows
    for labeled, raw_list in [(cal_rows, cal_raw), (hold_rows, hold_raw)]:
        for lr, rr in zip(labeled, raw_list):
            lr["_sync_k"] = rr["_sync_k"]

    print("Test 1: silence gate...")
    t1 = test1_silence_gate(eth_sell_ts, eth_sell_not, cal_rows, hold_rows)

    print("Test 2: sync_k threshold scan...")
    t2 = test2_sync_threshold(btc_sell_ts, btc_sell_not, sol_sell_ts, sol_sell_not,
                               cal_rows, hold_rows)

    print("Test 3: BULL_PULLBACK permutation null + anatomy...")
    t3 = test3_bull_pullback(cal_rows, hold_rows)

    print("Test 4: prior4h_bps trend gate...")
    t4 = test4_prior4h_gate(cal_rows, hold_rows)

    print("Test 5: KNN augmented (+ sync_k feature)...")
    t5 = test5_knn_augmented(cal_raw, hold_raw)

    print("Test 6: weekly breakdown...")
    t6 = test6_weekly(cal_rows, hold_rows)

    split_info = {
        "cal_n":     len(cal_rows),
        "hold_n":    len(hold_rows),
        "cal_start": ts_utc(cal_raw[0]["signal_ts_ms"]),
        "cal_end":   ts_utc(cal_raw[-1]["signal_ts_ms"]),
        "hold_start":ts_utc(hold_raw[0]["signal_ts_ms"]),
        "hold_end":  ts_utc(hold_raw[-1]["signal_ts_ms"]),
    }

    result = {
        "generated_at_utc": utc_now(),
        "status": "RESEARCH_ONLY_NO_LIVE_CHANGE",
        "split": split_info,
        "test1_silence":       t1,
        "test2_sync":          t2,
        "test3_bull_pullback": t3,
        "test4_prior4h":       t4,
        "test5_knn_aug":       t5,
        "test6_weekly":        t6,
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(result, indent=2, ensure_ascii=True), encoding="utf-8")
    md = render_md(result)
    OUT_MD.write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
