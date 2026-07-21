"""S34 Holdout Regime Probe — 4-test suite.

Status: RESEARCH_ONLY_NO_LIVE_CHANGE

Tests:
  1. Propagation-rate regime probe: did Jun 8-29 (holdout) have more
     same-side cascade propagation than cal? If yes, that explains why
     k5=CLEAN labels (built from cal neighbors) break in holdout.

  2. SYNC + k5=CLEAN composite: SYNCHRONIZED (BTC+SOL concurrent SELL >=200K
     in prior 10 min) AND k5=CLEAN — does this composite gate survive in holdout?

  3. Frequency expansion holdout calibration: compare event_end / reclaim
     timing vs raw anchor entry across cal and hold periods. Which timing
     survives in Jun 8-29?

  4. Monthly k5=CLEAN stability: show T3R and WR for k5=CLEAN events broken
     out by month, revealing when the structure broke.

SAF-02: no live change.
DAT-01: no lookahead (holdout KNN labels use cal neighbors only).
DAT-03: seeded permutation, seed=42.
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
OUT_JSON = ROOT / "reports" / "research" / "s34" / "S34_HOLDOUT_REGIME_PROBE.json"
OUT_MD = ROOT / "reports" / "research" / "s34" / "S34_HOLDOUT_REGIME_PROBE.md"

HOLDOUT_FRAC = 0.30
KS = (5, 8, 10, 20)
SEED = 42
MIN_N = 20
SYNC_THRESHOLD = 200_000.0
SYNC_WINDOW_MS = 10 * 60 * 1000
PROP_WINDOW_MS = 60 * 60 * 1000  # 60 min lookahead for propagation
PROP_THRESHOLD = 50_000.0


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ts_to_utc(ts: int) -> str:
    return datetime.fromtimestamp(int(ts) / 1000, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def month_of(ts: int) -> str:
    return datetime.fromtimestamp(int(ts) / 1000, tz=timezone.utc).strftime("%Y-%m")


def distance(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def knn_label(query_vec: list[float], ref_rows: list[dict], ref_vecs: list[list[float]],
              k: int, strictness: str = "base") -> str:
    dists = [(distance(query_vec, ref_vecs[j]), ref_rows[j]) for j in range(len(ref_rows))]
    nn = [r for _, r in sorted(dists, key=lambda x: x[0])[:k]]
    vals = [float(r["net_2h_bps"]) for r in nn if r.get("net_2h_bps") is not None]
    s = summary(vals)
    return "UNKNOWN" if s["n"] < k else classify_neighbor(s, strictness)


def build_labels(rows: list[dict], ref_rows: list[dict], ref_vecs: list[list[float]],
                 ks: tuple[int, ...]) -> list[dict]:
    query_vecs = [feature_vector(r) for r in rows]
    out = []
    for row, qv in zip(rows, query_vecs):
        row_id = row.get("event_id") or row.get("signal_ts_ms")
        ref_excl = [(rv, rr) for rv, rr in zip(ref_vecs, ref_rows)
                    if (rr.get("event_id") or rr.get("signal_ts_ms")) != row_id]
        if not ref_excl:
            out.append({**row, "labels": {f"k{k}": "UNKNOWN" for k in ks}})
            continue
        excl_vecs, excl_rows = zip(*ref_excl)
        item = dict(row)
        item["labels"] = {f"k{k}": knn_label(qv, list(excl_rows), list(excl_vecs), k) for k in ks}
        out.append(item)
    return out


def t3r(vals: list[float]) -> float:
    if len(vals) <= 3:
        return sum(vals)
    return sum(sorted(vals, reverse=True)[3:])


def quick_stats(vals: list[float]) -> dict:
    if not vals:
        return {"n": 0, "t3r": None, "sum": None, "median": None, "win": None, "max_loss": None}
    return {
        "n": len(vals),
        "t3r": r1(t3r(vals)) if len(vals) >= MIN_N else None,
        "sum": r1(sum(vals)),
        "median": r1(median(vals)),
        "win": r3(sum(1 for v in vals if v > 0) / len(vals)),
        "max_loss": r1(min(vals)),
    }


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def load_liq_index(conn: sqlite3.Connection, symbol: str, side: str) -> tuple[list[int], list[float]]:
    rows = conn.execute(
        "SELECT ts_ms, notional FROM liquidations WHERE symbol=? AND side=? ORDER BY ts_ms ASC",
        (symbol, side),
    ).fetchall()
    ts = [int(r[0]) for r in rows]
    notional = [float(r[1]) for r in rows]
    return ts, notional


def notional_in_window(ts_list: list[int], notional: list[float], lo: int, hi: int) -> float:
    a = bisect.bisect_left(ts_list, lo)
    b = bisect.bisect_right(ts_list, hi)
    return sum(notional[i] for i in range(a, b))


def load_mark_prices(conn: sqlite3.Connection, symbol: str) -> tuple[list[int], list[float]]:
    rows = conn.execute(
        "SELECT ts_ms, mark_price FROM mark_prices WHERE symbol=? ORDER BY ts_ms ASC",
        (symbol,),
    ).fetchall()
    ts = [int(r[0]) for r in rows]
    prices = [float(r[1]) for r in rows]
    return ts, prices


def mark_at_or_after(ts_list: list[int], prices: list[float], ts: int) -> float | None:
    idx = bisect.bisect_left(ts_list, ts)
    if idx >= len(ts_list):
        return None
    return prices[idx]


# ---------------------------------------------------------------------------
# Test 1: Propagation-rate regime probe
# ---------------------------------------------------------------------------

def test1_propagation_rate(conn: sqlite3.Connection, cal_rows: list[dict], hold_rows: list[dict]) -> dict:
    """Check same-side SELL cascade propagation rate in cal vs hold period."""
    eth_sell_ts, eth_sell_not = load_liq_index(conn, "ETHUSDT", "SELL")
    btc_sell_ts, btc_sell_not = load_liq_index(conn, "BTCUSDT", "SELL")
    sol_sell_ts, sol_sell_not = load_liq_index(conn, "SOLUSDT", "SELL")

    def prop_check(row: dict) -> dict:
        ts = int(row["signal_ts_ms"])
        # Exclude the event's own liq window (within 60s before/after signal)
        lo = ts + 60_000
        hi = ts + PROP_WINDOW_MS
        eth_next = notional_in_window(eth_sell_ts, eth_sell_not, lo, hi)
        btc_next = notional_in_window(btc_sell_ts, btc_sell_not, lo, hi)
        sol_next = notional_in_window(sol_sell_ts, sol_sell_not, lo, hi)
        # Sync in prior window
        btc_prior = notional_in_window(btc_sell_ts, btc_sell_not, ts - SYNC_WINDOW_MS, ts)
        sol_prior = notional_in_window(sol_sell_ts, sol_sell_not, ts - SYNC_WINDOW_MS, ts)
        sync_k = btc_prior + sol_prior
        return {
            "eth_next_prop": eth_next >= PROP_THRESHOLD,
            "any_next_prop": (eth_next + btc_next + sol_next) >= PROP_THRESHOLD,
            "cross_asset_next": (btc_next + sol_next) >= PROP_THRESHOLD,
            "is_sync": sync_k >= SYNC_THRESHOLD,
            "sync_k": sync_k,
        }

    def aggregate(rows: list[dict]) -> dict:
        props = [prop_check(r) for r in rows]
        n = len(props)
        if n == 0:
            return {}
        return {
            "n": n,
            "eth_prop_rate": r3(sum(1 for p in props if p["eth_next_prop"]) / n),
            "any_prop_rate": r3(sum(1 for p in props if p["any_next_prop"]) / n),
            "cross_asset_prop_rate": r3(sum(1 for p in props if p["cross_asset_next"]) / n),
            "sync_rate": r3(sum(1 for p in props if p["is_sync"]) / n),
            "mean_sync_k": r1(sum(p["sync_k"] for p in props) / n / 1000),
        }

    return {
        "cal": aggregate(cal_rows),
        "hold": aggregate(hold_rows),
        "interpretation": (
            "Higher prop_rate in hold -> more same-side follow-through -> "
            "k5=CLEAN (built on cal history) under-represents danger in hold"
        ),
    }


# ---------------------------------------------------------------------------
# Test 2: SYNC + k5=CLEAN composite holdout
# ---------------------------------------------------------------------------

def test2_sync_clean_composite(conn: sqlite3.Connection, cal_rows: list[dict],
                                hold_rows: list[dict]) -> dict:
    """SYNCHRONIZED AND k5=CLEAN: does composite gate survive holdout?"""
    btc_sell_ts, btc_sell_not = load_liq_index(conn, "BTCUSDT", "SELL")
    sol_sell_ts, sol_sell_not = load_liq_index(conn, "SOLUSDT", "SELL")

    def is_sync(row: dict) -> bool:
        ts = int(row["signal_ts_ms"])
        btc = notional_in_window(btc_sell_ts, btc_sell_not, ts - SYNC_WINDOW_MS, ts)
        sol = notional_in_window(sol_sell_ts, sol_sell_not, ts - SYNC_WINDOW_MS, ts)
        return (btc + sol) >= SYNC_THRESHOLD

    def seg(rows: list[dict], filter_fn) -> list[float]:
        sub = [r for r in rows if filter_fn(r) and r.get("net_2h_bps") is not None]
        return [float(r["net_2h_bps"]) for r in sub]

    def seg_n(rows: list[dict], filter_fn) -> int:
        return sum(1 for r in rows if filter_fn(r))

    is_k5clean = lambda r: r.get("labels", {}).get("k5") == "CLEAN"
    sync_clean = lambda r: is_sync(r) and is_k5clean(r)
    sync_danger = lambda r: is_sync(r) and r.get("labels", {}).get("k5") == "DANGER"
    idio_clean = lambda r: not is_sync(r) and is_k5clean(r)

    groups = {
        "sync_k5_clean": sync_clean,
        "sync_k5_danger_reverse": sync_danger,
        "idio_k5_clean": idio_clean,
        "k5_clean_all": is_k5clean,
        "sync_all": is_sync,
    }

    result = {}
    for name, fn in groups.items():
        if "reverse" in name:
            cal_vals = [-float(r["net_2h_bps"]) - 2 * FEE_BPS for r in cal_rows if fn(r) and r.get("net_2h_bps") is not None]
            hold_vals = [-float(r["net_2h_bps"]) - 2 * FEE_BPS for r in hold_rows if fn(r) and r.get("net_2h_bps") is not None]
        else:
            cal_vals = seg(cal_rows, fn)
            hold_vals = seg(hold_rows, fn)
        result[name] = {
            "cal": quick_stats(cal_vals),
            "hold": quick_stats(hold_vals),
        }

    return result


# ---------------------------------------------------------------------------
# Test 3: Frequency expansion holdout calibration
# ---------------------------------------------------------------------------

def test3_freq_expansion_holdout(conn: sqlite3.Connection, cal_rows: list[dict],
                                  hold_rows: list[dict]) -> dict:
    """Compare event_end / reclaim / raw timing across cal and hold.

    Uses NAV_EVENTS net_tp300_sl150_4h_bps for managed exit and net_2h_bps
    for raw exit. Also checks VDEPTH / tags to see if any tag combination
    survives in holdout.
    """
    # Timing comparison: net_2h vs net_4h vs managed (tp300_sl150)
    def timing_stats(rows: list[dict]) -> dict:
        v2h = [float(r["net_2h_bps"]) for r in rows if r.get("net_2h_bps") is not None]
        v4h = [float(r["net_4h_bps"]) for r in rows if r.get("net_4h_bps") is not None]
        vtp = [float(r["net_tp300_sl150_4h_bps"]) for r in rows if r.get("net_tp300_sl150_4h_bps") is not None]
        return {
            "n": len(rows),
            "raw_2h": quick_stats(v2h),
            "raw_4h": quick_stats(v4h),
            "tp300_sl150": quick_stats(vtp),
        }

    # Tag-based holdout check: which tags survive?
    all_tags: set[str] = set()
    for r in cal_rows + hold_rows:
        for t in (r.get("tags") or []):
            all_tags.add(str(t))

    SKIP = {"TAIL_REALIZED", "EXIT_2H_ACTUAL_BETTER", "EXIT_4H_ACTUAL_BETTER", "EXIT_4H_FAVORED"}
    tag_results = {}
    for tag in sorted(all_tags - SKIP):
        cal_sub = [r for r in cal_rows if tag in (r.get("tags") or [])]
        hold_sub = [r for r in hold_rows if tag in (r.get("tags") or [])]
        if len(cal_sub) < MIN_N and len(hold_sub) < MIN_N:
            continue
        cal_v = [float(r["net_2h_bps"]) for r in cal_sub if r.get("net_2h_bps") is not None]
        hold_v = [float(r["net_2h_bps"]) for r in hold_sub if r.get("net_2h_bps") is not None]
        c = quick_stats(cal_v)
        h = quick_stats(hold_v)
        # Only include if cal T3R > 0
        if c.get("t3r") and c["t3r"] > 0:
            tag_results[tag] = {"cal": c, "hold": h}

    # Sort by hold T3R descending
    sorted_tags = sorted(
        [(k, v) for k, v in tag_results.items() if v["hold"].get("t3r") is not None],
        key=lambda kv: float(kv[1]["hold"]["t3r"] or -1e9),
        reverse=True,
    )
    sorted_tags_all = sorted(
        [(k, v) for k, v in tag_results.items()],
        key=lambda kv: float(kv[1]["hold"].get("t3r") or float(kv[1]["hold"].get("sum") or -1e9)),
        reverse=True,
    )

    return {
        "cal_all": timing_stats(cal_rows),
        "hold_all": timing_stats(hold_rows),
        "tags_by_hold_t3r": {k: v for k, v in sorted_tags},
        "all_tag_results_count": len(tag_results),
        "top_hold_positive_tags": [k for k, v in sorted_tags if (v["hold"].get("t3r") or 0) > 0][:10],
        "top_tags_overall": [k for k, _ in sorted_tags_all[:10]],
    }


# ---------------------------------------------------------------------------
# Test 4: Monthly k5=CLEAN stability
# ---------------------------------------------------------------------------

def test4_monthly_stability(cal_rows: list[dict], hold_rows: list[dict]) -> dict:
    """Break k5=CLEAN events by month to reveal when the structure broke."""
    all_rows = cal_rows + hold_rows
    k5_clean = [r for r in all_rows if r.get("labels", {}).get("k5") == "CLEAN"]
    k5_danger = [r for r in all_rows if r.get("labels", {}).get("k5") == "DANGER"]
    k5_clean_rev_vals = {}  # k5=DANGER reverse by month
    k5_clean_vals = {}  # k5=CLEAN normal by month

    for r in k5_clean:
        m = month_of(int(r["signal_ts_ms"]))
        v = r.get("net_2h_bps")
        if v is not None:
            k5_clean_vals.setdefault(m, []).append(float(v))

    for r in k5_danger:
        m = month_of(int(r["signal_ts_ms"]))
        v = r.get("net_2h_bps")
        if v is not None:
            rev = -float(v) - 2 * FEE_BPS
            k5_clean_rev_vals.setdefault(m, []).append(rev)

    # Overall by month (all events)
    all_by_month: dict[str, list[float]] = {}
    for r in all_rows:
        m = month_of(int(r["signal_ts_ms"]))
        v = r.get("net_2h_bps")
        if v is not None:
            all_by_month.setdefault(m, []).append(float(v))

    months = sorted(set(k5_clean_vals.keys()) | set(all_by_month.keys()))
    result = {}
    for m in months:
        clean_v = k5_clean_vals.get(m, [])
        danger_rev_v = k5_clean_rev_vals.get(m, [])
        all_v = all_by_month.get(m, [])
        result[m] = {
            "all_n": len(all_v),
            "all_median": r1(median(all_v)) if all_v else None,
            "all_win": r3(sum(1 for v in all_v if v > 0) / len(all_v)) if all_v else None,
            "k5_clean_n": len(clean_v),
            "k5_clean_median": r1(median(clean_v)) if clean_v else None,
            "k5_clean_win": r3(sum(1 for v in clean_v if v > 0) / len(clean_v)) if clean_v else None,
            "k5_clean_t3r": r1(t3r(clean_v)) if len(clean_v) >= MIN_N else None,
            "k5_danger_rev_n": len(danger_rev_v),
            "k5_danger_rev_median": r1(median(danger_rev_v)) if danger_rev_v else None,
            "k5_danger_rev_win": r3(sum(1 for v in danger_rev_v if v > 0) / len(danger_rev_v)) if danger_rev_v else None,
            "in_holdout": m >= "2026-06",
        }
    return result


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------

def render_md(result: dict) -> str:
    lines = [
        "# S34 Holdout Regime Probe — 4-Test Suite",
        "",
        f"Generated: `{result['generated_at_utc']}`",
        "Status: `RESEARCH_ONLY_NO_LIVE_CHANGE`",
        "",
        f"Cal: {result['split']['cal_n']} events ({result['split']['cal_start']} to {result['split']['cal_end']})",
        f"Hold: {result['split']['hold_n']} events ({result['split']['hold_start']} to {result['split']['hold_end']})",
        "",
    ]

    # Test 1
    lines += [
        "## Test 1: Propagation-Rate Regime Probe",
        "",
        "| Metric | Cal | Hold | Delta |",
        "| --- | ---: | ---: | ---: |",
    ]
    t1 = result["test1_propagation"]
    cal1 = t1["cal"]
    hold1 = t1["hold"]
    for key, label in [
        ("eth_prop_rate", "ETH same-side prop rate (60min)"),
        ("cross_asset_prop_rate", "Cross-asset prop rate (60min)"),
        ("any_prop_rate", "Any prop rate (60min)"),
        ("sync_rate", "SYNC rate (BTC+SOL >=200K prior 10min)"),
        ("mean_sync_k", "Mean sync_k (K units)"),
    ]:
        c = cal1.get(key, "?")
        h = hold1.get(key, "?")
        delta = ""
        if isinstance(c, (int, float)) and isinstance(h, (int, float)):
            d = h - c
            delta = f"+{r3(d)}" if d > 0 else r3(d)
        lines.append(f"| {label} | {c} | {h} | {delta} |")
    lines += ["", f"*Interpretation*: {t1['interpretation']}", ""]

    # Test 2
    lines += [
        "## Test 2: SYNC + k5=CLEAN Composite Gate Holdout",
        "",
        "| Group | Cal N | Cal T3R | Cal median | Cal win | Hold N | Hold T3R | Hold median | Hold win | Hold maxL |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    t2 = result["test2_sync_clean"]
    for name, vals in t2.items():
        c = vals["cal"]
        h = vals["hold"]
        lines.append(
            f"| {name} | {c['n']} | {c.get('t3r')} | {c['median']} | {c['win']} |"
            f" {h['n']} | {h.get('t3r')} | {h['median']} | {h['win']} | {h['max_loss']} |"
        )
    lines.append("")

    # Test 3
    lines += [
        "## Test 3: Frequency Expansion / Timing Holdout",
        "",
        "### Exit-timing comparison (all events)",
        "",
        "| Split | N | 2h T3R | 2h median | 2h win | 4h T3R | 4h median | TP300/SL150 T3R | TP300/SL150 median |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    t3 = result["test3_freq_expansion"]
    for label, key in [("Cal", "cal_all"), ("Hold", "hold_all")]:
        d = t3[key]
        r2 = d["raw_2h"]
        r4 = d["raw_4h"]
        tp = d["tp300_sl150"]
        lines.append(
            f"| {label} | {d['n']} | {r2.get('t3r')} | {r2['median']} | {r2['win']} |"
            f" {r4.get('t3r')} | {r4['median']} | {tp.get('t3r')} | {tp['median']} |"
        )

    lines += [
        "",
        "### Tags with cal T3R > 0, ranked by hold T3R",
        "",
        "| Tag | Cal N | Cal T3R | Cal median | Hold N | Hold T3R | Hold median | Hold win |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for tag, vals in t3["tags_by_hold_t3r"].items():
        c = vals["cal"]
        h = vals["hold"]
        lines.append(
            f"| {tag} | {c['n']} | {c.get('t3r')} | {c['median']} |"
            f" {h['n']} | {h.get('t3r')} | {h['median']} | {h['win']} |"
        )
    if t3["top_hold_positive_tags"]:
        lines += ["", f"Hold-positive tags: `{t3['top_hold_positive_tags']}`"]
    lines.append("")

    # Test 4
    lines += [
        "## Test 4: Monthly k5=CLEAN Stability",
        "",
        "(*) = holdout period",
        "",
        "| Month | All N | All med | All win | k5=CLEAN N | CLEAN med | CLEAN win | CLEAN T3R | k5=DANGER REV med | REV win |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for m, d in sorted(result["test4_monthly"].items()):
        star = "(*)" if d.get("in_holdout") else ""
        lines.append(
            f"| {m}{star} | {d['all_n']} | {d['all_median']} | {d['all_win']} |"
            f" {d['k5_clean_n']} | {d['k5_clean_median']} | {d['k5_clean_win']} | {d.get('k5_clean_t3r', 'small-N')} |"
            f" {d['k5_danger_rev_median']} | {d['k5_danger_rev_win']} |"
        )
    lines.append("")

    lines += [
        "## Overall Verdict",
        "",
        "- If Test 1 shows hold prop rate >> cal -> regime change explains holdout failure.",
        "- If Test 2 SYNC+k5=CLEAN has hold T3R > 0 -> composite gate has residual signal.",
        "- If Test 3 TP300/SL150 T3R > 0 in hold -> exit management recovers some edge.",
        "- Test 4 monthly: which month did CLEAN structure break? That month = regime break.",
        "",
        "All results RESEARCH_ONLY. No live/paper promotion without OOS+ permutation-null.",
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
    n_cal = int(n_total * (1.0 - HOLDOUT_FRAC))
    cal_rows_raw = all_rows[:n_cal]
    hold_rows_raw = all_rows[n_cal:]
    print(f"Total: {n_total}  Cal: {len(cal_rows_raw)}  Hold: {len(hold_rows_raw)}")

    cal_vecs = [feature_vector(r) for r in cal_rows_raw]

    print("Building cal KNN labels (leave-one-out)...")
    cal_rows = build_labels(cal_rows_raw, cal_rows_raw, cal_vecs, KS)

    print("Building hold KNN labels (cal-neighbor pool)...")
    hold_rows = build_labels(hold_rows_raw, cal_rows_raw, cal_vecs, KS)

    split_info = {
        "n_total": n_total,
        "cal_n": len(cal_rows),
        "hold_n": len(hold_rows),
        "cal_start": ts_to_utc(cal_rows_raw[0]["signal_ts_ms"]),
        "cal_end": ts_to_utc(cal_rows_raw[-1]["signal_ts_ms"]),
        "hold_start": ts_to_utc(hold_rows_raw[0]["signal_ts_ms"]),
        "hold_end": ts_to_utc(hold_rows_raw[-1]["signal_ts_ms"]),
    }

    print("Connecting to DB...")
    with sqlite3.connect(f"file:{DEFAULT_DB}?mode=ro", uri=True) as conn:
        print("Test 1: Propagation rate regime probe...")
        t1 = test1_propagation_rate(conn, cal_rows, hold_rows)

        print("Test 2: SYNC + k5=CLEAN composite...")
        t2 = test2_sync_clean_composite(conn, cal_rows, hold_rows)

        print("Test 3: Frequency expansion holdout calibration...")
        t3 = test3_freq_expansion_holdout(conn, cal_rows, hold_rows)

    print("Test 4: Monthly stability...")
    t4 = test4_monthly_stability(cal_rows, hold_rows)

    result = {
        "generated_at_utc": utc_now(),
        "status": "RESEARCH_ONLY_NO_LIVE_CHANGE",
        "split": split_info,
        "test1_propagation": t1,
        "test2_sync_clean": t2,
        "test3_freq_expansion": t3,
        "test4_monthly": t4,
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(result, indent=2, ensure_ascii=True), encoding="utf-8")
    md = render_md(result)
    OUT_MD.write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
