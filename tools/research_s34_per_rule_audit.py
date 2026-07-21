"""
S34 Active Rule Bucket Refinement Audit
Per-rule validation of pooled findings. No runner/config/pre-reg changes.

Tests whether pooled patterns (day_trend >4% bad, 20-24 UTC weak,
cascade <200K bad) hold within each active candidate rule separately.
"""
from __future__ import annotations
import json
import math
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

ROOT    = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "s34_intelligence.db"
OUT_DIR = ROOT / "reports" / "research" / "s34"
OUT_MD  = OUT_DIR / "S34_PER_RULE_AUDIT.md"

MIN_N_FILTER = 20   # don't recommend a filter if N drops below this
MIN_N_BUCKET =  5   # don't report a bucket below this (mark as thin)

ACTIVE_RULES = [
    "ETH_BUY_LIQ_LONG_500K_DAYTREND0_TP60_SL40_BE30",
    "SOL_BUY_LIQ_LONG_200K_TP60_SL40_BE30",
    "ETH_BUY_LIQ_LONG_200K_TP60_SL40_BE30",
    "BTC_BUY_LIQ_LONG_1M_DISTRIBUTED_TP60_SL30_BE30",
    "ETH_BUY_LIQ_LONG_50K_TP120_SL40_BE30",   # included for diagnosis only
]

# ── helpers ──────────────────────────────────────────────────────────────────

def _r1(v): return round(float(v), 1) if v is not None and math.isfinite(float(v)) else None
def _r3(v): return round(float(v), 3) if v is not None and math.isfinite(float(v)) else None

def _median(vals):
    c = sorted(v for v in vals if v is not None and math.isfinite(v))
    if not c: return None
    i = (len(c) - 1) / 2
    lo, hi = math.floor(i), math.ceil(i)
    return c[lo] if lo == hi else (c[lo] + c[hi]) / 2

def _top3r(nets):
    s = sorted(nets)
    return sum(s[3:]) if len(s) > 3 else sum(s)

def _stats(nets: list[float]) -> dict:
    if not nets:
        return {"n": 0}
    wr = sum(1 for n in nets if n > 0) / len(nets)
    return {
        "n":      len(nets),
        "median": _r1(_median(nets)),
        "cum":    _r1(sum(nets)),
        "top3r":  _r1(_top3r(nets)),
        "wr":     round(wr, 3),
    }

def _bucket(val, edges: list[float]) -> str:
    if val is None:
        return "N/A"
    for i, edge in enumerate(edges):
        if val < edge:
            lo = edges[i - 1] if i > 0 else None
            lo_s = f"{lo:.0f}" if lo is not None else "-inf"
            return f"[{lo_s},{edge:.0f})"
    return f"[{edges[-1]:.0f},+inf)"

def _md_row(cells) -> str:
    return "| " + " | ".join(str(c) if c is not None else "" for c in cells) + " |"

def _md_table(header: list[str], rows: list[list]) -> list[str]:
    lines = [_md_row(header), _md_row(["---"] * len(header))]
    for r in rows:
        lines.append(_md_row(r))
    return lines


# ── data loading ──────────────────────────────────────────────────────────────

def load_trades(conn) -> list[dict]:
    rows = conn.execute(
        "SELECT trade_json FROM s34_trades WHERE status='CLOSED' AND trade_json IS NOT NULL"
    ).fetchall()
    out = []
    for (raw,) in rows:
        try:
            t = json.loads(raw)
            if t.get("net_bps") is not None:
                out.append(t)
        except Exception:
            pass
    return out


def featurize(t: dict) -> dict:
    regime = t.get("regime") or {}
    signal = t.get("signal") or {}
    rule   = t.get("rule") or {}
    entry_ms = int(t.get("entry_ts_ms") or t.get("signal_ts_ms") or 0)
    exit_ms  = int(t.get("exit_ts_ms") or 0)
    hour = datetime.fromtimestamp(entry_ms / 1000, tz=timezone.utc).hour if entry_ms else None
    return {
        "rule_name":         (rule.get("name") or t.get("rule_name", "")),
        "net_bps":           float(t["net_bps"]),
        "exit_reason":       t.get("exit_reason"),
        "day_trend_pct":     regime.get("trend_pct"),
        "day_range_pct":     regime.get("range_pct"),
        "buy_liq_notional":  regime.get("buy_liq_notional"),
        "cascade_notional":  signal.get("liq_total_notional"),
        "liq_count":         signal.get("liq_count"),
        "liq_max_notional":  signal.get("liq_max_notional"),
        "entry_hour_utc":    hour,
        "hold_sec":          (exit_ms - entry_ms) / 1000 if exit_ms > entry_ms else None,
        "max_single_share":  (
            (signal.get("liq_max_notional") / signal.get("liq_total_notional") * 100)
            if signal.get("liq_total_notional") and signal.get("liq_max_notional") else None
        ),
    }


# ── per-rule analysis ─────────────────────────────────────────────────────────

def analyze_rule(name: str, feats: list[dict]) -> list[str]:
    nets_all = [f["net_bps"] for f in feats]
    s        = _stats(nets_all)
    lines    = [
        f"## {name}",
        "",
        f"**N={s['n']}  Median={s['median']:+}  WR={s['wr']*100:.0f}%  Cum={s['cum']:+}  Top3R={s['top3r']:+}**",
        "",
    ]

    # ── Q1: day trend bins ────────────────────────────────────────────────────
    lines.append("### Day Trend Bins")
    lines.append("")
    trend_edges = [0.0, 100.0, 200.0, 400.0]  # in bps → convert below
    # trade stores trend_pct (percent, e.g. 1.5 means 1.5%)
    # edges in pct
    edges_pct = [0.0, 1.0, 2.0, 4.0]
    rows = []
    for i in range(len(edges_pct) + 1):
        if i == 0:
            lo, hi = None, edges_pct[0]
            label = f"<{hi:.0f}%"
        elif i == len(edges_pct):
            lo, hi = edges_pct[-1], None
            label = f">={lo:.0f}%"
        else:
            lo, hi = edges_pct[i - 1], edges_pct[i]
            label = f"{lo:.0f}–{hi:.0f}%"
        bucket_nets = [
            f["net_bps"] for f in feats
            if f["day_trend_pct"] is not None
            and (lo is None or f["day_trend_pct"] >= lo)
            and (hi is None or f["day_trend_pct"] < hi)
        ]
        st = _stats(bucket_nets)
        thin = "(thin)" if 0 < st["n"] < MIN_N_BUCKET else ""
        rows.append([label, st["n"], st.get("median"), st.get("cum"), st.get("top3r"),
                     f"{st['wr']*100:.0f}%" if st["n"] else "", thin])
    lines += _md_table(["Trend", "N", "Median", "Cum", "Top3R", "WR", "Note"], rows)
    lines.append("")

    # ── Q2: UTC session bins ──────────────────────────────────────────────────
    lines.append("### UTC Session Bins")
    lines.append("")
    sessions = [(0, 4), (4, 8), (8, 12), (12, 16), (16, 20), (20, 24)]
    s_rows = []
    for lo, hi in sessions:
        label = f"{lo:02d}-{hi:02d}"
        bucket_nets = [
            f["net_bps"] for f in feats
            if f["entry_hour_utc"] is not None
            and lo <= f["entry_hour_utc"] < hi
        ]
        st = _stats(bucket_nets)
        thin = "(thin)" if 0 < st["n"] < MIN_N_BUCKET else ""
        s_rows.append([label, st["n"], st.get("median"), st.get("cum"),
                       f"{st['wr']*100:.0f}%" if st["n"] else "", thin])
    lines += _md_table(["Session (UTC)", "N", "Median", "Cum", "WR", "Note"], s_rows)
    lines.append("")

    # ── Q3: cascade notional bins ─────────────────────────────────────────────
    lines.append("### Cascade Notional Bins")
    lines.append("")
    cas_edges = [100_000, 200_000, 500_000, 1_000_000]
    c_rows = []
    for i in range(len(cas_edges) + 1):
        if i == 0:
            lo, hi = None, cas_edges[0]
            label = f"<{hi//1000:.0f}K"
        elif i == len(cas_edges):
            lo, hi = cas_edges[-1], None
            label = f">{lo//1000:.0f}K"
        else:
            lo, hi = cas_edges[i - 1], cas_edges[i]
            label = f"{lo//1000:.0f}K-{hi//1000:.0f}K"
        bucket_nets = [
            f["net_bps"] for f in feats
            if f["cascade_notional"] is not None
            and (lo is None or f["cascade_notional"] >= lo)
            and (hi is None or f["cascade_notional"] < hi)
        ]
        st = _stats(bucket_nets)
        thin = "(thin)" if 0 < st["n"] < MIN_N_BUCKET else ""
        c_rows.append([label, st["n"], st.get("median"), st.get("cum"),
                       f"{st['wr']*100:.0f}%" if st["n"] else "", thin])
    lines += _md_table(["Cascade", "N", "Median", "Cum", "WR", "Note"], c_rows)
    lines.append("")

    # ── Q4: max single liq share ──────────────────────────────────────────────
    lines.append("### Max Single Liq Share Bins")
    lines.append("")
    share_edges = [50.0, 80.0]
    sh_rows = []
    for i in range(len(share_edges) + 1):
        if i == 0:
            lo, hi = None, share_edges[0]
            label = f"<{hi:.0f}%"
        elif i == len(share_edges):
            lo, hi = share_edges[-1], None
            label = f">={lo:.0f}%"
        else:
            lo, hi = share_edges[i - 1], share_edges[i]
            label = f"{lo:.0f}–{hi:.0f}%"
        bucket_nets = [
            f["net_bps"] for f in feats
            if f["max_single_share"] is not None
            and (lo is None or f["max_single_share"] >= lo)
            and (hi is None or f["max_single_share"] < hi)
        ]
        st = _stats(bucket_nets)
        thin = "(thin)" if 0 < st["n"] < MIN_N_BUCKET else ""
        sh_rows.append([label, st["n"], st.get("median"), st.get("cum"),
                        f"{st['wr']*100:.0f}%" if st["n"] else "", thin])
    lines += _md_table(["Single Share", "N", "Median", "Cum", "WR", "Note"], sh_rows)
    lines.append("")

    # ── Q5: candidate filters ─────────────────────────────────────────────────
    lines.append("### Candidate Filter Tests")
    lines.append("")
    filters = [
        ("max_day_trend <= 4%",    lambda f: f["day_trend_pct"] is not None and f["day_trend_pct"] <= 4.0),
        ("max_day_trend <= 3%",    lambda f: f["day_trend_pct"] is not None and f["day_trend_pct"] <= 3.0),
        ("exclude UTC 20-24",      lambda f: f["entry_hour_utc"] is not None and not (20 <= f["entry_hour_utc"] < 24)),
        ("min_liq_count >= 3",     lambda f: f["liq_count"] is not None and f["liq_count"] >= 3),
        ("min_liq_count >= 5",     lambda f: f["liq_count"] is not None and f["liq_count"] >= 5),
        ("max_single_share <= 80%",lambda f: f["max_single_share"] is not None and f["max_single_share"] <= 80.0),
        ("cascade >= 200K",        lambda f: f["cascade_notional"] is not None and f["cascade_notional"] >= 200_000),
        ("max_trend<=4% + UTC<20", lambda f: (
            f["day_trend_pct"] is not None and f["day_trend_pct"] <= 4.0
            and f["entry_hour_utc"] is not None and not (20 <= f["entry_hour_utc"] < 24)
        )),
    ]
    f_header = ["Filter", "N (kept)", "N (removed)", "Median", "Cum", "Top3R", "WR", "Note"]
    f_rows = []
    baseline = _stats(nets_all)
    for fname, fn in filters:
        kept   = [f for f in feats if fn(f)]
        k_nets = [f["net_bps"] for f in kept]
        k_st   = _stats(k_nets)
        n_removed = len(feats) - k_st["n"]
        notes = []
        if k_st["n"] < MIN_N_FILTER:
            notes.append("N<20 — too thin")
        if n_removed <= 2 and (k_st.get("median") or 0) > (baseline.get("median") or 0):
            notes.append("improvement from ≤2 removes")
        note_str = "; ".join(notes) if notes else "ok"
        f_rows.append([
            fname, k_st["n"], n_removed,
            k_st.get("median"), k_st.get("cum"), k_st.get("top3r"),
            f"{k_st['wr']*100:.0f}%" if k_st["n"] else "",
            note_str,
        ])
    lines += _md_table(f_header, f_rows)
    lines.append("")
    return lines


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    conn   = sqlite3.connect(f"file:{DB_PATH.as_posix()}?mode=ro", uri=True)
    trades = load_trades(conn)
    conn.close()

    now   = datetime.now(timezone.utc).isoformat()
    feats = [featurize(t) for t in trades]

    # group by rule
    by_rule: dict[str, list[dict]] = {}
    for f in feats:
        by_rule.setdefault(f["rule_name"], []).append(f)

    print(f"S34 Per-Rule Audit — {now}")
    print(f"Total trades: {len(feats)}  Rules seen: {sorted(by_rule)}\n")

    lines = [
        "# S34 Active Rule Bucket Refinement Audit",
        "",
        f"Generated: `{now}`",
        "",
        "Per-rule validation of pooled findings. **No runner/config changes.**",
        "Tests: day_trend >4%, UTC 20-24 weakness, cascade size, single-liq dominance.",
        "",
        "Pooled findings to validate:",
        "1. day_trend > 4% hurts BUY routes",
        "2. UTC 20-24 underperforms",
        "3. Cascade < 200K is bad",
        "4. ETH_BUY_50K is a drag (diagnosis only — confirm not in active allow list)",
        "",
    ]

    for rule_name in ACTIVE_RULES:
        rule_feats = by_rule.get(rule_name)
        if not rule_feats:
            lines += [f"## {rule_name}", "", "_No data_", ""]
            print(f"[SKIP] {rule_name} — no trades in DB")
            continue
        print(f"\n{'='*60}")
        print(f"[{rule_name}] N={len(rule_feats)}")
        nets = [f["net_bps"] for f in rule_feats]
        print(f"  median={_r1(_median(nets)):+}  WR={sum(1 for n in nets if n>0)/len(nets)*100:.0f}%  cum={_r1(sum(nets)):+}")
        lines += analyze_rule(rule_name, rule_feats)

    # ── summary ───────────────────────────────────────────────────────────────
    lines += [
        "---",
        "## Summary — Which Pooled Findings Survive Per-Rule?",
        "",
        "| Finding | ETH 500K | SOL 200K | ETH 200K | BTC 1M | ETH 50K | Verdict |",
        "| --- | --- | --- | --- | --- | --- | --- |",
        "| day_trend >4% bad | ? | ? | ? | ? | ? | see per-rule tables |",
        "| UTC 20-24 weak | ? | ? | ? | ? | ? | see per-rule tables |",
        "| cascade <200K bad | ? | ? | ? | ? | ? | see per-rule tables |",
        "| max_single_share | ? | ? | ? | ? | ? | see per-rule tables |",
        "",
        "> **No live rule change recommended** without consistent evidence across rule-specific data.",
        "",
    ]

    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nMD: {OUT_MD}")


if __name__ == "__main__":
    main()
