"""
S34 Live Paper Trade Analysis
Extracts market conditions from actual runner paper trades and compares
winners vs losers to find what separates good entries from bad ones.

No runner/config/pre-reg changes. Research only.
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
OUT_MD  = OUT_DIR / "S34_LIVE_TRADE_ANALYSIS.md"

PRELIMINARY_N = 10  # lower bar since real data is scarce


def _r1(v): return round(float(v), 1) if v is not None and math.isfinite(float(v)) else None
def _r3(v): return round(float(v), 3) if v is not None and math.isfinite(float(v)) else None

def _median(vals):
    c = sorted(v for v in vals if v is not None and math.isfinite(v))
    if not c: return None
    i = (len(c) - 1) / 2
    lo, hi = math.floor(i), math.ceil(i)
    return c[lo] if lo == hi else (c[lo] + c[hi]) / 2

def _mean(vals):
    c = [v for v in vals if v is not None and math.isfinite(v)]
    return sum(c) / len(c) if c else None


def load_trades(conn) -> list[dict]:
    rows = conn.execute(
        "SELECT trade_json FROM s34_trades WHERE status='CLOSED' AND trade_json IS NOT NULL"
    ).fetchall()
    trades = []
    for (raw,) in rows:
        try:
            t = json.loads(raw)
            if t.get("net_bps") is not None:
                trades.append(t)
        except Exception:
            pass
    return trades


def extract_features(t: dict) -> dict:
    regime  = t.get("regime") or {}
    signal  = t.get("signal") or {}
    rule    = t.get("rule") or {}

    entry_ms = int(t.get("entry_ts_ms") or t.get("signal_ts_ms") or 0)
    exit_ms  = int(t.get("exit_ts_ms") or 0)
    hold_sec = (exit_ms - entry_ms) / 1000 if exit_ms > entry_ms else None

    # Hour of entry (UTC)
    hour = datetime.fromtimestamp(entry_ms / 1000, tz=timezone.utc).hour if entry_ms else None

    return {
        "trade_id":         t.get("trade_id"),
        "rule_name":        rule.get("name") or t.get("rule_name"),
        "symbol":           t.get("symbol"),
        "direction":        t.get("direction"),
        "net_bps":          float(t["net_bps"]),
        "exit_reason":      t.get("exit_reason"),
        "hold_sec":         hold_sec,
        "entry_hour_utc":   hour,
        # Regime at entry
        "day_trend_pct":    regime.get("trend_pct"),
        "day_range_pct":    regime.get("range_pct"),
        "buy_liq_notional": regime.get("buy_liq_notional"),
        "agg_trade_count":  regime.get("agg_trade_count"),
        # Signal quality
        "cascade_notional": signal.get("liq_total_notional"),
        "liq_count":        signal.get("liq_count"),
        "liq_max_notional": signal.get("liq_max_notional"),
        # Rule params
        "threshold_usd":    rule.get("threshold_usd"),
        "tp_bps":           rule.get("tp_bps"),
        "sl_bps":           rule.get("sl_bps"),
    }


def group_stats(feats: list[dict], field: str, label: str) -> list[dict]:
    """Group by a categorical field and compute stats."""
    groups: dict[str, list[float]] = {}
    for f in feats:
        key = str(f.get(field) or "UNKNOWN")
        groups.setdefault(key, []).append(f["net_bps"])
    out = []
    for key, nets in sorted(groups.items(), key=lambda x: -len(x[1])):
        wr = sum(1 for n in nets if n > 0) / len(nets)
        out.append({
            "key":    key,
            "label":  label,
            "n":      len(nets),
            "median": _r1(_median(nets)),
            "mean":   _r1(_mean(nets)),
            "cum":    _r1(sum(nets)),
            "wr":     _r3(wr),
        })
    return out


def bin_stats(feats: list[dict], field: str, bins: list, label: str) -> list[dict]:
    """Group by numeric bins and compute stats."""
    def bucket(v):
        if v is None: return "N/A"
        for i, edge in enumerate(bins):
            if v < edge:
                lo = bins[i-1] if i > 0 else -math.inf
                return f"[{lo:.1f},{edge:.1f})"
        return f"[{bins[-1]:.1f},+inf)"

    groups: dict[str, list[float]] = {}
    for f in feats:
        key = bucket(f.get(field))
        groups.setdefault(key, []).append(f["net_bps"])

    out = []
    for key in sorted(groups):
        nets = groups[key]
        wr = sum(1 for n in nets if n > 0) / len(nets)
        out.append({
            "key":    key,
            "label":  label,
            "n":      len(nets),
            "median": _r1(_median(nets)),
            "mean":   _r1(_mean(nets)),
            "cum":    _r1(sum(nets)),
            "wr":     _r3(wr),
        })
    return out


def winner_loser_split(feats: list[dict]) -> tuple[list[dict], list[dict]]:
    winners = [f for f in feats if f["net_bps"] > 0]
    losers  = [f for f in feats if f["net_bps"] <= 0]
    return winners, losers


def compare_field(winners: list[dict], losers: list[dict], field: str) -> dict:
    wv = [f[field] for f in winners if f.get(field) is not None]
    lv = [f[field] for f in losers  if f.get(field) is not None]
    return {
        "field":          field,
        "winner_median":  _r3(_median(wv)),
        "loser_median":   _r3(_median(lv)),
        "winner_mean":    _r3(_mean(wv)),
        "loser_mean":     _r3(_mean(lv)),
        "winner_n":       len(wv),
        "loser_n":        len(lv),
    }


def md_table(rows: list[dict], cols: list[str]) -> list[str]:
    header = "| " + " | ".join(cols) + " |"
    sep    = "| " + " | ".join(["---"] * len(cols)) + " |"
    lines  = [header, sep]
    for r in rows:
        cells = []
        for c in cols:
            v = r.get(c)
            cells.append(str(v) if v is not None else "")
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    conn   = sqlite3.connect(f"file:{DB_PATH.as_posix()}?mode=ro", uri=True)
    trades = load_trades(conn)
    conn.close()

    now   = datetime.now(timezone.utc).isoformat()
    feats = [extract_features(t) for t in trades]

    all_nets  = [f["net_bps"] for f in feats]
    all_wr    = sum(1 for n in all_nets if n > 0) / len(all_nets) if all_nets else 0
    winners, losers = winner_loser_split(feats)

    print(f"S34 Live Trade Analysis — {now}")
    print(f"Total closed: {len(feats)}  WR={all_wr*100:.0f}%  median={_r1(_median(all_nets)):+}  cum={_r1(sum(all_nets)):+}")
    print()

    lines = [
        "# S34 Live Paper Trade Analysis",
        "", f"Generated: `{now}`", "",
        "Analyzes actual runner paper trades to find conditions separating winners from losers.",
        "",
        f"**Total closed:** {len(feats)}  **WR:** {all_wr*100:.0f}%  "
        f"**Median:** {_r1(_median(all_nets)):+} bps  **Cum:** {_r1(sum(all_nets)):+} bps",
        "",
    ]

    # ── 1. By rule ────────────────────────────────────────────────────────────
    print("=== BY RULE ===")
    rule_stats = group_stats(feats, "rule_name", "rule")
    for r in rule_stats:
        print(f"  {r['key'][:50]}  N={r['n']}  median={r['median']:+}  WR={r['wr']*100:.0f}%  cum={r['cum']:+}")

    lines += ["## By Rule", ""]
    lines += md_table(rule_stats, ["key", "n", "median", "cum", "wr"])
    lines.append("")

    # ── 2. Winner vs Loser: regime fields ────────────────────────────────────
    print("\n=== WINNER vs LOSER — REGIME CONDITIONS ===")
    compare_fields = [
        "day_trend_pct", "day_range_pct", "buy_liq_notional",
        "agg_trade_count", "cascade_notional", "liq_count",
        "liq_max_notional", "hold_sec", "entry_hour_utc",
    ]
    comparisons = [compare_field(winners, losers, f) for f in compare_fields]
    for c in comparisons:
        delta = None
        if c["winner_median"] is not None and c["loser_median"] is not None:
            delta = _r3(c["winner_median"] - c["loser_median"])
        print(f"  {c['field']:25s}  W={c['winner_median']}  L={c['loser_median']}  delta={delta}")

    lines += ["## Winner vs Loser — Regime Conditions", "",
              f"Winners N={len(winners)}, Losers N={len(losers)}", ""]
    lines += md_table(comparisons, ["field", "winner_median", "loser_median", "winner_mean", "loser_mean"])
    lines.append("")

    # ── 3. Day trend bins ─────────────────────────────────────────────────────
    print("\n=== BY DAY TREND PCT ===")
    trend_bins = bin_stats(feats, "day_trend_pct", [0.0, 1.0, 2.0, 4.0], "day_trend_pct")
    for r in trend_bins:
        print(f"  trend={r['key']}  N={r['n']}  median={r['median']:+}  WR={r['wr']*100:.0f}%  cum={r['cum']:+}")

    lines += ["## By Day Trend (%)", ""]
    lines += md_table(trend_bins, ["key", "n", "median", "cum", "wr"])
    lines.append("")

    # ── 4. Day range bins ─────────────────────────────────────────────────────
    print("\n=== BY DAY RANGE PCT ===")
    range_bins = bin_stats(feats, "day_range_pct", [2.5, 3.0, 4.0, 6.0], "day_range_pct")
    for r in range_bins:
        print(f"  range={r['key']}  N={r['n']}  median={r['median']:+}  WR={r['wr']*100:.0f}%  cum={r['cum']:+}")

    lines += ["## By Day Range (%)", ""]
    lines += md_table(range_bins, ["key", "n", "median", "cum", "wr"])
    lines.append("")

    # ── 5. Cascade notional bins ──────────────────────────────────────────────
    print("\n=== BY CASCADE NOTIONAL ===")
    cas_bins = bin_stats(feats, "cascade_notional", [100_000, 200_000, 500_000, 1_000_000], "cascade_notional")
    for r in cas_bins:
        print(f"  cascade={r['key']}  N={r['n']}  median={r['median']:+}  WR={r['wr']*100:.0f}%  cum={r['cum']:+}")

    lines += ["## By Cascade Notional", ""]
    lines += md_table(cas_bins, ["key", "n", "median", "cum", "wr"])
    lines.append("")

    # ── 6. Entry hour bins ────────────────────────────────────────────────────
    print("\n=== BY ENTRY HOUR (UTC) ===")
    hour_bins = bin_stats(feats, "entry_hour_utc", [4, 8, 12, 16, 20], "entry_hour_utc")
    for r in hour_bins:
        print(f"  hour={r['key']}  N={r['n']}  median={r['median']:+}  WR={r['wr']*100:.0f}%  cum={r['cum']:+}")

    lines += ["## By Entry Hour (UTC)", ""]
    lines += md_table(hour_bins, ["key", "n", "median", "cum", "wr"])
    lines.append("")

    # ── 7. Exit reason breakdown ──────────────────────────────────────────────
    print("\n=== BY EXIT REASON ===")
    exit_stats = group_stats(feats, "exit_reason", "exit_reason")
    for r in exit_stats:
        print(f"  {r['key']}  N={r['n']}  median={r['median']:+}  cum={r['cum']:+}")

    lines += ["## By Exit Reason", ""]
    lines += md_table(exit_stats, ["key", "n", "median", "cum", "wr"])
    lines.append("")

    # ── 8. Top winners and worst losers ───────────────────────────────────────
    sorted_feats = sorted(feats, key=lambda f: -f["net_bps"])
    print("\n=== TOP 10 WINNERS ===")
    lines += ["## Top 10 Winners", ""]
    for f in sorted_feats[:10]:
        line = (
            f"  {f['trade_id']}  {f['rule_name'][:40]}  "
            f"net={f['net_bps']:+.1f}  trend={f['day_trend_pct']:.2f}%  "
            f"range={f['day_range_pct']:.2f}%  cas={f['cascade_notional']:.0f}  "
            f"exit={f['exit_reason']}"
        )
        print(line)
        lines.append(
            f"- `{f['trade_id']}` {f['rule_name'][:40]} "
            f"**{f['net_bps']:+.1f}bps** exit={f['exit_reason']} "
            f"trend={f['day_trend_pct']:.2f}% range={f['day_range_pct']:.2f}% "
            f"cascade={f['cascade_notional']:.0f}"
        )

    print("\n=== BOTTOM 10 LOSERS ===")
    lines += ["", "## Bottom 10 Losers", ""]
    for f in sorted_feats[-10:]:
        line = (
            f"  {f['trade_id']}  {f['rule_name'][:40]}  "
            f"net={f['net_bps']:+.1f}  trend={f['day_trend_pct']:.2f}%  "
            f"range={f['day_range_pct']:.2f}%  cas={f['cascade_notional']:.0f}  "
            f"exit={f['exit_reason']}"
        )
        print(line)
        lines.append(
            f"- `{f['trade_id']}` {f['rule_name'][:40]} "
            f"**{f['net_bps']:+.1f}bps** exit={f['exit_reason']} "
            f"trend={f['day_trend_pct']:.2f}% range={f['day_range_pct']:.2f}% "
            f"cascade={f['cascade_notional']:.0f}"
        )

    lines.append("")
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nMD: {OUT_MD}")


if __name__ == "__main__":
    main()
