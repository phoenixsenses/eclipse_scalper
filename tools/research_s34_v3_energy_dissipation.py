from __future__ import annotations

import bisect
import json
import math
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.research_s34_cross_asset_absorption_pool import DEFAULT_DB
from tools.research_s34_knowable_anchor_continuation import load_liquidations
from tools.research_s34_wave_absorption import book_features_at


IN_JSON = ROOT / "reports" / "research" / "s34" / "S34_ABSORPTION_SYNC_2X2_POOL.json"
FALLBACK_JSON = ROOT / "reports" / "research" / "s34" / "S34_CROSS_ASSET_ABSORPTION_POOL.json"
OUT_JSON = ROOT / "reports" / "research" / "s34" / "S34_V3_ENERGY_DISSIPATION.json"
OUT_MD = ROOT / "reports" / "research" / "s34" / "S34_V3_ENERGY_DISSIPATION.md"

HORIZONS_SEC = (30, 60, 120)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def finite(value: Any) -> float | None:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    vals = [finite(r.get("net_bps")) for r in rows]
    vals = [v for v in vals if v is not None]
    if not vals:
        return {
            "n": 0,
            "sum_bps": 0.0,
            "mean_bps": None,
            "median_bps": None,
            "win_rate_pct": None,
            "t3r_bps": 0.0,
            "max_loss_bps": None,
            "tail_lt_100": 0,
        }
    ordered = sorted(vals, reverse=True)
    return {
        "n": len(vals),
        "sum_bps": round(sum(vals), 1),
        "mean_bps": round(sum(vals) / len(vals), 1),
        "median_bps": round(median(vals), 1),
        "win_rate_pct": round(100.0 * sum(1 for v in vals if v > 0.0) / len(vals), 1),
        "t3r_bps": round(sum(ordered[3:]) if len(ordered) > 3 else sum(vals), 1),
        "max_loss_bps": round(min(vals), 1),
        "tail_lt_100": sum(1 for v in vals if v < -100.0),
    }


def percentile(vals: list[float], q: float) -> float | None:
    xs = sorted(v for v in vals if math.isfinite(v))
    if not xs:
        return None
    idx = int(round((len(xs) - 1) * q))
    return xs[max(0, min(len(xs) - 1, idx))]


def window_liq(ts: list[int], rows: list[dict[str, Any]], start_ms: int, end_ms: int) -> float:
    lo = bisect.bisect_right(ts, int(start_ms))
    hi = bisect.bisect_right(ts, int(end_ms))
    return sum(float(rows[i]["notional"]) for i in range(lo, hi))


def load_liq_index(conn: sqlite3.Connection) -> dict[str, tuple[list[int], list[dict[str, Any]]]]:
    out = {}
    for symbol in ("BTCUSDT", "ETHUSDT", "SOLUSDT"):
        rows = load_liquidations(conn, symbol, "SELL", None, None)
        out[symbol] = ([int(r["ts_ms"]) for r in rows], rows)
    return out


def split_rows(rows: list[dict[str, Any]], payload: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], set[str]]:
    hold_months = set(payload.get("split", {}).get("holdout_months", []))
    cal = [r for r in rows if str(r.get("month")) not in hold_months]
    hold = [r for r in rows if str(r.get("month")) in hold_months]
    return cal, hold, hold_months


def annotate_rows(conn: sqlite3.Connection, rows: list[dict[str, Any]], *, max_book_staleness_sec: int) -> list[dict[str, Any]]:
    liq_idx = load_liq_index(conn)
    out = []
    for row in rows:
        r = dict(row)
        symbol = str(row["symbol"])
        entry_ts = int(row["entry_ts_ms"])
        base_bid = finite(row.get("bid_depth_usd"))
        base_total = finite(row.get("total_top_depth_usd"))
        base_spread = finite(row.get("spread_bps"))
        ts, lrows = liq_idx[symbol]
        for sec in HORIZONS_SEC:
            bf = book_features_at(conn, symbol, entry_ts + sec * 1000, int(max_book_staleness_sec))
            if bf and base_bid and base_bid > 0:
                r[f"bid_replenish_{sec}s_pct"] = (float(bf["bid_depth_usd"]) - base_bid) / base_bid * 100.0
            if bf and base_total and base_total > 0:
                r[f"total_replenish_{sec}s_pct"] = (float(bf["total_top_depth_usd"]) - base_total) / base_total * 100.0
            if bf and base_spread is not None:
                r[f"spread_change_{sec}s_bps"] = float(bf["spread_bps"]) - base_spread
            pre = window_liq(ts, lrows, entry_ts - sec * 1000, entry_ts)
            post = window_liq(ts, lrows, entry_ts, entry_ts + sec * 1000)
            r[f"pre_liq_{sec}s_k"] = pre / 1000.0
            r[f"post_liq_{sec}s_k"] = post / 1000.0
            r[f"liq_deceleration_{sec}s"] = (pre - post) / max(pre, 1.0)
            bid_repl = finite(r.get(f"bid_replenish_{sec}s_pct"))
            decel = finite(r.get(f"liq_deceleration_{sec}s"))
            spread_chg = finite(r.get(f"spread_change_{sec}s_bps"))
            if bid_repl is not None and decel is not None and spread_chg is not None:
                r[f"dissipation_score_{sec}s"] = bid_repl / 100.0 + decel - max(0.0, spread_chg) / 10.0
        out.append(r)
    return out


def gate_by_cal_cut(cal: list[dict[str, Any]], hold: list[dict[str, Any]], all_rows: list[dict[str, Any]], feature: str, q: float) -> dict[str, Any] | None:
    vals = [finite(r.get(feature)) for r in cal]
    vals = [v for v in vals if v is not None]
    cut = percentile(vals, q)
    if cut is None:
        return None

    def high(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [r for r in rows if (v := finite(r.get(feature))) is not None and v >= float(cut)]

    def low(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [r for r in rows if (v := finite(r.get(feature))) is not None and v < float(cut)]

    hold_hi = metrics(high(hold))
    hold_lo = metrics(low(hold))
    return {
        "feature": feature,
        "q": q,
        "cut": round(float(cut), 4),
        "cal_high": metrics(high(cal)),
        "cal_low": metrics(low(cal)),
        "hold_high": hold_hi,
        "hold_low": hold_lo,
        "all_high": metrics(high(all_rows)),
        "delta_hold_t3r_bps": round(float(hold_hi["t3r_bps"] or 0.0) - float(hold_lo["t3r_bps"] or 0.0), 1),
        "delta_hold_sum_bps": round(float(hold_hi["sum_bps"] or 0.0) - float(hold_lo["sum_bps"] or 0.0), 1),
    }


def by_symbol(hold: list[dict[str, Any]], feature: str, cut: float) -> list[dict[str, Any]]:
    out = []
    for symbol in sorted({str(r["symbol"]) for r in hold}):
        srows = [r for r in hold if str(r["symbol"]) == symbol]
        out.append(
            {
                "symbol": symbol,
                "all": metrics(srows),
                "high": metrics([r for r in srows if (v := finite(r.get(feature))) is not None and v >= cut]),
                "low": metrics([r for r in srows if (v := finite(r.get(feature))) is not None and v < cut]),
            }
        )
    return out


def build_report(payload: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    cal, hold, _ = split_rows(rows, payload)
    features = []
    for sec in HORIZONS_SEC:
        features.extend(
            [
                f"bid_replenish_{sec}s_pct",
                f"total_replenish_{sec}s_pct",
                f"liq_deceleration_{sec}s",
                f"dissipation_score_{sec}s",
            ]
        )
    tests = []
    for feature in features:
        for q in (0.5, 0.75, 0.9):
            row = gate_by_cal_cut(cal, hold, rows, feature, q)
            if row:
                tests.append(row)
    tests.sort(key=lambda r: (r["hold_high"]["t3r_bps"], r["hold_high"]["sum_bps"], r["hold_high"]["n"]), reverse=True)
    best = tests[0] if tests else None
    return {
        "generated_at_utc": utc_now(),
        "source": str(IN_JSON if IN_JSON.exists() else FALLBACK_JSON),
        "discipline": "Post-entry features are NOT legal entry inputs. Use only for management/diagnostics/forward observation.",
        "split": payload.get("split", {}),
        "coverage": {"rows": len(rows), "cal_rows": len(cal), "hold_rows": len(hold)},
        "overall": {"all": metrics(rows), "cal": metrics(cal), "hold": metrics(hold)},
        "tests": tests,
        "best_holdout_feature": best,
        "best_by_symbol_holdout": [] if not best else by_symbol(hold, best["feature"], float(best["cut"])),
        "rows": rows,
    }


def fmt(s: dict[str, Any]) -> str:
    return f"N={s['n']} sum={s['sum_bps']} med={s['median_bps']} T3R={s['t3r_bps']} max_loss={s['max_loss_bps']} tail<-100={s['tail_lt_100']}"


def render(report: dict[str, Any]) -> str:
    lines = [
        "# S34 v3 Energy Dissipation",
        "",
        f"Generated: `{report['generated_at_utc']}`",
        "",
        "Research-only. No live/paper/executor changes.",
        "",
        f"Discipline: {report['discipline']}",
        "",
        "## Overall",
        "",
        f"- All: {fmt(report['overall']['all'])}",
        f"- Calibration: {fmt(report['overall']['cal'])}",
        f"- Holdout: {fmt(report['overall']['hold'])}",
        "",
        "## Ranked Dissipation Tests",
        "",
        "| Rank | Feature | Cal cut | Cal high | Hold high | Hold low | Hold dT3R |",
        "| ---: | --- | ---: | --- | --- | --- | ---: |",
    ]
    for idx, row in enumerate(report["tests"], start=1):
        lines.append(
            f"| {idx} | `{row['feature']}:q{int(row['q']*100)}` | {row['cut']} | "
            f"{fmt(row['cal_high'])} | {fmt(row['hold_high'])} | {fmt(row['hold_low'])} | {row['delta_hold_t3r_bps']} |"
        )
    if report["best_holdout_feature"]:
        best = report["best_holdout_feature"]
        lines += [
            "",
            "## Best Holdout Feature By Symbol",
            "",
            f"Best feature: `{best['feature']}` with calibration q{int(best['q']*100)} cut `{best['cut']}`.",
            "",
            "| Symbol | All | High | Low |",
            "| --- | --- | --- | --- |",
        ]
        for row in report["best_by_symbol_holdout"]:
            lines.append(f"| `{row['symbol']}` | {fmt(row['all'])} | {fmt(row['high'])} | {fmt(row['low'])} |")
    lines += [
        "",
        "## Read",
        "",
        "- A positive dissipation test can become an exit/management observer, not an entry gate, because it is only known after entry.",
        "- Look for high holdout T3R with lower tails and enough N; otherwise it is another in-sample separator.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    source = IN_JSON if IN_JSON.exists() else FALLBACK_JSON
    payload = json.loads(source.read_text(encoding="utf-8"))
    conn = sqlite3.connect(DEFAULT_DB)
    try:
        rows = annotate_rows(conn, list(payload["rows"]), max_book_staleness_sec=10)
    finally:
        conn.close()
    report = build_report(payload, rows)
    OUT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    OUT_MD.write_text(render(report), encoding="utf-8")
    print(render(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
