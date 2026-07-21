"""Funding-extreme mean-reversion research.

Research-only. Tests whether knowable funding-rate extremes predict forward
returns across BTC/ETH/SOL without touching live or paper state.
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
from statistics import median
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.research_s34_knowable_anchor_continuation import file_fingerprint


DEFAULT_DB = ROOT / "data" / "microstructure.db"
OUT_JSON = ROOT / "reports" / "research" / "s34" / "FUNDING_EXTREME_MEAN_REVERSION.json"
OUT_MD = ROOT / "reports" / "research" / "s34" / "FUNDING_EXTREME_MEAN_REVERSION.md"
SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
HORIZONS_H = (8, 24)
FEE_BPS_SIDE = 3.05


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def iso_ms(ts_ms: int | None) -> str | None:
    if ts_ms is None:
        return None
    return datetime.fromtimestamp(int(ts_ms) / 1000.0, tz=timezone.utc).isoformat()


def finite(value: Any) -> float | None:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def pctile(vals: list[float], q: float) -> float | None:
    xs = sorted(v for v in vals if math.isfinite(v))
    if not xs:
        return None
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * float(q)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def metrics(vals: list[float]) -> dict[str, Any]:
    xs = [float(v) for v in vals if math.isfinite(float(v))]
    if not xs:
        return {
            "n": 0,
            "sum_bps": 0.0,
            "mean_bps": None,
            "median_bps": None,
            "win_rate_pct": None,
            "t3r_bps": 0.0,
            "max_loss_bps": None,
            "tail_lt_50": 0,
            "tail_lt_100": 0,
        }
    ordered = sorted(xs, reverse=True)
    return {
        "n": len(xs),
        "sum_bps": round(sum(xs), 1),
        "mean_bps": round(sum(xs) / len(xs), 2),
        "median_bps": round(median(xs), 2),
        "win_rate_pct": round(100.0 * sum(1 for v in xs if v > 0.0) / len(xs), 1),
        "t3r_bps": round(sum(ordered[3:]) if len(ordered) > 3 else sum(xs), 1),
        "max_loss_bps": round(min(xs), 1),
        "tail_lt_50": sum(1 for v in xs if v < -50.0),
        "tail_lt_100": sum(1 for v in xs if v < -100.0),
    }


def table_columns(conn: sqlite3.Connection, table: str) -> list[str]:
    try:
        return [str(r[1]) for r in conn.execute(f"PRAGMA table_info({table})").fetchall()]
    except sqlite3.Error:
        return []


def table_symbol_coverage(conn: sqlite3.Connection, table: str) -> dict[str, Any]:
    cols = table_columns(conn, table)
    if not cols:
        return {"present": False, "columns": [], "symbols": {}}
    symbol_col = "symbol" if "symbol" in cols else None
    ts_col = "ts_ms" if "ts_ms" in cols else None
    out: dict[str, Any] = {"present": True, "columns": cols, "symbols": {}}
    if not symbol_col or not ts_col:
        return out
    for sym in SYMBOLS:
        row = conn.execute(
            f"SELECT COUNT(*), MIN({ts_col}), MAX({ts_col}) FROM {table} WHERE {symbol_col}=?",
            (sym,),
        ).fetchone()
        out["symbols"][sym] = {
            "rows": int(row[0] or 0),
            "start": iso_ms(row[1]) if row and row[1] is not None else None,
            "end": iso_ms(row[2]) if row and row[2] is not None else None,
        }
    return out


class PriceIndex:
    def __init__(self, rows: list[tuple[int, float]]) -> None:
        data = [(int(ts), float(px)) for ts, px in rows if finite(px) is not None and float(px) > 0]
        self.ts = [x[0] for x in data]
        self.px = [x[1] for x in data]

    def at_or_after(self, ts_ms: int) -> tuple[int, float] | None:
        idx = bisect.bisect_left(self.ts, int(ts_ms))
        if idx >= len(self.ts):
            return None
        return self.ts[idx], self.px[idx]


def load_price_index(conn: sqlite3.Connection, symbol: str) -> PriceIndex:
    rows = conn.execute(
        "SELECT ts_ms, mark_price FROM mark_prices WHERE symbol=? ORDER BY ts_ms",
        (symbol,),
    ).fetchall()
    return PriceIndex([(int(r[0]), float(r[1])) for r in rows])


def sample_funding_rows(
    conn: sqlite3.Connection,
    symbol: str,
    *,
    step_sec: int,
    lookback: int,
) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT ts_ms, mark_price, funding_rate
        FROM mark_prices
        WHERE symbol=? AND funding_rate IS NOT NULL AND mark_price IS NOT NULL
        ORDER BY ts_ms
        """,
        (symbol,),
    ).fetchall()
    out: list[dict[str, Any]] = []
    last_keep = -10**30
    hist: list[float] = []
    for ts_ms, mark_price, funding_rate in rows:
        ts = int(ts_ms)
        fr = finite(funding_rate)
        px = finite(mark_price)
        if fr is None or px is None or px <= 0:
            continue
        if ts - last_keep < int(step_sec) * 1000:
            hist.append(fr)
            hist = hist[-lookback:]
            continue
        if len(hist) >= max(20, min(lookback, 20)):
            mu = sum(hist) / len(hist)
            var = sum((x - mu) ** 2 for x in hist) / len(hist)
            sd = math.sqrt(var)
            z = (fr - mu) / sd if sd > 0 else 0.0
            out.append(
                {
                    "symbol": symbol,
                    "ts_ms": ts,
                    "month": datetime.fromtimestamp(ts / 1000, tz=timezone.utc).strftime("%Y-%m"),
                    "mark_price": px,
                    "funding_rate": fr,
                    "funding_z": z,
                    "hist_n": len(hist),
                }
            )
            last_keep = ts
        hist.append(fr)
        hist = hist[-lookback:]
    return out


def add_forward_returns(rows: list[dict[str, Any]], px_index: PriceIndex, horizons_h: tuple[int, ...]) -> None:
    for row in rows:
        entry = px_index.at_or_after(int(row["ts_ms"]))
        if not entry:
            continue
        entry_ts, entry_px = entry
        row["entry_ts_ms"] = entry_ts
        row["entry_price"] = entry_px
        for h in horizons_h:
            exit_row = px_index.at_or_after(int(row["ts_ms"]) + int(h) * 3600 * 1000)
            if not exit_row:
                continue
            _, exit_px = exit_row
            raw_long = (exit_px - entry_px) / entry_px * 10_000.0
            # Mean-reversion direction: positive funding -> SHORT, negative funding -> LONG.
            direction = "SHORT" if float(row["funding_rate"]) > 0 else "LONG"
            gross = -raw_long if direction == "SHORT" else raw_long
            row[f"mr_{h}h_net_bps"] = gross - 2.0 * FEE_BPS_SIDE
            row[f"long_{h}h_gross_bps"] = raw_long
            row[f"direction_{h}h"] = direction


def chronological_split(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str], list[str]]:
    months = sorted({str(r["month"]) for r in rows})
    if len(months) <= 1:
        return rows, [], months, []
    hold_n = max(1, len(months) // 3)
    hold_months = months[-hold_n:]
    cal_months = months[:-hold_n]
    cal = [r for r in rows if str(r["month"]) in cal_months]
    hold = [r for r in rows if str(r["month"]) in hold_months]
    return cal, hold, cal_months, hold_months


def group_metrics(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    return metrics([float(r[key]) for r in rows if finite(r.get(key)) is not None])


def evaluate(rows: list[dict[str, Any]], *, horizon_h: int, z_cut: float) -> dict[str, Any]:
    key = f"mr_{horizon_h}h_net_bps"
    eligible = [r for r in rows if finite(r.get(key)) is not None and abs(float(r["funding_z"])) >= z_cut]
    pos = [r for r in eligible if float(r["funding_z"]) >= z_cut]
    neg = [r for r in eligible if float(r["funding_z"]) <= -z_cut]
    by_symbol = {
        sym: group_metrics([r for r in eligible if r["symbol"] == sym], key)
        for sym in SYMBOLS
    }
    return {
        "n": len(eligible),
        "all": group_metrics(eligible, key),
        "positive_funding_short": group_metrics(pos, key),
        "negative_funding_long": group_metrics(neg, key),
        "by_symbol": by_symbol,
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    with sqlite3.connect(f"file:{args.db}?mode=ro", uri=True) as conn:
        coverage = {
            t: table_symbol_coverage(conn, t)
            for t in ("mark_prices", "funding_rates", "open_interest", "spot_prices", "basis_reversion_candidates")
        }
        all_rows: list[dict[str, Any]] = []
        for sym in SYMBOLS:
            rows = sample_funding_rows(conn, sym, step_sec=args.step_sec, lookback=args.z_lookback)
            px = load_price_index(conn, sym)
            add_forward_returns(rows, px, tuple(args.horizons_h))
            all_rows.extend(rows)
    cal, hold, cal_months, hold_months = chronological_split(all_rows)
    configs = []
    for h in args.horizons_h:
        for z in args.z_cuts:
            configs.append(
                {
                    "config_id": f"funding_abs_z_ge_{z:g}_mr_{h}h",
                    "horizon_h": h,
                    "z_cut": z,
                    "all": evaluate(all_rows, horizon_h=h, z_cut=z),
                    "cal": evaluate(cal, horizon_h=h, z_cut=z),
                    "hold": evaluate(hold, horizon_h=h, z_cut=z),
                }
            )
    ranked = sorted(
        configs,
        key=lambda c: (
            float(c["hold"]["all"]["t3r_bps"] or 0.0),
            float(c["hold"]["all"]["sum_bps"] or 0.0),
            float(c["cal"]["all"]["sum_bps"] or 0.0),
        ),
        reverse=True,
    )
    consistent = [
        c
        for c in configs
        if c["cal"]["all"]["n"] >= args.min_n
        and c["hold"]["all"]["n"] >= args.min_n
        and float(c["cal"]["all"]["sum_bps"] or 0.0) > 0
        and float(c["hold"]["all"]["sum_bps"] or 0.0) > 0
        and float(c["cal"]["all"]["t3r_bps"] or 0.0) > 0
        and float(c["hold"]["all"]["t3r_bps"] or 0.0) > 0
    ]
    return {
        "generated_at_utc": utc_now(),
        "mode": "RESEARCH_ONLY_NO_LIVE_NO_PAPER",
        "source_db": file_fingerprint(args.db),
        "sampling": {
            "step_sec": args.step_sec,
            "z_lookback": args.z_lookback,
            "fee_bps_side": FEE_BPS_SIDE,
            "symbols": list(SYMBOLS),
            "horizons_h": list(args.horizons_h),
            "z_cuts": list(args.z_cuts),
            "min_n_gate": args.min_n,
        },
        "coverage": coverage,
        "split": {
            "cal_months": cal_months,
            "hold_months": hold_months,
            "rows": len(all_rows),
            "cal_rows": len(cal),
            "hold_rows": len(hold),
        },
        "consistent_passes": consistent,
        "ranked_configs": ranked,
    }


def fmt(m: dict[str, Any]) -> str:
    return (
        f"N={m['n']} sum={m['sum_bps']} med={m['median_bps']} "
        f"T3R={m['t3r_bps']} WR={m['win_rate_pct']} max_loss={m['max_loss_bps']}"
    )


def render_md(report: dict[str, Any]) -> str:
    lines = [
        "# Funding Extreme Mean-Reversion Research",
        "",
        f"Generated: `{report['generated_at_utc']}`",
        "",
        "`RESEARCH_ONLY_NO_LIVE_NO_PAPER` - no live or paper state was touched.",
        "",
        "## Coverage",
        "",
        f"- split: `{report['split']}`",
        f"- sampling: `{report['sampling']}`",
        "",
        "| Table | Symbol | Rows | Start | End |",
        "| --- | --- | ---: | --- | --- |",
    ]
    for table, info in report["coverage"].items():
        for sym, cov in (info.get("symbols") or {}).items():
            lines.append(f"| `{table}` | `{sym}` | {cov['rows']} | {cov['start']} | {cov['end']} |")
    lines.extend(
        [
            "",
            "## Consistent Passes",
            "",
            f"Configs with cal+hold N gate and positive sum/T3R in both splits: `{len(report['consistent_passes'])}`",
            "",
            "| Config | Cal | Hold |",
            "| --- | --- | --- |",
        ]
    )
    for c in report["consistent_passes"]:
        lines.append(f"| `{c['config_id']}` | {fmt(c['cal']['all'])} | {fmt(c['hold']['all'])} |")
    lines.extend(
        [
            "",
            "## Ranked Configs",
            "",
            "| Rank | Config | Cal | Hold | Positive funding -> SHORT hold | Negative funding -> LONG hold |",
            "| ---: | --- | --- | --- | --- | --- |",
        ]
    )
    for i, c in enumerate(report["ranked_configs"], 1):
        lines.append(
            f"| {i} | `{c['config_id']}` | {fmt(c['cal']['all'])} | {fmt(c['hold']['all'])} | "
            f"{fmt(c['hold']['positive_funding_short'])} | {fmt(c['hold']['negative_funding_long'])} |"
        )
    best = report["ranked_configs"][0] if report["ranked_configs"] else None
    if best:
        lines.extend(
            [
                "",
                "## Best Holdout By Symbol",
                "",
                f"Config: `{best['config_id']}`",
                "",
                "| Symbol | Hold metrics |",
                "| --- | --- |",
            ]
        )
        for sym, met in best["hold"]["by_symbol"].items():
            lines.append(f"| `{sym}` | {fmt(met)} |")
    lines.extend(
        [
            "",
            "## Read",
            "",
            "- This is a first-pass fresh-signal test, not a promotion.",
            "- Funding is knowable at the snapshot timestamp; forward returns are labels only.",
            "- A pass requires cal+hold consistency, N gate, total/T3R positivity, and later forward shadow.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_float_tuple(text: str) -> tuple[float, ...]:
    return tuple(float(x.strip()) for x in str(text).split(",") if x.strip())


def parse_int_tuple(text: str) -> tuple[int, ...]:
    return tuple(int(x.strip()) for x in str(text).split(",") if x.strip())


def main() -> int:
    p = argparse.ArgumentParser(description="Research funding-rate extreme mean reversion.")
    p.add_argument("--db", type=Path, default=DEFAULT_DB)
    p.add_argument("--out-json", type=Path, default=OUT_JSON)
    p.add_argument("--out-md", type=Path, default=OUT_MD)
    p.add_argument("--step-sec", type=int, default=3600)
    p.add_argument("--z-lookback", type=int, default=72)
    p.add_argument("--horizons-h", type=parse_int_tuple, default=HORIZONS_H)
    p.add_argument("--z-cuts", type=parse_float_tuple, default=(1.0, 1.5, 2.0))
    p.add_argument("--min-n", type=int, default=40)
    args = p.parse_args()
    report = build_report(args)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    md = render_md(report)
    args.out_md.write_text(md, encoding="utf-8")
    print(md)
    print(f"Wrote {args.out_json}")
    print(f"Wrote {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
