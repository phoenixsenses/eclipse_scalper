"""Unified forced-flow candidate harness.

Runs liquidation-side directional tests with chronological walk-forward folds,
fee sensitivity, and simple execution modes.

Examples:
  python tools/forced_flow_candidate_harness.py --symbol SOLUSDT --liq-side BUY --direction SHORT --thresholds 25000,50000,100000 --horizons 300,900
  python tools/forced_flow_candidate_harness.py --symbol ETHUSDT --liq-side BUY --direction SHORT --thresholds 200000,500000 --horizons 120,900
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from statistics import mean, median
from typing import Iterable

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def _parse_csv_float(raw: str) -> list[float]:
    return [float(x.strip()) for x in str(raw or "").split(",") if x.strip()]


def _parse_csv_int(raw: str) -> list[int]:
    return [int(float(x.strip())) for x in str(raw or "").split(",") if x.strip()]


def _wr(rets: Iterable[float]) -> float | None:
    vals = list(rets)
    if not vals:
        return None
    return 100.0 * sum(1 for x in vals if x > 0) / len(vals)


def _fmt(x: object, digits: int = 2) -> str:
    if x is None:
        return "n/a"
    if isinstance(x, float):
        return f"{x:.{digits}f}"
    return str(x)


def _mark_at(conn: sqlite3.Connection, symbol: str, ts_ms: int, *, before: bool) -> float | None:
    op = "<=" if before else ">="
    order = "DESC" if before else "ASC"
    row = conn.execute(
        f"SELECT mark_price FROM mark_prices WHERE symbol=? AND ts_ms {op} ? ORDER BY ts_ms {order} LIMIT 1",
        (symbol, ts_ms),
    ).fetchone()
    return float(row[0]) if row and row[0] is not None else None


def _trade_fill(
    conn: sqlite3.Connection,
    symbol: str,
    ts_ms: int,
    entry_px: float,
    direction: str,
    wait_ms: int,
) -> tuple[int, float] | None:
    if direction == "SHORT":
        row = conn.execute(
            """
            SELECT ts_ms, price FROM agg_trades
            WHERE symbol=? AND ts_ms BETWEEN ? AND ? AND price >= ?
            ORDER BY ts_ms ASC LIMIT 1
            """,
            (symbol, ts_ms, ts_ms + wait_ms, entry_px),
        ).fetchone()
    else:
        row = conn.execute(
            """
            SELECT ts_ms, price FROM agg_trades
            WHERE symbol=? AND ts_ms BETWEEN ? AND ? AND price <= ?
            ORDER BY ts_ms ASC LIMIT 1
            """,
            (symbol, ts_ms, ts_ms + wait_ms, entry_px),
        ).fetchone()
    if not row:
        return None
    return int(row[0]), float(row[1])


def _event_return(
    conn: sqlite3.Connection,
    *,
    symbol: str,
    ts_ms: int,
    direction: str,
    horizon_sec: int,
    exec_model: str,
    passive_wait_sec: int,
) -> float | None:
    entry = _mark_at(conn, symbol, ts_ms, before=True)
    if entry is None or entry <= 0:
        return None
    fill_ts = ts_ms
    fill_px = entry

    if exec_model in {"passive", "passive_then_taker"}:
        # Conservative one-tick-free proxy: try to fill at current mark or better.
        fill = _trade_fill(conn, symbol, ts_ms, entry, direction, passive_wait_sec * 1000)
        if fill:
            fill_ts, fill_px = fill
        elif exec_model == "passive":
            return None
        else:
            fill_ts = ts_ms + passive_wait_sec * 1000
            fill_px = _mark_at(conn, symbol, fill_ts, before=False) or entry

    exit_px = _mark_at(conn, symbol, fill_ts + horizon_sec * 1000, before=False)
    if exit_px is None or fill_px <= 0:
        return None
    raw = (exit_px - fill_px) / fill_px
    return -raw if direction == "SHORT" else raw


def _fold_stats(rets: list[float], folds: int) -> list[dict]:
    if not rets:
        return []
    folds = max(1, min(folds, len(rets)))
    out = []
    for i in range(folds):
        lo = int(i * len(rets) / folds)
        hi = int((i + 1) * len(rets) / folds)
        sub = rets[lo:hi]
        out.append(
            {
                "fold": i + 1,
                "n": len(sub),
                "wr": _wr(sub),
                "mean_bps": mean(sub) * 1e4 if sub else None,
                "median_bps": median(sub) * 1e4 if sub else None,
            }
        )
    return out


def run(args: argparse.Namespace) -> dict:
    conn = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    symbol = args.symbol.upper()
    liq_side = args.liq_side.upper()
    direction = args.direction.upper()
    thresholds = _parse_csv_float(args.thresholds)
    horizons = _parse_csv_int(args.horizons)
    fees = _parse_csv_float(args.fee_rt_bps)

    results = []
    for threshold in thresholds:
        rows = conn.execute(
            """
            SELECT ts_ms, notional FROM liquidations
            WHERE symbol=? AND side=? AND notional>=?
            ORDER BY ts_ms ASC
            """,
            (symbol, liq_side, threshold),
        ).fetchall()
        for horizon in horizons:
            gross = []
            filled = 0
            for ts_ms, _notional in rows:
                ret = _event_return(
                    conn,
                    symbol=symbol,
                    ts_ms=int(ts_ms),
                    direction=direction,
                    horizon_sec=horizon,
                    exec_model=args.exec_model,
                    passive_wait_sec=int(args.passive_wait_sec),
                )
                if ret is None:
                    continue
                filled += 1
                gross.append(float(ret))
            gross_bps = [x * 1e4 for x in gross]
            row = {
                "symbol": symbol,
                "liq_side": liq_side,
                "direction": direction,
                "threshold": threshold,
                "horizon_sec": horizon,
                "exec_model": args.exec_model,
                "events": len(rows),
                "filled": filled,
                "fill_rate": filled / len(rows) if rows else None,
                "gross_wr": _wr(gross),
                "gross_mean_bps": mean(gross_bps) if gross_bps else None,
                "gross_median_bps": median(gross_bps) if gross_bps else None,
                "folds": _fold_stats(gross, int(args.folds)),
                "fees": {},
            }
            for fee in fees:
                net_bps = [x - fee for x in gross_bps]
                row["fees"][str(fee)] = {
                    "net_mean_bps": mean(net_bps) if net_bps else None,
                    "net_median_bps": median(net_bps) if net_bps else None,
                    "net_wr": _wr([x / 1e4 for x in net_bps]),
                    "folds_positive": sum(
                        1 for f in row["folds"] if f["mean_bps"] is not None and float(f["mean_bps"]) - fee > 0
                    ),
                }
            results.append(row)
    conn.close()

    qualifying = [r for r in results if int(r["filled"]) >= int(args.min_n)]
    best = max(
        qualifying,
        key=lambda r: (
            float(r["gross_mean_bps"] or -1e9),
            float(r["gross_wr"] or 0.0),
            int(r["filled"]),
        ),
        default=None,
    )
    verdict = "NO_PROMOTION"
    if best and float(best["gross_mean_bps"] or 0.0) >= float(args.min_mean_bps) and float(best["gross_wr"] or 0.0) >= float(args.min_wr):
        verdict = "SHADOW_CANDIDATE"
    return {
        "inputs": vars(args),
        "results": results,
        "best": best,
        "verdict": verdict,
    }


def write_report(payload: dict, out_md: Path, out_json: Path) -> None:
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = ["# Forced Flow Candidate Harness", "", f"- verdict: `{payload['verdict']}`", ""]
    lines.append("## Best")
    lines.append("")
    lines.append(f"`{payload.get('best')}`")
    lines.append("")
    lines.append("## Results")
    lines.append("")
    lines.append("| symbol | side | dir | threshold | h | exec | events | filled | fill_rate | WR | mean_bps | median_bps |")
    lines.append("|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|")
    for r in payload["results"]:
        lines.append(
            f"| {r['symbol']} | {r['liq_side']} | {r['direction']} | {int(r['threshold'])} | {r['horizon_sec']} | "
            f"{r['exec_model']} | {r['events']} | {r['filled']} | {_fmt((r['fill_rate'] or 0)*100)}% | "
            f"{_fmt(r['gross_wr'])}% | {_fmt(r['gross_mean_bps'])} | {_fmt(r['gross_median_bps'])} |"
        )
    out_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description="Unified forced-flow candidate harness")
    p.add_argument("--db", default="data/microstructure.db")
    p.add_argument("--symbol", required=True)
    p.add_argument("--liq-side", required=True, choices=["BUY", "SELL"])
    p.add_argument("--direction", required=True, choices=["LONG", "SHORT"])
    p.add_argument("--thresholds", default="25000,50000,100000,200000")
    p.add_argument("--horizons", default="60,300,900")
    p.add_argument("--exec-model", choices=["taker", "passive", "passive_then_taker"], default="taker")
    p.add_argument("--passive-wait-sec", type=int, default=10)
    p.add_argument("--fee-rt-bps", default="2,4,8,10")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--min-n", type=int, default=20)
    p.add_argument("--min-wr", type=float, default=60.0)
    p.add_argument("--min-mean-bps", type=float, default=5.0)
    p.add_argument("--out-md", default="reports/FORCED_FLOW_CANDIDATE.md")
    p.add_argument("--out-json", default="reports/FORCED_FLOW_CANDIDATE.json")
    args = p.parse_args()
    payload = run(args)
    write_report(payload, Path(args.out_md), Path(args.out_json))
    print(f"Wrote {args.out_md}")
    print(f"Wrote {args.out_json}")
    print(f"Verdict: {payload['verdict']}")
    print(f"Best: {payload.get('best')}")


if __name__ == "__main__":
    main()
