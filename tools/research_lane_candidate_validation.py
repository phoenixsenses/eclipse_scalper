"""Targeted validation for lane-conditioned alpha candidates.

Uses only the existing SQLite DB. The output is meant to separate promising
lane-conditioned edges from small-sample uplift artifacts.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.research_narrow_event_lanes import _context_lanes, _ret_bps


def _wr(vals: list[float]) -> float | None:
    if not vals:
        return None
    return 100.0 * sum(1 for v in vals if v > 0) / len(vals)


def _stats(vals: list[float]) -> dict[str, Any]:
    if not vals:
        return {"n": 0, "wr": None, "mean_bps": None, "median_bps": None}
    return {"n": len(vals), "wr": _wr(vals), "mean_bps": mean(vals), "median_bps": median(vals)}


def _folds(vals: list[float], folds: int = 5) -> list[dict[str, Any]]:
    if not vals:
        return []
    folds = max(1, min(int(folds), len(vals)))
    out = []
    for i in range(folds):
        lo = int(i * len(vals) / folds)
        hi = int((i + 1) * len(vals) / folds)
        sub = vals[lo:hi]
        st = _stats(sub)
        out.append({"fold": i + 1, **st})
    return out


def _hour(ts_ms: int) -> int:
    return datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc).hour


def _session_us(ts_ms: int) -> bool:
    return 14 <= _hour(ts_ms) < 21


def _funding_negative(lanes: list[str]) -> bool:
    return "funding_negative" in lanes


def _has_lane(name: str) -> Callable[[list[str], int], bool]:
    return lambda lanes, _ts: name in lanes


def _hour_is(hour: int) -> Callable[[list[str], int], bool]:
    return lambda _lanes, ts: _hour(ts) == hour


def _session_us_filter(_lanes: list[str], ts_ms: int) -> bool:
    return _session_us(ts_ms)


def _load_liq(
    conn: sqlite3.Connection,
    *,
    symbol: str,
    side: str,
    threshold: float,
    direction: str,
    horizon_sec: int,
    max_events: int,
) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT ts_ms, notional FROM liquidations
        WHERE symbol=? AND side=? AND notional>=?
        ORDER BY ts_ms DESC
        LIMIT ?
        """,
        (symbol, side, float(threshold), int(max_events)),
    ).fetchall()
    rows = list(reversed(rows))
    out = []
    for ts_ms, notional in rows:
        ts = int(ts_ms)
        rb = _ret_bps(conn, symbol, ts, direction, horizon_sec)
        if rb is None:
            continue
        out.append({"ts_ms": ts, "return_bps": rb, "notional": float(notional or 0.0)})
    return out


def _load_s34(conn: sqlite3.Connection, *, horizon_sec: int) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT signal_ts_ms, basis_at_entry, liq_composition, confidence_band, session_tag
        FROM detector_signals
        WHERE symbol='ETHUSDT' AND signal_ts_ms IS NOT NULL AND entry_price IS NOT NULL
        ORDER BY signal_ts_ms ASC
        """
    ).fetchall()
    out = []
    for ts_ms, basis, comp, conf, session in rows:
        ts = int(ts_ms)
        rb = _ret_bps(conn, "ETHUSDT", ts, "SHORT", horizon_sec)
        if rb is None:
            continue
        out.append(
            {
                "ts_ms": ts,
                "return_bps": rb,
                "basis_at_entry": float(basis) if basis is not None else None,
                "liq_composition": str(comp or ""),
                "confidence_band": str(conf or ""),
                "session_tag": str(session or ""),
            }
        )
    return out


def _score_candidate(
    conn: sqlite3.Connection,
    *,
    name: str,
    rows: list[dict[str, Any]],
    symbol: str,
    predicate: Callable[[list[str], int, dict[str, Any]], bool],
    include_book: bool,
    fees: list[float],
) -> dict[str, Any]:
    filtered = []
    for row in rows:
        ts = int(row["ts_ms"])
        lanes = _context_lanes(conn, symbol, ts, include_book=include_book)
        if predicate(lanes, ts, row):
            filtered.append(row)
    vals = [float(r["return_bps"]) for r in rows]
    fvals = [float(r["return_bps"]) for r in filtered]
    base = _stats(vals)
    filt = _stats(fvals)
    fee_stats = {}
    for fee in fees:
        net = [v - float(fee) for v in fvals]
        fee_stats[str(float(fee))] = {
            "net_mean_bps": mean(net) if net else None,
            "net_wr": _wr(net),
            "folds_positive": sum(1 for f in _folds(net) if f["mean_bps"] is not None and float(f["mean_bps"]) > 0),
        }
    verdict = "REJECT"
    if int(filt["n"]) >= 20 and float(filt["mean_bps"] or 0.0) >= 8.0:
        positive = sum(1 for f in _folds(fvals) if f["mean_bps"] is not None and float(f["mean_bps"]) > 0)
        verdict = "SHADOW_TEST" if positive >= 4 else "WATCH_ONLY"
    return {
        "name": name,
        "baseline": base,
        "filtered": filt,
        "kept_ratio": (int(filt["n"]) / int(base["n"])) if int(base["n"]) else 0.0,
        "uplift_bps": float(filt["mean_bps"] or 0.0) - float(base["mean_bps"] or 0.0),
        "folds": _folds(fvals),
        "fees": fee_stats,
        "verdict": verdict,
    }


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    conn = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    fees = [float(x.strip()) for x in str(args.fee_rt_bps).split(",") if x.strip()]
    try:
        specs = []
        eth_buy_500 = _load_liq(conn, symbol="ETHUSDT", side="BUY", threshold=500000, direction="SHORT", horizon_sec=900, max_events=args.max_events)
        eth_buy_250 = _load_liq(conn, symbol="ETHUSDT", side="BUY", threshold=250000, direction="SHORT", horizon_sec=900, max_events=args.max_events)
        sol_buy_50 = _load_liq(conn, symbol="SOLUSDT", side="BUY", threshold=50000, direction="SHORT", horizon_sec=900, max_events=args.max_events)
        btc_sell_100 = _load_liq(conn, symbol="BTCUSDT", side="SELL", threshold=100000, direction="LONG", horizon_sec=900, max_events=args.max_events)
        s34_900 = _load_s34(conn, horizon_sec=900)

        specs.append(("ETH_BUY500K_SHORT_900_SESSION_US", eth_buy_500, "ETHUSDT", lambda lanes, ts, row: _session_us_filter(lanes, ts)))
        specs.append(("ETH_BUY500K_SHORT_900_UTC14", eth_buy_500, "ETHUSDT", lambda lanes, ts, row: _hour_is(14)(lanes, ts)))
        specs.append(("ETH_BUY250K_SHORT_900_UTC14", eth_buy_250, "ETHUSDT", lambda lanes, ts, row: _hour_is(14)(lanes, ts)))
        specs.append(("S34_SHORT_900_BASIS_POSITIVE", s34_900, "ETHUSDT", lambda lanes, ts, row: (row.get("basis_at_entry") is not None and float(row["basis_at_entry"]) > 0)))
        specs.append(("S34_SHORT_900_SESSION_US", s34_900, "ETHUSDT", lambda lanes, ts, row: _session_us_filter(lanes, ts)))
        specs.append(("S34_SHORT_900_SINGLE_LARGE", s34_900, "ETHUSDT", lambda lanes, ts, row: str(row.get("liq_composition")) == "single_large"))
        specs.append(("SOL_BUY50K_SHORT_900_FUNDING_NEGATIVE", sol_buy_50, "SOLUSDT", lambda lanes, ts, row: _funding_negative(lanes)))
        specs.append(("BTC_SELL100K_LONG_900_UTC13", btc_sell_100, "BTCUSDT", lambda lanes, ts, row: _hour_is(13)(lanes, ts)))

        rows = [
            _score_candidate(
                conn,
                name=name,
                rows=event_rows,
                symbol=symbol,
                predicate=pred,
                include_book=bool(args.include_book),
                fees=fees,
            )
            for name, event_rows, symbol, pred in specs
        ]
    finally:
        conn.close()
    rows.sort(key=lambda r: (r["verdict"] == "SHADOW_TEST", float(r["filtered"]["mean_bps"] or -1e9), int(r["filtered"]["n"])), reverse=True)
    return {"inputs": vars(args), "rows": rows}


def _fmt(x: Any) -> str:
    if x is None:
        return "n/a"
    return f"{float(x):.2f}"


def write_md(payload: dict[str, Any], path: Path) -> None:
    lines = [
        "# Lane Candidate Validation",
        "",
        "| candidate | verdict | n | WR | mean_bps | uplift_bps | folds_pos | net8_mean |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["rows"]:
        folds_pos = sum(1 for f in row["folds"] if f["mean_bps"] is not None and float(f["mean_bps"]) > 0)
        net8 = row["fees"].get("8.0", {}).get("net_mean_bps")
        lines.append(
            f"| {row['name']} | {row['verdict']} | {row['filtered']['n']} | {_fmt(row['filtered']['wr'])}% | "
            f"{_fmt(row['filtered']['mean_bps'])} | {_fmt(row['uplift_bps'])} | {folds_pos}/5 | {_fmt(net8)} |"
        )
    lines.append("")
    lines.append("## Fold Detail")
    for row in payload["rows"]:
        lines.append("")
        lines.append(f"### {row['name']}")
        lines.append("| fold | n | WR | mean_bps |")
        lines.append("|---:|---:|---:|---:|")
        for f in row["folds"]:
            lines.append(f"| {f['fold']} | {f['n']} | {_fmt(f['wr'])}% | {_fmt(f['mean_bps'])} |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description="Validate targeted lane-conditioned candidates.")
    p.add_argument("--db", default="data/microstructure.db")
    p.add_argument("--max-events", type=int, default=500)
    p.add_argument("--fee-rt-bps", default="2,4,8,10")
    p.add_argument("--include-book", action="store_true")
    p.add_argument("--out-md", default="reports/LANE_CANDIDATE_VALIDATION.md")
    p.add_argument("--out-json", default="reports/LANE_CANDIDATE_VALIDATION.json")
    args = p.parse_args()
    payload = build_payload(args)
    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_md(payload, out_md)
    print(f"Wrote {out_md}")
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()
