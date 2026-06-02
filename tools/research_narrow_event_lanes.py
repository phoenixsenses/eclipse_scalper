"""Narrow event-lane alpha scanner over the existing microstructure DB.

This intentionally does not collect new data or touch live execution. It takes
known event families, labels each event with local context lanes, then ranks
which lanes improve forward returns versus the family baseline.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any


def _mark_at(conn: sqlite3.Connection, symbol: str, ts_ms: int, *, before: bool) -> float | None:
    op = "<=" if before else ">="
    order = "DESC" if before else "ASC"
    row = conn.execute(
        f"SELECT mark_price FROM mark_prices WHERE symbol=? AND ts_ms {op} ? ORDER BY ts_ms {order} LIMIT 1",
        (symbol, ts_ms),
    ).fetchone()
    return float(row[0]) if row and row[0] is not None else None


def _book_at(conn: sqlite3.Connection, symbol: str, ts_ms: int) -> dict[str, float] | None:
    row = conn.execute(
        """
        SELECT spread_pct, book_imbalance, bid_depth_usd, ask_qty, bid_qty
        FROM book_ticker
        WHERE symbol=? AND ts_ms<=?
        ORDER BY ts_ms DESC
        LIMIT 1
        """,
        (symbol, ts_ms),
    ).fetchone()
    if not row:
        return None
    return {
        "spread_pct": float(row[0] or 0.0),
        "book_imbalance": float(row[1] or 0.0),
        "bid_depth_usd": float(row[2] or 0.0),
        "ask_qty": float(row[3] or 0.0),
        "bid_qty": float(row[4] or 0.0),
    }


def _side_overlap(
    conn: sqlite3.Connection,
    *,
    symbol: str,
    side: str,
    ts_ms: int,
    window_sec: int,
    min_notional: float,
) -> bool:
    n = conn.execute(
        """
        SELECT COUNT(*) FROM liquidations
        WHERE symbol=? AND side=? AND notional>=? AND ABS(ts_ms - ?) <= ?
        """,
        (symbol, side, float(min_notional), int(ts_ms), int(window_sec * 1000)),
    ).fetchone()[0]
    return int(n or 0) > 0


def _funding_at(conn: sqlite3.Connection, symbol: str, ts_ms: int) -> float | None:
    row = conn.execute(
        """
        SELECT funding_rate FROM mark_prices
        WHERE symbol=? AND ts_ms<=? AND funding_rate IS NOT NULL
        ORDER BY ts_ms DESC LIMIT 1
        """,
        (symbol, ts_ms),
    ).fetchone()
    return float(row[0]) if row and row[0] is not None else None


def _vol_decile_at(conn: sqlite3.Connection, symbol: str, ts_ms: int) -> int | None:
    row = conn.execute(
        """
        SELECT vol_decile FROM vol_state
        WHERE symbol=? AND ts_ms<=?
        ORDER BY ts_ms DESC LIMIT 1
        """,
        (symbol, ts_ms),
    ).fetchone()
    return int(row[0]) if row and row[0] is not None else None


def _ret_bps(conn: sqlite3.Connection, symbol: str, ts_ms: int, direction: str, horizon_sec: int) -> float | None:
    entry = _mark_at(conn, symbol, ts_ms, before=True)
    exit_px = _mark_at(conn, symbol, ts_ms + horizon_sec * 1000, before=False)
    if entry is None or exit_px is None or entry <= 0:
        return None
    raw = (exit_px - entry) / entry * 1e4
    return -raw if direction == "SHORT" else raw


def _time_lanes(ts_ms: int) -> list[str]:
    dt = datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc)
    lanes = [f"utc_hour_{dt.hour:02d}", f"weekday_{dt.weekday()}"]
    if 0 <= dt.hour < 8:
        lanes.append("session_asia")
    elif 8 <= dt.hour < 14:
        lanes.append("session_europe")
    elif 14 <= dt.hour < 21:
        lanes.append("session_us")
    else:
        lanes.append("session_late_us")
    return lanes


def _book_lanes(book: dict[str, float] | None) -> list[str]:
    if not book:
        return ["book_missing"]
    lanes = []
    imb = float(book.get("book_imbalance", 0.0))
    spread = float(book.get("spread_pct", 0.0))
    bid_depth = float(book.get("bid_depth_usd", 0.0))
    if imb >= 0.25:
        lanes.append("book_bid_imbalance")
    elif imb <= -0.25:
        lanes.append("book_ask_imbalance")
    else:
        lanes.append("book_balanced")
    if spread <= 0.0001:
        lanes.append("spread_tight")
    elif spread <= 0.00025:
        lanes.append("spread_mid")
    else:
        lanes.append("spread_wide")
    if bid_depth >= 100000:
        lanes.append("bid_depth_deep")
    elif bid_depth > 0:
        lanes.append("bid_depth_thin")
    return lanes


def _context_lanes(conn: sqlite3.Connection, symbol: str, ts_ms: int, *, include_book: bool) -> list[str]:
    lanes = _time_lanes(ts_ms)
    if include_book:
        lanes.extend(_book_lanes(_book_at(conn, symbol, ts_ms)))

    for other in ("ETHUSDT", "BTCUSDT", "SOLUSDT"):
        if other == symbol:
            continue
        for side in ("BUY", "SELL"):
            if _side_overlap(conn, symbol=other, side=side, ts_ms=ts_ms, window_sec=60, min_notional=100000):
                lanes.append(f"{other.lower()}_{side.lower()}_100k_overlap_60s")

    if _side_overlap(conn, symbol=symbol, side="BUY", ts_ms=ts_ms, window_sec=60, min_notional=100000):
        lanes.append("same_symbol_buy_100k_overlap_60s")
    if _side_overlap(conn, symbol=symbol, side="SELL", ts_ms=ts_ms, window_sec=60, min_notional=100000):
        lanes.append("same_symbol_sell_100k_overlap_60s")

    funding = _funding_at(conn, symbol, ts_ms)
    if funding is not None:
        if funding > 0:
            lanes.append("funding_positive")
        elif funding < 0:
            lanes.append("funding_negative")
        else:
            lanes.append("funding_zero")

    vol_decile = _vol_decile_at(conn, symbol, ts_ms)
    if vol_decile is not None:
        if vol_decile >= 8:
            lanes.append("vol_high")
        elif vol_decile <= 2:
            lanes.append("vol_low")
        else:
            lanes.append("vol_mid")
    return sorted(set(lanes))


def _liq_events(
    conn: sqlite3.Connection,
    *,
    symbol: str,
    side: str,
    threshold: float,
    direction: str,
    horizon_sec: int,
    include_book: bool,
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
    events: list[dict[str, Any]] = []
    for ts_ms, notional in rows:
        rb = _ret_bps(conn, symbol, int(ts_ms), direction, horizon_sec)
        if rb is None:
            continue
        lanes = _context_lanes(conn, symbol, int(ts_ms), include_book=include_book)
        events.append(
            {
                "family": f"{symbol}_{side}_liq_{int(threshold)}_{direction}_{horizon_sec}s",
                "symbol": symbol,
                "ts_ms": int(ts_ms),
                "direction": direction,
                "return_bps": rb,
                "notional": float(notional or 0.0),
                "lanes": lanes,
            }
        )
    return events


def _s34_events(conn: sqlite3.Connection, *, horizon_sec: int, include_book: bool) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT signal_ts_ms, entry_price, liq_composition, basis_at_entry,
               fingerprint_class, entry_book_state, session_tag, confidence_band
        FROM detector_signals
        WHERE symbol='ETHUSDT' AND signal_ts_ms IS NOT NULL AND entry_price IS NOT NULL
        ORDER BY signal_ts_ms ASC
        """
    ).fetchall()
    events: list[dict[str, Any]] = []
    for row in rows:
        ts_ms = int(row[0])
        rb = _ret_bps(conn, "ETHUSDT", ts_ms, "SHORT", horizon_sec)
        if rb is None:
            continue
        comp = str(row[2] or "unknown")
        basis = float(row[3]) if row[3] is not None else None
        fp = str(row[4] or "fp_unknown")
        book_state = str(row[5] or "book_unknown")
        session = str(row[6] or "session_unknown")
        conf = str(row[7] or "confidence_unknown")
        lanes = _context_lanes(conn, "ETHUSDT", ts_ms, include_book=include_book)
        lanes.extend([f"liq_comp_{comp}", f"fingerprint_{fp}", f"entry_book_{book_state}", f"s34_session_{session}", f"confidence_{conf}"])
        if basis is not None:
            lanes.append("basis_positive" if basis > 0 else "basis_nonpositive")
            if basis > 0.5:
                lanes.append("basis_gt_0p5bps")
        events.append(
            {
                "family": f"ETHUSDT_S34_detector_SHORT_{horizon_sec}s",
                "symbol": "ETHUSDT",
                "ts_ms": ts_ms,
                "direction": "SHORT",
                "return_bps": rb,
                "lanes": sorted(set(lanes)),
            }
        )
    return events


def _stats(vals: list[float]) -> dict[str, Any]:
    if not vals:
        return {"n": 0, "wr": None, "mean_bps": None, "median_bps": None}
    return {
        "n": len(vals),
        "wr": 100.0 * sum(1 for v in vals if v > 0) / len(vals),
        "mean_bps": mean(vals),
        "median_bps": median(vals),
    }


def _rank_lanes(events: list[dict[str, Any]], min_n: int) -> list[dict[str, Any]]:
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        by_family[str(event["family"])].append(event)

    rows: list[dict[str, Any]] = []
    for family, fam_events in sorted(by_family.items()):
        base_vals = [float(e["return_bps"]) for e in fam_events]
        base = _stats(base_vals)
        lane_map: dict[str, list[float]] = defaultdict(list)
        for event in fam_events:
            for lane in event.get("lanes", []):
                lane_map[str(lane)].append(float(event["return_bps"]))
        for lane, vals in sorted(lane_map.items()):
            if len(vals) < min_n:
                continue
            st = _stats(vals)
            rows.append(
                {
                    "family": family,
                    "lane": lane,
                    "baseline_n": int(base["n"]),
                    "baseline_wr": base["wr"],
                    "baseline_mean_bps": base["mean_bps"],
                    "n": int(st["n"]),
                    "kept_ratio": len(vals) / len(base_vals) if base_vals else 0.0,
                    "wr": st["wr"],
                    "mean_bps": st["mean_bps"],
                    "median_bps": st["median_bps"],
                    "uplift_bps": float(st["mean_bps"] or 0.0) - float(base["mean_bps"] or 0.0),
                }
            )
    rows.sort(key=lambda r: (float(r["uplift_bps"]), float(r["mean_bps"] or -1e9), int(r["n"])), reverse=True)
    return rows


def _write_md(payload: dict[str, Any], path: Path) -> None:
    lines = [
        "# Narrow Event Lane Alpha Scan",
        "",
        f"- db: `{payload['inputs']['db']}`",
        f"- min_n: `{payload['inputs']['min_n']}`",
        f"- events: `{payload['event_count']}`",
        "",
        "## Baselines",
        "",
        "| family | n | WR | mean_bps | median_bps |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in payload["baselines"]:
        lines.append(
            f"| {row['family']} | {row['n']} | {_fmt(row['wr'])}% | {_fmt(row['mean_bps'])} | {_fmt(row['median_bps'])} |"
        )
    lines.extend(
        [
            "",
            "## Top Lane Uplifts",
            "",
            "| family | lane | n | kept | WR | mean_bps | uplift_bps |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in payload["top_lanes"][:40]:
        lines.append(
            f"| {row['family']} | {row['lane']} | {row['n']} | {row['kept_ratio']*100:.1f}% | "
            f"{_fmt(row['wr'])}% | {_fmt(row['mean_bps'])} | {_fmt(row['uplift_bps'])} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt(x: Any) -> str:
    if x is None:
        return "n/a"
    return f"{float(x):.2f}"


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    conn = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    try:
        events: list[dict[str, Any]] = []
        events.extend(_s34_events(conn, horizon_sec=120, include_book=bool(args.include_book)))
        events.extend(_s34_events(conn, horizon_sec=900, include_book=bool(args.include_book)))
        for symbol in ("BTCUSDT", "ETHUSDT", "SOLUSDT"):
            for side, direction in (("BUY", "SHORT"), ("SELL", "LONG")):
                thresholds = [25000, 50000, 100000] if symbol == "SOLUSDT" else [100000, 250000, 500000, 1000000]
                for threshold in thresholds:
                    events.extend(
                        _liq_events(
                            conn,
                            symbol=symbol,
                            side=side,
                            threshold=float(threshold),
                            direction=direction,
                            horizon_sec=900,
                            include_book=bool(args.include_book),
                            max_events=int(args.max_events_per_family),
                        )
                    )
    finally:
        conn.close()

    by_family: dict[str, list[float]] = defaultdict(list)
    for event in events:
        by_family[str(event["family"])].append(float(event["return_bps"]))
    baselines = [{"family": fam, **_stats(vals)} for fam, vals in sorted(by_family.items())]
    top_lanes = _rank_lanes(events, int(args.min_n))
    promoted = [
        row
        for row in top_lanes
        if int(row["n"]) >= int(args.min_n)
        and float(row["mean_bps"] or 0.0) >= float(args.min_mean_bps)
        and float(row["uplift_bps"] or 0.0) >= float(args.min_uplift_bps)
    ]
    return {
        "inputs": vars(args),
        "event_count": len(events),
        "baseline_count": len(baselines),
        "lane_count": len(top_lanes),
        "candidate_count": len(promoted),
        "baselines": baselines,
        "top_lanes": top_lanes,
        "candidate_lanes": promoted[:25],
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Research narrow event lanes against existing local data only.")
    p.add_argument("--db", default="data/microstructure.db")
    p.add_argument("--min-n", type=int, default=8)
    p.add_argument("--min-mean-bps", type=float, default=8.0)
    p.add_argument("--min-uplift-bps", type=float, default=3.0)
    p.add_argument("--max-events-per-family", type=int, default=500)
    p.add_argument("--include-book", action="store_true")
    p.add_argument("--out-md", default="reports/NARROW_EVENT_LANE_ALPHA_SCAN.md")
    p.add_argument("--out-json", default="reports/NARROW_EVENT_LANE_ALPHA_SCAN.json")
    args = p.parse_args()
    payload = build_payload(args)
    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_md(payload, out_md)
    print(f"Wrote {out_md}")
    print(f"Wrote {out_json}")
    print(f"candidate_lanes={payload['candidate_count']}")


if __name__ == "__main__":
    main()
