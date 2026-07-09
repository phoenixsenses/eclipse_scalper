"""Non-predictive value tests: funding carry and direction-agnostic provision.

Research-only. This explicitly avoids directional alpha mining:

1. Funding carry harvest: hold the side that receives funding for one 8h slot.
2. Direction-agnostic maker provision: quote both sides around cascade anchors and
   measure spread/rebate vs adverse inventory markout.

No live, paper, runtime, or executor state is touched.
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

from tools.research_funding_nonoverlap import SLOT_SEC, funding_at
from tools.research_s34_knowable_anchor_continuation import load_mark_index, signed_return_bps

from ami.storage import production as PR
from ami.storage import research_reader as RR

DEFAULT_DB = ROOT / "data" / "microstructure.db"
POOL_JSON = ROOT / "reports" / "research" / "s34" / "S34_ABSORPTION_SYNC_2X2_POOL.json"
OUT_JSON = ROOT / "reports" / "research" / "s34" / "NONPREDICTIVE_CARRY_PROVISION.json"
OUT_MD = ROOT / "reports" / "research" / "s34" / "NONPREDICTIVE_CARRY_PROVISION.md"
SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def iso_month(ts_ms: int) -> str:
    return datetime.fromtimestamp(int(ts_ms) / 1000.0, tz=timezone.utc).strftime("%Y-%m")


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


def r1(value: float | None) -> float | None:
    return round(float(value), 1) if value is not None and math.isfinite(float(value)) else None


def r3(value: float | None) -> float | None:
    return round(float(value), 3) if value is not None and math.isfinite(float(value)) else None


def metrics(vals: list[float]) -> dict[str, Any]:
    xs = [float(v) for v in vals if math.isfinite(float(v))]
    if not xs:
        return {
            "n": 0,
            "sum_bps": 0.0,
            "mean_bps": None,
            "median_bps": None,
            "win_rate": None,
            "t3r_bps": 0.0,
            "max_loss_bps": None,
            "tail_lt_50": 0,
            "tail_lt_100": 0,
        }
    ordered = sorted(xs, reverse=True)
    return {
        "n": len(xs),
        "sum_bps": r1(sum(xs)),
        "mean_bps": r1(sum(xs) / len(xs)),
        "median_bps": r1(median(xs)),
        "win_rate": r3(sum(1 for v in xs if v > 0.0) / len(xs)),
        "t3r_bps": r1(sum(ordered[3:]) if len(ordered) > 3 else sum(xs)),
        "max_loss_bps": r1(min(xs)),
        "tail_lt_50": sum(1 for v in xs if v < -50.0),
        "tail_lt_100": sum(1 for v in xs if v < -100.0),
    }


def split_months(rows: list[dict[str, Any]]) -> tuple[set[str], dict[str, Any]]:
    months = sorted({str(r["month"]) for r in rows})
    hold_n = max(1, len(months) // 3) if months else 0
    hold_months = set(months[-hold_n:]) if hold_n else set()
    return hold_months, {"method": "chronological_month_tail", "months": months, "holdout_months": sorted(hold_months)}


def summarize(rows: list[dict[str, Any]], value_key: str, hold_months: set[str]) -> dict[str, Any]:
    return {
        "all": metrics([float(r[value_key]) for r in rows if finite(r.get(value_key)) is not None]),
        "cal": metrics([float(r[value_key]) for r in rows if r.get("month") not in hold_months and finite(r.get(value_key)) is not None]),
        "hold": metrics([float(r[value_key]) for r in rows if r.get("month") in hold_months and finite(r.get(value_key)) is not None]),
    }


def by_symbol(rows: list[dict[str, Any]], value_key: str, hold_months: set[str]) -> dict[str, Any]:
    out = {}
    for sym in sorted({str(r.get("symbol")) for r in rows if r.get("symbol")}):
        out[sym] = summarize([r for r in rows if str(r.get("symbol")) == sym], value_key, hold_months)
    return out


def funding_carry_rows(conn: sqlite3.Connection, *, cost_bps_rt: float, min_abs_funding_bps: float) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for symbol in SYMBOLS:
        marks = load_mark_index(conn, symbol)
        if not marks.ts:
            continue
        start = (marks.ts[0] // (SLOT_SEC * 1000)) * SLOT_SEC * 1000
        end = marks.ts[-1]
        for g in range(int(start), int(end), SLOT_SEC * 1000):
            f = funding_at(conn, symbol, g)
            e = marks.at_or_after(g)
            x = marks.at_or_after(g + SLOT_SEC * 1000)
            if f is None or not e or not x or float(e[1]) <= 0:
                continue
            funding_bps = abs(float(f)) * 10_000.0
            if funding_bps < float(min_abs_funding_bps):
                continue
            direction = "SHORT" if float(f) > 0 else "LONG"
            price_bps = signed_return_bps(direction, float(e[1]), float(x[1]))
            net_bps = price_bps + funding_bps - float(cost_bps_rt)
            out.append(
                {
                    "symbol": symbol,
                    "ts_ms": int(g),
                    "month": iso_month(int(g)),
                    "funding_rate": float(f),
                    "funding_bps": funding_bps,
                    "direction": direction,
                    "price_bps": price_bps,
                    "net_bps": net_bps,
                }
            )
    return out


def book_row(row: Any) -> dict[str, float] | None:
    if not row:
        return None
    return {"ts_ms": int(row[0]), "bid": float(row[1]), "ask": float(row[2]), "mid": float(row[3])}


def first_touch(
    conn: sqlite3.Connection,
    symbol: str,
    start_ms: int,
    end_ms: int,
    *,
    side: str,
    quote: float,
) -> dict[str, float] | None:
    # OUT-OF-SCOPE for ASOF V9: forward-scanning bounded range read with a
    # price predicate (`ask_price<=?`/`bid_price>=?`, ORDER BY ts_ms ASC
    # LIMIT 1 -- the *first touch*), not the `ORDER BY ts_ms DESC LIMIT 1`
    # as-of point lookup the helper migrates. Left on direct SQL.
    if side == "BID":
        row = conn.execute(
            """
            SELECT ts_ms, bid_price, ask_price, mid_price
            FROM book_ticker
            WHERE symbol=? AND ts_ms>=? AND ts_ms<=? AND ask_price<=?
            ORDER BY ts_ms
            LIMIT 1
            """,
            (symbol, int(start_ms), int(end_ms), float(quote)),
        ).fetchone()
        return book_row(row)
    if side == "ASK":
        row = conn.execute(
            """
            SELECT ts_ms, bid_price, ask_price, mid_price
            FROM book_ticker
            WHERE symbol=? AND ts_ms>=? AND ts_ms<=? AND bid_price>=?
            ORDER BY ts_ms
            LIMIT 1
            """,
            (symbol, int(start_ms), int(end_ms), float(quote)),
        ).fetchone()
        return book_row(row)
    raise ValueError(side)


def book_at_or_before(conn: sqlite3.Connection, symbol: str, ts_ms: int) -> dict[str, float] | None:
    """Direct-SQL oracle -- kept unchanged as the parity reference for
    `book_at_or_before_v2` (BATCH-STORAGE-ROTATION-RETENTION-ASOF-LOOKUP-
    CONSUMER-MIGRATION-V9). No longer called by `provision_one_event`; the
    reader-backed path is used instead."""
    row = conn.execute(
        """
        SELECT ts_ms, bid_price, ask_price, mid_price
        FROM book_ticker
        WHERE symbol=? AND ts_ms<=?
        ORDER BY ts_ms DESC
        LIMIT 1
        """,
        (symbol, int(ts_ms)),
    ).fetchone()
    return book_row(row)


_BOOK_COLS = ("ts_ms", "bid_price", "ask_price", "mid_price")


def book_at_or_before_v2(root, symbol: str, ts_ms: int, source_db_path=None) -> dict[str, float] | None:
    """Reader-backed replacement for `book_at_or_before`, via
    lookup_latest_at_or_before. `symbol` is a genuine runtime parameter
    here -- this file's rows come from `row["symbol"]` in the absorption
    pool JSON, which spans BTCUSDT/ETHUSDT/SOLUSDT (confirmed via direct
    inspection). The helper is always called with whichever symbol the
    row actually names; it never assumes SOLUSDT's archive coverage
    applies to a different symbol. Reuses `book_row` for identical
    {ts_ms,bid,ask,mid} shaping."""
    result = RR.lookup_latest_at_or_before(
        root, table="book_ticker", symbol=symbol, ts_ms=int(ts_ms),
        columns=_BOOK_COLS, source_db_path=source_db_path)
    if not result.found:
        return None
    return book_row(result.row)


def provision_one_event(
    conn: sqlite3.Connection,
    row: dict[str, Any],
    *,
    root,
    source_db_path=None,
    offset_bps: float,
    horizon_sec: int,
    maker_fee_bps: float,
    taker_fee_bps: float,
) -> dict[str, Any] | None:
    symbol = str(row["symbol"])
    ts = int(row["entry_ts_ms"])
    mid = finite(row.get("mid"))
    if mid is None or mid <= 0:
        return None
    quote_bid = mid * (1.0 - float(offset_bps) / 10_000.0)
    quote_ask = mid * (1.0 + float(offset_bps) / 10_000.0)
    end_ts = ts + int(horizon_sec) * 1000
    bid_fill = first_touch(conn, symbol, ts, end_ts, side="BID", quote=quote_bid)
    ask_fill = first_touch(conn, symbol, ts, end_ts, side="ASK", quote=quote_ask)
    end = book_at_or_before_v2(root, symbol, end_ts, source_db_path=source_db_path)
    if end is None:
        return None
    fill_count = int(bid_fill is not None) + int(ask_fill is not None)
    if fill_count == 0:
        net = 0.0
        state = "NO_FILL"
        adverse_bps = 0.0
    elif fill_count == 2:
        gross_spread = (quote_ask - quote_bid) / mid * 10_000.0
        net = gross_spread - 2.0 * float(maker_fee_bps)
        state = "BOTH_FILLED"
        adverse_bps = 0.0
    elif bid_fill is not None:
        # Bought passively; flatten at horizon bid.
        gross = (float(end["bid"]) - quote_bid) / quote_bid * 10_000.0
        net = gross - float(maker_fee_bps) - float(taker_fee_bps)
        state = "BID_ONLY_LONG_INVENTORY"
        adverse_bps = gross
    else:
        # Sold passively; flatten at horizon ask.
        gross = (quote_ask - float(end["ask"])) / quote_ask * 10_000.0
        net = gross - float(maker_fee_bps) - float(taker_fee_bps)
        state = "ASK_ONLY_SHORT_INVENTORY"
        adverse_bps = gross
    return {
        "symbol": symbol,
        "entry_ts_ms": ts,
        "month": str(row["month"]),
        "route_id": row.get("route_id"),
        "offset_bps": float(offset_bps),
        "horizon_sec": int(horizon_sec),
        "quote_bid": quote_bid,
        "quote_ask": quote_ask,
        "fill_state": state,
        "fill_count": fill_count,
        "net_bps": net,
        "adverse_or_inventory_gross_bps": adverse_bps,
    }


def provision_rows(
    conn: sqlite3.Connection,
    pool_rows: list[dict[str, Any]],
    *,
    root,
    source_db_path=None,
    offset_bps: float,
    horizon_sec: int,
    maker_fee_bps: float,
    taker_fee_bps: float,
) -> list[dict[str, Any]]:
    out = []
    seen: set[tuple[str, int]] = set()
    for row in pool_rows:
        # Use one anchor per symbol/time to avoid threshold/band duplicates inflating N.
        key = (str(row.get("symbol")), int(row.get("entry_ts_ms") or 0))
        if key in seen:
            continue
        seen.add(key)
        if str(row.get("vdepth_band")) not in {"v28_40", "v40_60", "v60_plus"}:
            continue
        item = provision_one_event(
            conn,
            row,
            root=root,
            source_db_path=source_db_path,
            offset_bps=offset_bps,
            horizon_sec=horizon_sec,
            maker_fee_bps=maker_fee_bps,
            taker_fee_bps=taker_fee_bps,
        )
        if item:
            out.append(item)
    return out


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    pool = json.loads(args.pool_json.read_text(encoding="utf-8"))
    pool_rows = pool.get("rows") or []
    root, _ = PR.resolve_production_root()
    source_db_path = str(args.db)
    with sqlite3.connect(f"file:{args.db}?mode=ro", uri=True) as conn:
        carry_configs = []
        if not args.skip_carry:
            for min_abs in args.min_abs_funding_bps:
                rows = funding_carry_rows(conn, cost_bps_rt=float(args.carry_cost_bps_rt), min_abs_funding_bps=float(min_abs))
                hold_months, split = split_months(rows)
                carry_configs.append(
                    {
                        "config_id": f"carry_paid_side_min_abs_funding_{min_abs:g}bps",
                        "min_abs_funding_bps": float(min_abs),
                        "split": split,
                        "summary": summarize(rows, "net_bps", hold_months),
                        "funding_only": summarize([{**r, "funding_only_bps": r["funding_bps"] - float(args.carry_cost_bps_rt)} for r in rows], "funding_only_bps", hold_months),
                        "price_component": summarize(rows, "price_bps", hold_months),
                        "by_symbol": by_symbol(rows, "net_bps", hold_months),
                    }
                )
        provision_configs = []
        if not args.skip_provision:
            for offset in args.provision_offsets_bps:
                for horizon in args.provision_horizons_sec:
                    rows = provision_rows(
                        conn,
                        pool_rows,
                        root=root,
                        source_db_path=source_db_path,
                        offset_bps=float(offset),
                        horizon_sec=int(horizon),
                        maker_fee_bps=float(args.maker_fee_bps),
                        taker_fee_bps=float(args.taker_fee_bps),
                    )
                    hold_months, split = split_months(rows)
                    fill_counts: dict[str, int] = {}
                    for r in rows:
                        fill_counts[str(r["fill_state"])] = fill_counts.get(str(r["fill_state"]), 0) + 1
                    provision_configs.append(
                        {
                            "config_id": f"provision_o{offset:g}_h{horizon}s_fee{float(args.maker_fee_bps):g}",
                            "offset_bps": float(offset),
                            "horizon_sec": int(horizon),
                            "maker_fee_bps": float(args.maker_fee_bps),
                            "split": split,
                            "fill_counts": fill_counts,
                            "summary": summarize(rows, "net_bps", hold_months),
                            "by_symbol": by_symbol(rows, "net_bps", hold_months),
                        }
                    )
    carry_configs.sort(key=lambda c: float(c["summary"]["hold"]["sum_bps"] or 0.0), reverse=True)
    provision_configs.sort(key=lambda c: float(c["summary"]["hold"]["sum_bps"] or 0.0), reverse=True)
    return {
        "generated_at_utc": utc_now(),
        "mode": "RESEARCH_ONLY_NO_LIVE_NO_PAPER",
        "config": {
            "carry_cost_bps_rt": float(args.carry_cost_bps_rt),
            "maker_fee_bps": float(args.maker_fee_bps),
            "taker_fee_bps": float(args.taker_fee_bps),
            "pool_json": str(args.pool_json),
            "permutation_scope_note": "v5 shuffles directional PnL labels/signs; it does not model exogenous funding cashflows or two-sided spread capture.",
        },
        "funding_carry": carry_configs,
        "maker_provision": provision_configs,
    }


def fmt(m: dict[str, Any]) -> str:
    return (
        f"N={m['n']} sum={m['sum_bps']} mean={m['mean_bps']} med={m['median_bps']} "
        f"T3R={m['t3r_bps']} WR={m['win_rate']} maxL={m['max_loss_bps']}"
    )


def render_md(report: dict[str, Any]) -> str:
    lines = [
        "# Non-Predictive Carry + Provision Research",
        "",
        f"Generated: `{report['generated_at_utc']}`",
        "",
        "`RESEARCH_ONLY_NO_LIVE_NO_PAPER` - no live executor, paper state, or runtime state was touched.",
        "",
        "## Scope Sanity",
        "",
        f"- {report['config']['permutation_scope_note']}",
        "- Therefore v5 kills directional prediction; it does not by itself kill carry/provision mechanisms.",
        "",
        "## Funding Carry Harvest",
        "",
        "Paid side: positive funding -> SHORT receives; negative funding -> LONG receives. Net = price P&L + funding cashflow - round-trip cost.",
        "",
        "| Rank | Config | All | Cal | Hold | Hold funding-only | Hold price component |",
        "| ---: | --- | --- | --- | --- | --- | --- |",
    ]
    for i, cfg in enumerate(report["funding_carry"], 1):
        lines.append(
            f"| {i} | `{cfg['config_id']}` | {fmt(cfg['summary']['all'])} | {fmt(cfg['summary']['cal'])} | "
            f"{fmt(cfg['summary']['hold'])} | {fmt(cfg['funding_only']['hold'])} | {fmt(cfg['price_component']['hold'])} |"
        )
    best_carry = report["funding_carry"][0] if report["funding_carry"] else None
    if best_carry:
        lines.extend(["", "### Best Carry By Symbol", "", "| Symbol | Hold |", "| --- | --- |"])
        for sym, val in best_carry["by_symbol"].items():
            lines.append(f"| `{sym}` | {fmt(val['hold'])} |")
    lines.extend(
        [
            "",
            "## Direction-Agnostic Maker Provision",
            "",
            "At each cascade anchor, quote both sides around mid. Both-filled = spread capture; one-side-filled = inventory flattened at horizon.",
            "",
            "| Rank | Config | Fill counts | All | Cal | Hold |",
            "| ---: | --- | --- | --- | --- | --- |",
        ]
    )
    for i, cfg in enumerate(report["maker_provision"], 1):
        lines.append(
            f"| {i} | `{cfg['config_id']}` | `{cfg['fill_counts']}` | {fmt(cfg['summary']['all'])} | "
            f"{fmt(cfg['summary']['cal'])} | {fmt(cfg['summary']['hold'])} |"
        )
    best_prov = report["maker_provision"][0] if report["maker_provision"] else None
    if best_prov:
        lines.extend(["", "### Best Provision By Symbol", "", "| Symbol | Hold |", "| --- | --- |"])
        for sym, val in best_prov["by_symbol"].items():
            lines.append(f"| `{sym}` | {fmt(val['hold'])} |")
    lines.extend(
        [
            "",
            "## Read",
            "",
            "- Funding carry is not directional mean-reversion; price P&L and funding cashflow are reported separately.",
            "- Maker provision is a first-pass touch-fill model and is optimistic about queue priority. If it fails here, the real version is unlikely to improve.",
            "- A positive result here would still need forward shadow and exchange-fee/rebate-specific validation before any live decision.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_float_tuple(text: str) -> tuple[float, ...]:
    return tuple(float(x.strip()) for x in str(text).split(",") if x.strip())


def parse_int_tuple(text: str) -> tuple[int, ...]:
    return tuple(int(x.strip()) for x in str(text).split(",") if x.strip())


def main() -> int:
    p = argparse.ArgumentParser(description="Research non-predictive carry/provision mechanisms.")
    p.add_argument("--db", type=Path, default=DEFAULT_DB)
    p.add_argument("--pool-json", type=Path, default=POOL_JSON)
    p.add_argument("--out-json", type=Path, default=OUT_JSON)
    p.add_argument("--out-md", type=Path, default=OUT_MD)
    p.add_argument("--carry-cost-bps-rt", type=float, default=8.0)
    p.add_argument("--min-abs-funding-bps", type=parse_float_tuple, default=(0.0, 1.0, 2.0, 5.0))
    p.add_argument("--provision-offsets-bps", type=parse_float_tuple, default=(1.0, 2.0, 5.0, 10.0))
    p.add_argument("--provision-horizons-sec", type=parse_int_tuple, default=(60, 300, 900))
    p.add_argument("--maker-fee-bps", type=float, default=2.0)
    p.add_argument("--taker-fee-bps", type=float, default=3.05)
    p.add_argument("--skip-carry", action="store_true")
    p.add_argument("--skip-provision", action="store_true")
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
