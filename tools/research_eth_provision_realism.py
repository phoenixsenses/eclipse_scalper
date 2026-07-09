"""ETH-only provision realism test.

Research-only. Tests whether the last remaining optimistic ETH-only provision
lead survives stricter fill realism:

- no touch-fill: quote fills only if top-of-book crosses beyond the limit by a
  queue penetration threshold;
- adverse multiplier on one-sided inventory markout;
- maker fee/rebate breakeven curve.

No live, paper, runtime, or executor state is touched.
"""

from __future__ import annotations

import argparse
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

from ami.storage import production as PR
from ami.storage import research_reader as RR

POOL_JSON = ROOT / "reports" / "research" / "s34" / "S34_ABSORPTION_SYNC_2X2_POOL.json"
OUT_JSON = ROOT / "reports" / "research" / "s34" / "ETH_PROVISION_REALISM.json"
OUT_MD = ROOT / "reports" / "research" / "s34" / "ETH_PROVISION_REALISM.md"
DEFAULT_DB = ROOT / "data" / "microstructure.db"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def finite(value: Any) -> float | None:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def r1(value: float | None) -> float | None:
    return round(float(value), 1) if value is not None and math.isfinite(float(value)) else None


def r3(value: float | None) -> float | None:
    return round(float(value), 3) if value is not None and math.isfinite(float(value)) else None


def metrics(vals: list[float]) -> dict[str, Any]:
    xs = [float(v) for v in vals if math.isfinite(float(v))]
    if not xs:
        return {"n": 0, "sum_bps": 0.0, "mean_bps": None, "median_bps": None, "win_rate": None, "t3r_bps": 0.0, "max_loss_bps": None}
    ordered = sorted(xs, reverse=True)
    return {
        "n": len(xs),
        "sum_bps": r1(sum(xs)),
        "mean_bps": r1(sum(xs) / len(xs)),
        "median_bps": r1(median(xs)),
        "win_rate": r3(sum(1 for v in xs if v > 0.0) / len(xs)),
        "t3r_bps": r1(sum(ordered[3:]) if len(ordered) > 3 else sum(xs)),
        "max_loss_bps": r1(min(xs)),
    }


def split_months(rows: list[dict[str, Any]]) -> tuple[set[str], dict[str, Any]]:
    months = sorted({str(r["month"]) for r in rows})
    hold_n = max(1, len(months) // 3) if months else 0
    hold_months = set(months[-hold_n:]) if hold_n else set()
    return hold_months, {"method": "chronological_month_tail", "months": months, "holdout_months": sorted(hold_months)}


def summarize(rows: list[dict[str, Any]], hold_months: set[str]) -> dict[str, Any]:
    return {
        "all": metrics([float(r["net_bps"]) for r in rows if finite(r.get("net_bps")) is not None]),
        "cal": metrics([float(r["net_bps"]) for r in rows if r["month"] not in hold_months and finite(r.get("net_bps")) is not None]),
        "hold": metrics([float(r["net_bps"]) for r in rows if r["month"] in hold_months and finite(r.get("net_bps")) is not None]),
    }


def book_row(row: Any) -> dict[str, float] | None:
    if not row:
        return None
    return {"ts_ms": int(row[0]), "bid": float(row[1]), "ask": float(row[2]), "mid": float(row[3])}


def first_cross(
    conn: sqlite3.Connection,
    start_ms: int,
    end_ms: int,
    *,
    side: str,
    quote: float,
    queue_cross_bps: float,
) -> dict[str, float] | None:
    # OUT-OF-SCOPE for ASOF V6: this is a forward-scanning bounded range read
    # with a price predicate (`ask_price<=?`/`bid_price>=?`, ORDER BY ts_ms
    # ASC LIMIT 1 -- the *first crossing*), not the `ORDER BY ts_ms DESC
    # LIMIT 1` as-of point lookup the helper migrates. Left on direct SQL.
    if side == "BID":
        required = float(quote) * (1.0 - float(queue_cross_bps) / 10_000.0)
        row = conn.execute(
            """
            SELECT ts_ms, bid_price, ask_price, mid_price
            FROM book_ticker
            WHERE symbol='ETHUSDT' AND ts_ms>=? AND ts_ms<=? AND ask_price<=?
            ORDER BY ts_ms
            LIMIT 1
            """,
            (int(start_ms), int(end_ms), required),
        ).fetchone()
        return book_row(row)
    if side == "ASK":
        required = float(quote) * (1.0 + float(queue_cross_bps) / 10_000.0)
        row = conn.execute(
            """
            SELECT ts_ms, bid_price, ask_price, mid_price
            FROM book_ticker
            WHERE symbol='ETHUSDT' AND ts_ms>=? AND ts_ms<=? AND bid_price>=?
            ORDER BY ts_ms
            LIMIT 1
            """,
            (int(start_ms), int(end_ms), required),
        ).fetchone()
        return book_row(row)
    raise ValueError(side)


def book_at_or_before(conn: sqlite3.Connection, ts_ms: int) -> dict[str, float] | None:
    """Direct-SQL oracle -- kept unchanged as the parity reference for
    `book_at_or_before_v2` (BATCH-STORAGE-ROTATION-RETENTION-ASOF-LOOKUP-
    CONSUMER-MIGRATION-V6). No longer called by the main path; the reader-
    backed path is used instead."""
    return book_row(
        conn.execute(
            """
            SELECT ts_ms, bid_price, ask_price, mid_price
            FROM book_ticker
            WHERE symbol='ETHUSDT' AND ts_ms<=?
            ORDER BY ts_ms DESC
            LIMIT 1
            """,
            (int(ts_ms),),
        ).fetchone()
    )


def book_at_or_before_v2(root, ts_ms: int, source_db_path=None) -> dict[str, float] | None:
    """Reader-backed replacement for `book_at_or_before`, via
    lookup_latest_at_or_before. Symbol is hardcoded ETHUSDT, exactly as in
    the oracle's SQL (this file never varies it). book_ticker has no
    ETHUSDT archive partition (only SOLUSDT/2026-04 is archived), so real
    production use of this file resolves SQLITE_ONLY -- confirmed, not
    assumed. Reuses `book_row` for identical {ts_ms,bid,ask,mid} shaping;
    the helper's returned column order (ts_ms, bid_price, ask_price,
    mid_price) matches `book_row`'s positional expectations exactly."""
    result = RR.lookup_latest_at_or_before(
        root, table="book_ticker", symbol="ETHUSDT", ts_ms=int(ts_ms),
        columns=("ts_ms", "bid_price", "ask_price", "mid_price"), source_db_path=source_db_path)
    if not result.found:
        return None
    return book_row(result.row)


def load_eth_events(pool_json: Path) -> list[dict[str, Any]]:
    payload = json.loads(pool_json.read_text(encoding="utf-8"))
    rows = payload.get("rows") or []
    out = []
    seen: set[int] = set()
    for r in rows:
        if str(r.get("symbol")) != "ETHUSDT":
            continue
        if str(r.get("vdepth_band")) not in {"v28_40", "v40_60", "v60_plus"}:
            continue
        ts = int(r.get("entry_ts_ms") or 0)
        if ts <= 0 or ts in seen:
            continue
        seen.add(ts)
        mid = finite(r.get("mid"))
        if mid is None or mid <= 0:
            continue
        out.append(r)
    return sorted(out, key=lambda r: int(r["entry_ts_ms"]))


def simulate_event(
    conn: sqlite3.Connection,
    event: dict[str, Any],
    *,
    root,
    source_db_path,
    offset_bps: float,
    horizon_sec: int,
    queue_cross_bps: float,
    maker_fee_bps: float,
    taker_fee_bps: float,
    adverse_mult: float,
) -> dict[str, Any] | None:
    ts = int(event["entry_ts_ms"])
    end_ts = ts + int(horizon_sec) * 1000
    mid = float(event["mid"])
    quote_bid = mid * (1.0 - float(offset_bps) / 10_000.0)
    quote_ask = mid * (1.0 + float(offset_bps) / 10_000.0)
    bid_fill = first_cross(conn, ts, end_ts, side="BID", quote=quote_bid, queue_cross_bps=queue_cross_bps)
    ask_fill = first_cross(conn, ts, end_ts, side="ASK", quote=quote_ask, queue_cross_bps=queue_cross_bps)
    end = book_at_or_before_v2(root, end_ts, source_db_path=source_db_path)
    if not end:
        return None
    fill_count = int(bid_fill is not None) + int(ask_fill is not None)
    if fill_count == 0:
        net = 0.0
        state = "NO_FILL"
        inventory_gross = 0.0
    elif fill_count == 2:
        gross_spread = (quote_ask - quote_bid) / mid * 10_000.0
        net = gross_spread - 2.0 * float(maker_fee_bps)
        state = "BOTH_FILLED"
        inventory_gross = 0.0
    elif bid_fill is not None:
        gross = (float(end["bid"]) - quote_bid) / quote_bid * 10_000.0
        penalty = max(0.0, -gross) * (float(adverse_mult) - 1.0)
        net = gross - penalty - float(maker_fee_bps) - float(taker_fee_bps)
        state = "BID_ONLY_LONG_INVENTORY"
        inventory_gross = gross
    else:
        gross = (quote_ask - float(end["ask"])) / quote_ask * 10_000.0
        penalty = max(0.0, -gross) * (float(adverse_mult) - 1.0)
        net = gross - penalty - float(maker_fee_bps) - float(taker_fee_bps)
        state = "ASK_ONLY_SHORT_INVENTORY"
        inventory_gross = gross
    return {
        "entry_ts_ms": ts,
        "month": str(event["month"]),
        "offset_bps": float(offset_bps),
        "horizon_sec": int(horizon_sec),
        "queue_cross_bps": float(queue_cross_bps),
        "maker_fee_bps": float(maker_fee_bps),
        "adverse_mult": float(adverse_mult),
        "fill_state": state,
        "fill_count": fill_count,
        "net_bps": net,
        "inventory_gross_bps": inventory_gross,
    }


def simulate_config(conn: sqlite3.Connection, events: list[dict[str, Any]], *, root, source_db_path, **kwargs: Any) -> list[dict[str, Any]]:
    out = []
    for ev in events:
        row = simulate_event(conn, ev, root=root, source_db_path=source_db_path, **kwargs)
        if row:
            out.append(row)
    return out


def fill_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in rows:
        out[str(r["fill_state"])] = out.get(str(r["fill_state"]), 0) + 1
    return out


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    events = load_eth_events(args.pool_json)
    configs = []
    root, _ = PR.resolve_production_root()
    source_db_path = str(args.db)
    # `conn` stays for the still-direct forward-scan helper (first_cross);
    # the point-lookup path (book_at_or_before_v2) reads via the reader over
    # source_db_path instead.
    with sqlite3.connect(f"file:{args.db}?mode=ro", uri=True) as conn:
        for queue_cross in args.queue_cross_bps:
            for adverse in args.adverse_mult:
                for fee in args.maker_fee_bps:
                    rows = simulate_config(
                        conn,
                        events,
                        root=root,
                        source_db_path=source_db_path,
                        offset_bps=float(args.offset_bps),
                        horizon_sec=int(args.horizon_sec),
                        queue_cross_bps=float(queue_cross),
                        maker_fee_bps=float(fee),
                        taker_fee_bps=float(args.taker_fee_bps),
                        adverse_mult=float(adverse),
                    )
                    hold_months, split = split_months(rows)
                    configs.append(
                        {
                            "config_id": f"eth_provision_o{args.offset_bps:g}_h{args.horizon_sec}s_q{queue_cross:g}_fee{fee:g}_adv{adverse:g}",
                            "queue_cross_bps": float(queue_cross),
                            "maker_fee_bps": float(fee),
                            "adverse_mult": float(adverse),
                            "split": split,
                            "fill_counts": fill_counts(rows),
                            "summary": summarize(rows, hold_months),
                        }
                    )
    configs.sort(key=lambda c: (float(c["summary"]["hold"]["sum_bps"] or 0.0), float(c["summary"]["hold"]["t3r_bps"] or 0.0)), reverse=True)
    pass_like = [
        c
        for c in configs
        if int(c["summary"]["hold"]["n"]) >= 40
        and float(c["summary"]["cal"]["sum_bps"] or 0.0) > 0
        and float(c["summary"]["hold"]["sum_bps"] or 0.0) > 0
        and float(c["summary"]["cal"]["t3r_bps"] or 0.0) > 0
        and float(c["summary"]["hold"]["t3r_bps"] or 0.0) > 0
    ]
    return {
        "generated_at_utc": utc_now(),
        "mode": "RESEARCH_ONLY_NO_LIVE_NO_PAPER",
        "event_count": len(events),
        "config": {
            "offset_bps": float(args.offset_bps),
            "horizon_sec": int(args.horizon_sec),
            "taker_fee_bps": float(args.taker_fee_bps),
            "queue_cross_bps": list(args.queue_cross_bps),
            "maker_fee_bps": list(args.maker_fee_bps),
            "adverse_mult": list(args.adverse_mult),
            "model": "fill requires quote penetration by queue_cross_bps; one-sided losses scaled by adverse_mult",
        },
        "pass_like_configs": pass_like,
        "ranked_configs": configs,
    }


def fmt(m: dict[str, Any]) -> str:
    return f"N={m['n']} sum={m['sum_bps']} mean={m['mean_bps']} med={m['median_bps']} T3R={m['t3r_bps']} WR={m['win_rate']} maxL={m['max_loss_bps']}"


def render_md(report: dict[str, Any]) -> str:
    lines = [
        "# ETH Provision Realism Test",
        "",
        f"Generated: `{report['generated_at_utc']}`",
        "",
        "`RESEARCH_ONLY_NO_LIVE_NO_PAPER` - no live executor, paper state, or runtime state was touched.",
        "",
        f"- ETH events: `{report['event_count']}`",
        f"- model: `{report['config']['model']}`",
        "",
        "## Pass-Like Configs",
        "",
        f"Configs with cal+hold positive sum/T3R and hold N>=40: `{len(report['pass_like_configs'])}`",
        "",
        "| Config | Fill counts | Cal | Hold |",
        "| --- | --- | --- | --- |",
    ]
    for cfg in report["pass_like_configs"]:
        lines.append(f"| `{cfg['config_id']}` | `{cfg['fill_counts']}` | {fmt(cfg['summary']['cal'])} | {fmt(cfg['summary']['hold'])} |")
    lines.extend(
        [
            "",
            "## Ranked Configs",
            "",
            "| Rank | Config | Fill counts | All | Cal | Hold |",
            "| ---: | --- | --- | --- | --- | --- |",
        ]
    )
    for i, cfg in enumerate(report["ranked_configs"], 1):
        lines.append(
            f"| {i} | `{cfg['config_id']}` | `{cfg['fill_counts']}` | "
            f"{fmt(cfg['summary']['all'])} | {fmt(cfg['summary']['cal'])} | {fmt(cfg['summary']['hold'])} |"
        )
    lines.extend(
        [
            "",
            "## Read",
            "",
            "- The prior optimistic lead was touch-fill with maker rebate. This test requires price to cross beyond the quote.",
            "- If only queue_cross=0 survives, the lead is mostly touch-fill/queue-priority artifact.",
            "- If it survives queue_cross>=0.5 with fee/rebate stress and adverse_mult>=1.0, it becomes a real shadow candidate, not live.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_float_tuple(text: str) -> tuple[float, ...]:
    return tuple(float(x.strip()) for x in str(text).split(",") if x.strip())


def main() -> int:
    p = argparse.ArgumentParser(description="ETH-only provision realistic fill sensitivity.")
    p.add_argument("--db", type=Path, default=DEFAULT_DB)
    p.add_argument("--pool-json", type=Path, default=POOL_JSON)
    p.add_argument("--out-json", type=Path, default=OUT_JSON)
    p.add_argument("--out-md", type=Path, default=OUT_MD)
    p.add_argument("--offset-bps", type=float, default=2.0)
    p.add_argument("--horizon-sec", type=int, default=300)
    p.add_argument("--queue-cross-bps", type=parse_float_tuple, default=(0.0, 0.25, 0.5, 1.0, 2.0))
    p.add_argument("--maker-fee-bps", type=parse_float_tuple, default=(-0.5, 0.0, 0.5, 1.0, 2.0))
    p.add_argument("--taker-fee-bps", type=float, default=3.05)
    p.add_argument("--adverse-mult", type=parse_float_tuple, default=(1.0, 1.2, 1.5))
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
