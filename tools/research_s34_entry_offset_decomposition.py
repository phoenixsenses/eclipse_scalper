from __future__ import annotations

import json
import math
import sqlite3
import argparse
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FEATURE_DB = ROOT / "data" / "s34_feature_factory.db"
MICRO_DB = ROOT / "data" / "microstructure.db"
OUT_DIR = ROOT / "reports" / "research" / "s34"

SYMBOL = "ETHUSDT"
LIQ_SIDE = "BUY"
DIRECTION = "LONG"
THRESHOLD_USD = 500_000.0
MIN_DAY_TREND_BPS = 0.0
TP_BPS = 60.0
SL_BPS = 40.0
BE_BPS = 30.0
MAX_HORIZON_SEC = 3600
TAKER_FEE_BPS = 4.0
MAX_BOOK_STALENESS_SEC = 5
OFFSETS_SEC = [0.0, 0.5, 1.0, 2.0, 5.0]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="S34 entry-offset and knowable-anchor execution diagnostic.")
    parser.add_argument("--symbol", default=SYMBOL)
    parser.add_argument("--liq-side", default=LIQ_SIDE)
    parser.add_argument("--direction", choices=["LONG", "SHORT"], default=DIRECTION)
    parser.add_argument("--threshold-usd", type=float, default=THRESHOLD_USD)
    parser.add_argument("--min-day-trend-bps", type=float, default=MIN_DAY_TREND_BPS)
    parser.add_argument("--tp-bps", type=float, default=TP_BPS)
    parser.add_argument("--sl-bps", type=float, default=SL_BPS)
    parser.add_argument("--be-bps", type=float, default=BE_BPS)
    parser.add_argument("--out-prefix", default="S34_ENTRY_OFFSET_DECOMPOSITION")
    return parser


def iso_ms(ts_ms: int | None) -> str | None:
    if ts_ms is None:
        return None
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).isoformat()


def pctile(values: list[float], q: float) -> float | None:
    vals = sorted(v for v in values if v is not None and math.isfinite(v))
    if not vals:
        return None
    if len(vals) == 1:
        return vals[0]
    pos = (len(vals) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return vals[lo]
    return vals[lo] + (vals[hi] - vals[lo]) * (pos - lo)


def mean(values: list[float]) -> float | None:
    vals = [v for v in values if v is not None and math.isfinite(v)]
    return sum(vals) / len(vals) if vals else None


def r1(value: float | None) -> float | None:
    return round(float(value), 1) if value is not None and math.isfinite(float(value)) else None


def r3(value: float | None) -> float | None:
    return round(float(value), 3) if value is not None and math.isfinite(float(value)) else None


def signed_ret_bps(entry: float, price: float) -> float:
    if DIRECTION == "SHORT":
        return (float(entry) - float(price)) / float(entry) * 10000.0
    return (float(price) - float(entry)) / float(entry) * 10000.0


def book_ticker_at(con: sqlite3.Connection, ts_ms: int) -> dict | None:
    row = con.execute(
        """
        select ts_ms, bid_price, ask_price, mid_price
        from book_ticker
        where symbol=? and ts_ms<=?
        order by ts_ms desc
        limit 1
        """,
        (SYMBOL, int(ts_ms)),
    ).fetchone()
    if not row:
        return None
    staleness_ms = int(ts_ms) - int(row[0])
    if staleness_ms > MAX_BOOK_STALENESS_SEC * 1000:
        return None
    return {
        "ts_ms": int(row[0]),
        "bid": float(row[1]),
        "ask": float(row[2]),
        "mid": float(row[3]),
        "staleness_ms": staleness_ms,
    }


def mark_path(con: sqlite3.Connection, start_ms: int, end_ms: int) -> list[tuple[int, float]]:
    return [
        (int(ts), float(price))
        for ts, price in con.execute(
            """
            select ts_ms, mark_price
            from mark_prices
            where symbol=? and ts_ms>=? and ts_ms<=?
            order by ts_ms
            """,
            (SYMBOL, int(start_ms), int(end_ms)),
        )
    ]


def threshold_cross_ts(con: sqlite3.Connection, event: dict) -> int | None:
    total = 0.0
    for ts_ms, notional in con.execute(
        """
        select ts_ms, notional
        from liquidations
        where symbol=? and side=? and ts_ms>=? and ts_ms<=?
        order by ts_ms
        """,
        (
            SYMBOL,
            LIQ_SIDE,
            int(event["cluster_start_ts_ms"]),
            int(event["cluster_end_ts_ms"]),
        ),
    ):
        total += float(notional)
        if total >= THRESHOLD_USD:
            return int(ts_ms)
    return None


def load_events(feature_con: sqlite3.Connection) -> list[dict]:
    feature_con.row_factory = sqlite3.Row
    return [
        dict(row)
        for row in feature_con.execute(
            """
            select *
            from liq_event_features
            where symbol=?
              and liq_side=?
              and cluster_notional>=?
              and day_trend_bps>=?
            order by event_ts_ms
            """,
            (SYMBOL, LIQ_SIDE, THRESHOLD_USD, MIN_DAY_TREND_BPS),
        )
    ]


def simulate_real_fill(
    con: sqlite3.Connection,
    event: dict,
    offset_sec: float = 0.0,
    *,
    anchor_ts_ms: int | None = None,
) -> dict:
    entry_ts_ms = int(anchor_ts_ms) if anchor_ts_ms is not None else int(event["event_ts_ms"]) + int(round(float(offset_sec) * 1000))
    entry_book = book_ticker_at(con, entry_ts_ms)
    if not entry_book:
        return {
            "event_id": event["event_id"],
            "entry_ts_ms": entry_ts_ms,
            "status": "NO_ENTRY_FILL",
            "net_bps": None,
        }

    entry_fill = float(entry_book["bid"] if DIRECTION == "SHORT" else entry_book["ask"])
    if DIRECTION == "SHORT":
        tp_px = entry_fill * (1.0 - TP_BPS / 10000.0)
        sl_px = entry_fill * (1.0 + SL_BPS / 10000.0)
        be_px = entry_fill * (1.0 - BE_BPS / 10000.0)
    else:
        tp_px = entry_fill * (1.0 + TP_BPS / 10000.0)
        sl_px = entry_fill * (1.0 - SL_BPS / 10000.0)
        be_px = entry_fill * (1.0 + BE_BPS / 10000.0)
    be_active = False
    mfe = -1e9
    mae = 1e9
    exit_reason = "TIME"
    exit_ts_ms = entry_ts_ms + MAX_HORIZON_SEC * 1000

    path = mark_path(con, entry_ts_ms, exit_ts_ms)
    if not path:
        return {
            "event_id": event["event_id"],
            "entry_ts_ms": entry_ts_ms,
            "status": "NO_MARK_PATH",
            "net_bps": None,
        }

    for ts_ms, mark in path:
        ret = signed_ret_bps(entry_fill, mark)
        mfe = max(mfe, ret)
        mae = min(mae, ret)
        if DIRECTION == "SHORT":
            if not be_active and mark <= be_px:
                be_active = True
            if mark <= tp_px:
                exit_reason = "TP"
                exit_ts_ms = int(ts_ms)
                break
            if mark >= sl_px:
                exit_reason = "SL"
                exit_ts_ms = int(ts_ms)
                break
            if be_active and mark >= entry_fill:
                exit_reason = "BE"
                exit_ts_ms = int(ts_ms)
                break
        else:
            if not be_active and mark >= be_px:
                be_active = True
            if mark >= tp_px:
                exit_reason = "TP"
                exit_ts_ms = int(ts_ms)
                break
            if mark <= sl_px:
                exit_reason = "SL"
                exit_ts_ms = int(ts_ms)
                break
            if be_active and mark <= entry_fill:
                exit_reason = "BE"
                exit_ts_ms = int(ts_ms)
                break

    exit_book = book_ticker_at(con, exit_ts_ms)
    if not exit_book:
        return {
            "event_id": event["event_id"],
            "entry_ts_ms": entry_ts_ms,
            "exit_ts_ms": exit_ts_ms,
            "status": "NO_EXIT_FILL",
            "exit_reason": exit_reason,
            "net_bps": None,
            "mfe_bps": mfe,
            "mae_bps": mae,
        }

    exit_fill = float(exit_book["ask"] if DIRECTION == "SHORT" else exit_book["bid"])
    gross_bps = signed_ret_bps(entry_fill, exit_fill)
    net_bps = gross_bps - (TAKER_FEE_BPS * 2.0)
    return {
        "event_id": event["event_id"],
        "entry_ts_ms": entry_ts_ms,
        "entry_book_ts_ms": int(entry_book["ts_ms"]),
        "entry_staleness_ms": int(entry_book["staleness_ms"]),
        "entry_fill": entry_fill,
        "exit_ts_ms": exit_ts_ms,
        "exit_book_ts_ms": int(exit_book["ts_ms"]),
        "exit_staleness_ms": int(exit_book["staleness_ms"]),
        "exit_fill": exit_fill,
        "status": "FILLED",
        "exit_reason": exit_reason,
        "gross_bps": gross_bps,
        "net_bps": net_bps,
        "mfe_bps": mfe,
        "mae_bps": mae,
        "hold_sec": (int(exit_ts_ms) - int(entry_ts_ms)) / 1000.0,
    }


def simulate_mark_counterfactual(
    con: sqlite3.Connection,
    event: dict,
    offset_sec: float = 0.0,
    *,
    anchor_ts_ms: int | None = None,
) -> dict:
    entry_ts_ms = int(anchor_ts_ms) if anchor_ts_ms is not None else int(event["event_ts_ms"]) + int(round(float(offset_sec) * 1000))
    path = mark_path(con, entry_ts_ms, entry_ts_ms + MAX_HORIZON_SEC * 1000)
    if not path:
        return {"event_id": event["event_id"], "status": "NO_MARK_PATH", "net_bps": None}
    entry_ts_ms, entry = path[0]
    if DIRECTION == "SHORT":
        tp_px = entry * (1.0 - TP_BPS / 10000.0)
        sl_px = entry * (1.0 + SL_BPS / 10000.0)
        be_px = entry * (1.0 - BE_BPS / 10000.0)
    else:
        tp_px = entry * (1.0 + TP_BPS / 10000.0)
        sl_px = entry * (1.0 - SL_BPS / 10000.0)
        be_px = entry * (1.0 + BE_BPS / 10000.0)
    be_active = False
    exit_reason = "TIME"
    exit_price = path[-1][1]
    exit_ts_ms = path[-1][0]
    mfe = -1e9
    mae = 1e9
    for ts_ms, mark in path:
        ret = signed_ret_bps(entry, mark)
        mfe = max(mfe, ret)
        mae = min(mae, ret)
        if DIRECTION == "SHORT":
            if not be_active and mark <= be_px:
                be_active = True
            if mark <= tp_px:
                exit_reason = "TP"
                exit_ts_ms = ts_ms
                exit_price = tp_px
                break
            if mark >= sl_px:
                exit_reason = "SL"
                exit_ts_ms = ts_ms
                exit_price = sl_px
                break
            if be_active and mark >= entry:
                exit_reason = "BE"
                exit_ts_ms = ts_ms
                exit_price = entry
                break
        else:
            if not be_active and mark >= be_px:
                be_active = True
            if mark >= tp_px:
                exit_reason = "TP"
                exit_ts_ms = ts_ms
                exit_price = tp_px
                break
            if mark <= sl_px:
                exit_reason = "SL"
                exit_ts_ms = ts_ms
                exit_price = sl_px
                break
            if be_active and mark <= entry:
                exit_reason = "BE"
                exit_ts_ms = ts_ms
                exit_price = entry
                break
    gross_bps = signed_ret_bps(entry, exit_price)
    return {
        "event_id": event["event_id"],
        "status": "SIMULATED",
        "entry_ts_ms": entry_ts_ms,
        "exit_ts_ms": exit_ts_ms,
        "exit_reason": exit_reason,
        "gross_bps": gross_bps,
        "net_bps": gross_bps - (TAKER_FEE_BPS * 2.0),
        "mfe_bps": mfe,
        "mae_bps": mae,
        "hold_sec": (int(exit_ts_ms) - int(entry_ts_ms)) / 1000.0,
    }


def summarize(rows: list[dict], key: str = "net_bps") -> dict:
    vals = [float(r[key]) for r in rows if r.get(key) is not None and math.isfinite(float(r[key]))]
    exits: dict[str, int] = {}
    for row in rows:
        if row.get(key) is None:
            continue
        reason = str(row.get("exit_reason") or "NA")
        exits[reason] = exits.get(reason, 0) + 1
    return {
        "n": len(vals),
        "mean": r1(mean(vals)),
        "median": r1(pctile(vals, 0.5)),
        "p25": r1(pctile(vals, 0.25)),
        "p75": r1(pctile(vals, 0.75)),
        "cum": r1(sum(vals)) if vals else None,
        "wr": r3(sum(v > 0 for v in vals) / len(vals)) if vals else None,
        "top3_winner_removed_cum": r1(sum(sorted(vals, reverse=True)[3:])) if len(vals) > 3 else r1(sum(vals)),
        "exits": dict(sorted(exits.items())),
        "mfe_median": r1(pctile([float(r["mfe_bps"]) for r in rows if r.get("mfe_bps") is not None], 0.5)),
        "mae_median": r1(pctile([float(r["mae_bps"]) for r in rows if r.get("mae_bps") is not None], 0.5)),
        "hold_median_sec": r1(pctile([float(r["hold_sec"]) for r in rows if r.get("hold_sec") is not None], 0.5)),
    }


def main() -> None:
    global SYMBOL, LIQ_SIDE, DIRECTION, THRESHOLD_USD, MIN_DAY_TREND_BPS, TP_BPS, SL_BPS, BE_BPS
    args = build_parser().parse_args()
    SYMBOL = str(args.symbol).upper()
    LIQ_SIDE = str(args.liq_side).upper()
    DIRECTION = str(args.direction).upper()
    THRESHOLD_USD = float(args.threshold_usd)
    MIN_DAY_TREND_BPS = float(args.min_day_trend_bps)
    TP_BPS = float(args.tp_bps)
    SL_BPS = float(args.sl_bps)
    BE_BPS = float(args.be_bps)
    out_json = OUT_DIR / f"{args.out_prefix}.json"
    out_md = OUT_DIR / f"{args.out_prefix}.md"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now(timezone.utc).isoformat()

    feature_con = sqlite3.connect(f"file:{FEATURE_DB.as_posix()}?mode=ro", uri=True)
    micro_con = sqlite3.connect(f"file:{MICRO_DB.as_posix()}?mode=ro", uri=True)
    micro_con.execute("pragma query_only=1")
    events = load_events(feature_con)

    timing_rows = []
    for event in events:
        cross = threshold_cross_ts(micro_con, event)
        timing_rows.append(
            {
                "event_id": event["event_id"],
                "event_ts_ms": int(event["event_ts_ms"]),
                "event_utc": iso_ms(int(event["event_ts_ms"])),
                "cluster_end_ts_ms": int(event["cluster_end_ts_ms"]),
                "cluster_end_utc": iso_ms(int(event["cluster_end_ts_ms"])),
                "threshold_cross_ts_ms": cross,
                "threshold_cross_utc": iso_ms(cross),
                "first_to_threshold_sec": ((cross - int(event["event_ts_ms"])) / 1000.0) if cross else None,
                "first_to_end_sec": (int(event["cluster_end_ts_ms"]) - int(event["event_ts_ms"])) / 1000.0,
                "cluster_notional": float(event["cluster_notional"]),
                "day_trend_bps": float(event["day_trend_bps"]),
            }
        )

    offset_results = {}
    for offset in OFFSETS_SEC:
        real_rows = [simulate_real_fill(micro_con, event, offset) for event in events]
        cf_rows = [simulate_mark_counterfactual(micro_con, event, offset) for event in events]
        filled = [row for row in real_rows if row.get("status") == "FILLED"]
        no_fill = [row for row in real_rows if row.get("status") != "FILLED"]
        no_fill_ids = {row["event_id"] for row in no_fill}
        cf_no_fill = [row for row in cf_rows if row.get("event_id") in no_fill_ids and row.get("net_bps") is not None]
        offset_results[str(offset)] = {
            "offset_sec": offset,
            "total_events": len(events),
            "real_fill": summarize(filled),
            "real_fill_rows": filled,
            "no_fill_rows": no_fill,
            "no_fill_count": len(no_fill),
            "no_fill_rate": r3(len(no_fill) / len(events)) if events else None,
            "no_fill_reasons": dict(sorted({s: sum(1 for row in no_fill if row.get("status") == s) for s in {row.get("status") for row in no_fill}}.items())),
            "mark_counterfactual_all": summarize([row for row in cf_rows if row.get("net_bps") is not None]),
            "mark_counterfactual_no_fill_only": summarize(cf_no_fill),
        }

    anchor_results = {}
    anchors = {
        "threshold_cross": {row["event_id"]: row["threshold_cross_ts_ms"] for row in timing_rows},
        "cluster_end": {row["event_id"]: row["cluster_end_ts_ms"] for row in timing_rows},
    }
    for anchor_name, anchor_by_event in anchors.items():
        eligible_events = [event for event in events if anchor_by_event.get(event["event_id"]) is not None]
        real_rows = [
            simulate_real_fill(micro_con, event, anchor_ts_ms=int(anchor_by_event[event["event_id"]]))
            for event in eligible_events
        ]
        cf_rows = [
            simulate_mark_counterfactual(micro_con, event, anchor_ts_ms=int(anchor_by_event[event["event_id"]]))
            for event in eligible_events
        ]
        filled = [row for row in real_rows if row.get("status") == "FILLED"]
        no_fill = [row for row in real_rows if row.get("status") != "FILLED"]
        no_fill_ids = {row["event_id"] for row in no_fill}
        cf_no_fill = [row for row in cf_rows if row.get("event_id") in no_fill_ids and row.get("net_bps") is not None]
        anchor_results[anchor_name] = {
            "anchor": anchor_name,
            "total_events": len(eligible_events),
            "real_fill": summarize(filled),
            "real_fill_rows": filled,
            "no_fill_rows": no_fill,
            "no_fill_count": len(no_fill),
            "no_fill_rate": r3(len(no_fill) / len(eligible_events)) if eligible_events else None,
            "no_fill_reasons": dict(sorted({s: sum(1 for row in no_fill if row.get("status") == s) for s in {row.get("status") for row in no_fill}}.items())),
            "mark_counterfactual_all": summarize([row for row in cf_rows if row.get("net_bps") is not None]),
            "mark_counterfactual_no_fill_only": summarize(cf_no_fill),
        }

    feature_con.close()
    micro_con.close()

    threshold_lags = [row["first_to_threshold_sec"] for row in timing_rows if row["first_to_threshold_sec"] is not None]
    end_lags = [row["first_to_end_sec"] for row in timing_rows if row["first_to_end_sec"] is not None]
    timing_summary = {
        "events": len(events),
        "threshold_cross_observed": len(threshold_lags),
        "first_to_threshold_sec": {
            "median": r1(pctile(threshold_lags, 0.5)),
            "p25": r1(pctile(threshold_lags, 0.25)),
            "p75": r1(pctile(threshold_lags, 0.75)),
            "mean": r1(mean(threshold_lags)),
        },
        "first_to_cluster_end_sec": {
            "median": r1(pctile(end_lags, 0.5)),
            "p25": r1(pctile(end_lags, 0.25)),
            "p75": r1(pctile(end_lags, 0.75)),
            "mean": r1(mean(end_lags)),
        },
    }

    payload = {
        "generated_at_utc": generated_at,
        "scope": {
            "symbol": SYMBOL,
            "liq_side": LIQ_SIDE,
            "cluster_notional_gte": THRESHOLD_USD,
            "day_trend_bps_gte": MIN_DAY_TREND_BPS,
            "route": f"LONG_TP{int(TP_BPS)}_SL{int(SL_BPS)}_BE{int(BE_BPS)}",
            "offsets_sec": OFFSETS_SEC,
            "book_staleness_sec": MAX_BOOK_STALENESS_SEC,
        },
        "timing_summary": timing_summary,
        "timing_rows": timing_rows,
        "offset_results": offset_results,
        "anchor_results": anchor_results,
    }
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# S34 Entry-Offset Decomposition",
        "",
        f"Generated: `{generated_at}`",
        "",
        "Research-only shadow analysis. No runner/config/live-rule changes.",
        "",
        f"Scope: `{SYMBOL} {LIQ_SIDE}`, `{DIRECTION}`, `cluster_notional >= {THRESHOLD_USD:,.0f}`, "
        f"`day_trend_bps >= {MIN_DAY_TREND_BPS:g}`, `TP{TP_BPS:g}/SL{SL_BPS:g}/BE{BE_BPS:g}`, "
        "entry anchored to feature-factory `event_ts_ms` plus offset.",
        "",
        "## First-To-Threshold Timing",
        "",
        "| Metric | N | Median | P25 | P75 | Mean |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        (
            f"| first_ts -> threshold_cross | {timing_summary['threshold_cross_observed']} "
            f"| {timing_summary['first_to_threshold_sec']['median']} "
            f"| {timing_summary['first_to_threshold_sec']['p25']} "
            f"| {timing_summary['first_to_threshold_sec']['p75']} "
            f"| {timing_summary['first_to_threshold_sec']['mean']} |"
        ),
        (
            f"| first_ts -> cluster_end | {timing_summary['events']} "
            f"| {timing_summary['first_to_cluster_end_sec']['median']} "
            f"| {timing_summary['first_to_cluster_end_sec']['p25']} "
            f"| {timing_summary['first_to_cluster_end_sec']['p75']} "
            f"| {timing_summary['first_to_cluster_end_sec']['mean']} |"
        ),
        "",
        "## Real-Fill Offset Curve",
        "",
        "| Offset sec | Filled N | No-fill % | Median | Mean | Cum | WR | Top3W removed | Exits | MFE med | MAE med | Hold med |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |",
    ]
    for offset in OFFSETS_SEC:
        result = offset_results[str(offset)]
        st = result["real_fill"]
        wr = f"{float(st['wr']) * 100:.1f}%" if st.get("wr") is not None else ""
        nf = f"{float(result['no_fill_rate']) * 100:.1f}%" if result.get("no_fill_rate") is not None else ""
        lines.append(
            f"| {offset:g} | {st['n']} | {nf} | {st['median']} | {st['mean']} | {st['cum']} | {wr} "
            f"| {st['top3_winner_removed_cum']} | {st['exits']} | {st['mfe_median']} | {st['mae_median']} | {st['hold_median_sec']} |"
        )

    lines += [
        "",
        "## Knowable-Anchor Curve",
        "",
        f"`threshold_cross` is the first liquidation timestamp where cumulative cluster notional reaches {THRESHOLD_USD:,.0f}. `cluster_end` is the retrospective end of the 300s feature-factory cluster.",
        "",
        "| Anchor | Filled N | No-fill % | Median | Mean | Cum | WR | Top3W removed | Exits | MFE med | MAE med | Hold med |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |",
    ]
    for anchor_name in ["threshold_cross", "cluster_end"]:
        result = anchor_results[anchor_name]
        st = result["real_fill"]
        wr = f"{float(st['wr']) * 100:.1f}%" if st.get("wr") is not None else ""
        nf = f"{float(result['no_fill_rate']) * 100:.1f}%" if result.get("no_fill_rate") is not None else ""
        lines.append(
            f"| {anchor_name} | {st['n']} | {nf} | {st['median']} | {st['mean']} | {st['cum']} | {wr} "
            f"| {st['top3_winner_removed_cum']} | {st['exits']} | {st['mfe_median']} | {st['mae_median']} | {st['hold_median_sec']} |"
        )

    lines += [
        "",
        "## No-Fill Counterfactual",
        "",
        "Counterfactual uses mark-price path for events without executable book entry/exit fill at the same offset.",
        "",
        "| Offset sec | No-fill N | CF Median | CF Mean | CF Cum | CF WR | CF Exits |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for offset in OFFSETS_SEC:
        result = offset_results[str(offset)]
        st = result["mark_counterfactual_no_fill_only"]
        wr = f"{float(st['wr']) * 100:.1f}%" if st.get("wr") is not None else ""
        lines.append(
            f"| {offset:g} | {result['no_fill_count']} | {st['median']} | {st['mean']} | {st['cum']} | {wr} | {st['exits']} |"
        )
    lines += [
        "",
        "## Knowable-Anchor No-Fill Counterfactual",
        "",
        "| Anchor | No-fill N | CF Median | CF Mean | CF Cum | CF WR | CF Exits |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for anchor_name in ["threshold_cross", "cluster_end"]:
        result = anchor_results[anchor_name]
        st = result["mark_counterfactual_no_fill_only"]
        wr = f"{float(st['wr']) * 100:.1f}%" if st.get("wr") is not None else ""
        lines.append(
            f"| {anchor_name} | {result['no_fill_count']} | {st['median']} | {st['mean']} | {st['cum']} | {wr} | {st['exits']} |"
        )

    lines += [
        "",
        "## Read",
        "",
        "- This is an execution-realism diagnostic, not a new rule.",
        "- `event_ts_ms` is the feature-factory first timestamp. The threshold-cross lag table estimates how much of the cascade has already elapsed before the 500K condition is knowable.",
        "- If the real-fill offset curve decays sharply from 0s to 0.5/1/2s, the apparent edge is highly latency-sensitive.",
        "- If no-fill counterfactuals outperform filled rows, the real-fill subset is likely adversely biased by missed fast winners.",
    ]
    out_md.write_text("\n".join(lines), encoding="utf-8")

    print(f"events={len(events)}")
    for offset in OFFSETS_SEC:
        result = offset_results[str(offset)]
        st = result["real_fill"]
        print(
            f"offset={offset:g}s filled={st['n']} no_fill={result['no_fill_count']} "
            f"median={st['median']} mean={st['mean']} wr={st['wr']}"
        )
    for anchor_name in ["threshold_cross", "cluster_end"]:
        result = anchor_results[anchor_name]
        st = result["real_fill"]
        print(
            f"anchor={anchor_name} filled={st['n']} no_fill={result['no_fill_count']} "
            f"median={st['median']} mean={st['mean']} wr={st['wr']}"
        )
    print(f"MD: {out_md}")
    print(f"JSON: {out_json}")


if __name__ == "__main__":
    main()
