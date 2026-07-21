from __future__ import annotations

import datetime as dt
import itertools
import json
import sqlite3
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.s34_shadow_paper_runner import S34Rule, _bucket_events


SOURCE_DB = f"file:{(ROOT / 'data' / 'microstructure.db').as_posix()}?mode=ro"
OUT_JSON = ROOT / "reports" / "research" / "s34" / "S34_BTC_1M_ROUTE_SWEEP.json"
OUT_MD = ROOT / "reports" / "research" / "s34" / "S34_BTC_1M_ROUTE_SWEEP.md"

SYMBOL = "BTCUSDT"
LIQ_SIDE = "BUY"
THRESHOLD_USD = 1_000_000.0
ENTRY_DELAY_SEC = 0
MAX_HORIZON_SEC = 3600
BUCKET_SEC = 300
MIN_GAP_SEC = 900
MAX_SINGLE_LIQ_SHARE_PCT = 50.0   # pre-reg: distributed filter
TP_GRID = [40.0, 60.0, 80.0, 100.0, 120.0]
SL_GRID = [20.0, 30.0, 40.0, 50.0]
BE_GRID = [20.0, 25.0, 30.0, 35.0, 40.0]
CURRENT = {"tp": 60.0, "sl": 30.0, "be": 30.0}  # pre-reg: SL=30
ROUND_TRIP_FEE_BPS = 8.0
TAKER_FEE_BPS = 4.0
MAX_BOOK_STALENESS_SEC = 5
LOOKBACK_DAYS = 120
SIGNAL_LIMIT = 100_000


def median(vals: list[float]) -> float | None:
    return statistics.median(vals) if vals else None


def iso_day(ts_ms: int) -> str:
    return dt.datetime.fromtimestamp(ts_ms / 1000, tz=dt.timezone.utc).date().isoformat()


def iso_ts(ts_ms: int) -> str:
    return dt.datetime.fromtimestamp(ts_ms / 1000, tz=dt.timezone.utc).isoformat()


def signed_ret(entry_price: float, price: float) -> float:
    return (float(price) - float(entry_price)) / float(entry_price) * 10000.0


def price_from_ret(entry_price: float, bps: float) -> float:
    return float(entry_price) * (1.0 + float(bps) / 10000.0)


def count_by(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for row in rows:
        value = str(row[key])
        out[value] = out.get(value, 0) + 1
    return dict(sorted(out.items()))


def summarize(rows: list[dict[str, Any]], key: str = "net_bps") -> dict[str, Any]:
    vals = [float(row[key]) for row in rows if row.get(key) is not None]
    days = sorted({row["day"] for row in rows})
    day_cums = {day: sum(float(row[key]) for row in rows if row["day"] == day and row.get(key) is not None) for day in days}
    if not vals:
        return {
            "n": 0,
            "days": 0,
            "mean": None,
            "median": None,
            "cum": 0.0,
            "wr": None,
            "top3_removed_cum": 0.0,
            "positive_days": 0,
            "worst_day_cum": None,
            "exit_counts": {},
        }
    return {
        "n": len(vals),
        "days": len(days),
        "mean": sum(vals) / len(vals),
        "median": median(vals),
        "cum": sum(vals),
        "wr": sum(v > 0 for v in vals) / len(vals),
        "top3_removed_cum": sum(sorted(vals, reverse=True)[3:]) if len(vals) > 3 else 0.0,
        "positive_days": sum(v > 0 for v in day_cums.values()),
        "worst_day_cum": min(day_cums.values()) if day_cums else None,
        "exit_counts": count_by(rows, "exit_reason"),
    }


def mark_at(con: sqlite3.Connection, ts_ms: int):
    return con.execute(
        """
        SELECT ts_ms, mark_price
        FROM mark_prices
        WHERE symbol=? AND ts_ms>=?
        ORDER BY ts_ms ASC
        LIMIT 1
        """,
        (SYMBOL, int(ts_ms)),
    ).fetchone()


def path_marks(con: sqlite3.Connection, entry_ts_ms: int) -> list[tuple[int, float]]:
    return [
        (int(ts), float(price))
        for ts, price in con.execute(
            """
            SELECT ts_ms, mark_price
            FROM mark_prices
            WHERE symbol=? AND ts_ms>=? AND ts_ms<=?
            ORDER BY ts_ms ASC
            """,
            (SYMBOL, int(entry_ts_ms), int(entry_ts_ms) + MAX_HORIZON_SEC * 1000),
        )
    ]


def book_ticker_at(con: sqlite3.Connection, ts_ms: int):
    row = con.execute(
        """
        SELECT ts_ms, bid_price, ask_price, mid_price
        FROM book_ticker
        WHERE symbol=? AND ts_ms<=?
        ORDER BY ts_ms DESC
        LIMIT 1
        """,
        (SYMBOL, int(ts_ms)),
    ).fetchone()
    if not row:
        return None
    if int(ts_ms) - int(row[0]) > MAX_BOOK_STALENESS_SEC * 1000:
        return None
    return {"ts_ms": int(row[0]), "bid": float(row[1]), "ask": float(row[2]), "mid": float(row[3])}


def real_fill_net(con: sqlite3.Connection, row: dict[str, Any]) -> dict[str, Any] | None:
    entry_book = book_ticker_at(con, int(row["entry_ts_ms"]))
    exit_book = book_ticker_at(con, int(row["exit_ts_ms"]))
    if not entry_book or not exit_book:
        return None
    basis = float(row["entry_price"])
    exit_ref = float(row["exit_price"])
    entry_fill = float(entry_book["ask"])
    exit_fill = float(exit_book["bid"])
    entry_mid = float(entry_book["mid"])
    exit_mid = float(exit_book["mid"])
    gross_bps = signed_ret(basis, exit_ref)
    executable_bps = signed_ret(entry_fill, exit_fill) * (entry_fill / basis)
    entry_adverse_bps = (entry_mid - basis) / basis * 10000.0
    exit_adverse_bps = (exit_ref - exit_mid) / basis * 10000.0
    spread_cost_bps = ((entry_fill - entry_mid) + (exit_mid - exit_fill)) / basis * 10000.0
    fee_cost_bps = TAKER_FEE_BPS * 2.0
    net_bps = gross_bps - entry_adverse_bps - exit_adverse_bps - spread_cost_bps - fee_cost_bps
    executable_net = executable_bps - fee_cost_bps
    if abs(net_bps - executable_net) > 1e-6:
        raise RuntimeError(f"identity mismatch {net_bps} != {executable_net}")
    return {
        **row,
        "real_gross_bps": gross_bps,
        "entry_adverse_bps": entry_adverse_bps,
        "exit_adverse_bps": exit_adverse_bps,
        "spread_cost_bps": spread_cost_bps,
        "fee_cost_bps": fee_cost_bps,
        "real_net_bps": net_bps,
    }


def simulate_path(event: dict[str, Any], marks: list[tuple[int, float]], tp: float, sl: float, be: float) -> dict[str, Any] | None:
    if not marks:
        return None
    entry_ts_ms, entry_price = marks[0]
    be_active = False
    mfe = -1e9
    mae = 1e9
    exit_reason = "TIME"
    exit_ts_ms, exit_price = marks[-1]
    for ts_ms, price in marks:
        ret = signed_ret(entry_price, price)
        mfe = max(mfe, ret)
        mae = min(mae, ret)
        if not be_active and ret >= be:
            be_active = True
        if ret >= tp:
            exit_reason = "TP"
            exit_ts_ms, exit_price = ts_ms, price
            break
        if ret <= -sl:
            exit_reason = "SL"
            exit_ts_ms, exit_price = ts_ms, price
            break
        if be_active and ret <= 0:
            exit_reason = "BE"
            exit_ts_ms = ts_ms
            exit_price = price_from_ret(entry_price, 0.0)
            break
    gross = signed_ret(entry_price, exit_price)
    return {
        "event_id": event["event_id"],
        "event_ts_ms": int(event["ts_ms"]),
        "event_utc": event["ts_utc"],
        "day": iso_day(int(event["ts_ms"])),
        "liq_total_notional": float(event.get("liq_total_notional") or 0.0),
        "liq_count": int(event.get("liq_count") or 0),
        "cluster_duration_sec": float(event.get("cluster_duration_sec") or 0.0),
        "cluster_shape_label": event.get("cluster_shape_label"),
        "entry_ts_ms": int(entry_ts_ms),
        "entry_price": float(entry_price),
        "exit_ts_ms": int(exit_ts_ms),
        "exit_price": float(exit_price),
        "exit_reason": exit_reason,
        "gross_bps": gross,
        "net_bps": gross - ROUND_TRIP_FEE_BPS,
        "mfe_bps": mfe,
        "mae_bps": mae,
    }


def split_rows(rows: list[dict[str, Any]], split_ts_ms: int) -> dict[str, list[dict[str, Any]]]:
    return {
        "train": [row for row in rows if int(row["event_ts_ms"]) <= split_ts_ms],
        "test": [row for row in rows if int(row["event_ts_ms"]) > split_ts_ms],
        "all": rows,
    }


def route_id(tp: float, sl: float, be: float) -> str:
    return f"TP{int(tp)}_SL{int(sl)}_BE{int(be)}"


def fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "NA"
    return f"{float(value):+.{digits}f}"


def main() -> int:
    source_con = sqlite3.connect(SOURCE_DB, uri=True, timeout=30)
    source_con.execute("PRAGMA query_only=1")
    max_ts = source_con.execute("SELECT MAX(ts_ms) FROM liquidations WHERE symbol=? AND side=?", (SYMBOL, LIQ_SIDE)).fetchone()[0]
    if not max_ts:
        raise RuntimeError("no BTC liquidation rows")
    end_ms = int(max_ts)
    start_ms = end_ms - LOOKBACK_DAYS * 24 * 3600 * 1000

    base_rule = S34Rule(
        name="BTC_BUY_LIQ_LONG_1M_DISTRIBUTED_TP60_SL30_BE30",
        symbol=SYMBOL,
        liq_side=LIQ_SIDE,
        direction="LONG",
        threshold_usd=THRESHOLD_USD,
        bucket_sec=BUCKET_SEC,
        min_gap_sec=MIN_GAP_SEC,
        tp_bps=CURRENT["tp"],
        sl_bps=CURRENT["sl"],
        be_trigger_bps=CURRENT["be"],
        max_horizon_sec=MAX_HORIZON_SEC,
        max_single_liq_share_pct=MAX_SINGLE_LIQ_SHARE_PCT,
        use_global_regime=False,
    )
    signals = _bucket_events(source_con, base_rule, start_ms, end_ms, SIGNAL_LIMIT)
    events = []
    for idx, signal in enumerate(signals, 1):
        if signal.get("fill_error"):
            continue
        signal = dict(signal)
        signal["event_id"] = idx
        events.append(signal)
    split_ts_ms = events[len(events) // 2]["ts_ms"] if events else end_ms

    event_paths = {}
    for event in events:
        entry = mark_at(source_con, int(event["entry_ts_ms"]))
        if not entry:
            continue
        event_paths[event["event_id"]] = path_marks(source_con, int(entry[0]))

    combinations = list(itertools.product(TP_GRID, SL_GRID, BE_GRID))
    # SL_GRID now has 4 values: 4*5*5 = 100 combinations
    route_rows: dict[str, list[dict[str, Any]]] = {}
    route_summary: dict[str, dict[str, Any]] = {}
    for tp, sl, be in combinations:
        rid = route_id(tp, sl, be)
        rows = []
        for event in events:
            row = simulate_path(event, event_paths.get(event["event_id"], []), tp, sl, be)
            if row:
                row.update({"route": rid, "tp": tp, "sl": sl, "be": be})
                rows.append(row)
        route_rows[rid] = rows
        periods = split_rows(rows, int(split_ts_ms))
        route_summary[rid] = {
            "tp": tp,
            "sl": sl,
            "be": be,
            "train": summarize(periods["train"]),
            "test": summarize(periods["test"]),
            "all": summarize(periods["all"]),
        }

    ranked_train = sorted(
        route_summary.values(),
        key=lambda r: (
            r["train"]["n"] >= 20,
            r["train"]["median"] is not None and r["train"]["median"] > 0,
            r["train"]["top3_removed_cum"],
            r["train"]["median"] if r["train"]["median"] is not None else -1e9,
            r["train"]["mean"] if r["train"]["mean"] is not None else -1e9,
        ),
        reverse=True,
    )
    top5 = ranked_train[:5]
    current_rid = route_id(CURRENT["tp"], CURRENT["sl"], CURRENT["be"])
    top_ids = {route_id(r["tp"], r["sl"], r["be"]) for r in top5}
    top_ids.add(current_rid)

    real_fill = {}
    for rid in sorted(top_ids):
        rows = route_rows[rid]
        filled = []
        no_fill = 0
        for row in rows:
            rf = real_fill_net(source_con, row)
            if not rf:
                no_fill += 1
                continue
            filled.append(rf)
        periods = split_rows(filled, int(split_ts_ms))
        real_fill[rid] = {
            "total_rows": len(rows),
            "real_fill_rows": len(filled),
            "no_fill_rows": no_fill,
            "no_fill_rate": no_fill / len(rows) if rows else None,
            "train": summarize(periods["train"], key="real_net_bps"),
            "test": summarize(periods["test"], key="real_net_bps"),
            "all": summarize(periods["all"], key="real_net_bps"),
            "mean_entry_adverse": sum(r["entry_adverse_bps"] for r in filled) / len(filled) if filled else None,
            "mean_exit_adverse": sum(r["exit_adverse_bps"] for r in filled) / len(filled) if filled else None,
            "mean_spread": sum(r["spread_cost_bps"] for r in filled) / len(filled) if filled else None,
            "mean_fee": sum(r["fee_cost_bps"] for r in filled) / len(filled) if filled else None,
        }

    payload = {
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "scope": {
            "symbol": SYMBOL,
            "liq_side": LIQ_SIDE,
            "threshold_usd": THRESHOLD_USD,
            "entry_delay_sec": ENTRY_DELAY_SEC,
            "signals": len(signals),
            "events": len(events),
            "simulated_events": len(event_paths),
            "combination_count": len(combinations),
            "start_utc": iso_ts(start_ms),
            "end_utc": iso_ts(end_ms),
            "split_ts_ms": int(split_ts_ms),
            "split_utc": iso_ts(int(split_ts_ms)),
            "current_route": current_rid,
            "cost_note": "OOS grid uses mark path + flat 8 bps. Real-fill parity uses historical book_ticker ask entry / bid exit where available.",
        },
        "top5_train": top5,
        "current": route_summary[current_rid],
        "all_routes": route_summary,
        "real_fill": real_fill,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# S34 BTC 1M Route Sweep",
        "",
        f"Generated: `{payload['generated_at_utc']}`",
        "",
        "Scope: `BTC BUY`, `cluster_notional >= 1M`, `delay0`; live runner/config unchanged.",
        "",
        f"Combinations evaluated: `{len(combinations)}`",
        f"Signals: `{len(signals)}`; real-entry events: `{len(events)}`; simulated events: `{len(event_paths)}`",
        f"Temporal split: `{payload['scope']['split_utc']}`",
        "",
        "## 1. Train-Selected Top 5, With Test Performance",
        "",
        "| Rank | Route | Train N | Train Median | Train Cum | Train Top3 Removed | Test N | Test Median | Test Mean | Test Cum | Test WR | Test Top3 Removed | Test Exits |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for idx, row in enumerate(top5, 1):
        rid = route_id(row["tp"], row["sl"], row["be"])
        tr = row["train"]
        te = row["test"]
        lines.append(
            f"| {idx} | {rid} | {tr['n']} | {fmt(tr['median'])} | {fmt(tr['cum'])} | {fmt(tr['top3_removed_cum'])} | "
            f"{te['n']} | {fmt(te['median'])} | {fmt(te['mean'])} | {fmt(te['cum'])} | "
            f"{(te['wr'] or 0)*100:.1f}% | {fmt(te['top3_removed_cum'])} | {te['exit_counts']} |"
        )
    lines.extend(
        [
            "",
            "## 2. Current BTC Route Comparator",
            "",
            "| Route | Period | N | Median | Mean | Cum | WR | Top3 Removed | Exits |",
            "|---|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    cur = route_summary[current_rid]
    for period in ["train", "test", "all"]:
        row = cur[period]
        lines.append(
            f"| {current_rid} | {period} | {row['n']} | {fmt(row['median'])} | {fmt(row['mean'])} | "
            f"{fmt(row['cum'])} | {(row['wr'] or 0)*100:.1f}% | {fmt(row['top3_removed_cum'])} | {row['exit_counts']} |"
        )
    lines.extend(
        [
            "",
            "## 3. Real-Fill Parity: Top 5 + Current",
            "",
            "| Route | Total | Real Fill | No Fill | No Fill Rate | Test N | Test Median | Test Mean | Test Cum | Test Top3 Removed | Test Positive Days | Entry Adv | Exit Adv | Spread | Fee |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    ordered_real = [route_id(r["tp"], r["sl"], r["be"]) for r in top5]
    if current_rid not in ordered_real:
        ordered_real.append(current_rid)
    for rid in ordered_real:
        rf = real_fill[rid]
        te = rf["test"]
        lines.append(
            f"| {rid} | {rf['total_rows']} | {rf['real_fill_rows']} | {rf['no_fill_rows']} | "
            f"{(rf['no_fill_rate'] or 0)*100:.1f}% | {te['n']} | {fmt(te['median'])} | {fmt(te['mean'])} | "
            f"{fmt(te['cum'])} | {fmt(te['top3_removed_cum'])} | {te['positive_days']}/{te['days']} | "
            f"{fmt(rf['mean_entry_adverse'])} | {fmt(rf['mean_exit_adverse'])} | {fmt(rf['mean_spread'])} | {fmt(rf['mean_fee'])} |"
        )
    best_real = sorted(
        [(rid, real_fill[rid]["test"]) for rid in ordered_real],
        key=lambda x: (
            x[1]["n"] >= 20,
            x[1]["median"] if x[1]["median"] is not None else -1e9,
            x[1]["top3_removed_cum"],
            x[1]["mean"] if x[1]["mean"] is not None else -1e9,
        ),
        reverse=True,
    )[0]
    lines.extend(
        [
            "",
            "## Read",
            "",
            f"Best real-fill test median among reported routes: `{best_real[0]}`. This is retrospective over 75 combinations; promotion would require a separate exploratory pre-registration. Existing live runner remains unchanged by this report.",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    source_con.close()
    print(OUT_MD)
    print(OUT_JSON)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
