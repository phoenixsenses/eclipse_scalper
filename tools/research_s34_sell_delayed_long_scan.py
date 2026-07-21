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
OUT_JSON = ROOT / "reports" / "research" / "s34" / "S34_SELL_DELAYED_LONG_SCAN.json"
OUT_MD = ROOT / "reports" / "research" / "s34" / "S34_SELL_DELAYED_LONG_SCAN.md"

SYMBOL = "ETHUSDT"
LIQ_SIDE = "SELL"
DIRECTION = "LONG"
THRESHOLD_GRID = [50_000.0, 100_000.0, 200_000.0, 500_000.0]
DELAY_GRID_SEC = [0, 60, 120, 300, 600]
TP_GRID = [40.0, 60.0, 80.0]
SL_BPS = 40.0
BE_BPS = 30.0
MAX_HORIZON_SEC = 3600
BUCKET_SEC = 300
MIN_GAP_SEC = 900
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


def simulate_path(event: dict[str, Any], marks: list[tuple[int, float]], tp: float) -> dict[str, Any] | None:
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
        if not be_active and ret >= BE_BPS:
            be_active = True
        if ret >= tp:
            exit_reason = "TP"
            exit_ts_ms, exit_price = ts_ms, price
            break
        if ret <= -SL_BPS:
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


def route_id(threshold: float, delay: int, tp: float) -> str:
    return f"TH{int(threshold / 1000)}K_DELAY{int(delay)}_TP{int(tp)}_SL40_BE30"


def fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "NA"
    return f"{float(value):+.{digits}f}"


def load_events(con: sqlite3.Connection, start_ms: int, end_ms: int) -> dict[float, list[dict[str, Any]]]:
    out = {}
    for threshold in THRESHOLD_GRID:
        rule = S34Rule(
            name=f"ETH_SELL_LIQ_LONG_{int(threshold)}_RESEARCH",
            symbol=SYMBOL,
            liq_side=LIQ_SIDE,
            direction=DIRECTION,
            threshold_usd=threshold,
            bucket_sec=BUCKET_SEC,
            min_gap_sec=MIN_GAP_SEC,
            use_global_regime=False,
        )
        rows = []
        for idx, signal in enumerate(_bucket_events(con, rule, start_ms, end_ms, SIGNAL_LIMIT), 1):
            if signal.get("fill_error"):
                continue
            signal = dict(signal)
            signal["event_id"] = f"{int(threshold)}:{idx}"
            rows.append(signal)
        out[threshold] = rows
    return out


def main() -> int:
    con = sqlite3.connect(SOURCE_DB, uri=True, timeout=30)
    con.execute("PRAGMA query_only=1")
    max_ts = con.execute("SELECT MAX(ts_ms) FROM liquidations WHERE symbol=? AND side=?", (SYMBOL, LIQ_SIDE)).fetchone()[0]
    if not max_ts:
        raise RuntimeError("no ETH SELL liquidation rows")
    end_ms = int(max_ts)
    start_ms = end_ms - LOOKBACK_DAYS * 24 * 3600 * 1000
    events_by_threshold = load_events(con, start_ms, end_ms)
    all_event_ts = sorted({int(event["ts_ms"]) for rows in events_by_threshold.values() for event in rows})
    split_ts_ms = all_event_ts[len(all_event_ts) // 2] if all_event_ts else end_ms

    route_rows: dict[str, list[dict[str, Any]]] = {}
    route_summary: dict[str, dict[str, Any]] = {}
    for threshold, delay, tp in itertools.product(THRESHOLD_GRID, DELAY_GRID_SEC, TP_GRID):
        rid = route_id(threshold, delay, tp)
        rows = []
        for event in events_by_threshold[threshold]:
            entry = mark_at(con, int(event["ts_ms"]) + int(delay) * 1000)
            if not entry:
                continue
            path = path_marks(con, int(entry[0]))
            row = simulate_path(event, path, tp)
            if row:
                row.update({"route": rid, "threshold": threshold, "delay_sec": delay, "tp": tp, "sl": SL_BPS, "be": BE_BPS})
                rows.append(row)
        route_rows[rid] = rows
        periods = split_rows(rows, int(split_ts_ms))
        route_summary[rid] = {
            "threshold": threshold,
            "delay_sec": delay,
            "tp": tp,
            "sl": SL_BPS,
            "be": BE_BPS,
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
    top_ids = {route_id(r["threshold"], r["delay_sec"], r["tp"]) for r in top5}

    real_fill = {}
    for rid in sorted(top_ids):
        rows = route_rows[rid]
        filled = []
        no_fill = 0
        for row in rows:
            rf = real_fill_net(con, row)
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
            "direction": DIRECTION,
            "threshold_grid": THRESHOLD_GRID,
            "delay_grid_sec": DELAY_GRID_SEC,
            "tp_grid": TP_GRID,
            "sl_bps": SL_BPS,
            "be_bps": BE_BPS,
            "combination_count": len(THRESHOLD_GRID) * len(DELAY_GRID_SEC) * len(TP_GRID),
            "start_utc": iso_ts(start_ms),
            "end_utc": iso_ts(end_ms),
            "split_ts_ms": int(split_ts_ms),
            "split_utc": iso_ts(int(split_ts_ms)),
        },
        "event_counts_by_threshold": {str(int(k)): len(v) for k, v in events_by_threshold.items()},
        "top5_train": top5,
        "all_routes": route_summary,
        "real_fill": real_fill,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# S34 SELL-Liq Delayed LONG Scan",
        "",
        f"Generated: `{payload['generated_at_utc']}`",
        "",
        "Scope: `ETH SELL liquidation cluster -> LONG`, threshold x delay x TP grid; live runner/config unchanged.",
        "",
        f"Combinations evaluated: `{payload['scope']['combination_count']}`",
        f"Event counts by threshold: `{payload['event_counts_by_threshold']}`",
        f"Temporal split: `{payload['scope']['split_utc']}`",
        "",
        "## 1. Train-Selected Top 5, With Test Performance",
        "",
        "| Rank | Route | Train N | Train Median | Train Cum | Train Top3 Removed | Test N | Test Median | Test Mean | Test Cum | Test WR | Test Top3 Removed | Test Exits |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for idx, row in enumerate(top5, 1):
        rid = route_id(row["threshold"], row["delay_sec"], row["tp"])
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
            "## 2. Real-Fill Parity: Top 5",
            "",
            "| Route | Total | Real Fill | No Fill | No Fill Rate | Test N | Test Median | Test Mean | Test Cum | Test Top3 Removed | Test Positive Days | Entry Adv | Exit Adv | Spread | Fee |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in top5:
        rid = route_id(row["threshold"], row["delay_sec"], row["tp"])
        rf = real_fill[rid]
        te = rf["test"]
        lines.append(
            f"| {rid} | {rf['total_rows']} | {rf['real_fill_rows']} | {rf['no_fill_rows']} | "
            f"{(rf['no_fill_rate'] or 0)*100:.1f}% | {te['n']} | {fmt(te['median'])} | {fmt(te['mean'])} | "
            f"{fmt(te['cum'])} | {fmt(te['top3_removed_cum'])} | {te['positive_days']}/{te['days']} | "
            f"{fmt(rf['mean_entry_adverse'])} | {fmt(rf['mean_exit_adverse'])} | {fmt(rf['mean_spread'])} | {fmt(rf['mean_fee'])} |"
        )
    best_real = sorted(
        [(route_id(r["threshold"], r["delay_sec"], r["tp"]), real_fill[route_id(r["threshold"], r["delay_sec"], r["tp"])]["test"]) for r in top5],
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
            f"Best real-fill test median among train-selected routes: `{best_real[0]}`. This is a broad retrospective scan; any promotion would need a separate pre-registration and should be treated as a new alpha family.",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    con.close()
    print(OUT_MD)
    print(OUT_JSON)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
