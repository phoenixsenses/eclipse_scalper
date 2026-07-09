import datetime as dt
import json
import math
import sqlite3
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ami.storage import production as PR
from ami.storage import research_reader as RR

FEATURE_DB = "data/s34_feature_factory.db"
SOURCE_DB = "file:data/microstructure.db?mode=ro"
OUT_JSON = Path("reports/research/s34/S34_EXIT_GIVEBACK_SWEEP.json")
OUT_MD = Path("reports/research/s34/S34_EXIT_GIVEBACK_SWEEP.md")

FEE_BPS = 8.0
MAX_HORIZON_SEC = 3600
BE_SWEEP = [20.0, 25.0, 30.0, 35.0, 40.0]

ROUTES = {
    "LONG_DELAY0_TP60": {"direction": "LONG", "entry_delay_sec": 0, "tp_bps": 60.0, "sl_bps": 40.0},
    "LONG_DELAY60_TP120": {"direction": "LONG", "entry_delay_sec": 60, "tp_bps": 120.0, "sl_bps": 40.0},
    "SHORT_DELAY0_TP40_CONTROL": {"direction": "SHORT", "entry_delay_sec": 0, "tp_bps": 40.0, "sl_bps": 40.0},
}


def median(vals: list[float]) -> float:
    return statistics.median(vals) if vals else 0.0


def mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def mark_at(con: sqlite3.Connection, symbol: str, ts_ms: int, before: bool = False):
    """Out of scope for BATCH-STORAGE-ROTATION-RETENTION-RANGE-READ-
    CONSUMER-MIGRATION-V7 (range-read only). This function is called
    exclusively with `before=False` in this file, i.e. an
    `ORDER BY ts_ms ASC LIMIT 1` "first row at-or-after ts_ms" lookup --
    the opposite direction from `research_reader.lookup_latest_at_or_before`,
    which only supports "latest row at-or-before ts_ms"
    (`ORDER BY ts_ms DESC LIMIT 1`). No forward-ASOF primitive exists in
    the reader yet, and this gate is not authorized to add one (no new
    helper). Left on direct SQL, untouched; proven untouched by
    `test_mark_at_asof_forward_lookup_left_untouched`."""
    op = "<=" if before else ">="
    order = "desc" if before else "asc"
    return con.execute(
        f"""
        select ts_ms, mark_price
        from mark_prices
        where symbol=? and ts_ms {op} ?
        order by ts_ms {order}
        limit 1
        """,
        (symbol, ts_ms),
    ).fetchone()


def signed_ret(direction: str, entry_price: float, price: float) -> float:
    raw = (float(price) - float(entry_price)) / float(entry_price) * 10000.0
    return raw if direction == "LONG" else -raw


def exit_price_from_signed_ret(direction: str, entry_price: float, signed_bps: float) -> float:
    if direction == "LONG":
        return entry_price * (1.0 + signed_bps / 10000.0)
    return entry_price * (1.0 - signed_bps / 10000.0)


def net_from_exit(direction: str, entry_price: float, exit_price: float) -> float:
    return signed_ret(direction, entry_price, exit_price) - FEE_BPS


def load_feature_events() -> list[dict]:
    con = sqlite3.connect(FEATURE_DB)
    con.row_factory = sqlite3.Row
    rows = [
        dict(r)
        for r in con.execute(
            """
            select *
            from liq_event_features
            where symbol='ETHUSDT' and liq_side='BUY' and cluster_notional>=200000
            order by event_ts_ms
            """
        )
    ]
    con.close()
    return rows


def load_outcome_labels() -> list[dict]:
    con = sqlite3.connect(FEATURE_DB)
    con.row_factory = sqlite3.Row
    rows = [dict(r) for r in con.execute("select * from liq_event_outcome_labels order by event_id, route_id")]
    con.close()
    return rows


def path_marks(con: sqlite3.Connection, symbol: str, entry_ts_ms: int) -> list[tuple[int, float]]:
    """Direct-SQL oracle -- kept unchanged as the parity reference for
    `path_marks_v2` (BATCH-STORAGE-ROTATION-RETENTION-RANGE-READ-
    CONSUMER-MIGRATION-V7). No longer called by main()'s live path."""
    return [
        (int(ts), float(price))
        for ts, price in con.execute(
            """
            select ts_ms, mark_price
            from mark_prices
            where symbol=? and ts_ms>=? and ts_ms<=?
            order by ts_ms
            """,
            (symbol, int(entry_ts_ms), int(entry_ts_ms) + MAX_HORIZON_SEC * 1000),
        )
    ]


def path_marks_v2(root: str, symbol: str, entry_ts_ms: int, source_db_path: str | None = None) -> list[tuple[int, float]]:
    """Reader-backed replacement for `path_marks`, via
    `plan_read`/`execute_read`. `symbol` is a genuine runtime parameter
    in principle, but in this file's only call path it is always
    "ETHUSDT" (the feature-event query is hard-filtered to
    `symbol='ETHUSDT'`) -- real ARCHIVE_ONLY/HYBRID production smoke is
    therefore possible, not merely synthetic.

    Range semantics: the oracle's SQL uses an INCLUSIVE upper bound
    (`ts_ms<=entry_ts_ms+MAX_HORIZON_SEC*1000`); `plan_read`/`execute_read`
    use the reader's half-open `[start_ms, end_ms)` convention. Since
    `ts_ms` is always an integer column, `ts_ms<=end_ms` is exactly
    equivalent to `ts_ms<end_ms+1` -- so `end_ms+1` is passed here to
    reproduce the original inclusive boundary bit-for-bit."""
    start_ms = int(entry_ts_ms)
    end_ms = int(entry_ts_ms) + MAX_HORIZON_SEC * 1000
    plan = RR.plan_read(root, table="mark_prices", symbol=symbol, start_ms=start_ms, end_ms=end_ms + 1)
    result = RR.execute_read(plan, columns=("ts_ms", "mark_price"), source_db_path=source_db_path)
    return [(int(ts), float(price)) for ts, price in result.iter_rows()]


def simulate_standard(
    con: sqlite3.Connection,
    event: dict,
    route: dict,
    *,
    be_bps: float | None,
    tp_bps: float | None = None,
    root: str = "",
    source_db_path: str | None = None,
) -> dict | None:
    direction = str(route["direction"])
    tp = float(route["tp_bps"] if tp_bps is None else tp_bps)
    sl = float(route["sl_bps"])
    entry_target = int(event["event_ts_ms"]) + int(route["entry_delay_sec"]) * 1000
    entry = mark_at(con, str(event["symbol"]), entry_target, before=False)
    if not entry:
        return None
    entry_ts, entry_price = int(entry[0]), float(entry[1])
    marks = path_marks_v2(root, str(event["symbol"]), entry_ts, source_db_path=source_db_path)
    if not marks:
        return None
    be_active = False
    mfe = -1e9
    mae = 1e9
    time_to_mfe = 0.0
    exit_reason = "TIME"
    exit_ts, exit_price = marks[-1]
    for ts, price in marks:
        ret = signed_ret(direction, entry_price, price)
        if ret > mfe:
            mfe = ret
            time_to_mfe = (ts - entry_ts) / 1000.0
        if ret < mae:
            mae = ret
        if be_bps is not None and (not be_active) and ret >= float(be_bps):
            be_active = True
        if ret >= tp:
            exit_reason = "TP"
            exit_ts, exit_price = ts, price
            break
        if ret <= -sl:
            exit_reason = "SL"
            exit_ts, exit_price = ts, price
            break
        if be_active and ret <= 0:
            exit_reason = "BE"
            exit_ts = ts
            exit_price = exit_price_from_signed_ret(direction, entry_price, 0.0)
            break
    return {
        "exit_reason": exit_reason,
        "exit_ts_ms": exit_ts,
        "exit_price": exit_price,
        "gross_bps": signed_ret(direction, entry_price, exit_price),
        "net_bps": net_from_exit(direction, entry_price, exit_price),
        "mfe_bps": mfe,
        "mae_bps": mae,
        "time_to_mfe_sec": time_to_mfe,
    }


def simulate_trailing_half_mfe(con: sqlite3.Connection, event: dict, route: dict, *, arm_bps: float = 30.0,
                                root: str = "", source_db_path: str | None = None) -> dict | None:
    direction = str(route["direction"])
    tp = float(route["tp_bps"])
    sl = float(route["sl_bps"])
    entry_target = int(event["event_ts_ms"]) + int(route["entry_delay_sec"]) * 1000
    entry = mark_at(con, str(event["symbol"]), entry_target, before=False)
    if not entry:
        return None
    entry_ts, entry_price = int(entry[0]), float(entry[1])
    marks = path_marks_v2(root, str(event["symbol"]), entry_ts, source_db_path=source_db_path)
    if not marks:
        return None
    mfe = -1e9
    mae = 1e9
    stop_bps = -sl
    exit_reason = "TIME"
    exit_ts, exit_price = marks[-1]
    for ts, price in marks:
        ret = signed_ret(direction, entry_price, price)
        if ret > mfe:
            mfe = ret
            if mfe >= arm_bps:
                stop_bps = max(stop_bps, mfe * 0.5)
        if ret < mae:
            mae = ret
        if ret >= tp:
            exit_reason = "TP"
            exit_ts, exit_price = ts, price
            break
        if ret <= stop_bps:
            exit_reason = "TRAIL" if stop_bps > 0 else "SL"
            exit_ts = ts
            exit_price = exit_price_from_signed_ret(direction, entry_price, stop_bps)
            break
    return {
        "exit_reason": exit_reason,
        "exit_ts_ms": exit_ts,
        "exit_price": exit_price,
        "gross_bps": signed_ret(direction, entry_price, exit_price),
        "net_bps": net_from_exit(direction, entry_price, exit_price),
        "mfe_bps": mfe,
        "mae_bps": mae,
    }


def simulate_partial_tp60_rest_tp120(con: sqlite3.Connection, event: dict, route: dict,
                                      root: str = "", source_db_path: str | None = None) -> dict | None:
    direction = str(route["direction"])
    sl = float(route["sl_bps"])
    entry_target = int(event["event_ts_ms"]) + int(route["entry_delay_sec"]) * 1000
    entry = mark_at(con, str(event["symbol"]), entry_target, before=False)
    if not entry:
        return None
    entry_ts, entry_price = int(entry[0]), float(entry[1])
    marks = path_marks_v2(root, str(event["symbol"]), entry_ts, source_db_path=source_db_path)
    if not marks:
        return None
    first_exit = None
    be_active_rest = False
    rest_exit = None
    mfe = -1e9
    mae = 1e9
    for ts, price in marks:
        ret = signed_ret(direction, entry_price, price)
        mfe = max(mfe, ret)
        mae = min(mae, ret)
        if first_exit is None:
            if ret >= 60.0:
                first_exit = {"reason": "TP60_HALF", "ts": ts, "gross_bps": 60.0}
                be_active_rest = True
                continue
            if ret <= -sl:
                first_exit = {"reason": "SL_FULL", "ts": ts, "gross_bps": -sl}
                rest_exit = {"reason": "SL_FULL", "ts": ts, "gross_bps": -sl}
                break
        else:
            if ret >= 120.0:
                rest_exit = {"reason": "TP120_REST", "ts": ts, "gross_bps": 120.0}
                break
            if be_active_rest and ret <= 0:
                rest_exit = {"reason": "BE_REST", "ts": ts, "gross_bps": 0.0}
                break
            if ret <= -sl:
                rest_exit = {"reason": "SL_REST", "ts": ts, "gross_bps": -sl}
                break
    if first_exit is None:
        final_ret = signed_ret(direction, entry_price, marks[-1][1])
        first_exit = {"reason": "TIME_FULL", "ts": marks[-1][0], "gross_bps": final_ret}
        rest_exit = {"reason": "TIME_FULL", "ts": marks[-1][0], "gross_bps": final_ret}
    if rest_exit is None:
        final_ret = signed_ret(direction, entry_price, marks[-1][1])
        rest_exit = {"reason": "TIME_REST", "ts": marks[-1][0], "gross_bps": final_ret}
    gross = 0.5 * float(first_exit["gross_bps"]) + 0.5 * float(rest_exit["gross_bps"])
    return {
        "exit_reason": f"{first_exit['reason']}+{rest_exit['reason']}",
        "gross_bps": gross,
        "net_bps": gross - FEE_BPS,
        "mfe_bps": mfe,
        "mae_bps": mae,
    }


def summarize(vals: list[float]) -> dict:
    return {
        "n": len(vals),
        "mean": mean(vals),
        "median": median(vals),
        "cum": sum(vals),
        "positive_rate": sum(1 for v in vals if v > 0) / len(vals) if vals else 0.0,
    }


def main() -> None:
    events = load_feature_events()
    labels = load_outcome_labels()
    event_by_id = {event["event_id"]: event for event in events}
    labels_by_route: dict[str, list[dict]] = defaultdict(list)
    for label in labels:
        labels_by_route[str(label["route_id"])].append(label)

    giveback = {}
    for route_id, rows in labels_by_route.items():
        route_tp = float(rows[0]["tp_bps"]) if rows else 0.0
        hits = [r for r in rows if float(r["mfe_bps"]) >= route_tp * 0.5 and float(r["net_bps"]) < 0.0]
        giveback[route_id] = {
            "n": len(rows),
            "tp_bps": route_tp,
            "giveback_loss_count": len(hits),
            "giveback_loss_rate": len(hits) / len(rows) if rows else 0.0,
            "giveback_loss_cum_bps": sum(float(r["net_bps"]) for r in hits),
            "total_negative_count": sum(1 for r in rows if float(r["net_bps"]) < 0.0),
            "total_negative_cum_bps": sum(float(r["net_bps"]) for r in rows if float(r["net_bps"]) < 0.0),
        }

    root, _ = PR.resolve_production_root()
    source_db_path = str(ROOT / "data" / "microstructure.db")

    source = sqlite3.connect(SOURCE_DB, uri=True)
    be_sweep = defaultdict(dict)
    trailing = {}
    partial = {}
    for route_id, route in ROUTES.items():
        route_events = [event_by_id[row["event_id"]] for row in labels_by_route[route_id] if row["event_id"] in event_by_id]
        no_be = [simulate_standard(source, event, route, be_bps=None, root=root, source_db_path=source_db_path) for event in route_events]
        no_be = [row for row in no_be if row]
        no_be_by_event = {
            event["event_id"]: row
            for event, row in zip(route_events, no_be)
        }
        for be in BE_SWEEP:
            sims = []
            protected = 0
            missed_profit_to_sl = 0
            for event in route_events:
                sim = simulate_standard(source, event, route, be_bps=be, root=root, source_db_path=source_db_path)
                if not sim:
                    continue
                sims.append(sim)
                nob = no_be_by_event.get(event["event_id"])
                if sim["exit_reason"] == "BE" and nob and nob["exit_reason"] == "SL":
                    protected += 1
                if sim["exit_reason"] == "SL" and float(sim["mfe_bps"]) >= 20.0:
                    missed_profit_to_sl += 1
            vals = [float(row["net_bps"]) for row in sims]
            be_sweep[route_id][str(int(be))] = {
                **summarize(vals),
                "exit_counts": dict(Counter(row["exit_reason"] for row in sims)),
                "protected_from_sl_count": protected,
                "missed_profit_to_sl_count": missed_profit_to_sl,
            }
        trail_sims = [simulate_trailing_half_mfe(source, event, route, arm_bps=30.0, root=root, source_db_path=source_db_path) for event in route_events]
        trail_sims = [row for row in trail_sims if row]
        current = [float(row["net_bps"]) for row in labels_by_route[route_id]]
        trail_vals = [float(row["net_bps"]) for row in trail_sims]
        trailing[route_id] = {
            "current_be30": summarize(current),
            "trail_half_mfe_arm30": summarize(trail_vals),
            "delta_cum_bps": sum(trail_vals) - sum(current),
            "better_count": sum(1 for a, b in zip(trail_vals, current) if a > b),
            "worse_count": sum(1 for a, b in zip(trail_vals, current) if a < b),
            "exit_counts": dict(Counter(row["exit_reason"] for row in trail_sims)),
        }
        if route["direction"] == "LONG":
            partial_sims = [simulate_partial_tp60_rest_tp120(source, event, route, root=root, source_db_path=source_db_path) for event in route_events]
            partial_sims = [row for row in partial_sims if row]
            part_vals = [float(row["net_bps"]) for row in partial_sims]
            full_tp60_vals = [
                float(simulate_standard(source, event, {**route, "tp_bps": 60.0}, be_bps=30.0, root=root, source_db_path=source_db_path)["net_bps"])
                for event in route_events
            ]
            full_tp120_vals = [
                float(simulate_standard(source, event, {**route, "tp_bps": 120.0}, be_bps=30.0, root=root, source_db_path=source_db_path)["net_bps"])
                for event in route_events
            ]
            partial[route_id] = {
                "partial_50_tp60_rest_tp120": summarize(part_vals),
                "full_tp60": summarize(full_tp60_vals),
                "full_tp120": summarize(full_tp120_vals),
                "partial_vs_full_tp60_delta_cum_bps": sum(part_vals) - sum(full_tp60_vals),
                "partial_vs_full_tp120_delta_cum_bps": sum(part_vals) - sum(full_tp120_vals),
                "better_than_full_tp60_count": sum(1 for a, b in zip(part_vals, full_tp60_vals) if a > b),
                "worse_than_full_tp60_count": sum(1 for a, b in zip(part_vals, full_tp60_vals) if a < b),
                "better_than_full_tp120_count": sum(1 for a, b in zip(part_vals, full_tp120_vals) if a > b),
                "worse_than_full_tp120_count": sum(1 for a, b in zip(part_vals, full_tp120_vals) if a < b),
                "exit_counts": dict(Counter(row["exit_reason"] for row in partial_sims)),
            }
    source.close()

    payload = {
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "scope": {
            "feature_db": FEATURE_DB,
            "symbol": "ETHUSDT",
            "liq_side": "BUY",
            "event_count": len(events),
            "routes": list(ROUTES),
            "fee_bps": FEE_BPS,
            "definitions": {
                "giveback_loss": "mfe_bps >= 50% of route TP and net_bps < 0",
                "protected_from_sl": "BE exit where no-BE replay would have exited SL",
                "missed_profit_to_sl": "SL exit after seeing at least +20 bps MFE",
                "trail_half_mfe": "after +30 bps MFE, stop trails at 50% of max MFE",
            },
        },
        "giveback": giveback,
        "be_sweep": be_sweep,
        "partial_exit": partial,
        "trailing": trailing,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# S34 Exit Giveback Sweep",
        "",
        f"Generated: {payload['generated_at_utc']}",
        "",
        "Scope: ETHUSDT BUY feature factory outcomes, 450 events, Phase-1 simplified cost model.",
        "",
        "No runner/config changes. This is descriptive + retrospective sweep only.",
        "",
        "## Definitions",
        "",
        "- Giveback loss: MFE reached at least 50% of route TP, but final net_bps < 0.",
        "- Protected from SL: BE exit where the same path without BE would have reached SL.",
        "- Missed profit to SL: SL exit after the path had at least +20 bps MFE.",
        "- Trailing half-MFE: after +30 bps MFE, stop follows at 50% of maximum MFE.",
        "",
        "## 1. Giveback Loss Rate",
        "",
        "| Route | N | TP | Giveback Losses | Rate | Giveback Cum Loss | Total Negative | Total Negative Cum |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for route_id, row in giveback.items():
        lines.append(
            f"| {route_id} | {row['n']} | {row['tp_bps']:.0f} | {row['giveback_loss_count']} | "
            f"{pct(row['giveback_loss_rate'])} | {row['giveback_loss_cum_bps']:+.2f} | "
            f"{row['total_negative_count']} | {row['total_negative_cum_bps']:+.2f} |"
        )
    lines.extend(
        [
            "",
            "## 2. BE Threshold Sweep",
            "",
            "| Route | BE | N | Cum | Mean | Median | WR | TP/BE/SL/TIME | Protected From SL | Missed Profit To SL |",
            "|---|---:|---:|---:|---:|---:|---:|---|---:|---:|",
        ]
    )
    for route_id, rows in be_sweep.items():
        for be, row in rows.items():
            exits = row["exit_counts"]
            lines.append(
                f"| {route_id} | {be} | {row['n']} | {row['cum']:+.2f} | {row['mean']:+.2f} | "
                f"{row['median']:+.2f} | {pct(row['positive_rate'])} | "
                f"{exits.get('TP',0)}/{exits.get('BE',0)}/{exits.get('SL',0)}/{exits.get('TIME',0)} | "
                f"{row['protected_from_sl_count']} | {row['missed_profit_to_sl_count']} |"
            )
    lines.extend(
        [
            "",
            "## 3. Partial Exit Sweep",
            "",
            "| Route | Scenario | N | Cum | Mean | Median | WR | Delta vs Full TP60 | Delta vs Full TP120 | Better/Worse vs TP60 | Better/Worse vs TP120 |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for route_id, row in partial.items():
        part = row["partial_50_tp60_rest_tp120"]
        lines.append(
            f"| {route_id} | 50% TP60 + 50% TP120/BE | {part['n']} | {part['cum']:+.2f} | {part['mean']:+.2f} | "
            f"{part['median']:+.2f} | {pct(part['positive_rate'])} | "
            f"{row['partial_vs_full_tp60_delta_cum_bps']:+.2f} | {row['partial_vs_full_tp120_delta_cum_bps']:+.2f} | "
            f"{row['better_than_full_tp60_count']}/{row['worse_than_full_tp60_count']} | "
            f"{row['better_than_full_tp120_count']}/{row['worse_than_full_tp120_count']} |"
        )
    lines.extend(
        [
            "",
            "## 4. Trailing Half-MFE Sweep",
            "",
            "| Route | Current Cum | Trailing Cum | Delta | Current Median | Trailing Median | Better/Worse Count | Trailing Exits |",
            "|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for route_id, row in trailing.items():
        current = row["current_be30"]
        trail = row["trail_half_mfe_arm30"]
        exits = row["exit_counts"]
        lines.append(
            f"| {route_id} | {current['cum']:+.2f} | {trail['cum']:+.2f} | {row['delta_cum_bps']:+.2f} | "
            f"{current['median']:+.2f} | {trail['median']:+.2f} | {row['better_count']}/{row['worse_count']} | "
            f"{exits} |"
        )
    lines.extend(
        [
            "",
            "## Honest Read",
            "",
            "These are retrospective sweeps on the same sample used to inspect the issue. They are useful for identifying failure modes, not for changing the live runner directly.",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUT_MD)
    print(json.dumps({k: payload[k] for k in ["giveback", "partial_exit", "trailing"]}, indent=2)[:6000])


if __name__ == "__main__":
    main()
