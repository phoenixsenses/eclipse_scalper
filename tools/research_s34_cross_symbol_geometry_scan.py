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
OUT_JSON = ROOT / "reports" / "research" / "s34" / "S34_CROSS_SYMBOL_GEOMETRY_SCAN.json"
OUT_MD = ROOT / "reports" / "research" / "s34" / "S34_CROSS_SYMBOL_GEOMETRY_SCAN.md"

LOOKBACK_DAYS = 120
SIGNAL_LIMIT = 100_000
MAX_HORIZON_SEC = 3600
BUCKET_SEC = 300
MIN_GAP_SEC = 900
TAKER_FEE_BPS = 4.0
ROUND_TRIP_FEE_BPS = 8.0
MAX_BOOK_STALENESS_SEC = 5

SCOPES = [
    {
        "name": "SOL_200K_TP60_SL40_BE30",
        "symbol": "SOLUSDT",
        "threshold": 200_000.0,
        "tp": 60.0,
        "sl": 40.0,
        "be": 30.0,
    },
    {
        "name": "BTC_1M_TP60_SL40_BE30",
        "symbol": "BTCUSDT",
        "threshold": 1_000_000.0,
        "tp": 60.0,
        "sl": 40.0,
        "be": 30.0,
    },
    {
        "name": "BTC_1M_TP60_SL30_BE30",
        "symbol": "BTCUSDT",
        "threshold": 1_000_000.0,
        "tp": 60.0,
        "sl": 30.0,
        "be": 30.0,
    },
]


def median(vals: list[float]) -> float | None:
    return statistics.median(vals) if vals else None


def quantile(vals: list[float], q: float) -> float:
    vals = sorted(vals)
    if not vals:
        return 0.0
    pos = (len(vals) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(vals) - 1)
    frac = pos - lo
    return vals[lo] * (1 - frac) + vals[hi] * frac


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


def mark_at(con: sqlite3.Connection, symbol: str, ts_ms: int):
    return con.execute(
        """
        SELECT ts_ms, mark_price
        FROM mark_prices
        WHERE symbol=? AND ts_ms>=?
        ORDER BY ts_ms ASC
        LIMIT 1
        """,
        (symbol, int(ts_ms)),
    ).fetchone()


def path_marks(con: sqlite3.Connection, symbol: str, entry_ts_ms: int) -> list[tuple[int, float]]:
    return [
        (int(ts), float(price))
        for ts, price in con.execute(
            """
            SELECT ts_ms, mark_price
            FROM mark_prices
            WHERE symbol=? AND ts_ms>=? AND ts_ms<=?
            ORDER BY ts_ms ASC
            """,
            (symbol, int(entry_ts_ms), int(entry_ts_ms) + MAX_HORIZON_SEC * 1000),
        )
    ]


def book_ticker_at(con: sqlite3.Connection, symbol: str, ts_ms: int):
    row = con.execute(
        """
        SELECT ts_ms, bid_price, ask_price, mid_price
        FROM book_ticker
        WHERE symbol=? AND ts_ms<=?
        ORDER BY ts_ms DESC
        LIMIT 1
        """,
        (symbol, int(ts_ms)),
    ).fetchone()
    if not row:
        return None
    if int(ts_ms) - int(row[0]) > MAX_BOOK_STALENESS_SEC * 1000:
        return None
    return {"ts_ms": int(row[0]), "bid": float(row[1]), "ask": float(row[2]), "mid": float(row[3])}


def real_fill_net(con: sqlite3.Connection, row: dict[str, Any]) -> dict[str, Any] | None:
    symbol = row["symbol"]
    entry_book = book_ticker_at(con, symbol, int(row["entry_ts_ms"]))
    exit_book = book_ticker_at(con, symbol, int(row["exit_ts_ms"]))
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


def simulate_path(scope: dict[str, Any], event: dict[str, Any], marks: list[tuple[int, float]]) -> dict[str, Any] | None:
    if not marks:
        return None
    tp = float(scope["tp"])
    sl = float(scope["sl"])
    be = float(scope["be"])
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
    duration = float(event.get("cluster_duration_sec") or 0.0)
    notional = float(event.get("liq_total_notional") or 0.0)
    max_notional = float(event.get("liq_max_notional") or 0.0)
    return {
        "scope": scope["name"],
        "symbol": scope["symbol"],
        "event_id": event["event_id"],
        "event_ts_ms": int(event["ts_ms"]),
        "event_utc": event["ts_utc"],
        "day": iso_day(int(event["ts_ms"])),
        "liq_total_notional": notional,
        "liq_count": int(event.get("liq_count") or 0),
        "cluster_duration_sec": duration,
        "max_single_liq_share": (max_notional / notional * 100.0) if notional > 0 else 0.0,
        "intensity_per_sec": notional / max(duration, 1.0),
        "inter_cluster_gap_sec": event.get("inter_cluster_gap_sec"),
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


def build_predicates(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    counts = [float(r["liq_count"]) for r in rows]
    durations = [float(r["cluster_duration_sec"]) for r in rows]
    intensities = [float(r["intensity_per_sec"]) for r in rows]
    gaps = [float(r["inter_cluster_gap_sec"]) for r in rows if r.get("inter_cluster_gap_sec") is not None]
    preds = [
        {"label": "max_share_ge_80", "feature": "max_single_liq_share", "op": ">=", "value": 80.0},
        {"label": "max_share_lt_50", "feature": "max_single_liq_share", "op": "<", "value": 50.0},
        {"label": f"liq_count_ge_p75_{quantile(counts, 0.75):.0f}", "feature": "liq_count", "op": ">=", "value": quantile(counts, 0.75)},
        {"label": f"liq_count_le_p25_{quantile(counts, 0.25):.0f}", "feature": "liq_count", "op": "<=", "value": quantile(counts, 0.25)},
        {"label": f"duration_ge_p75_{quantile(durations, 0.75):.0f}s", "feature": "cluster_duration_sec", "op": ">=", "value": quantile(durations, 0.75)},
        {"label": f"intensity_ge_p75_{quantile(intensities, 0.75):.0f}", "feature": "intensity_per_sec", "op": ">=", "value": quantile(intensities, 0.75)},
        {"label": f"intensity_le_p25_{quantile(intensities, 0.25):.0f}", "feature": "intensity_per_sec", "op": "<=", "value": quantile(intensities, 0.25)},
    ]
    if gaps:
        preds.extend(
            [
                {"label": "gap_lt_30m", "feature": "inter_cluster_gap_sec", "op": "<", "value": 1800.0},
                {"label": "gap_ge_2h", "feature": "inter_cluster_gap_sec", "op": ">=", "value": 7200.0},
            ]
        )
    for label in ("single_dominant_80pct", "stretched_120s", "distributed_mid_duration"):
        preds.append({"label": f"shape_{label}", "feature": "cluster_shape_label", "op": "==", "value": label})
    return preds


def matches(row: dict[str, Any], pred: dict[str, Any]) -> bool:
    val = row.get(pred["feature"])
    if val is None:
        return False
    op = pred["op"]
    target = pred["value"]
    if op == "==":
        return str(val) == str(target)
    val_f = float(val)
    target_f = float(target)
    if op == ">=":
        return val_f >= target_f
    if op == "<=":
        return val_f <= target_f
    if op == "<":
        return val_f < target_f
    raise ValueError(op)


def apply_candidate(rows: list[dict[str, Any]], preds: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if all(matches(row, pred) for pred in preds)]


def split_rows(rows: list[dict[str, Any]], split_ts_ms: int) -> dict[str, list[dict[str, Any]]]:
    return {
        "train": [row for row in rows if int(row["event_ts_ms"]) <= split_ts_ms],
        "test": [row for row in rows if int(row["event_ts_ms"]) > split_ts_ms],
        "all": rows,
    }


def load_events(con: sqlite3.Connection, scope: dict[str, Any], start_ms: int, end_ms: int) -> list[dict[str, Any]]:
    rule = S34Rule(
        name=scope["name"],
        symbol=scope["symbol"],
        liq_side="BUY",
        direction="LONG",
        threshold_usd=float(scope["threshold"]),
        bucket_sec=BUCKET_SEC,
        min_gap_sec=MIN_GAP_SEC,
        use_global_regime=False,
    )
    events = []
    prev_ts = None
    for idx, signal in enumerate(_bucket_events(con, rule, start_ms, end_ms, SIGNAL_LIMIT), 1):
        if signal.get("fill_error"):
            continue
        row = dict(signal)
        row["event_id"] = f"{scope['name']}:{idx}"
        row["inter_cluster_gap_sec"] = None if prev_ts is None else (int(row["ts_ms"]) - int(prev_ts)) / 1000.0
        prev_ts = int(row["ts_ms"])
        events.append(row)
    return events


def evaluate_scope(con: sqlite3.Connection, scope: dict[str, Any], start_ms: int, end_ms: int) -> dict[str, Any]:
    events = load_events(con, scope, start_ms, end_ms)
    rows = []
    for event in events:
        entry = mark_at(con, scope["symbol"], int(event["entry_ts_ms"]))
        if not entry:
            continue
        path = path_marks(con, scope["symbol"], int(entry[0]))
        row = simulate_path(scope, event, path)
        if row:
            rows.append(row)
    split_ts_ms = rows[len(rows) // 2]["event_ts_ms"] if rows else end_ms
    baseline_periods = split_rows(rows, int(split_ts_ms))
    baseline = {k: summarize(v) for k, v in baseline_periods.items()}
    predicates = build_predicates(rows)
    candidates = [{"label": pred["label"], "predicates": [pred]} for pred in predicates]
    for left, right in itertools.combinations(predicates, 2):
        if left["feature"] == right["feature"]:
            continue
        candidates.append({"label": f"{left['label']} AND {right['label']}", "predicates": [left, right]})

    scored = []
    for candidate in candidates:
        selected = apply_candidate(rows, candidate["predicates"])
        periods = split_rows(selected, int(split_ts_ms))
        train = summarize(periods["train"])
        test = summarize(periods["test"])
        if train["n"] < 5 or test["n"] < 5:
            continue
        scored.append(
            {
                "label": candidate["label"],
                "predicates": candidate["predicates"],
                "train": train,
                "test": test,
                "all": summarize(selected),
            }
        )
    scored.sort(
        key=lambda c: (
            c["train"]["median"] is not None and c["train"]["median"] > 0,
            c["test"]["median"] if c["test"]["median"] is not None else -1e9,
            c["test"]["top3_removed_cum"],
            c["test"]["mean"] if c["test"]["mean"] is not None else -1e9,
        ),
        reverse=True,
    )
    top5 = scored[:5]
    real_fill = {}
    for candidate in top5:
        selected = apply_candidate(rows, candidate["predicates"])
        filled = []
        no_fill = 0
        for row in selected:
            rf = real_fill_net(con, row)
            if rf:
                filled.append(rf)
            else:
                no_fill += 1
        periods = split_rows(filled, int(split_ts_ms))
        real_fill[candidate["label"]] = {
            "total_rows": len(selected),
            "real_fill_rows": len(filled),
            "no_fill_rows": no_fill,
            "no_fill_rate": no_fill / len(selected) if selected else None,
            "train": summarize(periods["train"], key="real_net_bps"),
            "test": summarize(periods["test"], key="real_net_bps"),
            "all": summarize(periods["all"], key="real_net_bps"),
        }
    baseline_filled = []
    baseline_no_fill = 0
    for row in rows:
        rf = real_fill_net(con, row)
        if rf:
            baseline_filled.append(rf)
        else:
            baseline_no_fill += 1
    baseline_real_periods = split_rows(baseline_filled, int(split_ts_ms))
    return {
        "scope": scope,
        "events": len(events),
        "simulated_rows": len(rows),
        "split_ts_ms": int(split_ts_ms),
        "split_utc": iso_ts(int(split_ts_ms)),
        "baseline": baseline,
        "baseline_real_fill": {
            "total_rows": len(rows),
            "real_fill_rows": len(baseline_filled),
            "no_fill_rows": baseline_no_fill,
            "no_fill_rate": baseline_no_fill / len(rows) if rows else None,
            "train": summarize(baseline_real_periods["train"], key="real_net_bps"),
            "test": summarize(baseline_real_periods["test"], key="real_net_bps"),
            "all": summarize(baseline_filled, key="real_net_bps"),
        },
        "top5": top5,
        "real_fill": real_fill,
    }


def fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "NA"
    return f"{float(value):+.{digits}f}"


def main() -> int:
    con = sqlite3.connect(SOURCE_DB, uri=True, timeout=30)
    con.execute("PRAGMA query_only=1")
    max_ts = con.execute("SELECT MAX(ts_ms) FROM liquidations").fetchone()[0]
    if not max_ts:
        raise RuntimeError("no liquidation rows")
    end_ms = int(max_ts)
    start_ms = end_ms - LOOKBACK_DAYS * 24 * 3600 * 1000
    scopes = [evaluate_scope(con, scope, start_ms, end_ms) for scope in SCOPES]
    con.close()

    payload = {
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "start_utc": iso_ts(start_ms),
        "end_utc": iso_ts(end_ms),
        "lookback_days": LOOKBACK_DAYS,
        "scopes": scopes,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# S34 Cross-Symbol Geometry Scan",
        "",
        f"Generated: `{payload['generated_at_utc']}`",
        "",
        "Scope: geometry filters for current/new BUY-liq continuation candidates. Research-only; live runner/config unchanged.",
        "",
        "## Baselines",
        "",
        "| Scope | Rows | Real Fill | No Fill | Test N | Test Median | Test Mean | Test Cum | Test Top3 Removed | Test Positive Days | Exits |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for scope in scopes:
        base = scope["baseline_real_fill"]
        test = base["test"]
        lines.append(
            f"| {scope['scope']['name']} | {base['total_rows']} | {base['real_fill_rows']} | "
            f"{base['no_fill_rows']} ({(base['no_fill_rate'] or 0)*100:.1f}%) | {test['n']} | "
            f"{fmt(test['median'])} | {fmt(test['mean'])} | {fmt(test['cum'])} | "
            f"{fmt(test['top3_removed_cum'])} | {test['positive_days']}/{test['days']} | {test['exit_counts']} |"
        )
    lines.extend(
        [
            "",
            "## Top Geometry Candidates",
            "",
            "| Scope | Rank | Candidate | Train N | Train Median | Test N | Test Median | Test Mean | Test Cum | Test Top3 Removed |",
            "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for scope in scopes:
        for idx, candidate in enumerate(scope["top5"], 1):
            lines.append(
                f"| {scope['scope']['name']} | {idx} | {candidate['label']} | "
                f"{candidate['train']['n']} | {fmt(candidate['train']['median'])} | "
                f"{candidate['test']['n']} | {fmt(candidate['test']['median'])} | "
                f"{fmt(candidate['test']['mean'])} | {fmt(candidate['test']['cum'])} | "
                f"{fmt(candidate['test']['top3_removed_cum'])} |"
            )
    lines.extend(
        [
            "",
            "## Real-Fill Parity For Top Geometry Candidates",
            "",
            "| Scope | Candidate | Total | Real Fill | No Fill | Test N | Test Median | Test Mean | Test Cum | Test Top3 Removed | Test Positive Days |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for scope in scopes:
        for candidate in scope["top5"]:
            rf = scope["real_fill"][candidate["label"]]
            test = rf["test"]
            lines.append(
                f"| {scope['scope']['name']} | {candidate['label']} | {rf['total_rows']} | {rf['real_fill_rows']} | "
                f"{rf['no_fill_rows']} ({(rf['no_fill_rate'] or 0)*100:.1f}%) | {test['n']} | "
                f"{fmt(test['median'])} | {fmt(test['mean'])} | {fmt(test['cum'])} | "
                f"{fmt(test['top3_removed_cum'])} | {test['positive_days']}/{test['days']} |"
            )
    lines.extend(
        [
            "",
            "## Read",
            "",
            "These filters are retrospective geometry screens. Use them to decide whether a narrower exploratory variant deserves pre-registration; do not mutate existing live rules directly from this report.",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUT_MD)
    print(OUT_JSON)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
