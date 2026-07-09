"""Control test for ETH pre-liquidation book-pressure scalp.

Research only. Compares the ETH pre-liq book-pressure pocket against matched
non-liquidation control timestamps. The point is to separate a tradable
continuous book-pressure edge from a future-label artifact.
"""

from __future__ import annotations

import json
import random
import sqlite3
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.s34_shadow_paper_runner import RiskConfig, S34Rule, _close_trade, _fill_quote, _paper_trade_from_signal

from ami.storage import production as PR
from ami.storage import research_reader as RR

MICRO_DB = f"file:{(ROOT / 'data' / 'microstructure.db').as_posix()}?mode=ro"
SOURCE_DB_PATH = (ROOT / "data" / "microstructure.db").as_posix()
FEATURE_DB = ROOT / "data" / "s34_feature_factory.db"
OUT_JSON = ROOT / "reports" / "research" / "s34" / "S34_ETH_PRELIQ_CONTROL.json"
OUT_MD = ROOT / "reports" / "research" / "s34" / "S34_ETH_PRELIQ_CONTROL.md"

PRESSURE = {"name": "lead5_lb10_down5", "lead_sec": 5, "lookback_sec": 10, "move_bps": 5.0, "max_spread_bps": 1.0}
ROUTES = [
    {"name": "TP15_SL20_BE10_H60", "tp": 15.0, "sl": 20.0, "be": 10.0, "hold": 60},
    {"name": "TP20_SL20_BE10_H60", "tp": 20.0, "sl": 20.0, "be": 10.0, "hold": 60},
]
THRESHOLDS = [500_000.0, 1_000_000.0]
CONTROL_MULTIPLE = 5
RANDOM_SEED = 34027
EXCLUDE_NEAR_LIQ_SEC = 900


def _iso(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).isoformat()


def _day(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).date().isoformat()


def _fmt(v: Any, digits: int = 1) -> str:
    return "-" if v is None else f"{float(v):+.{digits}f}"


def _pct(v: Any) -> str:
    return "-" if v is None else f"{float(v) * 100:.0f}%"


_BOOK_COLS = ("ts_ms", "bid_price", "bid_qty", "ask_price", "ask_qty", "mid_price", "spread_pct", "book_imbalance")


def _book_at_or_before(con: sqlite3.Connection, ts_ms: int) -> sqlite3.Row | None:
    """Direct-SQL oracle -- kept unchanged as the parity reference for
    `_book_at_or_before_v2` (BATCH-STORAGE-ROTATION-RETENTION-ASOF-LOOKUP-
    CONSUMER-MIGRATION-V6). No longer called by main(); the reader-backed
    path is used instead."""
    return con.execute(
        """
        SELECT ts_ms, bid_price, bid_qty, ask_price, ask_qty, mid_price, spread_pct, book_imbalance
        FROM book_ticker INDEXED BY idx_bt_symbol_ts
        WHERE symbol='ETHUSDT' AND ts_ms<=?
        ORDER BY ts_ms DESC
        LIMIT 1
        """,
        (int(ts_ms),),
    ).fetchone()


def _book_at_or_before_v2(root, ts_ms: int, source_db_path=None) -> dict[str, float] | None:
    """Reader-backed replacement for `_book_at_or_before`, via
    lookup_latest_at_or_before. Symbol is hardcoded ETHUSDT, exactly as in
    the oracle's SQL (this file never varies it). book_ticker has no
    ETHUSDT archive partition (only SOLUSDT/2026-04 is archived), so real
    production use of this file resolves SQLITE_ONLY -- confirmed, not
    assumed. Returns a dict keyed identically to the oracle Row's selected
    columns, so downstream `row["col"]` access is unchanged."""
    result = RR.lookup_latest_at_or_before(
        root, table="book_ticker", symbol="ETHUSDT", ts_ms=int(ts_ms),
        columns=_BOOK_COLS, source_db_path=source_db_path)
    if not result.found:
        return None
    return dict(zip(_BOOK_COLS, result.row))


def _book_series(con: sqlite3.Connection, start_ms: int, end_ms: int) -> list[sqlite3.Row]:
    # OUT-OF-SCOPE for ASOF V6: this is a bounded range read (BETWEEN, ORDER
    # BY ts_ms ASC), not the `ORDER BY ts_ms DESC LIMIT 1` point-lookup the
    # helper migrates. Left on the direct-SQL path deliberately; belongs to
    # the range-read migration track.
    return con.execute(
        """
        SELECT ts_ms, bid_price, bid_qty, ask_price, ask_qty, mid_price, spread_pct, book_imbalance
        FROM book_ticker INDEXED BY idx_bt_symbol_ts
        WHERE symbol='ETHUSDT' AND ts_ms>=? AND ts_ms<=?
        ORDER BY ts_ms ASC
        """,
        (int(start_ms), int(end_ms)),
    ).fetchall()


def _rule(route: dict[str, Any]) -> S34Rule:
    return S34Rule(
        name=f"ETH_PRELIQ_CONTROL_{route['name']}",
        symbol="ETHUSDT",
        liq_side="SELL",
        direction="SHORT",
        threshold_usd=500_000.0,
        bucket_sec=300,
        min_gap_sec=900,
        tp_bps=float(route["tp"]),
        sl_bps=float(route["sl"]),
        be_trigger_bps=float(route["be"]),
        max_horizon_sec=int(route["hold"]),
        use_global_regime=False,
        require_book_ticker_fill=True,
    )


def _pressure_pass(root, entry_target_ts: int, source_db_path=None) -> dict[str, Any] | None:
    start_ts = entry_target_ts - int(PRESSURE["lookback_sec"]) * 1000
    start = _book_at_or_before_v2(root, start_ts, source_db_path=source_db_path)
    end = _book_at_or_before_v2(root, entry_target_ts, source_db_path=source_db_path)
    if start is None or end is None:
        return None
    if abs(start_ts - int(start["ts_ms"])) > 1000 or abs(entry_target_ts - int(end["ts_ms"])) > 1000:
        return None
    start_mid = float(start["mid_price"])
    entry_mid = float(end["mid_price"])
    move_bps = (start_mid - entry_mid) / start_mid * 10_000.0
    spread_bps = float(end["spread_pct"]) * 10_000.0
    if move_bps < float(PRESSURE["move_bps"]) or spread_bps > float(PRESSURE["max_spread_bps"]):
        return None
    return {
        "entry_ts_ms": int(end["ts_ms"]),
        "entry_reference_price": entry_mid,
        "pre_move_bps": move_bps,
        "spread_bps": spread_bps,
        "book_imbalance": float(end["book_imbalance"]),
    }


def _make_signal(base: dict[str, Any], rule: S34Rule, con: sqlite3.Connection) -> dict[str, Any] | None:
    try:
        fill = _fill_quote(
            con,
            rule,
            "ETHUSDT",
            int(base["entry_ts_ms"]),
            float(base["entry_reference_price"]),
            "SHORT",
            "ENTRY",
            "taker",
        )
    except RuntimeError:
        return None
    signal_ts = int(base.get("label_ts_ms") or base["entry_ts_ms"])
    return {
        "signal_key": str(base["key"]),
        "bucket": int(signal_ts // 1000),
        "symbol": "ETHUSDT",
        "liq_side": "SELL",
        "direction": "SHORT",
        "ts_ms": signal_ts,
        "ts_utc": _iso(signal_ts),
        "signal_ts_ms": signal_ts,
        "signal_ts_utc": _iso(signal_ts),
        "entry_ts_ms": int(fill["ts_ms"]),
        "entry_ts_utc": _iso(int(fill["ts_ms"])),
        "mark_ts_ms": int(base["entry_ts_ms"]),
        "mark_ts_utc": _iso(int(base["entry_ts_ms"])),
        "entry_reference_price": float(base["entry_reference_price"]),
        "entry_price": float(fill["fill_price"]),
        "entry_fill": fill,
        "fill_error": "",
        "pre_move_bps": float(base["pre_move_bps"]),
        "book_imbalance": float(base["book_imbalance"]),
        "sample_type": str(base["sample_type"]),
    }


def _eval_trade(con: sqlite3.Connection, trade: dict[str, Any], hold: int) -> dict[str, Any]:
    entry_ms = int(trade["entry_ts_ms"])
    rows = _book_series(con, entry_ms + 1, entry_ms + int(hold) * 1000)
    if not rows:
        return trade
    for row in rows:
        ts = int(row["ts_ms"])
        mid = float(row["mid_price"])
        if not trade["be_active"] and mid <= float(trade["be_trigger_price"]):
            trade["be_active"] = True
            trade["be_activated_ts_ms"] = ts
            trade["be_activated_ts_utc"] = _iso(ts)
        if mid <= float(trade["tp_price"]):
            return _close_trade(con, trade, ts, mid, "TP")
        stop = float(trade.get("entry_reference_price") or trade["entry_price"]) if trade["be_active"] else float(trade["sl_price"])
        if mid >= stop:
            return _close_trade(con, trade, ts, mid, "BE" if trade["be_active"] else "SL")
    row = rows[-1]
    return _close_trade(con, trade, int(row["ts_ms"]), float(row["mid_price"]), "TIME")


def _simulate(con: sqlite3.Connection, samples: list[dict[str, Any]], route: dict[str, Any]) -> list[dict[str, Any]]:
    rule = _rule(route)
    rows = []
    for base in samples:
        sig = _make_signal(base, rule, con)
        if sig is None:
            continue
        trade = _paper_trade_from_signal(rule, sig, RiskConfig())
        try:
            out = _eval_trade(con, trade, int(route["hold"]))
        except RuntimeError as exc:
            if "no_fill_data" in str(exc):
                continue
            raise
        if out.get("status") == "CLOSED":
            out["sample_type"] = base["sample_type"]
            out["pre_move_bps"] = base["pre_move_bps"]
            out["book_imbalance"] = base["book_imbalance"]
            rows.append(out)
    return rows


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    vals = [float(r["net_bps"]) for r in rows]
    if not vals:
        return {"n": 0, "median": None, "mean": None, "cum": 0.0, "wr": None}
    days = defaultdict(float)
    exits = defaultdict(int)
    holds = []
    moves = []
    imbs = []
    for row in rows:
        days[_day(int(row["signal_ts_ms"]))] += float(row["net_bps"])
        exits[str(row.get("exit_reason") or "?")] += 1
        holds.append((int(row["exit_ts_ms"]) - int(row["entry_ts_ms"])) / 1000.0)
        moves.append(float(row["pre_move_bps"]))
        imbs.append(float(row["book_imbalance"]))
    sorted_vals = sorted(vals, reverse=True)
    return {
        "n": len(vals),
        "median": statistics.median(vals),
        "mean": statistics.mean(vals),
        "cum": sum(vals),
        "wr": sum(v > 0 for v in vals) / len(vals),
        "top3_removed_cum": sum(sorted_vals[3:]) if len(sorted_vals) > 3 else 0.0,
        "positive_days": sum(v > 0 for v in days.values()),
        "total_days": len(days),
        "avg_hold_sec": statistics.mean(holds) if holds else None,
        "mean_pre_move_bps": statistics.mean(moves) if moves else None,
        "mean_book_imbalance": statistics.mean(imbs) if imbs else None,
        "exit_counts": dict(exits),
    }


def _load_preliq_samples(root, ff: sqlite3.Connection, threshold: float, source_db_path=None) -> list[dict[str, Any]]:
    samples = []
    events = ff.execute(
        """
        SELECT event_id, event_ts_ms
        FROM liq_event_features
        WHERE symbol='ETHUSDT' AND liq_side='SELL' AND cluster_notional>=?
        ORDER BY event_ts_ms
        """,
        (float(threshold),),
    ).fetchall()
    for event in events:
        entry_target = int(event["event_ts_ms"]) - 5_000
        p = _pressure_pass(root, entry_target, source_db_path=source_db_path)
        if p is None:
            continue
        p.update({"key": f"preliq:{threshold}:{event['event_id']}", "sample_type": f"preliq_{int(threshold/1000)}K", "label_ts_ms": int(event["event_ts_ms"])})
        samples.append(p)
    return samples


def _load_control_samples(root, ff: sqlite3.Connection, target_n: int, min_ts: int, max_ts: int, source_db_path=None) -> list[dict[str, Any]]:
    rng = random.Random(RANDOM_SEED)
    # Exclude times too close to known ETH SELL liquidation clusters, otherwise
    # controls accidentally include the label events.
    event_ts = [
        int(r[0])
        for r in ff.execute(
            "SELECT event_ts_ms FROM liq_event_features WHERE symbol='ETHUSDT' AND liq_side='SELL' ORDER BY event_ts_ms"
        ).fetchall()
    ]
    controls = []
    attempts = 0
    max_attempts = max(10_000, target_n * 500)
    while len(controls) < target_n and attempts < max_attempts:
        attempts += 1
        ts = rng.randrange(min_ts + 120_000, max_ts - 120_000)
        if any(abs(ts - e) <= EXCLUDE_NEAR_LIQ_SEC * 1000 for e in event_ts):
            continue
        p = _pressure_pass(root, ts, source_db_path=source_db_path)
        if p is None:
            continue
        p.update({"key": f"control:{len(controls)}:{ts}", "sample_type": "control", "label_ts_ms": ts})
        controls.append(p)
    return controls


def main() -> None:
    root, _ = PR.resolve_production_root()
    # `micro` stays for the still-direct range/fill helpers (_book_series,
    # _fill_quote, _close_trade); only the point-lookup path moved to the
    # reader (via SOURCE_DB_PATH), so no book_ticker point read runs on it.
    micro = sqlite3.connect(MICRO_DB, uri=True, timeout=60)
    micro.row_factory = sqlite3.Row
    micro.execute("PRAGMA query_only=1")
    ff = sqlite3.connect(FEATURE_DB)
    ff.row_factory = sqlite3.Row

    min_ts, max_ts = ff.execute(
        "SELECT MIN(event_ts_ms), MAX(event_ts_ms) FROM liq_event_features WHERE symbol='ETHUSDT'"
    ).fetchone()

    preliq_by_threshold = {thr: _load_preliq_samples(root, ff, thr, source_db_path=SOURCE_DB_PATH) for thr in THRESHOLDS}
    total_preliq = sum(len(v) for v in preliq_by_threshold.values())
    controls = _load_control_samples(root, ff, max(total_preliq * CONTROL_MULTIPLE, 100), int(min_ts), int(max_ts), source_db_path=SOURCE_DB_PATH)

    results = []
    for route in ROUTES:
        control_rows = _simulate(micro, controls, route)
        results.append({"sample": "control", "route": route, "raw_samples": len(controls), "closed": len(control_rows), "summary": _summary(control_rows)})
        for thr, samples in preliq_by_threshold.items():
            rows = _simulate(micro, samples, route)
            results.append({"sample": f"preliq_{int(thr/1000)}K", "route": route, "raw_samples": len(samples), "closed": len(rows), "summary": _summary(rows)})

    ff.close()
    micro.close()
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "pressure": PRESSURE,
        "control_multiple": CONTROL_MULTIPLE,
        "exclude_near_liq_sec": EXCLUDE_NEAR_LIQ_SEC,
        "preliq_sample_counts": {str(int(k)): len(v) for k, v in preliq_by_threshold.items()},
        "control_count": len(controls),
        "results": results,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# S34 ETH Pre-Liq Book Pressure Control Test",
        "",
        f"Generated: `{payload['generated_at']}`",
        "",
        "Research only. Tests whether the pre-liq book-pressure pocket exists outside future liquidation labels.",
        "",
        f"Pressure: `{PRESSURE['name']}`. Control timestamps exclude +/-{EXCLUDE_NEAR_LIQ_SEC}s around ETH SELL liq events.",
        "",
        "## Results",
        "",
        "| Sample | Route | Raw samples | Closed | Median | Mean | WR | Cum | T3R | Pos days | Avg hold | Pre-move | Imb | Exits |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for item in results:
        s = item["summary"]
        exits = " ".join(f"{k}={v}" for k, v in sorted((s.get("exit_counts") or {}).items()))
        lines.append(
            "| "
            + " | ".join(
                [
                    item["sample"],
                    item["route"]["name"],
                    str(item["raw_samples"]),
                    str(item["closed"]),
                    _fmt(s.get("median")),
                    _fmt(s.get("mean")),
                    _pct(s.get("wr")),
                    _fmt(s.get("cum"), 0),
                    _fmt(s.get("top3_removed_cum"), 0),
                    f"{s.get('positive_days')}/{s.get('total_days')}",
                    "-" if s.get("avg_hold_sec") is None else f"{s['avg_hold_sec']:.0f}s",
                    _fmt(s.get("mean_pre_move_bps")),
                    _fmt(s.get("mean_book_imbalance"), 2),
                    exits,
                ]
            )
            + " |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "- If control performance is similar to pre-liq, the edge may be a standalone book-pressure alpha.",
        "- If control is much worse, the result depends on knowing a future liquidation cluster and is not directly tradable.",
        "- This does not change runner/live rules.",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT_MD}")


if __name__ == "__main__":
    main()
