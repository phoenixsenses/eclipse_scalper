"""Executable check for ETH pre-liquidation book-pressure scalp.

Research only. Future ETH SELL liquidation clusters are used as labels. Entry
is before the liquidation event, using only prior bookTicker state. Fills use
the shadow runner's real bookTicker taker fill path. No runner/config changes.
"""

from __future__ import annotations

import json
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
OUT_JSON = ROOT / "reports" / "research" / "s34" / "S34_ETH_PRELIQ_EXECUTABLE.json"
OUT_MD = ROOT / "reports" / "research" / "s34" / "S34_ETH_PRELIQ_EXECUTABLE.md"

PRESSURES = [
    {"name": "lead5_lb10_down5", "lead_sec": 5, "lookback_sec": 10, "move_bps": 5.0, "imb_min": None, "imb_max": None, "max_spread_bps": 1.0},
    {"name": "lead5_lb10_down3_bidheavy", "lead_sec": 5, "lookback_sec": 10, "move_bps": 3.0, "imb_min": 0.2, "imb_max": None, "max_spread_bps": 1.0},
    {"name": "lead5_lb10_down3_askheavy", "lead_sec": 5, "lookback_sec": 10, "move_bps": 3.0, "imb_min": None, "imb_max": -0.2, "max_spread_bps": 1.0},
]

THRESHOLDS = [500_000.0, 1_000_000.0]
TP_GRID = [10.0, 15.0, 20.0]
SL_GRID = [10.0, 15.0, 20.0]
BE_GRID = [6.0, 10.0]
HOLD_GRID = [60, 120]


def _iso(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).isoformat()


def _day(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).date().isoformat()


def _fmt(v: Any, digits: int = 1) -> str:
    if v is None:
        return "-"
    return f"{float(v):+.{digits}f}"


def _pct(v: Any) -> str:
    if v is None:
        return "-"
    return f"{float(v) * 100:.0f}%"


_BOOK_COLS = ("ts_ms", "bid_price", "bid_qty", "ask_price", "ask_qty", "mid_price", "spread_pct", "book_imbalance")


def _book_at_or_before(con: sqlite3.Connection, ts_ms: int) -> sqlite3.Row | None:
    """Direct-SQL oracle -- kept unchanged as the parity reference for
    `_book_at_or_before_v2` (BATCH-STORAGE-ROTATION-RETENTION-ASOF-LOOKUP-
    CONSUMER-MIGRATION-V8). No longer called by main(); the reader-backed
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
    lookup_latest_at_or_before. Symbol is hardcoded ETHUSDT directly in the
    oracle's SQL (there is no symbol parameter at all in this function --
    it never varies). book_ticker has no ETHUSDT archive partition (only
    SOLUSDT/2026-04 is archived), so real production use of this file
    resolves SQLITE_ONLY -- confirmed, not assumed. Returns a dict keyed
    identically to the oracle Row's selected columns, so downstream
    `row["col"]` access is unchanged."""
    result = RR.lookup_latest_at_or_before(
        root, table="book_ticker", symbol="ETHUSDT", ts_ms=int(ts_ms),
        columns=_BOOK_COLS, source_db_path=source_db_path)
    if not result.found:
        return None
    return dict(zip(_BOOK_COLS, result.row))


def _book_series(con: sqlite3.Connection, start_ms: int, end_ms: int) -> list[sqlite3.Row]:
    # OUT-OF-SCOPE for ASOF V8: this is a bounded range read (BETWEEN, ORDER
    # BY ts_ms ASC), not the point-lookup semantics the helper migrates.
    # Left on the direct-SQL path deliberately; belongs to the range-read
    # migration track.
    return con.execute(
        """
        SELECT ts_ms, bid_price, bid_qty, ask_price, ask_qty, mid_price, spread_pct, book_imbalance
        FROM book_ticker INDEXED BY idx_bt_symbol_ts
        WHERE symbol='ETHUSDT' AND ts_ms>=? AND ts_ms<=?
        ORDER BY ts_ms ASC
        """,
        (int(start_ms), int(end_ms)),
    ).fetchall()


def _rule(name: str, tp: float, sl: float, be: float, hold: int) -> S34Rule:
    return S34Rule(
        name=name,
        symbol="ETHUSDT",
        liq_side="SELL",
        direction="SHORT",
        threshold_usd=500_000.0,
        bucket_sec=300,
        min_gap_sec=900,
        tp_bps=float(tp),
        sl_bps=float(sl),
        be_trigger_bps=float(be),
        max_horizon_sec=int(hold),
        use_global_regime=False,
        require_book_ticker_fill=True,
    )


def _load_events(ff: sqlite3.Connection, threshold: float) -> list[sqlite3.Row]:
    return ff.execute(
        """
        SELECT event_id, event_ts_ms, event_utc, cluster_notional, cluster_liq_count,
               max_single_liq_share, day_trend_bps, day_range_bps
        FROM liq_event_features
        WHERE symbol='ETHUSDT' AND liq_side='SELL' AND cluster_notional>=?
        ORDER BY event_ts_ms ASC
        """,
        (float(threshold),),
    ).fetchall()


def _entry_signal(
    con: sqlite3.Connection,
    event: sqlite3.Row,
    pressure: dict[str, Any],
    rule: S34Rule,
    *,
    root,
    source_db_path=None,
) -> dict[str, Any] | None:
    event_ts = int(event["event_ts_ms"])
    entry_target_ts = event_ts - int(pressure["lead_sec"]) * 1000
    start_ts = entry_target_ts - int(pressure["lookback_sec"]) * 1000
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
    imb = float(end["book_imbalance"])
    if move_bps < float(pressure["move_bps"]):
        return None
    if pressure.get("imb_min") is not None and imb < float(pressure["imb_min"]):
        return None
    if pressure.get("imb_max") is not None and imb > float(pressure["imb_max"]):
        return None
    if spread_bps > float(pressure["max_spread_bps"]):
        return None

    try:
        fill = _fill_quote(con, rule, "ETHUSDT", int(end["ts_ms"]), entry_mid, "SHORT", "ENTRY", "taker")
    except RuntimeError as exc:
        return {"fill_error": str(exc)}

    return {
        "signal_key": f"PRELIQ:{event['event_id']}:{pressure['name']}",
        "bucket": int(event_ts // 1000),
        "symbol": "ETHUSDT",
        "liq_side": "SELL",
        "direction": "SHORT",
        "ts_ms": event_ts,
        "ts_utc": event["event_utc"],
        "signal_ts_ms": event_ts,
        "signal_ts_utc": event["event_utc"],
        "entry_ts_ms": int(fill["ts_ms"]),
        "entry_ts_utc": _iso(int(fill["ts_ms"])),
        "entry_reference_price": entry_mid,
        "entry_price": float(fill["fill_price"]),
        "entry_fill": fill,
        "fill_error": "",
        "mark_ts_ms": int(end["ts_ms"]),
        "mark_ts_utc": _iso(int(end["ts_ms"])),
        "cluster_notional": float(event["cluster_notional"]),
        "cluster_liq_count": event["cluster_liq_count"],
        "day_trend_bps": event["day_trend_bps"],
        "day_range_bps": event["day_range_bps"],
        "preliq_pressure": pressure["name"],
        "preliq_move_bps": move_bps,
        "preliq_spread_bps": spread_bps,
        "preliq_book_imbalance": imb,
        "entry_lead_sec": (event_ts - int(fill["ts_ms"])) / 1000.0,
    }


def _eval(con: sqlite3.Connection, trade: dict[str, Any], hold: int) -> dict[str, Any]:
    entry_ms = int(trade["entry_ts_ms"])
    rows = _book_series(con, entry_ms + 1, entry_ms + hold * 1000)
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


def _simulate(
    con: sqlite3.Connection,
    events: list[sqlite3.Row],
    pressure: dict[str, Any],
    rule: S34Rule,
    *,
    root,
    source_db_path=None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows: list[dict[str, Any]] = []
    counts = {"events": len(events), "filtered": 0, "no_fill": 0, "closed": 0}
    for event in events:
        sig = _entry_signal(con, event, pressure, rule, root=root, source_db_path=source_db_path)
        if sig is None:
            counts["filtered"] += 1
            continue
        if sig.get("fill_error"):
            counts["no_fill"] += 1
            continue
        trade = _paper_trade_from_signal(rule, sig, RiskConfig())
        try:
            out = _eval(con, trade, int(rule.max_horizon_sec))
        except RuntimeError as exc:
            if "no_fill_data" in str(exc):
                counts["no_fill"] += 1
                continue
            raise
        if out.get("status") == "CLOSED":
            out.update(
                {
                    "preliq_pressure": sig["preliq_pressure"],
                    "preliq_move_bps": sig["preliq_move_bps"],
                    "preliq_book_imbalance": sig["preliq_book_imbalance"],
                    "cluster_notional": sig["cluster_notional"],
                    "cluster_liq_count": sig["cluster_liq_count"],
                    "day_trend_bps": sig["day_trend_bps"],
                    "day_range_bps": sig["day_range_bps"],
                }
            )
            rows.append(out)
            counts["closed"] += 1
    return rows, counts


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    vals = [float(row["net_bps"]) for row in rows if row.get("net_bps") is not None]
    if not vals:
        return {"n": 0, "median": None, "mean": None, "cum": 0.0, "wr": None}
    days: dict[str, float] = defaultdict(float)
    exits: dict[str, int] = defaultdict(int)
    holds: list[float] = []
    moves: list[float] = []
    imbs: list[float] = []
    adverse: list[float] = []
    for row in rows:
        days[_day(int(row["signal_ts_ms"]))] += float(row["net_bps"])
        exits[str(row.get("exit_reason") or "?")] += 1
        holds.append((int(row["exit_ts_ms"]) - int(row["entry_ts_ms"])) / 1000.0)
        moves.append(float(row["preliq_move_bps"]))
        imbs.append(float(row["preliq_book_imbalance"]))
        if row.get("mark_to_fill_cost_bps") is not None:
            adverse.append(float(row["mark_to_fill_cost_bps"]))
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
        "mean_preliq_move_bps": statistics.mean(moves) if moves else None,
        "mean_book_imbalance": statistics.mean(imbs) if imbs else None,
        "mean_mark_to_fill_cost_bps": statistics.mean(adverse) if adverse else None,
        "exit_counts": dict(exits),
    }


def _verdict(s: dict[str, Any], counts: dict[str, int]) -> str:
    n = int(s.get("n") or 0)
    if n < 25:
        return "thin"
    if (s.get("median") or 0) <= 0:
        return "reject_negative_median"
    if (s.get("top3_removed_cum") or 0) <= 0:
        return "reject_outlier_dependent"
    if s.get("total_days") and s.get("positive_days") / s.get("total_days") < 0.6:
        return "watch_day_consistency"
    if counts["events"] and counts["closed"] / counts["events"] < 0.2:
        return "watch_too_selective"
    return "candidate"


def main() -> None:
    if not FEATURE_DB.exists():
        raise SystemExit(f"missing {FEATURE_DB}")
    con = sqlite3.connect(MICRO_DB, uri=True, timeout=60)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA query_only=1")
    ff = sqlite3.connect(FEATURE_DB)
    ff.row_factory = sqlite3.Row
    root, _ = PR.resolve_production_root()
    results: list[dict[str, Any]] = []
    for threshold in THRESHOLDS:
        events = _load_events(ff, threshold)
        print(f"[threshold] ETH SELL {threshold:,.0f} events={len(events)}", flush=True)
        for pressure in PRESSURES:
            for tp in TP_GRID:
                for sl in SL_GRID:
                    for be in BE_GRID:
                        for hold in HOLD_GRID:
                            rule = _rule(f"ETH_PRELIQ_{int(threshold)}", tp, sl, be, hold)
                            rows, counts = _simulate(con, events, pressure, rule, root=root, source_db_path=SOURCE_DB_PATH)
                            s = _summary(rows)
                            results.append(
                                {
                                    "threshold": threshold,
                                    "pressure": pressure,
                                    "tp_bps": tp,
                                    "sl_bps": sl,
                                    "be_bps": be,
                                    "max_hold_sec": hold,
                                    "counts": counts,
                                    "summary": s,
                                    "verdict": _verdict(s, counts),
                                }
                            )
    ff.close()
    con.close()
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope": "ETH SELL pre-liquidation executable book-pressure check",
        "results": results,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    best = sorted(
        results,
        key=lambda r: (
            r["verdict"] == "candidate",
            float(r["summary"].get("median") or -9999),
            float(r["summary"].get("top3_removed_cum") or -999999),
        ),
        reverse=True,
    )
    lines = [
        "# S34 ETH Pre-Liquidation Executable Check",
        "",
        f"Generated: `{payload['generated_at']}`",
        "",
        "Research only. Future liquidation clusters are labels; entry uses only prior bookTicker state. Fills use real bid/ask bookTicker.",
        "",
        "## Top 30 Rows",
        "",
        "| Threshold | Pressure | TP/SL/BE/H | Events | Closed | Filtered | No-fill | Median | Mean | WR | Cum | T3R | Pos days | Avg hold | Pre-move | Imb | Adverse | Exits | Verdict |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for r in best[:30]:
        s = r["summary"]
        c = r["counts"]
        exits = " ".join(f"{k}={v}" for k, v in sorted((s.get("exit_counts") or {}).items()))
        lines.append(
            "| "
            + " | ".join(
                [
                    f"{r['threshold']/1000:.0f}K",
                    r["pressure"]["name"],
                    f"TP{int(r['tp_bps'])}/SL{int(r['sl_bps'])}/BE{int(r['be_bps'])}/H{int(r['max_hold_sec'])}",
                    str(c["events"]),
                    str(c["closed"]),
                    str(c["filtered"]),
                    str(c["no_fill"]),
                    _fmt(s.get("median")),
                    _fmt(s.get("mean")),
                    _pct(s.get("wr")),
                    _fmt(s.get("cum"), 0),
                    _fmt(s.get("top3_removed_cum"), 0),
                    f"{s.get('positive_days')}/{s.get('total_days')}",
                    "-" if s.get("avg_hold_sec") is None else f"{s['avg_hold_sec']:.0f}s",
                    _fmt(s.get("mean_preliq_move_bps")),
                    _fmt(s.get("mean_book_imbalance"), 2),
                    _fmt(s.get("mean_mark_to_fill_cost_bps")),
                    exits,
                    r["verdict"],
                ]
            )
            + " |"
        )
    lines += [
        "",
        "## Best Per Threshold",
        "",
        "| Threshold | Pressure | TP/SL/BE/H | Closed | Median | WR | T3R | Pos days | Verdict |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for threshold in THRESHOLDS:
        rows = [r for r in best if r["threshold"] == threshold]
        if not rows:
            continue
        r = rows[0]
        s = r["summary"]
        lines.append(
            f"| {threshold/1000:.0f}K | {r['pressure']['name']} | TP{int(r['tp_bps'])}/SL{int(r['sl_bps'])}/BE{int(r['be_bps'])}/H{int(r['max_hold_sec'])} | {s.get('n')} | {_fmt(s.get('median'))} | {_pct(s.get('wr'))} | {_fmt(s.get('top3_removed_cum'),0)} | {s.get('positive_days')}/{s.get('total_days')} | {r['verdict']} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "- A `candidate` here is not deployable by itself because the sample is conditioned on future liquidation labels.",
        "- A robust candidate would justify building a continuous pre-liq book-pressure detector and forward logging it.",
        "- If rows are only thin or outlier-dependent, the idea stays research-only.",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT_MD}")


if __name__ == "__main__":
    main()
