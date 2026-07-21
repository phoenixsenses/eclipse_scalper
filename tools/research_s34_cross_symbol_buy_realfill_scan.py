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

from tools.s34_shadow_paper_runner import (
    RiskConfig,
    S34Rule,
    _bucket_events,
    _evaluate_trade,
    _paper_trade_from_signal,
)


SOURCE_DB = f"file:{(ROOT / 'data' / 'microstructure.db').as_posix()}?mode=ro"
OUT_JSON = ROOT / "reports" / "research" / "s34" / "S34_CROSS_SYMBOL_BUY_REALFILL_SCAN.json"
OUT_MD = ROOT / "reports" / "research" / "s34" / "S34_CROSS_SYMBOL_BUY_REALFILL_SCAN.md"

SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
THRESHOLDS = (50_000.0, 100_000.0, 200_000.0, 500_000.0, 1_000_000.0)
TP_BPS = 60.0
SL_BPS = 40.0
BE_BPS = 30.0
MAX_HORIZON_SEC = 3600
BUCKET_SEC = 300
MIN_GAP_SEC = 900
LOOKBACK_DAYS = 120
SIGNAL_LIMIT = 100_000


def _utc_iso(ts_ms: int) -> str:
    return datetime.fromtimestamp(int(ts_ms) / 1000.0, tz=timezone.utc).isoformat()


def _day(ts_ms: int) -> str:
    return datetime.fromtimestamp(int(ts_ms) / 1000.0, tz=timezone.utc).date().isoformat()


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    return float(statistics.median(values))


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    vals = [float(row["net_bps"]) for row in rows if row.get("net_bps") is not None]
    day_cums: dict[str, float] = defaultdict(float)
    for row in rows:
        if row.get("net_bps") is not None:
            day_cums[_day(int(row["signal_ts_ms"]))] += float(row["net_bps"])
    if not vals:
        return {
            "n": 0,
            "days": 0,
            "cum": 0.0,
            "mean": None,
            "median": None,
            "wr": None,
            "top3_removed": 0.0,
            "positive_days": 0,
            "exit_counts": {},
            "mean_entry_adverse": None,
            "median_entry_adverse": None,
            "mean_fill_penalty": None,
        }
    entry_adv = [float(row.get("entry_adverse_bps") or 0.0) for row in rows]
    fill_penalty = [
        float(row.get("entry_adverse_bps") or 0.0)
        + float(row.get("exit_adverse_bps") or 0.0)
        + float(row.get("spread_cost_bps") or 0.0)
        + float(row.get("fee_cost_bps") or 0.0)
        for row in rows
    ]
    return {
        "n": len(vals),
        "days": len(day_cums),
        "cum": sum(vals),
        "mean": statistics.mean(vals),
        "median": _median(vals),
        "wr": sum(v > 0 for v in vals) / len(vals),
        "top3_removed": sum(sorted(vals, reverse=True)[3:]) if len(vals) > 3 else 0.0,
        "positive_days": sum(v > 0 for v in day_cums.values()),
        "exit_counts": dict(sorted((reason, sum(1 for row in rows if row.get("exit_reason") == reason)) for reason in {row.get("exit_reason") for row in rows})),
        "mean_entry_adverse": statistics.mean(entry_adv),
        "median_entry_adverse": _median(entry_adv),
        "mean_fill_penalty": statistics.mean(fill_penalty),
    }


def _evaluate_signals(conn: sqlite3.Connection, rule: S34Rule, start_ms: int, end_ms: int) -> dict[str, Any]:
    signals = _bucket_events(conn, rule, start_ms, end_ms, SIGNAL_LIMIT)
    rows: list[dict[str, Any]] = []
    no_fill = 0
    incomplete = 0
    for signal in signals:
        if signal.get("fill_error"):
            no_fill += 1
            continue
        trade = _paper_trade_from_signal(rule, signal, RiskConfig())
        try:
            evaluated = _evaluate_trade(conn, trade, min(end_ms, int(signal["entry_ts_ms"]) + MAX_HORIZON_SEC * 1000))
        except RuntimeError as exc:
            if "no_fill_data" in str(exc):
                no_fill += 1
                continue
            raise
        if evaluated.get("status") != "CLOSED":
            incomplete += 1
            continue
        rows.append(evaluated)
    split_ts = signals[len(signals) // 2]["ts_ms"] if signals else None
    first_half = [row for row in rows if split_ts is not None and int(row["signal_ts_ms"]) < int(split_ts)]
    second_half = [row for row in rows if split_ts is not None and int(row["signal_ts_ms"]) >= int(split_ts)]
    return {
        "total_signals": len(signals),
        "real_closed": len(rows),
        "no_fill": no_fill,
        "incomplete": incomplete,
        "no_fill_rate": no_fill / len(signals) if signals else None,
        "split_ts_ms": split_ts,
        "split_utc": _utc_iso(split_ts) if split_ts is not None else None,
        "all": _summarize(rows),
        "first_half": _summarize(first_half),
        "second_half": _summarize(second_half),
        "trades": [
            {
                "signal_ts_ms": int(row["signal_ts_ms"]),
                "signal_utc": row.get("signal_ts_utc"),
                "exit_ts_ms": int(row.get("exit_ts_ms") or 0),
                "exit_utc": row.get("exit_ts_utc"),
                "exit_reason": row.get("exit_reason"),
                "net_bps": row.get("net_bps"),
                "gross_bps": row.get("gross_bps"),
                "entry_adverse_bps": row.get("entry_adverse_bps"),
                "exit_adverse_bps": row.get("exit_adverse_bps"),
                "spread_cost_bps": row.get("spread_cost_bps"),
                "fee_cost_bps": row.get("fee_cost_bps"),
                "liq_total_notional": (row.get("signal") or {}).get("liq_total_notional"),
                "liq_count": (row.get("signal") or {}).get("liq_count"),
                "cluster_duration_sec": (row.get("signal") or {}).get("cluster_duration_sec"),
                "cluster_shape_label": (row.get("signal") or {}).get("cluster_shape_label"),
            }
            for row in rows
        ],
    }


def _fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:+.{digits}f}" if value < 0 else f"{value:.{digits}f}"
    return str(value)


def _pct(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value) * 100:.1f}%"


def _render_table(rows: list[list[Any]]) -> list[str]:
    if not rows:
        return []
    header = rows[0]
    out = [
        "| " + " | ".join(str(x) for x in header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    for row in rows[1:]:
        out.append("| " + " | ".join(str(x) for x in row) + " |")
    return out


def main() -> int:
    conn = sqlite3.connect(SOURCE_DB, uri=True, timeout=30)
    conn.execute("PRAGMA query_only=1")
    max_ts = conn.execute("SELECT MAX(ts_ms) FROM liquidations").fetchone()[0]
    if not max_ts:
        raise RuntimeError("liquidations table has no rows")
    end_ms = int(max_ts)
    start_ms = end_ms - LOOKBACK_DAYS * 24 * 3600 * 1000

    results: list[dict[str, Any]] = []
    for symbol in SYMBOLS:
        for threshold in THRESHOLDS:
            rule = S34Rule(
                name=f"{symbol}_BUY_LIQ_LONG_{int(threshold)}_TP60_SL40_BE30_RESEARCH",
                symbol=symbol,
                liq_side="BUY",
                direction="LONG",
                threshold_usd=threshold,
                bucket_sec=BUCKET_SEC,
                min_gap_sec=MIN_GAP_SEC,
                tp_bps=TP_BPS,
                sl_bps=SL_BPS,
                be_trigger_bps=BE_BPS,
                max_horizon_sec=MAX_HORIZON_SEC,
                use_global_regime=False,
            )
            outcome = _evaluate_signals(conn, rule, start_ms, end_ms)
            results.append(
                {
                    "symbol": symbol,
                    "threshold_usd": threshold,
                    "rule": rule.name,
                    **outcome,
                }
            )

    conn.close()
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_db": SOURCE_DB,
        "lookback_days": LOOKBACK_DAYS,
        "start_utc": _utc_iso(start_ms),
        "end_utc": _utc_iso(end_ms),
        "route": {
            "liq_side": "BUY",
            "direction": "LONG",
            "tp_bps": TP_BPS,
            "sl_bps": SL_BPS,
            "be_trigger_bps": BE_BPS,
            "bucket_sec": BUCKET_SEC,
            "min_gap_sec": MIN_GAP_SEC,
            "max_horizon_sec": MAX_HORIZON_SEC,
        },
        "results": results,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    rows = [["Symbol", "Threshold", "Signals", "Real", "No Fill", "No Fill %", "Median", "Mean", "Cum", "WR", "Days", "Second Median", "Second Cum", "Top3 Removed"]]
    for row in results:
        all_s = row["all"]
        second = row["second_half"]
        rows.append(
            [
                row["symbol"],
                f"{row['threshold_usd'] / 1000:.0f}K",
                row["total_signals"],
                row["real_closed"],
                row["no_fill"],
                _pct(row["no_fill_rate"]),
                _fmt(all_s["median"]),
                _fmt(all_s["mean"]),
                _fmt(all_s["cum"]),
                _pct(all_s["wr"]),
                all_s["days"],
                _fmt(second["median"]),
                _fmt(second["cum"]),
                _fmt(all_s["top3_removed"]),
            ]
        )

    candidates = sorted(
        [
            row
            for row in results
            if row["all"]["n"] >= 25
            and row["second_half"]["n"] >= 10
            and (row["all"]["median"] or -10**9) > 0
            and (row["second_half"]["median"] or -10**9) > 0
            and (row["all"]["top3_removed"] or -10**9) > 0
        ],
        key=lambda row: (
            row["second_half"]["median"] or -10**9,
            row["all"]["median"] or -10**9,
            row["all"]["n"],
        ),
        reverse=True,
    )
    cand_rows = [["Rank", "Symbol", "Threshold", "N", "Median", "Second N", "Second Median", "Second Cum", "No Fill", "Reason"]]
    for idx, row in enumerate(candidates[:8], 1):
        cand_rows.append(
            [
                idx,
                row["symbol"],
                f"{row['threshold_usd'] / 1000:.0f}K",
                row["all"]["n"],
                _fmt(row["all"]["median"]),
                row["second_half"]["n"],
                _fmt(row["second_half"]["median"]),
                _fmt(row["second_half"]["cum"]),
                _pct(row["no_fill_rate"]),
                "passes min-N + positive median split screen",
            ]
        )

    lines = [
        "# S34 Cross-Symbol BUY-Liq Real-Fill Scan",
        "",
        f"Generated: `{payload['generated_at_utc']}`",
        f"Window: `{payload['start_utc']}` to `{payload['end_utc']}` ({LOOKBACK_DAYS}d lookback)",
        "",
        "Scope: BUY liquidation cluster -> LONG, real historical bookTicker entry/exit fills, TP60/SL40/BE30. No live runner or config changes.",
        "",
        "## Full Grid",
        "",
        *_render_table(rows),
        "",
        "## Candidate Screen",
        "",
        "Screen: N>=25, second-half N>=10, full median >0, second-half median >0, top3-removed cumulative >0.",
        "",
        *_render_table(cand_rows),
        "",
        "## Notes",
        "",
        "- This is research-only and not a paper-runner change.",
        "- Second-half split is chronological by detected signal order, not randomized.",
        "- No-fill rows are excluded from real-fill stats; high no-fill remains a selection-bias risk.",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {OUT_MD}")
    print(f"Wrote {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
