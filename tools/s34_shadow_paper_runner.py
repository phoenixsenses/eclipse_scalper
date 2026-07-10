"""Paper-only S34 shadow runner over restored liquidation data.

This tool reads local DB rows, creates synthetic paper trades, and writes
evidence files. It never places orders and never calls Binance.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sqlite3
import sys
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import s34_intelligence_ledger as intelligence_ledger
from tools.s34_feature_availability import assert_feature_set_available, signal_entry_features

# OD-018 (2026-07-10) min-gap semantics v2: the per-rule timestamp of the last
# EMITTED signal is persisted in the runner state file and seeded back into
# _bucket_events, so live loop cycles, restarts, and backfill enforce the same
# independent-cycle gap (parity proven in tests/test_s34_shadow_paper_min_gap_parity.py).
# Pre-v2 live results were produced under invocation-local min-gap semantics —
# do not pool signal populations across the 2026-07-10 boundary (forward N=0).
# Emission consumes the gap regardless of downstream fate (regime/governance/
# risk/no-fill), keeping the reference data-deterministic and replay-safe.
MIN_GAP_SEMANTICS_VERSION = "persistent-v2"

# OD-018-FOLLOWUP (2026-07-10, independent-review corrective): first-activation
# migration. On the very first v2 run for a given rule, last_signal_ts_ms is
# derived once from the EXISTING trade/signal history rather than assumed
# absent (-inf), which would otherwise let one post-restart signal falsely
# pass the 900s gate if the true last pre-v2 emission's bucket has already
# scrolled behind the persisted cursor (reproduced in
# tests/test_s34_shadow_paper_min_gap_migration.py). The trade store is a
# complete emission log under normal operation: every candidate _bucket_events
# returns (after its own internal min-gap check) is persisted exactly once as
# an OPEN/CLOSED/SKIPPED trade record before any regime/governance/risk/fill
# gate runs (see the candidate loop in run_once) -- so max(signal_ts_ms) per
# exact rule.name is a faithful reconstruction of _bucket_events' internal
# last_signal_ms at the moment v2 activates. This assumes the trade store has
# not been truncated/hand-edited since inception (governed by the standing
# no-silent-data-deletion rule). A rule whose history shows the SAME name
# under a DIFFERENT (symbol, threshold_usd, liq_side) identity is treated as
# ambiguous and fails closed: no new signal is emitted for that rule until an
# operator resolves it manually.
MIN_GAP_STATE_MIGRATION_VERSION = "v1-derived-from-trade-history"


@dataclass(frozen=True)
class S34Rule:
    name: str
    symbol: str = "ETHUSDT"
    liq_side: str = "BUY"
    direction: str = "LONG"
    threshold_usd: float = 50_000.0
    bucket_sec: int = 300
    min_gap_sec: int = 900
    tp_bps: float = 120.0
    sl_bps: float = 40.0
    be_trigger_bps: float = 30.0
    max_horizon_sec: int = 3600
    taker_fee_bps: float = 4.0
    maker_fee_bps: float = 2.0
    tp_fill_mode: str = "taker"
    modeled_spread_bps: float = 0.0
    max_book_staleness_sec: int = 5
    require_book_ticker_fill: bool = True
    max_open_trades: int = 1
    daily_max_sl: int = 3
    entry_delay_sec: int = 0
    btc_confirm_symbol: str = "BTCUSDT"
    btc_pre_window_sec: int = 0
    btc_pre_min_return_bps: float | None = None
    use_global_regime: bool = True
    min_day_trend_bps: float | None = None
    max_day_trend_bps: float | None = None
    min_cluster_liq_count: int | None = None
    required_shape_label: str | None = None
    max_single_liq_share_pct: float | None = None
    priority: int = 100


@dataclass(frozen=True)
class RiskConfig:
    simulated_equity_usdt: float = 100.0
    leverage: float = 10.0
    risk_per_trade_pct: float = 0.25
    max_open_trades: int = 1
    daily_max_loss_pct: float = 1.0
    daily_max_sl: int = 3
    cooldown_after_consecutive_sl: int = 2
    cooldown_hours: float = 6.0
    max_mark_staleness_sec: int = 30


@dataclass(frozen=True)
class RegimeConfig:
    enabled: bool = False
    min_trend_pct: float = 1.0
    min_range_pct: float = 2.5
    min_buy_liq_notional: float = 5_000_000.0
    min_agg_trade_count: int = 250_000


DEFAULT_RULES: tuple[S34Rule, ...] = (
    S34Rule(name="ETH_BUY_LIQ_LONG_50K_TP120_SL40_BE30", threshold_usd=50_000.0, tp_bps=120.0, priority=50),
    S34Rule(name="ETH_BUY_LIQ_LONG_200K_TP60_SL40_BE30", threshold_usd=200_000.0, tp_bps=60.0, priority=30),
    S34Rule(
        name="ETH_BUY_LIQ_LONG_200K_BTC_PRE15_TP120_SL40_BE30_DELAY60",
        threshold_usd=200_000.0,
        tp_bps=120.0,
        entry_delay_sec=60,
        btc_pre_window_sec=900,
        btc_pre_min_return_bps=0.0,
        priority=40,
    ),
    S34Rule(
        name="ETH_BUY_LIQ_LONG_500K_DAYTREND0_TP60_SL40_BE30",
        threshold_usd=500_000.0,
        tp_bps=60.0,
        use_global_regime=False,
        min_day_trend_bps=0.0,
        priority=10,
    ),
    S34Rule(
        name="ETH_BUY_LIQ_LONG_500K_NEGTREND_STRETCHED_TP60_SL40_BE30",
        threshold_usd=500_000.0,
        tp_bps=60.0,
        use_global_regime=False,
        max_day_trend_bps=0.0,
        required_shape_label="stretched_120s",
        priority=20,
    ),
    S34Rule(
        name="SOL_BUY_LIQ_LONG_100K_TP60_SL40_BE30",
        symbol="SOLUSDT",
        threshold_usd=100_000.0,
        tp_bps=60.0,
        use_global_regime=False,
        priority=15,
    ),
    S34Rule(
        name="SOL_BUY_LIQ_LONG_200K_TP60_SL40_BE30",
        symbol="SOLUSDT",
        threshold_usd=200_000.0,
        tp_bps=60.0,
        use_global_regime=False,
        priority=10,
    ),
    S34Rule(
        name="BTC_BUY_LIQ_LONG_1M_DISTRIBUTED_TP60_SL30_BE30",
        symbol="BTCUSDT",
        threshold_usd=1_000_000.0,
        tp_bps=60.0,
        sl_bps=30.0,
        use_global_regime=False,
        max_single_liq_share_pct=50.0,
        priority=10,
    ),
    S34Rule(
        name="ETH_SELL_LIQ_SHORT_500K_TP60_SL40_BE40",
        liq_side="SELL",
        direction="SHORT",
        threshold_usd=500_000.0,
        tp_bps=60.0,
        sl_bps=40.0,
        be_trigger_bps=40.0,
        use_global_regime=False,
        priority=10,
    ),
    S34Rule(
        name="ETH_SELL_LIQ_SHORT_1M_TP80_SL40_BE40",
        liq_side="SELL",
        direction="SHORT",
        threshold_usd=1_000_000.0,
        tp_bps=80.0,
        sl_bps=40.0,
        be_trigger_bps=40.0,
        use_global_regime=False,
        priority=8,
    ),
    S34Rule(
        name="SOL_SELL_LIQ_SHORT_200K_TP60_SL30_BE30",
        symbol="SOLUSDT",
        liq_side="SELL",
        direction="SHORT",
        threshold_usd=200_000.0,
        tp_bps=60.0,
        sl_bps=30.0,
        be_trigger_bps=30.0,
        use_global_regime=False,
        priority=10,
    ),
    S34Rule(
        name="SOL_SELL_LIQ_SHORT_100K_TP60_SL30_BE40",
        symbol="SOLUSDT",
        liq_side="SELL",
        direction="SHORT",
        threshold_usd=100_000.0,
        tp_bps=60.0,
        sl_bps=30.0,
        be_trigger_bps=40.0,
        use_global_regime=False,
        priority=15,
    ),
)


DEFAULT_RISK = RiskConfig()
DEFAULT_REGIME = RegimeConfig()
DEPRECATED_PAPER_RULES = frozenset(
    {
        "ETH_BUY_LIQ_LONG_50K_TP120_SL40_BE30",
        "ETH_BUY_LIQ_LONG_500K_DAYTREND0_TP60_SL40_BE30",
    }
)
DEPRECATED_PAPER_RULE_REASON = "ARCHIVED_CONTAMINATED_LOOKAHEAD_ANCHOR"
DEPRECATED_PAPER_RULE_NOTE = (
    "BUY continuation rules 50K/TP120 and 500K/daytrend remain logged for "
    "signal/prediction audit, but are blocked from opening new shadow paper "
    "positions after feature-availability and anchor-integrity contamination."
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_print_summary(summary: dict[str, Any]) -> None:
    try:
        stream = getattr(sys, "stdout", None)
        if stream is None or getattr(stream, "closed", False):
            return
        print(json.dumps(summary, sort_keys=True))
    except (AttributeError, BrokenPipeError, OSError, ValueError):
        return


def _deprecated_paper_rule_reason(rule: S34Rule) -> str | None:
    if rule.name in DEPRECATED_PAPER_RULES:
        return DEPRECATED_PAPER_RULE_REASON
    return None


def _iso_from_ms(ts_ms: int) -> str:
    return datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc).isoformat()


def _ms_from_iso(value: str) -> int:
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    return int(datetime.fromisoformat(text).timestamp() * 1000)


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, events: list[dict[str, Any]]) -> None:
    if not events:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for event in events:
            fh.write(json.dumps(event, ensure_ascii=True, separators=(",", ":")) + "\n")


def _mark_at(conn: sqlite3.Connection, symbol: str, ts_ms: int, *, before: bool) -> tuple[int, float] | None:
    op = "<=" if before else ">="
    order = "DESC" if before else "ASC"
    row = conn.execute(
        f"""
        SELECT ts_ms, mark_price FROM mark_prices
        WHERE symbol=? AND ts_ms {op} ?
        ORDER BY ts_ms {order}
        LIMIT 1
        """,
        (symbol, int(ts_ms)),
    ).fetchone()
    if not row:
        return None
    return int(row[0]), float(row[1])


def _latest_mark_ts(conn: sqlite3.Connection, symbol: str) -> int:
    row = conn.execute("SELECT MAX(ts_ms) FROM mark_prices WHERE symbol=?", (symbol,)).fetchone()
    return int(row[0] or 0)


def _latest_liq_ts(conn: sqlite3.Connection, symbol: str) -> int:
    row = conn.execute("SELECT MAX(ts_ms) FROM liquidations WHERE symbol=?", (symbol,)).fetchone()
    return int(row[0] or 0)


def _mark_return_bps(
    conn: sqlite3.Connection,
    symbol: str,
    start_ts_ms: int,
    end_ts_ms: int,
) -> tuple[float, int, int] | None:
    start = _mark_at(conn, symbol, int(start_ts_ms), before=True)
    end = _mark_at(conn, symbol, int(end_ts_ms), before=True)
    if not start or not end:
        return None
    start_mark_ts, start_px = start
    end_mark_ts, end_px = end
    if float(start_px) <= 0:
        return None
    return ((float(end_px) - float(start_px)) / float(start_px) * 1e4, int(start_mark_ts), int(end_mark_ts))


def _pnl_bps(direction: str, entry: float, px: float) -> float:
    raw = (px - entry) / entry * 1e4
    return raw if direction.upper() == "LONG" else -raw


def _book_ticker_at(conn: sqlite3.Connection, symbol: str, ts_ms: int) -> dict[str, Any] | None:
    try:
        row = conn.execute(
            """
            SELECT ts_ms, bid_price, ask_price, mid_price FROM book_ticker
            WHERE symbol=? AND ts_ms<=?
            ORDER BY ts_ms DESC
            LIMIT 1
            """,
            (symbol, int(ts_ms)),
        ).fetchone()
    except sqlite3.OperationalError:
        return None
    if not row:
        return None
    return {
        "ts_ms": int(row[0]),
        "bid": float(row[1]),
        "ask": float(row[2]),
        "mid": float(row[3]),
    }


def _modeled_book(reference_price: float, spread_bps: float) -> dict[str, Any]:
    half = float(reference_price) * float(spread_bps) / 2.0 / 1e4
    return {
        "ts_ms": None,
        "bid": float(reference_price) - half,
        "ask": float(reference_price) + half,
        "mid": float(reference_price),
    }


def _fill_quote(
    conn: sqlite3.Connection,
    rule: S34Rule | dict[str, Any],
    symbol: str,
    ts_ms: int,
    reference_price: float,
    direction: str,
    leg: str,
    mode: str,
    *,
    limit_price: float | None = None,
) -> dict[str, Any]:
    if isinstance(rule, dict):
        modeled_spread_bps = float(rule.get("modeled_spread_bps") or 0.0)
        max_stale_sec = int(rule.get("max_book_staleness_sec") or 5)
        taker_fee_bps = float(rule.get("taker_fee_bps") or 4.0)
        maker_fee_bps = float(rule.get("maker_fee_bps") or 2.0)
        require_book = bool(rule.get("require_book_ticker_fill", True))
    else:
        modeled_spread_bps = float(rule.modeled_spread_bps)
        max_stale_sec = int(rule.max_book_staleness_sec)
        taker_fee_bps = float(rule.taker_fee_bps)
        maker_fee_bps = float(rule.maker_fee_bps)
        require_book = bool(rule.require_book_ticker_fill)

    book = _book_ticker_at(conn, symbol, int(ts_ms))
    source = "BOOK_TICKER"
    if not book or book["ts_ms"] is None or int(ts_ms) - int(book["ts_ms"]) > max_stale_sec * 1000:
        if require_book:
            raise RuntimeError(
                f"no_fill_data symbol={symbol} ts_ms={int(ts_ms)} "
                f"max_book_staleness_sec={max_stale_sec}"
            )
        book = _modeled_book(float(reference_price), modeled_spread_bps)
        source = "MODELED_SPREAD"

    direction = direction.upper()
    mode = mode.lower()
    leg = leg.upper()
    if mode == "maker":
        if limit_price is None:
            raise ValueError("maker fill requires limit_price")
        fill_price = float(limit_price)
        fee_bps = maker_fee_bps
    elif mode == "taker":
        if direction == "LONG":
            fill_price = float(book["ask"] if leg == "ENTRY" else book["bid"])
        else:
            fill_price = float(book["bid"] if leg == "ENTRY" else book["ask"])
        fee_bps = taker_fee_bps
    else:
        raise ValueError(f"unknown fill mode {mode!r}")

    return {
        "ts_ms": int(ts_ms),
        "ts_utc": _iso_from_ms(int(ts_ms)),
        "book_ts_ms": book["ts_ms"],
        "book_ts_utc": None if book["ts_ms"] is None else _iso_from_ms(int(book["ts_ms"])),
        "source": source,
        "mode": mode,
        "leg": leg,
        "reference_price": float(reference_price),
        "bid": float(book["bid"]),
        "ask": float(book["ask"]),
        "mid": float(book["mid"]),
        "fill_price": float(fill_price),
        "fee_bps": float(fee_bps),
    }


def _cost_decomposition(
    direction: str,
    entry_reference: float,
    exit_reference: float,
    entry_fill: dict[str, Any],
    exit_fill: dict[str, Any],
    entry_fee_bps: float,
    exit_fee_bps: float,
) -> dict[str, float]:
    basis = float(entry_reference)
    gross_bps = _pnl_bps(direction, basis, float(exit_reference))
    entry_fill_px = float(entry_fill["fill_price"])
    exit_fill_px = float(exit_fill["fill_price"])
    entry_mid = float(entry_fill.get("mid", entry_fill_px))
    exit_mid = float(exit_fill.get("mid", exit_fill_px))
    executable_bps = _pnl_bps(direction, entry_fill_px, exit_fill_px) * (entry_fill_px / basis)
    mid_to_mid_bps = _pnl_bps(direction, entry_mid, exit_mid) * (entry_mid / basis)
    if direction.upper() == "LONG":
        entry_adverse_bps = (entry_mid - basis) / basis * 1e4
        exit_adverse_bps = (float(exit_reference) - exit_mid) / basis * 1e4
        entry_spread_bps = (entry_fill_px - entry_mid) / basis * 1e4
        exit_spread_bps = (exit_mid - exit_fill_px) / basis * 1e4
    else:
        entry_adverse_bps = (basis - entry_mid) / basis * 1e4
        exit_adverse_bps = (exit_mid - float(exit_reference)) / basis * 1e4
        entry_spread_bps = (entry_mid - entry_fill_px) / basis * 1e4
        exit_spread_bps = (exit_fill_px - exit_mid) / basis * 1e4
    spread_cost_bps = entry_spread_bps + exit_spread_bps
    mark_to_fill_cost_bps = entry_adverse_bps + exit_adverse_bps + spread_cost_bps
    fee_cost_bps = float(entry_fee_bps) + float(exit_fee_bps)
    net_bps = gross_bps - entry_adverse_bps - exit_adverse_bps - spread_cost_bps - fee_cost_bps
    identity = gross_bps - entry_adverse_bps - exit_adverse_bps - spread_cost_bps - fee_cost_bps
    if abs(identity - net_bps) > 1e-9:
        raise RuntimeError(f"cost_identity_failed identity={identity} net_bps={net_bps}")
    if abs((gross_bps - executable_bps) - mark_to_fill_cost_bps) > 1e-6:
        raise RuntimeError("mark_to_fill_identity_failed")
    return {
        "gross_bps": gross_bps,
        "mid_to_mid_bps": mid_to_mid_bps,
        "executable_gross_bps": executable_bps,
        "entry_adverse_bps": entry_adverse_bps,
        "exit_adverse_bps": exit_adverse_bps,
        "mark_to_fill_cost_bps": mark_to_fill_cost_bps,
        "spread_cost_bps": spread_cost_bps,
        "entry_spread_bps": entry_spread_bps,
        "exit_spread_bps": exit_spread_bps,
        "fee_cost_bps": fee_cost_bps,
        "net_bps": net_bps,
    }


def _risk_payload(rule: S34Rule, config: RiskConfig) -> dict[str, float]:
    risk_usdt = float(config.simulated_equity_usdt) * float(config.risk_per_trade_pct) / 100.0
    stop_bps = max(float(rule.sl_bps), 1.0)
    notional = risk_usdt * 1e4 / stop_bps
    margin = notional / max(float(config.leverage), 1.0)
    return {
        "simulated_equity_usdt": float(config.simulated_equity_usdt),
        "leverage": float(config.leverage),
        "risk_per_trade_pct": float(config.risk_per_trade_pct),
        "risk_usdt": risk_usdt,
        "notional_usdt": notional,
        "margin_required_usdt": margin,
    }


def _utc_day_start_ms(ts_ms: int) -> int:
    dt = datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc)
    start = datetime(dt.year, dt.month, dt.day, tzinfo=timezone.utc)
    return int(start.timestamp() * 1000)


def _regime_snapshot(conn: sqlite3.Connection, symbol: str, ts_ms: int) -> dict[str, Any]:
    start_ms = _utc_day_start_ms(int(ts_ms))
    first = conn.execute(
        """
        SELECT mark_price FROM mark_prices
        WHERE symbol=? AND ts_ms>=? AND ts_ms<=?
        ORDER BY ts_ms ASC LIMIT 1
        """,
        (symbol, start_ms, int(ts_ms)),
    ).fetchone()
    last = conn.execute(
        """
        SELECT mark_price FROM mark_prices
        WHERE symbol=? AND ts_ms>=? AND ts_ms<=?
        ORDER BY ts_ms DESC LIMIT 1
        """,
        (symbol, start_ms, int(ts_ms)),
    ).fetchone()
    minmax = conn.execute(
        """
        SELECT MIN(mark_price), MAX(mark_price) FROM mark_prices
        WHERE symbol=? AND ts_ms>=? AND ts_ms<=?
        """,
        (symbol, start_ms, int(ts_ms)),
    ).fetchone()
    buy_liq = conn.execute(
        """
        SELECT COUNT(*), COALESCE(SUM(notional), 0.0) FROM liquidations
        WHERE symbol=? AND side='BUY' AND ts_ms>=? AND ts_ms<=?
        """,
        (symbol, start_ms, int(ts_ms)),
    ).fetchone()
    try:
        agg = conn.execute(
            """
            SELECT COUNT(*), COALESCE(SUM(notional), 0.0) FROM agg_trades
            WHERE symbol=? AND ts_ms>=? AND ts_ms<=?
            """,
            (symbol, start_ms, int(ts_ms)),
        ).fetchone()
    except sqlite3.OperationalError as exc:
        if "no such table: agg_trades" not in str(exc):
            raise
        agg = (0, 0.0)
    trend_pct = None
    range_pct = None
    if first and last and first[0]:
        trend_pct = (float(last[0]) - float(first[0])) / float(first[0]) * 100.0
    if first and minmax and first[0] and minmax[0] is not None and minmax[1] is not None:
        range_pct = (float(minmax[1]) - float(minmax[0])) / float(first[0]) * 100.0
    return {
        "window": "utc_day_so_far",
        "day_start_ts_ms": start_ms,
        "trend_pct": trend_pct,
        "range_pct": range_pct,
        "buy_liq_count": int((buy_liq or [0, 0.0])[0] or 0),
        "buy_liq_notional": float((buy_liq or [0, 0.0])[1] or 0.0),
        "agg_trade_count": int((agg or [0, 0.0])[0] or 0),
        "agg_trade_notional": float((agg or [0, 0.0])[1] or 0.0),
    }


def _cluster_shape_label(duration_sec: float, max_single_liq_share: float) -> str:
    if float(max_single_liq_share) >= 80.0:
        return "single_dominant_80pct"
    if float(duration_sec) >= 120.0:
        return "stretched_120s"
    return "distributed_mid_duration"


def _previous_liq_gap_sec(conn: sqlite3.Connection, rule: S34Rule, ts_ms: int) -> float | None:
    row = conn.execute(
        """
        SELECT MAX(ts_ms) FROM liquidations
        WHERE symbol=? AND side=? AND ts_ms<?
        """,
        (rule.symbol, rule.liq_side, int(ts_ms)),
    ).fetchone()
    prev_ts = row[0] if row else None
    if prev_ts is None:
        return None
    return max(0.0, (int(ts_ms) - int(prev_ts)) / 1000.0)


def _quality_gate(
    conn: sqlite3.Connection,
    signal_ts_ms: int,
    *,
    min_eclipse_score: float = 42.0,
    allow_standard_confidence: bool = False,
    lookup_window_ms: int = 300_000,
) -> tuple[bool, str]:
    """Quality gate based on detector_signals enrichment.

    Looks up detector_signals within ±lookup_window_ms of signal_ts_ms.
    Falls through (pass) if no detector signal found — permissive fallback
    so the gate doesn't block signals when the detector hasn't enriched them yet.

    Returns (pass, skip_reason). skip_reason is empty string on pass.

    Bucket analysis (61 pre-reg trades, 2026-04-14 cutoff) findings:
      - confidence=standard: 36/61 trades, WinR 55.6%, avg -0.054  <<< KILL
      - eclipse_score Q1+Q2 (<42): 31/61 trades, avg -0.188          <<< KILL
    """
    try:
        row = conn.execute(
            """
            SELECT confidence_band, eclipse_score
            FROM detector_signals
            WHERE signal_ts_ms BETWEEN ? AND ?
            ORDER BY ABS(signal_ts_ms - ?) ASC
            LIMIT 1
            """,
            (signal_ts_ms - lookup_window_ms, signal_ts_ms + lookup_window_ms, signal_ts_ms),
        ).fetchone()
    except Exception:
        return True, ""

    if row is None:
        return True, ""

    confidence = str(row[0] or "")
    eclipse = row[1]

    if not allow_standard_confidence and confidence == "standard":
        return False, f"QUALITY_GATE_confidence_{confidence}"

    if eclipse is not None and float(eclipse) < min_eclipse_score:
        return False, f"QUALITY_GATE_eclipse_{float(eclipse):.1f}_lt_{min_eclipse_score:.0f}"

    return True, ""


def _regime_gate(conn: sqlite3.Connection, rule: S34Rule, signal: dict[str, Any], config: RegimeConfig) -> tuple[bool, str, dict[str, Any]]:
    has_rule_gate = rule.min_day_trend_bps is not None or rule.max_day_trend_bps is not None
    if not bool(config.enabled) and not has_rule_gate:
        return True, "", {}
    snap = _regime_snapshot(conn, rule.symbol, int(signal["ts_ms"]))
    checks: dict[str, bool] = {}
    thresholds: dict[str, Any] = {}
    if bool(config.enabled) and bool(rule.use_global_regime):
        checks.update(
            {
                "trend_pct_gte": snap["trend_pct"] is not None
                and float(snap["trend_pct"]) >= float(config.min_trend_pct),
                "range_pct_gte": snap["range_pct"] is not None
                and float(snap["range_pct"]) >= float(config.min_range_pct),
                "buy_liq_notional_gte": float(snap["buy_liq_notional"]) >= float(config.min_buy_liq_notional),
                "agg_trade_count_gte": int(snap["agg_trade_count"]) >= int(config.min_agg_trade_count),
            }
        )
        thresholds["global_regime"] = asdict(config)
    if has_rule_gate:
        trend_bps = None if snap["trend_pct"] is None else float(snap["trend_pct"]) * 100.0
        if rule.min_day_trend_bps is not None:
            checks["day_trend_bps_gte"] = trend_bps is not None and trend_bps >= float(rule.min_day_trend_bps)
        if rule.max_day_trend_bps is not None:
            checks["day_trend_bps_lte"] = trend_bps is not None and trend_bps < float(rule.max_day_trend_bps)
        thresholds["rule_regime"] = {
            "use_global_regime": bool(rule.use_global_regime),
            "min_day_trend_bps": None if rule.min_day_trend_bps is None else float(rule.min_day_trend_bps),
            "max_day_trend_bps": None if rule.max_day_trend_bps is None else float(rule.max_day_trend_bps),
        }
        snap["day_trend_bps"] = trend_bps
    snap["thresholds"] = thresholds
    snap["checks"] = checks
    return all(checks.values()), "REGIME_FILTER", snap


def _derive_min_gap_seed_from_history(trades: dict[str, dict[str, Any]], rule: S34Rule) -> dict[str, Any]:
    """OD-018-FOLLOWUP first-activation migration oracle (read of already
    in-memory `trades`, no extra file access). Never raises; fails closed and
    reports on malformed/ambiguous input rather than approximating.

    Returns one of:
      DERIVED_FROM_HISTORY  -- seed_ts_ms is the max signal_ts_ms found for
                                this exact rule identity (name+symbol+
                                threshold+liq_side all match)
      NO_PRIOR_EMISSION     -- no trade record for this rule at all (fresh
                                rule; equivalent to the pre-migration -inf
                                default, not an error)
      AMBIGUOUS_FAILED      -- the same rule.name appears with a DIFFERENT
                                identity, or a record is missing signal_ts_ms;
                                seeding is refused
    """
    valid_ts: list[int] = []
    malformed: list[str] = []
    for trade_id, trade in trades.items():
        rule_blob = trade.get("rule")
        if not isinstance(rule_blob, dict) or rule_blob.get("name") != rule.name:
            continue
        try:
            same_identity = (
                str(rule_blob.get("symbol")) == str(rule.symbol)
                and str(rule_blob.get("liq_side", "")).upper() == str(rule.liq_side).upper()
                and abs(float(rule_blob.get("threshold_usd", -1.0)) - float(rule.threshold_usd)) < 1e-6
            )
        except (TypeError, ValueError):
            same_identity = False
        if not same_identity:
            malformed.append(str(trade_id))
            continue
        ts = trade.get("signal_ts_ms")
        if not isinstance(ts, (int, float)):
            malformed.append(str(trade_id))
            continue
        valid_ts.append(int(ts))
    if malformed:
        return {
            "status": "AMBIGUOUS_FAILED",
            "seed_ts_ms": None,
            "malformed_trade_ids": sorted(malformed),
            "source": "trade_history",
        }
    if not valid_ts:
        return {"status": "NO_PRIOR_EMISSION", "seed_ts_ms": None, "source": "trade_history"}
    return {"status": "DERIVED_FROM_HISTORY", "seed_ts_ms": max(valid_ts), "source": "trade_history"}


def _annotate_trade_pnl_usdt(trade: dict[str, Any]) -> dict[str, Any]:
    risk = trade.get("risk") or {}
    notional = float(risk.get("notional_usdt") or 0.0)
    gross_bps = trade.get("gross_bps")
    net_bps = trade.get("net_bps")
    trade["gross_usdt"] = None if gross_bps is None else float(gross_bps) / 1e4 * notional
    trade["net_usdt"] = None if net_bps is None else float(net_bps) / 1e4 * notional
    return trade


def _bucket_events(
    conn: sqlite3.Connection,
    rule: S34Rule,
    start_ms: int,
    end_ms: int,
    limit: int,
    last_signal_ms_seed: int | None = None,
) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT CAST(ts_ms / ? AS INTEGER) AS bucket, ts_ms, price, notional
        FROM liquidations
        WHERE symbol=? AND side=? AND ts_ms>=? AND ts_ms<=?
        ORDER BY ts_ms ASC
        """,
        (
            int(rule.bucket_sec * 1000),
            rule.symbol,
            rule.liq_side,
            int(start_ms),
            int(end_ms),
        ),
    ).fetchall()

    buckets: dict[int, dict[str, Any]] = {}
    for bucket, ts_ms, price, notional in rows:
        bucket_i = int(bucket)
        state = buckets.setdefault(
            bucket_i,
            {
                "first_ts": int(ts_ms),
                "last_ts": int(ts_ms),
                "count": 0,
                "total_notional": 0.0,
                "max_notional": 0.0,
                "max_price": 0.0,
                "crossed": False,
            },
        )
        if state["crossed"]:
            continue
        state["last_ts"] = int(ts_ms)
        state["count"] = int(state["count"]) + 1
        state["total_notional"] = float(state["total_notional"]) + float(notional or 0.0)
        if float(notional or 0.0) >= float(state["max_notional"]):
            state["max_notional"] = float(notional or 0.0)
            state["max_price"] = float(price or 0.0)
        if float(state["total_notional"]) >= float(rule.threshold_usd):
            state["crossed"] = True

    out: list[dict[str, Any]] = []
    last_signal_ms = -10**18 if last_signal_ms_seed is None else int(last_signal_ms_seed)
    for bucket, state in sorted(buckets.items(), key=lambda item: int(item[1]["first_ts"])):
        if not state.get("crossed"):
            continue
        first_ts = int(state["first_ts"])
        last_ts = int(state["last_ts"])
        count = int(state["count"])
        total_notional = float(state["total_notional"])
        max_notional = float(state["max_notional"])
        max_price = float(state["max_price"])
        ts_ms = last_ts
        if ts_ms - last_signal_ms < rule.min_gap_sec * 1000:
            continue
        duration_sec = max(0.0, (float(last_ts) - float(first_ts)) / 1000.0)
        max_single_share = 0.0
        if total_notional:
            max_single_share = float(max_notional or 0.0) / float(total_notional) * 100.0
        shape_label = _cluster_shape_label(duration_sec, max_single_share)
        if rule.min_cluster_liq_count is not None and int(count or 0) < int(rule.min_cluster_liq_count):
            continue
        if rule.required_shape_label is not None and shape_label != str(rule.required_shape_label):
            continue
        if rule.max_single_liq_share_pct is not None and max_single_share >= float(rule.max_single_liq_share_pct):
            continue
        btc_pre_return_bps = None
        btc_pre_start_mark_ts_ms = None
        btc_pre_end_mark_ts_ms = None
        if rule.btc_pre_min_return_bps is not None and int(rule.btc_pre_window_sec) > 0:
            btc_window_start = ts_ms - int(rule.btc_pre_window_sec) * 1000
            btc_return = _mark_return_bps(conn, rule.btc_confirm_symbol, btc_window_start, ts_ms)
            if btc_return is None:
                continue
            btc_pre_return_bps, btc_pre_start_mark_ts_ms, btc_pre_end_mark_ts_ms = btc_return
            if float(btc_pre_return_bps) < float(rule.btc_pre_min_return_bps):
                continue
        entry_ts_ms = ts_ms + max(0, int(rule.entry_delay_sec)) * 1000
        mark = _mark_at(conn, rule.symbol, entry_ts_ms, before=False)
        if not mark:
            continue
        mark_ts, entry_reference_price = mark
        fill_error = ""
        try:
            entry_quote = _fill_quote(
                conn,
                rule,
                rule.symbol,
                mark_ts,
                entry_reference_price,
                rule.direction,
                "ENTRY",
                "taker",
            )
        except RuntimeError as exc:
            entry_quote = None
            fill_error = str(exc)
        day_context = _regime_snapshot(conn, rule.symbol, ts_ms)
        day_trend_bps = None if day_context.get("trend_pct") is None else float(day_context["trend_pct"]) * 100.0
        day_range_bps = None if day_context.get("range_pct") is None else float(day_context["range_pct"]) * 100.0
        intensity_per_sec = float(total_notional or 0.0) / max(float(duration_sec), 1.0)
        inter_cluster_gap_sec = None if last_signal_ms <= -10**17 else max(0.0, (ts_ms - last_signal_ms) / 1000.0)
        prev_liq_gap_sec = _previous_liq_gap_sec(conn, rule, ts_ms)
        if inter_cluster_gap_sec is None:
            inter_cluster_gap_sec = prev_liq_gap_sec
        out.append(
            {
                "min_gap_semantics": MIN_GAP_SEMANTICS_VERSION,
                "bucket": int(bucket),
                "ts_ms": ts_ms,
                "ts_utc": _iso_from_ms(ts_ms),
                "mark_ts_ms": mark_ts,
                "entry_delay_sec": int(rule.entry_delay_sec),
                "entry_ts_ms": int(mark_ts),
                "entry_ts_utc": _iso_from_ms(int(mark_ts)),
                "entry_reference_price": float(entry_reference_price),
                "entry_price": float(entry_quote["fill_price"]) if entry_quote else float(entry_reference_price),
                "entry_fill": entry_quote,
                "fill_error": fill_error,
                "btc_confirm_symbol": rule.btc_confirm_symbol,
                "btc_pre_window_sec": int(rule.btc_pre_window_sec),
                "btc_pre_min_return_bps": rule.btc_pre_min_return_bps,
                "btc_pre_return_bps": btc_pre_return_bps,
                "btc_pre_start_mark_ts_ms": btc_pre_start_mark_ts_ms,
                "btc_pre_end_mark_ts_ms": btc_pre_end_mark_ts_ms,
                "liq_count": int(count or 0),
                "liq_total_notional": float(total_notional or 0.0),
                "liq_max_notional": float(max_notional or 0.0),
                "liq_max_price": float(max_price or 0.0),
                "cluster_start_ts_ms": int(first_ts),
                "cluster_end_ts_ms": int(last_ts),
                "threshold_cross_ts_ms": int(ts_ms),
                "threshold_cross_ts_utc": _iso_from_ms(int(ts_ms)),
                "cluster_duration_sec": float(duration_sec),
                "cluster_liq_count": int(count or 0),
                "cluster_max_single_liq_share": float(max_single_share),
                "max_single_liq_share": float(max_single_share),
                "intensity_per_sec": float(intensity_per_sec),
                "inter_cluster_gap_sec": inter_cluster_gap_sec,
                "prev_liq_gap_sec": prev_liq_gap_sec,
                "cluster_shape_label": shape_label,
                "day_context_window": day_context.get("window"),
                "day_start_ts_ms": day_context.get("day_start_ts_ms"),
                "day_trend_bps": day_trend_bps,
                "day_range_bps": day_range_bps,
                "day_buy_liq_count": day_context.get("buy_liq_count"),
                "day_buy_liq_notional": day_context.get("buy_liq_notional"),
                "day_agg_trade_count": day_context.get("agg_trade_count"),
                "day_agg_trade_notional": day_context.get("agg_trade_notional"),
            }
        )
        last_signal_ms = ts_ms
    return out


def _paper_trade_from_signal(rule: S34Rule, signal: dict[str, Any], risk_config: RiskConfig) -> dict[str, Any]:
    entry_ts_ms = int(signal.get("entry_ts_ms") or signal.get("mark_ts_ms") or signal["ts_ms"])
    assert_feature_set_available(
        signal_entry_features(signal),
        entry_ts_ms,
        context=f"s34_shadow_paper_runner:{rule.name}",
    )
    entry = float(signal["entry_price"])
    entry_reference = float(signal.get("entry_reference_price") or signal["entry_price"])
    if rule.direction == "LONG":
        tp = entry_reference * (1.0 + rule.tp_bps / 1e4)
        sl = entry_reference * (1.0 - rule.sl_bps / 1e4)
        be_trigger = entry_reference * (1.0 + rule.be_trigger_bps / 1e4)
    else:
        tp = entry_reference * (1.0 - rule.tp_bps / 1e4)
        sl = entry_reference * (1.0 + rule.sl_bps / 1e4)
        be_trigger = entry_reference * (1.0 - rule.be_trigger_bps / 1e4)
    signal_key = f"{rule.name}:{int(signal['bucket'])}"
    return {
        "trade_id": "",
        "trial_id": "",
        "signal_key": signal_key,
        "status": "OPEN",
        "opened_at_utc": _utc_now_iso(),
        "signal_ts_ms": int(signal["ts_ms"]),
        "signal_ts_utc": signal["ts_utc"],
        "entry_ts_ms": entry_ts_ms,
        "entry_ts_utc": signal.get("entry_ts_utc") or _iso_from_ms(entry_ts_ms),
        "symbol": rule.symbol,
        "direction": rule.direction,
        "rule": asdict(rule),
        "entry_reference_price": entry_reference,
        "entry_price": entry,
        "entry_fill": signal.get("entry_fill"),
        "tp_price": tp,
        "sl_price": sl,
        "be_trigger_price": be_trigger,
        "be_active": False,
        "exit_price": None,
        "exit_ts_ms": None,
        "exit_ts_utc": None,
        "exit_reason": None,
        "gross_bps": None,
        "mid_to_mid_bps": None,
        "executable_gross_bps": None,
        "entry_adverse_bps": None,
        "exit_adverse_bps": None,
        "mark_to_fill_cost_bps": None,
        "spread_cost_bps": None,
        "entry_spread_bps": None,
        "exit_spread_bps": None,
        "fee_cost_bps": None,
        "net_bps": None,
        "gross_usdt": None,
        "net_usdt": None,
        "risk": _risk_payload(rule, risk_config),
        "risk_gate_status": "ACCEPTED",
        "risk_gate_reason": "",
        # OD-018-FOLLOWUP (C): top-level, durable protocol tag. Absent on all
        # 1,338 pre-v2 trades (verified); every trade created from this point
        # on carries it, so pre-v2/v2 populations are filterable without
        # descending into the nested `signal` dict.
        "min_gap_semantics": signal.get("min_gap_semantics") or MIN_GAP_SEMANTICS_VERSION,
        "signal": signal,
    }


def _legacy_entry_fill(trade: dict[str, Any]) -> dict[str, Any]:
    ts_ms = int(trade.get("entry_ts_ms") or trade.get("signal_ts_ms") or 0)
    price = float(trade["entry_price"])
    rule = trade.get("rule") or {}
    return {
        "ts_ms": ts_ms,
        "ts_utc": _iso_from_ms(ts_ms) if ts_ms else None,
        "book_ts_ms": None,
        "book_ts_utc": None,
        "source": "LEGACY_MARK_OR_MODELED",
        "mode": "taker",
        "leg": "ENTRY",
        "reference_price": float(trade.get("entry_reference_price") or price),
        "bid": price,
        "ask": price,
        "mid": float(trade.get("entry_reference_price") or price),
        "fill_price": price,
        "fee_bps": float(rule.get("taker_fee_bps") or 4.0),
    }


def _close_trade(conn: sqlite3.Connection, trade: dict[str, Any], ts_ms: int, px: float, reason: str) -> dict[str, Any]:
    if reason == "BE":
        be_ts = int(trade.get("be_activated_ts_ms") or 0)
        if not be_ts:
            raise RuntimeError(f"invalid_be_close_missing_activation trade_id={trade.get('trade_id')}")
        if int(ts_ms) < be_ts:
            raise RuntimeError(
                f"invalid_be_close_rewind trade_id={trade.get('trade_id')} "
                f"exit_ts_ms={int(ts_ms)} be_activated_ts_ms={be_ts}"
            )
    rule = trade["rule"]
    direction = str(trade["direction"])
    entry_reference = float(trade.get("entry_reference_price") or trade["entry_price"])
    entry_fill = trade.get("entry_fill") or _legacy_entry_fill(trade)
    trade["entry_fill"] = entry_fill
    exit_mode = "taker"
    limit_price = None
    if reason == "TP":
        exit_mode = str(rule.get("tp_fill_mode") or "taker").lower()
        if exit_mode == "maker":
            # Optimistic: assumes the resting TP limit fills once the mark/last touches the TP level.
            limit_price = float(trade["tp_price"])
    exit_fill = _fill_quote(
        conn,
        rule,
        str(trade["symbol"]),
        int(ts_ms),
        float(px),
        direction,
        "EXIT",
        exit_mode,
        limit_price=limit_price,
    )
    cost = _cost_decomposition(
        direction,
        entry_reference,
        float(px),
        entry_fill,
        exit_fill,
        float(entry_fill.get("fee_bps") or 0.0),
        float(exit_fill.get("fee_bps") or 0.0),
    )
    trade["status"] = "CLOSED"
    trade["closed_at_utc"] = _utc_now_iso()
    trade["exit_ts_ms"] = int(ts_ms)
    trade["exit_ts_utc"] = _iso_from_ms(int(ts_ms))
    trade["exit_reference_price"] = float(px)
    trade["exit_price"] = float(exit_fill["fill_price"])
    trade["exit_fill"] = exit_fill
    trade["exit_reason"] = reason
    trade["gross_bps"] = cost["gross_bps"]
    trade["mid_to_mid_bps"] = cost["mid_to_mid_bps"]
    trade["executable_gross_bps"] = cost["executable_gross_bps"]
    trade["entry_adverse_bps"] = cost["entry_adverse_bps"]
    trade["exit_adverse_bps"] = cost["exit_adverse_bps"]
    trade["mark_to_fill_cost_bps"] = cost["mark_to_fill_cost_bps"]
    trade["spread_cost_bps"] = cost["spread_cost_bps"]
    trade["entry_spread_bps"] = cost["entry_spread_bps"]
    trade["exit_spread_bps"] = cost["exit_spread_bps"]
    trade["fee_cost_bps"] = cost["fee_cost_bps"]
    trade["net_bps"] = cost["net_bps"]
    return _annotate_trade_pnl_usdt(trade)


def _evaluate_trade(conn: sqlite3.Connection, trade: dict[str, Any], end_ms: int) -> dict[str, Any]:
    if trade.get("status") != "OPEN":
        return trade
    signal_ms = int(trade["signal_ts_ms"])
    entry_ms = int(trade.get("entry_ts_ms") or signal_ms)
    start_ms = max(entry_ms, int(trade.get("last_evaluated_mark_ts_ms") or entry_ms))
    max_horizon_ms = int(trade["rule"]["max_horizon_sec"]) * 1000
    horizon_end_ms = entry_ms + max_horizon_ms
    cutoff_ms = min(int(end_ms), horizon_end_ms)
    if cutoff_ms <= start_ms:
        return trade
    rows = conn.execute(
        """
        SELECT ts_ms, mark_price FROM mark_prices
        WHERE symbol=? AND ts_ms>? AND ts_ms<=?
        ORDER BY ts_ms ASC
        """,
        (str(trade["symbol"]), start_ms, cutoff_ms),
    ).fetchall()
    if not rows:
        return trade

    direction = str(trade["direction"]).upper()
    for ts_ms, px in rows:
        ts = int(ts_ms)
        price = float(px)
        trade["last_evaluated_mark_ts_ms"] = ts
        trade["last_evaluated_mark_ts_utc"] = _iso_from_ms(ts)
        if direction == "LONG":
            if not trade["be_active"] and price >= float(trade["be_trigger_price"]):
                trade["be_active"] = True
                trade["be_activated_ts_ms"] = ts
                trade["be_activated_ts_utc"] = _iso_from_ms(ts)
            if price >= float(trade["tp_price"]):
                return _close_trade(conn, trade, ts, price, "TP")
            stop_price = float(trade.get("entry_reference_price") or trade["entry_price"]) if trade["be_active"] else float(trade["sl_price"])
            if price <= stop_price:
                return _close_trade(conn, trade, ts, price, "BE" if trade["be_active"] else "SL")
        else:
            if not trade["be_active"] and price <= float(trade["be_trigger_price"]):
                trade["be_active"] = True
                trade["be_activated_ts_ms"] = ts
                trade["be_activated_ts_utc"] = _iso_from_ms(ts)
            if price <= float(trade["tp_price"]):
                return _close_trade(conn, trade, ts, price, "TP")
            stop_price = float(trade.get("entry_reference_price") or trade["entry_price"]) if trade["be_active"] else float(trade["sl_price"])
            if price >= stop_price:
                return _close_trade(conn, trade, ts, price, "BE" if trade["be_active"] else "SL")

    if cutoff_ms >= horizon_end_ms:
        last_ts, last_px = rows[-1]
        return _close_trade(conn, trade, int(last_ts), float(last_px), "TIME")
    return trade


def _load_trades(path: Path) -> dict[str, dict[str, Any]]:
    payload = _read_json(path, {"trades": []})
    trades = payload.get("trades", []) if isinstance(payload, dict) else []
    return {str(t.get("trade_id")): t for t in trades if t.get("trade_id")}


def _rule_from_trade(trade: dict[str, Any]) -> S34Rule:
    raw = trade.get("rule") or {}
    return S34Rule(
        name=str(raw.get("name") or "UNKNOWN"),
        symbol=str(raw.get("symbol") or trade.get("symbol") or "ETHUSDT"),
        liq_side=str(raw.get("liq_side") or "BUY"),
        direction=str(raw.get("direction") or trade.get("direction") or "LONG"),
        threshold_usd=float(raw.get("threshold_usd") or 50_000.0),
        bucket_sec=int(raw.get("bucket_sec") or 300),
        min_gap_sec=int(raw.get("min_gap_sec") or 900),
        tp_bps=float(raw.get("tp_bps") or 120.0),
        sl_bps=float(raw.get("sl_bps") or 40.0),
        be_trigger_bps=float(raw.get("be_trigger_bps") or 30.0),
        max_horizon_sec=int(raw.get("max_horizon_sec") or 3600),
        taker_fee_bps=float(raw.get("taker_fee_bps") or (float(raw.get("round_trip_cost_bps") or 8.0) / 2.0)),
        maker_fee_bps=float(raw.get("maker_fee_bps") or 2.0),
        tp_fill_mode=str(raw.get("tp_fill_mode") or "taker"),
        modeled_spread_bps=float(raw.get("modeled_spread_bps") or 0.0),
        max_book_staleness_sec=int(raw.get("max_book_staleness_sec") or 5),
        require_book_ticker_fill=bool(raw.get("require_book_ticker_fill", True)),
        max_open_trades=int(raw.get("max_open_trades") or 1),
        daily_max_sl=int(raw.get("daily_max_sl") or 3),
        entry_delay_sec=int(raw.get("entry_delay_sec") or 0),
        btc_confirm_symbol=str(raw.get("btc_confirm_symbol") or "BTCUSDT"),
        btc_pre_window_sec=int(raw.get("btc_pre_window_sec") or 0),
        btc_pre_min_return_bps=(
            None if raw.get("btc_pre_min_return_bps") is None else float(raw.get("btc_pre_min_return_bps"))
        ),
        use_global_regime=bool(raw.get("use_global_regime", True)),
        min_day_trend_bps=(None if raw.get("min_day_trend_bps") is None else float(raw.get("min_day_trend_bps"))),
        max_day_trend_bps=(None if raw.get("max_day_trend_bps") is None else float(raw.get("max_day_trend_bps"))),
        min_cluster_liq_count=(
            None if raw.get("min_cluster_liq_count") is None else int(raw.get("min_cluster_liq_count"))
        ),
        required_shape_label=(None if raw.get("required_shape_label") is None else str(raw.get("required_shape_label"))),
        max_single_liq_share_pct=(
            None if raw.get("max_single_liq_share_pct") is None else float(raw.get("max_single_liq_share_pct"))
        ),
        priority=int(raw.get("priority") or 100),
    )


def _normalize_trades(trades: dict[str, dict[str, Any]], risk_config: RiskConfig) -> None:
    for trade in trades.values():
        if not trade.get("trial_id"):
            trade["trial_id"] = trade.get("trade_id")
        if not trade.get("risk"):
            trade["risk"] = _risk_payload(_rule_from_trade(trade), risk_config)
        trade.setdefault("risk_gate_status", "ACCEPTED" if trade.get("status") != "SKIPPED" else "SKIPPED")
        trade.setdefault("risk_gate_reason", "")
        if trade.get("net_bps") is not None and trade.get("net_usdt") is None:
            _annotate_trade_pnl_usdt(trade)


def _next_trial_number(trades: dict[str, dict[str, Any]]) -> int:
    max_seen = 0
    for trade in trades.values():
        raw = str(trade.get("trial_id") or trade.get("trade_id") or "")
        if raw.startswith("P") and raw[1:].isdigit():
            max_seen = max(max_seen, int(raw[1:]))
    return max_seen + 1


def _assign_trial_id(trade: dict[str, Any], trial_number: int) -> dict[str, Any]:
    trial_id = f"P{trial_number:03d}"
    trade["trade_id"] = trial_id
    trade["trial_id"] = trial_id
    return trade


def _trade_day(ts_ms: int) -> str:
    return datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc).date().isoformat()


def _trade_rule_name(trade: dict[str, Any]) -> str:
    rule = trade.get("rule") or {}
    if isinstance(rule, dict):
        return str(rule.get("name") or "")
    return ""


def _trade_symbol(trade: dict[str, Any]) -> str:
    rule = trade.get("rule") or {}
    if isinstance(rule, dict):
        return str(trade.get("symbol") or rule.get("symbol") or "")
    return str(trade.get("symbol") or "")


def _trade_direction(trade: dict[str, Any]) -> str:
    rule = trade.get("rule") or {}
    if isinstance(rule, dict):
        return str(trade.get("direction") or rule.get("direction") or "").upper()
    return str(trade.get("direction") or "").upper()


def _signal_cluster_key(rule: S34Rule, signal: dict[str, Any]) -> tuple[str, str, str, int]:
    return (str(rule.symbol), str(rule.direction).upper(), str(rule.liq_side).upper(), int(signal["bucket"]))


def _trade_cluster_key(trade: dict[str, Any]) -> tuple[str, str, str, int] | None:
    signal = trade.get("signal") or {}
    if not isinstance(signal, dict) or signal.get("bucket") is None:
        return None
    rule = trade.get("rule") or {}
    liq_side = str(rule.get("liq_side") or signal.get("liq_side") or "BUY") if isinstance(rule, dict) else "BUY"
    return (_trade_symbol(trade), _trade_direction(trade), liq_side.upper(), int(signal["bucket"]))


def _cluster_owner(trades: dict[str, dict[str, Any]], cluster_key: tuple[str, str, str, int]) -> dict[str, Any] | None:
    for trade in trades.values():
        if trade.get("status") == "SKIPPED":
            continue
        if _trade_cluster_key(trade) == cluster_key:
            return trade
    return None


def _candidate_sort_key(item: tuple[S34Rule, dict[str, Any]]) -> tuple[int, str, str, int, int, str]:
    rule, signal = item
    return (
        int(signal["ts_ms"]),
        str(rule.symbol),
        str(rule.direction).upper(),
        int(signal["bucket"]),
        int(rule.priority),
        str(rule.name),
    )


def _closed_before(
    trades: dict[str, dict[str, Any]], ts_ms: int, rule_name: str | None = None
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for trade in trades.values():
        if trade.get("status") != "CLOSED":
            continue
        if rule_name is not None and _trade_rule_name(trade) != rule_name:
            continue
        exit_ts = int(trade.get("exit_ts_ms") or 0)
        if 0 < exit_ts <= int(ts_ms):
            out.append(trade)
    return sorted(out, key=lambda t: int(t.get("exit_ts_ms") or 0))


def _same_day_closed_before(
    trades: dict[str, dict[str, Any]], ts_ms: int, rule_name: str | None = None
) -> list[dict[str, Any]]:
    day = _trade_day(ts_ms)
    return [t for t in _closed_before(trades, ts_ms, rule_name) if _trade_day(int(t.get("exit_ts_ms") or 0)) == day]


def _consecutive_sl_count(closed: list[dict[str, Any]]) -> int:
    count = 0
    for trade in reversed(closed):
        if str(trade.get("exit_reason")) == "SL":
            count += 1
        else:
            break
    return count


def _risk_gate(
    trades: dict[str, dict[str, Any]],
    signal: dict[str, Any],
    config: RiskConfig,
    rule: S34Rule,
) -> tuple[bool, str]:
    ts_ms = int(signal["ts_ms"])
    rule_name = str(rule.name)
    open_count = sum(
        1 for t in trades.values() if t.get("status") == "OPEN" and _trade_rule_name(t) == rule_name
    )
    if open_count >= int(rule.max_open_trades):
        return False, "MAX_OPEN_TRADES"
    same_symbol_direction_open = sum(
        1
        for t in trades.values()
        if t.get("status") == "OPEN"
        and _trade_symbol(t) == str(rule.symbol)
        and _trade_direction(t) == str(rule.direction).upper()
    )
    if same_symbol_direction_open:
        return False, "MAX_SYMBOL_DIRECTION_OPEN_TRADES"
    mark_lag_sec = max(0.0, (ts_ms - int(signal.get("mark_ts_ms") or ts_ms)) / 1000.0)
    if mark_lag_sec > float(config.max_mark_staleness_sec):
        return False, "STALE_MARK_AT_ENTRY"

    same_day = _same_day_closed_before(trades, ts_ms)
    day_net = sum(float(t.get("net_usdt") or 0.0) for t in same_day)
    daily_limit = -float(config.simulated_equity_usdt) * float(config.daily_max_loss_pct) / 100.0
    if day_net <= daily_limit:
        return False, "DAILY_MAX_LOSS"
    same_rule_day = _same_day_closed_before(trades, ts_ms, rule_name)
    if sum(1 for t in same_rule_day if str(t.get("exit_reason")) == "SL") >= int(rule.daily_max_sl):
        return False, "DAILY_MAX_SL"

    closed = _closed_before(trades, ts_ms, rule_name)
    if _consecutive_sl_count(closed) >= int(config.cooldown_after_consecutive_sl):
        last_exit = int(closed[-1].get("exit_ts_ms") or 0) if closed else 0
        cooldown_ms = int(float(config.cooldown_hours) * 3600 * 1000)
        if ts_ms - last_exit < cooldown_ms:
            return False, "COOLDOWN_AFTER_CONSECUTIVE_SL"
    return True, ""


def _skipped_trade_from_signal(
    rule: S34Rule,
    signal: dict[str, Any],
    risk_config: RiskConfig,
    reason: str,
) -> dict[str, Any]:
    trade = _paper_trade_from_signal(rule, signal, risk_config)
    trade["status"] = "SKIPPED"
    trade["risk_gate_status"] = "SKIPPED"
    trade["risk_gate_reason"] = reason
    trade["exit_reason"] = reason
    return trade


def _prediction_stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0, "median_net_bps": None, "mean_net_bps": None, "win_rate": None, "cum_net_bps": 0.0}
    ordered = sorted(values)
    n = len(ordered)
    median = ordered[n // 2] if n % 2 else (ordered[n // 2 - 1] + ordered[n // 2]) / 2.0
    return {
        "n": n,
        "median_net_bps": median,
        "mean_net_bps": sum(values) / n,
        "win_rate": sum(1 for value in values if value > 0) / n,
        "cum_net_bps": sum(values),
    }


def _base_rate_prediction(ledger_conn: sqlite3.Connection | None, rule: S34Rule, signal: dict[str, Any]) -> dict[str, Any]:
    prediction = {
        "model": "base_rate_v1",
        "model_version": "2026-06-22",
        "rule_name": rule.name,
        "symbol": rule.symbol,
        "direction": rule.direction,
        "signal_ts_ms": int(signal.get("ts_ms") or 0),
        "cluster_notional": float(signal.get("liq_total_notional") or 0.0),
        "cluster_liq_count": int(signal.get("liq_count") or 0),
        "cluster_shape_label": signal.get("cluster_shape_label") or "",
        "base_rates": {},
        "confidence_note": "no_ledger",
    }
    signal_ts_ms = int(signal.get("ts_ms") or 0)
    if ledger_conn is None:
        return prediction
    try:
        same_rule = [
            float(row[0])
            for row in ledger_conn.execute(
                "SELECT net_bps FROM s34_outcomes WHERE rule_name=? AND exit_ts_ms<? AND net_bps IS NOT NULL",
                (rule.name, signal_ts_ms),
            ).fetchall()
        ]
        same_shape = [
            float(row[0])
            for row in ledger_conn.execute(
                """
                SELECT o.net_bps
                FROM s34_outcomes o
                JOIN s34_signals s ON s.signal_id=o.signal_id
                WHERE s.cluster_shape_label=? AND o.net_bps IS NOT NULL
                  AND o.exit_ts_ms<?
                """,
                (signal.get("cluster_shape_label") or "", signal_ts_ms),
            ).fetchall()
        ]
        same_symbol_direction = [
            float(row[0])
            for row in ledger_conn.execute(
                """
                SELECT o.net_bps
                FROM s34_outcomes o
                JOIN s34_signals s ON s.signal_id=o.signal_id
                WHERE s.symbol=? AND s.direction=? AND o.net_bps IS NOT NULL
                  AND o.exit_ts_ms<?
                """,
                (rule.symbol, rule.direction, signal_ts_ms),
            ).fetchall()
        ]
        prediction["base_rates"] = {
            "same_rule": _prediction_stats(same_rule),
            "same_shape": _prediction_stats(same_shape),
            "same_symbol_direction": _prediction_stats(same_symbol_direction),
        }
        max_n = max((int(v.get("n") or 0) for v in prediction["base_rates"].values()), default=0)
        prediction["confidence_note"] = "usable" if max_n >= 20 else "thin"
        prediction["expected_net_bps"] = prediction["base_rates"]["same_rule"].get("median_net_bps")
        if prediction["expected_net_bps"] is None:
            prediction["expected_net_bps"] = prediction["base_rates"]["same_symbol_direction"].get("median_net_bps")
    except sqlite3.Error as exc:
        prediction["confidence_note"] = "ledger_error"
        prediction["error"] = repr(exc)
    return prediction


def _knn_prediction(rule: S34Rule, signal: dict[str, Any], audit: dict[str, Any], model_key: str) -> dict[str, Any]:
    knn = audit.get(model_key) or {}
    return {
        "model": model_key,
        "model_version": "2026-06-22",
        "rule_name": rule.name,
        "symbol": rule.symbol,
        "direction": rule.direction,
        "signal_ts_ms": int(signal.get("ts_ms") or 0),
        "cluster_notional": float(signal.get("liq_total_notional") or 0.0),
        "cluster_liq_count": int(signal.get("liq_count") or 0),
        "cluster_shape_label": signal.get("cluster_shape_label") or "",
        "expected_net_bps": knn.get("median_net_bps"),
        "win_rate": knn.get("win_rate"),
        "k": knn.get("k"),
        "avg_similarity": knn.get("avg_similarity"),
        "feature_set": knn.get("feature_set"),
        "confidence_note": "usable" if int(knn.get("k") or 0) >= 5 and (knn.get("avg_similarity") or 0) >= 0.4 else "thin",
    }


def _model_guardrail(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    values: list[dict[str, Any]] = []
    for prediction in predictions:
        expected = prediction.get("expected_net_bps")
        if expected is None:
            continue
        try:
            expected_value = float(expected)
        except (TypeError, ValueError):
            continue
        values.append(
            {
                "model_name": prediction.get("model"),
                "expected_net_bps": expected_value,
                "confidence_note": prediction.get("confidence_note"),
                "k": prediction.get("k"),
                "win_rate": prediction.get("win_rate"),
            }
        )
    if not values:
        return {
            "version": "2026-06-22",
            "level": "unknown",
            "headline": "No usable model prediction yet.",
            "reasons": [],
            "models": [],
        }

    negative = [row for row in values if row["expected_net_bps"] < 0]
    strongly_negative = [row for row in values if row["expected_net_bps"] <= -30]
    positive = [row for row in values if row["expected_net_bps"] > 0]
    reasons: list[str] = []
    if len(negative) >= 3:
        reasons.append(f"{len(negative)}/{len(values)} models expect negative net bps")
    if strongly_negative:
        names = ", ".join(str(row["model_name"]) for row in strongly_negative[:3])
        reasons.append(f"strong negative warning from {names}")
    if positive and negative:
        reasons.append("models disagree; treat confidence as low")

    if len(negative) >= 3 or len(strongly_negative) >= 2:
        level = "warning"
        headline = "MODEL WARNING: similar signals have negative expectancy."
    elif len(negative) >= 1 and len(positive) >= 1:
        level = "caution"
        headline = "MODEL CAUTION: predictions disagree."
    else:
        level = "ok"
        headline = "MODEL OK: no negative consensus."
    return {"version": "2026-06-22", "level": level, "headline": headline, "reasons": reasons, "models": values}


def _shadow_hard_block_v2(signal: dict[str, Any], model_guardrail: dict[str, Any]) -> dict[str, Any]:
    cluster_notional = float(signal.get("liq_total_notional") or 0.0)
    would_block = (
        str(model_guardrail.get("level") or "") == "warning"
        and 100_000.0 <= cluster_notional < 200_000.0
    )
    if would_block:
        level = "hard_block_candidate"
        action = "would_block"
        headline = "SHADOW V2: warning 100K-200K cluster would be blocked."
    else:
        level = "observe"
        action = "observe"
        headline = "SHADOW V2: no hard-block candidate."
    return {
        "name": "guardrail_v2_warning_100k_200k",
        "version": "2026-06-23",
        "action": action,
        "level": level,
        "headline": headline,
        "cluster_notional": cluster_notional,
        "rule_name": signal.get("rule_name"),
        "model_guardrail_level": model_guardrail.get("level"),
        "definition": "model_guardrail=warning AND 100K <= cluster_notional < 200K",
        "live_effect": "none_shadow_only",
    }


def _shadow_hard_block_v4_50k_weak_cluster(
    rule_name: str, signal: dict[str, Any], model_guardrail: dict[str, Any]
) -> dict[str, Any]:
    cluster_notional = float(signal.get("liq_total_notional") or 0.0)
    would_block = (
        rule_name == "ETH_BUY_LIQ_LONG_50K_TP120_SL40_BE30"
        and str(model_guardrail.get("level") or "") == "warning"
        and cluster_notional < 200_000.0
    )
    if would_block:
        level = "hard_block_candidate"
        action = "would_block"
        headline = "SHADOW V4: 50K warning cluster below 200K would be blocked."
    else:
        level = "observe"
        action = "observe"
        headline = "SHADOW V4: no 50K weak-cluster candidate."
    return {
        "name": "guardrail_v4_50k_warning_lt200k",
        "version": "2026-06-24",
        "action": action,
        "level": level,
        "headline": headline,
        "cluster_notional": cluster_notional,
        "rule_name": rule_name,
        "model_guardrail_level": model_guardrail.get("level"),
        "definition": "rule=50K/TP120 AND model_guardrail=warning AND cluster_notional < 200K",
        "live_effect": "none_shadow_only",
        "source_report": "S34_50K_LOSS_POSTMORTEM",
    }


def _neighbor_audit(ledger_conn: sqlite3.Connection | None, rule: S34Rule, signal: dict[str, Any]) -> dict[str, Any]:
    audit = {
        "audit_version": "neighbor_audit_v1",
        "model": "base_rate_v1",
        "rule_name": rule.name,
        "signal_ts_ms": int(signal.get("ts_ms") or 0),
        "cluster_notional": float(signal.get("liq_total_notional") or 0.0),
        "cluster_liq_count": int(signal.get("liq_count") or 0),
        "cluster_shape_label": signal.get("cluster_shape_label") or "",
        "neighbors": [],
        "explanation": "No ledger connection; neighbors unavailable.",
    }
    if ledger_conn is None:
        return audit
    target_notional = max(0.0, float(signal.get("liq_total_notional") or 0.0))
    signal_ts_ms = int(signal.get("ts_ms") or 0)
    target_count = max(0, int(signal.get("liq_count") or 0))
    target_shape = str(signal.get("cluster_shape_label") or "")
    try:
        rows = ledger_conn.execute(
            """
            SELECT o.trade_id, o.exit_reason, o.net_bps, s.signal_ts_ms,
                   s.cluster_notional, s.cluster_liq_count, s.cluster_shape_label, s.features_json
            FROM s34_outcomes o
            JOIN s34_signals s ON s.signal_id=o.signal_id
            WHERE o.rule_name=? AND o.exit_ts_ms<? AND o.net_bps IS NOT NULL
            """,
            (rule.name, signal_ts_ms),
        ).fetchall()
    except sqlite3.Error as exc:
        audit["explanation"] = f"Ledger query failed: {exc!r}"
        return audit
    scored: list[dict[str, Any]] = []
    for row in rows:
        row_notional = max(0.0, float(row["cluster_notional"] or 0.0))
        row_count = max(0, int(row["cluster_liq_count"] or 0))
        row_shape = str(row["cluster_shape_label"] or "")
        try:
            row_features = json.loads(str(row["features_json"] or "{}"))
        except json.JSONDecodeError:
            row_features = {}
        notional_distance = abs(math.log1p(target_notional) - math.log1p(row_notional))
        count_distance = abs(target_count - row_count) / max(1.0, float(max(target_count, row_count, 1)))
        shape_penalty = 0.0 if row_shape == target_shape else 0.75
        score = notional_distance + count_distance + shape_penalty
        similarity = 1.0 / (1.0 + score)
        duration_distance = _scaled_abs_distance(signal.get("cluster_duration_sec"), row_features.get("cluster_duration_sec"), 180.0)
        max_share_distance = _scaled_abs_distance(
            signal.get("cluster_max_single_liq_share"), row_features.get("cluster_max_single_liq_share"), 100.0
        )
        btc_distance = _scaled_abs_distance(signal.get("btc_pre_return_bps"), row_features.get("btc_pre_return_bps"), 100.0)
        v1_score = score + duration_distance + max_share_distance + btc_distance
        v2_score = count_distance + duration_distance + max_share_distance
        scored.append(
            {
                "trade_id": row["trade_id"],
                "exit_reason": row["exit_reason"],
                "net_bps": float(row["net_bps"]),
                "signal_ts_ms": int(row["signal_ts_ms"] or 0),
                "cluster_notional": row_notional,
                "cluster_liq_count": row_count,
                "cluster_shape_label": row_shape,
                "distance": score,
                "similarity": similarity,
                "v1_distance": v1_score,
                "v1_similarity": 1.0 / (1.0 + v1_score),
                "v2_distance": v2_score,
                "v2_similarity": 1.0 / (1.0 + v2_score),
                "cluster_duration_sec": row_features.get("cluster_duration_sec"),
                "cluster_max_single_liq_share": row_features.get("cluster_max_single_liq_share"),
                "btc_pre_return_bps": row_features.get("btc_pre_return_bps"),
            }
        )
    neighbors = sorted(scored, key=lambda item: (float(item["distance"]), -int(item["signal_ts_ms"])))[:5]
    v1_neighbors = sorted(scored, key=lambda item: (float(item["v1_distance"]), -int(item["signal_ts_ms"])))[:5]
    v2_neighbors = sorted(scored, key=lambda item: (float(item["v2_distance"]), -int(item["signal_ts_ms"])))[:5]
    audit["neighbors"] = neighbors
    audit["knn_v0"] = {
        "k": len(neighbors),
        "feature_set": ["log_cluster_notional", "cluster_liq_count_ratio", "shape_match"],
        "distance": "abs(log1p(notional_diff))+normalized_count_diff+shape_penalty",
        "median_net_bps": None,
        "mean_net_bps": None,
        "win_rate": None,
        "avg_similarity": None,
    }
    audit["knn_v1"] = {
        "k": len(v1_neighbors),
        "feature_set": [
            "log_cluster_notional",
            "cluster_liq_count_ratio",
            "shape_match",
            "cluster_duration_sec",
            "max_single_liq_share",
            "btc_pre_return_bps_if_available",
        ],
        "distance": "knn_v0_distance+duration_distance+max_single_share_distance+btc_pre_return_distance",
        "median_net_bps": None,
        "mean_net_bps": None,
        "win_rate": None,
        "avg_similarity": None,
        "neighbors": v1_neighbors,
    }
    audit["knn_v2"] = {
        "k": len(v2_neighbors),
        "feature_set": ["cluster_liq_count_ratio", "cluster_duration_sec", "max_single_liq_share"],
        "distance": "normalized_count_diff+duration_distance+max_single_share_distance",
        "median_net_bps": None,
        "mean_net_bps": None,
        "win_rate": None,
        "avg_similarity": None,
        "neighbors": v2_neighbors,
    }
    if neighbors:
        values = sorted(float(n["net_bps"]) for n in neighbors)
        median_neighbor = values[len(values) // 2]
        audit["knn_v0"].update(
            {
                "median_net_bps": median_neighbor,
                "mean_net_bps": sum(values) / len(values),
                "win_rate": sum(1 for value in values if value > 0) / len(values),
                "avg_similarity": sum(float(n["similarity"]) for n in neighbors) / len(neighbors),
            }
        )
        audit["explanation"] = (
            f"Selected {len(neighbors)} same-rule neighbors by cluster notional, liq_count, and shape. "
            f"Neighbor median net {median_neighbor:.2f} bps."
        )
    if v1_neighbors:
        v1_values = sorted(float(n["net_bps"]) for n in v1_neighbors)
        audit["knn_v1"].update(
            {
                "median_net_bps": v1_values[len(v1_values) // 2],
                "mean_net_bps": sum(v1_values) / len(v1_values),
                "win_rate": sum(1 for value in v1_values if value > 0) / len(v1_values),
                "avg_similarity": sum(float(n["v1_similarity"]) for n in v1_neighbors) / len(v1_neighbors),
            }
        )
    if v2_neighbors:
        v2_values = sorted(float(n["net_bps"]) for n in v2_neighbors)
        audit["knn_v2"].update(
            {
                "median_net_bps": v2_values[len(v2_values) // 2],
                "mean_net_bps": sum(v2_values) / len(v2_values),
                "win_rate": sum(1 for value in v2_values if value > 0) / len(v2_values),
                "avg_similarity": sum(float(n["v2_similarity"]) for n in v2_neighbors) / len(v2_neighbors),
            }
        )
    else:
        audit["explanation"] = "No same-rule closed outcomes available for neighbor audit."
    return audit


def _scaled_abs_distance(left: Any, right: Any, scale: float) -> float:
    if left is None or right is None or left == "" or right == "":
        return 0.0
    try:
        return abs(float(left) - float(right)) / max(1.0, float(scale))
    except (TypeError, ValueError):
        return 0.0


def _save_trades(path: Path, trades: dict[str, dict[str, Any]]) -> None:
    ordered = sorted(trades.values(), key=lambda t: (int(t.get("signal_ts_ms", 0)), str(t.get("trade_id", ""))))
    _write_json(path, {"updated_at_utc": _utc_now_iso(), "trades": ordered})


def _write_journal(path: Path, trades: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "trial_id",
        "status",
        "signal_ts_utc",
        "symbol",
        "direction",
        "rule_name",
        "liq_side",
        "liq_total_notional",
        "liq_count",
        "entry_price",
        "entry_reference_price",
        "entry_fill_source",
        "tp_price",
        "sl_price",
        "be_trigger_price",
        "be_active",
        "exit_ts_utc",
        "exit_price",
        "exit_reference_price",
        "exit_fill_source",
        "exit_reason",
        "gross_bps",
        "mid_to_mid_bps",
        "executable_gross_bps",
        "entry_adverse_bps",
        "exit_adverse_bps",
        "mark_to_fill_cost_bps",
        "spread_cost_bps",
        "entry_spread_bps",
        "exit_spread_bps",
        "fee_cost_bps",
        "net_bps",
        "gross_usdt",
        "net_usdt",
        "simulated_equity_usdt",
        "leverage",
        "risk_per_trade_pct",
        "risk_usdt",
        "notional_usdt",
        "margin_required_usdt",
        "risk_gate_status",
        "risk_gate_reason",
        "regime_filter_enabled",
        "regime_pass",
        "regime_trend_pct",
        "regime_range_pct",
        "regime_buy_liq_notional",
        "regime_agg_trade_count",
        "notes",
    ]
    ordered = sorted(trades.values(), key=lambda t: (int(t.get("signal_ts_ms", 0)), str(t.get("trial_id", ""))))
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for trade in ordered:
            signal = trade.get("signal") or {}
            rule = trade.get("rule") or {}
            risk = trade.get("risk") or {}
            writer.writerow(
                {
                    "trial_id": trade.get("trial_id") or trade.get("trade_id"),
                    "status": trade.get("status"),
                    "signal_ts_utc": trade.get("signal_ts_utc"),
                    "symbol": trade.get("symbol"),
                    "direction": trade.get("direction"),
                    "rule_name": rule.get("name"),
                    "liq_side": rule.get("liq_side"),
                    "liq_total_notional": signal.get("liq_total_notional"),
                    "liq_count": signal.get("liq_count"),
                    "entry_price": trade.get("entry_price"),
                    "entry_reference_price": trade.get("entry_reference_price"),
                    "entry_fill_source": (trade.get("entry_fill") or {}).get("source"),
                    "tp_price": trade.get("tp_price"),
                    "sl_price": trade.get("sl_price"),
                    "be_trigger_price": trade.get("be_trigger_price"),
                    "be_active": trade.get("be_active"),
                    "exit_ts_utc": trade.get("exit_ts_utc"),
                    "exit_price": trade.get("exit_price"),
                    "exit_reference_price": trade.get("exit_reference_price"),
                    "exit_fill_source": (trade.get("exit_fill") or {}).get("source"),
                    "exit_reason": trade.get("exit_reason"),
                    "gross_bps": trade.get("gross_bps"),
                    "mid_to_mid_bps": trade.get("mid_to_mid_bps"),
                    "executable_gross_bps": trade.get("executable_gross_bps"),
                    "entry_adverse_bps": trade.get("entry_adverse_bps"),
                    "exit_adverse_bps": trade.get("exit_adverse_bps"),
                    "mark_to_fill_cost_bps": trade.get("mark_to_fill_cost_bps"),
                    "spread_cost_bps": trade.get("spread_cost_bps"),
                    "entry_spread_bps": trade.get("entry_spread_bps"),
                    "exit_spread_bps": trade.get("exit_spread_bps"),
                    "fee_cost_bps": trade.get("fee_cost_bps"),
                    "net_bps": trade.get("net_bps"),
                    "gross_usdt": trade.get("gross_usdt"),
                    "net_usdt": trade.get("net_usdt"),
                    "simulated_equity_usdt": risk.get("simulated_equity_usdt"),
                    "leverage": risk.get("leverage"),
                    "risk_per_trade_pct": risk.get("risk_per_trade_pct"),
                    "risk_usdt": risk.get("risk_usdt"),
                    "notional_usdt": risk.get("notional_usdt"),
                    "margin_required_usdt": risk.get("margin_required_usdt"),
                    "risk_gate_status": trade.get("risk_gate_status"),
                    "risk_gate_reason": trade.get("risk_gate_reason"),
                    "regime_filter_enabled": (trade.get("regime") or {}).get("thresholds", {}).get("enabled"),
                    "regime_pass": (trade.get("regime") or {}).get("pass"),
                    "regime_trend_pct": (trade.get("regime") or {}).get("trend_pct"),
                    "regime_range_pct": (trade.get("regime") or {}).get("range_pct"),
                    "regime_buy_liq_notional": (trade.get("regime") or {}).get("buy_liq_notional"),
                    "regime_agg_trade_count": (trade.get("regime") or {}).get("agg_trade_count"),
                    "notes": "SYSTEM_GENERATED_PAPER_ONLY_NOT_MANUAL",
                }
            )


def _write_status(path: Path, trades: dict[str, dict[str, Any]], summary: dict[str, Any]) -> None:
    closed = [t for t in trades.values() if t.get("status") == "CLOSED"]
    open_trades = [t for t in trades.values() if t.get("status") == "OPEN"]
    skipped = [t for t in trades.values() if t.get("status") == "SKIPPED"]
    by_reason: dict[str, int] = {}
    net_sum = 0.0
    net_usdt_sum = 0.0
    for trade in closed:
        by_reason[str(trade.get("exit_reason"))] = by_reason.get(str(trade.get("exit_reason")), 0) + 1
        net_sum += float(trade.get("net_bps") or 0.0)
        net_usdt_sum += float(trade.get("net_usdt") or 0.0)
    lines = [
        "# S34 Shadow Paper Status",
        "",
        f"- updated_at_utc: `{_utc_now_iso()}`",
        f"- mode: `PAPER_ONLY_NO_ORDERS`",
        f"- total_trades: `{len(trades)}`",
        f"- open_trades: `{len(open_trades)}`",
        f"- closed_trades: `{len(closed)}`",
        f"- skipped_trades: `{len(skipped)}`",
        f"- net_bps_sum_closed: `{net_sum:.2f}`",
        f"- net_usdt_sum_closed: `{net_usdt_sum:.4f}`",
        f"- exits: `{json.dumps(by_reason, sort_keys=True)}`",
        f"- last_run_new_signals: `{summary.get('new_signals', 0)}`",
        f"- last_run_new_trades: `{summary.get('new_trades', 0)}`",
        "",
        "## Open Trades",
        "",
    ]
    if open_trades:
        lines.append("| trade_id | signal | direction | entry | tp | sl | be_active |")
        lines.append("| --- | --- | --- | ---: | ---: | ---: | --- |")
        for trade in open_trades[-20:]:
            lines.append(
                f"| `{trade['trade_id']}` | `{trade['signal_ts_utc']}` | `{trade['direction']}` | "
                f"{float(trade['entry_price']):.4f} | {float(trade['tp_price']):.4f} | "
                f"{float(trade['sl_price']):.4f} | `{trade['be_active']}` |"
            )
    else:
        lines.append("_No open shadow paper trades._")
    lines.extend(["", "## Recent Closed Trades", ""])
    recent = closed[-20:]
    if recent:
        lines.append("| trade_id | signal | exit | reason | gross bps | net bps |")
        lines.append("| --- | --- | --- | --- | ---: | ---: |")
        for trade in recent:
            lines.append(
                f"| `{trade['trade_id']}` | `{trade['signal_ts_utc']}` | `{trade['exit_ts_utc']}` | "
                f"`{trade['exit_reason']}` | {float(trade['gross_bps']):.2f} | {float(trade['net_bps']):.2f} |"
            )
    else:
        lines.append("_No closed shadow paper trades yet._")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_daily_report(path: Path, trades: dict[str, dict[str, Any]], summary: dict[str, Any]) -> None:
    today = datetime.now(timezone.utc).date().isoformat()
    rows = []
    for trade in trades.values():
        ts = int(trade.get("signal_ts_ms") or 0)
        if ts and _trade_day(ts) == today:
            rows.append(trade)
    closed = [t for t in rows if t.get("status") == "CLOSED"]
    open_trades = [t for t in rows if t.get("status") == "OPEN"]
    skipped = [t for t in rows if t.get("status") == "SKIPPED"]
    net_bps = sum(float(t.get("net_bps") or 0.0) for t in closed)
    net_usdt = sum(float(t.get("net_usdt") or 0.0) for t in closed)
    exits: dict[str, int] = {}
    skipped_reasons: dict[str, int] = {}
    for trade in closed:
        exits[str(trade.get("exit_reason"))] = exits.get(str(trade.get("exit_reason")), 0) + 1
    for trade in skipped:
        reason = str(trade.get("risk_gate_reason") or "UNKNOWN")
        skipped_reasons[reason] = skipped_reasons.get(reason, 0) + 1
    v2_shadow = _daily_shadow_guardrail_summary(today)
    today_median = _format_optional_bps(v2_shadow["today"]["median_net_bps"])
    all_time_median = _format_optional_bps(v2_shadow["all_time"]["median_net_bps"])
    lines = [
        f"# S34 Daily Execution Report - {today}",
        "",
        f"- updated_at_utc: `{_utc_now_iso()}`",
        f"- mode: `PAPER_ONLY_NO_ORDERS`",
        f"- day_trials: `{len(rows)}`",
        f"- closed: `{len(closed)}`",
        f"- open: `{len(open_trades)}`",
        f"- skipped_by_risk: `{len(skipped)}`",
        f"- net_bps_closed: `{net_bps:.2f}`",
        f"- net_usdt_closed: `{net_usdt:.4f}`",
        f"- exits: `{json.dumps(exits, sort_keys=True)}`",
        f"- skipped_reasons: `{json.dumps(skipped_reasons, sort_keys=True)}`",
        f"- v2_shadow_today: `{json.dumps(v2_shadow.get('today', {}), sort_keys=True)}`",
        f"- v2_shadow_all_time: `{json.dumps(v2_shadow.get('all_time', {}), sort_keys=True)}`",
        f"- last_run_new_trades: `{summary.get('new_trades', 0)}`",
        "",
        "## Guardrail V2 Shadow",
        "",
        "| scope | signals | would_block | closed | cum net bps | median net bps | latest |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
        (
            f"| today | {v2_shadow['today']['signals']} | {v2_shadow['today']['would_block']} | "
            f"{v2_shadow['today']['closed']} | {v2_shadow['today']['cum_net_bps']:.2f} | "
            f"{today_median} | "
            f"`{v2_shadow['today']['latest'] or ''}` |"
        ),
        (
            f"| all_time | {v2_shadow['all_time']['signals']} | {v2_shadow['all_time']['would_block']} | "
            f"{v2_shadow['all_time']['closed']} | {v2_shadow['all_time']['cum_net_bps']:.2f} | "
            f"{all_time_median} | "
            f"`{v2_shadow['all_time']['latest'] or ''}` |"
        ),
        "",
        "## Recent Day Trials",
        "",
    ]
    if rows:
        lines.append("| trial | status | signal | direction | entry | exit | reason | net bps | net usdt | risk gate |")
        lines.append("| --- | --- | --- | --- | ---: | ---: | --- | ---: | ---: | --- |")
        for trade in rows[-30:]:
            exit_price = trade.get("exit_price")
            lines.append(
                f"| `{trade.get('trial_id') or trade.get('trade_id')}` | `{trade.get('status')}` | "
                f"`{trade.get('signal_ts_utc')}` | `{trade.get('direction')}` | "
                f"{float(trade.get('entry_price') or 0.0):.4f} | "
                f"{'' if exit_price is None else f'{float(exit_price):.4f}'} | "
                f"`{trade.get('exit_reason') or ''}` | "
                f"{float(trade.get('net_bps') or 0.0):.2f} | "
                f"{float(trade.get('net_usdt') or 0.0):.4f} | "
                f"`{trade.get('risk_gate_status')}` |"
            )
    else:
        lines.append("_No S34 paper trials today._")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _format_optional_bps(value: Any) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return ""


def _daily_shadow_guardrail_summary(today: str) -> dict[str, dict[str, Any]]:
    empty = {
        "signals": 0,
        "would_block": 0,
        "closed": 0,
        "cum_net_bps": 0.0,
        "median_net_bps": None,
        "latest": "",
    }
    db_path = intelligence_ledger.DEFAULT_LEDGER_PATH
    if not db_path.exists():
        return {"today": dict(empty), "all_time": dict(empty)}

    def summarize(rows: list[sqlite3.Row]) -> dict[str, Any]:
        would_block = [row for row in rows if str(row["action"] or "") == "would_block"]
        closed = [row for row in would_block if row["net_bps"] is not None]
        nets = sorted(float(row["net_bps"] or 0.0) for row in closed)
        latest = rows[0] if rows else None
        return {
            "signals": len(rows),
            "would_block": len(would_block),
            "closed": len(closed),
            "cum_net_bps": float(sum(nets)) if nets else 0.0,
            "median_net_bps": None if not nets else (nets[len(nets) // 2] if len(nets) % 2 else (nets[len(nets) // 2 - 1] + nets[len(nets) // 2]) / 2.0),
            "latest": ""
            if latest is None
            else f"{latest['signal_ts_utc']} {latest['action']} {latest['trade_id'] or ''} {'' if latest['net_bps'] is None else f'{float(latest['net_bps']):.2f}'}",
        }

    con = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True, timeout=3.0)
    con.row_factory = sqlite3.Row
    try:
        exists = con.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='s34_shadow_guardrails'"
        ).fetchone()
        if not exists:
            return {"today": dict(empty), "all_time": dict(empty)}
        sql = """
            SELECT
              s.signal_ts_utc,
              sg.action,
              sg.level,
              o.trade_id,
              o.exit_reason,
              o.net_bps
            FROM s34_shadow_guardrails sg
            JOIN s34_signals s ON s.signal_id=sg.signal_id
            LEFT JOIN s34_outcomes o ON o.signal_id=sg.signal_id
            WHERE sg.guardrail_name='guardrail_v2_warning_100k_200k'
            ORDER BY s.signal_ts_ms DESC
        """
        all_rows = list(con.execute(sql).fetchall())
    except sqlite3.Error:
        return {"today": dict(empty), "all_time": dict(empty)}
    finally:
        con.close()
    today_rows = [row for row in all_rows if str(row["signal_ts_utc"] or "").startswith(today)]
    return {"today": summarize(today_rows), "all_time": summarize(all_rows)}


def _state_window(args: argparse.Namespace, conn: sqlite3.Connection, state_path: Path) -> tuple[int, int, dict[str, Any]]:
    state = _read_json(state_path, {})
    if args.start_utc:
        start_ms = _ms_from_iso(str(args.start_utc))
    elif args.backfill_restore_window:
        restore = _read_json(Path("reports/runtime_validation/s34_liq_restore_window.json"), {})
        start_ms = _ms_from_iso(str(restore.get("window_start_utc")))
    elif bool(args.backfill_existing):
        start_ms = 0
    else:
        start_ms = int(state.get("cursor_ts_ms") or _latest_liq_ts(conn, "ETHUSDT") or 0)

    if args.end_utc:
        end_ms = _ms_from_iso(str(args.end_utc))
    else:
        end_ms = min(_latest_liq_ts(conn, "ETHUSDT"), _latest_mark_ts(conn, "ETHUSDT"))

    return int(start_ms), int(end_ms), state


def run_once(args: argparse.Namespace) -> dict[str, Any]:
    state_path = Path(str(args.state_json))
    trades_path = Path(str(args.trades_json))
    jsonl_path = Path(str(args.events_jsonl))
    status_path = Path(str(args.status_md))
    journal_path = Path(str(args.journal_csv))
    daily_report_path = Path(str(args.daily_report_md))
    risk_config = RiskConfig(
        simulated_equity_usdt=float(args.simulated_equity_usdt),
        leverage=float(args.leverage),
        risk_per_trade_pct=float(args.risk_per_trade_pct),
        max_open_trades=int(args.max_open_trades),
        daily_max_loss_pct=float(args.daily_max_loss_pct),
        daily_max_sl=int(args.daily_max_sl),
        cooldown_after_consecutive_sl=int(args.cooldown_after_consecutive_sl),
        cooldown_hours=float(args.cooldown_hours),
        max_mark_staleness_sec=int(args.max_mark_staleness_sec),
    )
    regime_config = RegimeConfig(
        enabled=bool(args.regime_filter_enabled),
        min_trend_pct=float(args.regime_min_trend_pct),
        min_range_pct=float(args.regime_min_range_pct),
        min_buy_liq_notional=float(args.regime_min_buy_liq_notional),
        min_agg_trade_count=int(args.regime_min_agg_trade_count),
    )
    rules = tuple(
        replace(
            rule,
            taker_fee_bps=float(args.taker_fee_bps),
            maker_fee_bps=float(args.maker_fee_bps),
            tp_fill_mode=str(args.tp_fill_mode),
            modeled_spread_bps=float(args.modeled_spread_bps),
            max_book_staleness_sec=int(args.max_book_staleness_sec),
            require_book_ticker_fill=not bool(args.allow_modeled_spread_fill),
            max_open_trades=int(args.max_open_trades),
            daily_max_sl=int(args.daily_max_sl),
        )
        for rule in DEFAULT_RULES
    )
    trades = _load_trades(trades_path)
    _normalize_trades(trades, risk_config)
    events: list[dict[str, Any]] = []
    new_signals = 0
    new_trades = 0
    skipped_trades = 0
    next_trial = _next_trial_number(trades)

    ledger_conn: sqlite3.Connection | None = None
    intelligence_db = str(getattr(args, "intelligence_db", "") or "")
    if intelligence_db and not bool(args.dry_run):
        ledger_conn = intelligence_ledger.connect(intelligence_db)

    conn = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    try:
        start_ms, end_ms, state = _state_window(args, conn, state_path)
        for trade_id, trade in list(trades.items()):
            before = trade.get("status")
            trades[trade_id] = _evaluate_trade(conn, trade, end_ms)
            if before != trades[trade_id].get("status"):
                events.append({"ts": time.time(), "event": "s34_shadow_paper.trade_closed", "data": trades[trade_id]})
                if ledger_conn is not None:
                    intelligence_ledger.record_trade_lifecycle(ledger_conn, trades[trade_id], "CLOSE", str(trades[trade_id].get("exit_reason") or ""))

        candidates: list[tuple[S34Rule, dict[str, Any]]] = []
        # OD-018 persistent-v2: seed per-rule min-gap state from the state file and
        # rescan from the current bucket's floor so a bucket straddling the cursor
        # keeps its full cluster identity (signal_key dedup absorbs re-emissions).
        last_signal_map: dict[str, int] = {
            str(k): int(v)
            for k, v in (state.get("last_signal_ts_ms_by_rule") or {}).items()
        }
        # OD-018-FOLLOWUP: one-time per-rule first-activation migration. Runs
        # exactly once per rule (guarded by presence in migration_provenance,
        # which is persisted every cycle alongside last_signal_map) so restart
        # after a successful migration always reuses the persisted value
        # rather than re-deriving a possibly different one.
        migration_provenance: dict[str, Any] = dict(state.get("min_gap_state_provenance_by_rule") or {})
        migration_initialized_at_utc = state.get("min_gap_state_initialized_at_utc")
        for rule in rules:
            if rule.name in migration_provenance:
                continue
            prov = _derive_min_gap_seed_from_history(trades, rule)
            prov["migrated_at_utc"] = _utc_now_iso()
            prov["migration_version"] = MIN_GAP_STATE_MIGRATION_VERSION
            migration_provenance[rule.name] = prov
            if migration_initialized_at_utc is None:
                migration_initialized_at_utc = prov["migrated_at_utc"]
            if prov["status"] == "DERIVED_FROM_HISTORY" and rule.name not in last_signal_map:
                last_signal_map[rule.name] = int(prov["seed_ts_ms"])
        for rule in rules:
            prov = migration_provenance.get(rule.name) or {}
            if prov.get("status") == "AMBIGUOUS_FAILED":
                continue  # fail closed: no candidates generated until an operator resolves the ambiguity
            bucket_ms = int(rule.bucket_sec) * 1000
            scan_start_ms = (int(start_ms) // bucket_ms) * bucket_ms
            seed = last_signal_map.get(str(rule.name))
            signals = _bucket_events(
                conn,
                rule,
                scan_start_ms,
                end_ms,
                int(args.limit_per_rule),
                last_signal_ms_seed=seed,
            )
            for signal in signals:
                candidates.append((rule, signal))
            if signals:
                newest = max(int(s["ts_ms"]) for s in signals)
                if seed is None or newest > int(seed):
                    last_signal_map[str(rule.name)] = newest
        existing_keys = {str(t.get("signal_key")) for t in trades.values() if t.get("signal_key")}
        for rule, signal in sorted(candidates, key=_candidate_sort_key):
            trade = _paper_trade_from_signal(rule, signal, risk_config)
            if trade["signal_key"] in existing_keys:
                continue
            new_signals += 1
            if ledger_conn is not None:
                signal_id = intelligence_ledger.record_signal(ledger_conn, rule, signal)
                prediction = _base_rate_prediction(ledger_conn, rule, signal)
                audit = _neighbor_audit(ledger_conn, rule, signal)
                knn = audit.get("knn_v0") or {}
                prediction["knn_v0_expected_net_bps"] = knn.get("median_net_bps")
                prediction["knn_v0_win_rate"] = knn.get("win_rate")
                prediction["knn_v0_k"] = knn.get("k")
                prediction["knn_v0_avg_similarity"] = knn.get("avg_similarity")
                knn_v0_prediction = _knn_prediction(rule, signal, audit, "knn_v0")
                knn_v1_prediction = _knn_prediction(rule, signal, audit, "knn_v1")
                knn_v2_prediction = _knn_prediction(rule, signal, audit, "knn_v2")
                intelligence_ledger.record_prediction(
                    ledger_conn,
                    signal_id,
                    "base_rate_v1",
                    "2026-06-22",
                    prediction,
                )
                intelligence_ledger.record_prediction(
                    ledger_conn,
                    signal_id,
                    "knn_v0",
                    "2026-06-22",
                    knn_v0_prediction,
                )
                intelligence_ledger.record_prediction(
                    ledger_conn,
                    signal_id,
                    "knn_v1",
                    "2026-06-22",
                    knn_v1_prediction,
                )
                intelligence_ledger.record_prediction(
                    ledger_conn,
                    signal_id,
                    "knn_v2",
                    "2026-06-22",
                    knn_v2_prediction,
                )
                intelligence_ledger.record_model_audit(
                    ledger_conn,
                    signal_id,
                    "base_rate_v1",
                    audit,
                )
                model_guardrail = _model_guardrail([prediction, knn_v0_prediction, knn_v1_prediction, knn_v2_prediction])
                intelligence_ledger.record_model_guardrail(
                    ledger_conn,
                    signal_id,
                    model_guardrail,
                )
                intelligence_ledger.record_shadow_guardrail(
                    ledger_conn,
                    signal_id,
                    _shadow_hard_block_v2(signal, model_guardrail),
                )
                intelligence_ledger.record_shadow_guardrail(
                    ledger_conn,
                    signal_id,
                    _shadow_hard_block_v4_50k_weak_cluster(rule.name, signal, model_guardrail),
                )
            deprecated_reason = _deprecated_paper_rule_reason(rule)
            if deprecated_reason:
                skipped_trades += 1
                trade = _skipped_trade_from_signal(rule, signal, risk_config, deprecated_reason)
                trade["deprecated_rule"] = True
                trade["deprecation_note"] = DEPRECATED_PAPER_RULE_NOTE
                trade = _assign_trial_id(trade, next_trial)
                next_trial += 1
                trades[trade["trade_id"]] = trade
                existing_keys.add(str(trade["signal_key"]))
                events.append({"ts": time.time(), "event": "s34_shadow_paper.trade_skipped", "data": trade})
                if ledger_conn is not None:
                    intelligence_ledger.record_trade_lifecycle(ledger_conn, trade, "REJECT", deprecated_reason)
                continue
            if signal.get("fill_error"):
                skipped_trades += 1
                trade = _skipped_trade_from_signal(rule, signal, risk_config, "NO_FILL_DATA")
                trade["fill_error"] = signal.get("fill_error")
                trade = _assign_trial_id(trade, next_trial)
                next_trial += 1
                trades[trade["trade_id"]] = trade
                existing_keys.add(str(trade["signal_key"]))
                events.append({"ts": time.time(), "event": "s34_shadow_paper.trade_skipped", "data": trade})
                if ledger_conn is not None:
                    intelligence_ledger.record_trade_lifecycle(ledger_conn, trade, "REJECT", "NO_FILL_DATA")
                continue
            regime_ok, regime_reason, regime = _regime_gate(conn, rule, signal, regime_config)
            if regime:
                regime["pass"] = bool(regime_ok)
                trade["regime"] = regime
            if not regime_ok:
                skipped_trades += 1
                trade = _skipped_trade_from_signal(rule, signal, risk_config, regime_reason)
                trade["regime"] = regime
                trade = _assign_trial_id(trade, next_trial)
                next_trial += 1
                trades[trade["trade_id"]] = trade
                existing_keys.add(str(trade["signal_key"]))
                events.append({"ts": time.time(), "event": "s34_shadow_paper.trade_skipped", "data": trade})
                if ledger_conn is not None:
                    intelligence_ledger.record_trade_lifecycle(ledger_conn, trade, "REJECT", str(regime_reason))
                continue
            cluster_key = _signal_cluster_key(rule, signal)
            owner = _cluster_owner(trades, cluster_key)
            if owner:
                skipped_trades += 1
                trade = _skipped_trade_from_signal(rule, signal, risk_config, "SAME_CLUSTER_LOWER_PRIORITY")
                trade["cluster_owner_trade_id"] = owner.get("trade_id")
                trade["cluster_owner_rule"] = _trade_rule_name(owner)
                trade = _assign_trial_id(trade, next_trial)
                next_trial += 1
                trades[trade["trade_id"]] = trade
                existing_keys.add(str(trade["signal_key"]))
                events.append({"ts": time.time(), "event": "s34_shadow_paper.trade_skipped", "data": trade})
                if ledger_conn is not None:
                    intelligence_ledger.record_trade_lifecycle(ledger_conn, trade, "REJECT", "SAME_CLUSTER_LOWER_PRIORITY")
                continue
            if bool(getattr(args, "quality_gate_enabled", False)):
                qg_pass, qg_reason = _quality_gate(
                    conn,
                    int(signal["ts_ms"]),
                    min_eclipse_score=float(getattr(args, "quality_gate_min_eclipse", 42.0)),
                    allow_standard_confidence=bool(getattr(args, "quality_gate_allow_standard", False)),
                )
                if not qg_pass:
                    skipped_trades += 1
                    trade = _skipped_trade_from_signal(rule, signal, risk_config, qg_reason)
                    trade = _assign_trial_id(trade, next_trial)
                    next_trial += 1
                    trades[trade["trade_id"]] = trade
                    existing_keys.add(str(trade["signal_key"]))
                    events.append({"ts": time.time(), "event": "s34_shadow_paper.trade_skipped", "data": trade})
                    if ledger_conn is not None:
                        intelligence_ledger.record_trade_lifecycle(ledger_conn, trade, "REJECT", qg_reason)
                    continue

            accepted, reason = _risk_gate(trades, signal, risk_config, rule)
            if not accepted:
                skipped_trades += 1
                trade = _skipped_trade_from_signal(rule, signal, risk_config, reason)
                trade = _assign_trial_id(trade, next_trial)
                next_trial += 1
                trades[trade["trade_id"]] = trade
                existing_keys.add(str(trade["signal_key"]))
                events.append({"ts": time.time(), "event": "s34_shadow_paper.trade_skipped", "data": trade})
                if ledger_conn is not None:
                    intelligence_ledger.record_trade_lifecycle(ledger_conn, trade, "REJECT", str(reason))
                continue
            new_trades += 1
            trade = _assign_trial_id(trade, next_trial)
            next_trial += 1
            opposite_dir = "SHORT" if str(rule.direction).upper() == "LONG" else "LONG"
            cross_open = [
                t for t in trades.values()
                if t.get("status") == "OPEN"
                and _trade_symbol(t) == str(rule.symbol)
                and _trade_direction(t) == opposite_dir
            ]
            if cross_open:
                trade["cross_direction_conflict"] = True
                trade["cross_direction_open_rules"] = [_trade_rule_name(t) for t in cross_open]
                events.append({"ts": time.time(), "event": "s34_shadow_paper.cross_direction_conflict", "data": {
                    "symbol": rule.symbol,
                    "new_direction": str(rule.direction).upper(),
                    "new_rule": rule.name,
                    "conflicting_rules": trade["cross_direction_open_rules"],
                }})
            trade = _evaluate_trade(conn, trade, end_ms)
            trades[trade["trade_id"]] = trade
            existing_keys.add(str(trade["signal_key"]))
            events.append({"ts": time.time(), "event": "s34_shadow_paper.trade_opened", "data": trade})
            if ledger_conn is not None:
                intelligence_ledger.record_trade_lifecycle(ledger_conn, trade, "ACCEPT", "")
            if trade.get("status") == "CLOSED":
                events.append({"ts": time.time(), "event": "s34_shadow_paper.trade_closed", "data": trade})
                if ledger_conn is not None:
                    intelligence_ledger.record_trade_lifecycle(ledger_conn, trade, "CLOSE", str(trade.get("exit_reason") or ""))
    finally:
        conn.close()
        if ledger_conn is not None:
            ledger_conn.commit()
            ledger_conn.close()

    cursor_ts_ms = end_ms
    if bool(args.dry_run):
        summary = {
            "dry_run": True,
            "new_signals": new_signals,
            "new_trades": new_trades,
            "skipped_trades": skipped_trades,
            "events": events[:10],
        }
        return summary

    _save_trades(trades_path, trades)
    _write_journal(journal_path, trades)
    _append_jsonl(jsonl_path, events)
    _write_json(
        state_path,
        {
            "updated_at_utc": _utc_now_iso(),
            "cursor_ts_ms": cursor_ts_ms,
            "cursor_ts_utc": _iso_from_ms(cursor_ts_ms),
            "min_gap_semantics": MIN_GAP_SEMANTICS_VERSION,
            "last_signal_ts_ms_by_rule": last_signal_map,
            "min_gap_state_migration_version": MIN_GAP_STATE_MIGRATION_VERSION,
            "min_gap_state_initialized_at_utc": migration_initialized_at_utc,
            "min_gap_state_provenance_by_rule": migration_provenance,
        },
    )
    summary = {
        "updated_at_utc": _utc_now_iso(),
        "paper_only": True,
        "new_signals": new_signals,
        "new_trades": new_trades,
        "skipped_trades": skipped_trades,
        "total_trades": len(trades),
        "open_trades": sum(1 for t in trades.values() if t.get("status") == "OPEN"),
        "closed_trades": sum(1 for t in trades.values() if t.get("status") == "CLOSED"),
        "risk_skipped_trades": sum(1 for t in trades.values() if t.get("status") == "SKIPPED"),
        "cursor_ts_ms": cursor_ts_ms,
        "cursor_ts_utc": _iso_from_ms(cursor_ts_ms),
        "min_gap_semantics": MIN_GAP_SEMANTICS_VERSION,
        "events_jsonl": str(jsonl_path),
        "trades_json": str(trades_path),
        "journal_csv": str(journal_path),
        "daily_report_md": str(daily_report_path),
        "intelligence_db": intelligence_db,
        "risk_config": asdict(risk_config),
        "regime_config": asdict(regime_config),
        "fill_model": {
            "entry": "taker_executable_side",
            "sl_be_time_exit": "taker_executable_side",
            "tp_fill_mode": str(args.tp_fill_mode),
            "taker_fee_bps": float(args.taker_fee_bps),
            "maker_fee_bps": float(args.maker_fee_bps),
            "modeled_spread_bps": float(args.modeled_spread_bps),
            "max_book_staleness_sec": int(args.max_book_staleness_sec),
            "require_book_ticker_fill": not bool(args.allow_modeled_spread_fill),
            "cost_identity": "net=gross-entry_adverse-exit_adverse-spread-fee",
        },
    }
    _write_json(Path(str(args.summary_json)), summary)
    _write_status(status_path, trades, summary)
    _write_daily_report(daily_report_path, trades, summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run S34 shadow paper signals from restored liquidation data.")
    parser.add_argument("--db", default="data/microstructure.db")
    parser.add_argument("--state-json", default="reports/research/s34/S34_SHADOW_PAPER_STATE.json")
    parser.add_argument("--trades-json", default="reports/research/s34/S34_SHADOW_PAPER_TRADES.json")
    parser.add_argument("--events-jsonl", default="logs/s34_shadow_paper_events.jsonl")
    parser.add_argument("--summary-json", default="reports/research/s34/S34_SHADOW_PAPER_STATUS.json")
    parser.add_argument("--status-md", default="reports/research/s34/S34_SHADOW_PAPER_STATUS.md")
    parser.add_argument("--journal-csv", default="reports/research/s34/S34_SHADOW_PAPER_JOURNAL.csv")
    parser.add_argument("--daily-report-md", default="reports/research/s34/S34_DAILY_EXECUTION_REPORT.md")
    parser.add_argument("--intelligence-db", default="data/s34_intelligence.db")
    parser.add_argument("--simulated-equity-usdt", type=float, default=DEFAULT_RISK.simulated_equity_usdt)
    parser.add_argument("--leverage", type=float, default=DEFAULT_RISK.leverage)
    parser.add_argument("--risk-per-trade-pct", type=float, default=DEFAULT_RISK.risk_per_trade_pct)
    parser.add_argument("--max-open-trades", type=int, default=DEFAULT_RISK.max_open_trades)
    parser.add_argument("--daily-max-loss-pct", type=float, default=DEFAULT_RISK.daily_max_loss_pct)
    parser.add_argument("--daily-max-sl", type=int, default=DEFAULT_RISK.daily_max_sl)
    parser.add_argument("--cooldown-after-consecutive-sl", type=int, default=DEFAULT_RISK.cooldown_after_consecutive_sl)
    parser.add_argument("--cooldown-hours", type=float, default=DEFAULT_RISK.cooldown_hours)
    parser.add_argument("--max-mark-staleness-sec", type=int, default=DEFAULT_RISK.max_mark_staleness_sec)
    parser.add_argument("--regime-filter-enabled", action="store_true")
    parser.add_argument("--regime-min-trend-pct", type=float, default=DEFAULT_REGIME.min_trend_pct)
    parser.add_argument("--regime-min-range-pct", type=float, default=DEFAULT_REGIME.min_range_pct)
    parser.add_argument("--regime-min-buy-liq-notional", type=float, default=DEFAULT_REGIME.min_buy_liq_notional)
    parser.add_argument("--regime-min-agg-trade-count", type=int, default=DEFAULT_REGIME.min_agg_trade_count)
    parser.add_argument("--taker-fee-bps", type=float, default=DEFAULT_RULES[0].taker_fee_bps)
    parser.add_argument("--maker-fee-bps", type=float, default=DEFAULT_RULES[0].maker_fee_bps)
    parser.add_argument("--tp-fill-mode", choices=("maker", "taker"), default=DEFAULT_RULES[0].tp_fill_mode)
    parser.add_argument("--modeled-spread-bps", type=float, default=DEFAULT_RULES[0].modeled_spread_bps)
    parser.add_argument("--max-book-staleness-sec", type=int, default=DEFAULT_RULES[0].max_book_staleness_sec)
    parser.add_argument("--allow-modeled-spread-fill", action="store_true")
    parser.add_argument("--start-utc", default="")
    parser.add_argument("--end-utc", default="")
    parser.add_argument("--backfill-existing", action="store_true")
    parser.add_argument("--backfill-restore-window", action="store_true")
    parser.add_argument("--limit-per-rule", type=int, default=200)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--interval-sec", type=int, default=60)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--quality-gate-enabled", action="store_true",
                        help="Filter signals by confidence_band + eclipse_score from detector_signals")
    parser.add_argument("--quality-gate-min-eclipse", type=float, default=42.0,
                        help="Minimum eclipse_score to accept (default: 42.0, below Q3)")
    parser.add_argument("--quality-gate-allow-standard", action="store_true",
                        help="Allow confidence=standard signals (default: block them)")
    args = parser.parse_args()

    while True:
        summary = run_once(args)
        _safe_print_summary(summary)
        if not bool(args.loop):
            return 0
        time.sleep(max(5, int(args.interval_sec)))


if __name__ == "__main__":
    raise SystemExit(main())
