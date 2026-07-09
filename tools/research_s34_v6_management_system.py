"""S34 v6 live-rule management system report.

Research/risk-only. Builds management artifacts around the existing live-like
route without changing entry filters or live order logic:

- tail-aware sizing for the irreducible tail;
- defensive dissipation observer spec, shadow-only;
- regime degradation monitor from the shadow mirror ledger;
- descriptive failure-mode classifier for large historical losers;
- ETH provision queue realism using agg_trade volume through the quote.
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
V4_JSON = ROOT / "reports" / "research" / "s34" / "S34_V4_DISSIPATION_MANAGEMENT_BACKTEST.json"
SHADOW_LEDGER = ROOT / "reports" / "research" / "s34" / "S34_V_ENGINE_V0_2_SHADOW_MIRROR_LEDGER.jsonl"
SHADOW_BRIEF = ROOT / "reports" / "research" / "s34" / "S34_V_ENGINE_V0_2_SHADOW_MIRROR_BRIEF.json"
DEFAULT_DB = ROOT / "data" / "microstructure.db"
OUT_JSON = ROOT / "reports" / "research" / "s34" / "S34_V6_MANAGEMENT_SYSTEM.json"
OUT_MD = ROOT / "reports" / "research" / "s34" / "S34_V6_MANAGEMENT_SYSTEM.md"


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


def iso_week(ts_ms: int) -> str:
    dt = datetime.fromtimestamp(int(ts_ms) / 1000.0, tz=timezone.utc)
    y, w, _ = dt.isocalendar()
    return f"{y}-W{w:02d}"


def month_of(ts_ms: int) -> str:
    return datetime.fromtimestamp(int(ts_ms) / 1000.0, tz=timezone.utc).strftime("%Y-%m")


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            out.append(obj)
    return out


def split_months(rows: list[dict[str, Any]]) -> tuple[set[str], dict[str, Any]]:
    months = sorted({str(r["month"]) for r in rows if r.get("month")})
    hold_n = max(1, len(months) // 3) if months else 0
    hold_months = set(months[-hold_n:]) if hold_n else set()
    return hold_months, {"months": months, "holdout_months": sorted(hold_months), "method": "chronological_month_tail"}


def summarize_rows(rows: list[dict[str, Any]], key: str = "net_bps") -> dict[str, Any]:
    hold_months, split = split_months(rows)
    return {
        "split": split,
        "all": metrics([float(r[key]) for r in rows if finite(r.get(key)) is not None]),
        "cal": metrics([float(r[key]) for r in rows if str(r.get("month")) not in hold_months and finite(r.get(key)) is not None]),
        "hold": metrics([float(r[key]) for r in rows if str(r.get("month")) in hold_months and finite(r.get(key)) is not None]),
    }


def tail_aware_sizing(pool_rows: list[dict[str, Any]], shadow_rows: list[dict[str, Any]], *, equity_usdt: float, leverage: float) -> dict[str, Any]:
    hist_vals = [float(r["net_bps"]) for r in pool_rows if finite(r.get("net_bps")) is not None]
    hist_min = min(hist_vals) if hist_vals else -410.0
    live_tail_floor = -410.0
    stress_tail_bps = max(abs(hist_min), abs(live_tail_floor)) * 1.25
    shadow_vals = [float(r["net_bps"]) for r in shadow_rows if finite(r.get("net_bps")) is not None]
    shadow_stats = metrics(shadow_vals)
    rows = []
    for risk_pct in (0.25, 0.5, 1.0, 2.0, 5.0):
        risk_usdt = float(equity_usdt) * risk_pct / 100.0
        notional = risk_usdt / (stress_tail_bps / 10_000.0)
        rows.append(
            {
                "risk_pct_equity": risk_pct,
                "risk_usdt": r1(risk_usdt),
                "max_notional_usdt": r1(notional),
                "max_margin_usdt_at_leverage": r1(notional / float(leverage)),
            }
        )
    env_style_margin = float(equity_usdt) * 0.85
    env_style_notional = env_style_margin * float(leverage)
    env_style_tail_loss = env_style_notional * stress_tail_bps / 10_000.0
    return {
        "equity_usdt_assumption": float(equity_usdt),
        "leverage_assumption": float(leverage),
        "historical_min_bps": r1(hist_min),
        "tail_floor_bps": -abs(live_tail_floor),
        "stress_tail_bps_abs": r1(stress_tail_bps),
        "sizing_table": rows,
        "current_env_style_85pct_margin": {
            "margin_usdt": r1(env_style_margin),
            "notional_usdt": r1(env_style_notional),
            "stress_tail_loss_usdt": r1(env_style_tail_loss),
            "stress_tail_loss_pct_equity": r1(100.0 * env_style_tail_loss / float(equity_usdt)) if equity_usdt else None,
        },
        "kelly_note": "Kelly fraction forced to 0 for promotion because edge is unvalidated; use tail-budget sizing only.",
        "shadow_forward_summary": shadow_stats,
    }


def dissipation_observer_spec(v4: dict[str, Any]) -> dict[str, Any]:
    primary = v4.get("primary_tau120_dual_median") or {}
    consistent = v4.get("consistent_cal_hold_positive") or []
    return {
        "status": "SHADOW_ONLY_NO_ORDER_CHANGE",
        "reason": "v4 cut tails but did not validate positive expectancy; log recommendations only.",
        "observe_at_sec": [60, 120, 180],
        "primary_reference": {
            "config_id": primary.get("config_id"),
            "cuts": primary.get("cuts"),
            "hold_delta": (primary.get("hold") or {}).get("delta_sum_bps"),
            "cal_delta": (primary.get("cal") or {}).get("delta_sum_bps"),
            "read": "tail cut in holdout, calibration worsened; not an execution rule",
        },
        "conservative_shadow_reference": consistent[:3],
        "logged_decisions": ["HOLD", "TIGHTEN", "EXIT_EARLY"],
        "hard_rule": "Never alter live orders from this observer without operator sign-off and forward validation.",
    }


def regime_degradation_monitor(shadow_rows: list[dict[str, Any]], brief: dict[str, Any]) -> dict[str, Any]:
    closed = [r for r in shadow_rows if r.get("observation_status") == "CLOSED" and finite(r.get("net_bps")) is not None]
    by_week: dict[str, list[float]] = {}
    for r in closed:
        ts = int(r.get("exit_ts_ms") or r.get("maker_fill_ts_ms") or r.get("signal_ts_ms") or 0)
        by_week.setdefault(iso_week(ts), []).append(float(r["net_bps"]))
    weekly = [{"week": w, **metrics(v)} for w, v in sorted(by_week.items())]
    last5 = [float(r["net_bps"]) for r in sorted(closed, key=lambda x: int(x.get("exit_ts_ms") or x.get("signal_ts_ms") or 0))[-5:]]
    last10 = [float(r["net_bps"]) for r in sorted(closed, key=lambda x: int(x.get("exit_ts_ms") or x.get("signal_ts_ms") or 0))[-10:]]
    triggers = []
    if len(last5) >= 5 and sum(last5) < 0:
        triggers.append("ROLLING_5_SUM_NEGATIVE")
    if len(last10) >= 10 and sum(last10) < 0:
        triggers.append("ROLLING_10_SUM_NEGATIVE")
    if any(v <= -100.0 for v in last10):
        triggers.append("ROLLING_TAIL_LT_-100")
    n = len(closed)
    status = "DATA_INSUFFICIENT" if n < 20 else ("TRIGGERED" if triggers else "OK")
    return {
        "status": status,
        "closed_n": n,
        "all_closed": metrics([float(r["net_bps"]) for r in closed]),
        "last5": metrics(last5),
        "last10": metrics(last10),
        "weekly": weekly,
        "triggers": triggers,
        "brief_recent_kill_check": ((brief.get("recent") or {}).get("kill_check") if isinstance(brief, dict) else None),
        "rules": {
            "observe_only_data_min": 20,
            "rolling_5_sum_negative": "alarm",
            "rolling_10_sum_negative": "alarm",
            "any_rolling_tail_lt_-100": "alarm",
            "30_or_60_day_sum_negative_after_5_closed": "kill_or_dormant_review",
        },
    }


def classify_failure(row: dict[str, Any], accel_q75: float) -> str:
    net = float(row.get("net_bps") or 0.0)
    sync = str(row.get("sync_gate") or "")
    bid_gate = str(row.get("bid_depth_gate") or "")
    absorption = str(row.get("absorption_gate") or "")
    imb = float(row.get("book_imbalance") or 0.0)
    accel = float(row.get("running_accel") or 0.0)
    concurrent = float(row.get("market_concurrent_k") or 0.0)
    if net > -100:
        return "NOT_LARGE_LOSS"
    if sync == "sync" or concurrent >= 200.0:
        return "MARKET_WIDE_DELEVERAGING"
    if absorption == "vacuum_like" or bid_gate == "shallow_bid":
        return "LIQUIDITY_VACUUM_ADVERSE_SELECTION"
    if imb > 0.5 and net <= -100:
        return "BID_WALL_FAILED_TRAP"
    if accel >= accel_q75:
        return "ACCELERATION_RUNAWAY"
    return "UNCLASSIFIED_NEGATIVE_SKEW"


def failure_mode_classifier(pool_rows: list[dict[str, Any]]) -> dict[str, Any]:
    vals = [float(r.get("running_accel") or 0.0) for r in pool_rows if finite(r.get("running_accel")) is not None]
    vals = sorted(vals)
    accel_q75 = vals[int(0.75 * (len(vals) - 1))] if vals else 0.0
    losers = [r for r in pool_rows if finite(r.get("net_bps")) is not None and float(r["net_bps"]) <= -100.0]
    counts: dict[str, int] = {}
    samples = []
    for r in losers:
        label = classify_failure(r, accel_q75)
        counts[label] = counts.get(label, 0) + 1
        if len(samples) < 20:
            samples.append(
                {
                    "label": label,
                    "symbol": r.get("symbol"),
                    "route_id": r.get("route_id"),
                    "month": r.get("month"),
                    "net_bps": r1(float(r["net_bps"])),
                    "sync_gate": r.get("sync_gate"),
                    "bid_depth_gate": r.get("bid_depth_gate"),
                    "absorption_gate": r.get("absorption_gate"),
                    "book_imbalance": r1(float(r.get("book_imbalance") or 0.0)),
                    "running_accel": r1(float(r.get("running_accel") or 0.0)),
                }
            )
    return {
        "status": "DESCRIPTIVE_ONLY_NOT_ENTRY_FILTER",
        "large_loss_threshold_bps": -100.0,
        "large_loss_n": len(losers),
        "counts": dict(sorted(counts.items(), key=lambda kv: kv[1], reverse=True)),
        "accel_q75": r1(accel_q75),
        "samples": samples,
    }


def first_book_cross(conn: sqlite3.Connection, start_ms: int, end_ms: int, *, side: str, quote: float, cross_bps: float) -> dict[str, Any] | None:
    # OUT-OF-SCOPE for ASOF V9: forward-scanning bounded range read with a
    # price predicate (`ask_price<=?`/`bid_price>=?`, ORDER BY ts_ms ASC
    # LIMIT 1), not the `ORDER BY ts_ms DESC LIMIT 1` as-of point lookup the
    # helper migrates. Left on direct SQL.
    if side == "BID":
        px = float(quote) * (1.0 - float(cross_bps) / 10_000.0)
        row = conn.execute(
            """
            SELECT ts_ms, bid_price, ask_price, mid_price
            FROM book_ticker
            WHERE symbol='ETHUSDT' AND ts_ms>=? AND ts_ms<=? AND ask_price<=?
            ORDER BY ts_ms LIMIT 1
            """,
            (int(start_ms), int(end_ms), px),
        ).fetchone()
    else:
        px = float(quote) * (1.0 + float(cross_bps) / 10_000.0)
        row = conn.execute(
            """
            SELECT ts_ms, bid_price, ask_price, mid_price
            FROM book_ticker
            WHERE symbol='ETHUSDT' AND ts_ms>=? AND ts_ms<=? AND bid_price>=?
            ORDER BY ts_ms LIMIT 1
            """,
            (int(start_ms), int(end_ms), px),
        ).fetchone()
    if not row:
        return None
    return {"ts_ms": int(row[0]), "bid": float(row[1]), "ask": float(row[2]), "mid": float(row[3])}


def book_at_or_before(conn: sqlite3.Connection, ts_ms: int) -> dict[str, Any] | None:
    """Direct-SQL oracle -- kept unchanged as the parity reference for
    `book_at_or_before_v2` (BATCH-STORAGE-ROTATION-RETENTION-ASOF-LOOKUP-
    CONSUMER-MIGRATION-V9). No longer called by `simulate_tick_queue_event`;
    the reader-backed path is used instead."""
    row = conn.execute(
        """
        SELECT ts_ms, bid_price, ask_price, mid_price
        FROM book_ticker
        WHERE symbol='ETHUSDT' AND ts_ms<=?
        ORDER BY ts_ms DESC LIMIT 1
        """,
        (int(ts_ms),),
    ).fetchone()
    if not row:
        return None
    return {"ts_ms": int(row[0]), "bid": float(row[1]), "ask": float(row[2]), "mid": float(row[3])}


_BOOK_COLS = ("ts_ms", "bid_price", "ask_price", "mid_price")


def book_at_or_before_v2(root, ts_ms: int, source_db_path=None) -> dict[str, Any] | None:
    """Reader-backed replacement for `book_at_or_before`, via
    lookup_latest_at_or_before. Symbol is hardcoded ETHUSDT directly in the
    oracle's SQL (there is no symbol parameter at all in this function --
    it never varies). book_ticker has no ETHUSDT archive partition (only
    SOLUSDT/2026-04 is archived), so real production use of this file
    resolves SQLITE_ONLY -- confirmed, not assumed."""
    result = RR.lookup_latest_at_or_before(
        root, table="book_ticker", symbol="ETHUSDT", ts_ms=int(ts_ms),
        columns=_BOOK_COLS, source_db_path=source_db_path)
    if not result.found:
        return None
    row_ts, bid, ask, mid = result.row
    return {"ts_ms": int(row_ts), "bid": float(bid), "ask": float(ask), "mid": float(mid)}


def through_quote_notional(conn: sqlite3.Connection, start_ms: int, end_ms: int, *, side: str, quote: float) -> float:
    # OUT-OF-SCOPE for ASOF V9: aggregate SUM over a bounded range read on
    # agg_trades (ts_ms BETWEEN + price/is_buyer_maker predicate), not a
    # point lookup. Left on direct SQL.
    if side == "BID":
        # Buyer is maker => taker sell hitting bids.
        row = conn.execute(
            """
            SELECT COALESCE(SUM(notional), 0)
            FROM agg_trades
            WHERE symbol='ETHUSDT' AND ts_ms>=? AND ts_ms<=? AND is_buyer_maker=1 AND price<=?
            """,
            (int(start_ms), int(end_ms), float(quote)),
        ).fetchone()
    else:
        # Buyer is taker => taker buy lifting asks.
        row = conn.execute(
            """
            SELECT COALESCE(SUM(notional), 0)
            FROM agg_trades
            WHERE symbol='ETHUSDT' AND ts_ms>=? AND ts_ms<=? AND is_buyer_maker=0 AND price>=?
            """,
            (int(start_ms), int(end_ms), float(quote)),
        ).fetchone()
    return float(row[0] or 0.0)


def eth_events(pool_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    seen = set()
    for r in pool_rows:
        if str(r.get("symbol")) != "ETHUSDT":
            continue
        if str(r.get("vdepth_band")) not in {"v28_40", "v40_60", "v60_plus"}:
            continue
        ts = int(r.get("entry_ts_ms") or 0)
        if ts <= 0 or ts in seen:
            continue
        mid = finite(r.get("mid"))
        if mid is None or mid <= 0:
            continue
        seen.add(ts)
        out.append(r)
    return sorted(out, key=lambda r: int(r["entry_ts_ms"]))


def simulate_tick_queue_event(
    conn: sqlite3.Connection,
    event: dict[str, Any],
    *,
    root,
    source_db_path=None,
    offset_bps: float,
    horizon_sec: int,
    cross_bps: float,
    queue_notional_usd: float,
    maker_fee_bps: float,
    taker_fee_bps: float,
) -> dict[str, Any] | None:
    ts = int(event["entry_ts_ms"])
    end_ts = ts + int(horizon_sec) * 1000
    mid = float(event["mid"])
    quote_bid = mid * (1.0 - float(offset_bps) / 10_000.0)
    quote_ask = mid * (1.0 + float(offset_bps) / 10_000.0)
    bid_cross = first_book_cross(conn, ts, end_ts, side="BID", quote=quote_bid, cross_bps=cross_bps)
    ask_cross = first_book_cross(conn, ts, end_ts, side="ASK", quote=quote_ask, cross_bps=cross_bps)
    bid_fill = False
    ask_fill = False
    bid_through = ask_through = 0.0
    if bid_cross:
        bid_through = through_quote_notional(conn, ts, end_ts, side="BID", quote=quote_bid)
        bid_fill = bid_through >= float(queue_notional_usd)
    if ask_cross:
        ask_through = through_quote_notional(conn, ts, end_ts, side="ASK", quote=quote_ask)
        ask_fill = ask_through >= float(queue_notional_usd)
    end = book_at_or_before_v2(root, end_ts, source_db_path=source_db_path)
    if not end:
        return None
    fill_count = int(bid_fill) + int(ask_fill)
    if fill_count == 0:
        net = 0.0
        state = "NO_FILL"
    elif fill_count == 2:
        net = (quote_ask - quote_bid) / mid * 10_000.0 - 2.0 * float(maker_fee_bps)
        state = "BOTH_FILLED"
    elif bid_fill:
        gross = (float(end["bid"]) - quote_bid) / quote_bid * 10_000.0
        net = gross - float(maker_fee_bps) - float(taker_fee_bps)
        state = "BID_ONLY_LONG_INVENTORY"
    else:
        gross = (quote_ask - float(end["ask"])) / quote_ask * 10_000.0
        net = gross - float(maker_fee_bps) - float(taker_fee_bps)
        state = "ASK_ONLY_SHORT_INVENTORY"
    return {
        "month": str(event["month"]),
        "entry_ts_ms": ts,
        "fill_state": state,
        "fill_count": fill_count,
        "bid_through_usd": bid_through,
        "ask_through_usd": ask_through,
        "net_bps": net,
    }


def tick_level_execution_realism(db: Path, pool_rows: list[dict[str, Any]]) -> dict[str, Any]:
    events = eth_events(pool_rows)
    configs = []
    root, _ = PR.resolve_production_root()
    source_db_path = str(db)
    with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as conn:
        for queue_usd in (0.0, 1_000.0, 5_000.0, 10_000.0, 25_000.0):
            for fee in (-0.5, 0.0, 0.5):
                rows = []
                for ev in events:
                    item = simulate_tick_queue_event(
                        conn,
                        ev,
                        root=root,
                        source_db_path=source_db_path,
                        offset_bps=2.0,
                        horizon_sec=300,
                        cross_bps=0.5,
                        queue_notional_usd=queue_usd,
                        maker_fee_bps=fee,
                        taker_fee_bps=3.05,
                    )
                    if item:
                        rows.append(item)
                summary = summarize_rows(rows)
                counts: dict[str, int] = {}
                for r in rows:
                    counts[str(r["fill_state"])] = counts.get(str(r["fill_state"]), 0) + 1
                configs.append(
                    {
                        "config_id": f"eth_tick_queue_o2_h300_qcross0.5_queue{int(queue_usd)}_fee{fee:g}",
                        "queue_notional_usd": queue_usd,
                        "maker_fee_bps": fee,
                        "fill_counts": counts,
                        "summary": summary,
                    }
                )
    configs.sort(key=lambda c: float(c["summary"]["hold"]["sum_bps"] or 0.0), reverse=True)
    return {
        "status": "EXECUTION_REALISM_ONLY_NOT_ALPHA",
        "event_count": len(events),
        "model": "book cross beyond quote plus agg_trade notional through quote >= queue_notional_usd",
        "ranked_configs": configs,
    }


def explicit_kill_criteria(tail_sizing: dict[str, Any], regime: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": "RISK_POLICY_DRAFT_OPERATOR_SIGNOFF_REQUIRED",
        "criteria": [
            {"name": "KILL_30D_SUM_NEGATIVE", "rule": "30-day forward/live-like closed sum < 0 after >=5 closed fills"},
            {"name": "KILL_60D_SUM_NEGATIVE", "rule": "60-day forward/live-like closed sum < 0 after >=5 closed fills"},
            {"name": "PAUSE_ROLLING_5_NEGATIVE", "rule": "rolling last 5 closed fills sum < 0"},
            {"name": "PAUSE_TAIL_BUDGET_BREACH", "rule": "any realized or shadow loss exceeds pre-accepted tail budget"},
            {"name": "NO_SCALE_UNTIL_VALIDATED", "rule": "no size increase until forward OOS positive across >=2 regimes"},
        ],
        "current_regime_monitor_status": regime.get("status"),
        "recommended_max_margin_from_tail_table": tail_sizing.get("sizing_table", [None])[2],
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    pool = load_json(args.pool_json, {})
    pool_rows = list(pool.get("rows") or [])
    v4 = load_json(args.v4_json, {})
    shadow_rows = load_jsonl(args.shadow_ledger)
    brief = load_json(args.shadow_brief, {})
    tail = tail_aware_sizing(pool_rows, shadow_rows, equity_usdt=float(args.equity_usdt), leverage=float(args.leverage))
    diss = dissipation_observer_spec(v4)
    regime = regime_degradation_monitor(shadow_rows, brief)
    failure = failure_mode_classifier(pool_rows)
    tick = tick_level_execution_realism(args.db, pool_rows)
    kill = explicit_kill_criteria(tail, regime)
    return {
        "generated_at_utc": utc_now(),
        "mode": "RESEARCH_RISK_ONLY_NO_LIVE_ORDER_LOGIC_CHANGE",
        "inputs": {
            "pool_json": str(args.pool_json),
            "v4_json": str(args.v4_json),
            "shadow_ledger": str(args.shadow_ledger),
            "shadow_brief": str(args.shadow_brief),
            "db": str(args.db),
        },
        "tail_aware_sizing": tail,
        "dissipation_defensive_observer": diss,
        "regime_degradation_monitor": regime,
        "failure_mode_classifier": failure,
        "explicit_kill_criteria": kill,
        "tick_level_execution_realism": tick,
    }


def fmt(m: dict[str, Any]) -> str:
    return f"N={m['n']} sum={m['sum_bps']} mean={m['mean_bps']} med={m['median_bps']} T3R={m['t3r_bps']} WR={m['win_rate']} maxL={m['max_loss_bps']}"


def render_md(report: dict[str, Any]) -> str:
    tail = report["tail_aware_sizing"]
    regime = report["regime_degradation_monitor"]
    failure = report["failure_mode_classifier"]
    tick = report["tick_level_execution_realism"]
    lines = [
        "# S34 v6 Management System",
        "",
        f"Generated: `{report['generated_at_utc']}`",
        "",
        "`RESEARCH_RISK_ONLY_NO_LIVE_ORDER_LOGIC_CHANGE` - no entry filter, live order logic, or executor setting was changed.",
        "",
        "## 1. Tail-Aware Sizing",
        "",
        f"- historical min bps: `{tail['historical_min_bps']}`",
        f"- stress tail abs bps: `{tail['stress_tail_bps_abs']}`",
        f"- equity assumption: `${tail['equity_usdt_assumption']}`",
        f"- current 85% margin style stress loss: `{tail['current_env_style_85pct_margin']}`",
        f"- Kelly note: {tail['kelly_note']}",
        "",
        "| Risk % equity | Risk USDT | Max notional | Max margin @ leverage |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for row in tail["sizing_table"]:
        lines.append(
            f"| {row['risk_pct_equity']} | {row['risk_usdt']} | {row['max_notional_usdt']} | {row['max_margin_usdt_at_leverage']} |"
        )
    lines.extend(
        [
            "",
            "## 2. Defensive Dissipation Observer",
            "",
            f"- status: `{report['dissipation_defensive_observer']['status']}`",
            f"- primary reference: `{report['dissipation_defensive_observer']['primary_reference']}`",
            f"- hard rule: {report['dissipation_defensive_observer']['hard_rule']}",
            "",
            "## 3. Regime-Degradation Monitor",
            "",
            f"- status: `{regime['status']}`",
            f"- closed_n: `{regime['closed_n']}`",
            f"- all closed: {fmt(regime['all_closed'])}",
            f"- last5: {fmt(regime['last5'])}",
            f"- last10: {fmt(regime['last10'])}",
            f"- triggers: `{regime['triggers']}`",
            "",
            "## 4. Failure-Mode Classifier",
            "",
            f"- status: `{failure['status']}`",
            f"- large loss N: `{failure['large_loss_n']}`",
            f"- counts: `{failure['counts']}`",
            "",
            "| Label | Symbol | Month | Net bps | Route | Key state |",
            "| --- | --- | --- | ---: | --- | --- |",
        ]
    )
    for s in failure["samples"][:12]:
        state = f"sync={s['sync_gate']}, bid={s['bid_depth_gate']}, abs={s['absorption_gate']}, imb={s['book_imbalance']}, accel={s['running_accel']}"
        lines.append(f"| `{s['label']}` | `{s['symbol']}` | `{s['month']}` | {s['net_bps']} | `{s['route_id']}` | {state} |")
    lines.extend(
        [
            "",
            "## 5. Explicit Kill Criteria",
            "",
        ]
    )
    for c in report["explicit_kill_criteria"]["criteria"]:
        lines.append(f"- `{c['name']}`: {c['rule']}")
    lines.extend(
        [
            "",
            "## 6. Tick-Level Maker Execution Realism",
            "",
            f"- status: `{tick['status']}`",
            f"- model: {tick['model']}",
            f"- event_count: `{tick['event_count']}`",
            "",
            "| Rank | Config | Fill counts | Cal | Hold |",
            "| ---: | --- | --- | --- | --- |",
        ]
    )
    for i, cfg in enumerate(tick["ranked_configs"][:15], 1):
        lines.append(
            f"| {i} | `{cfg['config_id']}` | `{cfg['fill_counts']}` | {fmt(cfg['summary']['cal'])} | {fmt(cfg['summary']['hold'])} |"
        )
    lines.extend(
        [
            "",
            "## Read",
            "",
            "- Management is not an entry filter. Failure modes are descriptive and must not be wired into entry selection.",
            "- The only hard defense against the irreducible tail is sizing and kill/pause criteria.",
            "- Dissipation remains observer-only because v4 did not validate expectancy improvement.",
            "- 600GB data is used here for execution/queue realism, not for new directional alpha mining.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description="Build S34 v6 management system report.")
    p.add_argument("--db", type=Path, default=DEFAULT_DB)
    p.add_argument("--pool-json", type=Path, default=POOL_JSON)
    p.add_argument("--v4-json", type=Path, default=V4_JSON)
    p.add_argument("--shadow-ledger", type=Path, default=SHADOW_LEDGER)
    p.add_argument("--shadow-brief", type=Path, default=SHADOW_BRIEF)
    p.add_argument("--out-json", type=Path, default=OUT_JSON)
    p.add_argument("--out-md", type=Path, default=OUT_MD)
    p.add_argument("--equity-usdt", type=float, default=35.0)
    p.add_argument("--leverage", type=float, default=40.0)
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
