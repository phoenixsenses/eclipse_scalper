from __future__ import annotations

import json
import math
import sqlite3
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


from ami.storage import production as PR
from ami.storage import research_reader as RR

ROOT = Path(__file__).resolve().parents[1]
TRADES_PATH = ROOT / "reports" / "research" / "s34" / "S34_SHADOW_PAPER_TRADES.json"
MICRO_DB = ROOT / "data" / "microstructure.db"
ROUTE_SWEEP_PATH = ROOT / "reports" / "research" / "s34" / "S34_500K_DAYTREND_ROUTE_SWEEP.json"
OUT_PATH = ROOT / "reports" / "research" / "s34" / "visualization_export.json"

EXCLUDED_TRADE_IDS = {"P013", "P056"}

LIVE_ROUTES = {
    "ETH_BUY_LIQ_LONG_50K_TP120_SL40_BE30": {
        "route_id": "50K_TP120_PREREG",
        "label": "50K / TP120 (Pre-Reg)",
        "category": "pre_reg",
        "valid_filter": "pre_reg_valid",
    },
    "ETH_BUY_LIQ_LONG_200K_TP60_SL40_BE30": {
        "route_id": "200K_TP60_EXPLORATORY",
        "label": "200K / TP60 (Exploratory)",
        "category": "exploratory_live",
        "valid_filter": "closed_bookticker",
    },
    "ETH_BUY_LIQ_LONG_500K_DAYTREND0_TP60_SL40_BE30": {
        "route_id": "500K_DAYTREND_TP60_EXPLORATORY",
        "label": "500K / DayTrend TP60 (Exploratory)",
        "category": "exploratory_live",
        "valid_filter": "closed_bookticker",
    },
    "ETH_BUY_LIQ_LONG_200K_BTC_PRE15_TP120_SL40_BE30_DELAY60": {
        "route_id": "200K_BTC_PRE15_TP120_DELAY60_EXPLORATORY",
        "label": "200K / BTC Pre15 / TP120 / Delay60 (Exploratory)",
        "category": "exploratory_live",
        "valid_filter": "closed_bookticker",
    },
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def clean_float(value: Any, digits: int = 6) -> float | None:
    out = as_float(value)
    if out is None:
        return None
    return round(out, digits)


def load_trades() -> list[dict[str, Any]]:
    payload = json.loads(TRADES_PATH.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        return list(payload.get("trades") or [])
    if isinstance(payload, list):
        return payload
    return []


def rule_name(trade: dict[str, Any]) -> str:
    return str((trade.get("rule") or {}).get("name") or "")


def is_bookticker_closed(trade: dict[str, Any]) -> bool:
    if trade.get("status") != "CLOSED":
        return False
    if (trade.get("entry_fill") or {}).get("source") != "BOOK_TICKER":
        return False
    if (trade.get("exit_fill") or {}).get("source") != "BOOK_TICKER":
        return False
    return trade.get("net_bps") is not None


def is_pre_reg_valid(trade: dict[str, Any]) -> bool:
    tid = str(trade.get("trade_id") or trade.get("trial_id") or "")
    if tid in EXCLUDED_TRADE_IDS:
        return False
    if rule_name(trade) != "ETH_BUY_LIQ_LONG_50K_TP120_SL40_BE30":
        return False
    if not is_bookticker_closed(trade):
        return False
    needed = ("gross_bps", "entry_adverse_bps", "exit_adverse_bps", "spread_cost_bps", "fee_cost_bps", "net_bps")
    if any(trade.get(k) is None for k in needed):
        return False
    identity = (
        float(trade["gross_bps"])
        - float(trade["entry_adverse_bps"])
        - float(trade["exit_adverse_bps"])
        - float(trade["spread_cost_bps"])
        - float(trade["fee_cost_bps"])
    )
    return abs(identity - float(trade["net_bps"])) <= 1e-6


def summarize(values: list[float]) -> dict[str, Any]:
    if not values:
        return {
            "n": 0,
            "median_net_bps": None,
            "mean_net_bps": None,
            "win_rate": None,
            "cum_net_bps": 0.0,
        }
    return {
        "n": len(values),
        "median_net_bps": round(statistics.median(values), 6),
        "mean_net_bps": round(statistics.fmean(values), 6),
        "win_rate": round(sum(1 for v in values if v > 0.0) / len(values), 6),
        "cum_net_bps": round(sum(values), 6),
    }


def _mark_prices_in_range(con: sqlite3.Connection, symbol: str, start_ms: int, end_ms: int) -> list[tuple]:
    """Direct-SQL oracle -- kept unchanged as the parity reference for
    `_mark_prices_in_range_v2` (BATCH-STORAGE-ROTATION-RETENTION-RANGE-
    READ-CONSUMER-MIGRATION-V4). No longer called by `mark_path_stats`;
    the reader-backed path is used instead."""
    return con.execute(
        """
        SELECT mark_price
        FROM mark_prices
        WHERE symbol = ?
          AND ts_ms >= ?
          AND ts_ms <= ?
          AND mark_price IS NOT NULL
        """,
        (symbol, start_ms, end_ms),
    ).fetchall()


def _mark_prices_in_range_v2(root, symbol: str, start_ms: int, end_ms: int, source_db_path=None) -> list[tuple]:
    """Reader-backed replacement for `_mark_prices_in_range`, via
    `plan_read`/`execute_read`. `symbol` is a genuine runtime parameter
    (sourced from `trade["symbol"]`, defaulting to ETHUSDT); real call-site
    data spans ETHUSDT/SOLUSDT/BTCUSDT. Inclusive upper bound reproduced
    with `end_ms+1` (exact for integer ts_ms). No ORDER BY is needed here
    (the oracle has none either -- only min/max over the value set matter,
    not row order), so the reader's canonical `(ts_ms ASC, id ASC)`
    ordering is a superset guarantee, not a behavior change. The
    `mark_price IS NOT NULL` filter is reproduced via the reader's
    client-side `filters` (native Python `!=` against `None`, which
    correctly implements SQL's NULL-exclusion semantics)."""
    plan = RR.plan_read(root, table="mark_prices", symbol=symbol, start_ms=int(start_ms), end_ms=int(end_ms) + 1)
    result = RR.execute_read(plan, columns=("mark_price",), filters=(("mark_price", "!=", None),),
                             source_db_path=source_db_path)
    return list(result.iter_rows())


def mark_path_stats(root, symbol: str, entry_ts_ms: int, exit_ts_ms: int, entry_price: float, direction: str, source_db_path=None) -> tuple[float | None, float | None]:
    if exit_ts_ms < entry_ts_ms:
        return None, None
    rows = _mark_prices_in_range_v2(root, symbol, entry_ts_ms, exit_ts_ms, source_db_path=source_db_path)
    prices = [float(r[0]) for r in rows if r[0] is not None]
    if not prices or entry_price <= 0:
        return None, None
    if direction.upper() == "SHORT":
        mfe = (entry_price - min(prices)) / entry_price * 10000.0
        mae = (entry_price - max(prices)) / entry_price * 10000.0
    else:
        mfe = (max(prices) - entry_price) / entry_price * 10000.0
        mae = (min(prices) - entry_price) / entry_price * 10000.0
    return round(mfe, 6), round(mae, 6)


def trade_to_export(trade: dict[str, Any], root, source_db_path: str | None = None) -> dict[str, Any]:
    signal = trade.get("signal") or {}
    entry_ts_ms = int(trade.get("entry_ts_ms") or 0)
    exit_ts_ms = int(trade.get("exit_ts_ms") or 0)
    entry_price = as_float(trade.get("entry_price"))
    mfe_bps = clean_float(trade.get("mfe_bps"))
    mae_bps = clean_float(trade.get("mae_bps"))
    if root is not None and (mfe_bps is None or mae_bps is None) and entry_ts_ms and exit_ts_ms and entry_price:
        mfe_bps, mae_bps = mark_path_stats(
            root,
            str(trade.get("symbol") or "ETHUSDT"),
            entry_ts_ms,
            exit_ts_ms,
            float(entry_price),
            str(trade.get("direction") or "LONG"),
            source_db_path=source_db_path,
        )
    return {
        "trade_id": str(trade.get("trade_id") or trade.get("trial_id") or ""),
        "entry_ts": trade.get("entry_ts_utc") or trade.get("signal_ts_utc"),
        "exit_ts": trade.get("exit_ts_utc"),
        "exit_type": trade.get("exit_reason"),
        "entry_price": clean_float(trade.get("entry_price")),
        "exit_price": clean_float(trade.get("exit_price")),
        "net_bps": clean_float(trade.get("net_bps")),
        "mfe_bps": mfe_bps,
        "mae_bps": mae_bps,
        "cluster_notional": clean_float(signal.get("liq_total_notional"), 2),
        "cluster_liq_count": signal.get("liq_count"),
        "tp_price": clean_float(trade.get("tp_price")),
        "sl_price": clean_float(trade.get("sl_price")),
        "be_price": clean_float(trade.get("be_trigger_price")),
    }


def build_live_routes(trades: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    by_rule: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for trade in trades:
        name = rule_name(trade)
        if name in LIVE_ROUTES and trade.get("status") == "CLOSED":
            by_rule[name].append(trade)

    # Every mark_prices range read now goes through the reader (via
    # source_db_path), so the old direct sqlite3 connection to MICRO_DB is
    # fully dead here and is not opened (mirrors the ASOF/range-read
    # dead-connection-drop precedent -- `mark_path_stats` was the only
    # direct-SQL caller in this file).
    root = None
    source_db_path: str | None = None
    if MICRO_DB.exists():
        root, _ = PR.resolve_production_root()
        source_db_path = str(MICRO_DB)

    routes: list[dict[str, Any]] = []
    total_exported = 0
    for name, meta in LIVE_ROUTES.items():
        rows = sorted(by_rule.get(name, []), key=lambda t: (int(t.get("entry_ts_ms") or 0), str(t.get("trade_id") or "")))
        if meta["valid_filter"] == "pre_reg_valid":
            valid_rows = [t for t in rows if is_pre_reg_valid(t)]
        else:
            valid_rows = [t for t in rows if is_bookticker_closed(t)]
        nets = [float(t["net_bps"]) for t in valid_rows if t.get("net_bps") is not None]
        stats = summarize(nets)
        exported_trades = [trade_to_export(t, root, source_db_path=source_db_path) for t in valid_rows]
        total_exported += len(exported_trades)
        routes.append(
            {
                "route_id": meta["route_id"],
                "rule_name": name,
                "label": meta["label"],
                "category": meta["category"],
                "n_closed": len(rows),
                "n_valid": len(valid_rows),
                "median_net_bps": stats["median_net_bps"],
                "mean_net_bps": stats["mean_net_bps"],
                "win_rate": stats["win_rate"],
                "cum_net_bps": stats["cum_net_bps"],
                "trades": exported_trades,
            }
        )
    return routes, total_exported


def route_stat_from_summary(route_id: str, label: str, category: str, status: str, summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "route_id": route_id,
        "label": label,
        "category": category,
        "status": status,
        "n": summary.get("n"),
        "median_net_bps": clean_float(summary.get("median")),
        "mean_net_bps": clean_float(summary.get("mean")),
        "cum_net_bps": clean_float(summary.get("cum")),
        "win_rate": clean_float(summary.get("wr")),
        "top3_removed_cum_bps": clean_float(summary.get("top3_removed_cum")),
        "positive_days": summary.get("positive_days"),
        "days": summary.get("days"),
        "exit_counts": summary.get("exit_counts") or {},
    }


def build_research_routes() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not ROUTE_SWEEP_PATH.exists():
        return out
    payload = json.loads(ROUTE_SWEEP_PATH.read_text(encoding="utf-8"))
    real_fill = payload.get("real_fill") or {}
    live_current = "TP60_SL40_BE30"
    ranked: list[tuple[float, str, dict[str, Any]]] = []
    for key, row in real_fill.items():
        if key == live_current:
            continue
        test = row.get("test") or {}
        ranked.append((float(test.get("cum") or 0.0), key, row))
    ranked.sort(reverse=True)
    for _, key, row in ranked[:5]:
        test = row.get("test") or {}
        out.append(
            route_stat_from_summary(
                "500K_DAYTREND_" + key,
                f"500K/DayTrend {key.replace('_', '/')} (research candidate)",
                "research_only",
                "not_live",
                test,
            )
            | {
                "source": "S34_500K_DAYTREND_ROUTE_SWEEP.real_fill.test",
                "total_rows": row.get("total_rows"),
                "real_fill_rows": row.get("real_fill_rows"),
                "no_fill_rows": row.get("no_fill_rows"),
                "no_fill_rate": clean_float(row.get("no_fill_rate")),
            }
        )
    return out


def main() -> int:
    trades = load_trades()
    routes, total_trade_count = build_live_routes(trades)
    research_routes = build_research_routes()
    payload = {
        "generated_at": utc_now_iso(),
        "source_files": {
            "trades": str(TRADES_PATH.relative_to(ROOT)),
            "route_sweep": str(ROUTE_SWEEP_PATH.relative_to(ROOT)) if ROUTE_SWEEP_PATH.exists() else None,
            "microstructure_db_for_mfe_mae": str(MICRO_DB.relative_to(ROOT)) if MICRO_DB.exists() else None,
        },
        "notes": [
            "Export is read-only. Live runner/config and databases were not modified.",
            "Live trade MFE/MAE are computed from mark_prices when absent from the journal.",
            "Research routes are aggregate-only and not live.",
        ],
        "routes": routes,
        "research_routes": research_routes,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    size = OUT_PATH.stat().st_size
    print(json.dumps({"output": str(OUT_PATH), "bytes": size, "live_trade_count": total_trade_count, "research_routes": len(research_routes)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
