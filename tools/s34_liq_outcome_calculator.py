from __future__ import annotations

import argparse
import json
import math
import re
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
FEATURE_DB = ROOT / "data" / "s34_feature_factory.db"
MICRO_DB_URI = f"file:{(ROOT / 'data' / 'microstructure.db').as_posix()}?mode=ro"
INTELLIGENCE_DB = ROOT / "data" / "s34_intelligence.db"
OUT_DIR = ROOT / "reports" / "research" / "s34"

DEFAULT_HORIZONS_SEC = (60, 300, 900, 3600)
ROUTE_IDS = ("LONG_DELAY0_TP60", "LONG_DELAY60_TP120", "SHORT_DELAY0_TP40_CONTROL")


KNN_FEATURES = [
    ("log_cluster_notional", "cluster_notional", "target_notional", 2.0),
    ("cluster_duration_sec", "cluster_duration_sec", "target_duration_sec", 0.8),
    ("cluster_liq_count", "cluster_liq_count", "target_liq_count", 0.8),
    ("max_single_liq_share", "max_single_liq_share", "target_max_single_share", 0.8),
    ("intensity_per_sec", "intensity_per_sec", "target_intensity_per_sec", 1.0),
    ("inter_cluster_gap_sec", "inter_cluster_gap_sec", "target_inter_cluster_gap_sec", 0.7),
    ("day_trend_bps", "day_trend_bps", "target_day_trend_bps", 1.4),
    ("day_range_bps", "day_range_bps", "target_day_range_bps", 1.0),
    ("symbol_pre_5m_bps", "symbol_pre_5m_bps", "target_symbol_pre_5m_bps", 1.0),
    ("symbol_pre_15m_bps", "symbol_pre_15m_bps", "target_symbol_pre_15m_bps", 1.0),
    ("btc_pre_15m_bps", "btc_pre_15m_bps", "target_btc_pre_15m_bps", 0.8),
]

# Per-symbol/side notional thresholds used when --min-notional is not supplied explicitly.
SYMBOL_SIDE_NOTIONAL_DEFAULTS: dict[tuple[str, str], float] = {
    ("ETHUSDT", "BUY"):  500_000.0,
    ("ETHUSDT", "SELL"): 500_000.0,
    ("BTCUSDT", "BUY"):  1_000_000.0,
    ("BTCUSDT", "SELL"): 1_000_000.0,
    ("SOLUSDT", "BUY"):  100_000.0,
    ("SOLUSDT", "SELL"): 100_000.0,
}

# Per-combo KNN model configs (ETH SELL validated v2).
COMBO_MODEL_CONFIGS: dict[tuple[str, str], dict] = {
    ("ETHUSDT", "SELL"): {
        "k": 10,
        "metric": "manhattan",
        "exclude_features": frozenset({
            "cluster_liq_count",
            "btc_pre_15m_bps",
            "cluster_duration_sec",
            "day_trend_bps",
            "symbol_pre_15m_bps",
        }),
    },
}

MODEL_USEFULNESS: dict[tuple[str, str], str] = {
    ("ETHUSDT", "SELL"): "KNN_USEFUL",
    ("BTCUSDT", "SELL"): "BASE_RATE_ONLY",
    ("ETHUSDT", "BUY"): "REGIME_SHIFT_RECENCY_HELPFUL_PRELIMINARY",
    ("SOLUSDT", "BUY"): "BASE_RATE_ONLY",
    ("SOLUSDT", "SELL"): "BASE_RATE_ONLY",
    ("BTCUSDT", "BUY"): "BASE_RATE_ONLY",
}


PRESETS: dict[str, dict[str, Any]] = {
    "eth_buy_500k": {
        "symbol": "ETHUSDT",
        "side": "BUY",
        "min_notional": 500_000.0,
        "mode": "filter",
        "importance_route": "LONG_DELAY0_TP60",
    },
    "eth_sell_500k": {
        "symbol": "ETHUSDT",
        "side": "SELL",
        "min_notional": 500_000.0,
        "mode": "filter",
        "importance_route": "SHORT_DELAY0_TP60",
    },
    "eth_sell_1m": {
        "symbol": "ETHUSDT",
        "side": "SELL",
        "min_notional": 1_000_000.0,
        "mode": "filter",
        "importance_route": "SHORT_DELAY0_TP80",
    },
    "sol_buy_100k": {
        "symbol": "SOLUSDT",
        "side": "BUY",
        "min_notional": 100_000.0,
        "mode": "filter",
        "importance_route": "LONG_DELAY0_TP60",
    },
    "sol_buy_200k": {
        "symbol": "SOLUSDT",
        "side": "BUY",
        "min_notional": 200_000.0,
        "mode": "filter",
        "importance_route": "LONG_DELAY0_TP60",
    },
    "sol_sell_100k": {
        "symbol": "SOLUSDT",
        "side": "SELL",
        "min_notional": 100_000.0,
        "mode": "filter",
        "importance_route": "SHORT_DELAY0_TP40",
    },
    "sol_sell_200k": {
        "symbol": "SOLUSDT",
        "side": "SELL",
        "min_notional": 200_000.0,
        "mode": "filter",
        "importance_route": "SHORT_DELAY0_TP60",
    },
    "btc_buy_1m": {
        "symbol": "BTCUSDT",
        "side": "BUY",
        "min_notional": 1_000_000.0,
        "mode": "filter",
        "importance_route": "LONG_DELAY0_TP60",
    },
    "btc_sell_1m": {
        "symbol": "BTCUSDT",
        "side": "SELL",
        "min_notional": 1_000_000.0,
        "mode": "filter",
        "importance_route": "SHORT_DELAY0_TP40",
    },
}

DEPRECATED_SIGNAL_RULES = {
    "ETH_BUY_LIQ_LONG_50K_TP120_SL40_BE30",
}


def iso_ms(ts_ms: int | None) -> str | None:
    if ts_ms is None:
        return None
    return datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc).isoformat()


def pctile(values: list[float], q: float) -> float | None:
    vals = sorted(v for v in values if v is not None and math.isfinite(v))
    if not vals:
        return None
    if len(vals) == 1:
        return vals[0]
    pos = (len(vals) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return vals[lo]
    return vals[lo] + (vals[hi] - vals[lo]) * (pos - lo)


def mean(values: list[float]) -> float | None:
    vals = [v for v in values if v is not None and math.isfinite(v)]
    return sum(vals) / len(vals) if vals else None


def stdev(values: list[float]) -> float | None:
    vals = [v for v in values if v is not None and math.isfinite(v)]
    if len(vals) < 2:
        return None
    avg = sum(vals) / len(vals)
    return math.sqrt(sum((v - avg) ** 2 for v in vals) / (len(vals) - 1))


def _parser_defaults() -> argparse.Namespace:
    parser = build_parser()
    return parser.parse_args([])


def _apply_if_default(args: argparse.Namespace, defaults: argparse.Namespace, name: str, value: Any) -> None:
    if value is None:
        return
    explicit = getattr(args, "_explicit_options", set())
    if f"--{name.replace('_', '-')}" in explicit:
        return
    if getattr(args, name) == getattr(defaults, name):
        setattr(args, name, value)


def apply_preset(args: argparse.Namespace, defaults: argparse.Namespace) -> None:
    if not args.preset:
        return
    preset = PRESETS.get(str(args.preset).lower())
    if not preset:
        valid = ", ".join(sorted(PRESETS))
        raise SystemExit(f"Unknown preset {args.preset!r}. Valid presets: {valid}")
    for key, value in preset.items():
        _apply_if_default(args, defaults, key, value)
    args.preset_applied = str(args.preset).lower()


def _threshold_from_rule_name(rule_name: str | None) -> float | None:
    if not rule_name:
        return None
    match_m = re.search(r"_(\d+)M(?:_|$)", rule_name)
    if match_m:
        return float(match_m.group(1)) * 1_000_000.0
    match_k = re.search(r"_(\d+)K(?:_|$)", rule_name)
    if match_k:
        return float(match_k.group(1)) * 1_000.0
    return None


def _latest_signal(signal_id: str | None = None) -> dict[str, Any] | None:
    if not INTELLIGENCE_DB.exists():
        return None
    con = sqlite3.connect(INTELLIGENCE_DB)
    con.row_factory = sqlite3.Row
    if signal_id:
        row = con.execute("SELECT * FROM s34_signals WHERE signal_id=?", (signal_id,)).fetchone()
    else:
        placeholders = ",".join("?" for _ in DEPRECATED_SIGNAL_RULES)
        row = con.execute(
            f"""
            SELECT * FROM s34_signals
            WHERE rule_name NOT IN ({placeholders})
            ORDER BY signal_ts_ms DESC LIMIT 1
            """,
            list(DEPRECATED_SIGNAL_RULES),
        ).fetchone()
    con.close()
    if not row:
        return None
    data = dict(row)
    try:
        data["features"] = json.loads(data.get("features_json") or "{}")
    except json.JSONDecodeError:
        data["features"] = {}
    return data


def _trade_source(open_only: bool = False, trade_id: str | None = None) -> tuple[dict[str, Any], dict[str, Any]] | None:
    if not INTELLIGENCE_DB.exists():
        return None
    con = sqlite3.connect(INTELLIGENCE_DB)
    con.row_factory = sqlite3.Row
    if trade_id:
        trade = con.execute("SELECT * FROM s34_trades WHERE trade_id=?", (trade_id,)).fetchone()
    else:
        where = "WHERE status='OPEN'" if open_only else ""
        trade = con.execute(
            f"""
            SELECT * FROM s34_trades
            {where}
            ORDER BY COALESCE(entry_ts_ms, 0) DESC
            LIMIT 1
            """
        ).fetchone()
    if not trade:
        con.close()
        return None
    signal = con.execute("SELECT * FROM s34_signals WHERE signal_id=?", (trade["signal_id"],)).fetchone()
    con.close()
    if not signal:
        return None
    signal_data = dict(signal)
    try:
        signal_data["features"] = json.loads(signal_data.get("features_json") or "{}")
    except json.JSONDecodeError:
        signal_data["features"] = {}
    return signal_data, dict(trade)


def apply_signal_targets(args: argparse.Namespace, defaults: argparse.Namespace) -> None:
    if (
        not args.from_latest_signal
        and not args.signal_id
        and not args.from_open_trade
        and not args.from_latest_accepted_signal
        and not args.from_trade_id
    ):
        return
    source_trade = None
    if args.from_trade_id:
        loaded = _trade_source(trade_id=args.from_trade_id)
        if not loaded:
            raise SystemExit(f"Trade {args.from_trade_id!r} not found in {INTELLIGENCE_DB}")
        signal, source_trade = loaded
    elif args.from_open_trade:
        loaded = _trade_source(open_only=True)
        if not loaded:
            raise SystemExit(f"No OPEN trade found in {INTELLIGENCE_DB}")
        signal, source_trade = loaded
    elif args.from_latest_accepted_signal:
        loaded = _trade_source(open_only=False)
        if not loaded:
            raise SystemExit(f"No accepted trade found in {INTELLIGENCE_DB}")
        signal, source_trade = loaded
    else:
        signal = _latest_signal(args.signal_id)
    if not signal:
        target = f"signal_id={args.signal_id}" if args.signal_id else "latest signal"
        raise SystemExit(f"Could not load {target} from {INTELLIGENCE_DB}")
    features = signal.get("features") or {}
    _apply_if_default(args, defaults, "symbol", signal.get("symbol"))
    _apply_if_default(args, defaults, "side", signal.get("liq_side"))
    _apply_if_default(args, defaults, "mode", "knn")
    _apply_if_default(args, defaults, "min_notional", _threshold_from_rule_name(signal.get("rule_name")))
    _apply_if_default(args, defaults, "target_notional", signal.get("cluster_notional") or features.get("liq_total_notional"))
    _apply_if_default(args, defaults, "target_duration_sec", features.get("cluster_duration_sec"))
    _apply_if_default(args, defaults, "target_liq_count", signal.get("cluster_liq_count") or features.get("cluster_liq_count") or features.get("liq_count"))
    _apply_if_default(args, defaults, "target_max_single_share", features.get("max_single_liq_share") or features.get("cluster_max_single_liq_share"))
    _apply_if_default(args, defaults, "target_intensity_per_sec", features.get("intensity_per_sec"))
    _apply_if_default(args, defaults, "target_inter_cluster_gap_sec", features.get("inter_cluster_gap_sec") or features.get("prev_liq_gap_sec"))
    _apply_if_default(args, defaults, "target_day_trend_bps", features.get("day_trend_bps"))
    _apply_if_default(args, defaults, "target_day_range_bps", features.get("day_range_bps"))
    _apply_if_default(args, defaults, "target_symbol_pre_15m_bps", features.get("symbol_pre_15m_bps"))
    _apply_if_default(args, defaults, "target_btc_pre_15m_bps", features.get("btc_pre_return_bps") or features.get("btc_pre_15m_bps"))
    args.source_signal = {
        "signal_id": signal.get("signal_id"),
        "signal_ts_utc": signal.get("signal_ts_utc"),
        "rule_name": signal.get("rule_name"),
        "symbol": signal.get("symbol"),
        "liq_side": signal.get("liq_side"),
        "direction": signal.get("direction"),
        "cluster_notional": signal.get("cluster_notional"),
        "cluster_liq_count": signal.get("cluster_liq_count"),
        "cluster_shape_label": signal.get("cluster_shape_label"),
    }
    if source_trade:
        args.source_trade = {
            "trade_id": source_trade.get("trade_id"),
            "status": source_trade.get("status"),
            "rule_name": source_trade.get("rule_name"),
            "entry_ts_ms": source_trade.get("entry_ts_ms"),
            "entry_price": source_trade.get("entry_price"),
            "exit_ts_ms": source_trade.get("exit_ts_ms"),
            "exit_reason": source_trade.get("exit_reason"),
            "net_bps": source_trade.get("net_bps"),
        }


def summarize(values: list[float]) -> dict[str, Any]:
    vals = [v for v in values if v is not None and math.isfinite(v)]
    return {
        "n": len(vals),
        "mean": mean(vals),
        "median": pctile(vals, 0.50),
        "p25": pctile(vals, 0.25),
        "p75": pctile(vals, 0.75),
        "p10": pctile(vals, 0.10),
        "p90": pctile(vals, 0.90),
        "stdev": stdev(vals),
        "positive_rate": (sum(1 for v in vals if v > 0) / len(vals)) if vals else None,
    }


def mark_at(con: sqlite3.Connection, symbol: str, ts_ms: int, before: bool = False) -> tuple[int, float] | None:
    op = "<=" if before else ">="
    order = "DESC" if before else "ASC"
    row = con.execute(
        f"""
        SELECT ts_ms, mark_price
        FROM mark_prices
        WHERE symbol=? AND ts_ms {op} ?
        ORDER BY ts_ms {order}
        LIMIT 1
        """,
        (symbol, int(ts_ms)),
    ).fetchone()
    if not row:
        return None
    return int(row[0]), float(row[1])


def return_bps(source_con: sqlite3.Connection, symbol: str, start_ms: int, horizon_sec: int) -> float | None:
    start = mark_at(source_con, symbol, start_ms, before=False)
    end = mark_at(source_con, symbol, start_ms + horizon_sec * 1000, before=False)
    if not start or not end or not start[1]:
        return None
    return (end[1] - start[1]) / start[1] * 10000.0


def table_columns(con: sqlite3.Connection, table: str) -> set[str]:
    return {str(row[1]) for row in con.execute(f"PRAGMA table_info({table})").fetchall()}


def build_where(args: argparse.Namespace, columns: set[str]) -> tuple[str, list[Any], list[str]]:
    clauses = ["symbol=?", "liq_side=?"]
    params: list[Any] = [args.symbol, args.side]
    labels = [f"symbol={args.symbol}", f"side={args.side}"]

    def add(col: str, op: str, value: Any, label: str) -> None:
        if value is None:
            return
        if col not in columns:
            labels.append(f"{label}: unavailable")
            return
        clauses.append(f"{col} {op} ?")
        params.append(value)
        labels.append(label)

    add("cluster_notional", ">=", args.min_notional, f"cluster_notional>={args.min_notional:g}" if args.min_notional is not None else "")
    add("cluster_notional", "<", args.max_notional, f"cluster_notional<{args.max_notional:g}" if args.max_notional is not None else "")
    add("day_trend_bps", ">=", args.min_day_trend_bps, f"day_trend_bps>={args.min_day_trend_bps:g}" if args.min_day_trend_bps is not None else "")
    add("day_trend_bps", "<=", args.max_day_trend_bps, f"day_trend_bps<={args.max_day_trend_bps:g}" if args.max_day_trend_bps is not None else "")
    add("day_range_bps", ">=", args.min_day_range_bps, f"day_range_bps>={args.min_day_range_bps:g}" if args.min_day_range_bps is not None else "")
    add("symbol_pre_5m_bps", ">=", args.min_symbol_pre_5m_bps, f"symbol_pre_5m_bps>={args.min_symbol_pre_5m_bps:g}" if args.min_symbol_pre_5m_bps is not None else "")
    add("symbol_pre_15m_bps", ">=", args.min_symbol_pre_15m_bps, f"symbol_pre_15m_bps>={args.min_symbol_pre_15m_bps:g}" if args.min_symbol_pre_15m_bps is not None else "")
    add("btc_pre_15m_bps", ">=", args.min_btc_pre_15m_bps, f"btc_pre_15m_bps>={args.min_btc_pre_15m_bps:g}" if args.min_btc_pre_15m_bps is not None else "")
    add("cluster_duration_sec", "<=", args.max_duration_sec, f"cluster_duration_sec<={args.max_duration_sec:g}" if args.max_duration_sec is not None else "")
    add("inter_cluster_gap_sec", ">=", args.min_inter_cluster_gap_sec, f"inter_cluster_gap_sec>={args.min_inter_cluster_gap_sec:g}" if args.min_inter_cluster_gap_sec is not None else "")
    add("max_single_liq_share", ">=", args.min_max_single_share, f"max_single_liq_share>={args.min_max_single_share:g}" if args.min_max_single_share is not None else "")

    if args.shape_label:
        if "shape_label" in columns:
            clauses.append("shape_label=?")
            params.append(args.shape_label)
            labels.append(f"shape_label={args.shape_label}")
        else:
            labels.append("shape_label: unavailable")

    return " AND ".join(clauses), params, labels


def fetch_events(feature_con: sqlite3.Connection, where_sql: str, params: list[Any], limit: int | None) -> list[dict[str, Any]]:
    sql = f"""
    SELECT *
    FROM liq_event_features
    WHERE {where_sql}
    ORDER BY event_ts_ms ASC
    """
    if limit:
        sql += " LIMIT ?"
        params = [*params, int(limit)]
    feature_con.row_factory = sqlite3.Row
    return [dict(row) for row in feature_con.execute(sql, params).fetchall()]


def feature_value(event: dict[str, Any], column: str) -> float | None:
    raw = event.get(column)
    if raw is None:
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    return value


def transformed_value(event: dict[str, Any], feature_name: str, column: str) -> float | None:
    value = feature_value(event, column)
    if value is None:
        return None
    if feature_name == "log_cluster_notional":
        return math.log(max(value, 1.0))
    return value


def base_feature_weights() -> dict[str, float]:
    return {feature_name: float(weight) for feature_name, _column, _arg_name, weight in KNN_FEATURES}


def target_features(
    args: argparse.Namespace,
    weight_overrides: dict[str, float] | None = None,
    exclude_features: frozenset[str] | None = None,
) -> dict[str, dict[str, Any]]:
    _exclude = exclude_features or frozenset()
    out: dict[str, dict[str, Any]] = {}
    for feature_name, column, arg_name, weight in KNN_FEATURES:
        if feature_name in _exclude:
            continue
        raw = getattr(args, arg_name, None)
        if feature_name == "log_cluster_notional" and raw is None:
            raw = args.min_notional
        if raw is None:
            continue
        value = float(raw)
        transformed = math.log(max(value, 1.0)) if feature_name == "log_cluster_notional" else value
        out[feature_name] = {
            "column": column,
            "raw_value": value,
            "value": transformed,
            "weight": float((weight_overrides or {}).get(feature_name, weight)),
        }
    return out


def robust_scale(values: list[float]) -> float:
    vals = [v for v in values if v is not None and math.isfinite(v)]
    if len(vals) < 3:
        return 1.0
    p75 = pctile(vals, 0.75)
    p25 = pctile(vals, 0.25)
    if p75 is None or p25 is None:
        return 1.0
    iqr = abs(p75 - p25)
    if iqr > 1e-9:
        return iqr
    sd = stdev(vals)
    return sd if sd and sd > 1e-9 else 1.0


def knn_select(
    events: list[dict[str, Any]],
    args: argparse.Namespace,
    drop_feature: str | None = None,
    weight_overrides: dict[str, float] | None = None,
    metric: str = "euclidean",
    exclude_features: frozenset[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    targets = target_features(args, weight_overrides, exclude_features=exclude_features)
    if drop_feature:
        targets.pop(drop_feature, None)
    if not targets:
        return events[: args.k], {
            "enabled": True,
            "error": "no target features supplied; returned first K events from filtered universe",
            "drop_feature": drop_feature,
            "target_features": {},
            "feature_scales": {},
            "candidate_n": len(events),
            "k": args.k,
            "metric": metric,
        }

    scales: dict[str, float] = {}
    for feature_name, spec in targets.items():
        values = [
            transformed_value(event, feature_name, str(spec["column"]))
            for event in events
        ]
        scales[feature_name] = robust_scale([v for v in values if v is not None])

    scored: list[tuple[float, int, dict[str, Any], dict[str, float]]] = []
    for event in events:
        total_weight = 0.0
        total = 0.0
        components: dict[str, float] = {}
        for feature_name, spec in targets.items():
            value = transformed_value(event, feature_name, str(spec["column"]))
            if value is None:
                continue
            scale = scales.get(feature_name) or 1.0
            weight = float(spec["weight"])
            z = (value - float(spec["value"])) / scale
            if metric == "manhattan":
                dist_part = weight * abs(z)
                components[feature_name] = dist_part
            else:  # euclidean (default)
                dist_part = weight * z * z
                components[feature_name] = math.sqrt(dist_part)
            total += dist_part
            total_weight += weight
        if total_weight <= 0:
            continue
        distance = total / total_weight if metric == "manhattan" else math.sqrt(total / total_weight)
        scored.append((distance, int(event.get("event_ts_ms") or 0), event, components))

    scored.sort(key=lambda row: (row[0], row[1]))
    selected = []
    for distance, _ts, event, components in scored[: args.k]:
        row = dict(event)
        row["_knn_distance"] = distance
        row["_knn_components"] = components
        selected.append(row)

    return selected, {
        "enabled": True,
        "drop_feature": drop_feature,
        "candidate_n": len(events),
        "selected_n": len(selected),
        "k": args.k,
        "metric": metric,
        "excluded_features": sorted(exclude_features or frozenset()),
        "target_features": {
            name: {
                "column": spec["column"],
                "raw_value": spec["raw_value"],
                "weight": spec["weight"],
            }
            for name, spec in targets.items()
        },
        "feature_scales": scales,
        "distance": summarize([float(row["_knn_distance"]) for row in selected]),
    }


def outcome_rows(feature_con: sqlite3.Connection, event_ids: list[str]) -> list[dict[str, Any]]:
    if not event_ids:
        return []
    placeholders = ",".join("?" for _ in event_ids)
    feature_con.row_factory = sqlite3.Row
    rows = feature_con.execute(
        f"""
        SELECT *
        FROM liq_event_outcome_labels
        WHERE event_id IN ({placeholders})
        ORDER BY route_id, entry_delay_sec, tp_bps
        """,
        event_ids,
    ).fetchall()
    return [dict(row) for row in rows]


def route_summary(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for route_id in sorted({str(row["route_id"]) for row in rows}):
        route_rows = [row for row in rows if row["route_id"] == route_id]
        nets = [float(row["net_bps"]) for row in route_rows]
        top3_removed = sorted(nets, reverse=True)[3:] if len(nets) > 3 else []
        out[route_id] = {
            "n": len(route_rows),
            "net_bps": summarize(nets),
            "cum_net_bps": sum(nets),
            "top3_removed_cum_net_bps": sum(top3_removed) if top3_removed else None,
            "mfe_bps": summarize([float(row["mfe_bps"]) for row in route_rows]),
            "mae_bps": summarize([float(row["mae_bps"]) for row in route_rows]),
            "tp_touch_rate": sum(int(row["tp_touch"] or 0) for row in route_rows) / len(route_rows) if route_rows else None,
            "sl_touch_rate": sum(int(row["sl_touch"] or 0) for row in route_rows) / len(route_rows) if route_rows else None,
            "be_hit_rate": sum(int(row["be_hit"] or 0) for row in route_rows) / len(route_rows) if route_rows else None,
            "exit_mix": {
                key: sum(1 for row in route_rows if row["exit_reason"] == key)
                for key in ("TP", "BE", "SL", "TIME")
            },
        }
    return out


def primary_route_stats(summary: dict[str, dict[str, Any]], route_id: str) -> dict[str, Any]:
    row = summary.get(route_id) or {}
    net = row.get("net_bps") or {}
    return {
        "n": row.get("n"),
        "median_net_bps": net.get("median"),
        "mean_net_bps": net.get("mean"),
        "cum_net_bps": row.get("cum_net_bps"),
        "wr": net.get("positive_rate"),
    }


def run_knn_importance(
    feature_con: sqlite3.Connection,
    candidates: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    baseline_summary: dict[str, dict[str, Any]],
    args: argparse.Namespace,
    metric: str = "euclidean",
    exclude_features: frozenset[str] | None = None,
) -> list[dict[str, Any]]:
    if args.mode != "knn":
        return []
    baseline_ids = {str(event["event_id"]) for event in selected}
    baseline_stats = primary_route_stats(baseline_summary, args.importance_route)
    baseline_median = baseline_stats.get("median_net_bps")
    baseline_cum = baseline_stats.get("cum_net_bps")
    rows: list[dict[str, Any]] = []
    for feature_name in sorted(target_features(args, exclude_features=exclude_features)):
        dropped_events, meta = knn_select(candidates, args, drop_feature=feature_name,
                                          metric=metric, exclude_features=exclude_features)
        dropped_ids = {str(event["event_id"]) for event in dropped_events}
        dropped_outcomes = outcome_rows(feature_con, list(dropped_ids))
        dropped_summary = route_summary(dropped_outcomes)
        stats = primary_route_stats(dropped_summary, args.importance_route)
        overlap = len(baseline_ids & dropped_ids)
        median = stats.get("median_net_bps")
        cum = stats.get("cum_net_bps")
        rows.append(
            {
                "feature": feature_name,
                "route_id": args.importance_route,
                "baseline_median_net_bps": baseline_median,
                "drop_median_net_bps": median,
                "median_delta_bps": None if baseline_median is None or median is None else float(baseline_median) - float(median),
                "baseline_cum_net_bps": baseline_cum,
                "drop_cum_net_bps": cum,
                "cum_delta_bps": None if baseline_cum is None or cum is None else float(baseline_cum) - float(cum),
                "neighbor_overlap_n": overlap,
                "neighbor_overlap_pct": overlap / len(baseline_ids) if baseline_ids else None,
                "drop_selected_n": len(dropped_events),
                "drop_knn_distance_median": ((meta.get("distance") or {}).get("median") if isinstance(meta, dict) else None),
            }
        )
    rows.sort(
        key=lambda row: (
            abs(float(row["median_delta_bps"] or 0.0)),
            1.0 - float(row["neighbor_overlap_pct"] or 0.0),
            abs(float(row["cum_delta_bps"] or 0.0)),
        ),
        reverse=True,
    )
    return rows


def suggested_weights_from_importance(
    importance: list[dict[str, Any]],
    args: argparse.Namespace,
    exclude_features: frozenset[str] | None = None,
) -> dict[str, Any]:
    targets = target_features(args, exclude_features=exclude_features)
    if not importance or not targets:
        return {"available": False, "weights": {}, "reason": "importance unavailable"}

    impact_rows = []
    for row in importance:
        feature = str(row.get("feature") or "")
        if feature not in targets:
            continue
        median_delta = abs(float(row.get("median_delta_bps") or 0.0))
        cum_delta = abs(float(row.get("cum_delta_bps") or 0.0)) / max(1.0, float(args.k or 1))
        overlap = row.get("neighbor_overlap_pct")
        neighbor_shift = 1.0 - float(overlap) if overlap is not None else 0.0
        impact = median_delta + 0.25 * cum_delta + 15.0 * neighbor_shift
        impact_rows.append({"feature": feature, "impact": impact})

    if not impact_rows:
        return {"available": False, "weights": {}, "reason": "no target feature impact rows"}

    impacts = [row["impact"] for row in impact_rows]
    lo = min(impacts)
    hi = max(impacts)
    base = base_feature_weights()
    weights: dict[str, float] = {}
    details: list[dict[str, Any]] = []
    for row in impact_rows:
        feature = row["feature"]
        if hi > lo:
            score = (row["impact"] - lo) / (hi - lo)
        else:
            score = 0.5
        multiplier = 0.65 + 1.10 * score
        weight = max(0.25, min(3.0, base.get(feature, 1.0) * multiplier))
        weights[feature] = round(weight, 4)
        details.append(
            {
                "feature": feature,
                "base_weight": base.get(feature, 1.0),
                "impact": row["impact"],
                "score": score,
                "suggested_weight": weights[feature],
            }
        )

    return {
        "available": True,
        "method": "leave_one_feature_out impact normalized to 0.65x-1.75x base weight",
        "weights": weights,
        "details": sorted(details, key=lambda item: item["suggested_weight"], reverse=True),
    }


def split_by_time(events: list[dict[str, Any]], train_frac: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ordered = sorted(events, key=lambda event: int(event.get("event_ts_ms") or 0))
    if len(ordered) < 2:
        return ordered, []
    cut = int(len(ordered) * train_frac)
    cut = max(1, min(len(ordered) - 1, cut))
    return ordered[:cut], ordered[cut:]


def compact_eval(
    feature_con: sqlite3.Connection,
    source_con: sqlite3.Connection,
    events: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    event_ids = [str(event["event_id"]) for event in events]
    outcomes = outcome_rows(feature_con, event_ids)
    routes = route_summary(outcomes)
    return {
        "n": len(events),
        "confidence": confidence_label(len(events),
        ),
        "forward_returns_bps": horizon_summary(source_con, events, tuple(int(x) for x in args.horizons_sec)),
        "route_simulation": routes,
        "primary_route": primary_route_stats(routes, args.importance_route),
    }


def run_oos_validation(
    feature_con: sqlite3.Connection,
    source_con: sqlite3.Connection,
    candidates: list[dict[str, Any]],
    args: argparse.Namespace,
    metric: str = "euclidean",
    exclude_features: frozenset[str] | None = None,
) -> dict[str, Any]:
    if args.mode != "knn":
        return {"enabled": False, "reason": "OOS validation requires --mode knn"}
    train, test = split_by_time(candidates, args.oos_train_frac)
    if len(train) < args.k or len(test) < args.k:
        return {
            "enabled": True,
            "status": "INSUFFICIENT_SAMPLE",
            "train_n": len(train),
            "test_n": len(test),
            "k": args.k,
        }

    train_default, _train_meta = knn_select(train, args, metric=metric, exclude_features=exclude_features)
    train_outcomes = outcome_rows(feature_con, [str(event["event_id"]) for event in train_default])
    train_summary = route_summary(train_outcomes)
    train_importance = run_knn_importance(feature_con, train, train_default, train_summary, args,
                                          metric=metric, exclude_features=exclude_features)
    suggested = suggested_weights_from_importance(train_importance, args, exclude_features=exclude_features)
    weights = suggested.get("weights") if suggested.get("available") else {}

    test_default, test_default_meta = knn_select(test, args, metric=metric, exclude_features=exclude_features)
    test_auto, test_auto_meta = knn_select(test, args, weight_overrides=weights or {},
                                            metric=metric, exclude_features=exclude_features)
    default_eval = compact_eval(feature_con, source_con, test_default, args)
    auto_eval = compact_eval(feature_con, source_con, test_auto, args)

    default_primary = default_eval["primary_route"]
    auto_primary = auto_eval["primary_route"]
    default_median = default_primary.get("median_net_bps")
    auto_median = auto_primary.get("median_net_bps")
    default_cum = default_primary.get("cum_net_bps")
    auto_cum = auto_primary.get("cum_net_bps")

    verdict = "MIXED"
    if auto_median is not None and default_median is not None and auto_cum is not None and default_cum is not None:
        if auto_median > default_median and auto_cum > default_cum:
            verdict = "AUTO_WEIGHTS_IMPROVED_OOS"
        elif auto_median < default_median and auto_cum < default_cum:
            verdict = "AUTO_WEIGHTS_WORSE_OOS"

    return {
        "enabled": True,
        "status": "OK",
        "train_frac": args.oos_train_frac,
        "train_n": len(train),
        "test_n": len(test),
        "k": args.k,
        "importance_route": args.importance_route,
        "suggested_weights": suggested,
        "test_default": {
            "knn": test_default_meta,
            **default_eval,
        },
        "test_auto_weights": {
            "knn": test_auto_meta,
            **auto_eval,
        },
        "delta": {
            "median_net_bps": None if auto_median is None or default_median is None else float(auto_median) - float(default_median),
            "cum_net_bps": None if auto_cum is None or default_cum is None else float(auto_cum) - float(default_cum),
        },
        "verdict": verdict,
    }


def horizon_summary(source_con: sqlite3.Connection, events: list[dict[str, Any]], horizons: tuple[int, ...]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for horizon in horizons:
        vals = []
        for event in events:
            value = return_bps(source_con, str(event["symbol"]), int(event["event_ts_ms"]), horizon)
            if value is not None:
                vals.append(value)
        out[f"{horizon}s"] = summarize(vals)
    return out


def nearest_events(events: list[dict[str, Any]], args: argparse.Namespace, n: int) -> list[dict[str, Any]]:
    if not events:
        return []
    if any("_knn_distance" in event for event in events):
        rows = sorted(events, key=lambda event: (float(event.get("_knn_distance") or 0.0), int(event.get("event_ts_ms") or 0)))[:n]
        return [
            {
                "event_id": event.get("event_id"),
                "event_utc": event.get("event_utc"),
                "cluster_notional": event.get("cluster_notional"),
                "day_trend_bps": event.get("day_trend_bps"),
                "day_range_bps": event.get("day_range_bps"),
                "symbol_pre_15m_bps": event.get("symbol_pre_15m_bps"),
                "distance": round(float(event.get("_knn_distance") or 0.0), 4),
                "components": event.get("_knn_components") or {},
            }
            for event in rows
        ]
    target_notional = args.target_notional if args.target_notional is not None else args.min_notional
    target_trend = args.target_day_trend_bps
    target_range = args.target_day_range_bps
    target_pre15 = args.target_symbol_pre_15m_bps

    scored = []
    for event in events:
        score = 0.0
        if target_notional and float(event.get("cluster_notional") or 0.0) > 0:
            score += abs(math.log(float(event["cluster_notional"])) - math.log(float(target_notional))) * 2.0
        if target_trend is not None and event.get("day_trend_bps") is not None:
            score += abs(float(event["day_trend_bps"]) - float(target_trend)) / 100.0
        if target_range is not None and event.get("day_range_bps") is not None:
            score += abs(float(event["day_range_bps"]) - float(target_range)) / 150.0
        if target_pre15 is not None and event.get("symbol_pre_15m_bps") is not None:
            score += abs(float(event["symbol_pre_15m_bps"]) - float(target_pre15)) / 100.0
        scored.append((score, event))
    scored.sort(key=lambda x: (x[0], int(x[1].get("event_ts_ms") or 0)))
    return [
        {
            "event_id": event.get("event_id"),
            "event_utc": event.get("event_utc"),
            "cluster_notional": event.get("cluster_notional"),
            "day_trend_bps": event.get("day_trend_bps"),
            "day_range_bps": event.get("day_range_bps"),
            "symbol_pre_15m_bps": event.get("symbol_pre_15m_bps"),
            "distance": round(score, 4),
        }
        for score, event in scored[:n]
    ]


def confidence_label(n: int) -> str:
    if n >= 100:
        return "broad"
    if n >= 40:
        return "usable"
    if n >= 20:
        return "thin"
    return "too_thin"


def fmt_bps(value: Any) -> str:
    if value is None:
        return "-"
    return f"{float(value):+.2f}"


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    fwd = payload["forward_returns_bps"]
    routes = payload["route_simulation"]
    lines = [
        "# S34 Liquidation Outcome Calculator",
        "",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- preset: `{payload.get('preset') or '-'}`",
        f"- source_signal: `{(payload.get('source_signal') or {}).get('signal_id') or '-'}`",
        f"- source_trade: `{(payload.get('source_trade') or {}).get('trade_id') or '-'}`",
        f"- scope: `{payload['scope_note']}`",
        f"- selection_mode: `{payload['selection_mode']}`",
        f"- decision_card: `{(payload.get('decision_card') or {}).get('verdict', '-')}`",
        f"- model_tag: `{payload.get('model_tag', '-')}`",
        f"- candidate_events: `{payload['candidate_n']}`",
        f"- matched_events: `{payload['matched_n']}`",
        f"- confidence: `{payload['confidence']}`",
        f"- filters: `{'; '.join(payload['filters'])}`",
        "",
        "## Forward Return Distribution",
        "",
        "| Horizon | N | Mean | Median | P25 | P75 | Positive Rate |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for horizon, row in fwd.items():
        pos = row["positive_rate"]
        lines.append(
            f"| {horizon} | {row['n']} | {fmt_bps(row['mean'])} | {fmt_bps(row['median'])} | "
            f"{fmt_bps(row['p25'])} | {fmt_bps(row['p75'])} | "
            f"{'-' if pos is None else f'{pos*100:.1f}%'} |"
        )

    lines.extend(
        [
            "",
            "## Route Simulation",
            "",
            "| Route | N | Median Net | Mean Net | Cum Net | Top3 Removed | WR | TP/BE/SL/TIME | MFE Median | MAE Median |",
            "|---|---:|---:|---:|---:|---:|---:|---|---:|---:|",
        ]
    )
    card = payload.get("decision_card") or {}
    if card:
        lines.extend(
            [
                "",
                "## Decision Card",
                "",
                f"- verdict: `{card.get('verdict')}`",
                f"- model_tag: `{card.get('model_tag')}`",
                f"- recommended_route: `{card.get('recommended_route_id') or '-'}`",
                f"- reasons: `{'; '.join(card.get('reasons') or []) or '-'}`",
                f"- warnings: `{'; '.join(card.get('warnings') or []) or '-'}`",
            ]
        )
    for route_id, row in routes.items():
        net = row["net_bps"]
        mfe = row["mfe_bps"]
        mae = row["mae_bps"]
        mix = row["exit_mix"]
        lines.append(
            f"| `{route_id}` | {row['n']} | {fmt_bps(net['median'])} | {fmt_bps(net['mean'])} | "
            f"{fmt_bps(row['cum_net_bps'])} | {fmt_bps(row['top3_removed_cum_net_bps'])} | "
            f"{'-' if net['positive_rate'] is None else f'{net['positive_rate']*100:.1f}%'} | "
            f"{mix['TP']}/{mix['BE']}/{mix['SL']}/{mix['TIME']} | "
            f"{fmt_bps(mfe['median'])} | {fmt_bps(mae['median'])} |"
        )

    lines.extend(
        [
            "",
            "## Similarity",
            "",
        ]
    )
    knn = payload.get("knn") or {}
    if knn.get("enabled"):
        lines.append(f"- candidate_n: `{knn.get('candidate_n')}`")
        lines.append(f"- selected_n: `{knn.get('selected_n')}`")
        lines.append(f"- k: `{knn.get('k')}`")
        if knn.get("auto_weights_applied"):
            lines.append("- auto_weights_applied: `true`")
        lines.append("")
        lines.append("| Feature | Target | Weight | Scale |")
        lines.append("|---|---:|---:|---:|")
        targets = knn.get("target_features") or {}
        scales = knn.get("feature_scales") or {}
        for name, spec in targets.items():
            lines.append(
                f"| `{name}` | {fmt_bps(spec.get('raw_value'))} | {float(spec.get('weight') or 0):.2f} | {fmt_bps(scales.get(name))} |"
            )
    else:
        lines.append("Filter mode. No weighted KNN similarity was applied.")

    auto = payload.get("auto_weights") or {}
    suggested = auto.get("suggested") or {}
    if suggested.get("available"):
        lines.extend(
            [
                "",
                "## Suggested KNN Weights",
                "",
                f"- method: `{suggested.get('method')}`",
                f"- applied: `{'true' if auto.get('enabled') else 'false'}`",
                "",
                "| Feature | Base | Suggested | Impact |",
                "|---|---:|---:|---:|",
            ]
        )
        for row in suggested.get("details") or []:
            lines.append(
                f"| `{row['feature']}` | {float(row['base_weight']):.2f} | "
                f"{float(row['suggested_weight']):.2f} | {float(row['impact']):.2f} |"
            )

    importance = payload.get("knn_importance") or []
    if importance:
        lines.extend(
            [
                "",
                "## KNN Feature Importance",
                "",
                "Leave-one-feature-out impact on the selected neighbor set and primary route.",
                "",
                "| Dropped Feature | Median Delta | Cum Delta | Neighbor Overlap | Drop Median |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for row in importance:
            overlap = row.get("neighbor_overlap_pct")
            lines.append(
                f"| `{row['feature']}` | {fmt_bps(row.get('median_delta_bps'))} | "
                f"{fmt_bps(row.get('cum_delta_bps'))} | "
                f"{'-' if overlap is None else f'{overlap*100:.1f}%'} | "
                f"{fmt_bps(row.get('drop_median_net_bps'))} |"
            )

    oos = payload.get("oos_validation") or {}
    if oos.get("enabled"):
        lines.extend(
            [
                "",
                "## OOS Validation",
                "",
                f"- status: `{oos.get('status')}`",
                f"- verdict: `{oos.get('verdict', '-')}`",
                f"- train_n: `{oos.get('train_n')}`",
                f"- test_n: `{oos.get('test_n')}`",
                f"- k: `{oos.get('k')}`",
                "",
            ]
        )
        if oos.get("status") == "OK":
            default_primary = ((oos.get("test_default") or {}).get("primary_route") or {})
            auto_primary = ((oos.get("test_auto_weights") or {}).get("primary_route") or {})
            delta = oos.get("delta") or {}
            lines.extend(
                [
                    "| Model | N | Median Net | Mean Net | Cum Net | WR |",
                    "|---|---:|---:|---:|---:|---:|",
                    f"| Default KNN | {default_primary.get('n', '-')} | {fmt_bps(default_primary.get('median_net_bps'))} | {fmt_bps(default_primary.get('mean_net_bps'))} | {fmt_bps(default_primary.get('cum_net_bps'))} | {'-' if default_primary.get('wr') is None else f'{default_primary.get('wr')*100:.1f}%'} |",
                    f"| Auto-weight KNN | {auto_primary.get('n', '-')} | {fmt_bps(auto_primary.get('median_net_bps'))} | {fmt_bps(auto_primary.get('mean_net_bps'))} | {fmt_bps(auto_primary.get('cum_net_bps'))} | {'-' if auto_primary.get('wr') is None else f'{auto_primary.get('wr')*100:.1f}%'} |",
                    "",
                    f"- median_delta_bps: `{fmt_bps(delta.get('median_net_bps'))}`",
                    f"- cum_delta_bps: `{fmt_bps(delta.get('cum_net_bps'))}`",
                ]
            )

    lines.extend(
        [
            "",
            "## Nearest Analogs",
            "",
            "| Event | UTC | Notional | Day Trend | Day Range | Symbol Pre15 | Distance |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in payload["nearest_analogs"]:
        lines.append(
            f"| `{row['event_id']}` | {row['event_utc']} | {float(row['cluster_notional'] or 0):.0f} | "
            f"{fmt_bps(row['day_trend_bps'])} | {fmt_bps(row['day_range_bps'])} | "
            f"{fmt_bps(row['symbol_pre_15m_bps'])} | {row['distance']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Read",
            "",
            "This is a conditional historical distribution, not a price forecast. It is paper/research only.",
            "If confidence is `thin` or `too_thin`, treat the output as a hypothesis prompt, not evidence.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_decision_card(payload: dict[str, Any]) -> dict[str, Any]:
    routes = payload.get("route_simulation") or {}
    primary: list[dict[str, Any]] = []
    for route_id, row in routes.items():
        if "CONTROL" in str(route_id):
            continue
        net = row.get("net_bps") or {}
        primary.append(
            {
                "route_id": route_id,
                "n": row.get("n"),
                "median_net_bps": net.get("median"),
                "mean_net_bps": net.get("mean"),
                "win_rate": net.get("positive_rate"),
                "cum_net_bps": row.get("cum_net_bps"),
                "top3_removed_cum_net_bps": row.get("top3_removed_cum_net_bps"),
                "exit_mix": row.get("exit_mix") or {},
            }
        )
    best = max(
        primary,
        key=lambda r: (
            r.get("median_net_bps") if r.get("median_net_bps") is not None else -9999.0,
            r.get("top3_removed_cum_net_bps") if r.get("top3_removed_cum_net_bps") is not None else -999999.0,
        ),
        default=None,
    )
    filters = payload.get("filters") or []
    sym = next((f.split("=", 1)[1] for f in filters if f.startswith("symbol=")), None)
    side = next((f.split("=", 1)[1] for f in filters if f.startswith("side=")), None)
    model_tag = MODEL_USEFULNESS.get((sym, side), "UNKNOWN")
    reasons: list[str] = []
    warnings: list[str] = []
    confidence = payload.get("confidence")
    decision_grade = bool(payload.get("decision_grade"))
    prediction_mode = str(payload.get("prediction_mode") or "")
    median = best.get("median_net_bps") if best else None
    n = int(best.get("n") or 0) if best else 0

    if not best:
        verdict = "NO_ROUTE"
        warnings.append("no_primary_route")
    elif confidence in {"thin", "too_thin"} or n < 30:
        verdict = "RESEARCH_ONLY"
        warnings.append(f"sample_too_small_n={n}")
    elif median is None or float(median) <= 0:
        verdict = "AVOID"
        warnings.append("best_route_median_not_positive")
    elif not decision_grade:
        verdict = "BASE_RATE_ONLY" if prediction_mode == "filter_scan" else "RESEARCH_ONLY"
        reasons.append("not_event_specific")
    elif model_tag == "KNN_USEFUL":
        verdict = "KNN_SUPPORTED"
        reasons.append("event_specific_knn_supported")
    elif model_tag == "BASE_RATE_ONLY":
        verdict = "BASE_RATE_ONLY"
        reasons.append("knn_not_incremental_for_combo")
    else:
        verdict = "RESEARCH_ONLY"
        warnings.append(f"model_tag={model_tag}")

    if best and float(best.get("median_net_bps") or 0.0) > 0:
        reasons.append(f"best_route_median={float(best['median_net_bps']):+.1f}bps")
    if best and best.get("win_rate") is not None:
        reasons.append(f"best_route_wr={float(best['win_rate']) * 100:.0f}%")
    if payload.get("source_signal"):
        reasons.append("source_signal_targets_loaded")
    if payload.get("preset"):
        reasons.append(f"preset={payload['preset']}")

    if best:
        exit_mix = best.get("exit_mix") or {}
        total_exits = sum(int(v or 0) for v in exit_mix.values())
        be_time = int(exit_mix.get("BE") or 0) + int(exit_mix.get("TIME") or 0)
        if total_exits and be_time / total_exits > 0.35:
            warnings.append(f"high_be_time_rate={(be_time / total_exits) * 100:.0f}%")
        if best.get("top3_removed_cum_net_bps") is not None and float(best["top3_removed_cum_net_bps"]) <= 0:
            warnings.append("outlier_dependent_top3_removed_negative")

    return {
        "verdict": verdict,
        "model_tag": model_tag,
        "decision_grade": decision_grade,
        "prediction_mode": prediction_mode,
        "best_route": best,
        "recommended_route_id": best.get("route_id") if best else None,
        "reasons": reasons,
        "warnings": warnings,
    }


def generate(args: argparse.Namespace) -> dict[str, Any]:
    if not FEATURE_DB.exists():
        raise SystemExit(f"Missing feature DB: {FEATURE_DB}")

    # ── Step 3: resolve per-symbol/side notional threshold ────────────────────
    if args.min_notional is None:
        effective_notional = SYMBOL_SIDE_NOTIONAL_DEFAULTS.get(
            (args.symbol, args.side), 200_000.0
        )
        threshold_source = "default_by_combo"
    else:
        effective_notional = args.min_notional
        threshold_source = "explicit"
    args.min_notional = effective_notional  # downstream build_where uses args.min_notional

    # ── Step 4: resolve per-combo model config (K, metric, excluded features) ─
    combo_config = COMBO_MODEL_CONFIGS.get((args.symbol, args.side), {})
    effective_k = args.k if args.k is not None else combo_config.get("k", 50)
    effective_metric = args.metric or combo_config.get("metric", "euclidean")
    effective_exclude: frozenset[str] = combo_config.get("exclude_features", frozenset())
    args.k = effective_k  # downstream functions read args.k

    feature_con = sqlite3.connect(FEATURE_DB)
    source_con = sqlite3.connect(MICRO_DB_URI, uri=True, timeout=10)
    source_con.execute("PRAGMA query_only=1")

    columns = table_columns(feature_con, "liq_event_features")
    where_sql, params, labels = build_where(args, columns)
    candidates = fetch_events(feature_con, where_sql, params, args.limit)

    # ── Step 1: detect target-less mode ───────────────────────────────────────
    if args.mode == "knn":
        _all_targets = target_features(args, exclude_features=effective_exclude)
        # log_cluster_notional is auto-populated from min_notional (filter param), not a signal target
        has_targets = bool({k: v for k, v in _all_targets.items() if k != "log_cluster_notional"})
        if has_targets:
            prediction_mode = "event_prediction"
            decision_grade = True
            prediction_warning: str | None = None
        else:
            prediction_mode = "population_scan"
            decision_grade = False
            prediction_warning = "No target features supplied; this is not a live-event prediction."
    else:
        prediction_mode = "filter_scan"
        decision_grade = False
        prediction_warning = None

    knn_meta: dict[str, Any] = {"enabled": False}
    auto_weights: dict[str, Any] = {"enabled": False}
    importance: list[dict[str, Any]] = []
    if args.mode == "knn":
        events, knn_meta = knn_select(candidates, args, metric=effective_metric,
                                       exclude_features=effective_exclude)
        if args.importance or args.auto_weights:
            baseline_event_ids = [str(event["event_id"]) for event in events]
            baseline_outcomes = outcome_rows(feature_con, baseline_event_ids)
            baseline_route_sim = route_summary(baseline_outcomes)
            importance = run_knn_importance(feature_con, candidates, events, baseline_route_sim,
                                             args, metric=effective_metric,
                                             exclude_features=effective_exclude)
            suggested = suggested_weights_from_importance(importance, args,
                                                           exclude_features=effective_exclude)
            auto_weights = {"enabled": bool(args.auto_weights), "suggested": suggested}
            if args.auto_weights and suggested.get("available"):
                events, knn_meta = knn_select(candidates, args,
                                               weight_overrides=suggested.get("weights") or {},
                                               metric=effective_metric,
                                               exclude_features=effective_exclude)
                knn_meta["auto_weights_applied"] = True
                knn_meta["applied_weights"] = suggested.get("weights") or {}
    else:
        events = candidates
    event_ids = [str(event["event_id"]) for event in events]
    outcomes = outcome_rows(feature_con, event_ids)
    route_sim = route_summary(outcomes)
    if args.mode == "knn" and args.importance and args.auto_weights:
        importance = run_knn_importance(feature_con, candidates, events, route_sim, args,
                                         metric=effective_metric, exclude_features=effective_exclude)
    oos = (
        run_oos_validation(feature_con, source_con, candidates, args,
                           metric=effective_metric, exclude_features=effective_exclude)
        if args.oos else {"enabled": False}
    )

    horizons = tuple(int(x) for x in args.horizons_sec)
    active_feature_set = [f[0] for f in KNN_FEATURES if f[0] not in effective_exclude]
    model_tag = MODEL_USEFULNESS.get((args.symbol, args.side), "UNKNOWN")
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "scope_note": "  |  ".join(
            f"{row[0]} {row[1]} N={row[2]}"
            for row in feature_con.execute(
                "SELECT symbol, liq_side, COUNT(*) FROM liq_event_features GROUP BY symbol, liq_side ORDER BY symbol, liq_side"
            ).fetchall()
        ),
        "filters": labels,
        "preset": getattr(args, "preset_applied", None),
        "source_signal": getattr(args, "source_signal", None),
        "source_trade": getattr(args, "source_trade", None),
        "run_command": "python tools\\s34_liq_outcome_calculator.py " + " ".join(sys.argv[1:]),
        "model_tag": model_tag,
        "selection_mode": args.mode,
        "prediction_mode": prediction_mode,
        "decision_grade": decision_grade,
        **({"warning": prediction_warning} if prediction_warning else {}),
        "threshold_source": threshold_source,
        "min_notional_used": effective_notional,
        "model_config": {
            "combo": f"{args.symbol}_{args.side}",
            "k": effective_k,
            "metric": effective_metric,
            "feature_set": active_feature_set,
            "excluded_features": sorted(effective_exclude),
            "config_source": "combo_override" if combo_config else "default",
        },
        "candidate_n": len(candidates),
        "matched_n": len(events),
        "confidence": confidence_label(len(events)),
        "knn": knn_meta,
        "auto_weights": auto_weights,
        "oos_validation": oos,
        "coverage": {
            "feature_db": str(FEATURE_DB),
            "microstructure_db": str(ROOT / "data" / "microstructure.db"),
            "available_symbol_sides": [
                dict(symbol=row[0], liq_side=row[1], n=row[2], min_utc=row[3], max_utc=row[4])
                for row in feature_con.execute(
                    """
                    SELECT symbol, liq_side, COUNT(*), MIN(event_utc), MAX(event_utc)
                    FROM liq_event_features
                    GROUP BY symbol, liq_side
                    ORDER BY symbol, liq_side
                    """
                ).fetchall()
            ],
        },
        "forward_returns_bps": horizon_summary(source_con, events, horizons),
        "route_simulation": route_sim,
        "knn_importance": importance,
        "nearest_analogs": nearest_events(events, args, args.neighbors),
    }
    payload["decision_card"] = build_decision_card(payload)
    feature_con.close()
    source_con.close()
    return payload


def print_summary(payload: dict[str, Any]) -> None:
    print("=== S34 LIQ OUTCOME CALCULATOR ===")
    print(f"Mode           : {payload['selection_mode']}")
    print(f"Preset         : {payload.get('preset') or '-'}")
    source_signal = payload.get("source_signal") or {}
    if source_signal:
        print(f"Source signal  : {source_signal.get('signal_id')} @ {source_signal.get('signal_ts_utc')}")
    source_trade = payload.get("source_trade") or {}
    if source_trade:
        print(f"Source trade   : {source_trade.get('trade_id')} ({source_trade.get('status')})")
    card = payload.get("decision_card") or {}
    print(f"Decision card  : {card.get('verdict', '-')} ({card.get('model_tag', payload.get('model_tag', '-'))})")
    if card.get("recommended_route_id"):
        print(f"Recommended    : {card.get('recommended_route_id')}")
    if card.get("warnings"):
        print(f"Warnings       : {'; '.join(card.get('warnings') or [])}")
    print(f"Candidates     : {payload['candidate_n']}")
    print(f"Matched events : {payload['matched_n']} ({payload['confidence']})")
    print(f"Scope          : {payload['scope_note']}")
    print(f"Filters        : {'; '.join(payload['filters'])}")
    print("")
    print("Forward returns:")
    for horizon, row in payload["forward_returns_bps"].items():
        print(
            f"  {horizon:<5} N={row['n']:<4} "
            f"median={fmt_bps(row['median'])} bps  p25={fmt_bps(row['p25'])}  p75={fmt_bps(row['p75'])}"
        )
    print("")
    print("Route simulations:")
    for route_id, row in payload["route_simulation"].items():
        net = row["net_bps"]
        print(
            f"  {route_id:<26} N={row['n']:<4} "
            f"median={fmt_bps(net['median'])} mean={fmt_bps(net['mean'])} cum={fmt_bps(row['cum_net_bps'])}"
        )
    if payload.get("knn_importance"):
        print("")
        print("KNN feature importance:")
        for row in payload["knn_importance"][:8]:
            overlap = row.get("neighbor_overlap_pct")
            print(
                f"  drop {row['feature']:<24} "
                f"median_delta={fmt_bps(row.get('median_delta_bps'))} "
                f"overlap={'-' if overlap is None else f'{overlap*100:.1f}%'}"
            )
    auto = payload.get("auto_weights") or {}
    suggested = auto.get("suggested") or {}
    if suggested.get("available"):
        print("")
        print("Suggested weights" + (" (applied):" if auto.get("enabled") else " (not applied):"))
        for row in (suggested.get("details") or [])[:8]:
            print(f"  {row['feature']:<24} {float(row['base_weight']):.2f} -> {float(row['suggested_weight']):.2f}")
    oos = payload.get("oos_validation") or {}
    if oos.get("enabled"):
        print("")
        print(f"OOS validation: {oos.get('status')} {oos.get('verdict', '')}")
        if oos.get("status") == "OK":
            default_primary = ((oos.get("test_default") or {}).get("primary_route") or {})
            auto_primary = ((oos.get("test_auto_weights") or {}).get("primary_route") or {})
            delta = oos.get("delta") or {}
            print(
                f"  default median={fmt_bps(default_primary.get('median_net_bps'))} "
                f"cum={fmt_bps(default_primary.get('cum_net_bps'))}"
            )
            print(
                f"  auto    median={fmt_bps(auto_primary.get('median_net_bps'))} "
                f"cum={fmt_bps(auto_primary.get('cum_net_bps'))}"
            )
            print(
                f"  delta   median={fmt_bps(delta.get('median_net_bps'))} "
                f"cum={fmt_bps(delta.get('cum_net_bps'))}"
            )
    print("==================================")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Conditional outcome calculator over S34 liquidation feature factory events.")
    parser.add_argument("--preset", choices=sorted(PRESETS), help="Apply a named S34 bucket preset.")
    parser.add_argument("--from-latest-signal", action="store_true", help="Use the latest s34_intelligence signal as KNN target features.")
    parser.add_argument("--from-latest-accepted-signal", action="store_true", help="Use the latest signal that opened a paper trade as KNN target features.")
    parser.add_argument("--from-open-trade", action="store_true", help="Use the latest currently open paper trade as KNN target features.")
    parser.add_argument("--from-trade-id", help="Use the signal that produced the given paper trade id as KNN target features.")
    parser.add_argument("--signal-id", help="Use a specific s34_intelligence signal_id as KNN target features.")
    parser.add_argument("--mode", choices=["filter", "knn"], default="filter")
    parser.add_argument("--symbol", default="ETHUSDT")
    parser.add_argument("--side", default="BUY", choices=["BUY", "SELL"])
    parser.add_argument("--min-notional", type=float, default=None,
                        help="Notional threshold. Defaults to per-symbol/side value if omitted.")
    parser.add_argument("--max-notional", type=float)
    parser.add_argument("--min-day-trend-bps", type=float)
    parser.add_argument("--max-day-trend-bps", type=float)
    parser.add_argument("--min-day-range-bps", type=float)
    parser.add_argument("--min-symbol-pre-5m-bps", type=float)
    parser.add_argument("--min-symbol-pre-15m-bps", type=float)
    parser.add_argument("--min-btc-pre-15m-bps", type=float)
    parser.add_argument("--max-duration-sec", type=float)
    parser.add_argument("--min-inter-cluster-gap-sec", type=float)
    parser.add_argument("--min-max-single-share", type=float)
    parser.add_argument("--shape-label")
    parser.add_argument("--k", type=int, default=None,
                        help="K nearest events. Defaults to per-combo config (e.g. 10 for ETH SELL) or 50.")
    parser.add_argument("--metric", default=None, choices=["euclidean", "manhattan"],
                        help="Distance metric. Defaults to per-combo config (e.g. manhattan for ETH SELL) or euclidean.")
    parser.add_argument("--target-notional", type=float, help="Similarity target; defaults to --min-notional.")
    parser.add_argument("--target-duration-sec", type=float)
    parser.add_argument("--target-liq-count", type=float)
    parser.add_argument("--target-max-single-share", type=float)
    parser.add_argument("--target-intensity-per-sec", type=float)
    parser.add_argument("--target-inter-cluster-gap-sec", type=float)
    parser.add_argument("--target-day-trend-bps", type=float)
    parser.add_argument("--target-day-range-bps", type=float)
    parser.add_argument("--target-symbol-pre-5m-bps", type=float)
    parser.add_argument("--target-symbol-pre-15m-bps", type=float)
    parser.add_argument("--target-btc-pre-15m-bps", type=float)
    parser.add_argument("--horizons-sec", type=int, nargs="+", default=list(DEFAULT_HORIZONS_SEC))
    parser.add_argument("--neighbors", type=int, default=10)
    parser.add_argument("--importance", action="store_true", help="Run leave-one-feature-out KNN importance in knn mode.")
    parser.add_argument("--auto-weights", action="store_true", help="Apply KNN weights suggested by leave-one-feature-out importance.")
    parser.add_argument("--importance-route", default="LONG_DELAY0_TP60")
    parser.add_argument("--oos", action="store_true", help="Run time-split validation comparing default KNN vs auto-weight KNN.")
    parser.add_argument("--oos-train-frac", type=float, default=0.5)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output-json", type=Path, default=OUT_DIR / "S34_LIQ_OUTCOME_CALCULATOR_LATEST.json")
    parser.add_argument("--output-md", type=Path, default=OUT_DIR / "S34_LIQ_OUTCOME_CALCULATOR_LATEST.md")
    return parser


def parse_args() -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args()
    args._explicit_options = {item.split("=", 1)[0] for item in sys.argv[1:] if item.startswith("--")}
    return args


def main() -> None:
    args = parse_args()
    defaults = _parser_defaults()
    apply_preset(args, defaults)
    apply_signal_targets(args, defaults)
    payload = generate(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(args.output_md, payload)
    ts_str = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    archive_json = args.output_json.parent / f"S34_LIQ_OUTCOME_CALCULATOR_{ts_str}.json"
    archive_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print_summary(payload)
    print(f"JSON (latest) : {args.output_json}")
    print(f"JSON (archive): {archive_json}")
    print(f"MD            : {args.output_md}")


if __name__ == "__main__":
    main()
