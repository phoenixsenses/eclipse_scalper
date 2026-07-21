"""Shadow-only pre-liquidation detector telemetry.

This module computes the current ETH SELL pre-liquidation detector score from
live microstructure.db data. It does not trade and does not write to any DB.
It is intended for the S34 dashboard so we can observe detector behavior
forward before considering any executable strategy.
"""

from __future__ import annotations

import json
import math
import sqlite3
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MICRO_DB_DEFAULT = ROOT / "data" / "microstructure.db"
MODEL_PATH_DEFAULT = ROOT / "reports" / "research" / "s34" / "S34_PRELIQ_DETECTOR_V2.json"


def _iso_ms(ms: int | None) -> str | None:
    if ms is None:
        return None
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).isoformat()


def _book_at_or_before(con: sqlite3.Connection, symbol: str, ts_ms: int) -> sqlite3.Row | None:
    return con.execute(
        """
        SELECT ts_ms, bid_price, bid_qty, ask_price, ask_qty, mid_price,
               spread_pct, book_imbalance, bid_depth_usd
        FROM book_ticker INDEXED BY idx_bt_symbol_ts
        WHERE symbol=? AND ts_ms<=?
        ORDER BY ts_ms DESC
        LIMIT 1
        """,
        (symbol, int(ts_ms)),
    ).fetchone()


def _mark_at_or_before(con: sqlite3.Connection, symbol: str, ts_ms: int) -> sqlite3.Row | None:
    return con.execute(
        """
        SELECT ts_ms, mark_price
        FROM mark_prices INDEXED BY idx_mark_symbol_ts
        WHERE symbol=? AND ts_ms<=?
        ORDER BY ts_ms DESC
        LIMIT 1
        """,
        (symbol, int(ts_ms)),
    ).fetchone()


def _return_bps(con: sqlite3.Connection, symbol: str, ts_ms: int, lookback_sec: int) -> float | None:
    now = _mark_at_or_before(con, symbol, ts_ms)
    past = _mark_at_or_before(con, symbol, ts_ms - lookback_sec * 1000)
    if now is None or past is None:
        return None
    if abs(int(ts_ms) - int(now["ts_ms"])) > 5000:
        return None
    if abs((ts_ms - lookback_sec * 1000) - int(past["ts_ms"])) > 5000:
        return None
    prev = float(past["mark_price"])
    return (float(now["mark_price"]) - prev) / prev * 10_000.0 if prev else None


def _agg_stats(con: sqlite3.Connection, symbol: str, ts_ms: int, window_sec: int) -> dict[str, float]:
    row = con.execute(
        """
        SELECT
          COUNT(*) AS n,
          COALESCE(SUM(notional), 0.0) AS total_notional,
          COALESCE(SUM(CASE WHEN is_buyer_maker=1 THEN notional ELSE 0 END), 0.0) AS sell_taker_notional,
          COALESCE(SUM(CASE WHEN is_buyer_maker=0 THEN notional ELSE 0 END), 0.0) AS buy_taker_notional
        FROM agg_trades INDEXED BY idx_trade_symbol_ts
        WHERE symbol=? AND ts_ms>? AND ts_ms<=?
        """,
        (symbol, int(ts_ms - window_sec * 1000), int(ts_ms)),
    ).fetchone()
    prefix = symbol[:3].lower()
    total = float(row["total_notional"] or 0.0)
    sell = float(row["sell_taker_notional"] or 0.0)
    buy = float(row["buy_taker_notional"] or 0.0)
    return {
        f"{prefix}_agg_count_{window_sec}s": float(row["n"] or 0),
        f"{prefix}_agg_notional_{window_sec}s": total,
        f"{prefix}_sell_taker_notional_{window_sec}s": sell,
        f"{prefix}_buy_taker_notional_{window_sec}s": buy,
        f"{prefix}_taker_imbalance_{window_sec}s": (sell - buy) / total if total > 0 else 0.0,
    }


def _liq_stats(con: sqlite3.Connection, symbol: str, ts_ms: int, window_sec: int) -> dict[str, float]:
    rows = con.execute(
        """
        SELECT side, COUNT(*) AS n, COALESCE(SUM(notional), 0.0) AS notional
        FROM liquidations INDEXED BY idx_liq_symbol_ts
        WHERE symbol=? AND ts_ms>? AND ts_ms<=?
        GROUP BY side
        """,
        (symbol, int(ts_ms - window_sec * 1000), int(ts_ms)),
    ).fetchall()
    prefix = symbol[:3].lower()
    out = {
        f"{prefix}_liq_sell_count_{window_sec}s": 0.0,
        f"{prefix}_liq_buy_count_{window_sec}s": 0.0,
        f"{prefix}_liq_sell_notional_{window_sec}s": 0.0,
        f"{prefix}_liq_buy_notional_{window_sec}s": 0.0,
    }
    for row in rows:
        side = str(row["side"]).lower()
        if side in ("sell", "buy"):
            out[f"{prefix}_liq_{side}_count_{window_sec}s"] = float(row["n"] or 0)
            out[f"{prefix}_liq_{side}_notional_{window_sec}s"] = float(row["notional"] or 0.0)
    total = out[f"{prefix}_liq_sell_notional_{window_sec}s"] + out[f"{prefix}_liq_buy_notional_{window_sec}s"]
    out[f"{prefix}_liq_imbalance_{window_sec}s"] = (
        (out[f"{prefix}_liq_sell_notional_{window_sec}s"] - out[f"{prefix}_liq_buy_notional_{window_sec}s"]) / total
        if total > 0
        else 0.0
    )
    return out


def _current_features(con: sqlite3.Connection) -> dict[str, Any]:
    latest = con.execute("SELECT MAX(ts_ms) FROM book_ticker WHERE symbol='ETHUSDT'").fetchone()[0]
    if latest is None:
        raise RuntimeError("no ETHUSDT book_ticker rows")
    ts_ms = int(latest)
    now = _book_at_or_before(con, "ETHUSDT", ts_ms)
    if now is None:
        raise RuntimeError("missing current ETHUSDT book row")

    lookbacks = [1, 3, 5, 10, 15, 30]
    past: dict[int, sqlite3.Row] = {}
    for sec in lookbacks:
        row = _book_at_or_before(con, "ETHUSDT", ts_ms - sec * 1000)
        if row is None:
            raise RuntimeError(f"missing ETHUSDT book lookback {sec}s")
        past[sec] = row

    mid = float(now["mid_price"])
    bid_qty = float(now["bid_qty"])
    ask_qty = float(now["ask_qty"])
    features: dict[str, Any] = {
        "ts_ms": ts_ms,
        "ts_utc": _iso_ms(ts_ms),
        "mid": mid,
        "spread_bps": float(now["spread_pct"]) * 10_000.0,
        "book_imbalance": float(now["book_imbalance"]),
        "bid_qty": bid_qty,
        "ask_qty": ask_qty,
        "bid_depth_usd": float(now["bid_depth_usd"] or 0.0),
        "top_qty_usd": (bid_qty + ask_qty) * mid,
    }
    for sec, row in past.items():
        prev_mid = float(row["mid_price"])
        prev_bid_qty = float(row["bid_qty"])
        prev_ask_qty = float(row["ask_qty"])
        prev_imb = float(row["book_imbalance"])
        features[f"mid_down_{sec}s_bps"] = (prev_mid - mid) / prev_mid * 10_000.0
        features[f"imb_delta_{sec}s"] = float(now["book_imbalance"]) - prev_imb
        features[f"bid_qty_delta_{sec}s_pct"] = (bid_qty - prev_bid_qty) / max(prev_bid_qty, 1e-9)
        features[f"ask_qty_delta_{sec}s_pct"] = (ask_qty - prev_ask_qty) / max(prev_ask_qty, 1e-9)

    for sec in [1, 3, 5, 10, 30]:
        features.update(_agg_stats(con, "ETHUSDT", ts_ms, sec))
    for sym in ["ETHUSDT", "BTCUSDT", "SOLUSDT"]:
        for sec in [30, 60, 120]:
            features.update(_liq_stats(con, sym, ts_ms, sec))
    for sym in ["BTCUSDT", "SOLUSDT"]:
        for sec in [5, 10, 30]:
            features[f"{sym[:3].lower()}_ret_{sec}s_bps"] = _return_bps(con, sym, ts_ms, sec)
    return features


def _quantile(vals: list[float], q: float) -> float | None:
    if not vals:
        return None
    xs = sorted(vals)
    idx = (len(xs) - 1) * q
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return xs[int(idx)]
    return xs[lo] * (hi - idx) + xs[hi] * (idx - lo)


def _score(features: dict[str, Any], selected_features: list[dict[str, Any]]) -> float:
    # We do not persist the original scaler from the research run, so the live
    # card uses a directional normalized score. It is suitable for shadow
    # telemetry, not decision-grade trading.
    total = 0.0
    used = 0
    for item in selected_features:
        key = item.get("feature")
        if not key:
            continue
        value = features.get(str(key))
        if value is None:
            continue
        try:
            val = float(value)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(val):
            continue
        direction = 1.0 if int(item.get("direction", 1)) >= 0 else -1.0
        edge = abs(float(item.get("train_auc", 0.5)) - 0.5)
        total += direction * edge * math.asinh(val)
        used += 1
    return total / used if used else 0.0


def _level(score: float, q80_cutoff: float | None, q90_cutoff: float | None) -> str:
    # Research cutoffs are not directly comparable to the live normalized score,
    # but preserving them in payload keeps the panel honest. Use conservative
    # live score bands until a forward calibration file exists.
    _ = (q80_cutoff, q90_cutoff)
    if score >= 0.20:
        return "elevated"
    if score >= 0.08:
        return "watch"
    return "calm"


def current_preliq_shadow_payload(
    db_path: str | Path = MICRO_DB_DEFAULT,
    model_path: str | Path = MODEL_PATH_DEFAULT,
) -> dict[str, Any]:
    model_file = Path(model_path)
    if not model_file.exists():
        return {"available": False, "error": f"missing model file: {model_file}"}
    model = json.loads(model_file.read_text(encoding="utf-8"))
    db_uri = f"file:{Path(db_path).as_posix()}?mode=ro"
    con = sqlite3.connect(db_uri, uri=True, timeout=3)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA query_only=1")
    try:
        features = _current_features(con)
    finally:
        con.close()

    cards = []
    for result in model.get("threshold_results", []):
        threshold = float(result.get("threshold") or 0.0)
        selected = list(result.get("selected_features") or [])
        score = _score(features, selected)
        q80 = next((r for r in result.get("test_precision", []) if abs(float(r.get("score_quantile", 0)) - 0.80) < 1e-9), {})
        q90 = next((r for r in result.get("test_precision", []) if abs(float(r.get("score_quantile", 0)) - 0.90) < 1e-9), {})
        cards.append(
            {
                "threshold": threshold,
                "label": f"ETH SELL pre-liq {int(threshold / 1000)}K",
                "shadow_score": score,
                "level": _level(score, q80.get("cutoff"), q90.get("cutoff")),
                "test_auc": result.get("test_score_auc"),
                "test_q80_precision": q80.get("precision"),
                "test_q90_precision": q90.get("precision"),
                "test_q90_lift": q90.get("lift_vs_base"),
                "selected_features": selected,
            }
        )

    important_keys = [
        "mid_down_10s_bps",
        "mid_down_15s_bps",
        "btc_ret_5s_bps",
        "sol_ret_5s_bps",
        "btc_liq_imbalance_120s",
        "sol_liq_imbalance_120s",
        "eth_taker_imbalance_10s",
        "spread_bps",
    ]
    snapshot = {k: features.get(k) for k in important_keys}
    age_sec = max(0.0, (datetime.now(timezone.utc).timestamp() * 1000 - int(features["ts_ms"])) / 1000.0)
    return {
        "available": True,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_ts_utc": features.get("ts_utc"),
        "source_age_sec": age_sec,
        "symbol": "ETHUSDT",
        "side": "SELL",
        "mode": "SHADOW_ONLY",
        "decision_grade": False,
        "model_generated_at": model.get("generated_at"),
        "cards": cards,
        "feature_snapshot": snapshot,
        "warning": "Shadow telemetry only. Scores are not live trading signals and are not forward-calibrated.",
    }


def main() -> None:
    print(json.dumps(current_preliq_shadow_payload(), indent=2))


if __name__ == "__main__":
    main()
