from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from execution.sim.price_oracle import PriceOracle

def _parse_iso_utc(text: str) -> float:
    s = str(text or "").strip()
    if not s:
        raise ValueError("empty timestamp")
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def _to_iso(ts: float) -> str:
    return datetime.fromtimestamp(float(ts), tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _round(x: Optional[float], nd: int = 12) -> Optional[float]:
    if x is None:
        return None
    return round(float(x), nd)


@dataclass
class ExecSimConfig:
    fee_bps: float = 0.0
    spread_bps: float = 0.0
    use_spread_model: bool = False
    qty: float = 1.0
    horizon_sec: int = 120
    price_field_candidates: tuple[str, ...] = (
        "price",
        "mark_price",
        "trade_price",
        "px",
        "p",
        "close",
    )
    side_rule: str = "always_buy"  # always_buy | from_params
    skip_missing_price: bool = True
    horizon_or_before_enabled: bool = True
    fill_or_before_enabled: bool = False


def simulate_fills(
    decisions: List[Dict[str, Any]],
    events: List[Dict[str, Any]],
    cfg: ExecSimConfig,
) -> List[Dict[str, Any]]:
    fills, _ = simulate_fills_with_skips(decisions=decisions, events=events, cfg=cfg)
    return fills


def simulate_fills_with_skips(
    decisions: List[Dict[str, Any]],
    events: List[Dict[str, Any]],
    cfg: ExecSimConfig,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    oracle = PriceOracle.build(events=events, price_field_candidates=cfg.price_field_candidates)
    fills: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    fee_ratio = float(cfg.fee_bps) / 10000.0  # bps -> ratio
    spread_ratio_half = float(cfg.spread_bps) / 10000.0 / 2.0 if (cfg.use_spread_model or float(cfg.spread_bps) > 0.0) else 0.0

    for d in decisions:
        symbol = str(d.get("symbol") or "ALL")
        ts_utc = str(d.get("ts_utc") or "")
        decision_id = str(d.get("decision_id") or "")
        params = d.get("params") if isinstance(d.get("params"), dict) else {}
        side = "buy"
        if cfg.side_rule == "from_params":
            raw_side = str(params.get("side") or "").lower()
            if raw_side in ("buy", "sell"):
                side = raw_side

        if not oracle.has_symbol(symbol):
            skipped.append(
                {
                    "decision_id": decision_id,
                    "ts_utc": ts_utc,
                    "symbol": symbol,
                    "reason": "no_event_at_or_after_ts",
                    "details": {"phase": "symbol_index", "side": side},
                }
            )
            continue

        try:
            decision_ts = _parse_iso_utc(ts_utc)
        except Exception:
            skipped.append(
                {
                    "decision_id": decision_id,
                    "ts_utc": ts_utc,
                    "symbol": symbol,
                    "reason": "missing_fill_price",
                    "details": {"phase": "parse_ts", "side": side},
                }
            )
            continue
        fill_point = oracle.price_at_or_after(symbol=symbol, target_ts=decision_ts)
        fill_source = "after"
        if fill_point is None and cfg.fill_or_before_enabled:
            fill_point = oracle.price_at_or_before(symbol=symbol, target_ts=decision_ts)
            fill_source = "before" if fill_point is not None else "none"
        if fill_point is None:
            skipped.append(
                {
                    "decision_id": decision_id,
                    "ts_utc": ts_utc,
                    "symbol": symbol,
                    "reason": "no_event_at_or_after_ts",
                    "details": {"phase": "fill_lookup", "side": side, "oracle_used": fill_source},
                }
            )
            continue

        fill_ts = fill_point.ts
        fill_px_raw = float(fill_point.price)
        fill_iso = str(fill_point.ts_utc)
        horizon_target = fill_ts + float(max(1, int(cfg.horizon_sec)))
        # No lookahead: only exact horizon match (after bounded to horizon) or before-horizon fallback.
        horizon_point = oracle.price_at_or_after(symbol=symbol, target_ts=horizon_target, max_ts=horizon_target)
        horizon_source = "after" if horizon_point is not None else "none"
        if horizon_point is None and cfg.horizon_or_before_enabled:
            horizon_point = oracle.price_at_or_before(symbol=symbol, target_ts=horizon_target)
            horizon_source = "before" if horizon_point is not None else "none"
        if horizon_point is None:
            skipped.append(
                {
                    "decision_id": decision_id,
                    "ts_utc": ts_utc,
                    "symbol": symbol,
                    "reason": "missing_horizon_price",
                    "details": {"phase": "horizon_lookup", "side": side, "oracle_used": horizon_source},
                }
            )
            continue
        horizon_px_raw = float(horizon_point.price)
        horizon_iso = str(horizon_point.ts_utc)
        qty = float(cfg.qty)

        if side == "sell":
            fill_px = fill_px_raw * (1.0 - spread_ratio_half)
            horizon_px = horizon_px_raw * (1.0 + spread_ratio_half)
        else:
            fill_px = fill_px_raw * (1.0 + spread_ratio_half)
            horizon_px = horizon_px_raw * (1.0 - spread_ratio_half)

        fee = abs(fill_px * qty) * fee_ratio

        window = oracle.price_window(symbol=symbol, start_ts=fill_ts, end_ts=horizon_point.ts)
        window_prices = [p.price for p in window]
        adverse_samples = len(window_prices)
        adverse_min_px = min(window_prices) if adverse_samples > 0 else fill_px_raw
        adverse_max_px = max(window_prices) if adverse_samples > 0 else fill_px_raw
        if side == "sell":
            pnl_gross = (fill_px - horizon_px) * qty
            adverse = fill_px_raw - float(adverse_max_px)
        else:
            pnl_gross = (horizon_px - fill_px) * qty
            adverse = float(adverse_min_px) - fill_px_raw
        pnl = pnl_gross - fee
        spread_cost_est = abs(fill_px_raw - fill_px) * qty + abs(horizon_px_raw - horizon_px) * qty

        fills.append(
            {
                "decision_id": decision_id,
                "fill_ts_utc": fill_iso,
                "symbol": symbol,
                "side": side,
                "qty": _round(qty),
                "fill_px_raw": _round(fill_px_raw),
                "fill_px": _round(fill_px),
                "fee": _round(fee),
                "horizon_px_raw": _round(horizon_px_raw),
                "horizon_ts_utc": horizon_iso,
                "horizon_px": _round(horizon_px),
                "pnl_gross": _round(pnl_gross),
                "pnl": _round(pnl),
                "adverse_move": _round(adverse),
                "adverse_samples": int(adverse_samples),
                "adverse_min_px": _round(adverse_min_px),
                "adverse_max_px": _round(adverse_max_px),
                "spread_bps": _round(cfg.spread_bps),
                "spread_cost_est": _round(spread_cost_est),
                "fill_price_source": fill_source,
                "horizon_price_source": horizon_source,
                "status": "filled",
            }
        )

    return fills, skipped
