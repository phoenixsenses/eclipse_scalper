from __future__ import annotations

"""
Micro-edge paper simulation (research only).

Note:
  `intensity_spike_imbalance_cont` should be treated as research-only by default.
  Promote only after independent validation shows positive net edge under realistic costs.

Examples:
  python -m tools.micro_edge_backtest --db data/microstructure.db --symbols BTCUSDT,ETHUSDT --lookback-min 240 --bucket-sec 5 --horizon-sec 60
  python -m tools.micro_edge_backtest --db data/microstructure.db --symbols BTCUSDT --rule imbalance_gt_q90_up --side LONG
"""

import argparse
import json
import os
import sqlite3
import time
from pathlib import Path
from statistics import median
from typing import Any, Dict, List, Optional, Tuple

from execution.passive_execution_simulator import calibrate_passive_model, simulate_passive_fill
from config.costs import DEFAULT_MAKER_FEE_BPS
from src.microphys.execution.engine import ExecutionRequest, build_default_engines
from tools.micro_edge_lib import (
    append_jsonl,
    build_bucket_features,
    label_value_to_text,
    compute_rule_thresholds,
    evaluate_naive_rules,
    extract_best_rule_delta_min_n,
    filter_rules_min_n,
    infer_rule_side,
    rule_fires,
    rule_predicted_side,
    signal_aligned_labels,
    utc_now_iso,
)
from tools.micro_edge_smoke import _parse_symbols, _load_symbol_trades_and_marks
from tools.micro_edge_signal_v2 import enrich_rows_with_v2


def compute_gross_return(entry_price: float, exit_price: float, side: str) -> float:
    e = float(entry_price)
    x = float(exit_price)
    if e <= 0 or x <= 0:
        raise ValueError("entry_price and exit_price must be > 0")
    s = str(side).upper()
    if s == "SHORT":
        return (e / x) - 1.0
    return (x / e) - 1.0


def compute_trade_cost(fee_bps: float, slip_bps: float) -> float:
    cost_bps = 2.0 * (float(fee_bps) + float(slip_bps))
    return cost_bps / 10000.0


def _exec_engine_unified_enabled() -> bool:
    v = str(os.getenv("EXEC_ENGINE_UNIFIED", "0")).strip().lower()
    return v in {"1", "true", "yes", "on"}


def compute_net_return(entry_price: float, exit_price: float, side: str, fee_bps: float, slip_bps: float) -> tuple[float, float]:
    gross = compute_gross_return(entry_price, exit_price, side=side)
    cost = compute_trade_cost(fee_bps, slip_bps)
    return gross - cost, cost


def _pearson_corr(xs: List[float], ys: List[float]) -> Optional[float]:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    dx = [x - mx for x in xs]
    dy = [y - my for y in ys]
    sx2 = sum(v * v for v in dx)
    sy2 = sum(v * v for v in dy)
    if sx2 <= 0.0 or sy2 <= 0.0:
        return None
    cov = sum(a * b for a, b in zip(dx, dy))
    return cov / ((sx2**0.5) * (sy2**0.5))


def compute_exec_cost(
    *,
    exec_model: str,
    fee_bps: float,
    slip_bps: float,
    maker_fee_bps: float,
    maker_penalty_bps: float,
    spread_ratio: Optional[float],
) -> Optional[float]:
    em = str(exec_model or "taker").lower()
    if em == "taker":
        return compute_trade_cost(float(fee_bps), float(slip_bps))
    if em == "maker":
        return 2.0 * (float(maker_fee_bps) + float(maker_penalty_bps)) / 10000.0
    if em == "mid":
        return 2.0 * float(fee_bps) / 10000.0
    if em == "halfspread":
        if spread_ratio is None:
            return None
        return 2.0 * (float(fee_bps) / 10000.0 + 0.5 * float(spread_ratio))
    if em == "passive_realistic":
        return None
    return compute_trade_cost(float(fee_bps), float(slip_bps))


def _quantile(vals: List[float], q: float) -> Optional[float]:
    xs = sorted(float(v) for v in vals)
    if not xs:
        return None
    if q <= 0.0:
        return xs[0]
    if q >= 1.0:
        return xs[-1]
    pos = (len(xs) - 1) * q
    lo = int(pos)
    hi = min(len(xs) - 1, lo + 1)
    if lo == hi:
        return xs[lo]
    w = pos - lo
    return xs[lo] * (1.0 - w) + xs[hi] * w


def compute_regime_bins(rows: List[Dict[str, Optional[float]]]) -> Dict[str, Tuple[Optional[float], Optional[float], Optional[float]]]:
    spreads = [float(r["spread"]) for r in rows if r.get("spread") is not None]
    ints = [float(r["trade_intensity"]) for r in rows if r.get("trade_intensity") is not None]
    vol_src: List[float] = []
    for r in rows:
        v = r.get("micro_volatility")
        if v is not None:
            vol_src.append(float(v))
            continue
        r1 = r.get("ret_1")
        if r1 is not None:
            vol_src.append(abs(float(r1)))
    return {
        "spread": (_quantile(spreads, 0.25), _quantile(spreads, 0.50), _quantile(spreads, 0.75)),
        "intensity": (_quantile(ints, 0.25), _quantile(ints, 0.50), _quantile(ints, 0.75)),
        "vol": (_quantile(vol_src, 0.25), _quantile(vol_src, 0.50), _quantile(vol_src, 0.75)),
    }


def _bin_from_edges(v: Optional[float], e: Tuple[Optional[float], Optional[float], Optional[float]]) -> str:
    if v is None:
        return "missing"
    q25, q50, q75 = e
    if q25 is None or q50 is None or q75 is None:
        return "unknown"
    x = float(v)
    if x <= float(q25):
        return "<=p25"
    if x <= float(q50):
        return "p25-50"
    if x <= float(q75):
        return "p50-75"
    return ">p75"


def imbalance_bucket(v: Optional[float]) -> str:
    if v is None:
        return "missing"
    x = float(v)
    ax = abs(x)
    if ax < 0.3:
        return "abs<0.3"
    if ax < 0.5:
        return ("+" if x > 0 else "-") + "[0.3,0.5)"
    if ax < 0.7:
        return ("+" if x > 0 else "-") + "[0.5,0.7)"
    if ax < 0.9:
        return ("+" if x > 0 else "-") + "[0.7,0.9)"
    return ("+" if x > 0 else "-") + ">=0.9"


def build_regime_tags_for_row(
    row: Dict[str, Optional[float]],
    edges: Dict[str, Tuple[Optional[float], Optional[float], Optional[float]]],
) -> Dict[str, str]:
    vol_val = row.get("micro_volatility")
    if vol_val is None:
        r1 = row.get("ret_1")
        vol_val = (None if r1 is None else abs(float(r1)))
    return {
        "regime_spread_bin": _bin_from_edges(row.get("spread"), edges.get("spread", (None, None, None))),
        "regime_intensity_bin": _bin_from_edges(row.get("trade_intensity"), edges.get("intensity", (None, None, None))),
        "regime_vol_bin": _bin_from_edges(vol_val, edges.get("vol", (None, None, None))),
        "regime_imb_bin": imbalance_bucket(row.get("imbalance")),
    }


def make_event_id(symbol: str, ts_bucket: Any, signal_idx: int) -> str:
    return f"{symbol}|{ts_bucket}|{int(signal_idx)}"


def debug_cap_info(debug_samples: int, debug_rows_written: int) -> Dict[str, Any]:
    limit = int(debug_samples)
    capped = limit > 0 and int(debug_rows_written) >= limit
    return {
        "debug_out_capped": bool(capped),
        "debug_samples_limit": int(limit),
        "debug_rows_written": int(debug_rows_written),
    }


def parse_feature_bound(raw: str) -> Tuple[str, float]:
    s = str(raw or "").strip()
    if "=" not in s:
        raise ValueError(f"invalid feature bound '{raw}', expected name=value")
    name, val = s.split("=", 1)
    name = name.strip()
    if not name:
        raise ValueError(f"invalid feature name in '{raw}'")
    return name, float(val.strip())


def load_passive_profiles(path: str) -> Dict[str, Any]:
    p = Path(str(path or "").strip())
    if not str(p):
        return {}
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def resolve_symbol_profile(profiles: Dict[str, Any], symbol: str) -> Dict[str, Any]:
    if not isinstance(profiles, dict):
        return {}
    out: Dict[str, Any] = {}
    default = profiles.get("default")
    if isinstance(default, dict):
        out.update(default)
    sym = profiles.get(str(symbol))
    if isinstance(sym, dict):
        for k, v in sym.items():
            if isinstance(v, dict) and isinstance(out.get(k), dict):
                merged = dict(out[k])
                merged.update(v)
                out[k] = merged
            else:
                out[k] = v
    return out


def event_passes_feature_bounds(
    feature: Dict[str, Any],
    min_bounds: Dict[str, float],
    max_bounds: Dict[str, float],
    counters: Dict[str, int],
) -> bool:
    def _read_feature(name: str) -> Any:
        if name in feature:
            return feature.get(name)
        if name.startswith("abs_"):
            base = name[4:]
            v = feature.get(base)
            if v is None:
                return None
            try:
                return abs(float(v))
            except Exception:
                return None
        return None
    for k, thr in (min_bounds or {}).items():
        v = _read_feature(k)
        if v is None:
            counters["filter_drop_missing_key"] = int(counters.get("filter_drop_missing_key", 0)) + 1
            return False
        try:
            xv = float(v)
        except Exception:
            counters["filter_drop_missing_key"] = int(counters.get("filter_drop_missing_key", 0)) + 1
            return False
        if xv < float(thr):
            counters["filter_drop_below_min"] = int(counters.get("filter_drop_below_min", 0)) + 1
            return False
    for k, thr in (max_bounds or {}).items():
        v = _read_feature(k)
        if v is None:
            counters["filter_drop_missing_key"] = int(counters.get("filter_drop_missing_key", 0)) + 1
            return False
        try:
            xv = float(v)
        except Exception:
            counters["filter_drop_missing_key"] = int(counters.get("filter_drop_missing_key", 0)) + 1
            return False
        if xv > float(thr):
            counters["filter_drop_above_max"] = int(counters.get("filter_drop_above_max", 0)) + 1
            return False
    return True


def _attempt_gate_reject_reason(r: Dict[str, Any], gate: Dict[str, float]) -> Optional[str]:
    """Return rejection reason for pre-attempt strength thresholds (or None if pass).

    Uses only features available at signal time (no lookahead). DAT-01 safe.
    gate keys: min_trade_intensity_strong, min_imbalance_strong, max_spread_tight, max_volatility_extreme
    """
    if not gate:
        return None
    ti = float(r.get("trade_intensity") or 0.0)
    imb = abs(float(r.get("imbalance") or 0.0))
    spr = float(r.get("spread") or 0.0)
    min_int = float(gate.get("min_trade_intensity_strong", 0.0))
    min_imb = float(gate.get("min_imbalance_strong", 0.0))
    max_spr = float(gate.get("max_spread_tight", 0.0))
    max_vol = float(gate.get("max_volatility_extreme", 0.0))
    vol = r.get("micro_volatility")
    if vol is None:
        vol = abs(float(r.get("ret_1") or 0.0))
    if min_int > 0.0 and ti < min_int:
        return "intensity_too_low"
    if min_imb > 0.0 and imb < min_imb:
        return "imbalance_too_low"
    if max_spr > 0.0 and spr > max_spr:
        return "spread_too_wide"
    if max_vol > 0.0 and float(vol or 0.0) > max_vol:
        return "vol_quantile_reject"
    return None


def _passes_attempt_gate(r: Dict[str, Any], gate: Dict[str, float]) -> bool:
    return _attempt_gate_reject_reason(r, gate) is None


def resolve_and_emit_debug(
    *,
    debug_record: Dict[str, Any],
    debug_out_path: Optional[Path],
    debug_out_long_path: Optional[Path],
    debug_out_short_path: Optional[Path],
    seen_event_ids: set[str],
    stats: Dict[str, int],
    log_flow: bool,
) -> None:
    event_id = str(debug_record.get("event_id") or "")
    if event_id in seen_event_ids:
        stats["debug_dedupe_drop"] = int(stats.get("debug_dedupe_drop", 0)) + 1
        if log_flow:
            print(
                f"[DEBUG_FLOW source=tools.micro_edge_backtest:resolve_and_emit_debug] "
                f"stage=sink_dedupe_drop event_id={event_id}"
            )
        return
    seen_event_ids.add(event_id)

    side = str(debug_record.get("resolved_side") or "").upper()
    target: Optional[Path] = None
    if side == "LONG" and debug_out_long_path is not None:
        target = debug_out_long_path
    elif side == "SHORT" and debug_out_short_path is not None:
        target = debug_out_short_path
    else:
        target = debug_out_path
    if target is None:
        return
    append_jsonl(target, debug_record)
    stats["debug_written"] = int(stats.get("debug_written", 0)) + 1
    if log_flow:
        print(
            f"[DEBUG_FLOW source=tools.micro_edge_backtest:resolve_and_emit_debug] "
            f"stage=sink_write event_id={event_id} resolved_side={side} target={target}"
        )


def _passive_touch_depth_proxy(
    *,
    side: str,
    entry_price: float,
    spread_ratio: float,
    future_mids: List[float],
) -> Tuple[bool, float, int]:
    s = str(side).upper()
    ep = float(entry_price)
    sp = max(1e-9, float(spread_ratio))
    if ep <= 0.0 or not future_mids:
        return False, 0.0, -1
    if s == "SHORT":
        limit = ep * (1.0 + 0.5 * sp)
        for i, m in enumerate(future_mids):
            px = float(m)
            if px >= limit:
                depth = (px - limit) / (ep * sp)
                return True, max(0.0, depth), i
        return False, 0.0, -1
    limit = ep * (1.0 - 0.5 * sp)
    for i, m in enumerate(future_mids):
        px = float(m)
        if px <= limit:
            depth = (limit - px) / (ep * sp)
            return True, max(0.0, depth), i
    return False, 0.0, -1


def build_passive_calibration_samples(
    *,
    rows: List[Dict[str, Optional[float]]],
    rule_name: str,
    side: str,
    thresholds: Dict[str, Optional[float]],
    hold_buckets: int,
    min_feature_bounds: Optional[Dict[str, float]] = None,
    max_feature_bounds: Optional[Dict[str, float]] = None,
    max_wait_buckets: int = 0,
) -> List[Dict[str, Any]]:
    mids = [r.get("mid") for r in rows]
    n = len(rows)
    wait_buckets = int(max_wait_buckets) if int(max_wait_buckets) > 0 else int(hold_buckets)
    samples: List[Dict[str, Any]] = []
    counters: Dict[str, int] = {"filter_drop_missing_key": 0, "filter_drop_below_min": 0, "filter_drop_above_max": 0}
    for i in range(n):
        r = rows[i]
        if not rule_fires(rule_name, r, thresholds):
            continue
        feature_snapshot = {
            "imbalance": r.get("imbalance"),
            "ret_1": r.get("ret_1"),
            "spread": r.get("spread"),
            "trade_intensity": r.get("trade_intensity"),
            "micro_volatility": r.get("micro_volatility"),
        }
        if not event_passes_feature_bounds(
            feature=feature_snapshot,
            min_bounds=min_feature_bounds or {},
            max_bounds=max_feature_bounds or {},
            counters=counters,
        ):
            continue
        entry_idx = i + 1
        if entry_idx >= n:
            continue
        spread_ratio = r.get("spread")
        if spread_ratio is None or float(spread_ratio) <= 0.0:
            continue
        entry_px = mids[entry_idx]
        if entry_px is None or float(entry_px) <= 0.0:
            continue
        trade_side = rule_predicted_side(rule_name, r, default_side=str(side).upper())
        if trade_side is None:
            continue
        max_end = min(n - 1, entry_idx + max(1, wait_buckets) + max(1, int(hold_buckets)))
        future_mids = [float(v) for v in mids[entry_idx : max_end + 1] if v is not None and float(v) > 0.0]
        if len(future_mids) < 2:
            continue
        touched, depth, touch_idx = _passive_touch_depth_proxy(
            side=trade_side,
            entry_price=float(entry_px),
            spread_ratio=float(spread_ratio),
            future_mids=future_mids[: max(1, wait_buckets)],
        )
        full_proxy = bool(touched and depth >= 0.5)
        adverse_bps = 0.0
        if touched:
            at = min(len(future_mids) - 1, max(0, touch_idx))
            nxt = min(len(future_mids) - 1, at + 1)
            p_touch = float(future_mids[at])
            p_next = float(future_mids[nxt])
            if p_touch > 0.0:
                if trade_side == "SHORT":
                    adverse_bps = max(0.0, (p_touch - p_next) / p_touch * 10000.0)
                else:
                    adverse_bps = max(0.0, (p_next - p_touch) / p_touch * 10000.0)
        samples.append(
            {
                "spread": float(spread_ratio),
                "trade_intensity": float(r.get("trade_intensity") or 0.0),
                "vol_proxy": float(r.get("micro_volatility") or abs(float(r.get("ret_1") or 0.0))),
                "imbalance_for_fill": abs(float(r.get("imbalance") or 0.0)),
                "touched": bool(touched),
                "full_proxy": bool(full_proxy),
                "adverse_bps": float(adverse_bps),
                "depth": float(depth),
            }
        )
    return samples


def summarize_trade_components(trades: List[Dict[str, Any]]) -> Dict[str, float]:
    n = len(trades)
    if n == 0:
        return {"avg_gross": 0.0, "avg_cost": 0.0, "avg_net": 0.0, "fill_rate": 0.0, "avg_fill_fraction": 0.0}
    gross = [float(t.get("raw_return", 0.0)) for t in trades]
    cost = [float(t.get("cost", 0.0)) for t in trades]
    net = [float(t.get("net_return", 0.0)) for t in trades]
    fill_fracs = [float(t.get("fill_fraction", 1.0) or 1.0) for t in trades]
    return {
        "avg_gross": sum(gross) / n,
        "avg_cost": sum(cost) / n,
        "avg_net": sum(net) / n,
        "fill_rate": sum(1 for t in trades if bool(t.get("filled", True))) / n,
        "avg_fill_fraction": sum(fill_fracs) / n,
    }


def summarize_cost_breakdown(trades: List[Dict[str, Any]]) -> Dict[str, float]:
    n = len(trades)
    if n == 0:
        return {
            "avg_fee_bps": 0.0,
            "avg_spread_bps": 0.0,
            "avg_adverse_bps": 0.0,
            "avg_total_bps": 0.0,
        }
    fee = [float(t.get("cost_fee_bps", 0.0)) for t in trades]
    spr = [float(t.get("cost_spread_bps", 0.0)) for t in trades]
    adv = [float(t.get("cost_adverse_bps", 0.0)) for t in trades]
    tot = [float(t.get("cost_total_bps", 0.0)) for t in trades]
    return {
        "avg_fee_bps": sum(fee) / n,
        "avg_spread_bps": sum(spr) / n,
        "avg_adverse_bps": sum(adv) / n,
        "avg_total_bps": sum(tot) / n,
    }


def compute_backtest_metrics(returns: List[float], hold_buckets: List[int], total_buckets: int) -> Dict[str, float]:
    n = len(returns)
    if n == 0:
        return {
            "n_trades": 0,
            "win_rate": 0.0,
            "avg_return": 0.0,
            "median_return": 0.0,
            "pnl_sum": 0.0,
            "max_drawdown": 0.0,
            "profit_factor": 0.0,
            "avg_hold_buckets": 0.0,
            "exposure_pct": 0.0,
        }
    wins = sum(1 for r in returns if r > 0)
    avg_ret = sum(returns) / n
    med_ret = float(median(returns))
    pnl_sum = float(sum(returns))
    equity = 1.0
    peak = 1.0
    max_dd = 0.0
    for r in returns:
        equity *= (1.0 + float(r))
        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd
    gross_profit = sum(r for r in returns if r > 0)
    gross_loss = abs(sum(r for r in returns if r < 0))
    if gross_loss > 0:
        pf = gross_profit / gross_loss
    elif gross_profit > 0:
        pf = float("inf")
    else:
        pf = 0.0
    avg_hold = sum(hold_buckets) / max(1, len(hold_buckets))
    exposure = sum(max(0, int(h)) for h in hold_buckets) / max(1, int(total_buckets))
    return {
        "n_trades": int(n),
        "win_rate": wins / n,
        "avg_return": avg_ret,
        "median_return": med_ret,
        "pnl_sum": pnl_sum,
        "max_drawdown": max_dd,
        "profit_factor": pf,
        "avg_hold_buckets": avg_hold,
        "exposure_pct": exposure,
    }


def simulate_rule_trades(
    rows: List[Dict[str, Optional[float]]],
    rule_name: str,
    side: str,
    thresholds: Dict[str, Optional[float]],
    labels: Optional[List[Optional[int]]],
    hold_buckets: int,
    cooldown_buckets: int,
    fee_bps: float,
    slip_bps: float,
    debug_one_trade: bool = False,
    debug_samples: int = 0,
    debug_symbol: str = "",
    debug_out_path: Optional[Path] = None,
    debug_out_long_path: Optional[Path] = None,
    debug_out_short_path: Optional[Path] = None,
    min_feature_bounds: Optional[Dict[str, float]] = None,
    max_feature_bounds: Optional[Dict[str, float]] = None,
    exec_model: str = "taker",
    maker_fee_bps: float = 0.5,
    maker_penalty_bps: float = 0.5,
    passive_params: Optional[Dict[str, Any]] = None,
    passive_max_wait_buckets: int = 0,
    toxicity_cfg: Optional[Dict[str, Any]] = None,
    regime_edges: Optional[Dict[str, Tuple[Optional[float], Optional[float], Optional[float]]]] = None,
    attempt_gate_bounds: Optional[Dict[str, float]] = None,
    regime_filter: str = "",
    bucket_sec: int = 1,
    scratch_bps: float = 0.0,
    scratch_window_sec: int = 0,
    scratch_taker_fee_bps: float = 0.0,
    scratch_slippage_bps: float = 0.0,
    debug_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    use_unified_engine = _exec_engine_unified_enabled()
    engines = build_default_engines() if use_unified_engine else {}
    passive_params_effective = dict(passive_params or {})
    if (
        "touch_without_cross_factor" not in passive_params_effective
        and float(passive_params_effective.get("base_touch", 0.0) or 0.0) >= 0.999
        and float(passive_params_effective.get("base_full_cond_touch", 0.0) or 0.0) >= 0.999
    ):
        passive_params_effective["touch_without_cross_factor"] = 1.0
    mids = [r.get("mid") for r in rows]
    n = len(rows)
    next_allowed = 0
    i = 0
    rets: List[float] = []
    holds: List[int] = []
    trades: List[Dict[str, Any]] = []
    attempt_rows: List[Dict[str, Any]] = []
    debug_printed = False
    debug_emitted = 0
    seen_event_ids: set[str] = set()
    debug_stats: Dict[str, int] = {
        "debug_written": 0,
        "debug_dedupe_drop": 0,
        "filter_drop_missing_key": 0,
        "filter_drop_below_min": 0,
        "filter_drop_above_max": 0,
        "cost_drop_missing_spread": 0,
        "passive_attempts": 0,
        "passive_filled": 0,
        "passive_unfilled": 0,
        "passive_partial": 0,
        "toxicity_blocked": 0,
        "toxicity_block_vol_intensity": 0,
        "attempt_gate_blocked": 0,
        "attempt_gate_block_intensity_too_low": 0,
        "attempt_gate_block_imbalance_too_low": 0,
        "attempt_gate_block_spread_too_wide": 0,
        "attempt_gate_block_vol_quantile_reject": 0,
        "attempt_gate_block_other": 0,
    }
    n_pre_gate_signals = 0
    debug_limit = int(debug_samples)
    is_capped = debug_limit > 0
    while i < n:
        if i < next_allowed:
            i += 1
            continue
        r = rows[i]
        if not rule_fires(rule_name, r, thresholds):
            i += 1
            continue
        feature_snapshot = {
            "imbalance": r.get("imbalance"),
            "ret_1": r.get("ret_1"),
            "spread": r.get("spread"),
            "trade_intensity": r.get("trade_intensity"),
            "volume": r.get("volume"),
            "micro_volatility": r.get("micro_volatility"),
            "liquidity_proxy": r.get("liquidity_proxy"),
            "v2_score": r.get("v2_score"),
            "v2_confidence": r.get("v2_confidence"),
            "v2_imbalance_persist": r.get("v2_imbalance_persist"),
            "v2_flip_rate": r.get("v2_flip_rate"),
            "v2_meanrev_prob": r.get("v2_meanrev_prob"),
            "v3_score": r.get("v3_score"),
            "v3_confidence": r.get("v3_confidence"),
            "v3_spread_shrink_ratio": r.get("v3_spread_shrink_ratio"),
            "v3_intensity_slope": r.get("v3_intensity_slope"),
            "v3_follow_agree": r.get("v3_follow_agree"),
        }
        if not event_passes_feature_bounds(
            feature=feature_snapshot,
            min_bounds=min_feature_bounds or {},
            max_bounds=max_feature_bounds or {},
            counters=debug_stats,
        ):
            i += 1
            continue
        if regime_filter:
            _rlabel = str(r.get("_regime_label") or "")
            if _rlabel != str(regime_filter).upper():
                i += 1
                continue
        entry_idx = i + 1
        exit_idx = entry_idx + max(1, int(hold_buckets))
        if exit_idx >= n:
            break
        entry_px = mids[entry_idx]
        exit_px = mids[exit_idx]
        if entry_px is None or exit_px is None or entry_px <= 0 or exit_px <= 0:
            i += 1
            continue
        event_id = make_event_id(debug_symbol, rows[i].get("ts_ms"), i)
        log_flow = bool(is_capped and debug_emitted < debug_limit)
        if log_flow:
            print(
                f"[DEBUG_FLOW source=tools.micro_edge_backtest:simulate_rule_trades] "
                f"stage=signal_gen event_id={event_id} rule={rule_name} "
                f"imbalance={rows[i].get('imbalance')} ret_1={rows[i].get('ret_1')} "
                f"spread={rows[i].get('spread')} trade_intensity={rows[i].get('trade_intensity')}"
            )
        trade_side = rule_predicted_side(rule_name, r, default_side=str(side).upper())
        if trade_side is None:
            i += 1
            continue
        n_pre_gate_signals += 1
        reject_reason = _attempt_gate_reject_reason(r, attempt_gate_bounds or {})
        if attempt_gate_bounds and reject_reason is not None:
            debug_stats["attempt_gate_blocked"] += 1
            key = f"attempt_gate_block_{reject_reason}"
            if key in debug_stats:
                debug_stats[key] = int(debug_stats.get(key, 0)) + 1
            else:
                debug_stats["attempt_gate_block_other"] = int(debug_stats.get("attempt_gate_block_other", 0)) + 1
            i += 1
            continue
        # Attempt-level row starts here (post-signal, post-feature-filter, side resolved)
        attempted = {
            "signal_idx": int(i),
            "filled": False,
            "fill_fraction": 0.0,
            "net_return": 0.0,
            "adverse_selection_bps": 0.0,
        }
        if str(exec_model).lower() == "passive_realistic":
            tcfg = toxicity_cfg or {}
            enable_tox = bool(tcfg.get("enabled", True))
            vol_thr = float(tcfg.get("vol_high_threshold", 0.0))
            int_thr = float(tcfg.get("intensity_high_threshold", 0.0))
            imb_thr = float(tcfg.get("imbalance_min_threshold", 0.3))
            vol_val = r.get("micro_volatility")
            if vol_val is None:
                vol_val = abs(float(r.get("ret_1") or 0.0))
            int_val = float(r.get("trade_intensity") or 0.0)
            imb_val = abs(float(r.get("imbalance") or 0.0))
            if enable_tox and (vol_thr > 0.0 and int_thr > 0.0) and (float(vol_val or 0.0) >= vol_thr) and (int_val >= int_thr) and (imb_val < imb_thr):
                debug_stats["toxicity_blocked"] = int(debug_stats.get("toxicity_blocked", 0)) + 1
                debug_stats["toxicity_block_vol_intensity"] = int(debug_stats.get("toxicity_block_vol_intensity", 0)) + 1
                attempted["blocked_reason"] = "toxicity:vol_high+intensity_high+imbalance_low"
                if (not is_capped) or (debug_emitted < debug_limit):
                    dbg_block = {
                        "ts_utc": utc_now_iso(),
                        "symbol": debug_symbol,
                        "event_id": event_id,
                        "ts_bucket": rows[i].get("ts_ms"),
                        "signal_idx": i,
                        "rule_name": rule_name,
                        "resolved_side": trade_side,
                        "event": "passive_toxicity_block",
                        "blocked_reason": attempted["blocked_reason"],
                        "feature": feature_snapshot,
                        "timing": "signal at t blocked pre-post",
                        "source": "tools.micro_edge_backtest:toxicity_gate",
                    }
                    if isinstance(debug_meta, dict):
                        dbg_block.update(debug_meta)
                    resolve_and_emit_debug(
                        debug_record=dbg_block,
                        debug_out_path=debug_out_path,
                        debug_out_long_path=debug_out_long_path,
                        debug_out_short_path=debug_out_short_path,
                        seen_event_ids=seen_event_ids,
                        stats=debug_stats,
                        log_flow=log_flow,
                    )
                    debug_emitted += 1
                attempt_rows.append(attempted)
                i += 1
                continue
        fill_fraction = 1.0
        filled_flag = True
        adverse_bps = 0.0
        effective_cost_bps = 0.0
        exec_price_adj = 0.0
        fill_offset = 0
        cost_fee_ratio = 0.0
        cost_spread_ratio = 0.0
        cost_adverse_ratio = 0.0
        cost_total_ratio = 0.0
        scratch_triggered = False
        scratch_extra_cost_ratio = 0.0
        scratch_exit_idx = None
        queue_competition_score = 0.0
        toxicity_score = 0.0
        exec_mode = str(exec_model).lower()
        if exec_mode in {"passive_realistic", "passive_then_taker"}:
            debug_stats["passive_attempts"] = int(debug_stats.get("passive_attempts", 0)) + 1
            wait_buckets = int(passive_max_wait_buckets) if int(passive_max_wait_buckets) > 0 else int(hold_buckets)
            max_end = min(n - 1, entry_idx + max(1, wait_buckets) + max(1, int(hold_buckets)))
            future_mids = [float(v) for v in mids[entry_idx : max_end + 1] if v is not None and float(v) > 0.0]
            future_mids_for_sim = list(future_mids)
            optimistic_no_cross_touch = float(passive_params_effective.get("touch_without_cross_factor", 0.0) or 0.0) >= 0.999
            spread_ratio = float(r.get("spread") or 0.0)
            if optimistic_no_cross_touch and future_mids_for_sim and spread_ratio > 0.0:
                synthetic_touch_px = float(entry_px) * (1.0 - (0.5 * spread_ratio) if trade_side == "LONG" else 1.0 + (0.5 * spread_ratio))
                future_mids_for_sim = [synthetic_touch_px, *future_mids_for_sim]
            pfill = simulate_passive_fill(
                event={
                    "event_id": event_id,
                    "symbol": debug_symbol,
                    "side": trade_side,
                    "entry_price": float(entry_px),
                    "future_mids": future_mids_for_sim,
                },
                horizon_sec=max(1, int(hold_buckets)),
                features={
                    "spread": r.get("spread"),
                    "trade_intensity": r.get("trade_intensity"),
                    "vol_proxy": (r.get("micro_volatility") if r.get("micro_volatility") is not None else abs(float(r.get("ret_1") or 0.0))),
                    "imbalance_for_fill": abs(float(r.get("imbalance") or 0.0)),
                },
                params=passive_params_effective,
            )
            filled_flag = bool(pfill.get("filled"))
            if not filled_flag:
                if exec_mode == "passive_then_taker":
                    debug_stats["passive_fallback_taker"] = int(debug_stats.get("passive_fallback_taker", 0)) + 1
                    fallback_offset = max(1, int(passive_max_wait_buckets) if int(passive_max_wait_buckets) > 0 else 1)
                    entry_idx = entry_idx + fallback_offset
                    exit_idx = entry_idx + max(1, int(hold_buckets))
                    if exit_idx >= n:
                        break
                    entry_px = mids[entry_idx]
                    exit_px = mids[exit_idx]
                    if entry_px is None or exit_px is None or entry_px <= 0 or exit_px <= 0:
                        i += 1
                        continue
                    fallback_one_way_bps = max(0.0, float(scratch_taker_fee_bps)) + max(0.0, float(scratch_slippage_bps))
                    cost_fee_ratio = (2.0 * fallback_one_way_bps) / 10000.0
                    cost_adverse_ratio = 0.0
                    cost_spread_ratio = 0.0
                    cost_total_ratio = cost_fee_ratio
                    event_cost = cost_total_ratio
                    effective_cost_bps = 2.0 * fallback_one_way_bps
                    adverse_bps = 0.0
                    exec_price_adj = 0.0
                    fill_fraction = 1.0
                    filled_flag = True
                else:
                    debug_stats["passive_unfilled"] = int(debug_stats.get("passive_unfilled", 0)) + 1
                    attempt_rows.append(attempted)
                    i += 1
                    continue
            else:
                debug_stats["passive_filled"] = int(debug_stats.get("passive_filled", 0)) + 1
                fill_fraction = float(pfill.get("fill_fraction", 1.0) or 1.0)
                if fill_fraction < 0.999:
                    debug_stats["passive_partial"] = int(debug_stats.get("passive_partial", 0)) + 1
                adverse_bps = float(pfill.get("adverse_selection_bps", 0.0) or 0.0)
                effective_cost_bps = float(pfill.get("effective_cost_bps", 0.0) or 0.0)
                exec_price_adj = float(pfill.get("execution_price_adjustment", 0.0) or 0.0)
                queue_competition_score = float(pfill.get("queue_competition_score", 0.0) or 0.0)
                toxicity_score = float(pfill.get("toxicity_score", 0.0) or 0.0)
                fill_offset = max(0, int(pfill.get("fill_index_offset", 0) or 0))
                entry_idx = entry_idx + fill_offset
                exit_idx = entry_idx + max(1, int(hold_buckets))
                if exit_idx >= n:
                    break
                entry_px = mids[entry_idx]
                exit_px = mids[exit_idx]
                if entry_px is None or exit_px is None or entry_px <= 0 or exit_px <= 0:
                    i += 1
                    continue
                cost_fee_ratio = (2.0 * float(passive_params_effective.get("maker_fee_bps", maker_fee_bps))) / 10000.0
                cost_adverse_ratio = max(0.0, float(adverse_bps) / 10000.0)
                # Spread is modeled via execution price adjustment (price improvement/slippage path), not direct additive cost.
                cost_spread_ratio = 0.0
                cost_total_ratio = cost_fee_ratio + cost_spread_ratio + cost_adverse_ratio
                event_cost = cost_total_ratio
        else:
            event_cost = compute_exec_cost(
                exec_model=exec_model,
                fee_bps=float(fee_bps),
                slip_bps=float(slip_bps),
                maker_fee_bps=float(maker_fee_bps),
                maker_penalty_bps=float(maker_penalty_bps),
                spread_ratio=(None if r.get("spread") is None else float(r.get("spread"))),
            )
            if event_cost is None:
                debug_stats["cost_drop_missing_spread"] = int(debug_stats.get("cost_drop_missing_spread", 0)) + 1
                i += 1
                continue
            cost_total_ratio = float(event_cost)
        if log_flow:
            print(
                f"[DEBUG_FLOW source=tools.micro_edge_backtest:simulate_rule_trades] "
                f"stage=resolve_side event_id={event_id} resolved_side={trade_side}"
            )
        predicted_sign = 1 if trade_side == "LONG" else -1
        label_used = labels[i] if labels is not None and i < len(labels) else None
        direction_match = None
        if label_used in (-1, 1):
            direction_match = int(label_used) == int(predicted_sign)
        exec_entry_px = float(entry_px) * (1.0 + float(exec_price_adj))
        # Optional post-fill scratch/escape:
        # if adverse move exceeds threshold within scratch window, force early taker-like exit.
        if (
            str(exec_model).lower() in {"passive_realistic", "passive_then_taker"}
            and float(scratch_bps) > 0.0
            and int(scratch_window_sec) > 0
            and float(exec_entry_px) > 0.0
        ):
            sw_buckets = max(1, int(round(float(scratch_window_sec) / max(1, int(bucket_sec)))))
            sw_end = min(int(exit_idx), int(entry_idx) + sw_buckets)
            trigger_px_long = float(exec_entry_px) * (1.0 - float(scratch_bps) / 10000.0)
            trigger_px_short = float(exec_entry_px) * (1.0 + float(scratch_bps) / 10000.0)
            for j in range(int(entry_idx) + 1, int(sw_end) + 1):
                px_j = mids[j]
                if px_j is None or float(px_j) <= 0.0:
                    continue
                if trade_side == "LONG" and float(px_j) <= trigger_px_long:
                    scratch_triggered = True
                    scratch_exit_idx = int(j)
                    break
                if trade_side == "SHORT" and float(px_j) >= trigger_px_short:
                    scratch_triggered = True
                    scratch_exit_idx = int(j)
                    break
            if scratch_triggered and scratch_exit_idx is not None:
                exit_idx = int(scratch_exit_idx)
                exit_px = mids[exit_idx]
                if exit_px is None or float(exit_px) <= 0.0:
                    i += 1
                    continue
                scratch_extra_cost_ratio = (
                    max(0.0, float(scratch_taker_fee_bps)) + max(0.0, float(scratch_slippage_bps))
                ) / 10000.0
                event_cost = float(event_cost) + scratch_extra_cost_ratio
                cost_total_ratio = float(cost_total_ratio) + scratch_extra_cost_ratio
        raw_ret = compute_gross_return(float(exec_entry_px), float(exit_px), side=trade_side)
        if use_unified_engine:
            req = ExecutionRequest(
                symbol=str(debug_symbol),
                side=("buy" if trade_side == "LONG" else "sell"),
                entry_price=float(exec_entry_px),
                exit_price=float(exit_px),
                notional=1.0,
                fee_bps=0.0,
                slippage_bps=max(0.0, float(event_cost) * 10000.0),
                ts_ms=int(rows[i].get("ts_ms") or 0),
                order_id=str(event_id),
            )
            eres = engines["backtest"].execute(req)
            raw_ret = float(eres.gross_return)
            net_ret = float(eres.net_return)
        else:
            net_ret = raw_ret - float(event_cost)
        hold = int(exit_idx - entry_idx)
        rets.append(net_ret)
        holds.append(hold)
        trades.append(
            {
                "signal_idx": i,
                "entry_idx": entry_idx,
                "exit_idx": exit_idx,
                "entry_px": float(exec_entry_px),
                "exit_px": float(exit_px),
                "side": trade_side,
                "predicted_sign": predicted_sign,
                "label_used": label_used,
                "label_used_text": label_value_to_text(label_used),
                "direction_match": direction_match,
                "raw_return": raw_ret,
                "cost": float(event_cost),
                "net_return": net_ret,
                "hold_buckets": hold,
                "filled": bool(filled_flag),
                "fill_fraction": float(fill_fraction),
                "effective_cost_bps": float(effective_cost_bps),
                "adverse_selection_bps": float(adverse_bps),
                "execution_price_adjustment": float(exec_price_adj),
                "cost_fee_ratio": float(cost_fee_ratio),
                "cost_spread_ratio": float(cost_spread_ratio),
                "cost_adverse_ratio": float(cost_adverse_ratio),
                "cost_total_ratio": float(cost_total_ratio),
                "cost_fee_bps": float(cost_fee_ratio * 10000.0),
                "cost_spread_bps": float(cost_spread_ratio * 10000.0),
                "cost_adverse_bps": float(cost_adverse_ratio * 10000.0),
                "cost_total_bps": float(cost_total_ratio * 10000.0),
                "queue_competition_score": float(queue_competition_score),
                "toxicity_score": float(toxicity_score),
                "scratch_triggered": bool(scratch_triggered),
                "scratch_exit_idx": (int(scratch_exit_idx) if scratch_exit_idx is not None else None),
                "scratch_extra_cost_ratio": float(scratch_extra_cost_ratio),
                "scratch_extra_cost_bps": float(scratch_extra_cost_ratio * 10000.0),
            }
        )
        attempted["filled"] = bool(filled_flag)
        attempted["fill_fraction"] = float(fill_fraction)
        attempted["net_return"] = float(net_ret)
        attempted["adverse_selection_bps"] = float(adverse_bps)
        attempt_rows.append(attempted)
        if (not is_capped) or (debug_emitted < debug_limit):
            regime_tags = build_regime_tags_for_row(r, regime_edges or {})
            dbg = {
                "ts_utc": utc_now_iso(),
                "symbol": debug_symbol,
                "event_id": event_id,
                "ts_bucket": rows[i].get("ts_ms"),
                "signal_idx": i,
                "entry_idx": entry_idx,
                "exit_idx": exit_idx,
                "rule_name": rule_name,
                "feature": feature_snapshot,
                "resolved_side": trade_side,
                "score": (r.get("v3_score") if str(rule_name) == "micro_edge_v3_passive_alpha" else r.get("v2_score")),
                "confidence": (r.get("v3_confidence") if str(rule_name) == "micro_edge_v3_passive_alpha" else r.get("v2_confidence")),
                "entry_price": float(exec_entry_px),
                "exit_price": float(exit_px),
                "gross_ret": raw_ret,
                "cost": float(event_cost),
                "net_ret": net_ret,
                "filled_flag": bool(filled_flag),
                "fill_fraction": float(fill_fraction),
                "effective_cost_bps": float(effective_cost_bps),
                "adverse_selection_bps": float(adverse_bps),
                "execution_price_adjustment": float(exec_price_adj),
                "cost_fee_ratio": float(cost_fee_ratio),
                "cost_spread_ratio": float(cost_spread_ratio),
                "cost_adverse_ratio": float(cost_adverse_ratio),
                "cost_total_ratio": float(cost_total_ratio),
                "cost_fee_bps": float(cost_fee_ratio * 10000.0),
                "cost_spread_bps": float(cost_spread_ratio * 10000.0),
                "cost_adverse_bps": float(cost_adverse_ratio * 10000.0),
                "cost_total_bps": float(cost_total_ratio * 10000.0),
                "queue_competition_score": float(queue_competition_score),
                "toxicity_score": float(toxicity_score),
                "scratch_triggered": bool(scratch_triggered),
                "scratch_exit_idx": (int(scratch_exit_idx) if scratch_exit_idx is not None else None),
                "scratch_extra_cost_ratio": float(scratch_extra_cost_ratio),
                "scratch_extra_cost_bps": float(scratch_extra_cost_ratio * 10000.0),
                "label_used": label_used,
                "label_used_text": label_value_to_text(label_used),
                "direction_match": direction_match,
                "timing": "signal at t, entry at t+1 mark, exit at t+1+h mark",
                "source": "tools.micro_edge_backtest:simulate_rule_trades",
                **regime_tags,
            }
            if isinstance(debug_meta, dict):
                dbg.update(debug_meta)
            resolve_and_emit_debug(
                debug_record=dbg,
                debug_out_path=debug_out_path,
                debug_out_long_path=debug_out_long_path,
                debug_out_short_path=debug_out_short_path,
                seen_event_ids=seen_event_ids,
                stats=debug_stats,
                log_flow=log_flow,
            )
            debug_emitted += 1
        if debug_one_trade and not debug_printed:
            print(
                f"[DEBUG_TRADE] signal_idx={i} entry_idx={entry_idx} exit_idx={exit_idx} side={trade_side} "
                f"entry_price={float(entry_px):.8f} exit_price={float(exit_px):.8f} "
                f"gross_ret={raw_ret:+.8f} cost={float(event_cost):.8f} net_ret={net_ret:+.8f}"
            )
            debug_printed = True
        next_allowed = exit_idx + int(max(0, cooldown_buckets))
        i = max(i + 1, next_allowed)
    metrics = compute_backtest_metrics(rets, holds, total_buckets=n)
    filled_n = len(trades)
    filled_only = {
        "n": int(filled_n),
        "win_rate": (sum(1 for t in trades if float(t.get("net_return", 0.0)) > 0.0) / filled_n) if filled_n > 0 else 0.0,
        "avg_net": (sum(float(t.get("net_return", 0.0)) for t in trades) / filled_n) if filled_n > 0 else 0.0,
        "p90_net": (sorted(float(t.get("net_return", 0.0)) for t in trades)[int((filled_n - 1) * 0.9)] if filled_n > 0 else 0.0),
    }
    attempts_n = len(attempt_rows)
    filled_attempts = sum(1 for a in attempt_rows if bool(a.get("filled")))
    partial_attempts = sum(1 for a in attempt_rows if bool(a.get("filled")) and float(a.get("fill_fraction", 0.0)) < 0.999)
    attempt_level = {
        "n_attempts": int(attempts_n),
        "net_per_attempt": (sum(float(a.get("net_return", 0.0)) for a in attempt_rows) / attempts_n) if attempts_n > 0 else 0.0,
        "fill_rate": (filled_attempts / attempts_n) if attempts_n > 0 else 0.0,
        "partial_rate": (partial_attempts / attempts_n) if attempts_n > 0 else 0.0,
        "n_signals_before_gate": int(n_pre_gate_signals),
    }
    ff = [float(t.get("fill_fraction", 0.0)) for t in trades]
    nn = [float(t.get("net_return", 0.0)) for t in trades]
    adv = [float(t.get("adverse_selection_bps", 0.0)) for t in trades]
    correlations = {
        "fill_fraction_vs_net": _pearson_corr(ff, nn),
        "adverse_selection_bps_vs_net": _pearson_corr(adv, nn),
    }
    return {
        "metrics": metrics,
        "trades": trades,
        "attempt_rows": attempt_rows,
        "debug_stats": debug_stats,
        "filled_only_metrics": filled_only,
        "attempt_level_metrics": attempt_level,
        "trade_correlations": correlations,
    }


def _pick_best_rule(rules: Dict[str, Dict[str, Any]], min_rule_n: int) -> Optional[str]:
    rec = {"naive_rules": rules}
    best = extract_best_rule_delta_min_n(rec, min_rule_n=min_rule_n)
    if best is None:
        return None
    best_name = None
    best_val = None
    for k, v in rules.items():
        try:
            n = int(v.get("n", 0) or 0)
        except Exception:
            n = 0
        if n < min_rule_n:
            continue
        d = v.get("delta_vs_baseline")
        if d is None:
            continue
        dv = float(d)
        if best_val is None or dv > best_val:
            best_val = dv
            best_name = k
    return best_name


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Micro-edge paper simulation.")
    p.add_argument("--db", default="data/microstructure.db")
    p.add_argument("--symbols", default="BTCUSDT,ETHUSDT")
    p.add_argument("--lookback-min", type=int, default=240)
    p.add_argument("--bucket-sec", type=int, default=5)
    p.add_argument("--horizon-sec", type=int, default=60)
    p.add_argument("--rule", default="best")
    p.add_argument("--fee-bps", type=float, default=4.0)
    p.add_argument("--slip-bps", type=float, default=2.0)
    p.add_argument("--side", default="auto")
    p.add_argument("--max-hold-buckets", type=int, default=0)
    p.add_argument("--cooldown-buckets", type=int, default=0)
    p.add_argument("--min-rule-n", type=int, default=100)
    p.add_argument("--debug-one-trade", action="store_true")
    p.add_argument("--debug-samples", type=int, default=0, help="0 means no cap; otherwise cap debug rows written.")
    p.add_argument("--debug-out", default="logs/micro_edge_debug_trades.jsonl")
    p.add_argument("--debug-out-long", default="")
    p.add_argument("--debug-out-short", default="")
    p.add_argument("--debug-append", dest="debug_append", action="store_true", default=True, help="Append debug rows (default).")
    p.add_argument("--no-debug-append", dest="debug_append", action="store_false", help="Truncate debug output files before run.")
    p.add_argument("--min-feature", action="append", default=[], help="Repeatable: name=value gate, event feature must be >= value.")
    p.add_argument("--max-feature", action="append", default=[], help="Repeatable: name=value gate, event feature must be <= value.")
    p.add_argument("--exec-model", choices=["taker", "maker", "mid", "halfspread", "passive_realistic"], default="taker")
    p.add_argument("--maker-fee-bps", type=float, default=float(DEFAULT_MAKER_FEE_BPS))
    p.add_argument("--maker-penalty-bps", type=float, default=0.5)
    p.add_argument("--passive-seed", type=int, default=42)
    p.add_argument("--passive-max-wait-buckets", type=int, default=0, help="0 uses hold_buckets.")
    p.add_argument("--passive-adverse-mult", type=float, default=1.0)
    p.add_argument("--passive-latency-enabled", action="store_true", default=False, help="Enable latency-aware passive fill model.")
    p.add_argument("--passive-latency-decision-ms", type=float, default=0.0, help="Decision->ack latency mean (ms).")
    p.add_argument("--passive-latency-queue-ms", type=float, default=0.0, help="Ack->queue-entry latency mean (ms).")
    p.add_argument("--passive-latency-feed-ms", type=float, default=0.0, help="Feed lag latency mean (ms).")
    p.add_argument("--passive-latency-jitter-ms", type=float, default=0.0, help="+/- jitter applied to each latency component (ms).")
    p.add_argument("--passive-latency-touch-penalty-per-bar", type=float, default=0.06, help="Touch probability penalty per latency bar.")
    p.add_argument("--passive-latency-adverse-bps-per-sec", type=float, default=0.0, help="Additional adverse bps per second of total latency.")
    p.add_argument("--scratch-bps", type=float, default=0.0, help="Post-fill adverse move threshold (bps) for early scratch exit; 0 disables.")
    p.add_argument("--scratch-window-sec", type=int, default=0, help="Post-fill scratch window in seconds; 0 disables.")
    p.add_argument("--scratch-taker-fee-bps", type=float, default=0.0, help="Extra one-way taker fee bps applied when scratch exit triggers.")
    p.add_argument("--scratch-slippage-bps", type=float, default=0.0, help="Extra one-way slippage bps applied when scratch exit triggers.")
    p.add_argument("--v2-min-score", type=float, default=0.0, help="Optional override threshold for |v2_score|.")
    p.add_argument("--v2-min-persistence", type=float, default=0.0, help="Optional override threshold for |v2_imbalance_persist|.")
    p.add_argument("--v2-min-confidence", type=float, default=0.0, help="Optional override threshold for v2_confidence.")
    p.add_argument("--passive-profile-in", default="state/passive_realistic_profiles.json")
    p.add_argument("--out", default="logs/micro_edge_backtest.jsonl")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    symbols = _parse_symbols(args.symbols)
    out_path = Path(str(args.out))
    debug_out_path = Path(str(args.debug_out))
    debug_out_long_path = Path(str(args.debug_out_long)) if str(args.debug_out_long).strip() else None
    debug_out_short_path = Path(str(args.debug_out_short)) if str(args.debug_out_short).strip() else None
    if not bool(args.debug_append):
        for pth in [debug_out_path, debug_out_long_path, debug_out_short_path]:
            if pth is None:
                continue
            try:
                pth.unlink(missing_ok=True)
            except Exception:
                pass
    min_bounds: Dict[str, float] = {}
    max_bounds: Dict[str, float] = {}
    for raw in list(args.min_feature or []):
        k, v = parse_feature_bound(raw)
        min_bounds[k] = v
    for raw in list(args.max_feature or []):
        k, v = parse_feature_bound(raw)
        max_bounds[k] = v
    try:
        conn = sqlite3.connect(str(args.db), check_same_thread=False)
    except Exception as exc:
        print(f"micro_edge_backtest: unable to open db={args.db} err={exc}")
        return 0
    try:
        print(
            f"micro_edge_backtest db={args.db} symbols={symbols} lookback_min={args.lookback_min} "
            f"bucket_sec={args.bucket_sec} horizon_sec={args.horizon_sec} exec_model={args.exec_model}"
        )
        passive_profiles = load_passive_profiles(str(args.passive_profile_in))
        for sym in symbols:
            now_ms = int(time.time() * 1000)
            start_ms = now_ms - int(max(1, args.lookback_min) * 60 * 1000)
            trades, marks = _load_symbol_trades_and_marks(conn, sym, start_ms=start_ms, end_ms=now_ms)
            rows = build_bucket_features(trades, marks, bucket_sec=int(args.bucket_sec), vol_window=max(4, int(60 / max(1, args.bucket_sec))))
            rows = enrich_rows_with_v2(
                rows,
                bucket_sec=int(args.bucket_sec),
                cache_key=(str(args.db), str(sym), int(args.lookback_min), int(args.bucket_sec), str(args.rule)),
            )
            regime_edges = compute_regime_bins(rows)
            print(
                f"[{sym}] regime_bin_edges "
                f"spread={regime_edges.get('spread')} intensity={regime_edges.get('intensity')} vol={regime_edges.get('vol')}"
            )
            mids = [r.get("mid") for r in rows]
            horizon_steps = max(1, int(round(float(args.horizon_sec) / max(1, args.bucket_sec))))
            fwd, labels = signal_aligned_labels(mids, horizon_steps=horizon_steps, threshold=0.0002)
            lbl_valid = [int(x) for x in labels if x is not None]
            baseline = None
            if lbl_valid:
                nz = [x for x in lbl_valid if x != 0]
                if nz:
                    up = sum(1 for x in nz if x > 0)
                    baseline = max(up, len(nz) - up) / len(nz)
            rules_all = evaluate_naive_rules(rows, labels, baseline_hit_rate=baseline)
            rules_filtered = filter_rules_min_n(rules_all, int(args.min_rule_n))
            rule_name = str(args.rule)
            if rule_name.lower() == "best":
                rule_name = _pick_best_rule(rules_all, int(args.min_rule_n)) or ""
            if not rule_name:
                print(f"[{sym}] no eligible rule (min_rule_n={args.min_rule_n})")
                continue
            side = str(args.side).upper()
            if side == "AUTO":
                side = infer_rule_side(rule_name)
            hold = int(args.max_hold_buckets) if int(args.max_hold_buckets) > 0 else horizon_steps
            thresholds = compute_rule_thresholds(rows)
            if float(args.v2_min_score) > 0.0:
                thresholds["v2_min_score"] = float(args.v2_min_score)
            if float(args.v2_min_persistence) > 0.0:
                thresholds["v2_min_persistence"] = float(args.v2_min_persistence)
            if float(args.v2_min_confidence) > 0.0:
                thresholds["v2_min_confidence"] = float(args.v2_min_confidence)
            passive_params: Optional[Dict[str, Any]] = None
            if str(args.exec_model).lower() == "passive_realistic":
                sym_profile = resolve_symbol_profile(passive_profiles, sym)
                samples = build_passive_calibration_samples(
                    rows=rows,
                    rule_name=rule_name,
                    side=side,
                    thresholds=thresholds,
                    hold_buckets=hold,
                    min_feature_bounds=min_bounds,
                    max_feature_bounds=max_bounds,
                    max_wait_buckets=int(args.passive_max_wait_buckets),
                )
                passive_params = calibrate_passive_model(
                    samples,
                    maker_fee_bps=float(args.maker_fee_bps),
                    seed=int(args.passive_seed),
                )
                passive_over = sym_profile.get("passive", {}) if isinstance(sym_profile.get("passive", {}), dict) else {}
                passive_params.update(passive_over)
                passive_params["passive_adverse_mult"] = float(args.passive_adverse_mult)
                passive_params["latency_enabled"] = bool(args.passive_latency_enabled)
                passive_params["latency_decision_to_ack_ms"] = float(args.passive_latency_decision_ms)
                passive_params["latency_queue_entry_ms"] = float(args.passive_latency_queue_ms)
                passive_params["latency_feed_lag_ms"] = float(args.passive_latency_feed_ms)
                passive_params["latency_decision_to_ack_jitter_ms"] = float(args.passive_latency_jitter_ms)
                passive_params["latency_queue_entry_jitter_ms"] = float(args.passive_latency_jitter_ms)
                passive_params["latency_feed_lag_jitter_ms"] = float(args.passive_latency_jitter_ms)
                passive_params["latency_bucket_sec"] = float(args.bucket_sec)
                passive_params["latency_touch_penalty_per_bar"] = float(args.passive_latency_touch_penalty_per_bar)
                passive_params["latency_adverse_bps_per_sec"] = float(args.passive_latency_adverse_bps_per_sec)
                print(
                    f"[{sym}] passive_calibration samples={len(samples)} "
                    f"base_touch={float(passive_params.get('base_touch', 0.0)):.3f} "
                    f"base_full={float(passive_params.get('base_full_cond_touch', 0.0)):.3f} "
                    f"base_adverse_bps={float(passive_params.get('base_adverse_bps', 0.0)):.3f} "
                    f"latency_enabled={int(bool(passive_params.get('latency_enabled', False)))}"
                )
                tox_cfg = sym_profile.get("toxicity_gate", {}) if isinstance(sym_profile.get("toxicity_gate", {}), dict) else {}
                if "vol_high_threshold" not in tox_cfg:
                    tox_cfg["vol_high_threshold"] = float(regime_edges.get("vol", (None, None, 0.0))[2] or 0.0)
                if "intensity_high_threshold" not in tox_cfg:
                    tox_cfg["intensity_high_threshold"] = float(regime_edges.get("intensity", (None, None, 0.0))[2] or 0.0)
                tox_cfg.setdefault("imbalance_min_threshold", 0.3)
                tox_cfg.setdefault("enabled", True)
            else:
                tox_cfg = {}
            sim = simulate_rule_trades(
                rows=rows,
                rule_name=rule_name,
                side=side,
                thresholds=thresholds,
                labels=labels,
                hold_buckets=hold,
                cooldown_buckets=int(args.cooldown_buckets),
                fee_bps=float(args.fee_bps),
                slip_bps=float(args.slip_bps),
                debug_one_trade=bool(args.debug_one_trade),
                debug_samples=int(args.debug_samples),
                debug_symbol=sym,
                debug_out_path=debug_out_path,
                debug_out_long_path=debug_out_long_path,
                debug_out_short_path=debug_out_short_path,
                min_feature_bounds=min_bounds,
                max_feature_bounds=max_bounds,
                exec_model=str(args.exec_model),
                maker_fee_bps=float(args.maker_fee_bps),
                maker_penalty_bps=float(args.maker_penalty_bps),
                passive_params=passive_params,
                passive_max_wait_buckets=int(args.passive_max_wait_buckets),
                toxicity_cfg=tox_cfg,
                regime_edges=regime_edges,
                bucket_sec=int(args.bucket_sec),
                scratch_bps=float(args.scratch_bps),
                scratch_window_sec=int(args.scratch_window_sec),
                scratch_taker_fee_bps=float(args.scratch_taker_fee_bps),
                scratch_slippage_bps=float(args.scratch_slippage_bps),
                debug_meta={
                    "exec_model": str(args.exec_model),
                    "horizon_sec": int(args.horizon_sec),
                    "bucket_sec": int(args.bucket_sec),
                },
            )
            m = sim["metrics"]
            sim_trades = list(sim.get("trades", []))
            comp = summarize_trade_components(sim_trades)
            cost_breakdown = summarize_cost_breakdown(sim_trades)
            filled_only = sim.get("filled_only_metrics", {}) if isinstance(sim, dict) else {}
            attempt_level = sim.get("attempt_level_metrics", {}) if isinstance(sim, dict) else {}
            corrs = sim.get("trade_correlations", {}) if isinstance(sim, dict) else {}
            ds = sim.get("debug_stats", {}) if isinstance(sim, dict) else {}
            attempts = int(ds.get("passive_attempts", 0) or 0)
            filled = int(ds.get("passive_filled", 0) or 0)
            partial = int(ds.get("passive_partial", 0) or 0)
            print(
                f"[{sym}] rule={rule_name} side={side} n_trades={m['n_trades']} "
                f"win_rate={m['win_rate']:.2%} avg_ret={m['avg_return']:+.4f} "
                f"avg_gross={comp['avg_gross']:+.6f} avg_cost={comp['avg_cost']:+.6f} "
                f"pnl_sum={m['pnl_sum']:+.4f} max_dd={m['max_drawdown']:.2%} pf={m['profit_factor']:.3f}"
            )
            if str(args.exec_model).lower() == "passive_realistic":
                attempt_fill_rate = (filled / attempts) if attempts > 0 else 0.0
                print(
                    f"[{sym}] passive_fill attempt_fill_rate={attempt_fill_rate:.2%} trade_fill_rate={comp['fill_rate']:.2%} "
                    f"avg_fill_fraction={comp['avg_fill_fraction']:.3f} "
                    f"attempts={attempts} filled={filled} "
                    f"unfilled={int(ds.get('passive_unfilled', 0) or 0)} partial={partial}"
                )
                print(
                    f"[{sym}] FILLED_ONLY_METRICS n={int(filled_only.get('n', 0))} "
                    f"avg_net={float(filled_only.get('avg_net', 0.0)):+.6f} "
                    f"p90_net={float(filled_only.get('p90_net', 0.0)):+.6f} "
                    f"win_rate={float(filled_only.get('win_rate', 0.0)):.2%}"
                )
                print(
                    f"[{sym}] ATTEMPT_LEVEL_METRICS n={int(attempt_level.get('n_attempts', 0))} "
                    f"net_per_attempt={float(attempt_level.get('net_per_attempt', 0.0)):+.6f} "
                    f"fill_rate={float(attempt_level.get('fill_rate', 0.0)):.2%} "
                    f"partial_rate={float(attempt_level.get('partial_rate', 0.0)):.2%}"
                )
                print(
                    f"[{sym}] CORR fill_fraction_vs_net={corrs.get('fill_fraction_vs_net')} "
                    f"adverse_selection_bps_vs_net={corrs.get('adverse_selection_bps_vs_net')}"
                )
                print(
                    f"[{sym}] COST_BREAKDOWN avg_fee_bps={cost_breakdown['avg_fee_bps']:.3f} "
                    f"avg_spread_bps={cost_breakdown['avg_spread_bps']:.3f} "
                    f"avg_adverse_bps={cost_breakdown['avg_adverse_bps']:.3f} "
                    f"avg_total_bps={cost_breakdown['avg_total_bps']:.3f}"
                )
            if int(ds.get("debug_dedupe_drop", 0) or 0) > 0:
                print(f"[{sym}] debug_dedupe_drop={int(ds.get('debug_dedupe_drop', 0))}")
            fd_miss = int(ds.get("filter_drop_missing_key", 0) or 0)
            fd_min = int(ds.get("filter_drop_below_min", 0) or 0)
            fd_max = int(ds.get("filter_drop_above_max", 0) or 0)
            if fd_miss or fd_min or fd_max:
                print(
                    f"[{sym}] feature_filter_drops missing_key={fd_miss} below_min={fd_min} above_max={fd_max}"
                )
            tox_block = int(ds.get("toxicity_blocked", 0) or 0)
            if tox_block > 0:
                print(
                    f"[{sym}] toxicity_gate_blocks total={tox_block} "
                    f"vol_intensity={int(ds.get('toxicity_block_vol_intensity', 0) or 0)}"
                )
            cd_spread = int(ds.get("cost_drop_missing_spread", 0) or 0)
            if cd_spread > 0:
                print(f"[{sym}] cost_model_drops missing_spread={cd_spread}")
            cap = debug_cap_info(int(args.debug_samples), int(ds.get("debug_written", 0) or 0))
            print(
                f"[{sym}] debug_out_capped={'yes' if bool(cap['debug_out_capped']) else 'no'} "
                f"debug_samples_limit={int(cap['debug_samples_limit'])} "
                f"debug_rows_written={int(cap['debug_rows_written'])}"
            )
            rec = {
                "ts_utc": utc_now_iso(),
                "symbol": sym,
                "lookback_min": int(args.lookback_min),
                "bucket_sec": int(args.bucket_sec),
                "horizon_sec": int(args.horizon_sec),
                "rule": rule_name,
                "side": side,
                "fee_bps": float(args.fee_bps),
                "slip_bps": float(args.slip_bps),
                "exec_model": str(args.exec_model),
                "maker_fee_bps": float(args.maker_fee_bps),
                "maker_penalty_bps": float(args.maker_penalty_bps),
                "passive_adverse_mult": float(args.passive_adverse_mult),
                "max_hold_buckets": int(hold),
                "cooldown_buckets": int(args.cooldown_buckets),
                "min_rule_n": int(args.min_rule_n),
                "thresholds": thresholds,
                "rules_filtered": rules_filtered,
                "feature_filter": {"min": min_bounds, "max": max_bounds},
                "feature_filter_drops": {
                    "missing_key": fd_miss,
                    "below_min": fd_min,
                    "above_max": fd_max,
                    "cost_missing_spread": cd_spread,
                },
                "passive_stats": {
                    "attempts": attempts if str(args.exec_model).lower() == "passive_realistic" else int(ds.get("passive_attempts", 0) or 0),
                    "filled": filled if str(args.exec_model).lower() == "passive_realistic" else int(ds.get("passive_filled", 0) or 0),
                    "unfilled": int(ds.get("passive_unfilled", 0) or 0),
                    "partial": partial if str(args.exec_model).lower() == "passive_realistic" else int(ds.get("passive_partial", 0) or 0),
                    "attempt_fill_rate": (filled / attempts) if (str(args.exec_model).lower() == "passive_realistic" and attempts > 0) else 0.0,
                    "trade_fill_rate": float(comp.get("fill_rate", 0.0)),
                    "avg_fill_fraction": float(comp.get("avg_fill_fraction", 0.0)),
                },
                "filled_only_metrics": filled_only,
                "attempt_level_metrics": attempt_level,
                "trade_correlations": corrs,
                "toxicity_gate_counts": {
                    "blocked_total": int(ds.get("toxicity_blocked", 0) or 0),
                    "blocked_vol_intensity": int(ds.get("toxicity_block_vol_intensity", 0) or 0),
                },
                "trade_components": comp,
                "cost_breakdown": cost_breakdown,
                "passive_params": passive_params,
                "metrics": m,
            }
            append_jsonl(out_path, rec)
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
