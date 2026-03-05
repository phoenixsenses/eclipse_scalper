from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

from src.microphys.execution.cost_models import CostConfig, evaluate_trade_net
from src.microphys.execution.fill_models import HazardParams, simulate_maker_hazard_fill
from src.microphys.execution.queue_sim import QueueSimParams, simulate_maker_queue_fill

from .calibration import CalibrationContext
from .dsl import evaluate_expr
from .metrics import daily_sharpe, quantile_5, stability_score
from .spec import SignalSpec


@dataclass(frozen=True)
class SplitWindow:
    split_id: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int


def make_walkforward_splits(ts_ms: Sequence[int], splits: int) -> List[SplitWindow]:
    arr = np.asarray(list(ts_ms), dtype=np.int64)
    if arr.size < 30:
        return []
    uniq = np.unique(arr)
    n = int(len(uniq))
    k = max(1, int(splits))
    out: List[SplitWindow] = []
    chunk = max(10, n // (k + 2))
    for i in range(k):
        train_start_idx = i * chunk
        train_end_idx = min(n - 2, train_start_idx + chunk)
        test_start_idx = train_end_idx + 1
        test_end_idx = min(n - 1, test_start_idx + chunk - 1)
        if test_end_idx <= test_start_idx:
            break
        out.append(
            SplitWindow(
                split_id=i + 1,
                train_start=int(uniq[train_start_idx]),
                train_end=int(uniq[train_end_idx]),
                test_start=int(uniq[test_start_idx]),
                test_end=int(uniq[test_end_idx]),
            )
        )
    return out


def apply_signal_entries(df: pd.DataFrame, spec: SignalSpec, calibration: CalibrationContext | None = None) -> pd.Series:
    mask = evaluate_expr(df, spec.condition, calibration=calibration).fillna(False)
    if spec.regime_filter:
        mask = mask & df.get("regime_id", pd.Series(index=df.index, dtype=float)).isin(spec.regime_filter).fillna(False)
    if not bool(mask.any()):
        return mask
    cooldown = max(0, int(spec.cooldown_bars))
    if cooldown <= 0:
        return mask
    idx = np.flatnonzero(mask.to_numpy(dtype=bool))
    keep = np.zeros(len(mask), dtype=bool)
    next_allowed = -10**18
    for i in idx:
        if i < next_allowed:
            continue
        keep[i] = True
        next_allowed = i + cooldown
    return pd.Series(keep, index=df.index)


def _side_for_spec(spec: SignalSpec) -> float:
    s = str(spec.side).lower()
    if s == "buy":
        return 1.0
    if s == "sell":
        return -1.0
    return 0.0


def _ensure_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col, default in (
        ("r_1", 0.0),
        ("spread", 0.0),
        ("regime_id", -1),
        ("ts_utc", ""),
    ):
        if col not in out.columns:
            out[col] = default
    return out


def evaluate_spec_on_frame(
    df: pd.DataFrame,
    spec: SignalSpec,
    *,
    calibration: CalibrationContext | None = None,
    fee_bps: float,
    latency_bars: int,
    mode: str,
    fill_prob: float,
    max_trades_per_day: int = 500,
    execution_model: str = "simple",
    execution_params: Dict[str, Any] | None = None,
    ttl_bars: int = 10,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    frame = _ensure_cols(df).reset_index(drop=True)
    h = max(1, int(spec.horizon_bars))
    lat = max(0, int(latency_bars))

    mid = pd.to_numeric(frame.get("mid"), errors="coerce").replace(0.0, np.nan)
    ret_h = np.log(mid.shift(-(h + lat)) / mid.shift(lat))
    r1_lat = np.log(mid.shift(-(1 + lat)) / mid.shift(lat))

    entries = apply_signal_entries(frame, spec, calibration=calibration)
    side_scalar = _side_for_spec(spec)
    if side_scalar == 0.0:
        side = np.sign(pd.to_numeric(frame.get("F_ofi_z"), errors="coerce").fillna(0.0))
        side = side.replace(0.0, 1.0)
    else:
        side = pd.Series(np.full(len(frame), side_scalar, dtype=float), index=frame.index)

    gross = side * pd.to_numeric(ret_h, errors="coerce").fillna(0.0)
    net = evaluate_trade_net(
        ret=ret_h,
        spread=pd.to_numeric(frame.get("spread"), errors="coerce").fillna(0.0),
        side=side,
        r1=pd.to_numeric(r1_lat, errors="coerce").fillna(0.0),
        cfg=CostConfig(
            fee_bps=float(fee_bps),
            latency_bars=int(latency_bars),
            mode=str(mode),
            fill_prob=float(fill_prob),
        ),
    )

    t = pd.to_datetime(frame.get("ts_utc"), utc=True, errors="coerce")
    day = t.dt.strftime("%Y-%m-%d")
    trades = pd.DataFrame(
        {
            "ts_ms": frame.get("ts_ms"),
            "ts_utc": frame.get("ts_utc"),
            "day": day,
            "entry_idx": frame.index,
            "entry": entries,
            "regime_id": frame.get("regime_id"),
            "gross_ret": gross,
            "net_ret": net,
            "side": side,
            "signal": spec.name,
        }
    )
    trades = trades[trades["entry"]].reset_index(drop=True)
    attempted_entries = int(len(trades))
    if str(execution_model).lower() in {"maker_queue", "maker_hazard"} and not trades.empty:
        params = dict(execution_params or {})
        sim_rows = []
        for _, tr in trades.iterrows():
            idx = int(tr["entry_idx"])
            s = "buy" if float(tr["side"]) > 0 else "sell"
            if str(execution_model).lower() == "maker_hazard":
                hp = HazardParams(**{**{"ttl_bars": int(ttl_bars)}, **dict(params.get("maker_hazard", {}))})
                sim = simulate_maker_hazard_fill(frame, entry_idx=idx, side=s, params=hp)
            else:
                qp = QueueSimParams(**{**{"ttl_bars": int(ttl_bars)}, **dict(params.get("maker_queue", {}))})
                sim = simulate_maker_queue_fill(frame, entry_idx=idx, side=s, params=qp)
            sim_rows.append(sim)
        sim_df = pd.DataFrame(sim_rows)
        trades = pd.concat([trades.reset_index(drop=True), sim_df.reset_index(drop=True)], axis=1)
        if "filled" not in trades.columns:
            trades["filled"] = False
        if "ttl_expired" not in trades.columns:
            trades["ttl_expired"] = True
        trades["filled"] = trades["filled"].fillna(False).astype(bool)
        trades = trades[trades["filled"]].copy()
        if not trades.empty:
            trades["net_ret"] = pd.to_numeric(trades["net_ret"], errors="coerce").fillna(0.0)
            trades["gross_ret"] = pd.to_numeric(trades["gross_ret"], errors="coerce").fillna(0.0)
        else:
            trades = trades.iloc[:0].copy()
    capped = 0
    cap = max(0, int(max_trades_per_day))
    if cap > 0 and not trades.empty:
        kept = []
        for _, g in trades.groupby("day", sort=True):
            gg = g.sort_values("entry_idx").head(cap)
            capped += int(len(g) - len(gg))
            kept.append(gg)
        trades = pd.concat(kept, ignore_index=True) if kept else trades.iloc[:0].copy()
    if trades.empty:
        return trades, {
            "trade_count": 0.0,
            "gross_mean": 0.0,
            "net_mean": 0.0,
            "net_median": 0.0,
            "daily_sharpe": 0.0,
            "worst_day": 0.0,
            "q05_day": 0.0,
            "stability_score": 0.0,
            "regime_concentration": 0.0,
            "capped_trades": 0.0,
            "trade_density": 0.0,
            "fill_rate": 0.0,
        }

    daily = trades.groupby("day", as_index=False).agg(net_mean=("net_ret", "mean"))
    regime_share = trades["regime_id"].value_counts(normalize=True)
    bar_count = max(1, len(frame))
    stats = {
        "trade_count": float(len(trades)),
        "gross_mean": float(pd.to_numeric(trades["gross_ret"], errors="coerce").mean()),
        "net_mean": float(pd.to_numeric(trades["net_ret"], errors="coerce").mean()),
        "net_median": float(pd.to_numeric(trades["net_ret"], errors="coerce").median()),
        "daily_sharpe": float(daily_sharpe(daily["net_mean"])),
        "worst_day": float(pd.to_numeric(daily["net_mean"], errors="coerce").min()),
        "q05_day": float(quantile_5(daily["net_mean"])),
        "stability_score": float(stability_score(daily["net_mean"])),
        "regime_concentration": float(regime_share.max()) if not regime_share.empty else 0.0,
        "capped_trades": float(capped),
        "trade_density": float(len(trades) / bar_count),
        "fill_rate": float(len(trades) / max(1, attempted_entries)),
    }
    return trades, stats


def evaluate_walkforward(
    df: pd.DataFrame,
    specs: Iterable[SignalSpec],
    *,
    calibration: CalibrationContext | None = None,
    splits: int,
    fee_bps: float,
    latency_bars: int,
    mode: str,
    fill_prob: float,
    max_trades_per_day: int = 500,
    execution_model: str = "simple",
    execution_params: Dict[str, Any] | None = None,
    ttl_bars: int = 10,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    frame = _ensure_cols(df).copy()
    frame = frame.sort_values("ts_ms").reset_index(drop=True)
    windows = make_walkforward_splits(frame["ts_ms"].astype(int).to_list(), int(splits))
    eval_rows: List[Dict[str, Any]] = []
    trade_rows: List[pd.DataFrame] = []
    for spec in specs:
        for win in windows:
            train = frame[(frame["ts_ms"] >= win.train_start) & (frame["ts_ms"] <= win.train_end)].reset_index(drop=True)
            test = frame[(frame["ts_ms"] >= win.test_start) & (frame["ts_ms"] <= win.test_end)].reset_index(drop=True)
            train_trades, train_stats = evaluate_spec_on_frame(
                train,
                spec,
                calibration=calibration,
                fee_bps=fee_bps,
                latency_bars=latency_bars,
                mode=mode,
                fill_prob=fill_prob,
                max_trades_per_day=max_trades_per_day,
                execution_model=execution_model,
                execution_params=execution_params,
                ttl_bars=ttl_bars,
            )
            test_trades, test_stats = evaluate_spec_on_frame(
                test,
                spec,
                calibration=calibration,
                fee_bps=fee_bps,
                latency_bars=latency_bars,
                mode=mode,
                fill_prob=fill_prob,
                max_trades_per_day=max_trades_per_day,
                execution_model=execution_model,
                execution_params=execution_params,
                ttl_bars=ttl_bars,
            )
            overfit_gap = float(train_stats["net_mean"] - test_stats["net_mean"])
            eval_rows.append(
                {
                    "signal": spec.name,
                    "split_id": int(win.split_id),
                    "train_trade_count": int(train_stats["trade_count"]),
                    "test_trade_count": int(test_stats["trade_count"]),
                    "train_net_mean": float(train_stats["net_mean"]),
                    "test_net_mean": float(test_stats["net_mean"]),
                    "test_net_median": float(test_stats["net_median"]),
                    "test_sharpe": float(test_stats["daily_sharpe"]),
                    "worst_day": float(test_stats["worst_day"]),
                    "q05_day": float(test_stats["q05_day"]),
                    "stability_score": float(test_stats["stability_score"]),
                    "regime_concentration": float(test_stats["regime_concentration"]),
                    "capped_trades": float(test_stats["capped_trades"]),
                    "trade_density": float(test_stats["trade_density"]),
                    "fill_rate": float(test_stats.get("fill_rate", 1.0)),
                    "overfit_gap": overfit_gap,
                }
            )
            if not test_trades.empty:
                tt = test_trades.copy()
                tt["split_id"] = int(win.split_id)
                tt["sample"] = "test"
                trade_rows.append(tt)
            if not train_trades.empty:
                tr = train_trades.copy()
                tr["split_id"] = int(win.split_id)
                tr["sample"] = "train"
                trade_rows.append(tr)

    eval_df = pd.DataFrame(eval_rows).sort_values(["signal", "split_id"]).reset_index(drop=True)
    trades_df = pd.concat(trade_rows, ignore_index=True) if trade_rows else pd.DataFrame()
    return eval_df, trades_df
