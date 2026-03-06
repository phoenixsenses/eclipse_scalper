from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import pandas as pd
import os

from src.microphys.execution.engine import ExecutionRequest, build_default_engines
from src.microphys.execution.fill_models import HazardParams, simulate_maker_hazard_fill
from src.microphys.execution.queue_sim import QueueSimParams, simulate_maker_queue_fill


@dataclass(frozen=True)
class PaperTradeConfig:
    mode: str = "taker"
    fee_bps: float = 0.5
    execution_model: str = "simple"  # simple | maker_queue | maker_hazard
    execution_params: Dict[str, Any] | None = None
    ttl_bars: int = 10
    use_unified_engine: bool = False


def _flag_on(name: str, default: bool = False) -> bool:
    v = str(os.getenv(name, "1" if default else "0")).strip().lower()
    return v in {"1", "true", "yes", "on"}


def _use_unified_engine(cfg: PaperTradeConfig) -> bool:
    if bool(cfg.use_unified_engine):
        return True
    return _flag_on("EXEC_ENGINE_UNIFIED", default=False)


def generate_papertrades(
    frame: pd.DataFrame,
    *,
    horizon_bars: int,
    cfg: PaperTradeConfig,
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["entry_ts_utc", "side", "entry_price", "exit_ts_utc", "exit_price", "pnl_net"])
    df = frame.sort_values("ts_ms").reset_index(drop=True).copy()
    mid = pd.to_numeric(df.get("mid"), errors="coerce").replace(0.0, np.nan)
    spread = pd.to_numeric(df.get("spread"), errors="coerce").fillna(0.0)
    side = pd.to_numeric(df.get("ensemble_side"), errors="coerce").fillna(0.0)
    entry_mask = side != 0.0
    if not bool(entry_mask.any()):
        return pd.DataFrame(columns=["entry_ts_utc", "side", "entry_price", "exit_ts_utc", "exit_price", "pnl_net"])

    h = max(1, int(horizon_bars))
    exit_mid = mid.shift(-h)
    use_engine = _use_unified_engine(cfg)
    engines = build_default_engines() if use_engine else {}
    if str(cfg.execution_model).lower() == "simple":
        if str(cfg.mode).lower() == "taker":
            entry_px = mid + (np.sign(side) * (spread / 2.0))
            exit_px = exit_mid - (np.sign(side) * (spread.shift(-h).fillna(0.0) / 2.0))
        else:
            entry_px = mid - (np.sign(side) * (spread / 2.0))
            exit_px = exit_mid + (np.sign(side) * (spread.shift(-h).fillna(0.0) / 2.0))
        gross = np.sign(side) * ((exit_px - entry_px) / entry_px.replace(0.0, np.nan))
        fee = (float(cfg.fee_bps) / 10000.0)
        net = gross - fee

        out = pd.DataFrame(
            {
                "entry_ts_utc": df.get("ts_utc"),
                "side": np.where(side > 0, "buy", "sell"),
                "entry_price": entry_px,
                "fill_price": entry_px,
                "filled": True,
                "fill_delay_bars": 0,
                "ttl_expired": False,
                "exit_ts_utc": df.get("ts_utc").shift(-h),
                "exit_price": exit_px,
                "pnl_net": net,
                "pnl_gross": gross,
                "fee": fee,
                "signal_count": pd.to_numeric(df.get("signal_count"), errors="coerce").fillna(0).astype(int),
            }
        )
        out = out[entry_mask].dropna(subset=["exit_ts_utc", "entry_price", "exit_price"]).reset_index(drop=True)
        if use_engine and not out.empty:
            ereq = []
            for idx, r in out.iterrows():
                ereq.append(
                    ExecutionRequest(
                        symbol=str(df.iloc[int(idx)].get("symbol", "")),
                        side=str(r.get("side", "buy")).lower(),  # type: ignore[arg-type]
                        entry_price=float(r.get("entry_price", 0.0)),
                        exit_price=float(r.get("exit_price", 0.0)),
                        notional=1.0,
                        fee_bps=float(cfg.fee_bps),
                        slippage_bps=0.0,
                        ts_ms=int(pd.to_numeric(df.iloc[int(idx)].get("ts_ms"), errors="coerce") or 0),
                        order_id=f"paper_simple_{int(idx)}",
                    )
                )
            rows = []
            for req in ereq:
                res = engines["paper"].execute(req)
                rows.append(res.to_dict())
            res_df = pd.DataFrame(rows)
            if not res_df.empty:
                out["pnl_gross"] = pd.to_numeric(res_df.get("gross_return"), errors="coerce").fillna(out["pnl_gross"])
                out["pnl_net"] = pd.to_numeric(res_df.get("net_return"), errors="coerce").fillna(out["pnl_net"])
                out["fee"] = pd.to_numeric(res_df.get("fee_cost"), errors="coerce").fillna(out["fee"])
                out["order_id"] = res_df.get("order_id")
        return out

    # realistic maker path (queue/hazard)
    rows = []
    ex_model = str(cfg.execution_model).lower()
    params = dict(cfg.execution_params or {})
    for i in np.flatnonzero(entry_mask.to_numpy(dtype=bool)):
        s = "buy" if float(side.iloc[i]) > 0 else "sell"
        if ex_model == "maker_hazard":
            hp = HazardParams(**{**{"ttl_bars": int(cfg.ttl_bars)}, **dict(params.get("maker_hazard", {}))})
            sim = simulate_maker_hazard_fill(df, entry_idx=int(i), side=s, params=hp)
        else:
            qp = QueueSimParams(**{**{"ttl_bars": int(cfg.ttl_bars)}, **dict(params.get("maker_queue", {}))})
            sim = simulate_maker_queue_fill(df, entry_idx=int(i), side=s, params=qp)
        fill_idx = sim.get("fill_idx")
        filled = bool(sim.get("filled", False))
        entry_px = float(mid.iloc[i]) - (0.5 * float(spread.iloc[i]) if s == "buy" else -0.5 * float(spread.iloc[i]))
        fill_px = float(entry_px)
        if filled and fill_idx is not None:
            fi = int(fill_idx)
            fill_px = float(mid.iloc[fi]) - (0.5 * float(spread.iloc[fi]) if s == "buy" else -0.5 * float(spread.iloc[fi]))
            exit_i = fi + h
            if exit_i >= len(df):
                continue
            exit_px = float(mid.iloc[exit_i]) + (0.5 * float(spread.iloc[exit_i]) if s == "buy" else -0.5 * float(spread.iloc[exit_i]))
            gross = ((exit_px - fill_px) / max(1e-12, fill_px)) if s == "buy" else ((fill_px - exit_px) / max(1e-12, fill_px))
            fee = float(cfg.fee_bps) / 10000.0
            net = gross - fee
            order_id = f"paper_realistic_{int(i)}_{int(fi)}"
            if use_engine:
                req = ExecutionRequest(
                    symbol=str(df.iloc[int(i)].get("symbol", "")),
                    side=str(s),  # type: ignore[arg-type]
                    entry_price=float(fill_px),
                    exit_price=float(exit_px),
                    notional=1.0,
                    fee_bps=float(cfg.fee_bps),
                    slippage_bps=0.0,
                    ts_ms=int(pd.to_numeric(df.iloc[int(i)].get("ts_ms"), errors="coerce") or 0),
                    order_id=order_id,
                )
                res = engines["paper"].execute(req)
                gross = float(res.gross_return)
                net = float(res.net_return)
                fee = float(res.fee_cost)
            rows.append(
                {
                    "entry_ts_utc": str(df.iloc[i]["ts_utc"]),
                    "side": s,
                    "entry_price": entry_px,
                    "fill_price": fill_px,
                    "filled": True,
                    "fill_delay_bars": int(sim.get("fill_delay_bars", 0) or 0),
                    "ttl_expired": False,
                    "exit_ts_utc": str(df.iloc[exit_i]["ts_utc"]),
                    "exit_price": exit_px,
                    "pnl_net": float(net),
                    "pnl_gross": float(gross),
                    "fee": float(fee),
                    "order_id": str(order_id),
                    "signal_count": int(pd.to_numeric(df.get("signal_count"), errors="coerce").fillna(0).iloc[i]),
                }
            )
        else:
            rows.append(
                {
                    "entry_ts_utc": str(df.iloc[i]["ts_utc"]),
                    "side": s,
                    "entry_price": entry_px,
                    "fill_price": np.nan,
                    "filled": False,
                    "fill_delay_bars": np.nan,
                    "ttl_expired": True,
                    "exit_ts_utc": np.nan,
                    "exit_price": np.nan,
                    "pnl_net": 0.0,
                    "pnl_gross": 0.0,
                    "fee": 0.0,
                    "signal_count": int(pd.to_numeric(df.get("signal_count"), errors="coerce").fillna(0).iloc[i]),
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(
            columns=[
                "entry_ts_utc",
                "side",
                "entry_price",
                "fill_price",
                "filled",
                "fill_delay_bars",
                "ttl_expired",
                "exit_ts_utc",
                "exit_price",
                "pnl_net",
            ]
        )
    return out.reset_index(drop=True)
