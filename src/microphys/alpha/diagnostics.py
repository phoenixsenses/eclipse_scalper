from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

from src.microphys.execution.cost_models import CostConfig

from .dsl import evaluate_expr
from .eval import apply_signal_entries, evaluate_spec_on_frame, make_walkforward_splits
from .spec import SignalSpec


@dataclass(frozen=True)
class CoverageConfig:
    splits: int = 3


def split_windows(ts_ms: List[int], splits: int) -> List[tuple[int, int]]:
    wins = make_walkforward_splits(ts_ms, int(splits))
    out = [(int(w.test_start), int(w.test_end)) for w in wins]
    if out:
        return out
    if not ts_ms:
        return []
    uniq = sorted(set(int(x) for x in ts_ms))
    return [(int(uniq[0]), int(uniq[-1]))]


def candidate_coverage(frame: pd.DataFrame, specs: Iterable[SignalSpec], *, splits: int = 3) -> pd.DataFrame:
    df = frame.sort_values("ts_ms").reset_index(drop=True).copy()
    windows = split_windows(df["ts_ms"].astype(int).tolist(), splits)
    rows: List[Dict[str, float | int | str]] = []
    for spec in specs:
        raw = evaluate_expr(df, spec.condition).fillna(False)
        if spec.regime_filter:
            raw = raw & df.get("regime_id", pd.Series(index=df.index, dtype=float)).isin(spec.regime_filter).fillna(False)
        cooled = apply_signal_entries(df, spec).fillna(False)
        dropped = int(max(0, int(raw.sum()) - int(cooled.sum())))
        future_ok = pd.to_numeric(df.get("mid"), errors="coerce").shift(-max(1, int(spec.horizon_bars))).notna()
        effective = (cooled & future_ok).fillna(False)
        regime_counts = (
            df.loc[effective, "regime_id"].astype(int).value_counts(normalize=True).sort_values(ascending=False).head(3)
            if "regime_id" in df.columns
            else pd.Series(dtype=float)
        )
        top_reg = ",".join(f"{int(k)}:{float(v):.3f}" for k, v in regime_counts.items())
        for split_id, (a, b) in enumerate(windows, start=1):
            in_split = (df["ts_ms"] >= a) & (df["ts_ms"] <= b)
            rows.append(
                {
                    "signal": spec.name,
                    "split_id": int(split_id),
                    "triggered_events": int((raw & in_split).sum()),
                    "after_cooldown": int((cooled & in_split).sum()),
                    "effective_trades": int((effective & in_split).sum()),
                    "cooldown_drop_pct": float((dropped / max(1, int(raw.sum()))) * 100.0),
                    "missing_horizon_pct": float((((cooled & ~future_ok).sum()) / max(1, int(cooled.sum()))) * 100.0),
                    "regime_concentration_top3": top_reg,
                }
            )
    return pd.DataFrame(rows).sort_values(["signal", "split_id"]).reset_index(drop=True)


def _adverse_component(side: pd.Series, r1: pd.Series) -> pd.Series:
    s = pd.to_numeric(side, errors="coerce").fillna(0.0)
    nxt = pd.to_numeric(r1, errors="coerce").fillna(0.0)
    return pd.Series(np.where(s > 0, np.maximum(0.0, -nxt), np.maximum(0.0, nxt)), index=s.index)


def cost_decomposition(frame: pd.DataFrame, specs: Iterable[SignalSpec], *, fee_bps: float, latency_bars: int, mode: str, fill_prob: float) -> pd.DataFrame:
    out_rows: List[Dict[str, float | int | str]] = []
    for spec in specs:
        trades, stats = evaluate_spec_on_frame(
            frame,
            spec,
            fee_bps=fee_bps,
            latency_bars=latency_bars,
            mode=mode,
            fill_prob=fill_prob,
        )
        if trades.empty:
            out_rows.append(
                {
                    "signal": spec.name,
                    "mode": mode,
                    "trade_count": 0,
                    "gross_mean": 0.0,
                    "fee_cost_mean": 0.0,
                    "spread_cost_mean": 0.0,
                    "adverse_cost_mean": 0.0,
                    "net_mean": 0.0,
                }
            )
            continue
        cfg = CostConfig(fee_bps=float(fee_bps), latency_bars=int(latency_bars), mode=mode, fill_prob=float(fill_prob))
        fee = float(cfg.fee_bps) / 10000.0
        spread = pd.to_numeric(frame.get("spread"), errors="coerce").fillna(0.0).iloc[trades["entry_idx"].astype(int).to_list()].reset_index(drop=True)
        side = pd.to_numeric(trades.get("side"), errors="coerce").fillna(0.0).reset_index(drop=True)
        mid = pd.to_numeric(frame.get("mid"), errors="coerce").replace(0.0, np.nan)
        lat = max(0, int(latency_bars))
        r1_lat = np.log(mid.shift(-(1 + lat)) / mid.shift(lat))
        r1 = pd.to_numeric(r1_lat, errors="coerce").fillna(0.0).iloc[trades["entry_idx"].astype(int).to_list()].reset_index(drop=True)
        adverse = _adverse_component(side, r1)
        spread_cost = (0.5 * spread) if mode == "taker" else pd.Series(np.zeros(len(trades), dtype=float), index=trades.index)
        if mode == "maker":
            adverse = adverse * float(fill_prob)
        gross = pd.to_numeric(trades["gross_ret"], errors="coerce").fillna(0.0).reset_index(drop=True)
        net = gross - fee - spread_cost - adverse
        out_rows.append(
            {
                "signal": spec.name,
                "mode": mode,
                "trade_count": int(len(trades)),
                "gross_mean": float(gross.mean()),
                "fee_cost_mean": float(fee),
                "spread_cost_mean": float(spread_cost.mean() if len(spread_cost) else 0.0),
                "adverse_cost_mean": float(adverse.mean() if len(adverse) else 0.0),
                "net_mean": float(net.mean()),
            }
        )
    return pd.DataFrame(out_rows).sort_values(["signal", "mode"]).reset_index(drop=True)
