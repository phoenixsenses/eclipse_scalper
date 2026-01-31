#!/usr/bin/env python3
"""
Simple backtest harness for eclipse_scalper strategies.
Loads 1m CSV OHLCV data, runs scalper_signal, and evaluates multiple exit methods.
"""

from __future__ import annotations

import os
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import sys

# Avoid Windows console Unicode errors from logging setup
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import pandas as pd

from strategies import eclipse_scalper


@dataclass
class Trade:
    side: str
    entry_ts: int
    entry_px: float
    qty: float
    exit_ts: Optional[int] = None
    exit_px: Optional[float] = None
    pnl: Optional[float] = None


class DataShim:
    def __init__(self):
        self.ohlcv: Dict[str, List[list]] = {}
        self.raw_symbol: Dict[str, str] = {}
        self.funding: Dict[str, float] = {}


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "ts" not in df.columns:
        raise ValueError(f"missing ts in {path}")
    return df


def calc_atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    high = df["h"]
    low = df["l"]
    close = df["c"]
    prev_close = close.shift(1)
    tr = (high - low).abs()
    tr = tr.combine((high - prev_close).abs(), max)
    tr = tr.combine((low - prev_close).abs(), max)
    return tr.rolling(window).mean()


def backtest_symbol(
    sym: str,
    df: pd.DataFrame,
    *,
    enhanced: bool,
    exit_method: str,
    notional: float,
    fee_rate: float,
    slippage: float,
    stop_atr: float,
    take_atr: float,
    max_hold_bars: int,
    stride: int,
    window: int,
) -> Tuple[List[Trade], List[float]]:
    data = DataShim()
    k = eclipse_scalper._symkey(sym)
    data.raw_symbol[k] = sym

    trades: List[Trade] = []
    equity_curve: List[float] = []
    equity = 0.0

    open_trade: Optional[Trade] = None
    entry_index: Optional[int] = None

    # prepare rolling ATR for stops
    atr = calc_atr(df, 14)

    # set env for enhanced gates
    os.environ["SCALPER_ENHANCED"] = "1" if enhanced else "0"

    for i in range(len(df)):
        row = df.iloc[i]
        ts = int(row["ts"])
        o, h, l, c, v = float(row["o"]), float(row["h"]), float(row["l"]), float(row["c"]), float(row["v"])

        rows = data.ohlcv.get(k, [])
        rows.append([ts, o, h, l, c, v])
        if window > 0 and len(rows) > window:
            rows = rows[-window:]
        data.ohlcv[k] = rows

        long_sig = short_sig = False
        if stride <= 1 or (i % stride) == 0:
            long_sig, short_sig, _conf = eclipse_scalper.scalper_signal(k, data=data, cfg=None)

        # manage open trade
        if open_trade is not None:
            # exits
            exit_px = None
            if exit_method == "signal":
                if open_trade.side == "long" and short_sig:
                    exit_px = c * (1 - slippage)
                elif open_trade.side == "short" and long_sig:
                    exit_px = c * (1 + slippage)
            elif exit_method == "time":
                if entry_index is not None and (i - entry_index) >= max_hold_bars:
                    exit_px = c * (1 - slippage) if open_trade.side == "long" else c * (1 + slippage)
            elif exit_method == "atr":
                atr_now = atr.iloc[i]
                if not math.isfinite(atr_now):
                    atr_now = 0.0
                if open_trade.side == "long":
                    stop = open_trade.entry_px - atr_now * stop_atr
                    take = open_trade.entry_px + atr_now * take_atr
                    if l <= stop:
                        exit_px = stop * (1 - slippage)
                    elif h >= take:
                        exit_px = take * (1 - slippage)
                else:
                    stop = open_trade.entry_px + atr_now * stop_atr
                    take = open_trade.entry_px - atr_now * take_atr
                    if h >= stop:
                        exit_px = stop * (1 + slippage)
                    elif l <= take:
                        exit_px = take * (1 + slippage)

            if exit_px is not None:
                open_trade.exit_ts = ts
                open_trade.exit_px = exit_px
                if open_trade.side == "long":
                    pnl = (exit_px - open_trade.entry_px) * open_trade.qty
                else:
                    pnl = (open_trade.entry_px - exit_px) * open_trade.qty
                # fees (entry + exit)
                fees = (open_trade.entry_px * open_trade.qty + exit_px * open_trade.qty) * fee_rate
                pnl -= fees
                open_trade.pnl = pnl
                trades.append(open_trade)
                equity += pnl
                open_trade = None
                entry_index = None

        # entries (only if flat)
        if open_trade is None:
            if long_sig or short_sig:
                side = "long" if long_sig else "short"
                entry_px = c * (1 + slippage) if side == "long" else c * (1 - slippage)
                qty = notional / max(1e-9, entry_px)
                open_trade = Trade(side=side, entry_ts=ts, entry_px=entry_px, qty=qty)
                entry_index = i

        equity_curve.append(equity)

    return trades, equity_curve


def summarize(trades: List[Trade], equity_curve: List[float]) -> Dict[str, float]:
    if not trades:
        return {"trades": 0, "pnl": 0.0, "win_rate": 0.0, "max_dd": 0.0}
    pnl = sum(t.pnl or 0.0 for t in trades)
    wins = sum(1 for t in trades if (t.pnl or 0.0) > 0)
    win_rate = wins / len(trades)
    peak = -1e9
    max_dd = 0.0
    for x in equity_curve:
        peak = max(peak, x)
        dd = peak - x
        max_dd = max(max_dd, dd)
    return {
        "trades": len(trades),
        "pnl": pnl,
        "win_rate": win_rate,
        "max_dd": max_dd,
    }


def run():
    data_dir = os.path.join(os.getcwd(), "data_cache")
    sym_env = os.getenv("BT_SYMBOLS", "")
    if sym_env:
        symbols = [s.strip().replace("/", "_") + "_1m.csv" for s in sym_env.split(",") if s.strip()]
    else:
        symbols = ["DOGE_USDT_1m.csv", "BTC_USDT_1m.csv", "ETH_USDT_1m.csv"]
    methods_env = os.getenv("BT_METHODS", "")
    if methods_env:
        exit_methods = [m.strip() for m in methods_env.split(",") if m.strip()]
    else:
        exit_methods = ["atr", "signal", "time"]

    notional = float(os.getenv("BT_NOTIONAL", "100"))
    fee_rate = float(os.getenv("BT_FEE_RATE", "0.0004"))
    slippage = float(os.getenv("BT_SLIPPAGE", "0.0001"))
    stop_atr = float(os.getenv("BT_STOP_ATR", "1.5"))
    take_atr = float(os.getenv("BT_TAKE_ATR", "2.0"))
    max_hold_bars = int(os.getenv("BT_MAX_HOLD_BARS", "180"))
    max_bars = int(os.getenv("BT_MAX_BARS", "0"))
    stride = int(os.getenv("BT_STRIDE", "5"))
    window = int(os.getenv("BT_WINDOW", "800"))

    for fname in symbols:
        path = os.path.join(data_dir, fname)
        if not os.path.exists(path):
            print(f"missing {path}, skipping")
            continue
        sym = fname.replace("_1m.csv", "").replace("_", "/")
        df = load_csv(path)
        if max_bars > 0 and len(df) > max_bars:
            df = df.iloc[-max_bars:].reset_index(drop=True)
        print(f"\n=== {sym} bars={len(df)} ===")
        for enhanced in (True, False):
            for method in exit_methods:
                trades, equity_curve = backtest_symbol(
                    sym,
                    df,
                    enhanced=enhanced,
                    exit_method=method,
                    notional=notional,
                    fee_rate=fee_rate,
                    slippage=slippage,
                    stop_atr=stop_atr,
                    take_atr=take_atr,
                    max_hold_bars=max_hold_bars,
                    stride=stride,
                    window=window,
                )
                stats = summarize(trades, equity_curve)
                label = f"enhanced={'ON' if enhanced else 'OFF'} method={method}"
                print(
                    f"{label:32s} trades={stats['trades']:4d} "
                    f"pnl={stats['pnl']:+.2f} win={stats['win_rate']:.1%} "
                    f"maxDD={stats['max_dd']:.2f}"
                )


if __name__ == "__main__":
    run()
