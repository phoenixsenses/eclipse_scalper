#!/usr/bin/env python3
"""Scalper Stack parameter training — deterministic walk-forward backtest.

Mirrors tools/pine/scalper_stack_v1.pine logic against local agg_trades data:
  - 1m bars built from data/microstructure.db agg_trades (true taker delta)
  - daily-anchored VWAP (UTC), EMA fast/slow, ATR(14) Wilder
  - signal: price beyond VWAP +/- buffer*ATR, EMA separation >= min*ATR,
    alignment held N consecutive bars, delta confirmation, cooldown
  - entry at NEXT bar open (signal_idx < entry_idx < exit_idx, DAT-02)
  - exits: fixed horizon bars, or ATR stop/target with timeout (SL-first)
  - walk-forward: folds 0..k-2 train selection, last fold untouched holdout

Deterministic: no randomness; --seed is echoed into outputs for protocol
compliance. Same DB + same args => identical outputs (DAT-03 / VAL-02).

Usage:
  python tools/backtest_scalper_stack.py --symbol ETHUSDT
  python tools/backtest_scalper_stack.py --symbol BTCUSDT --fee-bps 2.0
"""

from __future__ import annotations

import argparse
import itertools
import json
import sqlite3
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

MS_PER_MIN = 60_000
GAP_TOLERANCE_MIN = 2  # bars further apart than this break run/entry continuity


# ──────────────────────────────────────────────────────────────────────
# Data loading: 1m bars from agg_trades
# ──────────────────────────────────────────────────────────────────────

def load_bars(db_path: str, symbol: str) -> dict:
    """Aggregate agg_trades into 1m bars with true taker delta.

    Returns dict of numpy arrays: ts_min, open, high, low, close, volume,
    delta (taker buy qty - taker sell qty).
    """
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=60.0)
    cur = con.cursor()
    rows = cur.execute(
        """
        SELECT ts_ms/60000 AS m,
               MIN(price) AS lo,
               MAX(price) AS hi,
               SUM(quantity) AS vol,
               SUM(CASE WHEN is_buyer_maker=0 THEN quantity ELSE -quantity END) AS delta,
               MIN(id) AS first_id,
               MAX(id) AS last_id,
               COUNT(*) AS n
        FROM agg_trades
        WHERE symbol = ?
        GROUP BY m
        ORDER BY m
        """,
        (symbol,),
    ).fetchall()
    if not rows:
        con.close()
        raise SystemExit(f"no agg_trades rows for symbol {symbol} in {db_path}")

    # open/close from first/last trade id per minute (insertion order follows stream)
    ids = []
    for r in rows:
        ids.append(r[5])
        ids.append(r[6])
    price_by_id = {}
    CHUNK = 900
    for i in range(0, len(ids), CHUNK):
        chunk = ids[i : i + CHUNK]
        q = ",".join("?" * len(chunk))
        for rid, price in cur.execute(
            f"SELECT id, price FROM agg_trades WHERE id IN ({q})", chunk
        ):
            price_by_id[rid] = price
    con.close()

    n = len(rows)
    out = {
        "ts_min": np.empty(n, dtype=np.int64),
        "open": np.empty(n),
        "high": np.empty(n),
        "low": np.empty(n),
        "close": np.empty(n),
        "volume": np.empty(n),
        "delta": np.empty(n),
        "trades": np.empty(n, dtype=np.int64),
    }
    for i, (m, lo, hi, vol, delta, fid, lid, cnt) in enumerate(rows):
        out["ts_min"][i] = m
        out["open"][i] = price_by_id[fid]
        out["high"][i] = hi
        out["low"][i] = lo
        out["close"][i] = price_by_id[lid]
        out["volume"][i] = vol
        out["delta"][i] = delta
        out["trades"][i] = cnt
    return out


# ──────────────────────────────────────────────────────────────────────
# Indicators (match Pine conventions)
# ──────────────────────────────────────────────────────────────────────

def ema(x: np.ndarray, length: int) -> np.ndarray:
    alpha = 2.0 / (length + 1.0)
    out = np.empty_like(x)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = alpha * x[i] + (1.0 - alpha) * out[i - 1]
    return out


def sma(x: np.ndarray, length: int) -> np.ndarray:
    out = np.full_like(x, np.nan)
    c = np.cumsum(np.insert(x, 0, 0.0))
    out[length - 1 :] = (c[length:] - c[:-length]) / length
    # warmup: expanding mean so early bars are usable
    for i in range(min(length - 1, len(x))):
        out[i] = c[i + 1] / (i + 1)
    return out


def atr_wilder(high: np.ndarray, low: np.ndarray, close: np.ndarray, length: int = 14) -> np.ndarray:
    n = len(close)
    tr = np.empty(n)
    tr[0] = high[0] - low[0]
    prev_close = close[:-1]
    tr[1:] = np.maximum(
        high[1:] - low[1:],
        np.maximum(np.abs(high[1:] - prev_close), np.abs(low[1:] - prev_close)),
    )
    out = np.empty(n)
    out[0] = tr[0]
    alpha = 1.0 / length
    for i in range(1, n):
        out[i] = alpha * tr[i] + (1.0 - alpha) * out[i - 1]
    return out


def daily_anchored_vwap(ts_min: np.ndarray, high: np.ndarray, low: np.ndarray,
                        close: np.ndarray, volume: np.ndarray) -> np.ndarray:
    """VWAP of hlc3, reset at each UTC day boundary (Pine timeframe.change('D'))."""
    hlc3 = (high + low + close) / 3.0
    day = ts_min // (24 * 60)
    out = np.empty_like(close)
    cum_pv = 0.0
    cum_v = 0.0
    cur_day = -1
    for i in range(len(close)):
        if day[i] != cur_day:
            cur_day = day[i]
            cum_pv = 0.0
            cum_v = 0.0
        v = volume[i] if volume[i] > 0 else 1e-12
        cum_pv += hlc3[i] * v
        cum_v += v
        out[i] = cum_pv / cum_v
    return out


# ──────────────────────────────────────────────────────────────────────
# Signal + trade simulation
# ──────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Config:
    fast_len: int
    slow_len: int
    vwap_buf_atr: float
    trend_min_atr: float
    confirm_bars: int
    delta_mode: str      # "any" | "strong<mult>"
    cooldown_bars: int
    exit_mode: str       # "h<bars>" | "atr"
    side_filter: str = "both"  # "both" | "long" | "short"


def generate_signals(bars: dict, ind: dict, cfg: Config) -> list[tuple[int, int]]:
    """Return [(bar_idx, side)] with side +1 long / -1 short.

    Mirrors Pine: longRun==confirmBars fires once, cooldown applies across
    both sides. Data gaps (> GAP_TOLERANCE_MIN) reset alignment runs.
    """
    close = bars["close"]
    ts = bars["ts_min"]
    vwap = ind["vwap"]
    ema_f = ind[f"ema{cfg.fast_len}"]
    ema_s = ind[f"ema{cfg.slow_len}"]
    atr = ind["atr"]
    delta = bars["delta"]
    avg_abs_delta = ind["avg_abs_delta"]

    if cfg.delta_mode == "any":
        d_long = delta > 0
        d_short = delta < 0
    else:
        mult = float(cfg.delta_mode.replace("strong", ""))
        d_long = delta > mult * avg_abs_delta
        d_short = delta < -mult * avg_abs_delta

    long_base = (close > vwap + cfg.vwap_buf_atr * atr) & (ema_f > ema_s) & \
                ((ema_f - ema_s) > cfg.trend_min_atr * atr)
    short_base = (close < vwap - cfg.vwap_buf_atr * atr) & (ema_f < ema_s) & \
                 ((ema_s - ema_f) > cfg.trend_min_atr * atr)

    signals = []
    long_run = 0
    short_run = 0
    last_sig = -10**9
    for i in range(1, len(close)):
        if ts[i] - ts[i - 1] > GAP_TOLERANCE_MIN:
            long_run = 0
            short_run = 0
        long_run = long_run + 1 if long_base[i] else 0
        short_run = short_run + 1 if short_base[i] else 0
        cooled = (i - last_sig) >= cfg.cooldown_bars
        allow_long = cfg.side_filter in ("both", "long")
        allow_short = cfg.side_filter in ("both", "short")
        if allow_long and long_run == cfg.confirm_bars and d_long[i] and cooled:
            signals.append((i, 1))
            last_sig = i
        elif allow_short and short_run == cfg.confirm_bars and d_short[i] and cooled:
            signals.append((i, -1))
            last_sig = i
    return signals


def simulate_trades(bars: dict, ind: dict, signals: list[tuple[int, int]],
                    cfg: Config, fee_bps: float, slip_bps: float) -> list[dict]:
    """Entry at next bar open; exit per cfg.exit_mode. Returns trade dicts."""
    o, h, l, c, ts = bars["open"], bars["high"], bars["low"], bars["close"], bars["ts_min"]
    atr = ind["atr"]
    n = len(o)
    cost_bps = 2.0 * (fee_bps + slip_bps)
    trades = []
    for sig_idx, side in signals:
        entry_idx = sig_idx + 1
        if entry_idx >= n or ts[entry_idx] - ts[sig_idx] > GAP_TOLERANCE_MIN:
            continue
        entry = o[entry_idx]
        exit_idx = -1
        exit_px = np.nan

        if cfg.exit_mode.startswith("h"):
            horizon = int(cfg.exit_mode[1:])
            target = entry_idx + horizon
            if target >= n:
                continue  # not enough forward data; drop, don't peek
            exit_idx = target
            exit_px = o[exit_idx]
        else:  # ATR stop/target, conservative: SL checked before TP each bar
            sl_mult, tp_mult, timeout = 1.2, 1.8, 30
            a = atr[sig_idx]
            sl = entry - side * sl_mult * a
            tp = entry + side * tp_mult * a
            last = min(entry_idx + timeout, n - 1)
            for j in range(entry_idx, last + 1):
                if side > 0:
                    if l[j] <= sl:
                        exit_idx, exit_px = j, sl
                        break
                    if h[j] >= tp:
                        exit_idx, exit_px = j, tp
                        break
                else:
                    if h[j] >= sl:
                        exit_idx, exit_px = j, sl
                        break
                    if l[j] <= tp:
                        exit_idx, exit_px = j, tp
                        break
            if exit_idx < 0:
                if last <= entry_idx:
                    continue
                exit_idx, exit_px = last, c[last]

        gross_bps = side * (exit_px / entry - 1.0) * 10_000.0
        trades.append({
            "signal_idx": int(sig_idx),
            "entry_idx": int(entry_idx),
            "exit_idx": int(exit_idx),
            "side": int(side),
            "gross_bps": float(gross_bps),
            "net_bps": float(gross_bps - cost_bps),
        })
    return trades


# ──────────────────────────────────────────────────────────────────────
# Walk-forward evaluation
# ──────────────────────────────────────────────────────────────────────

def fold_of(idx: int, edges: list[int]) -> int:
    for k in range(len(edges) - 1):
        if edges[k] <= idx < edges[k + 1]:
            return k
    return len(edges) - 2


def evaluate(bars: dict, ind: dict, cfg: Config, fee_bps: float, slip_bps: float,
             edges: list[int]) -> dict:
    signals = generate_signals(bars, ind, cfg)
    trades = simulate_trades(bars, ind, signals, cfg, fee_bps, slip_bps)
    n_folds = len(edges) - 1
    folds = [[] for _ in range(n_folds)]
    for t in trades:
        folds[fold_of(t["signal_idx"], edges)].append(t["net_bps"])
    fold_stats = []
    for f in folds:
        fold_stats.append({
            "trades": len(f),
            "net_mean_bps": float(np.mean(f)) if f else None,
            "win_rate": float(np.mean([1.0 if x > 0 else 0.0 for x in f])) if f else None,
        })
    train = [x for f in folds[:-1] for x in f]
    hold = folds[-1]
    return {
        "config": asdict(cfg),
        "signals": len(signals),
        "trades": len(trades),
        "train_trades": len(train),
        "train_net_mean_bps": float(np.mean(train)) if train else None,
        "train_win_rate": float(np.mean([1.0 if x > 0 else 0.0 for x in train])) if train else None,
        "train_pos_folds": sum(1 for fs in fold_stats[:-1] if fs["trades"] > 0 and fs["net_mean_bps"] is not None and fs["net_mean_bps"] > 0),
        "holdout_trades": len(hold),
        "holdout_net_mean_bps": float(np.mean(hold)) if hold else None,
        "holdout_win_rate": float(np.mean([1.0 if x > 0 else 0.0 for x in hold])) if hold else None,
        "fold_stats": fold_stats,
    }


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db-path", default="data/microstructure.db")
    ap.add_argument("--symbol", default="ETHUSDT")
    ap.add_argument("--fee-bps", type=float, default=5.0, help="taker fee per side")
    ap.add_argument("--slip-bps", type=float, default=1.0, help="slippage per side")
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--min-train-trades", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42, help="echoed for reproducibility protocol (no randomness used)")
    ap.add_argument("--out-dir", default="reports/research/scalper_stack")
    ap.add_argument("--profile", choices=["base", "deep"], default="base",
                    help="base = broad grid; deep = stricter filters, longer horizons, side split")
    args = ap.parse_args()

    bars = load_bars(args.db_path, args.symbol)
    n = len(bars["close"])
    ts = bars["ts_min"]
    gaps = int(np.sum(np.diff(ts) > GAP_TOLERANCE_MIN))
    t0 = datetime.fromtimestamp(ts[0] * 60, tz=timezone.utc)
    t1 = datetime.fromtimestamp(ts[-1] * 60, tz=timezone.utc)

    # Precompute shared indicators
    fast_lens = [5, 9, 12]
    slow_lens = [13, 21, 26]
    ind = {
        "vwap": daily_anchored_vwap(ts, bars["high"], bars["low"], bars["close"], bars["volume"]),
        "atr": atr_wilder(bars["high"], bars["low"], bars["close"], 14),
        "avg_abs_delta": sma(np.abs(bars["delta"]), 20),
    }
    for ln in sorted(set(fast_lens + slow_lens)):
        ind[f"ema{ln}"] = ema(bars["close"], ln)

    edges = [int(round(k * n / args.folds)) for k in range(args.folds)] + [n]

    if args.profile == "base":
        grid = list(itertools.product(
            [(5, 13), (9, 21), (12, 26)],
            [0.0, 0.15, 0.30, 0.50],        # vwap_buf_atr
            [0.0, 0.20, 0.40],              # trend_min_atr
            [1, 2, 3],                      # confirm_bars
            ["any", "strong1.2", "strong1.5"],
            [5, 15],                        # cooldown_bars
            ["h5", "h15", "h30", "atr"],    # exit_mode
            ["both"],                       # side_filter
        ))
    else:  # deep: chase the "stricter is better" gradient from the base run
        grid = list(itertools.product(
            [(5, 13), (9, 21)],
            [0.50, 0.75, 1.00],             # vwap_buf_atr
            [0.40, 0.60],                   # trend_min_atr
            [3, 4],                         # confirm_bars
            ["strong1.5", "strong2.0"],
            [15],                           # cooldown_bars
            ["h30", "h60", "h120"],         # exit_mode
            ["both", "long", "short"],      # side_filter
        ))

    results = []
    for (fl, sl), buf, tmin, conf, dmode, cool, emode, side in grid:
        cfg = Config(fl, sl, buf, tmin, conf, dmode, cool, emode, side)
        results.append(evaluate(bars, ind, cfg, args.fee_bps, args.slip_bps, edges))

    # Selection on train folds only; holdout reported, never used for ranking
    eligible = [r for r in results
                if r["train_trades"] >= args.min_train_trades
                and r["train_net_mean_bps"] is not None
                and r["train_pos_folds"] >= args.folds - 2]
    eligible.sort(key=lambda r: (-(r["train_net_mean_bps"] or -1e9), -r["train_trades"],
                                 json.dumps(r["config"], sort_keys=True)))
    top = eligible[:10]

    all_sorted = sorted(results, key=lambda r: (-(r["train_net_mean_bps"] if r["train_net_mean_bps"] is not None else -1e9),
                                                json.dumps(r["config"], sort_keys=True)))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = t1.strftime("%Y-%m-%d")
    payload = {
        "tool": "backtest_scalper_stack",
        "seed": args.seed,
        "config_echo": vars(args),
        "data": {
            "symbol": args.symbol,
            "bars": n,
            "gaps_gt_tolerance": gaps,
            "range_utc": [t0.isoformat(), t1.isoformat()],
            "fold_edges_bar_idx": edges,
        },
        "grid_size": len(grid),
        "eligible_configs": len(eligible),
        "top_by_train": top,
        "best_overall_train_unfiltered": all_sorted[:3],
    }
    json_path = out_dir / f"SCALPER_STACK_TRAINING_{args.symbol}_{args.profile}_{stamp}.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        f"# Scalper Stack Training — {args.symbol} ({stamp})",
        "",
        f"- seed: {args.seed} (deterministic; no randomness used)",
        f"- config: fee={args.fee_bps} bps/side, slip={args.slip_bps} bps/side, folds={args.folds}, min_train_trades={args.min_train_trades}",
        f"- data: {n} one-minute bars, {gaps} gaps > {GAP_TOLERANCE_MIN} min, range {t0:%Y-%m-%d %H:%M} → {t1:%Y-%m-%d %H:%M} UTC",
        f"- grid: {len(grid)} configs, eligible after train filters: {len(eligible)}",
        "",
        "Selection uses folds 1..N-1 (train) only. The final fold is holdout —",
        "reported for the top configs but never used for ranking.",
        "",
        "| rank | ema | buf | sep | hold | delta | cool | exit | side | train N | train bps | train WR | hold N | hold bps | hold WR |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for rank, r in enumerate(top, 1):
        cfg = r["config"]
        lines.append(
            f"| {rank} | {cfg['fast_len']}/{cfg['slow_len']} | {cfg['vwap_buf_atr']} | {cfg['trend_min_atr']} "
            f"| {cfg['confirm_bars']} | {cfg['delta_mode']} | {cfg['cooldown_bars']} | {cfg['exit_mode']} | {cfg['side_filter']} "
            f"| {r['train_trades']} | {r['train_net_mean_bps']:.2f} | {r['train_win_rate']:.0%} "
            f"| {r['holdout_trades']} | "
            f"{'-' if r['holdout_net_mean_bps'] is None else format(r['holdout_net_mean_bps'], '.2f')} | "
            f"{'-' if r['holdout_win_rate'] is None else format(r['holdout_win_rate'], '.0%')} |"
        )
    if not top:
        lines.append("")
        lines.append("**No config passed the train filters** (min trades + fold consistency).")
    md_path = out_dir / f"SCALPER_STACK_TRAINING_{args.symbol}_{args.profile}_{stamp}.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"bars={n} gaps={gaps} range={t0:%Y-%m-%d}..{t1:%Y-%m-%d} grid={len(grid)} eligible={len(eligible)}")
    if top:
        b = top[0]
        print("best-train:", json.dumps(b["config"]))
        print(f"  train: N={b['train_trades']} net={b['train_net_mean_bps']:.2f}bps WR={b['train_win_rate']:.0%}")
        hb = b["holdout_net_mean_bps"]
        print(f"  holdout: N={b['holdout_trades']} net={'-' if hb is None else format(hb, '.2f')}bps")
    print(f"wrote {md_path}")
    print(f"wrote {json_path}")


if __name__ == "__main__":
    sys.exit(main())
