from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.microphys.io.sqlite_reader import SQLiteMicroReader
from utils.symbols import canonical_symbol


EPS = 1e-12


@dataclass(frozen=True)
class BuildWindowResult:
    start_ts: float
    end_ts: float
    rows: int


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build microstructure bars parquet from sqlite events.")
    p.add_argument("--db", default="data/microstructure.db")
    p.add_argument("--symbol", required=True)
    p.add_argument("--interval-ms", type=int, default=100)
    p.add_argument("--out", default="data/derived/micro_bars")
    p.add_argument("--start-ts", type=float, default=None)
    p.add_argument("--end-ts", type=float, default=None)
    p.add_argument("--window-sec", type=int, default=86400)
    p.add_argument("--rv-window-sec", type=float, default=5.0)
    return p.parse_args()


def _to_df_trades(rows: list[Any]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=["ts", "symbol", "price", "qty", "side"])
    return pd.DataFrame([asdict(r) for r in rows])


def _to_df_book(rows: list[Any]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=["ts", "symbol", "bid_px", "bid_qty", "ask_px", "ask_qty"])
    return pd.DataFrame([asdict(r) for r in rows])


def _to_df_liq(rows: list[Any]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=["ts", "symbol", "side", "qty", "price"])
    return pd.DataFrame([asdict(r) for r in rows])


def _bucketize(ts: pd.Series, interval_ms: int) -> pd.Series:
    step = float(interval_ms) / 1000.0
    return (np.floor(ts.astype(float).to_numpy() / step) * step).astype(float)


def compute_micro_bars_from_frames(
    trades_df: pd.DataFrame,
    book_df: pd.DataFrame,
    liq_df: pd.DataFrame,
    *,
    symbol: str,
    start_ts: float,
    end_ts: float,
    interval_ms: int,
    rv_window_sec: float = 5.0,
) -> pd.DataFrame:
    sym = canonical_symbol(symbol)
    step = float(interval_ms) / 1000.0
    if end_ts <= start_ts:
        return pd.DataFrame()

    # deterministic bucket grid
    buckets = np.arange(start_ts, end_ts + (step * 0.5), step, dtype=float)
    out = pd.DataFrame({"bucket_ts": buckets})
    out["symbol"] = sym

    # Trades aggregation
    t = trades_df.copy()
    if not t.empty:
        t = t[t["symbol"].astype(str).str.upper() == sym]
        t = t[(t["ts"].astype(float) >= start_ts) & (t["ts"].astype(float) <= end_ts)]
    if not t.empty:
        t["bucket_ts"] = _bucketize(t["ts"], interval_ms)
        t["qty"] = pd.to_numeric(t["qty"], errors="coerce").fillna(0.0)
        t["price"] = pd.to_numeric(t["price"], errors="coerce")
        t["notional"] = t["qty"] * t["price"].fillna(0.0)
        side = t["side"].fillna("").astype(str).str.lower()
        t["buy_qty"] = np.where(side.eq("buy"), t["qty"], 0.0)
        t["sell_qty"] = np.where(side.eq("sell"), t["qty"], 0.0)

        tg = (
            t.groupby("bucket_ts", as_index=False)
            .agg(
                buy_qty=("buy_qty", "sum"),
                sell_qty=("sell_qty", "sum"),
                qty_sum=("qty", "sum"),
                trade_count=("qty", "size"),
                notional_sum=("notional", "sum"),
                price_last=("price", "last"),
            )
            .sort_values("bucket_ts")
        )
        tg["vwap"] = tg["notional_sum"] / (tg["qty_sum"] + EPS)
        out = out.merge(tg, on="bucket_ts", how="left")
    else:
        out["buy_qty"] = 0.0
        out["sell_qty"] = 0.0
        out["qty_sum"] = 0.0
        out["trade_count"] = 0
        out["notional_sum"] = 0.0
        out["price_last"] = np.nan
        out["vwap"] = np.nan

    # Top-of-book / mark aggregation
    b = book_df.copy()
    if not b.empty:
        b = b[b["symbol"].astype(str).str.upper() == sym]
        b = b[(b["ts"].astype(float) >= start_ts) & (b["ts"].astype(float) <= end_ts)]
    if not b.empty:
        b["bucket_ts"] = _bucketize(b["ts"], interval_ms)
        for c in ("bid_px", "ask_px", "bid_qty", "ask_qty"):
            b[c] = pd.to_numeric(b[c], errors="coerce")
        bg = (
            b.groupby("bucket_ts", as_index=False)
            .agg(
                bid_px=("bid_px", "last"),
                ask_px=("ask_px", "last"),
                bid_qty=("bid_qty", "last"),
                ask_qty=("ask_qty", "last"),
            )
            .sort_values("bucket_ts")
        )
        out = out.merge(bg, on="bucket_ts", how="left")
    else:
        out["bid_px"] = np.nan
        out["ask_px"] = np.nan
        out["bid_qty"] = 0.0
        out["ask_qty"] = 0.0

    # Liquidation aggregation
    l = liq_df.copy()
    if not l.empty:
        l = l[l["symbol"].astype(str).str.upper() == sym]
        l = l[(l["ts"].astype(float) >= start_ts) & (l["ts"].astype(float) <= end_ts)]
    if not l.empty:
        l["bucket_ts"] = _bucketize(l["ts"], interval_ms)
        l["qty"] = pd.to_numeric(l["qty"], errors="coerce").fillna(0.0)
        side = l["side"].fillna("").astype(str).str.lower()
        l["liq_sell_qty"] = np.where(side.eq("sell"), l["qty"], 0.0)
        l["liq_buy_qty"] = np.where(side.eq("buy"), l["qty"], 0.0)
        lg = (
            l.groupby("bucket_ts", as_index=False)
            .agg(
                liq_count=("qty", "size"),
                liq_qty=("qty", "sum"),
                liq_sell_qty=("liq_sell_qty", "sum"),
                liq_buy_qty=("liq_buy_qty", "sum"),
            )
        )
        out = out.merge(lg, on="bucket_ts", how="left")
    else:
        out["liq_count"] = 0
        out["liq_qty"] = 0.0
        out["liq_sell_qty"] = 0.0
        out["liq_buy_qty"] = 0.0

    # Derived columns
    out[["buy_qty", "sell_qty", "qty_sum", "notional_sum", "liq_qty", "liq_sell_qty", "liq_buy_qty"]] = out[
        ["buy_qty", "sell_qty", "qty_sum", "notional_sum", "liq_qty", "liq_sell_qty", "liq_buy_qty"]
    ].fillna(0.0)
    out[["trade_count", "liq_count"]] = out[["trade_count", "liq_count"]].fillna(0).astype(int)

    out["mid"] = (out["bid_px"] + out["ask_px"]) / 2.0
    out["mid"] = out["mid"].where(out["mid"] > 0, np.nan)
    # fallback: no bid/ask available -> use vwap/last trade
    out["mid"] = out["mid"].fillna(out["vwap"]).fillna(out["price_last"]) 

    out["spread_abs"] = (out["ask_px"] - out["bid_px"]).where((out["ask_px"] > 0) & (out["bid_px"] > 0), np.nan)
    out["spread"] = out["spread_abs"] / (out["mid"].abs() + EPS)
    # if no bid/ask, proxy spread from vwap-vs-mid mismatch
    spread_proxy = (out["vwap"] - out["mid"]).abs() / (out["mid"].abs() + EPS)
    out["spread"] = out["spread"].fillna(spread_proxy).fillna(0.0)

    out["microprice"] = (
        out["ask_px"] * out["bid_qty"] + out["bid_px"] * out["ask_qty"]
    ) / (out["bid_qty"] + out["ask_qty"] + EPS)
    out["microprice"] = out["microprice"].where(out["microprice"] > 0, out["mid"])

    out["ofi"] = out["buy_qty"] - out["sell_qty"]
    out["ofi_norm"] = out["ofi"] / (out["buy_qty"] + out["sell_qty"] + EPS)

    interval_sec = float(interval_ms) / 1000.0
    out["trade_intensity_qty_per_sec"] = out["qty_sum"] / max(EPS, interval_sec)
    out["trade_intensity_trades_per_sec"] = out["trade_count"] / max(EPS, interval_sec)
    out["top_depth_imbalance"] = (out["bid_qty"] - out["ask_qty"]) / (out["bid_qty"] + out["ask_qty"] + EPS)
    out["liq_imbalance"] = (out["liq_sell_qty"] - out["liq_buy_qty"]) / (out["liq_qty"] + EPS)
    out["liq_rate_per_sec"] = out["liq_qty"] / max(EPS, interval_sec)

    out["mid_ret"] = np.log((out["mid"].replace(0.0, np.nan)) / (out["mid"].shift(1).replace(0.0, np.nan)))
    rv_window = max(2, int(round(float(rv_window_sec) / interval_sec)))
    out["rv_short"] = np.sqrt((out["mid_ret"].fillna(0.0) ** 2).rolling(rv_window, min_periods=1).sum())

    out["ts_ms"] = (out["bucket_ts"] * 1000.0).round().astype(np.int64)
    out["ts_utc"] = pd.to_datetime(out["ts_ms"], unit="ms", utc=True).dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")

    cols = [
        "ts_ms",
        "ts_utc",
        "symbol",
        "mid",
        "spread",
        "microprice",
        "buy_qty",
        "sell_qty",
        "trade_count",
        "qty_sum",
        "vwap",
        "ofi",
        "ofi_norm",
        "trade_intensity_qty_per_sec",
        "trade_intensity_trades_per_sec",
        "top_depth_imbalance",
        "rv_short",
        "liq_count",
        "liq_qty",
        "liq_sell_qty",
        "liq_buy_qty",
        "liq_imbalance",
        "liq_rate_per_sec",
        "bid_px",
        "ask_px",
        "bid_qty",
        "ask_qty",
    ]
    out = out[cols].sort_values("ts_ms").reset_index(drop=True)
    return out


def _iter_windows(start_ts: float, end_ts: float, window_sec: int):
    cur = float(start_ts)
    step = float(max(1, int(window_sec)))
    while cur < end_ts:
        nxt = min(end_ts, cur + step)
        yield cur, nxt
        cur = nxt


def build_micro_features(
    db_path: Path,
    out_root: Path,
    symbol: str,
    interval_ms: int,
    window_sec: int,
    start_ts: float | None,
    end_ts: float | None,
    rv_window_sec: float,
) -> dict[str, Any]:
    reader = SQLiteMicroReader(db_path)
    sym = canonical_symbol(symbol)

    t_min, t_max = reader.get_ts_range("trades", sym)
    b_min, b_max = reader.get_ts_range("book", sym)

    inferred_start = min(x for x in (t_min, b_min) if x is not None) if any(x is not None for x in (t_min, b_min)) else None
    inferred_end = max(x for x in (t_max, b_max) if x is not None) if any(x is not None for x in (t_max, b_max)) else None

    if inferred_start is None or inferred_end is None:
        raise RuntimeError(f"no_data_for_symbol={sym}")

    s_ts = float(start_ts if start_ts is not None else inferred_start)
    e_ts = float(end_ts if end_ts is not None else inferred_end)
    if e_ts <= s_ts:
        raise RuntimeError("invalid_time_range")

    out_root = out_root / f"interval_ms={int(interval_ms)}" / f"symbol={sym}"
    out_root.mkdir(parents=True, exist_ok=True)

    per_date_frames: dict[str, list[pd.DataFrame]] = {}
    windows: list[BuildWindowResult] = []

    for w0, w1 in _iter_windows(s_ts, e_ts, window_sec):
        trades = _to_df_trades(reader.read_trades(sym, w0, w1))
        book = _to_df_book(reader.read_top_of_book(sym, w0, w1))
        liq = _to_df_liq(reader.read_liquidations(sym, w0, w1))
        bars = compute_micro_bars_from_frames(
            trades,
            book,
            liq,
            symbol=sym,
            start_ts=w0,
            end_ts=w1,
            interval_ms=interval_ms,
            rv_window_sec=rv_window_sec,
        )
        rows = int(len(bars))
        windows.append(BuildWindowResult(start_ts=w0, end_ts=w1, rows=rows))
        if rows <= 0:
            print(f"[build_micro_features] window {w0:.0f}-{w1:.0f} rows=0")
            continue
        bars["date"] = pd.to_datetime(bars["ts_ms"], unit="ms", utc=True).dt.strftime("%Y-%m-%d")
        for date_val, g in bars.groupby("date", sort=True):
            per_date_frames.setdefault(str(date_val), []).append(g.drop(columns=["date"]).copy())
        print(f"[build_micro_features] window {w0:.0f}-{w1:.0f} rows={rows}")

    manifests: list[dict[str, Any]] = []
    for date_val in sorted(per_date_frames):
        day_dir = out_root / f"date={date_val}"
        day_dir.mkdir(parents=True, exist_ok=True)
        day_df = pd.concat(per_date_frames[date_val], ignore_index=True)
        day_df = day_df.sort_values("ts_ms").drop_duplicates(subset=["ts_ms", "symbol"], keep="last")
        pq_path = day_dir / "bars.parquet"
        day_df.to_parquet(pq_path, index=False)

        manifest = {
            "symbol": sym,
            "date": date_val,
            "interval_ms": int(interval_ms),
            "rows": int(len(day_df)),
            "ts_min": int(day_df["ts_ms"].min()) if len(day_df) else None,
            "ts_max": int(day_df["ts_ms"].max()) if len(day_df) else None,
            "params": {
                "db": str(db_path),
                "window_sec": int(window_sec),
                "rv_window_sec": float(rv_window_sec),
            },
        }
        (day_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=True, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        manifests.append(manifest)

    run_manifest = {
        "symbol": sym,
        "interval_ms": int(interval_ms),
        "start_ts": s_ts,
        "end_ts": e_ts,
        "windows": [asdict(w) for w in windows],
        "dates": manifests,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    (out_root / "manifest.json").write_text(
        json.dumps(run_manifest, ensure_ascii=True, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return run_manifest


def main() -> int:
    args = _parse_args()
    try:
        manifest = build_micro_features(
            db_path=Path(str(args.db)),
            out_root=Path(str(args.out)),
            symbol=str(args.symbol),
            interval_ms=int(args.interval_ms),
            window_sec=int(args.window_sec),
            start_ts=args.start_ts,
            end_ts=args.end_ts,
            rv_window_sec=float(args.rv_window_sec),
        )
        dates = len(manifest.get("dates", []))
        rows = sum(int(x.get("rows", 0)) for x in manifest.get("dates", []))
        print(f"build_micro_features ok symbol={manifest['symbol']} dates={dates} rows={rows}")
        return 0
    except Exception as e:
        print(f"build_micro_features error runtime={type(e).__name__}:{e}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
