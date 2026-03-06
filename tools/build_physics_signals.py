from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from utils.symbols import canonical_symbol


EPS = 1e-12


@dataclass(frozen=True)
class PhysicsPartitionManifest:
    symbol: str
    date: str
    interval_ms: int
    rows: int
    ts_min: int | None
    ts_max: int | None


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build microstructure physics signals from state parquet.")
    p.add_argument("--state", default="data/derived/state")
    p.add_argument("--symbol", required=True)
    p.add_argument("--interval-ms", type=int, default=100)
    p.add_argument("--rolling", type=int, default=500)
    p.add_argument("--horizons", default="1,5,10,20")
    p.add_argument("--out", default="data/derived/physics")
    return p.parse_args()


def _parse_horizons(raw: str) -> list[int]:
    vals = [int(x.strip()) for x in str(raw).split(",") if x.strip()]
    uniq = sorted({v for v in vals if v > 0})
    return uniq or [1, 5, 10, 20]


def _rolling_z(series: pd.Series, window: int) -> pd.Series:
    min_p = max(2, min(int(window), max(20, int(window) // 10)))
    mu = series.rolling(window, min_periods=min_p).mean()
    sd = series.rolling(window, min_periods=min_p).std(ddof=0)
    return (series - mu) / (sd + EPS)


def compute_physics_signals_frame(df: pd.DataFrame, horizons: list[int], rolling: int) -> pd.DataFrame:
    out = df.copy()
    out = out.sort_values("ts_ms").reset_index(drop=True)

    mid = pd.to_numeric(out.get("mid"), errors="coerce").replace(0.0, np.nan)

    for h in horizons:
        out[f"r_{h}"] = np.log(mid.shift(-h) / mid)

    out["velocity"] = out.get("r_1", 0.0)
    out["acceleration"] = pd.to_numeric(out["velocity"], errors="coerce") - pd.to_numeric(out["velocity"], errors="coerce").shift(1)

    out["F_ofi"] = pd.to_numeric(out.get("ofi"), errors="coerce").fillna(0.0)
    out["F_ofi_z"] = _rolling_z(out["F_ofi"], rolling)

    out["F_intensity"] = pd.to_numeric(out.get("trade_intensity"), errors="coerce").fillna(0.0)
    out["F_intensity_z"] = _rolling_z(out["F_intensity"], rolling)

    out["F_liquidity"] = pd.to_numeric(out.get("top_depth_imbalance"), errors="coerce").fillna(0.0)
    out["F_liquidity_delta"] = out["F_liquidity"].diff()

    out["spread_z"] = _rolling_z(pd.to_numeric(out.get("spread"), errors="coerce").fillna(0.0), rolling)

    min_p = max(2, min(int(rolling), max(20, int(rolling) // 10)))
    spread_q20 = out["spread"].rolling(rolling, min_periods=min_p).quantile(0.2)
    spread_q80 = out["spread"].rolling(rolling, min_periods=min_p).quantile(0.8)
    depth_abs = out["F_liquidity"].abs()
    depth_q20 = depth_abs.rolling(rolling, min_periods=min_p).quantile(0.2)

    intensity_delta = out["F_intensity"].diff()
    out["compression_flag"] = ((out["spread"] <= spread_q20) & (intensity_delta > 0)).fillna(False)
    out["vacuum_flag"] = ((depth_abs <= depth_q20) & (out["spread"] >= spread_q80)).fillna(False)

    if "liq_rate" in out.columns:
        liq_rate = pd.to_numeric(out["liq_rate"], errors="coerce").fillna(0.0)
    else:
        liq_rate = pd.Series(np.zeros(len(out), dtype=float), index=out.index)
    out["liq_rate"] = liq_rate
    out["liq_rate_z"] = _rolling_z(liq_rate, rolling)
    liq_q90 = liq_rate.rolling(rolling, min_periods=min_p).quantile(0.9)
    out["liq_burst_flag"] = ((liq_rate >= liq_q90) & (out["liq_rate_z"] > 1.0)).fillna(False)

    qty_proxy = pd.to_numeric(out.get("qty_sum"), errors="coerce").fillna(0.0)
    out["volume_proxy"] = qty_proxy

    keep_cols = [
        "ts_ms",
        "ts_utc",
        "symbol",
        "mid",
        "microprice",
        "spread",
        "ofi",
        "trade_intensity",
        "top_depth_imbalance",
        "rv_short",
        "liq_rate",
        "qty_sum",
        "trade_count",
        "volume_proxy",
        "velocity",
        "acceleration",
        "F_ofi",
        "F_ofi_z",
        "F_intensity",
        "F_intensity_z",
        "F_liquidity",
        "F_liquidity_delta",
        "spread_z",
        "compression_flag",
        "vacuum_flag",
        "liq_rate_z",
        "liq_burst_flag",
    ] + [f"r_{h}" for h in horizons]

    for c in keep_cols:
        if c not in out.columns:
            out[c] = np.nan
    out = out[keep_cols].copy()
    out["symbol"] = out["symbol"].astype(str)
    return out


def _iter_state_partitions(state_root: Path, symbol: str, interval_ms: int) -> list[Path]:
    base = state_root / f"interval_ms={int(interval_ms)}" / f"symbol={symbol}"
    if not base.exists():
        return []
    return sorted(base.glob("date=*/state.parquet"))


def build_physics_signals(
    state_root: Path,
    out_root: Path,
    symbol: str,
    interval_ms: int,
    horizons: list[int],
    rolling: int,
) -> dict[str, Any]:
    sym = canonical_symbol(symbol)
    input_files = _iter_state_partitions(state_root, sym, interval_ms)
    if not input_files:
        raise RuntimeError(f"no_state_partitions symbol={sym} interval_ms={interval_ms}")

    out_base = out_root / f"interval_ms={int(interval_ms)}" / f"symbol={sym}"
    out_base.mkdir(parents=True, exist_ok=True)

    partitions: list[PhysicsPartitionManifest] = []
    for inp in input_files:
        date_tag = inp.parent.name
        day = date_tag.split("=", 1)[1] if "=" in date_tag else date_tag
        df = pd.read_parquet(inp)
        phys = compute_physics_signals_frame(df, horizons=horizons, rolling=rolling)
        day_dir = out_base / f"date={day}"
        day_dir.mkdir(parents=True, exist_ok=True)
        out_pq = day_dir / "physics.parquet"
        phys.to_parquet(out_pq, index=False)

        man = PhysicsPartitionManifest(
            symbol=sym,
            date=day,
            interval_ms=int(interval_ms),
            rows=int(len(phys)),
            ts_min=int(phys["ts_ms"].min()) if len(phys) else None,
            ts_max=int(phys["ts_ms"].max()) if len(phys) else None,
        )
        partitions.append(man)
        (day_dir / "manifest.json").write_text(
            json.dumps({
                **asdict(man),
                "input": str(inp),
                "output": str(out_pq),
                "rolling": int(rolling),
                "horizons": horizons,
            }, ensure_ascii=True, sort_keys=True, indent=2)
            + "\n",
            encoding="utf-8",
        )

    run_manifest = {
        "symbol": sym,
        "interval_ms": int(interval_ms),
        "rolling": int(rolling),
        "horizons": horizons,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "partitions": [asdict(p) for p in partitions],
    }
    (out_base / "manifest.json").write_text(
        json.dumps(run_manifest, ensure_ascii=True, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return run_manifest


def main() -> int:
    args = _parse_args()
    try:
        manifest = build_physics_signals(
            state_root=Path(str(args.state)),
            out_root=Path(str(args.out)),
            symbol=str(args.symbol),
            interval_ms=int(args.interval_ms),
            horizons=_parse_horizons(args.horizons),
            rolling=int(args.rolling),
        )
        rows = sum(int(x["rows"]) for x in manifest["partitions"])
        print(f"build_physics_signals ok symbol={manifest['symbol']} partitions={len(manifest['partitions'])} rows={rows}")
        return 0
    except Exception as e:
        print(f"build_physics_signals error runtime={type(e).__name__}:{e}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
