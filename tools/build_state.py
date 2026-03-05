from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from utils.symbols import canonical_symbol


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build state snapshots from micro bar parquet partitions.")
    p.add_argument("--in", dest="in_root", default="data/derived/micro_bars")
    p.add_argument("--symbol", required=True)
    p.add_argument("--interval-ms", type=int, default=100)
    p.add_argument("--out", default="data/derived/state")
    p.add_argument("--db", default="data/microstructure.db", help="reserved for future enrichments")
    return p.parse_args()


def _iter_input_partitions(in_root: Path, symbol: str, interval_ms: int) -> list[Path]:
    base = in_root / f"interval_ms={int(interval_ms)}" / f"symbol={symbol}"
    if not base.exists():
        return []
    out: list[Path] = []
    for day_dir in sorted(base.glob("date=*")):
        pq = day_dir / "bars.parquet"
        if pq.exists():
            out.append(pq)
    return out


def _build_state_frame(df: pd.DataFrame, interval_ms: int) -> pd.DataFrame:
    out = pd.DataFrame()
    out["ts_ms"] = df["ts_ms"].astype("int64")
    out["ts_utc"] = df["ts_utc"].astype(str)
    out["symbol"] = df["symbol"].astype(str)

    out["mid"] = pd.to_numeric(df.get("mid"), errors="coerce").fillna(0.0)
    out["microprice"] = pd.to_numeric(df.get("microprice"), errors="coerce").fillna(out["mid"])
    out["spread"] = pd.to_numeric(df.get("spread"), errors="coerce").fillna(0.0)

    out["ofi"] = pd.to_numeric(df.get("ofi"), errors="coerce").fillna(0.0)
    out["trade_intensity"] = pd.to_numeric(df.get("trade_intensity_qty_per_sec"), errors="coerce").fillna(0.0)
    out["top_depth_imbalance"] = pd.to_numeric(df.get("top_depth_imbalance"), errors="coerce").fillna(0.0)
    out["rv_short"] = pd.to_numeric(df.get("rv_short"), errors="coerce").fillna(0.0)

    liq_qty = pd.to_numeric(df.get("liq_qty"), errors="coerce").fillna(0.0)
    liq_count = pd.to_numeric(df.get("liq_count"), errors="coerce").fillna(0.0)
    sec = max(1e-9, float(interval_ms) / 1000.0)
    out["liq_rate"] = liq_qty / sec
    out["liq_count_rate"] = liq_count / sec

    out["trade_count"] = pd.to_numeric(df.get("trade_count"), errors="coerce").fillna(0).astype(int)
    out["qty_sum"] = pd.to_numeric(df.get("qty_sum"), errors="coerce").fillna(0.0)
    out = out.sort_values("ts_ms").reset_index(drop=True)
    return out


def build_state(in_root: Path, out_root: Path, symbol: str, interval_ms: int) -> dict[str, Any]:
    sym = canonical_symbol(symbol)
    input_files = _iter_input_partitions(in_root, sym, interval_ms)
    if not input_files:
        raise RuntimeError(f"no_input_partitions for symbol={sym} interval_ms={interval_ms}")

    out_base = out_root / f"interval_ms={int(interval_ms)}" / f"symbol={sym}"
    out_base.mkdir(parents=True, exist_ok=True)

    manifests: list[dict[str, Any]] = []
    for pq in input_files:
        day_dir_name = pq.parent.name
        day = day_dir_name.split("=", 1)[1] if "=" in day_dir_name else day_dir_name
        df = pd.read_parquet(pq)
        state_df = _build_state_frame(df, interval_ms=interval_ms)

        out_day = out_base / f"date={day}"
        out_day.mkdir(parents=True, exist_ok=True)
        out_pq = out_day / "state.parquet"
        state_df.to_parquet(out_pq, index=False)

        manifest = {
            "date": day,
            "symbol": sym,
            "interval_ms": int(interval_ms),
            "rows": int(len(state_df)),
            "ts_min": int(state_df["ts_ms"].min()) if len(state_df) else None,
            "ts_max": int(state_df["ts_ms"].max()) if len(state_df) else None,
            "input": str(pq),
            "output": str(out_pq),
        }
        (out_day / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=True, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        manifests.append(manifest)

    run_manifest = {
        "symbol": sym,
        "interval_ms": int(interval_ms),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "partitions": manifests,
    }
    (out_base / "manifest.json").write_text(
        json.dumps(run_manifest, ensure_ascii=True, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return run_manifest


def main() -> int:
    args = _parse_args()
    try:
        manifest = build_state(
            in_root=Path(str(args.in_root)),
            out_root=Path(str(args.out)),
            symbol=str(args.symbol),
            interval_ms=int(args.interval_ms),
        )
        rows = sum(int(p["rows"]) for p in manifest["partitions"])
        print(f"build_state ok symbol={manifest['symbol']} partitions={len(manifest['partitions'])} rows={rows}")
        return 0
    except Exception as e:
        print(f"build_state error runtime={type(e).__name__}:{e}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
