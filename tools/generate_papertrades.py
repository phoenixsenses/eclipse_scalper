from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from src.microphys.execution.calibration import load_execution_params
from src.microphys.sim.papertrade import PaperTradeConfig, generate_papertrades
from utils.symbols import canonical_symbol


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate lightweight papertrade log from ensemble scores.")
    p.add_argument("--physics", default="data/derived/physics")
    p.add_argument("--ensemble", required=True)
    p.add_argument("--symbol", required=True)
    p.add_argument("--interval-ms", type=int, default=100)
    p.add_argument("--horizon-bars", type=int, default=10)
    p.add_argument("--mode", choices=["taker", "maker"], default="taker")
    p.add_argument("--execution-model", choices=["simple", "maker_queue", "maker_hazard"], default="simple")
    p.add_argument("--execution-params", default="")
    p.add_argument("--ttl-bars", type=int, default=10)
    p.add_argument("--fee-bps", type=float, default=0.5)
    p.add_argument("--out", default="data/derived/alpha_eval")
    p.add_argument("--report", default="")
    return p.parse_args()


def _load_physics(root: Path, symbol: str, interval_ms: int) -> pd.DataFrame:
    base = root / f"interval_ms={int(interval_ms)}" / f"symbol={symbol}"
    files = sorted(base.glob("date=*/physics.parquet"))
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(p) for p in files], ignore_index=True).sort_values("ts_ms").reset_index(drop=True)


def main() -> int:
    args = _parse_args()
    try:
        symbol = canonical_symbol(args.symbol)
        physics = _load_physics(Path(str(args.physics)), symbol, int(args.interval_ms))
        if physics.empty:
            raise RuntimeError("physics_missing")
        ensemble = pd.read_parquet(Path(str(args.ensemble)))
        frame = physics.merge(ensemble[["ts_ms", "ensemble_side", "signal_count"]], on="ts_ms", how="left")
        frame["ensemble_side"] = pd.to_numeric(frame.get("ensemble_side"), errors="coerce").fillna(0.0)
        frame["signal_count"] = pd.to_numeric(frame.get("signal_count"), errors="coerce").fillna(0).astype(int)
        trades = generate_papertrades(
            frame,
            horizon_bars=int(args.horizon_bars),
            cfg=PaperTradeConfig(
                mode=str(args.mode),
                fee_bps=float(args.fee_bps),
                execution_model=str(args.execution_model),
                execution_params=(load_execution_params(Path(str(args.execution_params))) if str(args.execution_params).strip() else None),
                ttl_bars=int(args.ttl_bars),
            ),
        )
        out_base = Path(str(args.out)) / f"interval_ms={int(args.interval_ms)}" / f"symbol={symbol}"
        out_base.mkdir(parents=True, exist_ok=True)
        out_pq = out_base / "papertrades.parquet"
        trades.to_parquet(out_pq, index=False)
        (out_base / "papertrades_manifest.json").write_text(
            json.dumps(
                {
                    "symbol": symbol,
                    "interval_ms": int(args.interval_ms),
                    "rows": int(len(trades)),
                    "horizon_bars": int(args.horizon_bars),
                    "mode": str(args.mode),
                    "fee_bps": float(args.fee_bps),
                },
                ensure_ascii=True,
                sort_keys=True,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        report = Path(str(args.report)) if str(args.report).strip() else Path(f"reports/papertrades_{symbol}_{int(args.interval_ms)}ms.md")
        params_payload = (
            load_execution_params(Path(str(args.execution_params))) if str(args.execution_params).strip() else {}
        )
        params_hash = hashlib.sha1(
            json.dumps(params_payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:12]
        filled_rate = float(pd.to_numeric(trades.get("filled"), errors="coerce").fillna(1).mean()) if not trades.empty else 0.0
        ttl_rate = float(pd.to_numeric(trades.get("ttl_expired"), errors="coerce").fillna(0).mean()) if not trades.empty else 0.0
        lines = [
            f"# Paper Trades - {symbol} ({int(args.interval_ms)}ms)",
            "",
            f"- trades: `{len(trades)}`",
            f"- mode: `{args.mode}` fee_bps=`{float(args.fee_bps)}` horizon_bars=`{int(args.horizon_bars)}`",
            f"- execution_model: `{str(args.execution_model)}`",
            f"- execution_params_hash: `{params_hash}`",
            f"- fill_rate: `{filled_rate:.4f}`",
            f"- ttl_expired_rate: `{ttl_rate:.4f}`",
            f"- mean pnl_net: `{float(pd.to_numeric(trades.get('pnl_net'), errors='coerce').mean() if not trades.empty else 0.0):.8f}`",
            f"- median pnl_net: `{float(pd.to_numeric(trades.get('pnl_net'), errors='coerce').median() if not trades.empty else 0.0):.8f}`",
        ]
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
        print(f"generate_papertrades ok out={out_pq} report={report} rows={len(trades)}")
        return 0
    except Exception as e:
        print(f"generate_papertrades error runtime={type(e).__name__}:{e}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
