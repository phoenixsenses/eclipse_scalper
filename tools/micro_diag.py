from __future__ import annotations

import argparse
import json
import time

from core.micro_features import MicroFeatureEngine
from utils.symbols import canonical_symbol


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Micro feature diagnostic over sqlite tables.")
    p.add_argument("--db", default="data/microstructure.db")
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--window-sec", type=int, default=30)
    return p


def main() -> int:
    args = _build_parser().parse_args()
    sym = canonical_symbol(args.symbol)
    eng = MicroFeatureEngine(
        db_path=str(args.db),
        symbol=[sym],
        lookback_sec=max(30, int(args.window_sec)),
        update_interval_sec=1.0,
    )
    feat = eng._compute_once()
    ready, reason, detail = eng.get_readiness(sym)
    diag = eng.get_diag(sym)
    out = {
        "symbol": sym,
        "db": str(args.db),
        "window_sec": int(args.window_sec),
        "ready": bool(ready),
        "reason": str(reason),
        "detail": str(detail),
        "diag": diag,
        "features": (
            None
            if feat is None
            else {
                "timestamp": float(feat.timestamp),
                "age_sec": float(max(0.0, time.time() - float(feat.timestamp))),
                "imbalance_signed": float(feat.imbalance_signed),
                "trade_intensity": float(feat.trade_intensity),
                "spread": float(feat.spread),
                "mark_price": float(feat.mark_price),
            }
        ),
    }
    print(json.dumps(out, sort_keys=True, indent=2))
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())

