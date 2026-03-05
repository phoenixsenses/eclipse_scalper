from __future__ import annotations

import argparse
import json
from pathlib import Path

from utils.symbols import canonical_symbol


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Write per-symbol online calibration plans.")
    p.add_argument("--symbols", required=True, help="comma-separated")
    p.add_argument("--out-root", default="data/live")
    p.add_argument("--physics", default="data/derived/physics")
    p.add_argument("--interval-ms", type=int, default=100)
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        syms = [canonical_symbol(x) for x in str(args.symbols).split(",") if str(x).strip()]
        if not syms:
            raise RuntimeError("symbols_empty")
        out_root = Path(str(args.out_root))
        for sym in syms:
            plan = {
                "daily_online_refresh": {
                    "schedule": "daily",
                    "command": (
                        "python -m tools.activate_online_artifacts "
                        f"--live-root {out_root} "
                        "--build-calibration --build-execution "
                        f"--physics {args.physics} --symbol {sym} --interval-ms {int(args.interval_ms)} "
                        "--sanity-days 1 --days 14"
                    ),
                }
            }
            out = out_root / f"online_plan_{sym}.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(plan, ensure_ascii=True, sort_keys=True, indent=2) + "\n", encoding="utf-8")
            print(f"schedule_online_calibration_multi ok symbol={sym} out={out}")
        return 0
    except Exception as e:
        print(f"schedule_online_calibration_multi error runtime={type(e).__name__}:{e}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

