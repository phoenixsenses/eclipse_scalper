from __future__ import annotations

import argparse
import json
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Write alpha run plan and print scheduler commands.")
    p.add_argument("--out", default="data/runs/alpha/run_plan.json")
    p.add_argument("--symbol", default="ETHUSDT")
    p.add_argument("--interval-ms", type=int, default=100)
    p.add_argument("--physics", default="data/derived/physics")
    p.add_argument("--regimes", default="data/derived/regimes")
    p.add_argument("--out-root", default="data/runs/alpha")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        plan = {
            "daily_quick": {
                "schedule": "daily",
                "command": (
                    "python -m tools.run_alpha_pipeline "
                    f"--symbol {args.symbol} --interval-ms {int(args.interval_ms)} "
                    f"--physics {args.physics} --regimes {args.regimes} --out-root {args.out_root} --quick"
                ),
            },
            "weekly_full": {
                "schedule": "weekly",
                "command": (
                    "python -m tools.run_alpha_pipeline "
                    f"--symbol {args.symbol} --interval-ms {int(args.interval_ms)} "
                    f"--physics {args.physics} --regimes {args.regimes} --out-root {args.out_root}"
                ),
            },
        }
        out = Path(str(args.out))
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(plan, ensure_ascii=True, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        print(f"schedule_alpha_runs ok out={out}")
        print("Task Scheduler hints:")
        print(f"  Daily : {plan['daily_quick']['command']}")
        print(f"  Weekly: {plan['weekly_full']['command']}")
        return 0
    except Exception as e:
        print(f"schedule_alpha_runs error runtime={type(e).__name__}:{e}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
