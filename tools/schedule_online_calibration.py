from __future__ import annotations

import argparse
import json
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Write online calibration/execution schedule plan.")
    p.add_argument("--out", default="data/live/online_plan.json")
    p.add_argument("--symbol", default="ETHUSDT")
    p.add_argument("--interval-ms", type=int, default=100)
    p.add_argument("--physics", default="data/derived/physics")
    p.add_argument("--live-root", default="data/live")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        plan = {
            "daily_online_calibration": {
                "schedule": "daily",
                "command": (
                    "python -m tools.daily_execution_calibration "
                    f"--physics {args.physics} --symbol {args.symbol} --interval-ms {int(args.interval_ms)} --days 14 "
                    "--run-root-cause 1"
                ),
            },
            "daily_live_fill_root_cause": {
                "schedule": "daily",
                "command": "python -m tools.live_fill_drift_root_cause --run-pipeline",
            },
            "daily_canary_expansion_gate": {
                "schedule": "daily",
                "command": (
                    "python -m tools.evaluate_canary_expansion_gate "
                    "--report-dir reports/daily --window-days 7 --max-top-score 0.5"
                ),
            },
            "daily_canary_gate_wrapper": {
                "schedule": "daily",
                "command": (
                    "powershell -NoProfile -ExecutionPolicy Bypass "
                    "-File .\\tools\\run_daily_canary_gate.ps1 "
                    "-Symbol ETHUSDT -Days 14 -WindowDays 7 -MaxTopScore 0.5"
                ),
            },
            "weekly_execution_calibration": {
                "schedule": "weekly",
                "command": (
                    "python -m tools.activate_online_artifacts "
                    f"--live-root {args.live_root} "
                    "--build-calibration --build-execution "
                    f"--physics {args.physics} --symbol {args.symbol} --interval-ms {int(args.interval_ms)} "
                    "--sanity-days 1 --days 14"
                ),
            },
            "monthly_full_pipeline": {
                "schedule": "monthly",
                "command": (
                    "python -m tools.run_alpha_pipeline "
                    f"--symbol {args.symbol} --interval-ms {int(args.interval_ms)} "
                    f"--physics {args.physics} --regimes data/derived/regimes --out-root data/runs/alpha"
                ),
            },
        }
        out = Path(str(args.out))
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(plan, ensure_ascii=True, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        print(f"schedule_online_calibration ok out={out}")
        return 0
    except Exception as e:
        print(f"schedule_online_calibration error runtime={type(e).__name__}:{e}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
