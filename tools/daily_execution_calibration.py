from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


def _utc_day() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _run(cmd: List[str]) -> Dict[str, Any]:
    p = subprocess.run(cmd, capture_output=True, text=True)
    return {
        "cmd": cmd,
        "rc": int(p.returncode),
        "stdout_tail": str(p.stdout or "")[-3000:],
        "stderr_tail": str(p.stderr or "")[-3000:],
    }


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Daily execution calibration + drift pipeline.")
    p.add_argument("--symbol", default="ETHUSDT")
    p.add_argument("--interval-ms", type=int, default=100)
    p.add_argument("--physics", default="data/derived/physics")
    p.add_argument("--out", default="data/derived/execution_calibration")
    p.add_argument("--days", type=int, default=14)
    p.add_argument("--sim", default="logs/micro_edge_debug_trades.jsonl")
    p.add_argument("--live-db", default="data/paper_trades.db")
    p.add_argument("--live-parquet", default="data/live/papertrades_live.parquet")
    p.add_argument("--report-dir", default="reports/daily")
    p.add_argument("--run-root-cause", type=int, default=1, help="Run live fill drift root-cause analysis.")
    return p.parse_args()


def main() -> int:
    args = _args()
    day = _utc_day()
    out_dir = Path(str(args.report_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / f"{day}_EXEC_CALIBRATION.json"
    out_md = out_dir / f"{day}_EXEC_CALIBRATION.md"
    root_json = out_dir / f"{day}_LIVE_FILL_DRIFT_ROOT_CAUSE.json"
    root_md = out_dir / f"{day}_LIVE_FILL_DRIFT_ROOT_CAUSE.md"

    steps = []
    steps.append(
        _run(
            [
                sys.executable,
                "-m",
                "tools.calibrate_execution_models",
                "--physics",
                str(args.physics),
                "--symbol",
                str(args.symbol),
                "--interval-ms",
                str(int(args.interval_ms)),
                "--out",
                str(args.out),
                "--days",
                str(int(args.days)),
            ]
        )
    )
    steps.append(
        _run(
            [
                sys.executable,
                "-m",
                "tools.execution_e2e_pipeline",
                "--sim",
                str(args.sim),
                "--live-db",
                str(args.live_db),
                "--live-parquet",
                str(args.live_parquet),
            ]
        )
    )
    if int(args.run_root_cause) == 1:
        steps.append(
            _run(
                [
                    sys.executable,
                    "-m",
                    "tools.live_fill_drift_root_cause",
                    "--parity-json",
                    "reports/REPLAY_PARITY_REPORT.json",
                    "--diag-json",
                    "reports/EXECUTION_HEALTH.json",
                    "--tox-json",
                    "reports/TOXICITY_REPORT.json",
                    "--audit-json",
                    "reports/POST_ROLLOUT_AUDIT.json",
                    "--out-json",
                    str(root_json),
                    "--out-md",
                    str(root_md),
                ]
            )
        )
    ok = all(int(s.get("rc", 1)) == 0 for s in steps)
    payload = {
        "ts_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "symbol": str(args.symbol),
        "interval_ms": int(args.interval_ms),
        "days": int(args.days),
        "ok": bool(ok),
        "steps": steps,
    }
    out_json.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")

    lines = [
        f"# Daily Execution Calibration ({day})",
        "",
        f"- symbol: `{args.symbol}`",
        f"- interval_ms: `{int(args.interval_ms)}`",
        f"- lookback_days: `{int(args.days)}`",
        f"- ok: `{int(ok)}`",
        f"- root_cause_enabled: `{int(args.run_root_cause)}`",
        "",
        "## Steps",
    ]
    for i, s in enumerate(steps, 1):
        lines.append(f"- step_{i}: rc={int(s.get('rc',1))} cmd=`{' '.join(s.get('cmd', []))}`")
    lines.append("")
    if int(args.run_root_cause) == 1:
        lines.append("## Root Cause Artifacts")
        lines.append(f"- `{root_md}`")
        lines.append(f"- `{root_json}`")
        lines.append("")
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"daily_execution_calibration: ok={int(ok)} out_json={out_json} out_md={out_md}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
