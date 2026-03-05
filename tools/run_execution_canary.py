from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _run(cmd: List[str], *, env: Dict[str, str] | None = None) -> Dict[str, Any]:
    p = subprocess.run(cmd, capture_output=True, text=True, env=env)
    return {
        "cmd": cmd,
        "rc": int(p.returncode),
        "stdout_tail": str(p.stdout or "")[-4000:],
        "stderr_tail": str(p.stderr or "")[-4000:],
    }


def _safe_read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Single-command execution canary launcher.")
    p.add_argument("--symbol", default="ETHUSDT")
    p.add_argument("--interval-ms", type=int, default=100)
    p.add_argument("--db", default="data/microstructure.db")
    p.add_argument("--max-cycles", type=int, default=5, help="Canary daemon cycles.")
    p.add_argument("--refresh-sec", type=float, default=5.0)
    p.add_argument("--risk-policy", default="")
    p.add_argument("--report", default="reports/CANARY_EXECUTION_REPORT.json")
    p.add_argument("--report-md", default="reports/CANARY_EXECUTION_REPORT.md")
    p.add_argument("--sim", default="logs/micro_edge_debug_trades.jsonl")
    p.add_argument("--live-db", default="data/paper_trades.db")
    p.add_argument("--live-parquet", default="data/live/papertrades_live.parquet")
    return p.parse_args()


def _render_md(payload: Dict[str, Any]) -> str:
    checks = payload.get("checks", {})
    lines = [
        "# Canary Execution Report",
        "",
        f"- ts_utc: {payload.get('ts_utc','')}",
        f"- overall_ok: {int(bool(payload.get('overall_ok', False)))}",
        f"- symbol: {payload.get('symbol','')}",
        f"- max_cycles: {payload.get('max_cycles', 0)}",
        "",
        "## Checks",
    ]
    for k, v in checks.items():
        lines.append(f"- {k}: {int(bool(v))}")
    lines.append("")
    lines.append("## Steps")
    for s in payload.get("steps", []):
        lines.append(f"- rc={int(s.get('rc',1))} cmd=`{' '.join(s.get('cmd', []))}`")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = _args()
    start = time.time()
    env = dict(os.environ)
    # Force canary flags on for this process only.
    env["EXEC_ENGINE_UNIFIED"] = "1"
    env["EXEC_EVENT_BUS_ENABLED"] = "1"
    env["EXEC_RUNTIME_SUPERVISOR_ENABLED"] = "1"
    env.setdefault("EXEC_KILL_ON_CONTRACT_VIOLATION", "0")

    steps: List[Dict[str, Any]] = []
    daemon_cmd = [
        sys.executable,
        "-m",
        "tools.run_live_papertrade",
        "--db",
        str(args.db),
        "--symbol",
        str(args.symbol),
        "--interval-ms",
        str(int(args.interval_ms)),
        "--refresh-sec",
        str(float(args.refresh_sec)),
        "--max-cycles",
        str(int(args.max_cycles)),
        "--exec-engine-unified",
        "--exec-event-bus-enabled",
        "--exec-runtime-supervisor-enabled",
    ]
    if str(args.risk_policy).strip():
        daemon_cmd.extend(["--enable-risk-engine", "--risk-policy", str(args.risk_policy)])
    steps.append(_run(daemon_cmd, env=env))

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
            ],
            env=env,
        )
    )

    steps.append(_run([sys.executable, "-m", "tools.post_rollout_audit"], env=env))

    status = _safe_read_json(Path("data/live/status.json"))
    audit = _safe_read_json(Path("reports/POST_ROLLOUT_AUDIT.json"))
    checks = {
        "daemon_rc_ok": int(steps[0].get("rc", 1)) in {0, 2},  # 2 can be supervisor fail-safe
        "pipeline_ok": int(steps[1].get("rc", 1)) == 0,
        "audit_ok": int(steps[2].get("rc", 1)) == 0 and bool(audit.get("overall_ok", False)),
        "contract_violations_zero": int(status.get("exec_contract_violations", 0) or 0) == 0,
    }
    overall_ok = all(bool(v) for v in checks.values())
    payload = {
        "ts_utc": _utc(),
        "duration_sec": float(max(0.0, time.time() - start)),
        "symbol": str(args.symbol),
        "max_cycles": int(args.max_cycles),
        "flags": {
            "EXEC_ENGINE_UNIFIED": env.get("EXEC_ENGINE_UNIFIED"),
            "EXEC_EVENT_BUS_ENABLED": env.get("EXEC_EVENT_BUS_ENABLED"),
            "EXEC_RUNTIME_SUPERVISOR_ENABLED": env.get("EXEC_RUNTIME_SUPERVISOR_ENABLED"),
            "EXEC_KILL_ON_CONTRACT_VIOLATION": env.get("EXEC_KILL_ON_CONTRACT_VIOLATION"),
        },
        "steps": steps,
        "checks": checks,
        "overall_ok": bool(overall_ok),
    }
    out_json = Path(str(args.report))
    out_md = Path(str(args.report_md))
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    out_md.write_text(_render_md(payload), encoding="utf-8")
    print(f"run_execution_canary: overall_ok={int(overall_ok)} report={out_json} report_md={out_md}")
    return 0 if overall_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

