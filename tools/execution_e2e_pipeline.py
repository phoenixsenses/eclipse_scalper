from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List


def _run(cmd: List[str]) -> Dict[str, Any]:
    p = subprocess.run(cmd, capture_output=True, text=True)
    return {
        "cmd": cmd,
        "rc": int(p.returncode),
        "stdout": str(p.stdout or "")[-4000:],
        "stderr": str(p.stderr or "")[-4000:],
    }


def _write(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Execution upgrade E2E orchestration pipeline.")
    p.add_argument("--sim", default="logs/micro_edge_debug_trades.jsonl")
    p.add_argument("--live-db", default="data/paper_trades.db")
    p.add_argument("--live-parquet", default="data/live/papertrades_live.parquet")
    p.add_argument("--out-json", default="reports/EXECUTION_E2E_PIPELINE.json")
    return p.parse_args()


def main() -> int:
    args = _args()
    steps = []
    steps.append(
        _run(
            [
                sys.executable,
                "-m",
                "tools.replay_parity_report",
                "--sim",
                str(args.sim),
                "--live-db",
                str(args.live_db),
            ]
        )
    )
    steps.append(
        _run(
            [
                sys.executable,
                "-m",
                "tools.execution_diagnostics",
                "--in",
                str(args.live_parquet),
            ]
        )
    )
    steps.append(
        _run(
            [
                sys.executable,
                "-m",
                "tools.toxicity_report",
                "--in",
                str(args.live_parquet),
            ]
        )
    )
    steps.append(_run([sys.executable, "-m", "tools.post_rollout_audit"]))

    ok = all(int(s.get("rc", 1)) == 0 for s in steps)
    out = {
        "ok": bool(ok),
        "steps": steps,
    }
    _write(Path(str(args.out_json)), out)
    print(f"execution_e2e_pipeline: ok={int(ok)} out={args.out_json}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

