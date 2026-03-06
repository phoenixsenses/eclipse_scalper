from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from tools.run_summary import build_run_summary

def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _env_on(name: str) -> bool:
    return str(os.getenv(name, "0")).strip().lower() in {"1", "true", "yes", "on"}


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def build_audit(diag_json: Dict[str, Any], tox_json: Dict[str, Any]) -> Dict[str, Any]:
    flags = {
        "EXEC_LATENCY_V2": _env_on("EXEC_LATENCY_V2"),
        "QUEUE_MODEL_V2": _env_on("QUEUE_MODEL_V2"),
        "EXEC_ENGINE_UNIFIED": _env_on("EXEC_ENGINE_UNIFIED"),
    }
    checks = {
        "diag_rows_ok": int(diag_json.get("rows", 0)) > 0,
        "tox_rows_ok": int(tox_json.get("rows", 0)) > 0,
        "latency_p95_ok": float(diag_json.get("latency_fill_delay_sec_p95", 0.0)) <= 10.0,
        "fill_rate_ok": float(diag_json.get("fill_rate", 0.0)) >= 0.05,
    }
    overall = all(bool(v) for v in checks.values())
    return {
        "ts_utc": _utc(),
        "flags": flags,
        "checks": checks,
        "overall_ok": bool(overall),
    }


def _render_md(d: Dict[str, Any]) -> str:
    lines = [
        "# POST ROLLOUT AUDIT",
        "",
        f"- ts_utc: {d.get('ts_utc','')}",
        f"- overall_ok: {bool(d.get('overall_ok', False))}",
        "",
        "## Flags",
    ]
    for k, v in dict(d.get("flags", {})).items():
        lines.append(f"- {k}: {int(bool(v))}")
    lines.append("")
    lines.append("## Checks")
    for k, v in dict(d.get("checks", {})).items():
        lines.append(f"- {k}: {int(bool(v))}")
    if isinstance(d.get("run_summary"), dict):
        lines.extend(["", "## Run Summary", f"- `{d['run_summary']}`"])
    lines.append("")
    return "\n".join(lines)


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Post-rollout execution audit.")
    p.add_argument("--diag-json", default="reports/EXECUTION_HEALTH.json")
    p.add_argument("--tox-json", default="reports/TOXICITY_REPORT.json")
    p.add_argument("--out-md", default="reports/POST_ROLLOUT_AUDIT.md")
    p.add_argument("--out-json", default="reports/POST_ROLLOUT_AUDIT.json")
    return p.parse_args()


def main() -> int:
    args = _args()
    d = build_audit(_read_json(Path(str(args.diag_json))), _read_json(Path(str(args.tox_json))))
    out_md = Path(str(args.out_md))
    out_json = Path(str(args.out_json))
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    d["run_summary"] = build_run_summary(
        run_type="post_rollout_audit",
        inputs={"diag_json": str(args.diag_json), "tox_json": str(args.tox_json)},
        metrics={"overall_ok": bool(d.get("overall_ok", False)), "check_count": len(d.get("checks", {}))},
        artifacts={"json": str(out_json), "md": str(out_md)},
    )
    out_md.write_text(_render_md(d), encoding="utf-8")
    out_json.write_text(json.dumps(d, ensure_ascii=True, indent=2), encoding="utf-8")
    print(f"post_rollout_audit: overall_ok={int(bool(d.get('overall_ok', False)))} out_md={out_md} out_json={out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
