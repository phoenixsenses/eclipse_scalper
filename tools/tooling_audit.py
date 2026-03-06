from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List


SPECIAL_STATUS = {
    "_inspect_forward_json.py": "dev-only",
    "_inspect_rank_evals.py": "dev-only",
    "_inspect_rank_evals2.py": "dev-only",
    "_inspect_rank_json.py": "dev-only",
    "_inspect_rank_net2.py": "dev-only",
    "_inspect_rank_scores2.py": "dev-only",
    "_print_adv_sweep.py": "dev-only",
    "_print_fee_sweep.py": "dev-only",
    "_print_fee_sweep_newmetrics.py": "dev-only",
    "_summarize_gate_md.py": "dev-only",
}

RUN_SUMMARY_EXCLUDED = {
    "tools/build_presentation.py",
    "tools/write_risk_policy_doc.py",
}


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""


def classify_scope(name: str) -> str:
    if name.startswith("telemetry_") or name.startswith("dashboard_"):
        return "runtime"
    if name.startswith(("validate_", "report_", "run_", "build_", "analyze_", "sweep_", "micro_", "calibrate_")):
        return "research"
    if name.startswith(("health_", "ops_", "incident_", "collection_", "ingestion_", "preflight_", "post_rollout_")):
        return "ops"
    if name.startswith("_") or name.startswith("test_"):
        return "dev"
    return "mixed"


def classify_status(path: Path) -> str:
    name = path.name
    if name in SPECIAL_STATUS:
        return SPECIAL_STATUS[name]
    if name.startswith("test_"):
        return "dev-only"
    if name.startswith("_"):
        return "legacy"
    if name.startswith(("run_", "validate_", "report_", "smoke_")):
        return "core"
    if name.startswith(("build_", "analyze_", "calibrate_", "compare_", "eval_", "sweep_")):
        return "support"
    return "support"


def detect_contracts(text: str) -> Dict[str, bool]:
    return {
        "has_out_json": "--out-json" in text,
        "has_out_md": ("--out-md" in text) or ("--out-report" in text),
        "has_run_summary": "build_run_summary" in text,
        "uses_report_validator": "report_schema_validator" in text,
    }


def classify_family(name: str) -> str:
    if name.startswith("validate_"):
        return "validation"
    if name.startswith(("report_", "summarize_")):
        return "reporting"
    if name.startswith("run_"):
        return "runner"
    if name.startswith("build_"):
        return "builder"
    if name.startswith(("analyze_", "compare_", "eval_", "sweep_", "calibrate_")):
        return "research"
    if name.startswith(("smoke_", "health_", "ops_", "incident_", "collection_", "ingestion_", "preflight_")):
        return "ops"
    if name.startswith(("telemetry_", "dashboard_")):
        return "runtime"
    if name.startswith("_") or name.startswith("test_"):
        return "dev"
    return "misc"


def build_manifest(root: Path) -> Dict[str, Any]:
    tools_dir = root / "tools"
    scripts_dir = root / "scripts"
    tool_files = sorted(p for p in tools_dir.rglob("*.py") if "__pycache__" not in p.parts)
    script_files = sorted(p for p in scripts_dir.rglob("*") if p.is_file())

    tools_entries: List[Dict[str, Any]] = []
    for path in tool_files:
        text = _read_text(path)
        contracts = detect_contracts(text)
        tools_entries.append(
            {
                "path": str(path.relative_to(root)).replace("\\", "/"),
                "family": classify_family(path.name),
                "scope": classify_scope(path.name),
                "status": classify_status(path),
                "contracts": contracts,
            }
        )

    scripts_entries = [
        {
            "path": str(path.relative_to(root)).replace("\\", "/"),
            "kind": "launcher" if path.suffix in {".ps1", ".sh"} else "python",
        }
        for path in script_files
    ]

    family_counts = Counter(item["family"] for item in tools_entries)
    status_counts = Counter(item["status"] for item in tools_entries)
    run_summary_gaps = [
        item["path"]
        for item in tools_entries
        if item["contracts"]["has_out_json"] and not item["contracts"]["has_run_summary"] and item["path"] not in RUN_SUMMARY_EXCLUDED
    ]
    dev_only = [item["path"] for item in tools_entries if item["status"] == "dev-only"]

    return {
        "tools": tools_entries,
        "scripts": scripts_entries,
        "summary": {
            "tool_count": len(tools_entries),
            "script_count": len(scripts_entries),
            "family_counts": dict(sorted(family_counts.items())),
            "status_counts": dict(sorted(status_counts.items())),
            "run_summary_gap_count": len(run_summary_gaps),
            "dev_only_count": len(dev_only),
        },
        "cut_candidates": dev_only,
        "run_summary_gaps": run_summary_gaps,
    }


def render_markdown(manifest: Dict[str, Any]) -> str:
    summary = manifest["summary"]
    lines = [
        "# Tooling Audit",
        "",
        "## Summary",
        f"- tool_count: {summary['tool_count']}",
        f"- script_count: {summary['script_count']}",
        f"- run_summary_gap_count: {summary['run_summary_gap_count']}",
        f"- dev_only_count: {summary['dev_only_count']}",
        "",
        "## Status Counts",
    ]
    for key, value in summary["status_counts"].items():
        lines.append(f"- {key}: {value}")
    lines += ["", "## Family Counts"]
    for key, value in summary["family_counts"].items():
        lines.append(f"- {key}: {value}")
    lines += ["", "## Cut Candidates"]
    for path in manifest["cut_candidates"]:
        lines.append(f"- {path}")
    lines += ["", "## Run Summary Gaps"]
    if manifest["run_summary_gaps"]:
        for path in manifest["run_summary_gaps"]:
            lines.append(f"- {path}")
    else:
        lines.append("- none")
    return "\n".join(lines) + "\n"


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Audit tools/scripts inventory and emit a manifest.")
    p.add_argument("--root", default=".")
    p.add_argument("--out-json", default="reports/TOOL_MANIFEST.json")
    p.add_argument("--out-md", default="docs/TOOLING_AUDIT.md")
    return p.parse_args()


def main() -> int:
    args = _args()
    root = Path(str(args.root)).resolve()
    manifest = build_manifest(root)
    out_json = root / str(args.out_json)
    out_md = root / str(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    out_md.write_text(render_markdown(manifest), encoding="utf-8")
    print(f"tooling_audit tools={manifest['summary']['tool_count']} scripts={manifest['summary']['script_count']}")
    print(f"wrote {out_json}")
    print(f"wrote {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
