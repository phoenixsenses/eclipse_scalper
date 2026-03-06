from __future__ import annotations

import argparse
import json
from pathlib import Path


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate day-60 sweep outputs.")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--out", default="reports/DAY60_MASTER_RESULTS.md")
    return p.parse_args()


def main() -> int:
    args = _args()
    run_dir = Path(args.run_dir)
    manifest = run_dir / "manifest.json"
    if not manifest.exists():
        print(f"aggregate: missing {manifest}")
        return 2
    obj = json.loads(manifest.read_text(encoding="utf-8"))
    jobs = obj.get("jobs", [])
    ok = sum(1 for j in jobs if int(j.get("rc", 1)) == 0)
    total = len(jobs)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# DAY60 MASTER RESULTS",
        "",
        f"- run_dir: {run_dir}",
        f"- total_jobs: {total}",
        f"- succeeded: {ok}",
        f"- failed: {total-ok}",
        "",
        "## Jobs",
        "",
        "| id | rc | out_json |",
        "|---|---:|---|",
    ]
    for j in jobs:
        lines.append(f"| {j.get('id')} | {int(j.get('rc', 1))} | {j.get('out_json')} |")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"aggregate: wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

