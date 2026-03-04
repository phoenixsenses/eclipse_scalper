from __future__ import annotations

import argparse
import json
from pathlib import Path


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate go/no-go from sweep aggregate.")
    p.add_argument("--manifest", required=True, help="Sweep run manifest.json")
    p.add_argument("--min-success-ratio", type=float, default=0.8)
    p.add_argument("--out", default="reports/GO_NOGO_FRAMEWORK.md")
    return p.parse_args()


def main() -> int:
    args = _args()
    mf = Path(args.manifest)
    if not mf.exists():
        print(f"go_nogo: missing {mf}")
        return 2
    obj = json.loads(mf.read_text(encoding="utf-8"))
    jobs = obj.get("jobs", [])
    total = len(jobs)
    ok = sum(1 for j in jobs if int(j.get("rc", 1)) == 0)
    ratio = (ok / total) if total else 0.0
    verdict = "GO" if ratio >= float(args.min_success_ratio) else "NO_GO"
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "\n".join(
            [
                "# GO / NO-GO EVALUATION",
                "",
                f"- manifest: {mf}",
                f"- total_jobs: {total}",
                f"- successful_jobs: {ok}",
                f"- success_ratio: {ratio:.2%}",
                f"- threshold: {float(args.min_success_ratio):.2%}",
                f"- verdict: **{verdict}**",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"go_nogo: {verdict} (ratio={ratio:.2%})")
    print(f"go_nogo: wrote {out}")
    return 0 if verdict == "GO" else 1


if __name__ == "__main__":
    raise SystemExit(main())

