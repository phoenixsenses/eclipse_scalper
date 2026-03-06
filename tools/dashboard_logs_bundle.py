from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copy2(src, dst)
        return True
    except Exception:
        return False


def _tail_lines(path: Path, n: int) -> list[str]:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        return lines[-max(1, int(n)):]
    except Exception:
        return []


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export dashboard backend/supervisor log bundle")
    p.add_argument("--out-root", default="reports/incidents")
    p.add_argument("--tail-lines", type=int, default=400)
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    out_dir = repo_root / str(args.out_root) / f"dashboard_bundle_{_utc_stamp()}"
    out_dir.mkdir(parents=True, exist_ok=True)

    files = [
        repo_root / "logs" / "dashboard_backend_supervisor.log",
        repo_root / "logs" / "dashboard_backend_launcher.out.log",
        repo_root / "logs" / "dashboard_backend_launcher.err.log",
        repo_root / "runtime" / "dashboard_backend.json",
        repo_root / "runtime" / "dashboard_launcher.lock",
        repo_root / "logs" / "last_shutdown.json",
    ]

    copied: dict[str, str] = {}
    for src in files:
        dst = out_dir / src.name
        if _copy_if_exists(src, dst):
            copied[str(src.relative_to(repo_root))] = str(dst.relative_to(repo_root))

    # Tail snapshots to quickly inspect without opening huge files.
    for name in (
        "dashboard_backend_supervisor.log",
        "dashboard_backend_launcher.out.log",
        "dashboard_backend_launcher.err.log",
    ):
        src = repo_root / "logs" / name
        if not src.exists():
            continue
        tail = _tail_lines(src, int(args.tail_lines))
        (out_dir / f"tail_{name}").write_text("\n".join(tail) + ("\n" if tail else ""), encoding="utf-8")

    summary: dict[str, Any] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "bundle_dir": str(out_dir.relative_to(repo_root)),
        "copied_files": copied,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out_dir / "summary.md").write_text(
        "\n".join(
            [
                "# Dashboard Log Bundle",
                "",
                f"- created_utc: `{summary['created_utc']}`",
                f"- bundle_dir: `{summary['bundle_dir']}`",
                "",
                "## Copied Files",
                *([f"- `{k}` -> `{v}`" for k, v in copied.items()] or ["- none"]),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"dashboard_logs_bundle: wrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
