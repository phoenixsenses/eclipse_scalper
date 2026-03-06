from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


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


def _latest_log(logs_dir: Path) -> Path | None:
    cands = sorted(logs_dir.glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0] if cands else None


def _tail_lines(path: Path, n: int) -> List[str]:
    if not path.exists():
        return []
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        return lines[-max(1, int(n)) :]
    except Exception:
        return []


def _latest_run_dirs(run_root: Path, limit: int = 2) -> List[Path]:
    if not run_root.exists():
        return []
    runs = [p for p in run_root.iterdir() if p.is_dir() and p.name.startswith("run_")]
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return runs[: max(0, int(limit))]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Collect incident triage bundle.")
    p.add_argument("--out-root", default="reports/incidents")
    p.add_argument("--logs-dir", default="logs")
    p.add_argument("--run-root", default="data/runs/alpha")
    p.add_argument("--tail-lines", type=int, default=300)
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    ts = _utc_stamp()
    out_dir = Path(str(args.out_root)) / f"incident_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    copied: Dict[str, str] = {}
    logs_dir = Path(str(args.logs_dir))
    run_root = Path(str(args.run_root))

    files = [
        Path("logs/last_shutdown.json"),
        Path("data/live/status.json"),
        Path("data/live/active_artifacts.json"),
        Path("data/live/watermark.json"),
    ]
    for src in files:
        dst = out_dir / src.as_posix().replace("/", "_")
        if _copy_if_exists(src, dst):
            copied[src.as_posix()] = str(dst)

    latest = _latest_log(logs_dir)
    tail_path = out_dir / "latest_runtime_tail.log"
    if latest is not None:
        tail = _tail_lines(latest, int(args.tail_lines))
        tail_path.write_text("\n".join(tail) + ("\n" if tail else ""), encoding="utf-8")
        copied[str(latest)] = str(tail_path)

    run_copies: List[str] = []
    for r in _latest_run_dirs(run_root, limit=2):
        for name in ("manifest.json", "pointers.json"):
            src = r / name
            dst = out_dir / f"{r.name}_{name}"
            if _copy_if_exists(src, dst):
                run_copies.append(str(dst))

    summary: Dict[str, Any] = {
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "bundle_dir": str(out_dir),
        "copied_files": copied,
        "run_files": run_copies,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    md = [
        "# Incident Bundle",
        "",
        f"- created_utc: `{summary['created_utc']}`",
        f"- bundle_dir: `{summary['bundle_dir']}`",
        "",
        "## Files",
    ]
    if copied:
        for k, v in copied.items():
            md.append(f"- `{k}` -> `{v}`")
    else:
        md.append("- none")
    md.append("")
    md.append("## Run Artifacts")
    if run_copies:
        for p in run_copies:
            md.append(f"- `{p}`")
    else:
        md.append("- none")
    (out_dir / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"incident_bundle: wrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

