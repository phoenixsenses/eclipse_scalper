from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str]) -> tuple[int, str]:
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    out = (p.stdout or "") + (p.stderr or "")
    return int(p.returncode), out.strip()


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare and optionally create a local release tag.")
    p.add_argument("--tag", required=True, help="Release tag name, e.g. v2026.03.04-stable")
    p.add_argument("--create-tag", action="store_true", help="Create local annotated git tag when checks pass.")
    p.add_argument("--run-tests", action="store_true", help="Run pytest -q before creating tag.")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    root = Path(".").resolve()
    if not (root / ".git").exists():
        print("prepare_release_tag: not a git repository")
        return 2

    rc, branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    if rc != 0:
        print("prepare_release_tag: cannot resolve branch")
        return 2

    rc, sha = _run(["git", "rev-parse", "--short", "HEAD"])
    if rc != 0:
        print("prepare_release_tag: cannot resolve commit")
        return 2

    rc, dirty = _run(["git", "status", "--porcelain"])
    is_clean = rc == 0 and not dirty.strip()
    print(f"prepare_release_tag: branch={branch} commit={sha} clean={int(is_clean)}")
    if not is_clean:
        print("prepare_release_tag: working tree is dirty; commit/stash first")
        return 1

    if bool(args.run_tests):
        print("prepare_release_tag: running pytest -q ...")
        t = subprocess.run([sys.executable, "-m", "pytest", "-q"])
        if int(t.returncode) != 0:
            print("prepare_release_tag: tests failed")
            return 1

    if bool(args.create_tag):
        msg = f"Known-good release {args.tag} from {sha}"
        rc, out = _run(["git", "tag", "-a", str(args.tag), "-m", msg])
        if rc != 0:
            print(f"prepare_release_tag: failed to create tag: {out}")
            return 1
        print(f"prepare_release_tag: created local tag {args.tag}")
        print(f"prepare_release_tag: push with `git push origin {args.tag}`")
        return 0

    print("prepare_release_tag: dry-run complete (no tag created)")
    print(f"prepare_release_tag: create with `python -m tools.prepare_release_tag --tag {args.tag} --create-tag`")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

