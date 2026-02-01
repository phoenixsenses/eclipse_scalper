#!/usr/bin/env python3
"""
Run all unit-style tests in eclipse_scalper/tools.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    tools_dir = Path(__file__).resolve().parent
    repo_root = tools_dir.parent.parent
    tests = sorted(tools_dir.glob("test_*_unit.py"))

    if not tests:
        print("No unit tests found.")
        return 0

    failures = []
    for test in tests:
        print(f"\n==> {test.name}")
        result = subprocess.run([sys.executable, str(test)], cwd=repo_root)
        if result.returncode != 0:
            failures.append(test.name)

    print("\n==== Unit Test Summary ====")
    if failures:
        print(f"FAILED ({len(failures)}): " + ", ".join(failures))
        return 1

    print(f"OK ({len(tests)} tests)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
