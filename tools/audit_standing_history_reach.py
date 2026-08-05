"""How far back does each standing role actually read? (read-only audit)

SYSTEM_STATE section 265 answered this by grepping each role's own `*_LOOKBACK_MS`
constants and concluded "7 days". It was wrong by a factor of two: the deepest window
was `funding_pctile_14d`, defined in `tools/liq_indicator_library.py` and reached by
two standing roles through an import chain. A per-file audit cannot see that, and the
conclusion it produced was used to argue that an 836 GiB segment was safe to delete.

This walks the TRANSITIVE local import graph from every role that start_eclipse.ps1
launches, and reports the deepest declared backward window it can find.

WHAT THIS IS NOT
It is a lower bound, not a proof. Three reach classes are invisible to it and are
reported separately where detectable:

  * unbounded aggregates -- `SELECT MAX(ts_ms) FROM t` with no time predicate reads
    the whole estate; effectively infinite reach;
  * dynamically computed windows -- anything derived at runtime from a CLI argument
    or a persisted state file;
  * cold-start paths -- a bootstrap that seeds itself from `MIN(ts_ms)` walks the
    entire history regardless of any steady-state constant.

Treat a clean report as "no NEW declared window appeared", never as "nothing reaches
further".

    python -m tools.audit_standing_history_reach
    python -m tools.audit_standing_history_reach --json
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
START_SCRIPT = REPO_ROOT / "start_eclipse.ps1"
LOCAL_ROOTS = {"tools", "ami", "data", "scripts", "dashboard", "src"}

DAY_MS = 86_400_000
_DAYS = re.compile(r"(\d+)\s*\*\s*86_?400_?000")
_HOURS = re.compile(r"(\d+)\s*\*\s*3_?600_?000")
# an aggregate over a table with no ts_ms lower bound anywhere in the statement
_AGG = re.compile(r"SELECT\s+(?:MIN|MAX|COUNT|SUM|AVG)\s*\(", re.I)


def standing_roles() -> list[str]:
    """Module names start_eclipse.ps1 launches, taken from its -CommandNeedle values."""
    if not START_SCRIPT.exists():
        return []
    text = START_SCRIPT.read_text(encoding="utf-8", errors="replace")
    roles: set[str] = set()
    for raw in re.findall(r'-CommandNeedle\s+"([^"]+)"', text):
        needle = raw.split()[0]  # strip trailing CLI args, e.g. "... --serve"
        if needle.endswith(".py"):
            needle = needle.replace("\\", "/").replace("/", ".")[:-3]
        roles.add(needle)
    return sorted(roles)


def module_path(dotted: str) -> Path | None:
    p = REPO_ROOT / (dotted.replace(".", "/") + ".py")
    return p if p.exists() else None


def local_imports(path: Path) -> set[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except (SyntaxError, OSError):
        return set()
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
        elif isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
    return {m for m in found if m.split(".")[0] in LOCAL_ROOTS}


def scan_module(dotted: str) -> tuple[list[tuple[int, int, str]], list[tuple[int, str]]]:
    """(declared windows, unbounded aggregates) for one module."""
    path = module_path(dotted)
    if path is None:
        return [], []
    windows: list[tuple[int, int, str]] = []
    unbounded: list[tuple[int, str]] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        for m in _DAYS.finditer(line):
            windows.append((int(m.group(1)) * DAY_MS, lineno, stripped[:100]))
        for m in _HOURS.finditer(line):
            ms = int(m.group(1)) * 3_600_000
            if ms >= DAY_MS:
                windows.append((ms, lineno, stripped[:100]))
        if _AGG.search(line) and "ts_ms" not in line.split("WHERE")[-1]:
            if " FROM " in line.upper():
                unbounded.append((lineno, stripped[:100]))
    return windows, unbounded


def audit() -> list[dict]:
    report = []
    for role in standing_roles():
        if module_path(role) is None:
            report.append({"role": role, "status": "no module on disk"})
            continue
        seen: set[str] = set()
        stack = [role]
        windows: list[tuple[int, int, str, str]] = []
        unbounded: list[tuple[str, int, str]] = []
        while stack:
            mod = stack.pop()
            if mod in seen:
                continue
            seen.add(mod)
            mpath = module_path(mod)
            if mpath is None:
                continue
            w, u = scan_module(mod)
            windows.extend((ms, ln, src, mod) for ms, ln, src in w)
            unbounded.extend((mod, ln, src) for ln, src in u)
            stack.extend(local_imports(mpath))
        deepest = max(windows, default=(0, 0, "", role))
        report.append({
            "role": role,
            "modules_walked": len(seen),
            "deepest_ms": deepest[0],
            "deepest_days": round(deepest[0] / DAY_MS, 2),
            "deepest_at": f"{deepest[3]}:{deepest[1]}",
            "via_import": deepest[3] != role,
            "unbounded_aggregates": [f"{m}:{ln}" for m, ln, _ in unbounded],
        })
    report.sort(key=lambda r: r.get("deepest_ms", 0), reverse=True)
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    args = ap.parse_args(argv)

    report = audit()
    if args.json:
        print(json.dumps(report, indent=2))
        return 0

    print(f"{'role':46s} {'deepest':>9s} {'mods':>5s}  where")
    for row in report:
        if "status" in row:
            print(f"{row['role']:46s} {row['status']}")
            continue
        tag = "VIA IMPORT" if row["via_import"] else "own file"
        print(f"{row['role']:46s} {row['deepest_days']:>7.2f}d {row['modules_walked']:>5d}  "
              f"{row['deepest_at']} [{tag}]")
    deepest = max((r.get("deepest_ms", 0) for r in report), default=0)
    print(f"\nDEEPEST DECLARED STANDING REACH = {deepest / DAY_MS:.2f} days")

    flagged = [r for r in report if r.get("unbounded_aggregates")]
    if flagged:
        print("\nUNBOUNDED AGGREGATES (no ts_ms predicate -> reach is the whole estate):")
        for row in flagged:
            for site in row["unbounded_aggregates"]:
                print(f"  {row['role']:44s} {site}")
    print("\nLOWER BOUND ONLY: dynamic windows and cold-start bootstraps are not detected.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
