"""Assemble the Eclipse public mirror from an explicit allowlist.

    python docs/maintenance/tools/build_public_mirror.py --plan          # decide only
    python docs/maintenance/tools/build_public_mirror.py --build         # write the tree
    python docs/maintenance/tools/build_public_mirror.py --verify        # audit an existing tree

The public repository is **not** a redacted copy of the internal one. It is a
separate publication artifact assembled only from files an allowlist names.
That direction matters: with a blacklist, forgetting an entry publishes
something; with an allowlist, forgetting an entry publishes nothing.

This tool never touches the internal repository. It reads, it does not write
there, and it does not run `git` at all beyond `ls-files`.

The rules live in ``docs/maintenance/public_allowlist.json``, each with its reason.
Three of them are worth stating here because they carry the design:

* **Tools are excluded by default.** ``tools/`` is 701 files. Only the modules
  CI actually invokes are published, plus their first-party import closure.
  The seed is *what must work*, not *what looks safe*.

* **A test follows its subject.** A ``tests/`` file is published only if its
  entire transitive first-party import closure is also published. A test for an
  internal module is internal, whatever its fixtures look like — which is the
  only reliable way to tell a synthetic number from a copied one.

* **A boundary is documented, not fabricated.** If an included module reaches a
  file that is not published, that is reported as a violation and the build
  refuses. Nothing is stubbed to make a check pass.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

FIRST_PARTY = {
    "execution", "risk", "brain", "exchanges", "strategies", "core", "bot",
    "config", "utils", "monitoring", "integrations", "notifications", "data",
    "dashboard", "src", "tools", "ami", "tests", "web", "scripts",
}


def load_rules(root: Path) -> dict:
    return json.loads((root / "docs/maintenance/public_allowlist.json").read_text(encoding="utf-8"))


def tracked(root: Path) -> list[str]:
    """Everything that would exist on a remote after a commit.

    `git ls-files` alone lists only what is already in the index, which silently
    dropped the entire new public documentation set on the first build — the
    files most obviously belonging in the mirror were the ones missing from it.
    `--others --exclude-standard` adds untracked-but-not-ignored paths; ignored
    ones stay out, which is what keeps secrets and databases from arriving here.
    """
    out = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=root, capture_output=True, text=True, check=True, timeout=300,
    ).stdout
    return sorted(out.splitlines())


def _glob_re(pattern: str) -> re.Pattern[str]:
    """Translate a path glob to a regex with real `**` semantics.

    `fnmatch` maps `*` to `.*`, so `execution/**/*.py` fails to match
    `execution/order_router.py` — the `**` demands a directory that is not
    there. That silently published three files out of sixty on the first run,
    which is exactly the kind of quiet under-match an allowlist must not have.

      **/  matches any number of directories, including none
      **   matches anything, `/` included
      *    matches anything except `/`
      ?    matches one character except `/`
    """
    out, i = [], 0
    while i < len(pattern):
        if pattern.startswith("**/", i):
            out.append("(?:[^/]+/)*")
            i += 3
        elif pattern.startswith("**", i):
            out.append(".*")
            i += 2
        elif pattern[i] == "*":
            out.append("[^/]*")
            i += 1
        elif pattern[i] == "?":
            out.append("[^/]")
            i += 1
        else:
            out.append(re.escape(pattern[i]))
            i += 1
    return re.compile("^" + "".join(out) + "$")


_GLOB_CACHE: dict[str, re.Pattern[str]] = {}


def matches(path: str, patterns: list[str]) -> bool:
    for p in patterns:
        rx = _GLOB_CACHE.get(p)
        if rx is None:
            rx = _GLOB_CACHE[p] = _glob_re(p)
        if rx.match(path):
            return True
    return False


def toplevel_imports(root: Path, rel: str) -> set[str]:
    """Module names imported at module scope — the ones that must resolve at import time.

    An import nested inside a function or a `try` is a boundary the module
    already handles; an import at module scope is a hard dependency. The
    difference decides whether an unpublished dependency is a documented edge
    or a broken tree, so it is measured rather than asserted.
    """
    p = root / rel
    if p.suffix != ".py":
        return set()
    try:
        tree = ast.parse(p.read_text(encoding="utf-8-sig", errors="replace"), filename=rel)
    except (OSError, SyntaxError):
        return set()
    names: set[str] = set()
    for node in tree.body:  # module scope only
        if isinstance(node, ast.Import):
            names |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module)
    return names


def _is_toplevel(dep: str, hard_names: set[str]) -> bool:
    """Is this resolved dependency reached by a module-scope import?"""
    mod = dep[:-3].replace("/", ".") if dep.endswith(".py") else dep
    if mod.endswith(".__init__"):
        mod = mod[: -len(".__init__")]
    return any(h == mod or h.startswith(mod + ".") or mod.startswith(h + ".") for h in hard_names)


def first_party_imports(root: Path, rel: str, tracked_set: set[str]) -> set[str]:
    """Resolve a module's first-party imports to repository paths.

    Uses the AST rather than a regex so a string containing the word `import`
    is not mistaken for one.
    """
    p = root / rel
    if p.suffix != ".py":
        return set()
    try:
        tree = ast.parse(p.read_text(encoding="utf-8-sig", errors="replace"), filename=rel)
    except (OSError, SyntaxError):
        return set()

    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module)
            names |= {f"{node.module}.{a.name}" for a in node.names}

    out: set[str] = set()
    for name in names:
        if name.split(".")[0] not in FIRST_PARTY:
            continue
        stem = name.replace(".", "/")
        for cand in (f"{stem}.py", f"{stem}/__init__.py"):
            if cand in tracked_set:
                out.add(cand)
                break
        else:
            # `from pkg.mod import thing` — the module, not the symbol
            parent = "/".join(stem.split("/")[:-1])
            for cand in (f"{parent}.py", f"{parent}/__init__.py"):
                if cand in tracked_set:
                    out.add(cand)
                    break
    return out


def decide(root: Path) -> tuple[list[str], dict[str, list[str]], list[str]]:
    """Return (published, violations_by_file, dropped_tests)."""
    rules = load_rules(root)
    files = tracked(root)
    tracked_set = set(files)

    inc = [r["pattern"] for r in rules["include"]]
    exc = [r["pattern"] for r in rules["exclude"]]
    named_tools = set(rules["named_tools"]["paths"])
    named_exc = {e["path"] for e in rules["named_exceptions"]["paths"]}

    excluded_prefixes = ("dashboard/", "web/", "scripts/", "tools/", "eclipse_scalper/")

    base: set[str] = set()
    for f in files:
        if matches(f, exc):
            continue
        if f in named_tools or f in named_exc:
            base.add(f)
            continue
        if f.startswith(excluded_prefixes):
            continue
        if matches(f, inc):
            base.add(f)

    # tests are provisional until their closure is checked
    tests = {f for f in base if f.startswith("tests/")}
    non_tests = base - tests

    # A reach into the unpublished is a violation only if it is a *hard* one.
    # An import nested in a function or a try/except is a boundary the module
    # already handles; the tool checks that rather than taking the allowlist's
    # word for it, so a declared boundary that is really a module-level import
    # still fails the build.
    violations: dict[str, list[str]] = defaultdict(list)
    soft: dict[str, list[str]] = defaultdict(list)
    for f in sorted(non_tests):
        hard_names = toplevel_imports(root, f)
        for dep in first_party_imports(root, f, tracked_set):
            if dep in base:
                continue
            mod = dep[:-3].replace("/", ".") if dep.endswith(".py") else dep
            mod = mod[: -len(".__init__")] if mod.endswith(".__init__") else mod
            is_hard = any(h == mod or h.startswith(mod + ".") or mod.startswith(h + ".")
                          for h in hard_names)
            (violations if is_hard else soft)[f].append(dep)
    decide.soft_boundaries = dict(soft)  # type: ignore[attr-defined]

    # a test survives only if its full closure is inside the published set
    published_modules = non_tests
    kept_tests: set[str] = set()
    dropped: list[str] = []
    # The same hard/soft distinction the module pass uses. Following *every*
    # import here — lazy ones included — dropped a test whose only unpublished
    # reach was a guarded lazy import three modules away, and one of the tests
    # it cost was the paper-mode safety test, which is exactly the kind of thing
    # a public engineering repository exists to let a reader run.
    # Laziness means different things on either side of this walk.
    #
    # In a *production* module a lazy import is a guarded optional path: the
    # code already handles the module being absent, so it is a boundary, not a
    # dependency. Following those dropped the paper-mode safety test over a
    # guarded import three modules away.
    #
    # In a *test* a lazy import is the subject under test — deferring it to
    # inside the test function is a fixture convention, not a fallback. Treating
    # those as boundaries admitted a batch of research-pipeline tests whose
    # subject is not published and which would fail on import at run time.
    #
    # So: every import a test file makes counts; only top-level imports count
    # once the walk has crossed into production code.
    for t in sorted(tests):
        seen: set[str] = set()
        frontier = {t}
        ok = True
        while frontier:
            nxt: set[str] = set()
            for f in frontier:
                if f in seen:
                    continue
                seen.add(f)
                is_test = f.startswith("tests/")
                deps = (first_party_imports(root, f, tracked_set) if is_test
                        else {d for d in first_party_imports(root, f, tracked_set)
                              if _is_toplevel(d, toplevel_imports(root, f))})
                for dep in deps:
                    if dep in published_modules or dep.startswith("tests/"):
                        nxt.add(dep)
                    else:
                        ok = False
            frontier = nxt - seen
        (kept_tests.add(t) if ok else dropped.append(t))

    return sorted(non_tests | kept_tests), dict(violations), dropped


SANITIZE_ABS = re.compile(r"[A-Z]:[\\/]{1,2}eclipse_scalper[\\/]?")

# A veto, not a filter. The allowlist decides what may be published; this
# refuses to publish it anyway if the file carries research provenance. It
# exists because the allowlist's patterns are about *location* and these
# markers are prose: a collector whose docstring names the study it was built
# for, or a config carrying frozen window parameters, matches no numeric
# pattern and no path rule. Three such files reached a built mirror before this
# gate existed.
RESEARCH_PROVENANCE = [
    ("study-id", re.compile(r"\bS\d{2,3}\b(?!\s*(?:tatus|ystem))")),
    # Lower-case study ids hide inside identifiers: `research_s34_...`, `_s40_`.
    # The upper-case word-boundary pattern above walks straight past them, and a
    # test whose docstring named a study, a signal count and internal batch
    # tokens reached a built mirror before this line existed.
    ("study-id-in-identifier", re.compile(r"(?:^|[_/])s\d{2,3}[_/]", re.I | re.M)),
    ("state-section", re.compile(r"SYSTEM_STATE\s*§|§\s*\d{2,3}\b")),
    ("preregistration", re.compile(r"\bprereg(?:istration)?\b", re.I)),
    ("estimand", re.compile(r"\bestimand\b|\bNET_MEASURED\b", re.I)),
    ("frozen-window", re.compile(r"\bparent_window_minutes\b|\banchor_min_gap\w*\b|\banchor_bucket\w*\b")),
    ("sealed-arm", re.compile(r"\bqualified_t0\b|\bforward_ledger\b|\bsealed\s+arm\b", re.I)),
    ("internal-report-path", re.compile(r"reports/(?:research|shadow|governance)/")),
]
VETO_SUFFIXES = {".py", ".md", ".json", ".yml", ".yaml", ".ps1", ".bat", ".txt", ".ini"}


def veto(root: Path, rel: str, allowed: set[str], rules: dict) -> list[str]:
    """Research-provenance markers in the text that would actually be published.

    The veto reads the *post-sanitization* content, not the source. Checking the
    source would fail every file a declared redaction is about to fix, and —
    far worse the other way round — would pass a file whose redaction silently
    stopped matching. What ships is what gets judged.
    """
    if rel in allowed or Path(rel).suffix.lower() not in VETO_SUFFIXES:
        return []
    try:
        text = (root / rel).read_text(encoding="utf-8-sig", errors="replace")[:300_000]
    except OSError:
        return []
    text = sanitized_text(rel, text, rules)
    return [name for name, pat in RESEARCH_PROVENANCE if pat.search(text)]


REF_RE = re.compile(r"reports/(?:research|shadow|governance)/[A-Za-z0-9_./-]+")


JOB_KEY = re.compile(r"^  [a-z][a-z0-9-]*:\s*$")


def drop_yaml_jobs(text: str, names: list[str]) -> str:
    """Remove whole jobs from a workflow, by key, without a YAML library.

    Jobs are two-space keys under `jobs:`; a job ends where the next such key
    begins. Done textually so the file's formatting and comments survive
    verbatim — a round-trip through a YAML loader would rewrite the whole file
    and make the mirror's CI hard to diff against the internal one.
    """
    lines = text.splitlines(keepends=True)
    starts = [i for i, l in enumerate(lines) if JOB_KEY.match(l)]
    drop: set[int] = set()
    for idx, i in enumerate(starts):
        if lines[i].strip().rstrip(":") in names:
            end = starts[idx + 1] if idx + 1 < len(starts) else len(lines)
            drop |= set(range(i, end))
    return "".join(l for i, l in enumerate(lines) if i not in drop)


def sanitized_text(rel: str, text: str, rules: dict) -> str:
    """Apply the declared sanitizations for one file. Single source of truth for
    both the veto and the copy, so the two can never disagree."""
    for spec in rules.get("sanitize", []):
        if spec.get("kind") == "reference_redaction" and rel in spec.get("applies_to", []):
            text = REF_RE.sub("an internal research report", text)
        elif spec.get("path") == rel and spec.get("kind") == "replace":
            for r in spec["replacements"]:
                text = text.replace(r["find"], r["replace"])
        elif spec.get("path") == rel and spec.get("kind") == "drop_jobs":
            text = drop_yaml_jobs(text, spec["jobs"])
            for r in spec.get("replacements", []):
                text = text.replace(r["find"], r["replace"])
        elif spec.get("path") == rel and spec.get("kind") == "absolute_path":
            token = "$PSScriptRoot\\.." if rel.endswith(".ps1") else "%~dp0.."
            text = SANITIZE_ABS.sub(token + "\\\\", text)
    return text


def build(root: Path, target: Path, published: list[str], rules: dict) -> dict:
    """Copy the decided set, applying only the declared sanitizations.

    Every edit the mirror makes to a file is declared in the allowlist with its
    reason. A redaction that is not declared cannot happen here, and a declared
    one that fails to apply is an error rather than a silent no-op — otherwise a
    leaked line could survive a rename of the text around it.
    """
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)

    # One sanitizer, used by both the copy and the veto. An earlier version had
    # the logic written out twice; the copy then silently ignored a rule kind
    # the veto knew about, and reported "nothing changed" for a file it had in
    # fact never tried to change. Two implementations of one rule is one too
    # many.
    touched_paths = {s["path"] for s in rules.get("sanitize", [])
                     if s["path"] != "*" and s.get("kind") != "manual"}
    ref_rule = next((s for s in rules.get("sanitize", [])
                     if s.get("kind") == "reference_redaction"), None)
    touched_paths |= set(ref_rule["applies_to"]) if ref_rule else set()

    stats = Counter()
    sanitized: list[str] = []
    failed: list[str] = []

    for rel in published:
        src, dst = root / rel, target / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if rel not in touched_paths:
            shutil.copy2(src, dst)
            stats["copied"] += 1
            continue

        original = src.read_text(encoding="utf-8-sig", errors="replace")
        text = sanitized_text(rel, original, rules)
        if text == original:
            failed.append(f"{rel}: sanitization declared but nothing changed")
        dst.write_text(text, encoding="utf-8")
        sanitized.append(rel)
        stats["sanitized"] += 1

    return {"stats": dict(stats), "sanitized": sanitized, "failed": failed}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=".")
    ap.add_argument("--target", default=None)
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    if not (root / ".git").exists():
        for parent in Path(__file__).resolve().parents:
            if (parent / ".git").exists():
                root = parent
                break
    rules = load_rules(root)
    target = Path(args.target or rules["target"])

    published, violations, dropped = decide(root)

    # The content veto runs on the decided set, before anything is written.
    exempt = set(rules.get("veto_exempt", {}).get("paths", []))
    vetoed: dict[str, list[str]] = {}
    for rel in published:
        marks = veto(root, rel, exempt, rules)
        if marks:
            vetoed[rel] = marks

    if args.list:
        for p in published:
            print(p)
        return 0

    byd = Counter(p.split("/")[0] if "/" in p else "«root»" for p in published)
    print(f"allowlist decides {len(published)} files\n")
    for d, n in byd.most_common():
        print(f"  {d:16s} {n}")
    print(f"\n  tests dropped for reaching unpublished modules: {len(dropped)}")

    soft = getattr(decide, "soft_boundaries", {})
    if soft:
        declared = {
            b["where"].split(" -> ")[0]
            for b in rules.get("documented_boundaries", {}).get("boundaries", [])
        }
        print(f"\n  documented boundaries (lazy imports the code already guards): {len(soft)}")
        for f, deps in sorted(soft.items()):
            mark = "declared" if f in declared else "UNDECLARED"
            print(f"    [{mark}] {f} -> {', '.join(sorted(set(deps)))}")
        undeclared = [f for f in soft if f not in declared]
        if undeclared:
            print("\n  Every lazy boundary must be declared in `documented_boundaries`.")
            return 1

    if violations:
        print(f"\nBOUNDARY VIOLATIONS — {len(violations)} published file(s) import "
              f"something unpublished:\n")
        for f, deps in sorted(violations.items())[:40]:
            print(f"  {f}")
            for d in sorted(set(deps)):
                print(f"      -> {d}")
        print("\nRefusing to build. Either name the dependency in "
              "`named_exceptions` after reading it, or drop the importer.")
        return 1

    # Every published .py must parse. A public repository that ships a file
    # Python cannot read is broken in the most visible way available, and CI
    # will not necessarily catch it: the one file this found is not in any
    # required job's list, so the whole suite failed to collect while every
    # gate stayed green.
    unparsable: list[str] = []
    for rel in published:
        if not rel.endswith(".py"):
            continue
        try:
            ast.parse((root / rel).read_text(encoding="utf-8-sig", errors="replace"), filename=rel)
        except SyntaxError as e:
            unparsable.append(f"{rel}:{e.lineno}  {e.msg}")
    if unparsable:
        print(f"\nSYNTAX GATE — {len(unparsable)} published file(s) do not parse:\n")
        for u in unparsable:
            print("  " + u)
        print("\nRefusing to build. Fix it in the internal repository, or exclude it "
              "with a reason — do not publish a file Python cannot read.")
        return 1

    if vetoed:
        print(f"\nCONTENT VETO — {len(vetoed)} allowlisted file(s) carry research provenance:\n")
        for f, marks in sorted(vetoed.items()):
            print(f"  {f}\n      {', '.join(marks)}")
        print("\nRefusing to build. Either exclude the file, or exempt it in "
              "`veto_exempt` after reading it — the public documentation that "
              "*describes* this policy necessarily quotes its own vocabulary.")
        return 1

    if args.build:
        out = build(root, target, published, rules)
        print(f"\nbuilt {target}")
        print(f"  copied    {out['stats'].get('copied', 0)}")
        print(f"  sanitized {out['stats'].get('sanitized', 0)}")
        for s in out["sanitized"]:
            print(f"      {s}")
        if out["failed"]:
            print("\nSANITIZATION PROBLEMS:")
            for f in out["failed"]:
                print("  " + f)
            return 1
    elif not args.plan:
        print("\n(nothing written — pass --build to write the tree)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
