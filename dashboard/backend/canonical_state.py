"""Bounded, parse-safe readers for governance state.

Reads the canonical governance state from ``SYSTEM_STATE.md`` (bounded tail read,
never wholesale) and lightweight git identity (branch / short HEAD) by parsing
git *plumbing files directly* — NO subprocess is ever spawned, honouring the
dashboard's no-process-management / no-subprocess safety contract.

Ahead/behind and the full working-tree dirty partition genuinely require git
plumbing that is impractical without the git binary; those are reported as
``INFERRED`` / unavailable rather than shelling out.
"""
from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any

# Bound how much of SYSTEM_STATE.md we ever read (tail). The file grows section
# by section; the newest (highest-numbered) section is at the end.
SYSTEM_STATE_TAIL_BYTES = 96 * 1024

_SECTION_RE = re.compile(r"^##\s+(\d+)\.\s+(.*)$", re.MULTILINE)
_STATE_TOKEN_RE = re.compile(r"`([A-Z0-9_]{8,})`")


def _read_tail(path: Path, max_bytes: int) -> tuple[str, int, bool]:
    """Return (text, total_size, truncated). Fail-safe on any error."""
    try:
        size = path.stat().st_size
    except OSError:
        return "", 0, False
    try:
        with path.open("rb") as f:
            if size > max_bytes:
                f.seek(size - max_bytes)
                truncated = True
            else:
                truncated = False
            data = f.read()
        return data.decode("utf-8", errors="replace"), size, truncated
    except OSError:
        return "", size, False


def read_canonical_state(system_state_path: Path) -> dict[str, Any]:
    """Extract the latest governance section + canonical state token(s)."""
    read_ts = time.time()
    if not system_state_path.exists():
        return {
            "parse_status": "missing",
            "read_timestamp": read_ts,
            "source_path": str(system_state_path),
        }
    text, size, truncated = _read_tail(system_state_path, SYSTEM_STATE_TAIL_BYTES)
    try:
        mtime = system_state_path.stat().st_mtime
    except OSError:
        mtime = None

    sections = list(_SECTION_RE.finditer(text))
    latest_number = None
    latest_title = None
    latest_body = ""
    if sections:
        # If we truncated, the first captured section may be partial; use the
        # last (highest offset) which is always complete at the tail.
        last = sections[-1]
        try:
            latest_number = int(last.group(1))
        except ValueError:
            latest_number = None
        latest_title = last.group(2).strip()
        latest_body = text[last.end():]

    # Canonical state token: prefer an explicit "Kanonik ... durum" line, else
    # the last backticked ALL_CAPS token in the latest section body.
    canonical_state = None
    for m in re.finditer(r"[Kk]anonik[^\n]*durum[^\n]*", latest_body):
        toks = _STATE_TOKEN_RE.findall(latest_body[m.start(): m.start() + 400])
        if toks:
            canonical_state = toks[-1]
            break
    if canonical_state is None:
        toks = _STATE_TOKEN_RE.findall(latest_body)
        canonical_state = toks[-1] if toks else None

    return {
        "parse_status": "ok" if sections else "malformed",
        "read_timestamp": read_ts,
        "source_path": str(system_state_path),
        "source_timestamp": mtime,
        "size_bytes": size,
        "tail_truncated": truncated,
        "latest_section_number": latest_number,
        "latest_section_title": latest_title,
        "canonical_state": canonical_state,
        "section_count_in_tail": len(sections),
    }


def extract_open_findings(system_state_path: Path) -> dict[str, Any]:
    """Best-effort, bounded scan for open findings by severity in the tail."""
    text, _size, _trunc = _read_tail(system_state_path, SYSTEM_STATE_TAIL_BYTES)
    counts = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}
    findings: list[dict[str, str]] = []
    # Matches lines like: **F-DASH-01** (LOW): ...  /  F-NEW-03 ... MEDIUM
    for m in re.finditer(r"(F-[A-Z0-9-]+)\b[^\n]*?\b(CRITICAL|HIGH|MEDIUM|LOW)\b", text):
        fid, sev = m.group(1), m.group(2)
        counts[sev] = counts.get(sev, 0) + 1
        if len(findings) < 40:
            findings.append({"id": fid, "severity": sev})
    return {"counts": counts, "findings": findings}


# ---------------------------------------------------------------------------
# git plumbing (no subprocess)
# ---------------------------------------------------------------------------

def _resolve_git_dirs(repo_root: Path) -> tuple[Path | None, Path | None]:
    """Return (gitdir, commondir) for a normal checkout or a linked worktree."""
    dotgit = repo_root / ".git"
    try:
        if dotgit.is_dir():
            return dotgit, dotgit
        if dotgit.is_file():
            content = dotgit.read_text(encoding="utf-8", errors="replace").strip()
            if content.startswith("gitdir:"):
                gitdir = Path(content.split(":", 1)[1].strip())
                if not gitdir.is_absolute():
                    gitdir = (repo_root / gitdir).resolve()
                commondir = gitdir
                cf = gitdir / "commondir"
                if cf.exists():
                    rel = cf.read_text(encoding="utf-8", errors="replace").strip()
                    cd = Path(rel)
                    commondir = cd if cd.is_absolute() else (gitdir / rel).resolve()
                return gitdir, commondir
    except OSError:
        return None, None
    return None, None


def read_git_identity(repo_root: Path) -> dict[str, Any]:
    """Read branch + HEAD sha by parsing plumbing files (no subprocess)."""
    read_ts = time.time()
    out: dict[str, Any] = {
        "read_timestamp": read_ts,
        "branch": None,
        "head_sha": None,
        "head_short": None,
        "detached": False,
        "ahead_behind": None,  # requires git plumbing beyond static files
        "provenance": "direct_plumbing_read",
        "parse_status": "ok",
    }
    gitdir, commondir = _resolve_git_dirs(repo_root)
    if gitdir is None:
        out["parse_status"] = "missing"
        return out
    try:
        head = (gitdir / "HEAD").read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        out["parse_status"] = "malformed"
        return out

    sha = None
    if head.startswith("ref:"):
        ref = head.split(":", 1)[1].strip()
        out["branch"] = ref.removeprefix("refs/heads/")
        # loose ref first (check worktree gitdir then commondir), then packed-refs
        for base in (gitdir, commondir):
            rp = base / ref
            if rp.exists():
                try:
                    sha = rp.read_text(encoding="utf-8", errors="replace").strip()
                    break
                except OSError:
                    pass
        if sha is None and commondir is not None:
            pr = commondir / "packed-refs"
            if pr.exists():
                try:
                    for line in pr.read_text(encoding="utf-8", errors="replace").splitlines():
                        line = line.strip()
                        if not line or line.startswith("#") or line.startswith("^"):
                            continue
                        parts = line.split(" ", 1)
                        if len(parts) == 2 and parts[1] == ref:
                            sha = parts[0]
                            break
                except OSError:
                    pass
    else:
        out["detached"] = True
        sha = head or None

    if sha and re.fullmatch(r"[0-9a-f]{7,40}", sha):
        out["head_sha"] = sha
        out["head_short"] = sha[:12]
    return out
