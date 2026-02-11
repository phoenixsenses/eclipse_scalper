from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict


_CACHE: Dict[str, Any] = {
    "path": "",
    "mtime": -1.0,
    "result": {},
}


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return int(default)


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        if v != v:
            return float(default)
        return v
    except Exception:
        return float(default)


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, float(v)))


def _cfg(cfg: Any, name: str, default: Any) -> Any:
    try:
        return getattr(cfg, name, default)
    except Exception:
        return default


def _resolve_path(cfg: Any = None) -> Path:
    p = str(os.getenv("RELIABILITY_GATE_PATH", "")).strip()
    if not p:
        p = str(_cfg(cfg, "RELIABILITY_GATE_PATH", "") or "").strip()
    if not p:
        p = "logs/reliability_gate.txt"
    return Path(p)


def _parse_kv(path: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    try:
        raw = path.read_text(encoding="utf-8")
    except Exception:
        return out
    for line in raw.splitlines():
        s = str(line or "").strip()
        if not s or "=" not in s:
            continue
        k, v = s.split("=", 1)
        key = str(k or "").strip()
        if not key:
            continue
        out[key] = str(v or "").strip()
    return out


def _parse_replay_mismatch_ids(path: Path, limit: int = 5) -> list[str]:
    out: list[str] = []
    try:
        raw = path.read_text(encoding="utf-8")
    except Exception:
        return out
    in_ids = False
    for line in raw.splitlines():
        s = str(line or "").strip()
        if not s:
            if in_ids:
                break
            continue
        if s.lower() == "replay_mismatch_ids:":
            in_ids = True
            continue
        if in_ids:
            if not s.startswith("-"):
                break
            cid = str(s[1:] or "").strip()
            if cid:
                out.append(cid)
                if len(out) >= max(1, int(limit)):
                    break
    return out


def _categorize_mismatch_ids(ids: list[str]) -> Dict[str, int]:
    categories: Dict[str, int] = {
        "ledger": 0,
        "transition": 0,
        "belief": 0,
        "unknown": 0,
    }
    for raw in ids:
        cid = str(raw or "").strip().lower()
        if not cid:
            continue
        if cid.startswith("led-") or "ledger" in cid:
            categories["ledger"] = int(categories.get("ledger", 0)) + 1
        elif cid.startswith("trn-") or "transition" in cid:
            categories["transition"] = int(categories.get("transition", 0)) + 1
        elif cid.startswith("blf-") or "belief" in cid:
            categories["belief"] = int(categories.get("belief", 0)) + 1
        else:
            categories["unknown"] = int(categories.get("unknown", 0)) + 1
    return categories


def _parse_categories(raw: Any) -> Dict[str, int]:
    out = {"ledger": 0, "transition": 0, "belief": 0, "unknown": 0}
    s = str(raw or "").strip()
    if not s:
        return out
    try:
        payload = json.loads(s)
    except Exception:
        return out
    if not isinstance(payload, dict):
        return out
    for k in ("ledger", "transition", "belief", "unknown"):
        out[k] = max(0, _safe_int(payload.get(k), 0))
    return out


def get_runtime_gate(cfg: Any = None) -> Dict[str, Any]:
    path = _resolve_path(cfg)
    max_mismatch = max(0, _safe_int(os.getenv("RELIABILITY_GATE_MAX_REPLAY_MISMATCH", _cfg(cfg, "RELIABILITY_GATE_MAX_REPLAY_MISMATCH", 0)), 0))
    max_invalid = max(0, _safe_int(os.getenv("RELIABILITY_GATE_MAX_INVALID_TRANSITIONS", _cfg(cfg, "RELIABILITY_GATE_MAX_INVALID_TRANSITIONS", 0)), 0))
    min_cov = max(0.0, min(1.0, _safe_float(os.getenv("RELIABILITY_GATE_MIN_JOURNAL_COVERAGE", _cfg(cfg, "RELIABILITY_GATE_MIN_JOURNAL_COVERAGE", 0.90)), 0.90)))
    cat_score_threshold = _clamp01(
        _safe_float(os.getenv("RELIABILITY_GATE_CATEGORY_DEGRADE_SCORE", _cfg(cfg, "RELIABILITY_GATE_CATEGORY_DEGRADE_SCORE", 0.80)), 0.80)
    )

    if not path.exists():
        return {
            "available": False,
            "degraded": False,
            "reason": "missing",
            "replay_mismatch_count": 0,
            "invalid_transition_count": 0,
            "journal_coverage_ratio": 1.0,
            "mismatch_severity": 0.0,
            "invalid_severity": 0.0,
            "coverage_severity": 0.0,
            "mismatch_category_score": 0.0,
            "degrade_score": 0.0,
            "replay_mismatch_ids": [],
            "replay_mismatch_categories": {"ledger": 0, "transition": 0, "belief": 0, "unknown": 0},
            "path": str(path),
        }

    try:
        st = path.stat()
        mtime = float(st.st_mtime)
    except Exception:
        mtime = -1.0

    cache_path = str(path)
    if _CACHE.get("path") == cache_path and float(_CACHE.get("mtime", -1.0)) == mtime:
        cached = dict(_CACHE.get("result") or {})
        cached["path"] = cache_path
        return cached

    kv = _parse_kv(path)
    mismatch_ids = _parse_replay_mismatch_ids(path, limit=5)
    mismatch_categories = _parse_categories(kv.get("replay_mismatch_categories"))
    if not any(int(mismatch_categories.get(k, 0) or 0) > 0 for k in ("ledger", "transition", "belief", "unknown")):
        mismatch_categories = _categorize_mismatch_ids(mismatch_ids)
    mismatch = max(0, _safe_int(kv.get("replay_mismatch_count"), 0))
    invalid = max(0, _safe_int(kv.get("invalid_transition_count"), 0))
    coverage = max(0.0, min(1.0, _safe_float(kv.get("journal_coverage_ratio"), 0.0)))

    mismatch_excess = max(0, mismatch - max_mismatch)
    invalid_excess = max(0, invalid - max_invalid)
    mismatch_severity = max(0.0, min(1.0, float(mismatch_excess) / float(max(1, max_mismatch + 1))))
    invalid_severity = max(0.0, min(1.0, float(invalid_excess) / float(max(1, max_invalid + 1))))
    if coverage >= min_cov:
        coverage_severity = 0.0
    else:
        denom = max(1e-6, float(min_cov))
        coverage_severity = max(0.0, min(1.0, float(min_cov - coverage) / denom))
    cat_ledger = int(mismatch_categories.get("ledger", 0) or 0)
    cat_transition = int(mismatch_categories.get("transition", 0) or 0)
    cat_belief = int(mismatch_categories.get("belief", 0) or 0)
    cat_unknown = int(mismatch_categories.get("unknown", 0) or 0)
    mismatch_category_score = max(
        min(1.0, float(cat_ledger) / 2.0),
        min(1.0, float(cat_transition) / 2.0),
        min(1.0, float(cat_belief) / 3.0),
        min(1.0, float(cat_unknown) / 4.0),
    )
    degrade_score = max(mismatch_severity, invalid_severity, coverage_severity, mismatch_category_score)

    degraded = bool(
        mismatch > max_mismatch
        or invalid > max_invalid
        or coverage < min_cov
        or mismatch_category_score >= cat_score_threshold
    )
    reason_parts = []
    if mismatch > max_mismatch:
        reason_parts.append(f"mismatch>{max_mismatch}")
    if invalid > max_invalid:
        reason_parts.append(f"invalid>{max_invalid}")
    if coverage < min_cov:
        reason_parts.append(f"coverage<{min_cov:.2f}")
    if cat_ledger > 0:
        reason_parts.append(f"ledger={cat_ledger}")
    if cat_transition > 0:
        reason_parts.append(f"transition={cat_transition}")
    if cat_belief > 0:
        reason_parts.append(f"belief={cat_belief}")
    if cat_unknown > 0:
        reason_parts.append(f"unknown={cat_unknown}")
    if mismatch_category_score >= cat_score_threshold:
        reason_parts.append(f"cat_score>={cat_score_threshold:.2f}")
    reason = ",".join(reason_parts) if reason_parts else "ok"
    out = {
        "available": True,
        "degraded": degraded,
        "reason": reason,
        "replay_mismatch_count": mismatch,
        "invalid_transition_count": invalid,
        "journal_coverage_ratio": coverage,
        "mismatch_severity": float(mismatch_severity),
        "invalid_severity": float(invalid_severity),
        "coverage_severity": float(coverage_severity),
        "mismatch_category_score": float(mismatch_category_score),
        "degrade_score": float(degrade_score),
        "replay_mismatch_ids": list(mismatch_ids),
        "replay_mismatch_categories": dict(mismatch_categories),
        "path": cache_path,
    }
    _CACHE["path"] = cache_path
    _CACHE["mtime"] = mtime
    _CACHE["result"] = dict(out)
    return out
