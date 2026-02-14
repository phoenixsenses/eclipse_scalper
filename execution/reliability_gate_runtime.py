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
_UNSET = object()


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


def _cfg_opt(cfg: Any, name: str) -> Any:
    try:
        return getattr(cfg, name, _UNSET)
    except Exception:
        return _UNSET


def _pick(cfg: Any, cfg_name: str, env_name: str, default: Any) -> Any:
    # Prefer explicit runtime config in tests/bootstrap, then env, then default.
    v = _cfg_opt(cfg, cfg_name)
    if v is not _UNSET:
        return v
    ev = os.getenv(env_name, None)
    if ev is None:
        return default
    return ev


def _resolve_path(cfg: Any = None) -> Path:
    p = str(_pick(cfg, "RELIABILITY_GATE_PATH", "RELIABILITY_GATE_PATH", "") or "").strip()
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
        "position": 0,
        "orphan": 0,
        "coverage_gap": 0,
        "stage1_protection_fail": 0,
        "replace_race": 0,
        "contradiction": 0,
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
        elif cid.startswith("pos-") or "position" in cid:
            categories["position"] = int(categories.get("position", 0)) + 1
        elif "orphan" in cid:
            categories["orphan"] = int(categories.get("orphan", 0)) + 1
        elif "coverage" in cid or "protect" in cid:
            categories["coverage_gap"] = int(categories.get("coverage_gap", 0)) + 1
        elif "replace" in cid:
            categories["replace_race"] = int(categories.get("replace_race", 0)) + 1
        elif "contrad" in cid:
            categories["contradiction"] = int(categories.get("contradiction", 0)) + 1
        else:
            categories["unknown"] = int(categories.get("unknown", 0)) + 1
    return categories


def _parse_categories(raw: Any) -> Dict[str, int]:
    out = {
        "ledger": 0,
        "transition": 0,
        "belief": 0,
        "position": 0,
        "orphan": 0,
        "coverage_gap": 0,
        "stage1_protection_fail": 0,
        "replace_race": 0,
        "contradiction": 0,
        "unknown": 0,
    }
    s = str(raw or "").strip()
    if not s:
        return out
    try:
        payload = json.loads(s)
    except Exception:
        return out
    if not isinstance(payload, dict):
        return out
    for k in (
        "ledger",
        "transition",
        "belief",
        "position",
        "orphan",
        "coverage_gap",
        "stage1_protection_fail",
        "replace_race",
        "contradiction",
        "unknown",
    ):
        out[k] = max(0, _safe_int(payload.get(k), 0))
    return out


def get_runtime_gate(cfg: Any = None) -> Dict[str, Any]:
    path = _resolve_path(cfg)
    max_mismatch = max(
        0, _safe_int(_pick(cfg, "RELIABILITY_GATE_MAX_REPLAY_MISMATCH", "RELIABILITY_GATE_MAX_REPLAY_MISMATCH", 0), 0)
    )
    max_invalid = max(
        0, _safe_int(_pick(cfg, "RELIABILITY_GATE_MAX_INVALID_TRANSITIONS", "RELIABILITY_GATE_MAX_INVALID_TRANSITIONS", 0), 0)
    )
    min_cov = max(
        0.0,
        min(
            1.0,
            _safe_float(_pick(cfg, "RELIABILITY_GATE_MIN_JOURNAL_COVERAGE", "RELIABILITY_GATE_MIN_JOURNAL_COVERAGE", 0.90), 0.90),
        ),
    )
    cat_score_threshold = _clamp01(
        _safe_float(_pick(cfg, "RELIABILITY_GATE_CATEGORY_DEGRADE_SCORE", "RELIABILITY_GATE_CATEGORY_DEGRADE_SCORE", 0.80), 0.80)
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
            "replay_mismatch_categories": {
                "ledger": 0,
                "transition": 0,
                "belief": 0,
                "position": 0,
                "orphan": 0,
                "coverage_gap": 0,
                "stage1_protection_fail": 0,
                "replace_race": 0,
                "contradiction": 0,
                "unknown": 0,
            },
            "position_mismatch_count": 0,
            "position_mismatch_count_peak": 0,
            "orphan_count": 0,
            "intent_collision_count": 0,
            "protection_coverage_gap_seconds": 0.0,
            "protection_coverage_gap_seconds_peak": 0.0,
            "stage1_protection_fail_count": 0,
            "replace_race_count": 0,
            "evidence_contradiction_count": 0,
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
    if not any(
        int(mismatch_categories.get(k, 0) or 0) > 0
        for k in (
            "ledger",
            "transition",
            "belief",
            "position",
            "orphan",
            "coverage_gap",
            "stage1_protection_fail",
            "replace_race",
            "contradiction",
            "unknown",
        )
    ):
        mismatch_categories = _categorize_mismatch_ids(mismatch_ids)
    mismatch = max(0, _safe_int(kv.get("replay_mismatch_count"), 0))
    invalid = max(0, _safe_int(kv.get("invalid_transition_count"), 0))
    coverage = max(0.0, min(1.0, _safe_float(kv.get("journal_coverage_ratio"), 0.0)))
    position_mismatch = max(0, _safe_int(kv.get("position_mismatch_count"), _safe_int(mismatch_categories.get("position"), 0)))
    position_mismatch_peak = max(0, _safe_int(kv.get("position_mismatch_count_peak"), position_mismatch))
    orphan_count = max(0, _safe_int(kv.get("orphan_count"), _safe_int(mismatch_categories.get("orphan"), 0)))
    intent_collision_count = max(0, _safe_int(kv.get("intent_collision_count"), 0))
    coverage_gap_seconds = max(0.0, _safe_float(kv.get("protection_coverage_gap_seconds"), 0.0))
    coverage_gap_seconds_peak = max(0.0, _safe_float(kv.get("protection_coverage_gap_seconds_peak"), coverage_gap_seconds))
    stage1_protection_fail_count = max(
        0,
        _safe_int(kv.get("stage1_protection_fail_count"), _safe_int(mismatch_categories.get("stage1_protection_fail"), 0)),
    )
    replace_race_count = max(0, _safe_int(kv.get("replace_race_count"), _safe_int(mismatch_categories.get("replace_race"), 0)))
    contradiction_count = max(
        0, _safe_int(kv.get("evidence_contradiction_count"), _safe_int(mismatch_categories.get("contradiction"), 0))
    )
    mismatch_categories["position"] = int(position_mismatch)
    mismatch_categories["orphan"] = int(orphan_count)
    mismatch_categories["replace_race"] = int(replace_race_count)
    mismatch_categories["contradiction"] = int(contradiction_count)
    mismatch_categories["coverage_gap"] = max(
        int(mismatch_categories.get("coverage_gap", 0) or 0), (1 if coverage_gap_seconds > 0.0 else 0)
    )
    mismatch_categories["stage1_protection_fail"] = int(stage1_protection_fail_count)

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
    cat_position = int(mismatch_categories.get("position", 0) or 0)
    cat_orphan = int(mismatch_categories.get("orphan", 0) or 0)
    cat_cov_gap = int(mismatch_categories.get("coverage_gap", 0) or 0)
    cat_stage1_fail = int(mismatch_categories.get("stage1_protection_fail", 0) or 0)
    cat_replace = int(mismatch_categories.get("replace_race", 0) or 0)
    cat_contradiction = int(mismatch_categories.get("contradiction", 0) or 0)
    cat_unknown = int(mismatch_categories.get("unknown", 0) or 0)
    cat_position_score = min(1.0, float(cat_position) / 2.0)
    cat_orphan_score = min(1.0, float(cat_orphan))
    cat_cov_gap_score = min(1.0, max(float(cat_cov_gap), (coverage_gap_seconds / 30.0)))
    cat_stage1_fail_score = min(1.0, float(cat_stage1_fail))
    cat_replace_score = min(1.0, float(cat_replace) / 2.0)
    cat_contradiction_score = min(1.0, float(cat_contradiction) / 2.0)
    mismatch_category_score = max(
        min(1.0, float(cat_ledger) / 2.0),
        min(1.0, float(cat_transition) / 2.0),
        min(1.0, float(cat_belief) / 3.0),
        cat_position_score,
        cat_orphan_score,
        cat_cov_gap_score,
        cat_stage1_fail_score,
        cat_replace_score,
        cat_contradiction_score,
        min(1.0, float(cat_unknown) / 4.0),
    )
    degrade_score = max(mismatch_severity, invalid_severity, coverage_severity, mismatch_category_score)
    max_position = max(
        0, _safe_int(_pick(cfg, "RELIABILITY_GATE_MAX_POSITION_MISMATCH", "RELIABILITY_GATE_MAX_POSITION_MISMATCH", 1), 1)
    )
    max_position_peak = max(
        0,
        _safe_int(
            _pick(
                cfg,
                "RELIABILITY_GATE_MAX_POSITION_MISMATCH_PEAK",
                "RELIABILITY_GATE_MAX_POSITION_MISMATCH_PEAK",
                max_position,
            ),
            max_position,
        ),
    )
    max_orphan = max(0, _safe_int(_pick(cfg, "RELIABILITY_GATE_MAX_ORPHAN_COUNT", "RELIABILITY_GATE_MAX_ORPHAN_COUNT", 0), 0))
    max_intent_collision = max(
        0,
        _safe_int(
            _pick(cfg, "RELIABILITY_GATE_MAX_INTENT_COLLISION_COUNT", "RELIABILITY_GATE_MAX_INTENT_COLLISION_COUNT", 0),
            0,
        ),
    )
    max_cov_gap_sec = max(
        0.0,
        _safe_float(
            _pick(cfg, "RELIABILITY_GATE_MAX_COVERAGE_GAP_SECONDS", "RELIABILITY_GATE_MAX_COVERAGE_GAP_SECONDS", 0.0),
            0.0,
        ),
    )
    max_cov_gap_sec_peak = max(
        0.0,
        _safe_float(
            _pick(
                cfg,
                "RELIABILITY_GATE_MAX_COVERAGE_GAP_SECONDS_PEAK",
                "RELIABILITY_GATE_MAX_COVERAGE_GAP_SECONDS_PEAK",
                max_cov_gap_sec,
            ),
            max_cov_gap_sec,
        ),
    )
    max_stage1_fail = max(
        0,
        _safe_int(
            _pick(
                cfg,
                "RELIABILITY_GATE_MAX_STAGE1_PROTECTION_FAIL_COUNT",
                "RELIABILITY_GATE_MAX_STAGE1_PROTECTION_FAIL_COUNT",
                0,
            ),
            0,
        ),
    )
    max_replace = max(
        0, _safe_int(_pick(cfg, "RELIABILITY_GATE_MAX_REPLACE_RACE_COUNT", "RELIABILITY_GATE_MAX_REPLACE_RACE_COUNT", 1), 1)
    )
    max_contradiction = max(
        0,
        _safe_int(
            _pick(
                cfg,
                "RELIABILITY_GATE_MAX_EVIDENCE_CONTRADICTION_COUNT",
                "RELIABILITY_GATE_MAX_EVIDENCE_CONTRADICTION_COUNT",
                2,
            ),
            2,
        ),
    )

    degraded = bool(
        mismatch > max_mismatch
        or invalid > max_invalid
        or coverage < min_cov
        or position_mismatch > max_position
        or position_mismatch_peak > max_position_peak
        or orphan_count > max_orphan
        or intent_collision_count > max_intent_collision
        or (coverage_gap_seconds > max_cov_gap_sec)
        or (coverage_gap_seconds_peak > max_cov_gap_sec_peak)
        or stage1_protection_fail_count > max_stage1_fail
        or replace_race_count > max_replace
        or contradiction_count > max_contradiction
        or mismatch_category_score >= cat_score_threshold
    )
    reason_parts = []
    if mismatch > max_mismatch:
        reason_parts.append(f"mismatch>{max_mismatch}")
    if invalid > max_invalid:
        reason_parts.append(f"invalid>{max_invalid}")
    if coverage < min_cov:
        reason_parts.append(f"coverage<{min_cov:.2f}")
    if position_mismatch > max_position:
        reason_parts.append(f"position>{max_position}")
    if position_mismatch_peak > max_position_peak:
        reason_parts.append(f"position_peak>{max_position_peak}")
    if orphan_count > max_orphan:
        reason_parts.append(f"orphan>{max_orphan}")
    if intent_collision_count > max_intent_collision:
        reason_parts.append(f"intent_collision>{max_intent_collision}")
    if coverage_gap_seconds > max_cov_gap_sec:
        reason_parts.append(f"coverage_gap_sec>{max_cov_gap_sec:.1f}")
    if coverage_gap_seconds_peak > max_cov_gap_sec_peak:
        reason_parts.append(f"coverage_gap_sec_peak>{max_cov_gap_sec_peak:.1f}")
    if stage1_protection_fail_count > max_stage1_fail:
        reason_parts.append(f"stage1_protection_fail>{max_stage1_fail}")
    if replace_race_count > max_replace:
        reason_parts.append(f"replace_race>{max_replace}")
    if contradiction_count > max_contradiction:
        reason_parts.append(f"contradiction>{max_contradiction}")
    if cat_ledger > 0:
        reason_parts.append(f"ledger={cat_ledger}")
    if cat_transition > 0:
        reason_parts.append(f"transition={cat_transition}")
    if cat_belief > 0:
        reason_parts.append(f"belief={cat_belief}")
    if cat_position > 0:
        reason_parts.append(f"position={cat_position}")
    if cat_orphan > 0:
        reason_parts.append(f"orphan={cat_orphan}")
    if intent_collision_count > 0:
        reason_parts.append(f"intent_collision={intent_collision_count}")
    if cat_cov_gap > 0 or coverage_gap_seconds > 0.0:
        reason_parts.append(f"coverage_gap={max(cat_cov_gap, 1 if coverage_gap_seconds > 0.0 else 0)}")
    if cat_stage1_fail > 0:
        reason_parts.append(f"stage1_protection_fail={cat_stage1_fail}")
    if cat_replace > 0:
        reason_parts.append(f"replace_race={cat_replace}")
    if cat_contradiction > 0:
        reason_parts.append(f"contradiction={cat_contradiction}")
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
        "position_mismatch_count": int(position_mismatch),
        "position_mismatch_count_peak": int(position_mismatch_peak),
        "orphan_count": int(orphan_count),
        "intent_collision_count": int(intent_collision_count),
        "protection_coverage_gap_seconds": float(coverage_gap_seconds),
        "protection_coverage_gap_seconds_peak": float(coverage_gap_seconds_peak),
        "stage1_protection_fail_count": int(stage1_protection_fail_count),
        "replace_race_count": int(replace_race_count),
        "evidence_contradiction_count": int(contradiction_count),
        "path": cache_path,
    }
    _CACHE["path"] = cache_path
    _CACHE["mtime"] = mtime
    _CACHE["result"] = dict(out)
    return out
