from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Literal


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=True, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def _append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=True, sort_keys=True) + "\n")


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def get_active_artifacts(live_root: Path | str = "data/live") -> Dict[str, Any]:
    root = Path(str(live_root))
    p = root / "active_artifacts.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def activate_artifacts(
    *,
    live_root: Path | str = "data/live",
    calibration_path: str = "",
    execution_path: str = "",
    metadata: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    root = Path(str(live_root))
    active_path = root / "active_artifacts.json"
    prev = get_active_artifacts(root)
    rec = dict(prev)
    rec["updated_ts"] = _utc_now()
    if not rec.get("activated_ts"):
        rec["activated_ts"] = rec["updated_ts"]
    if metadata:
        rec.update(dict(metadata))

    cal = Path(str(calibration_path)) if str(calibration_path).strip() else None
    exe = Path(str(execution_path)) if str(execution_path).strip() else None
    if cal is not None:
        rec["calibration_json_path"] = str(cal)
        rec.setdefault("hashes", {})
        rec["hashes"]["calibration_sha256"] = _file_sha256(cal) if cal.exists() else ""
        _atomic_write_json(
            root / "calibration_active.json",
            {
                "path": rec.get("calibration_json_path", ""),
                "sha256": rec.get("hashes", {}).get("calibration_sha256", ""),
                "run_id": rec.get("run_id", ""),
                "activated_ts": rec.get("updated_ts", ""),
                "validation_report_path": rec.get("validation_report_path", ""),
                "validation_ts": rec.get("validation_ts", ""),
            },
        )
    if exe is not None:
        rec["execution_params_json_path"] = str(exe)
        rec.setdefault("hashes", {})
        rec["hashes"]["execution_sha256"] = _file_sha256(exe) if exe.exists() else ""
        _atomic_write_json(
            root / "execution_params_active.json",
            {
                "path": rec.get("execution_params_json_path", ""),
                "sha256": rec.get("hashes", {}).get("execution_sha256", ""),
                "run_id": rec.get("run_id", ""),
                "activated_ts": rec.get("updated_ts", ""),
                "validation_report_path": rec.get("validation_report_path", ""),
                "validation_ts": rec.get("validation_ts", ""),
            },
        )

    _atomic_write_json(active_path, rec)

    if cal is not None:
        _append_jsonl(
            root / "calibration_history.jsonl",
            {
                "ts_utc": rec["updated_ts"],
                "path": rec.get("calibration_json_path", ""),
                "sha256": rec.get("hashes", {}).get("calibration_sha256", ""),
                "run_id": rec.get("run_id", ""),
                "validation_passed": bool(rec.get("validation_passed", False)),
                "validation_reasons": list(rec.get("validation_reasons", []) or []),
                "probe_summary": dict(rec.get("calibration_probe_summary", {}) or {}),
                "probe_errors": list(rec.get("calibration_probe_errors", []) or []),
                "directional_sanity_enabled": bool(rec.get("directional_sanity_enabled", False)),
                "directional_summary": dict(rec.get("directional_probe_summary", {}) or {}),
                "directional_failed_count": int(rec.get("directional_failed_count", 0) or 0),
                "directional_errors": list(rec.get("directional_probe_errors", []) or []),
            },
        )
    if exe is not None:
        _append_jsonl(
            root / "execution_params_history.jsonl",
            {
                "ts_utc": rec["updated_ts"],
                "path": rec.get("execution_params_json_path", ""),
                "sha256": rec.get("hashes", {}).get("execution_sha256", ""),
                "run_id": rec.get("run_id", ""),
                "validation_passed": bool(rec.get("validation_passed", False)),
                "validation_reasons": list(rec.get("validation_reasons", []) or []),
            },
        )
    return rec


def rollback_to_previous(
    kind: Literal["calibration", "execution"],
    *,
    live_root: Path | str = "data/live",
) -> Dict[str, Any]:
    root = Path(str(live_root))
    hist_path = root / ("calibration_history.jsonl" if kind == "calibration" else "execution_params_history.jsonl")
    if not hist_path.exists():
        raise RuntimeError(f"rollback_missing_history:{kind}")
    rows = [json.loads(x) for x in hist_path.read_text(encoding="utf-8").splitlines() if x.strip()]
    if len(rows) < 2:
        raise RuntimeError(f"rollback_not_enough_history:{kind}")
    prev = rows[-2]
    if kind == "calibration":
        return activate_artifacts(live_root=root, calibration_path=str(prev.get("path", "")), metadata={"rollback_of": "calibration"})
    return activate_artifacts(live_root=root, execution_path=str(prev.get("path", "")), metadata={"rollback_of": "execution"})
