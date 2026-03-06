from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, sort_keys=True, indent=2) + "\n", encoding="utf-8")


@dataclass
class RunLog:
    run_dir: Path
    params: Dict[str, Any]
    manifest_path: Path = field(init=False)
    pointers_path: Path = field(init=False)
    params_path: Path = field(init=False)
    logs_path: Path = field(init=False)

    def __post_init__(self) -> None:
        self.manifest_path = self.run_dir / "manifest.json"
        self.pointers_path = self.run_dir / "pointers.json"
        self.params_path = self.run_dir / "params.json"
        self.logs_path = self.run_dir / "logs.jsonl"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        if not self.params_path.exists():
            _write_json(self.params_path, dict(self.params))
        if not self.pointers_path.exists():
            _write_json(
                self.pointers_path,
                {
                    "execution_params_json": "",
                    "execution_realism_report_md": "",
                },
            )
        if not self.manifest_path.exists():
            _write_json(
                self.manifest_path,
                {
                    "created_utc": _utc_now_iso(),
                    "status": "running",
                    "steps": {},
                },
            )

    def read_manifest(self) -> Dict[str, Any]:
        if not self.manifest_path.exists():
            return {"status": "running", "steps": {}}
        return json.loads(self.manifest_path.read_text(encoding="utf-8"))

    def set_step(self, step: str, status: str, *, error: str = "", detail: Dict[str, Any] | None = None) -> None:
        m = self.read_manifest()
        steps = dict(m.get("steps", {}))
        cur = dict(steps.get(step, {}))
        cur["status"] = str(status)
        if str(status) == "running":
            cur["started_utc"] = _utc_now_iso()
            cur.pop("finished_utc", None)
        else:
            cur["finished_utc"] = _utc_now_iso()
        if error:
            cur["error"] = str(error)
        elif "error" in cur:
            cur.pop("error", None)
        if detail:
            cur.update(detail)
        steps[step] = cur
        m["steps"] = steps
        m["updated_utc"] = _utc_now_iso()
        _write_json(self.manifest_path, m)

    def set_status(self, status: str) -> None:
        m = self.read_manifest()
        m["status"] = str(status)
        m["updated_utc"] = _utc_now_iso()
        _write_json(self.manifest_path, m)

    def log(self, event: str, **fields: Any) -> None:
        row = {"ts_utc": _utc_now_iso(), "event": str(event)}
        row.update(fields)
        self.logs_path.parent.mkdir(parents=True, exist_ok=True)
        with self.logs_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")

    def update_pointers(self, **pointers: str) -> None:
        cur = {}
        if self.pointers_path.exists():
            cur = json.loads(self.pointers_path.read_text(encoding="utf-8"))
        cur.update({k: v for k, v in pointers.items() if v})
        _write_json(self.pointers_path, cur)
