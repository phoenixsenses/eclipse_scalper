from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

from tools import report_schema_validator as rsv
from tools.run_summary import build_run_summary


def _expand_inputs(inputs: List[str]) -> List[Path]:
    out: List[Path] = []
    for item in inputs:
        raw = str(item or "").strip()
        if not raw:
            continue
        path = Path(raw)
        if any(ch in raw for ch in "*?[]"):
            out.extend(sorted(Path(".").glob(raw)))
            continue
        out.append(path)
    deduped: List[Path] = []
    seen = set()
    for path in out:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(path)
    return deduped


def check_reports(paths: List[Path]) -> Tuple[List[dict], dict]:
    results: List[dict] = []
    ok_count = 0
    fail_count = 0
    for path in paths:
        if not path.exists():
            results.append({"path": str(path), "ok": False, "schema": None, "errors": ["missing:file"]})
            fail_count += 1
            continue
        payload = rsv._load_payload(path)
        schema = ""
        if isinstance(payload, list) and payload and isinstance(payload[0], dict):
            schema = rsv.infer_schema_name(payload[0]) or ""
        elif isinstance(payload, dict):
            schema = rsv.infer_schema_name(payload) or ""
        if not schema:
            results.append({"path": str(path), "ok": False, "schema": None, "errors": ["unknown_schema:auto"]})
            fail_count += 1
            continue
        errors = rsv.validate_payload(payload, schema)
        ok = len(errors) == 0
        results.append({"path": str(path), "ok": ok, "schema": schema, "errors": errors})
        if ok:
            ok_count += 1
        else:
            fail_count += 1
    summary = {
        "checked": len(results),
        "ok_count": ok_count,
        "fail_count": fail_count,
    }
    return results, summary


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate multiple report JSON/JSONL files against known schemas.")
    p.add_argument("--inputs", nargs="+", required=True, help="One or more files or glob patterns.")
    p.add_argument("--out-json", default="")
    return p.parse_args()


def main() -> int:
    args = _args()
    paths = _expand_inputs(list(args.inputs))
    results, summary = check_reports(paths)
    print(f"report_check checked={summary['checked']} ok={summary['ok_count']} fail={summary['fail_count']}")
    for item in results:
        status = "PASS" if item["ok"] else "FAIL"
        print(f"- {status} path={item['path']} schema={item['schema'] or 'unknown'}")
        for err in item["errors"]:
            print(f"  {err}")
    if str(args.out_json).strip():
        out = Path(str(args.out_json))
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "results": results,
            "summary": summary,
        }
        payload["run_summary"] = build_run_summary(
            run_type="report_check",
            inputs={"inputs": [str(p) for p in paths]},
            metrics=summary,
            artifacts={"json": str(out)},
        )
        out.write_text(__import__("json").dumps(payload, indent=2), encoding="utf-8")
    return 0 if summary["fail_count"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
