from __future__ import annotations

import argparse
import json
from pathlib import Path


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Write calibration/execution activation rollup markdown.")
    p.add_argument("--live-root", default="data/live")
    p.add_argument("--out", default="reports/calibration_rollups.md")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        root = Path(str(args.live_root))
        cal = _read_jsonl(root / "calibration_history.jsonl")
        exe = _read_jsonl(root / "execution_params_history.jsonl")
        ev = _read_jsonl(Path("logs/calibration_events.jsonl"))
        active = {}
        p = root / "active_artifacts.json"
        if p.exists():
            active = json.loads(p.read_text(encoding="utf-8"))
        latest_probe = {}
        latest_directional = {}
        if cal:
            latest_probe = dict((cal[-1] or {}).get("probe_summary", {}) or {})
            latest_directional = dict((cal[-1] or {}).get("directional_summary", {}) or {})
        failed_validations = [r for r in ev if str(r.get("event", "")).startswith("activate") and (not bool(r.get("ok", True)))]
        suspect_events = [r for r in ev if str(r.get("event", "")) == "calibration_suspected_bad"]
        lines = [
            "# Calibration Rollups",
            "",
            f"- calibration_history_count: `{len(cal)}`",
            f"- execution_history_count: `{len(exe)}`",
            f"- calibration_events_count: `{len(ev)}`",
            f"- failed_validation_events: `{len(failed_validations)}`",
            f"- suspected_bad_events: `{len(suspect_events)}`",
            f"- active_calibration: `{active.get('calibration_json_path', '')}`",
            f"- active_execution: `{active.get('execution_params_json_path', '')}`",
            f"- active_probe_total_density: `{float(active.get('calibration_probe_total_density', 0.0) or 0.0):.6f}`",
            "",
            "## Latest Calibration Entries",
            "",
            "| ts_utc | path | sha256 |",
            "|---|---|---|",
        ]
        for r in cal[-5:][::-1]:
            lines.append(f"| {r.get('ts_utc','')} | {r.get('path','')} | {r.get('sha256','')} |")
        lines += [
            "",
            "## Latest Execution Entries",
            "",
            "| ts_utc | path | sha256 |",
            "|---|---|---|",
        ]
        for r in exe[-5:][::-1]:
            lines.append(f"| {r.get('ts_utc','')} | {r.get('path','')} | {r.get('sha256','')} |")
        lines += [
            "",
            "## Latest Probe Summary",
            "",
            f"- total_density: `{float(latest_probe.get('total_density', 0.0) or 0.0):.6f}`",
            f"- rows: `{int(latest_probe.get('total_rows', 0) or 0)}`",
            f"- days: `{float(latest_probe.get('days', 0.0) or 0.0):.6f}`",
            "",
            "| probe | triggers_per_day | density |",
            "|---|---:|---:|",
        ]
        for r in list(latest_probe.get("probe_stats", []) or []):
            lines.append(f"| {r.get('probe','')} | {float(r.get('triggers_per_day', 0.0)):.4f} | {float(r.get('density', 0.0)):.6f} |")
        lines += [
            "",
            "## Latest Directional Sanity",
            "",
            f"- failed_count: `{int(latest_directional.get('failed_count', 0) or 0)}`",
            "",
            "| dprobe | side | h | n | mean_signed | win_rate | failed |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
        for r in list(latest_directional.get("directional_probe_stats", []) or []):
            lines.append(
                f"| {r.get('probe','')} | {r.get('side','')} | {int(r.get('horizon_bars', 0))} | {int(r.get('n_triggers', 0))} | "
                f"{float(r.get('mean_signed_return', 0.0)):.8f} | {float(r.get('win_rate', 0.0)):.4f} | {int(bool(r.get('failed', False)))} |"
            )
        lines += [
            "",
            "## Recent Failed Validation Events",
            "",
            "| event | kind | path | errors |",
            "|---|---|---|---|",
        ]
        for r in failed_validations[-10:][::-1]:
            lines.append(
                f"| {r.get('event','')} | {r.get('kind','')} | {r.get('path','')} | {','.join(list(r.get('errors',[]) or []))} |"
            )
        out = Path(str(args.out))
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
        print(f"report_calibration_rollups ok out={out}")
        return 0
    except Exception as e:
        print(f"report_calibration_rollups error runtime={type(e).__name__}:{e}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
