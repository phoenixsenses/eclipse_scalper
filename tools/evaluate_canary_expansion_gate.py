from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

from tools.run_summary import build_run_summary

@dataclass
class DailyScore:
    date_key: str
    top_name: str
    top_score: float
    path: str


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _parse_daily_scores(report_dir: Path) -> List[DailyScore]:
    rows: List[DailyScore] = []
    for p in sorted(report_dir.glob("*_LIVE_FILL_DRIFT_ROOT_CAUSE.json")):
        d = _read_json(p)
        causes = list(d.get("causes") or [])
        if not causes:
            continue
        top = causes[0] if isinstance(causes[0], dict) else {}
        score = float(top.get("score", 9.9) or 9.9)
        name = str(top.get("name", "unknown"))
        date_key = p.name.split("_LIVE_FILL_DRIFT_ROOT_CAUSE.json")[0]
        rows.append(DailyScore(date_key=date_key, top_name=name, top_score=score, path=str(p)))
    return rows


def evaluate_gate(rows: List[DailyScore], *, window_days: int, max_top_score: float) -> Tuple[bool, Dict[str, Any]]:
    used = list(rows)[-int(max(1, window_days)) :]
    coverage_ok = len(used) >= int(window_days)
    score_ok = all(float(r.top_score) < float(max_top_score) for r in used)
    passed = bool(coverage_ok and score_ok)
    detail = {
        "window_days": int(window_days),
        "required_max_top_score": float(max_top_score),
        "coverage_ok": bool(coverage_ok),
        "score_ok": bool(score_ok),
        "days_observed": int(len(used)),
        "days": [
            {
                "date": r.date_key,
                "top_name": r.top_name,
                "top_score": float(r.top_score),
                "path": r.path,
            }
            for r in used
        ],
    }
    return passed, detail


def _render_md(payload: Dict[str, Any]) -> str:
    d = payload.get("gate", {})
    lines = [
        "# Canary Expansion Gate",
        "",
        f"- ts_utc: {payload.get('ts_utc', '')}",
        f"- verdict: {'GO' if bool(payload.get('passed', False)) else 'HOLD'}",
        f"- window_days: {int(d.get('window_days', 0))}",
        f"- required_max_top_score: {float(d.get('required_max_top_score', 0.0)):.3f}",
        f"- days_observed: {int(d.get('days_observed', 0))}",
        f"- coverage_ok: {int(bool(d.get('coverage_ok', False)))}",
        f"- score_ok: {int(bool(d.get('score_ok', False)))}",
        "",
        "## Daily Top Causes",
        "| date | top_cause | top_score |",
        "|---|---|---:|",
    ]
    for r in list(d.get("days") or []):
        lines.append(f"| {r.get('date','')} | {r.get('top_name','')} | {float(r.get('top_score',0.0)):.3f} |")
    lines.append("")
    lines.append("## Policy")
    lines.append("- GO only if all observed days in window have top_score below threshold and coverage is full.")
    lines.append("- HOLD otherwise; keep canary only and continue daily calibration.")
    if isinstance(payload.get("run_summary"), dict):
        lines.extend(["", "## Run Summary", f"- `{payload['run_summary']}`"])
    lines.append("")
    return "\n".join(lines)


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate canary expansion gate from daily root-cause reports.")
    p.add_argument("--report-dir", default="reports/daily")
    p.add_argument("--window-days", type=int, default=7)
    p.add_argument("--max-top-score", type=float, default=0.5)
    p.add_argument("--out-json", default="reports/CANARY_EXPANSION_GATE.json")
    p.add_argument("--out-md", default="reports/CANARY_EXPANSION_GATE.md")
    return p.parse_args()


def main() -> int:
    args = _args()
    rows = _parse_daily_scores(Path(str(args.report_dir)))
    passed, gate = evaluate_gate(rows, window_days=int(args.window_days), max_top_score=float(args.max_top_score))
    payload = {
        "ts_utc": _utc_now(),
        "passed": bool(passed),
        "gate": gate,
    }
    out_json = Path(str(args.out_json))
    out_md = Path(str(args.out_md))
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    payload["run_summary"] = build_run_summary(
        run_type="evaluate_canary_expansion_gate",
        inputs={"report_dir": str(args.report_dir), "window_days": int(args.window_days), "max_top_score": float(args.max_top_score)},
        metrics={"passed": bool(passed), "days_observed": int(gate.get("days_observed", 0))},
        artifacts={"json": str(out_json), "md": str(out_md)},
    )
    out_json.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    out_md.write_text(_render_md(payload), encoding="utf-8")
    print(
        "evaluate_canary_expansion_gate: "
        f"verdict={'GO' if passed else 'HOLD'} days={int(gate.get('days_observed', 0))} "
        f"out_md={out_md} out_json={out_json}"
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
