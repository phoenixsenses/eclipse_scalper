from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from tools.run_summary import build_run_summary

@dataclass
class CauseScore:
    name: str
    score: float
    evidence: List[str]
    actions: List[str]


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _f(d: Dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(d.get(key, default))
    except Exception:
        return float(default)


def _i(d: Dict[str, Any], key: str, default: int = 0) -> int:
    try:
        return int(d.get(key, default))
    except Exception:
        return int(default)


def rank_root_causes(
    parity: Dict[str, Any],
    diag: Dict[str, Any],
    tox: Dict[str, Any],
    audit: Dict[str, Any],
) -> List[CauseScore]:
    out: List[CauseScore] = []

    sim_count = _i(parity, "sim_count", 0)
    matched = _i(parity, "matched_count", 0)
    match_rate = _f(parity, "match_rate_vs_sim", 0.0)
    fill_rate_delta = abs(_f(parity, "fill_rate_delta", 0.0))
    delay_delta = abs(_f(parity, "mean_fill_delay_delta_sec", 0.0))
    adverse_delta = abs(_f(parity, "mean_adverse_bps_delta", 0.0))

    p95_delay = _f(diag, "latency_fill_delay_sec_p95", 0.0)
    tox_score = _f(diag, "toxicity_score", _f(tox, "toxicity_score", 0.0))
    diag_rows = _i(diag, "rows", 0)
    audit_ok = bool(audit.get("overall_ok", False))

    # 1) Join/clock/linkage mismatch.
    score_linkage = 0.0
    ev_linkage: List[str] = []
    if sim_count >= 20 and match_rate < 0.35:
        score_linkage += (0.35 - max(0.0, match_rate)) * 3.0
        ev_linkage.append(f"Low sim/live match rate: {match_rate:.2%} with sim_count={sim_count}.")
    if matched < 10 and sim_count >= 20:
        score_linkage += 0.4
        ev_linkage.append(f"Low matched_count={matched} despite active simulation rows.")
    if ev_linkage:
        out.append(
            CauseScore(
                name="Timestamp/Linkage Mismatch",
                score=score_linkage,
                evidence=ev_linkage,
                actions=[
                    "Validate event timestamps are UTC and monotonic across sim/live paths.",
                    "Re-run parity with wider `--match-window-sec` and inspect drift by hour.",
                    "Backfill missing symbol/side normalization in live trade export.",
                ],
            )
        )

    # 2) Latency model drift.
    score_latency = 0.0
    ev_latency: List[str] = []
    if p95_delay > 10.0:
        score_latency += min(2.0, (p95_delay - 10.0) / 10.0)
        ev_latency.append(f"High p95 fill delay: {p95_delay:.2f}s (>10s target).")
    if delay_delta > 1.0:
        score_latency += min(1.5, delay_delta / 4.0)
        ev_latency.append(f"Mean fill delay delta vs replay: {delay_delta:.2f}s.")
    if ev_latency:
        out.append(
            CauseScore(
                name="Latency Modeling Drift",
                score=score_latency,
                evidence=ev_latency,
                actions=[
                    "Recalibrate feed/order latency distributions from latest paper fills.",
                    "Segment latency profiles by session (Asia/EU/US) and volatility buckets.",
                    "Enable latency v2 canary only after parity delta improves for 3 consecutive days.",
                ],
            )
        )

    # 3) Queue/adverse selection drift.
    score_queue = 0.0
    ev_queue: List[str] = []
    if adverse_delta > 1.0:
        score_queue += min(2.0, adverse_delta / 4.0)
        ev_queue.append(f"Adverse excursion delta: {adverse_delta:.3f} bps.")
    if tox_score > 1.2:
        score_queue += min(1.0, tox_score / 4.0)
        ev_queue.append(f"Elevated toxicity score: {tox_score:.3f}.")
    if fill_rate_delta > 0.15:
        score_queue += min(1.0, fill_rate_delta / 0.5)
        ev_queue.append(f"Large fill-rate gap: {fill_rate_delta:.2%}.")
    if ev_queue:
        out.append(
            CauseScore(
                name="Queue/Hazard Miscalibration",
                score=score_queue,
                evidence=ev_queue,
                actions=[
                    "Refit queue depletion/join parameters using the latest 7-day paper fill set.",
                    "Check maker placement mode vs observed queue depth buckets.",
                    "Recompute adverse model by regime + time-of-day and compare MAE.",
                ],
            )
        )

    # 4) Data quality / insufficient sample.
    score_data = 0.0
    ev_data: List[str] = []
    if diag_rows < 30:
        score_data += 0.8
        ev_data.append(f"Low diagnostic sample size: rows={diag_rows}.")
    if not audit_ok:
        score_data += 0.4
        ev_data.append("Post-rollout audit has failing checks.")
    if sim_count < 20:
        score_data += 0.6
        ev_data.append(f"Insufficient simulated fills: sim_count={sim_count}.")
    if ev_data:
        out.append(
            CauseScore(
                name="Insufficient/Noisy Evidence",
                score=score_data,
                evidence=ev_data,
                actions=[
                    "Increase sample horizon (>=24h and >=50 matched fills) before retuning.",
                    "Run daily calibration pipeline and enforce artifact completeness checks.",
                    "Block parameter updates when audit or coverage checks fail.",
                ],
            )
        )

    if not out:
        out.append(
            CauseScore(
                name="No Material Drift Detected",
                score=0.0,
                evidence=["Current parity/diagnostic metrics are within configured tolerances."],
                actions=["Keep canary running and review 7-day stability before broad rollout."],
            )
        )

    out.sort(key=lambda x: x.score, reverse=True)
    return out


def _run_pipeline(sim: str, live_db: str, live_parquet: str) -> Dict[str, Any]:
    cmd = [
        sys.executable,
        "-m",
        "tools.execution_e2e_pipeline",
        "--sim",
        sim,
        "--live-db",
        live_db,
        "--live-parquet",
        live_parquet,
    ]
    p = subprocess.run(cmd, capture_output=True, text=True)
    return {
        "cmd": cmd,
        "rc": int(p.returncode),
        "stdout_tail": str(p.stdout or "")[-4000:],
        "stderr_tail": str(p.stderr or "")[-4000:],
    }


def _render_md(payload: Dict[str, Any]) -> str:
    causes = payload.get("causes", [])
    lines: List[str] = [
        "# LIVE FILL DRIFT ROOT-CAUSE REPORT",
        "",
        f"- ts_utc: {payload.get('ts_utc', '')}",
        f"- overall_status: {payload.get('overall_status', 'unknown')}",
        "",
        "## Inputs",
        f"- parity_json: `{payload.get('parity_json', '')}`",
        f"- diagnostics_json: `{payload.get('diagnostics_json', '')}`",
        f"- toxicity_json: `{payload.get('toxicity_json', '')}`",
        f"- audit_json: `{payload.get('audit_json', '')}`",
        "",
        "## Ranked Root Causes",
    ]
    for idx, c in enumerate(causes, 1):
        lines.append(f"### {idx}. {c.get('name', 'unknown')} (score={float(c.get('score', 0.0)):.3f})")
        lines.append("Evidence:")
        for ev in list(c.get("evidence") or []):
            lines.append(f"- {ev}")
        lines.append("Actions:")
        for ac in list(c.get("actions") or []):
            lines.append(f"- {ac}")
        lines.append("")
    if isinstance(payload.get("run_summary"), dict):
        lines.extend(["## Run Summary", f"- `{payload['run_summary']}`", ""])
    return "\n".join(lines).rstrip() + "\n"


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze live fill drift and rank likely root causes.")
    p.add_argument("--run-pipeline", action="store_true", help="Run execution_e2e_pipeline before analysis.")
    p.add_argument("--sim", default="logs/micro_edge_debug_trades.jsonl")
    p.add_argument("--live-db", default="data/paper_trades.db")
    p.add_argument("--live-parquet", default="data/live/papertrades_live.parquet")
    p.add_argument("--parity-json", default="reports/REPLAY_PARITY_REPORT.json")
    p.add_argument("--diag-json", default="reports/EXECUTION_HEALTH.json")
    p.add_argument("--tox-json", default="reports/TOXICITY_REPORT.json")
    p.add_argument("--audit-json", default="reports/POST_ROLLOUT_AUDIT.json")
    p.add_argument("--out-json", default="reports/LIVE_FILL_DRIFT_ROOT_CAUSE.json")
    p.add_argument("--out-md", default="reports/LIVE_FILL_DRIFT_ROOT_CAUSE.md")
    return p.parse_args()


def main() -> int:
    args = _args()
    pipeline = {}
    if bool(args.run_pipeline):
        pipeline = _run_pipeline(str(args.sim), str(args.live_db), str(args.live_parquet))

    parity = _read_json(Path(str(args.parity_json)))
    diag = _read_json(Path(str(args.diag_json)))
    tox = _read_json(Path(str(args.tox_json)))
    audit = _read_json(Path(str(args.audit_json)))
    causes = rank_root_causes(parity=parity, diag=diag, tox=tox, audit=audit)
    cause_dicts = [
        {"name": c.name, "score": c.score, "evidence": c.evidence, "actions": c.actions}
        for c in causes
    ]
    overall = "ok" if cause_dicts and float(cause_dicts[0].get("score", 0.0)) < 0.5 else "attention"

    payload: Dict[str, Any] = {
        "ts_utc": _utc_now(),
        "overall_status": overall,
        "parity_json": str(args.parity_json),
        "diagnostics_json": str(args.diag_json),
        "toxicity_json": str(args.tox_json),
        "audit_json": str(args.audit_json),
        "causes": cause_dicts,
        "pipeline": pipeline,
    }
    payload["run_summary"] = build_run_summary(
        run_type="live_fill_drift_root_cause",
        inputs={
            "parity_json": str(args.parity_json),
            "diag_json": str(args.diag_json),
            "tox_json": str(args.tox_json),
            "audit_json": str(args.audit_json),
            "run_pipeline": bool(args.run_pipeline),
        },
        metrics={
            "overall_status": str(overall),
            "cause_count": len(cause_dicts),
            "top_score": float(cause_dicts[0].get("score", 0.0)) if cause_dicts else 0.0,
        },
        artifacts={"json": str(args.out_json), "md": str(args.out_md)},
    )

    out_json = Path(str(args.out_json))
    out_md = Path(str(args.out_md))
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    out_md.write_text(_render_md(payload), encoding="utf-8")
    print(
        "live_fill_drift_root_cause: "
        f"status={overall} top={cause_dicts[0]['name'] if cause_dicts else 'none'} "
        f"out_md={out_md} out_json={out_json}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
