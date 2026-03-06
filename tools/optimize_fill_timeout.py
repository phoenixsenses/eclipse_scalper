from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from tools.run_summary import build_run_summary

def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        x = float(v)
        if x != x:
            return float(default)
        return x
    except Exception:
        return float(default)


def _safe_int(v: Any, default: int = 0) -> int:
    try:
        return int(round(float(v)))
    except Exception:
        return int(default)


def _parse_candidates(raw: str) -> list[int]:
    out: list[int] = []
    for part in str(raw or "").split(","):
        s = part.strip()
        if not s:
            continue
        try:
            x = int(round(float(s)))
            if x > 0:
                out.append(x)
        except Exception:
            continue
    uniq = sorted(set(out))
    return uniq or [5, 10, 15, 20, 30]


def _read_env_value(path: Path, key: str) -> int | None:
    if not path.exists():
        return None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        s = str(line).strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        k, v = s.split("=", 1)
        if str(k).strip() == key:
            try:
                return int(round(float(str(v).strip())))
            except Exception:
                return None
    return None


def _upsert_env_key(path: Path, key: str, value: int) -> bool:
    value_s = str(int(value))
    lines: list[str] = []
    updated = False
    if path.exists():
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    out: list[str] = []
    for ln in lines:
        s = str(ln).strip()
        if s and not s.startswith("#") and "=" in ln:
            k, _ = ln.split("=", 1)
            if str(k).strip() == key:
                out.append(f"{key}={value_s}")
                updated = True
                continue
        out.append(ln)
    if not updated:
        if out and out[-1].strip():
            out.append("")
        out.append(f"{key}={value_s}")
    text = "\n".join(out).rstrip() + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(str(tmp), str(path))
    return updated


@dataclass(frozen=True)
class TimeoutRecommendation:
    recommended: int
    raw_recommended: float
    source: str
    reason: str


def build_recommendation(payload: dict[str, Any], *, candidates: Iterable[int]) -> TimeoutRecommendation:
    cands = sorted({int(x) for x in candidates if int(x) > 0})
    if not cands:
        cands = [5, 10, 15, 20, 30]

    live = dict(payload.get("live_summary") or {})
    raw = _safe_float(live.get("recommended_timeout_sec"), 0.0)
    timeout_eval = list(live.get("timeout_eval") or [])
    if timeout_eval:
        best = sorted(
            timeout_eval,
            key=lambda r: (
                _safe_float(r.get("filled_pnl_mean"), 0.0),
                -_safe_float(r.get("filled_adverse_proxy_mean"), 0.0),
                _safe_float(r.get("fill_rate_within_timeout"), 0.0),
                _safe_float(r.get("eligible_frac"), 0.0),
            ),
            reverse=True,
        )[0]
        raw = _safe_float(best.get("timeout_sec"), raw if raw > 0 else 10.0)
        source = "timeout_eval"
        reason = (
            f"best filled_pnl_mean={_safe_float(best.get('filled_pnl_mean'), 0.0):+.8f} "
            f"adverse={_safe_float(best.get('filled_adverse_proxy_mean'), 0.0):.8f}"
        )
    else:
        if raw <= 0:
            raw = 10.0
        source = "live_summary"
        reason = "timeout_eval missing; fallback to live_summary.recommended_timeout_sec"

    rec = min(cands, key=lambda x: (abs(float(x) - raw), x))
    return TimeoutRecommendation(
        recommended=int(rec),
        raw_recommended=float(raw),
        source=str(source),
        reason=str(reason),
    )


def _render_md(
    *,
    analysis_path: Path,
    env_path: Path,
    current: int | None,
    rec: TimeoutRecommendation,
    applied: bool,
) -> str:
    lines = [
        "# Fill Timeout Recommendation",
        "",
        f"- analysis_json: `{analysis_path}`",
        f"- env_file: `{env_path}`",
        f"- current_ENTRY_WATCH_MAX_AGE_SEC: `{current if current is not None else 'missing'}`",
        f"- recommended_ENTRY_WATCH_MAX_AGE_SEC: `{rec.recommended}`",
        f"- raw_recommended_sec: `{rec.raw_recommended:.3f}`",
        f"- source: `{rec.source}`",
        f"- reason: `{rec.reason}`",
        f"- applied: `{int(bool(applied))}`",
        "",
        "## Apply",
        "",
        "```bash",
        f"python -m tools.optimize_fill_timeout --analysis-json {analysis_path} --env-file {env_path} --apply",
        "```",
        "",
    ]
    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Recommend and optionally apply ENTRY_WATCH_MAX_AGE_SEC from fill timing analysis.")
    p.add_argument("--analysis-json", default="reports/FILL_TIMING_ANALYSIS.json")
    p.add_argument("--env-file", default=".env.paper")
    p.add_argument("--out-md", default="reports/FILL_TIMEOUT_RECOMMENDATION.md")
    p.add_argument("--out-json", default="reports/FILL_TIMEOUT_RECOMMENDATION.json")
    p.add_argument("--candidates", default="5,10,15,20,30")
    p.add_argument("--apply", action="store_true", default=False)
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    analysis_path = Path(str(args.analysis_json))
    env_path = Path(str(args.env_file))
    out_md = Path(str(args.out_md))
    out_json = Path(str(args.out_json))
    if not analysis_path.exists():
        print(f"optimize_fill_timeout: analysis file missing: {analysis_path}")
        return 2
    payload = json.loads(analysis_path.read_text(encoding="utf-8", errors="replace"))
    rec = build_recommendation(payload, candidates=_parse_candidates(str(args.candidates)))
    current = _read_env_value(env_path, "ENTRY_WATCH_MAX_AGE_SEC")
    applied = False
    if bool(args.apply):
        _upsert_env_key(env_path, "ENTRY_WATCH_MAX_AGE_SEC", int(rec.recommended))
        applied = True

    report = {
        "analysis_json": str(analysis_path),
        "env_file": str(env_path),
        "current": current,
        "recommended": int(rec.recommended),
        "raw_recommended": float(rec.raw_recommended),
        "source": str(rec.source),
        "reason": str(rec.reason),
        "applied": bool(applied),
    }
    report["run_summary"] = build_run_summary(
        run_type="optimize_fill_timeout",
        inputs={"analysis_json": str(analysis_path), "env_file": str(env_path), "apply": bool(args.apply)},
        metrics={"recommended": int(rec.recommended), "current": int(current) if current is not None else None, "applied": bool(applied)},
        artifacts={"json": str(out_json), "md": str(out_md)},
    )
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, ensure_ascii=True, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    out_md.write_text(
        _render_md(analysis_path=analysis_path, env_path=env_path, current=current, rec=rec, applied=applied) + "\n",
        encoding="utf-8",
    )
    print(f"optimize_fill_timeout: wrote {out_md}")
    print(f"optimize_fill_timeout: wrote {out_json}")
    if applied:
        print(f"optimize_fill_timeout: applied ENTRY_WATCH_MAX_AGE_SEC={int(rec.recommended)} to {env_path}")
    else:
        print(f"optimize_fill_timeout: dry-run recommendation ENTRY_WATCH_MAX_AGE_SEC={int(rec.recommended)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
