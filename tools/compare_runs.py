from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def _flatten(obj: Any, prefix: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if isinstance(obj, dict):
        for k in sorted(obj.keys()):
            key = f"{prefix}.{k}" if prefix else str(k)
            out.update(_flatten(obj[k], key))
        return out
    if isinstance(obj, list):
        for i, v in enumerate(obj):
            key = f"{prefix}[{i}]"
            out.update(_flatten(v, key))
        return out
    out[prefix] = obj
    return out


def _safe_float(v: Any) -> float | None:
    try:
        if isinstance(v, bool):
            return None
        return float(v)
    except Exception:
        return None


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_run(run_dir: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    cfg = run_dir / "config.json"
    met = run_dir / "metrics.json"
    if not cfg.exists():
        raise FileNotFoundError(f"missing config.json in {run_dir}")
    if not met.exists():
        raise FileNotFoundError(f"missing metrics.json in {run_dir}")
    return _read_json(cfg), _read_json(met)


def compare_runs(a_dir: Path, b_dir: Path, top_k: int = 10) -> Dict[str, Any]:
    cfg_a, met_a = _load_run(a_dir)
    cfg_b, met_b = _load_run(b_dir)
    fa = _flatten(cfg_a)
    fb = _flatten(cfg_b)
    cfg_keys = sorted(set(fa.keys()) | set(fb.keys()))
    cfg_diff: List[Dict[str, Any]] = []
    for k in cfg_keys:
        va = fa.get(k)
        vb = fb.get(k)
        if va != vb:
            cfg_diff.append({"key": k, "a": va, "b": vb})

    ma = _flatten(met_a)
    mb = _flatten(met_b)
    mkeys = sorted(set(ma.keys()) | set(mb.keys()))
    metrics_diff: List[Dict[str, Any]] = []
    for k in mkeys:
        va = ma.get(k)
        vb = mb.get(k)
        if va == vb:
            continue
        da = _safe_float(va)
        db = _safe_float(vb)
        if da is None or db is None:
            continue
        delta = db - da
        pct = None
        if abs(da) > 0:
            pct = delta / abs(da)
        metrics_diff.append(
            {
                "key": k,
                "a": da,
                "b": db,
                "delta": delta,
                "pct": pct,
                "_abs_delta": abs(delta),
            }
        )
    metrics_diff.sort(key=lambda x: (-float(x["_abs_delta"]), str(x["key"])))
    top = metrics_diff[: max(1, int(top_k))]
    for x in top:
        x.pop("_abs_delta", None)

    highlights: List[str] = []
    for key in ("pnl_net_sum", "fills_count", "spread_cost_est_sum"):
        a_val = _safe_float(ma.get(key))
        b_val = _safe_float(mb.get(key))
        if a_val is None or b_val is None:
            continue
        d = b_val - a_val
        highlights.append(f"{key}: {a_val:.12g} -> {b_val:.12g} (delta={d:+.12g})")

    return {
        "a_run_dir": str(a_dir),
        "b_run_dir": str(b_dir),
        "config_diff": cfg_diff,
        "metrics_diff": top,
        "highlights": highlights,
    }


def _render_text(report: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# Compare Runs")
    lines.append("")
    lines.append(f"A: `{report['a_run_dir']}`")
    lines.append(f"B: `{report['b_run_dir']}`")
    lines.append("")
    lines.append("## Config changes")
    lines.append("")
    lines.append("| key | a | b |")
    lines.append("|---|---|---|")
    for d in report.get("config_diff", []):
        lines.append(f"| {d['key']} | `{d.get('a')}` | `{d.get('b')}` |")
    lines.append("")
    lines.append("## Metrics changes (top)")
    lines.append("")
    lines.append("| key | a | b | delta | pct |")
    lines.append("|---|---:|---:|---:|---:|")
    for d in report.get("metrics_diff", []):
        pct = d.get("pct")
        pct_txt = "-" if pct is None else f"{100.0*float(pct):+.2f}%"
        lines.append(
            f"| {d['key']} | {float(d['a']):.12g} | {float(d['b']):.12g} | {float(d['delta']):+.12g} | {pct_txt} |"
        )
    lines.append("")
    lines.append("## Highlights")
    lines.append("")
    for h in report.get("highlights", []):
        lines.append(f"- {h}")
    lines.append("")
    return "\n".join(lines)


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Compare two eval run folders (config+metrics diff).")
    p.add_argument("--a", required=True, help="Run A directory")
    p.add_argument("--b", required=True, help="Run B directory")
    p.add_argument("--out", default="", help="Optional JSON report output")
    p.add_argument("--format", choices=("text", "json"), default="text")
    p.add_argument("--top-k", type=int, default=10)
    return p


def main() -> int:
    args = _parser().parse_args()
    try:
        report = compare_runs(Path(str(args.a)), Path(str(args.b)), top_k=int(args.top_k))
        if str(args.out).strip():
            out = Path(str(args.out))
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(report, ensure_ascii=True, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        if str(args.format) == "json":
            print(json.dumps(report, ensure_ascii=True, sort_keys=True, indent=2))
        else:
            print(_render_text(report))
        return 0
    except Exception as e:
        print(f"compare_runs error runtime={type(e).__name__}:{e}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

