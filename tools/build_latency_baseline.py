from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build latency baseline markdown from latency_profile.jsonl.")
    p.add_argument("--in-jsonl", default="logs/latency_profile.jsonl")
    p.add_argument("--out-md", default="reports/LATENCY_BASELINE.md")
    p.add_argument("--last-n", type=int, default=5000)
    return p.parse_args()


def _load_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    out: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        s = line.strip()
        if not s:
            continue
        try:
            out.append(json.loads(s))
        except Exception:
            continue
    return out


def _mean(vals: List[float]) -> float:
    return float(sum(vals) / len(vals)) if vals else 0.0


def main() -> int:
    args = _parse_args()
    src = Path(str(args.in_jsonl))
    out = Path(str(args.out_md))
    out.parent.mkdir(parents=True, exist_ok=True)

    rows = _load_rows(src)
    if int(args.last_n) > 0 and len(rows) > int(args.last_n):
        rows = rows[-int(args.last_n) :]

    keys = ["e2e_ms", "feature_ms", "signal_ms", "regime_ms", "risk_ms", "submit_ms", "fill_ack_ms"]
    acc: Dict[str, List[float]] = {k: [] for k in keys}
    for r in rows:
        m = dict(r.get("metrics", {}) or {})
        for k in keys:
            v = dict(m.get(k, {}) or {}).get("mean_ms")
            try:
                if v is not None:
                    acc[k].append(float(v))
            except Exception:
                pass

    lines = [
        "# LATENCY BASELINE",
        "",
        f"- samples: `{len(rows)}`",
        f"- source: `{src}`",
        "",
        "| metric | mean_ms |",
        "|---|---:|",
    ]
    for k in keys:
        lines.append(f"| {k} | {_mean(acc[k]):.3f} |")
    if not rows:
        lines.extend(
            [
                "",
                "_No latency profile rows yet. Let paper bot run with entry loop for at least 1 week, then rerun this tool._",
            ]
        )
    out.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(f"build_latency_baseline: wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

