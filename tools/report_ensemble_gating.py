from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Report gating decisions for regime experts.")
    p.add_argument("--gating", required=True)
    p.add_argument("--out", default="reports/ensemble_gating.md")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        gating = pd.read_parquet(Path(str(args.gating)))
        fallback_rate = float(pd.to_numeric(gating.get("fallback_used"), errors="coerce").fillna(0.0).mean()) if not gating.empty else 0.0
        mean_conf = float(pd.to_numeric(gating.get("confidence_score"), errors="coerce").fillna(0.0).mean()) if not gating.empty else 0.0
        reason_counts = gating.get("reason", pd.Series([], dtype=str)).astype(str).value_counts().to_dict() if not gating.empty else {}
        lines = [
            "# Ensemble Gating Report",
            "",
            f"- rows: `{len(gating)}`",
            f"- fallback_rate: `{fallback_rate:.4f}`",
            f"- mean_confidence: `{mean_conf:.4f}`",
            "",
            "## Reason Counts",
            "",
            "| reason | count |",
            "|---|---:|",
        ]
        for k in sorted(reason_counts):
            lines.append(f"| {k} | {int(reason_counts[k])} |")
        out = Path(str(args.out))
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
        print(f"report_ensemble_gating ok rows={len(gating)} out={out}")
        return 0
    except Exception as e:
        print(f"report_ensemble_gating error runtime={type(e).__name__}:{e}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

