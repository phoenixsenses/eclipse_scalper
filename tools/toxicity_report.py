from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd


def _load(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    if path.suffix.lower() in {".parquet", ".pq"}:
        try:
            return pd.read_parquet(path)
        except Exception:
            return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _n(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(0.0)


def build_toxicity_report(df: pd.DataFrame) -> Dict[str, Any]:
    if df.empty:
        return {"rows": 0, "sides": {}}
    side_col = df.get("side", pd.Series(["unknown"] * len(df))).astype(str).str.lower()
    adv = _n(df.get("max_adverse_bps", pd.Series([0.0] * len(df)))).clip(lower=0.0)
    pnl = (_n(df.get("pnl_bps", df.get("pnl_net", pd.Series([0.0] * len(df))))) * 10000.0).astype(float)
    out: Dict[str, Any] = {"rows": int(len(df)), "sides": {}}
    for side in sorted(set(side_col.tolist())):
        m = side_col == side
        if not bool(m.any()):
            continue
        adv_m = adv[m]
        pnl_m = pnl[m]
        tox = (adv_m / (pnl_m.abs() + 1e-9)).clip(lower=0.0)
        out["sides"][side] = {
            "rows": int(m.sum()),
            "adverse_bps_mean": float(adv_m.mean()),
            "pnl_bps_mean": float(pnl_m.mean()),
            "toxicity_score": float(tox.mean()),
        }
    return out


def _render_md(d: Dict[str, Any]) -> str:
    lines = [
        "# TOXICITY REPORT",
        "",
        f"- rows: {int(d.get('rows', 0))}",
        "",
        "| side | rows | adverse_bps_mean | pnl_bps_mean | toxicity_score |",
        "|---|---:|---:|---:|---:|",
    ]
    sides = d.get("sides", {}) if isinstance(d, dict) else {}
    for side in sorted(sides.keys()):
        r = sides[side]
        lines.append(
            f"| {side} | {int(r.get('rows',0))} | {float(r.get('adverse_bps_mean',0.0)):.4f} | "
            f"{float(r.get('pnl_bps_mean',0.0)):+.4f} | {float(r.get('toxicity_score',0.0)):.4f} |"
        )
    lines.append("")
    return "\n".join(lines)


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Toxicity decomposition report.")
    p.add_argument("--in", dest="in_path", default="data/live/papertrades_live.parquet")
    p.add_argument("--out-md", default="reports/TOXICITY_REPORT.md")
    p.add_argument("--out-json", default="reports/TOXICITY_REPORT.json")
    return p.parse_args()


def main() -> int:
    args = _args()
    df = _load(Path(str(args.in_path)))
    d = build_toxicity_report(df)
    out_md = Path(str(args.out_md))
    out_json = Path(str(args.out_json))
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(_render_md(d), encoding="utf-8")
    out_json.write_text(json.dumps(d, ensure_ascii=True, indent=2), encoding="utf-8")
    print(f"toxicity_report: rows={int(d.get('rows',0))} out_md={out_md} out_json={out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

