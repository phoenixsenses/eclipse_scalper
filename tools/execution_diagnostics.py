from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd


def _load_rows(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    if path.suffix.lower() in {".parquet", ".pq"}:
        try:
            return pd.read_parquet(path)
        except Exception:
            return pd.DataFrame()
    if path.suffix.lower() in {".csv", ".txt"}:
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def _safe_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(0.0)


def compute_execution_diagnostics(df: pd.DataFrame) -> Dict[str, Any]:
    if df is None or df.empty:
        return {
            "rows": 0,
            "fill_rate": 0.0,
            "queue_competition_score": 0.0,
            "toxicity_score": 0.0,
            "adverse_selection_bps_mean": 0.0,
            "latency_fill_delay_sec_p50": 0.0,
            "latency_fill_delay_sec_p95": 0.0,
            "latency_impact_vs_net_corr": 0.0,
        }
    out: Dict[str, Any] = {"rows": int(len(df))}
    filled = _safe_num(df.get("filled", pd.Series([1] * len(df)))).clip(lower=0.0, upper=1.0)
    out["fill_rate"] = float(filled.mean())

    fill_delay_bars = _safe_num(df.get("fill_delay_bars", pd.Series([0.0] * len(df))))
    out["queue_competition_score"] = float((fill_delay_bars.clip(lower=0.0).mean()) / (1.0 + fill_delay_bars.clip(lower=0.0).mean()))

    pnl_bps = _safe_num(df.get("pnl_bps", df.get("pnl_net", pd.Series([0.0] * len(df)))) * 10000.0)
    adverse = _safe_num(df.get("max_adverse_bps", pd.Series([0.0] * len(df)))).clip(lower=0.0)
    out["adverse_selection_bps_mean"] = float(adverse.mean())

    # Toxicity proxy: adverse pressure relative to absolute pnl.
    denom = (pnl_bps.abs() + 1e-9)
    tox = (adverse / denom).clip(lower=0.0)
    out["toxicity_score"] = float(tox.mean())

    fill_delay_sec = _safe_num(df.get("fill_delay_sec", pd.Series([0.0] * len(df))))
    if (fill_delay_sec <= 0.0).all():
        # fallback from bars with bucket estimate of 1s
        fill_delay_sec = fill_delay_bars.clip(lower=0.0)
    out["latency_fill_delay_sec_p50"] = float(fill_delay_sec.quantile(0.50))
    out["latency_fill_delay_sec_p95"] = float(fill_delay_sec.quantile(0.95))
    if len(df) >= 2:
        corr = float(fill_delay_sec.corr(pnl_bps))
        if corr != corr:
            corr = 0.0
    else:
        corr = 0.0
    out["latency_impact_vs_net_corr"] = corr
    return out


def _render_md(d: Dict[str, Any]) -> str:
    lines = [
        "# EXECUTION DIAGNOSTICS",
        "",
        f"- rows: {int(d.get('rows', 0))}",
        f"- fill_rate: {float(d.get('fill_rate', 0.0)):.2%}",
        f"- queue_competition_score: {float(d.get('queue_competition_score', 0.0)):.4f}",
        f"- toxicity_score: {float(d.get('toxicity_score', 0.0)):.4f}",
        f"- adverse_selection_bps_mean: {float(d.get('adverse_selection_bps_mean', 0.0)):.4f}",
        f"- latency_fill_delay_sec_p50: {float(d.get('latency_fill_delay_sec_p50', 0.0)):.3f}",
        f"- latency_fill_delay_sec_p95: {float(d.get('latency_fill_delay_sec_p95', 0.0)):.3f}",
        f"- latency_impact_vs_net_corr: {float(d.get('latency_impact_vs_net_corr', 0.0)):+.4f}",
        "",
    ]
    return "\n".join(lines)


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Execution diagnostics report.")
    p.add_argument("--in", dest="in_path", default="data/live/papertrades_live.parquet")
    p.add_argument("--out-md", default="reports/EXECUTION_HEALTH.md")
    p.add_argument("--out-json", default="reports/EXECUTION_HEALTH.json")
    return p.parse_args()


def main() -> int:
    args = _args()
    inp = Path(str(args.in_path))
    df = _load_rows(inp)
    d = compute_execution_diagnostics(df)
    out_md = Path(str(args.out_md))
    out_json = Path(str(args.out_json))
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(_render_md(d), encoding="utf-8")
    out_json.write_text(json.dumps(d, ensure_ascii=True, indent=2), encoding="utf-8")
    print(f"execution_diagnostics: rows={int(d.get('rows',0))} out_md={out_md} out_json={out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

