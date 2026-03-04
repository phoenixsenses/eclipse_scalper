from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import pandas as pd


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Execution quality audit for paper/live trade logs.")
    p.add_argument("--in-parquet", default="data/live/papertrades_live.parquet")
    p.add_argument("--out-md", default="reports/execution_quality_audit.md")
    p.add_argument("--out-json", default="reports/execution_quality_audit.json")
    p.add_argument("--last-n", type=int, default=10000, help="Audit only the latest N rows after ts sort.")
    return p.parse_args()


def _safe_num(s: pd.Series | Any) -> pd.Series:
    if isinstance(s, pd.Series):
        return pd.to_numeric(s, errors="coerce")
    return pd.to_numeric(pd.Series([s]), errors="coerce")


def _base_metrics(df: pd.DataFrame) -> Dict[str, float]:
    if df.empty:
        return {
            "rows": 0.0,
            "fill_rate": 0.0,
            "ttl_expired_rate": 0.0,
            "mean_fill_delay_bars": 0.0,
            "pnl_net_mean": 0.0,
            "pnl_net_sum": 0.0,
            "pnl_gross_mean": 0.0,
            "fee_notional_sum": 0.0,
            "expected_slippage_bps_mean": 0.0,
        }

    filled = _safe_num(df.get("filled", pd.Series([1] * len(df))))
    ttl = _safe_num(df.get("ttl_expired", pd.Series([0] * len(df))))
    delay = _safe_num(df.get("fill_delay_bars", pd.Series([0] * len(df))))
    net = _safe_num(df.get("pnl_net_notional", df.get("pnl_net", pd.Series([0.0] * len(df))))).fillna(0.0)
    gross = _safe_num(df.get("pnl_gross_notional", df.get("pnl_gross", pd.Series([0.0] * len(df))))).fillna(0.0)
    fee = _safe_num(df.get("fee_notional", pd.Series([0.0] * len(df)))).fillna(0.0)
    entry = _safe_num(df.get("entry_price", pd.Series([0.0] * len(df))))
    fill = _safe_num(df.get("fill_price", df.get("entry_price", pd.Series([0.0] * len(df)))))
    slip_bps = (((fill - entry).abs() / entry.replace(0.0, pd.NA)) * 10000.0).fillna(0.0)
    return {
        "rows": float(len(df)),
        "fill_rate": float(filled.fillna(0).mean()),
        "ttl_expired_rate": float(ttl.fillna(0).mean()),
        "mean_fill_delay_bars": float(delay.fillna(0).mean()),
        "pnl_net_mean": float(net.mean()),
        "pnl_net_sum": float(net.sum()),
        "pnl_gross_mean": float(gross.mean()),
        "fee_notional_sum": float(fee.sum()),
        "expected_slippage_bps_mean": float(slip_bps.mean()),
    }


def _group_metrics(df: pd.DataFrame, col: str) -> Dict[str, Dict[str, float]]:
    if df.empty or col not in df.columns:
        return {}
    out: Dict[str, Dict[str, float]] = {}
    for key, g in df.groupby(df[col].astype(str), dropna=False):
        out[str(key)] = _base_metrics(g)
    return out


def _render_table(rows: Dict[str, Dict[str, float]]) -> list[str]:
    if not rows:
        return ["_empty_"]
    lines = [
        "| group | rows | fill_rate | ttl_expired | fill_delay | pnl_net_mean | pnl_net_sum | fee_sum | slip_bps |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for k in sorted(rows.keys()):
        r = rows[k]
        lines.append(
            f"| {k} | {int(r['rows'])} | {r['fill_rate']:.4f} | {r['ttl_expired_rate']:.4f} | {r['mean_fill_delay_bars']:.3f} | "
            f"{r['pnl_net_mean']:.8f} | {r['pnl_net_sum']:.8f} | {r['fee_notional_sum']:.6f} | {r['expected_slippage_bps_mean']:.4f} |"
        )
    return lines


def main() -> int:
    args = _parse_args()
    in_path = Path(str(args.in_parquet))
    out_md = Path(str(args.out_md))
    out_json = Path(str(args.out_json))
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    if not in_path.exists() or not in_path.is_file():
        payload = {"status": "skip", "reason": "missing_input", "input": str(in_path), "timestamp_utc": _utc_now()}
        out_json.write_text(json.dumps(payload, ensure_ascii=True, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        out_md.write_text("# Execution Quality Audit\n\n- status: `skip`\n- reason: `missing_input`\n", encoding="utf-8")
        print(f"execution_quality_audit: skip missing input {in_path}")
        return 0

    df = pd.read_parquet(in_path)
    if "entry_ts_utc" in df.columns:
        ts = pd.to_datetime(df["entry_ts_utc"], utc=True, errors="coerce")
        df = df.assign(_ts=ts).sort_values("_ts").drop(columns=["_ts"])
    if int(args.last_n) > 0 and len(df) > int(args.last_n):
        df = df.tail(int(args.last_n)).reset_index(drop=True)

    overall = _base_metrics(df)
    by_side = _group_metrics(df, "side")
    by_model = _group_metrics(df, "execution_model")
    by_reason = _group_metrics(df, "risk_reason")

    payload = {
        "status": "ok",
        "timestamp_utc": _utc_now(),
        "input": str(in_path),
        "rows": int(len(df)),
        "overall": overall,
        "by_side": by_side,
        "by_execution_model": by_model,
        "by_risk_reason": by_reason,
    }
    out_json.write_text(json.dumps(payload, ensure_ascii=True, sort_keys=True, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Execution Quality Audit",
        "",
        f"- timestamp_utc: `{payload['timestamp_utc']}`",
        f"- input: `{in_path}`",
        f"- rows: `{int(len(df))}`",
        "",
        "## Overall",
        "",
        *(
            [
                f"- `{k}`: `{(int(v) if k == 'rows' else f'{float(v):.8f}')}`"
                for k, v in overall.items()
            ]
        ),
        "",
        "## By Side",
        "",
        *_render_table(by_side),
        "",
        "## By Execution Model",
        "",
        *_render_table(by_model),
        "",
        "## By Risk Reason",
        "",
        *_render_table(by_reason),
    ]
    out_md.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(f"execution_quality_audit: ok out={out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

