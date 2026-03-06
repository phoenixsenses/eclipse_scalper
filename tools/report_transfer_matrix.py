from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build consolidated transfer matrix report.")
    p.add_argument("--matrix-manifest", required=True)
    p.add_argument("--out-md", default="reports/transfer/transfer_matrix.md")
    p.add_argument("--out-parquet", default="data/derived/transfer_matrix/transfer_matrix.parquet")
    return p.parse_args()


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_read_parquet(path: Path) -> pd.DataFrame:
    if not str(path).strip() or str(path) == ".":
        return pd.DataFrame()
    if not path.exists() or not path.is_file():
        return pd.DataFrame()
    return pd.read_parquet(path)


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        f = float(x)
        if pd.isna(f):
            return float(default)
        return f
    except Exception:
        return float(default)


def _summarize_pair(pair: Dict[str, Any]) -> Dict[str, Any]:
    eval_df = _safe_read_parquet(Path(str(pair.get("eval_transfer_parquet", "")).strip()))
    if eval_df.empty:
        return {
            "median_delta_sharpe": 0.0,
            "positive_net_frac": 0.0,
            "trigger_rate_ratio": 0.0,
            "fill_rate_target_mean": 0.0,
            "eval_rows": 0,
            "top_survivors": [],
        }
    if "delta_sharpe" not in eval_df.columns:
        eval_df["delta_sharpe"] = pd.to_numeric(eval_df.get("target_test_sharpe"), errors="coerce") - pd.to_numeric(
            eval_df.get("source_test_sharpe"), errors="coerce"
        )
    if "delta_net_mean" not in eval_df.columns:
        eval_df["delta_net_mean"] = pd.to_numeric(eval_df.get("target_test_net_mean"), errors="coerce") - pd.to_numeric(
            eval_df.get("source_test_net_mean"), errors="coerce"
        )
    src_trades = pd.to_numeric(eval_df.get("source_test_trade_count"), errors="coerce").replace(0.0, pd.NA)
    tgt_trades = pd.to_numeric(eval_df.get("target_test_trade_count"), errors="coerce")
    ratio = (tgt_trades / src_trades).astype(float).replace([float("inf"), float("-inf")], pd.NA).dropna()
    survivors = eval_df[pd.to_numeric(eval_df.get("target_test_net_mean"), errors="coerce") > 0].copy()
    top = survivors.sort_values(["target_test_net_mean", "signal"], ascending=[False, True]).head(10)
    return {
        "median_delta_sharpe": _safe_float(pd.to_numeric(eval_df["delta_sharpe"], errors="coerce").median(), 0.0),
        "positive_net_frac": _safe_float((pd.to_numeric(eval_df.get("target_test_net_mean"), errors="coerce") > 0).mean(), 0.0),
        "trigger_rate_ratio": _safe_float(ratio.median() if not ratio.empty else 0.0, 0.0),
        "fill_rate_target_mean": _safe_float(pd.to_numeric(eval_df.get("target_fill_rate"), errors="coerce").mean(), 0.0),
        "eval_rows": int(len(eval_df)),
        "top_survivors": top.get("signal", pd.Series([], dtype=str)).astype(str).tolist(),
    }


def main() -> int:
    args = _parse_args()
    try:
        manifest = _read_json(Path(str(args.matrix_manifest)))
        pairs = list(manifest.get("pairs", []) or [])
        rows: List[Dict[str, Any]] = []
        failure_rows: List[Dict[str, Any]] = []
        cal_rows: List[Dict[str, str]] = []
        for pair in pairs:
            summary = _summarize_pair(pair)
            transfer_manifest = _read_json(Path(str(pair.get("transfer_manifest_json", "")).strip())) if Path(str(pair.get("transfer_manifest_json", "")).strip()).exists() else {}
            row = {
                "pair_id": str(pair.get("pair_id", "")),
                "source_symbol": str(pair.get("source_symbol", "")),
                "target_symbol": str(pair.get("target_symbol", "")),
                "calibration_mode": str(pair.get("calibration_mode", "")),
                "ok": bool(pair.get("ok", False)),
                "export_exit_code": int(pair.get("export_exit_code", 0) or 0),
                "eval_exit_code": int(pair.get("eval_exit_code", 0) or 0),
                "median_delta_sharpe": float(summary["median_delta_sharpe"]),
                "positive_net_frac": float(summary["positive_net_frac"]),
                "trigger_rate_ratio": float(summary["trigger_rate_ratio"]),
                "fill_rate_target_mean": float(summary["fill_rate_target_mean"]),
                "eval_rows": int(summary["eval_rows"]),
                "top_survivors_csv": ",".join(summary["top_survivors"]),
                "report_md": str(pair.get("report_md", "")),
                "eval_transfer_parquet": str(pair.get("eval_transfer_parquet", "")),
                "calibration_path_used": str(transfer_manifest.get("calibration_path_used", "")),
            }
            rows.append(row)
            cal_rows.append(
                {
                    "pair_id": row["pair_id"],
                    "calibration_mode": row["calibration_mode"],
                    "calibration_path_used": row["calibration_path_used"] or "N/A",
                }
            )
            if not row["ok"]:
                failure_rows.append(
                    {
                        "pair_id": row["pair_id"],
                        "export_exit_code": row["export_exit_code"],
                        "eval_exit_code": row["eval_exit_code"],
                        "eval_transfer_parquet": row["eval_transfer_parquet"] or "N/A",
                    }
                )
        df = pd.DataFrame(rows).sort_values(["source_symbol", "target_symbol", "calibration_mode"]).reset_index(drop=True) if rows else pd.DataFrame()
        out_pq = Path(str(args.out_parquet))
        out_pq.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_pq, index=False)

        symbols = sorted(set(df.get("source_symbol", pd.Series([], dtype=str)).tolist()) | set(df.get("target_symbol", pd.Series([], dtype=str)).tolist()))
        lines = [
            "# Transfer Matrix Report",
            "",
            f"- source_manifest: `{args.matrix_manifest}`",
            f"- directions: `{len(df)}`",
            "",
            "## Matrix",
            "",
            "| source \\ target | " + " | ".join(symbols) + " |",
            "|---|" + "|".join(["---"] * max(1, len(symbols))) + "|",
        ]
        for src in symbols:
            cells = []
            for tgt in symbols:
                if src == tgt:
                    cells.append("-")
                    continue
                sub = df[(df["source_symbol"] == src) & (df["target_symbol"] == tgt)]
                if sub.empty:
                    cells.append("N/A")
                    continue
                r = sub.iloc[0]
                cells.append(
                    f"dS={float(r['median_delta_sharpe']):.4f}<br/>pos={float(r['positive_net_frac']):.2f}<br/>trr={float(r['trigger_rate_ratio']):.2f}"
                )
            lines.append(f"| {src} | " + " | ".join(cells) + " |")

        lines += [
            "",
            "## Top Survivors By Direction",
            "",
            "| pair | top_survivors |",
            "|---|---|",
        ]
        for _, r in df.iterrows():
            lines.append(f"| {r['pair_id']} | {str(r['top_survivors_csv']) or 'N/A'} |")

        trigger_collapse = int((pd.to_numeric(df.get("trigger_rate_ratio"), errors="coerce") < 0.25).sum()) if not df.empty else 0
        trigger_explosion = int((pd.to_numeric(df.get("trigger_rate_ratio"), errors="coerce") > 4.0).sum()) if not df.empty else 0
        fill_collapse = int((pd.to_numeric(df.get("fill_rate_target_mean"), errors="coerce") < 0.20).sum()) if not df.empty else 0
        lines += [
            "",
            "## Failure Modes",
            "",
            f"- trigger_collapse_count: `{trigger_collapse}`",
            f"- trigger_explosion_count: `{trigger_explosion}`",
            f"- fill_rate_collapse_count: `{fill_collapse}`",
            "- directional_sanity_fail_counts: `N/A`",
        ]

        lines += [
            "",
            "## Calibration Artifacts Used",
            "",
            "| pair | mode | calibration_path |",
            "|---|---|---|",
        ]
        for row in sorted(cal_rows, key=lambda x: (x["pair_id"], x["calibration_mode"])):
            lines.append(f"| {row['pair_id']} | {row['calibration_mode']} | {row['calibration_path_used']} |")

        if failure_rows:
            lines += [
                "",
                "## Failed Directions",
                "",
                "| pair | export_exit | eval_exit | eval_transfer_parquet |",
                "|---|---:|---:|---|",
            ]
            for fr in sorted(failure_rows, key=lambda x: x["pair_id"]):
                lines.append(
                    f"| {fr['pair_id']} | {int(fr['export_exit_code'])} | {int(fr['eval_exit_code'])} | {fr['eval_transfer_parquet']} |"
                )

        out_md = Path(str(args.out_md))
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
        print(f"report_transfer_matrix ok rows={len(df)} out={out_md}")
        return 0
    except Exception as e:
        print(f"report_transfer_matrix error runtime={type(e).__name__}:{e}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

