from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from src.microphys.alpha.transfer_regime import mismatch_diagnostic, summarize_transfer_by_regime


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Slice transfer results by aligned regime.")
    p.add_argument("--matrix-manifest", required=True)
    p.add_argument("--aligned-regimes", required=True)
    p.add_argument("--out-parquet", default="data/derived/regime_alignment/transfer_by_regime.parquet")
    p.add_argument("--out-md", default="reports/transfer/transfer_by_aligned_regime.md")
    return p.parse_args()


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_read_parquet(path: Path) -> pd.DataFrame:
    if not path.exists() or not path.is_file():
        return pd.DataFrame()
    return pd.read_parquet(path)


def _target_presence(aligned: pd.DataFrame, symbol: str) -> pd.DataFrame:
    sub = aligned[aligned["symbol"] == symbol].copy()
    if sub.empty:
        return pd.DataFrame(columns=["aligned_regime_id", "presence_frac"])
    vc = sub["aligned_regime_id"].value_counts(normalize=True).sort_index()
    return pd.DataFrame({"aligned_regime_id": vc.index.astype(int), "presence_frac": vc.values.astype(float)})


def main() -> int:
    args = _parse_args()
    try:
        matrix = _read_json(Path(str(args.matrix_manifest)))
        aligned = _safe_read_parquet(Path(str(args.aligned_regimes)))
        if aligned.empty:
            raise RuntimeError("aligned_regimes_empty")
        rows: List[Dict[str, Any]] = []
        diag_rows: List[Dict[str, Any]] = []
        for pair in list(matrix.get("pairs", []) or []):
            if not bool(pair.get("ok", False)):
                continue
            source = str(pair.get("source_symbol", ""))
            target = str(pair.get("target_symbol", ""))
            cal_mode = str(pair.get("calibration_mode", ""))
            tmanifest = _read_json(Path(str(pair.get("transfer_manifest_json", "")).strip()))
            source_trades = _safe_read_parquet(Path(str((tmanifest.get("source_eval_path", "") or "")).replace("eval.parquet", "trades.parquet")))
            target_trades = _safe_read_parquet(Path(str(tmanifest.get("target_trades_transfer_parquet", "")).strip()))
            by_target = summarize_transfer_by_regime(target_trades, aligned, target_symbol=target)
            by_target["source_symbol"] = source
            by_target["target_symbol"] = target
            by_target["calibration_mode"] = cal_mode
            rows.append(by_target)
            src_reg = summarize_transfer_by_regime(source_trades, aligned, target_symbol=source)
            pres = _target_presence(aligned, target)
            md = mismatch_diagnostic(src_reg, pres)
            diag_rows.append(
                {
                    "pair_id": str(pair.get("pair_id", "")),
                    "source_symbol": source,
                    "target_symbol": target,
                    "calibration_mode": cal_mode,
                    "has_mismatch": bool(md.get("has_mismatch", False)),
                    "source_focus_regime": int(md.get("source_focus_regime", -1)),
                    "target_presence": float(md.get("target_presence", 0.0)),
                }
            )

        out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
        out_diag = pd.DataFrame(diag_rows).sort_values(["source_symbol", "target_symbol", "calibration_mode"]) if diag_rows else pd.DataFrame()
        out_path = Path(str(args.out_parquet))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(out_path, index=False)
        diag_path = out_path.with_name("transfer_by_regime_diag.parquet")
        out_diag.to_parquet(diag_path, index=False)

        lines = [
            "# Transfer By Aligned Regime",
            "",
            f"- matrix_manifest: `{args.matrix_manifest}`",
            f"- aligned_regimes: `{args.aligned_regimes}`",
            f"- rows: `{len(out)}`",
            "",
            "| source | target | cal | aligned_regime_id | trade_count | trigger_rate | mean_net_ret | win_rate | fill_rate |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
        for _, r in out.sort_values(["source_symbol", "target_symbol", "calibration_mode", "aligned_regime_id"]).iterrows():
            lines.append(
                f"| {r['source_symbol']} | {r['target_symbol']} | {r['calibration_mode']} | {int(r['aligned_regime_id'])} | "
                f"{int(r['trade_count'])} | {float(r['trigger_rate']):.4f} | {float(r['mean_net_ret']):.8f} | {float(r['win_rate']):.4f} | {float(r['fill_rate']):.4f} |"
            )
        lines += ["", "## Regime Mismatch Diagnostics", "", "| pair | mismatch | source_focus_regime | target_presence |", "|---|---:|---:|---:|"]
        for _, r in out_diag.iterrows():
            lines.append(
                f"| {r['source_symbol']}->{r['target_symbol']} ({r['calibration_mode']}) | {int(bool(r['has_mismatch']))} | {int(r['source_focus_regime'])} | {float(r['target_presence']):.4f} |"
            )
        out_md = Path(str(args.out_md))
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
        print(f"report_transfer_by_aligned_regime ok rows={len(out)} out={out_md}")
        return 0
    except Exception as e:
        print(f"report_transfer_by_aligned_regime error runtime={type(e).__name__}:{e}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

