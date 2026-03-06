from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd

from src.microphys.alpha.spec import SignalSpec, signal_from_dict


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_run_pointers(run_dir: Path) -> Dict[str, str]:
    p = run_dir / "pointers.json"
    if not p.exists():
        return {}
    raw = _read_json(p)
    return {str(k): str(v) for k, v in dict(raw).items()}


def load_specs_jsonl(path: Path) -> List[SignalSpec]:
    specs: List[SignalSpec] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s:
            continue
        specs.append(signal_from_dict(json.loads(s)))
    return specs


def specs_jsonl_lines(specs: Iterable[SignalSpec]) -> List[str]:
    out: List[str] = []
    for spec in specs:
        out.append(json.dumps(spec.to_dict(), ensure_ascii=True, sort_keys=True, separators=(",", ":")))
    return out


def write_specs_jsonl(path: Path, specs: Iterable[SignalSpec]) -> str:
    lines = specs_jsonl_lines(specs)
    payload = "\n".join(lines) + ("\n" if lines else "")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def rank_source_signals(eval_df: pd.DataFrame, *, score_col: str, top_k: int) -> List[str]:
    if eval_df.empty or "signal" not in eval_df.columns:
        return []
    col = str(score_col)
    if col not in eval_df.columns:
        col = "test_sharpe" if "test_sharpe" in eval_df.columns else ("test_net_mean" if "test_net_mean" in eval_df.columns else "")
    if not col:
        return []
    agg = (
        eval_df.groupby("signal", as_index=False)
        .agg(score=(col, "mean"), test_trade_count=("test_trade_count", "sum"))
        .sort_values(["score", "test_trade_count", "signal"], ascending=[False, False, True])
        .reset_index(drop=True)
    )
    return agg.head(max(1, int(top_k)))["signal"].astype(str).tolist()


def load_partitioned_parquet(root: Path, *, symbol: str, interval_ms: int, name: str) -> pd.DataFrame:
    base = root / f"interval_ms={int(interval_ms)}" / f"symbol={symbol}"
    files = sorted(base.glob(f"date=*/{name}.parquet"))
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(p) for p in files], ignore_index=True).sort_values("ts_ms").reset_index(drop=True)


def merge_physics_regimes(physics: pd.DataFrame, regimes: pd.DataFrame) -> pd.DataFrame:
    if physics.empty:
        return physics
    if regimes.empty:
        out = physics.copy()
        if "regime_id" not in out.columns:
            out["regime_id"] = -1
        return out
    cols = [c for c in ("ts_ms", "regime_id", "regime_name", "regime_prob") if c in regimes.columns]
    reg = regimes[cols].drop_duplicates(subset=["ts_ms"], keep="last")
    out = physics.merge(reg, on="ts_ms", how="left")
    out["regime_id"] = pd.to_numeric(out.get("regime_id"), errors="coerce").fillna(-1).astype(int)
    return out


def missing_artifacts(pointers: Dict[str, str], keys: Iterable[str]) -> List[str]:
    out: List[str] = []
    for key in keys:
        raw = str(pointers.get(str(key), "")).strip()
        if not raw:
            out.append(str(key))
            continue
        p = Path(raw)
        if not p.exists():
            out.append(str(key))
    return out

