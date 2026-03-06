from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CalibrationContext:
    quantiles: Dict[str, Dict[str, float]]
    nan_ratio: Dict[str, float]
    sample_count: int

    def q(self, col: str, quantile: float, default: float = 0.0) -> float:
        key = f"{float(max(0.0, min(1.0, quantile))):.4f}"
        try:
            return float(self.quantiles.get(col, {}).get(key, default))
        except Exception:
            return float(default)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["quantiles"] = {
            str(k): {str(q): float(v) for q, v in sorted((vv or {}).items())}
            for k, vv in sorted((payload.get("quantiles") or {}).items())
        }
        payload["nan_ratio"] = {str(k): float(v) for k, v in sorted((payload.get("nan_ratio") or {}).items())}
        payload["sample_count"] = int(payload.get("sample_count", 0))
        return payload

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "CalibrationContext":
        qmap: Dict[str, Dict[str, float]] = {}
        for col, qv in dict(payload.get("quantiles", {}) or {}).items():
            qmap[str(col)] = {str(k): float(v) for k, v in dict(qv or {}).items()}
        nmap = {str(k): float(v) for k, v in dict(payload.get("nan_ratio", {}) or {}).items()}
        return cls(quantiles=qmap, nan_ratio=nmap, sample_count=int(payload.get("sample_count", 0)))


def compute_calibration(
    df: pd.DataFrame,
    *,
    columns: Iterable[str],
    quantile_grid: Iterable[float] | None = None,
) -> CalibrationContext:
    qgrid = list(quantile_grid or (0.01, 0.05, 0.10, 0.20, 0.50, 0.80, 0.90, 0.95, 0.99))
    quantiles: Dict[str, Dict[str, float]] = {}
    nan_ratio: Dict[str, float] = {}
    n = int(len(df))
    for col in columns:
        s = pd.to_numeric(df.get(col), errors="coerce")
        if s is None:
            continue
        valid = s.dropna()
        nan_ratio[str(col)] = float(1.0 - (len(valid) / max(1, n)))
        if valid.empty:
            quantiles[str(col)] = {}
            continue
        qvals = valid.quantile([max(0.0, min(1.0, float(q))) for q in qgrid]).to_dict()
        quantiles[str(col)] = {f"{float(k):.4f}": float(v) for k, v in qvals.items()}
        abs_valid = valid.abs()
        abs_qvals = abs_valid.quantile([max(0.0, min(1.0, float(q))) for q in qgrid]).to_dict()
        quantiles[f"abs({col})"] = {f"{float(k):.4f}": float(v) for k, v in abs_qvals.items()}
    return CalibrationContext(quantiles=quantiles, nan_ratio=nan_ratio, sample_count=n)


def save_calibration(ctx: CalibrationContext, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ctx.to_dict(), ensure_ascii=True, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def load_calibration(path: Path) -> CalibrationContext:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return CalibrationContext.from_dict(payload)
