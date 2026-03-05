from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Set

from .spec import SignalSpec


def collect_expr_columns(expr: Dict[str, Any]) -> Set[str]:
    cols: Set[str] = set()
    kind = str(expr.get("type", "")).lower()
    if kind in {"gt", "gte", "lt", "lte", "eq", "ne"}:
        left = str(expr.get("left", "")).strip()
        if left:
            cols.add(left)
    elif kind == "in":
        col = str(expr.get("col", "")).strip()
        if col:
            cols.add(col)
    elif kind == "fn":
        col = str(expr.get("col", "")).strip()
        if col:
            cols.add(col)
    elif kind in {"and", "or", "not"}:
        for sub in list(expr.get("args", []) or []):
            cols.update(collect_expr_columns(dict(sub)))
    return cols


@dataclass(frozen=True)
class ColumnValidationResult:
    ok: bool
    reason: str
    missing_columns: List[str]
    high_nan_columns: List[str]


def validate_signal_columns(
    spec: SignalSpec,
    *,
    available_columns: Iterable[str],
    nan_ratio: Dict[str, float] | None = None,
    max_nan_ratio: float = 0.95,
) -> ColumnValidationResult:
    available = {str(c) for c in available_columns}
    used = sorted(collect_expr_columns(spec.condition))
    missing = [c for c in used if c not in available]
    high_nan: List[str] = []
    nmap = dict(nan_ratio or {})
    for col in used:
        ratio = float(nmap.get(col, 0.0))
        if ratio > float(max_nan_ratio):
            high_nan.append(col)
    if missing:
        return ColumnValidationResult(False, "missing_columns", missing, high_nan)
    if high_nan:
        return ColumnValidationResult(False, "high_nan_columns", [], high_nan)
    return ColumnValidationResult(True, "ok", [], [])
