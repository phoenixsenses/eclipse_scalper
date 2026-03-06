from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd

from .calibration import CalibrationContext
from .eval import apply_signal_entries
from .spec import SignalSpec


def _phi(a: np.ndarray, b: np.ndarray) -> float:
    a1 = a.astype(bool)
    b1 = b.astype(bool)
    n11 = float(np.logical_and(a1, b1).sum())
    n10 = float(np.logical_and(a1, ~b1).sum())
    n01 = float(np.logical_and(~a1, b1).sum())
    n00 = float(np.logical_and(~a1, ~b1).sum())
    den = (n11 + n10) * (n11 + n01) * (n00 + n10) * (n00 + n01)
    if den <= 0:
        return 0.0
    return float((n11 * n00 - n10 * n01) / np.sqrt(den))


def build_trigger_matrix(
    frame: pd.DataFrame,
    specs: Sequence[SignalSpec],
    *,
    calibration: CalibrationContext | None = None,
    max_rows: int = 20_000,
) -> tuple[np.ndarray, list[str]]:
    if frame.empty or not specs:
        return np.zeros((0, 0), dtype=np.uint8), []
    stride = max(1, int(len(frame) // int(max_rows)))
    view = frame.iloc[::stride].reset_index(drop=True) if stride > 1 else frame
    names: list[str] = []
    cols: list[np.ndarray] = []
    for spec in specs:
        m = apply_signal_entries(view, spec, calibration=calibration).fillna(False).to_numpy(dtype=bool)
        names.append(spec.name)
        cols.append(m.astype(np.uint8))
    mat = np.vstack(cols) if cols else np.zeros((0, len(view)), dtype=np.uint8)
    return mat, names


def pairwise_overlap(
    frame: pd.DataFrame,
    specs: Sequence[SignalSpec],
    *,
    calibration: CalibrationContext | None = None,
    max_rows: int = 20_000,
    top_pairs: int = 20_000,
) -> pd.DataFrame:
    mat, names = build_trigger_matrix(frame, specs, calibration=calibration, max_rows=max_rows)
    n = int(mat.shape[0])
    rows: list[dict] = []
    if n <= 1:
        return pd.DataFrame(columns=["a", "b", "jaccard", "phi", "intersect", "union"])
    for i in range(n):
        ai = mat[i].astype(bool)
        for j in range(i + 1, n):
            bj = mat[j].astype(bool)
            inter = int(np.logical_and(ai, bj).sum())
            uni = int(np.logical_or(ai, bj).sum())
            jac = float(inter / uni) if uni > 0 else 0.0
            rows.append(
                {
                    "a": names[i],
                    "b": names[j],
                    "jaccard": jac,
                    "phi": _phi(ai, bj),
                    "intersect": inter,
                    "union": uni,
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.sort_values(["jaccard", "phi", "a", "b"], ascending=[False, False, True, True]).reset_index(drop=True)
    return out.head(int(max(1, top_pairs))).copy()


@dataclass(frozen=True)
class DedupResult:
    selected: List[SignalSpec]
    dropped: List[str]
    clusters: List[List[str]]


def dedupe_specs(
    specs: Sequence[SignalSpec],
    overlap_pairs: pd.DataFrame,
    *,
    jaccard_thr: float = 0.90,
    target_triggers_per_day: float = 200.0,
) -> DedupResult:
    by_name = {s.name: s for s in specs}
    names = sorted(by_name.keys())
    parent = {n: n for n in names}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if ra < rb:
            parent[rb] = ra
        else:
            parent[ra] = rb

    if not overlap_pairs.empty:
        for _, r in overlap_pairs.iterrows():
            if float(r.get("jaccard", 0.0)) >= float(jaccard_thr):
                a = str(r["a"])
                b = str(r["b"])
                if a in parent and b in parent:
                    union(a, b)

    groups: Dict[str, List[str]] = {}
    for n in names:
        groups.setdefault(find(n), []).append(n)

    selected: List[SignalSpec] = []
    dropped: List[str] = []
    clusters: List[List[str]] = []
    for root in sorted(groups):
        cluster = sorted(groups[root])
        clusters.append(cluster)
        best = cluster[0]
        best_score = -1e18
        for name in cluster:
            s = by_name[name]
            m = s.meta or {}
            tpd = float(m.get("trigger_rate_per_day", 0.0) or 0.0)
            trig = float(m.get("calibration_triggered", 0.0) or 0.0)
            closeness = -abs(tpd - float(target_triggers_per_day))
            score = (10.0 * closeness) + trig
            if score > best_score or (score == best_score and name < best):
                best = name
                best_score = score
        selected.append(by_name[best])
        dropped.extend([n for n in cluster if n != best])

    selected = sorted(selected, key=lambda s: s.name)
    dropped = sorted(set(dropped))
    clusters = sorted([sorted(c) for c in clusters], key=lambda c: (len(c), c[0]), reverse=True)
    return DedupResult(selected=selected, dropped=dropped, clusters=clusters)
