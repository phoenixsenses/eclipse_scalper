from __future__ import annotations

from typing import Dict, Iterable, List, Optional


def forward_return(mid_series: List[float], horizon_steps: int) -> List[Optional[float]]:
    if horizon_steps <= 0:
        raise ValueError("horizon_steps must be > 0")
    n = len(mid_series)
    out: List[Optional[float]] = [None] * n
    for i in range(n):
        j = i + horizon_steps
        if j >= n:
            out[i] = None
            continue
        a = float(mid_series[i] or 0.0)
        b = float(mid_series[j] or 0.0)
        if a <= 0.0:
            out[i] = None
            continue
        out[i] = (b / a) - 1.0
    return out


def direction_label(fwd_ret: float, threshold: float) -> int:
    x = float(fwd_ret)
    th = abs(float(threshold))
    if x > th:
        return 1
    if x < -th:
        return -1
    return 0


def make_labels(
    timestamps: Iterable[int],
    mids: Iterable[float],
    horizons: List[int] | None = None,
    threshold: float = 0.0002,
) -> Dict[str, Dict[int, List[Optional[float] | int | None]]]:
    ts_list = [int(t) for t in timestamps]
    mid_list = [float(m) for m in mids]
    if len(ts_list) != len(mid_list):
        raise ValueError("timestamps and mids must have same length")
    hs = list(horizons or [30, 60, 300])
    out: Dict[str, Dict[int, List[Optional[float] | int | None]]] = {
        "timestamps": {0: ts_list},
        "returns": {},
        "labels": {},
    }
    for h in hs:
        ret = forward_return(mid_list, int(h))
        lbl: List[Optional[int]] = [None if r is None else direction_label(float(r), float(threshold)) for r in ret]
        out["returns"][int(h)] = ret
        out["labels"][int(h)] = lbl
    return out

