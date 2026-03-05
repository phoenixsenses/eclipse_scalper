from __future__ import annotations

import numpy as np
import pandas as pd


EPS = 1e-12


def compute_response_kernel(mid: pd.Series, ofi: pd.Series, max_lag: int = 200) -> pd.DataFrame:
    m = pd.to_numeric(mid, errors="coerce").to_numpy(dtype=float)
    o = pd.to_numeric(ofi, errors="coerce").to_numpy(dtype=float)
    sign = np.sign(o)

    rows: list[dict[str, float | int]] = []
    for tau in range(1, int(max_lag) + 1):
        if tau >= len(m):
            rows.append({"tau": tau, "response": 0.0, "abs_response": 0.0, "count": 0})
            continue
        base = m[:-tau]
        fwd = m[tau:]
        valid = np.isfinite(base) & np.isfinite(fwd) & (base > 0) & (fwd > 0)
        if not np.any(valid):
            rows.append({"tau": tau, "response": 0.0, "abs_response": 0.0, "count": 0})
            continue
        ret = np.log(fwd[valid] / base[valid])
        s = sign[:-tau][valid]
        x = ret * s
        response = float(np.mean(x)) if len(x) else 0.0
        abs_response = float(np.mean(np.abs(x))) if len(x) else 0.0
        rows.append({"tau": tau, "response": response, "abs_response": abs_response, "count": int(len(x))})

    out = pd.DataFrame(rows)
    out["cumulative_response"] = out["response"].cumsum()
    return out


def summarize_kernel(kernel: pd.DataFrame) -> dict[str, float]:
    if kernel.empty:
        return {"max_response": 0.0, "min_response": 0.0, "final_cumulative": 0.0}
    return {
        "max_response": float(kernel["response"].max()),
        "min_response": float(kernel["response"].min()),
        "final_cumulative": float(kernel["cumulative_response"].iloc[-1]),
    }
