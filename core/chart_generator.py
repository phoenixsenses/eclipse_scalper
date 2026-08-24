from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Iterable


def _load_equity_points(db_path: Path, days: int = 7) -> tuple[list[float], list[float]]:
    if not db_path.exists():
        return [], []
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    try:
        cutoff = time.time() - (max(1, int(days)) * 86400.0)
        rows = conn.execute(
            "SELECT exit_time, pnl_bps FROM trades WHERE exit_time>=? ORDER BY exit_time ASC",
            (cutoff,),
        ).fetchall()
    finally:
        conn.close()
    if not rows:
        return [], []
    xs: list[float] = []
    ys: list[float] = []
    acc = 0.0
    for ts, pnl in rows:
        tsv = float(ts or 0.0)
        acc += float(pnl or 0.0)
        xs.append(tsv)
        ys.append(acc)
    return xs, ys


def build_equity_curve_png(db_path: str | Path, out_path: str | Path, *, days: int = 7) -> bool:
    xs, ys = _load_equity_points(Path(db_path), days=days)
    if not xs:
        return False
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 3))
    plt.plot(xs, ys, linewidth=1.6)
    plt.title(f"Paper Equity (cum pnl bps, {max(1, int(days))}d)")
    plt.xlabel("exit_time")
    plt.ylabel("cum pnl bps")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()
    return True

