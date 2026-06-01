"""Current SOL forced-flow transfer refresh.

Tests whether SOL liquidation events support either:
- SELL liquidation -> long reversal/continuation upward
- BUY liquidation -> short continuation downward

Outputs:
  reports/SOL_FORCED_FLOW_TRANSFER.md
  reports/SOL_FORCED_FLOW_TRANSFER.json
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path
from statistics import mean

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DB = "data/microstructure.db"
OUT_MD = Path("reports/SOL_FORCED_FLOW_TRANSFER.md")
OUT_JSON = Path("reports/SOL_FORCED_FLOW_TRANSFER.json")

THRESHOLDS = [25_000, 50_000, 100_000, 200_000, 500_000]
HORIZONS = [60, 300, 900]


def _wr(wins: int, n: int) -> float | None:
    return (wins / n * 100.0) if n else None


def _fmt(x: float | None, suffix: str = "") -> str:
    if x is None:
        return "n/a"
    return f"{x:.2f}{suffix}"


def _event_return(conn: sqlite3.Connection, ts_ms: int, side: str, horizon_sec: int) -> float | None:
    entry = conn.execute(
        "SELECT mark_price FROM mark_prices WHERE symbol='SOLUSDT' AND ts_ms <= ? ORDER BY ts_ms DESC LIMIT 1",
        (ts_ms,),
    ).fetchone()
    exit_ = conn.execute(
        "SELECT mark_price FROM mark_prices WHERE symbol='SOLUSDT' AND ts_ms >= ? ORDER BY ts_ms ASC LIMIT 1",
        (ts_ms + horizon_sec * 1000,),
    ).fetchone()
    if not entry or not exit_:
        return None
    ep = float(entry[0])
    xp = float(exit_[0])
    if ep <= 0:
        return None
    raw = (xp - ep) / ep
    if side == "SELL":
        return raw  # long after sell liquidation
    return -raw  # short after buy liquidation


def _stats(conn: sqlite3.Connection, side: str, threshold: float, horizon_sec: int) -> dict:
    rows = conn.execute(
        """
        SELECT ts_ms, notional
        FROM liquidations
        WHERE symbol='SOLUSDT' AND side=? AND notional>=?
        ORDER BY ts_ms
        """,
        (side, threshold),
    ).fetchall()
    rets = []
    for ts_ms, _notional in rows:
        r = _event_return(conn, int(ts_ms), side, horizon_sec)
        if r is not None:
            rets.append(float(r))
    n = len(rets)
    wins = sum(1 for r in rets if r > 0)
    return {
        "side": side,
        "threshold": threshold,
        "horizon_sec": horizon_sec,
        "n": n,
        "wr": _wr(wins, n),
        "mean_bps": (mean(rets) * 1e4) if rets else None,
        "median_bps": (sorted(rets)[n // 2] * 1e4) if rets else None,
    }


def main() -> None:
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB)

    coverage = {}
    for side in ["BUY", "SELL"]:
        coverage[side] = {}
        for threshold in THRESHOLDS:
            coverage[side][str(threshold)] = conn.execute(
                "SELECT COUNT(*) FROM liquidations WHERE symbol='SOLUSDT' AND side=? AND notional>=?",
                (side, threshold),
            ).fetchone()[0]

    results = [
        _stats(conn, side, threshold, horizon)
        for side in ["SELL", "BUY"]
        for threshold in THRESHOLDS
        for horizon in HORIZONS
    ]
    conn.close()

    qualifying = [r for r in results if int(r["n"]) >= 20]
    best = max(
        qualifying,
        key=lambda r: (float(r["mean_bps"] or -1e9), float(r["wr"] or 0.0), int(r["n"])),
        default=None,
    )

    verdict = "NO_PROMOTION"
    if best and float(best["mean_bps"] or 0.0) > 5.0 and float(best["wr"] or 0.0) >= 60.0:
        verdict = "SHADOW_CANDIDATE"

    payload = {
        "coverage": coverage,
        "results": results,
        "best_qualifying_n_ge_20": best,
        "verdict": verdict,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = ["# SOL Forced-Flow Transfer", "", f"- verdict: `{verdict}`", ""]
    lines.append("## Coverage")
    lines.append("")
    lines.append("| side | >=25k | >=50k | >=100k | >=200k | >=500k |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for side in ["SELL", "BUY"]:
        c = coverage[side]
        lines.append(
            f"| {side} | {c['25000']} | {c['50000']} | {c['100000']} | {c['200000']} | {c['500000']} |"
        )

    lines.append("")
    lines.append("## Results")
    lines.append("")
    lines.append("| side thesis | threshold | h | N | WR | mean_bps | median_bps |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for r in results:
        thesis = "SELL->LONG" if r["side"] == "SELL" else "BUY->SHORT"
        lines.append(
            f"| {thesis} | {int(r['threshold'])} | {r['horizon_sec']} | {r['n']} | "
            f"{_fmt(r['wr'], '%')} | {_fmt(r['mean_bps'])} | {_fmt(r['median_bps'])} |"
        )
    lines.append("")
    lines.append("## Best N>=20")
    lines.append("")
    lines.append(f"`{best}`")
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUT_MD}")
    print(f"Wrote {OUT_JSON}")
    print(f"Verdict: {verdict}")
    print(f"Best N>=20: {best}")


if __name__ == "__main__":
    main()
