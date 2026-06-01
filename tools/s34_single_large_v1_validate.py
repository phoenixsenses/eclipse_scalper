"""DB-native S34 single-large validation.

Validates detector_signals with liq_composition='single_large' against
clustered/other compositions using mark-price forward returns and optional
book-state fields already written by analyze_bookticker_recovery.py.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path
from statistics import mean, median

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DB = "data/microstructure.db"
OUT_MD = Path("reports/S34_SINGLE_LARGE_V1_VALIDATE.md")
OUT_JSON = Path("reports/S34_SINGLE_LARGE_V1_VALIDATE.json")
HORIZONS = [60, 120, 300, 900]


def _wr(vals: list[float]) -> float | None:
    return 100.0 * sum(1 for x in vals if x > 0) / len(vals) if vals else None


def _mark_after(conn: sqlite3.Connection, ts_ms: int) -> float | None:
    row = conn.execute(
        "SELECT mark_price FROM mark_prices WHERE symbol='ETHUSDT' AND ts_ms>=? ORDER BY ts_ms ASC LIMIT 1",
        (ts_ms,),
    ).fetchone()
    return float(row[0]) if row and row[0] is not None else None


def _fmt(x: object) -> str:
    if x is None:
        return "n/a"
    if isinstance(x, float):
        return f"{x:.2f}"
    return str(x)


def main() -> None:
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    rows = [
        dict(r)
        for r in conn.execute(
            """
            SELECT signal_ts_ms, signal_type, entry_price, liq_composition,
                   fragility_zone, regime_at_entry, basis_at_entry, ofi_at_entry,
                   entry_book_state, depth_recovery_pct
            FROM detector_signals
            WHERE symbol='ETHUSDT' AND signal_ts_ms IS NOT NULL AND entry_price IS NOT NULL
            ORDER BY signal_ts_ms ASC
            """
        )
    ]
    enriched = []
    for r in rows:
        ep = float(r["entry_price"])
        fwd = {}
        for h in HORIZONS:
            xp = _mark_after(conn, int(r["signal_ts_ms"]) + h * 1000)
            if xp:
                fwd[str(h)] = (ep - xp) / ep * 1e4  # ETH S34 short
        r["fwd_bps"] = fwd
        enriched.append(r)
    conn.close()

    def stat(label: str, subset: list[dict], h: int) -> dict:
        vals = [float(r["fwd_bps"][str(h)]) for r in subset if str(h) in r["fwd_bps"]]
        return {
            "label": label,
            "horizon_sec": h,
            "n": len(vals),
            "wr": _wr(vals),
            "mean_bps": mean(vals) if vals else None,
            "median_bps": median(vals) if vals else None,
        }

    groups = {
        "all": enriched,
        "single_large": [r for r in enriched if r.get("liq_composition") == "single_large"],
        "clustered": [r for r in enriched if r.get("liq_composition") == "clustered"],
        "other_or_null": [r for r in enriched if r.get("liq_composition") not in {"single_large", "clustered"}],
        "single_large_basis_pos": [
            r for r in enriched if r.get("liq_composition") == "single_large" and r.get("basis_at_entry") is not None and float(r["basis_at_entry"]) > 0
        ],
        "single_large_ofi_pos": [
            r for r in enriched if r.get("liq_composition") == "single_large" and r.get("ofi_at_entry") is not None and float(r["ofi_at_entry"]) > 0
        ],
        "single_large_book_partial": [
            r for r in enriched if r.get("liq_composition") == "single_large" and r.get("entry_book_state") == "partial"
        ],
        "single_large_book_recovered": [
            r for r in enriched if r.get("liq_composition") == "single_large" and r.get("entry_book_state") == "recovered"
        ],
    }
    results = [stat(label, subset, h) for label, subset in groups.items() for h in HORIZONS]
    best = max(
        [r for r in results if r["n"] >= 5],
        key=lambda r: (float(r["mean_bps"] or -1e9), float(r["wr"] or 0.0), int(r["n"])),
        default=None,
    )
    verdict = "MONITOR"
    sl120 = next((r for r in results if r["label"] == "single_large" and r["horizon_sec"] == 120), None)
    cl120 = next((r for r in results if r["label"] == "clustered" and r["horizon_sec"] == 120), None)
    if sl120 and cl120 and (sl120["n"] >= 10) and (float(sl120["mean_bps"] or 0) > float(cl120["mean_bps"] or 0)):
        verdict = "VALIDATE_FORWARD"

    payload = {"verdict": verdict, "rows": len(enriched), "results": results, "best": best}
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = ["# S34 Single Large V1 Validate", "", f"- verdict: `{verdict}`", f"- detector_rows: `{len(enriched)}`", ""]
    lines.append("## Results")
    lines.append("")
    lines.append("| group | h | N | WR | mean_bps | median_bps |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for r in results:
        lines.append(
            f"| {r['label']} | {r['horizon_sec']} | {r['n']} | {_fmt(r['wr'])}% | {_fmt(r['mean_bps'])} | {_fmt(r['median_bps'])} |"
        )
    lines.append("")
    lines.append("## Best N>=5")
    lines.append("")
    lines.append(f"`{best}`")
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUT_MD}")
    print(f"Wrote {OUT_JSON}")
    print(f"Verdict: {verdict}")
    print(f"Best: {best}")


if __name__ == "__main__":
    main()
