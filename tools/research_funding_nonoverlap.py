"""Funding mean-reversion, NON-OVERLAPPING (the make-or-break discipline test).

The first funding result (|z|>=1, 24h forward, every snapshot) showed huge N and
huge P&L -- but consecutive snapshots are autocorrelated and their forward windows
overlap, so N is fake and the P&L is inflated. This redoes it with NON-OVERLAPPING
trades on the funding grid: at each 8h funding boundary, z-score the funding rate
vs its trailing distribution; if |z|>=threshold, take ONE trade (SHORT if z>0 =
long-crowded -> revert down; LONG if z<0), hold exactly to the next 8h boundary
(horizon-matched), exit. One trade per slot, no overlap -> N is truly independent.

Reports per symbol AND per side (SHORT vs LONG separately = a beta control: a pure
directional/regime bet wins one side and loses the other; a real funding edge wins
both). Real round-trip cost. Chronological holdout. z sweep.
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.research_s34_knowable_anchor_continuation import load_mark_index, mean, pctile, r1, r3, signed_return_bps
from ami.storage import production as PR
from ami.storage import research_reader as RR

DEFAULT_DB = ROOT / "data" / "microstructure.db"
OUT_DIR = ROOT / "reports" / "research" / "s34"
OUT_JSON = OUT_DIR / "FUNDING_NONOVERLAP.json"
OUT_MD = OUT_DIR / "FUNDING_NONOVERLAP.md"

SYMBOLS = ("ETHUSDT", "BTCUSDT", "SOLUSDT")
SLOT_SEC = 8 * 3600
TRAIL = 90  # trailing funding obs (~30 days at 3/day) for the z baseline
Z_THRESHOLDS = (1.0, 1.5, 2.0)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def funding_at(conn, symbol, ts_ms):
    """Direct-SQL oracle -- kept as the parity reference for `funding_at_v2`
    (BATCH-STORAGE-ROTATION-RETENTION-ASOF-LOOKUP-CONSUMER-MIGRATION-V4).
    No longer called by build_trades(); the reader-backed path is used
    instead."""
    row = conn.execute(
        "SELECT funding_rate FROM mark_prices WHERE symbol=? AND ts_ms<=? AND funding_rate IS NOT NULL "
        "ORDER BY ts_ms DESC LIMIT 1",
        (symbol, int(ts_ms)),
    ).fetchone()
    return float(row[0]) if row and row[0] is not None else None


def funding_at_v2(root, symbol, ts_ms, source_db_path=None):
    """Reader-backed replacement for `funding_at`, via
    lookup_latest_at_or_before with the `funding_rate IS NOT NULL`
    predicate expressed as a helper filter. ETHUSDT resolves through the
    real mark_prices/ETHUSDT/2026-05 archive for historical timestamps;
    BTCUSDT/SOLUSDT have no mark_prices archive and resolve SQLITE_ONLY."""
    result = RR.lookup_latest_at_or_before(
        root, table="mark_prices", symbol=symbol, ts_ms=int(ts_ms), columns=("funding_rate",),
        filters=(("funding_rate", "!=", None),), source_db_path=source_db_path)
    return float(result.row[0]) if result.found and result.row[0] is not None else None


def metrics(vals):
    vals = [v for v in vals if v is not None and math.isfinite(v)]
    if not vals:
        return {"n": 0, "sum": 0.0, "mean": None, "median": None, "win_rate": None, "max_loss": None, "t3r": None}
    s = sorted(vals, reverse=True)
    return {"n": len(vals), "sum": r1(sum(vals)), "mean": r1(mean(vals)), "median": r1(pctile(vals, 0.5)),
            "win_rate": r3(sum(1 for v in vals if v > 0) / len(vals)), "max_loss": r1(min(vals)),
            "t3r": r1(sum(s[3:])) if len(s) > 3 else r1(sum(s))}


def build_trades(conn, root, symbol, z_thr, cost, source_db_path=None):
    marks = load_mark_index(conn, symbol)
    if not marks.ts:
        return []
    end = marks.ts[-1]
    start = (marks.ts[0] // SLOT_SEC // 1000) * SLOT_SEC * 1000
    grid = list(range(int(start), int(end), SLOT_SEC * 1000))
    fund = []
    for g in grid:
        f = funding_at_v2(root, symbol, g, source_db_path=source_db_path)
        fund.append((g, f))
    trades = []
    hist = []
    for i, (g, f) in enumerate(fund):
        if f is None:
            continue
        if len(hist) >= TRAIL:
            base = hist[-TRAIL:]
            mu = sum(base) / len(base)
            sd = math.sqrt(sum((x - mu) ** 2 for x in base) / (len(base) - 1)) if len(base) > 1 else 0.0
            if sd > 0:
                z = (f - mu) / sd
                if abs(z) >= z_thr:
                    direction = "SHORT" if z > 0 else "LONG"
                    e = marks.at_or_after(g)
                    x = marks.at_or_after(g + SLOT_SEC * 1000)
                    if e and x and float(e[1]) > 0:
                        net = signed_return_bps(direction, float(e[1]), float(x[1])) - cost
                        trades.append({"ts_ms": g, "side": direction, "z": r1(z), "net": net})
        hist.append(f)
    return trades


def split_side(trades, side, holdout_frac):
    sub = sorted([t for t in trades if (side is None or t["side"] == side)], key=lambda t: t["ts_ms"])
    if not sub:
        return {"cal": metrics([]), "hold": metrics([])}
    cut = sub[int(len(sub) * (1.0 - holdout_frac))]["ts_ms"]
    return {"cal": metrics([t["net"] for t in sub if t["ts_ms"] < cut]),
            "hold": metrics([t["net"] for t in sub if t["ts_ms"] >= cut])}


def render_md(report):
    cfg = report["config"]
    lines = [
        "# Funding Mean-Reversion — NON-OVERLAPPING (make-or-break)",
        "",
        f"Generated: `{report['generated_at_utc']}`  |  8h slot/hold (horizon-matched), trailing-{TRAIL} z baseline, "
        f"cost {cfg['cost']}bps RT, holdout {cfg['holdout_frac']}",
        "",
        "Non-overlapping: one trade per 8h funding slot, no overlap -> N is independent. SHORT vs LONG split = beta "
        "control (pure regime bet wins one side only; real funding edge wins BOTH).",
        "",
    ]
    for sym in report["symbols"]:
        lines.append(f"## {sym['symbol']}")
        for zt in sym["z"]:
            lines.append("")
            lines.append(f"### z>= {zt['z_thr']}  (trades={zt['n_total']})")
            lines.append("| Side | cal N | cal sum | cal mean | cal win | hold N | hold sum | hold mean | hold win | hold maxL | hold T3R |")
            lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
            for side in ("ALL", "SHORT", "LONG"):
                d = zt["sides"][side]
                cw = lambda m: None if m["win_rate"] is None else r1(m["win_rate"] * 100.0)
                lines.append(f"| {side} | {d['cal']['n']} | {d['cal']['sum']} | {d['cal']['mean']} | {cw(d['cal'])} | "
                             f"{d['hold']['n']} | {d['hold']['sum']} | {d['hold']['mean']} | {cw(d['hold'])} | {d['hold']['max_loss']} | {d['hold']['t3r']} |")
        lines.append("")
    return "\n".join(lines)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Non-overlapping funding mean-reversion test (discipline gate).")
    p.add_argument("--db", type=Path, default=DEFAULT_DB)
    p.add_argument("--cost-bps-rt", type=float, default=8.0)
    p.add_argument("--holdout-frac", type=float, default=0.30)
    p.add_argument("--symbols", default=",".join(SYMBOLS))
    p.add_argument("--json-out", type=Path, default=OUT_JSON)
    p.add_argument("--md-out", type=Path, default=OUT_MD)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    syms = [s.strip() for s in str(args.symbols).split(",") if s.strip()]
    root, _root_source = PR.resolve_production_root()
    out = []
    with sqlite3.connect(f"file:{args.db}?mode=ro", uri=True) as conn:
        for sym in syms:
            z_out = []
            for zt in Z_THRESHOLDS:
                trades = build_trades(conn, root, sym, zt, float(args.cost_bps_rt),
                                       source_db_path=str(args.db))
                z_out.append({"z_thr": zt, "n_total": len(trades),
                              "sides": {"ALL": split_side(trades, None, float(args.holdout_frac)),
                                        "SHORT": split_side(trades, "SHORT", float(args.holdout_frac)),
                                        "LONG": split_side(trades, "LONG", float(args.holdout_frac))}})
            out.append({"symbol": sym, "z": z_out})
    report = {"generated_at_utc": utc_now(),
              "config": {"cost": float(args.cost_bps_rt), "holdout_frac": float(args.holdout_frac)},
              "symbols": out}
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    args.md_out.write_text(render_md(report), encoding="utf-8")
    print(render_md(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
