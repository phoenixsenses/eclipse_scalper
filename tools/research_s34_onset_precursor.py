"""S34 Onset Precursor.

Horizon decay showed the cascade's price move is already spent by the knowable
threshold cross. The only honest way to capture it is to enter EARLIER -- at the
onset of the liquidation burst, before we know it will become a cascade.

This probe defines a lookahead-free onset: a liquidation that follows >= gap_sec
of quiet (no same-symbol/same-side liquidation). That is fully knowable in real
time -- you see the liq and you know the gap before it, with NO knowledge of
whether it will grow into a cascade. We enter at the onset (continuation
direction: LONG for BUY-liq, SHORT for SELL-liq) and measure the pure directional
mark return at a ladder of short horizons.

If onset entry shows a positive short-horizon directional move, that is the
precursor edge to develop. If onset is also flat/negative, the move is not
capturable anywhere and the liquidation family is truly dead.

It also splits by the onset liquidation's own size and by whether the burst
"builds" (>= build_n liqs within build_window_sec AFTER onset) -- the build flag
is NOT knowable at onset, so it is reported only as a forward diagnostic, never
as an entry condition.
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.research_s34_knowable_anchor_continuation import (
    load_liquidations,
    load_mark_index,
    mean,
    pctile,
    r1,
    r3,
    signed_return_bps,
)

DEFAULT_DB = ROOT / "data" / "microstructure.db"
OUT_DIR = ROOT / "reports" / "research" / "s34"
OUT_JSON = OUT_DIR / "S34_ONSET_PRECURSOR.json"
OUT_MD = OUT_DIR / "S34_ONSET_PRECURSOR.md"

HORIZONS_SEC = (15, 30, 60, 120, 300, 600)
LANES = (
    ("ETHUSDT", "BUY", "LONG"),
    ("ETHUSDT", "SELL", "SHORT"),
    ("SOLUSDT", "BUY", "LONG"),
    ("SOLUSDT", "SELL", "SHORT"),
    ("BTCUSDT", "BUY", "LONG"),
    ("BTCUSDT", "SELL", "SHORT"),
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def summ(vals: list[float], cost_bps: float) -> dict[str, Any]:
    vals = [v for v in vals if math.isfinite(v)]
    if not vals:
        return {"n": 0, "gross_median": None, "gross_mean": None, "win_rate": None, "net_median": None}
    return {
        "n": len(vals),
        "gross_median": r1(pctile(vals, 0.5)),
        "gross_mean": r1(mean(vals)),
        "win_rate": r3(sum(1 for v in vals if v > 0) / len(vals)),
        "net_median": r1(pctile(vals, 0.5) - cost_bps),
    }


def find_onsets(liqs: list[dict[str, Any]], gap_sec: int) -> list[dict[str, Any]]:
    """A liq preceded by >= gap_sec of quiet (knowable at the liq)."""
    onsets = []
    prev_ts = None
    gap_ms = int(gap_sec) * 1000
    for row in liqs:  # load_liquidations returns ts ascending
        ts = int(row["ts_ms"])
        if prev_ts is None or (ts - prev_ts) >= gap_ms:
            onsets.append(row)
        prev_ts = ts
    return onsets


def build_flag(liqs_ts: list[int], onset_ts: int, *, build_n: int, build_window_sec: int) -> bool:
    """Forward diagnostic ONLY (not knowable at onset): >= build_n liqs within window."""
    import bisect

    end = onset_ts + int(build_window_sec) * 1000
    lo = bisect.bisect_left(liqs_ts, onset_ts)
    hi = bisect.bisect_right(liqs_ts, end)
    return (hi - lo) >= int(build_n)


def decay_lane(
    conn, symbol, side, direction, *, gap_sec, cost_bps_rt, build_n, build_window_sec, size_split_usd
) -> dict[str, Any]:
    liqs = load_liquidations(conn, symbol, side, None, None)
    marks = load_mark_index(conn, symbol)
    liqs_ts = [int(r["ts_ms"]) for r in liqs]
    onsets = find_onsets(liqs, gap_sec)

    per_h: dict[int, list[float]] = {h: [] for h in HORIZONS_SEC}
    per_h_big: dict[int, list[float]] = {h: [] for h in HORIZONS_SEC}
    per_h_build: dict[int, list[float]] = {h: [] for h in HORIZONS_SEC}
    for row in onsets:
        ts = int(row["ts_ms"])
        entry = marks.at_or_after(ts)
        if not entry:
            continue
        is_big = float(row["notional"]) >= float(size_split_usd)
        is_build = build_flag(liqs_ts, ts, build_n=build_n, build_window_sec=build_window_sec)
        for h in HORIZONS_SEC:
            ex = marks.at_or_after(ts + h * 1000)
            if not ex:
                continue
            ret = signed_return_bps(direction, float(entry[1]), float(ex[1]))
            per_h[h].append(ret)
            if is_big:
                per_h_big[h].append(ret)
            if is_build:
                per_h_build[h].append(ret)
    return {
        "lane": {"symbol": symbol, "side": side, "direction": direction},
        "onset_n": len(onsets),
        "all": {str(h): summ(per_h[h], cost_bps_rt) for h in HORIZONS_SEC},
        "big_onset": {str(h): summ(per_h_big[h], cost_bps_rt) for h in HORIZONS_SEC},
        "builds_forward_diag": {str(h): summ(per_h_build[h], cost_bps_rt) for h in HORIZONS_SEC},
    }


def render_md(report: dict[str, Any]) -> str:
    cfg = report["config"]
    lines = [
        "# S34 Onset Precursor (lookahead-free onset entry)",
        "",
        f"Generated: `{report['generated_at_utc']}`  |  gap>= `{cfg['gap_sec']}`s, cost `{cfg['cost_bps_rt']}`bps, "
        f"big>= `{cfg['size_split_usd']}` USD, build= `{cfg['build_n']}` liqs / `{cfg['build_window_sec']}`s",
        "",
        "Continuation-direction mark return entered at a knowable liquidation onset. `all` = every onset; "
        "`big_onset` = onset liq notional above split (knowable); `builds` = onsets that became a burst "
        "(FORWARD diagnostic, NOT a tradeable condition).",
        "",
    ]
    for lane in report["lanes"]:
        L = lane["lane"]
        lines.append(f"## {L['symbol']} {L['side']} ({L['direction']}) -- onsets={lane['onset_n']}")
        lines.append("")
        lines.append("| Horizon | all N | all GrossMed | all Win% | big GrossMed | big Win% | builds GrossMed | builds Win% |")
        lines.append("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for h in HORIZONS_SEC:
            a = lane["all"][str(h)]
            b = lane["big_onset"][str(h)]
            c = lane["builds_forward_diag"][str(h)]
            wr = lambda s: None if s["win_rate"] is None else r1(s["win_rate"] * 100.0)
            lines.append(
                f"| {h}s | {a['n']} | {a['gross_median']} | {wr(a)} | {b['gross_median']} | {wr(b)} | "
                f"{c['gross_median']} | {wr(c)} |"
            )
        lines.append("")
    return "\n".join(lines)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Lookahead-free liquidation-onset directional decay (precursor probe).")
    p.add_argument("--db", type=Path, default=DEFAULT_DB)
    p.add_argument("--gap-sec", type=int, default=300)
    p.add_argument("--cost-bps-rt", type=float, default=6.1)
    p.add_argument("--size-split-usd", type=float, default=100_000.0)
    p.add_argument("--build-n", type=int, default=5)
    p.add_argument("--build-window-sec", type=int, default=60)
    p.add_argument("--json-out", type=Path, default=OUT_JSON)
    p.add_argument("--md-out", type=Path, default=OUT_MD)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    with sqlite3.connect(f"file:{args.db}?mode=ro", uri=True) as conn:
        lanes = [
            decay_lane(
                conn, sym, side, direction,
                gap_sec=int(args.gap_sec), cost_bps_rt=float(args.cost_bps_rt),
                build_n=int(args.build_n), build_window_sec=int(args.build_window_sec),
                size_split_usd=float(args.size_split_usd),
            )
            for sym, side, direction in LANES
        ]
    report = {
        "generated_at_utc": utc_now(),
        "config": {
            "gap_sec": int(args.gap_sec), "cost_bps_rt": float(args.cost_bps_rt),
            "size_split_usd": float(args.size_split_usd), "build_n": int(args.build_n),
            "build_window_sec": int(args.build_window_sec),
        },
        "lanes": lanes,
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    args.md_out.write_text(render_md(report), encoding="utf-8")
    print(render_md(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
