"""S34 Early-Build Velocity Entry.

Onset analysis showed the cascade's directional move is front-loaded into the
first ~30-60s of the burst, while the notional threshold cross arrives 40-92s in
-- after the move is spent. This tests a knowable EARLY trigger that fires on
liquidation *velocity* instead of cumulative notional: enter at the K-th
same-side liquidation within a W-second rolling window (fully knowable at that
K-th liq), well before any large threshold, and measure the forward directional
mark return.

Because this sweeps (K, W), every combo is split chronologically into calibration
(early 70%) and holdout (late 30%); a combo is only interesting if it is net
positive on BOTH. Sweeping inflates false positives, so a both-positive combo is
a lead to register and re-test, not a deployable strategy.
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
from collections import deque
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
OUT_JSON = OUT_DIR / "S34_EARLY_BUILD_ENTRY.json"
OUT_MD = OUT_DIR / "S34_EARLY_BUILD_ENTRY.md"

HORIZONS_SEC = (15, 30, 60, 120)
LANES = (
    ("ETHUSDT", "BUY", "LONG"),
    ("ETHUSDT", "SELL", "SHORT"),
    ("SOLUSDT", "BUY", "LONG"),
    ("SOLUSDT", "SELL", "SHORT"),
    ("BTCUSDT", "BUY", "LONG"),
    ("BTCUSDT", "SELL", "SHORT"),
)
SWEEP_K = (3, 4, 5, 6)
SWEEP_W = (20, 40)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def stat(vals: list[float], cost_bps: float) -> dict[str, Any]:
    vals = [v for v in vals if math.isfinite(v)]
    if not vals:
        return {"n": 0, "gross_median": None, "net_median": None, "win_rate": None, "gross_mean": None}
    med = pctile(vals, 0.5)
    return {
        "n": len(vals),
        "gross_median": r1(med),
        "net_median": r1(med - cost_bps),
        "gross_mean": r1(mean(vals)),
        "win_rate": r3(sum(1 for v in vals if v > 0) / len(vals)),
    }


def triggers_for(liqs: list[dict[str, Any]], *, k: int, w_sec: int, cooldown_sec: int) -> list[int]:
    """Timestamps where the K-th same-side liq within W sec lands (knowable), deduped by cooldown."""
    win: deque[int] = deque()
    w_ms = int(w_sec) * 1000
    cd_ms = int(cooldown_sec) * 1000
    last = None
    out = []
    for row in liqs:
        ts = int(row["ts_ms"])
        win.append(ts)
        while win and win[0] < ts - w_ms:
            win.popleft()
        if len(win) >= int(k) and (last is None or ts - last >= cd_ms):
            out.append(ts)
            last = ts
    return out


def eval_lane(conn, symbol, side, direction, *, cooldown_sec, cost_bps_rt, holdout_frac) -> dict[str, Any]:
    liqs = load_liquidations(conn, symbol, side, None, None)
    marks = load_mark_index(conn, symbol)
    combos = []
    for k in SWEEP_K:
        for w in SWEEP_W:
            trig = triggers_for(liqs, k=k, w_sec=w, cooldown_sec=cooldown_sec)
            if not trig:
                continue
            cut = trig[int(len(trig) * (1.0 - holdout_frac))] if len(trig) > 1 else trig[-1] + 1
            cal_h: dict[int, list[float]] = {h: [] for h in HORIZONS_SEC}
            hold_h: dict[int, list[float]] = {h: [] for h in HORIZONS_SEC}
            for ts in trig:
                entry = marks.at_or_after(ts)
                if not entry:
                    continue
                target = cal_h if ts < cut else hold_h
                for h in HORIZONS_SEC:
                    ex = marks.at_or_after(ts + h * 1000)
                    if not ex:
                        continue
                    target[h].append(signed_return_bps(direction, float(entry[1]), float(ex[1])))
            combos.append({
                "k": k, "w_sec": w, "trigger_n": len(trig),
                "calibration": {str(h): stat(cal_h[h], cost_bps_rt) for h in HORIZONS_SEC},
                "holdout": {str(h): stat(hold_h[h], cost_bps_rt) for h in HORIZONS_SEC},
            })
    return {"lane": {"symbol": symbol, "side": side, "direction": direction}, "combos": combos}


def both_positive(combo: dict[str, Any], horizon: int, min_n: int) -> bool:
    c = combo["calibration"][str(horizon)]
    h = combo["holdout"][str(horizon)]
    return (
        c["n"] >= min_n and h["n"] >= min_n
        and (c["net_median"] or -1) > 0 and (h["net_median"] or -1) > 0
        and (c["win_rate"] or 0) > 0.5 and (h["win_rate"] or 0) > 0.5
    )


def render_md(report: dict[str, Any]) -> str:
    cfg = report["config"]
    lines = [
        "# S34 Early-Build Velocity Entry",
        "",
        f"Generated: `{report['generated_at_utc']}`  |  cooldown `{cfg['cooldown_sec']}`s, cost `{cfg['cost_bps_rt']}`bps, "
        f"holdout `{cfg['holdout_frac']}`, min_n `{cfg['min_n']}`",
        "",
        "Enter at the K-th same-side liquidation within W seconds (knowable). Net = gross median - round-trip cost. "
        "`**` marks combos net-positive AND win>50% on BOTH calibration and holdout at the 60s horizon.",
        "",
    ]
    leads = []
    for lane in report["lanes"]:
        L = lane["lane"]
        lines.append(f"## {L['symbol']} {L['side']} ({L['direction']})")
        lines.append("")
        lines.append("| K | W | Trig N | cal net@30 | cal net@60 | cal win@60 | hold net@30 | hold net@60 | hold win@60 | |")
        lines.append("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
        for c in lane["combos"]:
            c30, c60 = c["calibration"]["30"], c["calibration"]["60"]
            h30, h60 = c["holdout"]["30"], c["holdout"]["60"]
            flag = "**" if both_positive(c, 60, cfg["min_n"]) else ""
            if flag:
                leads.append((L, c))
            wr = lambda s: None if s["win_rate"] is None else r1(s["win_rate"] * 100.0)
            lines.append(
                f"| {c['k']} | {c['w_sec']} | {c['trigger_n']} | {c30['net_median']} | {c60['net_median']} | "
                f"{wr(c60)} | {h30['net_median']} | {h60['net_median']} | {wr(h60)} | {flag} |"
            )
        lines.append("")
    lines.append("## Leads (both-split positive @60s)")
    if leads:
        for L, c in leads:
            lines.append(f"- **{L['symbol']} {L['side']} K={c['k']} W={c['w_sec']}**: "
                         f"cal net@60={c['calibration']['60']['net_median']}, hold net@60={c['holdout']['60']['net_median']}")
    else:
        lines.append("- none")
    lines.append("")
    return "\n".join(lines)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Knowable early-build (liquidation velocity) entry sweep with holdout split.")
    p.add_argument("--db", type=Path, default=DEFAULT_DB)
    p.add_argument("--cooldown-sec", type=int, default=300)
    p.add_argument("--cost-bps-rt", type=float, default=6.1)
    p.add_argument("--holdout-frac", type=float, default=0.30)
    p.add_argument("--min-n", type=int, default=30)
    p.add_argument("--json-out", type=Path, default=OUT_JSON)
    p.add_argument("--md-out", type=Path, default=OUT_MD)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    with sqlite3.connect(f"file:{args.db}?mode=ro", uri=True) as conn:
        lanes = [
            eval_lane(conn, sym, side, direction, cooldown_sec=int(args.cooldown_sec),
                      cost_bps_rt=float(args.cost_bps_rt), holdout_frac=float(args.holdout_frac))
            for sym, side, direction in LANES
        ]
    report = {
        "generated_at_utc": utc_now(),
        "config": {"cooldown_sec": int(args.cooldown_sec), "cost_bps_rt": float(args.cost_bps_rt),
                   "holdout_frac": float(args.holdout_frac), "min_n": int(args.min_n),
                   "sweep_k": list(SWEEP_K), "sweep_w": list(SWEEP_W)},
        "lanes": lanes,
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    args.md_out.write_text(render_md(report), encoding="utf-8")
    print(render_md(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
