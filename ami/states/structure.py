"""Structure transition matrix — Phase 2 (whitepaper §92).

Estimates the empirical phase-transition graph from history: classify each
4h (or chosen tf) bar into a StructurePhase with StateEngine rules, count
transitions, output probabilistic next-phase map + dual-direction estimates.
"""
from __future__ import annotations
import time
from collections import Counter, defaultdict
from pathlib import Path

from ami.enums import StructurePhase, TIMEFRAME_MS
from ami.states.engine import StateEngine


def estimate_transition_matrix(engine: StateEngine, sym: str = "ETHUSDT", tf: str = "4h",
                               days: int = 120, end_ms: int | None = None) -> dict:
    bar = TIMEFRAME_MS[tf]
    end = end_ms or int(time.time() * 1000)
    start = end - days * 86_400_000
    seq: list[tuple[int, str, str]] = []      # (ts, phase, direction)
    ts = start
    while ts <= end:
        ph, direction, _ = engine._structure(sym, ts, tf)
        seq.append((ts, ph.value, direction))
        ts += bar
    trans: dict[str, Counter] = defaultdict(Counter)
    dwell: Counter = Counter()
    fwd_ret: dict[str, list[float]] = defaultdict(list)
    for i in range(len(seq) - 1):
        a, b = seq[i][1], seq[i + 1][1]
        dwell[a] += 1
        if a != b:
            trans[a][b] += 1
        r = engine._ret_bps(sym, seq[i][0] + bar, bar)
        if r is not None:
            fwd_ret[a].append(r)
    matrix = {}
    for a, ctr in trans.items():
        tot = sum(ctr.values())
        matrix[a] = {b: round(c / tot, 3) for b, c in ctr.most_common(5)}
    directional = {}
    for ph, rets in fwd_ret.items():
        if len(rets) >= 10:
            n = len(rets); up = sum(1 for r in rets if r > 0)
            directional[ph] = {"n": n, "p_up_next": round(up / n, 3),
                               "avg_next_bps": round(sum(rets) / n, 1)}
    return {"tf": tf, "days": days, "n_bars": len(seq),
            "phase_freq": {k: v for k, v in dwell.most_common()},
            "transition_matrix": matrix,
            "dual_direction": directional}
