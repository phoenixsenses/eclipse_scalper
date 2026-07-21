"""S34 stress-reaction scalp gauntlet.

Research-only validation for the short-horizon reverse/scalp lead surfaced by
the navigation work. This script deliberately does not touch live execution,
order logic, size, leverage, or environment configuration.

Tests:
- fixed candidate cells across chronological walk-forward folds;
- final holdout summaries from fold 4-5 rows;
- max-statistic permutation over the tested candidate family;
- non-overlap grouping to check duplicate-threshold inflation;
- v0.2 navigation guard summaries;
- big winner / loser event cards for the leading candidate.
"""

from __future__ import annotations

import json
import math
import random
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.s34_navigation_full_followup import DEFAULT_DB, r1, r3, summary  # noqa: E402
from tools.s34_navigation_scalp_tail_tests import prepare_rows  # noqa: E402
from tools.s34_stress_reaction_deep_tests import (  # noqa: E402
    BASE_FEE_BPS,
    bracket_outcome,
    fixed_horizon,
    route_v02,
)

OUT_JSON = ROOT / "reports" / "research" / "s34" / "S34_STRESS_REACTION_GAUNTLET.json"
OUT_MD = ROOT / "reports" / "research" / "s34" / "S34_STRESS_REACTION_GAUNTLET.md"

SEED = 34044
PERMUTATIONS = 1000


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ts(row: dict[str, Any]) -> int:
    return int(row.get("signal_ts_ms") or 0)


def t3r(vals: list[float]) -> float:
    vals = [float(v) for v in vals if math.isfinite(float(v))]
    return float(sum(sorted(vals, reverse=True)[3:])) if len(vals) > 3 else float(sum(vals))


def stress3(row: dict[str, Any]) -> bool:
    return int(row.get("stress_score") or 0) >= 3


def btc_lt(row: dict[str, Any], threshold: float) -> bool:
    return float(row.get("btc4h_bps") or 0.0) < threshold


def v_lt(row: dict[str, Any], threshold: float) -> bool:
    return float(row.get("vdepth_bps") or 0.0) < threshold


def near_thresholds(row: dict[str, Any]) -> int:
    return int(row.get("chain_near_15m_thresholds") or 0)


def enrich_chain_counts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=ts)
    out = []
    for row in ordered:
        t = ts(row)
        near = [r for r in ordered if 0 <= abs(ts(r) - t) <= 900_000]
        item = dict(row)
        item["chain_near_15m_n"] = len(near)
        item["chain_near_15m_thresholds"] = len({int(float(r.get("threshold_usd") or 0)) for r in near})
        item["chain_prior_15m_n"] = len([r for r in ordered if 0 < t - ts(r) <= 900_000])
        out.append(item)
    return out


class Candidate:
    def __init__(
        self,
        name: str,
        selector: Callable[[dict[str, Any]], bool],
        *,
        direction: str,
        horizon_sec: int,
        tp: float | None = None,
        sl: float | None = None,
    ) -> None:
        self.name = name
        self.selector = selector
        self.direction = direction
        self.horizon_sec = horizon_sec
        self.tp = tp
        self.sl = sl

    @property
    def exit_name(self) -> str:
        if self.tp is None or self.sl is None:
            return f"fixed_{int(self.horizon_sec / 60)}m"
        return f"tp{self.tp:g}_sl{self.sl:g}_{int(self.horizon_sec / 60)}m"


def build_candidates() -> list[Candidate]:
    return [
        Candidate(
            "S3_BTC75_VLT50_REV_TP200_SL40_20M",
            lambda r: stress3(r) and btc_lt(r, -75.0) and v_lt(r, 50.0),
            direction="REVERSE",
            horizon_sec=1200,
            tp=200.0,
            sl=40.0,
        ),
        Candidate(
            "S3_BTC100_VLT50_REV_TP200_SL40_20M",
            lambda r: stress3(r) and btc_lt(r, -100.0) and v_lt(r, 50.0),
            direction="REVERSE",
            horizon_sec=1200,
            tp=200.0,
            sl=40.0,
        ),
        Candidate(
            "S3_BTC75_CHAIN3_REV_TP150_SL30_15M",
            lambda r: stress3(r) and btc_lt(r, -75.0) and near_thresholds(r) >= 3,
            direction="REVERSE",
            horizon_sec=900,
            tp=150.0,
            sl=30.0,
        ),
        Candidate(
            "S3_BTC75_VLT50_CHAIN3_REV_TP200_SL40_20M",
            lambda r: stress3(r) and btc_lt(r, -75.0) and v_lt(r, 50.0) and near_thresholds(r) >= 3,
            direction="REVERSE",
            horizon_sec=1200,
            tp=200.0,
            sl=40.0,
        ),
        Candidate(
            "S3_BTC75_REV_FIXED15M",
            lambda r: stress3(r) and btc_lt(r, -75.0),
            direction="REVERSE",
            horizon_sec=900,
        ),
        Candidate(
            "S3_BTC75_VLT50_REV_FIXED15M",
            lambda r: stress3(r) and btc_lt(r, -75.0) and v_lt(r, 50.0),
            direction="REVERSE",
            horizon_sec=900,
        ),
        Candidate(
            "V02_TAIL_LOW_NORMAL_FIXED2H",
            lambda r: route_v02(r) and "TAIL_LOW_CONTEXT" in set(r.get("tags") or []),
            direction="NORMAL",
            horizon_sec=7200,
        ),
        Candidate(
            "V02_BID_OK_NORMAL_FIXED2H",
            lambda r: route_v02(r) and "BID_DEPTH_OK" in set(r.get("tags") or []),
            direction="NORMAL",
            horizon_sec=7200,
        ),
        Candidate(
            "V02_TAIL_HIGH_NORMAL_FIXED2H",
            lambda r: route_v02(r) and "TAIL_HIGH_OR_UNKNOWN" in set(r.get("tags") or []),
            direction="NORMAL",
            horizon_sec=7200,
        ),
    ]


def value_for(conn: sqlite3.Connection, row: dict[str, Any], candidate: Candidate) -> tuple[float | None, str]:
    if candidate.tp is None or candidate.sl is None:
        return fixed_horizon(conn, row, candidate.horizon_sec, candidate.direction), "TIME"
    val, exit_, _ = bracket_outcome(
        conn,
        row,
        horizon_sec=candidate.horizon_sec,
        direction=candidate.direction,
        tp=candidate.tp,
        sl=candidate.sl,
        fee_bps=BASE_FEE_BPS,
    )
    return val, exit_


def eval_candidate(conn: sqlite3.Connection, rows: list[dict[str, Any]], candidate: Candidate) -> dict[str, Any]:
    vals: list[float] = []
    exits: dict[str, int] = defaultdict(int)
    matched = 0
    for row in rows:
        if not candidate.selector(row):
            continue
        matched += 1
        val, exit_ = value_for(conn, row, candidate)
        if val is None:
            continue
        vals.append(float(val))
        exits[str(exit_)] += 1
    return {"matched_n": matched, "summary": summary(vals), "exits": dict(exits)}


def by_fold(conn: sqlite3.Connection, rows: list[dict[str, Any]], candidate: Candidate) -> dict[str, Any]:
    out = {}
    for fold in sorted({int(r.get("fold") or 0) for r in rows}):
        fold_rows = [r for r in rows if int(r.get("fold") or 0) == fold]
        out[f"fold_{fold}"] = eval_candidate(conn, fold_rows, candidate)
    sums = [float(v["summary"].get("sum_bps") or 0.0) for v in out.values()]
    t3rs = [float(v["summary"].get("t3r_bps") or 0.0) for v in out.values()]
    return {
        "folds": out,
        "positive_sum_folds": sum(1 for v in sums if v > 0),
        "positive_t3r_folds": sum(1 for v in t3rs if v > 0),
        "fold_sum_total": r1(sum(sums)),
        "fold_t3r_total": r1(sum(t3rs)),
    }


def non_overlap_rows(rows: list[dict[str, Any]], window_sec: int, policy: str) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=ts)
    groups: list[list[dict[str, Any]]] = []
    cur: list[dict[str, Any]] = []
    end = -1
    for row in ordered:
        t = ts(row)
        if not cur or t <= end:
            cur.append(row)
            end = max(end, t + window_sec * 1000)
        else:
            groups.append(cur)
            cur = [row]
            end = t + window_sec * 1000
    if cur:
        groups.append(cur)

    selected = []
    for group in groups:
        if policy == "first":
            selected.append(min(group, key=ts))
        elif policy == "max_threshold":
            selected.append(max(group, key=lambda r: (float(r.get("threshold_usd") or 0.0), -ts(r))))
        elif policy == "min_vdepth":
            selected.append(min(group, key=lambda r: (float(r.get("vdepth_bps") or 0.0), ts(r))))
        else:
            raise ValueError(f"unknown policy {policy}")
    return selected


def non_overlap_eval(conn: sqlite3.Connection, rows: list[dict[str, Any]], candidate: Candidate) -> dict[str, Any]:
    target = [r for r in rows if candidate.selector(r)]
    out: dict[str, Any] = {"overlap": eval_candidate(conn, rows, candidate)}
    for window in (900, 1800, 3600):
        for policy in ("first", "max_threshold", "min_vdepth"):
            selected = non_overlap_rows(target, window, policy)
            # Evaluate all selected rows without re-applying candidate selector.
            vals = []
            exits: dict[str, int] = defaultdict(int)
            for row in selected:
                val, exit_ = value_for(conn, row, candidate)
                if val is not None:
                    vals.append(val)
                    exits[str(exit_)] += 1
            out[f"nonoverlap_{int(window/60)}m_{policy}"] = {
                "matched_n": len(selected),
                "summary": summary(vals),
                "exits": dict(exits),
            }
    return out


def candidate_values(conn: sqlite3.Connection, rows: list[dict[str, Any]], candidate: Candidate) -> tuple[list[int], list[float]]:
    idxs = []
    vals = []
    for idx, row in enumerate(rows):
        if not candidate.selector(row):
            continue
        val, _ = value_for(conn, row, candidate)
        if val is None:
            continue
        idxs.append(idx)
        vals.append(float(val))
    return idxs, vals


def max_stat_permutation(conn: sqlite3.Connection, rows: list[dict[str, Any]], candidates: list[Candidate]) -> dict[str, Any]:
    # Keep one outcome vector per exit family. For each family, shuffle realized
    # outcomes across rows, preserving candidate selectors and N.
    families: dict[str, list[float | None]] = {}
    idxs_by_candidate: dict[str, list[int]] = {}
    real_t3r: dict[str, float] = {}
    for cand in candidates:
        key = f"{cand.direction}_{cand.exit_name}"
        if key not in families:
            series: list[float | None] = []
            for row in rows:
                val, _ = value_for(conn, row, cand)
                series.append(None if val is None else float(val))
            families[key] = series
        idxs, vals = candidate_values(conn, rows, cand)
        idxs_by_candidate[cand.name] = idxs
        real_t3r[cand.name] = t3r(vals)

    rng = random.Random(SEED)
    max_stats = []
    for _ in range(PERMUTATIONS):
        shuffled: dict[str, list[float | None]] = {}
        for key, series in families.items():
            vals = [v for v in series if v is not None]
            rng.shuffle(vals)
            it = iter(vals)
            shuffled[key] = [next(it) if v is not None else None for v in series]
        perm_best = -1e18
        for cand in candidates:
            key = f"{cand.direction}_{cand.exit_name}"
            vals = [shuffled[key][i] for i in idxs_by_candidate[cand.name] if shuffled[key][i] is not None]
            stat = t3r([float(v) for v in vals if v is not None])
            perm_best = max(perm_best, stat)
        max_stats.append(perm_best)

    max_stats_sorted = sorted(max_stats)
    threshold95 = max_stats_sorted[int(0.95 * (len(max_stats_sorted) - 1))]
    out = {"permutations": PERMUTATIONS, "maxstat_95pct_t3r": r1(threshold95), "candidate_p": {}}
    for name, stat in real_t3r.items():
        p = (1 + sum(1 for v in max_stats if v >= stat)) / (PERMUTATIONS + 1)
        out["candidate_p"][name] = {"real_t3r": r1(stat), "mc_p": r3(p)}
    return out


def profile(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def avg(key: str) -> float | None:
        vals = [float(r[key]) for r in rows if r.get(key) is not None and math.isfinite(float(r[key]))]
        return r1(sum(vals) / len(vals)) if vals else None

    tag_counts: dict[str, int] = defaultdict(int)
    for row in rows:
        for tag in row.get("tags") or []:
            tag_counts[str(tag)] += 1
    return {
        "n": len(rows),
        "avg_threshold": avg("threshold_usd"),
        "avg_vdepth": avg("vdepth_bps"),
        "avg_prior4h": avg("prior4h_bps"),
        "avg_eth1h": avg("eth1h_bps"),
        "avg_btc4h": avg("btc4h_bps"),
        "avg_bid_depth": avg("bid_depth_usd"),
        "avg_chain_thresholds": avg("chain_near_15m_thresholds"),
        "tags": dict(sorted(tag_counts.items(), key=lambda kv: kv[1], reverse=True)[:8]),
    }


def event_card(row: dict[str, Any], value: float, exit_: str) -> dict[str, Any]:
    return {
        "event_id": row.get("event_id"),
        "signal_utc": row.get("signal_utc"),
        "fold": row.get("fold"),
        "value_bps": r1(value),
        "exit": exit_,
        "stress_score": row.get("stress_score"),
        "threshold": row.get("threshold_usd"),
        "vdepth": row.get("vdepth_bps"),
        "prior4h": row.get("prior4h_bps"),
        "eth1h": row.get("eth1h_bps"),
        "btc4h": row.get("btc4h_bps"),
        "bid_depth": row.get("bid_depth_usd"),
        "chain_thresholds": row.get("chain_near_15m_thresholds"),
        "tags": row.get("tags"),
    }


def anatomy(conn: sqlite3.Connection, rows: list[dict[str, Any]], candidate: Candidate) -> dict[str, Any]:
    scored = []
    for row in rows:
        if not candidate.selector(row):
            continue
        val, exit_ = value_for(conn, row, candidate)
        if val is not None:
            scored.append((float(val), exit_, row))
    scored.sort(key=lambda x: x[0])
    vals = [v for v, _, _ in scored]
    winners = [r for v, _, r in scored if v > 0]
    losers = [r for v, _, r in scored if v <= -40]
    tails = [r for v, _, r in scored if v <= -100]
    return {
        "summary": summary(vals),
        "winner_profile": profile(winners),
        "loser_profile": profile(losers),
        "tail_profile": profile(tails),
        "worst10": [event_card(r, v, e) for v, e, r in scored[:10]],
        "best10": [event_card(r, v, e) for v, e, r in reversed(scored[-10:])],
    }


def run() -> dict[str, Any]:
    rows = enrich_chain_counts(prepare_rows())
    candidates = build_candidates()
    with sqlite3.connect(DEFAULT_DB) as conn:
        final_hold = [r for r in rows if int(r.get("fold") or 0) >= 4]
        all_results = {}
        for cand in candidates:
            all_results[cand.name] = {
                "exit": cand.exit_name,
                "all": eval_candidate(conn, rows, cand),
                "final_hold_folds_4_5": eval_candidate(conn, final_hold, cand),
                "walkforward": by_fold(conn, rows, cand),
            }

        lead = candidates[0]
        chain_lead = candidates[3]
        result = {
            "generated_at_utc": utc_now(),
            "status": "RESEARCH_ONLY_NO_LIVE_CHANGE",
            "rows_n": len(rows),
            "final_hold_rows_n": len(final_hold),
            "candidates": all_results,
            "maxstat_permutation_final_hold": max_stat_permutation(conn, final_hold, candidates),
            "non_overlap_lead": non_overlap_eval(conn, rows, lead),
            "non_overlap_chain_lead": non_overlap_eval(conn, rows, chain_lead),
            "lead_anatomy_all": anatomy(conn, rows, lead),
            "lead_anatomy_final_hold": anatomy(conn, final_hold, lead),
        }
    return result


def fmt(s: dict[str, Any]) -> str:
    return (
        f"N={s.get('n')} sum={s.get('sum_bps')} med={s.get('median_bps')} "
        f"T3R={s.get('t3r_bps')} tail150={s.get('tail_lte_minus150_n')} maxLoss={s.get('max_loss_bps')}"
    )


def write_report(result: dict[str, Any]) -> None:
    lines = [
        "# S34 Stress Reaction Gauntlet",
        "",
        f"Generated: `{result['generated_at_utc']}`",
        "",
        f"Status: `{result['status']}`",
        "",
        f"Rows: `{result['rows_n']}`; final hold rows (fold 4-5): `{result['final_hold_rows_n']}`",
        "",
        "## Candidate Results",
        "",
        "| Candidate | Exit | All | Final hold | Positive folds | Fold T3R total | Exits final hold |",
        "| --- | --- | --- | --- | ---: | ---: | --- |",
    ]
    for name, row in result["candidates"].items():
        wf = row["walkforward"]
        lines.append(
            f"| `{name}` | `{row['exit']}` | {fmt(row['all']['summary'])} | "
            f"{fmt(row['final_hold_folds_4_5']['summary'])} | "
            f"{wf['positive_t3r_folds']}/5 | {wf['fold_t3r_total']} | "
            f"`{row['final_hold_folds_4_5']['exits']}` |"
        )

    lines.extend(["", "## Max-Statistic Permutation (Final Hold)", ""])
    perm = result["maxstat_permutation_final_hold"]
    lines.append(f"Permutations: `{perm['permutations']}`; 95pct max-stat T3R: `{perm['maxstat_95pct_t3r']}`")
    lines.append("")
    lines.append("| Candidate | Real T3R | MC p |")
    lines.append("| --- | ---: | ---: |")
    for name, row in perm["candidate_p"].items():
        lines.append(f"| `{name}` | {row['real_t3r']} | {row['mc_p']} |")

    for block_name in ("non_overlap_lead", "non_overlap_chain_lead"):
        lines.extend(["", f"## {block_name}", ""])
        lines.append("| Policy | Summary | Exits |")
        lines.append("| --- | --- | --- |")
        for name, row in result[block_name].items():
            lines.append(f"| `{name}` | {fmt(row['summary'])} | `{row.get('exits', {})}` |")

    lines.extend(["", "## Lead Anatomy: All Rows", ""])
    block = result["lead_anatomy_all"]
    lines.append(f"Summary: {fmt(block['summary'])}")
    lines.append(f"Winner profile: `{block['winner_profile']}`")
    lines.append(f"Loser profile: `{block['loser_profile']}`")
    lines.append(f"Tail profile: `{block['tail_profile']}`")
    lines.append("")
    lines.append("Worst 10:")
    for row in block["worst10"]:
        lines.append(f"- `{row}`")
    lines.append("Best 10:")
    for row in block["best10"]:
        lines.append(f"- `{row}`")

    lines.extend(["", "## Lead Anatomy: Final Hold", ""])
    block = result["lead_anatomy_final_hold"]
    lines.append(f"Summary: {fmt(block['summary'])}")
    lines.append(f"Winner profile: `{block['winner_profile']}`")
    lines.append(f"Loser profile: `{block['loser_profile']}`")
    lines.append(f"Tail profile: `{block['tail_profile']}`")
    lines.append("")
    lines.append("Worst 10:")
    for row in block["worst10"]:
        lines.append(f"- `{row}`")
    lines.append("Best 10:")
    for row in block["best10"]:
        lines.append(f"- `{row}`")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    result = run()
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    write_report(result)
    print(OUT_MD.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
