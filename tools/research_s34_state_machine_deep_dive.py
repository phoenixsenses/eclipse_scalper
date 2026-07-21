"""S34 state-machine candidate deep-dive tests.

Research-only. Takes the next-candidate gauntlet outputs one level deeper:
added-only value, chronological holdout, month stability, fee sensitivity,
tail/drawdown, no-overlap execution, overlap maps, and navigation labels.
No live executor, env, order logic, leverage, sizing, or runtime state is
modified.
"""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.research_s34_freq_tests import baseline_sync200, build_dataset, current_long_gate, fmt_stat, stat, time_exit, utc_now  # noqa: E402
from tools.research_s34_next_gauntlet import Candidate, build_candidates, combine_candidates, finite_rows, score_candidate  # noqa: E402


OUT_JSON = ROOT / "reports" / "research" / "s34" / "S34_STATE_MACHINE_DEEP_DIVE_TESTS.json"
OUT_MD = ROOT / "reports" / "research" / "s34" / "S34_STATE_MACHINE_DEEP_DIVE_TESTS.md"

SELECTED = [
    "C_current_live_long_short",
    "C_score_relax_short1m10",
    "C_no_btc7d_short1m10",
    "C_freq_balanced_btc4h_short1m10",
    "C_btc7d500_short1m10",
    "L_base_score1_added",
    "L_no_btc7d",
    "S_btc1m_delay10",
    "S_current_btc2m_delay5",
    "SEQ_silence_no_btc1m",
    "SEQ_noisy_with_btc1m",
]


def key(row: dict[str, Any]) -> tuple[str, int, int]:
    return (str(row.get("side", "")), int(row.get("anchor_ts_ms", 0)), int(row.get("entry_ts_ms", row.get("anchor_ts_ms", 0))))


def entry_ts(row: dict[str, Any]) -> int:
    return int(row.get("entry_ts_ms", row.get("anchor_ts_ms", 0)))


def exit_ts(row: dict[str, Any]) -> int:
    hold_ms = 4 * 3600_000 if str(row.get("side", "LONG")).upper() == "LONG" else 2 * 3600_000
    return entry_ts(row) + hold_ms


def month_key(row: dict[str, Any]) -> str:
    return datetime.fromtimestamp(entry_ts(row) / 1000.0, tz=timezone.utc).strftime("%Y-%m")


def chronological_split(rows: list[dict[str, Any]], frac: float = 0.70) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rs = sorted(finite_rows(rows), key=entry_ts)
    cut = int(round(len(rs) * frac))
    return rs[:cut], rs[cut:]


def fee_adjust(rows: list[dict[str, Any]], target_fee_bps: float, base_fee_bps: float = 5.0) -> list[dict[str, Any]]:
    # Existing net_bps already includes base fee. Move to target fee by subtracting the delta.
    delta = float(target_fee_bps) - float(base_fee_bps)
    return [{**r, "net_bps": float(r["net_bps"]) - delta} for r in finite_rows(rows)]


def max_drawdown(vals: list[float]) -> float:
    cum = 0.0
    peak = 0.0
    max_dd = 0.0
    for v in vals:
        cum += float(v)
        peak = max(peak, cum)
        max_dd = min(max_dd, cum - peak)
    return round(max_dd, 1)


def max_consecutive_losses(vals: list[float]) -> int:
    best = 0
    cur = 0
    for v in vals:
        if float(v) <= 0.0:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def risk_profile(rows: list[dict[str, Any]]) -> dict[str, Any]:
    rs = sorted(finite_rows(rows), key=entry_ts)
    vals = [float(r["net_bps"]) for r in rs]
    if not vals:
        return {"n": 0}
    return {
        "n": len(vals),
        "worst": round(min(vals), 1),
        "best": round(max(vals), 1),
        "tail_loss_100bps_n": sum(1 for v in vals if v <= -100.0),
        "tail_loss_100bps_rate": round(sum(1 for v in vals if v <= -100.0) / len(vals), 3),
        "max_drawdown_bps": max_drawdown(vals),
        "max_consecutive_losses": max_consecutive_losses(vals),
        "top3_removed_sum": stat(rs)["t3r"],
    }


def no_overlap(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    active_until = -1
    for r in sorted(finite_rows(rows), key=entry_ts):
        if entry_ts(r) < active_until:
            continue
        kept.append(r)
        active_until = exit_ts(r)
    return kept


def overlap_profile(rows: list[dict[str, Any]]) -> dict[str, Any]:
    rs = sorted(finite_rows(rows), key=entry_ts)
    points: list[tuple[int, int]] = []
    for r in rs:
        points.append((entry_ts(r), 1))
        points.append((exit_ts(r), -1))
    active = 0
    max_active = 0
    for _, delta in sorted(points):
        active += delta
        max_active = max(max_active, active)
    kept = no_overlap(rs)
    skipped = len(rs) - len(kept)
    return {
        "max_simultaneous": max_active,
        "no_overlap_n": len(kept),
        "skipped_by_no_overlap": skipped,
        "no_overlap_stats": stat(kept),
    }


def monthly(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in finite_rows(rows):
        groups[month_key(r)].append(r)
    return {m: stat(v) for m, v in sorted(groups.items())}


def selected_candidates(cands: list[Candidate]) -> dict[str, Candidate]:
    by = {c.name: c for c in cands}
    return {name: by[name] for name in SELECTED if name in by}


def added_only(candidate: Candidate, base: Candidate) -> list[dict[str, Any]]:
    base_keys = {key(r) for r in finite_rows(base.rows)}
    return [r for r in finite_rows(candidate.rows) if key(r) not in base_keys]


def overlap_matrix(cands: dict[str, Candidate]) -> dict[str, dict[str, Any]]:
    names = list(cands)
    out: dict[str, dict[str, Any]] = {}
    keysets = {n: {key(r) for r in finite_rows(c.rows)} for n, c in cands.items()}
    for a in names:
        out[a] = {}
        for b in names:
            if a == b:
                continue
            inter = keysets[a] & keysets[b]
            denom = max(1, len(keysets[a]))
            out[a][b] = {"overlap_n": len(inter), "share_of_a": round(len(inter) / denom, 3)}
    return out


def live_readiness(score: dict[str, Any], risk: dict[str, Any], hold: dict[str, Any], noov: dict[str, Any]) -> str:
    s = score["summary"]
    if int(s.get("n") or 0) < 20:
        return "RESEARCH_ONLY_LOW_N"
    if float(s.get("t3r") or 0.0) <= 0.0:
        return "REJECT_T3R"
    if int(score["folds"]["positive_t3r_folds"]) < 3:
        return "REJECT_FOLD_T3R"
    if float(hold["holdout"].get("sum") or 0.0) <= 0.0 or float(hold["holdout"].get("t3r") or 0.0) <= 0.0:
        return "RESEARCH_ONLY_HOLDOUT_WEAK"
    if int(risk.get("tail_loss_100bps_n") or 0) > 0:
        return "PAPER_SHADOW_ONLY_TAIL"
    if int(noov.get("skipped_by_no_overlap") or 0) > 0 and float(noov["no_overlap_stats"].get("t3r") or 0.0) <= 0.0:
        return "REJECT_NO_OVERLAP"
    return "PAPER_CANDIDATE"


def evaluate_candidate(c: Candidate, current_combo: Candidate) -> dict[str, Any]:
    cal, hold = chronological_split(c.rows, frac=0.70)
    sc = score_candidate(c.rows)
    rp = risk_profile(c.rows)
    noov = overlap_profile(c.rows)
    hold_stats = {"calibration": stat(cal), "holdout": stat(hold)}
    fee = {f"fee_{fee:g}bps": stat(fee_adjust(c.rows, fee)) for fee in [3.0, 5.0, 8.0, 10.0, 15.0]}
    added = added_only(c, current_combo)
    return {
        "name": c.name,
        "family": c.family,
        "note": c.note,
        "score": sc,
        "added_vs_current": stat(added),
        "added_vs_current_risk": risk_profile(added),
        "holdout_70_30": hold_stats,
        "monthly": monthly(c.rows),
        "fee_sensitivity": fee,
        "risk_profile": rp,
        "overlap_profile": noov,
        "readiness": live_readiness(sc, rp, hold_stats, noov),
    }


def navigation_tests(rows: list[dict[str, Any]], cands: dict[str, Candidate]) -> dict[str, Any]:
    baseline = baseline_sync200(rows)
    out: dict[str, Any] = {}
    out["baseline_sync200"] = stat(baseline)
    out["danger_noisy_with_btc1m_long4h"] = stat(cands["SEQ_noisy_with_btc1m"].rows) if "SEQ_noisy_with_btc1m" in cands else {}
    out["safe_silence_no_btc1m_long4h"] = stat(cands["SEQ_silence_no_btc1m"].rows) if "SEQ_silence_no_btc1m" in cands else {}
    if "SEQ_noisy_with_btc1m" in cands:
        danger_keys = {int(r["anchor_ts_ms"]) for r in finite_rows(cands["SEQ_noisy_with_btc1m"].rows)}
        out["baseline_excluding_danger_noisy_btc1m"] = stat([r for r in baseline if int(r["anchor_ts_ms"]) not in danger_keys])
    return out


def render(results: dict[str, Any]) -> str:
    lines = [
        "# S34 State Machine Deep-Dive Tests",
        "",
        f"Generated: `{results['generated_at_utc']}`",
        "",
        "Research-only. No live executor, env, order logic, leverage, sizing, or runtime state was changed.",
        "",
        "## Ideas Tested",
    ]
    for i, idea in enumerate(results["ideas"], 1):
        lines.append(f"{i}. {idea}")
    lines.extend(
        [
            "",
            "## Candidate Summary",
            "| Candidate | N | WR | Avg | T3R | Added N | Added Avg | Holdout Sum | Holdout T3R | Worst | TailN | NoOverlap N | Readiness |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for item in results["candidates"]:
        s = item["score"]["summary"]
        added = item["added_vs_current"]
        hold = item["holdout_70_30"]["holdout"]
        risk = item["risk_profile"]
        noov = item["overlap_profile"]
        wr = "" if s["wr"] is None else f"{float(s['wr']) * 100:.1f}%"
        avg = "" if s["avg"] is None else f"{float(s['avg']):+.1f}"
        addavg = "" if added["avg"] is None else f"{float(added['avg']):+.1f}"
        lines.append(
            f"| {item['name']} | {s['n']} | {wr} | {avg} | {float(s['t3r']):+.1f} | "
            f"{added['n']} | {addavg} | {float(hold['sum']):+.1f} | {float(hold['t3r']):+.1f} | "
            f"{float(risk.get('worst', 0.0)):+.1f} | {risk.get('tail_loss_100bps_n', 0)} | "
            f"{noov['no_overlap_n']} | {item['readiness']} |"
        )
    lines.extend(["", "## Navigation Tests"])
    for name, s in results["navigation"].items():
        if isinstance(s, dict) and "n" in s:
            lines.append(f"- {name}: {fmt_stat(s)} | T3R={float(s.get('t3r') or 0.0):+.1f}")
    lines.extend(["", "## Key Interpretation", *[f"- {x}" for x in results["interpretation"]]])
    return "\n".join(lines)


def main() -> int:
    rows, _all_rows = build_dataset()
    cands = build_candidates(rows)
    cands.extend(combine_candidates(cands))
    chosen = selected_candidates(cands)
    current_combo = chosen["C_current_live_long_short"]
    candidate_results = [evaluate_candidate(c, current_combo) for c in chosen.values()]
    nav = navigation_tests(rows, chosen)
    results = {
        "generated_at_utc": utc_now(),
        "dataset": {
            "anchors_200k": len(rows),
            "time_exit_200k": len(time_exit(rows)),
            "selected_candidate_count": len(chosen),
        },
        "ideas": [
            "Added-only value: each relaxed candidate's incremental trades vs current live combo.",
            "70/30 chronological holdout: calibration versus latest holdout behavior.",
            "Month stability: whether a candidate is one-regime/month dependent.",
            "Fee sensitivity: robustness if total costs rise from 5 bps to 8/10/15 bps.",
            "Tail and drawdown: worst loss, -100 bps tail count, max drawdown, losing streaks.",
            "No-overlap execution: one-position-at-a-time simulation, because live cannot hold infinite overlapping trades.",
            "Candidate overlap map: whether new candidates add independent events or mostly relabel current trades.",
            "State navigation: silence/noisy + BTC confirm as OK/DANGER context, not an entry alpha.",
            "Current-vs-relaxed delta: whether frequency comes from genuinely positive added trades.",
            "Live-readiness score: reject/paper/research-only classification from the above checks.",
        ],
        "baselines": {
            "sync200": stat(baseline_sync200(rows)),
            "current_long_only": stat(current_long_gate(rows)),
            "current_combo": stat(current_combo.rows),
        },
        "candidates": candidate_results,
        "overlap_matrix": overlap_matrix(chosen),
        "navigation": nav,
        "interpretation": [
            "Best raw candidate remains C_score_relax_short1m10, but it is still PAPER/SHADOW level because N is modest and it has a -100 bps tail.",
            "C_no_btc7d_short1m10 gets closest to 8 trades/month, but its added trades are materially weaker than the stricter score-relax candidate.",
            "Noisy+BTC-confirm is a strong DANGER navigation state for LONG, while silence without BTC confirm is the cleanest broad state label.",
            "No-overlap matters: frequency candidates must be evaluated as executable portfolios, not independent rows.",
            "No live changes were made; next step is forward-shadowing the leading candidates before promotion.",
        ],
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(results, indent=2, ensure_ascii=True), encoding="utf-8")
    OUT_MD.write_text(render(results), encoding="utf-8")
    print(render(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
