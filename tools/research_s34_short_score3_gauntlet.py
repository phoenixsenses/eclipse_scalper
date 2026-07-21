"""S34 SHORT score>=3 gauntlet.

Research-only. Tests relaxed SHORT score thresholds against BTC confirm and
hold-time variants on a narrow executable universe. No live files, env, order
logic, leverage, or sizing are modified.
"""

from __future__ import annotations

import json
import math
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.research_s34_freq_tests import (  # noqa: E402
    DB_PATH,
    SIL_HI_MS,
    build_dataset,
    first_liq_above,
    fmt_stat,
    load_liq_series,
    load_mark_series,
    signed_net,
    stat,
    utc_now,
)
from tools.research_s34_next_gauntlet import chronological_folds, finite_rows  # noqa: E402


OUT_JSON = ROOT / "reports" / "research" / "s34" / "S34_SHORT_SCORE3_GAUNTLET.json"
OUT_MD = ROOT / "reports" / "research" / "s34" / "S34_SHORT_SCORE3_GAUNTLET.md"


@dataclass(frozen=True)
class ShortCandidate:
    name: str
    score_min: int
    btc_thr: float
    delay_min: int
    hold_min: int
    rows: list[dict[str, Any]]


def short_rows(
    rows: list[dict[str, Any]],
    *,
    score_min: int,
    btc_thr: float,
    delay_min: int,
    hold_min: int,
) -> list[dict[str, Any]]:
    with sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True) as conn:
        btc_sell = load_liq_series(conn, "BTCUSDT", "SELL")
        eth_marks = load_mark_series(conn, "ETHUSDT")
    out: list[dict[str, Any]] = []
    for r in rows:
        if r["bull"] or r["session"] == "EUROPE" or r["dow"] == 6 or int(r["base_score"]) < int(score_min):
            continue
        ts = int(r["anchor_ts_ms"])
        hit = first_liq_above(btc_sell, ts + int(delay_min) * 60_000, ts + SIL_HI_MS, float(btc_thr))
        if hit is None:
            continue
        net = signed_net(eth_marks, "SHORT", int(hit[0]), int(hit[0]) + int(hold_min) * 60_000)
        if net is None or not math.isfinite(float(net)):
            continue
        out.append(
            {
                **r,
                "side": "SHORT",
                "entry_ts_ms": int(hit[0]),
                "btc_confirm_notional": float(hit[1]),
                "net_bps": float(net),
            }
        )
    return out


def split_70_30(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rs = sorted(finite_rows(rows), key=lambda r: int(r.get("entry_ts_ms", r["anchor_ts_ms"])))
    cut = int(round(len(rs) * 0.70))
    return rs[:cut], rs[cut:]


def fold_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    folds = chronological_folds(rows, 5)
    stats = [stat(f) for f in folds]
    return {
        "folds": stats,
        "positive_sum_folds": sum(1 for s in stats if float(s.get("sum") or 0.0) > 0.0),
        "positive_t3r_folds": sum(1 for s in stats if float(s.get("t3r") or 0.0) > 0.0),
        "worst_fold_sum": round(min((float(s.get("sum") or 0.0) for s in stats), default=0.0), 1),
    }


def risk(rows: list[dict[str, Any]]) -> dict[str, Any]:
    vals = [float(r["net_bps"]) for r in finite_rows(rows)]
    if not vals:
        return {"worst": None, "tail100_n": 0}
    return {
        "worst": round(min(vals), 1),
        "tail100_n": sum(1 for v in vals if v <= -100.0),
        "tail100_rate": round(sum(1 for v in vals if v <= -100.0) / len(vals), 3),
    }


def readiness(summary: dict[str, Any], holdout: dict[str, Any], folds: dict[str, Any], risk_block: dict[str, Any]) -> str:
    if int(summary.get("n") or 0) < 10:
        return "LOW_N_RESEARCH_ONLY"
    if float(summary.get("t3r") or 0.0) <= 0.0:
        return "REJECT_T3R"
    if float(holdout.get("sum") or 0.0) <= 0.0 or float(holdout.get("t3r") or 0.0) <= 0.0:
        return "RESEARCH_ONLY_HOLDOUT_WEAK"
    if int(folds.get("positive_sum_folds") or 0) < 3:
        return "REJECT_FOLD_STABILITY"
    if int(risk_block.get("tail100_n") or 0) > 0:
        return "SHADOW_ONLY_TAIL"
    return "PAPER_CANDIDATE"


def evaluate(c: ShortCandidate) -> dict[str, Any]:
    cal, ho = split_70_30(c.rows)
    s = stat(c.rows)
    h = stat(ho)
    f = fold_stats(c.rows)
    r = risk(c.rows)
    return {
        "name": c.name,
        "score_min": c.score_min,
        "btc_thr": c.btc_thr,
        "delay_min": c.delay_min,
        "hold_min": c.hold_min,
        "summary": s,
        "calibration": stat(cal),
        "holdout": h,
        "folds": f,
        "risk": r,
        "readiness": readiness(s, h, f, r),
    }


def render(results: dict[str, Any]) -> str:
    lines = [
        "# S34 SHORT Score>=3 Gauntlet",
        "",
        f"Generated: `{results['generated_at_utc']}`",
        "",
        "Research-only. No live executor, env, order logic, leverage, or sizing was changed.",
        "",
        "| Candidate | N | WR | Avg | T3R | Holdout N | Holdout Avg | Holdout T3R | Folds +sum | Worst | TailN | Readiness |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in results["ranked"]:
        s = row["summary"]
        h = row["holdout"]
        f = row["folds"]
        r = row["risk"]
        wr = "" if s["wr"] is None else f"{float(s['wr']) * 100:.1f}%"
        avg = "" if s["avg"] is None else f"{float(s['avg']):+.1f}"
        havg = "" if h["avg"] is None else f"{float(h['avg']):+.1f}"
        lines.append(
            f"| {row['name']} | {s['n']} | {wr} | {avg} | {float(s['t3r']):+.1f} | "
            f"{h['n']} | {havg} | {float(h['t3r']):+.1f} | {f['positive_sum_folds']}/5 | "
            f"{r['worst']} | {r['tail100_n']} | {row['readiness']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "- score>=3 expands frequency, but live promotion needs holdout + tail stability.",
            "- 4h hold is included here as research; live hold-time changes still require explicit operator sign-off.",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    rows, _ = build_dataset()
    candidates: list[ShortCandidate] = []
    for score_min in [3, 4]:
        for btc_thr in [1_000_000.0, 2_000_000.0]:
            for delay_min in [5, 10]:
                for hold_min in [120, 180, 240]:
                    name = f"S_score{score_min}_btc{int(btc_thr/1_000_000)}m_delay{delay_min}_hold{hold_min}m"
                    candidates.append(
                        ShortCandidate(
                            name=name,
                            score_min=score_min,
                            btc_thr=btc_thr,
                            delay_min=delay_min,
                            hold_min=hold_min,
                            rows=short_rows(rows, score_min=score_min, btc_thr=btc_thr, delay_min=delay_min, hold_min=hold_min),
                        )
                    )
    evaluated = [evaluate(c) for c in candidates]
    ranked = sorted(
        evaluated,
        key=lambda r: (
            r["readiness"] == "PAPER_CANDIDATE",
            float(r["summary"].get("t3r") or 0.0),
            float(r["summary"].get("sum") or 0.0),
            int(r["summary"].get("n") or 0),
        ),
        reverse=True,
    )
    results = {
        "generated_at_utc": utc_now(),
        "dataset": {"anchors_200k": len(rows), "candidate_count": len(candidates)},
        "ranked": ranked,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(results, indent=2, ensure_ascii=True), encoding="utf-8")
    OUT_MD.write_text(render(results), encoding="utf-8")
    print(render(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
