"""S34 state-machine next-candidate gauntlet.

Research-only. Builds candidate LONG/SHORT/state-sequence variants from the
existing offline dataset and evaluates overall stats, chronological folds,
top-3-removed, and a multiple-comparison permutation null. It does not touch
live executor state, env, orders, sizing, or config.
"""

from __future__ import annotations

import json
import math
import random
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.research_s34_freq_tests import (  # noqa: E402
    DB_PATH,
    FEE_BPS,
    SHORT_HOLD_MS,
    SIL_HI_MS,
    baseline_sync200,
    build_dataset,
    current_long_gate,
    first_liq_above,
    fmt_stat,
    load_liq_series,
    load_mark_series,
    signed_net,
    stat,
    time_exit,
    utc_now,
)


OUT_JSON = ROOT / "reports" / "research" / "s34" / "S34_NEXT_CANDIDATE_GAUNTLET.json"
OUT_MD = ROOT / "reports" / "research" / "s34" / "S34_NEXT_CANDIDATE_GAUNTLET.md"


@dataclass(frozen=True)
class Candidate:
    name: str
    family: str
    rows: list[dict[str, Any]]
    note: str


def finite_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in rows if r.get("net_bps") is not None and math.isfinite(float(r["net_bps"]))]


def t3r_sum(vals: list[float]) -> float:
    if not vals:
        return 0.0
    sv = sorted(vals)
    return float(sum(sv[:-3]) if len(sv) > 3 else sum(sv))


def chronological_folds(rows: list[dict[str, Any]], k: int = 5) -> list[list[dict[str, Any]]]:
    rs = sorted(finite_rows(rows), key=lambda r: int(r.get("entry_ts_ms", r["anchor_ts_ms"])))
    if not rs:
        return [[] for _ in range(k)]
    folds: list[list[dict[str, Any]]] = []
    n = len(rs)
    for i in range(k):
        lo = round(i * n / k)
        hi = round((i + 1) * n / k)
        folds.append(rs[lo:hi])
    return folds


def fold_summary(rows: list[dict[str, Any]], k: int = 5) -> dict[str, Any]:
    folds = chronological_folds(rows, k=k)
    fold_stats = [stat(f) for f in folds]
    positive_sum = sum(1 for s in fold_stats if float(s.get("sum") or 0.0) > 0.0)
    positive_t3r = sum(1 for s in fold_stats if float(s.get("t3r") or 0.0) > 0.0)
    return {
        "folds": fold_stats,
        "positive_sum_folds": positive_sum,
        "positive_t3r_folds": positive_t3r,
        "fold_t3r_sum": round(sum(float(s.get("t3r") or 0.0) for s in fold_stats), 1),
        "worst_fold_sum": round(min((float(s.get("sum") or 0.0) for s in fold_stats), default=0.0), 1),
        "worst_fold_t3r": round(min((float(s.get("t3r") or 0.0) for s in fold_stats), default=0.0), 1),
    }


def score_candidate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    rs = finite_rows(rows)
    s = stat(rs)
    folds = fold_summary(rs, k=5)
    pass_basic = (
        int(s["n"]) >= 10
        and float(s.get("wr") or 0.0) >= 0.70
        and float(s.get("avg") or 0.0) > 0.0
        and float(s.get("t3r") or 0.0) > 0.0
        and int(folds["positive_sum_folds"]) >= 3
        and int(folds["positive_t3r_folds"]) >= 3
    )
    return {
        "summary": s,
        "folds": folds,
        "basic_pass": pass_basic,
        "n_warning": "LOW_N" if int(s["n"]) < 20 else None,
    }


def long_prefilter(row: dict[str, Any]) -> bool:
    return (
        row["close_reason"] == "TIME_EXIT"
        and not row["bull"]
        and float(row["sync_k"]) < 200_000.0
        and row["session"] != "EUROPE"
        and not (row["session"] == "US" and int(row["hour"]) in {13, 14})
        and int(row["long_score"]) >= 3
    )


def long_rows(rows: list[dict[str, Any]], pred: Callable[[dict[str, Any]], bool], name: str = "") -> list[dict[str, Any]]:
    out = []
    for r in rows:
        if pred(r):
            out.append({**r, "entry_ts_ms": int(r["anchor_ts_ms"]), "candidate": name, "side": "LONG"})
    return out


def btc_confirm_rows(
    rows: list[dict[str, Any]],
    *,
    btc_thr: float,
    delay_min: int,
    hold_min: int,
    db_path: Path = DB_PATH,
) -> list[dict[str, Any]]:
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
        btc_sell = load_liq_series(conn, "BTCUSDT", "SELL")
        eth_marks = load_mark_series(conn, "ETHUSDT")
    out: list[dict[str, Any]] = []
    for r in rows:
        if r["bull"] or r["session"] == "EUROPE" or r["dow"] == 6 or int(r["base_score"]) < 4:
            continue
        ts = int(r["anchor_ts_ms"])
        hit = first_liq_above(btc_sell, ts + delay_min * 60_000, ts + SIL_HI_MS, btc_thr)
        if hit is None:
            continue
        net = signed_net(eth_marks, "SHORT", int(hit[0]), int(hit[0]) + hold_min * 60_000)
        if net is None:
            continue
        out.append(
            {
                **r,
                "entry_ts_ms": int(hit[0]),
                "btc_confirm_notional": float(hit[1]),
                "net_bps": float(net),
                "side": "SHORT",
            }
        )
    return out


def with_btc_confirm_flag(rows: list[dict[str, Any]], btc_thr: float = 1_000_000.0, delay_min: int = 0) -> list[dict[str, Any]]:
    with sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True) as conn:
        btc_sell = load_liq_series(conn, "BTCUSDT", "SELL")
    out = []
    for r in rows:
        ts = int(r["anchor_ts_ms"])
        hit = first_liq_above(btc_sell, ts + delay_min * 60_000, ts + SIL_HI_MS, btc_thr)
        out.append({**r, "btc_confirm_ts_ms": None if hit is None else int(hit[0]), "btc_confirm_notional": None if hit is None else float(hit[1])})
    return out


def build_candidates(rows: list[dict[str, Any]]) -> list[Candidate]:
    cands: list[Candidate] = []

    def add_long(name: str, pred: Callable[[dict[str, Any]], bool], note: str) -> None:
        cands.append(Candidate(name, "LONG", long_rows(rows, pred, name), note))

    add_long(
        "L_current_live_gate",
        lambda r: long_prefilter(r)
        and int(r["dow"]) not in {0, 2}
        and r.get("btc7d_bps") is not None
        and float(r["btc7d_bps"]) < 0.0,
        "Current conservative LONG gate.",
    )
    add_long(
        "L_btc7d_lt_500",
        lambda r: long_prefilter(r)
        and int(r["dow"]) not in {0, 2}
        and r.get("btc7d_bps") is not None
        and float(r["btc7d_bps"]) < 500.0,
        "Relax btc7d to +500 bps.",
    )
    add_long(
        "L_no_btc7d",
        lambda r: long_prefilter(r) and int(r["dow"]) not in {0, 2},
        "Remove btc7d regime gate.",
    )
    add_long(
        "L_btc4h_lt0_no_btc7d",
        lambda r: long_prefilter(r) and int(r["dow"]) not in {0, 2} and float(r["btc4h_bps"]) < 0.0,
        "Use btc4h<0 only.",
    )
    add_long(
        "L_btc3d_lt0",
        lambda r: long_prefilter(r)
        and int(r["dow"]) not in {0, 2}
        and r.get("btc3d_bps") is not None
        and float(r["btc3d_bps"]) < 0.0,
        "Use btc3d<0 instead of btc7d.",
    )
    add_long(
        "L_wed_only_block_btc7d0",
        lambda r: long_prefilter(r)
        and int(r["dow"]) != 2
        and r.get("btc7d_bps") is not None
        and float(r["btc7d_bps"]) < 0.0,
        "Remove Monday block; keep Wednesday block.",
    )
    add_long(
        "L_base_score1_added",
        lambda r: r["close_reason"] == "TIME_EXIT"
        and not r["bull"]
        and float(r["sync_k"]) < 200_000.0
        and r["session"] != "EUROPE"
        and not (r["session"] == "US" and int(r["hour"]) in {13, 14})
        and int(r["dow"]) not in {0, 2}
        and r.get("btc7d_bps") is not None
        and float(r["btc7d_bps"]) < 0.0
        and int(r["base_score"]) >= 1,
        "Relax long_score to include base_score1.",
    )
    add_long(
        "L_notional_300_500",
        lambda r: long_prefilter(r) and int(r["dow"]) not in {0, 2} and 300_000.0 <= float(r["running_notional"]) < 500_000.0,
        "Anchor running_notional sweet spot; no btc7d gate.",
    )
    add_long(
        "L_notional_300_500_btc7d0",
        lambda r: long_prefilter(r)
        and int(r["dow"]) not in {0, 2}
        and 300_000.0 <= float(r["running_notional"]) < 500_000.0
        and r.get("btc7d_bps") is not None
        and float(r["btc7d_bps"]) < 0.0,
        "Notional sweet spot with current btc7d gate.",
    )

    # SHORT candidates and hold sweeps.
    for name, thr, delay in [
        ("S_current_btc2m_delay5", 2_000_000.0, 5),
        ("S_btc1m_delay10", 1_000_000.0, 10),
        ("S_btc1m_delay15", 1_000_000.0, 15),
        ("S_btc1m_delay5", 1_000_000.0, 5),
        ("S_btc2m_delay10", 2_000_000.0, 10),
    ]:
        cands.append(Candidate(name, "SHORT", btc_confirm_rows(rows, btc_thr=thr, delay_min=delay, hold_min=120), f"SHORT BTC>={thr:.0f}, delay>={delay}m, hold=2h."))
    for hold_min in [90, 120, 150, 180]:
        cands.append(
            Candidate(
                f"S_current_hold_{hold_min}m",
                "SHORT_HOLD",
                btc_confirm_rows(rows, btc_thr=2_000_000.0, delay_min=5, hold_min=hold_min),
                f"Current SHORT confirm with hold={hold_min}m.",
            )
        )

    # State sequence diagnostics: use LONG 4h outcome to answer what each state does.
    seq_rows = with_btc_confirm_flag([r for r in rows if not r["bull"]], btc_thr=1_000_000.0, delay_min=0)
    seq_defs: list[tuple[str, Callable[[dict[str, Any]], bool], str]] = [
        ("SEQ_silence_no_btc1m", lambda r: r["close_reason"] == "TIME_EXIT" and r["btc_confirm_ts_ms"] is None, "Silence path with no BTC confirm."),
        ("SEQ_silence_with_btc1m", lambda r: r["close_reason"] == "TIME_EXIT" and r["btc_confirm_ts_ms"] is not None, "Silence path but BTC confirm appears."),
        ("SEQ_noisy_no_btc1m", lambda r: r["close_reason"] == "NOISY_EARLY_EXIT" and r["btc_confirm_ts_ms"] is None, "Noisy/follow-on without BTC confirm."),
        ("SEQ_noisy_with_btc1m", lambda r: r["close_reason"] == "NOISY_EARLY_EXIT" and r["btc_confirm_ts_ms"] is not None, "Noisy/follow-on plus BTC confirm."),
    ]
    for name, pred, note in seq_defs:
        cands.append(Candidate(name, "STATE_SEQUENCE_LONG4H", long_rows(seq_rows, pred, name), note))

    return cands


def candidate_index_universe(cands: list[Candidate], family: str) -> tuple[list[dict[str, Any]], list[list[int]]]:
    by_key: dict[tuple[str, int, float], int] = {}
    universe: list[dict[str, Any]] = []
    masks: list[list[int]] = []
    family_cands = [c for c in cands if c.family == family]
    for c in family_cands:
        idxs: list[int] = []
        for r in finite_rows(c.rows):
            key = (str(r.get("side", c.family)), int(r.get("entry_ts_ms", r["anchor_ts_ms"])), float(r["net_bps"]))
            if key not in by_key:
                by_key[key] = len(universe)
                universe.append(r)
            idxs.append(by_key[key])
        masks.append(idxs)
    return universe, masks


def mc_permutation(cands: list[Candidate], family: str, *, n_perm: int = 1000, seed: int = 3401) -> dict[str, float | None]:
    family_cands = [c for c in cands if c.family == family]
    universe, masks = candidate_index_universe(cands, family)
    vals = [float(r["net_bps"]) for r in universe]
    if not vals or not family_cands:
        return {c.name: None for c in family_cands}
    observed = [sum(vals[i] for i in idxs) for idxs in masks]
    rng = random.Random(seed)
    exceed = [0 for _ in family_cands]
    for _ in range(n_perm):
        shuffled = vals[:]
        rng.shuffle(shuffled)
        max_stat = max((sum(shuffled[i] for i in idxs) for idxs in masks), default=0.0)
        for i, obs in enumerate(observed):
            if max_stat >= obs:
                exceed[i] += 1
    return {c.name: round((exceed[i] + 1) / (n_perm + 1), 4) for i, c in enumerate(family_cands)}


def combine_candidates(cands: list[Candidate]) -> list[Candidate]:
    by_name = {c.name: c for c in cands}
    combos: list[Candidate] = []
    combo_defs = [
        ("C_current_live_long_short", "L_current_live_gate", "S_current_btc2m_delay5"),
        ("C_freq_balanced_btc4h_short1m10", "L_btc4h_lt0_no_btc7d", "S_btc1m_delay10"),
        ("C_no_btc7d_short1m10", "L_no_btc7d", "S_btc1m_delay10"),
        ("C_btc7d500_short1m10", "L_btc7d_lt_500", "S_btc1m_delay10"),
        ("C_score_relax_short1m10", "L_base_score1_added", "S_btc1m_delay10"),
    ]
    for name, l_name, s_name in combo_defs:
        if l_name in by_name and s_name in by_name:
            rows = sorted(by_name[l_name].rows + by_name[s_name].rows, key=lambda r: int(r.get("entry_ts_ms", r["anchor_ts_ms"])))
            combos.append(Candidate(name, "COMBINED", rows, f"{l_name} + {s_name}"))
    return combos


def render(results: dict[str, Any]) -> str:
    lines = [
        "# S34 Next Candidate Gauntlet",
        "",
        f"Generated: `{results['generated_at_utc']}`",
        "",
        "Research-only. No live executor, env, order logic, leverage, or sizing was changed.",
        "",
        "## Baselines",
        f"- sync<200K baseline: {fmt_stat(results['baselines']['sync200'])}",
        f"- current LONG gate: {fmt_stat(results['baselines']['current_long'])}",
        "",
        "## Candidate Results",
        "| Candidate | Family | N | WR | Avg | Sum | T3R | /mo | WF +sum | WF +T3R | Worst fold | MC p | Basic | Note |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for item in results["ranked"]:
        s = item["summary"]
        f = item["folds"]
        wr = "" if s["wr"] is None else f"{float(s['wr']) * 100:.1f}%"
        avg = "" if s["avg"] is None else f"{float(s['avg']):+.1f}"
        pval = item.get("mc_p")
        ptxt = "" if pval is None else f"{float(pval):.3f}"
        lines.append(
            f"| {item['name']} | {item['family']} | {s['n']} | {wr} | {avg} | {float(s['sum']):+.1f} | "
            f"{float(s['t3r']):+.1f} | {s['per_month']} | {f['positive_sum_folds']}/5 | {f['positive_t3r_folds']}/5 | "
            f"{float(f['worst_fold_sum']):+.1f} | {ptxt} | {item['basic_pass']} | {item['note']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "- `Basic=True` only means the candidate cleared simple in-sample robustness thresholds; MC p and fold stability still matter.",
            "- `MC p` is max-stat multiple-comparison permutation inside each family, so it is stricter than a single-cell shuffle.",
            "- Low-N candidates are hypotheses, not live promotions.",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    rows, _all_rows = build_dataset()
    cands = build_candidates(rows)
    cands.extend(combine_candidates(cands))

    pvals: dict[str, float | None] = {}
    for family in sorted({c.family for c in cands}):
        pvals.update(mc_permutation(cands, family, n_perm=1000))

    scored = []
    for c in cands:
        sc = score_candidate(c.rows)
        scored.append(
            {
                "name": c.name,
                "family": c.family,
                "note": c.note,
                "mc_p": pvals.get(c.name),
                **sc,
            }
        )
    ranked = sorted(
        scored,
        key=lambda x: (
            bool(x["basic_pass"]),
            float(x["summary"].get("t3r") or 0.0),
            float(x["summary"].get("sum") or 0.0),
            int(x["summary"].get("n") or 0),
        ),
        reverse=True,
    )
    results = {
        "generated_at_utc": utc_now(),
        "dataset": {
            "anchors_200k": len(rows),
            "time_exit_200k": len(time_exit(rows)),
            "candidate_count": len(cands),
        },
        "baselines": {
            "sync200": stat(baseline_sync200(rows)),
            "current_long": stat(current_long_gate(rows)),
        },
        "ranked": ranked,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(results, indent=2, ensure_ascii=True), encoding="utf-8")
    OUT_MD.write_text(render(results), encoding="utf-8")
    print(render(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
