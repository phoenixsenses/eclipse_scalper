"""S34 state-machine v8 robustness suite.

Research-only. No live executor, env, runtime state, orders, buckets, or
dashboard changes. This suite stress-tests the best V7 development leads before
any live consideration: profit-lock, confidence sizing, latency, fees, folds,
conflict policy, score/BTC grids, and early management.
"""

from __future__ import annotations

import bisect
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.research_s34_state_machine_v2_gauntlet import (  # noqa: E402
    FEE_BPS,
    Config,
    apply_conflict_policy,
    build_signals,
    fold_summaries,
    mark_at_or_after,
    summary_with_dd,
)
from tools.research_s34_state_machine_v4_promotion_gauntlet import build_base_rows  # noqa: E402
from tools.research_s34_state_machine_v6_development_ideas import (  # noqa: E402
    FINAL_CFG,
    confidence_sizing,
    mfe_mae_for_signal,
    split,
)


OUT_JSON = ROOT / "reports" / "research" / "s34" / "S34_STATE_MACHINE_V8_ROBUSTNESS_SUITE.json"
OUT_MD = ROOT / "reports" / "research" / "s34" / "S34_STATE_MACHINE_V8_ROBUSTNESS_SUITE.md"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def stat_line(s: dict[str, Any]) -> str:
    wr = s.get("wr")
    wrs = "NA" if wr is None else f"{float(wr) * 100:.1f}%"
    return (
        f"N={s.get('n')} WR={wrs} sum={s.get('sum')} mean={s.get('mean')} "
        f"med={s.get('median')} T3R={s.get('t3r')} maxL={s.get('max_loss')} DD={s.get('max_dd_bps')}"
    )


def hold_ms(side: str) -> int:
    return (4 if side.upper() == "LONG" else 2) * 3600_000


def with_cost(signals: list[dict[str, Any]], extra_bps: float) -> list[dict[str, Any]]:
    return [{**s, "net_bps": round(float(s["net_bps"]) - extra_bps, 1)} for s in signals]


def top_removed(signals: list[dict[str, Any]], n: int = 5) -> dict[str, Any]:
    rows = sorted(signals, key=lambda s: float(s["net_bps"]), reverse=True)[n:]
    return summary_with_dd(rows)


def delayed_entry(signals: list[dict[str, Any]], mk_ts: list[int], mk_px: list[float], delay_sec: int) -> list[dict[str, Any]]:
    rows = []
    for s in signals:
        entry_ts = int(s["entry_ts_ms"]) + delay_sec * 1000
        entry = mark_at_or_after(mk_ts, mk_px, entry_ts)
        exit_px = mark_at_or_after(mk_ts, mk_px, entry_ts + hold_ms(str(s["side"])))
        if not entry or not exit_px or entry <= 0:
            continue
        raw = (exit_px - entry) / entry * 10_000.0
        net = -raw if str(s["side"]).upper() == "SHORT" else raw
        rows.append({**s, "entry_ts_ms": entry_ts, "net_bps": round(net - FEE_BPS, 1), "entry_delay_sec": delay_sec})
    return rows


def profit_lock(
    signals: list[dict[str, Any]],
    mk_ts: list[int],
    mk_px: list[float],
    *,
    trigger: float,
    lock: float,
    side_filter: str | None = None,
    extra_exit_cost_bps: float = 0.0,
) -> list[dict[str, Any]]:
    managed = []
    for s in signals:
        side = str(s["side"]).upper()
        if side_filter and side != side_filter:
            managed.append(dict(s))
            continue
        entry_ts = int(s["entry_ts_ms"])
        entry = mark_at_or_after(mk_ts, mk_px, entry_ts)
        if not entry:
            continue
        a = bisect.bisect_left(mk_ts, entry_ts)
        b = bisect.bisect_right(mk_ts, entry_ts + hold_ms(side))
        armed = False
        net = float(s["net_bps"])
        for i in range(a, b):
            raw = (float(mk_px[i]) - entry) / entry * 10_000.0
            pnl = -raw if side == "SHORT" else raw
            if not armed and pnl >= trigger:
                armed = True
            if armed and pnl <= lock:
                net = float(lock) - FEE_BPS - extra_exit_cost_bps
                break
        managed.append({**s, "net_bps": round(net, 1), "profit_lock_triggered": armed})
    return managed


def early_cut(
    signals: list[dict[str, Any]],
    mk_ts: list[int],
    mk_px: list[float],
    *,
    minute: int,
    min_mfe: float | None = None,
    max_mae: float | None = None,
) -> list[dict[str, Any]]:
    out = []
    for s in signals:
        mm = mfe_mae_for_signal(s, mk_ts, mk_px, minute * 60_000)
        if not mm:
            out.append(dict(s))
            continue
        should_cut = False
        if min_mfe is not None and float(mm["mfe_bps"]) < min_mfe:
            should_cut = True
        if max_mae is not None and float(mm["mae_bps"]) <= max_mae:
            should_cut = True
        if not should_cut:
            out.append(dict(s))
            continue
        entry_ts = int(s["entry_ts_ms"])
        entry = mark_at_or_after(mk_ts, mk_px, entry_ts)
        exit_px = mark_at_or_after(mk_ts, mk_px, entry_ts + minute * 60_000)
        if not entry or not exit_px or entry <= 0:
            out.append(dict(s))
            continue
        raw = (exit_px - entry) / entry * 10_000.0
        net = -raw if str(s["side"]).upper() == "SHORT" else raw
        out.append({**s, "net_bps": round(net - FEE_BPS, 1), "early_cut": True})
    return out


def partial_scaleout(signals: list[dict[str, Any]], mk_ts: list[int], mk_px: list[float], trigger: float, fraction: float) -> list[dict[str, Any]]:
    out = []
    for s in signals:
        side = str(s["side"]).upper()
        entry_ts = int(s["entry_ts_ms"])
        entry = mark_at_or_after(mk_ts, mk_px, entry_ts)
        if not entry:
            continue
        a = bisect.bisect_left(mk_ts, entry_ts)
        b = bisect.bisect_right(mk_ts, entry_ts + hold_ms(side))
        hit = False
        for i in range(a, b):
            raw = (float(mk_px[i]) - entry) / entry * 10_000.0
            pnl = -raw if side == "SHORT" else raw
            if pnl >= trigger:
                hit = True
                break
        net = float(s["net_bps"])
        if hit:
            net = fraction * (trigger - FEE_BPS) + (1.0 - fraction) * net
        out.append({**s, "net_bps": round(net, 1), "scaleout_hit": hit})
    return out


def candidate_pack(signals: list[dict[str, Any]], mk_ts: list[int], mk_px: list[float]) -> dict[str, list[dict[str, Any]]]:
    conf = confidence_sizing(signals, mk_ts, mk_px)["sized_counterfactual"]["all"]
    # Rebuild confidence-sized rows, because the imported helper returns summaries.
    conf_rows = []
    for s in signals:
        mm = mfe_mae_for_signal(s, mk_ts, mk_px, 5 * 60_000)
        early_ok = bool(mm and mm["mfe_bps"] >= 20)
        score = int(s.get("score") or 0)
        c = 0
        c += int(score >= 4)
        c += int(early_ok)
        c += int(s["arm"] == "SILENCE_LONG" and s["row"].get("vd", 0) >= 30)
        c += int(s["arm"] == "NEITHER_SHORT" and (int(s["entry_ts_ms"]) - int(s["anchor_ts_ms"])) <= 15 * 60_000)
        c += int(abs(float(s["row"].get("b4h") or 0)) >= 50)
        mult = 0.5 if c <= 1 else 1.0 if c == 2 else 1.25 if c == 3 else 1.5
        conf_rows.append({**s, "net_bps": round(float(s["net_bps"]) * mult, 1), "confidence": c, "size_mult": mult})
    return {
        "baseline": signals,
        "profit_lock_100_50": profit_lock(signals, mk_ts, mk_px, trigger=100, lock=50),
        "profit_lock_long_only": profit_lock(signals, mk_ts, mk_px, trigger=100, lock=50, side_filter="LONG"),
        "profit_lock_short_only": profit_lock(signals, mk_ts, mk_px, trigger=100, lock=50, side_filter="SHORT"),
        "confidence_sized": conf_rows,
        "early_cut_5m_weak20": early_cut(signals, mk_ts, mk_px, minute=5, min_mfe=20),
        "early_cut_5m_adverse20": early_cut(signals, mk_ts, mk_px, minute=5, max_mae=-20),
        "early_cut_5m_weak20_or_adverse20": early_cut(signals, mk_ts, mk_px, minute=5, min_mfe=20, max_mae=-20),
        "scaleout_100_half": partial_scaleout(signals, mk_ts, mk_px, trigger=100, fraction=0.5),
        "scaleout_150_half": partial_scaleout(signals, mk_ts, mk_px, trigger=150, fraction=0.5),
    }


def summarize_candidates(cands: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    out = {}
    for name, rows in cands.items():
        out[name] = {
            "split": split(rows),
            "folds": fold_summaries(rows, folds=5),
            "top5_removed_all": top_removed(rows, 5),
            "top3_removed_hold": top_removed([s for s in rows if s["row"]["is_hold"]], 3),
        }
    return out


def fee_stress(cands: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    out = {}
    for name, rows in cands.items():
        out[name] = {f"extra_{extra}bps": split(with_cost(rows, extra)) for extra in [2, 5, 10, 20]}
    return out


def latency_stress(signals: list[dict[str, Any]], mk_ts: list[int], mk_px: list[float]) -> dict[str, Any]:
    return {f"delay_{sec}s": split(delayed_entry(signals, mk_ts, mk_px, sec)) for sec in [1, 2, 5, 10, 15, 30, 60]}


def conflict_policy_grid(raw: list[dict[str, Any]]) -> dict[str, Any]:
    out = {}
    for policy in ["one_pos_ignore", "short_replace", "all_independent"]:
        taken, blocked = apply_conflict_policy(raw, policy)
        out[policy] = {"taken": split(taken), "blocked": split(blocked), "blocked_n": len(blocked)}
    return out


def config_grid(rows: list[dict[str, Any]], mk_ts: list[int], mk_px: list[float]) -> dict[str, Any]:
    out = {}
    for btc in [750_000, 1_000_000, 1_250_000, 1_500_000]:
        for lscore in [3, 4, 5]:
            for sscore in [3, 4, 5]:
                cfg = Config(
                    f"btc{int(btc/1000)}_l{lscore}_s{sscore}",
                    btc_thr=btc,
                    long_score_min=lscore,
                    short_score_min=sscore,
                    exclude_long_dow=(0, 2),
                    exclude_short_dow=(6,),
                )
                sigs = apply_conflict_policy(build_signals(rows, cfg, mk_ts=mk_ts, mk_px=mk_px), "short_replace")[0]
                out[cfg.name] = {
                    "summary": split(sigs),
                    "long": split([s for s in sigs if s["side"] == "LONG"]),
                    "short": split([s for s in sigs if s["side"] == "SHORT"]),
                }
    return out


def rank_config_grid(grid: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for name, val in grid.items():
        h = val["summary"]["hold"]
        rows.append({"name": name, **h, "small_n": int(h.get("n") or 0) < 20})
    rows.sort(key=lambda x: (float(x.get("t3r") or -1e9), float(x.get("sum") or -1e9)), reverse=True)
    return rows


def render_md(report: dict[str, Any]) -> str:
    lines = [
        "# S34 State Machine V8 Robustness Suite",
        "",
        f"- generated_at_utc: `{report['generated_at_utc']}`",
        "- research_only: `true`",
        "- live_changes: `none`",
        "",
        "## Ideas / Questions Tested",
        "",
        "1. Does profit-lock 100/50 survive folds, fee stress, and side split?",
        "2. Is confidence sizing robust or just larger risk?",
        "3. Does execution latency destroy the state-machine edge?",
        "4. Is early weak/adverse movement useful as a defensive cut?",
        "5. Does partial scale-out improve skew?",
        "6. Which conflict policy is safest?",
        "7. Which BTC/score grid is best without overfitting?",
        "8. Does top-winner removal break candidates?",
        "9. Which candidates survive extra 2/5/10/20 bps cost?",
        "10. Which candidates deserve shadow only vs live consideration?",
        "",
        "## Candidate Robustness",
        "",
    ]
    for name, val in report["candidates"].items():
        lines.append(f"- {name}: hold `{stat_line(val['split']['hold'])}`, folds_pos={val['folds']['positive_folds']}/5, fold_t3r_sum={val['folds']['t3r_sum']}, hold_top3_removed `{stat_line(val['top3_removed_hold'])}`")
    lines += [
        "",
        "## Top Config Grid By Holdout T3R",
        "",
    ]
    for r in report["config_grid_ranked"][:10]:
        lines.append(f"- {r['name']}: `{stat_line(r)}` small_n={r['small_n']}")
    lines += [
        "",
        "## Full JSON",
        "",
        f"- `{OUT_JSON}`",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    rows, *_unused, mk_ts, mk_px = build_base_rows()
    raw = build_signals(rows, FINAL_CFG, mk_ts=mk_ts, mk_px=mk_px)
    signals, blocked = apply_conflict_policy(raw, "short_replace")
    cands = candidate_pack(signals, mk_ts, mk_px)
    grid = config_grid(rows, mk_ts, mk_px)
    report = {
        "generated_at_utc": utc_now(),
        "research_only": True,
        "live_changes": "none",
        "data": {
            "classified_rows": len(rows),
            "raw_signals": len(raw),
            "taken_signals": len(signals),
            "blocked_signals": len(blocked),
        },
        "candidates": summarize_candidates(cands),
        "fee_stress": fee_stress(cands),
        "latency_stress": latency_stress(signals, mk_ts, mk_px),
        "conflict_policy": conflict_policy_grid(raw),
        "config_grid": grid,
        "config_grid_ranked": rank_config_grid(grid),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    OUT_MD.write_text(render_md(report), encoding="utf-8")
    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_MD}")
    print(json.dumps({
        "baseline_hold": report["candidates"]["baseline"]["split"]["hold"],
        "profit_lock_hold": report["candidates"]["profit_lock_100_50"]["split"]["hold"],
        "confidence_hold": report["candidates"]["confidence_sized"]["split"]["hold"],
        "top_configs": report["config_grid_ranked"][:5],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
