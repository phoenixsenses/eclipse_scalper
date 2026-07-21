"""S34 state-machine v5 development suite.

Research-only. This does not touch live executor, env, runtime state, or orders.

The suite answers the next development questions around the promoted live state
machine: anatomy, transitions, exits, regime, score ablations, dose response,
frequency expansion, permission/navigation, and tail neighborhoods.
"""

from __future__ import annotations

import json
import math
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.research_s34_state_machine_v2_gauntlet import (  # noqa: E402
    DEFAULT_DB,
    DOW,
    FEE_BPS,
    PROP_THRESH,
    SIL_HI_MS,
    SIL_LO_MS,
    Config,
    apply_conflict_policy,
    build_signals,
    first_above,
    iso_ms,
    mark_at_or_after,
    recompute_score,
    signed_net,
    state_for,
    summarize,
    summary_with_dd,
)
from tools.research_s34_state_machine_v3_full_tests import (  # noqa: E402
    book_realism_suite,
    conflict_variants,
    dow_stability,
    horizon_exit_suite,
    latency_suite,
    mark_max,
    mark_min,
    monthly_stability,
    net_between,
    stop_suite,
    tail_cluster_suite,
)
from tools.research_s34_state_machine_v4_promotion_gauntlet import (  # noqa: E402
    build_base_rows,
    with_book_required,
    with_stops,
)


OUT_JSON = ROOT / "reports" / "research" / "s34" / "S34_STATE_MACHINE_V5_DEV_SUITE.json"
OUT_MD = ROOT / "reports" / "research" / "s34" / "S34_STATE_MACHINE_V5_DEV_SUITE.md"

FINAL_CFG = Config(
    "btc1000_dow_score3",
    btc_thr=1_000_000.0,
    long_score_min=3,
    short_score_min=3,
    exclude_long_dow=(0, 2),
    exclude_short_dow=(6,),
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def split(signals: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "all": summary_with_dd(signals),
        "cal": summary_with_dd([s for s in signals if not s["row"]["is_hold"]]),
        "hold": summary_with_dd([s for s in signals if s["row"]["is_hold"]]),
    }


def pct(vals: list[float], p: float) -> float | None:
    vals = sorted(float(v) for v in vals if math.isfinite(float(v)))
    if not vals:
        return None
    i = max(0, min(int((len(vals) - 1) * p / 100.0), len(vals) - 1))
    return round(vals[i], 1)


def avg(vals: list[float]) -> float | None:
    vals = [float(v) for v in vals if math.isfinite(float(v))]
    return round(mean(vals), 2) if vals else None


def feature_profile(signals: list[dict[str, Any]]) -> dict[str, Any]:
    if not signals:
        return {"n": 0}
    rows = [s["row"] for s in signals]
    return {
        "n": len(signals),
        "avg_net_bps": avg([s["net_bps"] for s in signals]),
        "median_net_bps": round(median([float(s["net_bps"]) for s in signals]), 1),
        "avg_score": avg([s.get("score", r.get("score_default", 0)) for s, r in zip(signals, rows, strict=False)]),
        "avg_sync_k": avg([r.get("sync_k", 0) for r in rows]),
        "avg_n2h": avg([r.get("n2h", 0) for r in rows]),
        "avg_btc4h_bps": avg([r.get("b4h", 0) for r in rows]),
        "avg_vdepth_bps": avg([r.get("vd", 0) for r in rows]),
        "avg_bid_depth": avg([r.get("bid", 0) for r in rows]),
        "session_counts": dict(sorted((str(k), sum(1 for r in rows if r.get("session") == k)) for k in {r.get("session") for r in rows})),
        "dow_counts": dict(
            sorted(
                ((DOW[int(k)], sum(1 for r in rows if int(r.get("dow", -1)) == int(k))) for k in {r.get("dow") for r in rows if r.get("dow") is not None}),
                key=lambda kv: DOW.index(kv[0]),
            )
        ),
    }


def side_anatomy(signals: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, pred in {
        "LONG": lambda s: s["side"] == "LONG",
        "SHORT": lambda s: s["side"] == "SHORT",
        "LONG_winners": lambda s: s["side"] == "LONG" and float(s["net_bps"]) > 0,
        "LONG_losers": lambda s: s["side"] == "LONG" and float(s["net_bps"]) <= 0,
        "SHORT_winners": lambda s: s["side"] == "SHORT" and float(s["net_bps"]) > 0,
        "SHORT_losers": lambda s: s["side"] == "SHORT" and float(s["net_bps"]) <= 0,
    }.items():
        subset = [s for s in signals if pred(s)]
        out[key] = {"summary": split(subset), "profile": feature_profile(subset)}
    return out


def sequence_model(rows: list[dict[str, Any]], signals: list[dict[str, Any]]) -> dict[str, Any]:
    by_state: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_state[state_for(r, FINAL_CFG.btc_thr)].append({"entry_ts_ms": r["ts"], "net_bps": r["long_t0_4h"] if r["sil_eth"] else r["short_anchor_2h"], "row": r})
    sig_by_state: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for s in signals:
        sig_by_state[state_for(s["row"], FINAL_CFG.btc_thr)].append(s)
    delays = []
    for s in signals:
        if s["side"] == "SHORT":
            delays.append((int(s["entry_ts_ms"]) - int(s["anchor_ts_ms"])) / 60_000.0)
    return {
        "raw_state_outcomes": {k: split([x for x in v if x["net_bps"] is not None]) for k, v in sorted(by_state.items())},
        "taken_by_state": {k: split(v) for k, v in sorted(sig_by_state.items())},
        "short_btc_confirm_delay_min": {
            "n": len(delays),
            "median": pct(delays, 50),
            "p75": pct(delays, 75),
            "p90": pct(delays, 90),
        },
    }


def silence_cause(rows: list[dict[str, Any]], signals: list[dict[str, Any]]) -> dict[str, Any]:
    silence_sigs = [s for s in signals if s["arm"] == "SILENCE_LONG"]
    raw_sil = [r for r in rows if r["sil_eth"] and not r["bull"]]
    raw_noisy = [r for r in rows if not r["sil_eth"] and not r["bull"]]
    return {
        "taken_silence": split(silence_sigs),
        "taken_silence_winners_profile": feature_profile([s for s in silence_sigs if float(s["net_bps"]) > 0]),
        "taken_silence_losers_profile": feature_profile([s for s in silence_sigs if float(s["net_bps"]) <= 0]),
        "raw_silence_profile": feature_profile([{"net_bps": r.get("long_t0_4h") or 0.0, "row": r, "side": "LONG"} for r in raw_sil]),
        "raw_noisy_profile": feature_profile([{"net_bps": r.get("short_anchor_2h") or 0.0, "row": r, "side": "SHORT"} for r in raw_noisy]),
        "read": "If silence winners mainly have lower sync/new-liq pressure than raw noisy rows, silence is acting as panic-ended / non-propagation context, not a pure directional alpha.",
    }


def lifecycle(signals: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(signals, key=lambda s: int(s["entry_ts_ms"]))
    transitions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    blocked: list[dict[str, Any]] = []
    active_side = None
    active_end = None
    for s in ordered:
        side = s["side"]
        hold = 4 * 3600_000 if side == "LONG" else 2 * 3600_000
        entry = int(s["entry_ts_ms"])
        if active_end is None or entry >= active_end:
            transitions[f"NONE->{side}"].append(s)
            active_side = side
            active_end = entry + hold
        elif side == active_side == "SHORT":
            transitions["SHORT->SHORT_REPLACE"].append(s)
            active_end = entry + hold
        elif side == "SHORT" and active_side == "LONG":
            transitions["LONG->SHORT_FLIP"].append(s)
            active_side = "SHORT"
            active_end = entry + hold
        else:
            transitions[f"{active_side}->{side}_BLOCK"].append(s)
            blocked.append(s)
    return {
        "transition_summaries": {k: split(v) for k, v in sorted(transitions.items())},
        "blocked_summary": split(blocked),
        "conflict_policy_duel": conflict_variants(signals),
    }


def early_danger(signals: list[dict[str, Any]], mk_ts: list[int], mk_px: list[float]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for minute in [5, 10, 15, 30]:
        rows = []
        for s in signals:
            entry = int(s["entry_ts_ms"])
            entry_px = mark_at_or_after(mk_ts, mk_px, entry)
            if not entry_px:
                continue
            hi = mark_max(mk_ts, mk_px, entry, entry + minute * 60_000)
            lo = mark_min(mk_ts, mk_px, entry, entry + minute * 60_000)
            if hi is None or lo is None:
                continue
            if s["side"] == "LONG":
                favorable = (hi - entry_px) / entry_px * 10_000.0
                adverse = (lo - entry_px) / entry_px * 10_000.0
            else:
                favorable = (entry_px - lo) / entry_px * 10_000.0
                adverse = (entry_px - hi) / entry_px * 10_000.0
            rows.append({**s, "early_fav": favorable, "early_adv": adverse})
        bins = {
            "adverse_le_-20": [s for s in rows if s["early_adv"] <= -20],
            "adverse_gt_-20": [s for s in rows if s["early_adv"] > -20],
            "favorable_ge_20": [s for s in rows if s["early_fav"] >= 20],
            "favorable_lt_20": [s for s in rows if s["early_fav"] < 20],
        }
        out[f"{minute}m"] = {k: split(v) for k, v in bins.items()} | {
            "fav_median_bps": pct([s["early_fav"] for s in rows], 50),
            "adv_median_bps": pct([s["early_adv"] for s in rows], 50),
        }
    return out


def bull_run_adaptation(rows: list[dict[str, Any]], mk_ts: list[int], mk_px: list[float]) -> dict[str, Any]:
    cfg_no_bull_filter = Config(
        "btc1000_dow_score3_include_bull",
        btc_thr=1_000_000.0,
        long_score_min=3,
        short_score_min=3,
        exclude_long_dow=(0, 2),
        exclude_short_dow=(6,),
    )
    # Counterfactual: clone rows with bull=False to let build_signals include them.
    cloned = [{**r, "bull": False} for r in rows]
    all_sigs = apply_conflict_policy(build_signals(cloned, cfg_no_bull_filter, mk_ts=mk_ts, mk_px=mk_px), "short_replace")[0]
    bull_sigs = [s for s in all_sigs if rows[int(s["row"]["idx"])]["bull"]]
    non_bull_sigs = [s for s in all_sigs if not rows[int(s["row"]["idx"])]["bull"]]
    trend_bins: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for s in all_sigs:
        r = rows[int(s["row"]["idx"])]
        eth_shift = float(r.get("eth_shift_30_bps") or 0.0)
        key = "eth_30m_up" if eth_shift > 15 else "eth_30m_flat_down"
        trend_bins[key].append(s)
    return {
        "include_bull_all": split(all_sigs),
        "bull_only": split(bull_sigs),
        "non_bull_only": split(non_bull_sigs),
        "eth_30m_context": {k: split(v) for k, v in sorted(trend_bins.items())},
        "read": "Current live rule excludes BULL_PULLBACK. This tests whether that is protective or leaving bull-run LONG opportunity.",
    }


def regime_recovery(signals: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(signals, key=lambda s: int(s["entry_ts_ms"]))
    windows = {}
    for w in [5, 10, 20]:
        chunks = []
        for i in range(0, max(0, len(ordered) - w + 1)):
            chunk = ordered[i : i + w]
            sm = summary_with_dd(chunk)
            chunks.append({"start": iso_ms(chunk[0]["entry_ts_ms"]), "end": iso_ms(chunk[-1]["entry_ts_ms"]), **sm})
        worst = sorted(chunks, key=lambda x: float(x.get("sum") or 0.0))[:5]
        windows[f"roll_{w}"] = {"n_windows": len(chunks), "worst5": worst}
    return {
        "monthly": monthly_stability(signals),
        "rolling_windows": windows,
        "tail_cluster": tail_cluster_suite(signals),
    }


def score_ablation(rows: list[dict[str, Any]], mk_ts: list[int], mk_px: list[float]) -> dict[str, Any]:
    components = {
        "sil_eth": lambda r: int(r["sil_eth"]),
        "n2h": lambda r: int(r["n2h"] >= 3),
        "btc4h_down": lambda r: int(r["b4h"] < 0),
        "vdepth": lambda r: int(r["vd"] >= 30),
        "us_session": lambda r: int(r["sess_us"]),
        "sync200": lambda r: int(r["sync_k"] >= 200_000),
    }

    def custom_score(row: dict[str, Any], drop: str | None = None) -> int:
        return sum(fn(row) for name, fn in components.items() if name != drop)

    out = {}
    for drop in [None, *components.keys()]:
        name = "full_score" if drop is None else f"drop_{drop}"
        sigs = []
        for r in rows:
            if r["bull"]:
                continue
            score = custom_score(r, drop)
            if r["sil_eth"]:
                if r["session"] == "EUROPE" or int(r["dow"]) in {0, 2} or score < 3:
                    continue
                if r.get("long_t0_4h") is not None:
                    sigs.append({"entry_ts_ms": int(r["ts"]), "anchor_ts_ms": int(r["ts"]), "side": "LONG", "arm": "SILENCE_LONG", "net_bps": float(r["long_t0_4h"]), "row": r, "score": score})
            else:
                entry_ts = r["first_btc_by_thr"].get("1000000")
                if entry_ts is None or int(r["dow"]) == 6 or score < 3:
                    continue
                net = net_between("SHORT", mk_ts, mk_px, int(entry_ts), int(entry_ts) + 2 * 3600_000)
                if net is not None:
                    sigs.append({"entry_ts_ms": int(entry_ts), "anchor_ts_ms": int(r["ts"]), "side": "SHORT", "arm": "NEITHER_SHORT", "net_bps": float(net), "row": r, "score": score})
        out[name] = split(apply_conflict_policy(sigs, "short_replace")[0])
    return out


def btc_dose_response(rows: list[dict[str, Any]], mk_ts: list[int], mk_px: list[float]) -> dict[str, Any]:
    out = {}
    for thr in [300_000, 500_000, 750_000, 1_000_000, 1_250_000, 1_500_000]:
        cfg = Config(
            f"btc{int(thr/1000)}k_dow_score3",
            btc_thr=float(thr),
            long_score_min=3,
            short_score_min=3,
            exclude_long_dow=(0, 2),
            exclude_short_dow=(6,),
        )
        sigs = apply_conflict_policy(build_signals(rows, cfg, mk_ts=mk_ts, mk_px=mk_px), "short_replace")[0]
        out[cfg.name] = split(sigs)
    return out


def dow_robustness(rows: list[dict[str, Any]], mk_ts: list[int], mk_px: list[float]) -> dict[str, Any]:
    cfg_no_dow = Config("btc1000_score3_no_dow", btc_thr=1_000_000, long_score_min=3, short_score_min=3)
    no_dow = apply_conflict_policy(build_signals(rows, cfg_no_dow, mk_ts=mk_ts, mk_px=mk_px), "short_replace")[0]
    final = apply_conflict_policy(build_signals(rows, FINAL_CFG, mk_ts=mk_ts, mk_px=mk_px), "short_replace")[0]
    excluded_counterfactual = [s for s in no_dow if s not in final]
    return {
        "with_dow_filter": split(final),
        "without_dow_filter": split(no_dow),
        "excluded_counterfactual": split(excluded_counterfactual),
        "per_dow_without_filter": dow_stability(no_dow),
    }


def frequency_expansion(rows: list[dict[str, Any]], mk_ts: list[int], mk_px: list[float]) -> dict[str, Any]:
    configs = [
        Config("final_btc1000_score3_dow", btc_thr=1_000_000, long_score_min=3, short_score_min=3, exclude_long_dow=(0, 2), exclude_short_dow=(6,)),
        Config("looser_btc750_score3_dow", btc_thr=750_000, long_score_min=3, short_score_min=3, exclude_long_dow=(0, 2), exclude_short_dow=(6,)),
        Config("looser_btc500_score3_dow", btc_thr=500_000, long_score_min=3, short_score_min=3, exclude_long_dow=(0, 2), exclude_short_dow=(6,)),
        Config("score2_btc1000_dow", btc_thr=1_000_000, long_score_min=2, short_score_min=2, exclude_long_dow=(0, 2), exclude_short_dow=(6,)),
        Config("include_europe_long", btc_thr=1_000_000, long_score_min=3, short_score_min=3, exclude_long_dow=(0, 2), exclude_short_dow=(6,), exclude_europe_long=False),
        Config("include_noisy_short", btc_thr=1_000_000, long_score_min=3, short_score_min=3, exclude_long_dow=(0, 2), exclude_short_dow=(6,), include_noisy_short=True),
    ]
    out = {}
    for cfg in configs:
        sigs = apply_conflict_policy(build_signals(rows, cfg, mk_ts=mk_ts, mk_px=mk_px), "short_replace")[0]
        out[cfg.name] = split(sigs)
    return out


def navigation_permission(signals: list[dict[str, Any]]) -> dict[str, Any]:
    by_score: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_sync: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for s in signals:
        score = int(s.get("score") or 0)
        by_score[f"score_{score}"].append(s)
        sync = float(s["row"].get("sync_k") or 0)
        if sync < 200_000:
            key = "sync_lt_200k"
        elif sync < 500_000:
            key = "sync_200_500k"
        elif sync < 1_000_000:
            key = "sync_500_1000k"
        else:
            key = "sync_ge_1000k"
        by_sync[key].append(s)
    return {
        "score_bins": {k: split(v) for k, v in sorted(by_score.items())},
        "sync_bins": {k: split(v) for k, v in sorted(by_sync.items())},
        "permission_candidate": "Use only as dashboard/nav labels until forward OOS confirms; do not add live filters from these bins in-sample.",
    }


def tail_neighborhood(signals: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(signals, key=lambda s: int(s["entry_ts_ms"]))
    before_tail = []
    after_tail_1 = []
    after_tail_2 = []
    same_day_after_tail = []
    for i, s in enumerate(ordered):
        if float(s["net_bps"]) <= -50:
            if i > 0:
                before_tail.append(ordered[i - 1])
            if i + 1 < len(ordered):
                after_tail_1.append(ordered[i + 1])
            if i + 2 < len(ordered):
                after_tail_2.append(ordered[i + 2])
            day = datetime.fromtimestamp(int(s["entry_ts_ms"]) / 1000.0, tz=timezone.utc).date()
            same_day_after_tail.extend([
                x for x in ordered[i + 1 :]
                if datetime.fromtimestamp(int(x["entry_ts_ms"]) / 1000.0, tz=timezone.utc).date() == day
            ])
    return {
        "tail_threshold_bps": -50,
        "tail_count": sum(1 for s in ordered if float(s["net_bps"]) <= -50),
        "before_tail": split(before_tail),
        "after_tail_next1": split(after_tail_1),
        "after_tail_next2": split(after_tail_2),
        "same_day_after_tail": split(same_day_after_tail),
        "cooldown_interpretation": "If after-tail subsets are worse, cooldown is management; if better, tail is isolated and cooldown may be harmful.",
    }


def render_stat(name: str, s: dict[str, Any]) -> str:
    return f"{name}: N={s.get('n',0)} WR={'' if s.get('wr') is None else round(float(s['wr'])*100,1)}% sum={s.get('sum')} med={s.get('median')} T3R={s.get('t3r')} maxLoss={s.get('max_loss')} DD={s.get('max_dd_bps')}"


def top_hold(table: dict[str, Any], n: int = 8) -> list[tuple[str, dict[str, Any]]]:
    items = []
    for k, v in table.items():
        hold = ((v or {}).get("hold") if isinstance(v, dict) else None) or ((v or {}).get("taken", {}).get("hold") if isinstance(v, dict) else None)
        if hold:
            items.append((k, hold))
    return sorted(items, key=lambda kv: float(kv[1].get("t3r") or -1e18), reverse=True)[:n]


def render_md(report: dict[str, Any]) -> str:
    p = report["primary"]
    lines = [
        "# S34 State Machine V5 Development Suite",
        "",
        f"- generated_at_utc: `{report['generated_at_utc']}`",
        "- research_only: `true`",
        f"- primary_config: `{report['primary_config']}`",
        f"- primary_hold: `{p['hold']}`",
        "",
        "## Executive Read",
        "",
        f"- Final live config remains strongest conservative lane: hold N={p['hold']['n']}, WR={p['hold']['wr']}, sum={p['hold']['sum']}bps, T3R={p['hold']['t3r']}bps.",
        "- Best development leads are management/navigation, not immediate live filter changes.",
        "- Frequency expansion candidates are research-only; no in-sample expansion should be promoted without a separate gauntlet.",
        "",
        "## Top Tables",
        "",
        "### BTC Dose Response",
    ]
    for name, s in top_hold(report["btc_dose_response"]):
        lines.append(f"- {render_stat(name, s)}")
    lines += ["", "### Frequency Expansion Counterfactuals"]
    for name, s in top_hold(report["frequency_expansion"]):
        lines.append(f"- {render_stat(name, s)}")
    lines += ["", "### Score Ablation"]
    for name, s in top_hold(report["score_ablation"]):
        lines.append(f"- {render_stat(name, s)}")
    lines += ["", "### DOW Robustness"]
    for k, v in report["dow_robustness"].items():
        if isinstance(v, dict) and "hold" in v:
            lines.append(f"- {render_stat(k, v['hold'])}")
    lines += [
        "",
        "## 15 Question Results",
        "",
        "1. LONG/SHORT anatomy: see `side_anatomy`.",
        "2. Follow-on sequence: see `sequence_model`.",
        "3. Silence cause: see `silence_cause`.",
        "4. SHORT replace: see `lifecycle.conflict_policy_duel`.",
        "5. Position lifecycle transitions: see `lifecycle.transition_summaries`.",
        "6. Exit timing: see `exit_timing`.",
        "7. Early danger monitor: see `early_danger`.",
        "8. Bull-run adaptation: see `bull_run_adaptation`.",
        "9. Regime recovery: see `regime_recovery`.",
        "10. Score ablation: see `score_ablation`.",
        "11. BTC threshold dose response: see `btc_dose_response`.",
        "12. DOW robustness: see `dow_robustness`.",
        "13. Frequency expansion: see `frequency_expansion`.",
        "14. Navigation permission: see `navigation_permission`.",
        "15. Tail neighborhood: see `tail_neighborhood`.",
        "",
        "## Full JSON",
        "",
        f"- `{OUT_JSON}`",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    rows, eth_ts, eth_not, btc_ts, btc_not, sol_ts, sol_not, mk_ts, mk_px = build_base_rows()
    raw = build_signals(rows, FINAL_CFG, mk_ts=mk_ts, mk_px=mk_px)
    signals, blocked = apply_conflict_policy(raw, "short_replace")
    report = {
        "generated_at_utc": utc_now(),
        "research_only": True,
        "primary_config": FINAL_CFG.name,
        "data": {"classified_rows": len(rows), "raw_signals": len(raw), "taken_signals": len(signals), "blocked_signals": len(blocked)},
        "primary": split(signals),
        "side_anatomy": side_anatomy(signals),
        "sequence_model": sequence_model(rows, signals),
        "silence_cause": silence_cause(rows, signals),
        "lifecycle": lifecycle(raw),
        "exit_timing": horizon_exit_suite(signals, mk_ts, mk_px),
        "stops": stop_suite(signals, mk_ts, mk_px),
        "latency": latency_suite(signals, mk_ts, mk_px),
        "book_realism": book_realism_suite(signals, DEFAULT_DB),
        "early_danger": early_danger(signals, mk_ts, mk_px),
        "bull_run_adaptation": bull_run_adaptation(rows, mk_ts, mk_px),
        "regime_recovery": regime_recovery(signals),
        "score_ablation": score_ablation(rows, mk_ts, mk_px),
        "btc_dose_response": btc_dose_response(rows, mk_ts, mk_px),
        "dow_robustness": dow_robustness(rows, mk_ts, mk_px),
        "frequency_expansion": frequency_expansion(rows, mk_ts, mk_px),
        "navigation_permission": navigation_permission(signals),
        "tail_neighborhood": tail_neighborhood(signals),
        "protective_stop_counterfactual": {
            "sl100": split(with_stops(signals, mk_ts, mk_px, 100.0)),
            "sl150": split(with_stops(signals, mk_ts, mk_px, 150.0)),
            "book_required_10s": split(with_book_required(signals, DEFAULT_DB, 10)),
        },
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    OUT_MD.write_text(render_md(report), encoding="utf-8")
    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_MD}")
    print(json.dumps({"primary_hold": report["primary"]["hold"], "top_btc": top_hold(report["btc_dose_response"], 3), "top_freq": top_hold(report["frequency_expansion"], 3)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
