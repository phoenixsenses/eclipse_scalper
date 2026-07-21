"""S34 state-machine v6 development ideas.

Research-only. Does not touch live executor, env, runtime state, or orders.

Tests the requested next development lines:
- top 5 development tracks from v5
- 10 additional questions around MFE/MAE, profit lock, replace quality,
  silence purity, BTC confirm speed, tail neighborhoods, sessions, volatility,
  BTC/ETH divergence, and confidence sizing.
"""

from __future__ import annotations

import bisect
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
    signed_net,
    summary_with_dd,
    win_cnt,
    win_sum,
)
from tools.research_s34_state_machine_v3_full_tests import (  # noqa: E402
    horizon_exit_suite,
    mark_max,
    mark_min,
    net_between,
)
from tools.research_s34_state_machine_v4_promotion_gauntlet import build_base_rows  # noqa: E402


OUT_JSON = ROOT / "reports" / "research" / "s34" / "S34_STATE_MACHINE_V6_DEVELOPMENT_IDEAS.json"
OUT_MD = ROOT / "reports" / "research" / "s34" / "S34_STATE_MACHINE_V6_DEVELOPMENT_IDEAS.md"

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
    idx = max(0, min(int((len(vals) - 1) * p / 100.0), len(vals) - 1))
    return round(vals[idx], 1)


def avg(vals: list[float]) -> float | None:
    vals = [float(v) for v in vals if math.isfinite(float(v))]
    return round(mean(vals), 2) if vals else None


def arm(signals: list[dict[str, Any]], name: str) -> list[dict[str, Any]]:
    return [s for s in signals if s.get("arm") == name]


def mfe_mae_for_signal(s: dict[str, Any], mk_ts: list[int], mk_px: list[float], horizon_ms: int | None = None) -> dict[str, Any] | None:
    entry_ts = int(s["entry_ts_ms"])
    side = str(s["side"]).upper()
    if horizon_ms is None:
        horizon_ms = 4 * 3600_000 if side == "LONG" else 2 * 3600_000
    entry = mark_at_or_after(mk_ts, mk_px, entry_ts)
    if not entry or entry <= 0:
        return None
    a = bisect.bisect_left(mk_ts, entry_ts)
    b = bisect.bisect_right(mk_ts, entry_ts + horizon_ms)
    if a >= b:
        return None
    best = -1e18
    worst = 1e18
    best_ts = None
    worst_ts = None
    for i in range(a, b):
        raw = (float(mk_px[i]) - entry) / entry * 10_000.0
        val = -raw if side == "SHORT" else raw
        if val > best:
            best = val
            best_ts = int(mk_ts[i])
        if val < worst:
            worst = val
            worst_ts = int(mk_ts[i])
    return {
        "mfe_bps": round(best, 1),
        "mae_bps": round(worst, 1),
        "mfe_min": round((int(best_ts) - entry_ts) / 60_000.0, 1) if best_ts else None,
        "mae_min": round((int(worst_ts) - entry_ts) / 60_000.0, 1) if worst_ts else None,
    }


def mfe_mae_exit_map(signals: list[dict[str, Any]], mk_ts: list[int], mk_px: list[float]) -> dict[str, Any]:
    rows = []
    for s in signals:
        mm = mfe_mae_for_signal(s, mk_ts, mk_px)
        if mm:
            rows.append({**s, **mm})
    out: dict[str, Any] = {}
    for key, subset in {
        "all": rows,
        "long": [s for s in rows if s["side"] == "LONG"],
        "short": [s for s in rows if s["side"] == "SHORT"],
        "winners": [s for s in rows if float(s["net_bps"]) > 0],
        "losers": [s for s in rows if float(s["net_bps"]) <= 0],
    }.items():
        out[key] = {
            "n": len(subset),
            "mfe_median_bps": pct([s["mfe_bps"] for s in subset], 50),
            "mae_median_bps": pct([s["mae_bps"] for s in subset], 50),
            "mfe_time_median_min": pct([s["mfe_min"] for s in subset if s["mfe_min"] is not None], 50),
            "mae_time_median_min": pct([s["mae_min"] for s in subset if s["mae_min"] is not None], 50),
            "final": split(subset),
        }
    return out


def profit_lock(signals: list[dict[str, Any]], mk_ts: list[int], mk_px: list[float]) -> dict[str, Any]:
    """Counterfactual: after reaching a profit trigger, exit at fixed lock if later falls below it."""
    out: dict[str, Any] = {}
    for trig, lock in [(50, 0), (75, 25), (100, 50), (150, 75)]:
        managed = []
        triggered = 0
        for s in signals:
            entry_ts = int(s["entry_ts_ms"])
            side = str(s["side"]).upper()
            horizon = 4 * 3600_000 if side == "LONG" else 2 * 3600_000
            entry = mark_at_or_after(mk_ts, mk_px, entry_ts)
            if not entry:
                continue
            a = bisect.bisect_left(mk_ts, entry_ts)
            b = bisect.bisect_right(mk_ts, entry_ts + horizon)
            armed = False
            net = float(s["net_bps"])
            for i in range(a, b):
                raw = (float(mk_px[i]) - entry) / entry * 10_000.0
                pnl = -raw if side == "SHORT" else raw
                if not armed and pnl >= trig:
                    armed = True
                    triggered += 1
                if armed and pnl <= lock:
                    net = float(lock) - FEE_BPS
                    break
            managed.append({**s, "net_bps": net, "profit_lock_triggered": armed, "trigger_bps": trig, "lock_bps": lock})
        res = split(managed)
        res["triggered"] = triggered
        res["triggered_pct"] = round(triggered / len(signals), 3) if signals else 0
        out[f"trig{trig}_lock{lock}"] = res
    return out


def early_momentum_observer(signals: list[dict[str, Any]], mk_ts: list[int], mk_px: list[float]) -> dict[str, Any]:
    out = {}
    for minute in [5, 10, 15]:
        rows = []
        for s in signals:
            mm = mfe_mae_for_signal(s, mk_ts, mk_px, minute * 60_000)
            if mm:
                rows.append({**s, "early_mfe": mm["mfe_bps"], "early_mae": mm["mae_bps"]})
        out[f"{minute}m"] = {
            "fav_ge_20": split([s for s in rows if s["early_mfe"] >= 20]),
            "fav_lt_20": split([s for s in rows if s["early_mfe"] < 20]),
            "adv_le_-20": split([s for s in rows if s["early_mae"] <= -20]),
            "adv_gt_-20": split([s for s in rows if s["early_mae"] > -20]),
            "fav_ge_20_and_adv_gt_-20": split([s for s in rows if s["early_mfe"] >= 20 and s["early_mae"] > -20]),
            "fav_lt_20_or_adv_le_-20": split([s for s in rows if s["early_mfe"] < 20 or s["early_mae"] <= -20]),
        }
    return out


def arm_specific_exit(signals: list[dict[str, Any]], mk_ts: list[int], mk_px: list[float]) -> dict[str, Any]:
    return {
        "long_silence": horizon_exit_suite(arm(signals, "SILENCE_LONG"), mk_ts, mk_px, include_by_arm=False),
        "short_neither": horizon_exit_suite(arm(signals, "NEITHER_SHORT"), mk_ts, mk_px, include_by_arm=False),
    }


def regime_degradation(signals: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(signals, key=lambda s: int(s["entry_ts_ms"]))
    out = {}
    for w in [5, 10, 20]:
        windows = []
        for i in range(0, max(0, len(ordered) - w + 1)):
            chunk = ordered[i : i + w]
            sm = summary_with_dd(chunk)
            windows.append({
                "start_utc": iso_ms(chunk[0]["entry_ts_ms"]),
                "end_utc": iso_ms(chunk[-1]["entry_ts_ms"]),
                **sm,
            })
        out[f"roll{w}"] = {
            "n_windows": len(windows),
            "worst_sum": sorted(windows, key=lambda x: float(x.get("sum") or 0))[:5],
            "worst_t3r": sorted(windows, key=lambda x: float(x.get("t3r") or 0))[:5],
            "negative_windows": sum(1 for x in windows if float(x.get("sum") or 0) < 0),
        }
    return out


def score4_shadow(rows: list[dict[str, Any]], mk_ts: list[int], mk_px: list[float]) -> dict[str, Any]:
    cfg = Config("btc1000_dow_score4", btc_thr=1_000_000, long_score_min=4, short_score_min=4, exclude_long_dow=(0, 2), exclude_short_dow=(6,))
    sigs = apply_conflict_policy(build_signals(rows, cfg, mk_ts=mk_ts, mk_px=mk_px), "short_replace")[0]
    return {"summary": split(sigs), "by_arm": {a: split([s for s in sigs if s["arm"] == a]) for a in sorted({s["arm"] for s in sigs})}}


def btc750_shadow(rows: list[dict[str, Any]], mk_ts: list[int], mk_px: list[float]) -> dict[str, Any]:
    cfg = Config("btc750_dow_score3", btc_thr=750_000, long_score_min=3, short_score_min=3, exclude_long_dow=(0, 2), exclude_short_dow=(6,))
    sigs = apply_conflict_policy(build_signals(rows, cfg, mk_ts=mk_ts, mk_px=mk_px), "short_replace")[0]
    return {"summary": split(sigs), "by_arm": {a: split([s for s in sigs if s["arm"] == a]) for a in sorted({s["arm"] for s in sigs})}}


def short_replace_quality(raw_signals: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(raw_signals, key=lambda s: int(s["entry_ts_ms"]))
    active_side = None
    active_end = None
    replaces = []
    first_shorts = []
    for s in ordered:
        side = s["side"]
        hold = 4 * 3600_000 if side == "LONG" else 2 * 3600_000
        entry = int(s["entry_ts_ms"])
        if active_end is None or entry >= active_end:
            if side == "SHORT":
                first_shorts.append(s)
            active_side = side
            active_end = entry + hold
        elif side == active_side == "SHORT":
            replaces.append(s)
            active_end = entry + hold
        elif side == "SHORT" and active_side == "LONG":
            first_shorts.append(s)
            active_side = "SHORT"
            active_end = entry + hold
    def prof(sub: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "summary": split(sub),
            "avg_score": avg([s.get("score", 0) for s in sub]),
            "avg_sync_k": avg([s["row"].get("sync_k", 0) for s in sub]),
            "avg_n2h": avg([s["row"].get("n2h", 0) for s in sub]),
            "avg_btc4h": avg([s["row"].get("b4h", 0) for s in sub]),
        }
    return {"first_shorts": prof(first_shorts), "replace_shorts": prof(replaces)}


def long_silence_purity(signals: list[dict[str, Any]]) -> dict[str, Any]:
    longs = arm(signals, "SILENCE_LONG")
    buckets = {
        "pure_depth_sync": [s for s in longs if s["row"].get("vd", 0) >= 30 and s["row"].get("sync_k", 0) < 500_000],
        "high_depth": [s for s in longs if s["row"].get("vd", 0) >= 35],
        "low_sync": [s for s in longs if s["row"].get("sync_k", 0) < 300_000],
        "high_bid": [s for s in longs if s["row"].get("bid", 0) >= 100_000],
        "weak_depth_or_bid": [s for s in longs if s["row"].get("vd", 0) < 30 or s["row"].get("bid", 0) < 100_000],
    }
    return {k: split(v) for k, v in buckets.items()}


def btc_confirm_speed(signals: list[dict[str, Any]]) -> dict[str, Any]:
    shorts = arm(signals, "NEITHER_SHORT")
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for s in shorts:
        delay = (int(s["entry_ts_ms"]) - int(s["anchor_ts_ms"])) / 60_000.0
        if delay < 5:
            key = "lt5m"
        elif delay < 15:
            key = "5_15m"
        else:
            key = "15_30m"
        buckets[key].append({**s, "confirm_delay_min": delay})
    return {k: split(v) for k, v in sorted(buckets.items())}


def post_tail_no_cooldown(signals: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(signals, key=lambda s: int(s["entry_ts_ms"]))
    out = {}
    for thr in [-30, -50, -75, -100]:
        next1 = []
        next2 = []
        next3 = []
        for i, s in enumerate(ordered):
            if float(s["net_bps"]) <= thr:
                if i + 1 < len(ordered):
                    next1.append(ordered[i + 1])
                if i + 2 < len(ordered):
                    next2.append(ordered[i + 2])
                if i + 3 < len(ordered):
                    next3.append(ordered[i + 3])
        out[f"tail_le_{thr}"] = {
            "tail_count": sum(1 for s in ordered if float(s["net_bps"]) <= thr),
            "next1": split(next1),
            "next2": split(next2),
            "next3": split(next3),
        }
    return out


def session_specific(signals: list[dict[str, Any]], mk_ts: list[int], mk_px: list[float]) -> dict[str, Any]:
    out = {}
    for sess in sorted({s["row"].get("session") for s in signals}):
        sub = [s for s in signals if s["row"].get("session") == sess]
        out[str(sess)] = {
            "summary": split(sub),
            "exit": horizon_exit_suite(sub, mk_ts, mk_px, include_by_arm=False),
            "by_side": {
                "long": split([s for s in sub if s["side"] == "LONG"]),
                "short": split([s for s in sub if s["side"] == "SHORT"]),
            },
        }
    return out


def volatility_context(signals: list[dict[str, Any]], mk_ts: list[int], mk_px: list[float]) -> dict[str, Any]:
    enriched = []
    for s in signals:
        ts = int(s["anchor_ts_ms"])
        p0 = mark_at_or_after(mk_ts, mk_px, ts - 3600_000)
        hi = mark_max(mk_ts, mk_px, ts - 3600_000, ts)
        lo = mark_min(mk_ts, mk_px, ts - 3600_000, ts)
        if p0 and hi and lo:
            vol = (hi - lo) / p0 * 10_000.0
            enriched.append({**s, "pre1h_range_bps": vol})
    med = median([s["pre1h_range_bps"] for s in enriched]) if enriched else 0
    return {
        "median_pre1h_range_bps": round(med, 1) if enriched else None,
        "high_vol": split([s for s in enriched if s["pre1h_range_bps"] >= med]),
        "low_vol": split([s for s in enriched if s["pre1h_range_bps"] < med]),
    }


def btc_eth_divergence(signals: list[dict[str, Any]], mk_ts: list[int], mk_px: list[float]) -> dict[str, Any]:
    # ETH 30m shift exists in rows. Rebuild BTC 30m shift from DB marks.
    ts_list = [int(s["anchor_ts_ms"]) for s in signals]
    with sqlite3.connect(f"file:{DEFAULT_DB}?mode=ro", uri=True) as conn:
        rows = conn.execute(
            "SELECT ts_ms, mark_price FROM mark_prices WHERE symbol='BTCUSDT' AND ts_ms>=? AND ts_ms<=? ORDER BY ts_ms",
            (min(ts_list) - 3600_000, max(ts_list) + 60_000),
        ).fetchall() if ts_list else []
    btc_ts = [int(r[0]) for r in rows]
    btc_px = [float(r[1]) for r in rows]
    enriched = []
    for s in signals:
        ts = int(s["anchor_ts_ms"])
        b0 = mark_at_or_after(btc_ts, btc_px, ts - 30 * 60_000)
        b1 = mark_at_or_after(btc_ts, btc_px, ts)
        btc30 = ((b1 - b0) / b0 * 10_000.0) if b0 and b1 and b0 > 0 else 0.0
        eth30 = float(s["row"].get("eth_shift_30_bps") or 0.0)
        div = eth30 - btc30
        enriched.append({**s, "btc30_bps": btc30, "eth_minus_btc_30m": div})
    return {
        "eth_weaker_than_btc": split([s for s in enriched if s["eth_minus_btc_30m"] < -20]),
        "eth_inline": split([s for s in enriched if -20 <= s["eth_minus_btc_30m"] <= 20]),
        "eth_stronger_than_btc": split([s for s in enriched if s["eth_minus_btc_30m"] > 20]),
    }


def confidence_sizing(signals: list[dict[str, Any]], mk_ts: list[int], mk_px: list[float]) -> dict[str, Any]:
    enriched = []
    for s in signals:
        mm = mfe_mae_for_signal(s, mk_ts, mk_px, 5 * 60_000)
        early_ok = bool(mm and mm["mfe_bps"] >= 20)
        score = int(s.get("score") or 0)
        conf = 0
        conf += int(score >= 4)
        conf += int(early_ok)
        conf += int(s["arm"] == "SILENCE_LONG" and s["row"].get("vd", 0) >= 30)
        conf += int(s["arm"] == "NEITHER_SHORT" and (int(s["entry_ts_ms"]) - int(s["anchor_ts_ms"])) <= 15 * 60_000)
        conf += int(abs(float(s["row"].get("b4h") or 0)) >= 50)
        enriched.append({**s, "confidence": conf})
    bins = {f"conf_{i}": [s for s in enriched if s["confidence"] == i] for i in range(0, 6)}
    bins["conf_ge3"] = [s for s in enriched if s["confidence"] >= 3]
    # Sizing simulation: 0/1=0.5x, 2=1x, 3=1.25x, 4+=1.5x.
    sized = []
    for s in enriched:
        c = int(s["confidence"])
        mult = 0.5 if c <= 1 else 1.0 if c == 2 else 1.25 if c == 3 else 1.5
        sized.append({**s, "net_bps": float(s["net_bps"]) * mult, "size_mult": mult})
    return {
        "bins": {k: split(v) for k, v in bins.items()},
        "sized_counterfactual": split(sized),
        "avg_size_mult": avg([s["size_mult"] for s in sized]),
    }


def render_stat(name: str, s: dict[str, Any]) -> str:
    return f"{name}: N={s.get('n', 0)} WR={'' if s.get('wr') is None else round(float(s['wr'])*100,1)}% sum={s.get('sum')} med={s.get('median')} T3R={s.get('t3r')} maxLoss={s.get('max_loss')} DD={s.get('max_dd_bps')}"


def render_md(r: dict[str, Any]) -> str:
    lines = [
        "# S34 State Machine V6 Development Ideas",
        "",
        f"- generated_at_utc: `{r['generated_at_utc']}`",
        "- research_only: `true`",
        f"- primary_hold: `{r['primary']['hold']}`",
        "",
        "## Executive Read",
        "",
        "- No live changes made.",
        "- Best immediate development lead: early momentum observer + arm-specific exit research.",
        "- Best shadow-only expansion leads: score4 and BTC750; neither should replace the live rule yet.",
        "",
        "## Key Results",
        "",
        f"- early_5m_fav_ge20: `{r['top5']['early_momentum_observer']['5m']['fav_ge_20']['hold']}`",
        f"- early_5m_fav_lt20: `{r['top5']['early_momentum_observer']['5m']['fav_lt_20']['hold']}`",
        f"- score4_shadow_hold: `{r['top5']['score4_shadow']['summary']['hold']}`",
        f"- btc750_shadow_hold: `{r['top5']['btc750_shadow']['summary']['hold']}`",
        f"- confidence_sized_hold: `{r['ideas']['confidence_sizing']['sized_counterfactual']['hold']}`",
        "",
        "## Report JSON",
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
        "top5": {
            "early_momentum_observer": early_momentum_observer(signals, mk_ts, mk_px),
            "arm_specific_exit": arm_specific_exit(signals, mk_ts, mk_px),
            "regime_degradation_monitor": regime_degradation(signals),
            "score4_shadow": score4_shadow(rows, mk_ts, mk_px),
            "btc750_shadow": btc750_shadow(rows, mk_ts, mk_px),
        },
        "ideas": {
            "mfe_mae_exit_map": mfe_mae_exit_map(signals, mk_ts, mk_px),
            "profit_lock_observer": profit_lock(signals, mk_ts, mk_px),
            "short_replace_quality": short_replace_quality(raw),
            "long_silence_purity": long_silence_purity(signals),
            "btc_confirm_speed": btc_confirm_speed(signals),
            "post_tail_no_cooldown": post_tail_no_cooldown(signals),
            "session_specific_parameters": session_specific(signals, mk_ts, mk_px),
            "volatility_expansion_context": volatility_context(signals, mk_ts, mk_px),
            "btc_eth_divergence": btc_eth_divergence(signals, mk_ts, mk_px),
            "confidence_sizing": confidence_sizing(signals, mk_ts, mk_px),
        },
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    OUT_MD.write_text(render_md(report), encoding="utf-8")
    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_MD}")
    print(json.dumps({
        "primary_hold": report["primary"]["hold"],
        "early_5m_fav_ge20_hold": report["top5"]["early_momentum_observer"]["5m"]["fav_ge_20"]["hold"],
        "score4_shadow_hold": report["top5"]["score4_shadow"]["summary"]["hold"],
        "btc750_shadow_hold": report["top5"]["btc750_shadow"]["summary"]["hold"],
        "confidence_sized_hold": report["ideas"]["confidence_sizing"]["sized_counterfactual"]["hold"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
