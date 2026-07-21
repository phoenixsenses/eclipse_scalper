"""S34 state-machine v7 full development suite.

Research-only. This script does not touch the live executor, env, runtime state,
orders, buckets, or dashboard. It expands the V6 idea suite into wider grids for
entry navigation, exit management, score/threshold dose-response, and regime
monitoring.
"""

from __future__ import annotations

import bisect
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.research_s34_state_machine_v2_gauntlet import (  # noqa: E402
    FEE_BPS,
    Config,
    apply_conflict_policy,
    build_signals,
    mark_at_or_after,
    summary_with_dd,
)
from tools.research_s34_state_machine_v4_promotion_gauntlet import build_base_rows  # noqa: E402
from tools.research_s34_state_machine_v6_development_ideas import (  # noqa: E402
    FINAL_CFG,
    arm,
    btc750_shadow,
    btc_eth_divergence,
    confidence_sizing,
    iso_ms,
    mfe_mae_for_signal,
    score4_shadow,
    split,
    volatility_context,
)


OUT_JSON = ROOT / "reports" / "research" / "s34" / "S34_STATE_MACHINE_V7_FULL_DEVELOPMENT_SUITE.json"
OUT_MD = ROOT / "reports" / "research" / "s34" / "S34_STATE_MACHINE_V7_FULL_DEVELOPMENT_SUITE.md"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def avg(vals: list[float]) -> float | None:
    vals = [float(v) for v in vals if math.isfinite(float(v))]
    return round(mean(vals), 2) if vals else None


def stat_line(s: dict[str, Any]) -> str:
    wr = s.get("wr")
    wrs = "NA" if wr is None else f"{float(wr) * 100:.1f}%"
    return (
        f"N={s.get('n')} WR={wrs} sum={s.get('sum')} mean={s.get('mean')} "
        f"med={s.get('median')} T3R={s.get('t3r')} maxL={s.get('max_loss')} DD={s.get('max_dd_bps')}"
    )


def net_at_horizon(s: dict[str, Any], mk_ts: list[int], mk_px: list[float], horizon_min: float) -> dict[str, Any] | None:
    entry_ts = int(s["entry_ts_ms"])
    entry = mark_at_or_after(mk_ts, mk_px, entry_ts)
    exit_px = mark_at_or_after(mk_ts, mk_px, entry_ts + int(horizon_min * 60_000))
    if not entry or not exit_px or entry <= 0:
        return None
    raw = (exit_px - entry) / entry * 10_000.0
    net = -raw if str(s["side"]).upper() == "SHORT" else raw
    return {**s, "net_bps": round(net - FEE_BPS, 1), "exit_horizon_min": horizon_min}


def horizon_grid(signals: list[dict[str, Any]], mk_ts: list[int], mk_px: list[float]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for label, subset in {
        "all": signals,
        "long": [s for s in signals if s["side"] == "LONG"],
        "short": [s for s in signals if s["side"] == "SHORT"],
        "long_silence": arm(signals, "SILENCE_LONG"),
        "short_neither": arm(signals, "NEITHER_SHORT"),
    }.items():
        out[label] = {}
        for h in [30, 60, 90, 120, 150, 180, 240, 360]:
            rows = [r for s in subset if (r := net_at_horizon(s, mk_ts, mk_px, h))]
            out[label][f"{h}m"] = split(rows)
    return out


def early_momentum_grid(signals: list[dict[str, Any]], mk_ts: list[int], mk_px: list[float]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for minute in [3, 5, 10, 15]:
        enriched = []
        for s in signals:
            mm = mfe_mae_for_signal(s, mk_ts, mk_px, minute * 60_000)
            if mm:
                enriched.append({**s, "early_mfe": mm["mfe_bps"], "early_mae": mm["mae_bps"]})
        out[f"{minute}m"] = {}
        for thr in [10, 20, 30, 50]:
            out[f"{minute}m"][f"fav_ge_{thr}"] = split([s for s in enriched if s["early_mfe"] >= thr])
            out[f"{minute}m"][f"fav_ge_{thr}_clean"] = split(
                [s for s in enriched if s["early_mfe"] >= thr and s["early_mae"] > -thr]
            )
            out[f"{minute}m"][f"weak_lt_{thr}"] = split([s for s in enriched if s["early_mfe"] < thr])
            out[f"{minute}m"][f"adverse_le_-{thr}"] = split([s for s in enriched if s["early_mae"] <= -thr])
    return out


def profit_lock_grid(signals: list[dict[str, Any]], mk_ts: list[int], mk_px: list[float]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for trig in [50, 75, 100, 125, 150]:
        for lock in [0, 25, 50, 75, 100]:
            if lock >= trig:
                continue
            managed = []
            triggered = 0
            for s in signals:
                entry_ts = int(s["entry_ts_ms"])
                side = str(s["side"]).upper()
                horizon = 240 if side == "LONG" else 120
                entry = mark_at_or_after(mk_ts, mk_px, entry_ts)
                if not entry:
                    continue
                a = bisect.bisect_left(mk_ts, entry_ts)
                b = bisect.bisect_right(mk_ts, entry_ts + horizon * 60_000)
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
                managed.append({**s, "net_bps": round(net, 1), "profit_lock_triggered": armed})
            key = f"trig{trig}_lock{lock}"
            out[key] = split(managed)
            out[key]["triggered"] = triggered
            out[key]["triggered_pct"] = round(triggered / len(signals), 3) if signals else 0.0
    return out


def score_quality_grid(signals: list[dict[str, Any]], mk_ts: list[int], mk_px: list[float]) -> dict[str, Any]:
    conf = confidence_sizing(signals, mk_ts, mk_px)
    out: dict[str, Any] = {"confidence": conf}
    for side in ["LONG", "SHORT"]:
        side_rows = [s for s in signals if s["side"] == side]
        out[f"{side.lower()}_score_bins"] = {
            f"score_{i}": split([s for s in side_rows if int(s.get("score") or 0) == i]) for i in range(1, 6)
        }
        out[f"{side.lower()}_score_ge"] = {
            f"score_ge_{i}": split([s for s in side_rows if int(s.get("score") or 0) >= i]) for i in range(2, 6)
        }
    return out


def btc_threshold_grid(rows: list[dict[str, Any]], mk_ts: list[int], mk_px: list[float]) -> dict[str, Any]:
    out = {}
    for thr in [500_000, 750_000, 1_000_000, 1_250_000, 1_500_000]:
        cfg = Config(
            f"btc{int(thr/1000)}k_dow_score3",
            btc_thr=thr,
            long_score_min=3,
            short_score_min=3,
            exclude_long_dow=(0, 2),
            exclude_short_dow=(6,),
        )
        sigs = apply_conflict_policy(build_signals(rows, cfg, mk_ts=mk_ts, mk_px=mk_px), "short_replace")[0]
        out[f"btc_{int(thr/1000)}k"] = {
            "summary": split(sigs),
            "long": split([s for s in sigs if s["side"] == "LONG"]),
            "short": split([s for s in sigs if s["side"] == "SHORT"]),
        }
    return out


def session_exit_grid(signals: list[dict[str, Any]], mk_ts: list[int], mk_px: list[float]) -> dict[str, Any]:
    out = {}
    for session in sorted({str(s["row"].get("session")) for s in signals}):
        sub = [s for s in signals if str(s["row"].get("session")) == session]
        out[session] = {
            "summary": split(sub),
            "horizons": horizon_grid(sub, mk_ts, mk_px),
        }
    return out


def vol_side_grid(signals: list[dict[str, Any]], mk_ts: list[int], mk_px: list[float]) -> dict[str, Any]:
    enriched = []
    for s in signals:
        ts = int(s["anchor_ts_ms"])
        p0 = mark_at_or_after(mk_ts, mk_px, ts - 3600_000)
        if not p0:
            continue
        a = bisect.bisect_left(mk_ts, ts - 3600_000)
        b = bisect.bisect_right(mk_ts, ts)
        if a >= b:
            continue
        hi = max(mk_px[a:b])
        lo = min(mk_px[a:b])
        enriched.append({**s, "pre1h_range_bps": (hi - lo) / p0 * 10_000.0})
    med = median([s["pre1h_range_bps"] for s in enriched]) if enriched else 0.0
    out = {"median_pre1h_range_bps": round(med, 1)}
    for vol_name, sub in {
        "low_vol": [s for s in enriched if s["pre1h_range_bps"] < med],
        "high_vol": [s for s in enriched if s["pre1h_range_bps"] >= med],
    }.items():
        out[vol_name] = {
            "summary": split(sub),
            "long": split([s for s in sub if s["side"] == "LONG"]),
            "short": split([s for s in sub if s["side"] == "SHORT"]),
        }
    return out


def regime_pause_sim(signals: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(signals, key=lambda s: int(s["entry_ts_ms"]))
    out = {}
    for window in [3, 5, 10]:
        for sum_thr in [-50, -100, -150, -200]:
            for pause_n in [1, 2, 3]:
                taken = []
                skipped = []
                pause_left = 0
                hist: list[dict[str, Any]] = []
                triggers = 0
                for s in ordered:
                    if pause_left > 0:
                        skipped.append(s)
                        pause_left -= 1
                        continue
                    taken.append(s)
                    hist.append(s)
                    if len(hist) >= window:
                        recent = hist[-window:]
                        if sum(float(x["net_bps"]) for x in recent) <= sum_thr:
                            triggers += 1
                            pause_left = pause_n
                out[f"roll{window}_sum{sum_thr}_pause{pause_n}"] = {
                    "taken": split(taken),
                    "skipped": split(skipped),
                    "triggers": triggers,
                    "skipped_n": len(skipped),
                }
    return out


def shortlist(report: dict[str, Any]) -> dict[str, Any]:
    primary_hold = report["primary"]["hold"]
    candidates = {
        "primary_live_baseline": primary_hold,
        "score4_shadow": report["shadow_candidates"]["score4"]["summary"]["hold"],
        "btc750_shadow": report["shadow_candidates"]["btc750"]["summary"]["hold"],
        "early_5m_fav20": report["early_momentum"]["5m"]["fav_ge_20"]["hold"],
        "early_5m_fav20_clean": report["early_momentum"]["5m"]["fav_ge_20_clean"]["hold"],
        "profit_lock_trig100_lock50": report["profit_lock"]["trig100_lock50"]["hold"],
        "confidence_sized": report["score_quality"]["confidence"]["sized_counterfactual"]["hold"],
        "low_vol": report["volatility"]["low_vol"]["summary"]["hold"],
        "eth_weaker_than_btc": report["divergence"]["eth_weaker_than_btc"]["hold"],
    }
    ranked = []
    for name, st in candidates.items():
        ranked.append({
            "name": name,
            **st,
            "caution": "small_N" if int(st.get("n") or 0) < 20 else "candidate",
        })
    ranked.sort(key=lambda x: (float(x.get("t3r") or -1e9), float(x.get("sum") or -1e9)), reverse=True)
    return {"ranked_by_hold_t3r": ranked}


def render_md(report: dict[str, Any]) -> str:
    ranked = report["shortlist"]["ranked_by_hold_t3r"]
    lines = [
        "# S34 State Machine V7 Full Development Suite",
        "",
        f"- generated_at_utc: `{report['generated_at_utc']}`",
        "- research_only: `true`",
        "- live_changes: `none`",
        f"- primary_config: `{report['primary_config']}`",
        f"- primary_hold: `{stat_line(report['primary']['hold'])}`",
        "",
        "## Questions Tested",
        "",
        "1. Early momentum threshold grid.",
        "2. Profit-lock exit grid.",
        "3. Arm-specific horizon grid.",
        "4. Score/confidence monotonicity.",
        "5. BTC threshold dose-response.",
        "6. BTC/ETH divergence navigation.",
        "7. Regime pause/kill simulation.",
        "8. Session-specific management.",
        "9. Volatility context.",
        "10. Shadow-candidate shortlist.",
        "",
        "## Shortlist By Holdout T3R",
        "",
    ]
    for r in ranked:
        lines.append(f"- {r['name']}: `{stat_line(r)}` caution={r['caution']}")
    lines += [
        "",
        "## Selected Results",
        "",
        f"- early 5m fav>=20: `{stat_line(report['early_momentum']['5m']['fav_ge_20']['hold'])}`",
        f"- early 5m fav>=20 clean: `{stat_line(report['early_momentum']['5m']['fav_ge_20_clean']['hold'])}`",
        f"- profit lock 100/50: `{stat_line(report['profit_lock']['trig100_lock50']['hold'])}`",
        f"- BTC 1000K: `{stat_line(report['btc_thresholds']['btc_1000k']['summary']['hold'])}`",
        f"- BTC 750K: `{stat_line(report['btc_thresholds']['btc_750k']['summary']['hold'])}`",
        f"- low vol: `{stat_line(report['volatility']['low_vol']['summary']['hold'])}`",
        f"- ETH weaker than BTC: `{stat_line(report['divergence']['eth_weaker_than_btc']['hold'])}`",
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

    report: dict[str, Any] = {
        "generated_at_utc": utc_now(),
        "research_only": True,
        "live_changes": "none",
        "primary_config": FINAL_CFG.name,
        "data": {
            "classified_rows": len(rows),
            "raw_signals": len(raw),
            "taken_signals": len(signals),
            "blocked_signals": len(blocked),
        },
        "primary": split(signals),
        "early_momentum": early_momentum_grid(signals, mk_ts, mk_px),
        "profit_lock": profit_lock_grid(signals, mk_ts, mk_px),
        "horizon_grid": horizon_grid(signals, mk_ts, mk_px),
        "score_quality": score_quality_grid(signals, mk_ts, mk_px),
        "btc_thresholds": btc_threshold_grid(rows, mk_ts, mk_px),
        "divergence": btc_eth_divergence(signals, mk_ts, mk_px),
        "regime_pause": regime_pause_sim(signals),
        "session_exit": session_exit_grid(signals, mk_ts, mk_px),
        "volatility": vol_side_grid(signals, mk_ts, mk_px),
        "shadow_candidates": {
            "score4": score4_shadow(rows, mk_ts, mk_px),
            "btc750": btc750_shadow(rows, mk_ts, mk_px),
        },
    }
    report["shortlist"] = shortlist(report)

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    OUT_MD.write_text(render_md(report), encoding="utf-8")

    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_MD}")
    print(json.dumps({
        "primary_hold": report["primary"]["hold"],
        "top_shortlist": report["shortlist"]["ranked_by_hold_t3r"][:8],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
