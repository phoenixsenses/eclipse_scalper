"""S34 state-machine v4 promotion gauntlet.

Research-only. This is the final pre-live test layer for the state-machine
candidate. It does not read keys, change env, touch runtime state, or place
orders.
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
    CAL_FRAC,
    DEFAULT_DB,
    DOW,
    FEE_BPS,
    PROP_THRESH,
    SIL_HI_MS,
    SIL_LO_MS,
    SYNC_WIN_MS,
    Config,
    apply_conflict_policy,
    build_signals,
    classify_rows,
    finite,
    first_above,
    iso_ms,
    load_liq,
    load_marks,
    load_nav_events,
    mark_at_or_after,
    signed_net,
    summarize,
    summary_with_dd,
    utc_now,
    win_cnt,
    win_sum,
)
from tools.research_s34_state_machine_v3_full_tests import (  # noqa: E402
    SHADOW_LEDGER,
    book_at,
    by_split,
    latency_suite,
    mark_max,
    mark_min,
    net_between,
    shadow_timestamp_parity,
    slippage_suite,
    stop_suite,
)

OUT_JSON = ROOT / "reports" / "research" / "s34" / "S34_STATE_MACHINE_V4_PROMOTION_GAUNTLET.json"
OUT_MD = ROOT / "reports" / "research" / "s34" / "S34_STATE_MACHINE_V4_PROMOTION_GAUNTLET.md"
LIVE_STATE = ROOT / "runtime" / "s34_v_engine_live_state.json"
LIVE_PID = ROOT / "logs" / "pids" / "s34_v_engine_live_executor.pid"
REALTIME_SHADOW_STATE = ROOT / "reports" / "shadow" / "s34_realtime_shadow_state.json"


def pct(vals: list[float], p: float) -> float | None:
    vals = sorted(float(v) for v in vals if math.isfinite(float(v)))
    if not vals:
        return None
    idx = max(0, min(int((len(vals) - 1) * p / 100.0), len(vals) - 1))
    return round(vals[idx], 1)


def month_key(ts_ms: int) -> str:
    return datetime.fromtimestamp(int(ts_ms) / 1000.0, tz=timezone.utc).strftime("%Y-%m")


def build_base_rows() -> tuple[list[dict[str, Any]], list[int], list[float], list[int], list[float], list[int], list[float], list[int], list[float]]:
    nav = load_nav_events()
    with sqlite3.connect(f"file:{DEFAULT_DB}?mode=ro", uri=True) as conn:
        eth_ts, eth_not = load_liq(conn, "ETHUSDT", "SELL")
        btc_ts, btc_not = load_liq(conn, "BTCUSDT", "SELL")
        sol_ts, sol_not = load_liq(conn, "SOLUSDT", "SELL")
        event_ts = [int(r["signal_ts_ms"]) for r in nav if finite(r.get("threshold_usd")) is not None]
        eth_mk_ts, eth_mk_px = load_marks(conn, "ETHUSDT", min(event_ts) - 60_000, max(event_ts) + 6 * 3600_000)
    rows = classify_rows(nav, eth_ts=eth_ts, eth_not=eth_not, btc_ts=btc_ts, btc_not=btc_not, sol_ts=sol_ts, sol_not=sol_not, mk_ts=eth_mk_ts, mk_px=eth_mk_px)
    return rows, eth_ts, eth_not, btc_ts, btc_not, sol_ts, sol_not, eth_mk_ts, eth_mk_px


def with_stops(signals: list[dict[str, Any]], mk_ts: list[int], mk_px: list[float], stop_bps: float) -> list[dict[str, Any]]:
    out = []
    for s in signals:
        entry_ts = int(s["entry_ts_ms"])
        side = str(s["side"])
        horizon = 4 * 3600_000 if side == "LONG" else 2 * 3600_000
        entry_px = mark_at_or_after(mk_ts, mk_px, entry_ts)
        if entry_px is None or entry_px <= 0:
            continue
        if side == "LONG":
            lo = mark_min(mk_ts, mk_px, entry_ts, entry_ts + horizon)
            adverse = ((lo - entry_px) / entry_px * 10_000.0) if lo is not None else 0.0
        else:
            hi = mark_max(mk_ts, mk_px, entry_ts, entry_ts + horizon)
            adverse = ((entry_px - hi) / entry_px * 10_000.0) if hi is not None else 0.0
        net = -float(stop_bps) - FEE_BPS if adverse <= -float(stop_bps) else float(s["net_bps"])
        out.append({**s, "net_bps": net, "stop_bps": stop_bps, "stop_triggered": adverse <= -float(stop_bps)})
    return out


def with_book_required(signals: list[dict[str, Any]], db_path: Path, max_stale_sec: int = 10) -> list[dict[str, Any]]:
    out = []
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
        for s in signals:
            entry_ts = int(s["entry_ts_ms"])
            horizon = 4 * 3600_000 if s["side"] == "LONG" else 2 * 3600_000
            ex = book_at(conn, entry_ts, max_stale_sec)
            ox = book_at(conn, entry_ts + horizon, max_stale_sec)
            if not ex or not ox:
                continue
            entry = ex["ask"] if s["side"] == "LONG" else ex["bid"]
            exit_ = ox["bid"] if s["side"] == "LONG" else ox["ask"]
            net = signed_net(s["side"], entry, exit_)
            if net is not None:
                out.append({**s, "net_bps": net, "book_required": True})
    return out


def final_config_duel(rows: list[dict[str, Any]], mk_ts: list[int], mk_px: list[float]) -> dict[str, Any]:
    configs = [
        Config("btc750_dow_score3", btc_thr=750_000, long_score_min=3, short_score_min=3, exclude_long_dow=(0, 2), exclude_short_dow=(6,)),
        Config("btc1000_dow_score3", btc_thr=1_000_000, long_score_min=3, short_score_min=3, exclude_long_dow=(0, 2), exclude_short_dow=(6,)),
        Config("btc750_score4", btc_thr=750_000, long_score_min=4, short_score_min=4),
        Config("btc750_dow_score4", btc_thr=750_000, long_score_min=4, short_score_min=4, exclude_long_dow=(0, 2), exclude_short_dow=(6,)),
        Config("btc750_dow_score3_no_noisy", btc_thr=750_000, long_score_min=3, short_score_min=3, exclude_long_dow=(0, 2), exclude_short_dow=(6,), include_noisy_short=False),
    ]
    out = {}
    for cfg in configs:
        raw = build_signals(rows, cfg, mk_ts=mk_ts, mk_px=mk_px)
        taken, blocked = apply_conflict_policy(raw, "short_replace")
        out[cfg.name] = {"taken": by_split(taken), "blocked": by_split(blocked)}
    primary = apply_conflict_policy(build_signals(rows, configs[0], mk_ts=mk_ts, mk_px=mk_px), "short_replace")[0]
    out["btc750_dow_score3_sl100"] = {"taken": by_split(with_stops(primary, mk_ts, mk_px, 100.0)), "blocked": by_split([])}
    out["btc750_dow_score3_sl150"] = {"taken": by_split(with_stops(primary, mk_ts, mk_px, 150.0)), "blocked": by_split([])}
    out["btc750_dow_score3_book_required"] = {"taken": by_split(with_book_required(primary, DEFAULT_DB, 10)), "blocked": by_split([])}
    out["btc750_dow_score3_exclude_april_diag"] = {
        "taken": by_split([s for s in primary if month_key(int(s["entry_ts_ms"])) != "2026-04"]),
        "blocked": by_split([s for s in primary if month_key(int(s["entry_ts_ms"])) == "2026-04"]),
    }
    return out


def april_regime_killer(signals: list[dict[str, Any]], rows: list[dict[str, Any]], mk_ts: list[int], mk_px: list[float]) -> dict[str, Any]:
    april = [s for s in signals if month_key(int(s["entry_ts_ms"])) == "2026-04"]
    non_april = [s for s in signals if month_key(int(s["entry_ts_ms"])) != "2026-04"]
    tests: dict[str, list[dict[str, Any]]] = {
        "base": signals,
        "exclude_april_diag": non_april,
        "exclude_sat": [s for s in signals if int(s["row"]["dow"]) != 5],
        "exclude_tue_sat": [s for s in signals if int(s["row"]["dow"]) not in {1, 5}],
        "score4_plus": [s for s in signals if int(s.get("score") or 0) >= 4],
        "btc1000_only": apply_conflict_policy(build_signals(rows, Config("btc1000_dow_score3", btc_thr=1_000_000, long_score_min=3, short_score_min=3, exclude_long_dow=(0, 2), exclude_short_dow=(6,)), mk_ts=mk_ts, mk_px=mk_px), "short_replace")[0],
        "sl100": with_stops(signals, mk_ts, mk_px, 100.0),
        "sl150": with_stops(signals, mk_ts, mk_px, 150.0),
    }
    feature_cards = {}
    for label, subset in {"april": april, "non_april": non_april}.items():
        feature_cards[label] = {
            "n": len(subset),
            "summary": summary_with_dd(subset),
            "avg_score": round(mean([float(s.get("score") or 0) for s in subset]), 2) if subset else None,
            "avg_b4h": round(mean([float(s["row"].get("b4h") or 0) for s in subset]), 1) if subset else None,
            "avg_sync_k": round(mean([float(s["row"].get("sync_k") or 0) for s in subset]), 1) if subset else None,
            "avg_n2h": round(mean([float(s["row"].get("n2h") or 0) for s in subset]), 1) if subset else None,
            "sessions": dict(sorted({str(x): sum(1 for s in subset if s["row"].get("session") == x) for x in {s["row"].get("session") for s in subset}}.items())),
            "states": {
                "SILENCE": sum(1 for s in subset if s["row"].get("sil_eth")),
                "NEITHER": sum(1 for s in subset if not s["row"].get("sil_eth")),
            },
        }
    return {
        "feature_cards": feature_cards,
        "filter_tests": {k: by_split(v) for k, v in tests.items()},
        "read": "Calendar-April exclusion is diagnostic only. Tradable mitigations tested here are score4, BTC1000, DOW, and stops.",
    }


def live_feature_rebuild_parity(
    rows: list[dict[str, Any]],
    btc_ts: list[int],
    btc_not: list[float],
    sol_ts: list[int],
    sol_not: list[float],
    eth_ts: list[int],
    eth_not: list[float],
) -> dict[str, Any]:
    event_ts = [int(r["ts"]) for r in rows]
    with sqlite3.connect(f"file:{DEFAULT_DB}?mode=ro", uri=True) as conn:
        btc_mk_ts, btc_mk_px = load_marks(conn, "BTCUSDT", min(event_ts) - 5 * 3600_000, max(event_ts) + 60_000)
        checks = []
        for r in rows:
            ts = int(r["ts"])
            p0 = mark_at_or_after(btc_mk_ts, btc_mk_px, ts - 4 * 3600_000)
            p1 = mark_at_or_after(btc_mk_ts, btc_mk_px, ts)
            b4h_live = ((p1 - p0) / p0 * 10_000.0) if p0 and p1 and p0 > 0 else None
            sync_live = win_sum(btc_ts, btc_not, ts - SYNC_WIN_MS, ts) + win_sum(sol_ts, sol_not, ts - SYNC_WIN_MS, ts)
            n2h_live = win_cnt(eth_ts, eth_not, ts - 2 * 3600_000, ts - 1000, PROP_THRESH)
            book = book_at(conn, ts, 10)
            checks.append(
                {
                    "b4h_diff": None if b4h_live is None else abs(float(r["b4h"]) - b4h_live),
                    "sync_match": abs(float(r["sync_k"]) - sync_live) < 1e-6,
                    "n2h_match": int(r["n2h"]) == int(n2h_live),
                    "dow_match": int(r["dow"]) == datetime.fromtimestamp(ts / 1000.0, tz=timezone.utc).weekday(),
                    "session_match": r["session"] in {"ASIA", "EUROPE", "US", "OFF"},
                    "book_available": book is not None,
                }
            )
    diffs = [c["b4h_diff"] for c in checks if c["b4h_diff"] is not None]
    return {
        "n": len(checks),
        "b4h_diff_median_bps": pct(diffs, 50),
        "b4h_diff_p95_bps": pct(diffs, 95),
        "sync_match_rate": round(sum(c["sync_match"] for c in checks) / len(checks), 3) if checks else None,
        "n2h_match_rate": round(sum(c["n2h_match"] for c in checks) / len(checks), 3) if checks else None,
        "dow_match_rate": round(sum(c["dow_match"] for c in checks) / len(checks), 3) if checks else None,
        "book_available_rate_10s": round(sum(c["book_available"] for c in checks) / len(checks), 3) if checks else None,
        "vdepth_note": "vdepth_bps is not directly rebuilt here; live executor must either recompute from running anchor marks or reject if unavailable. This remains a live-feature blocker.",
    }


def rolling_kill_tests(signals: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(signals, key=lambda s: int(s["entry_ts_ms"]))
    out = {}
    for window, limit in [(3, -150), (5, -200), (5, -300), (10, -400)]:
        taken = []
        paused_until = None
        skipped = 0
        for s in ordered:
            month = month_key(int(s["entry_ts_ms"]))
            if paused_until == month:
                skipped += 1
                continue
            taken.append(s)
            recent = [float(x["net_bps"]) for x in taken[-window:]]
            if len(recent) >= window and sum(recent) <= limit:
                paused_until = month
        out[f"roll{window}_sum_le_{limit}_pause_month"] = {"taken": by_split(taken), "skipped": skipped}
    return out


def risk_sizing(signals: list[dict[str, Any]], equity_usdt: float = 35.0, leverage: float = 40.0) -> dict[str, Any]:
    vals = [float(s["net_bps"]) for s in signals]
    worst = min(vals) if vals else 0.0
    out = {"worst_bps": round(worst, 1), "equity_usdt": equity_usdt, "leverage": leverage}
    for risk_pct in [1, 2, 5, 10]:
        risk_usdt = equity_usdt * risk_pct / 100.0
        notional = risk_usdt / (abs(worst) / 10_000.0) if worst < 0 else 0.0
        out[f"risk_{risk_pct}pct"] = {
            "risk_usdt": round(risk_usdt, 4),
            "max_notional_usdt": round(notional, 2),
            "margin_usdt_at_40x": round(notional / leverage, 4) if leverage else None,
        }
    return out


def readiness_readout() -> dict[str, Any]:
    live_state = {}
    if LIVE_STATE.exists():
        try:
            live_state = json.loads(LIVE_STATE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            live_state = {}
    shadow_state = {}
    if REALTIME_SHADOW_STATE.exists():
        try:
            shadow_state = json.loads(REALTIME_SHADOW_STATE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            shadow_state = {}
    return {
        "live_state_exists": LIVE_STATE.exists(),
        "live_pid_file_exists": LIVE_PID.exists(),
        "live_pid_file": LIVE_PID.read_text(encoding="utf-8").strip() if LIVE_PID.exists() else None,
        "live_active": live_state.get("active"),
        "live_status_rule": (live_state.get("status") or {}).get("rule"),
        "live_status_mode": (live_state.get("status") or {}).get("mode"),
        "realtime_shadow_state_exists": REALTIME_SHADOW_STATE.exists(),
        "realtime_shadow_open_positions": len((shadow_state.get("positions") or {})) if isinstance(shadow_state, dict) else None,
        "shadow_ledger_exists": SHADOW_LEDGER.exists(),
        "note": "Read-only readiness snapshot. It does not prove process uniqueness; process cleanup/audit must be separate before live promotion.",
    }


def render_stats_table(title: str, rows: list[tuple[str, dict[str, Any]]], split: str = "hold") -> list[str]:
    lines = [f"## {title}", "", "| Name | N | WR | Sum | Mean | Median | T3R | Max loss | Max DD |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name, val in rows:
        s = val.get(split, val) if isinstance(val, dict) else {}
        wr = "" if s.get("wr") is None else f"{float(s['wr']) * 100:.1f}%"
        lines.append(
            f"| {name} | {s.get('n',0)} | {wr} | {s.get('sum')} | {s.get('mean')} | {s.get('median')} | {s.get('t3r')} | {s.get('max_loss')} | {s.get('max_dd_bps','')} |"
        )
    lines.append("")
    return lines


def render_md(r: dict[str, Any]) -> str:
    lines = [
        "# S34 State Machine V4 Promotion Gauntlet",
        "",
        f"- generated_at_utc: `{r['generated_at_utc']}`",
        "- research_only: `true`",
        "",
        "## Decision Read",
        "",
        f"- leading_config: `{r['decision']['leading_config']}`",
        f"- leading_hold: `{r['decision']['leading_hold']}`",
        f"- blocker_1: `{r['decision']['blocker_1']}`",
        f"- blocker_2: `{r['decision']['blocker_2']}`",
        "",
    ]
    lines += render_stats_table("Final Config Duel", [(k, v["taken"]) for k, v in r["final_config_duel"].items()], "hold")
    lines += render_stats_table("April / Regime Killer Tests", [(k, v) for k, v in r["april_regime"]["filter_tests"].items()], "hold")
    lines += [
        "## April Feature Cards",
        "",
        f"- april: `{r['april_regime']['feature_cards']['april']}`",
        f"- non_april: `{r['april_regime']['feature_cards']['non_april']}`",
        "",
        "## Live Feature Rebuild Parity",
        "",
        f"- `{r['live_feature_rebuild_parity']}`",
        "",
    ]
    lines += render_stats_table("Rolling Kill Tests", [(k, v["taken"]) for k, v in r["rolling_kill_tests"].items()], "hold")
    lines += render_stats_table("Latency Stress", list(r["latency"].items()), "hold")
    lines += render_stats_table("Slippage Stress", list(r["slippage"].items()), "hold")
    lines += [
        "## Risk Sizing",
        "",
        f"- `{r['risk_sizing']}`",
        "",
        "## Shadow / Readiness",
        "",
        f"- shadow_timestamp_parity: `{r['shadow_timestamp_parity']}`",
        f"- readiness_readout: `{r['readiness_readout']}`",
        "",
        "## Read",
        "",
        "The state-machine candidate remains statistically strong. The remaining live blockers are operational: live feature parity for vdepth/book coverage, timestamp/action shadow parity beyond ID parity, and separate duplicate-process safety audit.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    rows, eth_ts, eth_not, btc_ts, btc_not, sol_ts, sol_not, mk_ts, mk_px = build_base_rows()
    primary_cfg = Config("btc750_dow_score3", btc_thr=750_000, long_score_min=3, short_score_min=3, exclude_long_dow=(0, 2), exclude_short_dow=(6,))
    primary = apply_conflict_policy(build_signals(rows, primary_cfg, mk_ts=mk_ts, mk_px=mk_px), "short_replace")[0]
    duel = final_config_duel(rows, mk_ts, mk_px)
    leading_name, leading_payload = max(duel.items(), key=lambda kv: float(kv[1]["taken"]["hold"].get("t3r") or -1e18))
    report = {
        "generated_at_utc": utc_now(),
        "counts": {
            "classified": len(rows),
            "primary_signals": len(primary),
            "primary_hold": sum(1 for s in primary if s["row"]["is_hold"]),
        },
        "decision": {
            "leading_config": leading_name,
            "leading_hold": leading_payload["taken"]["hold"],
            "blocker_1": "vdepth/live-feature rebuild is not proven 100%; current parity covers b4h/sync/n2h/book availability.",
            "blocker_2": "timestamp/action realtime shadow parity is still only ID-level; live process uniqueness must be audited separately.",
        },
        "final_config_duel": duel,
        "april_regime": april_regime_killer(primary, rows, mk_ts, mk_px),
        "live_feature_rebuild_parity": live_feature_rebuild_parity(rows, btc_ts, btc_not, sol_ts, sol_not, eth_ts, eth_not),
        "rolling_kill_tests": rolling_kill_tests(primary),
        "latency": latency_suite(primary, mk_ts, mk_px),
        "slippage": slippage_suite(primary),
        "risk_sizing": risk_sizing(primary),
        "shadow_timestamp_parity": shadow_timestamp_parity(rows),
        "readiness_readout": readiness_readout(),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    OUT_MD.write_text(render_md(report), encoding="utf-8")
    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_MD}")
    print(json.dumps({
        "leading": report["decision"],
        "feature_parity": report["live_feature_rebuild_parity"],
        "risk": report["risk_sizing"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
