"""S34 V Engine forward management monitor.

Risk/observation only. This tool does not place orders, edit executor config, or
change live order logic. It consumes the existing v0.2 shadow mirror ledger and
adds the management layer requested in S34 phase plan:

- forward OOS ledger split from a frozen start timestamp;
- tail-aware sizing recommendation and oversize alert;
- defensive dissipation observer, shadow-only;
- regime degradation and explicit kill criteria;
- descriptive failure-mode tracking for large losses;
- ETH-only maker-provision execution realism scenarios.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.research_s34_knowable_anchor_continuation import iso_ms, r1, r3
from tools.research_s34_v6_management_system import (
    book_at_or_before,
    simulate_tick_queue_event,
    summarize_rows,
    tail_aware_sizing,
)
from tools.research_s34_wave_absorption import book_features_at


DEFAULT_DB = ROOT / "data" / "microstructure.db"
DEFAULT_ENV = ROOT / ".env"
SHADOW_LEDGER = ROOT / "reports" / "research" / "s34" / "S34_V_ENGINE_V0_2_SHADOW_MIRROR_LEDGER.jsonl"
SHADOW_BRIEF = ROOT / "reports" / "research" / "s34" / "S34_V_ENGINE_V0_2_SHADOW_MIRROR_BRIEF.json"
POOL_JSON = ROOT / "reports" / "research" / "s34" / "S34_ABSORPTION_SYNC_2X2_POOL.json"
V6_JSON = ROOT / "reports" / "research" / "s34" / "S34_V6_MANAGEMENT_SYSTEM.json"
STOP_JSON = ROOT / "reports" / "research" / "s34" / "S34_V_ENGINE_PROTECTIVE_STOP.json"
LIVE_STATE = ROOT / "runtime" / "s34_v_engine_live_state.json"
STATE_PATH = ROOT / "runtime" / "s34_v_engine_forward_management_state.json"
OUT_LEDGER = ROOT / "reports" / "research" / "s34" / "S34_V_ENGINE_FORWARD_MANAGEMENT_LEDGER.jsonl"
OUT_JSON = ROOT / "reports" / "research" / "s34" / "S34_V_ENGINE_FORWARD_MANAGEMENT_READOUT.json"
OUT_MD = ROOT / "reports" / "research" / "s34" / "S34_V_ENGINE_FORWARD_MANAGEMENT_READOUT.md"
OUT_FRAGMENT = ROOT / "reports" / "research" / "s34" / "S34_CASCADE_NAVIGATION_MANAGEMENT_FRAGMENT.json"

RULE_ID = "S34_V_ENGINE_V0_2_ETH_SELL_MAKER_LONG_H2_O20_W300_O5_DEEPBID"
MODE = "OBSERVATION_RISK_ONLY_NO_LIVE_ORDER_LOGIC_CHANGE"
DISSIPATION_REPLENISH_CUT = 10.7903
DISSIPATION_DECEL_CUT = 0.4737
STRESS_TAIL_BPS = 634.0
DEFAULT_STOP_REALIZED_BPS = 175.7


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def now_ms() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True), encoding="utf-8")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n")


def parse_env(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        out[key.strip()] = value.strip().strip('"').strip("'")
    return out


def fenv(env: dict[str, str], key: str, default: float) -> float:
    try:
        return float(env.get(key, default))
    except (TypeError, ValueError):
        return float(default)


def summary(vals: list[float]) -> dict[str, Any]:
    vals = [float(v) for v in vals if math.isfinite(float(v))]
    if not vals:
        return {"n": 0, "sum_bps": 0.0, "mean_bps": None, "median_bps": None, "win_rate": None, "max_loss_bps": None, "t3r_bps": 0.0}
    ordered = sorted(vals, reverse=True)
    t3r = ordered[3:] if len(ordered) > 3 else []
    return {
        "n": len(vals),
        "sum_bps": r1(sum(vals)),
        "mean_bps": r1(sum(vals) / len(vals)),
        "median_bps": r1(median(vals)),
        "win_rate": r3(sum(1 for v in vals if v > 0) / len(vals)),
        "max_loss_bps": r1(min(vals)),
        "t3r_bps": r1(sum(t3r)) if t3r else 0.0,
    }


def load_or_init_state(path: Path, *, freeze_start_ms: int | None) -> dict[str, Any]:
    state = load_json(path, {})
    if freeze_start_ms is not None:
        state["frozen_start_ms"] = int(freeze_start_ms)
        state["frozen_start_utc"] = iso_ms(int(freeze_start_ms))
    elif not state.get("frozen_start_ms"):
        ts = now_ms()
        state["frozen_start_ms"] = ts
        state["frozen_start_utc"] = iso_ms(ts)
    state["updated_at_utc"] = utc_now()
    return state


def planned_live_size(env: dict[str, str], *, equity_usdt: float) -> dict[str, Any]:
    leverage = int(fenv(env, "S34_LIVE_MAX_LEVERAGE", 40.0))
    margin_pct = fenv(env, "S34_LIVE_MARGIN_PCT_ETH", fenv(env, "S34_LIVE_MARGIN_PCT", 85.0))
    fallback_margin = fenv(env, "S34_LIVE_MARGIN_USDT", 30.0)
    margin = round(float(equity_usdt) * margin_pct / 100.0, 2) if equity_usdt > 0 else fallback_margin
    notional = margin * leverage
    stress_loss = notional * STRESS_TAIL_BPS / 10_000.0
    return {
        "equity_usdt_assumption": float(equity_usdt),
        "env_margin_pct_eth": margin_pct,
        "fallback_margin_usdt": fallback_margin,
        "leverage": leverage,
        "planned_margin_usdt": r1(margin),
        "planned_notional_usdt": r1(notional),
        "stress_tail_loss_usdt": r1(stress_loss),
        "stress_tail_loss_pct_equity": r1(100.0 * stress_loss / equity_usdt) if equity_usdt else None,
    }


def stop_realized_bps(stop_json: Path) -> float:
    payload = load_json(stop_json, {})
    for row in payload.get("summaries") or []:
        if row.get("variant") == "fixed_sl_150":
            summary = row.get("summary") or {}
            max_loss = summary.get("max_loss_bps")
            if max_loss is not None:
                return abs(float(max_loss))
    return DEFAULT_STOP_REALIZED_BPS


def max_notional_for_risk(*, equity_usdt: float, risk_pct: float, stress_tail_bps: float = STRESS_TAIL_BPS) -> float:
    return (float(equity_usdt) * float(risk_pct) / 100.0) / (float(stress_tail_bps) / 10_000.0)


def sizing_monitor(
    env: dict[str, str],
    pool_rows: list[dict[str, Any]],
    shadow_rows: list[dict[str, Any]],
    *,
    equity_usdt: float,
    risk_pct: float,
    stop_json: Path = STOP_JSON,
) -> dict[str, Any]:
    live = planned_live_size(env, equity_usdt=equity_usdt)
    tail = tail_aware_sizing(pool_rows, shadow_rows, equity_usdt=equity_usdt, leverage=float(live["leverage"]))
    max_notional = max_notional_for_risk(equity_usdt=equity_usdt, risk_pct=risk_pct)
    max_margin = max_notional / float(live["leverage"])
    stop_bps = stop_realized_bps(stop_json)
    max_stop_notional = max_notional_for_risk(equity_usdt=equity_usdt, risk_pct=risk_pct, stress_tail_bps=stop_bps)
    max_stop_margin = max_stop_notional / float(live["leverage"])
    planned_notional = float(live["planned_notional_usdt"] or 0.0)
    oversize_multiple = planned_notional / max_notional if max_notional > 0 else None
    stop_oversize_multiple = planned_notional / max_stop_notional if max_stop_notional > 0 else None
    return {
        "status": "ALERT_OVERSIZE_OPERATOR_ACTION_REQUIRED" if oversize_multiple and oversize_multiple > 1.0 else "OK_WITHIN_TAIL_BUDGET",
        "action": "RECOMMENDATION_ONLY_NO_AUTO_SIZE_CHANGE",
        "risk_pct_equity": float(risk_pct),
        "leverage_kept": float(live["leverage"]),
        "max_tail_budget_notional_usdt": r1(max_notional),
        "max_tail_budget_margin_usdt": r1(max_margin),
        "stop_realized_bps_abs": r1(stop_bps),
        "max_stop_budget_notional_usdt": r1(max_stop_notional),
        "max_stop_budget_margin_usdt": r1(max_stop_margin),
        "planned_live_size_from_env": live,
        "oversize_multiple": r1(oversize_multiple) if oversize_multiple is not None else None,
        "stop_budget_oversize_multiple": r1(stop_oversize_multiple) if stop_oversize_multiple is not None else None,
        "v6_tail_reference": tail,
    }


def liq_notional(conn: sqlite3.Connection, symbol: str, side: str, start_ms: int, end_ms: int) -> float:
    row = conn.execute(
        "SELECT COALESCE(SUM(notional),0) FROM liquidations WHERE symbol=? AND side=? AND ts_ms>? AND ts_ms<=?",
        (symbol, side, int(start_ms), int(end_ms)),
    ).fetchone()
    return float(row[0] or 0.0)


def dissipation_at_tau(conn: sqlite3.Connection, row: dict[str, Any], tau_sec: int) -> dict[str, Any]:
    ts = int(row["signal_ts_ms"])
    symbol = str(row.get("symbol") or "ETHUSDT")
    side = str(row.get("liq_side") or "SELL")
    base_bid = float(row.get("bid_depth_usd") or 0.0)
    book = book_features_at(conn, symbol, ts + int(tau_sec) * 1000, 10)
    tau_bid = float((book or {}).get("bid_depth_usd") or 0.0)
    replenish = ((tau_bid - base_bid) / base_bid * 100.0) if base_bid > 0 else None
    pre_rate = float(row.get("running_rate_usd_per_sec") or 0.0)
    post_rate = liq_notional(conn, symbol, side, ts, ts + int(tau_sec) * 1000) / max(1.0, float(tau_sec))
    decel = (1.0 - (post_rate / pre_rate)) if pre_rate > 0 else None
    if replenish is None or decel is None:
        recommendation = "OBSERVE_ONLY_DATA_INCOMPLETE"
    elif tau_sec == 120 and replenish >= DISSIPATION_REPLENISH_CUT and decel >= DISSIPATION_DECEL_CUT:
        recommendation = "HOLD_SHADOW_ONLY"
    elif tau_sec == 120 and replenish < DISSIPATION_REPLENISH_CUT and decel < DISSIPATION_DECEL_CUT:
        recommendation = "EXIT_OR_TIGHTEN_SHADOW_ONLY"
    else:
        recommendation = "MIXED_SHADOW_ONLY"
    return {
        "tau_sec": int(tau_sec),
        "book_available": bool(book),
        "bid_depth_tau_usd": r1(tau_bid) if book else None,
        "replenish_pct": r1(replenish) if replenish is not None and math.isfinite(replenish) else None,
        "post_liq_rate_usd_per_sec": r1(post_rate),
        "liq_deceleration": r3(decel) if decel is not None and math.isfinite(decel) else None,
        "recommendation": recommendation,
    }


def classify_loss(row: dict[str, Any]) -> str:
    net = row.get("net_bps")
    if net is None or float(net) > -100.0:
        return "NOT_LARGE_LOSS"
    if float(row.get("running_accel_usd_per_sec") or 0.0) > 5_000.0:
        return "ACCELERATION_RUNAWAY_DESCRIPTIVE"
    if float(row.get("book_imbalance") or 0.0) < -0.25:
        return "LIQUIDITY_VACUUM_ADVERSE_SELECTION_DESCRIPTIVE"
    if float(row.get("bid_depth_usd") or 0.0) >= 135_423.8:
        return "BID_WALL_FAILED_TRAP_DESCRIPTIVE"
    return "MARKET_WIDE_DELEVERAGING_OR_NEGATIVE_SKEW_DESCRIPTIVE"


def provision_scenarios(conn: sqlite3.Connection, row: dict[str, Any]) -> list[dict[str, Any]]:
    if str(row.get("symbol") or "") != "ETHUSDT" or row.get("anchor_mark_price") is None:
        return []
    event = {
        "entry_ts_ms": int(row["signal_ts_ms"]),
        "mid": float(row["anchor_mark_price"]),
        "month": str(row.get("signal_utc") or "")[:7],
    }
    out = []
    for cross in (0.5, 1.0):
        for fee in (-0.5, 0.0, 0.5):
            sim = simulate_tick_queue_event(
                conn,
                event,
                offset_bps=2.0,
                horizon_sec=300,
                cross_bps=cross,
                queue_notional_usd=1_000.0,
                maker_fee_bps=fee,
                taker_fee_bps=3.05,
            )
            if not sim:
                continue
            out.append(
                {
                    "scenario": f"eth_provision_o2_h300_qcross{cross:g}_queue1000_fee{fee:g}",
                    "qcross_bps": cross,
                    "maker_fee_bps": fee,
                    "fill_state": sim["fill_state"],
                    "net_bps": r1(sim["net_bps"]),
                }
            )
    return out


def mark_min_between(conn: sqlite3.Connection, symbol: str, start_ms: int, end_ms: int) -> float | None:
    row = conn.execute(
        "SELECT MIN(mark_price) FROM mark_prices WHERE symbol=? AND ts_ms>=? AND ts_ms<=?",
        (symbol, int(start_ms), int(end_ms)),
    ).fetchone()
    return float(row[0]) if row and row[0] is not None else None


def atomicity_gap_observation(conn: sqlite3.Connection, row: dict[str, Any], *, poll_sec: float) -> dict[str, Any]:
    fill_ts = row.get("maker_fill_ts_ms")
    entry = row.get("entry_price")
    if fill_ts is None or entry is None:
        return {
            "status": "NO_FILL_OR_NO_ENTRY",
            "estimated_gap_sec": float(poll_sec),
            "adverse_move_bps": None,
            "alert": False,
        }
    start = int(fill_ts)
    end = start + int(float(poll_sec) * 1000)
    min_mark = mark_min_between(conn, str(row.get("symbol") or "ETHUSDT"), start, end)
    if min_mark is None:
        return {
            "status": "NO_MARK_IN_GAP",
            "fill_ts_ms": start,
            "estimated_stop_due_ts_ms": end,
            "estimated_gap_sec": float(poll_sec),
            "adverse_move_bps": None,
            "alert": False,
        }
    adverse = (float(min_mark) - float(entry)) / float(entry) * 10_000.0
    return {
        "status": "OBSERVED",
        "fill_ts_ms": start,
        "fill_utc": iso_ms(start),
        "estimated_stop_due_ts_ms": end,
        "estimated_stop_due_utc": iso_ms(end),
        "estimated_gap_sec": float(poll_sec),
        "min_mark_in_gap": r1(min_mark),
        "adverse_move_bps": r1(adverse),
        "alert": bool(adverse < -5.0),
        "read": "estimated fill-to-next-poll gap; live exchange ack latency is not included",
    }


def management_rows(db: Path, shadow_rows: list[dict[str, Any]], *, frozen_start_ms: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    env = parse_env(DEFAULT_ENV)
    poll_sec = fenv(env, "S34_LIVE_POLL_SEC", 2.0)
    with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as conn:
        for src in shadow_rows:
            ts = int(src.get("signal_ts_ms") or 0)
            if not ts:
                continue
            diss = [dissipation_at_tau(conn, src, tau) for tau in (60, 120, 180)]
            prov = provision_scenarios(conn, src)
            atomicity = atomicity_gap_observation(conn, src, poll_sec=poll_sec)
            net = src.get("net_bps")
            rows.append(
                {
                    "management_id": f"{src.get('observation_id')}|mgmt_v1",
                    "source_observation_id": src.get("observation_id"),
                    "sample_type": "FORWARD_OOS" if ts >= int(frozen_start_ms) else "PRE_FORWARD_REFERENCE",
                    "rule_id": src.get("protocol_id") or RULE_ID,
                    "signal_ts_ms": ts,
                    "signal_utc": src.get("signal_utc") or iso_ms(ts),
                    "observation_status": src.get("observation_status"),
                    "sim_status": src.get("sim_status"),
                    "net_bps": net,
                    "vdepth_bps": src.get("vdepth_bps"),
                    "prior_4h_bps": src.get("prior_4h_bps"),
                    "bid_depth_usd": src.get("bid_depth_usd"),
                    "book_imbalance": src.get("book_imbalance"),
                    "running_accel_usd_per_sec": src.get("running_accel_usd_per_sec"),
                    "dissipation_observer": diss,
                    "dissipation_tau120_recommendation": next((d["recommendation"] for d in diss if d["tau_sec"] == 120), None),
                    "atomicity_gap_observer": atomicity,
                    "failure_mode": classify_loss(src),
                    "provision_scenarios": prov,
                    "management_note": "recommendation_only_no_live_order_change",
                }
            )
    rows.sort(key=lambda r: (int(r["signal_ts_ms"]), str(r["source_observation_id"])))
    return rows


def closed_vals(rows: list[dict[str, Any]]) -> list[float]:
    vals = []
    for row in rows:
        if row.get("observation_status") == "CLOSED" and row.get("sim_status") == "FILLED" and row.get("net_bps") is not None:
            vals.append(float(row["net_bps"]))
    return vals


def regime_monitor(rows: list[dict[str, Any]]) -> dict[str, Any]:
    forward = [r for r in rows if r.get("sample_type") == "FORWARD_OOS"]
    ref = [r for r in rows if r.get("sample_type") == "PRE_FORWARD_REFERENCE"]
    fvals = closed_vals(forward)
    rvals = closed_vals(ref)
    triggers = []
    if len(fvals) >= 5 and sum(fvals[-5:]) < 0:
        triggers.append("PAUSE_ROLLING_5_NEGATIVE")
    if len(fvals) >= 5 and sum(fvals) < 0:
        triggers.append("KILL_FORWARD_SUM_NEGATIVE")
    if any(v <= -STRESS_TAIL_BPS for v in fvals):
        triggers.append("PAUSE_TAIL_BUDGET_BREACH")
    return {
        "status": "DATA_INSUFFICIENT" if len(fvals) < 20 else ("TRIGGERED" if triggers else "OK"),
        "forward_summary": summary(fvals),
        "reference_summary": summary(rvals),
        "triggers": triggers,
        "read": "forward OOS is the validator; reference rows are pre-freeze only",
    }


def atomicity_monitor(rows: list[dict[str, Any]]) -> dict[str, Any]:
    observed = [r.get("atomicity_gap_observer") or {} for r in rows if (r.get("atomicity_gap_observer") or {}).get("status") == "OBSERVED"]
    alerts = [r for r in observed if r.get("alert")]
    vals = [float(r["adverse_move_bps"]) for r in observed if r.get("adverse_move_bps") is not None]
    return {
        "status": "ALERT_ADVERSE_IN_GAP" if alerts else ("NO_OBSERVED_GAP_ALERTS" if observed else "NO_FILLED_GAP_OBSERVATIONS"),
        "observed_n": len(observed),
        "alert_n": len(alerts),
        "worst_adverse_bps": r1(min(vals)) if vals else None,
        "recommendation": "DOCUMENT_AND_RECOMMEND_ATOMIC_BRACKET_OPERATOR_SIGNOFF_REQUIRED",
        "read": "observation only; does not change bracket/order logic",
    }


def failure_forward(rows: list[dict[str, Any]]) -> dict[str, Any]:
    losses = [r for r in rows if r.get("sample_type") == "FORWARD_OOS" and r.get("failure_mode") != "NOT_LARGE_LOSS"]
    counts: dict[str, int] = {}
    for row in losses:
        label = str(row["failure_mode"])
        counts[label] = counts.get(label, 0) + 1
    return {
        "status": "DESCRIPTIVE_ONLY_NOT_ENTRY_FILTER",
        "forward_large_loss_n": len(losses),
        "counts": dict(sorted(counts.items())),
        "latest": losses[-10:],
    }


def provision_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by: dict[str, dict[str, list[float]]] = {}
    counts: dict[str, dict[str, int]] = {}
    for row in rows:
        sample = str(row.get("sample_type"))
        for sim in row.get("provision_scenarios") or []:
            scenario = str(sim["scenario"])
            by.setdefault(scenario, {}).setdefault(sample, []).append(float(sim["net_bps"]))
            key = f"{scenario}|{sample}"
            counts.setdefault(key, {})
            state = str(sim.get("fill_state"))
            counts[key][state] = counts[key].get(state, 0) + 1
    ranked = []
    for scenario, samples in by.items():
        ranked.append(
            {
                "scenario": scenario,
                "forward": summary(samples.get("FORWARD_OOS", [])),
                "reference": summary(samples.get("PRE_FORWARD_REFERENCE", [])),
                "fill_counts": {k.split("|", 1)[1]: v for k, v in counts.items() if k.startswith(scenario + "|")},
            }
        )
    ranked.sort(key=lambda r: float((r["forward"]["sum_bps"] if r["forward"]["n"] else r["reference"]["sum_bps"]) or 0.0), reverse=True)
    return {
        "status": "FORWARD_OBSERVATION_ONLY_NOT_ALPHA",
        "binding_question": "actual maker fee tier; positive maker fee kills the pocket",
        "ranked_scenarios": ranked,
    }


def kill_criteria(rows: list[dict[str, Any]], sizing: dict[str, Any], regime: dict[str, Any]) -> dict[str, Any]:
    fvals = closed_vals([r for r in rows if r.get("sample_type") == "FORWARD_OOS"])
    triggered = []
    if len(fvals) >= 5 and sum(fvals) < 0:
        triggered.append("KILL_30_60D_FORWARD_SUM_NEGATIVE")
    if any(v <= -STRESS_TAIL_BPS for v in fvals):
        triggered.append("PAUSE_TAIL_BUDGET_BREACH")
    if regime.get("status") == "TRIGGERED":
        triggered.append("PAUSE_REGIME_DEGRADATION")
    if sizing.get("status") == "ALERT_OVERSIZE_OPERATOR_ACTION_REQUIRED":
        triggered.append("OPERATOR_SIZE_REVIEW_REQUIRED")
    return {
        "status": "TRIGGERED_RECOMMENDATION_ONLY" if triggered else "NOT_TRIGGERED",
        "triggered": triggered,
        "criteria": [
            "30/60-day forward sum < 0 after >=5 closed fills",
            "rolling last 5 closed fills sum < 0",
            "realized/shadow loss breaches accepted tail budget",
            "regime-degradation monitor trips",
            "configured size exceeds tail-budget notional",
        ],
        "action_boundary": "recommendations only; no auto-disarm and no auto-size-change",
    }


def live_state_summary(path: Path) -> dict[str, Any]:
    state = load_json(path, {})
    status = state.get("status") if isinstance(state.get("status"), dict) else {}
    rec = state.get("reconciliation") if isinstance(state.get("reconciliation"), dict) else {}
    return {
        "path": str(path),
        "exists": path.exists(),
        "active": state.get("active"),
        "orders_n": len(state.get("orders") or []),
        "mode": status.get("mode"),
        "rule": status.get("rule"),
        "allowed_rules": status.get("allowed_rules"),
        "state_updated_at_utc": status.get("updated_at_utc"),
        "reconciliation": rec,
    }


def build_readout(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    env = parse_env(args.env)
    shadow_rows = load_jsonl(args.shadow_ledger)
    pool = load_json(args.pool_json, {})
    pool_rows = pool.get("rows") if isinstance(pool, dict) else []
    if not isinstance(pool_rows, list):
        pool_rows = []
    state = load_or_init_state(args.state_path, freeze_start_ms=args.freeze_start_ms)
    rows = management_rows(args.db, shadow_rows, frozen_start_ms=int(state["frozen_start_ms"]))
    sizing = sizing_monitor(env, pool_rows, shadow_rows, equity_usdt=float(args.equity_usdt), risk_pct=float(args.risk_pct))
    regime = regime_monitor(rows)
    atomicity = atomicity_monitor(rows)
    failures = failure_forward(rows)
    provision = provision_summary(rows)
    kills = kill_criteria(rows, sizing, regime)
    live_before = live_state_summary(args.live_state)
    forward_rows = [r for r in rows if r.get("sample_type") == "FORWARD_OOS"]
    ref_rows = [r for r in rows if r.get("sample_type") == "PRE_FORWARD_REFERENCE"]
    readout = {
        "generated_at_utc": utc_now(),
        "mode": MODE,
        "rule_id": RULE_ID,
        "frozen_start_ms": int(state["frozen_start_ms"]),
        "frozen_start_utc": state["frozen_start_utc"],
        "source_shadow_rows": len(shadow_rows),
        "management_ledger_rows": len(rows),
        "forward_rows": len(forward_rows),
        "reference_rows": len(ref_rows),
        "live_state_read_only_snapshot": live_before,
        "tail_aware_sizing_monitor": sizing,
        "regime_degradation_monitor": regime,
        "atomicity_gap_monitor": atomicity,
        "failure_mode_forward_tracking": failures,
        "eth_provision_forward_observation": provision,
        "explicit_kill_criteria": kills,
        "dashboard_line": dashboard_line(sizing, regime, kills, rows),
        "guardrails": [
            "no live order logic changed",
            "no live size changed",
            "no entry filters built from failure/trap/sync/absorption",
            "forward OOS only can validate; reference rows are descriptive",
        ],
    }
    state["last_run_utc"] = readout["generated_at_utc"]
    state["last_forward_rows"] = len(forward_rows)
    state["last_reference_rows"] = len(ref_rows)
    write_json(args.state_path, state)
    return readout, rows


def dashboard_line(sizing: dict[str, Any], regime: dict[str, Any], kills: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    fvals = closed_vals([r for r in rows if r.get("sample_type") == "FORWARD_OOS"])
    return {
        "rule_id": RULE_ID,
        "permission": "UNVALIDATED_OBSERVATION_ONLY",
        "size_status": sizing.get("status"),
        "regime_status": regime.get("status"),
        "kill_status": kills.get("status"),
        "forward_closed_n": len(fvals),
        "forward_sum_bps": r1(sum(fvals)) if fvals else 0.0,
        "max_tail_budget_margin_usdt": sizing.get("max_tail_budget_margin_usdt"),
        "max_stop_budget_margin_usdt": sizing.get("max_stop_budget_margin_usdt"),
        "env_planned_margin_usdt": (sizing.get("planned_live_size_from_env") or {}).get("planned_margin_usdt"),
        "oversize_multiple": sizing.get("oversize_multiple"),
        "recommendation": "OPERATOR_SIZE_REVIEW_OR_DISARM_UNTIL_FORWARD_VALIDATION"
        if sizing.get("status") == "ALERT_OVERSIZE_OPERATOR_ACTION_REQUIRED"
        else "CONTINUE_OBSERVATION_AT_TAIL_BUDGET_SIZE",
        "updated_at_utc": utc_now(),
    }


def render_md(readout: dict[str, Any]) -> str:
    sizing = readout["tail_aware_sizing_monitor"]
    live = sizing["planned_live_size_from_env"]
    regime = readout["regime_degradation_monitor"]
    kills = readout["explicit_kill_criteria"]
    atomicity = readout["atomicity_gap_monitor"]
    provision = readout["eth_provision_forward_observation"]
    failures = readout["failure_mode_forward_tracking"]
    lines = [
        "# S34 V Engine Forward Management Readout",
        "",
        f"Generated: `{readout['generated_at_utc']}`",
        "",
        f"Mode: `{readout['mode']}`",
        "",
        f"Frozen forward start: `{readout['frozen_start_utc']}`",
        "",
        f"Live rule: `{readout['rule_id']}`",
        "",
        "## Ledger",
        "",
        f"- source shadow rows: `{readout['source_shadow_rows']}`",
        f"- management rows: `{readout['management_ledger_rows']}`",
        f"- forward rows: `{readout['forward_rows']}`",
        f"- pre-forward reference rows: `{readout['reference_rows']}`",
        "",
        "## Live State Snapshot",
        "",
        f"- active: `{readout['live_state_read_only_snapshot'].get('active')}`",
        f"- open/order rows in state: `{readout['live_state_read_only_snapshot'].get('orders_n')}`",
        f"- reconciliation: `{readout['live_state_read_only_snapshot'].get('reconciliation')}`",
        "",
        "## Tail-Aware Size Monitor",
        "",
        f"- status: `{sizing['status']}`",
        f"- action: `{sizing['action']}`",
        f"- risk budget: `{sizing['risk_pct_equity']}%` equity",
        f"- max tail-budget notional: `${sizing['max_tail_budget_notional_usdt']}`",
        f"- max tail-budget margin: `${sizing['max_tail_budget_margin_usdt']}`",
        f"- max stop-budget notional: `${sizing['max_stop_budget_notional_usdt']}`",
        f"- max stop-budget margin: `${sizing['max_stop_budget_margin_usdt']}`",
        f"- env planned margin/notional: `${live['planned_margin_usdt']}` / `${live['planned_notional_usdt']}`",
        f"- env stress loss: `${live['stress_tail_loss_usdt']}` = `{live['stress_tail_loss_pct_equity']}%` equity",
        f"- oversize multiple vs budget: `{sizing['oversize_multiple']}`",
        f"- oversize multiple vs stop-budget: `{sizing['stop_budget_oversize_multiple']}`",
        "",
        "## Atomicity Gap",
        "",
        f"- status: `{atomicity['status']}`",
        f"- observed N: `{atomicity['observed_n']}`",
        f"- alert N: `{atomicity['alert_n']}`",
        f"- worst adverse bps: `{atomicity['worst_adverse_bps']}`",
        f"- recommendation: `{atomicity['recommendation']}`",
        "",
        "## Regime / Kill",
        "",
        f"- regime status: `{regime['status']}`",
        f"- forward summary: `{regime['forward_summary']}`",
        f"- reference summary: `{regime['reference_summary']}`",
        f"- kill status: `{kills['status']}`",
        f"- triggered: `{kills['triggered']}`",
        "",
        "## Failure-Mode Tracking",
        "",
        f"- status: `{failures['status']}`",
        f"- forward large loss N: `{failures['forward_large_loss_n']}`",
        f"- counts: `{failures['counts']}`",
        "",
        "## ETH Provision Observation",
        "",
        f"- status: `{provision['status']}`",
        f"- binding question: {provision['binding_question']}",
        "",
        "| Rank | Scenario | Forward | Reference |",
        "| ---: | --- | --- | --- |",
    ]
    for idx, row in enumerate(provision.get("ranked_scenarios", [])[:8], 1):
        lines.append(
            f"| {idx} | `{row['scenario']}` | {fmt(row['forward'])} | {fmt(row['reference'])} |"
        )
    lines.extend(
        [
            "",
            "## Dashboard Line",
            "",
            f"`{json.dumps(readout['dashboard_line'], sort_keys=True, ensure_ascii=True)}`",
            "",
            "## Guardrails",
            "",
            "- This readout emits recommendations only.",
            "- No live order logic, size, config, or executor state was changed.",
            "- Failure modes are descriptive and must not become entry filters.",
        ]
    )
    return "\n".join(lines) + "\n"


def fmt(s: dict[str, Any]) -> str:
    return f"N={s.get('n')} sum={s.get('sum_bps')} med={s.get('median_bps')} T3R={s.get('t3r_bps')} maxL={s.get('max_loss_bps')}"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="S34 V Engine forward management monitor (risk/observation only).")
    p.add_argument("--db", type=Path, default=DEFAULT_DB)
    p.add_argument("--env", type=Path, default=DEFAULT_ENV)
    p.add_argument("--shadow-ledger", type=Path, default=SHADOW_LEDGER)
    p.add_argument("--pool-json", type=Path, default=POOL_JSON)
    p.add_argument("--live-state", type=Path, default=LIVE_STATE)
    p.add_argument("--state-path", type=Path, default=STATE_PATH)
    p.add_argument("--out-ledger", type=Path, default=OUT_LEDGER)
    p.add_argument("--out-json", type=Path, default=OUT_JSON)
    p.add_argument("--out-md", type=Path, default=OUT_MD)
    p.add_argument("--out-fragment", type=Path, default=OUT_FRAGMENT)
    p.add_argument("--equity-usdt", type=float, default=35.0)
    p.add_argument("--risk-pct", type=float, default=2.0)
    p.add_argument("--freeze-start-ms", type=int, default=None)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    readout, rows = build_readout(args)
    write_jsonl(args.out_ledger, rows)
    write_json(args.out_json, readout)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(render_md(readout), encoding="utf-8")
    write_json(args.out_fragment, readout["dashboard_line"])
    print(render_md(readout))
    print(f"Wrote {args.out_ledger}")
    print(f"Wrote {args.out_json}")
    print(f"Wrote {args.out_md}")
    print(f"Wrote {args.out_fragment}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
