"""S34 v10 operational risk suite.

Observation/risk only. Produces:
- risk budget modes;
- Monte Carlo risk-of-ruin by mode;
- pre-trade risk card;
- post-trade autopsy cards;
- executor readiness watchdog snapshot;
- fee-tier reality check;
- stop-slippage forward tracker seed;
- kill-switch drill result;
- decision-journal template.

No live order, executor, .env, leverage, size, or config changes.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.research_s34_knowable_anchor_continuation import iso_ms

DB = ROOT / "data" / "microstructure.db"
ENV = ROOT / ".env"
LIVE_STATE = ROOT / "runtime" / "s34_v_engine_live_state.json"
LIVE_PID = ROOT / "logs" / "pids" / "s34_v_engine_live_executor.pid"
SHADOW_STATE = ROOT / "runtime" / "s34_v_engine_v02_shadow_mirror_state.json"
MGMT = ROOT / "reports" / "research" / "s34" / "S34_V_ENGINE_FORWARD_MANAGEMENT_READOUT.json"
MGMT_LEDGER = ROOT / "reports" / "research" / "s34" / "S34_V_ENGINE_FORWARD_MANAGEMENT_LEDGER.jsonl"
STOP = ROOT / "reports" / "research" / "s34" / "S34_V_ENGINE_PROTECTIVE_STOP.json"
V8 = ROOT / "reports" / "research" / "s34" / "S34_V8_STOP_RELIABILITY_SIZING.json"
V9 = ROOT / "reports" / "research" / "s34" / "S34_V9_KILL_FORWARD_REVIEW.json"
OUT_JSON = ROOT / "reports" / "research" / "s34" / "S34_V10_OPERATIONAL_RISK_SUITE.json"
OUT_MD = ROOT / "reports" / "research" / "s34" / "S34_V10_OPERATIONAL_RISK_SUITE.md"
DECISION_JOURNAL = ROOT / "reports" / "research" / "s34" / "S34_OPERATOR_DECISION_JOURNAL.jsonl"

EQUITY = 35.0
LEVERAGE = 40.0
TAIL_RATE = 101.0 / 539.0
TAIL_CUTOFF_BPS = -100.0
WEIGHTED_TAIL_BPS = -428.6
STOP_REALIZED_BPS = -175.7
TAIL_HARD_BPS = -634.0


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def now_ms() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def parse_env(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def fenv(env: dict[str, str], key: str, default: float) -> float:
    try:
        return float(env.get(key, default))
    except (TypeError, ValueError):
        return float(default)


def r1(v: float | None) -> float | None:
    if v is None or not math.isfinite(float(v)):
        return None
    return round(float(v), 1)


def r3(v: float | None) -> float | None:
    if v is None or not math.isfinite(float(v)):
        return None
    return round(float(v), 3)


def parse_iso_ms(text: str | None) -> int | None:
    if not text:
        return None
    t = str(text)
    if t.endswith("Z"):
        t = t[:-1] + "+00:00"
    return int(datetime.fromisoformat(t).timestamp() * 1000)


def fixed_stop_rows(stop: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [r for r in stop.get("rows") or [] if r.get("variant") == "fixed_sl_150"]
    rows.sort(key=lambda r: parse_iso_ms(r.get("signal_utc")) or 0)
    return rows


def risk_modes(v8: dict[str, Any], mgmt: dict[str, Any]) -> dict[str, Any]:
    rows = ((v8.get("stop_reliability_weighted_sizing") or {}).get("sizing_rows") or [])
    by = {r.get("basis"): r for r in rows}
    planned = ((mgmt.get("tail_aware_sizing_monitor") or {}).get("planned_live_size_from_env") or {})
    current_notional = float(planned.get("planned_notional_usdt") or 1190.0)
    modes = {
        "SURVIVAL": {"basis": "tail_only_hard_floor", "notional": 11.0, "margin": 0.3, "loss_bps": 634.0},
        "BALANCED": {"basis": "conservative_weighted", "notional": 16.3, "margin": 0.4, "loss_bps": 428.6},
        "STOP_ASSISTED": {"basis": "stop_only_unreliable_floor", "notional": 39.8, "margin": 1.0, "loss_bps": 175.7},
        "CURRENT_ENV": {"basis": "env", "notional": current_notional, "margin": current_notional / LEVERAGE, "loss_bps": 428.6},
    }
    for name, item in list(modes.items()):
        ref = by.get(item["basis"]) or {}
        if ref:
            item["notional"] = float(ref.get("max_notional_usdt") or item["notional"])
            item["margin"] = float(ref.get("max_margin_usdt_at_40x") or item["margin"])
            item["loss_bps"] = float(ref.get("loss_bps") or item["loss_bps"])
        item["oversize_vs_env"] = r1(current_notional / float(item["notional"])) if item["notional"] else None
    return modes


def positive_pool(stop_rows: list[dict[str, Any]]) -> list[float]:
    vals = [float(r.get("baseline_net_bps") or 0.0) for r in stop_rows]
    pool = [v for v in vals if v > TAIL_CUTOFF_BPS]
    return pool or vals or [0.0]


def percentile(vals: list[float], q: float) -> float | None:
    if not vals:
        return None
    vals = sorted(vals)
    idx = min(len(vals) - 1, max(0, int(round((len(vals) - 1) * q))))
    return vals[idx]


def monte_carlo_modes(stop_rows: list[dict[str, Any]], modes: dict[str, Any], *, trials: int, seed: int, equity: float) -> dict[str, Any]:
    rng = random.Random(seed)
    pool = positive_pool(stop_rows)
    horizons = (30, 60, 100)
    out: dict[str, Any] = {"trials": trials, "seed": seed, "tail_rate": r3(TAIL_RATE), "horizons": {}}
    for h in horizons:
        out["horizons"][str(h)] = {}
        for mode, cfg in modes.items():
            notional = float(cfg["notional"])
            tail_bps = -float(cfg["loss_bps"])
            finals = []
            maxdds = []
            ruins = 0
            min_balance_breaches = 0
            for _ in range(trials):
                bal = equity
                peak = equity
                maxdd = 0.0
                for _j in range(h):
                    bps = tail_bps if rng.random() < TAIL_RATE else rng.choice(pool)
                    bal += notional * bps / 10_000.0
                    peak = max(peak, bal)
                    maxdd = min(maxdd, bal - peak)
                finals.append(bal)
                maxdds.append(maxdd)
                if bal <= 0:
                    ruins += 1
                if bal <= 15.0:
                    min_balance_breaches += 1
            out["horizons"][str(h)][mode] = {
                "ruin_pct": r1(100.0 * ruins / trials),
                "min_balance_breach_pct": r1(100.0 * min_balance_breaches / trials),
                "median_final_equity": r1(percentile(finals, 0.5)),
                "p05_final_equity": r1(percentile(finals, 0.05)),
                "p95_max_dd_usdt": r1(percentile([abs(x) for x in maxdds], 0.95)),
                "p99_max_dd_usdt": r1(percentile([abs(x) for x in maxdds], 0.99)),
            }
    return out


def pre_trade_card(mgmt: dict[str, Any], v9: dict[str, Any], modes: dict[str, Any]) -> dict[str, Any]:
    sizing = mgmt.get("tail_aware_sizing_monitor") or {}
    planned = sizing.get("planned_live_size_from_env") or {}
    return {
        "status": "PRE_TRADE_RISK_CARD_TEMPLATE_READY",
        "planned_notional_usdt": planned.get("planned_notional_usdt"),
        "planned_margin_usdt": planned.get("planned_margin_usdt"),
        "configured_stop_bps": 150.0,
        "stop_realized_worst_bps": 175.7,
        "planned_loss_at_realized_stop_usdt": r1(float(planned.get("planned_notional_usdt") or 0.0) * 175.7 / 10_000.0),
        "risk_modes": modes,
        "kill_rule": ((v9.get("kill_criteria_redesign") or {}).get("recommended_kill_rule") or {}).get("name"),
        "operator_question": "Is this signal allowed at selected risk mode, or should it be observed only?",
    }


def post_trade_autopsy_cards(ledger: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    out = []
    for row in ledger[-limit:]:
        atomic = row.get("atomicity_gap_observer") or {}
        diss120 = None
        for item in row.get("dissipation_observer") or []:
            if item.get("tau_sec") == 120:
                diss120 = item
        out.append(
            {
                "signal_utc": row.get("signal_utc"),
                "sample_type": row.get("sample_type"),
                "net_bps": row.get("net_bps"),
                "failure_mode": row.get("failure_mode"),
                "atomicity_gap_status": atomic.get("status"),
                "atomicity_adverse_bps": atomic.get("adverse_move_bps"),
                "dissipation_tau120": diss120,
                "pnl_by_mode_usdt": {
                    "SURVIVAL": r1(11.0 * float(row.get("net_bps") or 0.0) / 10_000.0),
                    "BALANCED": r1(16.3 * float(row.get("net_bps") or 0.0) / 10_000.0),
                    "CURRENT_ENV": r1(1190.0 * float(row.get("net_bps") or 0.0) / 10_000.0),
                },
            }
        )
    return out


def pid_alive(pid: int | None) -> bool | None:
    if pid is None:
        return None
    if os.name == "nt":
        try:
            result = subprocess.run(
                ["tasklist", "/FI", f"PID eq {int(pid)}", "/FO", "CSV", "/NH"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            if result.returncode != 0 and "Access denied" in (result.stderr or ""):
                return None
            return str(pid) in (result.stdout or "")
        except Exception:
            return None
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def read_pid(path: Path) -> int | None:
    if not path.exists():
        return None
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except Exception:
        return None


def latest_db_ts(conn: sqlite3.Connection, table: str, symbol: str | None = None) -> int | None:
    try:
        if symbol:
            row = conn.execute(f"SELECT MAX(ts_ms) FROM {table} WHERE symbol=?", (symbol,)).fetchone()
        else:
            row = conn.execute(f"SELECT MAX(ts_ms) FROM {table}").fetchone()
    except sqlite3.Error:
        return None
    return int(row[0]) if row and row[0] is not None else None


def age_sec(ts_ms: int | None) -> float | None:
    if ts_ms is None:
        return None
    return (now_ms() - int(ts_ms)) / 1000.0


def executor_readiness(db: Path, env: dict[str, str], live_state: dict[str, Any], shadow_state: dict[str, Any]) -> dict[str, Any]:
    pid = read_pid(LIVE_PID)
    rec = live_state.get("reconciliation") if isinstance(live_state.get("reconciliation"), dict) else {}
    status = live_state.get("status") if isinstance(live_state.get("status"), dict) else {}
    state_updated_ms = parse_iso_ms(status.get("updated_at_utc"))
    state_age = age_sec(state_updated_ms)
    pid_status = pid_alive(pid)
    with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as conn:
        book_ts = latest_db_ts(conn, "book_ticker", "ETHUSDT")
        liq_ts = latest_db_ts(conn, "liquidations", "ETHUSDT")
        mark_ts = latest_db_ts(conn, "mark_prices", "ETHUSDT")
    checks = {
        "pid_file": pid,
        "pid_alive": pid_status,
        "pid_read": "unknown_access_denied_fallback_to_state_freshness" if pid_status is None else "direct_pid_check",
        "mode": status.get("mode"),
        "rule": status.get("rule"),
        "active": live_state.get("active"),
        "state_orders_n": len(live_state.get("orders") or []),
        "exchange_position_amount": rec.get("position_amount"),
        "exchange_s34_open_order_count": rec.get("s34ve_open_order_count"),
        "kill_switch_file": env.get("S34_LIVE_KILL_SWITCH_FILE"),
        "kill_switch_exists": (ROOT / str(env.get("S34_LIVE_KILL_SWITCH_FILE", "runtime/KILL_SWITCH"))).exists(),
        "book_age_sec": r1(age_sec(book_ts)),
        "liq_age_sec": r1(age_sec(liq_ts)),
        "mark_age_sec": r1(age_sec(mark_ts)),
        "shadow_mirror_updated_at_utc": shadow_state.get("updated_at_utc"),
        "state_age_sec": r1(state_age),
    }
    issues = []
    state_fresh = state_age is not None and state_age <= 15.0
    if checks["pid_alive"] is False and not state_fresh:
        issues.append("live_executor_pid_not_alive")
    if checks["pid_alive"] is None and not state_fresh:
        issues.append("live_executor_pid_unknown_and_state_stale")
    if checks["active"] is not None:
        issues.append("active_trade_present")
    if float(checks["exchange_position_amount"] or 0.0) != 0.0:
        issues.append("exchange_position_present")
    if int(checks["exchange_s34_open_order_count"] or 0) != 0:
        issues.append("open_s34_orders_present")
    if checks["kill_switch_exists"]:
        issues.append("kill_switch_blocks_new_entries")
    return {"status": "READY_NO_POSITION" if not issues else "ATTENTION", "checks": checks, "issues": issues}


def fee_tier_reality(env: dict[str, str]) -> dict[str, Any]:
    assumed_maker = fenv(env, "S34_V_ENGINE_MAKER_FEE_BPS", 2.0)
    assumed_taker = fenv(env, "S34_V_ENGINE_TAKER_FEE_BPS", 3.05)
    return {
        "status": "ACTUAL_FEE_TIER_UNKNOWN",
        "assumed_maker_fee_bps": assumed_maker,
        "assumed_taker_fee_bps": assumed_taker,
        "provision_pocket_requires": "maker fee <= -0.5 bps or very favorable queue; positive maker fee mostly kills it",
        "recommendation": "operator confirm actual exchange fee tier; do not promote provision pocket while UNKNOWN",
    }


def stop_slippage_tracker(stop_rows: list[dict[str, Any]]) -> dict[str, Any]:
    stopped = [r for r in stop_rows if r.get("exit_reason") == "SL"]
    vals = [float(r.get("net_bps") or 0.0) for r in stopped]
    return {
        "status": "FORWARD_TRACKER_SEEDED",
        "historical_stop_n": len(stopped),
        "historical_stop_summary": {
            "worst_realized_stop_bps": r1(min(vals)) if vals else None,
            "median_realized_stop_bps": r1(median(vals)) if vals else None,
        },
        "fields_to_log_forward": ["trigger_price", "realized_exit_price", "slippage_bps", "spread_bps_at_trigger", "book_age_ms", "volatility_context"],
    }


def kill_switch_drill(env: dict[str, str]) -> dict[str, Any]:
    ks = ROOT / str(env.get("S34_LIVE_KILL_SWITCH_FILE", "runtime/KILL_SWITCH"))
    return {
        "status": "SIMULATED_ONLY_NO_FILE_CREATED",
        "kill_switch_path": str(ks),
        "currently_exists": ks.exists(),
        "expected_executor_behavior": "blocks new entries when file exists; does not auto-close existing active position",
        "operator_drill_command": f"New-Item {ks} -ItemType File",
        "operator_clear_command": f"Remove-Item {ks}",
    }


def ensure_decision_journal(path: Path) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        template = {
            "ts_utc": utc_now(),
            "event": "decision_template",
            "decision": "ARMED_KEEP_SIZE | REDUCE_MARGIN | DISARM | OVERRIDE_KILL",
            "reason": "",
            "risk_mode": "SURVIVAL | BALANCED | STOP_ASSISTED | CURRENT_ENV",
            "evidence_refs": ["S34_V10_OPERATIONAL_RISK_SUITE.md"],
            "operator": "",
        }
        with path.open("w", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(template, sort_keys=True, ensure_ascii=True) + "\n")
        created = True
    else:
        created = False
    lines = sum(1 for _ in path.open("r", encoding="utf-8")) if path.exists() else 0
    return {"path": str(path), "created": created, "rows": lines, "status": "READY"}


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    env = parse_env(args.env)
    mgmt = load_json(args.management_json, {})
    v8 = load_json(args.v8_json, {})
    v9 = load_json(args.v9_json, {})
    stop = load_json(args.stop_json, {})
    live_state = load_json(args.live_state, {})
    shadow_state = load_json(SHADOW_STATE, {})
    ledger = load_jsonl(args.management_ledger)
    stop_rows = fixed_stop_rows(stop)
    modes = risk_modes(v8, mgmt)
    mc = monte_carlo_modes(stop_rows, modes, trials=int(args.trials), seed=int(args.seed), equity=float(args.equity_usdt))
    return {
        "generated_at_utc": utc_now(),
        "mode": "OBSERVATION_RISK_ONLY_NO_LIVE_CHANGE",
        "risk_budget_modes": modes,
        "risk_of_ruin": mc,
        "pre_trade_risk_card": pre_trade_card(mgmt, v9, modes),
        "post_trade_autopsy_cards": post_trade_autopsy_cards(ledger, limit=5),
        "executor_readiness": executor_readiness(args.db, env, live_state, shadow_state),
        "fee_tier_reality_check": fee_tier_reality(env),
        "stop_slippage_forward_tracker": stop_slippage_tracker(stop_rows),
        "kill_switch_drill": kill_switch_drill(env),
        "decision_journal": ensure_decision_journal(args.decision_journal),
    }


def render_md(report: dict[str, Any]) -> str:
    lines = [
        "# S34 v10 Operational Risk Suite",
        "",
        f"Generated: `{report['generated_at_utc']}`",
        "",
        f"Mode: `{report['mode']}`",
        "",
        "## Risk Budget Modes",
        "",
        "| Mode | Notional | Margin @40x | Loss bps basis | Oversize vs env |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for mode, cfg in report["risk_budget_modes"].items():
        lines.append(f"| `{mode}` | ${r1(cfg['notional'])} | ${r1(cfg['margin'])} | {r1(cfg['loss_bps'])} | {cfg.get('oversize_vs_env')}x |")
    lines.extend(["", "## Risk Of Ruin", ""])
    for h, modes in report["risk_of_ruin"]["horizons"].items():
        lines.append(f"### {h} Trades")
        lines.append("")
        lines.append("| Mode | Ruin% | MinBalance<=15% | P05 final equity | P99 max DD |")
        lines.append("| --- | ---: | ---: | ---: | ---: |")
        for mode, row in modes.items():
            lines.append(f"| `{mode}` | {row['ruin_pct']} | {row['min_balance_breach_pct']} | ${row['p05_final_equity']} | ${row['p99_max_dd_usdt']} |")
        lines.append("")
    pre = report["pre_trade_risk_card"]
    ready = report["executor_readiness"]
    fee = report["fee_tier_reality_check"]
    stop = report["stop_slippage_forward_tracker"]
    drill = report["kill_switch_drill"]
    journal = report["decision_journal"]
    lines.extend(
        [
            "## Pre-Trade Risk Card",
            "",
            f"- planned notional/margin: `${pre['planned_notional_usdt']}` / `${pre['planned_margin_usdt']}`",
            f"- planned loss at realized 150bps stop: `${pre['planned_loss_at_realized_stop_usdt']}`",
            f"- kill rule: `{pre['kill_rule']}`",
            "",
            "## Executor Readiness",
            "",
            f"- status: `{ready['status']}`",
            f"- issues: `{ready['issues']}`",
            f"- checks: `{ready['checks']}`",
            "",
            "## Fee Tier Reality",
            "",
            f"- status: `{fee['status']}`",
            f"- assumed maker/taker: `{fee['assumed_maker_fee_bps']}` / `{fee['assumed_taker_fee_bps']}` bps",
            f"- read: {fee['recommendation']}",
            "",
            "## Stop Slippage Tracker",
            "",
            f"- status: `{stop['status']}`",
            f"- historical stop N: `{stop['historical_stop_n']}`",
            f"- summary: `{stop['historical_stop_summary']}`",
            "",
            "## Kill Switch Drill",
            "",
            f"- status: `{drill['status']}`",
            f"- path: `{drill['kill_switch_path']}` exists=`{drill['currently_exists']}`",
            f"- expected: {drill['expected_executor_behavior']}",
            "",
            "## Decision Journal",
            "",
            f"- status: `{journal['status']}` path: `{journal['path']}` rows: `{journal['rows']}` created: `{journal['created']}`",
            "",
            "## Latest Autopsy Cards",
            "",
        ]
    )
    for card in report["post_trade_autopsy_cards"]:
        lines.append(
            f"- `{card['signal_utc']}` net={card['net_bps']} atomic={card['atomicity_adverse_bps']} failure={card['failure_mode']} pnl={card['pnl_by_mode_usdt']}"
        )
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="S34 v10 operational risk suite.")
    p.add_argument("--db", type=Path, default=DB)
    p.add_argument("--env", type=Path, default=ENV)
    p.add_argument("--management-json", type=Path, default=MGMT)
    p.add_argument("--management-ledger", type=Path, default=MGMT_LEDGER)
    p.add_argument("--stop-json", type=Path, default=STOP)
    p.add_argument("--v8-json", type=Path, default=V8)
    p.add_argument("--v9-json", type=Path, default=V9)
    p.add_argument("--live-state", type=Path, default=LIVE_STATE)
    p.add_argument("--decision-journal", type=Path, default=DECISION_JOURNAL)
    p.add_argument("--out-json", type=Path, default=OUT_JSON)
    p.add_argument("--out-md", type=Path, default=OUT_MD)
    p.add_argument("--equity-usdt", type=float, default=EQUITY)
    p.add_argument("--trials", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=734)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    report = build_report(args)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=True), encoding="utf-8")
    args.out_md.write_text(render_md(report), encoding="utf-8")
    print(render_md(report))
    print(f"Wrote {args.out_json}")
    print(f"Wrote {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
