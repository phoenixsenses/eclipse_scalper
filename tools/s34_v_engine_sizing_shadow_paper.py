"""S34 V Engine sizing shadow-paper ledger.

Observation only. Takes the live-matching v0.2 shadow mirror fills and replays
their P&L under separate sizing modes:

- CURRENT_ENV: live configured 40x plan, for comparison.
- BALANCED: stop-reliability-weighted system recommendation.
- SURVIVAL: tail-only hard floor.

This never places orders and never edits live executor state.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MIRROR_LEDGER = ROOT / "reports" / "research" / "s34" / "S34_V_ENGINE_V0_2_SHADOW_MIRROR_LEDGER.jsonl"
RISK_SUITE_JSON = ROOT / "reports" / "research" / "s34" / "S34_V10_OPERATIONAL_RISK_SUITE.json"
OUT_DIR = ROOT / "reports" / "research" / "s34"
OUT_JSONL = OUT_DIR / "S34_V_ENGINE_SIZING_SHADOW_PAPER_LEDGER.jsonl"
OUT_CSV = OUT_DIR / "S34_V_ENGINE_SIZING_SHADOW_PAPER_LEDGER.csv"
OUT_JSON = OUT_DIR / "S34_V_ENGINE_SIZING_SHADOW_PAPER.json"
OUT_MD = OUT_DIR / "S34_V_ENGINE_SIZING_SHADOW_PAPER.md"
STATE_PATH = ROOT / "runtime" / "s34_v_engine_sizing_shadow_paper_state.json"
PID_PATH = ROOT / "logs" / "pids" / "s34_v_engine_sizing_shadow_paper.pid"

RULE_ID = "S34_V_ENGINE_V0_2_ETH_SELL_MAKER_LONG_H2_O20_W300_O5_DEEPBID"
STATUS = "SHADOW_PAPER_SIZING_ONLY_NO_ORDER"
EQUITY_USDT = 35.0

FIELDS = (
    "shadow_trade_id",
    "risk_mode",
    "protocol_id",
    "signal_utc",
    "maker_fill_utc",
    "exit_utc",
    "observation_status",
    "sim_status",
    "exit_reason",
    "net_bps",
    "notional_usdt",
    "margin_usdt",
    "leverage",
    "pnl_usdt",
    "equity_after_usdt",
    "drawdown_usdt",
    "drawdown_pct_equity",
    "status",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_print(text: str) -> None:
    try:
        stream = getattr(sys, "stdout", None)
        if stream is None or getattr(stream, "closed", False):
            return
        print(text, flush=True)
    except (AttributeError, BrokenPipeError, OSError, ValueError):
        return


def r1(value: float | None) -> float | None:
    if value is None or not math.isfinite(float(value)):
        return None
    return round(float(value), 1)


def r2(value: float | None) -> float | None:
    if value is None or not math.isfinite(float(value)):
        return None
    return round(float(value), 2)


def r3(value: float | None) -> float | None:
    if value is None or not math.isfinite(float(value)):
        return None
    return round(float(value), 3)


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if text:
            rows.append(json.loads(text))
    return rows


def sizing_modes(risk_suite: dict[str, Any]) -> dict[str, dict[str, float]]:
    modes = risk_suite.get("risk_budget_modes") or {}
    if modes:
        return {
            name: {
                "notional": float(row["notional"]),
                "margin": float(row["margin"]),
                "leverage": float(row.get("leverage") or 40.0),
            }
            for name, row in modes.items()
            if name in {"CURRENT_ENV", "BALANCED", "SURVIVAL", "STOP_ASSISTED"}
            and row.get("notional") is not None
            and row.get("margin") is not None
        }
    return {
        "CURRENT_ENV": {"notional": 1190.0, "margin": 29.75, "leverage": 40.0},
        "BALANCED": {"notional": 16.3, "margin": 0.4, "leverage": 40.0},
        "SURVIVAL": {"notional": 11.0, "margin": 0.3, "leverage": 40.0},
        "STOP_ASSISTED": {"notional": 39.8, "margin": 1.0, "leverage": 40.0},
    }


def closed_fill_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        if row.get("observation_status") != "CLOSED":
            continue
        if row.get("sim_status") != "FILLED":
            continue
        if row.get("net_bps") is None:
            continue
        out.append(row)
    out.sort(key=lambda r: str(r.get("signal_utc") or ""))
    return out


def build_sizing_ledger(rows: list[dict[str, Any]], modes: dict[str, dict[str, float]]) -> list[dict[str, Any]]:
    closed = closed_fill_rows(rows)
    ledger: list[dict[str, Any]] = []
    for mode_name in sorted(modes):
        mode = modes[mode_name]
        equity = EQUITY_USDT
        peak = EQUITY_USDT
        for i, row in enumerate(closed, start=1):
            net_bps = float(row["net_bps"])
            pnl = float(mode["notional"]) * net_bps / 10_000.0
            equity += pnl
            peak = max(peak, equity)
            dd = equity - peak
            ledger.append(
                {
                    "shadow_trade_id": f"{mode_name}:{row.get('observation_id') or i}",
                    "risk_mode": mode_name,
                    "protocol_id": row.get("protocol_id") or RULE_ID,
                    "signal_utc": row.get("signal_utc"),
                    "maker_fill_utc": row.get("maker_fill_utc"),
                    "exit_utc": row.get("exit_utc"),
                    "observation_status": row.get("observation_status"),
                    "sim_status": row.get("sim_status"),
                    "exit_reason": row.get("exit_reason"),
                    "net_bps": r1(net_bps),
                    "notional_usdt": r2(mode["notional"]),
                    "margin_usdt": r2(mode["margin"]),
                    "leverage": r1(mode["leverage"]),
                    "pnl_usdt": r3(pnl),
                    "equity_after_usdt": r3(equity),
                    "drawdown_usdt": r3(dd),
                    "drawdown_pct_equity": r3(abs(dd) / EQUITY_USDT * 100.0),
                    "status": STATUS,
                }
            )
    return ledger


def summarize_mode(rows: list[dict[str, Any]]) -> dict[str, Any]:
    vals = [float(r["net_bps"]) for r in rows if r.get("net_bps") is not None]
    pnl = [float(r["pnl_usdt"]) for r in rows if r.get("pnl_usdt") is not None]
    if not rows:
        return {
            "n": 0,
            "sum_bps": 0.0,
            "median_bps": None,
            "win_rate": None,
            "sum_pnl_usdt": 0.0,
            "ending_equity_usdt": EQUITY_USDT,
            "max_drawdown_pct_equity": 0.0,
        }
    return {
        "n": len(rows),
        "notional_usdt": rows[0].get("notional_usdt"),
        "margin_usdt": rows[0].get("margin_usdt"),
        "leverage": rows[0].get("leverage"),
        "sum_bps": r1(sum(vals)),
        "median_bps": r1(median(vals)) if vals else None,
        "win_rate": r3(sum(1 for v in vals if v > 0) / len(vals)) if vals else None,
        "sum_pnl_usdt": r3(sum(pnl)),
        "ending_equity_usdt": rows[-1].get("equity_after_usdt"),
        "max_drawdown_pct_equity": r3(max(float(r.get("drawdown_pct_equity") or 0.0) for r in rows)),
        "max_loss_usdt": r3(min(pnl)) if pnl else None,
        "max_win_usdt": r3(max(pnl)) if pnl else None,
    }


def build_report(ledger: list[dict[str, Any]], *, source_rows: int, risk_suite_path: Path) -> dict[str, Any]:
    by_mode: dict[str, list[dict[str, Any]]] = {}
    for row in ledger:
        by_mode.setdefault(str(row["risk_mode"]), []).append(row)
    return {
        "generated_at_utc": utc_now(),
        "status": STATUS,
        "rule_id": RULE_ID,
        "source_shadow_mirror_ledger": str(MIRROR_LEDGER),
        "source_shadow_rows": int(source_rows),
        "source_risk_suite": str(risk_suite_path),
        "equity_assumption_usdt": EQUITY_USDT,
        "modes": {name: summarize_mode(rows) for name, rows in sorted(by_mode.items())},
        "read": (
            "Separate sizing shadow-paper for the same v0.2 alpha fills. "
            "CURRENT_ENV mirrors configured live sizing for comparison; BALANCED/SURVIVAL are system risk recommendations. "
            "No order is sent and no live config is changed."
        ),
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(FIELDS), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def render_md(report: dict[str, Any]) -> str:
    lines = [
        "# S34 V Engine Sizing Shadow Paper",
        "",
        f"Generated: `{report['generated_at_utc']}`",
        "",
        f"Status: `{report['status']}`. Same v0.2 shadow fills, separate sizing ledgers. No live order/config change.",
        "",
        "| Mode | N | Notional | Margin | Lev | Sum bps | Median | Win | PnL USDT | End Equity | Max DD % | Max Loss |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, row in report["modes"].items():
        lines.append(
            f"| {name} | {row.get('n')} | {row.get('notional_usdt')} | {row.get('margin_usdt')} | "
            f"{row.get('leverage')} | {row.get('sum_bps')} | {row.get('median_bps')} | "
            f"{row.get('win_rate')} | {row.get('sum_pnl_usdt')} | {row.get('ending_equity_usdt')} | "
            f"{row.get('max_drawdown_pct_equity')} | {row.get('max_loss_usdt')} |"
        )
    lines.extend(["", report["read"], ""])
    return "\n".join(lines)


def run(args: argparse.Namespace) -> dict[str, Any]:
    source_rows = load_jsonl(args.shadow_ledger)
    modes = sizing_modes(load_json(args.risk_suite_json, {}))
    ledger = build_sizing_ledger(source_rows, modes)
    report = build_report(ledger, source_rows=len(source_rows), risk_suite_path=args.risk_suite_json)
    write_jsonl(args.out_jsonl, ledger)
    write_csv(args.out_csv, ledger)
    args.out_json.write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    args.out_md.write_text(render_md(report), encoding="utf-8")
    write_state(args.state_path, report, loop=bool(args.loop))
    return report


def write_state(path: Path, report: dict[str, Any], *, loop: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    balanced = (report.get("modes") or {}).get("BALANCED") or {}
    payload = {
        "updated_at_utc": utc_now(),
        "pid": os.getpid(),
        "loop": bool(loop),
        "status": report.get("status"),
        "rule_id": report.get("rule_id"),
        "source_shadow_rows": report.get("source_shadow_rows"),
        "balanced_n": balanced.get("n"),
        "balanced_notional_usdt": balanced.get("notional_usdt"),
        "balanced_margin_usdt": balanced.get("margin_usdt"),
        "balanced_sum_pnl_usdt": balanced.get("sum_pnl_usdt"),
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build S34 v0.2 sizing shadow-paper ledgers.")
    p.add_argument("--shadow-ledger", type=Path, default=MIRROR_LEDGER)
    p.add_argument("--risk-suite-json", type=Path, default=RISK_SUITE_JSON)
    p.add_argument("--out-jsonl", type=Path, default=OUT_JSONL)
    p.add_argument("--out-csv", type=Path, default=OUT_CSV)
    p.add_argument("--out-json", type=Path, default=OUT_JSON)
    p.add_argument("--out-md", type=Path, default=OUT_MD)
    p.add_argument("--state-path", type=Path, default=STATE_PATH)
    p.add_argument("--pid-path", type=Path, default=PID_PATH)
    p.add_argument("--loop", action="store_true")
    p.add_argument("--interval-sec", type=int, default=60)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.loop:
        args.pid_path.parent.mkdir(parents=True, exist_ok=True)
        args.pid_path.write_text(str(os.getpid()), encoding="utf-8")
        while True:
            report = run(args)
            safe_print(
                f"{utc_now()} {RULE_ID} sizing_shadow rows={report.get('source_shadow_rows')} "
                f"balanced_pnl={(report.get('modes') or {}).get('BALANCED', {}).get('sum_pnl_usdt')}"
            )
            time.sleep(max(10, int(args.interval_sec)))
    report = run(args)
    safe_print(render_md(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
