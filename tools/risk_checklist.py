#!/usr/bin/env python3
"""
Print a risk checklist for live sessions.
"""

from __future__ import annotations

import os
import re
from pathlib import Path


def _env_line(name: str, default: str = "") -> str:
    val = os.getenv(name, default)
    return f"{name}={val}"


def _print_env_snapshot() -> None:
    print("Env Snapshot")
    keys = [
        "SCALPER_DRY_RUN",
        "FIRST_LIVE_SAFE",
        "FIRST_LIVE_SYMBOLS",
        "FIXED_NOTIONAL_USDT",
        "LEVERAGE",
        "LEVERAGE_MIN",
        "LEVERAGE_MAX",
        "LEVERAGE_BY_SYMBOL",
        "LEVERAGE_BY_GROUP",
        "LEVERAGE_GROUP_DYNAMIC",
        "LEVERAGE_GROUP_SCALE",
        "LEVERAGE_GROUP_SCALE_MIN",
        "LEVERAGE_GROUP_EXCLUDE_SELF",
        "LEVERAGE_GROUP_EXPOSURE",
        "LEVERAGE_GROUP_EXPOSURE_REF_PCT",
        "MARGIN_MODE",
        "MAX_DAILY_LOSS_PCT",
        "MAX_DRAWDOWN_PCT",
        "KILL_DAILY_HALT_SEC",
        "KILL_DRAWDOWN_HALT_SEC",
        "CORR_GROUPS",
        "CORR_GROUP_MAX_POSITIONS",
        "CORR_GROUP_MAX_NOTIONAL_USDT",
        "CORR_GROUP_LIMITS",
        "CORR_GROUP_NOTIONAL",
    ]
    for k in keys:
        print(f"- {_env_line(k, '')}")


def _parse_ps1_env(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    out: dict[str, str] = {}
    pattern = re.compile(r"^\s*\$env:([A-Za-z0-9_]+)\s*=\s*\"(.*)\"\s*$")
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            m = pattern.match(line)
            if not m:
                continue
            key = m.group(1)
            val = m.group(2)
            out[key] = val
    except Exception:
        return out
    return out


def _print_side_by_side() -> None:
    keys = [
        "SCALPER_DRY_RUN",
        "FIRST_LIVE_SAFE",
        "FIRST_LIVE_SYMBOLS",
        "FIXED_NOTIONAL_USDT",
        "LEVERAGE",
        "LEVERAGE_MIN",
        "LEVERAGE_MAX",
        "LEVERAGE_BY_SYMBOL",
        "LEVERAGE_BY_GROUP",
        "LEVERAGE_GROUP_DYNAMIC",
        "LEVERAGE_GROUP_SCALE",
        "LEVERAGE_GROUP_SCALE_MIN",
        "LEVERAGE_GROUP_EXCLUDE_SELF",
        "LEVERAGE_GROUP_EXPOSURE",
        "LEVERAGE_GROUP_EXPOSURE_REF_PCT",
        "MARGIN_MODE",
        "MAX_DAILY_LOSS_PCT",
        "MAX_DRAWDOWN_PCT",
        "KILL_DAILY_HALT_SEC",
        "KILL_DRAWDOWN_HALT_SEC",
        "CORR_GROUPS",
        "CORR_GROUP_MAX_POSITIONS",
        "CORR_GROUP_MAX_NOTIONAL_USDT",
        "CORR_GROUP_LIMITS",
        "CORR_GROUP_NOTIONAL",
    ]
    root = Path(__file__).resolve().parents[1]
    ps1_a = root / "run-bot.ps1"
    ps1_b = root / "run-bot-ps2.ps1"
    vals_a = _parse_ps1_env(ps1_a)
    vals_b = _parse_ps1_env(ps1_b)
    print("Script Snapshot (side-by-side)")
    header = f"{'KEY':<28} {'run-bot.ps1':<24} {'run-bot-ps2.ps1':<24}"
    print(header)
    print("-" * len(header))
    for k in keys:
        a = vals_a.get(k, "")
        b = vals_b.get(k, "")
        print(f"{k:<28} {a:<24} {b:<24}")


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Print a risk checklist.")
    parser.add_argument(
        "--env",
        action="store_true",
        help="Print current environment snapshot for key risk variables.",
    )
    parser.add_argument(
        "--scripts",
        action="store_true",
        help="Print run-bot.ps1 and run-bot-ps2.ps1 values side-by-side.",
    )
    args = parser.parse_args()

    print("Risk Checklist (Quick)")
    print("- Confirm SCALPER_DRY_RUN=0 is intentional.")
    print("- Verify FIRST_LIVE_SAFE=1 and FIRST_LIVE_SYMBOLS allowlist is set.")
    print("- Confirm FIXED_NOTIONAL_USDT and LEVERAGE are sized for account risk.")
    print("- If using per-symbol/group leverage, confirm LEVERAGE_BY_SYMBOL/LEVERAGE_BY_GROUP + min/max caps.")
    print("- If using dynamic group leverage, confirm LEVERAGE_GROUP_SCALE and min floor.")
    print("- Set MAX_DAILY_LOSS_PCT and MAX_DRAWDOWN_PCT.")
    print("- Set KILL_DAILY_HALT_SEC and KILL_DRAWDOWN_HALT_SEC (auto-pause).")
    print("- Set correlation caps (CORR_GROUP_*) for your symbol set.")
    print("- Ensure telemetry and audit logs are being written.")
    print("- Monitor roll/jitter alerts: python tools/telemetry_roll_alerts.py --since-min 120")
    print("- Watch counter thresholds: python tools/telemetry_threshold_alerts.py --thresholds entry.blocked=3,order.create.retry_alert=2")
    print("- Review error classes: python tools/telemetry_error_classes.py --since-min 60")
    if args.env:
        print("")
        _print_env_snapshot()
    if args.scripts:
        print("")
        _print_side_by_side()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
