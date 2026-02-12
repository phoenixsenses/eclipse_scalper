#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Mapping


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
        if out != out:
            return default
        return out
    except Exception:
        return default


def _split_symbols(raw: str) -> list[str]:
    text = str(raw or "").strip()
    if not text:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for item in text.replace(";", ",").split(","):
        sym = str(item or "").strip().upper()
        if not sym or sym in seen:
            continue
        seen.add(sym)
        out.append(sym)
    return out


def _parse_corr_groups(raw: str) -> tuple[dict[str, list[str]], list[str]]:
    text = str(raw or "").strip()
    if not text:
        return {}, []
    out: dict[str, list[str]] = {}
    errors: list[str] = []
    chunks = [c.strip() for c in text.split(";") if str(c or "").strip()]
    for chunk in chunks:
        if ":" not in chunk:
            errors.append(f"invalid CORR_GROUPS token '{chunk}' (missing ':').")
            continue
        gname, members_raw = chunk.split(":", 1)
        group = str(gname or "").strip().upper()
        members = _split_symbols(members_raw)
        if not group:
            errors.append("invalid CORR_GROUPS token with empty group name.")
            continue
        if not members:
            errors.append(f"group '{group}' has no symbols.")
            continue
        out[group] = members
    return out, errors


def _is_true(raw: str) -> bool:
    return str(raw or "").strip().lower() in ("1", "true", "yes", "on")


def validate_env(
    env: Mapping[str, str],
    *,
    safe_profile_max_leverage: float = 1.0,
    daily_loss_max: float = 0.05,
    drawdown_max: float = 0.20,
) -> tuple[list[str], list[str], dict[str, Any]]:
    errors: list[str] = []
    warnings: list[str] = []

    dry_run_raw = str(env.get("SCALPER_DRY_RUN", "") or "").strip()
    if dry_run_raw not in ("0", "1"):
        errors.append("SCALPER_DRY_RUN must be '0' or '1'.")
    dry_run = dry_run_raw == "1"

    active_symbols = _split_symbols(str(env.get("ACTIVE_SYMBOLS", "") or ""))
    if not active_symbols:
        errors.append("ACTIVE_SYMBOLS must be non-empty.")

    leverage = _safe_float(env.get("LEVERAGE", ""), -1.0)
    if leverage <= 0:
        errors.append("LEVERAGE must be a positive number.")
    elif leverage > float(max(1.0, safe_profile_max_leverage)):
        errors.append(
            f"LEVERAGE={leverage:g} exceeds SAFE_PROFILE_MAX_LEVERAGE={float(safe_profile_max_leverage):g}."
        )

    margin_mode = str(env.get("MARGIN_MODE", "") or "").strip().lower()
    if margin_mode != "isolated":
        errors.append("MARGIN_MODE must be 'isolated' for this safe live profile.")

    max_daily = _safe_float(env.get("MAX_DAILY_LOSS_PCT", ""), -1.0)
    max_dd = _safe_float(env.get("MAX_DRAWDOWN_PCT", ""), -1.0)
    if max_daily <= 0:
        errors.append("MAX_DAILY_LOSS_PCT must be set to a positive value.")
    elif max_daily > float(daily_loss_max):
        errors.append(f"MAX_DAILY_LOSS_PCT={max_daily:.4f} exceeds safe max {daily_loss_max:.4f}.")
    if max_dd <= 0:
        errors.append("MAX_DRAWDOWN_PCT must be set to a positive value.")
    elif max_dd > float(drawdown_max):
        errors.append(f"MAX_DRAWDOWN_PCT={max_dd:.4f} exceeds safe max {drawdown_max:.4f}.")
    if max_daily > 0 and max_dd > 0 and max_dd < max_daily:
        errors.append("MAX_DRAWDOWN_PCT must be >= MAX_DAILY_LOSS_PCT.")

    first_live = _is_true(env.get("FIRST_LIVE_SAFE", "0"))
    first_live_symbols = _split_symbols(str(env.get("FIRST_LIVE_SYMBOLS", "") or ""))
    first_live_cap = _safe_float(env.get("FIRST_LIVE_MAX_NOTIONAL_USDT", ""), -1.0)
    fixed_notional = _safe_float(env.get("FIXED_NOTIONAL_USDT", ""), -1.0)
    if fixed_notional <= 0:
        errors.append("FIXED_NOTIONAL_USDT must be a positive value.")
    if first_live:
        if not first_live_symbols:
            errors.append("FIRST_LIVE_SAFE=1 requires FIRST_LIVE_SYMBOLS to be non-empty.")
        if first_live_cap <= 0:
            errors.append("FIRST_LIVE_SAFE=1 requires FIRST_LIVE_MAX_NOTIONAL_USDT > 0.")
        if fixed_notional > 0 and first_live_cap > 0 and fixed_notional > first_live_cap:
            errors.append(
                "FIXED_NOTIONAL_USDT must be <= FIRST_LIVE_MAX_NOTIONAL_USDT when FIRST_LIVE_SAFE=1."
            )
        unknown = [s for s in active_symbols if s not in set(first_live_symbols)]
        if unknown:
            warnings.append(f"ACTIVE_SYMBOLS not in FIRST_LIVE_SYMBOLS allowlist: {', '.join(unknown)}")

    groups, group_errors = _parse_corr_groups(str(env.get("CORR_GROUPS", "") or ""))
    errors.extend(group_errors)
    corr_max_positions = int(_safe_float(env.get("CORR_GROUP_MAX_POSITIONS", "0"), 0.0))
    corr_max_notional = _safe_float(env.get("CORR_GROUP_MAX_NOTIONAL_USDT", "0"), 0.0)
    if groups and corr_max_positions <= 0 and corr_max_notional <= 0:
        warnings.append("CORR_GROUPS set but no global CORR_GROUP caps configured.")

    telemetry_path = str(env.get("TELEMETRY_PATH", "logs/telemetry.jsonl") or "logs/telemetry.jsonl").strip()
    reliability_path = str(env.get("RELIABILITY_GATE_PATH", "logs/reliability_gate.txt") or "logs/reliability_gate.txt").strip()
    reliability_window_seconds = _safe_float(env.get("RELIABILITY_GATE_WINDOW_SECONDS", "14400"), 14400.0)
    reliability_stale_seconds = _safe_float(env.get("RELIABILITY_GATE_STALE_SECONDS", "900"), 900.0)
    reliability_stage1_fail_max = _safe_float(env.get("RELIABILITY_GATE_MAX_STAGE1_PROTECTION_FAIL_COUNT", "0"), 0.0)
    if reliability_stage1_fail_max < 0:
        errors.append("RELIABILITY_GATE_MAX_STAGE1_PROTECTION_FAIL_COUNT must be >= 0.")
    journal_path = str(env.get("EVENT_JOURNAL_PATH", "logs/execution_journal.jsonl") or "logs/execution_journal.jsonl").strip()
    reliability_age_seconds = -1.0
    try:
        rp = Path(reliability_path)
        if rp.exists():
            reliability_age_seconds = max(0.0, float(time.time() - float(rp.stat().st_mtime)))
            if reliability_stale_seconds > 0.0 and reliability_age_seconds > reliability_stale_seconds:
                warnings.append(
                    (
                        "RELIABILITY_GATE_PATH appears stale "
                        f"(age={reliability_age_seconds:.0f}s > limit={reliability_stale_seconds:.0f}s). "
                        "Refresh before launch."
                    )
                )
        else:
            warnings.append("RELIABILITY_GATE_PATH missing; refresh before launch.")
    except Exception:
        warnings.append("Unable to read RELIABILITY_GATE_PATH metadata; refresh before launch.")

    summary: dict[str, Any] = {
        "dry_run": bool(dry_run),
        "active_symbols": active_symbols,
        "leverage": float(leverage if leverage > 0 else 0.0),
        "margin_mode": margin_mode,
        "fixed_notional_usdt": float(fixed_notional if fixed_notional > 0 else 0.0),
        "first_live_safe": bool(first_live),
        "first_live_symbols": first_live_symbols,
        "first_live_max_notional_usdt": float(first_live_cap if first_live_cap > 0 else 0.0),
        "max_daily_loss_pct": float(max_daily if max_daily > 0 else 0.0),
        "max_drawdown_pct": float(max_dd if max_dd > 0 else 0.0),
        "corr_group_count": int(len(groups)),
        "corr_group_max_positions": int(corr_max_positions),
        "corr_group_max_notional_usdt": float(corr_max_notional),
        "runtime_reliability_coupling": _is_true(env.get("RUNTIME_RELIABILITY_COUPLING", "1")),
        "telemetry_path": telemetry_path,
        "reliability_gate_path": reliability_path,
        "reliability_gate_window_seconds": float(reliability_window_seconds if reliability_window_seconds > 0 else 0.0),
        "reliability_gate_stale_seconds": float(reliability_stale_seconds if reliability_stale_seconds > 0 else 0.0),
        "reliability_gate_max_stage1_protection_fail_count": float(max(0.0, reliability_stage1_fail_max)),
        "reliability_gate_age_seconds": float(reliability_age_seconds if reliability_age_seconds >= 0 else -1.0),
        "event_journal_path": journal_path,
    }
    return errors, warnings, summary


def _format_summary(summary: Mapping[str, Any], errors: list[str], warnings: list[str]) -> str:
    lines = [
        "Safe Profile Preflight",
        "======================",
        f"mode={'dry-run' if bool(summary.get('dry_run')) else 'live'}",
        f"symbols={','.join(summary.get('active_symbols', []) or []) or 'n/a'}",
        f"notional={float(summary.get('fixed_notional_usdt', 0.0) or 0.0):.2f}",
        f"leverage={float(summary.get('leverage', 0.0) or 0.0):.2f}",
        f"margin_mode={str(summary.get('margin_mode') or '')}",
        f"first_live_safe={bool(summary.get('first_live_safe', False))}",
        f"first_live_symbols={','.join(summary.get('first_live_symbols', []) or []) or 'n/a'}",
        f"first_live_cap={float(summary.get('first_live_max_notional_usdt', 0.0) or 0.0):.2f}",
        f"max_daily_loss_pct={float(summary.get('max_daily_loss_pct', 0.0) or 0.0):.4f}",
        f"max_drawdown_pct={float(summary.get('max_drawdown_pct', 0.0) or 0.0):.4f}",
        f"corr_group_count={int(summary.get('corr_group_count', 0) or 0)}",
        f"runtime_reliability_coupling={bool(summary.get('runtime_reliability_coupling', True))}",
        f"telemetry_path={str(summary.get('telemetry_path') or '')}",
        f"reliability_gate_path={str(summary.get('reliability_gate_path') or '')}",
        f"reliability_gate_window_seconds={float(summary.get('reliability_gate_window_seconds', 0.0) or 0.0):.0f}",
        f"reliability_gate_stale_seconds={float(summary.get('reliability_gate_stale_seconds', 0.0) or 0.0):.0f}",
        f"reliability_gate_max_stage1_protection_fail_count={float(summary.get('reliability_gate_max_stage1_protection_fail_count', 0.0) or 0.0):.0f}",
        f"reliability_gate_age_seconds={float(summary.get('reliability_gate_age_seconds', -1.0) or -1.0):.0f}",
        f"event_journal_path={str(summary.get('event_journal_path') or '')}",
        f"errors={len(errors)} warnings={len(warnings)}",
    ]
    if errors:
        lines.append("errors_list:")
        lines.extend([f"- {e}" for e in errors])
    if warnings:
        lines.append("warnings_list:")
        lines.extend([f"- {w}" for w in warnings])
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Validate safe live run profile environment before launch.")
    ap.add_argument("--max-leverage", type=float, default=1.0)
    ap.add_argument("--daily-loss-max", type=float, default=0.05)
    ap.add_argument("--drawdown-max", type=float, default=0.20)
    ap.add_argument("--json-out", default="", help="Optional path to write structured summary JSON.")
    args = ap.parse_args(argv)

    errors, warnings, summary = validate_env(
        os.environ,
        safe_profile_max_leverage=max(1.0, float(args.max_leverage)),
        daily_loss_max=max(1e-6, float(args.daily_loss_max)),
        drawdown_max=max(1e-6, float(args.drawdown_max)),
    )
    report = _format_summary(summary, errors, warnings)
    print(report.strip())

    if args.json_out:
        payload = {"summary": summary, "errors": errors, "warnings": warnings}
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return 2 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
