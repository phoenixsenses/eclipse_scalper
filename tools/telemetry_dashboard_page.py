#!/usr/bin/env python3
"""
Telemetry dashboard page.

Produces a simple HTML summary combining the core health, signal data health,
and anomaly reports so you can open a single dashboard page.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from collections import Counter
from typing import Optional

try:
    from eclipse_scalper.tools import replay_trade as _replay_trade  # type: ignore
except Exception:
    try:
        from tools import replay_trade as _replay_trade  # type: ignore
    except Exception:
        _replay_trade = None


def _read_text_safe(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        try:
            return path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            try:
                return path.read_bytes().decode("utf-8", errors="replace")
            except Exception:
                return ""


def _read_section(path: Path, title: str) -> str:
    if not path.exists():
        return f"<h2>{title}</h2><p><em>Missing {path.name}</em></p>"
    text = _read_text_safe(path)
    if not text:
        return f"<h2>{title}</h2><p><em>Empty {path.name}</em></p>"
    return f"<h2>{title}</h2><pre>{text}</pre>"


def _read_json_section(path: Path, title: str) -> str:
    if not path.exists():
        return f"<h2>{title}</h2><p><em>Missing {path.name}</em></p>"
    raw = _read_text_safe(path)
    if not raw:
        return f"<h2>{title}</h2><p><em>Empty {path.name}</em></p>"
    try:
        payload = json.loads(raw)
        text = json.dumps(payload, indent=2)
    except Exception:
        text = raw
    return f"<h2>{title}</h2><pre>{text}</pre>"


def _read_csv_section(path: Path, title: str, limit: int = 5) -> str:
    if not path.exists():
        return f"<h2>{title}</h2><p><em>Missing {path.name}</em></p>"
    lines = _read_text_safe(path).splitlines()
    if not lines:
        return f"<h2>{title}</h2><p><em>Empty {path.name}</em></p>"
    header = lines[0]
    rows = lines[1:]
    tail = rows[-limit:] if rows else []
    text = "\n".join([header, *tail])
    return f"<h2>{title}</h2><pre>{text}</pre>"


def _read_reliability_gate_section(path: Path) -> str:
    if not path.exists():
        return f"<h2>Reliability Gate</h2><p><em>Missing {path.name}</em></p>"
    raw = _read_text_safe(path).strip()
    if not raw:
        return f"<h2>Reliability Gate</h2><p><em>Empty {path.name}</em></p>"
    kv: dict[str, str] = {}
    mismatch_ids: list[str] = []
    in_ids = False
    for line in raw.splitlines():
        s = str(line or "").strip()
        if not s:
            if in_ids:
                break
            continue
        if s.lower() == "replay_mismatch_ids:":
            in_ids = True
            continue
        if in_ids:
            if not s.startswith("-"):
                in_ids = False
            else:
                cid = str(s[1:] or "").strip()
                if cid:
                    mismatch_ids.append(cid)
                continue
        if "=" not in s:
            continue
        k, v = s.split("=", 1)
        key = str(k or "").strip()
        if key:
            kv[key] = str(v or "").strip()
    mismatch = int(float(kv.get("replay_mismatch_count", "0") or 0))
    invalid = int(float(kv.get("invalid_transition_count", "0") or 0))
    coverage = float(kv.get("journal_coverage_ratio", "0") or 0.0)
    position_mismatch = int(float(kv.get("position_mismatch_count", "0") or 0))
    position_mismatch_peak = int(float(kv.get("position_mismatch_count_peak", str(position_mismatch)) or position_mismatch))
    orphan_count = int(float(kv.get("orphan_count", "0") or 0))
    intent_collision_count = int(float(kv.get("intent_collision_count", "0") or 0))
    coverage_gap_seconds = float(kv.get("protection_coverage_gap_seconds", "0") or 0.0)
    coverage_gap_seconds_peak = float(
        kv.get("protection_coverage_gap_seconds_peak", str(coverage_gap_seconds)) or coverage_gap_seconds
    )
    replace_race_count = int(float(kv.get("replace_race_count", "0") or 0))
    contradiction_count = int(float(kv.get("evidence_contradiction_count", "0") or 0))
    categories = {
        "ledger": 0,
        "transition": 0,
        "belief": 0,
        "position": 0,
        "orphan": 0,
        "coverage_gap": 0,
        "replace_race": 0,
        "contradiction": 0,
        "unknown": 0,
    }
    raw_cats = str(kv.get("replay_mismatch_categories", "") or "").strip()
    if raw_cats:
        try:
            payload = json.loads(raw_cats)
            if isinstance(payload, dict):
                for k in list(categories.keys()):
                    categories[k] = int(float(payload.get(k, 0) or 0))
        except Exception:
            pass
    status = "OK" if mismatch <= 0 and invalid <= 0 and coverage >= 0.90 else "DEGRADED"
    lines = [
        f"status: {status}",
        f"replay_mismatch_count: {mismatch}",
        f"invalid_transition_count: {invalid}",
        f"journal_coverage_ratio: {coverage:.3f}",
        f"position_mismatch_count: {position_mismatch} (peak {position_mismatch_peak})",
        f"orphan_count: {orphan_count}",
        f"intent_collision_count: {intent_collision_count}",
        f"protection_coverage_gap_seconds: {coverage_gap_seconds:.1f} (peak {coverage_gap_seconds_peak:.1f})",
        f"replace_race_count: {replace_race_count}",
        f"evidence_contradiction_count: {contradiction_count}",
    ]
    if any(int(categories.get(k, 0)) > 0 for k in categories.keys()):
        lines.append(
            "mismatch_categories: "
            f"ledger={int(categories['ledger'])} "
            f"transition={int(categories['transition'])} "
            f"belief={int(categories['belief'])} "
            f"position={int(categories['position'])} "
            f"orphan={int(categories['orphan'])} "
            f"coverage_gap={int(categories['coverage_gap'])} "
            f"replace_race={int(categories['replace_race'])} "
            f"contradiction={int(categories['contradiction'])} "
            f"unknown={int(categories['unknown'])}"
        )
        ranked = sorted(
            [(str(k), int(v)) for (k, v) in categories.items() if int(v) > 0],
            key=lambda kv: int(kv[1]),
            reverse=True,
        )[:3]
        if ranked:
            lines.append("top_contributors: " + ", ".join(f"{k}={v}" for (k, v) in ranked))
        critical_keys = ("position", "orphan", "coverage_gap", "replace_race", "contradiction")
        critical_ranked = sorted(
            [(str(k), int(categories.get(k, 0))) for k in critical_keys if int(categories.get(k, 0)) > 0],
            key=lambda kv: int(kv[1]),
            reverse=True,
        )[:3]
        if critical_ranked:
            lines.append("critical_contributors: " + ", ".join(f"{k}={v}" for (k, v) in critical_ranked))
    if mismatch_ids:
        lines.append("top_missing_ids: " + ", ".join(mismatch_ids[:5]))
    return "<h2>Reliability Gate</h2><pre>" + "\n".join(lines) + "</pre>"


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    events: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s:
            continue
        try:
            events.append(json.loads(s))
        except Exception:
            continue
    return events


def _guard_reason_summary(events: list[dict], limit: int = 8) -> str:
    guard_events = {
        "entry.blocked",
        "exit.telemetry_guard",
        "order.create.retry_alert",
        "entry.partial_fill_escalation",
    }
    counts: Counter[str] = Counter()
    for ev in events:
        name = str(ev.get("event") or "")
        if name not in guard_events:
            continue
        data = ev.get("data") if isinstance(ev.get("data"), dict) else {}
        reason = (
            str(data.get("reason") or data.get("guard") or data.get("code") or "").strip().lower()
        )
        label = reason if reason else name
        counts[label] += 1
    if not counts:
        return "<h2>Top Guard Reasons</h2><p><em>No guard events found.</em></p>"
    lines = ["Top Guard Reasons", "-" * 18]
    for label, cnt in counts.most_common(limit):
        lines.append(f"{label}: {cnt}")
    return "<h2>Top Guard Reasons</h2><pre>" + "\n".join(lines) + "</pre>"


def _top_reliability_strip(events: list[dict]) -> str:
    latest_reconcile: dict = {}
    latest_belief: dict = {}
    latest_corr: dict = {}
    for ev in events:
        name = str(ev.get("event") or "")
        data = ev.get("data") if isinstance(ev.get("data"), dict) else {}
        if name == "reconcile.summary":
            latest_reconcile = data
        elif name == "execution.belief_state":
            latest_belief = data
        elif name == "execution.correlation_state":
            latest_corr = data
    if not latest_reconcile and not latest_belief:
        return "<div><em>No reliability telemetry yet.</em></div>"

    mode = str(latest_belief.get("guard_mode") or "n/a")
    allow_entries = bool(latest_belief.get("allow_entries", True))
    debt_sec = float(latest_belief.get("belief_debt_sec") or 0.0)
    mismatch_streak = int(latest_belief.get("mismatch_streak") or 0)
    coverage_gap = float(latest_reconcile.get("protection_coverage_gap_seconds") or 0.0)
    budget_blocked = int(latest_reconcile.get("protection_refresh_budget_blocked_count") or 0)
    budget_force = int(latest_reconcile.get("protection_refresh_budget_force_override_count") or 0)
    budget_blocked_level = float(latest_belief.get("guard_refresh_blocked_level") or 0.0)
    budget_force_level = float(latest_belief.get("guard_refresh_force_level") or 0.0)
    intent_collision_count = int(
        latest_belief.get("runtime_gate_intent_collision_count")
        or latest_reconcile.get("runtime_gate_intent_collision_count")
        or 0
    )
    intent_collision_streak = int(latest_belief.get("intent_collision_streak") or 0)
    corr_regime = str(latest_belief.get("corr_regime") or latest_corr.get("corr_regime") or "").strip().upper()
    corr_pressure = float(
        latest_belief.get("corr_pressure")
        or latest_corr.get("corr_pressure")
        or 0.0
    )
    mode_upper = mode.upper()

    def _env_float(name: str, default: float) -> float:
        raw = str(os.getenv(name, "") or "").strip()
        if not raw:
            return float(default)
        try:
            return float(raw)
        except Exception:
            return float(default)

    debt_warn = _env_float("TELEMETRY_DASHBOARD_DEBT_WARN_SEC", 30.0)
    debt_crit = _env_float("TELEMETRY_DASHBOARD_DEBT_CRIT_SEC", 90.0)
    mismatch_warn = _env_float("TELEMETRY_DASHBOARD_MISMATCH_STREAK_WARN", 3.0)
    mismatch_crit = _env_float("TELEMETRY_DASHBOARD_MISMATCH_STREAK_CRIT", 8.0)
    gap_warn = _env_float("TELEMETRY_DASHBOARD_COVERAGE_GAP_WARN_SEC", 2.0)
    gap_crit = _env_float("TELEMETRY_DASHBOARD_COVERAGE_GAP_CRIT_SEC", 5.0)
    blocked_warn = _env_float("TELEMETRY_DASHBOARD_REFRESH_BLOCKED_WARN", 1.0)
    blocked_crit = _env_float("TELEMETRY_DASHBOARD_REFRESH_BLOCKED_CRIT", 4.0)
    force_warn = _env_float("TELEMETRY_DASHBOARD_FORCE_OVERRIDE_WARN", 1.0)
    force_crit = _env_float("TELEMETRY_DASHBOARD_FORCE_OVERRIDE_CRIT", 3.0)
    collision_warn = _env_float("TELEMETRY_DASHBOARD_INTENT_COLLISION_WARN", 1.0)
    collision_crit = _env_float("TELEMETRY_DASHBOARD_INTENT_COLLISION_CRIT", 2.0)
    collision_streak_warn = _env_float("TELEMETRY_DASHBOARD_INTENT_COLLISION_STREAK_WARN", 1.0)
    collision_streak_crit = _env_float("TELEMETRY_DASHBOARD_INTENT_COLLISION_STREAK_CRIT", 2.0)
    corr_warn = _env_float("TELEMETRY_DASHBOARD_CORR_WARN", 0.60)
    corr_crit = _env_float("TELEMETRY_DASHBOARD_CORR_CRIT", 0.80)

    def _status_for_mode(v: str) -> str:
        if v in ("RED", "ORANGE"):
            return "crit"
        if v == "YELLOW":
            return "warn"
        return "ok"

    def _status_for_bool(v: bool) -> str:
        return "ok" if v else "crit"

    def _status_for_num(v: float, warn: float, crit: float) -> str:
        if v >= crit:
            return "crit"
        if v >= warn:
            return "warn"
        return "ok"

    def _status_for_corr(regime: str, pressure: float) -> str:
        r = str(regime or "").upper()
        if r == "STRESS":
            return "crit"
        if r == "TIGHTENING":
            return "warn"
        return _status_for_num(float(pressure), float(corr_warn), float(corr_crit))

    return (
        "<div class='top-strip'>"
        f"<div class='metric {_status_for_mode(mode_upper)}'><strong>mode</strong><br>{mode}</div>"
        f"<div class='metric {_status_for_bool(allow_entries)}'><strong>allow_entries</strong><br>{str(allow_entries).lower()}</div>"
        f"<div class='metric {_status_for_num(debt_sec, debt_warn, debt_crit)}'><strong>debt_sec</strong><br>{debt_sec:.1f}</div>"
        f"<div class='metric {_status_for_num(float(mismatch_streak), mismatch_warn, mismatch_crit)}'><strong>mismatch_streak</strong><br>{mismatch_streak}</div>"
        f"<div class='metric {_status_for_num(coverage_gap, gap_warn, gap_crit)}'><strong>coverage_gap_s</strong><br>{coverage_gap:.1f}</div>"
        f"<div class='metric {_status_for_num(float(intent_collision_count), collision_warn, collision_crit)}'><strong>intent_collision</strong><br>{intent_collision_count}</div>"
        f"<div class='metric {_status_for_num(float(intent_collision_streak), collision_streak_warn, collision_streak_crit)}'><strong>collision_streak</strong><br>{intent_collision_streak}</div>"
        f"<div class='metric {_status_for_num(float(budget_blocked), blocked_warn, blocked_crit)}'><strong>refresh_blocked</strong><br>{budget_blocked} ({budget_blocked_level:.2f})</div>"
        f"<div class='metric {_status_for_num(float(budget_force), force_warn, force_crit)}'><strong>force_override</strong><br>{budget_force} ({budget_force_level:.2f})</div>"
        f"<div class='metric {_status_for_corr(corr_regime, corr_pressure)}'><strong>corr_state</strong><br>{(corr_regime or 'NORMAL')} ({corr_pressure:.2f})</div>"
        "</div>"
        "<div class='metric-legend'>"
        f"corr_state thresholds: warn>={corr_warn:.2f}, crit>={corr_crit:.2f} "
        "(or regime=TIGHTENING/STRESS). "
        "Tune with TELEMETRY_DASHBOARD_CORR_WARN / TELEMETRY_DASHBOARD_CORR_CRIT."
        "</div>"
    )


def _guard_symbol_spark(events: list[dict], limit: int = 8, width: int = 20) -> str:
    guard_events = {
        "entry.blocked",
        "exit.telemetry_guard",
        "order.create.retry_alert",
        "entry.partial_fill_escalation",
    }
    counts: Counter[str] = Counter()
    for ev in events:
        name = str(ev.get("event") or "")
        if name not in guard_events:
            continue
        symbol = str(ev.get("symbol") or "")
        if not symbol:
            data = ev.get("data") if isinstance(ev.get("data"), dict) else {}
            symbol = str(data.get("symbol") or "")
        symbol = symbol.strip().upper() if symbol else "UNKNOWN"
        counts[symbol] += 1
    if not counts:
        return "<h2>Guard Hits By Symbol</h2><p><em>No guard events found.</em></p>"
    max_count = max(counts.values()) if counts else 1
    lines = ["Guard Hits By Symbol", "-" * 20]
    for symbol, cnt in counts.most_common(limit):
        bar_len = max(1, int(round((cnt / max_count) * width))) if cnt > 0 else 0
        bar = "#" * bar_len
        lines.append(f"{symbol:10} | {bar} {cnt}")
    return "<h2>Guard Hits By Symbol</h2><pre>" + "\n".join(lines) + "</pre>"


def _partial_fill_state_summary(events: list[dict], limit: int = 8) -> str:
    rows: list[dict] = []
    for ev in events:
        if str(ev.get("event") or "") != "entry.partial_fill_state":
            continue
        data = ev.get("data") if isinstance(ev.get("data"), dict) else {}
        symbol = str(ev.get("symbol") or data.get("symbol") or "UNKNOWN").strip().upper()
        outcome = str(data.get("outcome") or "unknown").strip().lower()
        cancel_ok = bool(data.get("cancel_ok"))
        flatten_ok = bool(data.get("flatten_ok"))
        rows.append(
            {
                "symbol": symbol or "UNKNOWN",
                "outcome": outcome or "unknown",
                "cancel_ok": cancel_ok,
                "flatten_ok": flatten_ok,
            }
        )

    if not rows:
        return "<h2>Partial Fill Outcomes</h2><p><em>No entry.partial_fill_state events found.</em></p>"

    by_outcome: Counter[str] = Counter()
    by_symbol: Counter[str] = Counter()
    cancel_ok_count = 0
    flatten_ok_count = 0
    for row in rows:
        by_outcome[row["outcome"]] += 1
        by_symbol[row["symbol"]] += 1
        if row["cancel_ok"]:
            cancel_ok_count += 1
        if row["flatten_ok"]:
            flatten_ok_count += 1

    lines = [f"Partial Fill Outcomes (events={len(rows)})", "-" * 36, "By outcome:"]
    for outcome, cnt in by_outcome.most_common(limit):
        lines.append(f"- {outcome}: {cnt}")
    lines.append("")
    lines.append("Top symbols:")
    for symbol, cnt in by_symbol.most_common(limit):
        lines.append(f"- {symbol}: {cnt}")
    lines.append("")
    lines.append(
        f"cancel_ok={cancel_ok_count}/{len(rows)} | flatten_ok={flatten_ok_count}/{len(rows)}"
    )
    return "<h2>Partial Fill Outcomes</h2><pre>" + "\n".join(lines) + "</pre>"


def _reconcile_first_gate_summary(events: list[dict], limit: int = 8, severity_threshold: float = 0.85) -> str:
    rows: list[dict] = []
    max_streak = 0
    cur_streak = 0
    max_severity = 0.0
    by_symbol_severity: dict[str, float] = {}
    for ev in events:
        if str(ev.get("event") or "") != "entry.reconcile_first_gate":
            continue
        data = ev.get("data") if isinstance(ev.get("data"), dict) else {}
        symbol = str(ev.get("symbol") or data.get("symbol") or "UNKNOWN").strip().upper()
        reason = str(data.get("reason") or "runtime_gate").strip().lower()
        severity = float(data.get("reconcile_first_severity") or data.get("runtime_gate_degrade_score") or 0.0)
        if severity > max_severity:
            max_severity = severity
        sym = (symbol or "UNKNOWN")
        if severity > float(by_symbol_severity.get(sym, 0.0)):
            by_symbol_severity[sym] = float(severity)
        if severity >= max(0.0, float(severity_threshold)):
            cur_streak += 1
            if cur_streak > max_streak:
                max_streak = cur_streak
        else:
            cur_streak = 0
        rows.append({"symbol": sym, "reason": (reason or "runtime_gate"), "severity": float(severity)})
    if not rows:
        return "<h2>Reconcile-First Gate</h2><p><em>No entry.reconcile_first_gate events found.</em></p>"
    by_symbol: Counter[str] = Counter()
    by_reason: Counter[str] = Counter()
    for row in rows:
        by_symbol[row["symbol"]] += 1
        by_reason[row["reason"]] += 1
    lines = [
        (
            f"Reconcile-First Gate (events={len(rows)} max_severity={max_severity:.2f} "
            f"max_streak={max_streak})"
        ),
        "-" * 38,
        "By reason:",
    ]
    for reason, cnt in by_reason.most_common(limit):
        lines.append(f"- {reason}: {cnt}")
    lines.append("")
    lines.append("Top symbols:")
    for symbol, cnt in by_symbol.most_common(limit):
        lines.append(f"- {symbol}: {cnt}")
    sev_sorted = sorted(by_symbol_severity.items(), key=lambda kv: kv[1], reverse=True)[: max(1, int(limit))]
    if sev_sorted:
        lines.append("")
        lines.append("Top severity symbols:")
        for symbol, sev in sev_sorted:
            lines.append(f"- {symbol}: {sev:.2f}")
    return "<h2>Reconcile-First Gate</h2><pre>" + "\n".join(lines) + "</pre>"


def _entry_budget_summary(events: list[dict], limit: int = 8) -> str:
    depleted_by_symbol: Counter[str] = Counter()
    scaled_by_symbol: Counter[str] = Counter()
    depleted_total = 0
    scaled_total = 0
    for ev in events:
        name = str(ev.get("event") or "")
        data = ev.get("data") if isinstance(ev.get("data"), dict) else {}
        symbol = str(ev.get("symbol") or data.get("symbol") or "UNKNOWN").strip().upper() or "UNKNOWN"
        if name == "entry.blocked":
            reason = str(data.get("reason") or "").strip().lower()
            if reason == "entry_budget_depleted":
                depleted_total += 1
                depleted_by_symbol[symbol] += 1
        elif name == "entry.notional_scaled":
            reason = str(data.get("reason") or "").strip().lower()
            if reason == "entry_budget_allocator":
                scaled_total += 1
                scaled_by_symbol[symbol] += 1
    if depleted_total <= 0 and scaled_total <= 0:
        return "<h2>Entry Budget Pressure</h2><p><em>No entry budget pressure events found.</em></p>"
    lines = [
        "Entry Budget Pressure",
        "-" * 24,
        f"depleted_total: {depleted_total}",
        f"scaled_total: {scaled_total}",
    ]
    if depleted_total > 0:
        lines.append("")
        lines.append("Depleted top symbols:")
        for symbol, cnt in depleted_by_symbol.most_common(limit):
            lines.append(f"- {symbol}: {cnt}")
    if scaled_total > 0:
        lines.append("")
        lines.append("Scaled top symbols:")
        for symbol, cnt in scaled_by_symbol.most_common(limit):
            lines.append(f"- {symbol}: {cnt}")
    return "<h2>Entry Budget Pressure</h2><pre>" + "\n".join(lines) + "</pre>"


def _protection_refresh_budget_summary(events: list[dict]) -> str:
    rows: list[dict] = []
    for ev in events:
        if str(ev.get("event") or "") != "reconcile.summary":
            continue
        data = ev.get("data") if isinstance(ev.get("data"), dict) else {}
        rows.append(data)
    if not rows:
        return "<h2>Protection Refresh Budget</h2><p><em>No reconcile.summary events found.</em></p>"

    def _mx(key: str) -> int:
        best = 0
        for row in rows:
            v = int(float(row.get(key) or 0.0))
            if v > best:
                best = v
        return best

    blocked_total = _mx("protection_refresh_budget_blocked_count")
    force_total = _mx("protection_refresh_budget_force_override_count")
    stop_blocked = _mx("protection_refresh_stop_budget_blocked_count")
    tp_blocked = _mx("protection_refresh_tp_budget_blocked_count")
    stop_force = _mx("protection_refresh_stop_force_override_count")
    tp_force = _mx("protection_refresh_tp_force_override_count")
    if (
        blocked_total <= 0
        and force_total <= 0
        and stop_blocked <= 0
        and tp_blocked <= 0
        and stop_force <= 0
        and tp_force <= 0
    ):
        return "<h2>Protection Refresh Budget</h2><p><em>No refresh budget activity found.</em></p>"

    lines = [
        "Protection Refresh Budget",
        "-" * 28,
        f"blocked_total: {blocked_total} (stop={stop_blocked}, tp={tp_blocked})",
        f"force_override_total: {force_total} (stop={stop_force}, tp={tp_force})",
    ]
    return "<h2>Protection Refresh Budget</h2><pre>" + "\n".join(lines) + "</pre>"


def _refresh_pressure_trend_summary(events: list[dict], limit: int = 8) -> str:
    rows: list[dict] = []
    for ev in events:
        if str(ev.get("event") or "") != "execution.belief_state":
            continue
        data = ev.get("data") if isinstance(ev.get("data"), dict) else {}
        rows.append(
            {
                "blocked_level": float(data.get("guard_refresh_blocked_level") or 0.0),
                "force_level": float(data.get("guard_refresh_force_level") or 0.0),
                "allow_entries": bool(data.get("allow_entries", True)),
                "mode": str(data.get("guard_mode") or ""),
                "stage": str(data.get("guard_recovery_stage") or ""),
            }
        )
    if not rows:
        return "<h2>Refresh Pressure Trend</h2><p><em>No execution.belief_state events found.</em></p>"
    latest = rows[-1]
    prev = rows[-2] if len(rows) > 1 else rows[-1]
    peak_blocked = max(float(r["blocked_level"]) for r in rows[-max(1, int(limit)) :])
    peak_force = max(float(r["force_level"]) for r in rows[-max(1, int(limit)) :])
    lines = [
        f"Refresh Pressure Trend (events={len(rows)})",
        "-" * 34,
        (
            f"latest blocked_level={float(latest['blocked_level']):.2f} "
            f"force_level={float(latest['force_level']):.2f}"
        ),
        (
            f"delta blocked={float(latest['blocked_level']) - float(prev['blocked_level']):+.2f} "
            f"force={float(latest['force_level']) - float(prev['force_level']):+.2f}"
        ),
        f"window peaks blocked={peak_blocked:.2f} force={peak_force:.2f}",
        (
            f"latest posture mode={str(latest['mode'] or 'n/a')} "
            f"stage={str(latest['stage'] or 'n/a')} allow_entries={bool(latest['allow_entries'])}"
        ),
    ]
    return "<h2>Refresh Pressure Trend</h2><pre>" + "\n".join(lines) + "</pre>"


def _replace_envelope_summary(events: list[dict], limit: int = 8) -> str:
    reason_counts: Counter[str] = Counter()
    symbol_counts: Counter[str] = Counter()
    total = 0
    for ev in events:
        if str(ev.get("event") or "") != "order.replace_envelope_block":
            continue
        data = ev.get("data") if isinstance(ev.get("data"), dict) else {}
        reason = str(data.get("reason") or "replace_envelope_block").strip().lower() or "replace_envelope_block"
        symbol = str(ev.get("symbol") or data.get("k") or data.get("symbol") or "UNKNOWN").strip().upper() or "UNKNOWN"
        reason_counts[reason] += 1
        symbol_counts[symbol] += 1
        total += 1
    if total <= 0:
        return "<h2>Replace Envelope Blocks</h2><p><em>No replace envelope block events found.</em></p>"
    lines = [f"Replace Envelope Blocks (events={total})", "-" * 38, "By reason:"]
    for reason, cnt in reason_counts.most_common(limit):
        lines.append(f"- {reason}: {cnt}")
    lines.append("")
    lines.append("Top symbols:")
    for symbol, cnt in symbol_counts.most_common(limit):
        lines.append(f"- {symbol}: {cnt}")
    return "<h2>Replace Envelope Blocks</h2><pre>" + "\n".join(lines) + "</pre>"


def _rebuild_orphan_summary(events: list[dict], limit: int = 8) -> str:
    by_action: Counter[str] = Counter()
    by_class: Counter[str] = Counter()
    by_symbol: Counter[str] = Counter()
    total = 0
    for ev in events:
        if str(ev.get("event") or "") != "rebuild.orphan_decision":
            continue
        data = ev.get("data") if isinstance(ev.get("data"), dict) else {}
        action = str(data.get("action") or "UNKNOWN").strip().upper() or "UNKNOWN"
        klass = str(data.get("class") or "unknown").strip().lower() or "unknown"
        symbol = str(data.get("symbol") or "UNKNOWN").strip().upper() or "UNKNOWN"
        by_action[action] += 1
        by_class[klass] += 1
        by_symbol[symbol] += 1
        total += 1
    if total <= 0:
        return "<h2>Rebuild Orphan Decisions</h2><p><em>No rebuild orphan decision events found.</em></p>"
    lines = [f"Rebuild Orphan Decisions (events={total})", "-" * 39, "By action:"]
    for action, cnt in by_action.most_common(limit):
        lines.append(f"- {action}: {cnt}")
    lines.append("")
    lines.append("By class:")
    for klass, cnt in by_class.most_common(limit):
        lines.append(f"- {klass}: {cnt}")
    lines.append("")
    lines.append("Top symbols:")
    for symbol, cnt in by_symbol.most_common(limit):
        lines.append(f"- {symbol}: {cnt}")
    return "<h2>Rebuild Orphan Decisions</h2><pre>" + "\n".join(lines) + "</pre>"


def _belief_state_summary(events: list[dict], limit: int = 8) -> str:
    rows: list[dict] = []
    for ev in events:
        if str(ev.get("event") or "") != "execution.belief_state":
            continue
        data = ev.get("data") if isinstance(ev.get("data"), dict) else {}
        rows.append(
            {
                "debt_sec": float(data.get("belief_debt_sec") or 0.0),
                "debt_symbols": int(data.get("belief_debt_symbols") or 0),
                "confidence": float(data.get("belief_confidence") or 0.0),
                "evidence_coverage_ratio": float(data.get("evidence_coverage_ratio") or 0.0),
                "evidence_ws_coverage_ratio": float(data.get("evidence_ws_coverage_ratio") or 0.0),
                "evidence_rest_coverage_ratio": float(data.get("evidence_rest_coverage_ratio") or 0.0),
                "evidence_fill_coverage_ratio": float(data.get("evidence_fill_coverage_ratio") or 0.0),
                "belief_envelope_symbols": int(data.get("belief_envelope_symbols") or 0),
                "belief_envelope_ambiguous_symbols": int(data.get("belief_envelope_ambiguous_symbols") or 0),
                "belief_position_interval_width_sum": float(data.get("belief_position_interval_width_sum") or 0.0),
                "belief_position_interval_width_max": float(data.get("belief_position_interval_width_max") or 0.0),
                "belief_live_unknown_symbols": int(data.get("belief_live_unknown_symbols") or 0),
                "belief_envelope_worst_symbol": str(data.get("belief_envelope_worst_symbol") or ""),
                "mismatch_streak": int(data.get("mismatch_streak") or 0),
                "repair_actions": int(data.get("repair_actions") or 0),
                "repair_skipped": int(data.get("repair_skipped") or 0),
                "guard_mode": str(data.get("guard_mode") or ""),
                "allow_entries": bool(data.get("allow_entries", True)),
                "guard_recovery_stage": str(data.get("guard_recovery_stage") or ""),
                "runtime_gate_cause_summary": str(data.get("runtime_gate_cause_summary") or ""),
                "guard_unlock_conditions": str(data.get("guard_unlock_conditions") or ""),
                "guard_next_unlock_sec": float(data.get("guard_next_unlock_sec") or 0.0),
                "guard_cause_tags": str(data.get("guard_cause_tags") or ""),
                "guard_dominant_contributors": str(data.get("guard_dominant_contributors") or ""),
                "guard_unlock_snapshot": (
                    dict(data.get("guard_unlock_snapshot") or {})
                    if isinstance(data.get("guard_unlock_snapshot"), dict)
                    else {}
                ),
            }
        )
    if not rows:
        return "<h2>Execution Belief State</h2><p><em>No execution.belief_state events found.</em></p>"
    latest = rows[-1]
    avg_conf = sum(r["confidence"] for r in rows) / float(len(rows))
    max_debt = max(r["debt_sec"] for r in rows)
    max_syms = max(r["debt_symbols"] for r in rows)
    lines = [
        f"Execution Belief State (events={len(rows)})",
        "-" * 36,
        (
            f"latest debt={latest['debt_sec']:.1f}s symbols={latest['debt_symbols']} "
            f"confidence={latest['confidence']:.2f} streak={latest['mismatch_streak']}"
        ),
        (
            f"latest evidence_coverage={latest['evidence_coverage_ratio']:.3f} "
            f"(ws={latest['evidence_ws_coverage_ratio']:.3f} "
            f"rest={latest['evidence_rest_coverage_ratio']:.3f} "
            f"fill={latest['evidence_fill_coverage_ratio']:.3f})"
        ),
        (
            f"latest envelope symbols={latest['belief_envelope_symbols']} "
            f"ambiguous={latest['belief_envelope_ambiguous_symbols']} "
            f"width_sum={latest['belief_position_interval_width_sum']:.3f} "
            f"width_max={latest['belief_position_interval_width_max']:.3f} "
            f"unknown={latest['belief_live_unknown_symbols']} "
            f"worst={latest['belief_envelope_worst_symbol'] or 'n/a'}"
        ),
        (
            f"latest guard_mode={latest['guard_mode'] or 'n/a'} "
            f"allow_entries={bool(latest['allow_entries'])}"
        ),
        (
            f"latest recovery_stage={latest['guard_recovery_stage'] or 'n/a'} "
            f"next_unlock_sec={float(latest['guard_next_unlock_sec']):.1f}"
        ),
        f"latest gate_cause_summary={latest['runtime_gate_cause_summary'] or 'stable'}",
        f"latest cause_tags={latest['guard_cause_tags'] or 'n/a'}",
        f"latest dominant_contributors={latest['guard_dominant_contributors'] or 'n/a'}",
        f"latest top_contributors={latest['guard_dominant_contributors'] or 'n/a'}",
        f"latest unlock_conditions={latest['guard_unlock_conditions'] or 'stable'}",
        (
            f"latest repair_actions={latest['repair_actions']} "
            f"repair_skipped={latest['repair_skipped']}"
        ),
        "",
        f"avg confidence: {avg_conf:.2f}",
        f"max debt sec: {max_debt:.1f}",
        f"max debt symbols: {max_syms}",
    ]
    snap = latest.get("guard_unlock_snapshot")
    if isinstance(snap, dict) and snap:
        ht_cur = int(float(snap.get("healthy_ticks_current") or 0.0))
        ht_req = int(float(snap.get("healthy_ticks_required") or 0.0))
        cov_cur = float(snap.get("journal_coverage_current") or 0.0)
        cov_req = float(snap.get("journal_coverage_required") or 0.0)
        cc_cur = float(snap.get("contradiction_clear_current_sec") or 0.0)
        cc_req = float(snap.get("contradiction_clear_required_sec") or 0.0)
        pg_cur = float(snap.get("protection_gap_current_sec") or 0.0)
        pg_req = float(snap.get("protection_gap_max_sec") or 0.0)
        lines.extend(
            [
                "",
                "unlock snapshot:",
                (
                    f"healthy_ticks={ht_cur}/{ht_req} "
                    f"journal_coverage={cov_cur:.3f}/{cov_req:.3f}"
                ),
                (
                    f"contradiction_clear={cc_cur:.0f}s/{cc_req:.0f}s "
                    f"protection_gap={pg_cur:.1f}s/{pg_req:.1f}s"
                ),
                (
                    "unlock_remaining "
                    f"healthy_ticks={max(0, ht_req - ht_cur)} "
                    f"journal_coverage={max(0.0, cov_req - cov_cur):.3f} "
                    f"contradiction_clear={max(0.0, cc_req - cc_cur):.0f}s "
                    f"protection_gap={max(0.0, pg_cur - pg_req):.1f}s"
                ),
            ]
        )
    return "<h2>Execution Belief State</h2><pre>" + "\n".join(lines) + "</pre>"


def _correlation_contribution_summary(events: list[dict], limit: int = 8) -> str:
    by_reason: Counter[str] = Counter()
    by_symbol: Counter[str] = Counter()
    blocked = 0
    scaled = 0
    stress = 0
    tightening = 0
    latest_regime = ""
    latest_pressure = 0.0
    latest_tags = ""
    for ev in events:
        name = str(ev.get("event") or "")
        data = ev.get("data") if isinstance(ev.get("data"), dict) else {}
        if name in ("entry.decision", "entry.blocked"):
            pressure = float(data.get("corr_pressure") or 0.0)
            regime = str(data.get("corr_regime") or "").strip().upper()
            if pressure <= 0.0 and not regime:
                continue
            symbol = str(ev.get("symbol") or data.get("symbol") or "UNKNOWN").strip().upper() or "UNKNOWN"
            reason = str(data.get("corr_reason_tags") or "stable").strip().lower() or "stable"
            by_symbol[symbol] += 1
            by_reason[reason] += 1
            if name == "entry.blocked":
                blocked += 1
            else:
                action = str(data.get("action") or "").strip().upper()
                if action == "SCALE":
                    scaled += 1
                elif action in ("DENY", "DEFER"):
                    blocked += 1
            if regime == "STRESS":
                stress += 1
            elif regime == "TIGHTENING":
                tightening += 1
        elif name == "execution.correlation_state":
            latest_regime = str(data.get("corr_regime") or latest_regime)
            latest_pressure = float(data.get("corr_pressure") or latest_pressure)
            latest_tags = str(data.get("corr_reason_tags") or latest_tags)
    total = blocked + scaled
    if total <= 0 and not latest_regime:
        return "<h2>Correlation Contribution</h2><p><em>No correlation contribution events found.</em></p>"
    lines = [
        f"Correlation Contribution (events={total})",
        "-" * 36,
        f"entry impact blocked={blocked} scaled={scaled}",
        f"regime pressure stress={stress} tightening={tightening}",
    ]
    if latest_regime:
        lines.append(f"latest regime={latest_regime} pressure={latest_pressure:.2f} tags={latest_tags}")
    if by_reason:
        lines.append("")
        lines.append("Top reason tags:")
        for reason, cnt in by_reason.most_common(limit):
            lines.append(f"- {reason}: {cnt}")
    if by_symbol:
        lines.append("")
        lines.append("Top symbols:")
        for symbol, cnt in by_symbol.most_common(limit):
            lines.append(f"- {symbol}: {cnt}")
    return "<h2>Correlation Contribution</h2><pre>" + "\n".join(lines) + "</pre>"


def _correlation_symbol_table(events: list[dict], limit: int = 8) -> str:
    by_symbol: dict[str, dict[str, object]] = {}
    for ev in events:
        name = str(ev.get("event") or "")
        if name not in ("entry.decision", "entry.blocked"):
            continue
        data = ev.get("data") if isinstance(ev.get("data"), dict) else {}
        pressure = float(data.get("corr_pressure") or 0.0)
        regime = str(data.get("corr_regime") or "").strip().upper()
        if pressure <= 0.0 and not regime:
            continue
        symbol = str(ev.get("symbol") or data.get("symbol") or "UNKNOWN").strip().upper() or "UNKNOWN"
        reason = str(data.get("corr_reason_tags") or "stable").strip().lower() or "stable"
        row = by_symbol.get(symbol)
        if row is None:
            row = {
                "blocked": 0,
                "scaled": 0,
                "reason_counts": Counter(),
                "reason_scores": Counter(),
                "max_pressure": 0.0,
                "stress_hits": 0,
                "tightening_hits": 0,
            }
            by_symbol[symbol] = row
        if name == "entry.blocked":
            row["blocked"] = int(row.get("blocked", 0) or 0) + 1
        else:
            action = str(data.get("action") or "").strip().upper()
            if action == "SCALE":
                row["scaled"] = int(row.get("scaled", 0) or 0) + 1
            elif action in ("DENY", "DEFER"):
                row["blocked"] = int(row.get("blocked", 0) or 0) + 1
        reasons = row.get("reason_counts")
        if isinstance(reasons, Counter):
            reasons[reason] += 1
        reason_scores = row.get("reason_scores")
        if isinstance(reason_scores, Counter):
            weight = 2 if name == "entry.blocked" else 1
            reason_scores[reason] += int(weight)
        p = float(pressure)
        if p > float(row.get("max_pressure", 0.0) or 0.0):
            row["max_pressure"] = p
        if regime == "STRESS":
            row["stress_hits"] = int(row.get("stress_hits", 0) or 0) + 1
        elif regime == "TIGHTENING":
            row["tightening_hits"] = int(row.get("tightening_hits", 0) or 0) + 1
    if not by_symbol:
        return "<h2>Correlation Impact By Symbol</h2><p><em>No symbol-level correlation impact found.</em></p>"
    sorted_rows = sorted(
        by_symbol.items(),
        key=lambda kv: (
            int(kv[1].get("blocked", 0) or 0) + int(kv[1].get("scaled", 0) or 0),
            float(kv[1].get("max_pressure", 0.0) or 0.0),
        ),
        reverse=True,
    )[: max(1, int(limit))]
    lines = [
        "Correlation Impact By Symbol",
        "-" * 38,
        f"{'SYMBOL':10} | {'BLK':>3} | {'SCL':>3} | {'MAX_P':>5} | {'REG':>8} | TOP_TAG",
        "-" * 78,
    ]
    for symbol, row in sorted_rows:
        blocked = int(row.get("blocked", 0) or 0)
        scaled = int(row.get("scaled", 0) or 0)
        max_p = float(row.get("max_pressure", 0.0) or 0.0)
        stress_hits = int(row.get("stress_hits", 0) or 0)
        tightening_hits = int(row.get("tightening_hits", 0) or 0)
        if stress_hits > 0:
            regime_label = "STRESS"
        elif tightening_hits > 0:
            regime_label = "TIGHTEN"
        else:
            regime_label = "NORMAL"
        reasons = row.get("reason_counts")
        top_tag = "stable"
        reason_scores = row.get("reason_scores")
        if isinstance(reason_scores, Counter) and reason_scores:
            top_tag = str(reason_scores.most_common(1)[0][0] or "stable")
        elif isinstance(reasons, Counter) and reasons:
            top_tag = str(reasons.most_common(1)[0][0] or "stable")
        lines.append(
            f"{symbol:10} | {blocked:>3} | {scaled:>3} | {max_p:>5.2f} | {regime_label:>8} | {top_tag}"
        )
    return "<h2>Correlation Impact By Symbol</h2><pre>" + "\n".join(lines) + "</pre>"


def _recovery_stage_summary(events: list[dict], limit: int = 8) -> str:
    stage_counts: Counter[str] = Counter()
    symbol_counts: Counter[str] = Counter()
    total = 0
    for ev in events:
        if str(ev.get("event") or "") != "execution.belief_state":
            continue
        data = ev.get("data") if isinstance(ev.get("data"), dict) else {}
        stage = str(data.get("guard_recovery_stage") or "UNKNOWN").strip().upper() or "UNKNOWN"
        stage_counts[stage] += 1
        total += 1
        worst = data.get("worst_symbols")
        if isinstance(worst, list):
            for item in worst:
                sym = ""
                if isinstance(item, (list, tuple)) and item:
                    sym = str(item[0] or "").strip().upper()
                elif isinstance(item, dict):
                    sym = str(item.get("symbol") or item.get("k") or "").strip().upper()
                else:
                    sym = str(item or "").strip().upper()
                if sym:
                    symbol_counts[sym] += 1
    if total <= 0:
        return "<h2>Recovery Stage Timeline</h2><p><em>No execution.belief_state events found.</em></p>"
    lines = [f"Recovery Stage Timeline (events={total})", "-" * 38, "By recovery stage:"]
    for stage, cnt in stage_counts.most_common(limit):
        lines.append(f"- {stage}: {cnt}")
    if symbol_counts:
        lines.append("")
        lines.append("Top symbols in staged recovery windows:")
        for sym, cnt in symbol_counts.most_common(limit):
            lines.append(f"- {sym}: {cnt}")
    return "<h2>Recovery Stage Timeline</h2><pre>" + "\n".join(lines) + "</pre>"


def _severity_debt_trend_summary(events: list[dict], severity_threshold: float = 0.85) -> str:
    ts_rows: list[tuple[float, float]] = []
    for ev in events:
        if str(ev.get("event") or "") not in ("entry.reconcile_first_gate", "execution.belief_state"):
            continue
        data = ev.get("data") if isinstance(ev.get("data"), dict) else {}
        ts = float(ev.get("ts") or 0.0)
        if ts <= 0:
            continue
        sev = float(data.get("reconcile_first_severity") or data.get("runtime_gate_degrade_score") or 0.0)
        if sev <= 0:
            continue
        ts_rows.append((ts, sev))
    if not ts_rows:
        return "<h2>Severity Debt Trend</h2><p><em>No severity telemetry found.</em></p>"
    ts_rows.sort(key=lambda x: x[0])
    now = ts_rows[-1][0]

    def _window(hours: float) -> list[float]:
        lo = now - (hours * 3600.0)
        return [sev for ts, sev in ts_rows if ts >= lo]

    def _stats(vals: list[float]) -> tuple[int, float, float, int]:
        if not vals:
            return 0, 0.0, 0.0, 0
        cnt = len(vals)
        avg = sum(vals) / float(cnt)
        mx = max(vals)
        breaches = sum(1 for v in vals if v >= float(severity_threshold))
        return cnt, avg, mx, breaches

    vals_1h = _window(1.0)
    vals_6h = _window(6.0)
    c1, a1, m1, b1 = _stats(vals_1h)
    c6, a6, m6, b6 = _stats(vals_6h)
    lines = [
        f"Severity Debt Trend (threshold={severity_threshold:.2f})",
        "-" * 40,
        f"1h: count={c1} avg={a1:.2f} max={m1:.2f} breaches={b1}",
        f"6h: count={c6} avg={a6:.2f} max={m6:.2f} breaches={b6}",
    ]
    if c6 > 0 and c1 > 0:
        lines.append(f"momentum(1h-6h avg): {(a1 - a6):+.2f}")
    return "<h2>Severity Debt Trend</h2><pre>" + "\n".join(lines) + "</pre>"


def _pick_latest_correlation_id(events: list[dict]) -> str:
    for ev in reversed(events):
        data = ev.get("data") if isinstance(ev.get("data"), dict) else {}
        corr = str(data.get("correlation_id") or ev.get("correlation_id") or "").strip()
        if corr:
            return corr
    return ""


def _pick_latest_symbol(events: list[dict]) -> str:
    for ev in reversed(events):
        data = ev.get("data") if isinstance(ev.get("data"), dict) else {}
        sym = str(ev.get("symbol") or data.get("symbol") or data.get("k") or "").strip().upper()
        if sym:
            return sym
    return ""


def _journal_replay_summary(events: list[dict], journal_path: Path, replay_limit: int = 8) -> str:
    if _replay_trade is None:
        return "<h2>Execution Journal Replay</h2><p><em>Replay helper unavailable.</em></p>"
    if not journal_path.exists():
        return f"<h2>Execution Journal Replay</h2><p><em>Missing {journal_path.name}</em></p>"
    corr = _pick_latest_correlation_id(events)
    sym = _pick_latest_symbol(events)
    try:
        out = _replay_trade.replay(journal_path, correlation_id=corr, symbol=("" if corr else sym))
    except Exception:
        return "<h2>Execution Journal Replay</h2><p><em>Replay failed.</em></p>"
    transitions = out.get("transitions", []) if isinstance(out, dict) else []
    if not transitions:
        return "<h2>Execution Journal Replay</h2><p><em>No matching transitions found.</em></p>"
    lines = [
        f"Execution Journal Replay (events={int(out.get('count', 0))})",
        "-" * 40,
        f"filter correlation_id={corr or 'n/a'} symbol={sym or 'n/a'}",
        f"last_state={str(out.get('last_state') or '')}",
        "",
        "latest transitions:",
    ]
    for tr in transitions[-max(1, int(replay_limit)):]:
        lines.append(
            f"- {tr.get('machine')} {tr.get('entity')} "
            f"{tr.get('from')}->{tr.get('to')} ({tr.get('reason')})"
        )
    return "<h2>Execution Journal Replay</h2><pre>" + "\n".join(lines) + "</pre>"


def _exit_quality_delta(path: Path) -> str:
    if not path.exists():
        return "<h2>Exit Quality Deltas</h2><p><em>Missing exit_quality_summary.json</em></p>"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return "<h2>Exit Quality Deltas</h2><p><em>Invalid exit_quality_summary.json</em></p>"
    w24 = payload.get("window_24h", {}) if isinstance(payload, dict) else {}
    w7 = payload.get("window_7d", {}) if isinstance(payload, dict) else {}
    win24 = float(w24.get("win_rate") or 0.0)
    win7 = float(w7.get("win_rate") or 0.0)
    pnl24 = float(w24.get("avg_pnl") or 0.0)
    pnl7 = float(w7.get("avg_pnl") or 0.0)
    win_delta = win24 - win7
    pnl_delta = pnl24 - pnl7
    lines = [
        "Exit Quality Deltas (24h vs 7d)",
        "-" * 34,
        f"win_rate: {win24:.1%} vs {win7:.1%} (delta {win_delta:+.1%})",
        f"avg_pnl: {pnl24:.4f} vs {pnl7:.4f} (delta {pnl_delta:+.4f})",
    ]
    return "<h2>Exit Quality Deltas</h2><pre>" + "\n".join(lines) + "</pre>"


def _corr_vs_exit_quality_summary(events: list[dict], limit: int = 8) -> str:
    corr_rows: list[tuple[float, str, float]] = []
    closes: list[dict[str, float | str]] = []
    for ev in events:
        name = str(ev.get("event") or "")
        data = ev.get("data") if isinstance(ev.get("data"), dict) else {}
        ts = float(ev.get("ts") or 0.0)
        if name == "execution.correlation_state":
            regime = str(data.get("corr_regime") or "NORMAL").strip().upper() or "NORMAL"
            pressure = float(data.get("corr_pressure") or 0.0)
            corr_rows.append((ts, regime, pressure))
        elif name == "position.closed":
            closes.append(
                {
                    "ts": ts,
                    "symbol": str(ev.get("symbol") or data.get("symbol") or "UNKNOWN").strip().upper() or "UNKNOWN",
                    "pnl": float(data.get("pnl_usdt") or 0.0),
                    "duration": float(data.get("duration_sec") or 0.0),
                }
            )
    if not closes:
        return "<h2>Corr vs Exit Quality</h2><p><em>No position.closed events found.</em></p>"
    corr_rows.sort(key=lambda x: x[0])
    by_regime: dict[str, dict[str, float]] = {}
    for row in closes:
        ts = float(row.get("ts") or 0.0)
        regime = "UNKNOWN"
        pressure = 0.0
        for cts, cregime, cpressure in corr_rows:
            if cts <= ts:
                regime = str(cregime)
                pressure = float(cpressure)
            else:
                break
        agg = by_regime.get(regime)
        if agg is None:
            agg = {"count": 0.0, "wins": 0.0, "pnl_sum": 0.0, "dur_sum": 0.0, "pressure_sum": 0.0}
            by_regime[regime] = agg
        pnl = float(row.get("pnl") or 0.0)
        agg["count"] += 1.0
        if pnl > 0:
            agg["wins"] += 1.0
        agg["pnl_sum"] += pnl
        agg["dur_sum"] += float(row.get("duration") or 0.0)
        agg["pressure_sum"] += pressure
    ranked = sorted(by_regime.items(), key=lambda kv: kv[1].get("count", 0.0), reverse=True)[: max(1, int(limit))]
    lines = [
        "Corr vs Exit Quality",
        "-" * 32,
        f"{'REGIME':10} | {'N':>3} | {'WIN%':>6} | {'AVG_PNL':>9} | {'AVG_DUR':>8} | {'AVG_CORR':>8}",
        "-" * 78,
    ]
    for regime, agg in ranked:
        n = max(1.0, float(agg.get("count", 0.0)))
        win_rate = float(agg.get("wins", 0.0)) / n
        avg_pnl = float(agg.get("pnl_sum", 0.0)) / n
        avg_dur = float(agg.get("dur_sum", 0.0)) / n
        avg_corr = float(agg.get("pressure_sum", 0.0)) / n
        lines.append(
            f"{regime:10} | {int(n):>3} | {win_rate:>6.1%} | {avg_pnl:>+9.3f} | {avg_dur:>8.0f}s | {avg_corr:>8.2f}"
        )
    return "<h2>Corr vs Exit Quality</h2><pre>" + "\n".join(lines) + "</pre>"


def _notify_state_summary(path: Path) -> str:
    if not path.exists():
        return "<h2>Notifier State</h2><p><em>Missing telemetry_dashboard_notify_state.json</em></p>"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return "<h2>Notifier State</h2><p><em>Invalid telemetry_dashboard_notify_state.json</em></p>"
    if not isinstance(payload, dict):
        return "<h2>Notifier State</h2><p><em>Invalid notify state payload.</em></p>"

    level = str(payload.get("level") or "unknown")
    prev_level = str(payload.get("previous_level") or "n/a")
    reason = str(payload.get("last_decision_reason") or "n/a")
    sent = bool(payload.get("last_decision_sent"))
    mismatch = int(float(payload.get("reliability_mismatch") or 0.0))
    prev_mismatch = int(float(payload.get("previous_reliability_mismatch") or 0.0))
    coverage = float(payload.get("reliability_coverage") or 0.0)
    prev_coverage = float(payload.get("previous_reliability_coverage") or 0.0)
    cat_ledger = int(float(payload.get("reliability_cat_ledger") or 0.0))
    prev_cat_ledger = int(float(payload.get("previous_reliability_cat_ledger") or 0.0))
    cat_transition = int(float(payload.get("reliability_cat_transition") or 0.0))
    prev_cat_transition = int(float(payload.get("previous_reliability_cat_transition") or 0.0))
    cat_belief = int(float(payload.get("reliability_cat_belief") or 0.0))
    prev_cat_belief = int(float(payload.get("previous_reliability_cat_belief") or 0.0))
    cat_position = int(float(payload.get("reliability_cat_position") or 0.0))
    prev_cat_position = int(float(payload.get("previous_reliability_cat_position") or 0.0))
    cat_orphan = int(float(payload.get("reliability_cat_orphan") or 0.0))
    prev_cat_orphan = int(float(payload.get("previous_reliability_cat_orphan") or 0.0))
    intent_collision = int(float(payload.get("reliability_intent_collision_count") or 0.0))
    prev_intent_collision = int(float(payload.get("previous_reliability_intent_collision_count") or 0.0))
    cat_cov_gap = int(float(payload.get("reliability_cat_coverage_gap") or 0.0))
    prev_cat_cov_gap = int(float(payload.get("previous_reliability_cat_coverage_gap") or 0.0))
    cat_replace = int(float(payload.get("reliability_cat_replace_race") or 0.0))
    prev_cat_replace = int(float(payload.get("previous_reliability_cat_replace_race") or 0.0))
    cat_contrad = int(float(payload.get("reliability_cat_contradiction") or 0.0))
    prev_cat_contrad = int(float(payload.get("previous_reliability_cat_contradiction") or 0.0))
    cat_unknown = int(float(payload.get("reliability_cat_unknown") or 0.0))
    prev_cat_unknown = int(float(payload.get("previous_reliability_cat_unknown") or 0.0))
    unlock_next = float(payload.get("unlock_next_sec") or 0.0)
    unlock_ht_rem = int(float(payload.get("unlock_healthy_ticks_remaining") or 0.0))
    unlock_cov_rem = float(payload.get("unlock_journal_coverage_remaining") or 0.0)
    unlock_contrad_rem = float(payload.get("unlock_contradiction_clear_remaining_sec") or 0.0)
    unlock_gap_rem = float(payload.get("unlock_protection_gap_remaining_sec") or 0.0)
    sev = float(payload.get("reconcile_gate_max_severity") or 0.0)
    prev_sev = float(payload.get("previous_reconcile_gate_max_severity") or 0.0)
    streak = int(float(payload.get("reconcile_gate_max_streak") or 0.0))
    prev_streak = int(float(payload.get("previous_reconcile_gate_max_streak") or 0.0))
    lines = [
        "Notifier State",
        "-" * 24,
        f"level: {level} (prev {prev_level})",
        f"last_decision: {'send' if sent else 'skip'} ({reason})",
        f"replay_mismatch: {mismatch} (prev {prev_mismatch})",
        f"journal_coverage: {coverage:.3f} (prev {prev_coverage:.3f})",
        (
            "mismatch_categories: "
            f"ledger={cat_ledger} (prev {prev_cat_ledger}) "
            f"transition={cat_transition} (prev {prev_cat_transition}) "
            f"belief={cat_belief} (prev {prev_cat_belief}) "
            f"position={cat_position} (prev {prev_cat_position}) "
            f"orphan={cat_orphan} (prev {prev_cat_orphan}) "
            f"intent_collision={intent_collision} (prev {prev_intent_collision}) "
            f"coverage_gap={cat_cov_gap} (prev {prev_cat_cov_gap}) "
            f"replace_race={cat_replace} (prev {prev_cat_replace}) "
            f"contradiction={cat_contrad} (prev {prev_cat_contrad}) "
            f"unknown={cat_unknown} (prev {prev_cat_unknown})"
        ),
        f"reconcile_max_severity: {sev:.2f} (prev {prev_sev:.2f})",
        f"reconcile_max_streak: {streak} (prev {prev_streak})",
        (
            "unlock_remaining: "
            f"next={unlock_next:.1f}s "
            f"healthy_ticks={unlock_ht_rem} "
            f"journal_coverage={unlock_cov_rem:.3f} "
            f"contradiction_clear={unlock_contrad_rem:.0f}s "
            f"protection_gap={unlock_gap_rem:.1f}s"
        ),
    ]
    return "<h2>Notifier State</h2><pre>" + "\n".join(lines) + "</pre>"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Telemetry dashboard HTML page")
    parser.add_argument("--core", default="logs/core_health.txt", help="Core health text")
    parser.add_argument("--signal", default="logs/signal_data_health.txt", help="Signal data health text")
    parser.add_argument("--anomaly", default="logs/telemetry_anomaly.txt", help="Anomaly report text")
    parser.add_argument("--signal-exit", default="logs/signal_exit_notify.txt", help="Signal/exit notify text")
    parser.add_argument("--guard-timeline", default="logs/telemetry_guard_timeline.txt", help="Guard timeline text")
    parser.add_argument("--guard-history", default="logs/telemetry_guard_history.csv", help="Guard history CSV")
    parser.add_argument("--telemetry", default="logs/telemetry.jsonl", help="Telemetry JSONL")
    parser.add_argument(
        "--guard-actions",
        default="logs/telemetry_guard_history_actions.json",
        help="Guard history actions JSON",
    )
    parser.add_argument(
        "--signal-feedback",
        default="logs/signal_exit_feedback.json",
        help="Signal exit feedback JSON",
    )
    parser.add_argument("--exit-quality", default="logs/exit_quality.txt", help="Exit quality dashboard text")
    parser.add_argument("--exit-quality-json", default="logs/exit_quality_summary.json", help="Exit quality JSON")
    parser.add_argument("--corr-group", default="logs/telemetry_corr_group.txt", help="Correlation guard dashboard text")
    parser.add_argument("--reliability-gate", default="logs/reliability_gate.txt", help="Reliability gate summary text")
    parser.add_argument(
        "--notify-state",
        default="logs/telemetry_dashboard_notify_state.json",
        help="Telemetry dashboard notify state JSON",
    )
    parser.add_argument("--journal", default="logs/execution_journal.jsonl", help="Execution journal JSONL")
    parser.add_argument("--replay-limit", type=int, default=8, help="Transition lines to show in replay panel")
    parser.add_argument(
        "--corr-symbol-limit",
        type=int,
        default=8,
        help="Rows to show in correlation contribution/symbol sections",
    )
    parser.add_argument(
        "--reconcile-first-gate-severity-threshold",
        type=float,
        default=float(os.getenv("RECONCILE_FIRST_GATE_SEVERITY_THRESHOLD", "0.85")),
        help="Severity threshold used for reconcile-first streak rendering",
    )
    parser.add_argument("--output", default="logs/telemetry_dashboard_page.html")
    args = parser.parse_args(argv)

    telemetry_events = _load_jsonl(Path(args.telemetry))

    sections = [
        _read_section(Path(args.core), "Core Health"),
        _read_section(Path(args.signal), "Signal Data Health"),
        _read_section(Path(args.anomaly), "Anomaly Detector"),
        _read_section(Path(args.signal_exit), "Signal/Exit Notify"),
        _read_section(Path(args.guard_timeline), "Guard Timeline"),
        _read_csv_section(Path(args.guard_history), "Guard History (Recent)"),
        _guard_reason_summary(telemetry_events),
        _guard_symbol_spark(telemetry_events),
        _reconcile_first_gate_summary(
            telemetry_events,
            severity_threshold=max(0.0, float(args.reconcile_first_gate_severity_threshold)),
        ),
        _entry_budget_summary(telemetry_events),
        _protection_refresh_budget_summary(telemetry_events),
        _refresh_pressure_trend_summary(telemetry_events),
        _replace_envelope_summary(telemetry_events),
        _rebuild_orphan_summary(telemetry_events),
        _partial_fill_state_summary(telemetry_events),
        _belief_state_summary(telemetry_events),
        _correlation_contribution_summary(telemetry_events, limit=max(1, int(args.corr_symbol_limit))),
        _correlation_symbol_table(telemetry_events, limit=max(1, int(args.corr_symbol_limit))),
        _recovery_stage_summary(telemetry_events),
        _severity_debt_trend_summary(
            telemetry_events,
            severity_threshold=max(0.0, float(args.reconcile_first_gate_severity_threshold)),
        ),
        _journal_replay_summary(telemetry_events, Path(args.journal), max(1, args.replay_limit)),
        _read_reliability_gate_section(Path(args.reliability_gate)),
        _notify_state_summary(Path(args.notify_state)),
        _exit_quality_delta(Path(args.exit_quality_json)),
        _corr_vs_exit_quality_summary(telemetry_events),
        _read_json_section(Path(args.guard_actions), "Guard History Actions"),
        _read_json_section(Path(args.signal_feedback), "Signal Exit Feedback"),
        _read_section(Path(args.exit_quality), "Exit Quality"),
        _read_section(Path(args.corr_group), "Correlation Guard"),
    ]
    html = f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <title>Telemetry Dashboard</title>
    <style>
      body {{ font-family: system-ui, sans-serif; background:#101218; color:#f7f7f7; padding:2rem; }}
      pre {{ background:#181c2a; padding:1rem; border-radius:0.5rem; overflow:auto; }}
      h1 {{ margin-bottom:1rem; }}
      .top-strip {{ display:grid; grid-template-columns:repeat(10,minmax(0,1fr)); gap:.5rem; margin-bottom:1rem; }}
      .metric {{ background:#181c2a; padding:.5rem; border-radius:.4rem; }}
      .metric.ok {{ border:1px solid #2f6f3f; }}
      .metric.warn {{ border:1px solid #8d7a2f; }}
      .metric.crit {{ border:1px solid #8d2f2f; }}
      .metric-legend {{ margin-bottom:1rem; color:#cfd6ea; font-size:.9rem; }}
    </style>
  </head>
  <body>
    <h1>Telemetry Dashboard Snapshot</h1>
    {_top_reliability_strip(telemetry_events)}
    {"".join(sections)}
  </body>
</html>"""
    Path(args.output).write_text(html, encoding="utf-8")
    print(f"Telemetry dashboard page saved to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
