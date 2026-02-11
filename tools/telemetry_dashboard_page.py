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
    if mismatch_ids:
        lines.append("top_missing_ids: " + ", ".join(mismatch_ids[:5]))
    return f"<h2>Reliability Gate</h2><pre>{'\n'.join(lines)}</pre>"


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
    return f"<h2>Top Guard Reasons</h2><pre>{'\n'.join(lines)}</pre>"


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
    return f"<h2>Guard Hits By Symbol</h2><pre>{'\n'.join(lines)}</pre>"


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
    return f"<h2>Partial Fill Outcomes</h2><pre>{'\n'.join(lines)}</pre>"


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
    return f"<h2>Reconcile-First Gate</h2><pre>{'\n'.join(lines)}</pre>"


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
    return f"<h2>Entry Budget Pressure</h2><pre>{'\n'.join(lines)}</pre>"


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
    return f"<h2>Replace Envelope Blocks</h2><pre>{'\n'.join(lines)}</pre>"


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
    return f"<h2>Rebuild Orphan Decisions</h2><pre>{'\n'.join(lines)}</pre>"


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
                "mismatch_streak": int(data.get("mismatch_streak") or 0),
                "repair_actions": int(data.get("repair_actions") or 0),
                "repair_skipped": int(data.get("repair_skipped") or 0),
                "guard_mode": str(data.get("guard_mode") or ""),
                "allow_entries": bool(data.get("allow_entries", True)),
                "guard_recovery_stage": str(data.get("guard_recovery_stage") or ""),
                "guard_unlock_conditions": str(data.get("guard_unlock_conditions") or ""),
                "guard_next_unlock_sec": float(data.get("guard_next_unlock_sec") or 0.0),
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
            f"latest guard_mode={latest['guard_mode'] or 'n/a'} "
            f"allow_entries={bool(latest['allow_entries'])}"
        ),
        (
            f"latest recovery_stage={latest['guard_recovery_stage'] or 'n/a'} "
            f"next_unlock_sec={float(latest['guard_next_unlock_sec']):.1f}"
        ),
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
    return f"<h2>Execution Belief State</h2><pre>{'\n'.join(lines)}</pre>"


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
    return f"<h2>Recovery Stage Timeline</h2><pre>{'\n'.join(lines)}</pre>"


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
    return f"<h2>Severity Debt Trend</h2><pre>{'\n'.join(lines)}</pre>"


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
    return f"<h2>Execution Journal Replay</h2><pre>{'\n'.join(lines)}</pre>"


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
    return f"<h2>Exit Quality Deltas</h2><pre>{'\n'.join(lines)}</pre>"


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
    cat_cov_gap = int(float(payload.get("reliability_cat_coverage_gap") or 0.0))
    prev_cat_cov_gap = int(float(payload.get("previous_reliability_cat_coverage_gap") or 0.0))
    cat_replace = int(float(payload.get("reliability_cat_replace_race") or 0.0))
    prev_cat_replace = int(float(payload.get("previous_reliability_cat_replace_race") or 0.0))
    cat_contrad = int(float(payload.get("reliability_cat_contradiction") or 0.0))
    prev_cat_contrad = int(float(payload.get("previous_reliability_cat_contradiction") or 0.0))
    cat_unknown = int(float(payload.get("reliability_cat_unknown") or 0.0))
    prev_cat_unknown = int(float(payload.get("previous_reliability_cat_unknown") or 0.0))
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
            f"coverage_gap={cat_cov_gap} (prev {prev_cat_cov_gap}) "
            f"replace_race={cat_replace} (prev {prev_cat_replace}) "
            f"contradiction={cat_contrad} (prev {prev_cat_contrad}) "
            f"unknown={cat_unknown} (prev {prev_cat_unknown})"
        ),
        f"reconcile_max_severity: {sev:.2f} (prev {prev_sev:.2f})",
        f"reconcile_max_streak: {streak} (prev {prev_streak})",
    ]
    return f"<h2>Notifier State</h2><pre>{'\n'.join(lines)}</pre>"


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
        _replace_envelope_summary(telemetry_events),
        _rebuild_orphan_summary(telemetry_events),
        _partial_fill_state_summary(telemetry_events),
        _belief_state_summary(telemetry_events),
        _recovery_stage_summary(telemetry_events),
        _severity_debt_trend_summary(
            telemetry_events,
            severity_threshold=max(0.0, float(args.reconcile_first_gate_severity_threshold)),
        ),
        _journal_replay_summary(telemetry_events, Path(args.journal), max(1, args.replay_limit)),
        _read_reliability_gate_section(Path(args.reliability_gate)),
        _notify_state_summary(Path(args.notify_state)),
        _exit_quality_delta(Path(args.exit_quality_json)),
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
    </style>
  </head>
  <body>
    <h1>Telemetry Dashboard Snapshot</h1>
    {"".join(sections)}
  </body>
</html>"""
    Path(args.output).write_text(html, encoding="utf-8")
    print(f"Telemetry dashboard page saved to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
