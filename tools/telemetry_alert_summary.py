#!/usr/bin/env python3
"""
Telemetry alert summary reporter.

Scans the generated telemetry artifacts, produces a short summary, and notifies
the Telegram channel whenever anomalies or pause states appear.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any
from collections import Counter

try:
    from eclipse_scalper.tools import replay_trade as _replay_trade  # type: ignore
except Exception:
    try:
        from tools import replay_trade as _replay_trade  # type: ignore
    except Exception:
        _replay_trade = None


def _read_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    try:
        return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except Exception:
        return []


def _load_actions(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _load_exit_quality_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return int(default)


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        if v != v:
            return float(default)
        return v
    except Exception:
        return float(default)


def _parse_kv_lines(lines: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in lines:
        s = str(line or "").strip()
        if not s or "=" not in s:
            continue
        k, v = s.split("=", 1)
        key = str(k or "").strip()
        if not key:
            continue
        out[key] = str(v or "").strip()
    return out


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


def _guard_reason_lines(events: list[dict], limit: int = 6) -> list[str]:
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
        return []
    return [f"- {label}: {cnt}" for label, cnt in counts.most_common(limit)]


def _guard_symbol_lines(events: list[dict], limit: int = 6) -> list[str]:
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
        return []
    return [f"- {symbol}: {cnt}" for symbol, cnt in counts.most_common(limit)]


def _partial_fill_state_lines(events: list[dict], limit: int = 6) -> list[str]:
    outcome_counts: Counter[str] = Counter()
    symbol_counts: Counter[str] = Counter()
    cancel_ok = 0
    flatten_ok = 0
    total = 0
    for ev in events:
        if str(ev.get("event") or "") != "entry.partial_fill_state":
            continue
        data = ev.get("data") if isinstance(ev.get("data"), dict) else {}
        outcome = str(data.get("outcome") or "unknown").strip().lower()
        symbol = str(ev.get("symbol") or data.get("symbol") or "UNKNOWN").strip().upper() or "UNKNOWN"
        outcome_counts[outcome] += 1
        symbol_counts[symbol] += 1
        total += 1
        if bool(data.get("cancel_ok")):
            cancel_ok += 1
        if bool(data.get("flatten_ok")):
            flatten_ok += 1
    if total <= 0:
        return []
    lines = [f"- events: {total} cancel_ok={cancel_ok}/{total} flatten_ok={flatten_ok}/{total}", "- outcomes:"]
    for outcome, cnt in outcome_counts.most_common(limit):
        lines.append(f"- {outcome}: {cnt}")
    lines.append("- top symbols:")
    for symbol, cnt in symbol_counts.most_common(limit):
        lines.append(f"- {symbol}: {cnt}")
    return lines


def _reconcile_first_gate_lines(events: list[dict], limit: int = 6) -> list[str]:
    by_symbol: Counter[str] = Counter()
    by_reason: Counter[str] = Counter()
    by_symbol_severity: dict[str, float] = {}
    total = 0
    for ev in events:
        if str(ev.get("event") or "") != "entry.reconcile_first_gate":
            continue
        data = ev.get("data") if isinstance(ev.get("data"), dict) else {}
        symbol = str(ev.get("symbol") or data.get("symbol") or "UNKNOWN").strip().upper() or "UNKNOWN"
        reason = str(data.get("reason") or "runtime_gate").strip().lower() or "runtime_gate"
        sev = _safe_float(data.get("reconcile_first_severity"), _safe_float(data.get("runtime_gate_degrade_score"), 0.0))
        by_symbol[symbol] += 1
        by_reason[reason] += 1
        if sev > float(by_symbol_severity.get(symbol, 0.0)):
            by_symbol_severity[symbol] = float(sev)
        total += 1
    if total <= 0:
        return []
    lines = [f"- events: {total}", "- reasons:"]
    for reason, cnt in by_reason.most_common(limit):
        lines.append(f"- {reason}: {cnt}")
    lines.append("- top symbols:")
    for symbol, cnt in by_symbol.most_common(limit):
        lines.append(f"- {symbol}: {cnt}")
    sev_sorted = sorted(by_symbol_severity.items(), key=lambda kv: kv[1], reverse=True)[: max(1, int(limit))]
    if sev_sorted:
        lines.append("- top severity symbols:")
        for symbol, sev in sev_sorted:
            lines.append(f"- {symbol}: {sev:.2f}")
    return lines


def _reconcile_first_gate_max_severity(events: list[dict]) -> float:
    mx = 0.0
    for ev in events:
        if str(ev.get("event") or "") != "entry.reconcile_first_gate":
            continue
        data = ev.get("data") if isinstance(ev.get("data"), dict) else {}
        sev = _safe_float(data.get("runtime_gate_degrade_score"), 0.0)
        if sev > mx:
            mx = sev
    return float(mx)


def _reconcile_first_gate_max_streak(events: list[dict], severity_threshold: float) -> int:
    cur = 0
    mx = 0
    th = max(0.0, float(severity_threshold))
    for ev in events:
        if str(ev.get("event") or "") != "entry.reconcile_first_gate":
            continue
        data = ev.get("data") if isinstance(ev.get("data"), dict) else {}
        sev = _safe_float(data.get("reconcile_first_severity"), _safe_float(data.get("runtime_gate_degrade_score"), 0.0))
        if sev >= th:
            cur += 1
            if cur > mx:
                mx = cur
        else:
            cur = 0
    return int(mx)


def _entry_budget_lines(events: list[dict], limit: int = 6) -> tuple[list[str], dict[str, int]]:
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
        return [], {"depleted_total": 0, "scaled_total": 0}
    lines = [
        f"- depleted_total: {depleted_total}",
        f"- scaled_total: {scaled_total}",
    ]
    if depleted_total > 0:
        lines.append("- depleted top symbols:")
        for symbol, cnt in depleted_by_symbol.most_common(limit):
            lines.append(f"- {symbol}: {cnt}")
    if scaled_total > 0:
        lines.append("- scaled top symbols:")
        for symbol, cnt in scaled_by_symbol.most_common(limit):
            lines.append(f"- {symbol}: {cnt}")
    return lines, {"depleted_total": int(depleted_total), "scaled_total": int(scaled_total)}


def _replace_envelope_lines(events: list[dict], limit: int = 6) -> tuple[list[str], int]:
    by_reason: Counter[str] = Counter()
    by_symbol: Counter[str] = Counter()
    total = 0
    for ev in events:
        if str(ev.get("event") or "") != "order.replace_envelope_block":
            continue
        data = ev.get("data") if isinstance(ev.get("data"), dict) else {}
        reason = str(data.get("reason") or "replace_envelope_block").strip().lower() or "replace_envelope_block"
        symbol = str(ev.get("symbol") or data.get("k") or data.get("symbol") or "UNKNOWN").strip().upper() or "UNKNOWN"
        by_reason[reason] += 1
        by_symbol[symbol] += 1
        total += 1
    if total <= 0:
        return [], 0
    lines = [f"- events: {total}", "- reasons:"]
    for reason, cnt in by_reason.most_common(limit):
        lines.append(f"- {reason}: {cnt}")
    lines.append("- top symbols:")
    for symbol, cnt in by_symbol.most_common(limit):
        lines.append(f"- {symbol}: {cnt}")
    return lines, int(total)


def _rebuild_orphan_lines(events: list[dict], limit: int = 6) -> tuple[list[str], dict[str, int]]:
    by_action: Counter[str] = Counter()
    by_class: Counter[str] = Counter()
    total = 0
    freeze_count = 0
    for ev in events:
        if str(ev.get("event") or "") != "rebuild.orphan_decision":
            continue
        data = ev.get("data") if isinstance(ev.get("data"), dict) else {}
        action = str(data.get("action") or "UNKNOWN").strip().upper() or "UNKNOWN"
        klass = str(data.get("class") or "unknown").strip().lower() or "unknown"
        by_action[action] += 1
        by_class[klass] += 1
        total += 1
        if action == "FREEZE":
            freeze_count += 1
    if total <= 0:
        return [], {"total": 0, "freeze_count": 0}
    lines = [f"- events: {total}", "- actions:"]
    for action, cnt in by_action.most_common(limit):
        lines.append(f"- {action}: {cnt}")
    lines.append("- classes:")
    for klass, cnt in by_class.most_common(limit):
        lines.append(f"- {klass}: {cnt}")
    return lines, {"total": int(total), "freeze_count": int(freeze_count)}


def _belief_state_lines(events: list[dict]) -> list[str]:
    rows: list[dict[str, Any]] = []
    for ev in events:
        if str(ev.get("event") or "") != "execution.belief_state":
            continue
        data = ev.get("data") if isinstance(ev.get("data"), dict) else {}
        rows.append(
            {
                "belief_debt_sec": float(data.get("belief_debt_sec") or 0.0),
                "belief_debt_symbols": int(data.get("belief_debt_symbols") or 0),
                "belief_confidence": float(data.get("belief_confidence") or 0.0),
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
                "runtime_gate_cause_summary": str(data.get("runtime_gate_cause_summary") or ""),
                "guard_unlock_conditions": str(data.get("guard_unlock_conditions") or ""),
                "guard_next_unlock_sec": float(data.get("guard_next_unlock_sec") or 0.0),
                "guard_unlock_snapshot": (
                    dict(data.get("guard_unlock_snapshot") or {})
                    if isinstance(data.get("guard_unlock_snapshot"), dict)
                    else {}
                ),
            }
        )
    if not rows:
        return []
    latest = rows[-1]
    avg_conf = sum(float(r["belief_confidence"]) for r in rows) / float(len(rows))
    lines = [
        (
            f"- latest debt={latest['belief_debt_sec']:.1f}s symbols={latest['belief_debt_symbols']} "
            f"confidence={latest['belief_confidence']:.2f} streak={latest['mismatch_streak']}"
        ),
        (
            f"- evidence_coverage={latest['evidence_coverage_ratio']:.3f} "
            f"(ws={latest['evidence_ws_coverage_ratio']:.3f} "
            f"rest={latest['evidence_rest_coverage_ratio']:.3f} "
            f"fill={latest['evidence_fill_coverage_ratio']:.3f})"
        ),
        (
            f"- envelope symbols={latest['belief_envelope_symbols']} "
            f"ambiguous={latest['belief_envelope_ambiguous_symbols']} "
            f"width_sum={latest['belief_position_interval_width_sum']:.3f} "
            f"width_max={latest['belief_position_interval_width_max']:.3f} "
            f"unknown={latest['belief_live_unknown_symbols']} "
            f"worst={latest['belief_envelope_worst_symbol'] or 'n/a'}"
        ),
        (
            f"- latest repairs actions={latest['repair_actions']} "
            f"skipped={latest['repair_skipped']} avg_confidence={avg_conf:.2f}"
        ),
        f"- runtime_gate_cause_summary={str(latest.get('runtime_gate_cause_summary') or 'stable')}",
    ]
    unlock_snapshot = latest.get("guard_unlock_snapshot")
    if isinstance(unlock_snapshot, dict) and unlock_snapshot:
        ht_cur = int(_safe_float(unlock_snapshot.get("healthy_ticks_current"), 0.0))
        ht_req = int(_safe_float(unlock_snapshot.get("healthy_ticks_required"), 0.0))
        cov_cur = _safe_float(unlock_snapshot.get("journal_coverage_current"), 0.0)
        cov_req = _safe_float(unlock_snapshot.get("journal_coverage_required"), 0.0)
        cc_cur = _safe_float(unlock_snapshot.get("contradiction_clear_current_sec"), 0.0)
        cc_req = _safe_float(unlock_snapshot.get("contradiction_clear_required_sec"), 0.0)
        pg_cur = _safe_float(unlock_snapshot.get("protection_gap_current_sec"), 0.0)
        pg_req = _safe_float(unlock_snapshot.get("protection_gap_max_sec"), 0.0)
        lines.append(
            "- unlock_snapshot "
            f"healthy_ticks={ht_cur}/{ht_req} "
            f"journal_coverage={cov_cur:.3f}/{cov_req:.3f} "
            f"contradiction_clear={cc_cur:.0f}s/{cc_req:.0f}s"
        )
        lines.append(
            "- unlock_remaining "
            f"healthy_ticks={max(0, ht_req - ht_cur)} "
            f"journal_coverage={max(0.0, cov_req - cov_cur):.3f} "
            f"contradiction_clear={max(0.0, cc_req - cc_cur):.0f}s "
            f"protection_gap={max(0.0, pg_cur - pg_req):.1f}s"
        )
    return lines


def _latest_belief_state(events: list[dict]) -> dict[str, Any] | None:
    latest = None
    for ev in events:
        if str(ev.get("event") or "") != "execution.belief_state":
            continue
        data = ev.get("data") if isinstance(ev.get("data"), dict) else {}
        latest = {
            "belief_debt_sec": float(data.get("belief_debt_sec") or 0.0),
            "belief_debt_symbols": int(data.get("belief_debt_symbols") or 0),
            "belief_confidence": float(data.get("belief_confidence") or 0.0),
            "mismatch_streak": int(data.get("mismatch_streak") or 0),
        }
    return latest


def _correlation_lines(events: list[dict], limit: int = 6) -> tuple[list[str], dict[str, int | float | str]]:
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
            corr_pressure = _safe_float(data.get("corr_pressure"), 0.0)
            corr_regime = str(data.get("corr_regime") or "").strip().upper()
            if corr_pressure <= 0.0 and not corr_regime:
                continue
            symbol = str(ev.get("symbol") or data.get("symbol") or "UNKNOWN").strip().upper() or "UNKNOWN"
            by_symbol[symbol] += 1
            reason = str(data.get("corr_reason_tags") or "stable").strip().lower() or "stable"
            by_reason[reason] += 1
            if name == "entry.blocked":
                blocked += 1
            else:
                action = str(data.get("action") or "").strip().upper()
                if action == "SCALE":
                    scaled += 1
                elif action in ("DENY", "DEFER"):
                    blocked += 1
            if corr_regime == "STRESS":
                stress += 1
            elif corr_regime == "TIGHTENING":
                tightening += 1
        elif name == "execution.correlation_state":
            latest_regime = str(data.get("corr_regime") or latest_regime)
            latest_pressure = _safe_float(data.get("corr_pressure"), latest_pressure)
            latest_tags = str(data.get("corr_reason_tags") or latest_tags)
    total = blocked + scaled
    if total <= 0 and not latest_regime:
        return [], {"blocked": 0, "scaled": 0, "stress": 0, "tightening": 0}
    lines = [
        f"- entry impact blocked={blocked} scaled={scaled}",
        f"- regime pressure stress={stress} tightening={tightening}",
    ]
    if latest_regime:
        lines.append(f"- latest regime={latest_regime} pressure={latest_pressure:.2f} tags={latest_tags}")
    if by_reason:
        lines.append("- top reason tags:")
        for reason, cnt in by_reason.most_common(limit):
            lines.append(f"- {reason}: {cnt}")
    if by_symbol:
        lines.append("- top symbols:")
        for symbol, cnt in by_symbol.most_common(limit):
            lines.append(f"- {symbol}: {cnt}")
    return lines, {
        "blocked": int(blocked),
        "scaled": int(scaled),
        "stress": int(stress),
        "tightening": int(tightening),
        "latest_regime": str(latest_regime),
        "latest_pressure": float(latest_pressure),
    }


def _latest_corr_or_symbol(events: list[dict]) -> tuple[str, str]:
    corr = ""
    sym = ""
    for ev in reversed(events):
        data = ev.get("data") if isinstance(ev.get("data"), dict) else {}
        if not corr:
            corr = str(data.get("correlation_id") or ev.get("correlation_id") or "").strip()
        if not sym:
            sym = str(ev.get("symbol") or data.get("symbol") or data.get("k") or "").strip().upper()
        if corr and sym:
            break
    return corr, sym


def _corr_vs_exit_quality_lines(
    events: list[dict], *, limit: int = 4
) -> tuple[list[str], dict[str, float | int]]:
    corr_rows: list[tuple[float, str, float]] = []
    closes: list[dict[str, float | str]] = []
    for ev in events:
        name = str(ev.get("event") or "")
        data = ev.get("data") if isinstance(ev.get("data"), dict) else {}
        ts = _safe_float(ev.get("ts"), 0.0)
        if name == "execution.correlation_state":
            regime = str(data.get("corr_regime") or "NORMAL").strip().upper() or "NORMAL"
            pressure = _safe_float(data.get("corr_pressure"), 0.0)
            corr_rows.append((ts, regime, pressure))
        elif name == "position.closed":
            closes.append(
                {
                    "ts": ts,
                    "symbol": str(ev.get("symbol") or data.get("symbol") or "UNKNOWN").strip().upper() or "UNKNOWN",
                    "pnl": _safe_float(data.get("pnl_usdt"), 0.0),
                    "duration": _safe_float(data.get("duration_sec"), 0.0),
                }
            )
    if not closes:
        return [], {}
    corr_rows.sort(key=lambda x: x[0])
    by_regime: dict[str, dict[str, float]] = {}
    for row in closes:
        ts = _safe_float(row.get("ts"), 0.0)
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
        pnl = _safe_float(row.get("pnl"), 0.0)
        agg["count"] += 1.0
        if pnl > 0:
            agg["wins"] += 1.0
        agg["pnl_sum"] += pnl
        agg["dur_sum"] += _safe_float(row.get("duration"), 0.0)
        agg["pressure_sum"] += pressure
    ranked = sorted(by_regime.items(), key=lambda kv: kv[1].get("count", 0.0), reverse=True)[: max(1, int(limit))]
    lines = ["- regimes:"]
    for regime, agg in ranked:
        n = max(1.0, _safe_float(agg.get("count"), 1.0))
        win_rate = _safe_float(agg.get("wins"), 0.0) / n
        avg_pnl = _safe_float(agg.get("pnl_sum"), 0.0) / n
        avg_dur = _safe_float(agg.get("dur_sum"), 0.0) / n
        avg_corr = _safe_float(agg.get("pressure_sum"), 0.0) / n
        lines.append(
            f"- {regime}: n={int(n)} win={win_rate:.0%} avg_pnl={avg_pnl:+.3f} avg_dur={avg_dur:.0f}s avg_corr={avg_corr:.2f}"
        )
    out: dict[str, float | int] = {}
    stress = by_regime.get("STRESS")
    normal = by_regime.get("NORMAL")
    if isinstance(stress, dict):
        sn = max(1.0, _safe_float(stress.get("count"), 1.0))
        out["stress_count"] = int(sn)
        out["stress_avg_pnl"] = _safe_float(stress.get("pnl_sum"), 0.0) / sn
    else:
        out["stress_count"] = 0
        out["stress_avg_pnl"] = 0.0
    if isinstance(normal, dict):
        nn = max(1.0, _safe_float(normal.get("count"), 1.0))
        out["normal_count"] = int(nn)
        out["normal_avg_pnl"] = _safe_float(normal.get("pnl_sum"), 0.0) / nn
    else:
        out["normal_count"] = 0
        out["normal_avg_pnl"] = 0.0
    out["stress_vs_normal_pnl_delta"] = _safe_float(out.get("normal_avg_pnl"), 0.0) - _safe_float(
        out.get("stress_avg_pnl"), 0.0
    )
    return lines, out


def _journal_replay_lines(events: list[dict], journal_path: Path, limit: int = 6) -> list[str]:
    if _replay_trade is None or not journal_path.exists():
        return []
    corr, sym = _latest_corr_or_symbol(events)
    try:
        out = _replay_trade.replay(journal_path, correlation_id=corr, symbol=("" if corr else sym))
    except Exception:
        return []
    transitions = out.get("transitions", []) if isinstance(out, dict) else []
    if not transitions:
        return []
    rows = [
        f"- filter correlation_id={corr or 'n/a'} symbol={sym or 'n/a'}",
        f"- last_state={str(out.get('last_state') or '')} events={int(out.get('count', 0) or 0)}",
    ]
    for tr in transitions[-max(1, int(limit)):]:
        rows.append(
            f"- {tr.get('machine')} {tr.get('entity')} "
            f"{tr.get('from')}->{tr.get('to')} ({tr.get('reason')})"
        )
    return rows


def _reliability_gate_lines(path: Path) -> tuple[list[str], dict[str, Any]]:
    lines = _read_lines(path)
    if not lines:
        return [], {}
    kv = _parse_kv_lines(lines)
    if not kv:
        return [], {}
    replay_mismatch = _safe_int(kv.get("replay_mismatch_count"), 0)
    invalid_transitions = _safe_int(kv.get("invalid_transition_count"), 0)
    coverage_ratio = _safe_float(kv.get("journal_coverage_ratio"), 0.0)
    position_mismatch = _safe_int(kv.get("position_mismatch_count"), 0)
    position_mismatch_peak = _safe_int(kv.get("position_mismatch_count_peak"), position_mismatch)
    orphan_count = _safe_int(kv.get("orphan_count"), 0)
    intent_collision_count = _safe_int(kv.get("intent_collision_count"), 0)
    coverage_gap_seconds = _safe_float(kv.get("protection_coverage_gap_seconds"), 0.0)
    coverage_gap_seconds_peak = _safe_float(kv.get("protection_coverage_gap_seconds_peak"), coverage_gap_seconds)
    replace_race_count = _safe_int(kv.get("replace_race_count"), 0)
    contradiction_count = _safe_int(kv.get("evidence_contradiction_count"), 0)
    cats = {
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
    raw_cats = str(kv.get("replay_mismatch_categories") or "").strip()
    if raw_cats:
        try:
            payload = json.loads(raw_cats)
            if isinstance(payload, dict):
                for k in cats.keys():
                    cats[k] = _safe_int(payload.get(k), 0)
        except Exception:
            pass
    rows = [
        f"- replay_mismatch_count: {replay_mismatch}",
        f"- invalid_transition_count: {invalid_transitions}",
        f"- journal_coverage_ratio: {coverage_ratio:.3f}",
        f"- position_mismatch_count: {position_mismatch} (peak {position_mismatch_peak})",
        f"- orphan_count: {orphan_count}",
        f"- intent_collision_count: {intent_collision_count}",
        f"- protection_coverage_gap_seconds: {coverage_gap_seconds:.1f} (peak {coverage_gap_seconds_peak:.1f})",
        f"- replace_race_count: {replace_race_count}",
        f"- evidence_contradiction_count: {contradiction_count}",
    ]
    if any(int(cats.get(k, 0)) > 0 for k in cats.keys()):
        rows.append(
            "- mismatch_categories: "
            f"ledger={int(cats['ledger'])} transition={int(cats['transition'])} "
            f"belief={int(cats['belief'])} position={int(cats['position'])} "
            f"orphan={int(cats['orphan'])} coverage_gap={int(cats['coverage_gap'])} "
            f"replace_race={int(cats['replace_race'])} contradiction={int(cats['contradiction'])} "
            f"unknown={int(cats['unknown'])}"
        )
        ranked = sorted(
            [(str(k), int(v)) for (k, v) in cats.items() if int(v) > 0],
            key=lambda kv: int(kv[1]),
            reverse=True,
        )[:3]
        if ranked:
            rows.append("- top_contributors: " + ", ".join(f"{k}={v}" for (k, v) in ranked))
        critical_keys = ("position", "orphan", "coverage_gap", "replace_race", "contradiction")
        critical_ranked = sorted(
            [(str(k), int(cats.get(k, 0))) for k in critical_keys if int(cats.get(k, 0)) > 0],
            key=lambda kv: int(kv[1]),
            reverse=True,
        )[:3]
        if critical_ranked:
            rows.append("- critical_contributors: " + ", ".join(f"{k}={v}" for (k, v) in critical_ranked))
    return rows, {
        "replay_mismatch_count": replay_mismatch,
        "invalid_transition_count": invalid_transitions,
        "journal_coverage_ratio": coverage_ratio,
        "position_mismatch_count": position_mismatch,
        "position_mismatch_count_peak": position_mismatch_peak,
        "orphan_count": orphan_count,
        "intent_collision_count": intent_collision_count,
        "protection_coverage_gap_seconds": coverage_gap_seconds,
        "protection_coverage_gap_seconds_peak": coverage_gap_seconds_peak,
        "replace_race_count": replace_race_count,
        "evidence_contradiction_count": contradiction_count,
        "replay_mismatch_categories": cats,
    }


def _intent_collision_policy_lines(
    *,
    reliability_metrics: dict[str, Any],
    notify_state_path: Path,
    reliability_max_intent_collision_count: int,
    intent_collision_critical_threshold: int,
    intent_collision_critical_streak: int,
) -> tuple[list[str], dict[str, Any]]:
    state = _load_json(notify_state_path)
    current_count = int(_safe_int(reliability_metrics.get("intent_collision_count"), 0))
    streak = int(_safe_int(state.get("intent_collision_streak"), 0))
    state_level = str(state.get("level") or "").strip().lower()
    lines = [
        f"- current_count: {current_count}",
        (
            f"- thresholds: reliability_max={max(0, int(reliability_max_intent_collision_count))} "
            f"critical={max(1, int(intent_collision_critical_threshold))} "
            f"critical_streak={max(1, int(intent_collision_critical_streak))}"
        ),
        f"- notifier_streak: {streak}",
        f"- notifier_level: {state_level or 'n/a'}",
    ]
    metrics = {
        "intent_collision_count": current_count,
        "intent_collision_streak": streak,
        "notify_level": state_level or "",
    }
    return lines, metrics


def _build_message(signal_lines, core_lines, anomaly_lines, signal_exit_lines) -> tuple[str, bool]:
    warn = False
    parts = []
    if anomaly_lines:
        parts.append("Anomaly report:")
        parts.extend(anomaly_lines)
        warn = warn or any("Anomalies detected" in l for l in anomaly_lines)
        warn = warn or any("pause" in l.lower() for l in anomaly_lines)
    if signal_lines:
        parts.append("\nSignal data health (top lines):")
        parts.extend(signal_lines[:4])
    if core_lines:
        parts.append("\nCore health (summary):")
        parts.extend(core_lines[:4])
    if not parts:
        parts.append("Telemetry artifacts were empty.")
    if signal_exit_lines:
        parts.append("\nSignal/exit context:")
        parts.extend(signal_exit_lines[:4])
    return "\n".join(parts), warn


def _build_notifier():
    token = os.getenv("TELEGRAM_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat_id:
        return None
    try:
        from eclipse_scalper.notifications.telegram import Notifier

        return Notifier(token=token, chat_id=chat_id)
    except Exception:
        return None


def _send_alert(message: str, notifier) -> None:
    if notifier is None or not message.strip():
        return
    try:
        asyncio.run(notifier.speak(message, "critical"))
    except Exception:
        pass


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Telemetry alert summary (artifact -> Telegram)")
    parser.add_argument("--signal", default="logs/signal_data_health.txt")
    parser.add_argument("--core", default="logs/core_health.txt")
    parser.add_argument("--anomaly", default="logs/telemetry_anomaly.txt")
    parser.add_argument("--exit-quality", default="logs/exit_quality.txt")
    parser.add_argument("--exit-quality-json", default="logs/exit_quality_summary.json")
    parser.add_argument("--exit-quality-win-drop", type=float, default=0.15)
    parser.add_argument("--exit-quality-pnl-drop", type=float, default=0.15)
    parser.add_argument("--telemetry", default="logs/telemetry.jsonl")
    parser.add_argument("--guard-reasons-top", type=int, default=6)
    parser.add_argument("--guard-symbols-top", type=int, default=6)
    parser.add_argument("--reconcile-first-gate-threshold", type=int, default=1)
    parser.add_argument(
        "--reconcile-first-gate-severity-threshold",
        type=float,
        default=float(os.getenv("RECONCILE_FIRST_GATE_SEVERITY_THRESHOLD", "1.01")),
    )
    parser.add_argument(
        "--reconcile-first-gate-severity-streak-threshold",
        type=int,
        default=int(os.getenv("RECONCILE_FIRST_GATE_SEVERITY_STREAK_THRESHOLD", "2")),
    )
    parser.add_argument("--entry-budget-depleted-threshold", type=int, default=1)
    parser.add_argument("--replace-envelope-threshold", type=int, default=1)
    parser.add_argument("--rebuild-orphan-freeze-threshold", type=int, default=1)
    parser.add_argument(
        "--corr-stress-threshold",
        type=int,
        default=int(os.getenv("CORR_STRESS_THRESHOLD", os.getenv("CORR_STRESS_CRITICAL_THRESHOLD", "1"))),
    )
    parser.add_argument(
        "--corr-pressure-threshold",
        type=float,
        default=float(os.getenv("CORR_PRESSURE_THRESHOLD", os.getenv("CORR_PRESSURE_CRITICAL_THRESHOLD", "0.90"))),
    )
    parser.add_argument(
        "--corr-symbol-limit",
        type=int,
        default=6,
        help="Rows to show in correlation contribution section",
    )
    parser.add_argument(
        "--corr-exit-pnl-drop-threshold",
        type=float,
        default=float(
            os.getenv(
                "CORR_EXIT_PNL_DROP_THRESHOLD",
                os.getenv("CORR_EXIT_PNL_DROP_CRITICAL_THRESHOLD", "0.10"),
            )
        ),
        help="Warn when stress avg pnl is worse than normal avg pnl by this absolute amount",
    )
    parser.add_argument("--reliability-gate", default="logs/reliability_gate.txt")
    parser.add_argument("--reliability-max-replay-mismatch", type=int, default=0)
    parser.add_argument("--reliability-max-invalid-transitions", type=int, default=0)
    parser.add_argument("--reliability-min-journal-coverage", type=float, default=0.90)
    parser.add_argument(
        "--reliability-max-intent-collision-count",
        type=int,
        default=int(os.getenv("RELIABILITY_GATE_MAX_INTENT_COLLISION_COUNT", "0")),
    )
    parser.add_argument(
        "--intent-collision-critical-threshold",
        type=int,
        default=int(os.getenv("INTENT_COLLISION_CRITICAL_THRESHOLD", "1")),
    )
    parser.add_argument(
        "--intent-collision-critical-streak",
        type=int,
        default=int(os.getenv("INTENT_COLLISION_CRITICAL_STREAK", "2")),
    )
    parser.add_argument(
        "--notify-state",
        default=os.getenv("TELEMETRY_DASHBOARD_NOTIFY_STATE_PATH", "logs/telemetry_dashboard_notify_state.json"),
    )
    parser.add_argument("--output", default="logs/telemetry_alert_summary.txt")
    parser.add_argument(
        "--actions-path",
        default="logs/telemetry_anomaly_actions.json",
        help="Path to anomaly mitigation metadata",
    )
    parser.add_argument("--no-notify", action="store_true")
    parser.add_argument("--signal-exit", default="logs/signal_exit_notify.txt")
    parser.add_argument("--journal", default="logs/execution_journal.jsonl")
    args = parser.parse_args(argv)

    signal_lines = _read_lines(Path(args.signal))
    core_lines = _read_lines(Path(args.core))
    anomaly_lines = _read_lines(Path(args.anomaly))
    exit_quality_lines = _read_lines(Path(args.exit_quality))
    exit_quality_json = _load_exit_quality_json(Path(args.exit_quality_json))
    signal_exit_lines = _read_lines(Path(args.signal_exit))
    actions = _load_actions(Path(args.actions_path))
    telemetry_events = _load_jsonl(Path(args.telemetry))

    message, warn = _build_message(signal_lines, core_lines, anomaly_lines, signal_exit_lines)
    if exit_quality_lines:
        message += "\n\nExit quality (top lines):\n" + "\n".join(exit_quality_lines[:6])
    if exit_quality_json:
        w24 = exit_quality_json.get("window_24h", {})
        w7 = exit_quality_json.get("window_7d", {})
        message += "\n\nExit window:"
        message += f"\n- 24h win rate {w24.get('win_rate', 0.0):.0%} avg pnl {w24.get('avg_pnl', 0.0):+.2f}"
        message += f"\n- 7d win rate {w7.get('win_rate', 0.0):.0%} avg pnl {w7.get('avg_pnl', 0.0):+.2f}"
        drift_notes = []
        win_drop = (w7.get("win_rate", 0.0) or 0.0) - (w24.get("win_rate", 0.0) or 0.0)
        if w7.get("win_rate", 0.0) > 0 and win_drop >= args.exit_quality_win_drop:
            drift_notes.append(f"win rate down {win_drop:.0%}")
            warn = True
        pnl7 = w7.get("avg_pnl", 0.0)
        pnl_drop = (pnl7 or 0.0) - (w24.get("avg_pnl", 0.0) or 0.0)
        if pnl7 > 0 and (pnl_drop / pnl7) >= args.exit_quality_pnl_drop:
            drift_notes.append(f"pnl down {pnl_drop:.2f}")
            warn = True
        if drift_notes:
            message += "\n" + " | ".join(drift_notes)
    guard_reason_lines = _guard_reason_lines(telemetry_events, max(1, args.guard_reasons_top))
    if guard_reason_lines:
        message += "\n\nTop guard reasons:\n" + "\n".join(guard_reason_lines)
    guard_symbol_lines = _guard_symbol_lines(telemetry_events, max(1, args.guard_symbols_top))
    if guard_symbol_lines:
        message += "\n\nGuard hits by symbol:\n" + "\n".join(guard_symbol_lines)
    partial_lines = _partial_fill_state_lines(telemetry_events, max(1, args.guard_symbols_top))
    if partial_lines:
        message += "\n\nPartial fill states:\n" + "\n".join(partial_lines)
    reconcile_gate_lines = _reconcile_first_gate_lines(telemetry_events, max(1, args.guard_symbols_top))
    reconcile_gate_max_sev = _reconcile_first_gate_max_severity(telemetry_events)
    reconcile_gate_max_streak = _reconcile_first_gate_max_streak(
        telemetry_events, max(0.0, float(args.reconcile_first_gate_severity_threshold))
    )
    if reconcile_gate_lines:
        message += "\n\nReconcile-first gate:\n" + "\n".join(reconcile_gate_lines)
        message += f"\n- max_severity: {reconcile_gate_max_sev:.2f}"
        message += f"\n- max_severity_streak: {reconcile_gate_max_streak}"
        gate_events = 0
        try:
            gate_events = int(str(reconcile_gate_lines[0]).split(":", 1)[1].strip())
        except Exception:
            gate_events = 0
        if gate_events >= max(1, int(args.reconcile_first_gate_threshold)):
            warn = True
        if reconcile_gate_max_sev >= max(0.0, float(args.reconcile_first_gate_severity_threshold)):
            warn = True
        if reconcile_gate_max_streak >= max(1, int(args.reconcile_first_gate_severity_streak_threshold)):
            warn = True
    entry_budget_lines, entry_budget_stats = _entry_budget_lines(telemetry_events, max(1, args.guard_symbols_top))
    if entry_budget_lines:
        message += "\n\nEntry budget pressure:\n" + "\n".join(entry_budget_lines)
        if int(entry_budget_stats.get("depleted_total", 0)) >= max(1, int(args.entry_budget_depleted_threshold)):
            warn = True
    replace_envelope_lines, replace_envelope_total = _replace_envelope_lines(
        telemetry_events, max(1, args.guard_symbols_top)
    )
    if replace_envelope_lines:
        message += "\n\nReplace envelope blocks:\n" + "\n".join(replace_envelope_lines)
        if int(replace_envelope_total) >= max(1, int(args.replace_envelope_threshold)):
            warn = True
    rebuild_orphan_lines, rebuild_orphan_stats = _rebuild_orphan_lines(
        telemetry_events, max(1, args.guard_symbols_top)
    )
    if rebuild_orphan_lines:
        message += "\n\nRebuild orphan decisions:\n" + "\n".join(rebuild_orphan_lines)
        if int(rebuild_orphan_stats.get("freeze_count", 0)) >= max(1, int(args.rebuild_orphan_freeze_threshold)):
            warn = True
    belief_lines = _belief_state_lines(telemetry_events)
    if belief_lines:
        message += "\n\nExecution belief state:\n" + "\n".join(belief_lines)
    corr_lines, corr_stats = _correlation_lines(telemetry_events, max(1, int(args.corr_symbol_limit)))
    if corr_lines:
        message += "\n\nCorrelation contribution:\n" + "\n".join(corr_lines)
        if int(corr_stats.get("stress", 0)) >= max(1, int(args.corr_stress_threshold)):
            warn = True
        if float(_safe_float(corr_stats.get("latest_pressure"), 0.0)) >= max(0.0, float(args.corr_pressure_threshold)):
            warn = True
    corr_exit_lines, corr_exit_stats = _corr_vs_exit_quality_lines(telemetry_events, limit=max(1, int(args.corr_symbol_limit)))
    if corr_exit_lines:
        message += "\n\nCorr vs exit quality:\n" + "\n".join(corr_exit_lines)
        stress_n = int(_safe_int(corr_exit_stats.get("stress_count"), 0))
        normal_n = int(_safe_int(corr_exit_stats.get("normal_count"), 0))
        pnl_delta = _safe_float(corr_exit_stats.get("stress_vs_normal_pnl_delta"), 0.0)
        if stress_n > 0 and normal_n > 0 and pnl_delta >= max(0.0, float(args.corr_exit_pnl_drop_threshold)):
            warn = True
    belief_latest = _latest_belief_state(telemetry_events)
    if belief_latest is not None:
        if float(belief_latest.get("belief_confidence") or 0.0) < 0.75:
            warn = True
        if float(belief_latest.get("belief_debt_sec") or 0.0) >= 300.0:
            warn = True
    replay_lines = _journal_replay_lines(telemetry_events, Path(args.journal))
    if replay_lines:
        message += "\n\nExecution replay:\n" + "\n".join(replay_lines[:8])
    reliability_lines, reliability_metrics = _reliability_gate_lines(Path(args.reliability_gate))
    if reliability_lines:
        message += "\n\nReliability gate:\n" + "\n".join(reliability_lines)
        if int(reliability_metrics.get("replay_mismatch_count") or 0) > max(0, int(args.reliability_max_replay_mismatch)):
            warn = True
        if int(reliability_metrics.get("invalid_transition_count") or 0) > max(0, int(args.reliability_max_invalid_transitions)):
            warn = True
        if float(reliability_metrics.get("journal_coverage_ratio") or 0.0) < max(0.0, min(1.0, float(args.reliability_min_journal_coverage))):
            warn = True
        intent_collision_count = int(_safe_int(reliability_metrics.get("intent_collision_count"), 0))
        if intent_collision_count > max(0, int(args.reliability_max_intent_collision_count)):
            warn = True
        if intent_collision_count >= max(1, int(args.intent_collision_critical_threshold)):
            warn = True
        collision_policy_lines, collision_metrics = _intent_collision_policy_lines(
            reliability_metrics=reliability_metrics,
            notify_state_path=Path(args.notify_state),
            reliability_max_intent_collision_count=max(0, int(args.reliability_max_intent_collision_count)),
            intent_collision_critical_threshold=max(1, int(args.intent_collision_critical_threshold)),
            intent_collision_critical_streak=max(1, int(args.intent_collision_critical_streak)),
        )
        if int(_safe_int(collision_metrics.get("intent_collision_streak"), 0)) >= max(
            1, int(args.intent_collision_critical_streak)
        ):
            warn = True
        message += "\n\nIntent collision policy:\n" + "\n".join(collision_policy_lines)
    action_notes = []
    now = time.time()
    pause_until = float(actions.get("pause_until", 0) or 0)
    if pause_until and pause_until > now:
        ts = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(pause_until))
        reason = str(actions.get("pause_reason") or "").strip()
        note = f"auto-pause until {ts}"
        if reason:
            note += f" ({reason})"
        action_notes.append(note)
    exit_mult = float(actions.get("classifier_exit_multiplier", 1.0))
    conf_mult = float(actions.get("classifier_confidence_multiplier", 1.0))
    if exit_mult != 1.0:
        action_notes.append(f"classifier exit mult x{exit_mult:.2f}")
    if conf_mult != 1.0:
        action_notes.append(f"classifier confidence mult x{conf_mult:.2f}")
    if actions.get("anomaly_messages"):
        msgs = actions.get("anomaly_messages")[:3]
        action_notes.append("anomalies: " + "; ".join(str(m) for m in msgs))
    if action_notes:
        message += "\n\nAuto mitigation:\n" + "\n".join(action_notes)
    Path(args.output).write_text(message + "\n", encoding="utf-8")
    print(message)

    if warn and not args.no_notify:
        notifier = _build_notifier()
        _send_alert(message, notifier)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
