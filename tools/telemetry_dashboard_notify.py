#!/usr/bin/env python3
"""
Telemetry dashboard notifier.

Runs the dashboard helper, prints the snapshot for the workflow log, and
sends the same summary over Telegram when `TELEGRAM_TOKEN` / `TELEGRAM_CHAT_ID`
are configured (e.g., via GitHub Secrets).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
TELEMETRY_SCRIPT = ROOT / "eclipse_scalper" / "tools" / "telemetry_dashboard.py"


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


def _send_alert(notifier, text: str, level: str = "critical") -> None:
    if notifier is None or not text.strip():
        return
    try:
        asyncio.run(notifier.speak(text, str(level or "critical")))
    except Exception:
        pass


def _truncate(text: str, max_len: int = 3800) -> str:
    if max_len <= 0:
        return ""
    if len(text) <= max_len:
        return text
    suffix = " …"
    return text[: max_len - len(suffix)] + suffix


def _load_jsonl(path: Path, limit: int = 5000) -> list[dict]:
    if not path.exists():
        return []
    out: list[dict] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if not s:
                continue
            try:
                out.append(json.loads(s))
            except Exception:
                continue
            if len(out) >= max(1, int(limit)):
                break
    except Exception:
        return out
    return out


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


def _extract_unlock_metrics(unlock: str, next_unlock_sec: float) -> dict[str, float]:
    text = str(unlock or "").strip()
    out: dict[str, float] = {
        "unlock_next_sec": float(max(0.0, _safe_float(next_unlock_sec, 0.0))),
        "unlock_healthy_ticks_current": 0.0,
        "unlock_healthy_ticks_required": 0.0,
        "unlock_healthy_ticks_remaining": 0.0,
        "unlock_journal_coverage_current": 0.0,
        "unlock_journal_coverage_required": 0.0,
        "unlock_journal_coverage_remaining": 0.0,
        "unlock_contradiction_clear_current_sec": 0.0,
        "unlock_contradiction_clear_required_sec": 0.0,
        "unlock_contradiction_clear_remaining_sec": 0.0,
        "unlock_protection_gap_current_sec": 0.0,
        "unlock_protection_gap_max_sec": 0.0,
        "unlock_protection_gap_remaining_sec": 0.0,
    }
    if not text:
        return out

    m = re.search(r"healthy_ticks\s+([0-9]+)\s*/\s*([0-9]+)", text, flags=re.IGNORECASE)
    if m:
        cur = _safe_float(m.group(1), 0.0)
        req = _safe_float(m.group(2), 0.0)
        out["unlock_healthy_ticks_current"] = cur
        out["unlock_healthy_ticks_required"] = req
        out["unlock_healthy_ticks_remaining"] = float(max(0.0, req - cur))

    m = re.search(r"journal_coverage\s+([0-9]*\.?[0-9]+)\s*/\s*([0-9]*\.?[0-9]+)", text, flags=re.IGNORECASE)
    if m:
        cur = _safe_float(m.group(1), 0.0)
        req = _safe_float(m.group(2), 0.0)
        out["unlock_journal_coverage_current"] = cur
        out["unlock_journal_coverage_required"] = req
        out["unlock_journal_coverage_remaining"] = float(max(0.0, req - cur))

    m = re.search(r"contradiction_clear\s+([0-9]*\.?[0-9]+)s\s*/\s*([0-9]*\.?[0-9]+)s", text, flags=re.IGNORECASE)
    if m:
        cur = _safe_float(m.group(1), 0.0)
        req = _safe_float(m.group(2), 0.0)
        out["unlock_contradiction_clear_current_sec"] = cur
        out["unlock_contradiction_clear_required_sec"] = req
        out["unlock_contradiction_clear_remaining_sec"] = float(max(0.0, req - cur))

    m = re.search(r"protection_gap\s+([0-9]*\.?[0-9]+)s\s*/\s*([0-9]*\.?[0-9]+)s", text, flags=re.IGNORECASE)
    if m:
        cur = _safe_float(m.group(1), 0.0)
        req = _safe_float(m.group(2), 0.0)
        out["unlock_protection_gap_current_sec"] = cur
        out["unlock_protection_gap_max_sec"] = req
        out["unlock_protection_gap_remaining_sec"] = float(max(0.0, cur - req))

    return out


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


def _parse_replay_mismatch_ids(lines: list[str], limit: int = 5) -> list[str]:
    out: list[str] = []
    in_ids = False
    for line in lines:
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
                break
            cid = str(s[1:] or "").strip()
            if cid:
                out.append(cid)
                if len(out) >= max(1, int(limit)):
                    break
    return out


def _reliability_gate_snippet(
    path: Path,
    *,
    max_replay_mismatch: int = 0,
    max_invalid_transitions: int = 0,
    min_journal_coverage: float = 0.90,
) -> tuple[str, bool, dict[str, float]]:
    if not path.exists():
        return "", False, {}
    try:
        lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except Exception:
        return "", False, {}
    kv = _parse_kv_lines(lines)
    if not kv:
        return "", False, {}
    mismatch_ids = _parse_replay_mismatch_ids(lines, limit=5)
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
    raw_cats = str(kv.get("replay_mismatch_categories") or "").strip()
    if raw_cats:
        try:
            payload = json.loads(raw_cats)
            if isinstance(payload, dict):
                for k in list(categories.keys()):
                    categories[k] = _safe_int(payload.get(k), 0)
        except Exception:
            pass
    mismatch = _safe_int(kv.get("replay_mismatch_count"), 0)
    invalid = _safe_int(kv.get("invalid_transition_count"), 0)
    coverage = _safe_float(kv.get("journal_coverage_ratio"), 0.0)
    degraded = not (
        mismatch <= max(0, int(max_replay_mismatch))
        and invalid <= max(0, int(max_invalid_transitions))
        and coverage >= max(0.0, min(1.0, float(min_journal_coverage)))
    )
    status = "DEGRADED" if degraded else "OK"
    msg = (
        "Execution reliability gate:\n"
        f"- status={status} replay_mismatch={mismatch} invalid_transitions={invalid} "
        f"journal_coverage={coverage:.3f}"
    )
    if any(int(categories.get(k, 0)) > 0 for k in categories.keys()):
        msg += (
            "\n- mismatch_categories: "
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
            msg += "\n- top_contributors: " + ", ".join(f"{k}={v}" for (k, v) in ranked)
        critical_keys = ("position", "orphan", "coverage_gap", "replace_race", "contradiction")
        critical_ranked = sorted(
            [(str(k), int(categories.get(k, 0))) for k in critical_keys if int(categories.get(k, 0)) > 0],
            key=lambda kv: int(kv[1]),
            reverse=True,
        )[:3]
        if critical_ranked:
            msg += "\n- critical_contributors: " + ", ".join(f"{k}={v}" for (k, v) in critical_ranked)
    if mismatch_ids:
        msg += "\n- missing_ids: " + ", ".join(mismatch_ids)
    return msg, degraded, {
        "replay_mismatch_count": float(mismatch),
        "invalid_transition_count": float(invalid),
        "journal_coverage_ratio": float(coverage),
        "replay_mismatch_cat_ledger": float(categories["ledger"]),
        "replay_mismatch_cat_transition": float(categories["transition"]),
        "replay_mismatch_cat_belief": float(categories["belief"]),
        "replay_mismatch_cat_position": float(categories["position"]),
        "replay_mismatch_cat_orphan": float(categories["orphan"]),
        "replay_mismatch_cat_coverage_gap": float(categories["coverage_gap"]),
        "replay_mismatch_cat_replace_race": float(categories["replace_race"]),
        "replay_mismatch_cat_contradiction": float(categories["contradiction"]),
        "replay_mismatch_cat_unknown": float(categories["unknown"]),
    }


def _belief_state_snippet(path: Path) -> tuple[str, dict[str, Any]]:
    events = _load_jsonl(path)
    rows = []
    for ev in events:
        if str(ev.get("event") or "") != "execution.belief_state":
            continue
        data = ev.get("data") if isinstance(ev.get("data"), dict) else {}
        rows.append(
            {
                "debt": float(data.get("belief_debt_sec") or 0.0),
                "symbols": int(data.get("belief_debt_symbols") or 0),
                "conf": float(data.get("belief_confidence") or 0.0),
                "streak": int(data.get("mismatch_streak") or 0),
                "guard_mode": str(data.get("guard_mode") or ""),
                "allow_entries": bool(data.get("allow_entries", True)),
                "reason": str((data.get("trace") or {}).get("reason") if isinstance(data.get("trace"), dict) else ""),
                "refresh_blocked_level": float(data.get("guard_refresh_blocked_level") or 0.0),
                "refresh_force_level": float(data.get("guard_refresh_force_level") or 0.0),
                "recovery_stage": str(data.get("guard_recovery_stage") or ""),
                "unlock": str(data.get("guard_unlock_conditions") or ""),
                "next_unlock_sec": float(data.get("guard_next_unlock_sec") or 0.0),
            }
        )
    if not rows:
        return "", {}
    latest = rows[-1]
    msg = (
        "Execution belief state:\n"
        f"- debt={latest['debt']:.1f}s symbols={latest['symbols']} "
        f"confidence={latest['conf']:.2f} streak={latest['streak']}\n"
        f"- guard_mode={latest['guard_mode'] or 'n/a'} allow_entries={bool(latest['allow_entries'])} "
        f"recovery_stage={latest['recovery_stage'] or 'n/a'} next_unlock_sec={float(latest['next_unlock_sec']):.1f}\n"
        f"- refresh_levels blocked={float(latest['refresh_blocked_level']):.2f} "
        f"force={float(latest['refresh_force_level']):.2f}\n"
        f"- unlock_conditions={latest['unlock'] or 'stable'}"
    )
    if "protection_refresh_budget_hard_block" in str(latest.get("reason") or ""):
        msg += "\n- mitigation: protection_refresh_budget_hard_block active (entries clamped)"
    return msg, latest


def _recovery_stage_snippet(path: Path, *, limit: int = 3) -> tuple[str, str]:
    events = _load_jsonl(path)
    stage_counts: dict[str, int] = {}
    latest_stage = ""
    latest_unlock = ""
    for ev in events:
        if str(ev.get("event") or "") != "execution.belief_state":
            continue
        data = ev.get("data") if isinstance(ev.get("data"), dict) else {}
        stage = str(data.get("guard_recovery_stage") or "").strip().upper()
        if stage:
            stage_counts[stage] = int(stage_counts.get(stage, 0)) + 1
            latest_stage = stage
            latest_unlock = str(data.get("guard_unlock_conditions") or "").strip()
    if not stage_counts:
        return "", ""
    top = sorted(stage_counts.items(), key=lambda kv: kv[1], reverse=True)[: max(1, int(limit))]
    parts = [f"{k}:{v}" for k, v in top]
    line = "Recovery stages: " + ", ".join(parts)
    if latest_stage:
        line += f" (latest={latest_stage})"
    if latest_unlock:
        line += f"\n- unlock={latest_unlock}"
    return line, latest_stage


def _reconcile_first_gate_snippet(
    path: Path, *, limit: int = 3, severity_threshold: float = 1.01
) -> tuple[str, int, float, int]:
    events = _load_jsonl(path)
    rows: list[dict[str, str]] = []
    max_degrade_score = 0.0
    current_streak = 0
    max_streak = 0
    by_symbol_severity: dict[str, float] = {}
    for ev in events:
        if str(ev.get("event") or "") != "entry.reconcile_first_gate":
            continue
        data = ev.get("data") if isinstance(ev.get("data"), dict) else {}
        symbol = str(ev.get("symbol") or data.get("symbol") or "UNKNOWN").strip().upper() or "UNKNOWN"
        reason = str(data.get("reason") or "runtime_gate").strip().lower() or "runtime_gate"
        score = _safe_float(data.get("reconcile_first_severity"), _safe_float(data.get("runtime_gate_degrade_score"), 0.0))
        if score > max_degrade_score:
            max_degrade_score = score
        if score > float(by_symbol_severity.get(symbol, 0.0)):
            by_symbol_severity[symbol] = float(score)
        if score >= float(severity_threshold):
            current_streak += 1
            if current_streak > max_streak:
                max_streak = current_streak
        else:
            current_streak = 0
        rows.append({"symbol": symbol, "reason": reason})
    if not rows:
        return "", 0, 0.0, 0
    reason_counts: dict[str, int] = {}
    symbol_counts: dict[str, int] = {}
    for row in rows:
        reason_counts[row["reason"]] = int(reason_counts.get(row["reason"], 0)) + 1
        symbol_counts[row["symbol"]] = int(symbol_counts.get(row["symbol"], 0)) + 1
    reasons_sorted = sorted(reason_counts.items(), key=lambda kv: kv[1], reverse=True)[: max(1, int(limit))]
    symbols_sorted = sorted(symbol_counts.items(), key=lambda kv: kv[1], reverse=True)[: max(1, int(limit))]
    sev_sorted = sorted(by_symbol_severity.items(), key=lambda kv: kv[1], reverse=True)[: max(1, int(limit))]
    lines = [
        (
            f"Reconcile-first gate: events={len(rows)} max_severity={max_degrade_score:.2f} "
            f"max_streak={max_streak}"
        ),
        "- top reasons:",
    ]
    for reason, cnt in reasons_sorted:
        lines.append(f"- {reason}: {cnt}")
    lines.append("- top symbols:")
    for sym, cnt in symbols_sorted:
        lines.append(f"- {sym}: {cnt}")
    if sev_sorted:
        lines.append("- top severity symbols:")
        for sym, sev in sev_sorted:
            lines.append(f"- {sym}: {sev:.2f}")
    return "\n".join(lines), int(len(rows)), float(max_degrade_score), int(max_streak)


def _entry_budget_pressure_snippet(path: Path, *, limit: int = 3) -> tuple[str, int]:
    events = _load_jsonl(path)
    depleted_by_symbol: dict[str, int] = {}
    scaled_by_symbol: dict[str, int] = {}
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
                depleted_by_symbol[symbol] = int(depleted_by_symbol.get(symbol, 0)) + 1
        elif name == "entry.notional_scaled":
            reason = str(data.get("reason") or "").strip().lower()
            if reason == "entry_budget_allocator":
                scaled_total += 1
                scaled_by_symbol[symbol] = int(scaled_by_symbol.get(symbol, 0)) + 1
    if depleted_total <= 0 and scaled_total <= 0:
        return "", 0
    dep_sorted = sorted(depleted_by_symbol.items(), key=lambda kv: kv[1], reverse=True)[: max(1, int(limit))]
    scl_sorted = sorted(scaled_by_symbol.items(), key=lambda kv: kv[1], reverse=True)[: max(1, int(limit))]
    lines = [
        f"Entry budget pressure: depleted={depleted_total} scaled={scaled_total}",
    ]
    if dep_sorted:
        lines.append("- depleted symbols:")
        for sym, cnt in dep_sorted:
            lines.append(f"- {sym}: {cnt}")
    if scl_sorted:
        lines.append("- scaled symbols:")
        for sym, cnt in scl_sorted:
            lines.append(f"- {sym}: {cnt}")
    return "\n".join(lines), int(depleted_total)


def _replace_envelope_snippet(path: Path, *, limit: int = 3) -> tuple[str, int]:
    events = _load_jsonl(path)
    reason_counts: dict[str, int] = {}
    symbol_counts: dict[str, int] = {}
    total = 0
    for ev in events:
        if str(ev.get("event") or "") != "order.replace_envelope_block":
            continue
        data = ev.get("data") if isinstance(ev.get("data"), dict) else {}
        reason = str(data.get("reason") or "replace_envelope_block").strip().lower() or "replace_envelope_block"
        symbol = str(ev.get("symbol") or data.get("k") or data.get("symbol") or "UNKNOWN").strip().upper() or "UNKNOWN"
        reason_counts[reason] = int(reason_counts.get(reason, 0)) + 1
        symbol_counts[symbol] = int(symbol_counts.get(symbol, 0)) + 1
        total += 1
    if total <= 0:
        return "", 0
    reason_sorted = sorted(reason_counts.items(), key=lambda kv: kv[1], reverse=True)[: max(1, int(limit))]
    symbol_sorted = sorted(symbol_counts.items(), key=lambda kv: kv[1], reverse=True)[: max(1, int(limit))]
    lines = [f"Replace envelope blocks: events={total}", "- top reasons:"]
    for reason, cnt in reason_sorted:
        lines.append(f"- {reason}: {cnt}")
    lines.append("- top symbols:")
    for sym, cnt in symbol_sorted:
        lines.append(f"- {sym}: {cnt}")
    return "\n".join(lines), int(total)


def _protection_refresh_budget_snippet(path: Path) -> tuple[str, int, int]:
    events = _load_jsonl(path)
    blocked_total = 0
    force_total = 0
    stop_blocked_total = 0
    tp_blocked_total = 0
    stop_force_total = 0
    tp_force_total = 0
    latest_blocked = 0
    latest_force = 0
    latest_stop_blocked = 0
    latest_tp_blocked = 0
    latest_stop_force = 0
    latest_tp_force = 0
    prev_blocked = 0
    prev_force = 0
    for ev in events:
        if str(ev.get("event") or "") != "reconcile.summary":
            continue
        data = ev.get("data") if isinstance(ev.get("data"), dict) else {}
        b = _safe_int(data.get("protection_refresh_budget_blocked_count"), 0)
        f = _safe_int(data.get("protection_refresh_budget_force_override_count"), 0)
        sb = _safe_int(data.get("protection_refresh_stop_budget_blocked_count"), 0)
        tb = _safe_int(data.get("protection_refresh_tp_budget_blocked_count"), 0)
        sf = _safe_int(data.get("protection_refresh_stop_force_override_count"), 0)
        tf = _safe_int(data.get("protection_refresh_tp_force_override_count"), 0)
        prev_blocked = latest_blocked
        prev_force = latest_force
        latest_blocked, latest_force = b, f
        latest_stop_blocked, latest_tp_blocked = sb, tb
        latest_stop_force, latest_tp_force = sf, tf
        blocked_total = max(blocked_total, b)
        force_total = max(force_total, f)
        stop_blocked_total = max(stop_blocked_total, sb)
        tp_blocked_total = max(tp_blocked_total, tb)
        stop_force_total = max(stop_force_total, sf)
        tp_force_total = max(tp_force_total, tf)
    if blocked_total <= 0 and force_total <= 0 and latest_blocked <= 0 and latest_force <= 0:
        return "", 0, 0
    lines = [
        "Protection refresh budget:",
        (
            f"- latest blocked={int(latest_blocked)} (prev {int(prev_blocked)}) "
            f"(stop={int(latest_stop_blocked)} tp={int(latest_tp_blocked)})"
        ),
        (
            f"- latest force_override={int(latest_force)} (prev {int(prev_force)}) "
            f"(stop={int(latest_stop_force)} tp={int(latest_tp_force)})"
        ),
        f"- peak blocked={int(blocked_total)} force_override={int(force_total)}",
    ]
    return "\n".join(lines), int(latest_blocked), int(latest_force)


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


def _replay_snippet(telemetry_path: Path, journal_path: Path) -> str:
    if not journal_path.exists():
        return ""
    try:
        from tools import replay_trade  # type: ignore
    except Exception:
        try:
            from eclipse_scalper.tools import replay_trade  # type: ignore
        except Exception:
            return ""
    events = _load_jsonl(telemetry_path)
    corr, sym = _latest_corr_or_symbol(events)
    try:
        out = replay_trade.replay(journal_path, correlation_id=corr, symbol=("" if corr else sym))
    except Exception:
        return ""
    transitions = out.get("transitions", []) if isinstance(out, dict) else []
    if not transitions:
        return ""
    tail = transitions[-3:]
    lines = [
        "Execution replay:",
        f"- filter correlation_id={corr or 'n/a'} symbol={sym or 'n/a'}",
        f"- last_state={str(out.get('last_state') or '')}",
    ]
    for tr in tail:
        lines.append(f"- {tr.get('from')}->{tr.get('to')} ({tr.get('reason')})")
    return "\n".join(lines)


def _run_dashboard(
    path: Path,
    limit: int,
    codes_per_symbol: bool,
    codes_top: int,
    guard_events: bool,
    guard_history: bool,
    guard_history_path: Path,
) -> tuple[str, int]:
    cmd = [
        sys.executable or "python",
        str(TELEMETRY_SCRIPT),
        "--path",
        str(path),
        "--limit",
        str(limit),
    ]
    if codes_per_symbol:
        cmd.append("--codes-per-symbol")
        cmd.extend(["--codes-top", str(codes_top)])
    if guard_events:
        cmd.append("--guard-events")
    if guard_history:
        cmd.append("--guard-history")
        cmd.extend(["--guard-history-path", str(guard_history_path)])

    result = subprocess.run(cmd, capture_output=True, text=True)
    stdout = result.stdout.strip()
    stderr = result.stderr.strip()
    if stdout:
        print(stdout)
    if stderr:
        print(stderr, file=sys.stderr)
    return stdout, result.returncode


def _load_notify_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return raw if isinstance(raw, dict) else {}
    except Exception:
        return {}


def _save_notify_state(path: Path, state: dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(state, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    except Exception:
        pass


def _is_worsened(prev: dict[str, Any], curr: dict[str, Any]) -> bool:
    if float(curr.get("reliability_coverage", 1.0) or 1.0) < float(prev.get("reliability_coverage", 1.0) or 1.0):
        return True
    keys_higher_is_worse = (
        "reliability_mismatch",
        "reliability_invalid",
        "reliability_cat_ledger",
        "reliability_cat_transition",
        "reliability_cat_belief",
        "reliability_cat_position",
        "reliability_cat_orphan",
        "reliability_cat_coverage_gap",
        "reliability_cat_replace_race",
        "reliability_cat_contradiction",
        "reliability_cat_unknown",
        "reconcile_gate_count",
        "reconcile_gate_max_severity",
        "reconcile_gate_max_streak",
        "recovery_red_lock_streak",
        "entry_budget_depleted",
        "replace_envelope_count",
        "protection_refresh_budget_blocked_count",
        "protection_refresh_budget_force_override_count",
    )
    for key in keys_higher_is_worse:
        if float(curr.get(key, 0.0) or 0.0) > float(prev.get(key, 0.0) or 0.0):
            return True
    return False


def _decide_notify(prev: dict[str, Any], curr: dict[str, Any]) -> tuple[bool, str]:
    if not prev:
        return True, "initial_state"
    prev_level = str(prev.get("level") or "normal")
    curr_level = str(curr.get("level") or "normal")
    if prev_level != curr_level:
        return True, f"level_transition:{prev_level}->{curr_level}"
    if curr_level == "critical" and _is_worsened(prev, curr):
        return True, "critical_worsened"
    if curr_level == "critical":
        return False, "critical_unchanged"
    return False, "normal_unchanged"


def _attach_prev_snapshot(curr: dict[str, Any], prev: dict[str, Any]) -> dict[str, Any]:
    out = dict(curr)
    if not prev:
        return out
    tracked = (
        "level",
        "reliability_mismatch",
        "reliability_invalid",
        "reliability_coverage",
        "reliability_cat_ledger",
        "reliability_cat_transition",
        "reliability_cat_belief",
        "reliability_cat_position",
        "reliability_cat_orphan",
        "reliability_cat_coverage_gap",
        "reliability_cat_replace_race",
        "reliability_cat_contradiction",
        "reliability_cat_unknown",
        "reconcile_gate_count",
        "reconcile_gate_max_severity",
        "reconcile_gate_max_streak",
        "entry_budget_depleted",
        "replace_envelope_count",
        "protection_refresh_budget_blocked_count",
        "protection_refresh_budget_force_override_count",
    )
    for key in tracked:
        if key in prev:
            out[f"previous_{key}"] = prev.get(key)
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Telemetry dashboard notifier")
    parser.add_argument("--path", default="logs/telemetry.jsonl", help="Telemetry JSONL path")
    parser.add_argument("--limit", type=int, default=100, help="Max events to load")
    parser.add_argument("--codes-per-symbol", action="store_true", help="Include per-symbol code summary")
    parser.add_argument("--codes-top", type=int, default=4, help="Top per-symbol codes when enabled")
    parser.add_argument("--guard-events", action="store_true", help="Include guard telemetry counts")
    parser.add_argument("--guard-history", action="store_true", help="Include guard history rows")
    parser.add_argument("--journal", default="logs/execution_journal.jsonl", help="Execution journal JSONL path")
    parser.add_argument("--reliability-gate", default="logs/reliability_gate.txt", help="Reliability gate summary path")
    parser.add_argument("--reliability-max-replay-mismatch", type=int, default=0)
    parser.add_argument("--reliability-max-invalid-transitions", type=int, default=0)
    parser.add_argument("--reliability-min-journal-coverage", type=float, default=0.90)
    parser.add_argument("--reconcile-first-gate-critical-threshold", type=int, default=1)
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
    parser.add_argument("--entry-budget-depleted-critical-threshold", type=int, default=1)
    parser.add_argument("--replace-envelope-critical-threshold", type=int, default=1)
    parser.add_argument(
        "--protection-refresh-budget-blocked-critical-threshold",
        type=int,
        default=int(os.getenv("PROTECTION_REFRESH_BUDGET_BLOCKED_CRITICAL_THRESHOLD", "2")),
    )
    parser.add_argument(
        "--protection-refresh-force-override-critical-threshold",
        type=int,
        default=int(os.getenv("PROTECTION_REFRESH_FORCE_OVERRIDE_CRITICAL_THRESHOLD", "2")),
    )
    parser.add_argument(
        "--recovery-red-lock-critical-streak",
        type=int,
        default=int(os.getenv("RECOVERY_RED_LOCK_CRITICAL_STREAK", "2")),
        help="Force critical level when latest recovery stage is RED_LOCK for this many consecutive runs",
    )
    parser.add_argument(
        "--guard-history-path",
        default=os.getenv("TELEMETRY_GUARD_HISTORY_PATH", "logs/telemetry_guard_history.csv"),
        help="CSV path used when --guard-history is set",
    )
    parser.add_argument(
        "--state-path",
        default=os.getenv("TELEMETRY_DASHBOARD_NOTIFY_STATE_PATH", "logs/telemetry_dashboard_notify_state.json"),
        help="Path to persisted notify state used for transition-based dedup",
    )
    parser.add_argument("--no-notify", action="store_true", help="Skip sending the Telegram notification")
    args = parser.parse_args(argv)

    telemetry_path = Path(args.path)
    journal_path = Path(args.journal)
    reliability_gate_path = Path(args.reliability_gate)
    guard_history_path = Path(args.guard_history_path)
    stdout, rc = _run_dashboard(
        telemetry_path,
        max(1, args.limit),
        args.codes_per_symbol,
        max(1, args.codes_top),
        args.guard_events,
        args.guard_history,
        guard_history_path,
    )
    notifier = None if args.no_notify else _build_notifier()
    if stdout:
        header = f"Telemetry snapshot: {telemetry_path.name}"
        belief, belief_latest = _belief_state_snippet(telemetry_path)
        recovery_stage, recovery_latest = _recovery_stage_snippet(telemetry_path)
        reconcile_gate, reconcile_gate_count, reconcile_gate_max_severity, reconcile_gate_max_streak = _reconcile_first_gate_snippet(
            telemetry_path,
            severity_threshold=max(0.0, float(args.reconcile_first_gate_severity_threshold)),
        )
        entry_budget, entry_budget_depleted = _entry_budget_pressure_snippet(telemetry_path)
        replace_envelope, replace_envelope_count = _replace_envelope_snippet(telemetry_path)
        refresh_budget, refresh_budget_blocked_count, refresh_budget_force_count = _protection_refresh_budget_snippet(
            telemetry_path
        )
        replay = _replay_snippet(telemetry_path, journal_path)
        reliability, reliability_degraded, reliability_metrics = _reliability_gate_snippet(
            reliability_gate_path,
            max_replay_mismatch=max(0, int(args.reliability_max_replay_mismatch)),
            max_invalid_transitions=max(0, int(args.reliability_max_invalid_transitions)),
            min_journal_coverage=max(0.0, min(1.0, float(args.reliability_min_journal_coverage))),
        )
        reconcile_gate_degraded = bool(
            reconcile_gate_count >= max(1, int(args.reconcile_first_gate_critical_threshold))
        )
        reconcile_gate_severity_degraded = bool(
            reconcile_gate_max_severity >= max(0.0, float(args.reconcile_first_gate_severity_threshold))
        )
        reconcile_gate_streak_degraded = bool(
            reconcile_gate_max_streak >= max(1, int(args.reconcile_first_gate_severity_streak_threshold))
        )
        entry_budget_degraded = bool(
            entry_budget_depleted >= max(1, int(args.entry_budget_depleted_critical_threshold))
        )
        replace_envelope_degraded = bool(
            replace_envelope_count >= max(1, int(args.replace_envelope_critical_threshold))
        )
        refresh_budget_blocked_degraded = bool(
            int(refresh_budget_blocked_count)
            >= max(1, int(args.protection_refresh_budget_blocked_critical_threshold))
        )
        refresh_budget_force_degraded = bool(
            int(refresh_budget_force_count)
            >= max(1, int(args.protection_refresh_force_override_critical_threshold))
        )
        merged = stdout
        if belief:
            merged = f"{merged}\n\n{belief}"
        if recovery_stage:
            merged = f"{merged}\n\n{recovery_stage}"
        if reconcile_gate:
            merged = f"{merged}\n\n{reconcile_gate}"
        if entry_budget:
            merged = f"{merged}\n\n{entry_budget}"
        if replace_envelope:
            merged = f"{merged}\n\n{replace_envelope}"
        if refresh_budget:
            merged = f"{merged}\n\n{refresh_budget}"
        if replay:
            merged = f"{merged}\n\n{replay}"
        if reliability:
            merged = f"{merged}\n\n{reliability}"
        max_body_len = max(0, 3600 - len(header))
        body = _truncate(merged, max_len=max_body_len)
        current_state = {
            "level": (
                "critical"
                if (
                    reliability_degraded
                    or reconcile_gate_degraded
                    or reconcile_gate_severity_degraded
                    or reconcile_gate_streak_degraded
                    or entry_budget_degraded
                    or replace_envelope_degraded
                    or refresh_budget_blocked_degraded
                    or refresh_budget_force_degraded
                )
                else "normal"
            ),
            "reliability_mismatch": int(_safe_float(reliability_metrics.get("replay_mismatch_count"), 0.0)),
            "reliability_invalid": int(_safe_float(reliability_metrics.get("invalid_transition_count"), 0.0)),
            "reliability_coverage": float(_safe_float(reliability_metrics.get("journal_coverage_ratio"), 1.0)),
            "reliability_cat_ledger": int(_safe_float(reliability_metrics.get("replay_mismatch_cat_ledger"), 0.0)),
            "reliability_cat_transition": int(_safe_float(reliability_metrics.get("replay_mismatch_cat_transition"), 0.0)),
            "reliability_cat_belief": int(_safe_float(reliability_metrics.get("replay_mismatch_cat_belief"), 0.0)),
            "reliability_cat_position": int(_safe_float(reliability_metrics.get("replay_mismatch_cat_position"), 0.0)),
            "reliability_cat_orphan": int(_safe_float(reliability_metrics.get("replay_mismatch_cat_orphan"), 0.0)),
            "reliability_cat_coverage_gap": int(
                _safe_float(reliability_metrics.get("replay_mismatch_cat_coverage_gap"), 0.0)
            ),
            "reliability_cat_replace_race": int(
                _safe_float(reliability_metrics.get("replay_mismatch_cat_replace_race"), 0.0)
            ),
            "reliability_cat_contradiction": int(
                _safe_float(reliability_metrics.get("replay_mismatch_cat_contradiction"), 0.0)
            ),
            "reliability_cat_unknown": int(_safe_float(reliability_metrics.get("replay_mismatch_cat_unknown"), 0.0)),
            "reconcile_gate_count": int(reconcile_gate_count),
            "reconcile_gate_max_severity": float(reconcile_gate_max_severity),
            "reconcile_gate_max_streak": int(reconcile_gate_max_streak),
            "recovery_stage_latest": str(recovery_latest or ""),
            "entry_budget_depleted": int(entry_budget_depleted),
            "replace_envelope_count": int(replace_envelope_count),
            "protection_refresh_budget_blocked_count": int(refresh_budget_blocked_count),
            "protection_refresh_budget_force_override_count": int(refresh_budget_force_count),
            "belief_allow_entries_latest": bool(belief_latest.get("allow_entries", True)),
            "belief_guard_mode_latest": str(belief_latest.get("guard_mode") or ""),
            "belief_reason_latest": str(belief_latest.get("reason") or ""),
        }
        current_state.update(
            _extract_unlock_metrics(
                str(belief_latest.get("unlock") or ""),
                _safe_float(belief_latest.get("next_unlock_sec"), 0.0),
            )
        )
        state_path = Path(args.state_path)
        prev_state = _load_notify_state(state_path)
        prev_red_streak = int(_safe_float(prev_state.get("recovery_red_lock_streak", 0), 0.0)) if isinstance(prev_state, dict) else 0
        if str(current_state.get("recovery_stage_latest") or "").upper() == "RED_LOCK":
            red_lock_streak = prev_red_streak + 1
        else:
            red_lock_streak = 0
        current_state["recovery_red_lock_streak"] = int(red_lock_streak)
        recovery_red_lock_degraded = bool(
            int(red_lock_streak) >= max(1, int(args.recovery_red_lock_critical_streak))
        )
        if recovery_red_lock_degraded:
            current_state["level"] = "critical"
        should_send, decision_reason = _decide_notify(prev_state, current_state)
        current_state = _attach_prev_snapshot(current_state, prev_state)
        current_state["last_decision_reason"] = decision_reason
        current_state["last_decision_sent"] = bool(should_send)
        current_state["updated_at"] = float(time.time())
        print(
            "Notify decision: "
            f"{'send' if should_send else 'skip'} "
            f"reason={decision_reason} "
            f"prev={str(prev_state.get('level') or 'none')} "
            f"curr={str(current_state.get('level') or 'none')}"
        )
        if should_send and notifier:
            _send_alert(notifier, f"{header}\n{body}", level=str(current_state["level"]))
        _save_notify_state(state_path, current_state)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
