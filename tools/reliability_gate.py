#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from execution import state_machine as sm  # type: ignore
except Exception:
    try:
        from eclipse_scalper.execution import state_machine as sm  # type: ignore
    except Exception:
        sm = None  # type: ignore


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        if v != v:
            return default
        return v
    except Exception:
        return default


def _load_jsonl(path: Path) -> list[dict]:
    out: list[dict] = []
    if not path.exists():
        return out
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        s = str(raw or "").strip()
        if not s:
            continue
        try:
            obj = json.loads(s)
            if isinstance(obj, dict):
                out.append(obj)
        except Exception:
            continue
    return out


def _event_ts(ev: dict) -> float:
    try:
        v = _safe_float(ev.get("ts"), 0.0)
        if v > 0.0:
            return v
        data = ev.get("data") if isinstance(ev.get("data"), dict) else {}
        v = _safe_float(data.get("ts"), 0.0)
        if v > 0.0:
            return v
        v = _safe_float(ev.get("timestamp"), 0.0)
        if v > 0.0:
            return (v / 1000.0) if v > 1e12 else v
        v = _safe_float(data.get("timestamp"), 0.0)
        if v > 0.0:
            return (v / 1000.0) if v > 1e12 else v
    except Exception:
        return 0.0
    return 0.0


def _apply_window(events: list[dict], window_seconds: float, now_ts: float = 0.0) -> list[dict]:
    w = _safe_float(window_seconds, 0.0)
    if w <= 0.0:
        return list(events)
    ref = _safe_float(now_ts, 0.0)
    if ref <= 0.0:
        for ev in events:
            t = _event_ts(ev)
            if t > ref:
                ref = t
    if ref <= 0.0:
        return []
    floor = ref - w
    out: list[dict] = []
    for ev in events:
        t = _event_ts(ev)
        if t <= 0.0:
            continue
        if t >= floor:
            out.append(ev)
    return out


def _corr_from_event(ev: dict) -> str:
    data = ev.get("data") if isinstance(ev.get("data"), dict) else {}
    corr = str(data.get("correlation_id") or ev.get("correlation_id") or "").strip()
    if corr:
        return corr
    ent = str(data.get("entity") or "").strip()
    return ent


def _is_orderish_event(ev: dict) -> bool:
    event = str(ev.get("event") or "").strip().lower()
    if event.startswith("order."):
        return True
    if event in ("state.transition",):
        return True
    return False


def _journal_corr_ids(journal_events: list[dict]) -> set[str]:
    ids: set[str] = set()
    for ev in journal_events:
        name = str(ev.get("event") or "").strip()
        data = ev.get("data") if isinstance(ev.get("data"), dict) else {}
        if name == "state.transition":
            corr = _corr_from_event(ev)
            if corr:
                ids.add(corr)
            continue
        if name == "intent.ledger":
            corr = str(data.get("intent_id") or "").strip()
            if corr:
                ids.add(corr)
    return ids


def _categorize_mismatch_ids(ids: list[str]) -> Dict[str, int]:
    out: Dict[str, int] = {
        "ledger": 0,
        "transition": 0,
        "belief": 0,
        "position": 0,
        "orphan": 0,
        "coverage_gap": 0,
        "stage1_protection_fail": 0,
        "replace_race": 0,
        "contradiction": 0,
        "unknown": 0,
    }
    for raw in ids:
        cid = str(raw or "").strip().lower()
        if not cid:
            continue
        if cid.startswith("led-") or "ledger" in cid:
            out["ledger"] += 1
        elif cid.startswith("trn-") or "transition" in cid:
            out["transition"] += 1
        elif cid.startswith("blf-") or "belief" in cid:
            out["belief"] += 1
        elif cid.startswith("pos-") or "position" in cid:
            out["position"] += 1
        elif "orphan" in cid:
            out["orphan"] += 1
        elif "coverage" in cid or "protect" in cid:
            out["coverage_gap"] += 1
        elif "replace" in cid:
            out["replace_race"] += 1
        elif "contrad" in cid:
            out["contradiction"] += 1
        else:
            out["unknown"] += 1
    return out


def _ignore_corr_id(corr_id: str, ignore_tokens: list[str]) -> bool:
    cid = str(corr_id or "").strip().upper()
    if not cid:
        return True
    for tok in ignore_tokens:
        t = str(tok or "").strip().upper()
        if not t:
            continue
        if t in cid:
            return True
    return False


def _resolve_ignore_tokens(raw: str = "") -> list[str]:
    # Keep default narrow: only explicit synthetic restart marker.
    base = ["RESTART-UNK"]
    txt = str(raw or "").strip()
    if not txt:
        txt = str(os.getenv("RELIABILITY_GATE_IGNORE_CORR_TOKENS", "") or "").strip()
    if not txt:
        return base
    extra = [x.strip() for x in txt.replace(";", ",").split(",") if str(x or "").strip()]
    return base + extra


def _telemetry_category_counts(telemetry_events: list[dict]) -> Dict[str, float]:
    out: Dict[str, float] = {
        "position_latest": 0.0,
        "position_peak": 0.0,
        "orphan": 0.0,
        "intent_collision": 0.0,
        "coverage_gap": 0.0,
        "stage1_protection_fail": 0.0,
        "replace_race": 0.0,
        "contradiction": 0.0,
        "coverage_gap_seconds_latest": 0.0,
        "coverage_gap_seconds_peak": 0.0,
    }
    replace_keys: set[str] = set()
    for ev in telemetry_events:
        name = str(ev.get("event") or "").strip().lower()
        data = ev.get("data") if isinstance(ev.get("data"), dict) else {}
        symbol = str(ev.get("symbol") or data.get("symbol") or "").strip().upper()
        corr = str(data.get("correlation_id") or ev.get("correlation_id") or "").strip()

        if name == "rebuild.orphan_decision":
            out["orphan"] += 1.0
            if bool(data.get("intent_collision", False)):
                out["intent_collision"] += 1.0

        if name == "rebuild.summary":
            out["intent_collision"] += max(0.0, _safe_float(data.get("intent_collision_count"), 0.0))

        if name in ("order.replace_envelope_block", "order.replace_reconcile_required"):
            key = corr or f"{symbol}:{name}"
            if key and key not in replace_keys:
                replace_keys.add(key)
                out["replace_race"] += 1.0

        if name == "execution.belief_state":
            mismatch_streak = _safe_float(data.get("mismatch_streak"), 0.0)
            debt_symbols = _safe_float(data.get("belief_debt_symbols"), 0.0)
            contradiction_score = _safe_float(data.get("evidence_contradiction_score"), 0.0)
            contradiction_streak = _safe_float(data.get("evidence_contradiction_streak"), 0.0)
            pos_now = float(max(0.0, debt_symbols if mismatch_streak > 0.0 else 0.0))
            out["position_latest"] = pos_now
            if pos_now > float(out.get("position_peak", 0.0)):
                out["position_peak"] = pos_now
            if contradiction_score >= 0.60 or contradiction_streak > 0.0:
                out["contradiction"] += 1.0

        if name == "entry.stage1_gap_assertion":
            placed = bool(data.get("placed", False))
            gap_active = bool(data.get("gap_active", False))
            ttl_breached = bool(data.get("ttl_breached", False))
            if (not placed) or gap_active or ttl_breached:
                out["stage1_protection_fail"] += 1.0

        has_cov_field = any(
            key in data for key in ("coverage_gap_seconds", "coverage_gap_sec", "protection_coverage_gap_seconds")
        )
        if has_cov_field:
            coverage_gap_sec = max(
                _safe_float(data.get("coverage_gap_seconds"), 0.0),
                _safe_float(data.get("coverage_gap_sec"), 0.0),
                _safe_float(data.get("protection_coverage_gap_seconds"), 0.0),
            )
            out["coverage_gap_seconds_latest"] = float(max(0.0, coverage_gap_sec))
            if coverage_gap_sec > float(out.get("coverage_gap_seconds_peak", 0.0)):
                out["coverage_gap_seconds_peak"] = float(coverage_gap_sec)
            if coverage_gap_sec > 0.0:
                out["coverage_gap"] += 1.0
    return out


def _telemetry_corr_ids(telemetry_events: list[dict]) -> set[str]:
    ids: set[str] = set()
    for ev in telemetry_events:
        if not _is_orderish_event(ev):
            continue
        corr = _corr_from_event(ev)
        if corr:
            ids.add(corr)
    return ids


def _invalid_transition_count(journal_events: list[dict]) -> int:
    if sm is None:
        return 0
    bad = 0
    for ev in journal_events:
        if str(ev.get("event") or "") != "state.transition":
            continue
        data = ev.get("data") if isinstance(ev.get("data"), dict) else {}
        machine = str(data.get("machine") or "").strip().lower()
        frm = str(data.get("state_from") or "").strip().upper()
        to = str(data.get("state_to") or "").strip().upper()
        if not machine or not frm or not to:
            continue
        try:
            if machine == "order_intent":
                ok = bool(sm.is_valid_transition(sm.MachineKind.ORDER_INTENT, frm, to))
            elif machine == "position_belief":
                ok = bool(sm.is_valid_transition(sm.MachineKind.POSITION_BELIEF, frm, to))
            else:
                ok = True
        except Exception:
            ok = True
        if not ok:
            bad += 1
    return int(bad)


def build_report(
    telemetry_events: list[dict],
    journal_events: list[dict],
    *,
    window_seconds: float = 0.0,
    now_ts: float = 0.0,
    ignore_corr_tokens: Optional[list[str]] = None,
) -> Dict[str, Any]:
    telemetry_events = _apply_window(telemetry_events, window_seconds, now_ts=now_ts)
    journal_events = _apply_window(journal_events, window_seconds, now_ts=now_ts)
    ignore_tokens = list(ignore_corr_tokens or [])
    tele_corr_raw = _telemetry_corr_ids(telemetry_events)
    tele_corr = {c for c in tele_corr_raw if not _ignore_corr_id(c, ignore_tokens)}
    journ_corr = _journal_corr_ids(journal_events)
    missing = sorted([c for c in tele_corr if c not in journ_corr])
    categories = _categorize_mismatch_ids(missing)
    telemetry_cats = _telemetry_category_counts(telemetry_events)
    categories["position"] = int(
        max(
            int(categories.get("position", 0) or 0),
            int(_safe_float(telemetry_cats.get("position_latest"), 0.0) or 0.0),
        )
    )
    categories["orphan"] = int(categories.get("orphan", 0) + int(_safe_float(telemetry_cats.get("orphan"), 0.0)))
    categories["coverage_gap"] = int(
        categories.get("coverage_gap", 0) + int(_safe_float(telemetry_cats.get("coverage_gap"), 0.0))
    )
    categories["stage1_protection_fail"] = int(
        categories.get("stage1_protection_fail", 0)
        + int(_safe_float(telemetry_cats.get("stage1_protection_fail"), 0.0))
    )
    categories["replace_race"] = int(
        categories.get("replace_race", 0) + int(_safe_float(telemetry_cats.get("replace_race"), 0.0))
    )
    categories["contradiction"] = int(
        categories.get("contradiction", 0) + int(_safe_float(telemetry_cats.get("contradiction"), 0.0))
    )
    cov = 1.0
    if tele_corr:
        cov = 1.0 - (float(len(missing)) / float(len(tele_corr)))
    invalid = _invalid_transition_count(journal_events)
    return {
        "telemetry_corr_ids_raw": int(len(tele_corr_raw)),
        "telemetry_corr_ids_ignored": int(max(0, len(tele_corr_raw) - len(tele_corr))),
        "telemetry_corr_ids": int(len(tele_corr)),
        "journal_corr_ids": int(len(journ_corr)),
        "replay_mismatch_count": int(len(missing)),
        "journal_coverage_ratio": float(max(0.0, min(1.0, cov))),
        "invalid_transition_count": int(invalid),
        "replay_mismatch_ids": missing[:20],
        "replay_mismatch_categories": categories,
        "position_mismatch_count": int(categories.get("position", 0) or 0),
        "position_mismatch_count_peak": int(_safe_float(telemetry_cats.get("position_peak"), 0.0) or 0.0),
        "orphan_count": int(categories.get("orphan", 0) or 0),
        "intent_collision_count": int(max(0.0, _safe_float(telemetry_cats.get("intent_collision"), 0.0))),
        "protection_coverage_gap_seconds": float(_safe_float(telemetry_cats.get("coverage_gap_seconds_latest"), 0.0)),
        "protection_coverage_gap_seconds_peak": float(_safe_float(telemetry_cats.get("coverage_gap_seconds_peak"), 0.0)),
        "stage1_protection_fail_count": int(categories.get("stage1_protection_fail", 0) or 0),
        "replace_race_count": int(categories.get("replace_race", 0) or 0),
        "evidence_contradiction_count": int(categories.get("contradiction", 0) or 0),
    }


def _render(report: Dict[str, Any], *, telemetry_path: Path, journal_path: Path) -> str:
    lines = [
        "Execution Reliability Gate",
        "==========================",
        f"telemetry: {telemetry_path}",
        f"journal: {journal_path}",
        (
        f"window_seconds={float(report.get('window_seconds', 0.0) or 0.0):.1f}"
            if float(report.get("window_seconds", 0.0) or 0.0) > 0.0
            else "window_seconds=all"
        ),
        f"telemetry_corr_ids_raw={int(report.get('telemetry_corr_ids_raw', report.get('telemetry_corr_ids', 0)) or 0)}",
        f"telemetry_corr_ids_ignored={int(report.get('telemetry_corr_ids_ignored', 0) or 0)}",
        f"telemetry_corr_ids={int(report.get('telemetry_corr_ids', 0) or 0)}",
        f"journal_corr_ids={int(report.get('journal_corr_ids', 0) or 0)}",
        f"replay_mismatch_count={int(report.get('replay_mismatch_count', 0) or 0)}",
        f"journal_coverage_ratio={float(report.get('journal_coverage_ratio', 0.0) or 0.0):.3f}",
        f"invalid_transition_count={int(report.get('invalid_transition_count', 0) or 0)}",
        f"position_mismatch_count={int(report.get('position_mismatch_count', 0) or 0)}",
        f"position_mismatch_count_peak={int(report.get('position_mismatch_count_peak', 0) or 0)}",
        f"orphan_count={int(report.get('orphan_count', 0) or 0)}",
        f"intent_collision_count={int(report.get('intent_collision_count', 0) or 0)}",
        f"protection_coverage_gap_seconds={float(report.get('protection_coverage_gap_seconds', 0.0) or 0.0):.1f}",
        f"protection_coverage_gap_seconds_peak={float(report.get('protection_coverage_gap_seconds_peak', 0.0) or 0.0):.1f}",
        f"stage1_protection_fail_count={int(report.get('stage1_protection_fail_count', 0) or 0)}",
        f"replace_race_count={int(report.get('replace_race_count', 0) or 0)}",
        f"evidence_contradiction_count={int(report.get('evidence_contradiction_count', 0) or 0)}",
    ]
    categories = report.get("replay_mismatch_categories")
    if isinstance(categories, dict) and categories:
        normalized = {
            "ledger": int(categories.get("ledger", 0) or 0),
            "transition": int(categories.get("transition", 0) or 0),
            "belief": int(categories.get("belief", 0) or 0),
            "position": int(categories.get("position", 0) or 0),
            "orphan": int(categories.get("orphan", 0) or 0),
            "coverage_gap": int(categories.get("coverage_gap", 0) or 0),
            "stage1_protection_fail": int(categories.get("stage1_protection_fail", 0) or 0),
            "replace_race": int(categories.get("replace_race", 0) or 0),
            "contradiction": int(categories.get("contradiction", 0) or 0),
            "unknown": int(categories.get("unknown", 0) or 0),
        }
        lines.append(
            "replay_mismatch_categories="
            + json.dumps(
                normalized,
                ensure_ascii=True,
                separators=(",", ":"),
            )
        )
        ranked = sorted(
            [(str(k), int(v)) for (k, v) in normalized.items() if int(v) > 0],
            key=lambda kv: int(kv[1]),
            reverse=True,
        )[:3]
        if ranked:
            lines.append("top_contributors=" + ",".join(f"{k}:{v}" for (k, v) in ranked))
        critical_keys = ("position", "orphan", "coverage_gap", "stage1_protection_fail", "replace_race", "contradiction")
        critical_ranked = sorted(
            [(str(k), int(normalized.get(k, 0))) for k in critical_keys if int(normalized.get(k, 0)) > 0],
            key=lambda kv: int(kv[1]),
            reverse=True,
        )[:3]
        if critical_ranked:
            lines.append("critical_contributors=" + ",".join(f"{k}:{v}" for (k, v) in critical_ranked))
    missing = report.get("replay_mismatch_ids")
    if isinstance(missing, list) and missing:
        lines.append("replay_mismatch_ids:")
        for cid in missing:
            lines.append(f"- {cid}")
    return "\n".join(lines) + "\n"


def _passes(
    report: Dict[str, Any],
    *,
    max_replay_mismatch: int,
    max_invalid_transitions: int,
    min_journal_coverage: float,
    max_intent_collision_count: int,
    max_stage1_protection_fail_count: int,
) -> bool:
    if int(report.get("replay_mismatch_count", 0) or 0) > int(max_replay_mismatch):
        return False
    if int(report.get("invalid_transition_count", 0) or 0) > int(max_invalid_transitions):
        return False
    if float(report.get("journal_coverage_ratio", 0.0) or 0.0) < float(min_journal_coverage):
        return False
    if int(report.get("intent_collision_count", 0) or 0) > int(max_intent_collision_count):
        return False
    if int(report.get("stage1_protection_fail_count", 0) or 0) > int(max_stage1_protection_fail_count):
        return False
    return True


def main(argv: Optional[list[str]] = None) -> int:
    env_max_intent_collision = int(max(0, int(os.getenv("RELIABILITY_GATE_MAX_INTENT_COLLISION_COUNT", "0") or 0)))
    env_max_stage1_fail = int(max(0, int(os.getenv("RELIABILITY_GATE_MAX_STAGE1_PROTECTION_FAIL_COUNT", "0") or 0)))
    ap = argparse.ArgumentParser(description="Compute execution replay mismatch / invariant gate metrics")
    ap.add_argument("--telemetry", default="logs/telemetry.jsonl")
    ap.add_argument("--journal", default="logs/execution_journal.jsonl")
    ap.add_argument("--output", default="logs/reliability_gate.txt")
    ap.add_argument("--max-replay-mismatch", type=int, default=0)
    ap.add_argument("--max-invalid-transitions", type=int, default=0)
    ap.add_argument("--min-journal-coverage", type=float, default=0.90)
    ap.add_argument("--max-intent-collision-count", type=int, default=env_max_intent_collision)
    ap.add_argument("--max-stage1-protection-fail-count", type=int, default=env_max_stage1_fail)
    ap.add_argument(
        "--window-seconds",
        type=float,
        default=0.0,
        help="Use only events from the most recent N seconds (0 = full file)",
    )
    ap.add_argument(
        "--ignore-corr-tokens",
        default="",
        help="Comma-separated correlation-id tokens to ignore for coverage matching (default includes RESTART-UNK).",
    )
    ap.add_argument("--allow-missing", action="store_true", help="Return success when telemetry/journal files are missing")
    ap.add_argument("--enforce", action="store_true", help="Exit non-zero on threshold breach")
    args = ap.parse_args(argv)

    telemetry_path = Path(args.telemetry)
    journal_path = Path(args.journal)
    if (not telemetry_path.exists() or not journal_path.exists()) and bool(args.allow_missing):
        msg = (
            "Execution Reliability Gate\n"
            "==========================\n"
            f"telemetry: {telemetry_path}\n"
            f"journal: {journal_path}\n"
            "status: skipped (missing input file)\n"
        )
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(msg, encoding="utf-8")
        print(msg.strip())
        return 0

    telemetry_events = _load_jsonl(telemetry_path)
    journal_events = _load_jsonl(journal_path)
    report = build_report(
        telemetry_events,
        journal_events,
        window_seconds=max(0.0, float(args.window_seconds)),
        ignore_corr_tokens=_resolve_ignore_tokens(str(args.ignore_corr_tokens or "")),
    )
    report["window_seconds"] = float(max(0.0, float(args.window_seconds)))
    text = _render(report, telemetry_path=telemetry_path, journal_path=journal_path)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(text.strip())

    if bool(args.enforce):
        ok = _passes(
            report,
            max_replay_mismatch=max(0, int(args.max_replay_mismatch)),
            max_invalid_transitions=max(0, int(args.max_invalid_transitions)),
            min_journal_coverage=max(0.0, min(1.0, float(args.min_journal_coverage))),
            max_intent_collision_count=max(0, int(args.max_intent_collision_count)),
            max_stage1_protection_fail_count=max(0, int(args.max_stage1_protection_fail_count)),
        )
        return 0 if ok else 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
