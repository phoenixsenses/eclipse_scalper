#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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
        if str(ev.get("event") or "") != "state.transition":
            continue
        corr = _corr_from_event(ev)
        if corr:
            ids.add(corr)
    return ids


def _categorize_mismatch_ids(ids: list[str]) -> Dict[str, int]:
    out: Dict[str, int] = {
        "ledger": 0,
        "transition": 0,
        "belief": 0,
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
        else:
            out["unknown"] += 1
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


def build_report(telemetry_events: list[dict], journal_events: list[dict]) -> Dict[str, Any]:
    tele_corr = _telemetry_corr_ids(telemetry_events)
    journ_corr = _journal_corr_ids(journal_events)
    missing = sorted([c for c in tele_corr if c not in journ_corr])
    categories = _categorize_mismatch_ids(missing)
    cov = 1.0
    if tele_corr:
        cov = 1.0 - (float(len(missing)) / float(len(tele_corr)))
    invalid = _invalid_transition_count(journal_events)
    return {
        "telemetry_corr_ids": int(len(tele_corr)),
        "journal_corr_ids": int(len(journ_corr)),
        "replay_mismatch_count": int(len(missing)),
        "journal_coverage_ratio": float(max(0.0, min(1.0, cov))),
        "invalid_transition_count": int(invalid),
        "replay_mismatch_ids": missing[:20],
        "replay_mismatch_categories": categories,
    }


def _render(report: Dict[str, Any], *, telemetry_path: Path, journal_path: Path) -> str:
    lines = [
        "Execution Reliability Gate",
        "==========================",
        f"telemetry: {telemetry_path}",
        f"journal: {journal_path}",
        f"telemetry_corr_ids={int(report.get('telemetry_corr_ids', 0) or 0)}",
        f"journal_corr_ids={int(report.get('journal_corr_ids', 0) or 0)}",
        f"replay_mismatch_count={int(report.get('replay_mismatch_count', 0) or 0)}",
        f"journal_coverage_ratio={float(report.get('journal_coverage_ratio', 0.0) or 0.0):.3f}",
        f"invalid_transition_count={int(report.get('invalid_transition_count', 0) or 0)}",
    ]
    categories = report.get("replay_mismatch_categories")
    if isinstance(categories, dict) and categories:
        lines.append(
            "replay_mismatch_categories="
            + json.dumps(
                {
                    "ledger": int(categories.get("ledger", 0) or 0),
                    "transition": int(categories.get("transition", 0) or 0),
                    "belief": int(categories.get("belief", 0) or 0),
                    "unknown": int(categories.get("unknown", 0) or 0),
                },
                ensure_ascii=True,
                separators=(",", ":"),
            )
        )
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
) -> bool:
    if int(report.get("replay_mismatch_count", 0) or 0) > int(max_replay_mismatch):
        return False
    if int(report.get("invalid_transition_count", 0) or 0) > int(max_invalid_transitions):
        return False
    if float(report.get("journal_coverage_ratio", 0.0) or 0.0) < float(min_journal_coverage):
        return False
    return True


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Compute execution replay mismatch / invariant gate metrics")
    ap.add_argument("--telemetry", default="logs/telemetry.jsonl")
    ap.add_argument("--journal", default="logs/execution_journal.jsonl")
    ap.add_argument("--output", default="logs/reliability_gate.txt")
    ap.add_argument("--max-replay-mismatch", type=int, default=0)
    ap.add_argument("--max-invalid-transitions", type=int, default=0)
    ap.add_argument("--min-journal-coverage", type=float, default=0.90)
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
    report = build_report(telemetry_events, journal_events)
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
        )
        return 0 if ok else 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
