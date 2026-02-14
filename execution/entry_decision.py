from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


DecisionAction = str  # ALLOW | SCALE | DENY | DEFER


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        if v != v:  # NaN
            return default
        return v
    except Exception:
        return default


def _norm_action(x: Any) -> str:
    s = str(x or "").strip().lower()
    if s in ("buy", "sell"):
        return s
    return ""


def _stable_guard_fingerprint(guard_knobs: Dict[str, Any]) -> str:
    snap = {
        "mode": str(guard_knobs.get("mode") or ""),
        "allow_entries": bool(guard_knobs.get("allow_entries", True)),
        "min_entry_conf": _safe_float(guard_knobs.get("min_entry_conf"), 0.0),
        "entry_cooldown_seconds": _safe_float(guard_knobs.get("entry_cooldown_seconds"), 0.0),
        "max_notional_usdt": _safe_float(guard_knobs.get("max_notional_usdt"), 0.0),
        "runtime_gate_degraded": bool(guard_knobs.get("runtime_gate_degraded", False)),
        "reconcile_first_gate_degraded": bool(guard_knobs.get("reconcile_first_gate_degraded", False)),
        "runtime_gate_degrade_score": _safe_float(guard_knobs.get("runtime_gate_degrade_score"), 0.0),
    }
    blob = json.dumps(snap, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:12]


def rank_reasons(reasons: List[str]) -> List[str]:
    priority = {
        "guard_blocked": 0,
        "signal_missing": 1,
        "signal_action": 2,
        "confidence_low": 3,
        "sizing_missing": 4,
        "signal_limit_price": 5,
        "posture_changed": 6,
        "risk_in_position": 7,
        "cooldown_pending": 8,
    }
    uniq: List[str] = []
    seen = set()
    for r in reasons:
        rr = str(r or "").strip()
        if not rr or rr in seen:
            continue
        seen.add(rr)
        uniq.append(rr)
    return sorted(uniq, key=lambda r: (priority.get(r, 999), r))


@dataclass
class EntryDecisionRecord:
    symbol: str
    ts: float
    stage: str
    action: DecisionAction
    signal_action: str
    confidence: float
    min_confidence: float
    order_type: str
    amount: float
    price: float
    planned_notional: float
    guard_mode: str
    allow_entries: bool
    guard_fingerprint: str
    reasons: List[str] = field(default_factory=list)
    reason_primary: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "symbol": self.symbol,
            "ts": float(self.ts),
            "stage": self.stage,
            "decision": self.action,
            "signal_action": self.signal_action,
            "confidence": float(self.confidence),
            "min_confidence": float(self.min_confidence),
            "order_type": self.order_type,
            "amount": float(self.amount),
            "price": float(self.price),
            "planned_notional": float(self.planned_notional),
            "guard_mode": self.guard_mode,
            "allow_entries": bool(self.allow_entries),
            "guard_fingerprint": self.guard_fingerprint,
            "reasons": list(self.reasons),
            "reason_primary": self.reason_primary,
        }
        if self.meta:
            d["meta"] = dict(self.meta)
        return d


def compute_entry_decision(
    *,
    symbol: str,
    signal: Optional[Dict[str, Any]],
    guard_knobs: Dict[str, Any],
    min_confidence: float,
    amount: Optional[float],
    order_type: str,
    price: Optional[float],
    planned_notional: float = 0.0,
    stage: str = "propose",
    meta: Optional[Dict[str, Any]] = None,
) -> EntryDecisionRecord:
    reasons: List[str] = []

    allow_entries = bool(guard_knobs.get("allow_entries", True))
    guard_mode = str(guard_knobs.get("mode") or "").upper()
    guard_fp = _stable_guard_fingerprint(guard_knobs)

    if not allow_entries:
        reasons.append("guard_blocked")

    sig_action = _norm_action((signal or {}).get("action"))
    if not signal:
        reasons.append("signal_missing")
    elif not sig_action:
        reasons.append("signal_action")

    conf = _safe_float((signal or {}).get("confidence", (signal or {}).get("conf", 0.0)), 0.0)
    if _safe_float(min_confidence, 0.0) > 0 and conf < _safe_float(min_confidence, 0.0):
        reasons.append("confidence_low")

    amt = _safe_float(amount, 0.0)
    if amt <= 0:
        reasons.append("sizing_missing")

    otype = str(order_type or "").strip().lower() or "market"
    px = _safe_float(price, 0.0)
    if otype == "limit" and px <= 0:
        reasons.append("signal_limit_price")

    ranked = rank_reasons(reasons)
    if ranked:
        if "confidence_low" in ranked and len(ranked) == 1:
            dec = "SCALE"
        else:
            dec = "DENY"
    else:
        dec = "ALLOW"

    return EntryDecisionRecord(
        symbol=str(symbol or "").upper(),
        ts=time.time(),
        stage=str(stage),
        action=dec,
        signal_action=sig_action,
        confidence=float(conf),
        min_confidence=float(_safe_float(min_confidence, 0.0)),
        order_type=otype,
        amount=float(amt),
        price=float(px),
        planned_notional=float(_safe_float(planned_notional, 0.0)),
        guard_mode=guard_mode,
        allow_entries=allow_entries,
        guard_fingerprint=guard_fp,
        reasons=ranked,
        reason_primary=(ranked[0] if ranked else ""),
        meta=dict(meta or {}),
    )


def commit_entry_intent(
    record: EntryDecisionRecord,
    *,
    current_guard_knobs: Dict[str, Any],
    in_position_fn: Callable[[], bool],
    pending_fn: Callable[[], bool],
) -> Tuple[bool, EntryDecisionRecord]:
    reasons: List[str] = []
    if pending_fn():
        reasons.append("cooldown_pending")
    if in_position_fn():
        reasons.append("risk_in_position")

    current_fp = _stable_guard_fingerprint(current_guard_knobs or {})
    if current_fp != record.guard_fingerprint:
        reasons.append("posture_changed")

    allow_now = bool((current_guard_knobs or {}).get("allow_entries", True))
    if not allow_now:
        reasons.append("guard_blocked")

    ranked = rank_reasons(reasons)
    if ranked:
        nxt = EntryDecisionRecord(
            symbol=record.symbol,
            ts=time.time(),
            stage="commit",
            action="DEFER",
            signal_action=record.signal_action,
            confidence=record.confidence,
            min_confidence=record.min_confidence,
            order_type=record.order_type,
            amount=record.amount,
            price=record.price,
            planned_notional=record.planned_notional,
            guard_mode=str((current_guard_knobs or {}).get("mode") or "").upper(),
            allow_entries=allow_now,
            guard_fingerprint=current_fp,
            reasons=ranked,
            reason_primary=ranked[0],
            meta=dict(record.meta or {}),
        )
        return False, nxt

    ok = EntryDecisionRecord(
        symbol=record.symbol,
        ts=time.time(),
        stage="commit",
        action="ALLOW",
        signal_action=record.signal_action,
        confidence=record.confidence,
        min_confidence=record.min_confidence,
        order_type=record.order_type,
        amount=record.amount,
        price=record.price,
        planned_notional=record.planned_notional,
        guard_mode=str((current_guard_knobs or {}).get("mode") or "").upper(),
        allow_entries=allow_now,
        guard_fingerprint=current_fp,
        reasons=[],
        reason_primary="",
        meta=dict(record.meta or {}),
    )
    return True, ok
