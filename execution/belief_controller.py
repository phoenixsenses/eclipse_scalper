from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

from execution.guard_knobs import GuardKnobs


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
        if out != out:
            return default
        return out
    except Exception:
        return default


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _symkey(sym: str) -> str:
    s = str(sym or "").upper().strip()
    s = s.replace("/USDT:USDT", "USDT").replace("/USDT", "USDT")
    s = s.replace(":USDT", "USDT").replace(":", "")
    s = s.replace("/", "")
    if s.endswith("USDTUSDT"):
        s = s[:-4]
    return s


@dataclass
class DecisionTrace:
    mode: str
    debt_score: float
    debt_growth_per_min: float
    reason: str
    transition: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "debt_score": float(self.debt_score),
            "debt_growth_per_min": float(self.debt_growth_per_min),
            "reason": self.reason,
            "transition": self.transition,
        }


class BeliefController:
    """
    Policy engine that converts epistemic status into entry-risk posture.
    """

    _MODES = ("GREEN", "YELLOW", "ORANGE", "RED")

    def __init__(self, *, clock=time.time):
        self.clock = clock
        self.mode = "GREEN"
        self.mode_since = float(clock())
        self.last_debt_score = 0.0
        self.last_update = 0.0
        self._up_since: Dict[str, float] = {}
        self._down_since: Dict[str, float] = {}
        self._last_trace: Optional[DecisionTrace] = None
        self._runtime_gate_recover_until = 0.0
        self._post_red_warmup_until = 0.0

    def explain(self) -> DecisionTrace:
        if self._last_trace is not None:
            return self._last_trace
        return DecisionTrace(mode=self.mode, debt_score=0.0, debt_growth_per_min=0.0, reason="bootstrap")

    def allows_intent(self, intent: str) -> bool:
        # Safety invariant: exits are never blocked by belief controller posture.
        it = str(intent or "").strip().lower()
        if it in ("exit", "reduce_only", "protective_exit", "stop", "tp"):
            return True
        return True

    def _cfg(self, cfg: Any, name: str, default: float) -> float:
        try:
            return _safe_float(getattr(cfg, name, default), default)
        except Exception:
            return default

    def update(self, belief_state: Dict[str, Any], cfg: Any = None) -> GuardKnobs:
        now = float(self.clock())
        debt_sec = _safe_float(belief_state.get("belief_debt_sec", 0.0), 0.0)
        debt_symbols = max(0, int(_safe_float(belief_state.get("belief_debt_symbols", 0), 0.0)))
        mismatch_streak = max(0, int(_safe_float(belief_state.get("mismatch_streak", 0), 0.0)))
        evidence_confidence = _clamp(_safe_float(belief_state.get("evidence_confidence", 1.0), 1.0), 0.0, 1.0)
        evidence_degraded_sources = max(0, int(_safe_float(belief_state.get("evidence_degraded_sources", 0), 0.0)))
        evidence_contradiction_score = _clamp(
            _safe_float(belief_state.get("evidence_contradiction_score", 0.0), 0.0), 0.0, 1.0
        )
        evidence_contradiction_streak = max(
            0.0, _safe_float(belief_state.get("evidence_contradiction_streak", 0), 0.0)
        )
        runtime_gate_degraded = bool(belief_state.get("runtime_gate_degraded", False))
        runtime_gate_reason = str(belief_state.get("runtime_gate_reason", "") or "").strip()
        runtime_gate_mismatch_count = max(
            0.0, _safe_float(belief_state.get("runtime_gate_replay_mismatch_count", 0), 0.0)
        )
        runtime_gate_invalid_count = max(
            0.0, _safe_float(belief_state.get("runtime_gate_invalid_transition_count", 0), 0.0)
        )
        runtime_gate_cov = _clamp(
            _safe_float(belief_state.get("runtime_gate_journal_coverage_ratio", 1.0), 1.0), 0.0, 1.0
        )
        runtime_gate_degrade_score = _clamp(
            _safe_float(belief_state.get("runtime_gate_degrade_score", 0.0), 0.0), 0.0, 1.0
        )
        runtime_gate_cat_ledger = max(0.0, _safe_float(belief_state.get("runtime_gate_cat_ledger", 0), 0.0))
        runtime_gate_cat_transition = max(
            0.0, _safe_float(belief_state.get("runtime_gate_cat_transition", 0), 0.0)
        )
        runtime_gate_cat_belief = max(0.0, _safe_float(belief_state.get("runtime_gate_cat_belief", 0), 0.0))
        runtime_gate_cat_unknown = max(0.0, _safe_float(belief_state.get("runtime_gate_cat_unknown", 0), 0.0))
        reconcile_first_gate_count = max(
            0.0, _safe_float(belief_state.get("reconcile_first_gate_count", 0), 0.0)
        )
        reconcile_first_gate_max_severity = _clamp(
            _safe_float(belief_state.get("reconcile_first_gate_max_severity", 0.0), 0.0), 0.0, 1.0
        )
        reconcile_first_gate_max_streak = max(
            0.0, _safe_float(belief_state.get("reconcile_first_gate_max_streak", 0), 0.0)
        )
        reconcile_first_gate_last_reason = str(
            belief_state.get("reconcile_first_gate_last_reason", "") or ""
        ).strip()

        debt_ref = max(1.0, self._cfg(cfg, "BELIEF_DEBT_REF_SEC", 300.0))
        growth_ref = max(1.0, self._cfg(cfg, "BELIEF_GROWTH_REF_PER_MIN", 120.0))
        symbol_weight = self._cfg(cfg, "BELIEF_SYMBOL_WEIGHT", 0.2)
        streak_weight = self._cfg(cfg, "BELIEF_STREAK_WEIGHT", 0.03)
        evidence_weight = _clamp(self._cfg(cfg, "BELIEF_EVIDENCE_WEIGHT", 0.35), 0.0, 2.0)
        evidence_source_weight = _clamp(self._cfg(cfg, "BELIEF_EVIDENCE_SOURCE_WEIGHT", 0.08), 0.0, 1.0)
        evidence_contradiction_weight = _clamp(
            self._cfg(cfg, "BELIEF_EVIDENCE_CONTRADICTION_WEIGHT", 0.5), 0.0, 2.0
        )
        evidence_contradiction_streak_weight = _clamp(
            self._cfg(cfg, "BELIEF_EVIDENCE_CONTRADICTION_STREAK_WEIGHT", 0.08), 0.0, 1.0
        )
        evidence_debt = (1.0 - evidence_confidence)
        gate_mismatch_ref = max(1.0, self._cfg(cfg, "BELIEF_RUNTIME_GATE_MISMATCH_REF", 1.0))
        gate_invalid_ref = max(1.0, self._cfg(cfg, "BELIEF_RUNTIME_GATE_INVALID_REF", 1.0))
        gate_weight = _clamp(self._cfg(cfg, "BELIEF_RUNTIME_GATE_WEIGHT", 0.35), 0.0, 3.0)
        gate_cov_weight = _clamp(self._cfg(cfg, "BELIEF_RUNTIME_GATE_COVERAGE_WEIGHT", 1.0), 0.0, 3.0)
        gate_score_weight = _clamp(self._cfg(cfg, "BELIEF_RUNTIME_GATE_SCORE_WEIGHT", 0.5), 0.0, 3.0)
        gate_cat_ledger_weight = _clamp(self._cfg(cfg, "BELIEF_RUNTIME_GATE_CAT_LEDGER_WEIGHT", 0.8), 0.0, 3.0)
        gate_cat_transition_weight = _clamp(
            self._cfg(cfg, "BELIEF_RUNTIME_GATE_CAT_TRANSITION_WEIGHT", 0.7), 0.0, 3.0
        )
        gate_cat_belief_weight = _clamp(self._cfg(cfg, "BELIEF_RUNTIME_GATE_CAT_BELIEF_WEIGHT", 0.4), 0.0, 3.0)
        gate_cat_unknown_weight = _clamp(self._cfg(cfg, "BELIEF_RUNTIME_GATE_CAT_UNKNOWN_WEIGHT", 0.2), 0.0, 3.0)
        reconcile_first_gate_weight = _clamp(
            self._cfg(cfg, "BELIEF_RECONCILE_FIRST_GATE_WEIGHT", 0.25), 0.0, 3.0
        )
        reconcile_first_gate_count_ref = max(
            1.0, self._cfg(cfg, "BELIEF_RECONCILE_FIRST_GATE_COUNT_REF", 2.0)
        )
        reconcile_first_gate_streak_ref = max(
            1.0, self._cfg(cfg, "BELIEF_RECONCILE_FIRST_GATE_STREAK_REF", 2.0)
        )
        gate_debt = (
            (runtime_gate_mismatch_count / gate_mismatch_ref)
            + (runtime_gate_invalid_count / gate_invalid_ref)
            + ((1.0 - runtime_gate_cov) * gate_cov_weight)
            + (runtime_gate_degrade_score * gate_score_weight)
            + (runtime_gate_cat_ledger * gate_cat_ledger_weight)
            + (runtime_gate_cat_transition * gate_cat_transition_weight)
            + (runtime_gate_cat_belief * gate_cat_belief_weight)
            + (runtime_gate_cat_unknown * gate_cat_unknown_weight)
        )
        reconcile_first_gate_debt = (
            (reconcile_first_gate_count / reconcile_first_gate_count_ref)
            + float(reconcile_first_gate_max_severity)
            + (reconcile_first_gate_max_streak / reconcile_first_gate_streak_ref)
        )

        debt_score = (
            (debt_sec / debt_ref)
            + (debt_symbols * symbol_weight)
            + (mismatch_streak * streak_weight)
            + (evidence_debt * evidence_weight)
            + (float(evidence_degraded_sources) * evidence_source_weight)
            + (float(evidence_contradiction_score) * evidence_contradiction_weight)
            + (float(evidence_contradiction_streak) * evidence_contradiction_streak_weight)
            + (gate_debt * gate_weight)
            + (reconcile_first_gate_debt * reconcile_first_gate_weight)
        )
        if self.last_update > 0:
            dt = max(1e-6, now - self.last_update)
            debt_growth_per_min = max(0.0, (debt_score - self.last_debt_score) / dt * 60.0)
        else:
            debt_growth_per_min = 0.0
        self.last_update = now
        self.last_debt_score = debt_score

        # Thresholds + hysteresis
        yellow_t = self._cfg(cfg, "BELIEF_YELLOW_SCORE", 0.8)
        orange_t = self._cfg(cfg, "BELIEF_ORANGE_SCORE", 1.35)
        red_t = self._cfg(cfg, "BELIEF_RED_SCORE", 2.1)
        yellow_g = self._cfg(cfg, "BELIEF_YELLOW_GROWTH", 0.25)
        orange_g = self._cfg(cfg, "BELIEF_ORANGE_GROWTH", 0.5)
        red_g = self._cfg(cfg, "BELIEF_RED_GROWTH", 1.0)
        persist_sec = self._cfg(cfg, "BELIEF_MODE_PERSIST_SEC", 20.0)
        recover_sec = self._cfg(cfg, "BELIEF_MODE_RECOVER_SEC", 90.0)
        down_hyst = _clamp(self._cfg(cfg, "BELIEF_DOWN_HYST", 0.75), 0.4, 0.99)

        target = "GREEN"
        reason = "stable"
        if debt_score >= red_t or debt_growth_per_min >= red_g:
            target = "RED"
            reason = "red threshold"
        elif debt_score >= orange_t or debt_growth_per_min >= orange_g:
            target = "ORANGE"
            reason = "orange threshold"
        elif debt_score >= yellow_t or debt_growth_per_min >= yellow_g:
            target = "YELLOW"
            reason = "yellow threshold"

        transition = ""
        current_idx = self._MODES.index(self.mode)
        target_idx = self._MODES.index(target)
        if target_idx > current_idx:
            key = f"up:{target}"
            since = self._up_since.get(key, now)
            if key not in self._up_since:
                self._up_since[key] = now
            # RED moves immediately on threshold breach; others require persistence.
            if target == "RED" or (now - since) >= persist_sec:
                self.mode = target
                self.mode_since = now
                self._up_since.clear()
                self._down_since.clear()
                transition = f"{self._MODES[current_idx]}->{self.mode}"
        elif target_idx < current_idx:
            # Downgrade only after sustained stability below hysteresis thresholds.
            down_target = self._MODES[current_idx - 1]
            threshold = {
                "YELLOW": yellow_t * down_hyst,
                "ORANGE": orange_t * down_hyst,
                "RED": red_t * down_hyst,
            }.get(self.mode, yellow_t * down_hyst)
            if debt_score <= threshold:
                key = f"down:{down_target}"
                since = self._down_since.get(key, now)
                if key not in self._down_since:
                    self._down_since[key] = now
                if (now - since) >= recover_sec:
                    prev = self.mode
                    self.mode = down_target
                    self.mode_since = now
                    self._down_since.clear()
                    transition = f"{prev}->{self.mode}"
                    if prev == "RED":
                        post_red_warmup_sec = max(
                            0.0, self._cfg(cfg, "BELIEF_POST_RED_WARMUP_SEC", 180.0)
                        )
                        self._post_red_warmup_until = float(now + post_red_warmup_sec)
            else:
                self._down_since.clear()

        # Monotone risk mapping.
        slope = _clamp(self._cfg(cfg, "BELIEF_RISK_SLOPE", 0.35), 0.05, 1.5)
        risk_multiplier = _clamp(1.0 - (slope * debt_score), 0.0, 1.0)
        if self.mode == "YELLOW":
            risk_multiplier = min(risk_multiplier, 0.8)
        elif self.mode == "ORANGE":
            risk_multiplier = min(risk_multiplier, 0.45)
        elif self.mode == "RED":
            risk_multiplier = 0.0

        base_notional = max(0.0, self._cfg(cfg, "FIXED_NOTIONAL_USDT", 0.0))
        base_lev = max(1, int(self._cfg(cfg, "LEVERAGE", 1.0)))
        base_conf = _clamp(self._cfg(cfg, "ENTRY_MIN_CONFIDENCE", 0.0), 0.0, 1.0)
        conf_spread = _clamp(self._cfg(cfg, "BELIEF_MIN_CONF_SPREAD", 0.35), 0.0, 1.0)
        base_cd = max(0.0, self._cfg(cfg, "ENTRY_LOCAL_COOLDOWN_SEC", 8.0))
        extra_cd = max(0.0, self._cfg(cfg, "BELIEF_MAX_COOLDOWN_EXTRA_SEC", 40.0))
        base_open = max(1, int(self._cfg(cfg, "BELIEF_MAX_OPEN_ORDERS_PER_SYMBOL", 2.0)))
        per_symbol_weight = _clamp(self._cfg(cfg, "BELIEF_PER_SYMBOL_WEIGHT", 1.0), 0.1, 3.0)

        min_conf = _clamp(base_conf + ((1.0 - risk_multiplier) * conf_spread), 0.0, 1.0)
        max_notional = base_notional * risk_multiplier if base_notional > 0 else 0.0
        max_lev = max(1, int(base_lev * max(risk_multiplier, 0.1)))
        max_open = max(1, int(base_open * max(risk_multiplier, 0.5)))
        cooldown = base_cd + ((1.0 - risk_multiplier) * extra_cd)
        per_symbol: Dict[str, Dict[str, Any]] = {}
        symbol_belief_debt = belief_state.get("symbol_belief_debt_sec", {})
        if isinstance(symbol_belief_debt, dict):
            for raw_sym, raw_debt in symbol_belief_debt.items():
                sym = _symkey(str(raw_sym or ""))
                if not sym:
                    continue
                sym_debt = max(0.0, _safe_float(raw_debt, 0.0))
                sym_score = (sym_debt / debt_ref) * per_symbol_weight
                sym_multiplier = _clamp(1.0 - (slope * sym_score), 0.0, 1.0)
                sym_risk = min(risk_multiplier, sym_multiplier)
                sym_allow_entries = sym_risk > 0.20
                sym_notional = base_notional * sym_risk if base_notional > 0 else 0.0
                sym_lev = max(1, int(base_lev * max(sym_risk, 0.1)))
                sym_min_conf = _clamp(base_conf + ((1.0 - sym_risk) * conf_spread), 0.0, 1.0)
                sym_cd = base_cd + ((1.0 - sym_risk) * extra_cd)
                per_symbol[sym] = {
                    "allow_entries": bool(sym_allow_entries),
                    "max_notional_usdt": float(sym_notional),
                    "max_leverage": int(sym_lev),
                    "min_entry_conf": float(sym_min_conf),
                    "entry_cooldown_seconds": float(sym_cd),
                    "debt_sec": float(sym_debt),
                    "debt_score": float(sym_score),
                    "reason": f"symbol_debt={sym_debt:.1f}s",
                }

        allow_entries = self.mode in ("GREEN", "YELLOW")
        kill_trip = self.mode == "RED"
        if not allow_entries:
            max_notional = 0.0

        gate_recover_sec = max(1.0, self._cfg(cfg, "BELIEF_RUNTIME_GATE_RECOVER_SEC", 120.0))
        gate_cd_extra = max(0.0, self._cfg(cfg, "BELIEF_RUNTIME_GATE_COOLDOWN_EXTRA_SEC", 30.0))
        gate_min_conf_extra = _clamp(self._cfg(cfg, "BELIEF_RUNTIME_GATE_MIN_CONF_EXTRA", 0.1), 0.0, 1.0)
        gate_warmup_notional_scale = _clamp(self._cfg(cfg, "BELIEF_RUNTIME_GATE_WARMUP_NOTIONAL_SCALE", 0.5), 0.0, 1.0)
        gate_warmup_lev_scale = _clamp(self._cfg(cfg, "BELIEF_RUNTIME_GATE_WARMUP_LEVERAGE_SCALE", 0.6), 0.1, 1.0)
        gate_trip_kill = bool(self._cfg(cfg, "BELIEF_RUNTIME_GATE_TRIP_KILL", False))
        reconcile_first_gate_count_threshold = max(
            1, int(self._cfg(cfg, "BELIEF_RECONCILE_FIRST_GATE_COUNT_THRESHOLD", 2.0))
        )
        reconcile_first_gate_severity_threshold = _clamp(
            self._cfg(cfg, "BELIEF_RECONCILE_FIRST_GATE_SEVERITY_THRESHOLD", 0.85), 0.0, 1.0
        )
        reconcile_first_gate_streak_threshold = max(
            1, int(self._cfg(cfg, "BELIEF_RECONCILE_FIRST_GATE_STREAK_THRESHOLD", 2.0))
        )
        reconcile_first_spike_degraded = bool(
            (int(reconcile_first_gate_count) >= reconcile_first_gate_count_threshold)
            or (float(reconcile_first_gate_max_severity) >= reconcile_first_gate_severity_threshold)
            or (int(reconcile_first_gate_max_streak) >= reconcile_first_gate_streak_threshold)
        )
        runtime_recovering = bool(now < float(self._runtime_gate_recover_until))
        post_red_recovering = bool(now < float(self._post_red_warmup_until))
        if runtime_gate_degraded:
            self._runtime_gate_recover_until = float(now + gate_recover_sec)
            allow_entries = False
            max_notional = 0.0
            max_lev = max(1, int(max_lev * gate_warmup_lev_scale))
            min_conf = _clamp(max(min_conf, base_conf) + gate_min_conf_extra, 0.0, 1.0)
            cooldown = float(cooldown + gate_cd_extra)
            for sym in list(per_symbol.keys()):
                per_symbol[sym]["allow_entries"] = False
                per_symbol[sym]["max_notional_usdt"] = 0.0
                per_symbol[sym]["reason"] = (
                    str(per_symbol[sym].get("reason") or "") + " | runtime_gate_degraded"
                ).strip()
            if self._MODES.index(self.mode) < self._MODES.index("ORANGE"):
                self.mode = "ORANGE"
            kill_trip = bool(kill_trip or gate_trip_kill)
            if runtime_gate_reason:
                reason = f"{reason} | runtime_gate_degraded:{runtime_gate_reason}"
            else:
                reason = f"{reason} | runtime_gate_degraded"
        elif reconcile_first_spike_degraded:
            self._runtime_gate_recover_until = float(now + gate_recover_sec)
            allow_entries = False
            max_notional = 0.0
            max_lev = max(1, int(max_lev * gate_warmup_lev_scale))
            min_conf = _clamp(max(min_conf, base_conf) + gate_min_conf_extra, 0.0, 1.0)
            cooldown = float(cooldown + gate_cd_extra)
            for sym in list(per_symbol.keys()):
                per_symbol[sym]["allow_entries"] = False
                per_symbol[sym]["max_notional_usdt"] = 0.0
                per_symbol[sym]["reason"] = (
                    str(per_symbol[sym].get("reason") or "") + " | reconcile_first_spike"
                ).strip()
            if self._MODES.index(self.mode) < self._MODES.index("ORANGE"):
                self.mode = "ORANGE"
            spike_reason = (
                f"count={int(reconcile_first_gate_count)} "
                f"sev={float(reconcile_first_gate_max_severity):.2f} "
                f"streak={int(reconcile_first_gate_max_streak)}"
            )
            if reconcile_first_gate_last_reason:
                spike_reason = f"{spike_reason} reason={reconcile_first_gate_last_reason}"
            reason = f"{reason} | reconcile_first_spike:{spike_reason}"
        elif runtime_recovering:
            # staged warm-up after degraded gate clears
            if self._MODES.index(self.mode) < self._MODES.index("YELLOW"):
                self.mode = "YELLOW"
            else:
                self.mode = "YELLOW"
            allow_entries = True
            if base_notional > 0 and max_notional <= 0:
                max_notional = float(base_notional)
            max_notional = float(max_notional * gate_warmup_notional_scale)
            max_lev = max(1, int(max_lev * gate_warmup_lev_scale))
            min_conf = _clamp(max(min_conf, base_conf) + (gate_min_conf_extra * 0.5), 0.0, 1.0)
            cooldown = float(cooldown + (gate_cd_extra * 0.5))
            for sym in list(per_symbol.keys()):
                per_symbol[sym]["allow_entries"] = bool(per_symbol[sym].get("allow_entries", True))
                per_symbol[sym]["max_notional_usdt"] = float(
                    _safe_float(per_symbol[sym].get("max_notional_usdt", 0.0), 0.0)
                    * gate_warmup_notional_scale
                )
                per_symbol[sym]["max_leverage"] = max(
                    1,
                    int(_safe_float(per_symbol[sym].get("max_leverage", 1), 1.0) * gate_warmup_lev_scale),
                )
                per_symbol[sym]["entry_cooldown_seconds"] = float(
                    _safe_float(per_symbol[sym].get("entry_cooldown_seconds", base_cd), base_cd)
                    + (gate_cd_extra * 0.5)
                )
                per_symbol[sym]["reason"] = (
                    str(per_symbol[sym].get("reason") or "") + " | runtime_gate_warmup"
                ).strip()
            reason = f"{reason} | runtime_gate_warmup"
        elif gate_debt > 0:
            # Non-zero runtime gate pressure should tighten posture before a full degraded trip.
            soft_scale = _clamp(1.0 - (gate_weight * gate_debt * 0.15), 0.25, 1.0)
            max_notional = float(max_notional * soft_scale)
            max_lev = max(1, int(max_lev * soft_scale))
            min_conf = _clamp(min_conf + ((1.0 - soft_scale) * gate_min_conf_extra), 0.0, 1.0)
            cooldown = float(cooldown + ((1.0 - soft_scale) * gate_cd_extra))
            for sym in list(per_symbol.keys()):
                per_symbol[sym]["max_notional_usdt"] = float(
                    _safe_float(per_symbol[sym].get("max_notional_usdt", 0.0), 0.0) * soft_scale
                )
                per_symbol[sym]["max_leverage"] = max(
                    1,
                    int(_safe_float(per_symbol[sym].get("max_leverage", 1), 1.0) * soft_scale),
                )
            reason = f"{reason} | runtime_gate_soft_scale={soft_scale:.2f}"

        if post_red_recovering:
            rem = max(0.0, float(self._post_red_warmup_until - now))
            post_notional_scale = _clamp(
                self._cfg(cfg, "BELIEF_POST_RED_WARMUP_NOTIONAL_SCALE", 0.6), 0.0, 1.0
            )
            post_lev_scale = _clamp(
                self._cfg(cfg, "BELIEF_POST_RED_WARMUP_LEVERAGE_SCALE", 0.7), 0.1, 1.0
            )
            post_conf_extra = _clamp(
                self._cfg(cfg, "BELIEF_POST_RED_WARMUP_MIN_CONF_EXTRA", 0.06), 0.0, 1.0
            )
            post_cd_extra = max(0.0, self._cfg(cfg, "BELIEF_POST_RED_WARMUP_COOLDOWN_EXTRA_SEC", 20.0))
            max_notional = float(max_notional * post_notional_scale)
            max_lev = max(1, int(max_lev * post_lev_scale))
            min_conf = _clamp(min_conf + post_conf_extra, 0.0, 1.0)
            cooldown = float(cooldown + post_cd_extra)
            for sym in list(per_symbol.keys()):
                per_symbol[sym]["max_notional_usdt"] = float(
                    _safe_float(per_symbol[sym].get("max_notional_usdt", 0.0), 0.0)
                    * post_notional_scale
                )
                per_symbol[sym]["max_leverage"] = max(
                    1,
                    int(_safe_float(per_symbol[sym].get("max_leverage", 1), 1.0) * post_lev_scale),
                )
                per_symbol[sym]["entry_cooldown_seconds"] = float(
                    _safe_float(per_symbol[sym].get("entry_cooldown_seconds", base_cd), base_cd)
                    + post_cd_extra
                )
                per_symbol[sym]["reason"] = (
                    str(per_symbol[sym].get("reason") or "") + " | post_red_warmup"
                ).strip()
            reason = f"{reason} | post_red_warmup={rem:.0f}s"

        recovery_stage = ""
        unlock_conditions = "stable"
        next_unlock_sec = 0.0
        if runtime_gate_degraded:
            recovery_stage = "RUNTIME_GATE_DEGRADED"
            unlock_conditions = (
                f"runtime gate clear required ({runtime_gate_reason})"
                if runtime_gate_reason
                else "runtime gate clear required"
            )
        elif reconcile_first_spike_degraded:
            recovery_stage = "RECONCILE_FIRST_GATE_DEGRADED"
            unlock_conditions = (
                f"reconcile-first pressure clear required "
                f"(count={int(reconcile_first_gate_count)}, "
                f"severity={float(reconcile_first_gate_max_severity):.2f}, "
                f"streak={int(reconcile_first_gate_max_streak)})"
            )
        elif runtime_recovering:
            rem = max(0.0, float(self._runtime_gate_recover_until - now))
            recovery_stage = "RUNTIME_GATE_WARMUP"
            unlock_conditions = f"runtime gate warmup remaining {rem:.0f}s"
            next_unlock_sec = rem
        elif post_red_recovering:
            rem = max(0.0, float(self._post_red_warmup_until - now))
            recovery_stage = "POST_RED_WARMUP"
            unlock_conditions = f"post-red warmup remaining {rem:.0f}s"
            next_unlock_sec = rem
        elif self.mode == "RED":
            recovery_stage = "RED_LOCK"
            unlock_conditions = f"debt/growth must stay below recovery thresholds for {recover_sec:.0f}s"
            next_unlock_sec = float(recover_sec)
        elif self.mode == "ORANGE":
            recovery_stage = "ORANGE_RECOVERY"
            unlock_conditions = f"debt must remain below hysteresis threshold for {recover_sec:.0f}s"
            next_unlock_sec = float(recover_sec)
        elif self.mode == "YELLOW":
            recovery_stage = "YELLOW_WATCH"
            unlock_conditions = "maintain low debt and stable evidence to return GREEN"
        else:
            recovery_stage = "GREEN"

        trace = DecisionTrace(
            mode=self.mode,
            debt_score=float(debt_score),
            debt_growth_per_min=float(debt_growth_per_min),
            reason=reason,
            transition=transition,
        )
        self._last_trace = trace
        return GuardKnobs(
            allow_entries=bool(allow_entries),
            max_notional_usdt=float(max_notional),
            max_leverage=int(max_lev),
            min_entry_conf=float(min_conf),
            entry_cooldown_seconds=float(cooldown),
            max_open_orders_per_symbol=int(max_open),
            mode=str(self.mode),
            kill_switch_trip=bool(kill_trip),
            reason=f"{reason}{(' | ' + transition) if transition else ''}",
            debt_score=float(debt_score),
            debt_growth_per_min=float(debt_growth_per_min),
            runtime_gate_degraded=bool(runtime_gate_degraded),
            runtime_gate_reason=str(runtime_gate_reason),
            runtime_gate_degrade_score=float(runtime_gate_degrade_score),
            reconcile_first_gate_degraded=bool(reconcile_first_spike_degraded),
            reconcile_first_gate_count=int(reconcile_first_gate_count),
            reconcile_first_gate_max_severity=float(reconcile_first_gate_max_severity),
            reconcile_first_gate_max_streak=int(reconcile_first_gate_max_streak),
            recovery_stage=str(recovery_stage),
            unlock_conditions=str(unlock_conditions),
            next_unlock_sec=float(next_unlock_sec),
            per_symbol=per_symbol,
        )
