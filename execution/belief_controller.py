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
    cause_tags: str = ""
    dominant_contributors: str = ""
    unlock_requirements: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "debt_score": float(self.debt_score),
            "debt_growth_per_min": float(self.debt_growth_per_min),
            "reason": self.reason,
            "transition": self.transition,
            "cause_tags": self.cause_tags,
            "dominant_contributors": self.dominant_contributors,
            "unlock_requirements": self.unlock_requirements,
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
        self._runtime_gate_critical_hold_until = 0.0
        self._post_red_warmup_until = 0.0
        self._healthy_ticks = 0
        self._last_contradiction_ts = 0.0

    def _compute_mode_target(
        self,
        *,
        debt_score: float,
        debt_growth_per_min: float,
        yellow_t: float,
        orange_t: float,
        red_t: float,
        yellow_g: float,
        orange_g: float,
        red_g: float,
    ) -> tuple[str, str]:
        if debt_score >= red_t or debt_growth_per_min >= red_g:
            return "RED", "red threshold"
        if debt_score >= orange_t or debt_growth_per_min >= orange_g:
            return "ORANGE", "orange threshold"
        if debt_score >= yellow_t or debt_growth_per_min >= yellow_g:
            return "YELLOW", "yellow threshold"
        return "GREEN", "stable"

    def _apply_mode_transition(
        self,
        *,
        now: float,
        target: str,
        debt_score: float,
        yellow_t: float,
        orange_t: float,
        red_t: float,
        persist_sec: float,
        recover_sec: float,
        down_hyst: float,
        post_red_warmup_sec: float,
    ) -> str:
        transition = ""
        current_idx = self._MODES.index(self.mode)
        target_idx = self._MODES.index(target)
        if target_idx > current_idx:
            key = f"up:{target}"
            since = self._up_since.get(key, now)
            if key not in self._up_since:
                self._up_since[key] = now
            if target == "RED" or (now - since) >= persist_sec:
                self.mode = target
                self.mode_since = now
                self._up_since.clear()
                self._down_since.clear()
                transition = f"{self._MODES[current_idx]}->{self.mode}"
        elif target_idx < current_idx:
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
                        self._post_red_warmup_until = float(now + post_red_warmup_sec)
            else:
                self._down_since.clear()
        return transition

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
        protection_coverage_gap_seconds = max(
            0.0, _safe_float(belief_state.get("protection_coverage_gap_seconds", 0.0), 0.0)
        )
        protection_coverage_gap_symbols = max(
            0.0, _safe_float(belief_state.get("protection_coverage_gap_symbols", 0), 0.0)
        )
        protection_coverage_ttl_breaches = max(
            0.0, _safe_float(belief_state.get("protection_coverage_ttl_breaches", 0), 0.0)
        )
        evidence_confidence = _clamp(_safe_float(belief_state.get("evidence_confidence", 1.0), 1.0), 0.0, 1.0)
        evidence_degraded_sources = max(0, int(_safe_float(belief_state.get("evidence_degraded_sources", 0), 0.0)))
        evidence_ws_gap_rate = max(0.0, _safe_float(belief_state.get("evidence_ws_gap_rate", 0.0), 0.0))
        evidence_rest_gap_rate = max(0.0, _safe_float(belief_state.get("evidence_rest_gap_rate", 0.0), 0.0))
        evidence_fill_gap_rate = max(0.0, _safe_float(belief_state.get("evidence_fill_gap_rate", 0.0), 0.0))
        evidence_ws_error_rate = max(0.0, _safe_float(belief_state.get("evidence_ws_error_rate", 0.0), 0.0))
        evidence_rest_error_rate = max(0.0, _safe_float(belief_state.get("evidence_rest_error_rate", 0.0), 0.0))
        evidence_fill_error_rate = max(0.0, _safe_float(belief_state.get("evidence_fill_error_rate", 0.0), 0.0))
        evidence_ws_source_conf = _clamp(
            _safe_float(
                belief_state.get("evidence_ws_confidence", belief_state.get("evidence_ws_score", 1.0)),
                1.0,
            ),
            0.0,
            1.0,
        )
        evidence_rest_source_conf = _clamp(
            _safe_float(
                belief_state.get("evidence_rest_confidence", belief_state.get("evidence_rest_score", 1.0)),
                1.0,
            ),
            0.0,
            1.0,
        )
        evidence_fill_source_conf = _clamp(
            _safe_float(
                belief_state.get("evidence_fill_confidence", belief_state.get("evidence_fill_score", 1.0)),
                1.0,
            ),
            0.0,
            1.0,
        )
        evidence_contradiction_score = _clamp(
            _safe_float(belief_state.get("evidence_contradiction_score", 0.0), 0.0), 0.0, 1.0
        )
        evidence_contradiction_count = max(
            0.0, _safe_float(belief_state.get("evidence_contradiction_count", 0), 0.0)
        )
        evidence_contradiction_streak = max(
            0.0, _safe_float(belief_state.get("evidence_contradiction_streak", 0), 0.0)
        )
        evidence_contradiction_burn_rate = max(
            0.0, _safe_float(belief_state.get("evidence_contradiction_burn_rate", 0.0), 0.0)
        )
        evidence_contradiction_tags = str(belief_state.get("evidence_contradiction_tags", "") or "").strip()
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
        runtime_gate_cat_position = max(0.0, _safe_float(belief_state.get("runtime_gate_cat_position", 0), 0.0))
        runtime_gate_cat_orphan = max(0.0, _safe_float(belief_state.get("runtime_gate_cat_orphan", 0), 0.0))
        runtime_gate_cat_coverage_gap = max(
            0.0, _safe_float(belief_state.get("runtime_gate_cat_coverage_gap", 0), 0.0)
        )
        runtime_gate_cat_replace_race = max(
            0.0, _safe_float(belief_state.get("runtime_gate_cat_replace_race", 0), 0.0)
        )
        runtime_gate_cat_contradiction = max(
            0.0, _safe_float(belief_state.get("runtime_gate_cat_contradiction", 0), 0.0)
        )
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
        protection_gap_ref = max(1.0, self._cfg(cfg, "BELIEF_PROTECTION_GAP_REF_SEC", 90.0))
        protection_gap_weight = _clamp(self._cfg(cfg, "BELIEF_PROTECTION_GAP_WEIGHT", 0.25), 0.0, 3.0)
        protection_gap_symbol_weight = _clamp(
            self._cfg(cfg, "BELIEF_PROTECTION_GAP_SYMBOL_WEIGHT", 0.15), 0.0, 3.0
        )
        protection_gap_ttl_weight = _clamp(
            self._cfg(cfg, "BELIEF_PROTECTION_GAP_TTL_BREACH_WEIGHT", 0.5), 0.0, 5.0
        )
        evidence_weight = _clamp(self._cfg(cfg, "BELIEF_EVIDENCE_WEIGHT", 0.35), 0.0, 2.0)
        evidence_source_weight = _clamp(self._cfg(cfg, "BELIEF_EVIDENCE_SOURCE_WEIGHT", 0.08), 0.0, 1.0)
        evidence_source_gap_weight = _clamp(self._cfg(cfg, "BELIEF_EVIDENCE_SOURCE_GAP_WEIGHT", 0.08), 0.0, 1.0)
        evidence_source_error_weight = _clamp(self._cfg(cfg, "BELIEF_EVIDENCE_SOURCE_ERROR_WEIGHT", 0.10), 0.0, 2.0)
        evidence_contradiction_weight = _clamp(
            self._cfg(cfg, "BELIEF_EVIDENCE_CONTRADICTION_WEIGHT", 0.5), 0.0, 2.0
        )
        evidence_contradiction_count_weight = _clamp(
            self._cfg(cfg, "BELIEF_EVIDENCE_CONTRADICTION_COUNT_WEIGHT", 0.04), 0.0, 1.0
        )
        evidence_contradiction_streak_weight = _clamp(
            self._cfg(cfg, "BELIEF_EVIDENCE_CONTRADICTION_STREAK_WEIGHT", 0.08), 0.0, 1.0
        )
        evidence_contradiction_burn_weight = _clamp(
            self._cfg(cfg, "BELIEF_EVIDENCE_CONTRADICTION_BURN_WEIGHT", 0.20), 0.0, 2.0
        )
        evidence_contradiction_burn_ref = max(
            1.0, self._cfg(cfg, "BELIEF_EVIDENCE_CONTRADICTION_BURN_REF", 3.0)
        )
        evidence_source_disagree_weight = _clamp(
            self._cfg(cfg, "BELIEF_EVIDENCE_SOURCE_DISAGREE_WEIGHT", 0.45), 0.0, 2.0
        )
        evidence_contradiction_tag_weight = _clamp(
            self._cfg(cfg, "BELIEF_EVIDENCE_CONTRADICTION_TAG_WEIGHT", 0.15), 0.0, 1.0
        )
        evidence_debt = (1.0 - evidence_confidence)
        evidence_gap_debt = (
            float(evidence_ws_gap_rate) + float(evidence_rest_gap_rate) + float(evidence_fill_gap_rate)
        ) / 3.0
        evidence_error_debt = (
            float(evidence_ws_error_rate) + float(evidence_rest_error_rate) + float(evidence_fill_error_rate)
        ) / 3.0
        evidence_contradiction_burn_norm = _clamp(
            float(evidence_contradiction_burn_rate) / float(evidence_contradiction_burn_ref), 0.0, 1.0
        )
        evidence_source_disagree = max(
            0.0,
            max(
                float(evidence_ws_source_conf),
                float(evidence_rest_source_conf),
                float(evidence_fill_source_conf),
            )
            - min(
                float(evidence_ws_source_conf),
                float(evidence_rest_source_conf),
                float(evidence_fill_source_conf),
            ),
        )
        evidence_tag_debt = 0.0
        if evidence_contradiction_tags:
            evidence_tag_debt = min(1.0, len([x for x in evidence_contradiction_tags.split(",") if x.strip()]) / 3.0)
        protection_gap_norm = _clamp(float(protection_coverage_gap_seconds) / float(protection_gap_ref), 0.0, 3.0)
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
        gate_cat_position_weight = _clamp(self._cfg(cfg, "BELIEF_RUNTIME_GATE_CAT_POSITION_WEIGHT", 0.5), 0.0, 3.0)
        gate_cat_orphan_weight = _clamp(self._cfg(cfg, "BELIEF_RUNTIME_GATE_CAT_ORPHAN_WEIGHT", 0.9), 0.0, 3.0)
        gate_cat_coverage_gap_weight = _clamp(
            self._cfg(cfg, "BELIEF_RUNTIME_GATE_CAT_COVERAGE_GAP_WEIGHT", 0.8), 0.0, 3.0
        )
        gate_cat_replace_race_weight = _clamp(
            self._cfg(cfg, "BELIEF_RUNTIME_GATE_CAT_REPLACE_RACE_WEIGHT", 0.6), 0.0, 3.0
        )
        gate_cat_contradiction_weight = _clamp(
            self._cfg(cfg, "BELIEF_RUNTIME_GATE_CAT_CONTRADICTION_WEIGHT", 0.7), 0.0, 3.0
        )
        gate_cat_unknown_weight = _clamp(self._cfg(cfg, "BELIEF_RUNTIME_GATE_CAT_UNKNOWN_WEIGHT", 0.2), 0.0, 3.0)
        gate_critical_weight = _clamp(self._cfg(cfg, "BELIEF_RUNTIME_GATE_CRITICAL_WEIGHT", 0.35), 0.0, 3.0)
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
            + (runtime_gate_cat_position * gate_cat_position_weight)
            + (runtime_gate_cat_orphan * gate_cat_orphan_weight)
            + (runtime_gate_cat_coverage_gap * gate_cat_coverage_gap_weight)
            + (runtime_gate_cat_replace_race * gate_cat_replace_race_weight)
            + (runtime_gate_cat_contradiction * gate_cat_contradiction_weight)
            + (runtime_gate_cat_unknown * gate_cat_unknown_weight)
        )
        critical_cat_base = (
            (runtime_gate_cat_position * gate_cat_position_weight)
            + (runtime_gate_cat_orphan * gate_cat_orphan_weight)
            + (runtime_gate_cat_coverage_gap * gate_cat_coverage_gap_weight)
            + (runtime_gate_cat_replace_race * gate_cat_replace_race_weight)
            + (runtime_gate_cat_contradiction * gate_cat_contradiction_weight)
        )
        gate_debt += (critical_cat_base * gate_critical_weight)
        gate_contributors = [
            ("mismatch", (runtime_gate_mismatch_count / gate_mismatch_ref)),
            ("invalid", (runtime_gate_invalid_count / gate_invalid_ref)),
            ("coverage", ((1.0 - runtime_gate_cov) * gate_cov_weight)),
            ("score", (runtime_gate_degrade_score * gate_score_weight)),
            ("ledger", (runtime_gate_cat_ledger * gate_cat_ledger_weight)),
            ("transition", (runtime_gate_cat_transition * gate_cat_transition_weight)),
            ("belief", (runtime_gate_cat_belief * gate_cat_belief_weight)),
            ("position", (runtime_gate_cat_position * gate_cat_position_weight)),
            ("orphan", (runtime_gate_cat_orphan * gate_cat_orphan_weight)),
            ("coverage_gap", (runtime_gate_cat_coverage_gap * gate_cat_coverage_gap_weight)),
            ("replace_race", (runtime_gate_cat_replace_race * gate_cat_replace_race_weight)),
            ("contradiction", (runtime_gate_cat_contradiction * gate_cat_contradiction_weight)),
            ("unknown", (runtime_gate_cat_unknown * gate_cat_unknown_weight)),
            ("critical", (critical_cat_base * gate_critical_weight)),
        ]
        gate_contributors = [x for x in gate_contributors if float(x[1]) > 0.0]
        gate_contributors.sort(key=lambda kv: float(kv[1]), reverse=True)
        gate_top = ",".join(f"{str(k)}={float(v):.2f}" for (k, v) in gate_contributors[:3])
        reconcile_first_gate_debt = (
            (reconcile_first_gate_count / reconcile_first_gate_count_ref)
            + float(reconcile_first_gate_max_severity)
            + (reconcile_first_gate_max_streak / reconcile_first_gate_streak_ref)
        )

        debt_score = (
            (debt_sec / debt_ref)
            + (debt_symbols * symbol_weight)
            + (mismatch_streak * streak_weight)
            + (protection_gap_norm * protection_gap_weight)
            + (float(protection_coverage_gap_symbols) * protection_gap_symbol_weight)
            + (float(protection_coverage_ttl_breaches) * protection_gap_ttl_weight)
            + (evidence_debt * evidence_weight)
            + (float(evidence_degraded_sources) * evidence_source_weight)
            + (float(evidence_gap_debt) * evidence_source_gap_weight)
            + (float(evidence_error_debt) * evidence_source_error_weight)
            + (float(evidence_contradiction_score) * evidence_contradiction_weight)
            + (float(evidence_contradiction_count) * evidence_contradiction_count_weight)
            + (float(evidence_contradiction_streak) * evidence_contradiction_streak_weight)
            + (float(evidence_contradiction_burn_norm) * evidence_contradiction_burn_weight)
            + (float(evidence_source_disagree) * evidence_source_disagree_weight)
            + (float(evidence_tag_debt) * evidence_contradiction_tag_weight)
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

        target, reason = self._compute_mode_target(
            debt_score=debt_score,
            debt_growth_per_min=debt_growth_per_min,
            yellow_t=yellow_t,
            orange_t=orange_t,
            red_t=red_t,
            yellow_g=yellow_g,
            orange_g=orange_g,
            red_g=red_g,
        )
        post_red_warmup_sec = max(0.0, self._cfg(cfg, "BELIEF_POST_RED_WARMUP_SEC", 180.0))
        transition = self._apply_mode_transition(
            now=now,
            target=target,
            debt_score=debt_score,
            yellow_t=yellow_t,
            orange_t=orange_t,
            red_t=red_t,
            persist_sec=persist_sec,
            recover_sec=recover_sec,
            down_hyst=down_hyst,
            post_red_warmup_sec=post_red_warmup_sec,
        )

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

        protection_gap_trip_sec = max(1.0, self._cfg(cfg, "BELIEF_PROTECTION_GAP_TRIP_SEC", 90.0))
        protection_gap_cd_extra = max(0.0, self._cfg(cfg, "BELIEF_PROTECTION_GAP_COOLDOWN_EXTRA_SEC", 20.0))
        protection_gap_min_conf_extra = _clamp(
            self._cfg(cfg, "BELIEF_PROTECTION_GAP_MIN_CONF_EXTRA", 0.06), 0.0, 1.0
        )
        protection_gap_degraded = bool(
            float(protection_coverage_gap_seconds) >= float(protection_gap_trip_sec)
            or float(protection_coverage_ttl_breaches) > 0.0
        )
        if protection_gap_degraded:
            allow_entries = False
            max_notional = 0.0
            max_lev = max(1, int(max_lev * 0.7))
            min_conf = _clamp(max(min_conf, base_conf) + protection_gap_min_conf_extra, 0.0, 1.0)
            cooldown = float(cooldown + protection_gap_cd_extra)
            for sym in list(per_symbol.keys()):
                per_symbol[sym]["allow_entries"] = False
                per_symbol[sym]["max_notional_usdt"] = 0.0
                per_symbol[sym]["reason"] = (
                    str(per_symbol[sym].get("reason") or "") + " | protection_gap_degraded"
                ).strip()
            if self._MODES.index(self.mode) < self._MODES.index("ORANGE"):
                self.mode = "ORANGE"
            reason = (
                f"{reason} | protection_gap_degraded:"
                f"gap={float(protection_coverage_gap_seconds):.1f}s"
                f" symbols={int(protection_coverage_gap_symbols)}"
                f" ttl_breaches={int(protection_coverage_ttl_breaches)}"
            )

        gate_recover_sec = max(1.0, self._cfg(cfg, "BELIEF_RUNTIME_GATE_RECOVER_SEC", 120.0))
        gate_cd_extra = max(0.0, self._cfg(cfg, "BELIEF_RUNTIME_GATE_COOLDOWN_EXTRA_SEC", 30.0))
        gate_min_conf_extra = _clamp(self._cfg(cfg, "BELIEF_RUNTIME_GATE_MIN_CONF_EXTRA", 0.1), 0.0, 1.0)
        gate_warmup_notional_scale = _clamp(self._cfg(cfg, "BELIEF_RUNTIME_GATE_WARMUP_NOTIONAL_SCALE", 0.5), 0.0, 1.0)
        gate_warmup_lev_scale = _clamp(self._cfg(cfg, "BELIEF_RUNTIME_GATE_WARMUP_LEVERAGE_SCALE", 0.6), 0.1, 1.0)
        gate_trip_kill = bool(self._cfg(cfg, "BELIEF_RUNTIME_GATE_TRIP_KILL", False))
        gate_critical_trip_threshold = max(
            0.0, self._cfg(cfg, "BELIEF_RUNTIME_GATE_CRITICAL_TRIP_THRESHOLD", 3.0)
        )
        gate_critical_clear_threshold = max(
            0.0,
            self._cfg(
                cfg,
                "BELIEF_RUNTIME_GATE_CRITICAL_CLEAR_THRESHOLD",
                (gate_critical_trip_threshold * 0.5),
            ),
        )
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
        critical_total = (
            float(runtime_gate_cat_position)
            + float(runtime_gate_cat_orphan)
            + float(runtime_gate_cat_coverage_gap)
            + float(runtime_gate_cat_replace_race)
            + float(runtime_gate_cat_contradiction)
        )
        critical_trip = bool(critical_total >= gate_critical_trip_threshold)
        if critical_trip:
            self._runtime_gate_critical_hold_until = float(now + gate_recover_sec)
        critical_hold_active = bool(
            (now < float(self._runtime_gate_critical_hold_until))
            and (critical_total >= gate_critical_clear_threshold)
        )
        runtime_gate_effective_degraded = bool(runtime_gate_degraded or critical_trip or critical_hold_active)
        runtime_gate_effective_reason = str(runtime_gate_reason or "")
        if (critical_trip or critical_hold_active) and not runtime_gate_reason:
            runtime_gate_effective_reason = (
                f"runtime_gate_critical(total={critical_total:.0f},"
                f"clear={gate_critical_clear_threshold:.0f},trip={gate_critical_trip_threshold:.0f})"
            )

        if runtime_gate_effective_degraded:
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
            if runtime_gate_effective_reason:
                reason = f"{reason} | runtime_gate_degraded:{runtime_gate_effective_reason}"
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
            if critical_cat_base > 0.0:
                crit_scale = _clamp(1.0 - (critical_cat_base * 0.12), 0.20, 1.0)
                soft_scale = min(soft_scale, crit_scale)
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
            if gate_top:
                reason = f"{reason} | runtime_gate_soft_scale={soft_scale:.2f} top={gate_top}"
            else:
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

        contradiction_active = bool(
            float(runtime_gate_cat_contradiction) > 0.0
            or float(evidence_contradiction_score) > 0.0
            or float(evidence_contradiction_streak) > 0.0
            or float(evidence_contradiction_burn_norm) > 0.0
            or float(evidence_source_disagree) > 0.0
            or bool(evidence_contradiction_tags)
        )
        if contradiction_active:
            self._last_contradiction_ts = float(now)
        unlock_min_cov = _clamp(self._cfg(cfg, "BELIEF_UNLOCK_MIN_JOURNAL_COVERAGE", 0.90), 0.0, 1.0)
        unlock_contrad_sec = max(0.0, self._cfg(cfg, "BELIEF_UNLOCK_CONTRADICTION_CLEAR_SEC", 60.0))
        unlock_max_protection_gap_sec = max(
            0.0, self._cfg(cfg, "BELIEF_UNLOCK_MAX_PROTECTION_GAP_SEC", 0.0)
        )
        unlock_healthy_ticks_required = max(1, int(self._cfg(cfg, "BELIEF_UNLOCK_HEALTHY_TICKS_REQUIRED", 3.0)))
        is_healthy_tick = bool(
            (not runtime_gate_effective_degraded)
            and (not reconcile_first_spike_degraded)
            and (float(runtime_gate_cov) >= float(unlock_min_cov))
            and (not contradiction_active)
            and (float(protection_coverage_gap_seconds) <= float(unlock_max_protection_gap_sec))
        )
        if is_healthy_tick:
            self._healthy_ticks += 1
        else:
            self._healthy_ticks = 0
        contradiction_clear_sec = (
            float(now - self._last_contradiction_ts) if float(self._last_contradiction_ts) > 0.0 else float(unlock_contrad_sec)
        )
        unlock_needs: list[str] = []
        if int(self._healthy_ticks) < int(unlock_healthy_ticks_required):
            unlock_needs.append(f"healthy_ticks {int(self._healthy_ticks)}/{int(unlock_healthy_ticks_required)}")
        if float(runtime_gate_cov) < float(unlock_min_cov):
            unlock_needs.append(f"journal_coverage {float(runtime_gate_cov):.3f}/{float(unlock_min_cov):.3f}")
        if float(contradiction_clear_sec) < float(unlock_contrad_sec):
            unlock_needs.append(
                f"contradiction_clear {float(contradiction_clear_sec):.0f}s/{float(unlock_contrad_sec):.0f}s"
            )
        if float(protection_coverage_gap_seconds) > float(unlock_max_protection_gap_sec):
            unlock_needs.append(
                f"protection_gap {float(protection_coverage_gap_seconds):.1f}s/"
                f"{float(unlock_max_protection_gap_sec):.1f}s"
            )
        unlock_requirements = "; ".join(unlock_needs) if unlock_needs else "stable"
        cause_tags: list[str] = []
        if runtime_gate_effective_degraded:
            cause_tags.append("runtime_gate")
        if critical_trip or critical_hold_active:
            cause_tags.append("runtime_gate_critical")
        if reconcile_first_spike_degraded:
            cause_tags.append("reconcile_first_spike")
        if contradiction_active:
            cause_tags.append("evidence_contradiction")
        if float(evidence_source_disagree) > 0.0:
            cause_tags.append("evidence_source_disagreement")
        if float(evidence_contradiction_burn_norm) > 0.0:
            cause_tags.append("evidence_contradiction_burn")
        if float(protection_coverage_gap_seconds) > 0.0:
            cause_tags.append("protection_gap")
        if float(runtime_gate_mismatch_count) > 0.0:
            cause_tags.append("replay_mismatch")
        if transition:
            cause_tags.append("mode_transition")

        recovery_stage = ""
        unlock_conditions = "stable"
        next_unlock_sec = 0.0
        if runtime_gate_effective_degraded:
            recovery_stage = "RUNTIME_GATE_DEGRADED"
            unlock_conditions = (
                f"runtime gate clear required ({runtime_gate_effective_reason})"
                if runtime_gate_effective_reason
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
        if unlock_requirements != "stable":
            if unlock_conditions == "stable":
                unlock_conditions = unlock_requirements
            elif unlock_requirements not in unlock_conditions:
                unlock_conditions = f"{unlock_conditions}; {unlock_requirements}"

        trace = DecisionTrace(
            mode=self.mode,
            debt_score=float(debt_score),
            debt_growth_per_min=float(debt_growth_per_min),
            reason=reason,
            transition=transition,
            cause_tags=",".join(cause_tags),
            dominant_contributors=str(gate_top),
            unlock_requirements=str(unlock_requirements),
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
            runtime_gate_degraded=bool(runtime_gate_effective_degraded),
            runtime_gate_reason=str(runtime_gate_effective_reason),
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
