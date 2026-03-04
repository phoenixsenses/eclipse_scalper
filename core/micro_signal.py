from __future__ import annotations

import time
import os
import math
from dataclasses import dataclass
from typing import Any, Optional

from core.micro_features import MicroFeatureEngine, MicroFeatures
from utils.symbols import canonical_symbol


@dataclass(frozen=True)
class PocketFilter:
    min_imbalance: float
    min_intensity: float
    max_spread: float
    priority: int = 0


@dataclass(frozen=True)
class MicroSignalConfig:
    pockets: list[PocketFilter]
    active_sides: list[str]
    required_regime: str
    max_feature_age_sec: float = 5.0
    signal_cooldown_sec: float = 120.0
    order_type: str = "limit"
    fill_timeout_sec: float = 10.0


@dataclass(frozen=True)
class MicroSignal:
    symbol: str
    side: str
    confidence: float
    pocket_name: str
    features: MicroFeatures
    regime: str
    regime_age_sec: float
    order_type: str = "limit"
    fill_timeout_sec: float = 10.0


@dataclass(frozen=True)
class MicroSignalResult:
    present: bool
    confidence: float
    reason: str
    side: str
    symbol: str
    meta: Optional[dict[str, Any]] = None


def _regime_matches(required: str, current: str) -> bool:
    req = str(required or "none").strip().lower()
    cur = str(current or "UNKNOWN").strip().upper()
    if cur in ("TRANSITION", "UNKNOWN"):
        return False
    if req == "none":
        return cur in ("UP", "DOWN")
    if req == "up":
        return cur == "UP"
    if req == "down":
        return cur == "DOWN"
    return False


def _confidence(features: MicroFeatures, pocket: PocketFilter) -> float:
    if abs(float(features.imbalance_signed)) < float(pocket.min_imbalance):
        return 0.0
    if float(features.trade_intensity) < float(pocket.min_intensity):
        return 0.0
    if float(features.spread) > float(pocket.max_spread):
        return 0.0
    return 1.0


def _pocket_name(p: PocketFilter) -> str:
    return (
        f"imb>={float(p.min_imbalance):.2f}_"
        f"int>={int(float(p.min_intensity))}_"
        f"spr<={float(p.max_spread):.6f}"
    )


class MicroSignalProvider:
    def __init__(self, feature_engine: MicroFeatureEngine, regime_classifier, config: MicroSignalConfig):
        self.feature_engine = feature_engine
        self.regime_classifier = regime_classifier
        self.config = config
        self._last_signal_ts = 0.0
        self._start_ts = time.time()

    @staticmethod
    def _env_bool(name: str, default: bool) -> bool:
        v = str(os.getenv(name, "") or "").strip().lower()
        if not v:
            return bool(default)
        return v in ("1", "true", "yes", "on", "y")

    @staticmethod
    def _env_float(name: str, default: float) -> float:
        v = str(os.getenv(name, "") or "").strip()
        if not v:
            return float(default)
        try:
            return float(v)
        except Exception:
            return float(default)

    def evaluate(self, regime_override: Optional[dict] = None) -> MicroSignalResult:
        try:
            feat = self.feature_engine.current_features
            if feat is None:
                return MicroSignalResult(False, 0.0, "not_ready", "", "", {"detail": "no_features"})
            sym = canonical_symbol(str(getattr(feat, "symbol", "") or ""))
            age_sec = float(getattr(feat, "age_sec", 1e9) or 1e9)
            if age_sec > float(self.config.max_feature_age_sec):
                return MicroSignalResult(
                    False,
                    0.0,
                    "stale_features",
                    "",
                    sym,
                    {"age_sec": age_sec, "max_age_sec": float(self.config.max_feature_age_sec)},
                )
            now = time.time()
            cool = float(self.config.signal_cooldown_sec)
            if float(self._last_signal_ts) > 0 and (now - float(self._last_signal_ts)) < cool:
                rem = max(0.0, cool - (now - float(self._last_signal_ts)))
                return MicroSignalResult(False, 0.0, "cooldown", "", sym, {"remaining_sec": rem})

            current_regime = "UNKNOWN"
            regime_age_sec = 0.0
            if isinstance(regime_override, dict):
                current_regime = str(regime_override.get("current_regime") or "UNKNOWN")
                regime_age_sec = float(regime_override.get("regime_age_sec") or 0.0)
            else:
                try:
                    current_regime = str(getattr(self.regime_classifier, "current_regime", "UNKNOWN") or "UNKNOWN")
                    regime_age_sec = float(getattr(self.regime_classifier, "regime_age_sec", 0.0) or 0.0)
                except Exception:
                    current_regime = "UNKNOWN"
                    regime_age_sec = 0.0

            cur_regime = str(current_regime or "UNKNOWN").strip().upper()
            require_regime = self._env_bool("MICRO_SIGNAL_REQUIRE_REGIME", True)
            allow_unknown = self._env_bool("MICRO_SIGNAL_ALLOW_UNKNOWN_REGIME", False)
            warmup_sec = max(0.0, self._env_float("MICRO_SIGNAL_REGIME_WARMUP_SEC", 0.0))
            in_warmup = (time.time() - float(self._start_ts)) < warmup_sec if warmup_sec > 0 else False
            if not math.isfinite(float(regime_age_sec)):
                regime_age_sec = 0.0

            if require_regime and cur_regime == "UNKNOWN" and not (allow_unknown or in_warmup):
                return MicroSignalResult(
                    False,
                    0.0,
                    "regime_unknown",
                    "",
                    sym,
                    {
                        "required_regime": str(self.config.required_regime),
                        "current_regime": cur_regime,
                        "allow_unknown": bool(allow_unknown),
                        "warmup_sec": float(warmup_sec),
                        "in_warmup": bool(in_warmup),
                    },
                )
            if require_regime and cur_regime != "UNKNOWN" and not _regime_matches(self.config.required_regime, cur_regime):
                return MicroSignalResult(
                    False,
                    0.0,
                    "regime_mismatch",
                    "",
                    sym,
                    {"required_regime": str(self.config.required_regime), "current_regime": cur_regime},
                )

            pockets = sorted(
                list(self.config.pockets or []),
                key=lambda p: (-int(p.priority), -float(p.min_imbalance), -float(p.min_intensity), float(p.max_spread)),
            )
            sides = {str(s or "").strip().lower() for s in (self.config.active_sides or [])}
            if not sides:
                return MicroSignalResult(False, 0.0, "disabled", "", sym, {"detail": "no_active_sides"})
            signed = float(feat.imbalance_signed)
            for p in pockets:
                if float(feat.trade_intensity) < float(p.min_intensity):
                    continue
                if float(feat.spread) > float(p.max_spread):
                    continue
                if "sell" in sides and signed >= float(p.min_imbalance):
                    self._last_signal_ts = now
                    sig = MicroSignal(
                        symbol=sym,
                        side="sell",
                        confidence=_confidence(feat, p),
                        pocket_name=_pocket_name(p),
                        features=feat,
                        regime=str(current_regime),
                        regime_age_sec=float(regime_age_sec),
                        order_type=str(self.config.order_type or "limit"),
                        fill_timeout_sec=float(self.config.fill_timeout_sec),
                    )
                    return MicroSignalResult(
                        True,
                        float(sig.confidence),
                        "match",
                        "sell",
                        sym,
                        {"matched_pocket": sig.pocket_name, "age_sec": age_sec, "signal": sig},
                    )
                if "buy" in sides and signed <= -float(p.min_imbalance):
                    self._last_signal_ts = now
                    sig = MicroSignal(
                        symbol=sym,
                        side="buy",
                        confidence=_confidence(feat, p),
                        pocket_name=_pocket_name(p),
                        features=feat,
                        regime=str(current_regime),
                        regime_age_sec=float(regime_age_sec),
                        order_type=str(self.config.order_type or "limit"),
                        fill_timeout_sec=float(self.config.fill_timeout_sec),
                    )
                    return MicroSignalResult(
                        True,
                        float(sig.confidence),
                        "match",
                        "buy",
                        sym,
                        {"matched_pocket": sig.pocket_name, "age_sec": age_sec, "signal": sig},
                    )
            return MicroSignalResult(False, 0.0, "no_match", "", sym, {"age_sec": age_sec})
        except Exception as e:
            return MicroSignalResult(False, 0.0, "error", "", "", {"error": f"{type(e).__name__}: {e}"})
