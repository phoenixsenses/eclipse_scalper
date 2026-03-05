from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List


_VALID_SIDES = {"buy", "sell", "both"}
_VALID_ENTRIES = {"market", "passive"}
_VALID_ENTRY_PREF = {"taker", "maker", "both"}


@dataclass(frozen=True)
class SignalSpec:
    name: str
    side: str
    condition: Dict[str, Any]
    entry: str = "market"
    horizon_bars: int = 10
    cooldown_bars: int = 0
    regime_filter: List[int] = field(default_factory=list)
    entry_mode_preference: str = "both"
    meta: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not str(self.name).strip():
            raise ValueError("name is required")
        if str(self.side).lower() not in _VALID_SIDES:
            raise ValueError(f"invalid side: {self.side}")
        if str(self.entry).lower() not in _VALID_ENTRIES:
            raise ValueError(f"invalid entry: {self.entry}")
        if int(self.horizon_bars) <= 0:
            raise ValueError("horizon_bars must be > 0")
        if int(self.cooldown_bars) < 0:
            raise ValueError("cooldown_bars must be >= 0")
        if not isinstance(self.condition, dict):
            raise ValueError("condition must be an object")
        if not isinstance(self.meta, dict):
            raise ValueError("meta must be an object")
        if not isinstance(self.regime_filter, list):
            raise ValueError("regime_filter must be a list")
        if str(self.entry_mode_preference).lower() not in _VALID_ENTRY_PREF:
            raise ValueError(f"invalid entry_mode_preference: {self.entry_mode_preference}")

    def to_dict(self) -> Dict[str, Any]:
        self.validate()
        payload = asdict(self)
        payload["side"] = str(payload["side"]).lower()
        payload["entry"] = str(payload["entry"]).lower()
        payload["horizon_bars"] = int(payload["horizon_bars"])
        payload["cooldown_bars"] = int(payload["cooldown_bars"])
        payload["regime_filter"] = [int(x) for x in payload.get("regime_filter", [])]
        payload["entry_mode_preference"] = str(payload.get("entry_mode_preference", "both")).lower()
        return payload


def signal_from_dict(payload: Dict[str, Any]) -> SignalSpec:
    spec = SignalSpec(
        name=str(payload.get("name", "")).strip(),
        side=str(payload.get("side", "both")).lower(),
        condition=dict(payload.get("condition", {}) or {}),
        entry=str(payload.get("entry", "market")).lower(),
        horizon_bars=int(payload.get("horizon_bars", 10)),
        cooldown_bars=int(payload.get("cooldown_bars", 0)),
        regime_filter=[int(x) for x in (payload.get("regime_filter", []) or [])],
        entry_mode_preference=str(payload.get("entry_mode_preference", "both")).lower(),
        meta=dict(payload.get("meta", {}) or {}),
    )
    spec.validate()
    return spec


def specs_to_jsonl(specs: List[SignalSpec]) -> str:
    lines = []
    for spec in specs:
        lines.append(json.dumps(spec.to_dict(), ensure_ascii=True, sort_keys=True, separators=(",", ":")))
    return "\n".join(lines) + ("\n" if lines else "")
