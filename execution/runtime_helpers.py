from __future__ import annotations

import os
from typing import Any


def cfg_value(bot: Any, name: str, default: Any) -> Any:
    try:
        return getattr(getattr(bot, "cfg", None), name, default)
    except Exception:
        return default


def env_get(name: str) -> str:
    try:
        return str(os.getenv(name, "")).strip()
    except Exception:
        return ""


def truthy(value: Any) -> bool:
    if value is True:
        return True
    if isinstance(value, (int, float)) and value != 0:
        return True
    if isinstance(value, str) and value.strip().lower() in ("true", "1", "yes", "y", "on"):
        return True
    return False


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
        if parsed != parsed:
            return float(default)
        return parsed
    except Exception:
        return float(default)


def symkey(sym: str) -> str:
    normalized = (sym or "").upper().strip()
    normalized = normalized.replace("/USDT:USDT", "USDT").replace("/USDT", "USDT")
    normalized = normalized.replace(":USDT", "USDT").replace(":", "")
    normalized = normalized.replace("/", "")
    if normalized.endswith("USDTUSDT"):
        normalized = normalized[:-4]
    return normalized


def cfg_env_float(bot: Any, name: str, default: float) -> float:
    value = env_get(name)
    if value != "":
        try:
            return float(value)
        except Exception:
            pass
    try:
        return float(cfg_value(bot, name, default) or default)
    except Exception:
        return float(default)


def cfg_env_bool(bot: Any, name: str, default: Any = False) -> bool:
    value = env_get(name)
    if value != "":
        return truthy(value)
    return truthy(cfg_value(bot, name, default))


def parse_group_kv(raw: str) -> dict[str, float]:
    out: dict[str, float] = {}
    try:
        text = str(raw or "").strip()
        if not text:
            return out
        parts = [p.strip() for p in text.replace(";", ",").split(",") if p.strip()]
        for part in parts:
            if "=" not in part:
                continue
            key, value = part.split("=", 1)
            group = str(key or "").strip().upper()
            if not group:
                continue
            out[group] = safe_float(value, 0.0)
    except Exception:
        return out
    return out


def parse_symbol_kv(raw: str) -> dict[str, float]:
    if raw is None:
        return {}
    text = str(raw).strip()
    if not text:
        return {}
    out: dict[str, float] = {}
    parts = [p.strip() for p in text.replace(";", ",").split(",") if p.strip()]
    for part in parts:
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        symbol = symkey(key)
        if not symbol:
            continue
        out[symbol] = safe_float(value, 0.0)
    return out
