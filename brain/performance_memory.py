from __future__ import annotations

import json
import os
import time
from typing import Any, Optional

DEFAULT_PATH = os.path.join("state", "performance_memory.json")


def _fresh(now_ts: Optional[int] = None) -> dict:
    ts = int(now_ts if now_ts is not None else time.time())
    return {
        "version": 1,
        "updated_ts": ts,
        "signals": {},
    }


def _rename_bad(path: str) -> None:
    bad = f"{path}.bad"
    if os.path.exists(bad):
        bad = f"{bad}.{int(time.time())}"
    try:
        os.replace(path, bad)
    except Exception:
        pass


def _ensure_signal(mem: dict, signal_type: str, ema_alpha: float) -> dict:
    sigs = mem.setdefault("signals", {})
    key = str(signal_type or "unknown")
    node = sigs.get(key)
    if not isinstance(node, dict):
        node = {
            "count": 0,
            "wins": 0,
            "losses": 0,
            "avg_return": 0.0,
            "ema_return": 0.0,
            "avg_return_net": 0.0,
            "ema_return_net": 0.0,
            "ema_alpha": float(ema_alpha),
            "last_ts": 0,
            "regimes": {},
        }
        sigs[key] = node
    return node


def _online_mean(prev_mean: float, prev_count: int, x: float) -> float:
    n = int(prev_count) + 1
    return float(prev_mean) + (float(x) - float(prev_mean)) / float(n)


def load_memory(path: str = DEFAULT_PATH) -> dict:
    if not os.path.exists(path):
        return _fresh()
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("memory root must be object")
        if int(data.get("version", 0) or 0) != 1:
            raise ValueError("unsupported version")
        if not isinstance(data.get("signals", {}), dict):
            raise ValueError("signals must be object")
        return data
    except Exception:
        _rename_bad(path)
        return _fresh()


def save_memory(mem: dict, path: str = DEFAULT_PATH, *, raise_on_error: bool = False) -> None:
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp = f"{path}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(mem, f, ensure_ascii=True, separators=(",", ":"))
        os.replace(tmp, path)
    except Exception as e:
        if raise_on_error:
            raise
        print(f"[performance_memory] save failed: {e}")


def record_trade_outcome(
    mem: dict,
    *,
    signal_type: str,
    regime: str | None,
    return_pct: float,
    return_pct_net: float | None = None,
    ts: int | None = None,
    ema_alpha: float = 0.2,
) -> dict:
    now_ts = int(ts if ts is not None else time.time())
    alpha = max(0.0, min(1.0, float(ema_alpha)))
    ret = float(return_pct)
    ret_net = float(return_pct_net) if return_pct_net is not None else ret
    node = _ensure_signal(mem, signal_type, alpha)

    prev_count = int(node.get("count", 0) or 0)
    prev_mean = float(node.get("avg_return", 0.0) or 0.0)
    prev_ema = float(node.get("ema_return", 0.0) or 0.0)
    prev_mean_net = float(node.get("avg_return_net", prev_mean) or prev_mean)
    prev_ema_net = float(node.get("ema_return_net", prev_ema) or prev_ema)
    node["count"] = prev_count + 1
    if ret > 0.0:
        node["wins"] = int(node.get("wins", 0) or 0) + 1
    else:
        node["losses"] = int(node.get("losses", 0) or 0) + 1
    node["avg_return"] = _online_mean(prev_mean, prev_count, ret)
    node["ema_return"] = ret if prev_count == 0 else (alpha * ret + (1.0 - alpha) * prev_ema)
    node["avg_return_net"] = _online_mean(prev_mean_net, prev_count, ret_net)
    node["ema_return_net"] = ret_net if prev_count == 0 else (alpha * ret_net + (1.0 - alpha) * prev_ema_net)
    node["ema_alpha"] = alpha
    node["last_ts"] = now_ts

    reg_key = str(regime or "unknown")
    regimes = node.setdefault("regimes", {})
    reg = regimes.get(reg_key)
    if not isinstance(reg, dict):
        reg = {
            "count": 0,
            "wins": 0,
            "avg_return": 0.0,
            "ema_return": 0.0,
            "avg_return_net": 0.0,
            "ema_return_net": 0.0,
            "last_ts": 0,
        }
        regimes[reg_key] = reg

    r_prev_count = int(reg.get("count", 0) or 0)
    r_prev_mean = float(reg.get("avg_return", 0.0) or 0.0)
    r_prev_ema = float(reg.get("ema_return", 0.0) or 0.0)
    r_prev_mean_net = float(reg.get("avg_return_net", r_prev_mean) or r_prev_mean)
    r_prev_ema_net = float(reg.get("ema_return_net", r_prev_ema) or r_prev_ema)
    reg["count"] = r_prev_count + 1
    if ret > 0.0:
        reg["wins"] = int(reg.get("wins", 0) or 0) + 1
    reg["avg_return"] = _online_mean(r_prev_mean, r_prev_count, ret)
    reg["ema_return"] = ret if r_prev_count == 0 else (alpha * ret + (1.0 - alpha) * r_prev_ema)
    reg["avg_return_net"] = _online_mean(r_prev_mean_net, r_prev_count, ret_net)
    reg["ema_return_net"] = ret_net if r_prev_count == 0 else (alpha * ret_net + (1.0 - alpha) * r_prev_ema_net)
    reg["last_ts"] = now_ts

    mem["updated_ts"] = now_ts
    return mem


def compute_signal_stats(mem: dict, signal_type: str) -> dict | None:
    sigs = mem.get("signals", {})
    node = sigs.get(str(signal_type or "unknown")) if isinstance(sigs, dict) else None
    if not isinstance(node, dict):
        return None
    count = int(node.get("count", 0) or 0)
    wins = int(node.get("wins", 0) or 0)
    out = {
        "count": count,
        "win_rate": (wins / count) if count > 0 else 0.0,
        "avg_return": float(node.get("avg_return", 0.0) or 0.0),
        "ema_return": float(node.get("ema_return", 0.0) or 0.0),
        "avg_return_net": float(node.get("avg_return_net", node.get("avg_return", 0.0)) or 0.0),
        "ema_return_net": float(node.get("ema_return_net", node.get("ema_return", 0.0)) or 0.0),
        "last_ts": int(node.get("last_ts", 0) or 0),
        "regimes": {},
    }
    regimes = node.get("regimes", {})
    if isinstance(regimes, dict):
        reg_summary = {}
        for k, v in regimes.items():
            if not isinstance(v, dict):
                continue
            c = int(v.get("count", 0) or 0)
            w = int(v.get("wins", 0) or 0)
            reg_summary[str(k)] = {
                "count": c,
                "win_rate": (w / c) if c > 0 else 0.0,
                "avg_return": float(v.get("avg_return", 0.0) or 0.0),
                "ema_return": float(v.get("ema_return", 0.0) or 0.0),
                "avg_return_net": float(v.get("avg_return_net", v.get("avg_return", 0.0)) or 0.0),
                "ema_return_net": float(v.get("ema_return_net", v.get("ema_return", 0.0)) or 0.0),
                "last_ts": int(v.get("last_ts", 0) or 0),
            }
        out["regimes"] = reg_summary
    return out


def detect_edge_decay(
    mem: dict,
    signal_type: str,
    *,
    min_count: int = 30,
    ema_floor: float = 0.0,
    win_rate_floor: float = 0.48,
) -> bool:
    stats = compute_signal_stats(mem, signal_type)
    if not stats:
        return False
    count = int(stats.get("count", 0) or 0)
    if count < int(min_count):
        return False
    ema = float(stats.get("ema_return", 0.0) or 0.0)
    wr = float(stats.get("win_rate", 0.0) or 0.0)
    return bool((ema < float(ema_floor)) or (wr < float(win_rate_floor)))
