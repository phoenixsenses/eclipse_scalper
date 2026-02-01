#!/usr/bin/env python3
"""
Validate correlation group settings from environment variables.
"""

from __future__ import annotations

import os


def _symkey(sym: str) -> str:
    s = (sym or "").upper().strip()
    s = s.replace("/USDT:USDT", "USDT").replace("/USDT", "USDT")
    s = s.replace(":USDT", "USDT").replace(":", "")
    s = s.replace("/", "")
    if s.endswith("USDTUSDT"):
        s = s[:-4]
    return s


def _safe_float(x, default=0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _parse_groups(raw: str) -> dict:
    out: dict = {}
    s = str(raw or "").strip()
    if not s:
        return out
    groups = [g.strip() for g in s.split(";") if g.strip()]
    for g in groups:
        if ":" not in g:
            continue
        name, syms = g.split(":", 1)
        gn = str(name or "").strip().upper()
        if not gn:
            continue
        members = []
        for p in syms.replace(" ", "").split(","):
            if not p:
                continue
            members.append(_symkey(p))
        if members:
            out[gn] = members
    return out


def _parse_group_kv(raw: str) -> dict:
    out: dict = {}
    s = str(raw or "").strip()
    if not s:
        return out
    parts = [p.strip() for p in s.replace(";", ",").split(",") if p.strip()]
    for p in parts:
        if "=" not in p:
            continue
        k, v = p.split("=", 1)
        kk = str(k or "").strip().upper()
        if not kk:
            continue
        out[kk] = _safe_float(v, 0.0)
    return out


def main() -> int:
    raw_groups = os.getenv("CORR_GROUPS", "")
    groups = _parse_groups(raw_groups)

    global_pos = int(_safe_float(os.getenv("CORR_GROUP_MAX_POSITIONS", 0), 0))
    global_notional = _safe_float(os.getenv("CORR_GROUP_MAX_NOTIONAL_USDT", 0.0), 0.0)
    per_pos = _parse_group_kv(os.getenv("CORR_GROUP_LIMITS", ""))
    per_notional = _parse_group_kv(os.getenv("CORR_GROUP_NOTIONAL", ""))

    if not groups:
        print("No CORR_GROUPS configured.")
        return 0

    print("Correlation Groups")
    for name, members in groups.items():
        dupe = len(set(members)) != len(members)
        print(f"- {name}: {', '.join(members)}")
        if dupe:
            print(f"  WARNING: duplicate symbols detected in group {name}.")

    if global_pos > 0 or global_notional > 0:
        print("\nGlobal caps")
        if global_pos > 0:
            print(f"- CORR_GROUP_MAX_POSITIONS={global_pos}")
        if global_notional > 0:
            print(f"- CORR_GROUP_MAX_NOTIONAL_USDT={global_notional}")

    if per_pos or per_notional:
        print("\nPer-group overrides")
        for name in sorted(set(list(per_pos.keys()) + list(per_notional.keys()))):
            if name not in groups:
                print(f"- {name}: WARNING override set but group not defined")
                continue
            pos = per_pos.get(name)
            notional = per_notional.get(name)
            parts = []
            if pos is not None and pos > 0:
                parts.append(f"max_positions={int(pos)}")
            if notional is not None and notional > 0:
                parts.append(f"max_notional={notional}")
            print(f"- {name}: {', '.join(parts) if parts else 'no effective limits'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
