from __future__ import annotations

import json
import math
import os
import statistics
import time
from typing import Any

DATA_QUALITY_JOURNAL_PATH = os.path.join("logs", "data_quality.jsonl")


def _is_finite_number(v: Any) -> bool:
    try:
        return math.isfinite(float(v))
    except Exception:
        return False


def _to_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


def validate_candles(candles: list[dict]) -> dict:
    stats = {
        "count": int(len(candles) if isinstance(candles, list) else 0),
        "missing_gaps": 0,
        "duplicate_timestamps": 0,
        "non_monotonic": 0,
        "invalid_ohlc": 0,
        "invalid_volume": 0,
    }
    issues: list[str] = []

    if not isinstance(candles, list):
        stats["invalid_ohlc"] += 1
        issues.append("candles_not_list")
        return {"ok": False, "severity": "block", "issues": issues, "stats": stats}

    prev_ts: int | None = None
    intervals: list[int] = []
    raw_intervals: list[int] = []
    req = ("timestamp", "open", "high", "low", "close", "volume")

    for i, row in enumerate(candles):
        if not isinstance(row, dict):
            stats["invalid_ohlc"] += 1
            issues.append(f"row_not_dict@{i}")
            continue

        missing = [k for k in req if k not in row]
        if missing:
            stats["invalid_ohlc"] += 1
            issues.append(f"missing_keys@{i}:{','.join(missing)}")
            continue

        ts_val = row.get("timestamp")
        if not _is_finite_number(ts_val):
            stats["invalid_ohlc"] += 1
            issues.append(f"invalid_timestamp@{i}")
            continue
        ts = int(_to_float(ts_val))

        o = row.get("open")
        h = row.get("high")
        l = row.get("low")
        c = row.get("close")
        v = row.get("volume")

        if not all(_is_finite_number(x) for x in (o, h, l, c)):
            stats["invalid_ohlc"] += 1
            issues.append(f"nan_or_missing_ohlc@{i}")
        else:
            of = _to_float(o)
            hf = _to_float(h)
            lf = _to_float(l)
            cf = _to_float(c)
            if (
                of <= 0.0
                or hf <= 0.0
                or lf <= 0.0
                or cf <= 0.0
                or hf < max(of, cf)
                or lf > min(of, cf)
                or lf > hf
            ):
                stats["invalid_ohlc"] += 1
                issues.append(f"invalid_ohlc_logic@{i}")

        if (not _is_finite_number(v)) or (_to_float(v) < 0.0):
            stats["invalid_volume"] += 1
            issues.append(f"invalid_volume@{i}")

        if prev_ts is not None:
            dt = ts - prev_ts
            raw_intervals.append(dt)
            if dt == 0:
                stats["duplicate_timestamps"] += 1
                issues.append(f"duplicate_timestamp@{i}")
            elif dt < 0:
                stats["non_monotonic"] += 1
                issues.append(f"non_monotonic_timestamp@{i}")
            else:
                intervals.append(dt)
        prev_ts = ts

    if intervals:
        med = int(statistics.median(intervals))
        if med > 0:
            gap_count = sum(1 for dt in raw_intervals if dt > (2 * med))
            stats["missing_gaps"] = int(gap_count)
            if gap_count > 0:
                issues.append(f"missing_gaps={gap_count}")

    if stats["non_monotonic"] > 0 or stats["invalid_ohlc"] > 0 or stats["duplicate_timestamps"] > 0:
        severity = "block"
    elif stats["missing_gaps"] > 0:
        severity = "warn"
    else:
        severity = "ok"

    return {
        "ok": bool(severity != "block"),
        "severity": severity,
        "issues": issues,
        "stats": stats,
    }


def journal_data_quality(symbol: str, timeframe: str, report: dict) -> None:
    rec = {
        "ts": int(time.time()),
        "symbol": str(symbol or ""),
        "timeframe": str(timeframe or ""),
        "severity": str((report or {}).get("severity", "block")),
        "issues": list((report or {}).get("issues", []) or []),
        "stats": dict((report or {}).get("stats", {}) or {}),
    }
    try:
        path = DATA_QUALITY_JOURNAL_PATH
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=True, separators=(",", ":")) + "\n")
    except Exception:
        # best-effort journaling only
        return

