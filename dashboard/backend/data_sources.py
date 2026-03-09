"""Data sources for Eclipse Scalper Dashboard.

Reads from repo structure: logs/, state/, data/.
All reads are graceful — missing files return empty/default values.
"""
from __future__ import annotations

import json
import os
import platform
import re
import shutil
import sqlite3
import sys
import time
import urllib.parse
import urllib.request
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


# ─────────────────────────────────────────────
# Repo root — portable: always resolves relative to this file
# ─────────────────────────────────────────────

REPO_ROOT = Path(__file__).parent.parent.parent.resolve()


def _env_path(env_var: str, default: Path) -> Path:
    """Return Path from env var (absolute or REPO_ROOT-relative), else default."""
    raw = os.environ.get(env_var, "").strip()
    if not raw:
        return default
    p = Path(raw)
    return p if p.is_absolute() else (REPO_ROOT / p).resolve()


# Standard dirs.  LOG_DIR env var may override LOGS_DIR.
LOGS_DIR  = _env_path("LOG_DIR", REPO_ROOT / "logs")
STATE_DIR = REPO_ROOT / "state"
DATA_DIR  = REPO_ROOT / "data"
REPORTS_DIR = REPO_ROOT / "reports"

_SENSITIVE_KEYWORDS = {"API_KEY", "API_SECRET", "SECRET", "TOKEN", "PASSWORD", "PASS", "PRIVATE"}


# ─────────────────────────────────────────────
# Runtime status — constants
# ─────────────────────────────────────────────

_COLLECTOR_ALIVE_MAX_AGE = 120   # seconds: 2 missed 60 s stat intervals → dead
_FRESHNESS_DEGRADED      = 30    # seconds since last trade → DEGRADED
_FRESHNESS_STALE         = 120   # seconds since last trade → STALE
_RUNTIME_CACHE_TTL       = 1.0   # seconds

# Collector log stats-line regex (matches the real format written by microstructure_collector.py):
#   [HH:MM:SS] Uptime: X.Xh | DB: X.XMB | Trades: X/s (...) | Mark: X/s (...) | Liqs: X.X/s (...)
_STATS_RE = re.compile(
    r"\[(\d{2}:\d{2}:\d{2})\]\s+Uptime:\s*([\d.]+)h"
    r".*?Trades:\s*([\d.,]+)/s"
    r".*?Mark:\s*([\d.,]+)/s"
    r".*?Liqs:\s*([\d.,]+)/s",
)

# agg_trades timestamp column candidates, in priority order.
# ts_ms  → integer epoch milliseconds
# ts_utc / ts / timestamp → ISO 8601 string
_TS_COL_PRIORITY: list[str] = ["ts_ms", "ts_utc", "ts", "timestamp"]

# Module-level runtime cache
_runtime_cache: dict[str, Any] = {}
_runtime_cache_ts: float = 0.0

# DB size history for 5-minute growth: deque of (monotonic_ts, size_bytes)
_db_size_history: deque[tuple[float, int]] = deque(maxlen=360)  # 1 s × 360 s > 5 min


# ─────────────────────────────────────────────
# Runtime path accessors (read env vars at call time for late-binding)
# ─────────────────────────────────────────────

def _get_db_path() -> Path:
    """Effective path to microstructure.db — overridable via MICROSTRUCTURE_DB_PATH."""
    return _env_path("MICROSTRUCTURE_DB_PATH", DATA_DIR / "microstructure.db")


def _get_collector_log() -> Path:
    """Effective path to collector log — overridable via COLLECTOR_LOG_PATH."""
    return _env_path("COLLECTOR_LOG_PATH", LOGS_DIR / "microstructure_collector.log")


# ─────────────────────────────────────────────
# Internal utilities
# ─────────────────────────────────────────────

def _safe_json(path: Path) -> dict[str, Any]:
    """Load JSON file; return {} on any error."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _ema(values: list[float], period: int) -> list[float | None]:
    if period <= 0 or not values:
        return [None] * len(values)
    out: list[float | None] = [None] * len(values)
    alpha = 2.0 / (period + 1.0)
    prev: float | None = None
    for idx, value in enumerate(values):
        prev = value if prev is None else (value * alpha) + (prev * (1.0 - alpha))
        if idx >= period - 1:
            out[idx] = round(prev, 6)
    return out


def _rsi(values: list[float], period: int = 14) -> list[float | None]:
    if period <= 0 or len(values) <= period:
        return [None] * len(values)
    gains = [0.0] * len(values)
    losses = [0.0] * len(values)
    for idx in range(1, len(values)):
        delta = values[idx] - values[idx - 1]
        gains[idx] = max(delta, 0.0)
        losses[idx] = max(-delta, 0.0)

    out: list[float | None] = [None] * len(values)
    avg_gain = sum(gains[1 : period + 1]) / period
    avg_loss = sum(losses[1 : period + 1]) / period
    out[period] = 100.0 if avg_loss == 0 else round(100.0 - (100.0 / (1.0 + (avg_gain / avg_loss))), 4)

    for idx in range(period + 1, len(values)):
        avg_gain = ((avg_gain * (period - 1)) + gains[idx]) / period
        avg_loss = ((avg_loss * (period - 1)) + losses[idx]) / period
        out[idx] = 100.0 if avg_loss == 0 else round(100.0 - (100.0 / (1.0 + (avg_gain / avg_loss))), 4)
    return out


def _interval_to_seconds(interval: str) -> int:
    mapping = {"1m": 60, "5m": 300, "15m": 900, "1h": 3600, "4h": 14400}
    return mapping.get(interval, 300)


def _build_pocket_markers(symbol: str, interval: str, candles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if symbol != "ETHUSDT" or not candles:
        return []

    db_path = _get_db_path()
    if not db_path.exists():
        return []

    interval_sec = _interval_to_seconds(interval)
    start_ts_ms = int(candles[0]["time"]) * 1000
    end_ts_ms = int(candles[-1]["time"] + interval_sec) * 1000
    regime_start_ts_ms = start_ts_ms - (3600 * 1000)
    candle_times = {int(c["time"]): c for c in candles}

    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        trades = cur.execute(
            """
            select (ts_ms / 1000) as bucket_s,
                   sum(case when is_buyer_maker = 0 then quantity else 0 end) as buy_qty,
                   sum(case when is_buyer_maker = 1 then quantity else 0 end) as sell_qty,
                   count(*) as trade_count,
                   sum(price * quantity) / nullif(sum(quantity), 0) as vwap
            from agg_trades
            where symbol = ? and ts_ms >= ? and ts_ms < ?
            group by bucket_s
            order by bucket_s
            """,
            (symbol, start_ts_ms, end_ts_ms),
        ).fetchall()
        marks = cur.execute(
            """
            select (ts_ms / 1000) as bucket_s,
                   avg(mark_price) as mark_price
            from mark_prices
            where symbol = ? and ts_ms >= ? and ts_ms < ?
            group by bucket_s
            order by bucket_s
            """,
            (symbol, regime_start_ts_ms, end_ts_ms),
        ).fetchall()
        conn.close()
    except Exception:
        try:
            conn.close()
        except Exception:
            pass
        return []

    mark_by_bucket = {int(row[0]): float(row[1]) for row in marks if row and row[1] is not None}
    if not trades or not mark_by_bucket:
        return []

    price_history: list[tuple[int, float]] = sorted(mark_by_bucket.items())
    history_times = [row[0] for row in price_history]
    history_prices = [row[1] for row in price_history]

    markers_by_candle: dict[int, dict[str, Any]] = {}
    for bucket_s, buy_qty, sell_qty, trade_count, vwap in trades:
        bucket_s = int(bucket_s)
        mark = mark_by_bucket.get(bucket_s)
        if mark is None or mark <= 0:
            continue
        buy_qty = float(buy_qty or 0.0)
        sell_qty = float(sell_qty or 0.0)
        total_qty = buy_qty + sell_qty
        if total_qty <= 0:
            continue
        imbalance = (buy_qty - sell_qty) / total_qty
        trade_intensity = int(trade_count or 0) * 60.0
        spread = abs(float(vwap or mark) - mark) / mark

        if abs(imbalance) < 0.5 or trade_intensity < 3500.0 or spread > 0.0003:
            continue

        regime = "UNKNOWN"
        window_start = bucket_s - 3600
        past_candidates = [price for ts, price in price_history if ts <= window_start]
        if past_candidates:
            prev_mark = past_candidates[-1]
            if prev_mark > 0:
                regime = "UP" if mark > prev_mark else "DOWN"

        side = "BUY" if imbalance > 0 else "SELL"
        verdict = "WAIT"
        if regime == "UP":
            verdict = "GO"
        elif regime == "DOWN":
            verdict = "MARGINAL" if side == "BUY" else "NO-GO"

        candle_time = (bucket_s // interval_sec) * interval_sec
        if candle_time not in candle_times:
            continue

        score = abs(imbalance) + (trade_intensity / 3500.0) + max(0.0, (0.0003 - spread) / 0.0003)
        existing = markers_by_candle.get(candle_time)
        marker = {
            "time": candle_time,
            "bucket_time": bucket_s,
            "side": side,
            "verdict": verdict,
            "regime": regime,
            "imbalance": round(imbalance, 4),
            "trade_intensity": round(trade_intensity, 2),
            "spread": round(spread, 6),
            "score": round(score, 4),
        }
        if existing is None or float(existing.get("score", 0.0)) < score:
            markers_by_candle[candle_time] = marker

    return [markers_by_candle[key] for key in sorted(markers_by_candle)]


def _safe_json_with_meta(path: Path, stale_after_sec: float = 900.0) -> dict[str, Any]:
    payload = _safe_json(path)
    exists = path.exists()
    mtime = None
    age_sec = None
    stale = True
    if exists:
        try:
            mtime = path.stat().st_mtime
            age_sec = round(max(0.0, time.time() - mtime), 1)
            stale = age_sec > max(30.0, float(stale_after_sec))
        except Exception:
            pass
    if isinstance(payload, dict):
        payload.setdefault("_meta", {})
        payload["_meta"] = {
            "path": str(path),
            "exists": exists,
            "age_sec": age_sec,
            "stale": stale if exists else True,
        }
    return payload if isinstance(payload, dict) else {}


def _tail_lines(path: Path, n: int = 200) -> list[str]:
    """Return last n lines of a text file efficiently."""
    try:
        buf: deque[str] = deque(maxlen=n)
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                buf.append(line.rstrip("\n"))
        return list(buf)
    except Exception:
        return []


def _read_jsonl_tail(
    path: Path,
    limit: int = 100,
    symbol_filter: Optional[str] = None,
) -> list[dict]:
    """Read last `limit` valid JSON lines from a JSONL file, with optional symbol filter."""
    buf: deque[dict] = deque(maxlen=max(limit * 5, 500))
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if symbol_filter:
                        sym = (obj.get("symbol") or obj.get("sym") or "")
                        if sym.upper() != symbol_filter.upper():
                            continue
                    buf.append(obj)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    pass
    except Exception:
        pass
    return list(buf)[-limit:]


def _is_sensitive(key: str) -> bool:
    ku = key.upper()
    return any(kw in ku for kw in _SENSITIVE_KEYWORDS)


def _mask(value: str) -> str:
    if len(value) <= 4:
        return "***"
    return value[:4] + "***"


def _safe_path(filename: str, base_dir: Path) -> Optional[Path]:
    """Resolve a basename-only filename safely; reject path traversal."""
    safe_name = Path(filename).name
    candidate = (base_dir / safe_name).resolve()
    try:
        candidate.relative_to(base_dir.resolve())
        return candidate
    except ValueError:
        return None


# ─────────────────────────────────────────────
# State readers
# ─────────────────────────────────────────────

def read_scoreboard() -> dict[str, Any]:
    return _safe_json(STATE_DIR / "paper_scoreboard.json")


def read_micro_edge_gates() -> dict[str, Any]:
    return _safe_json(STATE_DIR / "micro_edge_gates.json")


def read_passive_profiles() -> dict[str, Any]:
    return _safe_json(STATE_DIR / "passive_realistic_profiles.json")


_RESEARCH_STATE_FILES: dict[str, str] = {
    "liquidation": "LIQUIDATION_ALERT_STATE_REAL.json",
    "spread_stress": "SPREAD_STRESS_STATE_REAL.json",
    "fill_toxicity": "FILL_TOXICITY_STATE_REAL.json",
    "latency_stress": "LATENCY_STRESS_STATE_REAL.json",
    "return_shock": "RETURN_SHOCK_STATE_REAL.json",
    "volume_vacuum": "VOLUME_VACUUM_STATE_REAL.json",
    "volatility_burst": "VOLATILITY_BURST_STATE_REAL.json",
    "book_proxy_pressure": "BOOK_PROXY_PRESSURE_STATE_REAL.json",
}

_RESEARCH_WATCHLIST_FILES: dict[str, str] = {
    "liquidation": "LIQUIDATION_WATCHLIST_REAL.json",
    "spread_stress": "SPREAD_STRESS_WATCHLIST_REAL.json",
    "return_shock": "RETURN_SHOCK_WATCHLIST_REAL.json",
    "volume_vacuum": "VOLUME_VACUUM_WATCHLIST_REAL.json",
    "volatility_burst": "VOLATILITY_BURST_WATCHLIST_REAL.json",
    "book_proxy_pressure": "BOOK_PROXY_PRESSURE_WATCHLIST_REAL.json",
}


def _latest_daily_report() -> dict[str, Any]:
    try:
        candidates = sorted(REPORTS_DIR.glob("DAILY_*.json"), key=lambda path: path.name, reverse=True)
    except Exception:
        candidates = []
    if not candidates:
        return {"_meta": {"path": str(REPORTS_DIR / "DAILY_<date>.json"), "exists": False, "age_sec": None, "stale": True}}
    return _safe_json_with_meta(candidates[0], stale_after_sec=36 * 3600.0)


def read_research_events() -> dict[str, Any]:
    watchboard = _safe_json_with_meta(REPORTS_DIR / "RESEARCH_EVENT_WATCHBOARD_REAL.json")
    states = {name: _safe_json_with_meta(REPORTS_DIR / filename) for name, filename in _RESEARCH_STATE_FILES.items()}
    watchlists = {name: _safe_json_with_meta(REPORTS_DIR / filename) for name, filename in _RESEARCH_WATCHLIST_FILES.items()}
    return {
        "daily_report": _latest_daily_report(),
        "watchboard": watchboard,
        "states": states,
        "watchlists": watchlists,
    }


# ─────────────────────────────────────────────
# Log readers (JSON snapshots)
# ─────────────────────────────────────────────

def read_exit_quality() -> dict[str, Any]:
    return _safe_json(LOGS_DIR / "exit_quality_summary.json")


def read_preflight() -> dict[str, Any]:
    return _safe_json(LOGS_DIR / "preflight_check.json")


def read_reliability() -> dict[str, Any]:
    """Parse reliability_gate.txt key=value pairs into a dict."""
    path = LOGS_DIR / "reliability_gate.txt"
    result: dict[str, Any] = {}
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                for sep in ("=", ":"):
                    if sep in line:
                        k, _, v = line.partition(sep)
                        result[k.strip()] = v.strip()
                        break
    except Exception:
        pass
    return result


# ─────────────────────────────────────────────
# JSONL event readers
# ─────────────────────────────────────────────

def read_regime_events(limit: int = 100, symbol: Optional[str] = None) -> list[dict]:
    events = _read_jsonl_tail(LOGS_DIR / "regime_transitions.jsonl", limit=limit, symbol_filter=symbol)
    normalized: list[dict] = []
    for event in events:
        item = dict(event)
        ts = item.get("ts")
        if isinstance(ts, (int, float)):
            # Normalize epoch seconds/milliseconds into an ISO-8601 UTC string
            # so FastAPI response validation remains stable.
            epoch = float(ts)
            if epoch > 1_000_000_000_000:
                epoch /= 1000.0
            try:
                item["ts"] = datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat()
            except Exception:
                item["ts"] = str(ts)
        normalized.append(item)
    return normalized


def read_signal_events(limit: int = 100, symbol: Optional[str] = None) -> list[dict]:
    return _read_jsonl_tail(LOGS_DIR / "alpha_gate.jsonl", limit=limit, symbol_filter=symbol)


def read_stability_events(limit: int = 100, symbol: Optional[str] = None) -> list[dict]:
    return _read_jsonl_tail(LOGS_DIR / "signal_stability.jsonl", limit=limit, symbol_filter=symbol)


def read_quality_events(limit: int = 100, symbol: Optional[str] = None) -> list[dict]:
    return _read_jsonl_tail(LOGS_DIR / "data_quality.jsonl", limit=limit, symbol_filter=symbol)


# ─────────────────────────────────────────────
# Log file listing & tail
# ─────────────────────────────────────────────

_EXCLUDE_DIRS = {"archive", "test_tmp", "pids", "__pycache__"}
_ALLOWED_EXTS = {".log", ".jsonl", ".txt", ".json"}
_LOG_LIST_CACHE_TTL_SEC = float(os.environ.get("DASHBOARD_LOG_LIST_CACHE_TTL_SEC", "8") or "8")
_LOG_LIST_MAX_FILES = int(os.environ.get("DASHBOARD_LOG_LIST_MAX_FILES", "500") or "500")
_LOG_TAIL_MAX_BYTES = int(os.environ.get("DASHBOARD_LOG_TAIL_MAX_BYTES", str(2 * 1024 * 1024)) or str(2 * 1024 * 1024))
_log_list_cache: tuple[float, list[dict[str, Any]]] = (0.0, [])
_log_list_cache_last_hit: bool = False
_tail_last_source: str = "init"
_RATE_LIMIT_RE = re.compile(r"\[RATE_LIMIT\]\s+used_1m=(\d+)\s+cap_1m=(\d+)\s+usage_pct=([\d.]+)", re.IGNORECASE)
_LIQ_ALERT_STATE_PATH = _env_path("LIQ_ALERT_STATE_JSON", REPO_ROOT / "reports" / "LIQUIDATION_ALERT_STATE_REAL.json")
_SPREAD_STRESS_STATE_PATH = _env_path("SPREAD_STRESS_STATE_JSON", REPO_ROOT / "reports" / "SPREAD_STRESS_STATE_REAL.json")
_SPREAD_STRESS_WATCHLIST_PATH = _env_path("SPREAD_STRESS_WATCHLIST_JSON", REPO_ROOT / "reports" / "SPREAD_STRESS_WATCHLIST_REAL.json")
_FILL_TOXICITY_STATE_PATH = _env_path("FILL_TOXICITY_STATE_JSON", REPO_ROOT / "reports" / "FILL_TOXICITY_STATE_REAL.json")
_LATENCY_STRESS_STATE_PATH = _env_path("LATENCY_STRESS_STATE_JSON", REPO_ROOT / "reports" / "LATENCY_STRESS_STATE_REAL.json")
_WATCHBOARD_STATE_PATH = _env_path("WATCHBOARD_STATE_JSON", REPO_ROOT / "reports" / "RESEARCH_EVENT_WATCHBOARD_REAL.json")
_BOOK_PROXY_PRESSURE_STATE_PATH = _env_path("BOOK_PROXY_PRESSURE_STATE_JSON", REPO_ROOT / "reports" / "BOOK_PROXY_PRESSURE_STATE_REAL.json")
_RETURN_SHOCK_STATE_PATH = _env_path("RETURN_SHOCK_STATE_JSON", REPO_ROOT / "reports" / "RETURN_SHOCK_STATE_REAL.json")
_VOLATILITY_BURST_STATE_PATH = _env_path("VOLATILITY_BURST_STATE_JSON", REPO_ROOT / "reports" / "VOLATILITY_BURST_STATE_REAL.json")
_VOLUME_VACUUM_STATE_PATH = _env_path("VOLUME_VACUUM_STATE_JSON", REPO_ROOT / "reports" / "VOLUME_VACUUM_STATE_REAL.json")
_OPS_HEALTH_HISTORY_PATH = LOGS_DIR / "ops_health_history.jsonl"
_OPS_HEALTH_HISTORY_APPEND_SEC = float(os.environ.get("OPS_HEALTH_HISTORY_APPEND_SEC", "60") or "60")
_ops_health_last_append_ts: float = 0.0
_LIVE_METRICS_CACHE_TTL = float(os.environ.get("LIVE_METRICS_CACHE_TTL_SEC", "2") or "2")
_live_metrics_cache: dict[str, Any] = {}
_live_metrics_cache_ts: float = 0.0
_live_trades_series: deque[float] = deque(maxlen=30)
_live_fills_series: deque[float] = deque(maxlen=30)
_MARKET_CHART_CACHE_TTL = float(os.environ.get("MARKET_CHART_CACHE_TTL_SEC", "15") or "15")
_market_chart_cache: dict[tuple[str, str, int], tuple[float, dict[str, Any]]] = {}


def _tail_lines_fast(path: Path, n: int = 200, max_bytes: int = _LOG_TAIL_MAX_BYTES) -> list[str]:
    """Efficient reverse-tail reader bounded by max_bytes."""
    global _tail_last_source
    if n <= 0:
        _tail_last_source = "fast_empty"
        return []
    try:
        file_size = path.stat().st_size
        if file_size <= 0:
            _tail_last_source = "fast_empty"
            return []
        window = min(file_size, max(4096, max_bytes))
        with open(path, "rb") as f:
            f.seek(file_size - window)
            chunk = f.read(window)
        text = chunk.decode("utf-8", errors="replace")
        lines = text.splitlines()
        if len(lines) > n:
            lines = lines[-n:]
        _tail_last_source = "fast"
        return lines
    except Exception:
        _tail_last_source = "fallback"
        return _tail_lines(path, n=n)


def list_log_files() -> list[dict[str, Any]]:
    """List available log/jsonl files in logs/ (excluding archive/test dirs)."""
    global _log_list_cache, _log_list_cache_last_hit
    now = time.time()
    cache_ts, cache_rows = _log_list_cache
    if cache_rows and (now - cache_ts) <= max(0.5, _LOG_LIST_CACHE_TTL_SEC):
        _log_list_cache_last_hit = True
        return cache_rows
    _log_list_cache_last_hit = False

    files: list[dict[str, Any]] = []
    try:
        for p in LOGS_DIR.iterdir():
            if p.is_dir() and p.name in _EXCLUDE_DIRS:
                continue
            if p.is_file() and p.suffix in _ALLOWED_EXTS:
                try:
                    stat = p.stat()
                    files.append({
                        "name": p.name,
                        "path": str(p.relative_to(REPO_ROOT)),
                        "size_bytes": stat.st_size,
                        "mtime": stat.st_mtime,
                    })
                    if len(files) >= max(10, _LOG_LIST_MAX_FILES):
                        break
                except Exception:
                    pass
    except Exception:
        pass
    out = sorted(files, key=lambda x: x["mtime"], reverse=True)
    _log_list_cache = (now, out)
    return out


def tail_log_file(filename: str, limit: int = 200) -> list[str]:
    """Return last `limit` lines from a file in logs/."""
    global _tail_last_source
    path = _safe_path(filename, LOGS_DIR)
    if path is None or not path.exists():
        _tail_last_source = "missing"
        return []
    return _tail_lines_fast(path, n=limit)


def log_list_last_cache_hit() -> bool:
    return _log_list_cache_last_hit


def log_tail_last_source() -> str:
    return _tail_last_source


# ─────────────────────────────────────────────
# Config reader (masked)
# ─────────────────────────────────────────────

def read_config_entries() -> list[dict[str, Any]]:
    """Read .env + runtime env vars; mask sensitive values."""
    entries: list[dict[str, Any]] = []

    env_path = REPO_ROOT / ".env"
    try:
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    k, _, v = line.partition("=")
                    k, v = k.strip(), v.strip()
                    sensitive = _is_sensitive(k)
                    entries.append({
                        "key": k,
                        "value": _mask(v) if sensitive else v,
                        "sensitive": sensitive,
                        "source": "env_file",
                    })
    except Exception:
        pass

    SHOW_ENV = [
        "SCALPER_DRY_RUN", "ACTIVE_SYMBOLS", "SCALPER_SIGNAL_PROFILE",
        "SCALPER_ENHANCED", "SCALPER_QUALITY_MODE", "SCALPER_QUALITY_CONF_MIN",
        "FIRST_LIVE_SAFE", "FIRST_LIVE_SYMBOLS", "FIRST_LIVE_MAX_NOTIONAL_USDT",
    ]
    for k in SHOW_ENV:
        v = os.environ.get(k)
        if v is not None:
            entries.append({
                "key": k,
                "value": v,
                "sensitive": False,
                "source": "runtime_env",
            })

    return entries


# ─────────────────────────────────────────────
# Runtime status helpers
# ─────────────────────────────────────────────

def _parse_collector_log() -> dict[str, Any]:
    """
    Tail collector log and extract the most recent stats line.

    Resilience guarantees:
    - alive + last_log_ts are always derived from the file's mtime when the
      file exists, independent of whether the regex matches.
    - Rates are None (not 0.0) when the regex doesn't match — the frontend
      can distinguish "no data" from "genuinely zero".
    - A file-read or parse error leaves alive/last_log_ts intact and rates
      as None; it never propagates.
    """
    result: dict[str, Any] = {
        "alive": False,
        "last_log_ts": None,
        "uptime_sec": None,
        "trades_per_sec_60s": None,
        "mark_per_sec_60s": None,
        "liquidations_per_sec_60s": None,
    }
    try:
        path = _get_collector_log()
        if not path.exists():
            return result

        stat = path.stat()
        age_sec = time.time() - stat.st_mtime
        result["last_log_ts"] = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()
        result["alive"] = age_sec < _COLLECTOR_ALIVE_MAX_AGE

        # Read only the last 6 KB — sufficient for multiple recent stat lines.
        file_size = stat.st_size
        read_from = max(0, file_size - 6144)
        last_match: re.Match | None = None

        try:
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                if read_from > 0:
                    fh.seek(read_from)
                    fh.readline()  # discard potentially partial first line
                for line in fh:
                    m = _STATS_RE.search(line)
                    if m:
                        last_match = m
        except Exception:
            # File-read error: alive/last_log_ts already set; rates stay None.
            return result

        if last_match:
            try:
                result["uptime_sec"] = round(float(last_match.group(2)) * 3600)
                result["trades_per_sec_60s"] = float(last_match.group(3).replace(",", ""))
                result["mark_per_sec_60s"] = float(last_match.group(4).replace(",", ""))
                result["liquidations_per_sec_60s"] = float(last_match.group(5).replace(",", ""))
            except (ValueError, IndexError):
                pass  # partial parse: rates stay None

    except Exception:
        pass

    return result


def _pick_ts_column(conn: sqlite3.Connection) -> tuple[str | None, str]:
    """
    Discover the best timestamp column in agg_trades via PRAGMA table_info.

    Priority: ts_ms (epoch-ms int) > ts_utc (ISO str) > ts (ISO str) > timestamp (ISO str)

    Returns (column_name, kind) where kind is 'epoch_ms' or 'iso_str'.
    Returns (None, '') when the table is absent or no recognised column exists.
    PRAGMA table_info returns zero rows for a non-existent table (no exception).
    """
    try:
        rows = conn.execute("PRAGMA table_info(agg_trades)").fetchall()
    except Exception:
        return None, ""

    if not rows:
        # Zero rows → table does not exist in this DB file.
        return None, ""

    # PRAGMA table_info row layout: (cid, name, type, notnull, dflt_value, pk)
    col_names = {row[1].lower() for row in rows}

    for candidate in _TS_COL_PRIORITY:
        if candidate in col_names:
            kind = "epoch_ms" if candidate == "ts_ms" else "iso_str"
            return candidate, kind

    return None, ""


def _parse_iso_ts(raw: str) -> datetime | None:
    """
    Safely parse an ISO 8601 timestamp string to an aware datetime (UTC).

    Handles:
    - Trailing 'Z' (not accepted by fromisoformat before Python 3.11)
    - Naive timestamps (assumed UTC)
    - Fractional seconds

    Returns None on any parse failure instead of raising.
    """
    try:
        s = str(raw).strip().replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
        return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)
    except Exception:
        return None


def _query_db_freshness() -> dict[str, Any]:
    """
    Query the DB for the most recent trade timestamp.

    Schema-resilience guarantees:
    - PRAGMA table_info(agg_trades) selects the timestamp column; no hardcoded
      column name ever appears in a SELECT.
    - Missing table → STALE (no exception).
    - Missing recognised column → STALE (no exception).
    - epoch-ms integer parse failure → DEGRADED.
    - ISO 8601 string parse failure → DEGRADED.
    - Any unexpected exception → STALE (outer try/except, never 500).
    """
    result: dict[str, Any] = {
        "last_trade_ts": None,
        "seconds_since_last_trade": None,
        "status": "STALE",
    }
    db_path = _get_db_path()
    try:
        if not db_path.exists():
            return result

        # as_posix() converts Windows backslashes for the SQLite URI.
        uri = f"file:{db_path.as_posix()}?mode=ro"
        conn = sqlite3.connect(uri, uri=True, timeout=2.0, check_same_thread=False)
        try:
            # Explicit table-existence check via sqlite_master (readable in mode=ro).
            tbl = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='agg_trades'"
            ).fetchone()
            if tbl is None:
                return result

            col, kind = _pick_ts_column(conn)
            if col is None:
                return result

            # col is one of the hardcoded _TS_COL_PRIORITY identifiers — not user input.
            row = conn.execute(f'SELECT MAX("{col}") FROM agg_trades').fetchone()
        finally:
            conn.close()

        if not row or row[0] is None:
            return result

        raw_ts = row[0]

        if kind == "epoch_ms":
            try:
                dt: datetime | None = datetime.fromtimestamp(int(raw_ts) / 1000.0, tz=timezone.utc)
            except Exception:
                result["status"] = "DEGRADED"
                return result
        else:
            dt = _parse_iso_ts(str(raw_ts))
            if dt is None:
                result["status"] = "DEGRADED"
                return result

        result["last_trade_ts"] = dt.isoformat()
        age = (datetime.now(tz=timezone.utc) - dt).total_seconds()
        result["seconds_since_last_trade"] = round(age, 1)

        if age < _FRESHNESS_DEGRADED:
            result["status"] = "LIVE"
        elif age < _FRESHNESS_STALE:
            result["status"] = "DEGRADED"
        else:
            result["status"] = "STALE"

    except Exception:
        pass

    return result


def _db_file_stats() -> dict[str, Any]:
    """Return DB file size, last-write timestamp, and 5-minute byte growth."""
    db_path = _get_db_path()
    result: dict[str, Any] = {
        "path": str(db_path),
        "size_bytes": 0,
        "last_write_ts": None,
        "growth_bytes_5min": 0,
    }
    try:
        if not db_path.exists():
            return result

        stat = db_path.stat()
        size_now = stat.st_size
        now_mono = time.monotonic()

        result["size_bytes"] = size_now
        result["last_write_ts"] = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()

        _db_size_history.append((now_mono, size_now))

        # Walk deque (oldest-first) and keep updating old_size for every sample
        # that is still >= 5 minutes old — this gives us the most-recent such sample.
        cutoff = now_mono - 300.0
        old_size: int | None = None
        for ts, sz in _db_size_history:
            if ts <= cutoff:
                old_size = sz
        if old_size is not None:
            result["growth_bytes_5min"] = max(0, size_now - old_size)

    except Exception:
        pass

    return result


def read_runtime_status() -> dict[str, Any]:
    """Assemble collector + DB + freshness + system status. Cached for 1 s."""
    global _runtime_cache, _runtime_cache_ts

    now = time.monotonic()
    if now - _runtime_cache_ts < _RUNTIME_CACHE_TTL and _runtime_cache:
        return _runtime_cache

    _runtime_cache = {
        "collector":      _parse_collector_log(),
        "database":       _db_file_stats(),
        "data_freshness": _query_db_freshness(),
        "system": {
            "server_time":    datetime.now(tz=timezone.utc).isoformat(),
            "python_version": sys.version.split()[0],
            "platform":       platform.platform(terse=True),
        },
    }
    _runtime_cache_ts = now
    return _runtime_cache


def _safe_iso_from_mtime(path: Path) -> str | None:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()
    except Exception:
        return None


def _backup_stats() -> dict[str, Any]:
    backup_dir = _env_path("DB_BACKUP_DIR", DATA_DIR / "backups")
    out = {
        "backup_dir": str(backup_dir),
        "backup_count": 0,
        "last_backup_ts": None,
        "backup_age_sec": None,
    }
    try:
        if not backup_dir.exists():
            return out
        files = [p for p in backup_dir.glob("*.db") if p.is_file()]
        if not files:
            return out
        files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        last = files[0]
        out["backup_count"] = len(files)
        last_ts = _safe_iso_from_mtime(last)
        out["last_backup_ts"] = last_ts
        out["backup_age_sec"] = round(max(0.0, time.time() - last.stat().st_mtime), 1)
    except Exception:
        pass
    return out


def _maintenance_stats() -> dict[str, Any]:
    # best-effort; if dedicated maintenance log exists, use mtime as checkpoint activity marker
    log_path = LOGS_DIR / "db_maintenance.log"
    out = {"last_maintenance_ts": None, "maintenance_age_sec": None}
    try:
        if not log_path.exists():
            return out
        out["last_maintenance_ts"] = _safe_iso_from_mtime(log_path)
        out["maintenance_age_sec"] = round(max(0.0, time.time() - log_path.stat().st_mtime), 1)
    except Exception:
        pass
    return out


def _disk_and_wal_stats() -> dict[str, Any]:
    db_path = _get_db_path()
    wal_path = Path(str(db_path) + "-wal")
    out = {
        "db_path": str(db_path),
        "db_size_bytes": 0,
        "wal_path": str(wal_path),
        "wal_size_bytes": 0,
        "wal_ratio": 0.0,
        "disk_free_gb": None,
        "disk_total_gb": None,
    }
    try:
        if db_path.exists():
            out["db_size_bytes"] = int(db_path.stat().st_size)
        if wal_path.exists():
            out["wal_size_bytes"] = int(wal_path.stat().st_size)
        db_size = float(out["db_size_bytes"] or 0)
        wal_size = float(out["wal_size_bytes"] or 0)
        out["wal_ratio"] = round((wal_size / db_size), 4) if db_size > 0 else 0.0
        du = shutil.disk_usage(str(REPO_ROOT))
        out["disk_free_gb"] = round(float(du.free) / (1024.0 ** 3), 2)
        out["disk_total_gb"] = round(float(du.total) / (1024.0 ** 3), 2)
    except Exception:
        pass
    return out


def _health_overall_stats() -> dict[str, Any]:
    out = {
        "collector_connected": None,
        "reconnects_last_5m": 0,
        "errors_last_5m": 0,
    }
    path = LOGS_DIR / "health" / "overall.json"
    try:
        if not path.exists():
            return out
        payload = _safe_json(path)
        comps = payload.get("components") if isinstance(payload.get("components"), dict) else {}
        collector = comps.get("collector") if isinstance(comps.get("collector"), dict) else {}
        out["collector_connected"] = collector.get("connected")
        out["reconnects_last_5m"] = int(collector.get("reconnects_last_5m", 0) or 0)
        out["errors_last_5m"] = int(collector.get("errors_last_5m", 0) or 0)
    except Exception:
        pass
    return out


def _rate_limit_stats() -> dict[str, Any]:
    out: dict[str, Any] = {
        "used_1m": None,
        "cap_1m": None,
        "usage_pct": None,
        "samples": [],
    }
    path = LOGS_DIR / "paper_trading.log"
    try:
        lines = _tail_lines(path, n=1200)
        samples: list[dict[str, Any]] = []
        for line in lines:
            m = _RATE_LIMIT_RE.search(line)
            if not m:
                continue
            sample = {
                "used_1m": int(m.group(1)),
                "cap_1m": int(m.group(2)),
                "usage_pct": float(m.group(3)),
            }
            samples.append(sample)
        if samples:
            out["samples"] = samples[-30:]
            last = samples[-1]
            out["used_1m"] = last["used_1m"]
            out["cap_1m"] = last["cap_1m"]
            out["usage_pct"] = last["usage_pct"]
    except Exception:
        pass
    return out


def read_ops_health() -> dict[str, Any]:
    disk = _disk_and_wal_stats()
    backup = _backup_stats()
    maint = _maintenance_stats()
    health = _health_overall_stats()
    rate = _rate_limit_stats()

    disk_free_min = float(os.environ.get("OPS_DISK_FREE_MIN_GB", "10") or "10")
    wal_warn_mb = float(os.environ.get("OPS_WAL_WARN_MB", "2048") or "2048")
    backup_warn_sec = float(os.environ.get("OPS_BACKUP_WARN_SEC", str(24 * 3600)) or str(24 * 3600))
    reconnect_warn_5m = int(os.environ.get("OPS_RECONNECT_WARN_5M", "10") or "10")
    rate_warn_pct = float(os.environ.get("OPS_RATE_WARN_PCT", "80") or "80")

    wal_mb = float(disk.get("wal_size_bytes") or 0) / (1024.0 ** 2)
    disk_free_gb = float(disk.get("disk_free_gb") or 0.0)
    backup_age = float(backup.get("backup_age_sec") or 1e18)
    reconnects_5m = int(health.get("reconnects_last_5m", 0) or 0)
    usage_pct = float(rate.get("usage_pct") or 0.0)

    data_status = "ok"
    net_status = "ok"
    if disk.get("disk_free_gb") is not None and disk_free_gb < disk_free_min:
        data_status = "critical"
    elif wal_mb > wal_warn_mb or backup_age > backup_warn_sec:
        data_status = "warning"

    if reconnects_5m >= reconnect_warn_5m:
        net_status = "warning"
    if usage_pct >= rate_warn_pct:
        net_status = "warning" if net_status == "ok" else net_status

    payload = {
        "ts_utc": datetime.now(tz=timezone.utc).isoformat(),
        "status": {
            "data_integrity": data_status,
            "network": net_status,
        },
        "thresholds": {
            "disk_free_min_gb": disk_free_min,
            "wal_warn_mb": wal_warn_mb,
            "backup_warn_sec": backup_warn_sec,
            "reconnect_warn_5m": reconnect_warn_5m,
            "rate_warn_pct": rate_warn_pct,
        },
        "data_integrity": {
            **disk,
            **backup,
            **maint,
        },
        "network": {
            **health,
            **rate,
        },
    }
    _append_ops_health_history(payload)
    payload["history"] = _read_ops_health_history(limit=int(os.environ.get("OPS_HEALTH_HISTORY_LIMIT", "120") or "120"))
    return payload


def read_connectivity_diag() -> dict[str, Any]:
    """Best-effort connectivity diagnostics for dashboard troubleshooting."""
    now = time.time()
    db_path = _get_db_path()
    collector_log = _get_collector_log()
    overall_health = LOGS_DIR / "health" / "overall.json"
    paper_log = LOGS_DIR / "paper_trading.log"

    def _path_diag(path: Path) -> dict[str, Any]:
        exists = path.exists()
        out: dict[str, Any] = {
            "path": str(path),
            "exists": exists,
            "is_file": path.is_file() if exists else False,
            "readable": False,
            "size_bytes": None,
            "age_sec": None,
        }
        if not exists:
            return out
        try:
            stat = path.stat()
            out["size_bytes"] = int(stat.st_size)
            out["age_sec"] = round(max(0.0, now - stat.st_mtime), 1)
        except Exception:
            pass
        try:
            with open(path, "rb") as f:
                _ = f.read(1)
            out["readable"] = True
        except Exception:
            out["readable"] = False
        return out

    logs_dir_diag: dict[str, Any] = {
        "path": str(LOGS_DIR),
        "exists": LOGS_DIR.exists(),
        "writable": False,
    }
    if LOGS_DIR.exists():
        try:
            probe = LOGS_DIR / ".dashboard_write_probe"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink(missing_ok=True)
            logs_dir_diag["writable"] = True
        except Exception:
            logs_dir_diag["writable"] = False

    items = {
        "db": _path_diag(db_path),
        "collector_log": _path_diag(collector_log),
        "paper_log": _path_diag(paper_log),
        "overall_health": _path_diag(overall_health),
    }

    hints: list[str] = []
    if not bool(logs_dir_diag.get("exists")):
        hints.append("logs directory missing")
    if bool(logs_dir_diag.get("exists")) and not bool(logs_dir_diag.get("writable")):
        hints.append("logs directory not writable")
    if not bool(items["db"].get("exists")):
        hints.append("microstructure db missing")
    if bool(items["collector_log"].get("exists")) and (items["collector_log"].get("age_sec") or 0) > 120:
        hints.append("collector log stale >120s")
    if bool(items["paper_log"].get("exists")) and (items["paper_log"].get("age_sec") or 0) > 120:
        hints.append("paper_trading log stale >120s")
    if not bool(items["overall_health"].get("exists")):
        hints.append("overall health file missing")

    status = "ok" if not hints else "degraded"
    return {
        "ts_utc": datetime.now(tz=timezone.utc).isoformat(),
        "status": status,
        "logs_dir": logs_dir_diag,
        "items": items,
        "hints": hints,
    }


def _append_ops_health_history(payload: dict[str, Any]) -> None:
    global _ops_health_last_append_ts
    now = time.time()
    if (now - float(_ops_health_last_append_ts or 0.0)) < max(5.0, _OPS_HEALTH_HISTORY_APPEND_SEC):
        return
    row = {
        "ts_utc": payload.get("ts_utc"),
        "data_integrity": {
            "status": ((payload.get("status") or {}).get("data_integrity") if isinstance(payload.get("status"), dict) else None),
            "disk_free_gb": ((payload.get("data_integrity") or {}).get("disk_free_gb") if isinstance(payload.get("data_integrity"), dict) else None),
            "wal_size_bytes": ((payload.get("data_integrity") or {}).get("wal_size_bytes") if isinstance(payload.get("data_integrity"), dict) else None),
        },
        "network": {
            "status": ((payload.get("status") or {}).get("network") if isinstance(payload.get("status"), dict) else None),
            "reconnects_last_5m": ((payload.get("network") or {}).get("reconnects_last_5m") if isinstance(payload.get("network"), dict) else None),
            "usage_pct": ((payload.get("network") or {}).get("usage_pct") if isinstance(payload.get("network"), dict) else None),
        },
    }
    try:
        _OPS_HEALTH_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(_OPS_HEALTH_HISTORY_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        _ops_health_last_append_ts = now
    except Exception:
        pass


def _read_ops_health_history(limit: int = 120) -> list[dict[str, Any]]:
    if limit <= 0:
        return []
    _TAIL_BYTES = 512 * 1024  # 512 KB — enough for ~120 recent entries
    rows: list[dict[str, Any]] = []
    try:
        file_size = _OPS_HEALTH_HISTORY_PATH.stat().st_size
        with open(_OPS_HEALTH_HISTORY_PATH, "r", encoding="utf-8", errors="replace") as f:
            if file_size > _TAIL_BYTES:
                f.seek(file_size - _TAIL_BYTES)
                f.readline()  # skip partial first line
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        rows.append(obj)
                except Exception:
                    continue
    except Exception:
        return []
    return rows[-limit:]


def read_supervisor_status() -> dict[str, Any]:
    """Read dashboard backend supervisor status from runtime/log artifacts."""
    now = time.time()
    runtime_path = REPO_ROOT / "runtime" / "dashboard_backend.json"
    lock_path = REPO_ROOT / "runtime" / "dashboard_launcher.lock"
    sup_log = LOGS_DIR / "dashboard_backend_supervisor.log"

    out: dict[str, Any] = {
        "ts_utc": datetime.now(tz=timezone.utc).isoformat(),
        "supervisor_log_path": str(sup_log),
        "supervisor_running": False,
        "backend_runtime_present": False,
        "backend_pid": None,
        "backend_host": None,
        "backend_port": None,
        "backend_runtime_age_sec": None,
        "last_event": None,
        "last_event_age_sec": None,
        "restarts_last_1h": 0,
    }

    if runtime_path.exists():
        try:
            payload = _safe_json(runtime_path)
            out["backend_runtime_present"] = True
            out["backend_pid"] = payload.get("pid")
            out["backend_host"] = payload.get("host") or os.environ.get("DASHBOARD_HOST", "127.0.0.1")
            out["backend_port"] = payload.get("port") or int(os.environ.get("DASHBOARD_PORT", "8765") or "8765")
            out["backend_runtime_age_sec"] = round(max(0.0, now - runtime_path.stat().st_mtime), 1)
        except Exception:
            pass

    # Heuristic: lock file exists if launcher active. Supervisor can also run without launcher.
    if lock_path.exists():
        out["supervisor_running"] = True

    if sup_log.exists():
        lines = _tail_lines(sup_log, n=400)
        if lines:
            # Last non-empty line as current state event
            for i in range(len(lines) - 1, -1, -1):
                line = lines[i].strip()
                if line:
                    out["last_event"] = line
                    break
            try:
                out["last_event_age_sec"] = round(max(0.0, now - sup_log.stat().st_mtime), 1)
            except Exception:
                pass
            last_started_idx = -1
            last_stopped_idx = -1
            cutoff = now - 3600.0
            count = 0
            for idx, line in enumerate(lines):
                low = line.lower()
                if "started backend pid=" in low:
                    last_started_idx = idx
                if "supervisor stopped" in low:
                    last_stopped_idx = idx
                if "restarted backend pid=" not in line:
                    continue
                # format: [YYYY-mm-dd HH:MM:SS] ...
                if line.startswith("[") and "]" in line:
                    stamp = line[1:line.find("]")]
                    try:
                        dt = datetime.strptime(stamp, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
                        if dt.timestamp() >= cutoff:
                            count += 1
                        continue
                    except Exception:
                        pass
                count += 1
            out["restarts_last_1h"] = count
            # Running iff latest lifecycle event is a "started" after any "stopped".
            out["supervisor_running"] = last_started_idx > last_stopped_idx

    return out


# ─────────────────────────────────────────────
# Overview aggregation
# ─────────────────────────────────────────────

def read_live_monitor_tests_status(limit: int = 80) -> dict[str, Any]:
    """Read last run status for tools/run_live_monitor_tests.ps1."""
    now = time.time()
    status_path = REPO_ROOT / "runtime" / "live_monitor_tests_status.json"
    default_log = LOGS_DIR / "live_monitor_tests.log"
    out: dict[str, Any] = {
        "ts_utc": datetime.now(tz=timezone.utc).isoformat(),
        "state": "unknown",
        "stage": "unknown",
        "message": "no test run status yet",
        "strict_mode": False,
        "backend_ok": False,
        "frontend_typecheck_ok": False,
        "frontend_smoke_ok": False,
        "frontend_smoke_skipped": False,
        "pid": None,
        "run_command": "powershell -NoProfile -ExecutionPolicy Bypass -File .\\tools\\run_live_monitor_tests.ps1",
        "log_path": str(default_log),
        "status_path": str(status_path),
        "status_age_sec": None,
        "log_tail": _tail_lines(default_log, n=limit),
    }
    if not status_path.exists():
        return out
    payload = _safe_json(status_path)
    if not isinstance(payload, dict) or not payload:
        out["message"] = "status file exists but unreadable"
        return out

    out["ts_utc"] = str(payload.get("ts_utc") or out["ts_utc"])
    out["state"] = str(payload.get("state") or out["state"])
    out["stage"] = str(payload.get("stage") or out["stage"])
    out["message"] = str(payload.get("message") or out["message"])
    out["strict_mode"] = bool(payload.get("strict_mode"))
    out["backend_ok"] = bool(payload.get("backend_ok"))
    out["frontend_typecheck_ok"] = bool(payload.get("frontend_typecheck_ok"))
    out["frontend_smoke_ok"] = bool(payload.get("frontend_smoke_ok"))
    out["frontend_smoke_skipped"] = bool(payload.get("frontend_smoke_skipped"))

    pid_raw = payload.get("pid")
    if isinstance(pid_raw, (int, float)):
        out["pid"] = int(pid_raw)

    run_command = payload.get("run_command")
    if isinstance(run_command, str) and run_command.strip():
        out["run_command"] = run_command.strip()

    custom_log_path: Path | None = None
    log_raw = payload.get("log_path")
    if isinstance(log_raw, str) and log_raw.strip():
        custom_log_path = Path(log_raw.strip())
        out["log_path"] = str(custom_log_path)

    status_raw = payload.get("status_path")
    if isinstance(status_raw, str) and status_raw.strip():
        out["status_path"] = status_raw.strip()

    ts_obj = _parse_iso_ts(out["ts_utc"])
    if ts_obj is not None:
        out["status_age_sec"] = round(max(0.0, now - ts_obj.timestamp()), 1)

    if custom_log_path is not None:
        out["log_tail"] = _tail_lines(custom_log_path, n=limit)
    return out


def _pick_paper_file() -> str:
    files = list_log_files()
    names = [str(x.get("name", "")) for x in files]
    for preferred in ("paper_trades.jsonl", "paper_start_e2e.log", "run-bot-output.log"):
        if preferred in names:
            return preferred
    for n in names:
        low = n.lower()
        if "paper" in low and (low.endswith(".log") or low.endswith(".jsonl")):
            return n
    return "paper_trades.jsonl"


def _parse_num(v: Any) -> float | None:
    try:
        if v is None:
            return None
        if isinstance(v, (int, float)):
            return float(v)
        s = str(v).strip()
        if not s:
            return None
        return float(s)
    except Exception:
        return None


def _parse_ts_ms(v: Any) -> int | None:
    try:
        if v is None:
            return None
        if isinstance(v, (int, float)):
            n = float(v)
            if n <= 0:
                return None
            return int(n if n > 10_000_000_000 else n * 1000.0)
        s = str(v).strip()
        if not s:
            return None
        n = float(s)
        if n > 0:
            return int(n if n > 10_000_000_000 else n * 1000.0)
    except Exception:
        pass
    try:
        dt = _parse_iso_ts(str(v))
        if dt is None:
            return None
        return int(dt.timestamp() * 1000.0)
    except Exception:
        return None


def _parse_fill_from_json_obj(obj: dict[str, Any]) -> dict[str, Any] | None:
    typ = str(obj.get("type") or obj.get("event") or obj.get("action") or obj.get("status") or "").lower()
    is_fill = (
        "fill" in typ
        or "filled" in typ
        or obj.get("fill_price") is not None
        or obj.get("avg_fill_price") is not None
    )
    if not is_fill:
        return None
    ts_raw = obj.get("ts") or obj.get("ts_utc") or obj.get("time") or obj.get("timestamp") or obj.get("ts_ms")
    ts_ms = _parse_ts_ms(ts_raw)
    ts = datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc).isoformat() if ts_ms else str(ts_raw or "-")
    pnl_num = _parse_num(obj.get("pnl") or obj.get("pnl_bps") or obj.get("net_pnl_bps"))
    delay_ms = _parse_num(obj.get("fill_delay_ms") or obj.get("latency_ms") or obj.get("delay_ms") or obj.get("time_to_fill_ms"))
    adverse_bps = _parse_num(obj.get("adverse_bps") or obj.get("mae_bps") or obj.get("adverse_selection_bps"))
    return {
        "ts": ts,
        "symbol": str(obj.get("symbol") or obj.get("sym") or "-"),
        "side": str(obj.get("side") or obj.get("direction") or "-"),
        "price": str(obj.get("fill_price") or obj.get("avg_fill_price") or obj.get("price") or obj.get("entry_price") or obj.get("exit_price") or "-"),
        "qty": str(obj.get("qty") or obj.get("size") or obj.get("filled_qty") or obj.get("amount") or "-"),
        "pnl": str(obj.get("pnl") or obj.get("pnl_bps") or obj.get("net_pnl_bps") or "-"),
        "ts_ms": ts_ms,
        "pnl_num": pnl_num,
        "delay_ms": delay_ms,
        "adverse_bps": adverse_bps,
    }


def _parse_fill_from_text(line: str) -> dict[str, Any] | None:
    low = line.lower()
    if "fill" not in low:
        return None

    def pick1(rx: str) -> str:
        m = re.search(rx, line, re.IGNORECASE)
        return m.group(1) if m else "-"

    def pick2(rx: str) -> str:
        m = re.search(rx, line, re.IGNORECASE)
        return m.group(2) if m else "-"

    ts_guess = line[:24]
    ts_ms = _parse_ts_ms(ts_guess)
    return {
        "ts": line[:19] if line else "-",
        "symbol": pick1(r"\b([A-Z]{3,}USDT)\b"),
        "side": pick2(r"\b(side|direction)=([a-zA-Z]+)"),
        "price": pick2(r"\b(price|fill_price|avg_fill_price)=([0-9.]+)"),
        "qty": pick2(r"\b(qty|size|amount|filled_qty)=([0-9.]+)"),
        "pnl": pick2(r"\b(pnl_bps|pnl|net_pnl_bps)=([\-0-9.]+)"),
        "ts_ms": ts_ms,
        "pnl_num": _parse_num(pick2(r"\b(pnl_bps|pnl|net_pnl_bps)=([\-0-9.]+)")),
        "delay_ms": _parse_num(pick2(r"\b(fill_delay_ms|latency_ms|delay_ms|time_to_fill_ms)=([\-0-9.]+)")),
        "adverse_bps": _parse_num(pick2(r"\b(adverse_bps|mae_bps|adverse_selection_bps)=([\-0-9.]+)")),
    }


def _extract_fill_rows(lines: list[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for line in lines:
        s = line.strip()
        if not s:
            continue
        obj: dict[str, Any] | None = None
        if s.startswith("{") and s.endswith("}"):
            try:
                maybe = json.loads(s)
                if isinstance(maybe, dict):
                    obj = maybe
            except Exception:
                obj = None
        row = _parse_fill_from_json_obj(obj) if obj is not None else _parse_fill_from_text(line)
        if row is not None:
            out.append(row)
    return out


def read_live_metrics() -> dict[str, Any]:
    global _live_metrics_cache, _live_metrics_cache_ts
    now_mono = time.monotonic()
    if (now_mono - _live_metrics_cache_ts) < _LIVE_METRICS_CACHE_TTL and _live_metrics_cache:
        return _live_metrics_cache

    runtime = read_runtime_status()
    scoreboard = read_scoreboard()
    collector = runtime.get("collector") if isinstance(runtime.get("collector"), dict) else {}
    freshness = runtime.get("data_freshness") if isinstance(runtime.get("data_freshness"), dict) else {}

    paper_file = _pick_paper_file()
    short_lines = tail_log_file(paper_file, limit=80)
    long_lines = tail_log_file(paper_file, limit=2000)

    order_count = 0
    fill_count = 0
    blocked_count = 0
    for ln in short_lines:
        low = ln.lower()
        if "order" in low:
            order_count += 1
        if "fill" in low:
            fill_count += 1
        if "blocked" in low or "regime_mismatch" in low or "no_match" in low:
            blocked_count += 1
    fill_per_order = (fill_count / order_count * 100.0) if order_count > 0 else None

    fill_rows = _extract_fill_rows(long_lines)
    last5 = list(reversed(fill_rows[-5:]))

    now_ms = int(time.time() * 1000)
    start_day = datetime.now(tz=timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    start_day_ms = int(start_day.timestamp() * 1000)
    h24_ms = 24 * 3600 * 1000
    d7_ms = 7 * 24 * 3600 * 1000

    def _sum_pnl(window_ms: int | None = None, from_ms: int | None = None) -> float:
        total = 0.0
        for r in fill_rows:
            ts_ms = r.get("ts_ms")
            pnl_num = r.get("pnl_num")
            if not isinstance(pnl_num, (int, float)):
                continue
            if isinstance(from_ms, int):
                if not isinstance(ts_ms, int) or ts_ms < from_ms:
                    continue
            if isinstance(window_ms, int):
                if not isinstance(ts_ms, int) or (now_ms - ts_ms) > window_ms:
                    continue
            total += float(pnl_num)
        return round(total, 6)

    pnl_strip = {
        "today": _sum_pnl(from_ms=start_day_ms),
        "h24": _sum_pnl(window_ms=h24_ms),
        "d7": _sum_pnl(window_ms=d7_ms),
        "sample": len(fill_rows),
    }

    delays = [float(r["delay_ms"]) for r in fill_rows if isinstance(r.get("delay_ms"), (int, float))]
    adverse = [float(r["adverse_bps"]) for r in fill_rows if isinstance(r.get("adverse_bps"), (int, float))]
    fill_quality = {
        "avg_delay_ms": round(sum(delays) / len(delays), 3) if delays else None,
        "avg_adverse_bps": round(sum(adverse) / len(adverse), 4) if adverse else None,
        "with_delay": len(delays),
        "with_adverse": len(adverse),
    }

    blocked_raw = scoreboard.get("blocked_by_reason") if isinstance(scoreboard.get("blocked_by_reason"), dict) else {}
    blocked_reasons = sorted(
        [{"reason": str(k), "count": int(v) if isinstance(v, (int, float)) else int(float(v)) if str(v).strip() else 0} for k, v in blocked_raw.items()],
        key=lambda x: x["count"],
        reverse=True,
    )[:10]

    trade_age_alert_sec = int(os.environ.get("LIVE_TRADE_AGE_ALERT_SEC", "10") or "10")
    fill_flatline_alert_min = int(os.environ.get("LIVE_FILL_FLATLINE_ALERT_MIN", "15") or "15")
    trade_age = float(freshness.get("seconds_since_last_trade") or 99999.0)
    last_fill_ts_raw = scoreboard.get("last_fill_ts")
    last_fill_ms = _parse_ts_ms(last_fill_ts_raw)
    fill_age_min = ((now_ms - last_fill_ms) / 60000.0) if isinstance(last_fill_ms, int) and last_fill_ms > 0 else None
    trade_age_alert = trade_age > trade_age_alert_sec
    fill_flatline_alert = (fill_age_min if fill_age_min is not None else 99999.0) > fill_flatline_alert_min

    trade_rate = float(collector.get("trades_per_sec_60s") or 0.0)
    _live_trades_series.append(trade_rate)
    _live_fills_series.append(float(fill_count))

    _live_metrics_cache = {
        "ts_utc": datetime.now(tz=timezone.utc).isoformat(),
        "runtime": runtime,
        "scoreboard": scoreboard,
        "pnl_strip": pnl_strip,
        "fill_quality": fill_quality,
        "tail_kpis": {
            "window_lines": len(short_lines),
            "order_count": order_count,
            "fill_count": fill_count,
            "blocked_count": blocked_count,
            "fill_per_order_pct": round(fill_per_order, 3) if isinstance(fill_per_order, float) else None,
        },
        "blocked_reasons": blocked_reasons,
        "last_fills": last5,
        "alerts": {
            "any_alert": bool(trade_age_alert or fill_flatline_alert),
            "trade_age_alert": bool(trade_age_alert),
            "fill_flatline_alert": bool(fill_flatline_alert),
            "trade_age_sec": round(trade_age, 3),
            "fill_age_min": round(fill_age_min, 3) if isinstance(fill_age_min, float) else None,
            "config": {
                "trade_age_alert_sec": trade_age_alert_sec,
                "fill_flatline_alert_min": fill_flatline_alert_min,
            },
        },
        "trends": {
            "trades_per_sec": list(_live_trades_series),
            "fills_tail": list(_live_fills_series),
        },
        "paper_file": paper_file,
    }
    _live_metrics_cache_ts = now_mono
    return _live_metrics_cache


def read_market_chart(symbol: str = "BTCUSDT", interval: str = "5m", limit: int = 240) -> dict[str, Any]:
    symbol_clean = str(symbol or "BTCUSDT").upper().strip()
    interval_clean = str(interval or "5m").strip()
    limit_clean = max(50, min(int(limit or 240), 500))
    if symbol_clean not in {"BTCUSDT", "ETHUSDT"}:
        symbol_clean = "BTCUSDT"
    if interval_clean not in {"1m", "5m", "15m", "1h", "4h"}:
        interval_clean = "5m"

    cache_key = (symbol_clean, interval_clean, limit_clean)
    now_mono = time.monotonic()
    cached = _market_chart_cache.get(cache_key)
    if cached is not None and (now_mono - cached[0]) < _MARKET_CHART_CACHE_TTL:
        return cached[1]

    params = urllib.parse.urlencode(
        {
            "symbol": symbol_clean,
            "interval": interval_clean,
            "limit": limit_clean,
        }
    )
    url = f"https://api.binance.com/api/v3/klines?{params}"
    req = urllib.request.Request(url, headers={"User-Agent": "eclipse-scalper-dashboard/1.0"})

    with urllib.request.urlopen(req, timeout=8) as response:
        raw = json.loads(response.read().decode("utf-8"))

    candles: list[dict[str, Any]] = []
    closes: list[float] = []
    for row in raw:
        if not isinstance(row, list) or len(row) < 6:
            continue
        candle = {
            "time": int(float(row[0]) / 1000.0),
            "open": float(row[1]),
            "high": float(row[2]),
            "low": float(row[3]),
            "close": float(row[4]),
            "volume": float(row[5]),
        }
        candles.append(candle)
        closes.append(candle["close"])

    payload = {
        "source": "binance_spot",
        "symbol": symbol_clean,
        "interval": interval_clean,
        "limit": limit_clean,
        "generated_ts": datetime.now(tz=timezone.utc).isoformat(),
        "candles": candles,
        "overlays": [
            {"name": "EMA 20", "values": _ema(closes, 20)},
            {"name": "EMA 50", "values": _ema(closes, 50)},
        ],
        "oscillator": {"name": "RSI 14", "values": _rsi(closes, 14)},
        "pocket_markers": _build_pocket_markers(symbol_clean, interval_clean, candles),
    }
    _market_chart_cache[cache_key] = (now_mono, payload)
    return payload


def build_overview() -> dict[str, Any]:
    raw_gates = read_micro_edge_gates()
    symbols: list[dict] = []
    for sym, data in raw_gates.get("symbols", {}).items():
        rule = data.get("rule_name", "")
        rule_data = data.get("rules_filtered", {}).get(rule, {})
        symbols.append({
            "symbol": sym,
            "rule_name": rule,
            "hit_rate": rule_data.get("hit_rate"),
            "delta_vs_baseline": data.get("delta_vs_baseline"),
            "thresholds": data.get("thresholds", {}),
        })

    return {
        "scoreboard": read_scoreboard(),
        "gates": {
            "generated_utc": raw_gates.get("generated_utc"),
            "lookback_min": raw_gates.get("lookback_min"),
            "symbols": symbols,
        },
        "recent_regimes": read_regime_events(limit=10),
        "exit_quality": read_exit_quality(),
        "preflight": read_preflight(),
        "reliability": read_reliability(),
        "research_events": read_research_events(),
    }


# ---------------------------------------------------------------------------
# Liquidation alert state (monitoring/annotation card)
# ---------------------------------------------------------------------------

def read_liq_alert_state() -> dict[str, Any]:
    """Read the liquidation alert state payload produced by research tools.

    Returns a safe dict even if the file is missing or malformed.
    """
    empty: dict[str, Any] = {
        "available": False,
        "symbol": "",
        "state": {"level": "quiet", "reasons": [], "primary_side_bias": "NEUTRAL", "dominant_severity": "none"},
        "card": {},
        "summary_snapshot": {},
        "alerts": [],
    }
    try:
        if not _LIQ_ALERT_STATE_PATH.exists():
            return empty
        payload = json.loads(_LIQ_ALERT_STATE_PATH.read_text(encoding="utf-8", errors="replace"))
        if not isinstance(payload, dict):
            return empty
        state = payload.get("state") or {}
        card = payload.get("card") or {}
        summary = payload.get("summary_snapshot") or {}
        # Also try to load raw alerts for the table
        alerts: list[dict[str, Any]] = []
        source = str(payload.get("source_json") or "")
        if source:
            source_path = Path(source)
            if not source_path.is_absolute():
                source_path = REPO_ROOT / source_path
            if source_path.exists():
                try:
                    raw = json.loads(source_path.read_text(encoding="utf-8", errors="replace"))
                    alerts = list(raw.get("alerts") or [])[:50]
                except Exception:
                    pass
        # File age for staleness detection
        age_sec = max(0.0, time.time() - _LIQ_ALERT_STATE_PATH.stat().st_mtime)
        return {
            "available": True,
            "stale": age_sec > 600,  # >10 min = stale
            "age_sec": round(age_sec, 1),
            "symbol": str(payload.get("symbol") or ""),
            "rule": str(payload.get("rule") or ""),
            "state": {
                "level": str(state.get("level") or "quiet"),
                "reasons": list(state.get("reasons") or []),
                "primary_side_bias": str(state.get("primary_side_bias") or "NEUTRAL"),
                "dominant_severity": str(state.get("dominant_severity") or "none"),
            },
            "card": dict(card),
            "summary_snapshot": dict(summary),
            "alerts": alerts,
        }
    except Exception:
        return empty


# ---------------------------------------------------------------------------
# Spread stress state + watchlist (monitoring/annotation card)
# ---------------------------------------------------------------------------

def read_spread_stress_state() -> dict[str, Any]:
    """Read spread-stress state payload. Safe dict on missing/malformed."""
    empty: dict[str, Any] = {
        "available": False,
        "symbol": "",
        "state": {"level": "quiet", "reasons": []},
        "card": {},
        "watchlist": None,
    }
    try:
        # Single-symbol state
        state_payload: dict[str, Any] = {}
        if _SPREAD_STRESS_STATE_PATH.exists():
            state_payload = json.loads(_SPREAD_STRESS_STATE_PATH.read_text(encoding="utf-8", errors="replace"))
            if not isinstance(state_payload, dict):
                state_payload = {}

        # Watchlist
        watchlist_payload: dict[str, Any] | None = None
        if _SPREAD_STRESS_WATCHLIST_PATH.exists():
            raw_wl = json.loads(_SPREAD_STRESS_WATCHLIST_PATH.read_text(encoding="utf-8", errors="replace"))
            if isinstance(raw_wl, dict):
                watchlist_payload = {
                    "summary": dict(raw_wl.get("summary") or {}),
                    "top_summary": dict(raw_wl.get("top_summary") or {}),
                    "banner": dict(raw_wl.get("banner") or {}),
                    "rows": list(raw_wl.get("rows") or [])[:50],
                }

        if not state_payload and not watchlist_payload:
            return empty

        state = state_payload.get("state") or {}
        card = state_payload.get("card") or {}
        freshness = state.get("freshness") or {}

        # File age
        ref_path = _SPREAD_STRESS_STATE_PATH if _SPREAD_STRESS_STATE_PATH.exists() else _SPREAD_STRESS_WATCHLIST_PATH
        age_sec = max(0.0, time.time() - ref_path.stat().st_mtime) if ref_path.exists() else 9999.0

        return {
            "available": True,
            "stale": age_sec > 600,
            "age_sec": round(age_sec, 1),
            "symbol": str(state_payload.get("symbol") or ""),
            "state": {
                "level": str(state.get("level") or "quiet"),
                "reasons": list(state.get("reasons") or []),
                "freshness_status": str(freshness.get("status") or "unknown"),
            },
            "card": dict(card),
            "dashboard_summary": str(state_payload.get("dashboard_summary") or ""),
            "recommended_action": str(state_payload.get("recommended_action") or ""),
            "watchlist": watchlist_payload,
        }
    except Exception:
        return empty


# ---------------------------------------------------------------------------
# Fill toxicity state (monitoring/annotation card)
# ---------------------------------------------------------------------------

def read_fill_toxicity_state() -> dict[str, Any]:
    """Read fill-toxicity state payload. Safe dict on missing/malformed."""
    empty: dict[str, Any] = {
        "available": False,
        "state": {"level": "quiet", "reasons": []},
        "card": {},
    }
    try:
        if not _FILL_TOXICITY_STATE_PATH.exists():
            return empty
        payload = json.loads(_FILL_TOXICITY_STATE_PATH.read_text(encoding="utf-8", errors="replace"))
        if not isinstance(payload, dict):
            return empty
        state = payload.get("state") or {}
        card = payload.get("card") or {}
        age_sec = max(0.0, time.time() - _FILL_TOXICITY_STATE_PATH.stat().st_mtime)
        return {
            "available": True,
            "stale": age_sec > 600,
            "age_sec": round(age_sec, 1),
            "source": str(payload.get("source") or ""),
            "rows": int(payload.get("rows") or 0),
            "top_side": str(payload.get("top_side") or ""),
            "state": {
                "level": str(state.get("level") or "quiet"),
                "reasons": list(state.get("reasons") or []),
            },
            "card": dict(card),
            "dashboard_summary": str(payload.get("dashboard_summary") or ""),
            "recommended_action": str(payload.get("recommended_action") or ""),
        }
    except Exception:
        return empty


# ---------------------------------------------------------------------------
# Latency stress state (monitoring/annotation card)
# ---------------------------------------------------------------------------

def read_latency_stress_state() -> dict[str, Any]:
    """Read latency-stress state payload. Safe dict on missing/malformed."""
    empty: dict[str, Any] = {
        "available": False,
        "state": {"level": "quiet", "reasons": []},
        "card": {},
    }
    try:
        if not _LATENCY_STRESS_STATE_PATH.exists():
            return empty
        payload = json.loads(_LATENCY_STRESS_STATE_PATH.read_text(encoding="utf-8", errors="replace"))
        if not isinstance(payload, dict):
            return empty
        state = payload.get("state") or {}
        card = payload.get("card") or {}
        age_sec = max(0.0, time.time() - _LATENCY_STRESS_STATE_PATH.stat().st_mtime)
        return {
            "available": True,
            "stale": age_sec > 600,
            "age_sec": round(age_sec, 1),
            "source": str(payload.get("source") or ""),
            "state": {
                "level": str(state.get("level") or "quiet"),
                "reasons": list(state.get("reasons") or []),
            },
            "card": dict(card),
            "dashboard_summary": str(payload.get("dashboard_summary") or ""),
            "recommended_action": str(payload.get("recommended_action") or ""),
        }
    except Exception:
        return empty


# ---------------------------------------------------------------------------
# Research event watchboard (top-level aggregation)
# ---------------------------------------------------------------------------

def read_watchboard_state() -> dict[str, Any]:
    """Read the research event watchboard aggregation payload."""
    empty: dict[str, Any] = {
        "available": False,
        "summary": {"lane_count": 0, "state_counts": {}, "top_lane": ""},
        "top_event": None,
        "banner": None,
        "lanes": [],
    }
    try:
        if not _WATCHBOARD_STATE_PATH.exists():
            return empty
        payload = json.loads(_WATCHBOARD_STATE_PATH.read_text(encoding="utf-8", errors="replace"))
        if not isinstance(payload, dict):
            return empty
        summary = payload.get("summary") or {}
        top_event = payload.get("top_event")
        banner = payload.get("banner")
        lanes = list(payload.get("lanes") or [])[:20]
        age_sec = max(0.0, time.time() - _WATCHBOARD_STATE_PATH.stat().st_mtime)
        return {
            "available": True,
            "stale": age_sec > 600,
            "age_sec": round(age_sec, 1),
            "summary": {
                "lane_count": int(summary.get("lane_count") or 0),
                "state_counts": dict(summary.get("state_counts") or {}),
                "top_lane": str(summary.get("top_lane") or ""),
            },
            "top_event": dict(top_event) if isinstance(top_event, dict) else None,
            "banner": dict(banner) if isinstance(banner, dict) else None,
            "lanes": [dict(l) for l in lanes if isinstance(l, dict)],
        }
    except Exception:
        return empty


# ---------------------------------------------------------------------------
# Book proxy pressure state
# ---------------------------------------------------------------------------

def read_book_proxy_pressure_state() -> dict[str, Any]:
    """Read book-proxy-pressure state payload. Safe dict on missing/malformed."""
    empty: dict[str, Any] = {
        "available": False,
        "state": {"level": "quiet", "reasons": [], "primary_side_bias": "NEUTRAL"},
        "card": {},
    }
    try:
        if not _BOOK_PROXY_PRESSURE_STATE_PATH.exists():
            return empty
        payload = json.loads(_BOOK_PROXY_PRESSURE_STATE_PATH.read_text(encoding="utf-8", errors="replace"))
        if not isinstance(payload, dict):
            return empty
        state = payload.get("state") or {}
        card = payload.get("card") or {}
        age_sec = max(0.0, time.time() - _BOOK_PROXY_PRESSURE_STATE_PATH.stat().st_mtime)
        return {
            "available": True,
            "stale": age_sec > 600,
            "age_sec": round(age_sec, 1),
            "symbol": str(payload.get("symbol") or ""),
            "lane": str(payload.get("lane") or "book_proxy_pressure"),
            "state": {
                "level": str(state.get("level") or "quiet"),
                "reasons": list(state.get("reasons") or []),
                "primary_side_bias": str(state.get("primary_side_bias") or "NEUTRAL"),
                "freshness": dict(state.get("freshness") or {}),
            },
            "card": dict(card),
            "dashboard_summary": str(payload.get("dashboard_summary") or ""),
            "recommended_action": str(payload.get("recommended_action") or ""),
        }
    except Exception:
        return empty


# ---------------------------------------------------------------------------
# Return shock state
# ---------------------------------------------------------------------------

def read_return_shock_state() -> dict[str, Any]:
    """Read return-shock state payload. Safe dict on missing/malformed."""
    empty: dict[str, Any] = {
        "available": False,
        "state": {"level": "quiet", "reasons": [], "dominant_direction": ""},
        "card": {},
    }
    try:
        if not _RETURN_SHOCK_STATE_PATH.exists():
            return empty
        payload = json.loads(_RETURN_SHOCK_STATE_PATH.read_text(encoding="utf-8", errors="replace"))
        if not isinstance(payload, dict):
            return empty
        state = payload.get("state") or {}
        card = payload.get("card") or {}
        age_sec = max(0.0, time.time() - _RETURN_SHOCK_STATE_PATH.stat().st_mtime)
        return {
            "available": True,
            "stale": age_sec > 600,
            "age_sec": round(age_sec, 1),
            "symbol": str(payload.get("symbol") or ""),
            "state": {
                "level": str(state.get("level") or "quiet"),
                "reasons": list(state.get("reasons") or []),
                "dominant_direction": str(state.get("dominant_direction") or ""),
                "freshness": dict(state.get("freshness") or {}),
            },
            "card": dict(card),
            "dashboard_summary": str(payload.get("dashboard_summary") or ""),
            "recommended_action": str(payload.get("recommended_action") or ""),
        }
    except Exception:
        return empty


# ---------------------------------------------------------------------------
# Volatility burst state
# ---------------------------------------------------------------------------

def read_volatility_burst_state() -> dict[str, Any]:
    """Read volatility-burst state payload. Safe dict on missing/malformed."""
    empty: dict[str, Any] = {
        "available": False,
        "state": {"level": "quiet", "reasons": [], "dominant_direction": ""},
        "card": {},
    }
    try:
        if not _VOLATILITY_BURST_STATE_PATH.exists():
            return empty
        payload = json.loads(_VOLATILITY_BURST_STATE_PATH.read_text(encoding="utf-8", errors="replace"))
        if not isinstance(payload, dict):
            return empty
        state = payload.get("state") or {}
        card = payload.get("card") or {}
        age_sec = max(0.0, time.time() - _VOLATILITY_BURST_STATE_PATH.stat().st_mtime)
        return {
            "available": True,
            "stale": age_sec > 600,
            "age_sec": round(age_sec, 1),
            "symbol": str(payload.get("symbol") or ""),
            "lane": str(payload.get("lane") or "volatility_burst"),
            "state": {
                "level": str(state.get("level") or "quiet"),
                "reasons": list(state.get("reasons") or []),
                "dominant_direction": str(state.get("dominant_direction") or ""),
                "freshness": dict(state.get("freshness") or {}),
            },
            "card": dict(card),
            "dashboard_summary": str(payload.get("dashboard_summary") or ""),
            "recommended_action": str(payload.get("recommended_action") or ""),
        }
    except Exception:
        return empty


# ---------------------------------------------------------------------------
# Volume vacuum state
# ---------------------------------------------------------------------------

def read_volume_vacuum_state() -> dict[str, Any]:
    """Read volume-vacuum state payload. Safe dict on missing/malformed."""
    empty: dict[str, Any] = {
        "available": False,
        "state": {"level": "quiet", "reasons": []},
        "card": {},
    }
    try:
        if not _VOLUME_VACUUM_STATE_PATH.exists():
            return empty
        payload = json.loads(_VOLUME_VACUUM_STATE_PATH.read_text(encoding="utf-8", errors="replace"))
        if not isinstance(payload, dict):
            return empty
        state = payload.get("state") or {}
        card = payload.get("card") or {}
        age_sec = max(0.0, time.time() - _VOLUME_VACUUM_STATE_PATH.stat().st_mtime)
        return {
            "available": True,
            "stale": age_sec > 600,
            "age_sec": round(age_sec, 1),
            "symbol": str(payload.get("symbol") or ""),
            "lane": str(payload.get("lane") or "volume_vacuum"),
            "state": {
                "level": str(state.get("level") or "quiet"),
                "reasons": list(state.get("reasons") or []),
                "freshness": dict(state.get("freshness") or {}),
            },
            "card": dict(card),
            "dashboard_summary": str(payload.get("dashboard_summary") or ""),
            "recommended_action": str(payload.get("recommended_action") or ""),
        }
    except Exception:
        return empty
