# utils/logging.py — ECLIPSE ETERNAL LOGGING — 2026 v4.2 (FULL SURGERY)
# Enhancements (vs your current):
# - Idempotent setup (importing multiple times won't duplicate handlers / spam)
# - Optional JSON logs via env (LOG_JSON=1) for production + parsing
# - Optional file logging w/ rotation via env (LOG_FILE=..., LOG_FILE_MB=..., LOG_FILE_BACKUPS=...)
# - Per-logger level overrides via env (LOG_LEVEL, LOG_LEVEL_CORE, LOG_LEVEL_DATA, ...)
# - Safer color handling (disable colors when not a TTY or when NO_COLOR is set)
# - Micro-friendly: less noise, structured "extra" fields supported (won't crash if missing)
# - UTC timestamps optional (LOG_UTC=1) while keeping your default HH:MM:SS local style
# - Keeps existing public names: log, log_core, log_data, log_signal, log_entry, log_exit, log_risk, log_brain

from __future__ import annotations

import json
import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from datetime import datetime, timezone
from typing import Any, Dict, Optional


# ---------------------------
# Environment controls
# ---------------------------
def _env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "y", "on")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


def _env_level(name: str, default: str = "INFO") -> int:
    s = (os.getenv(name, default) or default).strip().upper()
    return getattr(logging, s, logging.INFO)


LOG_LEVEL = _env_level("LOG_LEVEL", "INFO")
LOG_JSON = _env_bool("LOG_JSON", False)
LOG_UTC = _env_bool("LOG_UTC", False)

LOG_FILE = os.getenv("LOG_FILE", "").strip()
LOG_FILE_MB = _env_int("LOG_FILE_MB", 25)
LOG_FILE_BACKUPS = _env_int("LOG_FILE_BACKUPS", 5)

NO_COLOR = _env_bool("NO_COLOR", False) or (os.getenv("NO_COLOR") is not None)


def _supports_color() -> bool:
    if NO_COLOR:
        return False
    try:
        return sys.stdout.isatty()
    except Exception:
        return False


USE_COLOR = _supports_color()


# ---------------------------
# Formatters
# ---------------------------
class ColorFormatter(logging.Formatter):
    # ANSI colors
    grey = "\x1b[38;21m"
    green = "\x1b[32;21m"
    yellow = "\x1b[33;21m"
    red = "\x1b[31;21m"
    bold_red = "\x1b[31;1m"
    cyan = "\x1b[36;21m"
    reset = "\x1b[0m"

    def __init__(self, *, use_color: bool = True, utc: bool = False):
        super().__init__()
        self.use_color = bool(use_color)
        self.utc = bool(utc)

        # Precompute formats (no per-record formatter allocations)
        if self.use_color:
            self.formats = {
                logging.DEBUG: f"%(asctime)s | Φ | {self.grey}%(levelname)-8s{self.reset} | %(name)-12s | %(message)s",
                logging.INFO: f"%(asctime)s | Φ | {self.green}%(levelname)-8s{self.reset} | %(name)-12s | %(message)s",
                logging.WARNING: f"%(asctime)s | Φ | {self.yellow}%(levelname)-8s{self.reset} | %(name)-12s | %(message)s",
                logging.ERROR: f"%(asctime)s | Φ | {self.red}%(levelname)-8s{self.reset} | %(name)-12s | %(message)s",
                logging.CRITICAL: f"%(asctime)s | Φ | {self.bold_red}%(levelname)-8s{self.reset} | %(name)-12s | %(message)s",
            }
        else:
            self.formats = {
                logging.DEBUG: "%(asctime)s | Φ | %(levelname)-8s | %(name)-12s | %(message)s",
                logging.INFO: "%(asctime)s | Φ | %(levelname)-8s | %(name)-12s | %(message)s",
                logging.WARNING: "%(asctime)s | Φ | %(levelname)-8s | %(name)-12s | %(message)s",
                logging.ERROR: "%(asctime)s | Φ | %(levelname)-8s | %(name)-12s | %(message)s",
                logging.CRITICAL: "%(asctime)s | Φ | %(levelname)-8s | %(name)-12s | %(message)s",
            }

        self.datefmt = "%H:%M:%S"

    def formatTime(self, record, datefmt=None):
        dt = datetime.fromtimestamp(record.created, tz=timezone.utc if self.utc else None)
        return dt.strftime(datefmt or self.datefmt)

    def format(self, record: logging.LogRecord) -> str:
        fmt = self.formats.get(record.levelno, self.formats[logging.INFO])
        # Add safe "extras" if present without crashing
        # Users can do: logger.info("msg", extra={"sym":"BTCUSDT"}) then include in message themselves.
        formatter = logging.Formatter(fmt, datefmt=self.datefmt)
        formatter.formatTime = self.formatTime  # type: ignore
        return formatter.format(record)


class JsonFormatter(logging.Formatter):
    """
    Structured logs for production ingestion.
    Won't crash if record has unserializable extras; will stringify.
    """

    def __init__(self, *, utc: bool = False):
        super().__init__()
        self.utc = bool(utc)

    def format(self, record: logging.LogRecord) -> str:
        ts = datetime.fromtimestamp(record.created, tz=timezone.utc if self.utc else None)
        base: Dict[str, Any] = {
            "ts": ts.isoformat(timespec="seconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }

        # Attach common extras if present
        for k in ("sym", "tf", "side", "conf", "pnl", "oid"):
            if hasattr(record, k):
                try:
                    base[k] = getattr(record, k)
                except Exception:
                    pass

        # If exception info exists, include it (compact)
        if record.exc_info:
            base["exc"] = self.formatException(record.exc_info)

        # Capture all non-standard attributes (best-effort)
        # Avoid duplicating standard LogRecord keys
        standard = set(vars(logging.LogRecord("", 0, "", 0, "", (), None)).keys())
        for k, v in record.__dict__.items():
            if k in standard or k in base:
                continue
            try:
                json.dumps(v)  # test serializable
                base[k] = v
            except Exception:
                base[k] = str(v)

        return json.dumps(base, ensure_ascii=False)


# ---------------------------
# Setup / handlers
# ---------------------------
_CONFIGURED = False


def _clear_handlers(logger: logging.Logger):
    # Remove handlers we own (or all handlers if you want full control)
    for h in list(logger.handlers):
        logger.removeHandler(h)


def _make_console_handler() -> logging.Handler:
    h = logging.StreamHandler(sys.stdout)
    if LOG_JSON:
        h.setFormatter(JsonFormatter(utc=LOG_UTC))
    else:
        h.setFormatter(ColorFormatter(use_color=USE_COLOR, utc=LOG_UTC))
    return h


def _make_file_handler(path: str) -> Optional[logging.Handler]:
    if not path:
        return None
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        max_bytes = max(1, LOG_FILE_MB) * 1024 * 1024
        h = RotatingFileHandler(path, maxBytes=max_bytes, backupCount=max(1, LOG_FILE_BACKUPS), encoding="utf-8")
        h.setFormatter(JsonFormatter(utc=True) if LOG_JSON else ColorFormatter(use_color=False, utc=True))
        return h
    except Exception:
        return None


def setup_logging(force: bool = False) -> None:
    """
    Idempotent global logging setup.
    Calling multiple times will NOT duplicate handlers unless force=True.
    """
    global _CONFIGURED
    if _CONFIGURED and not force:
        return

    root = logging.getLogger()
    root.setLevel(LOG_LEVEL)

    # Clear existing handlers to prevent duplicates (common in reloads)
    _clear_handlers(root)

    # Console always
    root.addHandler(_make_console_handler())

    # Optional file
    fh = _make_file_handler(LOG_FILE)
    if fh is not None:
        root.addHandler(fh)

    # Keep third-party libraries quieter unless user overrides
    # (ccxt, asyncio, websockets can be insanely chatty)
    logging.getLogger("ccxt").setLevel(_env_level("LOG_LEVEL_CCXT", "WARNING"))
    logging.getLogger("asyncio").setLevel(_env_level("LOG_LEVEL_ASYNCIO", "WARNING"))
    logging.getLogger("websockets").setLevel(_env_level("LOG_LEVEL_WEBSOCKETS", "WARNING"))

    _CONFIGURED = True


def _get_named_logger(name: str, env_level_key: str) -> logging.Logger:
    lg = logging.getLogger(name)
    # Allow per-logger override; otherwise inherit root
    lvl = os.getenv(env_level_key)
    if lvl:
        lg.setLevel(_env_level(env_level_key, lvl))
    return lg


# Ensure configured on import (safe/idempotent)
setup_logging()

# Public logger names preserved
log = _get_named_logger("EclipseEternal", "LOG_LEVEL_MAIN")
log_core = _get_named_logger("Core", "LOG_LEVEL_CORE")
log_data = _get_named_logger("Data", "LOG_LEVEL_DATA")
log_signal = _get_named_logger("Signal", "LOG_LEVEL_SIGNAL")
log_entry = _get_named_logger("Entry", "LOG_LEVEL_ENTRY")
log_exit = _get_named_logger("Exit", "LOG_LEVEL_EXIT")
log_risk = _get_named_logger("Risk", "LOG_LEVEL_RISK")
log_brain = _get_named_logger("Brain", "LOG_LEVEL_BRAIN")

# One-time banner (avoid repeating on reload)
if not getattr(log, "_ratio_banner_done", False):
    log.info("FULL LOGGING PACK ACTIVATED — THE RATIO NOW SPEAKS CLEARLY")
    setattr(log, "_ratio_banner_done", True)
