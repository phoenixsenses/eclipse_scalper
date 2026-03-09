# execution/flatten_intent.py — SCALPER ETERNAL — FLATTEN INTENT MANAGER — 2026 v1.0
# Persists flatten intent BEFORE execution so that:
# - If interrupted mid-flatten, restart can resume
# - Verification loop re-checks exposure after each attempt
# - Tracks which symbols have been successfully flattened
#
# Design principles:
# - WAL pattern: write to .tmp, fsync, rename (atomic)
# - Persist BEFORE any flatten operations begin
# - Track per-symbol completion for resume
# - Clear only on full success

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from utils.logging import log_core

# Configuration
_FLATTEN_INTENT_PATH = Path(os.path.expanduser("~/.blade_flatten_intent.json"))
_FLATTEN_INTENT_MAX_AGE_SEC = float(os.getenv("FLATTEN_INTENT_MAX_AGE_SEC", "3600"))  # 1 hour
_FLATTEN_INTENT_MAX_ATTEMPTS = int(os.getenv("FLATTEN_INTENT_MAX_ATTEMPTS", "5"))
_FLATTEN_INTENT_ENABLED = os.getenv("FLATTEN_INTENT_ENABLED", "1").lower() in ("1", "true", "yes", "on")


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        return default if v != v else v
    except Exception:
        return default


def _truthy(x: Any) -> bool:
    if x is True:
        return True
    if x is False or x is None:
        return False
    if isinstance(x, (int, float)):
        return x != 0
    if isinstance(x, str):
        return x.strip().lower() in ("1", "true", "yes", "y", "on", "t")
    return False


def _cfg(bot, name: str, default: Any) -> Any:
    try:
        cfg = getattr(bot, "cfg", None)
        return getattr(cfg, name, default) if cfg is not None else default
    except Exception:
        return default


@dataclass
class FlattenIntent:
    """
    Persisted flatten intent state.

    This is written to disk BEFORE any flatten operations begin,
    enabling restart recovery if the process is interrupted.
    """
    reason: str
    forced: bool
    started_ts: float
    target_symbols: List[str]
    completed_symbols: List[str] = field(default_factory=list)
    failed_symbols: List[str] = field(default_factory=list)
    attempt_count: int = 0
    max_attempts: int = _FLATTEN_INTENT_MAX_ATTEMPTS
    last_attempt_ts: float = 0.0
    verification_passes: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "reason": self.reason,
            "forced": self.forced,
            "started_ts": self.started_ts,
            "target_symbols": self.target_symbols,
            "completed_symbols": self.completed_symbols,
            "failed_symbols": self.failed_symbols,
            "attempt_count": self.attempt_count,
            "max_attempts": self.max_attempts,
            "last_attempt_ts": self.last_attempt_ts,
            "verification_passes": self.verification_passes,
            "version": "1.0",
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'FlattenIntent':
        return cls(
            reason=str(d.get("reason") or ""),
            forced=bool(d.get("forced", False)),
            started_ts=_safe_float(d.get("started_ts"), 0.0),
            target_symbols=list(d.get("target_symbols") or []),
            completed_symbols=list(d.get("completed_symbols") or []),
            failed_symbols=list(d.get("failed_symbols") or []),
            attempt_count=int(_safe_float(d.get("attempt_count"), 0)),
            max_attempts=int(_safe_float(d.get("max_attempts"), _FLATTEN_INTENT_MAX_ATTEMPTS)),
            last_attempt_ts=_safe_float(d.get("last_attempt_ts"), 0.0),
            verification_passes=int(_safe_float(d.get("verification_passes"), 0)),
        )

    @property
    def remaining_symbols(self) -> List[str]:
        """Get symbols that have not been completed yet."""
        completed_set = set(self.completed_symbols)
        return [s for s in self.target_symbols if s not in completed_set]

    @property
    def is_complete(self) -> bool:
        """True if all target symbols have been completed."""
        return len(self.remaining_symbols) == 0

    @property
    def progress_pct(self) -> float:
        """Completion percentage."""
        if not self.target_symbols:
            return 100.0
        return (len(self.completed_symbols) / len(self.target_symbols)) * 100.0

    @property
    def age_sec(self) -> float:
        """Age of this intent in seconds."""
        return time.time() - self.started_ts if self.started_ts > 0 else 0.0


class FlattenIntentManager:
    """
    Manages flatten intent persistence and recovery.

    Usage:
        mgr = FlattenIntentManager()

        # On emergency_flat start
        intent = await mgr.begin_flatten(reason, forced, target_symbols)

        # After each symbol flattened
        await mgr.mark_symbol_completed(intent, symbol)

        # On success
        await mgr.complete_flatten()

        # On startup, check for incomplete flatten
        intent = await mgr.check_incomplete_flatten()
        if intent:
            # Resume flatten with intent.remaining_symbols
    """

    def __init__(
        self,
        path: Path = _FLATTEN_INTENT_PATH,
        max_age_sec: float = _FLATTEN_INTENT_MAX_AGE_SEC,
    ):
        self.path = path
        self.max_age_sec = max_age_sec
        self._lock = asyncio.Lock()
        self._current_intent: Optional[FlattenIntent] = None

    async def begin_flatten(
        self,
        reason: str,
        forced: bool,
        target_symbols: List[str],
        max_attempts: int = _FLATTEN_INTENT_MAX_ATTEMPTS,
    ) -> FlattenIntent:
        """
        Create and persist flatten intent BEFORE starting flatten.

        This MUST be called before any flatten operations begin.
        The intent is written to disk with fsync to ensure durability.

        Args:
            reason: Why flatten was triggered
            forced: Whether this is a forced flatten
            target_symbols: List of symbols to flatten
            max_attempts: Maximum resume attempts

        Returns:
            FlattenIntent object

        Raises:
            Exception: If persistence fails (flatten should abort)
        """
        async with self._lock:
            intent = FlattenIntent(
                reason=reason,
                forced=forced,
                started_ts=time.time(),
                target_symbols=list(target_symbols),
                max_attempts=max_attempts,
            )

            await self._persist(intent)
            self._current_intent = intent

            log_core.critical(
                f"FLATTEN INTENT PERSISTED: {len(target_symbols)} symbols | "
                f"reason={reason} | forced={forced}"
            )

            return intent

    async def mark_symbol_completed(self, intent: FlattenIntent, symbol: str) -> None:
        """
        Mark a symbol as successfully flattened.

        Args:
            intent: Current flatten intent
            symbol: Symbol that was successfully flattened
        """
        async with self._lock:
            if symbol not in intent.completed_symbols:
                intent.completed_symbols.append(symbol)

            # Remove from failed if it was there
            if symbol in intent.failed_symbols:
                intent.failed_symbols.remove(symbol)

            await self._persist(intent)

            remaining = len(intent.remaining_symbols)
            log_core.info(
                f"[flatten_intent] COMPLETED: {symbol} | "
                f"progress={intent.progress_pct:.0f}% | remaining={remaining}"
            )

    async def mark_symbol_failed(self, intent: FlattenIntent, symbol: str) -> None:
        """
        Mark a symbol as failed to flatten.

        Args:
            intent: Current flatten intent
            symbol: Symbol that failed to flatten
        """
        async with self._lock:
            if symbol not in intent.failed_symbols:
                intent.failed_symbols.append(symbol)
            await self._persist(intent)

            log_core.warning(
                f"[flatten_intent] FAILED: {symbol} | "
                f"total_failed={len(intent.failed_symbols)}"
            )

    async def increment_attempt(self, intent: FlattenIntent) -> None:
        """
        Increment attempt counter (called on resume).

        Args:
            intent: Current flatten intent
        """
        async with self._lock:
            intent.attempt_count += 1
            intent.last_attempt_ts = time.time()
            await self._persist(intent)

            log_core.info(
                f"[flatten_intent] ATTEMPT: {intent.attempt_count}/{intent.max_attempts}"
            )

    async def increment_verification(self, intent: FlattenIntent) -> None:
        """
        Increment verification pass counter.

        Args:
            intent: Current flatten intent
        """
        async with self._lock:
            intent.verification_passes += 1
            await self._persist(intent)

    async def complete_flatten(self) -> None:
        """
        Clear flatten intent after successful completion.

        This removes the intent file from disk.
        """
        async with self._lock:
            if self.path.exists():
                try:
                    self.path.unlink()
                except Exception as e:
                    log_core.error(f"[flatten_intent] Failed to remove intent file: {e}")

            self._current_intent = None
            log_core.critical("FLATTEN INTENT CLEARED - flatten complete")

    async def check_incomplete_flatten(self) -> Optional[FlattenIntent]:
        """
        Check for incomplete flatten on startup.

        Returns:
            FlattenIntent if there's unfinished business, None otherwise

        Behavior:
            - Clears stale intents (> max_age_sec old)
            - Returns intent if max_attempts not exceeded and symbols remain
            - Logs critical warning if max_attempts exceeded (manual intervention)
        """
        if not _FLATTEN_INTENT_ENABLED:
            return None

        if not self.path.exists():
            return None

        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            intent = FlattenIntent.from_dict(data)

            # Check if intent is stale
            if intent.age_sec > self.max_age_sec:
                log_core.warning(
                    f"[flatten_intent] STALE (>{self.max_age_sec:.0f}s) - clearing | "
                    f"age={intent.age_sec:.0f}s"
                )
                await self.complete_flatten()
                return None

            # Check if max attempts exceeded
            if intent.attempt_count >= intent.max_attempts:
                log_core.critical(
                    f"FLATTEN INTENT MAX ATTEMPTS ({intent.max_attempts}) EXCEEDED | "
                    f"remaining={len(intent.remaining_symbols)} symbols | "
                    f"MANUAL INTERVENTION REQUIRED"
                )
                # Don't clear - keep for forensics
                return intent

            # Check if there are remaining symbols
            if intent.remaining_symbols:
                log_core.critical(
                    f"INCOMPLETE FLATTEN DETECTED: {len(intent.remaining_symbols)} symbols remaining | "
                    f"attempt={intent.attempt_count}/{intent.max_attempts} | "
                    f"reason={intent.reason}"
                )
                self._current_intent = intent
                return intent

            # All done, clear it
            log_core.info("[flatten_intent] All symbols completed - clearing")
            await self.complete_flatten()
            return None

        except json.JSONDecodeError as e:
            log_core.error(f"[flatten_intent] Corrupt intent file: {e} - removing")
            try:
                self.path.unlink()
            except Exception:
                pass
            return None
        except Exception as e:
            log_core.error(f"[flatten_intent] Failed to read intent file: {e}")
            return None

    async def _persist(self, intent: FlattenIntent) -> None:
        """
        Write intent to disk with WAL pattern (atomic write).

        Uses: write to .tmp, fsync, rename (atomic on POSIX)
        """
        tmp_path = self.path.with_suffix('.tmp')

        try:
            # Ensure parent directory exists
            tmp_path.parent.mkdir(parents=True, exist_ok=True)

            # Write to temp file
            with open(tmp_path, 'w', encoding='utf-8') as f:
                json.dump(intent.to_dict(), f, indent=2)
                f.flush()
                os.fsync(f.fileno())  # Force write to disk

            # Atomic rename (on POSIX systems)
            tmp_path.replace(self.path)

        except Exception as e:
            log_core.error(f"[flatten_intent] Failed to persist: {e}")
            # Clean up temp file if it exists
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except Exception:
                pass
            raise

    @property
    def current_intent(self) -> Optional[FlattenIntent]:
        """Get current active flatten intent."""
        return self._current_intent

    @property
    def has_active_flatten(self) -> bool:
        """True if there's an active flatten operation."""
        return self._current_intent is not None


# Module-level singleton for convenience
_default_manager: Optional[FlattenIntentManager] = None


def get_flatten_intent_manager() -> FlattenIntentManager:
    """Get the default FlattenIntentManager instance."""
    global _default_manager
    if _default_manager is None:
        _default_manager = FlattenIntentManager()
    return _default_manager


async def check_incomplete_flatten_on_startup() -> Optional[FlattenIntent]:
    """
    Convenience function to check for incomplete flatten on startup.

    Returns:
        FlattenIntent if resume needed, None otherwise
    """
    return await get_flatten_intent_manager().check_incomplete_flatten()
