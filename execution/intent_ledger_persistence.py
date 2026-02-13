# execution/intent_ledger_persistence.py — SCALPER ETERNAL — INTENT LEDGER PERSISTENCE — 2026 v1.0
# WAL-style persistence layer for intent ledger with crash safety.
#
# Implements:
# - WAL pattern: write to .tmp, fsync, rename
# - Periodic checkpoint (every 30s or 100 records)
# - I/O failure retry (3 attempts with exponential backoff)
# - Alert emission on persistent failures
# - Degraded mode flag when persistence unavailable
#
# Design principles:
# - Never block the hot path (fire-and-forget writes)
# - Crash safety via fsync on critical stages
# - Self-healing on transient I/O failures

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from utils.logging import log_core

# Configuration
_CHECKPOINT_INTERVAL_SEC = float(os.getenv("INTENT_LEDGER_CHECKPOINT_SEC", "30.0"))
_CHECKPOINT_RECORDS = int(os.getenv("INTENT_LEDGER_CHECKPOINT_RECORDS", "100"))
_WRITE_RETRIES = int(os.getenv("INTENT_LEDGER_WRITE_RETRIES", "3"))
_FSYNC_CRITICAL_ONLY = os.getenv("INTENT_LEDGER_FSYNC_CRITICAL", "1").lower() in ("1", "true", "yes", "on")

# Critical stages that require fsync
_CRITICAL_STAGES = {"DONE", "FAILED", "CANCEL_FAILED", "SUBMITTED_UNKNOWN"}


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        return default if v != v else v
    except Exception:
        return default


class IntentLedgerPersistence:
    """
    WAL-style persistence layer for intent ledger.

    Provides crash-safe persistence with periodic checkpoints
    and graceful degradation on I/O failures.

    Usage:
        persistence = IntentLedgerPersistence(Path("logs/intent_ledger.jsonl"))

        # Append record (async, fire-and-forget safe)
        success = await persistence.append_record(payload)

        # Periodic checkpoint
        if persistence.should_checkpoint():
            await persistence.checkpoint(store)

        # Recovery on startup
        store = await persistence.recover()
    """

    def __init__(
        self,
        path: Path,
        checkpoint_path: Optional[Path] = None,
        checkpoint_interval_sec: float = _CHECKPOINT_INTERVAL_SEC,
        checkpoint_records: int = _CHECKPOINT_RECORDS,
        write_retries: int = _WRITE_RETRIES,
        fsync_critical_only: bool = _FSYNC_CRITICAL_ONLY,
    ):
        self.path = path
        self.checkpoint_path = checkpoint_path or path.with_suffix('.checkpoint.json')
        self.wal_path = path.with_suffix('.wal')
        self.checkpoint_interval = checkpoint_interval_sec
        self.checkpoint_records = checkpoint_records
        self.write_retries = write_retries
        self.fsync_critical_only = fsync_critical_only

        self.last_checkpoint_ts = 0.0
        self.records_since_checkpoint = 0
        self.degraded = False
        self.degraded_since = 0.0
        self.consecutive_failures = 0
        self.total_writes = 0
        self.total_failures = 0

        self._write_lock = asyncio.Lock()
        self._checkpoint_lock = asyncio.Lock()

        # Callbacks for alerting
        self._on_degraded: Optional[Callable[[str], None]] = None
        self._on_recovered: Optional[Callable[[], None]] = None

    def set_callbacks(
        self,
        on_degraded: Optional[Callable[[str], None]] = None,
        on_recovered: Optional[Callable[[], None]] = None,
    ) -> None:
        """Set callbacks for state changes."""
        self._on_degraded = on_degraded
        self._on_recovered = on_recovered

    async def append_record(
        self,
        payload: Dict[str, Any],
        *,
        force_fsync: bool = False,
    ) -> bool:
        """
        Append record to ledger with WAL pattern.

        Args:
            payload: Record to append (must be JSON-serializable)
            force_fsync: Force fsync even for non-critical stages

        Returns:
            True if write succeeded, False on failure

        Note: This method never raises. Failures are logged and tracked.
        """
        stage = str(payload.get("stage") or "").upper()
        should_fsync = force_fsync or (not self.fsync_critical_only) or (stage in _CRITICAL_STAGES)

        async with self._write_lock:
            for attempt in range(self.write_retries):
                try:
                    success = await self._write_with_wal(payload, fsync=should_fsync)
                    if success:
                        self.total_writes += 1
                        self.records_since_checkpoint += 1
                        self.consecutive_failures = 0

                        # Clear degraded state on success
                        if self.degraded:
                            self.degraded = False
                            log_core.info("[intent_ledger_persistence] Recovered from degraded state")
                            if self._on_recovered:
                                try:
                                    self._on_recovered()
                                except Exception:
                                    pass

                        return True

                except Exception as e:
                    self.total_failures += 1
                    self.consecutive_failures += 1

                    if attempt < self.write_retries - 1:
                        # Exponential backoff
                        delay = 0.1 * (2 ** attempt)
                        log_core.warning(
                            f"[intent_ledger_persistence] Write failed (attempt {attempt + 1}/{self.write_retries}): {e} | "
                            f"retrying in {delay:.1f}s"
                        )
                        await asyncio.sleep(delay)
                    else:
                        # Final failure
                        log_core.error(
                            f"[intent_ledger_persistence] Write failed after {self.write_retries} attempts: {e}"
                        )

            # All retries exhausted
            if not self.degraded:
                self.degraded = True
                self.degraded_since = time.time()
                log_core.critical("[intent_ledger_persistence] DEGRADED: persistence unavailable")
                if self._on_degraded:
                    try:
                        self._on_degraded(f"Write failed after {self.write_retries} retries")
                    except Exception:
                        pass

            return False

    async def _write_with_wal(
        self,
        payload: Dict[str, Any],
        *,
        fsync: bool = False,
    ) -> bool:
        """
        Write record using WAL pattern.

        1. Write to .wal temp file
        2. Optionally fsync
        3. Append to main file
        4. Remove .wal
        """
        # Ensure directory exists
        self.path.parent.mkdir(parents=True, exist_ok=True)

        line = json.dumps(payload, ensure_ascii=True) + "\n"

        # Write to WAL first
        with open(self.wal_path, 'w', encoding='utf-8') as f:
            f.write(line)
            if fsync:
                f.flush()
                os.fsync(f.fileno())

        # Append to main file
        with open(self.path, 'a', encoding='utf-8') as f:
            f.write(line)
            if fsync:
                f.flush()
                os.fsync(f.fileno())

        # Remove WAL
        try:
            self.wal_path.unlink()
        except Exception:
            pass  # Non-fatal

        return True

    def should_checkpoint(self) -> bool:
        """Check if a checkpoint is due."""
        if self.records_since_checkpoint >= self.checkpoint_records:
            return True

        if self.last_checkpoint_ts > 0:
            elapsed = time.time() - self.last_checkpoint_ts
            if elapsed >= self.checkpoint_interval:
                return True

        return False

    async def checkpoint(self, store: Dict[str, Any]) -> bool:
        """
        Write full state snapshot to checkpoint file.

        Args:
            store: The full intent ledger store dict

        Returns:
            True if checkpoint succeeded, False on failure
        """
        async with self._checkpoint_lock:
            try:
                tmp_path = self.checkpoint_path.with_suffix('.tmp')

                # Serialize store
                checkpoint_data = {
                    "ts": time.time(),
                    "version": "1.0",
                    "intents": store.get("intents", {}),
                    "by_order_id": store.get("by_order_id", {}),
                    "by_client_order_id": store.get("by_client_order_id", {}),
                }

                # Write to temp file with fsync
                with open(tmp_path, 'w', encoding='utf-8') as f:
                    json.dump(checkpoint_data, f, indent=2)
                    f.flush()
                    os.fsync(f.fileno())

                # Atomic rename
                tmp_path.replace(self.checkpoint_path)

                self.last_checkpoint_ts = time.time()
                self.records_since_checkpoint = 0

                log_core.info(
                    f"[intent_ledger_persistence] Checkpoint created | "
                    f"intents={len(checkpoint_data.get('intents', {}))}"
                )

                return True

            except Exception as e:
                log_core.error(f"[intent_ledger_persistence] Checkpoint failed: {e}")
                return False

    async def recover(self) -> Dict[str, Any]:
        """
        Recover state from checkpoint + WAL replay.

        Returns:
            Recovered store dict

        Recovery order:
        1. Load checkpoint (if exists)
        2. Replay JSONL entries after checkpoint timestamp
        3. Replay WAL (if exists) - incomplete writes
        """
        store: Dict[str, Any] = {
            "intents": {},
            "by_order_id": {},
            "by_client_order_id": {},
        }

        checkpoint_ts = 0.0

        # 1. Load checkpoint
        if self.checkpoint_path.exists():
            try:
                data = json.loads(self.checkpoint_path.read_text(encoding="utf-8"))
                store["intents"] = data.get("intents", {})
                store["by_order_id"] = data.get("by_order_id", {})
                store["by_client_order_id"] = data.get("by_client_order_id", {})
                checkpoint_ts = _safe_float(data.get("ts"), 0.0)

                log_core.info(
                    f"[intent_ledger_persistence] Loaded checkpoint | "
                    f"intents={len(store['intents'])} | ts={checkpoint_ts:.0f}"
                )

            except Exception as e:
                log_core.warning(f"[intent_ledger_persistence] Failed to load checkpoint: {e}")

        # 2. Replay JSONL entries after checkpoint
        if self.path.exists():
            replayed = 0
            try:
                for line in self.path.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    # Skip entries before checkpoint
                    entry_ts = _safe_float(payload.get("ts"), 0.0)
                    if checkpoint_ts > 0 and entry_ts <= checkpoint_ts:
                        continue

                    self._apply_record(store, payload)
                    replayed += 1

                if replayed > 0:
                    log_core.info(
                        f"[intent_ledger_persistence] Replayed {replayed} entries from JSONL"
                    )

            except Exception as e:
                log_core.warning(f"[intent_ledger_persistence] Failed to replay JSONL: {e}")

        # 3. Replay WAL (incomplete write recovery)
        if self.wal_path.exists():
            try:
                line = self.wal_path.read_text(encoding="utf-8").strip()
                if line:
                    payload = json.loads(line)
                    self._apply_record(store, payload)
                    log_core.info("[intent_ledger_persistence] Recovered 1 entry from WAL")

                # Append recovered entry to main file
                with open(self.path, 'a', encoding='utf-8') as f:
                    f.write(line + "\n")

                # Remove WAL
                self.wal_path.unlink()

            except Exception as e:
                log_core.warning(f"[intent_ledger_persistence] Failed to replay WAL: {e}")

        self.last_checkpoint_ts = time.time()
        self.records_since_checkpoint = 0

        return store

    def _apply_record(self, store: Dict[str, Any], payload: Dict[str, Any]) -> None:
        """Apply a single record to the store."""
        intent_id = str(payload.get("intent_id") or "").strip()
        if not intent_id:
            return

        intents = store.get("intents")
        if not isinstance(intents, dict):
            intents = {}
            store["intents"] = intents

        # Get or create record
        prev = intents.get(intent_id, {})
        rec = dict(prev)
        rec.update(payload)
        rec["intent_id"] = intent_id
        intents[intent_id] = rec

        # Update indexes
        order_id = str(rec.get("order_id") or "").strip()
        if order_id:
            store.setdefault("by_order_id", {})[order_id] = intent_id

        client_order_id = str(rec.get("client_order_id") or "").strip()
        if client_order_id:
            store.setdefault("by_client_order_id", {})[client_order_id] = intent_id

    async def compact(self, store: Dict[str, Any], *, max_age_sec: float = 86400.0) -> bool:
        """
        Compact the ledger by rewriting only recent entries.

        Args:
            store: Current store state
            max_age_sec: Keep entries newer than this (default 24h)

        Returns:
            True if compaction succeeded
        """
        async with self._write_lock:
            async with self._checkpoint_lock:
                try:
                    now = time.time()
                    cutoff = now - max_age_sec

                    # Filter recent intents
                    intents = store.get("intents", {})
                    recent_intents = {
                        k: v for k, v in intents.items()
                        if _safe_float(v.get("ts"), 0.0) > cutoff
                    }

                    # Rebuild indexes
                    by_order_id = {}
                    by_client_order_id = {}
                    for intent_id, rec in recent_intents.items():
                        order_id = str(rec.get("order_id") or "").strip()
                        if order_id:
                            by_order_id[order_id] = intent_id
                        client_order_id = str(rec.get("client_order_id") or "").strip()
                        if client_order_id:
                            by_client_order_id[client_order_id] = intent_id

                    # Update store
                    store["intents"] = recent_intents
                    store["by_order_id"] = by_order_id
                    store["by_client_order_id"] = by_client_order_id

                    # Write new checkpoint
                    await self.checkpoint(store)

                    # Truncate JSONL
                    tmp_path = self.path.with_suffix('.compact')
                    with open(tmp_path, 'w', encoding='utf-8') as f:
                        for rec in recent_intents.values():
                            f.write(json.dumps(rec, ensure_ascii=True) + "\n")
                        f.flush()
                        os.fsync(f.fileno())

                    tmp_path.replace(self.path)

                    removed = len(intents) - len(recent_intents)
                    log_core.info(
                        f"[intent_ledger_persistence] Compacted | "
                        f"removed={removed} | kept={len(recent_intents)}"
                    )

                    return True

                except Exception as e:
                    log_core.error(f"[intent_ledger_persistence] Compaction failed: {e}")
                    return False

    @property
    def stats(self) -> Dict[str, Any]:
        """Get persistence statistics."""
        return {
            "total_writes": self.total_writes,
            "total_failures": self.total_failures,
            "consecutive_failures": self.consecutive_failures,
            "records_since_checkpoint": self.records_since_checkpoint,
            "last_checkpoint_ts": self.last_checkpoint_ts,
            "degraded": self.degraded,
            "degraded_since": self.degraded_since,
        }
