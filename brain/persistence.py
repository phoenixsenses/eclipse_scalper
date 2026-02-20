# brain/persistence.py — ETERNAL BRAIN — IMMORTALITY LAYER — 2026 v4.5 (HARDENED)
# Patch vs your v4.4c:
# - ✅ Canonical _symkey matches state.py (handles /USDT:USDT, /USDT, :USDT)
# - ✅ Adds entry_watches (canonicalize + cap) for entry_watch.py persistence
# - ✅ Fix blacklist_reason merge: prefer-new string merge (not dict-merge)
# - ✅ Normalizes/caps known_exit_order_ids deterministically
# - ✅ streak_history always stored as primitives [iso, int, float]
# - ✅ Keeps IO lock + atomic rollback write behavior

from __future__ import annotations

import os
import msgpack
import lz4.frame
import aiofiles
import time
import hashlib
import asyncio
from datetime import date, datetime
from typing import Optional, Dict, Any, List, Tuple, Union, Callable

from utils.logging import log_brain
from brain.state import PsycheState

BRAIN_PATH = os.path.expanduser("~/.blade_eternal.brain.lz4")

PERSISTENCE_VERSION = "god-emperor-immortal-v4.5-2026-jan06"

ACCEPTED_VERSIONS = {
    "god-emperor-immortal-v1.0-2025-dec26",
    "god-emperor-immortal-v4.0-2026-jan04",
    "god-emperor-immortal-v4.1-2026-jan05",
    "god-emperor-immortal-v4.2-2026-jan05",
    "god-emperor-immortal-v4.3-2026-jan06",
    "god-emperor-immortal-v4.4-2026-jan06",
    "god-emperor-immortal-v4.4c-2026-jan06",
    PERSISTENCE_VERSION,
}

MAX_BACKUPS = 3

_memory_fallback_payload: Optional[Dict[str, Any]] = None
_disk_failed: bool = False

_IO_LOCK = asyncio.Lock()

# Caps (bloat control)
KNOWN_EXIT_IDS_CAP = 50_000
ENTRY_WATCHES_CAP = 500


# -----------------------
# Hashing / utilities
# -----------------------
def _sha256_hex(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _safe_float(x, default: float = 0.0) -> float:
    try:
        v = float(x)
        if v != v:  # NaN
            return default
        return v
    except Exception:
        return default


def _safe_int(x, default: int = 0) -> int:
    try:
        v = int(x)
        return v if v >= 0 else default
    except Exception:
        return default


def _parse_iso_date(s) -> Optional[date]:
    if not s:
        return None
    if isinstance(s, date):
        return s
    if isinstance(s, str):
        try:
            return date.fromisoformat(s)
        except Exception:
            return None
    return None


def _fsync_best_effort(path: str) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except Exception:
        pass


# -----------------------
# Primitive-safe coercion (msgpack safety)
# -----------------------
_Prim = Union[None, bool, int, float, str, bytes]
_MsgpackSafe = Union[_Prim, Dict[str, Any], List[Any]]


def _to_primitive_safe(x: Any, *, _depth: int = 0, _max_depth: int = 40) -> _MsgpackSafe:
    """
    Convert arbitrary objects into msgpack-safe primitives.
    - dict keys -> str
    - set/tuple -> list
    - date/datetime -> isoformat string
    """
    if _depth > _max_depth:
        return str(x)

    if x is None or isinstance(x, (bool, int, float, str, bytes)):
        return x

    if isinstance(x, (date, datetime)):
        try:
            return x.isoformat()
        except Exception:
            return str(x)

    if isinstance(x, set):
        try:
            return [_to_primitive_safe(v, _depth=_depth + 1) for v in list(x)]
        except Exception:
            return []

    if isinstance(x, tuple):
        try:
            return [_to_primitive_safe(v, _depth=_depth + 1) for v in list(x)]
        except Exception:
            return []

    if isinstance(x, list):
        return [_to_primitive_safe(v, _depth=_depth + 1) for v in x]

    if isinstance(x, dict):
        out: Dict[str, Any] = {}
        for k, v in x.items():
            try:
                ks = str(k)
            except Exception:
                ks = "<?>"
            out[ks] = _to_primitive_safe(v, _depth=_depth + 1)
        return out

    try:
        d = vars(x)
        if isinstance(d, dict):
            return _to_primitive_safe(d, _depth=_depth + 1)
    except Exception:
        pass

    return str(x)


# -----------------------
# Canonical symbol law (MATCHES brain/state.py)
# -----------------------
def _symkey(sym: Any) -> str:
    if sym is None:
        return ""
    try:
        s = str(sym).strip().upper()
    except Exception:
        return ""
    if not s:
        return ""

    s = s.replace("/USDT:USDT", "USDT").replace("/USDT", "USDT")
    s = s.replace(":USDT", "USDT").replace(":", "")
    s = s.replace("/", "")

    if s.endswith("USDTUSDT"):
        s = s[:-4]
    return s


def _merge_max(a: Any, b: Any) -> Any:
    try:
        fa = _safe_float(a, 0.0)
        fb = _safe_float(b, 0.0)
        return a if fa >= fb else b
    except Exception:
        return a if a is not None else b


def _merge_dict_shallow(a: Any, b: Any) -> Any:
    if isinstance(a, dict) and isinstance(b, dict):
        out = dict(a)
        out.update(b)
        return out
    return b if b is not None else a


def _merge_pick_b(a: Any, b: Any) -> Any:
    """Prefer new value on collisions (good for str maps like blacklist_reason)."""
    return b if b is not None else a


def _canon_map_keys(m: Any, *, merge_fn: Optional[Callable[[Any, Any], Any]] = None) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if not isinstance(m, dict):
        return out
    for k, v in m.items():
        ck = _symkey(k)
        if not ck:
            continue
        if ck in out and merge_fn is not None:
            try:
                out[ck] = merge_fn(out[ck], v)
            except Exception:
                out[ck] = v
        else:
            out[ck] = v
    return out


def _apply_compat_map(d: Dict[str, Any], compat_map: Optional[Dict[str, str]]) -> Dict[str, Any]:
    if not compat_map or not isinstance(compat_map, dict):
        return d
    out = dict(d)
    try:
        for old, new in compat_map.items():
            if old in out and new not in out:
                out[new] = out.pop(old)
    except Exception:
        return out
    return out


def _canon_positions_map(pos: Any) -> Dict[str, Any]:
    """
    Canonicalize positions keys; collision policy keeps newest entry_ts.
    Ensures inner symbol is canonical and size is ABS on disk.
    """
    out: Dict[str, Any] = {}
    if not isinstance(pos, dict):
        return out

    def score(pv: Any) -> float:
        if isinstance(pv, dict):
            return _safe_float(pv.get("entry_ts"), _safe_float(pv.get("last_update_ts"), 0.0))
        return 0.0

    for k, pv in pos.items():
        ck = _symkey(k)
        if not ck:
            continue

        if isinstance(pv, dict):
            pv2 = dict(pv)
            pv2["symbol"] = ck
            if "size" in pv2:
                pv2["size"] = abs(_safe_float(pv2.get("size"), 0.0))
        else:
            pv2 = pv

        if ck not in out:
            out[ck] = pv2
        else:
            try:
                if score(pv2) >= score(out[ck]):
                    out[ck] = pv2
            except Exception:
                pass

    return out


def _canon_entry_watches_map(m: Any) -> Dict[str, Dict[str, Any]]:
    """
    Canonicalize entry_watches keys and keep newest created_ts per symbol.
    Caps size to ENTRY_WATCHES_CAP.
    """
    out: Dict[str, Dict[str, Any]] = {}
    if not isinstance(m, dict):
        return out

    for sym, w in m.items():
        k = _symkey(sym)
        if not k or not isinstance(w, dict):
            continue
        w2 = dict(w)
        w2["k"] = k
        try:
            w2["symbol_any"] = str(w2.get("symbol_any") or k)
        except Exception:
            w2["symbol_any"] = k

        ct = _safe_float(w2.get("created_ts", 0.0), 0.0)
        if k not in out:
            out[k] = w2
        else:
            prev_ct = _safe_float(out[k].get("created_ts", 0.0), 0.0)
            if ct >= prev_ct:
                out[k] = w2

    # deterministic cap: keep newest
    try:
        if len(out) > ENTRY_WATCHES_CAP:
            items = sorted(out.items(), key=lambda kv: _safe_float(kv[1].get("created_ts", 0.0), 0.0))
            out = dict(items[-ENTRY_WATCHES_CAP:])
    except Exception:
        pass

    return out


def _canonicalize_payload_state(s: Dict[str, Any]) -> Dict[str, Any]:
    """
    Canonicalize ALL symbol-keyed maps in payload_state.
    Runs on load + save so disk becomes permanently canonical.
    """
    if not isinstance(s, dict):
        return {}

    out = dict(s)

    out.setdefault("positions", {})
    out.setdefault("blacklist", {})
    out.setdefault("blacklist_reason", {})
    out.setdefault("consecutive_losses", {})
    out.setdefault("last_exit_time", {})
    out.setdefault("symbol_performance", {})
    out.setdefault("entry_confidence_history", {})
    out.setdefault("funding_rate_snapshot", {})
    out.setdefault("entry_watches", {})
    out.setdefault("streak_history", [])
    out.setdefault("run_context", {})

    out["positions"] = _canon_positions_map(out.get("positions"))

    out["blacklist"] = _canon_map_keys(out.get("blacklist"), merge_fn=_merge_max)
    out["last_exit_time"] = _canon_map_keys(out.get("last_exit_time"), merge_fn=_merge_max)
    out["consecutive_losses"] = _canon_map_keys(out.get("consecutive_losses"), merge_fn=_merge_max)

    # IMPORTANT: blacklist_reason values are strings; prefer new on collisions
    out["blacklist_reason"] = _canon_map_keys(out.get("blacklist_reason"), merge_fn=_merge_pick_b)

    out["symbol_performance"] = _canon_map_keys(out.get("symbol_performance"), merge_fn=_merge_dict_shallow)
    out["entry_confidence_history"] = _canon_map_keys(out.get("entry_confidence_history"), merge_fn=_merge_dict_shallow)
    out["funding_rate_snapshot"] = _canon_map_keys(out.get("funding_rate_snapshot"), merge_fn=_merge_dict_shallow)

    out["entry_watches"] = _canon_entry_watches_map(out.get("entry_watches"))

    # known_exit_order_ids -> list[str] capped
    kei = out.get("known_exit_order_ids")
    if isinstance(kei, set):
        kei_list = list(kei)
    elif isinstance(kei, tuple):
        kei_list = list(kei)
    elif isinstance(kei, list):
        kei_list = kei
    else:
        kei_list = []
    try:
        if len(kei_list) > KNOWN_EXIT_IDS_CAP:
            kei_list = kei_list[-KNOWN_EXIT_IDS_CAP:]
    except Exception:
        pass
    out["known_exit_order_ids"] = [str(x) for x in kei_list if x is not None]

    # streak_history disk rule: list of [iso_date_str, int, float]
    sh = out.get("streak_history")
    if isinstance(sh, list):
        fixed: List[List[Any]] = []
        for item in sh:
            if not isinstance(item, (list, tuple)) or len(item) != 3:
                continue
            d, streak_len, pnl = item
            if isinstance(d, date):
                d_str = d.isoformat()
            elif isinstance(d, str):
                d2 = _parse_iso_date(d)
                if d2 is None:
                    continue
                d_str = d2.isoformat()
            else:
                continue
            fixed.append([d_str, int(streak_len or 0), float(pnl or 0.0)])
        out["streak_history"] = fixed
    else:
        out["streak_history"] = []

    return out


# -----------------------
# Payload shaping
# -----------------------
def _state_to_payload(state: PsycheState) -> Dict[str, Any]:
    """
    Create the serializable payload_state dict.
    """
    try:
        payload = state.to_dict()
        if isinstance(payload, dict):
            payload.setdefault("schema_version", int(getattr(state, "schema_version", 1)))
            payload = _canonicalize_payload_state(payload)
            payload = _to_primitive_safe(payload)
            return payload  # type: ignore[return-value]
    except Exception:
        pass

    # Fallback minimal snapshot
    positions_payload: Dict[str, Any] = {}
    try:
        pos_map = getattr(state, "positions", {}) or {}
        if isinstance(pos_map, dict):
            for k, v in pos_map.items():
                ck = _symkey(k)
                if not ck:
                    continue
                positions_payload[ck] = {
                    "symbol": ck,
                    "side": str(getattr(v, "side", "long") or "long"),
                    "size": abs(_safe_float(getattr(v, "size", 0.0), 0.0)),
                    "entry_price": _safe_float(getattr(v, "entry_price", 0.0), 0.0),
                    "atr": _safe_float(getattr(v, "atr", 0.0), 0.0),
                    "leverage": int(getattr(v, "leverage", 0) or 0),
                    "entry_ts": _safe_float(getattr(v, "entry_ts", time.time()), time.time()),
                    "hard_stop_order_id": getattr(v, "hard_stop_order_id", None),
                    "trailing_active": bool(getattr(v, "trailing_active", False)),
                    "breakeven_moved": bool(getattr(v, "breakeven_moved", False)),
                    "confidence": _safe_float(getattr(v, "confidence", 0.0), 0.0),
                    "last_breakeven_move": _safe_float(getattr(v, "last_breakeven_move", 0.0), 0.0),
                    "last_update_ts": float(time.time()),
                }
    except Exception:
        positions_payload = {}

    cd = getattr(state, "current_day", None)
    if isinstance(cd, date):
        cd_iso = cd.isoformat()
    elif isinstance(cd, str):
        cd_iso = cd
    else:
        cd_iso = None

    payload = {
        "schema_version": int(getattr(state, "schema_version", 1)),
        "current_equity": _safe_float(getattr(state, "current_equity", 0.0), 0.0),
        "peak_equity": _safe_float(getattr(state, "peak_equity", 0.0), 0.0),
        "peak_equity_timestamp": _safe_float(getattr(state, "peak_equity_timestamp", time.time()), time.time()),
        "daily_pnl": _safe_float(getattr(state, "daily_pnl", 0.0), 0.0),
        "total_trades": _safe_int(getattr(state, "total_trades", 0), 0),
        "total_wins": _safe_int(getattr(state, "total_wins", 0), 0),
        "win_streak": _safe_int(getattr(state, "win_streak", 0), 0),
        "start_of_day_equity": _safe_float(getattr(state, "start_of_day_equity", 0.0), 0.0),
        "current_day": cd_iso,
        "positions": positions_payload,
        "blacklist": dict(getattr(state, "blacklist", {}) or {}),
        "blacklist_reason": dict(getattr(state, "blacklist_reason", {}) or {}),
        "consecutive_losses": dict(getattr(state, "consecutive_losses", {}) or {}),
        "last_exit_time": dict(getattr(state, "last_exit_time", {}) or {}),
        "known_exit_order_ids": list(getattr(state, "known_exit_order_ids", set()) or []),
        "symbol_performance": dict(getattr(state, "symbol_performance", {}) or {}),
        "entry_confidence_history": dict(getattr(state, "entry_confidence_history", {}) or {}),
        "streak_history": list(getattr(state, "streak_history", []) or []),
        "adaptive_risk_multiplier": _safe_float(getattr(state, "adaptive_risk_multiplier", 1.0), 1.0),
        "funding_paid": _safe_float(getattr(state, "funding_paid", 0.0), 0.0),
        "funding_rate_snapshot": dict(getattr(state, "funding_rate_snapshot", {}) or {}),
        "entry_watches": dict(getattr(state, "entry_watches", {}) or {}),
        "run_context": dict(getattr(state, "run_context", {}) or {}),
        "session_start_timestamp": _safe_float(getattr(state, "session_start_timestamp", time.time()), time.time()),
        "uptime_seconds": _safe_float(getattr(state, "uptime_seconds", 0.0), 0.0),
        "win_rate": _safe_float(getattr(state, "win_rate", 0.0), 0.0),
        "max_drawdown": _safe_float(getattr(state, "max_drawdown", 0.0), 0.0),
    }

    payload = _canonicalize_payload_state(payload)
    payload = _to_primitive_safe(payload)
    return payload  # type: ignore[return-value]


def _migrate_payload_state(payload: Dict[str, Any], persistence_version: str) -> Dict[str, Any]:
    """
    Migration hook for payload dict before PsycheState.from_loaded().
    """
    s = dict(payload or {})

    compat_map = s.get("compat_map")
    if isinstance(compat_map, dict):
        s = _apply_compat_map(s, compat_map)

    s.setdefault("blacklist", {})
    s.setdefault("blacklist_reason", {})
    s.setdefault("consecutive_losses", {})
    s.setdefault("last_exit_time", {})
    s.setdefault("known_exit_order_ids", [])
    s.setdefault("positions", {})
    s.setdefault("symbol_performance", {})
    s.setdefault("entry_confidence_history", {})
    s.setdefault("funding_rate_snapshot", {})
    s.setdefault("entry_watches", {})
    s.setdefault("run_context", {})
    s.setdefault("streak_history", [])

    # known_exit_order_ids -> list[str] capped
    kei = s.get("known_exit_order_ids")
    if isinstance(kei, set):
        kei_list = list(kei)
    elif isinstance(kei, tuple):
        kei_list = list(kei)
    elif isinstance(kei, list):
        kei_list = kei
    else:
        kei_list = []
    if len(kei_list) > KNOWN_EXIT_IDS_CAP:
        kei_list = kei_list[-KNOWN_EXIT_IDS_CAP:]
    s["known_exit_order_ids"] = [str(x) for x in kei_list if x is not None]

    # streak_history -> primitives [iso_str, int, float]
    sh = s.get("streak_history")
    if isinstance(sh, list):
        fixed: List[List[Any]] = []
        for item in sh:
            if not isinstance(item, (list, tuple)) or len(item) != 3:
                continue
            d, streak_len, pnl = item
            if isinstance(d, date):
                d_str = d.isoformat()
            elif isinstance(d, str):
                d2 = _parse_iso_date(d)
                if d2 is None:
                    continue
                d_str = d2.isoformat()
            else:
                continue
            fixed.append([d_str, int(streak_len or 0), float(pnl or 0.0)])
        s["streak_history"] = fixed
    else:
        s["streak_history"] = []

    s.setdefault("schema_version", int(s.get("schema_version", 1) or 1))

    # Enforce canonical law BEFORE PsycheState.from_loaded()
    s = _canonicalize_payload_state(s)

    # Final hardening
    s = _to_primitive_safe(s)  # type: ignore[assignment]
    return s  # type: ignore[return-value]


# -----------------------
# Envelope IO
# -----------------------
def _pack_envelope(payload_bytes: bytes) -> bytes:
    compressed = lz4.frame.compress(payload_bytes)
    checksum = _sha256_hex(compressed)
    payload_sha = _sha256_hex(payload_bytes)
    env = {"checksum": checksum, "payload_sha": payload_sha, "blob": compressed}
    return msgpack.packb(env, use_bin_type=True)


def _unpack_envelope(raw: bytes) -> Optional[Dict[str, Any]]:
    try:
        env = msgpack.unpackb(raw, raw=False)
    except Exception:
        return None

    if not isinstance(env, dict):
        return None
    if "checksum" not in env or "blob" not in env:
        return None

    checksum = env.get("checksum")
    blob = env.get("blob")

    if isinstance(checksum, (bytes, bytearray)):
        try:
            checksum = checksum.decode("utf-8", errors="ignore")
        except Exception:
            return None
    if not isinstance(checksum, str):
        return None

    if not isinstance(blob, (bytes, bytearray)):
        return None

    blob = bytes(blob)
    if _sha256_hex(blob) != checksum:
        return None

    payload_sha = env.get("payload_sha")
    if isinstance(payload_sha, (bytes, bytearray)):
        try:
            payload_sha = payload_sha.decode("utf-8", errors="ignore")
        except Exception:
            payload_sha = None
    if not isinstance(payload_sha, str):
        payload_sha = None

    return {"blob": blob, "checksum": checksum, "payload_sha": payload_sha}


def _rotate_backups():
    oldest = f"{BRAIN_PATH}.bak{MAX_BACKUPS}"
    if os.path.exists(oldest):
        try:
            os.unlink(oldest)
        except Exception:
            pass

    for i in range(MAX_BACKUPS - 1, 0, -1):
        src = f"{BRAIN_PATH}.bak{i}"
        dst = f"{BRAIN_PATH}.bak{i+1}"
        if os.path.exists(src):
            try:
                os.replace(src, dst)
            except Exception:
                pass


async def _atomic_write(path: str, data: bytes) -> None:
    """
    Atomic-ish write with rollback:
      1) write tmp + fsync tmp
      2) rotate backups
      3) move main->bak1 (if exists)
      4) move tmp->main
      5) fsync dir best-effort
    """
    tmp = path + ".tmp"

    async with aiofiles.open(tmp, "wb") as f:
        await f.write(data)
        try:
            await f.flush()
        except Exception:
            pass

    try:
        fd = os.open(tmp, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except Exception:
        pass

    _rotate_backups()

    bak1 = f"{path}.bak1"
    main_existed = os.path.exists(path)

    if main_existed:
        try:
            os.replace(path, bak1)
        except Exception:
            pass

    try:
        os.replace(tmp, path)
    except Exception as e:
        try:
            if os.path.exists(bak1):
                os.replace(bak1, path)
        except Exception:
            pass
        try:
            if os.path.exists(tmp):
                os.unlink(tmp)
        except Exception:
            pass
        raise e

    try:
        d = os.path.dirname(path) or "."
        _fsync_best_effort(d)
    except Exception:
        pass


# -----------------------
# Public API
# -----------------------
async def save_brain(state: PsycheState, force: bool = False):
    """SAVE — IMMORTALITY LAYER"""
    global _memory_fallback_payload, _disk_failed

    async with _IO_LOCK:
        if _disk_failed and not force:
            try:
                _memory_fallback_payload = _state_to_payload(state)
            except Exception:
                _memory_fallback_payload = None
            log_brain.info("Disk previously failed — saving to memory fallback (not forced)")
            return

        try:
            payload_state = _state_to_payload(state)
            payload_state = _canonicalize_payload_state(payload_state)
            payload_state = _to_primitive_safe(payload_state)

            data = {
                "v": PERSISTENCE_VERSION,
                "timestamp": float(time.time()),
                "meta": {
                    "core_version_seen": getattr(state, "version", None),
                    "schema_version": int(getattr(state, "schema_version", 1)),
                },
                "state": payload_state,
            }

            data = _to_primitive_safe(data)  # type: ignore[assignment]

            payload_bytes = msgpack.packb(data, use_bin_type=True)
            envelope = _pack_envelope(payload_bytes)

            await _atomic_write(BRAIN_PATH, envelope)

            _disk_failed = False
            _memory_fallback_payload = None

            log_brain.critical(
                f"BRAIN SAVED | Size: {len(envelope)/1024:.1f}KB | v={PERSISTENCE_VERSION}"
            )
        except Exception as e:
            _disk_failed = True
            try:
                _memory_fallback_payload = _state_to_payload(state)
            except Exception:
                _memory_fallback_payload = None
            log_brain.critical(f"BRAIN SAVE FAILED: {e} — FALLING BACK TO MEMORY")


async def load_brain(state: PsycheState) -> bool:
    """LOAD — RESURRECTION"""
    global _memory_fallback_payload, _disk_failed

    async with _IO_LOCK:
        runtime_version = getattr(state, "version", None)

        # 1) Memory fallback
        if _memory_fallback_payload is not None:
            try:
                payload = _migrate_payload_state(_memory_fallback_payload, PERSISTENCE_VERSION)
                loaded = PsycheState.from_loaded(payload)
                _apply_state_in_place(dst=state, src=loaded, preserve_version=runtime_version)
                log_brain.critical("RESURRECTED FROM MEMORY FALLBACK")
                _memory_fallback_payload = None
                return True
            except Exception as e:
                log_brain.critical(f"Memory fallback resurrection failed: {e}")

        # 2) Disk resurrection: main + backups
        for i in range(MAX_BACKUPS + 1):
            path = BRAIN_PATH if i == 0 else f"{BRAIN_PATH}.bak{i}"
            if not os.path.exists(path):
                continue

            try:
                async with aiofiles.open(path, "rb") as f:
                    raw = await f.read()

                env = _unpack_envelope(raw)
                if env is None:
                    log_brain.warning(f"Invalid/corrupt envelope in {path} — skipping")
                    continue

                blob = env["blob"]
                payload_bytes = lz4.frame.decompress(blob)

                if env.get("payload_sha"):
                    try:
                        if _sha256_hex(payload_bytes) != env["payload_sha"]:
                            log_brain.warning(f"Payload SHA mismatch in {path} (continuing anyway)")
                    except Exception:
                        pass

                data = msgpack.unpackb(payload_bytes, raw=False)
                if not isinstance(data, dict):
                    log_brain.warning(f"Invalid payload root in {path} — skipping")
                    continue

                v = data.get("v")
                if v not in ACCEPTED_VERSIONS:
                    log_brain.warning(f"Unsupported persistence version {v} in {path} — skipping")
                    continue

                payload_state = data.get("state") or {}
                if not isinstance(payload_state, dict):
                    log_brain.warning(f"Invalid payload state in {path} — skipping")
                    continue

                payload_state = _migrate_payload_state(payload_state, str(v))
                loaded = PsycheState.from_loaded(payload_state)

                _apply_state_in_place(dst=state, src=loaded, preserve_version=runtime_version)

                _disk_failed = False
                log_brain.critical(
                    f"BRAIN RESURRECTED FROM {path} — {len(getattr(state, 'positions', {}) or {})} positions"
                )

                # Heal forward: if loaded from backup, re-save main
                try:
                    if i != 0:
                        await save_brain(state, force=True)
                except Exception:
                    pass

                return True

            except Exception as e:
                log_brain.critical(f"Resurrection failed from {path}: {e}")

        log_brain.info("No brain found — fresh consciousness")
        return False


# -----------------------
# In-place apply helper
# -----------------------
def _apply_state_in_place(dst: PsycheState, src: PsycheState, preserve_version: Optional[str]) -> None:
    """
    Copy fields from src into dst, preserving dst.version if provided.
    Preserves dst object identity.
    """
    try:
        for k, v in vars(src).items():
            if k == "version":
                continue
            setattr(dst, k, v)

        if preserve_version is not None:
            dst.version = preserve_version
    except Exception:
        pass

    try:
        dst.validate()
    except Exception:
        pass
    try:
        dst.recompute_derived()
    except Exception:
        pass
